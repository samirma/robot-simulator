"""The simulation process over its control port, without Docker: scene, rendering,
readings, admission, placement refusals, removal on a lost lease, state preservation."""

import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import SHARED, running_sim

import protocol

PORT = 9281


@pytest.fixture(scope="module")
def sim(logdir):
    with running_sim("molmospaces", "ithor:1", PORT, logdir) as s:
        yield s


def client():
    return protocol.Client("127.0.0.1", PORT)


def wait(pred, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


def robots(c):
    return {r["id"]: r for r in c.call("robots")["robots"]}


def test_hello_and_scene(sim):
    c = client()
    h = c.call("hello")
    assert h["engine"] == "molmospaces" and h["scene"] == "ithor:1"
    # the worktop is the reference survey's: FloorPlan1's island, top at 1.10 m
    assert abs(h["worktop"]["z"] - 1.10001) < 1e-4 and abs(h["floor_z"]) < 0.01
    assert h["worktop"]["name"].startswith("standardislandheight")
    s = c.call("scene")
    assert s["scene_nbody"] == 242 + 6 and s["nlight"] == 1 and len(s["free_bodies"]) > 10
    # the six worktop objects are the scene's own, there with no robot
    assert set(s["staging"]) == {"scene"}
    assert sorted(s["staging"]["scene"]["staged"]) == sorted(
        ["apple", "plate", "bowl", "mug", "banana", "lemon"])
    c.close()


def test_worktop_staging_and_its_removal(sim):
    """The six objects and the cleared loose objects are the scene's, there before a robot
    is; an arm stands among them and, when it goes, nothing of them has changed."""
    c = client()
    s0 = c.call("scene")
    scene = s0["staging"]["scene"]
    names = list(s0["free_bodies"])
    assert sorted(n.split("_")[0] for n in scene["cleared"]) == ["apple", "book", "bread"]
    sunk = c.call("bodies", names=scene["cleared"])["bodies"]
    assert all(b["pos"][2] < -40 for b in sunk.values())     # parked under the scene
    before = c.call("bodies", names=names)["bodies"]
    res = c.call("spawn", robot="so101", placement="worktop", ports=[])
    c.call("commit")
    assert res["staged"] == scene["staged"] and res["cleared"] == scene["cleared"]
    row = robots(c)["so101"]
    assert row["staged"] == res["staged"] and row["cleared"] == res["cleared"]
    # the arm stands at the scene's objects: they have not moved
    mid = c.call("bodies", names=names)["bodies"]
    for n in names:
        assert np.linalg.norm(np.subtract(mid[n]["pos"], before[n]["pos"])) < 0.02, n
    c.call("remove")
    assert wait(lambda: robots(c) == {})
    after = c.call("bodies", names=names)["bodies"]
    for n in names:
        assert np.linalg.norm(np.subtract(after[n]["pos"], before[n]["pos"])) < 0.02, n
    sc = c.call("scene")
    assert sc["nbody"] == s0["nbody"] and sc["staging"] == s0["staging"]
    assert sorted(sc["free_bodies"]) == sorted(names)
    c.close()


def test_render_png_rgb_depth(sim):
    c = client()
    wt = c.call("hello")["worktop"]
    r = c.call("render", view={"lookat": wt["centroid"] + [wt["z"]], "distance": 2.5,
                               "azimuth": 45, "elevation": -35}, width=320, height=240)
    assert r["_payload"][:8] == b"\x89PNG\r\n\x1a\n"
    r = c.call("render", view={"pos": [0.3, -1.2, 2.0], "target": [-1.6, -1.4, 0.9]},
               width=320, height=240, format="rgb")
    assert len(r["_payload"]) == 320 * 240 * 3
    # straight down onto the open floor from 1 m: the floor is 1 m away
    r = c.call("render", view={"pos": [1.47, 1.52, 1.0], "target": [1.47, 1.5201, 0.0]},
               width=64, height=48, format="depth")
    depth = np.frombuffer(r["_payload"], np.float32).reshape(48, 64)
    assert abs(depth[24, 32] - 1.0) < 0.02
    c.close()


def test_admission_duplicate_busy_and_release(sim):
    a, b = client(), client()
    ra = a.call("spawn", robot="myagv", placement="floor", ports=[9990])
    assert ra["robot"] == "myagv"
    with pytest.raises(protocol.RemoteError, match="another spawn \\(myagv\\) is starting up"):
        b.call("spawn", robot="so101", placement="worktop", ports=[9991])
    with pytest.raises(protocol.RemoteError, match="another spawn"):
        b.call("precheck", robot="so101", placement="worktop", ports=[9991])
    a.call("commit")
    with pytest.raises(protocol.RemoteError, match="already spawned"):
        b.call("spawn", robot="myagv", placement="floor", ports=[9992])
    with pytest.raises(protocol.RemoteError, match="port 9990 is reserved"):
        b.call("spawn", robot="so101", placement="worktop", ports=[9990])
    assert b.call("spawn", robot="so101", placement="worktop", ports=[9991])["robot"] == "so101"
    b.call("commit")
    c = client()
    assert set(robots(c)) == {"myagv", "so101"}
    # the worktop is taken: another arm is refused and nothing is left behind
    with pytest.raises(protocol.RemoteError, match="worktop already holds so101"):
        c.call("spawn", robot="mycobot280", placement="worktop", ports=[9993])
    assert set(robots(c)) == {"myagv", "so101"} and c.call("robots")["starting"] is None
    # ending a spawn (its connection) removes its robot and releases its id and port
    a.close()
    assert wait(lambda: "myagv" not in robots(c))
    d = client()
    assert d.call("spawn", robot="myagv", placement="floor", ports=[9990])["robot"] == "myagv"
    d.call("remove")
    b.call("remove")
    assert wait(lambda: robots(c) == {})
    for x in (b, c, d):
        x.close()


def test_unknown_robot_is_refused_with_the_list(sim):
    c = client()
    with pytest.raises(protocol.RemoteError, match="(?s)unknown robot id .foo.*myagv"):
        c.call("spawn", robot="foo", placement="floor", ports=[])
    c.close()


def test_sigkilled_spawn_releases_robot_id_and_placement(sim):
    code = ("import sys, time; sys.path.insert(0, %r); import protocol; "
            "c = protocol.Client('127.0.0.1', %d); c.call('spawn', robot='so101', "
            "placement='worktop', ports=[9994]); c.call('commit'); print('up', flush=True); "
            "time.sleep(600)" % (str(SHARED), PORT))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "up"
    c = client()
    assert "so101" in robots(c)
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    assert wait(lambda: "so101" not in robots(c))
    # id, port and worktop are free again
    assert c.call("spawn", robot="so101", placement="worktop", ports=[9994])["robot"] == "so101"
    c.call("remove")
    c.close()


def test_readings_ctrl_and_state_preserved_across_spawn_and_removal(sim):
    owner = client()
    res = owner.call("spawn", robot="myagv", placement="floor", ports=[])
    owner.call("commit")
    wire = client()
    desc = wire.call("wire", token=res["token"], role="main")["describe"]
    wheels = [a["name"] for a in desc["actuators"] if a["kind"] == "velocity"]
    assert len(wheels) == 4
    wire.call("ctrl", values={w: 5.0 for w in wheels})       # the myAGV drives forward
    time.sleep(1.0)
    r0 = owner.call("readings", robot="myagv")
    other = client()
    other.call("spawn", robot="rosmaster_x3_plus", placement="floor", ports=[])
    other.call("commit")
    r1 = owner.call("readings", robot="myagv")
    # neither time nor the moving robot was reset by the recompile; it keeps its speed
    assert r1["sim_time"] > r0["sim_time"]
    v0 = np.linalg.norm(r0["base"]["linvel_world"][:2])
    v1 = np.linalg.norm(r1["base"]["linvel_world"][:2])
    assert v0 > 0.1 and abs(v1 - v0) < 0.05
    for k in wheels:
        assert abs(r1["ctrl"][k] - 5.0) < 1e-9
    other.call("remove")
    r2 = owner.call("readings", robot="myagv")
    assert r2["sim_time"] > r1["sim_time"]
    assert abs(np.linalg.norm(r2["base"]["linvel_world"][:2]) - v0) < 0.05
    d = np.array(r2["base"]["pos"][:2]) - np.array(r0["base"]["pos"][:2])
    assert np.linalg.norm(d) > 0.05       # it kept driving
    wire.call("ctrl", values={w: 0.0 for w in wheels})
    owner.call("remove")
    for x in (owner, wire, other):
        x.close()


def test_robot_camera_render_matches_its_model_camera(sim):
    c = client()
    c.call("spawn", robot="so101", placement="worktop", ports=[])
    c.call("commit")
    r = c.call("render", view={"camera": "so101/default_cam"}, width=640, height=480,
               format="rgb")
    img = np.frombuffer(r["_payload"], np.uint8).reshape(480, 640, 3)
    assert img.std() > 1.0
    rd = c.call("readings", robot="so101")
    assert set(rd["joints"]) >= {"shoulder_pan_joint", "gripper_joint"}
    assert "base_link" in rd["bodies"]
    fr = c.call("render", view={"frame_robot": "so101"}, width=320, height=240)
    assert fr["_payload"][:4] == b"\x89PNG"
    c.call("remove")
    c.close()
