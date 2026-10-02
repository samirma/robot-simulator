"""spawn.sh's lifecycle (spec §2.3, §5 Lifecycle), on MolmoSpaces ithor:1: readiness line and
exit statuses, admission, all-or-nothing startup, cleanup after every kind of end, and
isolation of the robots that stay."""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from conftest import (REPO, SIM, Spawn, docker_containers, docker_ok, port_free, running_sim,
                      wait_no_containers, wait_port_free)

import protocol

SIM_PORT = 9381
pytestmark = pytest.mark.skipif(not docker_ok(), reason="Docker is not running")


@pytest.fixture(scope="module")
def sim(logdir):
    with running_sim("molmospaces", "ithor:1", SIM_PORT, logdir) as s:
        yield s


def robots(sim_port=SIM_PORT):
    c = protocol.Client("127.0.0.1", sim_port)
    try:
        return {r["id"]: r for r in c.call("robots")["robots"]}
    finally:
        c.close()


def wait(pred, timeout=30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.3)
    return False


def containers(rid, sim_port=SIM_PORT):
    return docker_containers(f"label=rsim.robot={rid}") and \
        [c for c in docker_containers(f"label=rsim.sim_port={sim_port}") if f"-{rid}-" in c]


def assert_released(rid, ports, sim_port=SIM_PORT):
    assert wait(lambda: rid not in robots(sim_port)), f"{rid} still in the simulation"
    assert wait(lambda: not containers(rid, sim_port)), f"{rid} containers remain"
    for p in ports:
        assert wait_port_free(p, 30), f"port {p} still held"


def test_ready_line_and_sigint_exit_zero(sim, logdir):
    sp = Spawn("myagv", SIM_PORT, 9391, placement="floor", log=logdir / "lc-myagv.log")
    line = sp.wait_ready()
    assert line.startswith("spawn ready: myagv") and "ws://127.0.0.1:9391" in line
    assert "myagv" in robots() and robots()["myagv"]["state"] == "running"
    assert containers("myagv")
    assert sp.stop(signal.SIGINT) == 0
    assert_released("myagv", [9391])


def test_sigterm_exit_zero_and_composite_ports(sim, logdir):
    sp = Spawn("myagv_mycobot280", SIM_PORT, 9392, placement="floor", arm_port=9394,
               log=logdir / "lc-comp.log")
    line = sp.wait_ready()
    assert "ws://127.0.0.1:9392" in line and "ws://127.0.0.1:9394" in line
    assert sp.stop(signal.SIGTERM) == 0
    assert_released("myagv_mycobot280", [9392, 9394])


def test_duplicate_and_busy_refusals_and_isolation(sim, logdir):
    a = Spawn("myagv", SIM_PORT, 9391, placement="floor", log=logdir / "lc-a.log")
    a.wait_ready()
    c = protocol.Client("127.0.0.1", SIM_PORT)
    before = c.call("readings", robot="myagv")
    # a duplicate id is refused, and leaves the running one alone
    dup = Spawn("myagv", SIM_PORT, 9395, placement="floor", log=logdir / "lc-dup.log")
    assert dup.wait_exit(120) != 0
    assert "already spawned" in dup.text()
    assert port_free(9395)
    # a second spawn while another starts up is refused
    b = Spawn("so101", SIM_PORT, 9396, log=logdir / "lc-b.log")
    assert wait(lambda: "placed on" in b.text() or b.proc.poll() is not None, 300)
    busy = Spawn("mycobot280", SIM_PORT, 9397, log=logdir / "lc-busy.log")
    code = busy.wait_exit(120)
    if "spawn ready" not in b.text():
        assert code != 0 and "starting up" in busy.text()
    b.wait_ready()
    # the worktop now holds so101: another arm is refused there, nothing left behind
    arm = Spawn("mycobot280", SIM_PORT, 9397, log=logdir / "lc-arm.log")
    assert arm.wait_exit(120) != 0 and "worktop already holds so101" in arm.text()
    assert "mycobot280" not in robots() and not containers("mycobot280")
    # removing so101 does not disturb the myAGV
    assert b.stop() == 0
    assert_released("so101", [9396])
    after = c.call("readings", robot="myagv")
    assert after["sim_time"] > before["sim_time"]
    moved = sum((x - y) ** 2 for x, y in zip(after["base"]["pos"], before["base"]["pos"])) ** 0.5
    assert moved < 0.01
    c.close()
    assert a.stop() == 0
    assert_released("myagv", [9391])


def test_worktop_objects_come_and_go_with_the_arm(sim, logdir):
    """The six worktop objects are the scene's, there before spawn.sh so101 and after it
    ends, with the island's own loose objects cleared for them throughout."""
    c = protocol.Client("127.0.0.1", SIM_PORT)
    s0 = c.call("scene")
    names = s0["free_bodies"]
    assert {n for n in names if n.startswith("task_")} >= {"task_apple", "task_bowl"}
    before = c.call("bodies", names=names)["bodies"]
    sp = Spawn("so101", SIM_PORT, 9396, log=logdir / "lc-objects.log")
    sp.wait_ready()
    me = robots()["so101"]
    assert set(me["staged"]) == {"apple", "plate", "bowl", "mug", "banana", "lemon"}
    assert sorted(n.split("_")[0] for n in me["cleared"]) == ["apple", "book", "bread"]
    assert "worktop objects staged" in sp.text()
    sunk = c.call("bodies", names=me["cleared"])["bodies"]
    assert all(b["pos"][2] < -40 for b in sunk.values())
    assert sp.stop() == 0
    assert_released("so101", [9396])
    sc = c.call("scene")
    assert sc["staging"] == s0["staging"] and sorted(sc["free_bodies"]) == sorted(names)
    after = c.call("bodies", names=names)["bodies"]
    for n in names:
        moved = sum((a - b) ** 2 for a, b in zip(after[n]["pos"], before[n]["pos"])) ** 0.5
        assert moved < 0.005, n
    c.close()


def test_sigkill_releases_robot_containers_ports_and_id(sim, logdir):
    sp = Spawn("so101", SIM_PORT, 9396, log=logdir / "lc-kill.log")
    sp.wait_ready()
    os.kill(sp.proc.pid, signal.SIGKILL)
    sp.proc.wait()
    assert_released("so101", [9396])
    again = Spawn("so101", SIM_PORT, 9396, log=logdir / "lc-kill2.log")
    again.wait_ready()
    assert again.stop() == 0
    assert_released("so101", [9396])


def test_container_exit_ends_the_spawn(sim, logdir):
    sp = Spawn("myagv", SIM_PORT, 9391, placement="floor", log=logdir / "lc-cexit.log")
    sp.wait_ready()
    subprocess.run(["docker", "kill", f"rsim-{SIM_PORT}-myagv-main"], stdout=subprocess.DEVNULL)
    assert sp.wait_exit(60) != 0
    assert "container exited" in sp.text()
    assert_released("myagv", [9391])


def test_serving_process_failure_ends_the_spawn(sim, logdir):
    sp = Spawn("myagv", SIM_PORT, 9391, placement="floor", log=logdir / "lc-proc.log")
    sp.wait_ready()
    name = f"rsim-{SIM_PORT}-myagv-main"
    subprocess.run(["docker", "exec", name, "pkill", "-9", "-f", "rosbridge_websocket"])
    # the container itself keeps running; the spawn notices the wire stopped serving
    assert sp.wait_exit(60) != 0
    assert "stopped serving" in sp.text() or "lost node" in sp.text()
    assert_released("myagv", [9391])


def test_vendor_node_failure_ends_the_spawn(sim, logdir):
    sp = Spawn("myagv", SIM_PORT, 9391, placement="floor", log=logdir / "lc-node.log")
    sp.wait_ready()
    subprocess.run(["docker", "exec", f"rsim-{SIM_PORT}-myagv-main", "pkill", "-9", "-f",
                    "robot_pose_ekf"])
    assert sp.wait_exit(60) != 0 and "lost node(s) /robot_pose_ekf" in sp.text()
    assert_released("myagv", [9391])


def test_one_of_two_wires_failing_ends_the_whole_spawn(sim, logdir):
    sp = Spawn("myagv_mycobot280", SIM_PORT, 9392, placement="floor",
               log=logdir / "lc-onewire.log")
    sp.wait_ready()
    subprocess.run(["docker", "kill", f"rsim-{SIM_PORT}-myagv_mycobot280-arm"],
                   stdout=subprocess.DEVNULL)
    assert sp.wait_exit(60) != 0
    assert_released("myagv_mycobot280", [9392, 9393])


def test_partially_failed_startup_leaves_nothing(sim, logdir):
    env_before = os.environ.get("RSIM_TEST_FAIL_ROLE")
    os.environ["RSIM_TEST_FAIL_ROLE"] = "arm"
    try:
        sp = Spawn("myagv_mycobot280", SIM_PORT, 9392, placement="floor",
                   log=logdir / "lc-partial.log")
    finally:
        if env_before is None:
            del os.environ["RSIM_TEST_FAIL_ROLE"]
        else:
            os.environ["RSIM_TEST_FAIL_ROLE"] = env_before
    assert sp.wait_exit(400) != 0
    assert "failed to start" in sp.text()
    assert_released("myagv_mycobot280", [9392, 9393])
    assert robots() == {} or "myagv_mycobot280" not in robots()
    c = protocol.Client("127.0.0.1", SIM_PORT)
    assert c.call("robots")["starting"] is None
    c.close()


def test_simulation_end_ends_every_spawn_nonzero(logdir):
    with running_sim("molmospaces", "ithor:1", 9382, logdir) as other:
        sp = Spawn("myagv", 9382, 9398, placement="floor", log=logdir / "lc-simend.log")
        sp.wait_ready()
        assert other.stop() == 0
        assert sp.wait_exit(60) != 0
        assert "simulation ended" in sp.text()
        assert wait(lambda: not docker_containers("label=rsim.sim_port=9382"))
        assert wait_port_free(9398, 30)


def test_refusals_without_docker_or_files(sim, tmp_path):
    out = subprocess.run([str(SIM / "spawn.sh"), "myagv", "--sim-port", str(SIM_PORT),
                          "--port", "9391"], env=dict(os.environ, DOCKER_HOST="tcp://127.0.0.1:1"),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert out.returncode != 0 and "Docker is not running" in out.stdout
    # a robot whose required files are missing is refused, naming run.sh setup
    fake = tmp_path / "repo"
    (fake / "robots_specs").mkdir(parents=True)
    shutil.copy(REPO / "robots_specs" / "high_level_spec.md", fake / "robots_specs")
    out = subprocess.run([str(SIM / "spawn.sh"), "ainex", "--sim-port", str(SIM_PORT)],
                         env=dict(os.environ, RSIM_REPO_ROOT=str(fake)), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=120)
    assert out.returncode != 0 and "run.sh setup" in out.stdout and "missing" in out.stdout
    assert "ainex" not in robots()
