"""RoboCasa scenes carry RoboCasa's own kitchen objects (spec §2.1, §5 Reference parity):
graspable objects drawn from its object library, standing on counter tops -- the
`Counter.get_reset_regions()` RoboCasa's own API reports (`reference/counter_tops.py`),
not this project's reading of them -- at most 24 and about eight per square metre of
counter, not touching one another, the same on every start. Those in the worktop objects' area are cleared.
They come to rest where they are set and stay there, the scene solved by constraint island
(§2.1, amended 2026-10-03)."""

import json
import math
import os
import subprocess
import time

import pytest

from conftest import SIM, TESTS, engine_ready, running_sim

#: the default scene (the reference's kitchen), and a layout with an island and more counters
SCENES = ("robocasa:1-1", "robocasa:50-1")
PORTS = (9284, 9285)
MAX_OBJECTS, PER_SQUARE_METRE = 24, 8.0
#: an object's origin stands at most this high above the top it stands on (RoboCasa
#: objects are drawn no larger than 0.30 m)
MAX_ORIGIN_HEIGHT = 0.32


def counter_tops(layout: int, style: int) -> list:
    cmd = (f'source "{SIM / "robocasa" / "env.sh"}" >/dev/null; '
           f'exec "{SIM / "robocasa" / ".venv" / "bin" / "python"}" '
           f'"{TESTS / "reference" / "counter_tops.py"}" {layout} {style}')
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, text=True, timeout=600, env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


#: the scene as `scenes.load` builds it, compiled and forwarded before any step: the pairs
#: of RoboCasa objects in contact
TOUCHING = """
import contextlib, json, sys
import mujoco
sys.path.insert(0, sys.argv[3])
import scenes
with contextlib.redirect_stdout(sys.stderr):
    _, spec = scenes.build_kitchen_arena(int(sys.argv[1]), int(sys.argv[2]), objects=True)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_rootid[m.geom_bodyid[g]]) or ""
pairs = set()
for c in d.contact[:d.ncon]:
    a, b = name(c.geom1), name(c.geom2)
    if a != b and a.startswith("rc_obj_") and b.startswith("rc_obj_"):
        pairs.add(tuple(sorted((a, b))))
print(json.dumps(sorted(pairs)))
"""


def touching(layout: int, style: int) -> list:
    cmd = (f'source "{SIM / "robocasa" / "env.sh"}" >/dev/null; '
           f'exec "{SIM / "robocasa" / ".venv" / "bin" / "python"}" -c \'{TOUCHING}\' '
           f'{layout} {style} "{SIM / "shared"}"')
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, text=True, timeout=600, env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def on_top(pos, top) -> bool:
    """Inside the top's rectangle and standing on it."""
    dx, dy = pos[0] - top["centre"][0], pos[1] - top["centre"][1]
    c, s = math.cos(top["rot"]), math.sin(top["rot"])
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return (abs(lx) <= top["half"][0] and abs(ly) <= top["half"][1]
            and -0.005 <= pos[2] - top["top_z"] <= MAX_ORIGIN_HEIGHT)


#: the simulation time the objects are read at on every start, so two starts are compared
#: at the same physics time, not at whatever wall-clock moment each start became ready
SAMPLE_AT = 8.0


def objects_of(sim):
    """The RoboCasa objects standing in the scene ({name: pos}) at simulation time
    `SAMPLE_AT`, and those cleared."""
    c = sim.client()
    s = c.call("scene")
    names = [n for n in s["free_bodies"] if n.startswith("rc_obj_")]
    reply = c.call("bodies", names=names)
    assert reply["sim_time"] < SAMPLE_AT, f"started too slowly: sim time {reply['sim_time']:.1f} s"
    while reply["sim_time"] < SAMPLE_AT:
        time.sleep(0.01)
        reply = c.call("bodies", names=names)
    c.close()
    assert reply["sim_time"] < SAMPLE_AT + 0.2, f"read at sim time {reply['sim_time']:.2f} s"
    cleared = [n for n in s["staging"].get("scene", {}).get("cleared", []) if n.startswith("rc_obj_")]
    return {n: reply["bodies"][n]["pos"] for n in names}, cleared


@pytest.mark.parametrize("scene", SCENES)
def test_robocasa_objects_stand_on_counters_and_repeat(scene, logdir):
    if not engine_ready("robocasa"):
        pytest.skip("robocasa is not set up")
    tops = counter_tops(*(int(v) for v in scene.split(":")[1].split("-")))
    area = sum(4.0 * t["half"][0] * t["half"][1] for t in tops)
    runs = []
    for port in PORTS:
        with running_sim("robocasa", scene, port, logdir) as sim:
            runs.append(objects_of(sim))
    (first, cleared), (second, cleared2) = runs
    drawn = len(first) + len(cleared)
    assert drawn >= 5 and first, f"only {drawn} RoboCasa objects loaded ({len(first)} standing)"
    assert drawn <= min(MAX_OBJECTS, round(area * PER_SQUARE_METRE)), (drawn, area)
    # every one on a counter top, the same objects at the same places on every start
    for name, pos in first.items():
        assert any(on_top(pos, t) for t in tops), (name, pos)
    assert first.keys() == second.keys() and cleared == cleared2
    for name, (x, y, z) in first.items():
        assert max(abs(a - b) for a, b in zip(second[name], (x, y, z))) < 0.02, name
    # stood not touching one another (on one counter or across neighbouring ones)
    assert touching(*(int(v) for v in scene.split(":")[1].split("-"))) == []


#: §2.1 (amended 2026-10-03): RoboCasa's objects rest where they are set and stay there
REST_SCENES = ("robocasa:1-1", "robocasa:3-5", "robocasa:5-2")
REST_PORT = 9287
#: how far a resting object may move between the two readings, metres
AT_REST = 0.005

#: the RoboCasa scene as `scenes.load` builds it: whether its solver works by constraint island
ISLANDS = """
import contextlib, json, sys
import mujoco
sys.path.insert(0, sys.argv[2])
import scenes
with contextlib.redirect_stdout(sys.stderr):
    m = scenes.load("robocasa", sys.argv[1]).spec.compile()
if hasattr(mujoco.mjtEnableBit, "mjENBL_ISLAND"):
    on = bool(m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_ISLAND))
else:
    on = not m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
print(json.dumps({"islands": on, "mujoco": mujoco.__version__}))
"""


def test_robocasa_scene_solves_by_island():
    if not engine_ready("robocasa"):
        pytest.skip("robocasa is not set up")
    cmd = (f'source "{SIM / "robocasa" / "env.sh"}" >/dev/null; '
           f'exec "{SIM / "robocasa" / ".venv" / "bin" / "python"}" -c \'{ISLANDS}\' '
           f'robocasa:1-1 "{SIM / "shared"}"')
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, text=True, timeout=600, env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["islands"], got


@pytest.mark.parametrize("scene", REST_SCENES)
def test_robocasa_objects_stay_at_rest(scene, logdir):
    """Every RoboCasa object and staged worktop object stands still from the first reading
    after start to one 9 s of physics later, still on a counter top."""
    if not engine_ready("robocasa"):
        pytest.skip("robocasa is not set up")
    tops = counter_tops(*(int(v) for v in scene.split(":")[1].split("-")))
    with running_sim("robocasa", scene, REST_PORT, logdir) as sim:
        c = sim.client()
        s = c.call("scene")
        cleared = set(s["staging"].get("scene", {}).get("cleared", []))
        names = [n for n in s["free_bodies"] if n not in cleared]
        first = c.call("bodies", names=names)
        while first["sim_time"] < 1.0:
            time.sleep(0.05)
            first = c.call("bodies", names=names)
        assert first["sim_time"] < 5.0, f"started too slowly: sim time {first['sim_time']:.1f} s"
        later = first
        while later["sim_time"] < first["sim_time"] + 9.0:
            time.sleep(0.2)
            later = c.call("bodies", names=names)
        c.close()
    moved = {n: round(math.dist(first["bodies"][n]["pos"], later["bodies"][n]["pos"]), 4)
             for n in names}
    assert {n: d for n, d in moved.items() if d > AT_REST} == {}, \
        f"objects moved over {later['sim_time'] - first['sim_time']:.1f} s of physics"
    for n in names:
        if n.startswith("rc_obj_"):
            assert any(on_top(later["bodies"][n]["pos"], t) for t in tops), (n, later["bodies"][n])
