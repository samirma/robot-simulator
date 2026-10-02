"""RoboCasa scenes carry RoboCasa's own kitchen objects (spec §2.1): graspable objects drawn
from its object library, standing on counter tops, the same on every start."""

import pytest

from conftest import engine_ready, running_sim

SCENE = "robocasa:50-1"
PORTS = (9284, 9285)


def objects_of(sim):
    c = sim.client()
    names = [n for n in c.call("scene")["free_bodies"] if n.startswith("rc_obj_")]
    bodies = c.call("bodies", names=names)["bodies"]
    c.close()
    return {n: bodies[n]["pos"] for n in names}


def test_robocasa_objects_stand_on_counters_and_repeat(logdir):
    if not engine_ready("robocasa"):
        pytest.skip("robocasa is not set up")
    runs = []
    for port in PORTS:
        with running_sim("robocasa", SCENE, port, logdir) as sim:
            runs.append(objects_of(sim))
    first, second = runs
    assert len(first) >= 5, f"only {len(first)} RoboCasa objects loaded"
    # on a counter top (RoboCasa's counters are 0.92 m high; the worktop island too), the
    # same objects at the same places on every start
    assert first.keys() == second.keys()
    for name, (x, y, z) in first.items():
        assert 0.85 < z < 1.3, (name, z)
        assert max(abs(a - b) for a, b in zip(second[name], (x, y, z))) < 0.02, name
