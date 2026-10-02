"""Reference parity of the worktop (spec §5 *Reference parity*): on each engine's default
scene the SO-101 stands where the reference project stood it, with the reference's six
objects at the reference's poses and the same scene objects cleared.

The reference is run from a checkout of github.com/samirma/robot-simulator rev 34547ae
named by `RSIM_REF_DIR` (skipped without it), through `reference/ref_driver.py`, in this
engine's own environment; this project's side is a real simulation's `spawn` reply.

    RSIM_REF_DIR=<ref> simulator/molmospaces/.venv/bin/python -m pytest \\
        simulator/tests/integration/test_reference_parity.py
"""

import json
import math
import os
import subprocess

import numpy as np
import pytest

from conftest import SIM, TESTS, engine_ready, running_sim

REF = os.environ.get("RSIM_REF_DIR")
ENGINES = {"molmospaces": ("ithor:1", 9285), "robocasa": ("robocasa:1-1", 9286)}

pytestmark = pytest.mark.skipif(not REF, reason="RSIM_REF_DIR (the reference checkout) is not set")


def reference(engine: str) -> dict:
    cmd = (f'source "{SIM / engine / "env.sh"}" >/dev/null; '
           f'exec "{SIM / engine / ".venv" / "bin" / "python"}" '
           f'"{TESTS / "reference" / "ref_driver.py"}" {engine} --robot so101')
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, text=True, timeout=900,
                         env=dict(os.environ, RSIM_REF_DIR=REF))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def angle(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


@pytest.mark.parametrize("engine", list(ENGINES))
def test_so101_stands_where_the_reference_stood_it(engine, logdir):
    if not engine_ready(engine):
        pytest.skip(f"{engine} is not set up")
    ref = reference(engine)
    scene, port = ENGINES[engine]
    with running_sim(engine, scene, port, logdir) as sim:
        c = sim.client()
        res = c.call("spawn", robot="so101", placement="worktop", ports=[])
        c.call("commit")
        try:
            report = {"engine": engine, "ref": ref, "ours": res}
            (logdir / f"parity-{engine}.json").write_text(json.dumps(report, indent=1))
            # the spot: xy and surface height within 1 um, heading within 1e-9 rad
            assert np.max(np.abs(np.subtract(res["xyz"][:2], ref["xy"]))) <= 1e-6, (res["xyz"], ref["xy"])
            assert abs(res["surface_z"] - ref["z"]) <= 1e-6, (res["surface_z"], ref["z"])
            assert angle(res["yaw"], ref["yaw"]) <= 1e-9, (res["yaw"], ref["yaw"])
            # the base: within 3 mm of the reference's graft height
            assert abs(res["xyz"][2] - ref["mount_z"]) <= 0.003, (res["xyz"][2], ref["mount_z"])
            # the six objects, as staged, within 1 um (and the same orientation)
            assert set(res["staged"]) == set(ref["objects"])
            for name, o in ref["objects"].items():
                mine = res["staged"][name]
                assert np.max(np.abs(np.subtract(mine["pos"], o["pos"]))) <= 1e-6, name
                q, r = np.asarray(mine["quat"]), np.asarray(o["quat"])
                assert min(np.max(np.abs(q - r)), np.max(np.abs(q + r))) <= 1e-6, name
            # the same scene objects cleared from the working area
            assert sorted(res["cleared"]) == sorted(ref["cleared"])
        finally:
            c.call("remove")
            c.close()
