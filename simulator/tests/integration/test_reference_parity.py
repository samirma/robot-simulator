"""Reference parity (spec §5 *Reference parity*): on each engine's default scene the six
worktop objects are staged where the reference project stood its arm, at the reference's
poses, with the same scene objects cleared; and the scene `start` loads, with
no robot, is the reference's default scene with its robot left out -- on MolmoSpaces the
same bodies, geoms and meshes, the six objects at the same poses and pixel-identical
renders; on RoboCasa the reference kitchen's fixtures -- every one where the reference has
it but those RoboCasa's own fixture placement sets on a counter, which the reference left
where the layout put them -- and six objects, plus RoboCasa's own kitchen objects, which
the reference did not load.

The reference is run from a checkout of github.com/samirma/robot-simulator rev 34547ae
-- the one named by `RSIM_REF_DIR`, or else extracted from this repository's own history
(`git archive 34547ae simulator robots_specs`, the reference being this workspace's prior
version), so the check runs wherever that revision is at hand and is skipped only where
it is not -- through `reference/ref_driver.py`, in this engine's own environment, for the
reference's own arm. This project's side is `reference/our_scene.py`: `Simulation.load()`
as `start` runs it, before any physics step, so both scenes are compared as compiled and
drawn by the same code (`reference/scene_views.py`).

    simulator/molmospaces/.venv/bin/python -m pytest \\
        simulator/tests/integration/test_reference_parity.py     # [RSIM_REF_DIR=<ref>]
"""

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO, SIM, TESTS, engine_ready

#: the reference project's revision (spec Terms)
REF_REV = "34547ae"
ENGINES = {"molmospaces": ("ithor:1", 9285), "robocasa": ("robocasa:1-1", 9286)}
#: RoboCasa's own kitchen objects (spec §2.1), which the reference did not load
RC = "rc_obj_"


@pytest.fixture(scope="module")
def ref_dir(tmp_path_factory) -> str:
    """A checkout of the reference: `RSIM_REF_DIR`, else its revision's `simulator/` and
    `robots_specs/` extracted from this repository's history."""
    if os.environ.get("RSIM_REF_DIR"):
        return os.environ["RSIM_REF_DIR"]
    git = ["git", "-C", str(REPO)]
    if subprocess.run(git + ["cat-file", "-e", f"{REF_REV}^{{commit}}"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        pytest.skip(f"the reference ({REF_REV}) is not in this repository's history and "
                    "RSIM_REF_DIR is not set")
    dest = tmp_path_factory.mktemp(f"reference-{REF_REV}")
    archive = subprocess.Popen(git + ["archive", REF_REV, "simulator", "robots_specs"],
                               stdout=subprocess.PIPE)
    untar = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout)
    archive.stdout.close()
    assert archive.wait() == 0 and untar.returncode == 0, "extracting the reference failed"
    assert (Path(dest) / "simulator" / "shared" / "tasks" / "apple_on_plate.py").is_file()
    return str(dest)


def driver(engine: str, script: str, *args, ref: str) -> dict:
    """The JSON a `reference/` driver prints, run in the engine's environment."""
    cmd = (f'source "{SIM / engine / "env.sh"}" >/dev/null; '
           f'exec "{SIM / engine / ".venv" / "bin" / "python"}" '
           f'"{TESTS / "reference" / script}" ' + " ".join(f'"{a}"' for a in args))
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, text=True, timeout=900,
                         env=dict(os.environ, RSIM_REF_DIR=ref))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def reference(engine: str, *args, ref: str) -> dict:
    """The reference's own choice for its own arm (`ref_driver.py`'s default robot)."""
    return driver(engine, "ref_driver.py", engine, *args, ref=ref)


def angle(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def same_pose(mine: dict, ref: dict) -> bool:
    q, r = np.asarray(mine["quat"]), np.asarray(ref["quat"])
    return (np.max(np.abs(np.subtract(mine["pos"], ref["pos"]))) <= 1e-6
            and min(np.max(np.abs(q - r)), np.max(np.abs(q + r))) <= 1e-6)


def the_references(names) -> list:
    """The names the reference's scene has too: all but RoboCasa's own objects'."""
    return sorted(n for n in names if not n.startswith(RC))


@pytest.mark.parametrize("engine", list(ENGINES))
def test_the_staging_spot_is_where_the_reference_stood_its_arm(engine, logdir, ref_dir):
    if not engine_ready(engine):
        pytest.skip(f"{engine} is not set up")
    ref = reference(engine, ref=ref_dir)
    ours = driver(engine, "our_scene.py", engine, "--scene", ENGINES[engine][0], ref=ref_dir)
    frame = ours["frame"]
    report = {"engine": engine, "ref": ref, "ours": {k: ours[k] for k in
                                                     ("frame", "objects", "cleared")}}
    (logdir / f"parity-{engine}.json").write_text(json.dumps(report, indent=1))
    # the spot: xy and surface height within 1 um, heading within 1e-9 rad
    assert np.max(np.abs(np.subtract(frame["xy"], ref["xy"]))) <= 1e-6, (frame, ref["xy"])
    assert abs(frame["z"] - ref["z"]) <= 1e-6, (frame["z"], ref["z"])
    assert angle(frame["yaw"], ref["yaw"]) <= 1e-9, (frame["yaw"], ref["yaw"])
    # the six objects, as staged, within 1 um (and the same orientation)
    assert set(ours["objects"]) == set(ref["objects"])
    for name, o in ref["objects"].items():
        assert same_pose(ours["objects"][name], o), name
    # the same scene objects cleared from the working area -- of the reference's scene:
    # on RoboCasa, RoboCasa's own objects there are cleared as well
    assert the_references(ours["cleared"]) == sorted(ref["cleared"])


@pytest.mark.parametrize("engine", list(ENGINES))
def test_start_scene_is_the_reference_scene_without_its_robot(engine, tmp_path, ref_dir):
    if not engine_ready(engine):
        pytest.skip(f"{engine} is not set up")
    full = reference(engine, "--views", tmp_path / "ref", ref=ref_dir)
    ref, ref_cleared = full["scene"], full["cleared"]
    ours = driver(engine, "our_scene.py", engine, "--scene", ENGINES[engine][0],
                  "--views", tmp_path / "ours", ref=ref_dir)
    # the same bodies, geoms and meshes (on RoboCasa: the reference kitchen's fixtures),
    # every body where the reference has it
    for kind in ("bodies", "geoms", "meshes"):
        mine = ours[kind] if engine == "molmospaces" else the_references(ours[kind])
        assert mine == ref[kind], (kind, sorted(set(mine) ^ set(ref[kind]))[:20])
    moved = [n for n, pos in ref["positions"].items()
             if np.max(np.abs(np.subtract(ours["positions"][n], pos))) > 1e-6]
    if engine == "robocasa":
        # except the fixtures RoboCasa's own fixture placement sets on a counter (a toaster,
        # a knife block...), which the reference left where the layout put them (on 1-1, at
        # the world origin): each stands off the floor now (§2.1)
        placed = ours["placed_fixtures"]
        assert placed, "RoboCasa placed no fixture"
        moved = [n for n in moved if not any(n.startswith(f + "_") for f in placed)]
        for f in placed:
            assert ours["positions"][f + "_main"][2] > 0.5, (f, ours["positions"][f + "_main"])
    assert moved == [], moved[:20]
    # the six objects at the reference's poses, and the same scene objects cleared
    assert set(ours["objects"]) == set(ref["objects"]) and len(ref["objects"]) == 6
    for name, o in ref["objects"].items():
        assert same_pose(ours["objects"][name], o), (name, ours["objects"][name], o)
    assert the_references(ours["cleared"]) == sorted(ref_cleared)
    if engine == "robocasa":
        # plus RoboCasa's own kitchen objects, which the reference did not load
        assert any(n.startswith(RC) for n in ours["bodies"])
        return
    assert ours["counts"] == ref["counts"]
    # pixel-identical renders from the same viewpoints
    assert len(ours["views"]) == len(ref["views"]) > 0
    for i, (a, b) in enumerate(zip(ours["views"], ref["views"])):
        mine, theirs = np.load(a), np.load(b)
        assert mine.shape == theirs.shape and mine.std() > 1.0, i
        differ = np.count_nonzero(np.any(mine != theirs, axis=-1))
        assert differ == 0, f"view {i}: {differ} pixels differ"
