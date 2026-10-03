"""Each engine's default scene loads as the reference project loaded it (model counts
recorded from the reference loader, plus the six worktop objects the scene holds), with no
robot and the reference survey's worktop; both worktop robots stand at the scene's spot,
among its six objects. (`test_reference_parity.py` compares the scene with the reference's
own, body by body and render by render, when a reference checkout is at hand.)"""

import math

import numpy as np
import pytest

from conftest import engine_ready, running_sim

#: Counts of the scene built by the reference project's own loader (MolmoSpaces
#: `MjSpec.from_file` of the resolved FloorPlan1; RoboCasa `KitchenArena(1, 1)` in a bare
#: `ManipulationTask`), measured when this project's loader was checked against it.
REFERENCE = {
    ("molmospaces", "ithor:1"): {"nbody": 242, "ngeom": 2116, "nmesh": 1493, "nlight": 1,
                                 "nq": 289, "ncam": 0, "nu": 0},
    ("robocasa", "robocasa:1-1"): {"nbody": 150, "ngeom": 825, "nmesh": 102, "nlight": 1,
                                   "nq": 46, "ncam": 0, "nu": 0},
}

#: The worktop (spec §2.2): FloorPlan1's island; RoboCasa layout 1's counter run.
WORKTOP = {
    ("molmospaces", "ithor:1"): ("standardislandheight_7cc63329b1f0a38cd8c2450298404ab3_1_0_0",
                                 1.1000100382472997),
    ("robocasa", "robocasa:1-1"): ("counter_main_main_group/geom_1", 0.92),
}

#: Golden spots (x, y, surface z, yaw) and cleared objects, recorded 2026-10-01. so101's
#: are the reference project's own (integration/test_reference_parity.py checks them
#: against it); the myCobot 280 stands at the same spot on both engines, where the scene's
#: worktop objects are: in its natural ready pose it fits there (upright, all zeros, it
#: ran into the wall cabinet on RoboCasa layout 1).
SPOTS = {
    ("molmospaces", "so101"): ((-0.3819999726666255, 0.2180100565565969, 1.1000100382472997, 0.0),
                               ["apple", "bread", "book"]),
    # fits where the scene's objects are, so stands there (its own survey spot, at
    # pi/6, is only for an arm that does not)
    ("molmospaces", "mycobot280"): ((-0.3819999726666255, 0.2180100565565969, 1.1000100382472997,
                                     0.0), ["apple", "bread", "book"]),
    ("robocasa", "so101"): ((2.231088192318004, -0.2, 0.92, -math.pi / 2), []),
    ("robocasa", "mycobot280"): ((2.231088192318004, -0.2, 0.92, -math.pi / 2), []),
}
#: The objects in the robot's base frame (worktop_objects.OBJECT_POSES), frame at the
#: surface + 4 mm.
OBJECTS = {"apple": (0.30, 0.10, 0.020), "plate": (0.226, -0.226, 0.0),
           "bowl": (0.14, -0.40, 0.0271), "mug": (0.05, 0.27, 0.0272),
           "banana": (0.156, 0.156, 0.0172), "lemon": (0.07, -0.26, 0.0294)}

#: What the six worktop objects add to a scene: bodies, geoms (apple 2, plate 2 + 24 rim
#: boxes, four distractors 2 each), meshes, and the free joints' qpos (apple + 4 x 7).
OBJECT_COUNTS = {"nbody": 6, "ngeom": 36, "nmesh": 6, "nq": 35}

PORTS = {"molmospaces": 9282, "robocasa": 9283}


@pytest.mark.parametrize("engine,scene", list(REFERENCE))
def test_default_scene_matches_reference_and_has_a_worktop(engine, scene, logdir, tmp_path):
    if not engine_ready(engine):
        pytest.skip(f"{engine} is not set up")
    with running_sim(engine, scene, PORTS[engine], logdir) as sim:
        c = sim.client()
        h = c.call("hello")
        assert h["scene"] == scene and h["robots"] == []
        s = c.call("scene")
        # the reference loader's scene plus the six worktop objects the scene itself holds
        # (a scene object cleared for them loses its free joint: 7 qpos each)
        n_cleared = len(s["staging"]["scene"]["cleared"])
        counts = dict(OBJECT_COUNTS, nq=OBJECT_COUNTS["nq"] - 7 * n_cleared)
        for k, v in REFERENCE[(engine, scene)].items():
            if engine == "robocasa" and k in ("nbody", "ngeom", "nmesh", "nq"):
                # RoboCasa's own kitchen objects stand on the counters as well (they
                # come from its asset library, so only a lower bound is fixed)
                assert s[k] > v + counts.get(k, 0), (k, s[k], v)
            else:
                assert s[k] == v + counts.get(k, 0), (k, s[k], v)
        if engine == "robocasa":
            assert any(n.startswith("rc_obj_") for n in s["free_bodies"]), s["free_bodies"]
        assert sorted(s["staging"]["scene"]["staged"]) == sorted(OBJECTS)
        name, z = WORKTOP[(engine, scene)]
        assert h["worktop"]["name"] == name and abs(h["worktop"]["z"] - z) < 1e-4
        png = c.call("render", view={"lookat": h["worktop"]["centroid"] + [h["worktop"]["z"]],
                                     "distance": 2.5, "azimuth": 45, "elevation": -35},
                     width=640, height=360)["_payload"]
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        (tmp_path / f"{engine}.png").write_bytes(png)
        # each worktop robot at its golden spot with the six objects around it; removing
        # it leaves the scene as it was loaded
        for rid in ("so101", "mycobot280"):
            (x, y, sz, yaw), cleared = SPOTS[(engine, rid)]
            res = c.call("spawn", robot=rid, placement="worktop", ports=[])
            c.call("commit")
            try:
                assert abs(res["xyz"][0] - x) < 1e-6 and abs(res["xyz"][1] - y) < 1e-6, (rid, res["xyz"])
                assert abs(res["surface_z"] - sz) < 1e-6
                assert abs(math.atan2(math.sin(res["yaw"] - yaw), math.cos(res["yaw"] - yaw))) < 1e-9
                got = sorted(n.split("_")[0] for n in res["cleared"])
                # the scene's own staging already cleared its area; an arm standing
                # elsewhere clears only what that leaves
                if engine == "robocasa":
                    # only RoboCasa's own objects can stand in the area (the kitchen's
                    # fixtures are not loose); which ones follows from the layout's draw
                    assert all(n.startswith("rc_obj_") for n in res["cleared"]), res["cleared"]
                else:
                    assert got == sorted(cleared) if rid == "so101" else set(got) <= set(cleared)
                c_, s_ = math.cos(yaw), math.sin(yaw)
                for obj, (ox, oy, oz) in OBJECTS.items():
                    want = [x + c_ * ox - s_ * oy, y + s_ * ox + c_ * oy, sz + 0.004 + oz]
                    assert np.max(np.abs(np.subtract(res["staged"][obj]["pos"], want))) < 1e-6, obj
            finally:
                c.call("remove")
            after = c.call("scene")
            for k in REFERENCE[(engine, scene)]:
                assert after[k] == s[k], (rid, k, after[k], s[k])
        c.close()
