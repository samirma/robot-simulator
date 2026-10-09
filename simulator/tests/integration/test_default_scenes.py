"""Each engine's default scene loads as the reference project loaded it (model counts
recorded from the reference loader, plus the six worktop objects the scene holds, staged
with no robot at the reference's spot), with no robot and the reference survey's worktop;
each worktop robot stands at its spot -- among the scene's objects where it fits there --
with the six objects around it. (`test_reference_parity.py` compares the scene with the
reference's own, body by body and render by render, when a reference checkout is at
hand.)"""

import math

import numpy as np
import pytest

import registry
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

#: Golden staging spots of the scene's six objects (x, y, surface z, yaw) and the scene
#: objects cleared for them: the reference project's own arm spots
#: (integration/test_reference_parity.py checks them against it).
SCENE_SPOTS = {
    "molmospaces": ((-0.3819999726666255, 0.2180100565565969, 1.1000100382472997, 0.0),
                    ["apple", "bread", "book"]),
    "robocasa": ((2.231088192318004, -0.2, 0.92, -math.pi / 2), None),
}
#: Golden spots of each worktop robot (x, y, surface z, yaw), recorded 2026-10-09, and
#: whether it stands at the scene's objects. On robocasa:1-1 the reBot's folded home pose
#: reaches into the wall behind the scene's spot, so it stands at the first of the
#: reference survey's spots where it fits: with support and clearance, and with its D405
#: wrist camera clear of the furniture within 0.8 m (before the camera was added the arm
#: stood at (2.2522, -0.41, yaw pi/2), which the camera clearance no longer admits).
SPOTS = {
    ("molmospaces", "rebot_b601_rs"): ((-0.3819999726666255, 0.2180100565565969,
                                        1.1000100382472997, 0.0), True),
    ("robocasa", "rebot_b601_rs"): ((2.202176384636008, -0.26, 0.92, -math.pi / 3), False),
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
        # staged with no robot, at the scene's staging spot
        (x, y, sz, yaw), cleared = SCENE_SPOTS[engine]
        frame = s["staging"]["scene"]["frame"]
        assert np.max(np.abs(np.subtract(frame["xy"], [x, y]))) < 1e-6, frame
        assert abs(frame["z"] - sz) < 1e-6 and abs(frame["yaw"] - yaw) < 1e-9, frame
        got = s["staging"]["scene"]["cleared"]
        if cleared is None:
            # on RoboCasa only RoboCasa's own objects stand loose in the area
            assert got and all(n.startswith("rc_obj_") for n in got), got
        else:
            assert sorted(n.split("_")[0] for n in got) == sorted(cleared)
        name, z = WORKTOP[(engine, scene)]
        assert h["worktop"]["name"] == name and abs(h["worktop"]["z"] - z) < 1e-4
        png = c.call("render", view={"lookat": h["worktop"]["centroid"] + [h["worktop"]["z"]],
                                     "distance": 2.5, "azimuth": 45, "elevation": -35},
                     width=640, height=360)["_payload"]
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        (tmp_path / f"{engine}.png").write_bytes(png)
        # each worktop robot at its golden spot with the six objects around it; removing
        # it leaves the scene as it was loaded
        for rid in sorted(r.id for r in registry.robots() if r.kind == "arm"):
            (x, y, sz, yaw), at_scene = SPOTS[(engine, rid)]
            res = c.call("spawn", robot=rid, placement="worktop", ports=[])
            c.call("commit")
            try:
                assert abs(res["xyz"][0] - x) < 1e-6 and abs(res["xyz"][1] - y) < 1e-6, (rid, res["xyz"])
                assert abs(res["surface_z"] - sz) < 1e-6
                assert abs(math.atan2(math.sin(res["yaw"] - yaw), math.cos(res["yaw"] - yaw))) < 1e-9
                if at_scene:
                    # among the scene's own objects: their staging, untouched
                    assert res["frame"] == frame and res["cleared"] == s["staging"]["scene"]["cleared"]
                else:
                    # the scene's own staging already cleared its area; an arm standing
                    # elsewhere clears only what that leaves -- on RoboCasa only its own
                    # objects can stand in the area (the kitchen's fixtures are not loose)
                    assert res["frame"] != frame
                    assert all(n.startswith("rc_obj_") for n in res["cleared"]), res["cleared"]
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
