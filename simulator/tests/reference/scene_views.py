"""What the scene-parity check (`integration/test_reference_parity.py`) reads off a compiled
scene, and the one way both sides draw it: the reference's scene (`ref_driver.py`) and the
one `start` builds (`our_scene.py`) go through these same functions, in the same engine
venv, as compiled -- before any physics step -- so a difference in a render is a
difference in the scene.

Stdlib, NumPy and MuJoCo only: `ref_driver.py` imports it with this project's own modules
off the import path.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

WIDTH, HEIGHT = 640, 480
#: the geom groups each engine's scenes show (RoboCasa hides its collision hulls, group 0)
GEOMGROUP = {"molmospaces": (1, 1, 1, 0, 0, 0), "robocasa": (0, 1, 1, 0, 0, 0)}
#: free-camera viewpoints (lookat, distance, azimuth, elevation) on each default scene:
#: the worktop from two sides and from above
VIEWS = {
    "molmospaces": [((-0.38, 0.22, 1.10), 1.2, 30.0, -35.0),
                    ((-0.38, 0.22, 1.10), 2.5, 210.0, -25.0),
                    ((-0.06, 0.0, 1.10), 1.3, 0.0, -65.0)],
    "robocasa": [((2.23, -0.20, 0.92), 1.2, 60.0, -35.0),
                 ((2.23, -0.20, 0.92), 2.5, 240.0, -25.0),
                 ((2.23, -0.33, 0.92), 1.3, 90.0, -65.0)],
}
#: the six worktop objects' bodies (worktop_objects.BODIES; the reference's names too)
OBJECTS = ("apple", "plate", "bowl", "mug", "banana", "lemon")


def _names(model, kind, count) -> list:
    return sorted(mujoco.mj_id2name(model, kind, i) or "" for i in range(count))


def describe(model, data) -> dict:
    """Counts, the sorted body, geom and mesh names, every named body's position, and the
    six objects' poses."""
    objects = {}
    for name in OBJECTS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"task_{name}")
        if b >= 0:
            objects[name] = {"pos": data.xpos[b].tolist(), "quat": data.xquat[b].tolist()}
    return {"counts": {"nbody": model.nbody, "ngeom": model.ngeom, "nmesh": model.nmesh,
                       "nlight": model.nlight},
            "bodies": _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
            "geoms": _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
            "meshes": _names(model, mujoco.mjtObj.mjOBJ_MESH, model.nmesh),
            "positions": {model.body(i).name: data.xpos[i].tolist() for i in range(model.nbody)
                          if model.body(i).name},
            "objects": objects}


def render(model, data, engine: str, out: Path) -> list:
    """Each of the engine's `VIEWS` as raw RGB, saved as `out/view<i>.npy`; the paths."""
    out.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    opt = mujoco.MjvOption()
    for i, g in enumerate(GEOMGROUP[engine]):
        opt.geomgroup[i] = g
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    paths = []
    try:
        for i, (lookat, distance, azimuth, elevation) in enumerate(VIEWS[engine]):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = lookat
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
            renderer.update_scene(data, camera=cam, scene_option=opt)
            path = out / f"view{i}.npy"
            np.save(path, renderer.render())
            paths.append(str(path))
    finally:
        renderer.close()
    return paths
