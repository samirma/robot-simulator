#!/usr/bin/env python
"""Run the reference project's own worktop placement and task staging on one engine's
default scene, and print what it chose as JSON (for `integration/test_reference_parity.py`).

    RSIM_REF_DIR=<checkout of github.com/samirma/robot-simulator rev 34547ae> \\
        <engine venv python> ref_driver.py molmospaces|robocasa [--robot so101] [--views DIR]

Run it in the engine's environment (`simulator/<engine>/env.sh`). Only the reference's
code runs here -- `tools/spawn_robot.py`'s `find_worktop` (MolmoSpaces:
`place_arm_on_table`; RoboCasa: `find_counter_mount`), its `placement.BASE_CLEARANCE` and
`tasks/apple_on_plate.stage` -- on the scene the reference's own loader builds; this
project's modules are taken off the import path first, since both have a `placement`.

Output: {"surface", "xy", "z", "yaw", "mount_z", "cleared": [body names],
"objects": {name: {"pos", "quat"}}} -- the objects' poses as compiled, before any step.
With `--views DIR`, also "scene": the staged scene (the reference's default scene with its
robot left out, its apple-on-plate staging included) as `scene_views.describe` reads it,
and its renders from `scene_views.VIEWS`, saved in DIR.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve()
SIM = HERE.parents[2]                     # simulator/
sys.path.insert(0, str(HERE.parent))

import scene_views  # noqa: E402  (before the reference's modules take the import path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=("molmospaces", "robocasa"))
    ap.add_argument("--robot", default="so101")
    ap.add_argument("--views", type=Path, default=None)
    args = ap.parse_args()
    ref = Path(os.environ["RSIM_REF_DIR"]).resolve()
    ours = {str(SIM / "shared"), str(SIM / args.engine)}
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) not in ours]
    sys.path[:0] = [str(ref / "simulator" / args.engine), str(ref / "simulator" / "shared")]
    sys.argv = [sys.argv[0]]

    import mujoco

    with contextlib.redirect_stdout(sys.stderr):
        sr = importlib.import_module("tools.spawn_robot")
        refpl = importlib.import_module("placement")
        task = importlib.import_module("tasks.apple_on_plate")
        if args.engine == "molmospaces":
            # the scene file the reference's resolve_scene.py installs for ithor:1
            out = subprocess.run([sys.executable, str(SIM / "molmospaces" / "tools" / "resolve_scene.py"),
                                  "ithor", "1"], stdout=subprocess.PIPE, text=True, check=True)
            xml = out.stdout.strip().splitlines()[-1]
            scene = sr.MolmoSpacesEngine.load_scene(types.SimpleNamespace(scene=xml))
            wt = sr.MolmoSpacesEngine.find_worktop(scene, args.robot)
        else:
            import robocasa.models

            robocasa.models.assets_root = os.environ["ROBOCASA_ASSETS_DIR"]
            scene = sr.RoboCasaEngine.load_scene(types.SimpleNamespace(layout=1, style=1, seed=0))
            wt = sr.RoboCasaEngine.find_worktop(scene, args.robot)
        mount_z = float(wt.z) + refpl.BASE_CLEARANCE.get(args.robot, 0.0)
        cleared = task.stage(scene.spec, [float(wt.xy[0]), float(wt.xy[1]), mount_z], float(wt.yaw))
        m = scene.spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        extra = {}
        if args.views is not None:
            extra["scene"] = scene_views.describe(m, d)
            extra["scene"]["views"] = scene_views.render(m, d, args.engine, args.views)
    objects = {}
    for name in task.TASK_OBJECTS:
        b = m.body(f"task_{name}").id
        objects[name] = {"pos": d.xpos[b].tolist(), "quat": d.xquat[b].tolist()}
    print(json.dumps({"surface": wt.name, "xy": [float(v) for v in wt.xy], "z": float(wt.z),
                      "yaw": float(wt.yaw), "mount_z": mount_z, "cleared": list(cleared),
                      "objects": objects, **extra}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
