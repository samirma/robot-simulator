#!/usr/bin/env python3
"""Headless smoke check: build a RoboCasa kitchen for each robot and save a screenshot.

This verifies the engine actually composes and renders each shared robot in a kitchen,
without needing a live robot_console connection. For every robot it builds the world
(the same `build_world` the wire surface uses), renders one offscreen frame from a free
camera framed on the robot, and writes a PNG. Run headless:

    ./run.sh view ...            # normal use
    .venv/bin/python tools/screenshot.py --scene kitchen:1 --out /tmp/shots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spawn_robot import ROBOTS, PREFIX, build_world  # noqa: E402


def _robot_xy(model, data, robot: str):
    """World (x, y, z) of the robot's root body, to aim the camera at."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            PREFIX + ROBOTS[robot]["root"])
    if bid < 0:
        return np.array([0.0, 0.0, 0.5])
    return np.array(data.xpos[bid], dtype=float)


def shoot(robot: str, scene: str, seed: int, size, out: Path) -> Path:
    model, data = build_world(scene, robot, seed)
    mujoco.mj_forward(model, data)

    width, height = size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    target = _robot_xy(model, data, robot)
    # A floor base (myagv) and a counter-height arm (so101) need different framing so the
    # robot is not hidden behind a counter.
    if robot == "myagv":
        target[2] = 0.25
        cam.distance = 2.4
        cam.azimuth = 90.0
        cam.elevation = -35.0
    else:
        target[2] = 0.95
        cam.distance = 1.6
        cam.azimuth = 135.0
        cam.elevation = -15.0
    cam.lookat[:] = target
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()
    renderer.close()

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"robocasa_{robot}.png"
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, frame)
    except Exception:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"[{robot}] {scene}: {model.nbody} bodies -> {path}", file=sys.stderr)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", choices=sorted(ROBOTS), default=None,
                    help="one robot; default is every robot")
    ap.add_argument("--scene", default="kitchen:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/robocasa_shots", type=Path)
    ap.add_argument("--size", type=int, nargs=2, default=[960, 720])
    args = ap.parse_args()

    robots = [args.robot] if args.robot else sorted(ROBOTS)
    for r in robots:
        shoot(r, args.scene, args.seed, args.size, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
