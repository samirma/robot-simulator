#!/usr/bin/env python
"""Standalone check that the SO-101 attaches to a scene and its move groups work.

Runs without the datagen pipeline, so failures point at the robot definition rather
than at task sampling.

    python robots/so101/test_attach.py [--scene /path/to/house.xml] [--render out.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="house MJCF to attach into (default: empty world)")
    ap.add_argument("--render", default=None, help="write a PNG of the result")
    ap.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--yaw", type=float, default=0.0, help="facing, in degrees")
    args = ap.parse_args()

    from robots.so101 import SO101Robot, SO101RobotConfig, SO101RobotView

    config = SO101RobotConfig()

    spec = mujoco.MjSpec.from_file(args.scene) if args.scene else mujoco.MjSpec()
    yaw = np.radians(args.yaw)
    SO101Robot.add_robot_to_scene(
        config,
        spec,
        prefix=config.robot_namespace,
        pos=list(args.pos),
        quat=[float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"compiled: {model.nbody} bodies, {model.ngeom} geoms, {model.nu} actuators")

    view = SO101RobotView(data, config.robot_namespace)
    print(f"move groups: {view.move_group_ids()}")

    arm = view.get_move_group("arm")
    gripper = view.get_move_group("gripper")

    arm.joint_pos = config.init_qpos["arm"]
    gripper.joint_pos = config.init_qpos["gripper"]
    mujoco.mj_forward(model, data)
    print(f"arm qpos       : {np.round(arm.joint_pos, 4)}")
    print(f"gripper qpos   : {np.round(gripper.joint_pos, 4)}")
    print(f"TCP pose       : {np.round(arm.leaf_frame_to_world[:3, 3], 4)}")
    print(f"inter-finger   : {gripper.inter_finger_dist:.4f} m (open={gripper.is_open})")

    gripper.set_gripper_ctrl_open(False)
    for _ in range(400):
        mujoco.mj_step(model, data)
    print(f"after close    : {gripper.inter_finger_dist:.4f} m (open={gripper.is_open})")

    gripper.set_gripper_ctrl_open(True)
    for _ in range(400):
        mujoco.mj_step(model, data)
    print(f"after open     : {gripper.inter_finger_dist:.4f} m (open={gripper.is_open})")

    # Drive the arm to a new joint target and confirm the controllers track it.
    target = [0.6, -0.4, 0.8, 0.3, 0.2]
    arm.ctrl = target
    for _ in range(1500):
        mujoco.mj_step(model, data)
    reached = np.asarray(arm.joint_pos)
    err = np.abs(reached - np.asarray(target))
    print(f"commanded arm  : {np.round(target, 3)}")
    print(f"reached arm    : {np.round(reached, 3)}  max err {err.max():.4f} rad")

    jac = arm.get_jacobian()
    print(f"jacobian shape : {jac.shape}, rank {np.linalg.matrix_rank(jac)}")

    if args.render:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 800)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = data.site(f"{config.robot_namespace}tcp").xpos
        cam.distance = 1.4
        cam.azimuth = 140
        cam.elevation = -20
        with mujoco.Renderer(model, 800, 1280) as r:
            r.update_scene(data, camera=cam)
            from PIL import Image

            Image.fromarray(r.render()).save(args.render)
        print(f"wrote {args.render}")

    ok = err.max() < 0.05
    print("\nRESULT:", "ok" if ok else f"arm did not track target (max err {err.max():.4f} rad)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
