#!/usr/bin/env python
"""Standalone check that LeKiwi attaches, drives and moves its arm.

    python robots/lekiwi/test_attach.py [--scene house.xml] [--render out.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FAIL = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(label)


def settle(model, data, steps=3000):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--render", default=None)
    args = ap.parse_args()

    from robots.lekiwi import LeKiwiRobot, LeKiwiRobotConfig, LeKiwiRobotView

    config = LeKiwiRobotConfig()

    spec = mujoco.MjSpec.from_file(args.scene) if args.scene else mujoco.MjSpec()
    if args.scene is None:
        spec.worldbody.add_light(
            pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.55, 0.56, 0.58, 1]
        )

    LeKiwiRobot.add_robot_to_scene(
        config, spec, prefix=config.robot_namespace, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    ns = config.robot_namespace
    view = LeKiwiRobotView(data, ns)
    print(f"compiled: {model.nbody} bodies, {model.nu} actuators")

    check("move groups", view.move_group_ids() == ["base", "arm", "gripper"],
          str(view.move_group_ids()))
    check("front camera", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{ns}front") >= 0)

    base = view.get_move_group("base")
    arm = view.get_move_group("arm")
    gripper = view.get_move_group("gripper")

    origin = np.zeros(2)
    heading = np.array([1.0, 0.0])
    reach, strafe_reach = 1.0, 0.5
    if args.scene:
        from tools.spawn_robot import find_open_spot

        origin, yaw = find_open_spot(args.scene)
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        reach, strafe_reach = 0.5, 0.25
        pose = np.eye(4)
        pose[:2, 3] = origin
        base.pose = pose
        print(f"starting from open floor at {np.round(origin, 3)}")

    for group, qpos in config.init_qpos.items():
        # Skip "base" entirely: its init_qpos is the origin, and applying it here would
        # undo the placement above and leave the robot inside the furniture at (0, 0).
        if group != "base" and group in view.move_group_ids() and len(qpos):
            mg = view.get_move_group(group)
            mg.joint_pos = qpos
            mg.ctrl = qpos
    base.ctrl = np.array([origin[0], origin[1], 0.0])
    mujoco.mj_forward(model, data)

    left = np.array([-heading[1], heading[0]])
    fwd = origin + heading * reach
    strafed = fwd + left * strafe_reach

    print("\ndriving:")
    base.ctrl = np.array([fwd[0], fwd[1], 0.0])
    settle(model, data, 4000)
    reached = np.asarray(base.joint_pos)
    check(f"drive {reach:.2f} m", np.linalg.norm(reached[:2] - fwd) < 0.03,
          f"reached {np.round(reached[:2], 3)} wanted {np.round(fwd, 3)}")

    base.ctrl = np.array([strafed[0], strafed[1], 0.0])
    settle(model, data, 4000)
    reached = np.asarray(base.joint_pos)
    check(f"strafe {strafe_reach:.2f} m", np.linalg.norm(reached[:2] - strafed) < 0.03,
          f"reached {np.round(reached[:2], 3)} wanted {np.round(strafed, 3)}")
    check("heading held during strafe", abs(reached[2]) < np.radians(2),
          f"{np.degrees(reached[2]):.2f} deg")

    base.ctrl = np.array([strafed[0], strafed[1], np.pi / 2])
    settle(model, data, 4000)
    reached = np.asarray(base.joint_pos)
    check("rotate to 90 deg", abs(reached[2] - np.pi / 2) < np.radians(2),
          f"{np.degrees(reached[2]):.2f} deg")

    print("\narm:")
    target = np.asarray(config.init_qpos["arm"], dtype=float).copy()
    target[0] += 0.5
    target[1] += 0.3
    arm.ctrl = target
    settle(model, data, 3000)
    err = float(np.max(np.abs(np.asarray(arm.joint_pos) - target)))
    check("arm tracks a command", err < 0.08, f"max joint error {err:.4f} rad")

    gripper.set_gripper_ctrl_open(False)
    settle(model, data, 2000)
    closed = gripper.inter_finger_dist
    gripper.set_gripper_ctrl_open(True)
    settle(model, data, 2000)
    opened = gripper.inter_finger_dist
    check("gripper opens wider than it closes", opened > closed + 0.01,
          f"closed {closed:.4f} m, open {opened:.4f} m")

    if args.render:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 900)
        for gid in range(model.ngeom):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            if "ceiling" in name:
                model.geom_rgba[gid, 3] = 0.0
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = data.body(f"{ns}{LeKiwiRobot.robot_model_root_name()}").xpos + np.array(
            [0, 0, 0.15]
        )
        cam.distance, cam.azimuth, cam.elevation = 1.4, 135, -18
        with mujoco.Renderer(model, 900, 1280) as r:
            r.update_scene(data, camera=cam)
            from PIL import Image

            Image.fromarray(r.render()).save(args.render)
        print(f"\nwrote {args.render}")

    print("\nRESULT:", "ok" if not FAIL else f"{len(FAIL)} check(s) failed: {FAIL}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
