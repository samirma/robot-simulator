#!/usr/bin/env python
"""Standalone check that the B601 attaches and is controllable.

    python robots/rebot_b601/test_attach.py [--scene house.xml] [--render out.png]
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
    ap.add_argument("--pos", type=float, nargs=2, default=None)
    args = ap.parse_args()

    from robots.rebot_b601 import B601Robot, B601RobotConfig, B601RobotView

    config = B601RobotConfig()

    spec = mujoco.MjSpec.from_file(args.scene) if args.scene else mujoco.MjSpec()
    pos = [0.0, 0.0, 0.0]
    if args.scene is None:
        spec.worldbody.add_light(
            pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.55, 0.56, 0.58, 1]
        )
    else:
        from tools.spawn_robot import find_open_spot

        spot, _ = find_open_spot(args.scene)
        pos = [float(spot[0]), float(spot[1]), 0.0]
        print(f"placing at {np.round(spot, 3)}")

    B601Robot.add_robot_to_scene(
        config, spec, prefix=config.robot_namespace, pos=pos, quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    ns = config.robot_namespace
    view = B601RobotView(data, ns)
    print(f"compiled: {model.nbody} bodies, {model.nu} actuators")

    check("move groups", view.move_group_ids() == ["base", "arm", "gripper"],
          str(view.move_group_ids()))
    check("6 arm actuators", view.get_move_group("arm").n_actuators == 6)
    check("2 gripper actuators", view.get_move_group("gripper").n_actuators == 2)

    arm = view.get_move_group("arm")
    gripper = view.get_move_group("gripper")

    for group, qpos in config.init_qpos.items():
        if group in view.move_group_ids() and len(qpos):
            mg = view.get_move_group(group)
            mg.joint_pos = qpos
            mg.ctrl = qpos
    mujoco.mj_forward(model, data)

    tcp0 = view.get_move_group("arm").leaf_frame_to_world[:3, 3].copy()
    print(f"\nrest pose TCP at {np.round(tcp0, 4)}")

    settle(model, data, 3000)
    drift = float(np.linalg.norm(np.asarray(arm.joint_pos) - np.asarray(config.init_qpos["arm"])))
    check("holds its rest pose under gravity", drift < 0.05, f"drift {drift:.4f} rad")

    # Track a commanded joint change.
    target = np.asarray(config.init_qpos["arm"], dtype=float).copy()
    target[0] += 0.6
    target[2] -= 0.4
    arm.ctrl = target
    settle(model, data, 4000)
    err = float(np.max(np.abs(np.asarray(arm.joint_pos) - target)))
    check("arm tracks a command", err < 0.05, f"max joint error {err:.4f} rad")

    tcp1 = view.get_move_group("arm").leaf_frame_to_world[:3, 3]
    check("TCP moved with the arm", float(np.linalg.norm(tcp1 - tcp0)) > 0.05,
          f"moved {float(np.linalg.norm(tcp1 - tcp0)):.4f} m")

    gripper.set_gripper_ctrl_open(False)
    settle(model, data, 2000)
    closed = gripper.inter_finger_dist
    gripper.set_gripper_ctrl_open(True)
    settle(model, data, 2000)
    opened = gripper.inter_finger_dist
    check("gripper opens wider than it closes", opened > closed + 0.02,
          f"closed {closed:.4f} m, open {opened:.4f} m")
    check("is_open agrees with the measurement", gripper.is_open)

    jac = arm.get_jacobian()
    check("arm jacobian is full rank (6 DoF)", np.linalg.matrix_rank(jac) == 6,
          f"rank {np.linalg.matrix_rank(jac)}")

    if args.render:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 900)
        for gid in range(model.ngeom):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            if "ceiling" in name:
                model.geom_rgba[gid, 3] = 0.0
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = data.body(f"{ns}mount").xpos + np.array([0, 0, 0.7])
        cam.distance, cam.azimuth, cam.elevation = 2.0, 135, -15
        with mujoco.Renderer(model, 900, 1280) as r:
            r.update_scene(data, camera=cam)
            from PIL import Image

            Image.fromarray(r.render()).save(args.render)
        print(f"\nwrote {args.render}")

    print("\nRESULT:", "ok" if not FAIL else f"{len(FAIL)} check(s) failed: {FAIL}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
