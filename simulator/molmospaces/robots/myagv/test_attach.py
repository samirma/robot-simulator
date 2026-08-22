#!/usr/bin/env python
"""Standalone check that the myAGV attaches and drives.

Runs standalone, so a failure points at the robot definition.

    python robots/myagv/test_attach.py [--scene /path/to/house.xml] [--render out.png]
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


def drive_to(model, data, view, target, steps: int = 4000) -> np.ndarray:
    """Command an absolute world pose and settle. Returns the reached [x, y, theta]."""
    base = view.get_move_group("base")
    base.ctrl = np.asarray(target, dtype=np.float64)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return np.asarray(base.joint_pos, dtype=np.float64).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="house MJCF to attach into")
    ap.add_argument("--render", default=None)
    args = ap.parse_args()

    from robots.myagv import MyAGVRobot, MyAGVRobotConfig, MyAGVRobotView

    config = MyAGVRobotConfig()

    spec = mujoco.MjSpec.from_file(args.scene) if args.scene else mujoco.MjSpec()
    if args.scene is None:
        spec.worldbody.add_light(
            pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.55, 0.56, 0.58, 1]
        )

    MyAGVRobot.add_robot_to_scene(
        config, spec, prefix=config.robot_namespace, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"compiled: {model.nbody} bodies, {model.ngeom} geoms, {model.nu} actuators")

    view = MyAGVRobotView(data, config.robot_namespace)
    base = view.get_move_group("base")

    origin = np.zeros(2)
    # Direction with room to move, and how far we can ask for. In an empty world that
    # is +x for a metre; in a house we head for the middle of the room and keep the
    # distances short, because "drove into a wall and stopped" is correct behaviour
    # that would otherwise read as a failure.
    heading = np.array([1.0, 0.0])
    reach, strafe_reach = 1.0, 0.5

    if args.scene:
        # The house origin is usually inside furniture (in FloorPlan1 it is inside the
        # kitchen island), and a robot embedded in geometry cannot move - which looks
        # exactly like a broken actuator. Start from open floor instead.
        from tools.spawn_robot import find_open_spot

        origin, yaw = find_open_spot(args.scene)
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        reach, strafe_reach = 0.5, 0.25
        start_pose = np.eye(4)
        start_pose[:2, 3] = origin
        base.pose = start_pose
        base.ctrl = np.array([origin[0], origin[1], 0.0])
        mujoco.mj_forward(model, data)
        print(f"starting from open floor at {np.round(origin, 3)}, heading {np.degrees(yaw):.0f} deg")

    left = np.array([-heading[1], heading[0]])
    fwd_target = origin + heading * reach
    strafe_target = fwd_target + left * strafe_reach
    check("move groups", view.move_group_ids() == ["base"], str(view.move_group_ids()))
    check("3 base actuators", base.n_actuators == 3, f"{base.n_actuators}")

    ns = config.robot_namespace
    check("front_camera present", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{ns}front_camera") >= 0)
    check("world site present", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{ns}world") >= 0)

    print("\ndriving:")
    # Translate along the open heading.
    reached = drive_to(model, data, view, [fwd_target[0], fwd_target[1], 0.0])
    err = np.linalg.norm(reached[:2] - fwd_target)
    check(f"drive {reach:.2f} m forward", err < 0.02,
          f"reached {np.round(reached[:2], 3)}, wanted {np.round(fwd_target, 3)}")

    # Strafe perpendicular to the heading with no heading change - the Mecanum capability.
    reached = drive_to(model, data, view, [strafe_target[0], strafe_target[1], 0.0])
    err = np.linalg.norm(reached[:2] - strafe_target)
    check(f"strafe {strafe_reach:.2f} m sideways", err < 0.02,
          f"reached {np.round(reached[:2], 3)}, wanted {np.round(strafe_target, 3)}")
    check("heading held during strafe", abs(reached[2]) < np.radians(1.0),
          f"theta={np.degrees(reached[2]):.3f} deg")

    # Rotate in place.
    reached = drive_to(model, data, view, [strafe_target[0], strafe_target[1], np.pi / 2])
    check("rotate to 90 deg", abs(reached[2] - np.pi / 2) < np.radians(1.0),
          f"theta={np.degrees(reached[2]):.3f} deg")
    check("position held during rotation",
          np.linalg.norm(reached[:2] - strafe_target) < 0.02,
          f"xy={np.round(reached[:2], 4)}")

    # The pose property must agree with the joints.
    pose = base.pose
    check("pose matches joints",
          np.allclose(pose[:2, 3], reached[:2], atol=1e-6)
          and abs(np.arctan2(pose[1, 0], pose[0, 0]) - reached[2]) < 1e-6)

    # Teleporting via the pose setter is how spawn_robot places it.
    target = np.eye(4)
    target[:2, 3] = [2.0, -1.0]
    base.pose = target
    mujoco.mj_forward(model, data)
    check("pose setter teleports", np.allclose(np.asarray(base.joint_pos)[:2], [2.0, -1.0], atol=1e-9))

    if args.render:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 800)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = data.body(f"{ns}base").xpos
        cam.distance, cam.azimuth, cam.elevation = 1.2, 135, -20
        with mujoco.Renderer(model, 800, 1280) as r:
            r.update_scene(data, camera=cam)
            from PIL import Image

            Image.fromarray(r.render()).save(args.render)
        print(f"\nwrote {args.render}")

    print("\nRESULT:", "ok" if not FAIL else f"{len(FAIL)} check(s) failed: {FAIL}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
