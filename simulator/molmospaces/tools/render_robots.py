#!/usr/bin/env python
"""Render every loadable robot on a neutral stage, one PNG each.

Doubles as a smoke test: a robot that cannot be attached and compiled shows up as a
failure line rather than a missing image.

    python tools/render_robots.py --outdir /tmp/robots
    python tools/render_robots.py --only so101,rby1
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import mujoco
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))


def robot_registry() -> dict[str, tuple]:
    """label -> (config class, human description). Imported lazily; molmo_spaces is slow."""
    from molmo_spaces.configs import robot_configs as rc

    from robots.ainex import AiNexRobotConfig
    from robots.myagv import MyAGVRobotConfig
    from robots.rebot_b601 import B601RobotConfig
    from robots.so101 import SO101RobotConfig

    return {
        "franka_droid": (rc.FrankaRobotConfig, "Franka FR3 + Robotiq 2F-85 (DROID)"),
        "franka_cap": (rc.FrankaCAPRobotConfig, "Franka FR3 + CAP gripper"),
        "mobile_franka": (rc.MobileFrankaRobotConfig, "Franka on a mobile base"),
        "rby1": (rc.RBY1Config, "Rainbow Robotics RB-Y1"),
        "rby1m": (rc.RBY1MConfig, "Rainbow Robotics RB-Y1M"),
        "yam": (rc.I2rtYamRobotConfig, "I2RT YAM"),
        "bimanual_yam": (rc.BimanualYamRobotConfig, "I2RT YAM (bimanual)"),
        "rum": (rc.FloatingRUMRobotConfig, "Floating RUM gripper"),
        "floating_robotiq": (rc.FloatingRobotiq2f85RobotConfig, "Floating Robotiq 2F-85"),
        "so101": (SO101RobotConfig, "TheRobotStudio SO-101 (out-of-tree)"),
        "myagv": (MyAGVRobotConfig, "Elephant Robotics myAGV Pi (out-of-tree)"),
        "rebot_b601": (B601RobotConfig, "Seeed reBot Arm B601-DM (out-of-tree)"),
        "ainex": (AiNexRobotConfig, "Hiwonder AiNex 24-DoF humanoid (out-of-tree)"),
    }


def build_stage() -> mujoco.MjSpec:
    """An empty world with a floor and enough light to photograph a robot on."""
    spec = mujoco.MjSpec()
    # MuJoCo 3.5 replaced the `directional` flag with an explicit light `type`.
    spec.worldbody.add_light(
        pos=[0, 0, 4.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.worldbody.add_light(
        pos=[2.0, -2.0, 2.5],
        dir=[-0.6, 0.6, -0.7],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[8, 8, 0.1],
        rgba=[0.55, 0.56, 0.58, 1.0],
    )
    return spec


def robot_bounds(model, data, prefix: str) -> tuple[np.ndarray, float]:
    """Centre and radius of the geoms belonging to the attached robot."""
    pts = []
    for gid in range(model.ngeom):
        body = model.geom_bodyid[gid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.startswith(prefix):
            pts.append(data.geom_xpos[gid])
    if not pts:
        return np.array([0.0, 0.0, 0.5]), 1.0
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 0.3)
    return centre, radius


def render_one(label: str, config_cls, out_path: Path, width: int, height: int) -> str:
    config = config_cls()
    robot_cls = config.robot_cls
    prefix = config.robot_namespace

    spec = build_stage()
    robot_cls.add_robot_to_scene(
        config, spec, prefix=prefix, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    # Rest pose: set both joint state and actuator targets, otherwise position
    # actuators immediately drive the robot out of the pose we want to photograph.
    view = config.robot_view_factory(data, prefix)
    for group_id, qpos in (config.init_qpos or {}).items():
        # init_qpos values are lists for some robots and numpy arrays for others
        # (e.g. RB-Y1), so test emptiness by length rather than truthiness.
        if qpos is None or len(qpos) == 0 or group_id not in view.move_group_ids():
            continue
        group = view.get_move_group(group_id)
        try:
            group.joint_pos = qpos
        except Exception:
            pass
        try:
            group.ctrl = qpos
        except Exception:
            # Groups whose actuators do not map 1:1 onto joints (mocap bases, tendon
            # driven grippers) reject a joint-width ctrl vector; the pose still holds.
            pass
    mujoco.mj_forward(model, data)

    centre, radius = robot_bounds(model, data, prefix)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = centre
    cam.distance = radius * 3.0
    cam.azimuth = 135
    cam.elevation = -15

    with mujoco.Renderer(model, height, width) as renderer:
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

    from PIL import Image

    Image.fromarray(pixels).save(out_path)

    dof = sum(
        len(view.get_move_group(g).joint_ids)
        for g in view.move_group_ids()
        if hasattr(view.get_move_group(g), "joint_ids")
    )
    return f"{model.nbody - 2} bodies, {model.nu} actuators, {dof} joints in move groups"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="robot_renders")
    ap.add_argument("--only", default=None, help="comma-separated subset of labels")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    registry = robot_registry()
    wanted = args.only.split(",") if args.only else list(registry)

    failures = []
    for label in wanted:
        if label not in registry:
            print(f"{label:18s} SKIP  unknown robot")
            continue
        config_cls, description = registry[label]
        out_path = outdir / f"{label}.png"
        try:
            info = render_one(label, config_cls, out_path, args.width, args.height)
            print(f"{label:18s} ok    {description} - {info}")
        except Exception as exc:
            failures.append((label, exc))
            print(f"{label:18s} FAIL  {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3, file=sys.stderr)

    print(f"\n{len(wanted) - len(failures)}/{len(wanted)} robots rendered into {outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
