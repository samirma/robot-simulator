"""Experiment config for the Hiwonder AiNex."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from mujoco import MjData

from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

from . import servos
from .ainex import CAMERA_FOVY_DEG, CAMERA_NAME, AiNexRobot
from .ainex_view import AiNexRobotView

ROBOT_DIR = Path(__file__).resolve().parent


class AiNexCameraSystem(CameraSystemConfig):
    """The single monocular USB camera, which on this robot rides the pan/tilt head.

    640x480 is the hardware's own resolution (Hiwonder quote 640x480 at 30 fps), and it is
    also about as much as a frame-by-frame JSON websocket will carry.
    """

    img_resolution: tuple[int, int] = (640, 480)  # (width, height)
    cameras: list[Any] = [
        MjcfCameraConfig(
            name="front_camera",
            mjcf_name=CAMERA_NAME,
            robot_namespace="robot_0/",
            fov=CAMERA_FOVY_DEG,
        )
    ]


def _pose(names: tuple[str, ...]) -> np.ndarray:
    return np.array([servos.INIT_POSE[n] for n in names], dtype=float)


class AiNexRobotConfig(BaseRobotConfig):
    robot_cls: type[AiNexRobot] | None = AiNexRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = AiNexRobot
    robot_view_factory: RobotViewFactory | None = AiNexRobotView
    robot_namespace: str = "robot_0/"
    name: str = "ainex"

    # The flattened vendor URDF. `AiNexRobot._load_robot_spec` edits the spec in memory --
    # see its docstring for why a patched copy on disk would be worse.
    robot_xml_path: Path = Path("urdf/ainex.urdf")
    robot_dir: Path = ROBOT_DIR

    # It stands on the floor; there is no pedestal.
    base_size: list[float] | None = None

    # The vendor's own init_pose.yaml, split per move group. This is a slight crouch
    # rather than a straight stand -- hips and knees already loaded, which is what the
    # gait swings from.
    init_qpos: dict[str, Any] = {
        "base": np.array([0.0, 0.0, 0.0]),  # x, y, theta
        "legs": _pose(servos.LEG_JOINTS),
        "left_arm": _pose(tuple(j for j in servos.ARM_JOINTS if j.startswith("l_"))),
        "right_arm": _pose(tuple(j for j in servos.ARM_JOINTS if j.startswith("r_"))),
        "left_gripper": np.array([servos.INIT_POSE["l_gripper"]]),
        "right_gripper": np.array([servos.INIT_POSE["r_gripper"]]),
        "head": _pose(servos.HEAD_JOINTS),
    }
    init_qpos_noise_range: dict[str, Any] | None = None

    # Relative for the base, as for myagv: the command is added to the current pose each
    # step, so a robot held up by a wall does not accumulate an ever-growing target.
    # The limbs are absolute -- an action group names the pose it wants, not a delta.
    command_mode: dict[str, str | None] = {
        "base": "holo_joint_rel_planar_position",
        "legs": "joint_position",
        "left_arm": "joint_position",
        "right_arm": "joint_position",
        "left_gripper": "joint_position",
        "right_gripper": "joint_position",
        "head": "joint_position",
    }

    # The real robot sags under its own weight; this one does not. See `ainex.py` step 0
    # for why that is the right trade when the gait is animated rather than balanced.
    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert self.command_mode.get("base") in (
            "holo_joint_planar_position",
            "holo_joint_rel_planar_position",
        ), f"unsupported base command mode {self.command_mode.get('base')!r}"
