"""Experiment config for LeKiwi."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from mujoco import MjData

from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

from .lekiwi import LeKiwiRobot
from .lekiwi_view import LeKiwiRobotView

ROBOT_DIR = Path(__file__).resolve().parent


class LeKiwiCameraSystem(CameraSystemConfig):
    """The chassis camera the upstream model already ships (`front`)."""

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[Any] = [
        MjcfCameraConfig(
            name="front_camera",
            mjcf_name="front",
            robot_namespace="robot_0/",
            fov=70.0,
        )
    ]


class LeKiwiRobotConfig(BaseRobotConfig):
    robot_cls: type[LeKiwiRobot] | None = LeKiwiRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = LeKiwiRobot
    robot_view_factory: RobotViewFactory | None = LeKiwiRobotView
    robot_namespace: str = "robot_0/"
    name: str = "lekiwi"

    # Load the upstream model untouched; LeKiwiRobot._load_robot_spec edits the spec in
    # memory (see its docstring for why a patched copy on disk is worse here).
    robot_xml_path: Path = Path("upstream/lekiwi/lekiwi.xml")
    robot_dir: Path = ROBOT_DIR

    # Drives on the floor.
    base_size: list[float] | None = None

    init_qpos: dict[str, Any] = {
        "base": np.array([0.0, 0.0, 0.0]),  # x, y, theta
        # Arm folded in so it does not clip the chassis or trail on the floor.
        "arm": np.array([0.0, -1.4, 1.4, 0.6, 0.0]),
        "gripper": np.array([0.3]),
    }
    init_qpos_noise_range: dict[str, Any] | None = None

    command_mode: dict[str, str | None] = {
        "base": "holo_joint_rel_planar_position",
        "arm": "joint_position",
        "gripper": "joint_position",
    }

    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert self.command_mode.get("base") in (
            "holo_joint_planar_position",
            "holo_joint_rel_planar_position",
        )
        assert self.command_mode.get("arm") in ("joint_position", "joint_rel_position")
