"""Experiment config for the Seeed Studio reBot Arm B601-DM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from mujoco import MjData

from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

from .b601 import B601Robot
from .b601_view import B601RobotView

ROBOT_DIR = Path(__file__).resolve().parent


class B601RobotConfig(BaseRobotConfig):
    robot_cls: type[B601Robot] | None = B601Robot
    robot_factory: Callable[[MjData, Any], Robot] | None = B601Robot
    robot_view_factory: RobotViewFactory | None = B601RobotView
    robot_namespace: str = "robot_0/"
    name: str = "rebot_b601"

    # The vendor URDF, loaded untouched; B601Robot._load_robot_spec finishes it.
    robot_xml_path: Path = Path("urdf/00-arm-rs_asm-v3.urdf")
    robot_dir: Path = ROBOT_DIR

    # 767 mm reach, so a taller pedestal than the SO-101's to give it a useful workspace.
    base_size: list[float] | None = [0.3, 0.3, 0.6]

    # Elbow-up rest pose within the URDF joint limits
    # (joint2 and joint3 are limited to [0, pi]).
    init_qpos: dict[str, Any] = {
        "base": np.array([]),
        "arm": np.array([0.0, 0.9, 1.2, 0.0, 0.6, 0.0]),
        "gripper": np.array([0.025, 0.035]),
    }
    init_qpos_noise_range: dict[str, Any] | None = None

    command_mode: dict[str, str | None] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }

    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert self.command_mode.get("arm") in ("joint_position", "joint_rel_position")
        assert self.command_mode.get("gripper") == "joint_position"
