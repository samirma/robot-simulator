"""Experiment config for the SO-101 arm.

``robot_dir`` points at this directory, so MolmoSpaces loads the model from here rather
than from the managed asset tree — no fork of the upstream repo is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mujoco import MjData

from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

from .so101 import SO101Robot
from .so101_view import SO101RobotView

ROBOT_DIR = Path(__file__).resolve().parent

# Elbow-up rest pose that keeps the arm clear of its own mount and points the gripper
# forward and slightly down, so the wrist camera sees the workspace.
REST_ARM_QPOS = [0.0, -0.6, 1.0, 0.6, 0.0]

# Gripper joint, near-open. See so101_view.INTER_FINGER_DIST_RANGE.
REST_GRIPPER_QPOS = [1.2]


class SO101RobotConfig(BaseRobotConfig):
    robot_cls: type[SO101Robot] | None = SO101Robot
    robot_factory: Callable[[MjData, Any], Robot] | None = SO101Robot
    robot_view_factory: RobotViewFactory | None = SO101RobotView
    robot_namespace: str = "robot_0/"
    name: str = "so101"

    robot_xml_path: Path = Path("model.xml")
    robot_dir: Path = ROBOT_DIR

    # The SO-101 is a small tabletop arm (~0.4 m reach), so it needs a pedestal to put
    # its workspace at counter height in a house scene.
    base_size: list[float] | None = [0.3, 0.3, 0.7]

    init_qpos: dict[str, list[float]] = {
        "base": [],
        "arm": REST_ARM_QPOS,
        "gripper": REST_GRIPPER_QPOS,
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None

    command_mode: dict[str, str | None] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }

    # The MJCF's sts3215 actuators already carry tuned position gains, so let the model
    # values stand rather than overriding them here.
    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ("joint_position", "joint_rel_position")
