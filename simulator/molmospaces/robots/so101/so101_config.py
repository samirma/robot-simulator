"""Experiment config for the SO-101 arm.

``robot_dir`` points at ``shared/robots/so101/``, so MolmoSpaces loads the model from
there rather than from the managed asset tree — no fork of the upstream repo is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mujoco import MjData

from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

import robots_spec

from .so101 import SO101Robot
from .so101_view import SO101RobotView

# The generated model.xml, shared across simulator engines; its official meshes are
# loaded from robots_specs/so101/.
ROBOT_XML = robots_spec.model_xml("so101")

# Upright rest pose: upper arm and forearm both vertical, square to the base, and the
# wrist bent so the wrist camera looks level along the base's +x (its front) with an
# upright horizon. "Vertical" is each link's own long axis, not the joint-to-joint line:
# the elbow sits 28 mm off the upper arm's axis, so lining up the joints instead leaves
# the link visibly leaning 14 deg. The camera is mounted 0.57 rad off the jaw axis, so a
# level view has the jaw pointing ~33 deg up. wrist_roll is the -1.52 branch; its +1.62
# twin gives the same jaw but an upside-down image, and needs a wrist_flex past its limit.
REST_ARM_QPOS = [0.0, 0.0, -1.5708, 1.0008, -1.5221]

# Gripper joint, near-open. See so101_view.INTER_FINGER_DIST_RANGE.
REST_GRIPPER_QPOS = [1.2]


class SO101RobotConfig(BaseRobotConfig):
    robot_cls: type[SO101Robot] | None = SO101Robot
    robot_factory: Callable[[MjData, Any], Robot] | None = SO101Robot
    robot_view_factory: RobotViewFactory | None = SO101RobotView
    robot_namespace: str = "robot_0/"
    name: str = "so101"

    robot_xml_path: Path = Path(ROBOT_XML.name)
    robot_dir: Path = ROBOT_XML.parent

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
