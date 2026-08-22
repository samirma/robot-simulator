"""Robot class for the Elephant Robotics myAGV Pi 2023.

A base-only mobile robot: no arm, no gripper, no IK. Modelled on
`molmo_spaces/robots/mobile_franka.py` for the holonomic base handling, minus
everything arm-related.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from mujoco import MjData, MjSpec

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

    from .myagv_config import MyAGVRobotConfig

log = logging.getLogger(__name__)

# Both are world-frame [x, y, theta] in metres/radians. The relative mode adds the
# command to the base's *current* pose each step, so driving into a wall does not wind
# up an ever-growing target; it also has a deadband that parks the base when the
# command goes to zero, which is what makes key-release stop the robot cleanly.
BASE_COMMAND_MODES = {
    "holo_joint_planar_position": JointPosController,
    "holo_joint_rel_planar_position": JointRelPosController,
}


class MyAGVRobot(Robot):
    def __init__(self, mj_data: MjData, config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, config)
        robot_config = config.robot_config
        self._robot_view = robot_config.robot_view_factory(mj_data, robot_config.robot_namespace)

        mode = robot_config.command_mode.get("base") or "holo_joint_rel_planar_position"
        if mode not in BASE_COMMAND_MODES:
            raise ValueError(
                f"Base command mode {mode!r} is not supported for the myAGV; "
                f"expected one of {sorted(BASE_COMMAND_MODES)}"
            )
        self._base_command_mode = mode
        self._controllers = {"base": BASE_COMMAND_MODES[mode](self._robot_view.get_move_group("base"))}

    @property
    def base_command_mode(self) -> str:
        return self._base_command_mode

    @property
    def namespace(self):
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        # There is no articulated chain to solve for: the base is three virtual joints
        # that already are the pose.
        raise NotImplementedError("The myAGV has no arm, so no kinematics solver")

    @property
    def parallel_kinematics(self):
        raise NotImplementedError("The myAGV has no arm, so no kinematics solver")

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def get_arm_move_group_ids(self) -> list[str]:
        return []

    def reset(self) -> None:
        init = self.exp_config.robot_config.init_qpos.get("base")
        if init is not None and len(init) == 3:
            self._robot_view.get_move_group("base").joint_pos = np.asarray(init, dtype=np.float64)
        for controller in self._controllers.values():
            controller.reset()

    @staticmethod
    def robot_model_root_name() -> str:
        return "base"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "MyAGVRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        robot_config = cast("MyAGVRobotConfig", robot_config)
        pos = list(pos) + [0.0] if len(pos) == 2 else list(pos)

        # The base joints are world-aligned slide/hinge joints, so the robot must be
        # grafted in at the origin with identity rotation; anywhere else and driving
        # base_x would no longer mean "move along world +x". Placement is done instead
        # by writing the joints, i.e. `robot_view.base.pose = ...`. RB-Y1 has the same
        # constraint (see molmo_spaces/robots/rby1.py).
        if not np.allclose(pos, [0.0, 0.0, 0.0]) or not np.allclose(quat, [1.0, 0.0, 0.0, 0.0]):
            raise ValueError(
                "The myAGV must be attached at the origin with identity rotation; "
                "set its pose via robot_view.base.pose instead of pos/quat "
                f"(got pos={pos}, quat={quat})"
            )

        # Reference site the base actuators and HoloJointsRobotBaseGroup measure against.
        spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root = robot_spec.body(cls.robot_model_root_name())
        if robot_root is None:
            raise ValueError("Robot root body 'base' not found in the myAGV spec")
        spec.worldbody.add_frame().attach_body(robot_root, prefix, "")
