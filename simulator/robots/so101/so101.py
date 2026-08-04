"""Robot class for the SO-101 arm.

Modelled on ``molmospaces/examples/add_robot/xarm7.py``. Joint-position control of a
5-DoF arm plus a one-actuator gripper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from mujoco import MjData, MjSpec, mjtGeom

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

    from .so101_config import SO101RobotConfig

log = logging.getLogger(__name__)


class SO101Robot(Robot):
    def __init__(self, mj_data: MjData, config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, config)
        robot_config = config.robot_config
        self._robot_view = robot_config.robot_view_factory(mj_data, robot_config.robot_namespace)
        self._kinematics = MlSpacesKinematics(robot_config)

        command_mode = robot_config.command_mode
        arm_controller_cls = (
            JointPosController
            if not command_mode or command_mode["arm"] == "joint_position"
            else JointRelPosController
        )
        self._controllers = {
            "arm": arm_controller_cls(self._robot_view.get_move_group("arm")),
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }

    @property
    def namespace(self):
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        # Batched GPU IK (SimpleWarpKinematics) is only useful with CUDA, and this
        # install is CPU-only. Nothing in the joint-control path needs it.
        raise NotImplementedError("Parallel kinematics is not implemented for the SO-101")

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def create_robot_sensors(self):
        return super().create_robot_sensors() + [TCPPoseSensor(uuid="tcp_pose")]

    def get_arm_move_group_ids(self) -> list[str]:
        return ["arm"]

    def reset(self) -> None:
        for mg_id, default_pos in self.exp_config.robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = default_pos

    @staticmethod
    def robot_model_root_name() -> str:
        return "base"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "SO101RobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        robot_config = cast("SO101RobotConfig", robot_config)
        add_base = robot_config.base_size is not None
        pos = pos + [0.0] if len(pos) == 2 else pos

        # Named "mount", not "base": the SO-101's own root body is called "base" and
        # gets the same prefix on attach, so "base" here would be a duplicate name.
        robot_body = spec.worldbody.add_body(
            name=f"{prefix}mount",
            pos=pos,
            quat=quat,
            mocap=True,
        )
        if add_base:
            base_height = robot_config.base_size[2]
            robot_body.add_geom(
                type=mjtGeom.mjGEOM_BOX,
                size=[x / 2 for x in robot_config.base_size],
                pos=[0, 0, base_height / 2],
                rgba=[0.4, 0.4, 0.4, 1.0],
                group=0,  # visual group
            )
            attach_frame = robot_body.add_frame(pos=[0, 0, base_height])
        else:
            attach_frame = robot_body.add_frame()

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root_name = cls.robot_model_root_name()
        robot_root = robot_spec.body(robot_root_name)
        if robot_root is None:
            raise ValueError(f"Robot root body {robot_root_name!r} not found in the SO-101 spec")
        attach_frame.attach_body(robot_root, prefix, "")
