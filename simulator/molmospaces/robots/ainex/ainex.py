"""Robot class for the Hiwonder AiNex: a 24-DoF biped humanoid.

The AiNex is the first robot here that walks rather than rolls, and the thing to
understand before reading the surgery below is that **it does not walk in simulation**.
Its gait engine on the real robot is `walking_module.so`, a precompiled ARM binary with
no source, so there is nothing to port; and authoring a balance controller for a 2.35 kg
biped is a research project, not an integration.

Instead the torso rides the same virtual slide-X / slide-Y / hinge-Z base that `myagv`
uses for its Mecanum drive, and the twelve leg joints are animated over the top by
`gait.py` at a phase and stride matched to the base's velocity.
The result navigates reliably, never falls, and puts its feet where a walking robot would
-- see `gait.py` on why the stance foot does not skate. What it does not do is balance.
The arms, grippers and head are genuinely actuated and are what grasping uses.

The vendor URDF is loaded untouched and the spec edited in memory, as for `rebot_b601`:
MuJoCo strips the directory from URDF mesh filenames, so a patched copy
elsewhere on disk would silently fail to find the 25 STLs
(`robots/URDF.md`). See `robots/ainex/urdf/PROVENANCE.md` for what was vendored.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco import MjData, MjSpec

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.abstract import Robot

import ainex_model
from ros_surfaces.ainex.gait import LegGeometry
from ainex_model import (  # re-exported: this module was where they lived
    CAMERA_FOVY_DEG,
    CAMERA_NAME,
    GRIPPER_ANGLES,
    TORSO_BODY,
)
from ros_surfaces.ainex import servos

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

    from .ainex_config import AiNexRobotConfig

log = logging.getLogger(__name__)

BASE_MODES = {
    "holo_joint_planar_position": JointPosController,
    "holo_joint_rel_planar_position": JointRelPosController,
}










class AiNexRobot(Robot):
    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, exp_config)
        robot_config = exp_config.robot_config
        self._namespace = robot_config.robot_namespace
        self._robot_view = robot_config.robot_view_factory(mj_data, self._namespace)
        self._kinematics = MlSpacesKinematics(robot_config)

        base_mode = robot_config.command_mode.get("base") or "holo_joint_rel_planar_position"
        if base_mode not in BASE_MODES:
            raise ValueError(f"unsupported base command mode {base_mode!r}")

        self._controllers: dict[str, Controller] = {
            "base": BASE_MODES[base_mode](self._robot_view.get_move_group("base"))
        }
        # Every limb group is plain joint position control. There is no Cartesian or IK
        # controller here on purpose: the real robot has no IK service either, and its
        # manipulation is replayed servo trajectories (see actions.py).
        for group in ("legs", "left_arm", "right_arm", "left_gripper", "right_gripper", "head"):
            self._controllers[group] = JointPosController(
                self._robot_view.get_move_group(group)
            )

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    @property
    def namespace(self):
        return self._namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        raise NotImplementedError("Parallel kinematics is not implemented for the AiNex")

    def get_arm_move_group_ids(self) -> list[str]:
        return ["left_arm", "right_arm"]

    def reset(self) -> None:
        for group, qpos in self.exp_config.robot_config.init_qpos.items():
            if group in self._robot_view.move_group_ids() and len(qpos):
                self._robot_view.get_move_group(group).joint_pos = np.asarray(qpos, dtype=float)
        for controller in self._controllers.values():
            controller.reset()


    # ------------------------------------------------------------------ spec surgery



    # --- the model ------------------------------------------------------------------
    #
    # It is built by `shared/ainex_model.py`, not here: the corrections that turn the
    # vendor URDF into a usable robot belong to the robot, and both engines compile the
    # same one. What is left in this class is what MolmoSpaces asks of a robot.

    @staticmethod
    def robot_model_root_name() -> str:
        return ainex_model.robot_model_root_name()

    @classmethod
    def _load_robot_spec(cls, robot_config, strip_meshes: bool = False) -> MjSpec:
        return ainex_model.build_spec(
            Path(robot_config.robot_dir) / robot_config.robot_xml_path
        )

    @classmethod
    def ride_height(cls, spec: MjSpec) -> float:
        return ainex_model.ride_height(spec)

    @classmethod
    def leg_geometry(cls, model, namespace: str = "") -> LegGeometry:
        return ainex_model.leg_geometry(model, namespace)












    # ------------------------------------------------------------------ attachment

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "AiNexRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        robot_config = cast("AiNexRobotConfig", robot_config)
        pos = list(pos) + [0.0] if len(pos) == 2 else list(pos)

        # World-aligned slide joints, so the robot is grafted in over the origin and
        # driven to its spawn pose (`robot_view.base.pose = ...`), exactly as myagv is.
        # Attaching it at an x/y offset or a rotation would silently give it a wrong
        # "forward".
        #
        # z is the exception, and it is what lets this robot stand on something: the slide
        # joints are x/y/yaw only, so lifting the graft cannot rotate or shift the axes the
        # base drives along. A caller mounting it on a worktop passes that worktop's height
        # here, and the ride height below is added to it -- the same rule
        # `mujoco_bridge.PlanarJointBase` enforces on the other engine.
        if not np.allclose(pos[:2], [0.0, 0.0]) or not np.allclose(quat, [1.0, 0.0, 0.0, 0.0]):
            raise ValueError(
                "AiNex must be attached over the origin with identity rotation -- a z "
                "offset is fine, and is how it is stood on a worktop; set its x/y pose "
                f"via robot_view.base.pose instead (got pos={pos}, quat={quat})"
            )

        # The gains assume an implicit integrator, and every MolmoSpaces house already
        # uses `implicitfast` -- but a bare MjSpec defaults to Euler, where 24 servos on
        # ~1e-4 kg.m^2 links go NaN. The same trap b601 records; it only bites in
        # standalone scenes such as test_attach.py's empty world.
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root = robot_spec.body(cls.robot_model_root_name())
        if robot_root is None:
            raise ValueError(f"Robot root body {TORSO_BODY!r} not found in the AiNex spec")

        # A pure z offset is safe: the virtual joints are x/y/yaw only, so lifting cannot
        # rotate or shift the axes the base drives along.
        spec.worldbody.add_frame(
            pos=[0.0, 0.0, float(pos[2]) + cls.ride_height(robot_spec)]
        ).attach_body(
            robot_root, prefix, ""
        )
