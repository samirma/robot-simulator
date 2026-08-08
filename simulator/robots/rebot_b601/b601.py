"""Robot class for the Seeed Studio reBot Arm B601-DM.

The vendor model (`vectorBH6/reBotArm_control_py`, `urdf/00-arm-rs_asm-v3/`) is a raw
SolidWorks URDF export. It has real link geometry, joint limits and inertias, but
**no actuators, no TCP frame and no MuJoCo-specific tuning** — so this class loads the
URDF and finishes the model in memory. Editing the spec on load beats generating a
patched file: MuJoCo strips the directory from URDF mesh paths, so a copy elsewhere on
disk would silently fail to find its meshes.

Two consequences of importing a URDF are worth knowing:

* MuJoCo **merges the URDF root link into the worldbody** when it carries no joint, so
  `base_link` disappears and `link1` becomes the root. Its visual is replaced by the
  mocap pedestal that `add_robot_to_scene` builds, exactly as for the SO-101.
* Collision uses the **convex hull** of each visual mesh, which MuJoCo computes on
  import. That is a reasonable fit for an arm whose links are each roughly convex, and
  it avoids running a separate convex-decomposition pass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtGeom

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

    from .b601_config import B601RobotConfig

log = logging.getLogger(__name__)

# The URDF's mesh filenames are bare names — MuJoCo strips the directory on import — so
# meshdir resolves them relative to the URDF's own location.
MESHDIR = "meshes"

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINTS = ("joint_left", "joint_right")

# Gains are sized per joint, from two competing requirements, and both matter:
#
#  * Stiffness — the servo must be able to hold the link against gravity, so kp is set
#    from the joint's torque capacity: full rated torque at STIFFNESS_ERROR of error.
#    (Effort limits are the vendor URDF's own: 36 N.m for the shoulder and elbow,
#    14 N.m for the wrist, 500 N for the prismatic fingers.)
#  * Stability — with an explicit integrator, kv*dt/I < 2. Hardcoding a kv is exactly
#    what breaks here: the wrist inertias are of order 1e-3 kg m^2, so a kv that looks
#    sensible beside the shoulder's makes joints 4-6 oscillate instead of hold.
#
# So kp is clamped to a safe multiple of the measured inertia and kv follows from
# critical damping. Effective inertia comes from `dof_M0` on a throwaway compile.
JOINT_EFFORT = {
    "joint1": 36.0, "joint2": 36.0, "joint3": 36.0,
    "joint4": 14.0, "joint5": 14.0, "joint6": 14.0,
    "joint_left": 500.0, "joint_right": 500.0,
}
STIFFNESS_ERROR = 0.08  # rad (or m) of error at which the joint commands full torque
# kp <= STABILITY_MARGIN * I / dt^2 keeps 2*dt*sqrt(kp/I) comfortably under 2.
STABILITY_MARGIN = 0.4
SIM_TIMESTEP = 0.002
MIN_INERTIA = 1e-5  # guard against a degenerate dof_M0 entry

# Grasp centre, measured along link6's +z (its frame already has +z = approach and the
# fingers separating along y, so the site needs no rotation).
TCP_OFFSET_Z = 0.1243
FINGER_TIP_LOCAL = 0.0


class B601Robot(Robot):
    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, exp_config)
        robot_config = exp_config.robot_config
        self._namespace = robot_config.robot_namespace
        self._robot_view = robot_config.robot_view_factory(mj_data, self._namespace)
        self._kinematics = MlSpacesKinematics(robot_config)

        arm_mode = robot_config.command_mode.get("arm") or "joint_position"
        arm_cls = JointPosController if arm_mode == "joint_position" else JointRelPosController
        self._controllers = {
            "arm": arm_cls(self._robot_view.get_move_group("arm")),
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }

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
        raise NotImplementedError("Parallel kinematics is not implemented for the B601")

    def create_robot_sensors(self):
        return super().create_robot_sensors() + [TCPPoseSensor(uuid="tcp_pose")]

    def get_arm_move_group_ids(self) -> list[str]:
        return ["arm"]

    def reset(self) -> None:
        for group, qpos in self.exp_config.robot_config.init_qpos.items():
            if group in self._robot_view.move_group_ids() and len(qpos):
                self._robot_view.get_move_group(group).joint_pos = np.asarray(qpos, dtype=float)
        for controller in self._controllers.values():
            controller.reset()

    @staticmethod
    def robot_model_root_name() -> str:
        # base_link is merged into the worldbody on URDF import, so link1 is the root.
        return "link1"

    # ------------------------------------------------------------------ spec surgery

    @classmethod
    def _load_robot_spec(cls, robot_config, strip_meshes: bool = False) -> MjSpec:
        spec = super()._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        spec.meshdir = MESHDIR

        # 0. Gravity compensation. The config asks for it, but MolmoSpaces only applies
        #    that through the datagen pipeline's control overrides — so a robot spawned
        #    directly (tools/spawn_robot.py, the self-test) would sag. Setting it on the
        #    bodies makes the behaviour the same either way. Without it this arm droops
        #    over a radian at the shoulder: its links are heavy relative to the modest
        #    effort limits the vendor URDF declares.
        if getattr(robot_config, "gravcomp", False):
            for body in spec.bodies:
                body.gravcomp = 1.0

        # 1. Measure first, before anything named is added. `spec.compile()` clears the
        #    name table of elements added beforehand, so a site or geom created above
        #    this line comes out anonymous in the final scene and cannot be looked up.
        inertia, resting_contacts = cls._measure(spec, robot_config)

        # 2. TCP site at the grasp centre, on link6 so it does not move with the fingers.
        link6 = spec.body("link6")
        if link6 is None:
            raise ValueError("link6 not found; upstream URDF changed?")
        link6.add_site(name="tcp", pos=[0.0, 0.0, TCP_OFFSET_Z], group=3)

        # 3. Tip markers so inter_finger_dist has something to measure. Non-colliding:
        #    they are rulers, not contact geometry.
        for geom_name, body_name in (("left_finger_tip", "gripper_left"),
                                     ("right_finger_tip", "gripper_right")):
            body = spec.body(body_name)
            if body is None:
                raise ValueError(f"{body_name} not found; upstream URDF changed?")
            body.add_geom(
                name=geom_name,
                type=mjtGeom.mjGEOM_SPHERE,
                size=[0.003, 0, 0],
                pos=[0.0, 0.0, FINGER_TIP_LOCAL],
                contype=0,
                conaffinity=0,
                group=3,
                rgba=[1, 0, 0, 0.4],
                mass=0.0,
            )

        # 4. Position actuators. The URDF declares effort and velocity limits but no
        #    transmission, so nothing would move without these.
        for name in ARM_JOINTS + GRIPPER_JOINTS:
            cls._add_position_actuator(spec, f"{name}_act", name, inertia[name])

        # 5. Turn off self-collision. MuJoCo collides the convex hull of each visual
        #    mesh, and on a raw CAD export those hulls overlap wherever parts nest —
        #    the two gripper fingers worst of all. The phantom contacts are violent:
        #    they drove joint6 a full radian off target with 81 N.m of actuator force
        #    fighting them, and excluding only the pairs touching at rest was not enough
        #    because more pairs collide as the arm moves. Excluding every pair is the
        #    deterministic option; the arm can now pass through itself, which is noted
        #    as a limitation in robots/README.md. Collision with the world is unaffected.
        bodies = [b.name for b in spec.bodies if b.name and b.name != "world"]
        for i, body_a in enumerate(bodies):
            for body_b in bodies[i + 1 :]:
                spec.add_exclude(bodyname1=body_a, bodyname2=body_b)
        return spec

    @staticmethod
    def _measure(spec: MjSpec, robot_config) -> tuple[dict[str, float], set[tuple[str, str]]]:
        """Per-joint effective inertia and rest-pose self-contacts, from one compile.

        `dof_M0` is the diagonal of the mass matrix in the reference configuration —
        exactly the quantity the gains have to be scaled against.
        """
        model = spec.compile()
        data = mujoco.MjData(model)

        inertia = {
            name: max(float(model.dof_M0[model.jnt_dofadr[model.joint(name).id]]), MIN_INERTIA)
            for name in ARM_JOINTS + GRIPPER_JOINTS
        }

        for group, qpos in (robot_config.init_qpos or {}).items():
            names = {"arm": ARM_JOINTS, "gripper": GRIPPER_JOINTS}.get(group)
            if not names or len(qpos) != len(names):
                continue
            for name, value in zip(names, qpos):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = float(value)
        mujoco.mj_forward(model, data)

        pairs = set()
        for con in data.contact[: data.ncon]:
            a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[con.geom1])
            b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[con.geom2])
            if a and b and a != b:
                pairs.add(tuple(sorted((a, b))))
        return inertia, pairs

    @staticmethod
    def _add_position_actuator(spec: MjSpec, name: str, joint: str, inertia: float) -> None:
        wanted = JOINT_EFFORT[joint] / STIFFNESS_ERROR
        stable = STABILITY_MARGIN * inertia / (SIM_TIMESTEP**2)
        kp = min(wanted, stable)
        kv = 2.0 * float(np.sqrt(kp * inertia))  # critically damped
        act = spec.add_actuator()
        act.name = name
        act.target = joint
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = kp
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kv
        # Left unset, MuJoCo would clamp ctrl to [0, 0]; mirror the joint's own range.
        jnt = spec.joint(joint)
        act.ctrlrange = np.array(jnt.range, dtype=float)

    # ------------------------------------------------------------------ attachment

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "B601RobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        robot_config = cast("B601RobotConfig", robot_config)
        add_base = robot_config.base_size is not None
        pos = list(pos) + [0.0] if len(pos) == 2 else list(pos)

        robot_body = spec.worldbody.add_body(
            name=f"{prefix}mount", pos=pos, quat=quat, mocap=True
        )
        if add_base:
            base_height = robot_config.base_size[2]
            robot_body.add_geom(
                type=mjtGeom.mjGEOM_BOX,
                size=[x / 2 for x in robot_config.base_size],
                pos=[0, 0, base_height / 2],
                rgba=[0.4, 0.4, 0.4, 1.0],
                group=0,
            )
            attach_frame = robot_body.add_frame(pos=[0, 0, base_height])
        else:
            attach_frame = robot_body.add_frame()

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root = robot_spec.body(cls.robot_model_root_name())
        if robot_root is None:
            raise ValueError("Robot root body 'link1' not found in the B601 spec")
        attach_frame.attach_body(robot_root, prefix, "")

        # The gains assume an implicit integrator, which every MolmoSpaces house already
        # uses; a bare MjSpec defaults to Euler, where the same damping goes unstable
        # (NaN in QACC at the wrist). MjSpec.attach_body merges body subtrees only, never
        # <option>, so this has to be set on the target scene explicitly.
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
