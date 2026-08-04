"""Robot class for LeKiwi: an SO-ARM100 arm on a three-wheel kiwi-drive base.

The upstream model (Ekumen-OS/lekiwi, `packages/lekiwi_sim/.../assets`) is a genuine
three-wheel omni drive: three hinge joints with velocity actuators. This adapter
**replaces that with a virtual holonomic base**, for the same reasons as `myagv` — a
planar slide/slide/hinge base reproduces the motion envelope exactly, is stable, and
plugs into MolmoSpaces' tested `JointPosController` path, whereas driving three omni
wheels needs a kiwi inverse-kinematics controller that MolmoSpaces does not have
(`controllers/base_pose.py` sketches a swerve variant but is marked untested and its
`pose` handling is broken).

Rather than generating a patched MJCF, the upstream file is loaded untouched and the
spec is edited in memory. That keeps `upstream/` pristine and, more practically, avoids
rewriting the relative `meshdir` and `<attach file=...>` paths that a copied XML would
break.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco import MjData, MjSpec

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

    from .lekiwi_config import LeKiwiRobotConfig

log = logging.getLogger(__name__)

BASE_MODES = {
    "holo_joint_planar_position": JointPosController,
    "holo_joint_rel_planar_position": JointRelPosController,
}

WHEEL_JOINTS = ("base_left_wheel_joint", "base_right_wheel_joint", "base_back_wheel_joint")
WHEEL_ACTUATORS = ("base_left_wheel", "base_right_wheel", "base_back_wheel")
BASE_BODY = "base_plate_layer_1_link"

# Keep the chassis hull just clear of the floor; the base has no vertical DoF so it
# cannot fall, and this stops it scraping while still colliding with walls.
CHASSIS_CLEARANCE = 0.005

# Bodies belonging to the SO-ARM100 arm rather than the chassis.
ARM_BODIES = {
    "Base", "Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll",
    "Fixed_Jaw", "Moving_Jaw", "wrist_camera_mount_link", "wrist_camera_link",
}

# Jaw tip contact geoms we add so inter_finger_dist has something to measure.
FIXED_JAW_TIP = ("fixed_jaw_tip", "Fixed_Jaw", (0.0, -0.10, 0.0))
MOVING_JAW_TIP = ("moving_jaw_tip", "Moving_Jaw", (0.0, -0.075, 0.0))


class LeKiwiRobot(Robot):
    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, exp_config)
        robot_config = exp_config.robot_config
        self._namespace = robot_config.robot_namespace
        self._robot_view = robot_config.robot_view_factory(mj_data, self._namespace)
        self._kinematics = MlSpacesKinematics(robot_config)

        base_mode = robot_config.command_mode.get("base") or "holo_joint_rel_planar_position"
        if base_mode not in BASE_MODES:
            raise ValueError(f"unsupported base command mode {base_mode!r}")
        arm_mode = robot_config.command_mode.get("arm") or "joint_position"
        arm_cls = JointPosController if arm_mode == "joint_position" else JointRelPosController

        self._controllers = {
            "base": BASE_MODES[base_mode](self._robot_view.get_move_group("base")),
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
        raise NotImplementedError("Parallel kinematics is not implemented for LeKiwi")

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
        return BASE_BODY

    # ------------------------------------------------------------------ spec surgery

    @classmethod
    def _load_robot_spec(cls, robot_config, strip_meshes: bool = False) -> MjSpec:
        spec = super()._load_robot_spec(robot_config, strip_meshes=strip_meshes)

        # 0. Drop the low-friction wheel/floor contact pairs. They name a `floor` geom
        #    that only exists in upstream's own scene.xml, so the model cannot even
        #    compile standalone with them; and once the wheels stop driving they are
        #    meaningless anyway.
        for pair in list(spec.pairs):
            spec.delete(pair)

        # 1. Drop the real wheel drive. The joints go too, not just the actuators: left
        #    in place they would be free-spinning hinges adding three unactuated DoF that
        #    jitter under contact for no benefit, since the base is driven virtually.
        for name in WHEEL_ACTUATORS:
            actuator = spec.actuator(name)
            if actuator is not None:
                spec.delete(actuator)
        for name in WHEEL_JOINTS:
            joint = spec.joint(name)
            if joint is not None:
                spec.delete(joint)

        base = spec.body(BASE_BODY)
        if base is None:
            raise ValueError(f"{BASE_BODY!r} not found; upstream model changed?")

        # 2. Remove the chassis free joint. Upstream floats the base and drives it via
        #    wheel friction; we replace that with virtual joints, and a body may carry
        #    at most 6 DoF so the free joint and the three virtual ones cannot coexist.
        for joint in list(base.joints):
            spec.delete(joint)

        # 3. Virtual holonomic base, world-aligned, in the (x, y, theta) order
        #    HoloJointsRobotBaseGroup requires.
        base.add_site(name="base_site", pos=[0, 0, 0], group=3)
        base.add_joint(name="base_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], damping=5)
        base.add_joint(name="base_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], damping=5)
        base.add_joint(
            name="base_theta", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.5
        )

        # 3. Jaw tip geoms, so the gripper has an inter-finger distance to report.
        for geom_name, body_name, pos in (FIXED_JAW_TIP, MOVING_JAW_TIP):
            body = spec.body(body_name)
            if body is None:
                raise ValueError(f"{body_name!r} not found; upstream model changed?")
            body.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.004, 0, 0],
                pos=list(pos),
                contype=0,
                conaffinity=0,
                group=3,
                rgba=[1, 0, 0, 0.4],
                mass=0.0,
            )

        # 4. TCP site between the jaws, on the fixed jaw so it does not move with the
        #    gripper opening.
        fixed_jaw = spec.body("Fixed_Jaw")
        fixed_jaw.add_site(name="tcp", pos=[0.0, -0.09, 0.0], group=3)

        # 5. Replace every chassis collider with one lifted box hull.
        #    Upstream gave the omni wheels their sideways slip through the low-friction
        #    contact pairs deleted in step 0. Without those, the wheels and the plate grip
        #    the floor at default friction and fight the virtual joints: the base
        #    under-shoots badly and picks up ~20 degrees of uncommanded yaw from
        #    asymmetric contact. Since the base has no vertical DoF it cannot fall, so
        #    nothing needs to carry it — one hull that never touches the floor but still
        #    stops it at walls is both cleaner and much cheaper. Arm geoms keep their
        #    own collisions so the arm can still touch the world.
        for geom in list(spec.geoms):
            body_name = geom.parent.name if geom.parent is not None else ""
            if body_name not in ARM_BODIES:
                geom.contype = 0
                geom.conaffinity = 0

        dx, dy, z_min, z_max = cls._measure_chassis(spec)
        # Upstream floats the chassis on a free joint, so with that joint gone the body
        # sits at its declared z=0 and the whole robot is buried in the floor. The hull
        # spans the chassis but starts just above its lowest point, and add_robot_to_scene
        # lifts the whole robot by -z_min so that point rests on the ground.
        base.add_geom(
            name="chassis_hull",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[dx / 2, dy / 2, max((z_max - z_min - CHASSIS_CLEARANCE) / 2, 0.01)],
            pos=[0, 0, (z_min + CHASSIS_CLEARANCE + z_max) / 2],
            group=3,
            rgba=[1, 0, 0, 0.0],
            mass=0.0,
            contype=1,
            conaffinity=1,
        )

        # 6. Base actuators, gains derived from the compiled model rather than guessed —
        #    the same lesson as myagv, where a hand-set kv broke the explicit-integration
        #    stability bound and made the simulation diverge.
        mass, izz = cls._measure_base(spec)
        cls._add_base_actuator(spec, "base_x_act", "base_x", kp=600.0 * mass, inertia=mass)
        cls._add_base_actuator(spec, "base_y_act", "base_y", kp=600.0 * mass, inertia=mass)
        cls._add_base_actuator(spec, "base_theta_act", "base_theta", kp=2000.0 * izz, inertia=izz)
        return spec

    @staticmethod
    def _measure_chassis(spec: MjSpec) -> np.ndarray:
        """(x, y, z) extent of the chassis, from a throwaway compile.

        Only the base plate and the parts bolted to it are measured; the arm is excluded
        so a raised arm does not inflate the collision hull.
        """
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        pts = []
        for gid in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
            if body in ARM_BODIES:
                continue
            pts.append(data.geom_xpos[gid])
        pts = np.array(pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (
            max(float(hi[0] - lo[0]), 0.05),
            max(float(hi[1] - lo[1]), 0.05),
            float(lo[2]),
            float(hi[2]),
        )

    @classmethod
    def ride_height(cls, spec: MjSpec) -> float:
        """How far to lift the robot so its lowest chassis point rests on the floor."""
        _, _, z_min, _ = cls._measure_chassis(spec)
        return -z_min

    @staticmethod
    def _measure_base(spec: MjSpec) -> tuple[float, float]:
        """Total mass and yaw inertia, from a throwaway compile of the edited spec."""
        model = spec.compile()
        mass = float(model.body_mass.sum())
        # Yaw inertia about the base origin, parallel-axis summed over the bodies.
        izz = 0.0
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        origin = data.xpos[model.body(BASE_BODY).id][:2]
        for bid in range(1, model.nbody):
            r = data.xipos[bid][:2] - origin
            izz += float(model.body_inertia[bid][2] + model.body_mass[bid] * float(r @ r))
        return mass, max(izz, 1e-3)

    @staticmethod
    def _add_base_actuator(spec: MjSpec, name: str, joint: str, kp: float, inertia: float) -> None:
        act = spec.add_actuator()
        act.name = name
        act.target = joint
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = kp
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        # Critically damped: kv = 2*sqrt(kp*I).
        kv = 2.0 * float(np.sqrt(kp * inertia))
        act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kv
        act.ctrlrange = np.array([-25.0, 25.0])

    # ------------------------------------------------------------------ attachment

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "LeKiwiRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        robot_config = cast("LeKiwiRobotConfig", robot_config)
        pos = list(pos) + [0.0] if len(pos) == 2 else list(pos)

        # World-aligned slide joints, so the robot must be grafted in at the origin and
        # driven to its spawn pose (`robot_view.base.pose = ...`), exactly as for myagv.
        if not np.allclose(pos, [0.0, 0.0, 0.0]) or not np.allclose(quat, [1.0, 0.0, 0.0, 0.0]):
            raise ValueError(
                "LeKiwi must be attached at the origin with identity rotation; set its "
                f"pose via robot_view.base.pose instead (got pos={pos}, quat={quat})"
            )

        spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root = robot_spec.body(cls.robot_model_root_name())
        if robot_root is None:
            raise ValueError(f"Robot root body {BASE_BODY!r} not found in the LeKiwi spec")

        # Lift so the wheels sit on the floor rather than through it. A z offset is safe
        # here — the virtual joints are x/y/yaw only, so this cannot rotate or shift the
        # axes the base drives along.
        spec.worldbody.add_frame(pos=[0.0, 0.0, cls.ride_height(robot_spec)]).attach_body(
            robot_root, prefix, ""
        )
