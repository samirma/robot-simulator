"""Move groups and robot view for the SO-101 arm.

The SO-101 (TheRobotStudio / LeRobot) is a 5-DoF tabletop arm with a single hinged
jaw. Structure, as named in model.xml:

    base -> shoulder -> upper_arm -> lower_arm -> wrist -> gripper -> moving_jaw_so101_v1

    arm joints:  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
    gripper:     one hinge joint + one actuator ("gripper")
    TCP site:    "tcp"  (+z = approach, +y = finger axis; added by make_model.py)

Two things differ from the Franka/xarm7 pattern this is modelled on:

* Only **5** arm DoF, so the arm is not able to reach an arbitrary 6-DoF pose. Full
  Cartesian IK will be under-determined; joint-space control is exact.
* A **single hinged jaw** closing against a fixed jaw, rather than two symmetric
  parallel fingers. `inter_finger_dist` is therefore measured between the two jaw
  *tips*, and it varies non-linearly with the joint angle.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    MJCFFrameMixin,
    MocapRobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Measured from model.xml by sweeping the gripper joint over its range and taking the
# distance between the jaw-tip geoms; see tools/measure_gripper.py.
INTER_FINGER_DIST_RANGE = (0.007, 0.127)

# Actuator ctrlrange for the gripper joint. Larger = more open.
GRIPPER_CTRL_CLOSED = -0.17453
GRIPPER_CTRL_OPEN = 1.74533


class SO101BaseGroup(MocapRobotBaseGroup):
    """Unactuated mocap mount the arm is bolted to; teleported during task sampling.

    Note the body is named ``mount`` rather than ``base``: the SO-101's own root body is
    already called ``base``, and both get the same namespace prefix when the robot is
    attached to a scene, so reusing the name would collide.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        body_id: int = mj_data.model.body(f"{namespace}mount").id
        super().__init__(mj_data, body_id)


class SO101ArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    def __init__(
        self,
        mj_data: MjData,
        base_group: SO101BaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}{j}").id for j in ARM_JOINTS]
        # The SO-101 names each actuator after the joint it drives.
        act_ids = [model.actuator(f"{namespace}{j}").id for j in ARM_JOINTS]
        self._arm_root_id = model.body(f"{namespace}base").id
        self._ee_site_id = model.site(f"{namespace}tcp").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base_group)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class SO101GripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(self, mj_data: MjData, base_group: SO101BaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}gripper").id]
        act_ids = [model.actuator(f"{namespace}gripper").id]
        # The fixed jaw is part of the "gripper" body; only the other jaw moves.
        root_body_id = model.body(f"{namespace}gripper").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}tcp").id
        self._finger_1_geom_id = model.geom(f"{namespace}fixed_jaw_sph_tip2").id
        self._finger_2_geom_id = model.geom(f"{namespace}moving_jaw_sph_tip2").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = [GRIPPER_CTRL_OPEN if open else GRIPPER_CTRL_CLOSED]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return INTER_FINGER_DIST_RANGE

    @property
    def inter_finger_dist(self) -> float:
        dist = mujoco.mj_geomDistance(
            self.mj_model, self.mj_data, self._finger_1_geom_id, self._finger_2_geom_id, 0.5, None
        )
        return max(0.0, dist)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class SO101RobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = SO101BaseGroup(mj_data, namespace)
        move_groups = {
            "base": base,
            "arm": SO101ArmGroup(mj_data, base, namespace),
            "gripper": SO101GripperGroup(mj_data, base, namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return "so101"

    @property
    def base(self) -> SO101BaseGroup:
        return self._move_groups["base"]
