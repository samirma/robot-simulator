"""Move groups and robot view for the Seeed Studio reBot Arm B601-DM.

A 6-DoF arm with a two-finger prismatic gripper ("6+1 DoF" in Seeed's marketing).
Built from the vendor URDF (`vectorBH6/reBotArm_control_py`), which is a SolidWorks
export: real link geometry and inertias, but no actuators and no MuJoCo tuning. See
`b601.py` for what is added on load.

Convenient accident of the CAD: link6's own frame already matches the MolmoSpaces
gripper convention — +z is the approach direction and the fingers separate along y —
so the TCP site needs no rotation, only an offset along z to the grasp centre.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    MJCFFrameMixin,
    MocapRobotBaseGroup,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINTS = ("joint_left", "joint_right")

# Measured by sweeping both finger joints over their (asymmetric) ranges.
INTER_FINGER_DIST_RANGE = (0.0, 0.1216)


class B601BaseGroup(MocapRobotBaseGroup):
    """Unactuated mocap pedestal the arm is bolted to."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        super().__init__(mj_data, mj_data.model.body(f"{namespace}mount").id)


class B601ArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}{n}").id for n in ARM_JOINTS]
        act_ids = [model.actuator(f"{namespace}{n}_act").id for n in ARM_JOINTS]
        self._arm_root_id = model.body(f"{namespace}link1").id
        self._tcp_site_id = model.site(f"{namespace}tcp").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._tcp_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class B601GripperGroup(MJCFFrameMixin, GripperGroup):
    """Two independently actuated prismatic fingers.

    Unlike the SO-101's single hinged jaw, both fingers move — but their travel is
    *asymmetric* in the vendor URDF (0.050 m left, 0.0715 m right), so `set_gripper_ctrl_open`
    drives each to its own limit rather than to a shared value.
    """

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}{n}").id for n in GRIPPER_JOINTS]
        act_ids = [model.actuator(f"{namespace}{n}_act").id for n in GRIPPER_JOINTS]
        super().__init__(mj_data, joint_ids, act_ids, model.body(f"{namespace}link6").id, base)
        self._tcp_site_id = model.site(f"{namespace}tcp").id
        self._left_geom_id = model.geom(f"{namespace}left_finger_tip").id
        self._right_geom_id = model.geom(f"{namespace}right_finger_tip").id
        self._open_ctrl = [float(model.jnt_range[j][1]) for j in joint_ids]
        self._closed_ctrl = [float(model.jnt_range[j][0]) for j in joint_ids]

    @property
    def leaf_frame_id(self) -> int:
        return self._tcp_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = np.array(self._open_ctrl if open else self._closed_ctrl)

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return INTER_FINGER_DIST_RANGE

    @property
    def inter_finger_dist(self) -> float:
        dist = mujoco.mj_geomDistance(
            self.mj_model, self.mj_data, self._left_geom_id, self._right_geom_id, 0.5, None
        )
        return max(0.0, dist)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class B601RobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = B601BaseGroup(mj_data, namespace)
        super().__init__(
            mj_data,
            {
                "base": base,
                "arm": B601ArmGroup(mj_data, base, namespace),
                "gripper": B601GripperGroup(mj_data, base, namespace),
            },
        )

    @property
    def name(self) -> str:
        return "rebot_b601"

    @property
    def base(self) -> B601BaseGroup:
        return self._move_groups["base"]
