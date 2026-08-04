"""Move groups and robot view for LeKiwi.

LeKiwi (SIGRobotics / Ekumen) is an SO-ARM100 arm on a three-wheel kiwi-drive base —
so it is exactly the combination of the two robots already here: `myagv`'s holonomic
planar base and `so101`'s arm-plus-jaw.

Base: as with `myagv`, the three omni wheels are decorative and motion comes from
virtual slide-X / slide-Y / hinge-Z joints via `HoloJointsRobotBaseGroup`. See
`lekiwi.py` for why the real wheel joints are removed.

Arm: the Ekumen model embeds SO-ARM100 (the SO-101's predecessor) with joints
`Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll` plus a `Jaw` gripper.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

BASE_AXES = ("x", "y", "theta")
ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
GRIPPER_JOINT = "Jaw"

# Measured from the model by sweeping the Jaw joint; see lekiwi.py::measure_gripper.
INTER_FINGER_DIST_RANGE = (0.0, 0.09)


class LeKiwiBaseGroup(HoloJointsRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        super().__init__(
            mj_data,
            model.site(f"{namespace}world").id,
            model.site(f"{namespace}base_site").id,
            [model.joint(f"{namespace}base_{axis}").id for axis in BASE_AXES],
            [model.actuator(f"{namespace}base_{axis}_act").id for axis in BASE_AXES],
            model.body(f"{namespace}base_plate_layer_1_link").id,
        )


class LeKiwiArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}{n}").id for n in ARM_JOINTS]
        act_ids = [model.actuator(f"{namespace}{n}").id for n in ARM_JOINTS]
        self._arm_root_id = model.body(f"{namespace}Base").id
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


class LeKiwiGripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        super().__init__(
            mj_data,
            [model.joint(f"{namespace}{GRIPPER_JOINT}").id],
            [model.actuator(f"{namespace}{GRIPPER_JOINT}").id],
            model.body(f"{namespace}Fixed_Jaw").id,
            base,
        )
        self._tcp_site_id = model.site(f"{namespace}tcp").id
        self._fixed_geom_id = model.geom(f"{namespace}fixed_jaw_tip").id
        self._moving_geom_id = model.geom(f"{namespace}moving_jaw_tip").id
        rng = model.jnt_range[model.joint(f"{namespace}{GRIPPER_JOINT}").id]
        self._open_ctrl, self._closed_ctrl = float(rng[1]), float(rng[0])

    @property
    def leaf_frame_id(self) -> int:
        return self._tcp_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = [self._open_ctrl if open else self._closed_ctrl]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return INTER_FINGER_DIST_RANGE

    @property
    def inter_finger_dist(self) -> float:
        dist = mujoco.mj_geomDistance(
            self.mj_model, self.mj_data, self._fixed_geom_id, self._moving_geom_id, 0.5, None
        )
        return max(0.0, dist)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class LeKiwiRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = LeKiwiBaseGroup(mj_data, namespace)
        super().__init__(
            mj_data,
            {
                "base": base,
                "arm": LeKiwiArmGroup(mj_data, base, namespace),
                "gripper": LeKiwiGripperGroup(mj_data, base, namespace),
            },
        )

    @property
    def name(self) -> str:
        return "lekiwi"

    @property
    def base(self) -> LeKiwiBaseGroup:
        return self._move_groups["base"]
