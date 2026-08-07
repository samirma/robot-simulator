"""Move groups and robot view for the AiNex.

Seven groups, which is more than any other robot here, because the AiNex has more
independently commanded limbs than any other robot here:

    base            the virtual planar joints the torso rides on
    legs            all 12 leg joints, driven by gait.py rather than by a planner
    left_arm        4 joints, shoulder to elbow
    right_arm       4 joints
    left_gripper    1 joint
    right_gripper   1 joint
    head            pan and tilt -- and the camera rides on it (see ainex.py step 4)

The arms are `SimplyActuatedMoveGroup` with a TCP frame, and the grippers are
`GripperGroup`, so the usual MolmoSpaces tooling can address them. But note what is
deliberately *not* here: the real AiNex exposes no inverse-kinematics service and no
Cartesian arm interface at all. Its manipulation is replayed servo trajectories
(`actions.py`). The TCP frames exist so a grasp can be measured and reasoned about, not
because the hardware offers a Cartesian command.

The legs are a `SimplyActuatedMoveGroup` too, with no TCP: nothing plans for them, and
`ros_surface.py` writes their joint targets directly from the gait.
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

from .servos import ARM_JOINTS, HEAD_JOINTS, LEG_JOINTS

BASE_AXES = ("x", "y", "theta")
TORSO_BODY = "body_link"

LEFT_ARM_JOINTS = tuple(j for j in ARM_JOINTS if j.startswith("l_"))
RIGHT_ARM_JOINTS = tuple(j for j in ARM_JOINTS if j.startswith("r_"))


class AiNexBaseGroup(HoloJointsRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        super().__init__(
            mj_data,
            model.site(f"{namespace}world").id,
            model.site(f"{namespace}base_site").id,
            [model.joint(f"{namespace}base_{axis}").id for axis in BASE_AXES],
            [model.actuator(f"{namespace}base_{axis}_act").id for axis in BASE_AXES],
            model.body(f"{namespace}{TORSO_BODY}").id,
        )


class _JointGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """A plain set of position-controlled joints rooted at some body.

    The actuator for each joint is named after the joint, which is what `ainex.py`
    creates -- so a client that says "servo 5" and a move group that says `l_knee` are
    addressing the same thing by construction.
    """

    joints: tuple[str, ...] = ()
    root_body: str = TORSO_BODY
    leaf_site: str | None = None

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}{n}").id for n in self.joints]
        act_ids = [model.actuator(f"{namespace}{n}").id for n in self.joints]
        self._root_id = model.body(f"{namespace}{self.root_body}").id
        self._leaf_id = (
            model.site(f"{namespace}{self.leaf_site}").id if self.leaf_site else self._root_id
        )
        super().__init__(mj_data, joint_ids, act_ids, self._root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._leaf_id

    @property
    def leaf_frame_type(self):
        return "site" if self.leaf_site else "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_id)


class AiNexLegsGroup(_JointGroup):
    joints = LEG_JOINTS


class AiNexLeftArmGroup(_JointGroup):
    joints = LEFT_ARM_JOINTS
    leaf_site = "l_tcp"


class AiNexRightArmGroup(_JointGroup):
    joints = RIGHT_ARM_JOINTS
    leaf_site = "r_tcp"


class AiNexHeadGroup(_JointGroup):
    joints = HEAD_JOINTS


class AiNexGripperGroup(MJCFFrameMixin, GripperGroup):
    """One hinged claw per hand.

    A single joint, like the SO-101's jaw and unlike a parallel-jaw gripper, so
    `inter_finger_dist` is measured between the claw tip and the palm it closes against
    rather than between two symmetric fingers. Which direction is "open" is read off the
    joint's own range at construction, not hardcoded -- see `ainex.py`, where the range
    comes from the vendor servo table.
    """

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, side: str, namespace: str = ""
                 ) -> None:
        model = mj_data.model
        self._namespace = namespace
        self._side = side
        joint = f"{namespace}{side}_gripper"
        super().__init__(
            mj_data,
            [model.joint(joint).id],
            [model.actuator(joint).id],
            model.body(f"{namespace}{side}_gripper_link").id,
            base,
        )
        self._tcp_site_id = model.site(f"{namespace}{side}_tcp").id
        self._tip_geom_id = model.geom(f"{namespace}{side}_claw_tip").id
        self._palm_geom_id = model.geom(f"{namespace}{side}_claw_palm").id
        rng = model.jnt_range[model.joint(joint).id]
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
        return self._span

    @property
    def inter_finger_dist(self) -> float:
        dist = mujoco.mj_geomDistance(
            self.mj_model, self.mj_data, self._tip_geom_id, self._palm_geom_id, 0.5, None
        )
        return max(0.0, dist)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class AiNexRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = AiNexBaseGroup(mj_data, namespace)
        super().__init__(
            mj_data,
            {
                "base": base,
                "legs": AiNexLegsGroup(mj_data, base, namespace),
                "left_arm": AiNexLeftArmGroup(mj_data, base, namespace),
                "right_arm": AiNexRightArmGroup(mj_data, base, namespace),
                "left_gripper": AiNexGripperGroup(mj_data, base, "l", namespace),
                "right_gripper": AiNexGripperGroup(mj_data, base, "r", namespace),
                "head": AiNexHeadGroup(mj_data, base, namespace),
            },
        )

    @property
    def name(self) -> str:
        return "ainex"

    @property
    def base(self) -> AiNexBaseGroup:
        return self._move_groups["base"]
