"""Move groups and robot view for the Elephant Robotics myAGV Pi 2023.

The myAGV is a wheeled mobile base with no arm: a single `base` move group, driven by
three virtual holonomic joints (slide-X, slide-Y, hinge-Z) that stand in for the real
Mecanum drive. See make_model.py for why the wheels are not simulated as rolling
contacts.

This reuses MolmoSpaces' `HoloJointsRobotBaseGroup` unchanged — the same class RB-Y1's
base uses — which requires exactly the joint order (x, y, theta) and a `world` site to
measure against.
"""

from __future__ import annotations

from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    HoloJointsRobotBaseGroup,
    RobotView,
)

BASE_AXES = ("x", "y", "theta")


class MyAGVBaseGroup(HoloJointsRobotBaseGroup):
    """Holonomic planar base: x and y translation plus yaw, all world-aligned.

    Unlike a differential-drive base, y is directly commandable — that is the Mecanum
    capability, and it is why teleop can strafe.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        world_site_id = model.site(f"{namespace}world").id
        base_site_id = model.site(f"{namespace}base_site").id
        joint_ids = [model.joint(f"{namespace}base_{axis}").id for axis in BASE_AXES]
        act_ids = [model.actuator(f"{namespace}base_{axis}_act").id for axis in BASE_AXES]
        root_body_id = model.body(f"{namespace}base").id
        super().__init__(mj_data, world_site_id, base_site_id, joint_ids, act_ids, root_body_id)


class MyAGVRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = MyAGVBaseGroup(mj_data, namespace)
        super().__init__(mj_data, {"base": base})

    @property
    def name(self) -> str:
        return "myagv"

    @property
    def base(self) -> MyAGVBaseGroup:
        return self._move_groups["base"]
