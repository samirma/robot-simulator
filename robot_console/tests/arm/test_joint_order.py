"""``/joint_states`` arrives alphabetically sorted. Index it by name, never by position.

``INTERFACE.md`` records the simulator's actual ``name`` array:

    ['elbow_flex_joint', 'gripper_joint', 'shoulder_lift_joint',
     'shoulder_pan_joint', 'wrist_flex_joint', 'wrist_roll_joint']

while the contract's action-vector order is ``[shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper]``. The two agree on **no** index,
so any positional read is wrong on every joint. These tests pin the resolution
path that the whole stack depends on.
"""

from __future__ import annotations

import numpy as np
from inspect_robots_ros._msgs import parse_joint_state
from robot_console.arm.kinematics import JOINT_ORDER


#: What the simulator's `/joint_states` carries: the contract joints, sorted, which is what
#: a real `joint_state_broadcaster` returns and what `shared/ros_surfaces/so101.py`
#: reproduces on purpose.
JOINT_STATE_NAMES: tuple[str, ...] = tuple(sorted(JOINT_ORDER))


def _joint_state_msg(stamp, positions, velocities):
    """The message shape `shared/ros_surfaces/so101.py` puts on the wire."""
    by_name = dict(zip(JOINT_ORDER, positions, strict=True))
    vel_by_name = dict(zip(JOINT_ORDER, velocities, strict=True))
    return {
        "header": {"stamp": {"sec": int(stamp), "nanosec": 0}, "frame_id": ""},
        "name": list(JOINT_STATE_NAMES),
        "position": [float(by_name[n]) for n in JOINT_STATE_NAMES],
        "velocity": [float(vel_by_name[n]) for n in JOINT_STATE_NAMES],
        "effort": [],
    }
from robot_console.arm.ros_settings import RosSettings

#: Verbatim from INTERFACE.md, "Names that are easy to get wrong".
AS_BUILT_NAMES = [
    "elbow_flex_joint",
    "gripper_joint",
    "shoulder_lift_joint",
    "shoulder_pan_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]


def test_the_two_orders_share_no_index() -> None:
    assert AS_BUILT_NAMES != list(JOINT_ORDER)
    assert all(a != b for a, b in zip(AS_BUILT_NAMES, JOINT_ORDER, strict=True))


def test_joint_state_is_resolved_by_name_into_contract_order() -> None:
    # One distinct value per joint, so a positional read cannot pass by luck.
    positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    msg = {"name": AS_BUILT_NAMES, "position": positions}
    by_name = dict(zip(AS_BUILT_NAMES, positions, strict=True))
    resolved = parse_joint_state(msg, JOINT_ORDER)
    assert list(resolved) == [by_name[name] for name in JOINT_ORDER]
    assert list(resolved) != positions


def test_the_adapter_asks_for_the_arm_joints_then_the_gripper() -> None:
    settings = RosSettings()
    requested = (*settings.joints, settings.gripper_joint)
    assert requested == JOINT_ORDER


def test_the_fake_simulator_sorts_its_joint_names_like_the_real_one() -> None:
    assert list(JOINT_STATE_NAMES) == AS_BUILT_NAMES
    values = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    msg = _joint_state_msg(1.0, values, np.zeros(6))
    assert msg["name"] == AS_BUILT_NAMES
    # And the values still round-trip back to contract order through the parser.
    assert list(parse_joint_state(msg, JOINT_ORDER)) == list(values)
