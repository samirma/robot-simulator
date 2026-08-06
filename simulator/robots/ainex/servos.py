"""The AiNex's servo table: ids, the count<->radian mapping, limits and the home pose.

Transcribed from the vendor's own configuration so that a client speaking the AiNex's
bus-servo topics addresses the same joint by the same id with the same number here as on
the robot:

* `ainex_kinematics/config/servo_controller.yaml` -- id, `init`, `min`, `max` per joint;
* `ainex_kinematics/config/init_pose.yaml` -- the home pose, in radians;
* `ainex_kinematics/scripts/ainex_controller.py` -- `ENCODER_TICKS_PER_RADIAN` and
  `angle2pulse`/`pulse2angle`, which are the authority on how the two relate.

Pure: no MuJoCo, no ROS. `test_attach.py` checks it round-trips.
"""

from __future__ import annotations

import math

# ainex_controller.py:25, written out the way the vendor writes it:
#     ENCODER_TICKS_PER_RADIAN = 180 / 3.1415926 / 240 * 1000
# i.e. 1000 counts spanning the HX-series servo's 240 degrees of travel. ~238.73.
TICKS_PER_RADIAN = 180.0 / math.pi / 240.0 * 1000.0

# Electrical travel of a bus servo, in counts. Every joint in servo_controller.yaml has
# the same 0..1000 -- the per-joint difference is `init`, not the span.
COUNT_MIN = 0
COUNT_MAX = 1000

# The URDF gives every joint this same limit, written literally as `2.09`. That is not a
# per-joint calibration: it is half the servo's 240 degrees (2.094395 rad), rounded down
# to two decimals. It is still a real bound -- nothing travels further than the servo can
# -- so it is intersected with the servo-derived range rather than discarded. Being a
# hair tighter than the exact half-travel, it is what the symmetric joints end up with.
# See joint_limits().
URDF_PLACEHOLDER_LIMIT = 2.09


# joint name -> (servo id, init count, flipped)
#
# `init` is the count the joint's *zero radians* maps to (ainex_controller.py::angle2pulse
# adds the angle to it), so it is a per-joint offset and not a calibration constant. Four
# joints have a non-500 init, and those are exactly the four whose reachable range comes
# out asymmetric about the URDF zero.
#
# `flipped` is servo_controller.yaml's `min > max`, which is how the vendor encodes a
# servo mounted the other way round; ainex_controller.py negates the ticks coefficient for
# those. It is read off the transcription rather than decided here, so this table stays a
# plain copy of the yaml and can be diffed against it.
SERVOS: dict[str, tuple[int, int, bool]] = {
    # id, init, flipped                        left        right
    "l_ank_roll": (1, 500, False),  # noqa: E241   1           2
    "r_ank_roll": (2, 500, False),
    "l_ank_pitch": (3, 500, False),  #             3           4
    "r_ank_pitch": (4, 500, False),
    "l_knee": (5, 240, False),  #                  5           6
    "r_knee": (6, 760, False),
    "l_hip_pitch": (7, 500, False),  #             7           8
    "r_hip_pitch": (8, 500, False),
    "l_hip_roll": (9, 500, False),  #              9          10
    "r_hip_roll": (10, 500, False),
    "l_hip_yaw": (11, 500, False),  #             11          12
    "r_hip_yaw": (12, 500, False),
    "l_sho_pitch": (13, 875, True),  #            13          14   <- both flipped
    "r_sho_pitch": (14, 125, True),
    "l_sho_roll": (15, 500, False),  #            15          16
    "r_sho_roll": (16, 500, False),
    "l_el_pitch": (17, 500, False),  #            17          18
    "r_el_pitch": (18, 500, False),
    "l_el_yaw": (19, 500, False),  #              19          20
    "r_el_yaw": (20, 500, False),
    "l_gripper": (21, 500, False),  #             21          22
    "r_gripper": (22, 500, False),
    "head_pan": (23, 500, False),  #              23 (pan)    24 (tilt)
    "head_tilt": (24, 500, False),
}

# The vendor's `init_pose.yaml`, verbatim: the pose `move_to_init_pose` drives to at
# start-up and returns to after every action group. It is a slight crouch, not a straight
# stand -- hips and knees are already loaded, which is what the gait needs to swing from.
# Left and right are mirrored with opposite signs, which is the sign convention the whole
# file depends on.
INIT_POSE: dict[str, float] = {
    "l_ank_pitch": 0.625, "r_ank_pitch": -0.625,
    "l_ank_roll": 0.016, "r_ank_roll": -0.016,
    "l_hip_pitch": -0.828, "r_hip_pitch": 0.828,
    "l_hip_roll": 0.016, "r_hip_roll": -0.016,
    "l_hip_yaw": 0.00, "r_hip_yaw": 0.00,
    "l_knee": 1.192, "r_knee": -1.192,
    "l_sho_pitch": 0.00, "r_sho_pitch": 0.00,
    "l_sho_roll": 1.293, "r_sho_roll": -1.293,
    "l_el_pitch": -0.10, "r_el_pitch": 0.10,
    "l_el_yaw": -1.926, "r_el_yaw": 1.926,
    "l_gripper": 0.00, "r_gripper": 0.00,
    "head_pan": 0.00, "head_tilt": 0.00,
}

LEG_JOINTS = (
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
)
ARM_JOINTS = (
    "l_sho_pitch", "l_sho_roll", "l_el_pitch", "l_el_yaw",
    "r_sho_pitch", "r_sho_roll", "r_el_pitch", "r_el_yaw",
)
GRIPPER_JOINTS = ("l_gripper", "r_gripper")
HEAD_JOINTS = ("head_pan", "head_tilt")

# Servo ids in ascending order, which is the order /joint_states is published in and the
# order an action-group frame's 24 counts are stored in.
BY_ID: tuple[str, ...] = tuple(sorted(SERVOS, key=lambda n: SERVOS[n][0]))


def _coefficient(name: str) -> tuple[int, float]:
    """(offset, ticks-per-radian) for a joint, as ainex_controller builds it."""
    _, init, flipped = SERVOS[name]
    return init, (-TICKS_PER_RADIAN if flipped else TICKS_PER_RADIAN)


def angle_to_count(name: str, angle: float) -> int:
    """Radians -> servo count. ainex_controller.py::angle2pulse."""
    offset, ticks = _coefficient(name)
    return offset + int(round(angle * ticks))


def count_to_angle(name: str, count: float) -> float:
    """Servo count -> radians. ainex_controller.py::pulse2angle."""
    offset, ticks = _coefficient(name)
    return (count - offset) / ticks


def joint_limits(name: str) -> tuple[float, float]:
    """The reachable range of a joint, in radians.

    Two independent bounds, and neither source is authoritative on its own, so the
    intersection is used:

    * the **servo**, which can only reach counts 0..1000 -- and because zero radians sits
      at the joint's `init` count rather than mid-travel, that window is asymmetric for
      the four joints whose init is not 500. The knees get roughly -1.0..+3.2 rad, which
      is electrical travel, not something a knee can do;
    * the **URDF**, whose uniform +/-2.09 is a placeholder in the sense that it is not
      per-joint, but is a true statement about the mechanism: nothing travels further
      than the servo's own 240 degrees about its centre.

    Taking the tighter of the two per side gives a range both sources agree is reachable.
    For the twenty joints with `init` 500 that is the URDF's own +/-2.09; for `l_knee`,
    `r_knee`, `l_sho_pitch` and `r_sho_pitch` it is narrower on one side. Those four are
    the only externally visible sign that this table was applied at all, which is why
    test_attach.py checks for exactly them rather than for "some joint changed".
    """
    lo, hi = sorted((count_to_angle(name, COUNT_MIN), count_to_angle(name, COUNT_MAX)))
    return max(lo, -URDF_PLACEHOLDER_LIMIT), min(hi, URDF_PLACEHOLDER_LIMIT)


def all_limits() -> dict[str, tuple[float, float]]:
    return {name: joint_limits(name) for name in SERVOS}


def clamp(name: str, angle: float) -> float:
    lo, hi = joint_limits(name)
    return min(max(angle, lo), hi)
