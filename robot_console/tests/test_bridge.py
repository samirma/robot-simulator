import math

import pytest

from robot_console.bridge import Odom, parse_odom, quaternion_yaw, wrap_angle
from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TYPE_COMPRESSED_IMAGE,
    TYPE_ODOM,
    TYPE_TWIST,
    normalise,
)


def odom_message(x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, wz=0.0):
    return {
        "header": {"seq": 1, "stamp": {"secs": 10, "nsecs": 500_000_000}, "frame_id": "odom"},
        "child_frame_id": "base_footprint",
        "pose": {
            "pose": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)},
            },
            "covariance": [0.0] * 36,
        },
        "twist": {
            "twist": {"linear": {"x": vx, "y": vy, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": wz}},
            "covariance": [0.0] * 36,
        },
    }


def test_topic_names_match_the_myagv_contract():
    assert (TOPIC_CMD_VEL, TYPE_TWIST) == ("/cmd_vel", "geometry_msgs/Twist")
    assert (TOPIC_ODOM, TYPE_ODOM) == ("/odom", "nav_msgs/Odometry")
    assert (TOPIC_CAMERA, TYPE_COMPRESSED_IMAGE) == (
        "/camera/image_raw/compressed",
        "sensor_msgs/CompressedImage",
    )


def test_normalise():
    assert normalise("cmd_vel") == "/cmd_vel"
    assert normalise("/cmd_vel") == "/cmd_vel"


@pytest.mark.parametrize("yaw", [0.0, 0.5, -0.5, math.pi / 2, -math.pi / 2, 2.9, -2.9])
def test_quaternion_yaw_round_trip(yaw):
    got = quaternion_yaw(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    assert wrap_angle(got - yaw) == pytest.approx(0.0, abs=1e-9)


def test_quaternion_yaw_handles_roll_and_pitch():
    # A real myAGV's odometry is not perfectly planar, unlike the simulator's.
    roll = 0.05
    yaw = 1.0
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # Roll-yaw composition with zero pitch.
    got = quaternion_yaw(sr * cy, sr * sy, cr * sy, cr * cy)
    assert wrap_angle(got - yaw) == pytest.approx(0.0, abs=1e-6)


def test_parse_odom():
    odom = parse_odom(odom_message(x=1.5, y=-2.0, yaw=1.0, vx=0.15, vy=-0.05, wz=0.3))
    assert odom.x == pytest.approx(1.5)
    assert odom.y == pytest.approx(-2.0)
    assert odom.yaw == pytest.approx(1.0)
    assert odom.vx == pytest.approx(0.15)
    assert odom.vy == pytest.approx(-0.05)
    assert odom.wz == pytest.approx(0.3)
    assert odom.stamp == pytest.approx(10.5)
    # The frame pair myagv_odometry_node publishes, and what robot_pose_ekf expects.
    assert odom.frame_id == "odom"
    assert odom.child_frame_id == "base_footprint"


def test_parse_odom_tolerates_missing_fields():
    assert parse_odom({}) == Odom(frame_id="", child_frame_id="")
    assert parse_odom({"pose": None, "twist": 5}).x == 0.0
    # An absent orientation must read as identity, not as a divide-by-zero.
    assert parse_odom({"pose": {"pose": {}}}).yaw == pytest.approx(0.0)


def test_wrap_angle():
    # Half a turn lands on either end of the range, depending on float rounding.
    assert abs(wrap_angle(3 * math.pi)) == pytest.approx(math.pi)
    assert abs(wrap_angle(-3 * math.pi)) == pytest.approx(math.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(2 * math.pi + 0.5) == pytest.approx(0.5)
    assert wrap_angle(-2 * math.pi - 0.5) == pytest.approx(-0.5)
