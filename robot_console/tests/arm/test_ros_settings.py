"""The ROS names must be the SO-101's official interface (``robots_specs/so101/ros2.yml``).

Each expected string below is copied from that file (bare, as a single bringup
presents it), or is the workspace-owned ``/reset`` and ``/scene`` rig of the
simulator's spec §3 -- not from memory.
"""

from __future__ import annotations

import pytest

from robot_console.arm import ros_settings as rs

# From robots_specs/so101/ros2.yml, `topics`.
OFFICIAL_TOPICS = {
    "/joint_trajectory_controller/joint_trajectory": "trajectory_msgs/msg/JointTrajectory",
    "/joint_states": "sensor_msgs/msg/JointState",
    "/wrist/image_raw/compressed": "sensor_msgs/msg/CompressedImage",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}
# From ros2.yml, `actions`.
OFFICIAL_ACTIONS = {
    "/gripper_controller/gripper_cmd": "control_msgs/action/ParallelGripperCommand",
}
# Workspace-owned (spec §3), under /scene.
RIG = {
    "/scene/overhead/color/compressed": "sensor_msgs/msg/CompressedImage",
    "/scene/side/color/compressed": "sensor_msgs/msg/CompressedImage",
    "/scene/overhead/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/scene/side/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/scene/tf_static": "tf2_msgs/msg/TFMessage",
}


def test_the_arm_interface_is_the_official_one_plus_reset() -> None:
    bare = rs.arm_interface("")
    assert bare["topics"] == OFFICIAL_TOPICS
    assert bare["actions"] == OFFICIAL_ACTIONS
    assert bare["services"] == {"/reset": "std_srvs/srv/Trigger"}
    named = rs.arm_interface("so101")
    assert set(named["topics"]) == {"/so101" + t for t in OFFICIAL_TOPICS}
    assert named["services"] == {"/so101/reset": "std_srvs/srv/Trigger"}


def test_the_rig_is_under_scene_whatever_the_robot_is_called() -> None:
    assert rs.rig_interface() == RIG
    for namespace in ("so101", "", "robot_2"):
        cameras = rs.RosSettings(namespace=namespace).cameras()
        assert cameras["overhead"][0] == "/scene/overhead/color/compressed"
        assert cameras["side"][0] == "/scene/side/color/compressed"
        assert cameras["wrist"][0] == rs.namespaced(rs.WRIST_CAMERA_TOPIC, namespace)


def test_no_success_topic_and_no_engine_names() -> None:
    assert not hasattr(rs, "TASK_SUCCESS_TOPIC")
    assert not hasattr(rs, "FREE_JOINT_STATES_TOPIC")
    assert not any("success" in name for name in rs.RosSettings().base_kwargs())


def test_the_default_views_are_overhead_side_wrist_in_that_order() -> None:
    cameras = rs.RosSettings().cameras()
    assert cameras == {
        "overhead": ("/scene/overhead/color/compressed", 480, 640),
        "side": ("/scene/side/color/compressed", 480, 640),
        "wrist": ("/so101/wrist/image_raw/compressed", 480, 640),
    }
    assert list(cameras) == ["overhead", "side", "wrist"]


def test_the_rig_views_are_always_subscribed_because_the_scorer_needs_them() -> None:
    assert list(rs.RosSettings(views=("wrist",)).cameras()) == ["wrist", "overhead", "side"]


def test_unknown_and_duplicate_views_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown"):
        rs.RosSettings(views=("overhead", "trainlow"))
    with pytest.raises(ValueError, match="duplicate"):
        rs.RosSettings(views=("overhead", "overhead"))


def test_every_rig_frame_is_delivered_not_one_per_step() -> None:
    assert rs.RosSettings().base_kwargs()["camera_throttle_ms"] == 0


def test_action_bounds_follow_the_contract_joint_order() -> None:
    settings = rs.RosSettings()
    assert settings.joints == (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_flex_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    )
    assert len(settings.action_low) == len(settings.action_high) == 5
    assert all(low < high for low, high in zip(settings.action_low, settings.action_high))
