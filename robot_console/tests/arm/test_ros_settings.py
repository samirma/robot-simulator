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
# Workspace-owned (spec §3).
SCENE_TOPICS = {
    "/overhead/color/compressed": "sensor_msgs/msg/CompressedImage",
    "/side/color/compressed": "sensor_msgs/msg/CompressedImage",
}
WORKSPACE_SERVICES = {"/reset"}


def test_every_configured_name_is_in_the_official_interface() -> None:
    settings = rs.RosSettings()
    assert {settings.command_topic, settings.joint_states_topic} <= set(OFFICIAL_TOPICS)
    assert settings.gripper_topic in OFFICIAL_ACTIONS
    cameras = {settings.camera_topic, *(topic for _, topic, _, _ in settings.extra_cameras)}
    assert cameras <= set(SCENE_TOPICS)
    assert rs.WRIST_CAMERA_TOPIC in OFFICIAL_TOPICS
    assert rs.GRIPPER_ACTION_TYPE == OFFICIAL_ACTIONS[rs.GRIPPER_ACTION]
    assert (rs.TF_TOPIC, rs.TF_STATIC_TOPIC) == ("/tf", "/tf_static")


def test_the_success_topic_is_deliberately_not_consumed() -> None:
    """The episode is graded from the camera, so nothing subscribes to a success topic."""
    assert not hasattr(rs, "TASK_SUCCESS_TOPIC")
    assert not any("success" in name for name in rs.RosSettings().base_kwargs())


def test_the_reset_service_is_the_workspaces() -> None:
    assert rs.RESET_SERVICE in WORKSPACE_SERVICES
    assert rs.RosSettings().reset_service == rs.RESET_SERVICE
    assert not hasattr(rs, "RESET_WORLD_SERVICE")


def test_the_gripper_is_the_parallel_gripper_action() -> None:
    settings = rs.RosSettings()
    assert settings.gripper_mode == "action"
    assert settings.gripper_topic == "/gripper_controller/gripper_cmd"
    # The base adapter still builds a Float64MultiArray; the client turns it into a goal.
    assert settings.base_kwargs()["gripper_command_type"] == "float64_multi_array"
    assert not hasattr(rs, "GRIPPER_COMMAND_TOPIC")


def test_the_client_sends_a_gripper_publish_as_an_action_goal() -> None:
    from robot_console.arm.ros_client import HeaderStampingClient

    sent: list = []
    client = HeaderStampingClient("ws://127.0.0.1:1",
                                  gripper_actions={rs.GRIPPER_ACTION: rs.GRIPPER_ACTION_TYPE})
    client._send = sent.append  # type: ignore[method-assign]
    client.advertise(rs.GRIPPER_ACTION, message_type="std_msgs/msg/Float64MultiArray")
    client.publish(rs.GRIPPER_ACTION, {"data": [0.42]})
    assert len(sent) == 1
    goal = sent[0]
    assert goal["op"] == "send_action_goal"
    assert (goal["action"], goal["action_type"]) == (rs.GRIPPER_ACTION, rs.GRIPPER_ACTION_TYPE)
    assert goal["args"]["command"]["position"] == [0.42]


def test_an_unknown_gripper_mode_is_rejected_with_a_reason() -> None:
    with pytest.raises(ValueError, match="GripperActionController"):
        rs.RosSettings(gripper_mode="topic")


def test_the_default_cameras_are_the_only_published_pair() -> None:
    # 640x480, the rig's constants; the upstream adapter validates the first frame
    # against this, so it must track.
    #
    # overhead+side are the worktop rig, and the default pair a policy is handed.
    # Order is load-bearing: MolmoAct2 consumes views positionally.
    # They come out under the *scene's* namespace, not the arm's: they watch the
    # work surface and would still be there with the arm unbolted. The view
    # *names* are slot labels a policy matches on and are never prefixed.
    assert rs.RosSettings().cameras() == {
        "overhead": ("/scene/overhead/color/compressed", 480, 640),
        "side": ("/scene/side/color/compressed", 480, 640),
    }
    cameras = rs.RosSettings().cameras()
    assert list(cameras) == ["overhead", "side"]
    assert rs.RosSettings().base_kwargs()["cameras"] == cameras


def test_the_scene_rig_keeps_its_namespace_whatever_the_robot_is_called() -> None:
    """The rig is not the robot's, so the robot's namespace does not reach it.

    This is the half that would fail silently: a scene camera composed under the arm's
    name subscribes to a topic nobody publishes, and rosbridge answers a subscription to
    a topic nobody publishes by simply never sending anything.
    """
    for namespace in ("so101", "", "robot_2"):
        cameras = rs.settings_for_views(
            ("overhead", "side", "wrist"), namespace=namespace
        ).cameras()
        assert cameras["overhead"][0] == "/scene/overhead/color/compressed"
        assert cameras["side"][0] == "/scene/side/color/compressed"
        # ...while the arm's own camera follows the arm, wherever it is.
        assert cameras["wrist"][0] == rs.namespaced(
            rs.WRIST_CAMERA_TOPIC, namespace
        )


def test_the_overhead_side_pair_is_wirable_explicitly() -> None:
    # Naming the default pair field by field must give the same wiring as the
    # dataclass default, so a caller that spells it out does not drift from it.
    settings = rs.RosSettings(
        camera_name=rs.OVERHEAD_CAMERA_NAME,
        camera_topic=rs.OVERHEAD_CAMERA_TOPIC,
        extra_cameras=((rs.SIDE_CAMERA_NAME, rs.SIDE_CAMERA_TOPIC, 640, 480),),
    )
    assert list(settings.cameras()) == ["overhead", "side"]
    assert settings.cameras()["side"] == ("/scene/side/color/compressed", 480, 640)


def test_extra_cameras_append_in_declaration_order() -> None:
    settings = rs.RosSettings(
        camera_name=rs.OVERHEAD_CAMERA_NAME,
        camera_topic=rs.OVERHEAD_CAMERA_TOPIC,
        extra_cameras=((rs.SIDE_CAMERA_NAME, rs.SIDE_CAMERA_TOPIC, 640, 480),),
    )
    # Order is load-bearing: MolmoAct2 consumes views positionally.
    assert list(settings.cameras()) == ["overhead", "side"]


def test_a_duplicate_camera_name_is_refused() -> None:
    # The default primary is ``overhead``, so an extra camera of that name
    # collides with it.
    settings = rs.RosSettings(extra_cameras=(("overhead", "/other", 640, 480),))
    with pytest.raises(ValueError, match="duplicate camera name"):
        settings.cameras()


def test_camera_can_be_disabled() -> None:
    # ``extra_cameras`` carries a default (slot 1, ``side``), so honouring only
    # ``camera_topic=None`` would leave that slot subscribed and a "run without
    # images" caller would still block ``obs_timeout_s`` on a camera it asked
    # not to have. Clearing the primary therefore clears every camera.
    assert rs.RosSettings(camera_topic=None, extra_cameras=()).cameras() == {}
    assert rs.RosSettings(camera_topic=None).cameras() == {}
    assert rs.RosSettings(camera_topic=None).extra_cameras == ()


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
