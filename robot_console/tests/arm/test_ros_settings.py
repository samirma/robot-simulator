"""The ROS names must match ``simulator/INTERFACE.md``, which is the as-built truth.

Each expected string below is copied from the ``ros2 topic list -t`` /
``ros2 service list -t`` output recorded in that file, not from ``CONTRACT.md``
and not from memory. Four of them were wrong before: the free-joint topic was
missing its plugin namespace, the reset service named the wrong node, the
gripper was configured as an action, and there was no camera at all.
"""

from __future__ import annotations

import pytest

from robot_console.arm import ros_settings as rs

# Verbatim from INTERFACE.md section 6, "ros2 topic list -t".
AS_BUILT_TOPICS = {
    "/joint_trajectory_controller/joint_trajectory": "trajectory_msgs/msg/JointTrajectory",
    "/gripper_controller/commands": "std_msgs/msg/Float64MultiArray",
    "/joint_states": "sensor_msgs/msg/JointState",
    "/free_joint_publisher/free_joint_states": (
        "mujoco_ros2_control_msgs/msg/FreeJointStateArray"
    ),
    "/task_success": "std_msgs/msg/Bool",
    # The only two camera topics the container publishes. The `trainlow`,
    # `trainhigh`, `policylow` and `policyhigh` cameras were deleted from the
    # scene on 2026-08-31, so their topics are gone from the wire and from
    # here; `overhead` and `side` were untouched by that change.
    "/overhead/color/compressed": "sensor_msgs/msg/CompressedImage",
    "/side/color/compressed": "sensor_msgs/msg/CompressedImage",
}
AS_BUILT_SERVICES = {
    "/mujoco_ros2_control_node/reset_world",
    "/reset",
}


def test_every_configured_topic_exists_in_the_running_container() -> None:
    settings = rs.RosSettings()
    configured = {
        settings.command_topic,
        settings.gripper_topic,
        settings.joint_states_topic,
        settings.object_state_topic,
        settings.success_topic,
        settings.camera_topic,
        # extra_cameras carries slot 1 now, so it has to be checked too: a
        # mis-typed second view is exactly the failure this test exists for.
        *(topic for _, topic, _, _ in settings.extra_cameras),
    }
    assert configured <= set(AS_BUILT_TOPICS)


def test_free_joint_topic_carries_its_plugin_namespace() -> None:
    # The plugin key in config/mujoco_plugins.yaml becomes a sub-namespace.
    assert rs.FREE_JOINT_STATES_TOPIC == "/free_joint_publisher/free_joint_states"


def test_reset_service_names_the_mujoco_node() -> None:
    assert rs.RESET_WORLD_SERVICE in AS_BUILT_SERVICES
    assert rs.RESET_WORLD_SERVICE.startswith("/mujoco_ros2_control_node/")
    assert rs.TASK_MANAGER_RESET_SERVICE in AS_BUILT_SERVICES


def test_gripper_is_a_topic_not_an_action() -> None:
    settings = rs.RosSettings()
    assert settings.gripper_mode == "topic"
    assert settings.gripper_topic == "/gripper_controller/commands"
    kwargs = settings.base_kwargs()
    assert kwargs["gripper_command_type"] == "float64_multi_array"
    assert not any(key.startswith("gripper_action") for key in kwargs)
    # The sentinel-topic + send_action_goal indirection is gone entirely.
    for gone in ("GRIPPER_ACTION", "GRIPPER_ACTION_TYPE", "GRIPPER_SENTINEL_TOPIC"):
        assert not hasattr(rs, gone), f"{gone} should have been deleted with the action hack"
    assert not hasattr(settings, "gripper_action")


def test_the_action_gripper_client_is_gone() -> None:
    import robot_console.arm.embodiment as emb

    assert not hasattr(emb, "ActionGripperClient")


def test_action_gripper_mode_is_rejected_with_a_reason() -> None:
    with pytest.raises(ValueError, match="ForwardCommandController"):
        rs.RosSettings(gripper_mode="action")


def test_the_default_cameras_are_the_only_published_pair() -> None:
    # 640x480, measured on the wire 2026-08-30. The overhead camera was 1280x720
    # when INTERFACE.md was written; the simulator changed it, and the upstream
    # adapter validates the first frame against this, so it must track.
    #
    # overhead+side are the only cameras the simulator still publishes, so the
    # default is the only pair that can be subscribed at all. Order is
    # load-bearing: MolmoAct2 consumes views positionally.
    cameras = rs.RosSettings().cameras()
    assert cameras == {
        "overhead": ("/overhead/color/compressed", 480, 640),
        "side": ("/side/color/compressed", 480, 640),
    }
    assert list(cameras) == ["overhead", "side"]
    assert rs.RosSettings().base_kwargs()["cameras"] == cameras


def test_the_overhead_side_pair_is_wirable_explicitly() -> None:
    # Naming the default pair field by field must give the same wiring as the
    # dataclass default, so a caller that spells it out does not drift from it.
    settings = rs.RosSettings(
        camera_name=rs.OVERHEAD_CAMERA_NAME,
        camera_topic=rs.OVERHEAD_CAMERA_TOPIC,
        extra_cameras=((rs.SIDE_CAMERA_NAME, rs.SIDE_CAMERA_TOPIC, 640, 480),),
    )
    assert list(settings.cameras()) == ["overhead", "side"]
    assert settings.cameras()["side"] == ("/side/color/compressed", 480, 640)


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
