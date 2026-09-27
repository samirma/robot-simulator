"""The arm's names, used consistently by the console -- with no simulator checkout.

The console's copies of the SO-101 and rig contract (``ros_settings``, ``kinematics``,
``task``, ``vision_success``) are held equal to the simulator's contract modules and to
``robots_specs/so101/ros2.yml`` by the workspace tests in ``../tests/`` (console spec §4,
Contract parity), which read both source trees. This file checks only what the console
alone can: which names its modules read, and how it composes them onto the wire.

The one that matters most is the joint order. ``/joint_states`` comes back alphabetically
sorted, and for this arm the sorted order and the contract order share *no* index -- so a
positional read is wrong about every joint while looking entirely plausible.
"""

from __future__ import annotations

from robot_console.arm.kinematics import JOINT_ORDER
from robot_console.arm.ros_settings import (
    OVERHEAD_CAMERA_TOPIC,
    SCENE_NAMESPACE,
    SIDE_CAMERA_NAME,
    SIDE_CAMERA_TOPIC,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_NAME,
    WRIST_CAMERA_TOPIC,
    WRIST_CAMERA_WIDTH,
    CAMERA_SPECS,
)
from robot_console.topics import namespaced


def test_the_console_reads_nothing_outside_the_official_interface() -> None:
    """No free-joint stream, no engine reset: every name is the robot's, `/reset` or the rig's."""
    import inspect as _inspect

    from robot_console.arm import embodiment, preflight, ros_settings, scorer, vision_success

    for module in (embodiment, preflight, ros_settings, scorer, vision_success):
        source = _inspect.getsource(module)
        assert "free_joint" not in source
        assert "reset_world" not in source


def test_the_camera_sizes_are_contract_terms() -> None:
    """The VLA's preprocessor stretches to 4:3 without preserving aspect, which is why the
    two scene cameras are 640x480; the wrist is usb_cam 0.8.1 at its default 640x480."""
    sizes = {topic: (w, h) for topic, w, h in CAMERA_SPECS.values()}
    assert sizes[OVERHEAD_CAMERA_TOPIC] == sizes[SIDE_CAMERA_TOPIC] == (640, 480)
    assert sizes[WRIST_CAMERA_TOPIC] == (WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT) == (640, 480)


def test_the_sorted_wire_order_shares_no_index_with_the_contract_order() -> None:
    """Which is why this is worth a test on both sides rather than a comment."""
    wire = sorted(JOINT_ORDER)
    assert wire != list(JOINT_ORDER)
    assert not any(a == b for a, b in zip(wire, JOINT_ORDER))


def test_the_scene_rig_is_not_the_robots() -> None:
    """`/so101/*` holds what the SO-101 presents, and an overhead view of the room is not."""
    from robot_console.arm.ros_settings import RosSettings

    subscribed = {
        spec[0]
        for spec in RosSettings(
            namespace="so101", views=("overhead", SIDE_CAMERA_NAME, WRIST_CAMERA_NAME),
        ).cameras().values()
    }
    assert subscribed == {namespaced(OVERHEAD_CAMERA_TOPIC, SCENE_NAMESPACE),
                          namespaced(SIDE_CAMERA_TOPIC, SCENE_NAMESPACE),
                          namespaced(WRIST_CAMERA_TOPIC, "so101")}
    assert not any(t.startswith("/so101/") for t in subscribed if "wrist" not in t)


def test_the_arm_settings_put_the_namespace_on_every_wire_name() -> None:
    """What the embodiment actually subscribes and publishes, end to end.

    The dataclass fields stay bare -- they are the transcript of `ros2 topic list -t`
    inside the reference container -- and the prefix is applied where they are handed to
    the adapter.
    """
    from robot_console.arm.ros_settings import RosSettings

    settings = RosSettings(namespace="so101")
    kwargs = settings.base_kwargs()
    assert kwargs["joint_states_topic"] == "/so101/joint_states"
    assert kwargs["command_topic"] == "/so101/joint_trajectory_controller/joint_trajectory"
    assert kwargs["gripper_topic"] == "/so101/gripper_controller/gripper_cmd"
    assert kwargs["reset_service"] == "/so101/reset"
    assert list(kwargs["cameras"]) == ["overhead", "side", "wrist"]
    assert kwargs["cameras"]["wrist"][0] == "/so101/wrist/image_raw/compressed"
    assert set(settings.camera_info_topics().values()) == {
        "/scene/overhead/color/camera_info", "/scene/side/color/camera_info"}
    # The rig's own namespace, not the arm's.
    assert kwargs["cameras"]["overhead"][0] == "/scene/overhead/color/compressed"

    # Stripping the namespace back off must reproduce the container's own names exactly.
    bare = RosSettings(namespace="").base_kwargs()
    for key in ("joint_states_topic", "command_topic", "gripper_topic", "reset_service"):
        assert kwargs[key] == "/so101" + bare[key]


def test_only_the_robots_that_boot_with_a_tree_are_required_to_have_one() -> None:
    """Three robots, three different true answers, and the console must not average them.

    * the myAGV's bringup starts `robot_state_publisher` and `robot_pose_ekf`, so `/tf` and
      the latched (empty) `/tf_static` are required of a base;
    * the SO-101's ROS 2 bringups run `robot_state_publisher` beside the controller
      manager, so both are required of an arm;
    * the AiNex's shipped boot chain runs none, so neither is required.
    """
    from robot_console.ainex_topics import CONTRACT_TOPICS
    from robot_console.arm.ros_settings import TF_STATIC_TOPIC, TF_TOPIC
    from robot_console.fleet import BASE_TOPICS, arm_topics
    from robot_console.topics import TOPIC_TF, TOPIC_TF_STATIC

    assert TOPIC_TF not in CONTRACT_TOPICS and TOPIC_TF_STATIC not in CONTRACT_TOPICS
    assert TOPIC_TF in BASE_TOPICS and TOPIC_TF_STATIC in BASE_TOPICS
    assert {TF_TOPIC, TF_STATIC_TOPIC} <= set(arm_topics())
