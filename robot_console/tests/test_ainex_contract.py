"""The console's AiNex contract, used consistently -- with no simulator checkout.

`ainex_topics.py` is the console's copy of the facts it consumes from
`robots_specs/ainex/ros.yml`. Holding that copy equal to the simulator's
`ros_surfaces/ainex/topics.py` and to the ROS file is a workspace test
(`../tests/test_contract_parity.py`, class `AiNex`), because it needs both source trees;
this file checks only what the console alone can: that its modules use the one copy and
compose it the way the rest of the console does.
"""

from __future__ import annotations

from robot_console import ainex_topics as ac


def test_the_link_keeps_no_copy_of_the_contract() -> None:
    """`ainex_link` must read these names, not re-type them. Identity, not equality: a
    literal that happens to match today is the same drift risk tomorrow."""
    from robot_console import ainex_link

    assert ainex_link.TOPIC_SET_WALKING_PARAM is ac.TOPIC_SET_WALKING_PARAM
    assert ainex_link.SRV_WALKING_COMMAND is ac.SRV_WALKING_COMMAND
    assert ainex_link.TYPE_WALKING_PARAM is ac.TYPE_WALKING_PARAM
    assert ainex_link.SRV_TYPE_SET_WALKING_COMMAND is ac.SRV_TYPE_SET_WALKING_COMMAND
    assert ainex_link.TOPIC_HEAD_PAN is ac.TOPIC_HEAD_PAN
    assert ainex_link.TOPIC_HEAD_TILT is ac.TOPIC_HEAD_TILT


def test_the_link_puts_the_drive_names_under_the_namespace() -> None:
    """The camera was namespaced and the drive names were not, which is a robot that
    shows you its view and ignores every key. No test saw it because nothing checked the
    two together."""
    from robot_console.ainex_link import AiNexLink

    link = AiNexLink("127.0.0.1", 9090, camera_topic="/ainex/camera/image_raw/compressed",
                     namespace="ainex")
    assert link._param_name == "/ainex/walking/set_param"
    assert link._command_name == "/ainex/walking/command"
    assert link._head_names == ("/ainex/head_pan_controller/command",
                                "/ainex/head_tilt_controller/command")

    bare = AiNexLink("127.0.0.1", 9090)
    assert bare._param_name == ac.TOPIC_SET_WALKING_PARAM
    assert bare._command_name == ac.SRV_WALKING_COMMAND


def test_the_stop_handshake_uses_known_walking_commands() -> None:
    """The handshake `ainex_link` sends and the two the gait runs on."""
    for command in ("enable_control", "enable", "start", "stop"):
        assert command in ac.WALKING_COMMANDS


def test_servo_ids_are_positions_in_the_joint_table() -> None:
    assert ac.servo_id(ac.JOINT_NAMES[0]) == 1
    assert ac.servo_id("head_pan") == 23 and ac.servo_id("head_tilt") == 24
    assert len(ac.JOINT_NAMES) == len(set(ac.JOINT_NAMES)) == 24


def test_a_humanoid_is_not_a_mobile_base() -> None:
    """The AiNex is commanded as a walking state machine and has no wheels."""
    from robot_console.topics import TOPIC_CMD_VEL, TOPIC_ODOM

    assert TOPIC_CMD_VEL not in ac.CONTRACT_TOPICS
    assert TOPIC_ODOM not in ac.CONTRACT_TOPICS


def test_nothing_outside_the_boot_chain_is_required() -> None:
    """No lidar, no joint states and no transform tree: the shipped robot presents none."""
    for name in ("/scan", "/joint_states", "/tf", "/tf_static"):
        assert name not in ac.CONTRACT_TOPICS
