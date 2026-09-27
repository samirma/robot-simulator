"""The two projects' copies of the AiNex's contract, held equal.

Same mechanism and same reason as `tests/arm/test_ros_contract.py`: the console cannot
import the simulator -- it has to install and run with no simulator checkout at all -- so
the vendor's names are written down on both sides, and a duplicated constant that drifts
is worse than no constant. The failure it prevents is silent: a fleet check asking for a
topic nobody publishes reports a robot missing that is right there.

The simulator's `topics.py` transcribes `robots_specs/ainex/ros.yml` and is itself held
to that file by `simulator/shared/contracts/test_ainex_contract.py`; this file holds the
console's subset to the simulator's tables -- name, type and direction -- so every fact
the console uses is one the robot's ROS file states.

Skips when the sibling simulator is not checked out, exactly as the arm's does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from robot_console import ainex_topics as ac

#: The simulator's copy. Stdlib-only on that side, so it loads here by path the same way
#: the SO-101 surface does.
SIM_TOPICS = (
    Path(__file__).resolve().parents[2]
    / "simulator" / "shared" / "ros_surfaces" / "ainex" / "topics.py"
)


def _sim_module(name: str):
    """One of the simulator's stdlib-only AiNex modules, loaded by path.

    By path and not by import, for the reason at the top of this file: this project must
    install and run with no simulator checkout, so the sibling can only ever be read
    opportunistically and skipped when absent.
    """
    path = SIM_TOPICS.with_name(f"{name}.py")
    if not path.exists():
        pytest.skip(f"sibling simulator checkout not present at {path}")
    key = f"_ainex_sim_{name}"
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[key]
        raise
    return module


def _sim():
    return _sim_module("topics")


def _published(s) -> dict[str, str]:
    return {t.name: t.type for t in s.TOPICS if t.direction == "out"}


def _subscribed(s) -> dict[str, str]:
    return {t.name: t.type for t in s.TOPICS if t.direction == "in"}


def _services(s) -> dict[str, str]:
    return {x.name: x.type for x in s.SERVICES}


def test_every_topic_the_console_sends_is_one_the_robot_subscribes_with_that_type() -> None:
    s = _sim()
    sub = _subscribed(s)
    assert sub[ac.TOPIC_SET_WALKING_PARAM] == ac.TYPE_WALKING_PARAM
    assert sub[ac.TOPIC_APP_ACTION] == ac.TYPE_STRING
    assert sub[ac.TOPIC_HEAD_PAN] == ac.TYPE_HEAD_STATE
    assert sub[ac.TOPIC_HEAD_TILT] == ac.TYPE_HEAD_STATE


def test_every_topic_the_console_reads_is_one_the_robot_publishes_with_that_type() -> None:
    s = _sim()
    pub = _published(s)
    assert pub[ac.TOPIC_IS_WALKING] == ac.TYPE_BOOL
    assert pub[ac.TOPIC_IMU] == ac.TYPE_IMU
    assert pub[ac.TOPIC_CAMERA] == ac.TYPE_COMPRESSED_IMAGE


def test_every_service_the_console_calls_is_the_robots_with_that_type() -> None:
    """A service name is exactly as easy to get wrong as a topic name."""
    srv = _services(_sim())
    assert srv[ac.SRV_WALKING_COMMAND] == ac.SRV_TYPE_SET_WALKING_COMMAND
    assert srv[ac.SRV_IS_WALKING] == ac.SRV_TYPE_GET_WALKING_STATE
    assert srv[ac.SRV_BUS_SERVO_GET] == ac.SRV_TYPE_GET_BUS_SERVOS_POSITION


def test_the_names_match_the_simulators_constants() -> None:
    s = _sim()
    for name in ("TOPIC_SET_WALKING_PARAM", "TOPIC_APP_ACTION", "TOPIC_HEAD_PAN",
                 "TOPIC_HEAD_TILT", "TOPIC_IS_WALKING", "TOPIC_IMU", "TOPIC_CAMERA",
                 "SRV_WALKING_COMMAND", "SRV_IS_WALKING", "SRV_BUS_SERVO_GET"):
        assert getattr(ac, name) == getattr(s, name), name


def test_the_walking_commands_match_the_simulators() -> None:
    s = _sim()
    assert ac.WALKING_COMMANDS == s.WALKING_COMMANDS
    # The handshake `ainex_link.connect` sends, and the two the gait state machine runs
    # on. Named individually because the link depends on these four strings specifically.
    for command in ("enable_control", "enable", "start", "stop"):
        assert command in ac.WALKING_COMMANDS


def test_the_joint_table_and_servo_scale_match_the_simulators() -> None:
    """Servo ids by position, and the count<->radian scale the read-back is decoded with."""
    s = _sim()
    servos = _sim_module("servos")
    assert ac.JOINT_NAMES == s.JOINT_NAMES
    for joint in ("head_pan", "head_tilt", "l_knee"):
        assert ac.servo_id(joint) == servos.SERVOS[joint][0]
    assert ac.SERVO_TICKS_PER_RADIAN == pytest.approx(servos.TICKS_PER_RADIAN)
    assert servos.SERVOS["head_pan"][1:] == (ac.HEAD_SERVO_CENTRE, False)
    assert servos.SERVOS["head_tilt"][1:] == (ac.HEAD_SERVO_CENTRE, False)


def test_the_contract_topics_are_all_names_the_robot_presents() -> None:
    """Nothing may be required of the robot that its ROS file does not list."""
    s = _sim()
    assert set(ac.CONTRACT_TOPICS) <= {t.name for t in s.TOPICS}


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


def test_the_head_limits_match_the_simulators() -> None:
    """The console clamps to the robot's own range, so the two copies must agree."""
    servos = _sim_module("servos")
    assert servos.joint_limits("head_pan") == (-ac.HEAD_PAN_LIMIT, ac.HEAD_PAN_LIMIT)
    assert servos.joint_limits("head_tilt") == (-ac.HEAD_TILT_LIMIT, ac.HEAD_TILT_LIMIT)


def test_a_humanoid_is_not_a_mobile_base() -> None:
    """The AiNex is commanded as a walking state machine and has no wheels."""
    from robot_console.topics import TOPIC_CMD_VEL, TOPIC_ODOM

    assert TOPIC_CMD_VEL not in ac.CONTRACT_TOPICS
    assert TOPIC_ODOM not in ac.CONTRACT_TOPICS


def test_nothing_outside_the_boot_chain_is_required() -> None:
    """No lidar, no joint states and no transform tree: the shipped robot presents none."""
    for name in ("/scan", "/joint_states", "/tf", "/tf_static"):
        assert name not in ac.CONTRACT_TOPICS
        assert name not in {t.name for t in _sim().TOPICS}
