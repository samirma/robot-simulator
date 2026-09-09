"""The two projects' copies of the AiNex's topic names, held equal.

Same mechanism and same reason as `tests/arm/test_ros_contract.py`: the console cannot
import the simulator -- it has to install and run with no simulator checkout at all -- so
the vendor's names are written down on both sides, and a duplicated constant that drifts
is worse than no constant. The failure it prevents is silent: a fleet check asking for a
topic nobody publishes reports a robot missing that is right there.

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


def test_every_name_matches_the_simulators() -> None:
    s = _sim()
    assert ac.TOPIC_SET_WALKING_PARAM == s.TOPIC_SET_WALKING_PARAM
    assert ac.TOPIC_APP_ACTION == s.TOPIC_APP_ACTION
    assert ac.TOPIC_BUS_SERVO_SET == s.TOPIC_BUS_SERVO_SET
    assert ac.TOPIC_IS_WALKING == s.TOPIC_IS_WALKING
    assert ac.TOPIC_JOINT_STATES == s.TOPIC_JOINT_STATES
    assert ac.TOPIC_IMU == s.TOPIC_IMU
    assert ac.TOPIC_CAMERA == s.TOPIC_CAMERA


def test_the_walking_service_and_its_commands_match_the_simulators() -> None:
    """A service name is exactly as easy to get wrong as a topic name.

    None of this was held equal until `ainex_link` stopped keeping its own copies: the
    link re-typed the service name and two type strings as literals, so a rename on the
    simulator's side would have passed every test here while teleop called into nothing.
    """
    s = _sim()
    assert ac.SRV_WALKING_COMMAND == s.SRV_WALKING_COMMAND
    assert ac.WALKING_COMMANDS == s.WALKING_COMMANDS
    # The handshake `ainex_link.connect` sends, and the two the gait state machine runs
    # on. Named individually because the link depends on these four strings specifically.
    for command in ("enable_control", "enable", "start", "stop"):
        assert command in ac.WALKING_COMMANDS


def test_every_type_string_matches_the_simulators() -> None:
    """The dialect too: ROS 1 single-slash, and the vendor's own `ainex_interfaces`.

    A type string that disagrees is not cosmetic -- `rosapi/topics_for_type` matches it
    exactly, which is how asking for one dialect found half the cameras on the wire.
    """
    s = _sim()
    assert ac.TYPE_WALKING_PARAM == s.TYPE_WALKING_PARAM
    assert ac.TYPE_HEAD_STATE == s.TYPE_HEAD_STATE
    assert ac.TYPE_BOOL == s.TYPE_BOOL
    assert ac.TYPE_FLOAT64 == s.TYPE_FLOAT64
    assert ac.TYPE_JOINT_STATE == s.TYPE_JOINT_STATE
    assert ac.TYPE_IMU == s.TYPE_IMU
    assert ac.TYPE_COMPRESSED_IMAGE == s.TYPE_COMPRESSED_IMAGE
    assert ac.SRV_TYPE_SET_WALKING_COMMAND == s.SRV_TYPE_SET_WALKING_COMMAND


def test_the_link_keeps_no_copy_of_the_contract() -> None:
    """`ainex_link` must read these names, not re-type them.

    It used to declare its own `TOPIC_SET_WALKING_PARAM`, `SRV_WALKING_COMMAND` and two
    type strings, which the test above could not see. Identity, not equality: a literal
    that happens to match today is the same drift risk tomorrow.
    """
    from robot_console import ainex_link

    assert ainex_link.TOPIC_SET_WALKING_PARAM is ac.TOPIC_SET_WALKING_PARAM
    assert ainex_link.SRV_WALKING_COMMAND is ac.SRV_WALKING_COMMAND
    assert ainex_link.TYPE_WALKING_PARAM is ac.TYPE_WALKING_PARAM
    assert ainex_link.SRV_TYPE_SET_WALKING_COMMAND is ac.SRV_TYPE_SET_WALKING_COMMAND


def test_the_link_puts_the_drive_names_under_the_namespace() -> None:
    """The camera was namespaced and the drive names were not, which is a robot that
    shows you its view and ignores every key. No test saw it because nothing checked the
    two together."""
    from robot_console.ainex_link import AiNexLink

    link = AiNexLink("127.0.0.1", 9090, camera_topic="/ainex/camera/image_raw/compressed",
                     namespace="ainex")
    assert link._param_name == "/ainex/walking/set_param"
    assert link._command_name == "/ainex/walking/command"

    bare = AiNexLink("127.0.0.1", 9090)
    assert bare._param_name == ac.TOPIC_SET_WALKING_PARAM
    assert bare._command_name == ac.SRV_WALKING_COMMAND


def test_the_contract_topics_are_all_names_the_simulator_knows() -> None:
    """Nothing may be required of the robot that the other side has never heard of."""
    s = _sim()
    known = {v for k, v in vars(s).items() if k.startswith("TOPIC_")}
    # The per-joint controllers are a table, not a constant each.
    known |= set(s.JOINT_COMMAND_TOPICS.values())
    assert set(ac.CONTRACT_TOPICS) <= known


def test_the_joint_table_matches_the_simulators() -> None:
    """24 joints, same names, same servo-id order, same topic per joint."""
    s = _sim()
    assert ac.JOINT_NAMES == s.JOINT_NAMES
    assert ac.JOINT_COMMAND_TOPICS == tuple(s.JOINT_COMMAND_TOPICS[j] for j in s.JOINT_NAMES)


def test_the_head_topics_are_two_of_the_joint_controllers() -> None:
    """The arrows drive per-joint controllers, not a topic of their own.

    Named rather than indexed, but they still have to *be* in the table: a head topic the
    simulator does not subscribe to fails as a head that never moves, with no error, which
    is the same silence namespacing the drive topics once produced.
    """
    assert ac.TOPIC_HEAD_PAN in ac.JOINT_COMMAND_TOPICS
    assert ac.TOPIC_HEAD_TILT in ac.JOINT_COMMAND_TOPICS
    assert ac.TOPIC_HEAD_PAN != ac.TOPIC_HEAD_TILT


def test_the_head_limits_match_the_simulators() -> None:
    """The console clamps to the robot's own range, so the two copies must agree.

    Clamping short would make part of the head's travel unreachable from teleop; clamping
    long would have the console asking for angles the robot silently folds back, which
    reads as a head that stops responding near the end of its travel.
    """
    servos = _sim_module("servos")
    assert servos.joint_limits("head_pan") == (-ac.HEAD_PAN_LIMIT, ac.HEAD_PAN_LIMIT)
    assert servos.joint_limits("head_tilt") == (-ac.HEAD_TILT_LIMIT, ac.HEAD_TILT_LIMIT)


def test_a_humanoid_is_not_a_mobile_base() -> None:
    """The AiNex is commanded as a walking state machine and has no wheels.

    Checking it against the myAGV's contract demanded `/cmd_vel` and `/odom` of a biped,
    which is how `--robots so101,ainex` failed its fleet check with every topic it does
    present sitting on the wire.
    """
    from robot_console.topics import TOPIC_CMD_VEL, TOPIC_ODOM

    assert TOPIC_CMD_VEL not in ac.CONTRACT_TOPICS
    assert TOPIC_ODOM not in ac.CONTRACT_TOPICS


def test_the_invented_lidar_is_not_required() -> None:
    """The simulator publishes `/scan` for this robot and its own constants call that a
    departure -- the real AiNex has no lidar. Requiring it here would make the console
    assert a simulator's invention as though it were the hardware."""
    assert "/scan" not in ac.CONTRACT_TOPICS
