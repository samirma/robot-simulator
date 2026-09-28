from pathlib import Path

import pytest

from robot_console.cli import build_parser, parse_args
from robot_console.teleop import HOLD_TIMEOUT, SPEED_MAX, SPEED_MIN
from robot_console.topics import TOPIC_CAMERA
from robot_console.wire import DEFAULT_URL, parse_url

SPEC_FLAGS = {
    "--robot", "--namespace", "--url", "--record", "--speed", "--max-speed", "--latch",
}


def test_the_flags_are_exactly_the_spec_synopsis():
    """Console spec §2.1, plus argparse's own --help and --version."""
    flags = {s for a in build_parser()._actions for s in a.option_strings if s.startswith("--")}
    assert flags - {"--help", "--version"} == SPEC_FLAGS


@pytest.mark.parametrize("gone", ["--host", "--port", "--hold-timeout", "--publish-hz",
                                  "--cmd-topic", "--odom-topic", "--camera-topic",
                                  "--connect-timeout", "--record-fps", "--safety-timeout",
                                  "--no-preflight", "--reinstall"])
def test_flags_outside_the_spec_are_rejected(gone):
    with pytest.raises(SystemExit):
        parse_args([gone, "1"])


def test_defaults():
    options = parse_args([])
    assert options.url == DEFAULT_URL == "ws://127.0.0.1:9090"
    assert (options.host, options.port) == ("127.0.0.1", 9090)
    assert options.robot is None and options.namespace is None
    assert options.needs_discovery
    assert not options.latch
    assert options.record is None


def test_url_parses_and_is_canonical():
    assert parse_args(["--url", "ws://192.168.1.42:9091"]).url == "ws://192.168.1.42:9091"
    assert parse_args(["--url", "ws://robot.local"]).url == "ws://robot.local:9090"
    assert parse_url("ws://[::1]:9092") == ("::1", 9092)


@pytest.mark.parametrize("bad", ["127.0.0.1:9090", "http://x:1", "ws://", "ws://h:notaport",
                                 "ws://h:1/path"])
def test_a_bad_url_is_a_usage_error(bad):
    with pytest.raises(SystemExit):
        parse_args(["--url", bad])


def test_an_explicitly_empty_namespace_is_the_bare_contract():
    options = parse_args(["--namespace", ""])
    assert options.namespace == ""
    assert options.resolved("myagv", "ignored").namespace == ""


def test_both_named_needs_no_discovery():
    assert not parse_args(["--namespace", "myagv", "--robot", "myagv"]).needs_discovery
    assert parse_args(["--namespace", "myagv"]).needs_discovery


def test_the_wire_fills_in_what_was_not_given():
    options = parse_args([]).resolved("ainex", "ainex", camera_topic="/ainex/cam")
    assert (options.robot, options.namespace, options.camera_topic) == ("ainex", "ainex", "/ainex/cam")
    assert parse_args([]).resolved("myagv", "").camera_topic == TOPIC_CAMERA


def test_speeds_are_reclamped_into_the_robot_the_wire_named():
    import robot_console.ainex_link as ainex

    options = parse_args(["--speed", "99"])
    assert options.speed == pytest.approx(SPEED_MAX)
    assert options.resolved("ainex", "ainex").speed == pytest.approx(ainex.SPEED_MAX)


def test_record_becomes_a_path():
    assert parse_args(["--record", "runs/drive1"]).record == Path("runs/drive1")


def test_speed_is_clamped_into_range():
    assert parse_args(["--speed", "99"]).speed == pytest.approx(SPEED_MAX)
    assert parse_args(["--speed", "0"]).speed == pytest.approx(SPEED_MIN)


def test_max_speed_override(capsys):
    options = parse_args(["--max-speed", "0.6", "--speed", "0.5"])
    assert options.max_speed == pytest.approx(0.6)
    assert options.speed == pytest.approx(0.5)
    assert "exceeds the real myAGV limit" in capsys.readouterr().err


@pytest.mark.parametrize(
    "robot, fragment",
    [("myagv", "the real myAGV limit"), ("ainex", "the AiNex gait envelope")],
)
def test_the_max_speed_warning_names_the_right_robots_limit(robot, fragment, capsys):
    parse_args(["--robot", robot, "--max-speed", "9"])
    assert fragment in capsys.readouterr().err


def test_release_after_0_6_s_unless_latched():
    assert HOLD_TIMEOUT == pytest.approx(0.6)
    assert parse_args([]).hold_timeout == pytest.approx(0.6)
    assert parse_args(["--latch"]).hold_timeout is None


def test_the_safety_timeout_is_a_fixed_constant():
    """Console spec §2.1: 0.25 s, not a flag."""
    from robot_console.supervisor import SAFETY_TIMEOUT

    assert SAFETY_TIMEOUT == 0.25


def test_help_mentions_the_keys():
    text = build_parser().format_help()
    for fragment in ("W/S", "A/D", "Q/E", "Space", "Esc", "Hold a key", "focus"):
        assert fragment in text


# ------------------------------------------------------------------ robot profiles


def test_both_profiles_resolve():
    from robot_console.robots import PROFILES

    assert list(PROFILES) == ["myagv", "ainex", "myagv_mycobot280", "rosmaster_x3_plus"]
    assert PROFILES["myagv"].has_odom and not PROFILES["myagv"].has_head
    assert PROFILES["ainex"].has_head and not PROFILES["ainex"].has_odom
    assert PROFILES["ainex"].speed_max < PROFILES["myagv"].speed_max
    for robot in ("myagv_mycobot280", "rosmaster_x3_plus"):
        assert PROFILES[robot].has_odom and not PROFILES[robot].has_head, robot


def test_the_spec_speed_envelopes():
    from robot_console.robots import PROFILES

    agv, ainex = PROFILES["myagv"], PROFILES["ainex"]
    assert (agv.speed_default, agv.speed_max, agv.speed_step, agv.turn_ratio) == (0.15, 0.28, 0.05, 2.0)
    assert (ainex.speed_default, ainex.speed_max, ainex.speed_step) == (0.10, 0.20, 0.02)
    assert (ainex.turn_ratio, ainex.turn_max) == (4.0, 1.0)
    # The composite drives as the myAGV it stands on; the X3 PLUS from Yahboom's own
    # keyboard teleop (0.2 m/s, 1.0 rad/s) up to its board's 0.7 m/s and 3.2 rad/s.
    comp, x3 = PROFILES["myagv_mycobot280"], PROFILES["rosmaster_x3_plus"]
    assert (comp.speed_default, comp.speed_max, comp.speed_step, comp.turn_ratio,
            comp.turn_max) == (agv.speed_default, agv.speed_max, agv.speed_step,
                               agv.turn_ratio, agv.turn_max)
    assert (x3.speed_default, x3.speed_max, x3.speed_step) == (0.2, 0.7, 0.05)
    assert (x3.turn_ratio, x3.turn_max) == (5.0, 3.2)


def test_walking_params_invert_the_gait_model():
    from robot_console.ainex_link import PERIOD_S, STRIDE_FACTOR, walking_params
    from robot_console.teleop import Command

    wp = walking_params(Command(vx=0.1))
    assert wp["x_move_amplitude"] == pytest.approx(0.1 * PERIOD_S / STRIDE_FACTOR)
    assert wp["period_time"] == pytest.approx(PERIOD_S * 1000.0), "milliseconds on the wire"
    assert walking_params(Command(vx=99.0))["x_move_amplitude"] == pytest.approx(0.02)
    assert walking_params(Command(wz=-99.0))["angle_move_amplitude"] == pytest.approx(-10.0)


def test_an_unknown_robot_is_a_keyerror_naming_the_known_ones():
    from robot_console.robots import PROFILES

    with pytest.raises(KeyError, match="myagv"):
        PROFILES["forklift"]


@pytest.mark.parametrize("robot", ["myagv", "ainex", "myagv_mycobot280", "rosmaster_x3_plus"])
def test_robot_flag_selects_each_supported_robot(robot):
    assert parse_args(["--robot", robot]).robot == robot


def test_an_unsupported_robot_is_a_usage_error_listing_the_supported_ones(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--robot", "so101"])
    err = capsys.readouterr().err
    for robot in ("myagv", "ainex"):
        assert robot in err


@pytest.mark.parametrize(
    "robot, module",
    [("myagv", "robot_console.teleop"), ("ainex", "robot_console.ainex_link")],
)
def test_speed_defaults_and_caps_follow_the_robot(robot, module):
    import importlib

    envelope = importlib.import_module(module)
    options = parse_args(["--robot", robot])
    assert options.speed == pytest.approx(envelope.SPEED_DEFAULT)
    assert options.max_speed == pytest.approx(envelope.SPEED_MAX)
    assert parse_args(["--robot", robot, "--speed", "99"]).speed == pytest.approx(envelope.SPEED_MAX)
    assert parse_args(["--robot", robot, "--speed", "0"]).speed == pytest.approx(envelope.SPEED_MIN)
