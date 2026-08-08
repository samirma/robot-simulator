from pathlib import Path

import pytest

from robot_console.cli import DEFAULT_PORT, build_parser, parse_args, split_host_port
from robot_console.teleop import HOLD_TIMEOUT, SPEED_MAX, SPEED_MIN
from robot_console.topics import TOPIC_CAMERA, TOPIC_CMD_VEL, TOPIC_ODOM


def test_defaults():
    options = parse_args([])
    assert (options.host, options.port) == ("127.0.0.1", 9090)
    assert options.record is None
    assert options.preflight is True
    assert options.publish_hz == 20.0
    assert (options.cmd_topic, options.odom_topic, options.camera_topic) == (
        TOPIC_CMD_VEL,
        TOPIC_ODOM,
        TOPIC_CAMERA,
    )


def test_publish_rate_beats_the_watchdog():
    # The simulator's bridge stops the base after 0.5 s of silence.
    assert 1.0 / parse_args([]).publish_hz < 0.5


@pytest.mark.parametrize(
    "value, expected",
    [
        ("192.168.1.42", ("192.168.1.42", DEFAULT_PORT)),
        ("192.168.1.42:9091", ("192.168.1.42", 9091)),
        ("localhost:1234", ("localhost", 1234)),
        ("[::1]:9091", ("::1", 9091)),
        ("[::1]", ("::1", DEFAULT_PORT)),
        ("host:notaport", ("host:notaport", DEFAULT_PORT)),
    ],
)
def test_split_host_port(value, expected):
    assert split_host_port(value) == expected


def test_host_with_port():
    options = parse_args(["--host", "192.168.1.42:9091"])
    assert (options.host, options.port) == ("192.168.1.42", 9091)


def test_explicit_port_wins():
    options = parse_args(["--host", "10.0.0.1:9091", "--port", "9092"])
    assert (options.host, options.port) == ("10.0.0.1", 9092)


def test_record_becomes_a_path():
    options = parse_args(["--record", "runs/drive1"])
    assert options.record == Path("runs/drive1")


def test_no_preflight():
    assert parse_args(["--no-preflight"]).preflight is False


def test_speed_is_clamped_into_range():
    assert parse_args(["--speed", "99"]).speed == pytest.approx(SPEED_MAX)
    assert parse_args(["--speed", "0"]).speed == pytest.approx(SPEED_MIN)


def test_max_speed_override(capsys):
    options = parse_args(["--max-speed", "0.6", "--speed", "0.5"])
    assert options.max_speed == pytest.approx(0.6)
    assert options.speed == pytest.approx(0.5)
    # Going past the real myAGV limit should say so rather than fail silently.
    assert "exceeds the real myAGV limit" in capsys.readouterr().err


@pytest.mark.parametrize(
    "robot, fragment",
    [
        ("myagv", "the real myAGV limit"),
        ("ainex", "the AiNex gait envelope"),
    ],
)
def test_the_max_speed_warning_names_the_right_robots_limit(robot, fragment, capsys):
    """Telling someone they have passed the myAGV's limit while they drive an AiNex is
    worse than saying nothing: the number they are being measured against is not theirs."""
    parse_args(["--robot", robot, "--max-speed", "9"])
    assert fragment in capsys.readouterr().err


def test_hold_timeout_defaults_to_release_on_key_up():
    assert parse_args([]).hold_timeout == pytest.approx(HOLD_TIMEOUT)


def test_latch_disables_the_release_timeout():
    assert parse_args(["--latch"]).hold_timeout is None


def test_hold_timeout_override():
    assert parse_args(["--hold-timeout", "1.2"]).hold_timeout == pytest.approx(1.2)
    # Zero would stop the robot between key repeats, so it is floored.
    assert parse_args(["--hold-timeout", "0"]).hold_timeout > 0


def test_url():
    assert parse_args(["--host", "1.2.3.4", "--port", "9091"]).url == "ws://1.2.3.4:9091"


def test_help_mentions_the_keys():
    text = build_parser().format_help()
    for fragment in ("W/S", "A/D", "Q/E", "Space", "Esc", "Hold a key", "focus"):
        assert fragment in text


# ------------------------------------------------------------------ robot profiles


def test_robots_imports_without_the_missing_ainex_link():
    """`robots.py` was committed without `ainex_link.py`; that must not break the myAGV.

    A module-scope import of the missing module made `robot_console.robots` -- and every
    camera-mapping entry point behind it -- unimportable, for a robot none of them use.
    """
    from robot_console.robots import DEFAULT_ROBOT, PROFILES

    assert DEFAULT_ROBOT == "myagv"
    assert sorted(PROFILES) == ["ainex", "myagv"]
    assert PROFILES["myagv"].name == "myagv"
    assert PROFILES["myagv"].has_odom


def test_the_ainex_profile_resolves_with_its_link_present():
    """`ainex_link.py` was reconstructed from the simulator's transcription of the
    vendor stack; the lazy-profile machinery must now hand out a working profile
    (and would raise a clear RuntimeError again if the module ever went missing)."""
    from robot_console.robots import PROFILES

    profile = PROFILES["ainex"]
    assert profile.name == "ainex"
    assert not profile.has_odom, "the AiNex publishes no odometry at all"
    assert profile.speed_max < PROFILES["myagv"].speed_max, "a biped walks, not rolls"


def test_walking_params_invert_the_gait_model():
    """v = 4A/T, so A = vT/4 -- and the envelope caps, never raises."""
    from robot_console.ainex_link import PERIOD_S, STRIDE_FACTOR, walking_params
    from robot_console.teleop import Command

    wp = walking_params(Command(vx=0.1))
    assert wp["x_move_amplitude"] == pytest.approx(0.1 * PERIOD_S / STRIDE_FACTOR)
    assert wp["period_time"] == pytest.approx(PERIOD_S * 1000.0), "milliseconds on the wire"

    flat_out = walking_params(Command(vx=99.0))
    assert flat_out["x_move_amplitude"] == pytest.approx(0.02), "clamped to the envelope"

    turn = walking_params(Command(wz=-99.0))
    assert turn["angle_move_amplitude"] == pytest.approx(-10.0), "degrees, vendor units"


def test_an_unknown_robot_is_a_keyerror_naming_the_known_ones():
    from robot_console.robots import PROFILES

    with pytest.raises(KeyError, match="myagv"):
        PROFILES["forklift"]


# ------------------------------------------------------------------ --robot


def test_robot_defaults_to_the_myagv():
    from robot_console.robots import DEFAULT_ROBOT

    assert parse_args([]).robot == DEFAULT_ROBOT == "myagv"


@pytest.mark.parametrize("robot", ["myagv", "ainex"])
def test_robot_flag_selects_each_supported_robot(robot):
    assert parse_args(["--robot", robot]).robot == robot


def test_an_unsupported_robot_is_a_usage_error_listing_the_supported_ones(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--robot", "forklift"])
    err = capsys.readouterr().err
    for robot in ("myagv", "ainex"):
        assert robot in err


def test_listing_the_robots_does_not_construct_them(monkeypatch):
    """`--robot`'s choices come from the factory table, not from calling it. Otherwise a
    robot with a broken link module would take `--help` down with it."""
    import robot_console.robots as robots

    def explode() -> None:
        raise AssertionError("a profile was constructed just to list the choices")

    monkeypatch.setitem(robots._FACTORIES, "ainex", explode)
    assert "ainex" in build_parser().format_help()


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
    # And an out-of-range request lands on that robot's limits, not the myAGV's.
    assert parse_args(["--robot", robot, "--speed", "99"]).speed == pytest.approx(envelope.SPEED_MAX)
    assert parse_args(["--robot", robot, "--speed", "0"]).speed == pytest.approx(envelope.SPEED_MIN)


def test_a_missing_link_module_exits_rather_than_tracebacks(monkeypatch, capsys):
    """The lazy factories exist so a robot whose link module is absent produces an
    actionable sentence. `main` must not turn that back into a traceback."""
    import robot_console.cli as cli
    import robot_console.robots as robots

    def missing() -> None:
        raise RuntimeError("AiNex support is incomplete: ainex_link.py is missing")

    monkeypatch.setitem(robots._FACTORIES, "ainex", missing)
    assert cli.main(["--robot", "ainex"]) == 2
    assert "ainex_link.py is missing" in capsys.readouterr().err
