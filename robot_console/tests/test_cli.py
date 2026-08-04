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
