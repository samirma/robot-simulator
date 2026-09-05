"""Argument parsing for the three SLAM commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_console.slam.cli import DEFAULT_MAP_DIR, SlamOptions, parse_args
from robot_console.teleop import SPEED_MAX, SPEED_MIN


def test_defaults():
    options = parse_args([], mode="map")
    assert options.mode == "map"
    assert options.host == "127.0.0.1"
    assert options.port == 9090
    assert options.out == DEFAULT_MAP_DIR
    assert options.preflight
    assert options.url == "ws://127.0.0.1:9090"


@pytest.mark.parametrize("mode", ["explore", "map", "navigate"])
def test_each_mode_parses(mode):
    assert parse_args([], mode=mode).mode == mode


def test_the_mode_can_come_from_the_command_line():
    assert parse_args(["explore"]).mode == "explore"


def test_an_unknown_mode_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["fly"])


def test_host_may_carry_the_port():
    options = parse_args(["--host", "192.168.1.42:9091"], mode="map")
    assert (options.host, options.port) == ("192.168.1.42", 9091)


def test_an_explicit_port_wins():
    options = parse_args(["--host", "10.0.0.1:1111", "--port", "2222"], mode="map")
    assert options.port == 2222


def test_paths_become_paths():
    options = parse_args(["--out", "runs/house", "--map", "runs/old"], mode="navigate")
    assert options.out == Path("runs/house")
    assert options.load == Path("runs/old")


def test_navigate_reads_the_output_map_when_none_is_named():
    """`slam.sh navigate` with no flags should open whatever the last run wrote."""
    assert parse_args(["--out", "runs/house"], mode="navigate").map_source == Path("runs/house")


def test_mapping_starts_a_fresh_map_by_default():
    assert parse_args(["--out", "runs/house"], mode="map").map_source is None


def test_an_explicit_map_is_loaded_in_any_mode():
    options = parse_args(["--out", "runs/new", "--map", "runs/old"], mode="map")
    assert options.map_source == Path("runs/old")


def test_speed_above_the_hardware_limit_warns(capsys):
    options = parse_args(["--max-speed", "1.5"], mode="explore")
    assert "exceeds the real myAGV limit" in capsys.readouterr().err
    assert options.max_speed == 1.5


def test_speed_is_kept_inside_the_cap():
    options = parse_args(["--speed", "9.0", "--max-speed", "0.2"], mode="explore")
    assert options.speed == pytest.approx(0.2)
    assert parse_args(["--speed", "0.001"], mode="explore").speed == pytest.approx(SPEED_MIN)


def test_the_default_cap_is_the_hardware_limit():
    assert parse_args([], mode="explore").max_speed == SPEED_MAX


def test_latch_disables_the_hold_timeout():
    assert parse_args(["--latch"], mode="map").hold_timeout is None


def test_no_preflight():
    assert parse_args(["--no-preflight"], mode="map").preflight is False


def test_resolution_and_range_are_kept_sane():
    options = parse_args(["--resolution", "-1", "--max-range", "0", "--robot-radius", "-5"], mode="map")
    assert options.resolution > 0 and options.max_range > 0 and options.robot_radius > 0


def test_slam_hz_has_a_floor():
    """A zero rate would make the keyframe interval infinite and matching would never run."""
    assert parse_args(["--slam-hz", "0"], mode="explore").slam_hz >= 0.5


def test_topics_can_be_overridden():
    # A relative name comes back absolute: resolution now happens in `parse_args`, so
    # `Options` holds the name that actually goes on the wire. `RobotLink` normalised it
    # anyway, so this is the same topic -- just settled one layer earlier, which is what
    # lets `--namespace` and an explicit topic be combined without guessing.
    options = parse_args(["--scan-topic", "/laser", "--odom-topic", "odom2"], mode="map")
    assert options.scan_topic == "/laser"
    assert options.odom_topic == "/odom2"


def test_a_namespace_prefixes_every_topic_but_an_explicit_one_wins():
    """One flag for a robot on a shared graph; naming a topic still overrides it."""
    options = parse_args(["--namespace", "myagv"], mode="map")
    assert options.cmd_topic == "/myagv/cmd_vel"
    assert options.odom_topic == "/myagv/odom"
    assert options.scan_topic == "/myagv/scan"
    assert options.camera_topic == "/myagv/camera/image_raw/compressed"

    # An explicit topic is left exactly as given, and one already carrying the namespace
    # is not prefixed twice.
    explicit = parse_args(
        ["--namespace", "myagv", "--odom-topic", "/elsewhere/odom",
         "--scan-topic", "/myagv/scan"],
        mode="map",
    )
    assert explicit.odom_topic == "/elsewhere/odom"
    assert explicit.scan_topic == "/myagv/scan"
    assert explicit.cmd_topic == "/myagv/cmd_vel"

    # And no namespace is the bare contract a real myAGV bringup presents.
    assert parse_args([], mode="map").cmd_topic == "/cmd_vel"


def test_help_does_not_need_a_display(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--help"], mode="explore")
    assert exit_info.value.code == 0
    assert "explore" in capsys.readouterr().out


def test_options_are_frozen():
    with pytest.raises(Exception):
        SlamOptions().mode = "explore"


# --------------------------------------------------------- explore recovery behaviour


def _session(mode="explore"):
    """A _Session with no network, for testing decision logic directly."""
    import numpy as np
    from robot_console.slam.app import _Session
    from robot_console.slam.controller import PathFollower
    from robot_console.slam.grid import OccupancyGrid
    from robot_console.slam.mapview import MapView
    from robot_console.slam.pose import PoseTracker
    from robot_console.teleop import TeleopState

    options = SlamOptions(mode=mode, out=Path("/tmp/_unused"))
    return _Session(options, OccupancyGrid(0.05), PoseTracker(), PathFollower(speed=0.2),
                    MapView(), None, TeleopState(), None)


def test_rerouting_to_the_same_goal_keeps_the_stuck_watchdog_running():
    """Otherwise a robot pinned against furniture reroutes every tick, resets the
    watchdog every tick, and spins in place forever instead of giving up on the goal."""
    import numpy as np

    session = _session()
    session._reset_follower_for(np.array([3.0, 4.0]))
    session.follower._best_at = 100.0
    session._reset_follower_for(np.array([3.0, 4.0]))
    assert session.follower._best_at == 100.0


def test_a_new_goal_does_reset_the_stuck_watchdog():
    import numpy as np

    session = _session()
    session._reset_follower_for(np.array([3.0, 4.0]))
    session.follower._best_at = 100.0
    session._reset_follower_for(np.array([9.0, 9.0]))
    assert session.follower._best_at is None


def test_explore_does_not_finish_before_any_scan_has_arrived():
    """`no frontiers` on an empty map means `no data`, not `done`."""
    session = _session()
    session.tracker.update_odom((0.0, 0.0, 0.0))
    assert session.integrated == 0
    session.decide(now=1.0)
    assert session.finished is None
