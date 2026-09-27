"""`slam.sh`'s command line (console spec §2.2) and the session behind it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_console.slam.cli import DEFAULT_MAX_DURATION, SlamOptions, build_parser, parse_args
from robot_console.slam.explorer import DEFAULT_MAX_GOALS


def _flags(mode):
    return {s for a in build_parser(mode)._actions for s in a.option_strings
            if s.startswith("--")} - {"--help", "--version"}


def test_each_mode_takes_exactly_the_spec_flags():
    common = {"--namespace", "--url", "--safety-timeout"}
    assert _flags("explore") == common | {"--out", "--max-duration", "--max-goals"}
    assert _flags("map") == common | {"--out"}
    assert _flags("navigate") == common | {"--map"}


@pytest.mark.parametrize("gone", ["--host", "--port", "--timeout", "--stall-timeout",
                                  "--resolution", "--speed", "--cmd-topic", "--record"])
def test_flags_outside_the_spec_are_rejected(gone):
    with pytest.raises(SystemExit):
        parse_args(["--out", "x", gone, "1"], mode="explore")


def test_defaults():
    options = parse_args(["--out", "runs/house"], mode="explore")
    assert options.url == "ws://127.0.0.1:9090"
    assert options.namespace is None
    assert options.max_duration == DEFAULT_MAX_DURATION == 3600.0
    assert options.max_goals == DEFAULT_MAX_GOALS == 500
    assert options.safety_timeout == pytest.approx(0.25)
    assert options.resolution == 0.05, "the grid is 0.05 m per cell"


def test_the_map_directory_is_required():
    with pytest.raises(SystemExit):
        parse_args([], mode="map")
    with pytest.raises(SystemExit):
        parse_args(["--out", "x"], mode="navigate")


def test_the_mode_can_come_from_the_command_line():
    assert parse_args(["navigate", "--map", "m"]).mode == "navigate"
    with pytest.raises(SystemExit):
        parse_args(["survey", "--out", "x"])


def test_limits_parse_and_must_be_positive():
    options = parse_args(["--out", "x", "--max-duration", "60", "--max-goals", "7"], mode="explore")
    assert (options.max_duration, options.max_goals) == (60.0, 7)
    for bad in (["--max-duration", "0"], ["--max-goals", "0"], ["--safety-timeout", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(["--out", "x", *bad], mode="explore")


def test_the_url_and_namespace():
    options = parse_args(["--map", "m", "--url", "ws://10.0.0.2:9091", "--namespace", ""],
                         mode="navigate")
    assert (options.host, options.port, options.namespace) == ("10.0.0.2", 9091, "")


def test_where_the_map_comes_from_and_goes(tmp_path):
    from robot_console.slam import mapio
    from robot_console.slam.grid import OccupancyGrid

    fresh = parse_args(["--out", str(tmp_path / "new")], mode="map")
    assert fresh.map_source is None and fresh.save_to == tmp_path / "new"
    mapio.save_map(OccupancyGrid(0.05), tmp_path / "old")
    again = parse_args(["--out", str(tmp_path / "old")], mode="explore")
    assert again.map_source == tmp_path / "old", "a map already in --out is continued"
    nav = parse_args(["--map", str(tmp_path / "old")], mode="navigate")
    assert nav.map_source == nav.save_to == tmp_path / "old"


def test_help_does_not_need_a_display(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--help"], mode="explore")
    assert exit_info.value.code == 0
    assert "--max-goals" in capsys.readouterr().out


def test_options_are_frozen():
    with pytest.raises(Exception):
        SlamOptions().mode = "explore"


# --------------------------------------------------------- the session


def _session(mode="explore", **kw):
    from robot_console.slam.app import _Session
    from robot_console.slam.controller import PathFollower
    from robot_console.slam.grid import OccupancyGrid
    from robot_console.slam.mapview import MapView
    from robot_console.slam.pose import PoseTracker
    from robot_console.teleop import TeleopState

    options = SlamOptions(mode=mode, out=Path("/tmp/_unused"), **kw)
    return _Session(options, OccupancyGrid(0.05), PoseTracker(), PathFollower(speed=0.2),
                    MapView(), None, TeleopState(), None)


def test_rerouting_to_the_same_goal_keeps_the_stuck_watchdog_running():
    session = _session()
    session._reset_follower_for(np.array([3.0, 4.0]))
    session.follower._best_at = 100.0
    session._reset_follower_for(np.array([3.0, 4.0]))
    assert session.follower._best_at == 100.0


def test_a_new_goal_does_reset_the_stuck_watchdog():
    session = _session()
    session._reset_follower_for(np.array([3.0, 4.0]))
    session.follower._best_at = 100.0
    session._reset_follower_for(np.array([9.0, 9.0]))
    assert session.follower._best_at is None


def test_explore_does_not_finish_before_any_scan_has_arrived():
    session = _session()
    session.tracker.update_odom((0.0, 0.0, 0.0))
    session.decide(now=1.0)
    assert session.finished is None


def test_the_run_reports_elapsed_time_and_goals():
    session = _session()
    session.started = 10.0
    session.finished = "limit"
    assert session.report(70.0) == "explore ended: limit after 60.0 s, 0 goals attempted"
