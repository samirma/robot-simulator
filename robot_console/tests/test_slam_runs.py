"""Whole SLAM runs on deterministic scan/odometry fixtures (console spec §4, SLAM).

`simworld` supplies the scans and perfect odometry; a real `_Session` decides. Covered:
map persistence (a continued map starts at the saved pose), both planning modes, and the
two runs that must end `limit` rather than go on forever -- a goal held in continuous
progress, and a world that keeps generating frontiers -- each saving a loadable map.
"""

from __future__ import annotations

import numpy as np
import pytest
from simworld import HOUSE, World, _Odom, offline_session

from robot_console.slam import mapio
from robot_console.slam.explorer import LIMIT
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import costmap_for, plan

DT = 0.2


def _drive(session, world, *, max_ticks, move=True):
    tick = 0
    for tick in range(max_ticks):
        now = tick * DT
        session.on_odom(_Odom(world.pose))
        session.on_scan(world.scan(), now)
        command = session.decide(now)
        if session.finished:
            break
        if move:
            world.drive(command)
    return tick * DT


def _assert_saved_and_loadable(session, directory):
    saved = session.save(session.finished)
    assert saved is not None and saved.exists()
    grid = mapio.load_map(directory)
    assert grid.resolution == pytest.approx(0.05)
    assert (grid.classify() != 0).sum() > 100, "the map has content"
    pose = mapio.load_pose(directory)
    assert pose is not None and np.allclose(pose, session.tracker.pose)


def test_a_goal_in_continuous_progress_still_ends_with_limit_at_max_duration(tmp_path):
    """The robot is held still while the map keeps growing, so the one goal it chose
    progresses for ever (0.25 m² every 2.5 s). Only --max-duration ends the run."""
    max_duration = 120.0
    session = offline_session(out=tmp_path, max_duration=max_duration)
    world = World(start=(1.0, 1.0, 0.0), dt=DT)

    def growing():
        return 50.0 + 0.1 * session.clock

    session.clock = 0.0
    session.area = growing
    tick = 0
    for tick in range(5000):
        session.clock = now = tick * DT
        session.on_odom(_Odom(world.pose))
        session.on_scan(world.scan(), now)
        session.decide(now)
        if session.finished:
            break
    assert session.finished == LIMIT
    elapsed = tick * DT - session.started
    assert max_duration <= elapsed <= max_duration + DT
    assert session.explorer.attempted == 1, "it was one goal, progressing throughout"
    assert session.explorer.records[0].progress_events > 10
    assert "limit after 120" in session.report(tick * DT)
    _assert_saved_and_loadable(session, tmp_path)


def test_continuously_generated_frontiers_still_end_with_limit_at_max_goals(tmp_path):
    """A field of pillars far larger than the lidar's reach never runs out of frontiers.

    Pillars rather than open ground because a beam that returns nothing traces nothing:
    the mapper only learns free space along beams that hit something within range."""
    size, pitch, half = 40.0, 2.0, 0.1
    segments = [(0, 0, size, 0), (0, size, size, size), (0, 0, 0, size), (size, 0, size, size)]
    for cx in np.arange(pitch / 2, size, pitch):
        for cy in np.arange(pitch / 2, size, pitch):
            x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
            segments += [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
    field = np.array(segments, dtype=np.float64)
    session = offline_session(out=tmp_path, max_goals=4)
    world = World(walls=field, start=(size / 2, size / 2, 0.0), dt=DT)
    _drive(session, world, max_ticks=5000)
    assert session.finished == LIMIT
    assert session.explorer.attempted == 4
    _assert_saved_and_loadable(session, tmp_path)


def test_explore_with_a_short_duration_saves_and_ends_with_limit(tmp_path):
    session = offline_session(out=tmp_path, max_duration=20.0)
    world = World(walls=HOUSE, dt=DT)
    elapsed = _drive(session, world, max_ticks=2000)
    assert session.finished == LIMIT and elapsed == pytest.approx(20.0, abs=2 * DT)
    _assert_saved_and_loadable(session, tmp_path)


# ------------------------------------------------------------------ persistence


def test_a_continued_map_starts_at_the_saved_pose(tmp_path):
    first = offline_session(out=tmp_path, max_duration=15.0)
    world = World(walls=HOUSE, dt=DT)
    _drive(first, world, max_ticks=500)
    _assert_saved_and_loadable(first, tmp_path)
    saved_pose = mapio.load_pose(tmp_path)
    before = (first.grid.classify() != 0).sum()

    from robot_console.slam.app import _initial_grid

    second = offline_session(out=tmp_path, mode="map")
    assert second.options.map_source == tmp_path
    grid, pose = _initial_grid(second.options)
    assert np.allclose(pose, saved_pose)
    second.grid, second._seed_pose, second._localized = grid, pose, False
    # The robot's own odometry restarts at zero; the continued map puts it where it was.
    second.on_odom(_Odom((0.0, 0.0, 0.0)))
    second.on_scan(World(walls=HOUSE, start=tuple(saved_pose)).scan(), 0.0)
    assert np.allclose(second.tracker.pose, saved_pose, atol=1e-9)
    assert (second.grid.classify() != 0).sum() >= before * 0.95, "continued, not restarted"


def test_the_sidecar_holds_full_precision_and_the_pose(tmp_path):
    grid = OccupancyGrid(0.05, width=20, height=20, origin=(-0.5, -0.5))
    grid.data[5, 5] = 1.2345678
    mapio.save_map(grid, tmp_path, pose=(0.1, 0.2, 0.3), crop=False)
    for name in ("map.pgm", "map.yaml", "map.npz"):
        assert (tmp_path / name).exists()
    assert mapio.load_map(tmp_path).data[5, 5] == pytest.approx(1.2345678, abs=1e-6)
    assert np.allclose(mapio.load_pose(tmp_path), (0.1, 0.2, 0.3))
    assert "resolution: 0.050000" in (tmp_path / "map.yaml").read_text()


# ------------------------------------------------------------------ planning modes


def _half_known():
    """Free on the left, unknown on the right, the goal in the unknown half."""
    grid = OccupancyGrid(0.05, width=80, height=40, origin=(0.0, 0.0))
    grid.data[:] = 0.0
    grid.data[:, :40] = -5.0
    grid.revision += 1
    return grid


def test_navigate_treats_unknown_as_blocked_and_explore_as_free():
    grid = _half_known()
    start, goal = (0.5, 1.0), (3.5, 1.0)
    assert plan(costmap_for(grid, None, allow_unknown=True, radius=0.25), start, goal) is not None
    assert plan(costmap_for(grid, None, allow_unknown=False, radius=0.25), start, goal) is None


def test_the_navigate_session_plans_with_unknown_blocked(tmp_path):
    session = offline_session(out=tmp_path)
    session.options = session.options.__class__(**{**session.options.__dict__, "mode": "navigate"})
    session.grid = _half_known()
    session.tracker.update_odom((0.5, 1.0, 0.0))
    session._set_goal(np.array([3.5, 1.0]), 0.0)
    assert session.path is None and "no path" in session.note
    session._set_goal(np.array([1.5, 1.0]), 0.0)
    assert session.path is not None
