"""Does exploration actually map the whole house?

The question every other frontier test dodges. They are single shots at a static
half-seen grid, and a strategy that abandons an entire room passes all of them. These
drive a real `_Session` around a floorplan until it stops, then ask what it mapped.

Numbers below are what the code being tested achieves with headroom, not its exact
scores; they are thresholds, not a golden file.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import simworld
from simworld import (
    HOUSE,
    ROOMS,
    coverage,
    reachable_mask,
    reachable_residual_frontiers,
    room_coverage,
    run_exploration,
    sealed_house,
    visited,
)

from robot_console.teleop import Command


@pytest.fixture(scope="module")
def run():
    return run_exploration(max_ticks=8000)


@pytest.fixture(scope="module")
def sealed_run():
    return run_exploration(walls=sealed_house(), max_ticks=8000)


# ------------------------------------------------------------------ the world itself


def test_the_robot_moves_in_the_world_frame_not_its_own():
    pose = simworld.step_pose((5.0, 4.0, np.pi / 2), Command(vx=1.0), 1.0, walls=HOUSE)
    assert pose[0] == pytest.approx(5.0, abs=1e-6)
    assert pose[1] == pytest.approx(5.0, abs=1e-6)


def test_the_base_cannot_drive_through_a_wall():
    """Without this a planner that aims through walls would quietly look like it works."""
    against_the_wall = simworld.step_pose((5.0, 6.5, np.pi / 2), Command(vx=1.0), 1.0)
    assert against_the_wall[1] == pytest.approx(6.5)


def test_rotating_on_the_spot_is_always_allowed():
    turned = simworld.step_pose((5.0, 6.5, 0.0), Command(wz=1.0), 1.0)
    assert turned[2] == pytest.approx(1.0)


def test_a_sealed_room_is_not_counted_as_reachable():
    from robot_console.slam.grid import OccupancyGrid

    grid = OccupancyGrid(0.05, width=260, height=180, origin=(-1.0, -1.0))
    open_house = reachable_mask(grid, walls=HOUSE)
    shut = reachable_mask(grid, walls=sealed_house())
    assert shut.sum() < open_house.sum(), "bricking up a doorway removes reachable floor"


# ------------------------------------------------------------------ coverage


def test_exploration_finishes_rather_than_giving_up(run):
    """`explored` means the map ran out of frontiers, not that the run ran out of patience.

    `stalled` and `None` are the two failures this whole change is about: the first is the
    progress watchdog cutting off a livelock, the second is never terminating at all.
    """
    assert run["reason"] == "explored"


def test_exploration_maps_essentially_all_reachable_floor(run):
    assert coverage(run["grid"], reachable_mask(run["grid"])) >= 0.95


def test_exploration_leaves_no_frontier_behind(run):
    """A residual frontier is somewhere the robot knew about and never went."""
    assert reachable_residual_frontiers(run["grid"]) == 0


@pytest.mark.parametrize("room", sorted(ROOMS))
def test_every_room_is_entered_and_mapped(run, room):
    """The assertion a whole-house percentage hides.

    A run can score well overall while never entering one room -- the corridor and the
    rooms it did visit carry the average.
    """
    assert visited(run["history"], ROOMS[room]), f"never drove into {room}"
    assert room_coverage(run["grid"], ROOMS[room]) >= 0.9, room


def test_the_room_behind_the_narrow_doorway_is_reached(run):
    """The regression for the suppression radius.

    The east room's only entrance is a 0.7 m gap. One strike against a goal in it used to
    poison a 0.5 m radius -- more than half the doorway -- and the room stayed unmapped
    for the rest of the run.
    """
    assert visited(run["history"], ROOMS["east"])


# ------------------------------------------------------------------ knowing when to stop


def test_a_sealed_room_does_not_stop_the_run_finishing(sealed_run):
    assert sealed_run["reason"] == "explored"


def test_a_sealed_room_is_not_falsely_reported_as_mapped(sealed_run):
    """Coverage is measured against reachable floor, so the honest answer is 'complete'."""
    grid = sealed_run["grid"]
    assert coverage(grid, reachable_mask(grid, walls=sealed_house())) >= 0.95
    assert room_coverage(grid, ROOMS["east"]) < 0.5, "it cannot have seen inside"
    assert not visited(sealed_run["history"], ROOMS["east"])


def test_a_run_that_starts_facing_a_wall_still_finishes():
    """The hang regression, end to end.

    A goal whose first follower step trips the obstacle brake used to leave the progress
    watchdog unarmed, so `is_stuck` never fired, the explorer rerouted to the same goal
    every tick, and the run neither blacklisted it nor ever terminated.
    """
    result = run_exploration(start=(0.35, 0.35, np.pi), max_ticks=8000)
    assert result["reason"] is not None, "did not terminate"
    assert coverage(result["grid"], reachable_mask(result["grid"], start=(0.35, 0.35, np.pi))) >= 0.9


# ------------------------------------------------------------------ budget


def test_deciding_stays_inside_the_publish_period(run):
    """`/cmd_vel` goes out every 50 ms, and a tick that overruns it publishes late.

    `_Budget` warns after five overruns, so the bar is that a whole house-sized run stays
    under that. Typical ticks are microseconds; only the replans cost anything.
    """
    assert run["worst_decide_ms"] < 100.0


def test_the_whole_run_is_fast_enough_to_be_a_test():
    started = time.perf_counter()
    run_exploration(max_ticks=8000)
    assert time.perf_counter() - started < 30.0
