"""The give-up ladder, goal commitment, and the progress watchdog."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import raycast

from robot_console.slam.explorer import Explorer, Stage
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import CostMap
from robot_console.slam.scan import scan_points, transform_points


def half_seen_room() -> OccupancyGrid:
    grid = OccupancyGrid(0.05)
    pose = (1.0, 1.0, 0.0)
    pts = scan_points(raycast(*pose), max_range=2.5, offset=(0.0, 0.0))
    grid.integrate(pose, transform_points(pts, pose), max_range=2.5)
    return grid


def finished_map() -> OccupancyGrid:
    grid = OccupancyGrid(0.05, width=60, height=60, origin=(0.0, 0.0))
    grid.data[:] = -5.0
    grid.data[0, :] = grid.data[-1, :] = 5.0
    grid.data[:, 0] = grid.data[:, -1] = 5.0
    grid.revision += 1
    return grid


def cost_for(grid):
    return CostMap(grid, allow_unknown=True)


def test_a_map_with_nothing_left_finishes():
    """It takes one last look round first, then stops. Both halves matter."""
    grid = finished_map()
    explorer = Explorer(sweep_seconds=5.0)
    cost = cost_for(grid)

    looking = explorer.replan(grid, cost, (1.0, 1.0, 0.0), 0.0)
    assert looking.finished is None and looking.spin != 0.0

    explorer.replan(grid, cost, (1.0, 1.0, 0.0), 10.0)   # sweep window elapses
    done = explorer.replan(grid, cost, (1.0, 1.0, 0.0), 20.0)
    assert done.finished == "explored"
    assert done.path is None


def test_an_ordinary_map_yields_a_goal_at_the_first_rung():
    grid = half_seen_room()
    explorer = Explorer()
    decision = explorer.replan(grid, cost_for(grid), (1.0, 1.0, 0.0), 0.0)
    assert decision.path is not None and decision.goal is not None
    assert explorer.stage is Stage.NORMAL, "a productive rung resets the ladder"


def test_the_ladder_relaxes_the_threshold_before_giving_up():
    """Defect F: a doorway seen edge-on at range is a two- or three-cell frontier.

    With `min_cells` pinned high, the first rung finds nothing; the explorer must try
    harder rather than call the map finished.
    """
    grid = half_seen_room()
    explorer = Explorer(min_cells=10_000)
    decision = explorer.replan(grid, cost_for(grid), (1.0, 1.0, 0.0), 0.0)
    assert decision.finished is None
    assert decision.path is not None, "a lower threshold should still find something"


def test_a_freshly_chosen_goal_is_not_immediately_reconsidered():
    """Defect G: full re-selection every 0.6 m made comparable goals trade places."""
    grid = half_seen_room()
    explorer = Explorer(commit=5.0)
    explorer.replan(grid, cost_for(grid), (1.0, 1.0, 0.0), 0.0)
    assert not explorer.needs_replan((9.0, 9.0, 0.0), 1.0), "still inside the commitment"
    assert explorer.needs_replan((9.0, 9.0, 0.0), 10.0)


def test_a_replan_is_forced_once_the_path_is_gone():
    explorer = Explorer(commit=5.0)
    grid = half_seen_room()
    explorer.replan(grid, cost_for(grid), (1.0, 1.0, 0.0), 0.0)
    explorer.on_blocked(1.0)
    assert explorer.needs_replan((1.0, 1.0, 0.0), 1.0), "no path means replan regardless"


def test_being_blocked_keeps_the_goal_but_being_stuck_strikes_it():
    """Blocked is a local obstruction; stuck is the goal's fault.

    Suppressing on every brake would burn through every frontier in the house in seconds
    while the robot never moved.
    """
    grid = half_seen_room()
    explorer = Explorer()
    explorer.replan(grid, cost_for(grid), (1.0, 1.0, 0.0), 0.0)
    goal = explorer.goal.copy()

    explorer.on_blocked(1.0)
    assert explorer.goal is not None and not explorer.blacklist.blocks(goal, 1.0)

    explorer.goal = goal
    explorer.on_stuck(2.0)
    assert explorer.blacklist.blocks(goal, 2.0)


def test_arriving_retires_the_point_that_was_arrived_at():
    """A frontier you are standing on has shown you everything it is going to."""
    grid = half_seen_room()
    explorer = Explorer()
    explorer.replan(grid, cost_for(grid), (1.0, 1.0, 0.0), 0.0)
    goal = explorer.goal.copy()
    explorer.on_arrived(1.0)
    assert explorer.blacklist.blocks(goal, 10_000.0), "retirement outlasts the TTL"


def test_the_watchdog_fires_when_nothing_is_happening():
    grid = finished_map()
    # Carve a little unknown so there is always something to aim at, and never map it.
    grid.data[20:24, 20:24] = 0.0
    grid.revision += 1
    explorer = Explorer(stall_seconds=10.0)
    cost = cost_for(grid)
    explorer.replan(grid, cost, (1.0, 1.0, 0.0), 0.0)
    late = explorer.replan(grid, cost, (1.0, 1.0, 0.0), 1000.0)
    assert late.finished == "stalled"


def test_the_watchdog_does_not_fire_while_ground_is_being_mapped():
    """Replans a few seconds apart, as a real run has them, while the map grows."""
    explorer = Explorer(stall_seconds=30.0)
    grid = half_seen_room()
    explorer.replan(grid, cost_for(grid), (1.0, 1.0, 0.0), 0.0)
    for step, pose in enumerate([(1.6, 1.2), (2.4, 1.6), (3.4, 2.2), (4.4, 2.6)], start=1):
        pts = scan_points(raycast(*pose, 0.0), max_range=2.5, offset=(0.0, 0.0))
        grid.integrate((*pose, 0.0), transform_points(pts, (*pose, 0.0)), max_range=2.5)
        decision = explorer.replan(grid, cost_for(grid), (*pose, 0.0), step * 3.0)
        assert decision.finished is None, "mapping new ground is not a stall"


def test_retiring_goals_counts_as_progress_for_the_watchdog():
    """Closing the last sensor holes maps almost no new area, by definition.

    Without this the watchdog fires on exactly the endgame it exists to protect, and a
    run that was two goals from finishing reports `stalled` instead.
    """
    grid = half_seen_room()
    explorer = Explorer(stall_seconds=30.0)
    cost = cost_for(grid)
    explorer.replan(grid, cost, (1.0, 1.0, 0.0), 0.0)
    for step in range(1, 6):
        explorer.goal = np.array([1.0 + 0.3 * step, 1.0])
        explorer.on_arrived(step * 20.0)
        decision = explorer.replan(grid, cost, (1.0, 1.0, 0.0), step * 20.0)
        assert decision.finished != "stalled"


def test_the_sweep_happens_at_most_once():
    grid = finished_map()
    explorer = Explorer(sweep_seconds=5.0)
    cost = cost_for(grid)
    # Force the ladder to the sweep rung by hand, then let it run to completion twice.
    explorer._stage = Stage.SWEEP
    first = explorer.replan(grid, cost, (1.0, 1.0, 0.0), 0.0)
    assert first.spin != 0.0
    explorer._stage = Stage.SWEEP
    explorer.replan(grid, cost, (1.0, 1.0, 0.0), 100.0)   # sweep window elapsed
    explorer._stage = Stage.SWEEP
    again = explorer.replan(grid, cost, (1.0, 1.0, 0.0), 200.0)
    assert again.spin == 0.0, "a second sweep would be a spin loop"


def test_the_robot_radius_option_reaches_the_costmap():
    """`--robot-radius` was parsed, clamped, stored -- and never passed anywhere."""
    from robot_console.slam.planner import CLEARANCE_M, costmap_for

    grid = half_seen_room()
    narrow = costmap_for(grid, None, allow_unknown=True, radius=0.10 + CLEARANCE_M)
    wide = costmap_for(grid, None, allow_unknown=True, radius=0.40 + CLEARANCE_M)
    assert wide.blocked.sum() > narrow.blocked.sum()
