"""The explorer's goal rules and give-up ladder (console spec §2.2), on hand-built grids.

Every fixture is a 4 x 4 m walled room, fully observed, at the 0.05 m grid. Openings are
cut in its east wall: `gap(rows)` makes those wall cells free and leaves unknown space
behind them, so the frontier is exactly that many cells -- which is what lets each rung
be reached on purpose.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_console.slam import frontier as frontier_mod
from robot_console.slam.explorer import (
    EXPLORED,
    GOAL_PROGRESS_AREA_M2,
    GOAL_PROGRESS_M,
    GOAL_STALL_S,
    GOAL_TOTAL_S,
    LIMIT,
    SWEEP_SECONDS,
    Explorer,
    GoalBlacklist,
    Rung,
)
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import CostMap

ROOM = 80        # cells: 4 m
WIDTH = 120      # the east 40 columns are outside the room
FREE, WALL = -5.0, 5.0
POSE = (1.0, 2.0, 0.0)


def room(*gaps, hole=False) -> OccupancyGrid:
    """The room, with each `gaps` entry a (row0, row1) opening in the east wall."""
    grid = OccupancyGrid(0.05, width=WIDTH, height=ROOM, origin=(0.0, 0.0))
    grid.data[:] = 0.0
    grid.data[:, :ROOM + 1] = FREE
    grid.data[0, :ROOM + 1] = grid.data[-1, :ROOM + 1] = WALL
    grid.data[:, 0] = grid.data[:, ROOM] = WALL
    for row0, row1 in gaps:
        grid.data[row0:row1, ROOM] = FREE
    if hole:
        grid.data[20:22, 30:32] = 0.0    # a 2 x 2 sensor hole, enclosed by free space
    grid.revision += 1
    return grid


def cost_for(grid):
    return CostMap(grid, allow_unknown=True)


def replan(explorer, grid, now=0.0, pose=POSE):
    return explorer.replan(grid, cost_for(grid), pose, now)


def frontier_at(grid, rows):
    """The frontier cluster for the gap at `rows`, as `find_frontiers` sees it."""
    y = (rows[0] + rows[1]) / 2 * 0.05
    return min(frontier_mod.find_frontiers(grid, min_cells=1),
               key=lambda f: abs(f.centroid[1] - y))


# ------------------------------------------------------------------ the fixtures themselves


@pytest.mark.parametrize("width", [2, 4, 10])
def test_a_gap_is_a_frontier_of_exactly_its_width(width):
    grid = room((30, 30 + width))
    sizes = [f.size for f in frontier_mod.find_frontiers(grid, min_cells=1)]
    assert sizes == [width]


def test_a_closed_room_has_no_frontier_and_a_hole_is_only_a_hole():
    assert frontier_mod.find_frontiers(room(), min_cells=1) == []
    assert frontier_mod.find_frontiers(room(hole=True), min_cells=1) == []
    assert len(frontier_mod.unknown_pockets(room(hole=True))) == 1


# ------------------------------------------------------------------ each rung


def test_rung_1_takes_a_frontier_of_six_cells_or_more():
    decision = replan(Explorer(), room((30, 40)))
    assert decision.rung == Rung.LARGE and decision.path is not None


def test_rung_2_takes_three_to_five_cells_when_rung_1_finds_nothing():
    explorer = Explorer()
    decision = replan(explorer, room((30, 34)))
    assert decision.rung == Rung.SMALL and decision.path is not None
    assert not explorer.flushed, "rung 3 is only tried when rung 2 found nothing"


def test_rung_3_clears_timed_out_goals_but_not_fruitless_ones():
    grid = room((10, 20), (60, 70))
    timed_out, fruitless = frontier_at(grid, (10, 20)), frontier_at(grid, (60, 70))
    explorer = Explorer()
    explorer.blacklist.add([timed_out.centroid], GoalBlacklist.TIMEOUT)
    explorer.blacklist.add([fruitless.centroid], GoalBlacklist.FRUITLESS)

    decision = replan(explorer, grid)
    assert decision.rung == Rung.FLUSH
    assert abs(decision.frontier.centroid[1] - timed_out.centroid[1]) < 0.1
    assert explorer.blacklist.count(GoalBlacklist.TIMEOUT) == 0
    assert explorer.blacklist.count(GoalBlacklist.FRUITLESS) == 1


def test_rung_3_runs_once_per_run():
    grid = room((10, 20))
    frontier = frontier_at(grid, (10, 20))
    explorer = Explorer()
    explorer.blacklist.add([frontier.centroid], GoalBlacklist.TIMEOUT)
    assert replan(explorer, grid).rung == Rung.FLUSH
    explorer.blacklist.add([frontier.centroid], GoalBlacklist.TIMEOUT)
    again = replan(explorer, grid, now=10.0)
    assert again.rung != Rung.FLUSH and again.path is None


def test_rung_4_takes_a_frontier_of_one_or_two_cells():
    explorer = Explorer()
    decision = replan(explorer, room((30, 32)))
    assert decision.rung == Rung.ANY and decision.path is not None
    assert explorer.flushed, "rung 3 came first and found nothing"


def test_rung_5_goes_to_an_enclosed_hole():
    decision = replan(Explorer(), room(hole=True))
    assert decision.rung == Rung.HOLES and decision.path is not None
    assert np.allclose(decision.goal, (1.55, 1.05), atol=0.1)


def test_rung_6_is_one_rotation_once_then_explored():
    explorer = Explorer()
    grid = room()
    sweep = replan(explorer, grid, now=0.0)
    assert sweep.rung == Rung.SWEEP and sweep.spin > 0
    assert sweep.spin * SWEEP_SECONDS == pytest.approx(2 * math.pi), "one full rotation"
    assert explorer.sweeping(SWEEP_SECONDS - 0.1) and not explorer.sweeping(SWEEP_SECONDS)
    done = replan(explorer, grid, now=SWEEP_SECONDS + 1)
    assert done.finished == EXPLORED


def test_rung_6_returns_to_rung_1():
    explorer = Explorer()
    replan(explorer, room(), now=0.0)                       # rung 6: the sweep
    after = replan(explorer, room((30, 40)), now=SWEEP_SECONDS + 1)
    assert after.rung == Rung.LARGE, "the sweep uncovered a frontier; back at rung 1"


def test_a_goal_from_any_rung_restarts_the_ladder():
    explorer = Explorer()
    assert replan(explorer, room((30, 32))).rung == Rung.ANY
    assert replan(explorer, room((30, 32), (50, 60)), now=5.0).rung == Rung.LARGE


def test_explored_needs_rung_5_empty_and_rung_6_done():
    explorer = Explorer()
    first = replan(explorer, room(hole=True))
    assert first.rung == Rung.HOLES, "a hole is still something to do"
    explorer.on_arrived(room(), 1.0, area=1e9)             # resolved
    assert replan(explorer, room(), now=2.0).rung == Rung.SWEEP
    assert replan(explorer, room(), now=2.0 + SWEEP_SECONDS).finished == EXPLORED


# ------------------------------------------------------------------ goal progress


def _chosen(explorer, grid):
    decision = replan(explorer, grid)
    assert decision.goal is not None
    return decision


def test_ninety_seconds_without_progress_blacklists_the_goal():
    grid = room((30, 40))
    explorer = Explorer()
    goal = _chosen(explorer, grid).goal.copy()
    for t in range(0, 91):
        assert explorer.track(grid, POSE, float(t), area=10.0) is None
    assert explorer.track(grid, POSE, 90.5, area=10.0) == "timeout"
    assert explorer.goal is None and explorer.blacklist.blocks(goal)
    assert explorer.blacklist.count(GoalBlacklist.TIMEOUT) >= 1


def test_coming_closer_is_progress_but_only_by_0_45_m_past_the_closest_approach():
    grid = room((30, 40))
    explorer = Explorer()
    goal = _chosen(explorer, grid).goal
    start = np.array(POSE[:2])
    towards = (goal - start) / np.linalg.norm(goal - start)
    explorer.track(grid, POSE, 0.0, area=10.0)
    # Creeping closer 0.4 m, then back and forth within it: never progress.
    x = start + towards * (GOAL_PROGRESS_M - 0.05)
    explorer.track(grid, (*x, 0.0), 30.0, area=10.0)
    explorer.track(grid, POSE, 60.0, area=10.0)
    assert explorer.current.progress_events == 0
    # 0.45 m past the closest approach so far is.
    x = start + towards * (2 * GOAL_PROGRESS_M)
    explorer.track(grid, (*x, 0.0), 80.0, area=10.0)
    assert explorer.current.progress_events == 1
    assert explorer.track(grid, (*x, 0.0), 160.0, area=10.0) is None, "80 s since progress"


def test_uncovering_map_is_progress():
    grid = room((30, 40))
    explorer = Explorer()
    _chosen(explorer, grid)
    area = frontier_mod.explored_area(grid)
    for t in range(0, 250, 60):
        area += GOAL_PROGRESS_AREA_M2
        assert explorer.track(grid, POSE, float(t), area=area) is None
    assert explorer.current.progress_events >= 4


def test_three_hundred_seconds_in_total_blacklists_even_a_progressing_goal():
    grid = room((30, 40))
    explorer = Explorer()
    goal = _chosen(explorer, grid).goal.copy()
    area, t, result = frontier_mod.explored_area(grid), 0.0, None
    while result is None and t < 400:
        area += GOAL_PROGRESS_AREA_M2
        result = explorer.track(grid, POSE, t, area=area)
        t += 10.0
    assert result == "timeout"
    assert GOAL_TOTAL_S < t - 10.0 <= GOAL_TOTAL_S + 10.0
    assert explorer.blacklist.blocks(goal)


def test_a_goal_reached_without_uncovering_map_is_never_chosen_again():
    grid = room((30, 40))
    explorer = Explorer()
    goal = _chosen(explorer, grid).goal.copy()
    assert explorer.on_arrived(grid, 5.0, area=frontier_mod.explored_area(grid)) is True
    assert explorer.blacklist.count(GoalBlacklist.FRUITLESS) >= 1
    # Not at rung 1, not after rung 3 clears the timeouts, not at any rung.
    for t in (6.0, 7.0, 8.0):
        decision = replan(explorer, grid, now=t)
        assert decision.goal is None or not np.allclose(decision.goal, goal, atol=0.25)


def test_a_goal_reached_while_uncovering_map_is_not_blacklisted():
    grid = room((30, 40))
    explorer = Explorer()
    goal = _chosen(explorer, grid).goal.copy()
    area = frontier_mod.explored_area(grid) + 1.0
    assert explorer.on_arrived(grid, 5.0, area=area) is False
    assert not explorer.blacklist.blocks(goal)


def test_the_goal_clock_is_kept_across_re_selection():
    """Switching away and back does not reset a goal's 300 s budget."""
    grid = room((30, 40))
    explorer = Explorer()
    _chosen(explorer, grid)
    record = explorer.current
    explorer.track(grid, POSE, 0.0, area=10.0)
    explorer.track(grid, POSE, 50.0, area=10.0)
    explorer.on_blocked(50.0)
    replan(explorer, grid, now=60.0)
    assert explorer.current is record and record.active_s == pytest.approx(50.0)
    assert explorer.attempted == 1


# ------------------------------------------------------------------ the goal limit


def test_max_goals_ends_the_run_with_limit():
    explorer = Explorer(max_goals=2)
    replan(explorer, room((10, 20)))
    explorer.on_arrived(room(), 1.0, area=1e9)
    replan(explorer, room((60, 70)), now=2.0)
    explorer.on_arrived(room(), 3.0, area=1e9)
    decision = replan(explorer, room((35, 45)), now=4.0)
    assert decision.finished == LIMIT and explorer.attempted == 2


# ------------------------------------------------------------------ commitment


def test_a_freshly_chosen_goal_is_not_immediately_reconsidered():
    grid = room((30, 40))
    explorer = Explorer(commit=5.0)
    replan(explorer, grid)
    assert not explorer.needs_replan((3.0, 3.0, 0.0), 1.0)
    assert explorer.needs_replan((3.0, 3.0, 0.0), 10.0)


def test_being_blocked_or_stuck_keeps_the_goal():
    """Neither is a blacklisting rule; the goal's own clock is."""
    grid = room((30, 40))
    explorer = Explorer()
    goal = replan(explorer, grid).goal.copy()
    explorer.on_blocked(1.0)
    explorer.on_stuck(2.0)
    assert explorer.path is None and np.allclose(explorer.goal, goal)
    assert not explorer.blacklist.blocks(goal)
