"""Obstacle inflation, A*, and path simplification."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import mapped_room

from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import (
    ROBOT_RADIUS_M,
    CostMap,
    costmap_for,
    line_of_sight,
    path_length,
    plan,
)


def corridor_grid(resolution: float = 0.1) -> OccupancyGrid:
    """A 6 x 4 m box split by a wall with a 0.8 m gap at y = 2."""
    grid = OccupancyGrid(resolution, width=60, height=40, origin=(0.0, 0.0))
    free, occupied = -5.0, 5.0
    grid.data[:] = free
    grid.data[0, :] = occupied
    grid.data[-1, :] = occupied
    grid.data[:, 0] = occupied
    grid.data[:, -1] = occupied
    wall = int(3.0 / resolution)
    grid.data[:, wall] = occupied
    gap = slice(int(1.6 / resolution), int(2.4 / resolution))
    grid.data[gap, wall] = free
    grid.revision += 1
    return grid


@pytest.fixture(scope="module")
def corridor():
    return corridor_grid()


def test_the_footprint_matches_the_myagv_chassis():
    """0.311 x 0.230 m gives a ~0.194 m circumscribed radius (robots/myagv/make_model.py)."""
    assert ROBOT_RADIUS_M == pytest.approx(0.20, abs=0.01)


def test_inflation_blocks_the_cells_next_to_a_wall(corridor):
    cost = CostMap(corridor, radius=0.25)
    assert cost.is_blocked(cost.world_to_cell((0.05, 2.0))), "hard against the west wall"
    assert not cost.is_blocked(cost.world_to_cell((1.5, 2.0)))


def test_a_path_goes_through_the_gap(corridor):
    cost = CostMap(corridor)
    path = plan(cost, (1.0, 2.0), (5.0, 2.0))
    assert path is not None
    assert path_length(path) >= 4.0
    # Every waypoint pair must cross the wall only where the gap is.
    for point in path:
        assert not cost.is_blocked(cost.world_to_cell(point))


def test_no_path_when_the_gap_is_closed(corridor):
    sealed = corridor.copy()
    sealed.data[:, int(3.0 / sealed.resolution)] = 5.0
    sealed.revision += 1
    assert plan(CostMap(sealed), (1.0, 2.0), (5.0, 2.0)) is None


def test_a_goal_outside_the_map_has_no_path(corridor):
    assert plan(CostMap(corridor), (1.0, 2.0), (-8.0, -8.0)) is None


def test_a_path_never_enters_the_inflation_radius(corridor):
    cost = CostMap(corridor)
    path = plan(cost, (0.6, 0.6), (5.4, 3.4))
    assert path is not None
    dense = np.vstack([
        a + (b - a) * np.linspace(0, 1, 40)[:, None] for a, b in zip(path[:-1], path[1:])
    ])
    assert not any(cost.is_blocked(cost.world_to_cell(p)) for p in dense)


def test_unknown_space_blocks_navigation_but_not_exploration():
    """Navigating a saved map must not route through what no sensor has ever seen."""
    grid = mapped_room()
    strict = CostMap(grid, allow_unknown=False)
    loose = CostMap(grid, allow_unknown=True)
    outside = strict.world_to_cell((-2.0, -2.0))
    assert strict.is_blocked(outside)
    assert not loose.is_blocked(outside)


def test_a_start_wedged_against_a_wall_is_snapped_out(corridor):
    """The robot's own footprint is inflated too, so where it stands reads as blocked."""
    cost = CostMap(corridor)
    start = (0.12, 2.0)
    assert cost.is_blocked(cost.world_to_cell(start))
    assert plan(cost, start, (2.0, 2.0)) is not None
    assert plan(cost, start, (2.0, 2.0), snap=False) is None


def test_the_path_starts_and_ends_where_it_was_asked_to(corridor):
    path = plan(CostMap(corridor), (1.0, 1.0), (5.0, 3.0))
    assert path[0] == pytest.approx([1.0, 1.0])
    assert path[-1] == pytest.approx([5.0, 3.0])


def test_a_straight_run_collapses_to_two_points(corridor):
    path = plan(CostMap(corridor), (1.0, 2.0), (2.5, 2.0))
    assert len(path) == 2, "a staircase of waypoints makes the base weave"


def test_line_of_sight_sees_through_open_space_and_not_through_walls(corridor):
    cost = CostMap(corridor)
    assert line_of_sight(cost, (1.0, 1.0), (2.5, 1.0))
    assert not line_of_sight(cost, (1.0, 1.0), (5.0, 1.0)), "the dividing wall is between them"


def test_path_length_of_nothing_is_zero():
    assert path_length(None) == 0.0
    assert path_length(np.empty((0, 2))) == 0.0


def test_the_costmap_is_cached_until_the_map_changes(corridor):
    first = costmap_for(corridor, None)
    assert costmap_for(corridor, first) is first
    assert costmap_for(corridor, first, allow_unknown=True) is not first
    changed = corridor.copy()
    changed.integrate((1.0, 1.0, 0.0), np.array([[1.5, 1.0]]))
    assert costmap_for(changed, first) is not first


def test_planning_to_where_you_already_are(corridor):
    path = plan(CostMap(corridor), (2.0, 2.0), (2.0, 2.0))
    assert path is not None and len(path) >= 1
