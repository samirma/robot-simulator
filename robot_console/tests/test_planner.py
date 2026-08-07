"""Obstacle inflation, A*, and path simplification."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import mapped_room

from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import (
    ROBOT_RADIUS_M,
    CostMap,
    choose_decimation,
    costmap_for,
    descend,
    distance_field,
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


# ------------------------------------------------------------------ distance field


def test_the_wavefront_reaches_the_far_side_through_the_gap(corridor):
    cost = CostMap(corridor, radius=0.25)
    field = distance_field(cost, (1.0, 2.0))
    assert field.converged
    assert field.reachable((5.0, 2.0)), "the gap at y=2 is open"


def test_the_wavefront_does_not_leak_through_a_closed_wall():
    grid = corridor_grid()
    wall = int(3.0 / grid.resolution)
    grid.data[:, wall] = 5.0  # seal the gap
    grid.revision += 1
    field = distance_field(CostMap(grid, radius=0.25), (1.0, 2.0))
    assert field.reachable((1.5, 2.0))
    assert not field.reachable((5.0, 2.0))


def test_the_wavefront_cost_grows_with_distance(corridor):
    field = distance_field(CostMap(corridor, radius=0.25), (1.0, 2.0))
    assert field.at((1.0, 2.0)) == pytest.approx(0.0)
    assert field.at((2.5, 2.0)) < field.at((5.0, 2.0))


def test_a_detour_costs_more_than_the_straight_line_suggests(corridor):
    """The whole reason for a geodesic field: the gap is the only way through.

    A point just past the wall is barely a metre away in a straight line, but every route
    to it goes via y=2. Ranking it by straight-line distance is what sends the explorer at
    frontiers it cannot reach.
    """
    field = distance_field(CostMap(corridor, radius=0.25), (2.5, 0.5))
    near_as_the_crow_flies = field.at((3.5, 0.5))
    through_the_gap = field.at((3.5, 2.0))
    assert near_as_the_crow_flies > through_the_gap


def test_the_wavefront_agrees_with_astar_about_reachability(corridor):
    cost = CostMap(corridor, radius=0.25)
    start = (1.0, 2.0)
    field = distance_field(cost, start)
    for goal in ((1.5, 1.0), (5.0, 2.0), (5.5, 3.5), (2.0, 3.0)):
        assert field.reachable(goal) == (plan(cost, start, goal) is not None), goal


def test_the_field_is_seeded_even_when_the_start_is_inflated(corridor):
    """The robot's own footprint is inflated, so its cell is routinely 'blocked'."""
    cost = CostMap(corridor, radius=0.25)
    hard_against_the_wall = (0.15, 2.0)
    assert cost.is_blocked(cost.world_to_cell(hard_against_the_wall))
    field = distance_field(cost, hard_against_the_wall)
    assert field.at(hard_against_the_wall) == pytest.approx(0.0)


def test_a_start_outside_the_map_reaches_nothing(corridor):
    field = distance_field(CostMap(corridor, radius=0.25), (-50.0, -50.0))
    assert not field.reachable((1.0, 2.0))


def test_at_cells_matches_at_for_the_same_points(corridor):
    cost = CostMap(corridor, radius=0.25)
    field = distance_field(cost, (1.0, 2.0))
    points = [(1.5, 2.0), (2.5, 1.0), (5.0, 2.0)]
    cells = np.array([cost.world_to_cell(p) for p in points])
    np.testing.assert_allclose(
        field.at_cells(cells), [field.at(p) for p in points], rtol=0, atol=0
    )


def test_descending_the_field_arrives_at_the_source(corridor):
    field = distance_field(CostMap(corridor, radius=0.25), (1.0, 2.0))
    walk = descend(field, (5.0, 2.0))
    assert walk is not None and len(walk) > 1
    assert np.hypot(*(walk[-1] - np.array([1.0, 2.0]))) < 0.5


def test_descending_from_an_unreachable_cell_gives_nothing():
    grid = corridor_grid()
    wall = int(3.0 / grid.resolution)
    grid.data[:, wall] = 5.0
    grid.revision += 1
    field = distance_field(CostMap(grid, radius=0.25), (1.0, 2.0))
    assert descend(field, (5.0, 2.0)) is None


def test_a_coarser_field_is_cheaper_and_still_reaches(corridor):
    cost = CostMap(corridor, radius=0.25)
    coarse = distance_field(cost, (1.0, 2.0), decimate=3)
    assert coarse.shape[0] < distance_field(cost, (1.0, 2.0), decimate=1).shape[0]
    assert coarse.reachable((5.0, 2.0))


def test_the_costmap_cache_notices_a_changed_radius(corridor):
    wide = costmap_for(corridor, None, radius=0.25)
    assert costmap_for(corridor, wide, radius=0.25) is wide
    assert costmap_for(corridor, wide, radius=0.40) is not wide


def sealed_corridor() -> OccupancyGrid:
    grid = corridor_grid()
    grid.data[:, int(3.0 / grid.resolution)] = 5.0
    grid.revision += 1
    return grid


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_coarsening_never_seals_a_doorway(corridor, stride):
    """The asymmetry that sets the coarsening policy.

    Calling a route closed abandons every frontier behind it -- a whole room -- while
    calling it open costs one A* that returns None. So coarsening is optimistic, and this
    pins that a 0.8 m gap survives every stride the budget can pick.
    """
    cost = CostMap(corridor, radius=0.25)
    assert distance_field(cost, (1.0, 2.0), decimate=stride).reachable((5.0, 2.0))


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_coarsening_does_not_leak_through_a_sealed_wall(stride):
    """The other side of optimism: a wall is inflated to ~10 cells, so it must survive."""
    cost = CostMap(sealed_corridor(), radius=0.25)
    assert not distance_field(cost, (1.0, 2.0), decimate=stride).reachable((5.0, 2.0))


def test_the_stride_grows_with_the_open_area(corridor):
    small = CostMap(corridor, radius=0.25)
    assert choose_decimation(small, budget=10_000_000) == 1
    assert choose_decimation(small, budget=1) == 4
    assert choose_decimation(small, budget=int((~small.blocked).sum() / 4) + 1) == 2


def test_the_default_field_sizes_its_own_stride(corridor):
    cost = CostMap(corridor, radius=0.25)
    field = distance_field(cost, (1.0, 2.0))
    assert field.decimate == choose_decimation(cost)
    assert field.converged
