"""The occupancy grid: coordinates, growth, and what a scan does to it."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import mapped_room, raycast, wall_distance

from robot_console.slam.grid import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid,
    _bresenham_rays,
)
from robot_console.slam.scan import scan_points, transform_points


def test_a_new_grid_is_entirely_unknown():
    grid = OccupancyGrid(0.05, width=10, height=10)
    assert (grid.classify() == UNKNOWN).all()
    assert grid.probability() == pytest.approx(0.5)


def test_world_and_cell_coordinates_round_trip():
    grid = OccupancyGrid(0.05, width=20, height=20, origin=(-1.0, 2.0))
    cells = np.array([[0, 0], [5, 7], [19, 19]])
    back = grid.world_to_cell(grid.cell_to_world(cells))
    assert np.array_equal(back, cells)


def test_cell_zero_sits_at_the_origin():
    """map_server puts the image's lower-left pixel at `origin`; so does this."""
    grid = OccupancyGrid(0.1, origin=(-2.0, 3.0))
    assert grid.cell_to_world(np.array([[0, 0]]))[0] == pytest.approx([-1.95, 3.05])
    assert tuple(grid.world_to_cell(np.array([[-2.0, 3.0]]))[0]) == (0, 0)


def test_a_point_just_below_the_origin_lands_in_a_negative_cell():
    grid = OccupancyGrid(0.1, origin=(0.0, 0.0))
    assert tuple(grid.world_to_cell(np.array([[-0.01, -0.01]]))[0]) == (-1, -1)


def test_growth_keeps_existing_cells_where_they_were():
    """A map that shifts under a pose already computed against it is worse than no map."""
    grid = OccupancyGrid(0.05, width=16, height=16, origin=(0.0, 0.0))
    grid.integrate((0.4, 0.4, 0.0), np.array([[0.6, 0.4]]))
    marked = grid.cell_to_world(np.argwhere(grid.data != 0.0)[:, ::-1])

    assert grid.grow_to_include(np.array([[-3.0, -3.0], [5.0, 5.0]]))
    still_marked = grid.cell_to_world(np.argwhere(grid.data != 0.0)[:, ::-1])
    assert sorted(map(tuple, np.round(marked, 6))) == sorted(map(tuple, np.round(still_marked, 6)))


def test_growth_reports_when_nothing_was_needed():
    grid = OccupancyGrid(0.05, width=40, height=40, origin=(0.0, 0.0))
    assert grid.grow_to_include(np.array([[1.0, 1.0]])) is False


def test_a_ray_carves_free_space_and_marks_its_endpoint():
    grid = OccupancyGrid(0.1, width=40, height=40, origin=(0.0, 0.0))
    for _ in range(6):  # enough observations to pass the decision thresholds
        grid.integrate((0.5, 0.5, 0.0), np.array([[2.5, 0.5]]))
    classes = grid.classify()

    end = grid.world_to_cell(np.array([[2.5, 0.5]]))[0]
    middle = grid.world_to_cell(np.array([[1.5, 0.5]]))[0]
    aside = grid.world_to_cell(np.array([[1.5, 2.5]]))[0]

    assert classes[end[1], end[0]] == OCCUPIED
    assert classes[middle[1], middle[0]] == FREE
    assert classes[aside[1], aside[0]] == UNKNOWN, "nothing was observed off the beam"


def test_repeated_free_observations_do_not_erase_a_wall():
    """Hits weigh more than misses, or traffic in front of a wall would rub it out."""
    grid = OccupancyGrid(0.1, width=40, height=40, origin=(0.0, 0.0))
    for _ in range(30):
        grid.integrate((0.5, 0.5, 0.0), np.array([[2.5, 0.5]]))
    end = grid.world_to_cell(np.array([[2.5, 0.5]]))[0]
    assert grid.classify()[end[1], end[0]] == OCCUPIED


def test_evidence_is_clamped_so_a_door_can_open_again():
    grid = OccupancyGrid(0.1, width=40, height=40, origin=(0.0, 0.0))
    for _ in range(500):
        grid.integrate((0.5, 0.5, 0.0), np.array([[2.5, 0.5]]))
    # Now the obstacle is gone and rays pass through it.
    for _ in range(60):
        grid.integrate((0.5, 0.5, 0.0), np.array([[3.5, 0.5]]))
    end = grid.world_to_cell(np.array([[2.5, 0.5]]))[0]
    assert grid.classify()[end[1], end[0]] == FREE


def test_a_mapped_room_puts_its_walls_where_the_walls_are():
    grid = mapped_room()
    occupied = np.argwhere(grid.classify() == OCCUPIED)
    assert len(occupied) > 100
    distances = wall_distance(grid.cell_to_world(occupied[:, ::-1]))
    # One cell diagonal of slack; the walls are zero-thickness lines and the cells are not.
    assert np.percentile(distances, 95) < grid.resolution * 1.5


def test_a_mapped_room_has_free_space_inside_it():
    grid = mapped_room()
    assert grid.is_free((1.0, 1.0))
    assert grid.is_free((4.5, 2.5))
    assert not grid.is_free((-5.0, -5.0)), "never observed"


def test_integrating_an_empty_cloud_changes_nothing():
    grid = OccupancyGrid(0.05, width=10, height=10)
    grid.integrate((0.0, 0.0, 0.0), np.empty((0, 2)))
    assert (grid.data == 0).all()


def test_max_range_drops_returns_beyond_it():
    grid = OccupancyGrid(0.1, width=80, height=80, origin=(0.0, 0.0))
    grid.integrate((0.5, 0.5, 0.0), np.array([[6.5, 0.5]]), max_range=2.0)
    assert (grid.data == 0).all()


def test_a_zero_length_ray_is_harmless():
    """The endpoint landing in the robot's own cell must not divide by zero."""
    grid = OccupancyGrid(0.1, width=10, height=10, origin=(0.0, 0.0))
    with np.errstate(all="raise"):
        grid.integrate((0.55, 0.55, 0.0), np.array([[0.56, 0.56]]))
    assert np.isfinite(grid.data).all()


def test_bresenham_excludes_the_endpoint():
    cells = _bresenham_rays(np.array([0, 0]), np.array([[4, 0]]))
    assert sorted(map(tuple, cells)) == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_bresenham_covers_a_diagonal_without_gaps():
    cells = {tuple(c) for c in _bresenham_rays(np.array([0, 0]), np.array([[5, 5]]))}
    assert cells == {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}


def test_copy_is_independent():
    grid = mapped_room()
    clone = grid.copy()
    clone.integrate((1.0, 1.0, 0.0), np.array([[2.0, 1.0]]))
    assert not np.array_equal(grid.data, clone.data)


def test_revision_advances_on_change():
    grid = OccupancyGrid(0.05, width=10, height=10)
    before = grid.revision
    grid.integrate((0.1, 0.1, 0.0), np.array([[0.3, 0.1]]))
    assert grid.revision > before
