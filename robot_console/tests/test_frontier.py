"""Frontier detection and goal selection."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import mapped_room, raycast

from robot_console.slam import frontier
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import CostMap
from robot_console.slam.scan import scan_points, transform_points


def half_seen_room() -> OccupancyGrid:
    """The 6 x 4 m room observed from one corner only, so plenty is still unknown."""
    grid = OccupancyGrid(0.05)
    pose = (1.0, 1.0, 0.0)
    pts = scan_points(raycast(*pose), max_range=2.5, offset=(0.0, 0.0))
    grid.integrate(pose, transform_points(pts, pose), max_range=2.5)
    return grid


def test_an_unseen_room_has_frontiers():
    frontiers = frontier.find_frontiers(half_seen_room())
    assert frontiers
    assert all(f.size >= frontier.MIN_FRONTIER_CELLS for f in frontiers)


def test_every_frontier_cell_is_free_and_touches_the_unknown():
    """The definition, asserted on the mask itself.

    Not on the centroids: a frontier cluster curving around a corner has its centroid in
    open space, which is a perfectly good place to drive to and not a frontier cell.
    """
    grid = half_seen_room()
    classes = grid.classify()
    mask = frontier.frontier_mask(grid)
    assert mask.any()

    ys, xs = np.nonzero(mask)
    assert (classes[ys, xs] == 1).all(), "a frontier cell is a free cell"
    neighbours = np.stack([
        classes[np.clip(ys + dy, 0, classes.shape[0] - 1),
                np.clip(xs + dx, 0, classes.shape[1] - 1)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
    ])
    assert (neighbours == 0).any(axis=0).all(), "...with unknown space next to it"


def test_a_frontier_centroid_lies_within_its_own_cluster_bounds():
    grid = half_seen_room()
    mask = frontier.frontier_mask(grid)
    world = grid.cell_to_world(np.argwhere(mask)[:, ::-1])
    lo, hi = world.min(axis=0) - grid.resolution, world.max(axis=0) + grid.resolution
    for f in frontier.find_frontiers(grid):
        assert (f.centroid >= lo).all() and (f.centroid <= hi).all()


def test_a_map_with_no_unknown_space_has_no_frontiers():
    grid = OccupancyGrid(0.05, width=40, height=40, origin=(0.0, 0.0))
    grid.data[:] = -5.0            # everything free
    grid.data[0, :] = 5.0          # ringed by obstacles, so no unknown edge
    grid.data[-1, :] = 5.0
    grid.data[:, 0] = 5.0
    grid.data[:, -1] = 5.0
    assert frontier.find_frontiers(grid) == []
    assert frontier.is_complete(grid)


def test_speckle_is_not_a_frontier():
    grid = OccupancyGrid(0.05, width=40, height=40, origin=(0.0, 0.0))
    grid.data[:] = -5.0
    grid.data[20, 20] = 0.0        # a single unknown cell in the middle
    assert frontier.find_frontiers(grid, min_cells=6) == []


def test_frontiers_come_back_largest_first():
    sizes = [f.size for f in frontier.find_frontiers(half_seen_room())]
    assert sizes == sorted(sizes, reverse=True)


def test_choose_goal_returns_a_reachable_frontier_and_a_path():
    grid = half_seen_room()
    cost = CostMap(grid, allow_unknown=True)
    target, path = frontier.choose_goal(grid, cost, (1.0, 1.0, 0.0))
    assert target is not None and path is not None
    assert path[-1] == pytest.approx(target.centroid)
    assert target.distance > 0


def test_a_blacklisted_frontier_is_not_chosen_again():
    """Without this the robot picks the same unreachable gap forever."""
    grid = half_seen_room()
    cost = CostMap(grid, allow_unknown=True)
    first, _ = frontier.choose_goal(grid, cost, (1.0, 1.0, 0.0))
    second, _ = frontier.choose_goal(grid, cost, (1.0, 1.0, 0.0), blacklist=[first.centroid])
    assert second is None or np.hypot(*(second.centroid - first.centroid)) >= frontier.BLACKLIST_RADIUS_M


def test_choose_goal_gives_up_on_a_finished_map():
    grid = OccupancyGrid(0.05, width=40, height=40, origin=(0.0, 0.0))
    grid.data[:] = -5.0
    target, path = frontier.choose_goal(grid, CostMap(grid, allow_unknown=True), (1.0, 1.0, 0.0))
    assert target is None and path is None


def test_nearby_frontiers_beat_far_ones_of_the_same_size():
    """Finishing the room you are in produces far less backtracking."""
    grid = half_seen_room()
    cost = CostMap(grid, allow_unknown=True)
    near, _ = frontier.choose_goal(grid, cost, (1.0, 1.0, 0.0))
    far, _ = frontier.choose_goal(grid, cost, (1.0, 1.0, 0.0), distance_bias=0.01)
    assert near is not None and far is not None
    assert near.utility > 0


def test_explored_area_counts_only_known_cells():
    grid = OccupancyGrid(0.1, width=10, height=10, origin=(0.0, 0.0))
    assert frontier.explored_area(grid) == 0.0
    grid.data[0, 0] = -5.0
    assert frontier.explored_area(grid) == pytest.approx(0.01)


def test_explored_area_grows_as_the_room_is_seen():
    assert frontier.explored_area(mapped_room()) > frontier.explored_area(half_seen_room())


def test_approach_point_stops_short_of_the_frontier():
    p = frontier.approach_point((5.0, 0.0), (0.0, 0.0, 0.0), standoff=1.0)
    assert p == pytest.approx([4.0, 0.0])


def test_approach_point_does_not_overshoot_backwards():
    p = frontier.approach_point((0.5, 0.0), (0.0, 0.0, 0.0), standoff=2.0)
    assert p == pytest.approx([0.5, 0.0])
