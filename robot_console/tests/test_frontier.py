"""Frontier detection and goal selection."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import mapped_room, raycast

from robot_console.slam import frontier
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import CostMap, distance_field
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
    # The path ends at the standoff goal, not the centroid: driving the base's centre onto
    # the boundary of the unknown parks it inside its own obstacle brake.
    assert path[-1] == pytest.approx(target.target)
    assert target.goal is not None
    assert target.distance > 0


def test_the_goal_stands_off_the_boundary_it_is_aimed_at():
    """Measured against the nearest frontier *cell*, not the centroid.

    A cluster that curves around the robot has its centroid in the middle of the arc,
    metres from any cell of it -- which is exactly why the goal is no longer the centroid.
    """
    grid = half_seen_room()
    target, _ = frontier.choose_goal(grid, CostMap(grid, allow_unknown=True), (1.0, 1.0, 0.0))
    cells = grid.cell_to_world(np.argwhere(frontier.frontier_mask(grid))[:, ::-1])
    gap = float(np.linalg.norm(cells - target.goal, axis=1).min())
    assert gap <= frontier.FRONTIER_STANDOFF_M + 2 * grid.resolution


def test_the_goal_is_somewhere_the_wavefront_actually_reached():
    grid = half_seen_room()
    cost = CostMap(grid, allow_unknown=True)
    field = distance_field(cost, (1.0, 1.0))
    target, _ = frontier.choose_goal(grid, cost, (1.0, 1.0, 0.0), field=field)
    assert field.reachable(target.goal)


def test_a_corridor_frontier_beats_a_nearer_one_through_a_wall():
    """Defect D, the reason the ranking is geodesic.

    Straight-line distance made a frontier one metre away through a wall outrank one three
    metres down an open corridor, so the robot repeatedly chose goals it then could not
    plan to.
    """
    res = 0.05
    grid = OccupancyGrid(res, width=int(8 / res), height=int(4 / res), origin=(0.0, 0.0))
    grid.data[:] = -5.0
    grid.data[0, :] = 5.0
    grid.data[-1, :] = 5.0
    grid.data[:, 0] = 5.0
    grid.data[:, -1] = 5.0
    wall = int(4.0 / res)
    grid.data[:, wall] = 5.0                       # sealed, no way through
    grid.data[int(3.0 / res):, int(1.0 / res):int(2.0 / res)] = 0.0   # unknown, same side
    grid.data[int(0.5 / res):int(1.5 / res), wall + 4:wall + 20] = 0.0  # unknown, far side
    grid.revision += 1

    pose = (2.0, 2.0, 0.0)
    ranked = frontier.rank_frontiers(
        grid, distance_field(CostMap(grid, allow_unknown=True), pose[:2])
    )
    same_side = [f for f in ranked if f.centroid[0] < 4.0]
    far_side = [f for f in ranked if f.centroid[0] > 4.0]
    assert same_side and same_side[0].reachable
    assert all(not f.reachable for f in far_side), "nothing behind a sealed wall is reachable"
    assert ranked[0].centroid[0] < 4.0


def test_a_curved_cluster_gets_a_goal_off_its_own_centroid():
    """Defect C: a C-shaped cluster's centroid sits in the wall the cluster wraps."""
    res = 0.05
    grid = OccupancyGrid(res, width=int(6 / res), height=int(6 / res), origin=(0.0, 0.0))
    grid.data[:] = -5.0
    # A blob of occupied space with unknown wrapped around three of its sides.
    lo, hi = int(2.0 / res), int(4.0 / res)
    grid.data[lo:hi, lo:hi] = 5.0
    grid.data[lo - 4:hi + 4, lo - 4:lo] = 0.0
    grid.data[lo - 4:lo, lo - 4:hi + 4] = 0.0
    grid.data[hi:hi + 4, lo - 4:hi + 4] = 0.0
    grid.revision += 1
    cost = CostMap(grid, allow_unknown=True)
    ranked = frontier.rank_frontiers(grid, distance_field(cost, (0.5, 0.5)))
    assert ranked
    for f in ranked:
        if f.reachable:
            assert not cost.is_blocked(cost.world_to_cell(f.goal)), "goal must be drivable"


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
    """Finishing the room you are in produces far less backtracking.

    The version this replaces asserted only `near.utility > 0` and never compared the two,
    so the distance weighting it was named for was in fact unpinned.
    """
    res = 0.05
    grid = OccupancyGrid(res, width=int(10 / res), height=int(3 / res), origin=(0.0, 0.0))
    grid.data[:] = -5.0
    grid.data[0, :] = 5.0
    grid.data[-1, :] = 5.0
    grid.data[:, 0] = 5.0
    grid.data[:, -1] = 5.0
    band = slice(int(1.2 / res), int(1.8 / res))
    grid.data[band, int(2.0 / res):int(2.3 / res)] = 0.0   # near
    grid.data[band, int(8.0 / res):int(8.3 / res)] = 0.0   # far, identical size
    grid.revision += 1

    ranked = frontier.rank_frontiers(
        grid, distance_field(CostMap(grid, allow_unknown=True), (1.0, 1.5))
    )
    near = min(ranked, key=lambda f: f.centroid[0])
    far = max(ranked, key=lambda f: f.centroid[0])
    assert near.size == pytest.approx(far.size, rel=0.3), "comparable clusters"
    assert near.distance < far.distance
    assert near.utility > far.utility


def test_the_incumbent_goal_wins_a_tie():
    """Defect G: without a bonus the robot dithers between two comparable frontiers."""
    res = 0.05
    grid = OccupancyGrid(res, width=int(10 / res), height=int(3 / res), origin=(0.0, 0.0))
    grid.data[:] = -5.0
    grid.data[0, :] = 5.0
    grid.data[-1, :] = 5.0
    grid.data[:, 0] = 5.0
    grid.data[:, -1] = 5.0
    band = slice(int(1.2 / res), int(1.8 / res))
    grid.data[band, int(2.0 / res):int(2.3 / res)] = 0.0
    grid.data[band, int(8.0 / res):int(8.3 / res)] = 0.0
    grid.revision += 1
    field = distance_field(CostMap(grid, allow_unknown=True), (5.0, 1.5))

    plain = frontier.rank_frontiers(grid, field)
    far = max(plain, key=lambda f: f.centroid[0])
    held = frontier.rank_frontiers(grid, field, incumbent=far.centroid)
    assert max(held, key=lambda f: f.centroid[0]).utility > far.utility


# ------------------------------------------------------------------ blacklist


def test_a_blacklisted_goal_comes_back_after_its_ttl():
    """Append-only suppression is why whole rooms went unmapped."""
    bl = frontier.Blacklist(ttl=10.0, strikes=3)
    bl.strike((1.0, 1.0), now=0.0)
    assert bl.blocks((1.0, 1.0), now=5.0)
    assert not bl.blocks((1.0, 1.0), now=20.0)


def test_repeated_failures_make_suppression_permanent():
    bl = frontier.Blacklist(ttl=10.0, strikes=2)
    bl.strike((1.0, 1.0), now=0.0)
    bl.strike((1.0, 1.0), now=1.0)
    assert bl.blocks((1.0, 1.0), now=10_000.0)


def test_suppression_does_not_span_a_doorway():
    """A doorway is ~0.8 m; the old 0.5 m radius sealed one from a single strike."""
    bl = frontier.Blacklist()
    bl.strike((0.0, 0.0), now=0.0)
    assert not bl.blocks((0.0, 0.45), now=1.0)


def test_expire_drops_stale_entries_but_keeps_struck_out_ones():
    bl = frontier.Blacklist(ttl=10.0, strikes=2)
    bl.strike((0.0, 0.0), now=0.0)
    bl.strike((5.0, 5.0), now=0.0)
    bl.strike((5.0, 5.0), now=0.0)
    bl.expire(now=100.0)
    assert len(bl) == 1
    assert bl.blocks((5.0, 5.0), now=100.0)


def test_clearing_the_blacklist_frees_everything():
    bl = frontier.Blacklist()
    bl.strike((1.0, 1.0), now=0.0)
    bl.clear()
    assert not bl.blocks((1.0, 1.0), now=0.0)


# ------------------------------------------------------------------ outcomes


def test_a_finished_map_is_reported_as_finished():
    grid = OccupancyGrid(0.05, width=40, height=40, origin=(0.0, 0.0))
    grid.data[:] = -5.0
    choice = frontier.survey(grid, CostMap(grid, allow_unknown=True), (1.0, 1.0, 0.0))
    assert choice.reason == "finished" and not choice.found


def test_a_suppressed_map_is_not_reported_as_finished():
    """Defect B: the two used to be indistinguishable, so the run called itself done."""
    grid = half_seen_room()
    cost = CostMap(grid, allow_unknown=True)
    ranked = frontier.rank_frontiers(grid, distance_field(cost, (1.0, 1.0)))
    bl = frontier.Blacklist(radius=50.0)
    bl.strike(ranked[0].centroid, now=0.0)
    for f in ranked:
        bl.strike(f.centroid, now=0.0)
    choice = frontier.survey(grid, cost, (1.0, 1.0, 0.0), blacklist=bl, now=1.0)
    assert choice.reason == "blacklisted"


def test_relaxing_min_cells_finds_frontiers_a_strict_threshold_hides():
    """Defect F: a doorway seen edge-on at range is a two- or three-cell frontier."""
    grid = half_seen_room()
    cost = CostMap(grid, allow_unknown=True)
    strict = frontier.survey(grid, cost, (1.0, 1.0, 0.0), min_cells=10_000)
    assert not strict.found and strict.reason in ("unreachable", "blacklisted")
    assert frontier.survey(grid, cost, (1.0, 1.0, 0.0), min_cells=1).found


# ------------------------------------------------------------------ pockets


def test_an_enclosed_sensor_hole_is_a_pocket():
    grid = OccupancyGrid(0.05, width=100, height=100, origin=(0.0, 0.0))
    grid.data[:] = -5.0
    grid.data[50:53, 50:53] = 0.0        # a few weakly-seen cells in the middle
    grid.revision += 1
    pockets = frontier.unknown_pockets(grid)
    assert len(pockets) == 1 and pockets[0].size == 9


def test_unexplored_space_at_the_map_edge_is_not_a_pocket():
    grid = OccupancyGrid(0.05, width=100, height=100, origin=(0.0, 0.0))
    grid.data[:] = -5.0
    grid.data[:, 90:] = 0.0              # touches the edge: unexplored, not a hole
    grid.revision += 1
    assert frontier.unknown_pockets(grid) == []


def test_a_large_enclosed_region_is_a_room_not_a_hole():
    grid = OccupancyGrid(0.05, width=200, height=200, origin=(0.0, 0.0))
    grid.data[:] = -5.0
    grid.data[50:150, 50:150] = 0.0
    grid.revision += 1
    assert frontier.unknown_pockets(grid, max_cells=400) == []


# ------------------------------------------------------------------ compatibility


def test_choose_goal_accepts_the_kwargs_vslam_passes():
    """`vslam/app.py` passes `snap_to_frontier_cell`, which used to be a TypeError."""
    grid = half_seen_room()
    target, path = frontier.choose_goal(
        grid, CostMap(grid, allow_unknown=True), (1.0, 1.0, 0.0),
        snap_to_frontier_cell=True, max_candidates=12,
    )
    assert target is not None and path is not None


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
