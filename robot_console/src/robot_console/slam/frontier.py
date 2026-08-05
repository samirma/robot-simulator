"""Where to go next: the boundary between what has been mapped and what has not.

A frontier is a free cell adjacent to an unknown one. Drive to frontiers until there are
none and the map is complete by construction -- which is why exploration is posed this
way rather than as a coverage or lawnmower pattern that has no idea when it is finished.

Everything is morphology on the classified grid, so a house-sized map costs well under a
millisecond and this can run on every replan.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence

import cv2
import numpy as np

from robot_console.slam.grid import FREE, UNKNOWN, OccupancyGrid
from robot_console.slam.planner import CostMap, path_length, plan

# A frontier smaller than this is noise at the edge of the sensor's reach, or a gap
# behind a chair leg the robot cannot fit through anyway.
MIN_FRONTIER_CELLS = 6

# Utility is size / (distance + BIAS). BIAS in metres sets how much a big far frontier is
# worth against a small near one; at 2 m the robot finishes the room it is in before
# crossing the house, which produces far less backtracking.
DISTANCE_BIAS_M = 2.0

# A goal that could not be reached is not retried within this radius. Without it the
# robot picks the same unreachable frontier forever.
BLACKLIST_RADIUS_M = 0.5


@dataclasses.dataclass(frozen=True)
class Frontier:
    centroid: np.ndarray
    size: int
    distance: float = 0.0
    utility: float = 0.0

    def with_scores(self, distance: float, utility: float) -> "Frontier":
        return Frontier(self.centroid, self.size, distance, utility)


def frontier_mask(grid: OccupancyGrid) -> np.ndarray:
    """Boolean mask of free cells that touch unknown space."""
    classes = grid.classify()
    free = (classes == FREE).astype(np.uint8)
    unknown = (classes == UNKNOWN).astype(np.uint8)
    # A cell is a frontier if it is free and any 4-neighbour is unknown. Dilating the
    # unknown region and intersecting is the same test, done in one pass.
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    near_unknown = cv2.dilate(unknown, kernel)
    return (free & near_unknown).astype(bool)


def find_frontiers(
    grid: OccupancyGrid,
    *,
    min_cells: int = MIN_FRONTIER_CELLS,
) -> List[Frontier]:
    """Cluster the frontier cells and return one candidate per cluster, largest first."""
    mask = frontier_mask(grid).astype(np.uint8)
    if not mask.any():
        return []
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[Frontier] = []
    for i in range(1, count):  # label 0 is the background
        size = int(stats[i, cv2.CC_STAT_AREA])
        if size < min_cells:
            continue
        cx, cy = centroids[i]
        world = grid.origin + (np.array([cx, cy]) + 0.5) * grid.resolution
        out.append(Frontier(world, size))
    out.sort(key=lambda f: f.size, reverse=True)
    return out


def choose_goal(
    grid: OccupancyGrid,
    cost: CostMap,
    pose: Sequence[float],
    *,
    blacklist: Optional[Sequence[np.ndarray]] = None,
    min_cells: int = MIN_FRONTIER_CELLS,
    max_candidates: int = 12,
    distance_bias: float = DISTANCE_BIAS_M,
):
    """Pick the best reachable frontier and the path to it.

    Returns `(frontier, path)`, or `(None, None)` when the map is finished or everything
    left is unreachable. Candidates are scored by straight-line distance first and only
    the promising ones are planned to: A* is the expensive part, and a frontier across the
    house is rarely the answer.
    """
    frontiers = find_frontiers(grid, min_cells=min_cells)
    if not frontiers:
        return None, None

    origin = np.asarray(pose[:2], dtype=np.float64)
    blocked_goals = [np.asarray(b, dtype=np.float64)[:2] for b in (blacklist or [])]

    scored: List[Frontier] = []
    for f in frontiers:
        if any(np.hypot(*(f.centroid - b)) < BLACKLIST_RADIUS_M for b in blocked_goals):
            continue
        d = float(np.hypot(*(f.centroid - origin)))
        scored.append(f.with_scores(d, f.size / (d + distance_bias)))
    if not scored:
        return None, None
    scored.sort(key=lambda f: f.utility, reverse=True)

    for candidate in scored[:max_candidates]:
        path = plan(cost, origin, candidate.centroid)
        if path is None:
            continue
        # Re-score on the true path length: a frontier through a wall is metres away and
        # a corridor away, and only planning tells the two apart.
        travelled = path_length(path)
        return candidate.with_scores(travelled, candidate.size / (travelled + distance_bias)), path
    return None, None


def explored_area(grid: OccupancyGrid) -> float:
    """Square metres of the map that are no longer unknown."""
    return float((grid.classify() != UNKNOWN).sum()) * grid.resolution ** 2


def is_complete(grid: OccupancyGrid, *, min_cells: int = MIN_FRONTIER_CELLS) -> bool:
    return not find_frontiers(grid, min_cells=min_cells)


def approach_point(
    frontier: Sequence[float], pose: Sequence[float], standoff: float = 0.0
) -> np.ndarray:
    """A point `standoff` metres short of the frontier, along the line from `pose`.

    Driving onto a frontier cell means driving into the boundary of what is known; a
    little standoff lets the lidar see past it without the base having to get there.
    """
    target = np.asarray(frontier, dtype=np.float64)[:2]
    origin = np.asarray(pose, dtype=np.float64)[:2]
    delta = target - origin
    d = float(np.hypot(*delta))
    if d <= standoff or d < 1e-6:
        return target
    return target - delta / d * standoff


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
