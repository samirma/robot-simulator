"""Turning a grid and a goal into a path the base can actually drive.

Two pieces: inflate obstacles by the robot's radius so the planner can treat the base as
a point, then A* over what is left. Inflation is `cv2.dilate` -- a Minkowski sum with a
disc is exactly a morphological dilation, and doing it in OpenCV rather than per-cell is
the difference between microseconds and seconds on a house-sized map.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from robot_console.slam.grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid

# The myAGV chassis is 0.311 x 0.230 m (simulator/robots/myagv/make_model.py), so the
# circumscribed radius is ~0.194 m. Rounded up, plus a little, because the base is
# holonomic and can be driven sideways through a gap it would not fit through diagonally.
ROBOT_RADIUS_M = 0.20
CLEARANCE_M = 0.05

# Cells nearer than this to an obstacle cost extra without being forbidden, which keeps
# paths off the walls when there is room and still lets them through a doorway when there
# is not. Pure hard-blocking produces wall-hugging paths that any pose error turns into a
# collision.
SOFT_MARGIN_M = 0.25
SOFT_PENALTY = 2.5

SQRT2 = math.sqrt(2.0)


class CostMap:
    """An inflated, planner-ready view of an `OccupancyGrid`."""

    def __init__(
        self,
        grid: OccupancyGrid,
        *,
        radius: float = ROBOT_RADIUS_M + CLEARANCE_M,
        soft_margin: float = SOFT_MARGIN_M,
        allow_unknown: bool = False,
    ) -> None:
        self.resolution = grid.resolution
        self.origin = grid.origin.copy()
        self.revision = grid.revision
        self.allow_unknown = bool(allow_unknown)

        classes = grid.classify()
        blocked = (classes == OCCUPIED).astype(np.uint8)
        if not allow_unknown:
            # Unexplored space is not free space. Exploration relaxes this deliberately --
            # its whole job is to drive into the unknown -- but navigating a saved map
            # must not route through a region no sensor has ever seen.
            blocked |= (classes == UNKNOWN).astype(np.uint8)

        cells = max(1, int(round(radius / grid.resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cells + 1, 2 * cells + 1))
        self.blocked = cv2.dilate(blocked, kernel).astype(bool)

        # Distance to the nearest blocked cell, for the soft cost.
        free_u8 = np.where(self.blocked, 0, 255).astype(np.uint8)
        if self.blocked.any():
            distance = cv2.distanceTransform(free_u8, cv2.DIST_L2, 5) * grid.resolution
        else:
            distance = np.full(self.blocked.shape, soft_margin, dtype=np.float32)
        self.extra = (
            SOFT_PENALTY * np.clip(1.0 - distance / max(soft_margin, 1e-6), 0.0, 1.0)
        ).astype(np.float32)
        self.free = (classes == FREE)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.blocked.shape

    def is_stale(self, grid: OccupancyGrid) -> bool:
        return grid.revision != self.revision or grid.data.shape != self.blocked.shape

    def world_to_cell(self, point: Sequence[float]) -> Tuple[int, int]:
        ix = int(math.floor((point[0] - self.origin[0]) / self.resolution))
        iy = int(math.floor((point[1] - self.origin[1]) / self.resolution))
        return ix, iy

    def cell_to_world(self, cell: Sequence[int]) -> np.ndarray:
        return self.origin + (np.asarray(cell, dtype=np.float64) + 0.5) * self.resolution

    def in_bounds(self, cell: Sequence[int]) -> bool:
        h, w = self.blocked.shape
        return 0 <= cell[0] < w and 0 <= cell[1] < h

    def is_blocked(self, cell: Sequence[int]) -> bool:
        if not self.in_bounds(cell):
            return True
        return bool(self.blocked[cell[1], cell[0]])

    def nearest_open(self, cell: Sequence[int], max_radius_cells: int = 12) -> Optional[Tuple[int, int]]:
        """The closest non-blocked cell to `cell`, or None.

        The robot's own footprint is inflated along with everything else, so a start pose
        one cell from a wall is "blocked" even though the robot is demonstrably standing
        there. Snapping out is what makes replanning from a tight spot possible at all.
        """
        cell = (int(cell[0]), int(cell[1]))
        if not self.is_blocked(cell):
            return cell
        for r in range(1, max_radius_cells + 1):
            best: Optional[Tuple[int, int]] = None
            best_d = math.inf
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if self.is_blocked(candidate):
                        continue
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best, best_d = candidate, d
            if best is not None:
                return best
        return None


def plan(
    cost: CostMap,
    start: Sequence[float],
    goal: Sequence[float],
    *,
    snap: bool = True,
) -> Optional[np.ndarray]:
    """A* from world `start` to world `goal`. Returns (N, 2) world waypoints, or None.

    None means "no path", and callers must treat it as a stop rather than as an empty
    path they can drive: an empty path and a missing one are the same shape and very
    different instructions.
    """
    start_cell = cost.world_to_cell(start)
    goal_cell = cost.world_to_cell(goal)
    if snap:
        snapped_start = cost.nearest_open(start_cell)
        snapped_goal = cost.nearest_open(goal_cell)
        if snapped_start is None or snapped_goal is None:
            return None
        start_cell, goal_cell = snapped_start, snapped_goal
    elif cost.is_blocked(start_cell) or cost.is_blocked(goal_cell):
        return None

    cells = _astar(cost, start_cell, goal_cell)
    if cells is None:
        return None
    world = np.array([cost.cell_to_world(c) for c in cells], dtype=np.float64)
    # Pin the true endpoints: A* works in cell centres, and a goal the user clicked
    # deserves to be driven to rather than to the middle of its cell.
    if world.shape[0] >= 1:
        world[-1] = np.asarray(goal, dtype=np.float64)[:2]
        world[0] = np.asarray(start, dtype=np.float64)[:2]
    return simplify(cost, world)


def _astar(cost: CostMap, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]
    h, w = cost.blocked.shape
    blocked = cost.blocked
    extra = cost.extra

    def heuristic(a: Tuple[int, int]) -> float:
        dx, dy = abs(a[0] - goal[0]), abs(a[1] - goal[1])
        # Octile distance: exact for 8-connected movement, so A* stays admissible and
        # expands far fewer cells than Euclidean would.
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

    open_heap: List[Tuple[float, Tuple[int, int]]] = [(heuristic(start), start)]
    came: dict = {}
    g = {start: 0.0}
    closed = set()

    neighbours = (
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
    )

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)
        cx, cy = current
        base = g[current]
        for dx, dy, step in neighbours:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or blocked[ny, nx]:
                continue
            if dx and dy:
                # No cutting corners diagonally between two blocked cells: the base is
                # 0.31 m long and would clip both.
                if blocked[cy, nx] or blocked[ny, cx]:
                    continue
            tentative = base + step * (1.0 + float(extra[ny, nx]))
            node = (nx, ny)
            if tentative < g.get(node, math.inf):
                g[node] = tentative
                came[node] = current
                heapq.heappush(open_heap, (tentative + heuristic(node), node))
    return None


def simplify(cost: CostMap, path: np.ndarray) -> np.ndarray:
    """Drop waypoints that a straight line already covers.

    A raw A* path is a staircase; following it literally makes the base weave. Greedy
    line-of-sight collapsing turns it into the handful of corners that actually matter.
    """
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] <= 2:
        return pts
    out = [pts[0]]
    anchor = 0
    for i in range(2, pts.shape[0]):
        if not line_of_sight(cost, pts[anchor], pts[i]):
            out.append(pts[i - 1])
            anchor = i - 1
    out.append(pts[-1])
    return np.array(out, dtype=np.float64)


def line_of_sight(cost: CostMap, a: Sequence[float], b: Sequence[float]) -> bool:
    """True if the straight segment a->b crosses no blocked cell."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    length = float(np.hypot(*(b - a)))
    if length <= 1e-9:
        return not cost.is_blocked(cost.world_to_cell(a))
    # Half-cell sampling: a full-cell step can jump the corner of a blocked cell.
    n = int(math.ceil(length / (cost.resolution * 0.5))) + 1
    ts = np.linspace(0.0, 1.0, n)
    for t in ts:
        if cost.is_blocked(cost.world_to_cell(a + t * (b - a))):
            return False
    return True


def path_length(path: Optional[np.ndarray]) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())


def costmap_for(
    grid: OccupancyGrid, cached: Optional[CostMap], *, allow_unknown: bool = False
) -> CostMap:
    """Reuse `cached` while the map is unchanged; dilation is not free."""
    if cached is not None and cached.allow_unknown == allow_unknown and not cached.is_stale(grid):
        return cached
    return CostMap(grid, allow_unknown=allow_unknown)
