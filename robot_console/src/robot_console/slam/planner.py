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

from robot_console.slam.grid import OCCUPIED, UNKNOWN, OccupancyGrid

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

# The wavefront runs on a decimated grid. Relaxation costs roughly `passes x area`, and a
# stride shrinks both, so stride 2 is about six times cheaper than full resolution --
# measured on a 326x360 house map, 124 ms undecimated against 20 ms at stride 2, against
# 19 ms for the euclidean-and-A* ranking it replaces. Same budget, true geodesic distance
# to every cell instead of twelve straight-line guesses.
#
# The stride is chosen per call from this cell budget rather than fixed, because the map
# grows during a run: a fixed stride that suits a room overruns on a house.
WAVEFRONT_CELL_BUDGET = 20_000

# Fallback stride when a caller does not want the budget consulted.
WAVEFRONT_DECIMATE = 2

# Each relaxation pass moves information one cell, so a converged field needs about as many
# passes as the longest route is long. That O(diameter x area) shape is why this is worth
# measuring rather than assuming: relaxation beats a heap when the region is squat and
# loses when it is a long corridor. Decimation shrinks the diameter too, which is what
# keeps it comfortably ahead here. The cap is a backstop against a pathological map, not a
# tuning knob; `DistanceField.converged` reports whether it was hit.
WAVEFRONT_MAX_ITER = 4000

INF = np.float32(np.inf)


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
        self.radius = float(radius)

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


class DistanceField:
    """Geodesic cost-to-come from one start pose to everywhere reachable.

    Straight-line distance is the wrong way to rank exploration goals: a frontier one
    metre away through a wall is twenty metres of corridor, and only a route knows that.
    A* answers the question for *one* goal, so ranking N frontiers that way costs N
    searches and tempts you to cap N -- which is how a run declares itself finished while
    a reachable frontier sits behind candidate thirteen.

    One wavefront answers it for every cell at once, and "unreachable" stops being a
    guess: a cell the wave never arrived at has no route, full stop.

    Values are in the same units as `_astar`'s `g` -- cells scaled by the soft cost -- so
    ranking here and planning there agree about what a metre costs. They are *not* metres.
    """

    def __init__(
        self,
        dist: np.ndarray,
        *,
        origin: np.ndarray,
        resolution: float,
        decimate: int,
        converged: bool,
        start_cell: Tuple[int, int] = (0, 0),
    ) -> None:
        self.dist = dist
        # Kept because callers need the true geometric distance from the robot, and the
        # field cannot give it: values here are scaled by the soft wall penalty, so a cell
        # 0.2 m away beside a wall scores like one 0.7 m away in open space.
        self.start_cell = (int(start_cell[0]), int(start_cell[1]))
        self.origin = np.asarray(origin, dtype=np.float64).copy()
        # The *coarse* cell size, so world_to_cell needs no further scaling.
        self.resolution = float(resolution)
        self.decimate = int(decimate)
        self.converged = bool(converged)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.dist.shape

    @property
    def source(self) -> np.ndarray:
        """World XY the field was flooded from."""
        return self.origin + (np.asarray(self.start_cell, dtype=np.float64) + 0.5) * self.resolution

    def cell_of(self, point: Sequence[float]) -> Tuple[int, int]:
        ix = int(math.floor((point[0] - self.origin[0]) / self.resolution))
        iy = int(math.floor((point[1] - self.origin[1]) / self.resolution))
        return ix, iy

    def at(self, point: Sequence[float]) -> float:
        """Cost to reach `point`, or `inf` if the wave never got there."""
        ix, iy = self.cell_of(point)
        h, w = self.dist.shape
        if not (0 <= ix < w and 0 <= iy < h):
            return math.inf
        return float(self.dist[iy, ix])

    def reachable(self, point: Sequence[float]) -> bool:
        return math.isfinite(self.at(point))

    def at_cells(self, cells: np.ndarray) -> np.ndarray:
        """Vectorised `at` for (N, 2) *fine-resolution* (ix, iy) cells.

        Frontier clusters arrive as fine cells from `connectedComponents`; converting the
        whole cluster at once is what keeps scoring off the Python interpreter.
        """
        idx = np.asarray(cells, dtype=np.int64).reshape(-1, 2) // self.decimate
        h, w = self.dist.shape
        out = np.full(idx.shape[0], np.inf, dtype=np.float64)
        inside = (
            (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
        )
        if inside.any():
            good = idx[inside]
            out[inside] = self.dist[good[:, 1], good[:, 0]]
        return out


def distance_field(
    cost: CostMap,
    start: Sequence[float],
    *,
    decimate: int = 0,
    max_iter: int = WAVEFRONT_MAX_ITER,
) -> DistanceField:
    """Flood the reachable free space with cost-to-come from `start`.

    `decimate=0` sizes the stride to `WAVEFRONT_CELL_BUDGET`; pass a positive stride to
    pin it.

    Jacobi relaxation rather than a heap: the update is eight shifted `np.minimum` passes,
    which is the same trade the rest of this package makes -- `cv2.dilate` for inflation,
    vectorised Bresenham for ray casting -- for the same reason. A pure-Python heap over
    tens of thousands of cells is an interpreter loop, and this is not.
    """
    decimate = choose_decimation(cost) if decimate <= 0 else max(1, int(decimate))
    blocked, step_cost = _coarsen(cost, decimate)
    h, w = blocked.shape

    sx = int(math.floor((start[0] - cost.origin[0]) / cost.resolution)) // decimate
    sy = int(math.floor((start[1] - cost.origin[1]) / cost.resolution)) // decimate
    field = np.full((h, w), INF, dtype=np.float32)
    coarse_res = cost.resolution * decimate
    if not (0 <= sx < w and 0 <= sy < h):
        return DistanceField(
            field, origin=cost.origin, resolution=coarse_res,
            decimate=decimate, converged=True, start_cell=(sx, sy),
        )
    # The robot's own footprint is inflated along with everything else, so the cell it is
    # demonstrably standing in is routinely "blocked" -- and coarsening, which blocks a
    # coarse cell if any fine cell in it is, makes that more likely rather than less.
    # Seeding anyway is the same concession `nearest_open` makes for A*.
    blocked[sy, sx] = False

    field[sy, sx] = 0.0
    converged = False
    for _ in range(max_iter):
        previous = field
        field = _relax(field, step_cost)
        field[blocked] = INF
        field[sy, sx] = 0.0
        if np.array_equal(field, previous):
            converged = True
            break

    return DistanceField(
        field, origin=cost.origin, resolution=coarse_res,
        decimate=decimate, converged=converged, start_cell=(sx, sy),
    )


# A coarse cell is blocked when more than this fraction of its fine cells are. Majority
# rule, and the two obvious alternatives were both measured worse: taking a coarse cell as
# blocked if *any* fine cell is seals a 0.8 m doorway at stride 3, and taking it as open if
# any fine cell is lets the wave leak straight through a one-cell wall at stride 4.
COARSE_BLOCKED_FRACTION = 0.5


def _coarsen(cost: CostMap, decimate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Down-sample the costmap by majority vote over each block of fine cells.

    The two errors are not symmetric, which is what rules out the pessimistic version:
    calling a route closed abandons every frontier behind it -- a whole room, the failure
    this field exists to fix -- while calling one open costs a single A* that returns None
    before the next candidate is tried. Full-resolution `plan` stays the authority on
    whether the base actually fits.

    Majority rather than outright optimism because a wall must survive too. `INTER_AREA`
    on the boolean mask is exactly the blocked fraction per block, so the test is one
    resize and a compare.
    """
    if decimate == 1:
        return cost.blocked.copy(), (1.0 + cost.extra).astype(np.float32)
    shape = (cost.blocked.shape[1] // decimate, cost.blocked.shape[0] // decimate)
    fraction = cv2.resize(
        cost.blocked.astype(np.float32), shape, interpolation=cv2.INTER_AREA
    )
    coarse = fraction > COARSE_BLOCKED_FRACTION
    step_cost = cv2.resize(
        (1.0 + cost.extra).astype(np.float32),
        (coarse.shape[1], coarse.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    return coarse, step_cost


def choose_decimation(cost: CostMap, *, budget: int = WAVEFRONT_CELL_BUDGET) -> int:
    """The coarsest stride that keeps the wavefront inside its tick budget.

    Cost is roughly `passes x area`, and both shrink with the stride, so a map four times
    the area is eight times the work at the same stride. A fixed stride therefore either
    wastes time on a small map or overruns on a big one; sizing to a cell budget is what
    keeps a house and a single room both inside the publish period.
    """
    open_cells = int((~cost.blocked).sum())
    for stride in (1, 2, 3, 4):
        if open_cells / float(stride * stride) <= budget:
            return stride
    return 4


# (dx, dy, step) for the eight neighbours, matching `_astar`'s move set and costs.
_NEIGHBOURS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
)


def _relax(field: np.ndarray, step_cost: np.ndarray) -> np.ndarray:
    """One Jacobi pass: every cell takes the cheapest of its neighbours' cost plus entry."""
    h, w = field.shape
    out = field.copy()
    for dx, dy, step in _NEIGHBOURS:
        shifted = np.full_like(field, INF)
        y0, y1 = max(0, dy), h + min(0, dy)
        x0, x1 = max(0, dx), w + min(0, dx)
        shifted[y0:y1, x0:x1] = field[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        np.minimum(out, shifted + np.float32(step) * step_cost, out=out)
    return out


def descend(
    field: DistanceField, start: Sequence[float], *, max_steps: int = 10000
) -> Optional[np.ndarray]:
    """Walk downhill from `start` to the field's source, returning world waypoints.

    A converged cost-to-come field has no local minimum but the source, so steepest
    descent always arrives. Handy when the wavefront has already been paid for and an
    approximate route is enough; `plan` still owns the full-resolution answer.
    """
    ix, iy = field.cell_of(start)
    h, w = field.dist.shape
    if not (0 <= ix < w and 0 <= iy < h) or not math.isfinite(field.dist[iy, ix]):
        return None
    cells = [(ix, iy)]
    for _ in range(max_steps):
        cx, cy = cells[-1]
        here = field.dist[cy, cx]
        if here <= 0.0:
            break
        best, best_value = None, here
        for dx, dy, _step in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            value = field.dist[ny, nx]
            if value < best_value:
                best, best_value = (nx, ny), value
        if best is None:
            break
        cells.append(best)
    world = field.origin + (np.array(cells, dtype=np.float64) + 0.5) * field.resolution
    return world


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
    # Sampled in one shot rather than a Python loop: `simplify` calls this greedily, so it
    # runs O(waypoints^2) times per plan, and on a long house-crossing path that loop was
    # the single largest contributor to a replan tick overrunning the publish period.
    ts = np.linspace(0.0, 1.0, n)[:, None]
    points = a + ts * (b - a)
    cells = np.floor((points - cost.origin) / cost.resolution).astype(np.int64)
    h, w = cost.blocked.shape
    inside = (
        (cells[:, 0] >= 0) & (cells[:, 0] < w)
        & (cells[:, 1] >= 0) & (cells[:, 1] < h)
    )
    if not inside.all():
        return False  # off the map counts as blocked, as `is_blocked` has it
    return not cost.blocked[cells[:, 1], cells[:, 0]].any()


def path_length(path: Optional[np.ndarray]) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())


def costmap_for(
    grid: OccupancyGrid,
    cached: Optional[CostMap],
    *,
    allow_unknown: bool = False,
    radius: float = ROBOT_RADIUS_M + CLEARANCE_M,
) -> CostMap:
    """Reuse `cached` while the map is unchanged; dilation is not free."""
    if (
        cached is not None
        and cached.allow_unknown == allow_unknown
        and abs(cached.radius - radius) < 1e-9
        and not cached.is_stale(grid)
    ):
        return cached
    return CostMap(grid, allow_unknown=allow_unknown, radius=radius)
