"""A log-odds occupancy grid that grows to fit wherever the robot goes.

Log-odds rather than probabilities because the update is then an addition, which is what
makes integrating a 360-beam scan at 10 Hz a couple of vectorised numpy calls instead of
a loop. Clamping keeps a cell from becoming so certain that no amount of contrary
evidence can move it -- a door that opens must be able to become free again.

Cell convention throughout: `cells` are integer (col, row) == (ix, iy), and `data[iy, ix]`
is the log-odds. Row 0 is the *lowest* y in world coordinates, matching the ROS map_server
convention that a map image's bottom-left pixel sits at `origin`.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

# 5 cm is what myagv_navigation's gmapping uses, so a map built here is directly
# comparable with one built on the robot.
DEFAULT_RESOLUTION = 0.05

# Per-observation evidence, in log-odds. Hits weigh more than misses: a wall seen once is
# worth more than the many rays that pass near it, and the asymmetry is what keeps thin
# obstacles from being erased by traffic in front of them.
L_OCCUPIED = 0.85
L_FREE = -0.4
L_CLAMP = 12.0

# Decision thresholds, matching map_server's yaml defaults (0.65 / 0.196) once converted.
OCCUPIED_LOGODDS = math.log(0.65 / 0.35)
FREE_LOGODDS = math.log(0.196 / 0.804)

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

# How much slack to allocate when the map has to grow, in cells. Growing costs a full
# copy, so it is worth over-allocating to make it rare.
GROW_MARGIN = 64


class OccupancyGrid:
    """A resizable 2D log-odds map in metres.

    `origin` is the world coordinate of the *lower-left corner* of cell (0, 0).
    """

    def __init__(
        self,
        resolution: float = DEFAULT_RESOLUTION,
        *,
        width: int = 256,
        height: int = 256,
        origin: Sequence[float] = (0.0, 0.0),
        data: Optional[np.ndarray] = None,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.resolution = float(resolution)
        if data is not None:
            self.data = np.ascontiguousarray(data, dtype=np.float32)
        else:
            self.data = np.zeros((int(height), int(width)), dtype=np.float32)
        self.origin = np.asarray(origin, dtype=np.float64).copy()
        # Bumped on every change so consumers (the likelihood field, the renderer) know
        # when a cached derivative is stale without diffing the array.
        self.revision = 0

    # ------------------------------------------------------------------ geometry

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def world_to_cell(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) world XY -> (N, 2) integer (ix, iy). Floor, so -0.01 m lands in cell -1."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return np.floor((pts - self.origin) / self.resolution).astype(np.int64)

    def cell_to_world(self, cells: np.ndarray) -> np.ndarray:
        """(N, 2) integer cells -> (N, 2) world XY at each cell's **centre**."""
        idx = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
        return self.origin + (idx + 0.5) * self.resolution

    def contains(self, cells: np.ndarray) -> np.ndarray:
        idx = np.asarray(cells).reshape(-1, 2)
        return (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.width)
            & (idx[:, 1] >= 0) & (idx[:, 1] < self.height)
        )

    # ------------------------------------------------------------------ growth

    def grow_to_include(self, points: np.ndarray) -> bool:
        """Enlarge so every world point in `points` is inside. True if anything changed.

        Growth only ever adds space on the outside, so existing cells keep their world
        coordinates -- the map does not shift under a pose already computed against it.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return False
        lo = self.world_to_cell(pts).min(axis=0)
        hi = self.world_to_cell(pts).max(axis=0)
        pad_lo = np.maximum(0, -np.minimum(lo, 0))
        pad_hi = np.maximum(0, hi + 1 - np.array([self.width, self.height]))
        if not (pad_lo.any() or pad_hi.any()):
            return False
        pad_lo = np.where(pad_lo > 0, pad_lo + GROW_MARGIN, 0)
        pad_hi = np.where(pad_hi > 0, pad_hi + GROW_MARGIN, 0)
        self.data = np.pad(
            self.data,
            ((int(pad_lo[1]), int(pad_hi[1])), (int(pad_lo[0]), int(pad_hi[0]))),
            mode="constant",
            constant_values=0.0,
        )
        # Padding below/left moves cell (0, 0), so the origin moves with it.
        self.origin = self.origin - pad_lo * self.resolution
        self.revision += 1
        return True

    # ------------------------------------------------------------------ update

    def integrate(
        self,
        pose: Sequence[float],
        points: np.ndarray,
        *,
        max_range: Optional[float] = None,
    ) -> None:
        """Fold one scan in: free along each beam, occupied at each endpoint.

        `points` are already in the **map** frame (see `scan.transform_points`); `pose` is
        (x, y, yaw) and only its position is used, as the ray origin.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return
        origin_xy = np.asarray(pose[:2], dtype=np.float64)
        if max_range is not None:
            keep = np.linalg.norm(pts - origin_xy, axis=1) <= max_range
            pts = pts[keep]
            if pts.size == 0:
                return

        self.grow_to_include(np.vstack((pts, origin_xy[None, :])))

        start = self.world_to_cell(origin_xy[None, :])[0]
        ends = self.world_to_cell(pts)

        free_cells = _bresenham_rays(start, ends)
        if free_cells.size:
            inside = self.contains(free_cells)
            fc = free_cells[inside]
            # `np.add.at` rather than fancy-index `+=`: a cell crossed by several beams
            # must accumulate every one of them, and plain indexing keeps only the last.
            np.add.at(self.data, (fc[:, 1], fc[:, 0]), L_FREE)

        inside = self.contains(ends)
        ec = ends[inside]
        if ec.size:
            # Plain L_OCCUPIED, not a correction: `_bresenham_rays` already excluded each
            # ray's own endpoint from the free pass.
            np.add.at(self.data, (ec[:, 1], ec[:, 0]), L_OCCUPIED)

        np.clip(self.data, -L_CLAMP, L_CLAMP, out=self.data)
        self.revision += 1

    # ------------------------------------------------------------------ readout

    def probability(self) -> np.ndarray:
        """Occupancy probability per cell, in [0, 1]."""
        return 1.0 / (1.0 + np.exp(-self.data))

    def classify(self) -> np.ndarray:
        """uint8 array of UNKNOWN / FREE / OCCUPIED."""
        out = np.full(self.data.shape, UNKNOWN, dtype=np.uint8)
        out[self.data <= FREE_LOGODDS] = FREE
        out[self.data >= OCCUPIED_LOGODDS] = OCCUPIED
        return out

    def occupied_mask(self) -> np.ndarray:
        return self.data >= OCCUPIED_LOGODDS

    def is_free(self, point: Sequence[float]) -> bool:
        cell = self.world_to_cell(np.asarray(point, dtype=np.float64)[None, :])[0]
        if not self.contains(cell[None, :])[0]:
            return False
        return bool(self.data[cell[1], cell[0]] <= FREE_LOGODDS)

    def copy(self) -> "OccupancyGrid":
        clone = OccupancyGrid(self.resolution, origin=self.origin, data=self.data.copy())
        clone.revision = self.revision
        return clone


def _bresenham_rays(start: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Cells along the line from `start` to each of `ends`, endpoints **excluded**.

    Vectorised across beams rather than looped: 360 separate Bresenham walks in Python
    would not fit in the loop's time budget. Each ray is sampled at `steps` evenly spaced
    parameters where `steps` is its Chebyshev length, which visits exactly the same cells
    a Bresenham walk would for a line on a grid, and duplicates are harmless -- a cell
    counted twice on one beam is not meaningfully different from two beams crossing it.

    The endpoint is dropped because `integrate` marks it occupied; carving it free first
    and then re-marking it would work, but only by cancellation.
    """
    ends = np.asarray(ends, dtype=np.int64).reshape(-1, 2)
    if ends.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    delta = ends - np.asarray(start, dtype=np.int64)
    steps = np.abs(delta).max(axis=1)
    n = int(steps.max())
    if n <= 0:
        return np.empty((0, 2), dtype=np.int64)

    # t[i, k] = k / steps[i], masked past each ray's own length so short rays do not
    # keep emitting their endpoint.
    k = np.arange(n, dtype=np.float64)[None, :]
    live = k < steps[:, None]
    # A zero-length ray (endpoint in the robot's own cell) would divide by zero and then
    # multiply inf by zero below. It contributes no free cells, so clamp the denominator
    # and let `live` drop it.
    t = k / np.maximum(steps, 1)[:, None]
    cells = np.rint(np.asarray(start, dtype=np.float64) + t[..., None] * delta[:, None, :])
    return cells[live].astype(np.int64)
