"""Correlative scan matching: correct the odometry prior against the map so far.

The simulator's `/odom` is ground truth, so this looks like ceremony there. It is not:
the real myAGV dead-reckons from wheel encoders on a Mecanum base, which is the worst
case for slip -- rollers scrub sideways on every strafe -- and `myagv_active.launch` runs
`robot_pose_ekf` precisely because the raw estimate walks away. Without this, a map built
on hardware shears as soon as the robot turns.

The method is Olson's correlative matcher, reduced to what fits the loop's time budget:
score candidate poses against a likelihood field, coarse pass then fine pass. No pose
graph, no loop closure -- those need an optimizer, and a house-sized map that is locally
consistent is worth more here than a globally optimal one that arrives too late.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence

import cv2
import numpy as np

from robot_console.slam.grid import OccupancyGrid

# Beyond this distance from an obstacle a point tells us nothing useful, and letting the
# field keep falling would make a badly wrong pose look merely mediocre instead of
# obviously bad.
MAX_DISTANCE_M = 0.5

# Search half-widths around the odom prior. Sized for one keyframe of drift at walking
# pace, not for kidnapping: 0.2 m and 10 deg is generous for 0.15 m of travel.
SEARCH_XY_M = 0.20
SEARCH_YAW_RAD = math.radians(10.0)
COARSE_STEPS = 5      # per axis, per pass
FINE_STEPS = 5

# Below this mean score the match is rejected and odom is kept. A score is the mean of
# exp(-d/sigma) over beams, so 0.35 means the typical beam is landing ~10 cm from anything
# in the map -- consistent with a room the robot has not seen before, where trusting the
# match would teleport it.
MIN_SCORE = 0.35
SIGMA_M = 0.15


@dataclasses.dataclass(frozen=True)
class MatchResult:
    pose: np.ndarray
    score: float
    accepted: bool
    prior_score: float = 0.0

    @property
    def correction(self) -> np.ndarray:
        return self.pose


class LikelihoodField:
    """Distance-to-nearest-obstacle over the map, cached against `grid.revision`.

    `cv2.distanceTransform` does the whole field in one pass, which is why this is
    affordable at all: the alternative, a nearest-neighbour query per beam per candidate
    pose, is several orders of magnitude more work.
    """

    def __init__(self, grid: OccupancyGrid) -> None:
        self.resolution = grid.resolution
        self.origin = grid.origin.copy()
        self.revision = grid.revision
        occupied = grid.occupied_mask()
        if not occupied.any():
            # Nothing to match against yet. A uniform "far" field scores every candidate
            # identically, so the matcher rejects and the caller keeps odom -- which is
            # the right answer for the first scan.
            self.distance = np.full(grid.data.shape, MAX_DISTANCE_M, dtype=np.float32)
        else:
            free = np.where(occupied, 0, 255).astype(np.uint8)
            dist_cells = cv2.distanceTransform(free, cv2.DIST_L2, 5)
            self.distance = np.minimum(dist_cells * grid.resolution, MAX_DISTANCE_M).astype(
                np.float32
            )
        self.height, self.width = self.distance.shape

    def is_stale(self, grid: OccupancyGrid) -> bool:
        return (
            grid.revision != self.revision
            or grid.data.shape != self.distance.shape
            or not np.array_equal(grid.origin, self.origin)
        )

    def score(self, points: np.ndarray) -> float:
        """Mean `exp(-d/sigma)` over map-frame points. 1.0 is a perfect fit, 0 is hopeless."""
        if points.size == 0:
            return 0.0
        return float(self._scores(points).mean())

    def _scores(self, points: np.ndarray) -> np.ndarray:
        cells = np.floor((points.reshape(-1, 2) - self.origin) / self.resolution).astype(np.int64)
        ix = np.clip(cells[:, 0], 0, self.width - 1)
        iy = np.clip(cells[:, 1], 0, self.height - 1)
        d = self.distance[iy, ix]
        # A beam that falls outside the map has no evidence either way; scoring it as a
        # miss would penalise exactly the poses that push into unexplored space, which is
        # where the robot spends most of an exploration run.
        outside = (cells[:, 0] < 0) | (cells[:, 0] >= self.width) | (cells[:, 1] < 0) | (
            cells[:, 1] >= self.height
        )
        s = np.exp(-d / SIGMA_M)
        s[outside] = 0.0
        return s


def match(
    field: LikelihoodField,
    points: np.ndarray,
    prior: Sequence[float],
    *,
    search_xy: float = SEARCH_XY_M,
    search_yaw: float = SEARCH_YAW_RAD,
    min_score: float = MIN_SCORE,
) -> MatchResult:
    """Find the pose near `prior` that best explains base-frame `points`.

    `points` are (N, 2) in the robot's own frame; `prior` is the odometry-propagated
    (x, y, yaw) guess in the map frame.
    """
    prior = np.asarray(prior, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 8:
        # Too few returns to constrain three degrees of freedom; anything this finds is
        # noise fitting noise.
        return MatchResult(prior.copy(), 0.0, False)

    prior_score = field.score(_transform(pts, prior))

    best_pose, best_score = prior.copy(), prior_score
    for span_xy, span_yaw, steps in (
        (search_xy, search_yaw, COARSE_STEPS),
        (search_xy / COARSE_STEPS, search_yaw / COARSE_STEPS, FINE_STEPS),
    ):
        best_pose, best_score = _search(field, pts, best_pose, span_xy, span_yaw, steps)

    accepted = best_score >= min_score and best_score >= prior_score
    return MatchResult(
        best_pose if accepted else prior.copy(), best_score, accepted, prior_score
    )


def _search(
    field: LikelihoodField,
    pts: np.ndarray,
    centre: np.ndarray,
    span_xy: float,
    span_yaw: float,
    steps: int,
):
    offsets = np.linspace(-span_xy, span_xy, steps)
    yaws = np.linspace(-span_yaw, span_yaw, steps) + centre[2]
    # dx/dy are evaluated as one vectorised block per yaw: rotating the cloud is the
    # expensive part and depends only on yaw, so it happens `steps` times rather than
    # `steps**3` times.
    dx, dy = np.meshgrid(offsets + centre[0], offsets + centre[1], indexing="ij")
    translations = np.column_stack((dx.ravel(), dy.ravel()))

    best_score = -1.0
    best_pose = centre.copy()
    for yaw in yaws:
        c, s = math.cos(yaw), math.sin(yaw)
        rotated = pts @ np.array([[c, s], [-s, c]])
        # (T, N, 2): every translation applied to the rotated cloud at once.
        candidates = rotated[None, :, :] + translations[:, None, :]
        scores = field._scores(candidates.reshape(-1, 2)).reshape(len(translations), -1).mean(axis=1)
        k = int(np.argmax(scores))
        if scores[k] > best_score:
            best_score = float(scores[k])
            best_pose = np.array([translations[k, 0], translations[k, 1], yaw])
    return best_pose, best_score


def _transform(pts: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    return pts @ np.array([[c, s], [-s, c]]) + np.asarray(pose[:2], dtype=np.float64)


def field_for(grid: OccupancyGrid, cached: Optional[LikelihoodField]) -> LikelihoodField:
    """Reuse `cached` unless the map moved on. Rebuilding is ~5 ms; doing it per tick is not."""
    if cached is not None and not cached.is_stale(grid):
        return cached
    return LikelihoodField(grid)
