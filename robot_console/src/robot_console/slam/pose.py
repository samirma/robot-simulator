"""Where the robot is on the map, as opposed to where its wheels think it is.

Two frames, which is the whole idea. `/odom` is smooth and continuous but drifts; the map
frame is the one the grid is drawn in. `PoseTracker` holds the `map <- odom` correction:
every tick it propagates the latest odom reading through the current correction (cheap,
exact, no jumps), and on a keyframe it re-solves the correction by scan matching.

Keyframing is not an optimisation detail, it is what makes this run on the same thread as
`cv2.waitKey` and a 20 Hz `/cmd_vel`. Matching every scan would cost ~10 ms of every 50 ms
tick for no benefit: consecutive scans 5 cm apart carry almost no new constraint.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from robot_console.bridge import wrap_angle
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.matcher import LikelihoodField, MatchResult, field_for, match

# A keyframe is due after this much motion. Roughly one map cell of translation at the
# 5 cm resolution times three, which is where a new scan starts to say something new.
KEYFRAME_DISTANCE_M = 0.15
KEYFRAME_ANGLE_RAD = math.radians(10.0)
# ...but never more often than this, whatever the robot does.
MIN_KEYFRAME_INTERVAL_S = 0.2


def compose(correction: Sequence[float], odom: Sequence[float]) -> np.ndarray:
    """Apply the `map <- odom` transform to an odom-frame pose."""
    c, s = math.cos(float(correction[2])), math.sin(float(correction[2]))
    x = correction[0] + c * odom[0] - s * odom[1]
    y = correction[1] + s * odom[0] + c * odom[1]
    return np.array([x, y, wrap_angle(float(correction[2]) + float(odom[2]))])


def relative(target: Sequence[float], source: Sequence[float]) -> np.ndarray:
    """The transform T with `compose(T, source) == target`."""
    yaw = wrap_angle(float(target[2]) - float(source[2]))
    c, s = math.cos(yaw), math.sin(yaw)
    x = target[0] - (c * source[0] - s * source[1])
    y = target[1] - (s * source[0] + c * source[1])
    return np.array([x, y, yaw])


@dataclasses.dataclass
class TrackerStats:
    keyframes: int = 0
    matches: int = 0
    rejected: int = 0
    last_score: float = 0.0
    last_shift: float = 0.0
    last_ms: float = 0.0
    worst_ms: float = 0.0


class PoseTracker:
    """Fuses odometry with scan matching against a live map."""

    def __init__(
        self,
        *,
        keyframe_distance: float = KEYFRAME_DISTANCE_M,
        keyframe_angle: float = KEYFRAME_ANGLE_RAD,
        min_interval: float = MIN_KEYFRAME_INTERVAL_S,
        match_enabled: bool = True,
    ) -> None:
        self.correction = np.zeros(3)
        self.keyframe_distance = float(keyframe_distance)
        self.keyframe_angle = float(keyframe_angle)
        self.min_interval = float(min_interval)
        self.match_enabled = bool(match_enabled)
        self._odom: Optional[np.ndarray] = None
        self._last_key_odom: Optional[np.ndarray] = None
        self._last_key_at = 0.0
        self._field: Optional[LikelihoodField] = None
        self.stats = TrackerStats()

    # ------------------------------------------------------------------ odometry

    def update_odom(self, odom) -> np.ndarray:
        """Take a fresh `bridge.Odom` (or an (x, y, yaw) triple) and return the map pose."""
        if odom is None:
            return self.pose
        if hasattr(odom, "x"):
            self._odom = np.array([odom.x, odom.y, odom.yaw])
        else:
            self._odom = np.asarray(odom, dtype=np.float64).copy()
        if self._last_key_odom is None:
            self._last_key_odom = self._odom.copy()
        return self.pose

    @property
    def pose(self) -> np.ndarray:
        """Best estimate in the map frame."""
        if self._odom is None:
            return self.correction.copy()
        return compose(self.correction, self._odom)

    @property
    def has_odom(self) -> bool:
        return self._odom is not None

    def seed(self, pose: Sequence[float]) -> None:
        """Declare that the robot is at `pose` on the map right now.

        Used when a saved map is loaded: the odom frame's origin is wherever the robot's
        driver last booted, which has nothing to do with where the map's origin is.
        """
        odom = self._odom if self._odom is not None else np.zeros(3)
        self.correction = relative(np.asarray(pose, dtype=np.float64), odom)
        self._last_key_odom = odom.copy()

    # ------------------------------------------------------------------ keyframes

    def keyframe_due(self, now: Optional[float] = None) -> bool:
        if self._odom is None or not self.match_enabled:
            return False
        now = time.monotonic() if now is None else now
        if now - self._last_key_at < self.min_interval:
            return False
        if self._last_key_odom is None:
            return True
        delta = self._odom - self._last_key_odom
        return (
            math.hypot(delta[0], delta[1]) >= self.keyframe_distance
            or abs(wrap_angle(delta[2])) >= self.keyframe_angle
        )

    def refine(
        self,
        grid: OccupancyGrid,
        points: np.ndarray,
        *,
        now: Optional[float] = None,
    ) -> MatchResult:
        """Scan-match `points` (base frame) against `grid` and adopt the result if sane."""
        now = time.monotonic() if now is None else now
        started = time.monotonic()
        prior = self.pose
        self._field = field_for(grid, self._field)
        result = match(self._field, points, prior)

        self.stats.keyframes += 1
        self.stats.last_score = result.score
        if result.accepted:
            self.stats.matches += 1
            self.correction = relative(result.pose, self._odom if self._odom is not None else np.zeros(3))
            self.stats.last_shift = float(np.hypot(*(result.pose[:2] - prior[:2])))
        else:
            self.stats.rejected += 1
            self.stats.last_shift = 0.0

        self._last_key_at = now
        if self._odom is not None:
            self._last_key_odom = self._odom.copy()

        self.stats.last_ms = (time.monotonic() - started) * 1000.0
        self.stats.worst_ms = max(self.stats.worst_ms, self.stats.last_ms)
        return result

    def invalidate_field(self) -> None:
        """Force the likelihood field to be rebuilt, e.g. after loading a map."""
        self._field = None


def pose_delta(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """(distance, |angle|) between two poses."""
    return math.hypot(a[0] - b[0], a[1] - b[1]), abs(wrap_angle(float(a[2]) - float(b[2])))
