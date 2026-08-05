"""`sensor_msgs/LaserScan` -> points in the robot's base frame.

Pure, like `bridge.parse_odom`: everything here is testable from a dict literal, which
matters because a scan misread by one index or one mount offset produces a map that looks
right and is wrong.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Optional, Sequence

import numpy as np

# base_footprint -> laser_frame, from the static_transform_publisher in
# myagv_odometry/launch/myagv_active.launch on the myagv_ros_2023Pi branch:
#   args="0.065 0.0 0.08 3.14159265 0.0 0.0 /base_footprint /laser_frame"
# Only x matters in 2D. The roll of pi is the mounting flip and is already cancelled by
# the driver's `inverted: true` (see simulator/robots/README.md), so bearings need no
# further sign change -- the scan is counter-clockwise in the base frame either way.
LASER_OFFSET_X = 0.065
LASER_OFFSET_Y = 0.0

# ydlidar_ros_driver/launch/X2.launch, used only when a message omits them.
DEFAULT_RANGE_MIN = 0.1
DEFAULT_RANGE_MAX = 12.0


@dataclasses.dataclass(frozen=True)
class LaserScan:
    """The parts of `sensor_msgs/LaserScan` the mapper uses. Angles in radians."""

    ranges: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0, dtype=np.float32))
    angle_min: float = -math.pi
    angle_increment: float = 0.0
    range_min: float = DEFAULT_RANGE_MIN
    range_max: float = DEFAULT_RANGE_MAX
    stamp: float = 0.0
    frame_id: str = ""

    @property
    def count(self) -> int:
        return int(self.ranges.size)

    @property
    def angles(self) -> np.ndarray:
        """Bearing of every beam in the laser frame, valid or not."""
        return self.angle_min + np.arange(self.ranges.size, dtype=np.float64) * self.angle_increment

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of beams that actually returned something.

        One test covers every "no return" convention in play: the X2 driver runs with
        `invalid_range_is_inf: false` and reports **0.0**, a stock driver reports **inf**,
        and the simulator sends **range_max + 1** because JSON has no infinity. All three
        fall outside [range_min, range_max], and so does NaN.
        """
        with np.errstate(invalid="ignore"):
            return (
                np.isfinite(self.ranges)
                & (self.ranges >= self.range_min)
                & (self.ranges <= self.range_max)
            )


def _seq(value: object) -> np.ndarray:
    """Coerce a JSON array of numbers to float64, tolerating nulls and junk entries."""
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return np.empty(0, dtype=np.float64)
    out = np.empty(len(value), dtype=np.float64)
    for i, item in enumerate(value):
        try:
            out[i] = float(item)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out[i] = math.nan  # an unreadable beam is a missing beam, not a crash
    return out


def parse_scan(msg: Mapping) -> LaserScan:
    """Pure `sensor_msgs/LaserScan` -> `LaserScan`. Never raises.

    A corrupt scan degrades to an empty one for the same reason
    `camera.decode_compressed_image` returns None: the loop doing the parsing is also the
    loop keeping the robot's command stream alive.
    """
    from robot_console.bridge import _f

    if not isinstance(msg, Mapping):
        return LaserScan()
    header = msg.get("header")
    ranges = _seq(msg.get("ranges"))
    increment = _f(msg, "angle_increment")
    angle_min = _f(msg, "angle_min", default=-math.pi)
    if increment == 0.0 and ranges.size > 1:
        # Some drivers omit angle_increment; angle_max/angle_min still pin it down.
        angle_max = _f(msg, "angle_max", default=math.pi)
        increment = (angle_max - angle_min) / (ranges.size - 1)
    return LaserScan(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=increment,
        range_min=_f(msg, "range_min", default=DEFAULT_RANGE_MIN),
        range_max=_f(msg, "range_max", default=DEFAULT_RANGE_MAX),
        stamp=_f(header, "stamp", "secs") + _f(header, "stamp", "nsecs") * 1e-9,
        frame_id=str(header.get("frame_id", "")) if isinstance(header, Mapping) else "",
    )


def scan_points(
    scan: LaserScan,
    *,
    max_range: Optional[float] = None,
    offset: Sequence[float] = (LASER_OFFSET_X, LASER_OFFSET_Y),
) -> np.ndarray:
    """Valid returns as an (N, 2) array of XY in the **base** frame.

    The mount offset is applied here rather than left to the caller because forgetting it
    is silent: 65 mm is more than a whole cell at the 5 cm resolution the map uses, so
    every wall ends up smeared by a cell in whichever direction the robot was facing.

    `max_range` trims long returns further than the sensor's own limit. Far beams are the
    least accurate and the most expensive to ray-trace into the map, and a 12 m return in
    a house is usually a sightline down a corridor rather than a wall worth carving.
    """
    if scan.count == 0:
        return np.empty((0, 2), dtype=np.float64)
    mask = scan.valid
    if max_range is not None:
        mask = mask & (scan.ranges <= max_range)
    if not mask.any():
        return np.empty((0, 2), dtype=np.float64)
    r = scan.ranges[mask]
    a = scan.angles[mask]
    return np.column_stack((r * np.cos(a) + offset[0], r * np.sin(a) + offset[1]))


def transform_points(points: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    """Rotate and translate (N, 2) base-frame points into the map frame by (x, y, yaw)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return points.reshape(0, 2)
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T + np.asarray(pose[:2], dtype=np.float64)


def nearest_obstacle(
    scan: LaserScan, half_angle: float = math.pi / 4, bearing: float = 0.0
) -> float:
    """Closest valid return within `half_angle` of `bearing`, or `inf`.

    `bearing` matters because the base is holonomic. A differential-drive robot only ever
    moves along its own +x, so "what is in front" and "what is in the way" are the same
    question; a Mecanum base can strafe or reverse, and asking about straight ahead while
    driving sideways brakes for obstacles it is moving away from -- and wedges the robot
    permanently in any spot where something happens to sit in front of it.
    """
    if scan.count == 0:
        return math.inf
    delta = scan.angles - bearing
    mask = scan.valid & (np.abs(np.arctan2(np.sin(delta), np.cos(delta))) <= half_angle)
    if not mask.any():
        return math.inf
    return float(scan.ranges[mask].min())
