"""A synthetic room and lidar, so map tests assert against known geometry.

Shared by the SLAM tests. Every expected value here is arithmetic off `WALLS`, not a
recorded fixture: a fixture would happily keep passing after the convention it encodes
was broken.
"""

from __future__ import annotations

import math

import numpy as np

from robot_console.slam.scan import LaserScan

BEAMS = 360
RANGE_MAX = 12.0

# A 6 x 4 m room with a stub of interior wall, as (x0, y0, x1, y1) segments.
WALLS = np.array(
    [
        (0.0, 0.0, 6.0, 0.0),
        (0.0, 4.0, 6.0, 4.0),
        (0.0, 0.0, 0.0, 4.0),
        (6.0, 0.0, 6.0, 4.0),
        (3.0, 0.0, 3.0, 1.5),
    ],
    dtype=np.float64,
)

_ANGLES = -math.pi + np.arange(BEAMS) * (2 * math.pi / BEAMS)


def raycast(x: float, y: float, yaw: float = 0.0, *, walls: np.ndarray = WALLS) -> LaserScan:
    """A LaserScan from (x, y, yaw), with bearings in the robot's own frame."""
    a = _ANGLES + yaw
    dx, dy = np.cos(a), np.sin(a)
    best = np.full(BEAMS, RANGE_MAX + 1.0)
    for x0, y0, x1, y1 in walls:
        ex, ey = x1 - x0, y1 - y0
        den = dx * ey - dy * ex
        ok = np.abs(den) > 1e-12
        t = np.full(BEAMS, np.inf)
        u = np.full(BEAMS, np.inf)
        t[ok] = ((x0 - x) * ey - (y0 - y) * ex) / den[ok]
        u[ok] = ((x0 - x) * dy[ok] - (y0 - y) * dx[ok]) / den[ok]
        hit = ok & (t > 0.05) & (t < best) & (u >= 0.0) & (u <= 1.0)
        best[hit] = t[hit]
    # Misses as range_max + 1, the simulator's convention.
    best[best > RANGE_MAX] = RANGE_MAX + 1.0
    return LaserScan(
        ranges=best,
        angle_min=-math.pi,
        angle_increment=2 * math.pi / BEAMS,
        range_min=0.1,
        range_max=RANGE_MAX,
    )


def scan_message(x: float, y: float, yaw: float = 0.0) -> dict:
    """The same scan as a rosbridge `sensor_msgs/LaserScan` dict."""
    scan = raycast(x, y, yaw)
    return {
        "header": {"seq": 1, "stamp": {"secs": 100, "nsecs": 500000000}, "frame_id": "laser_frame"},
        "angle_min": scan.angle_min,
        "angle_max": scan.angle_min + (BEAMS - 1) * scan.angle_increment,
        "angle_increment": scan.angle_increment,
        "range_min": scan.range_min,
        "range_max": scan.range_max,
        "ranges": [float(r) for r in scan.ranges],
        "intensities": [],
    }


def mapped_room(resolution: float = 0.05, poses=((1.0, 1.0, 0.0), (4.5, 2.5, 1.2), (2.0, 3.0, -0.7))):
    """An OccupancyGrid built by observing WALLS from `poses`."""
    from robot_console.slam.grid import OccupancyGrid
    from robot_console.slam.scan import scan_points, transform_points

    grid = OccupancyGrid(resolution)
    for pose in poses:
        pts = scan_points(raycast(*pose), max_range=8.0, offset=(0.0, 0.0))
        grid.integrate(pose, transform_points(pts, pose), max_range=8.0)
    return grid


def wall_distance(points: np.ndarray, walls: np.ndarray = WALLS) -> np.ndarray:
    """Distance from each (N, 2) point to the nearest wall segment."""
    a, b = walls[:, :2], walls[:, 2:]
    ab = b - a
    out = np.empty(len(points))
    for i, p in enumerate(np.asarray(points, dtype=np.float64).reshape(-1, 2)):
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0.0, 1.0)
        out[i] = np.linalg.norm(a + t[:, None] * ab - p, axis=1).min()
    return out
