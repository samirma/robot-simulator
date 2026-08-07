"""Following a path with a holonomic base.

The myAGV is Mecanum, so this is not a car: it can translate in any direction while
holding or changing its heading independently. That makes the follower much simpler than
a differential-drive one -- no turn-in-place-then-go, no arc planning -- but it also means
"drive toward the carrot" and "point where you are going" are two separate, simultaneous
outputs rather than one coupled decision.

Pure: `PathFollower.step` takes a pose, a path and a scan, and returns a `Command`. No
clock, no network, no state that a test cannot construct.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from robot_console.bridge import wrap_angle
from robot_console.slam.scan import LaserScan, nearest_obstacle
from robot_console.teleop import SPEED_MAX, TURN_MAX, Command

# Carrot distance. Short enough to track a corner, long enough that the base does not
# chase quantisation noise in the path.
LOOKAHEAD_M = 0.45

# Close enough to the final waypoint to call it arrived. One robot radius: asking a base
# with a 0.05 m setpoint lead to land inside a few centimetres just makes it hunt.
ARRIVAL_M = 0.20

# Heading is servoed toward the direction of travel with this gain, but only once the
# robot is actually going somewhere -- yawing to face a goal 5 cm away is pointless.
YAW_GAIN = 1.6
MIN_HEADING_SPEED = 0.05

# Obstacle braking. Below STOP_M the base holds still and waits for a replan; between
# STOP_M and SLOW_M speed scales linearly. The lidar sits 65 mm ahead of centre and the
# chassis reaches 0.156 m, so 0.35 m is roughly two chassis-lengths of warning.
SLOW_M = 0.7
STOP_M = 0.35
BRAKE_HALF_ANGLE = math.radians(35.0)

# Progress watchdog: if the robot has not closed this much distance in this long, the
# path is not working and the caller should replan rather than keep pushing.
STUCK_DISTANCE_M = 0.08
STUCK_SECONDS = 4.0


@dataclasses.dataclass(frozen=True)
class FollowResult:
    command: Command
    arrived: bool = False
    blocked: bool = False
    target: Optional[np.ndarray] = None
    remaining: float = 0.0

    @property
    def should_replan(self) -> bool:
        return self.blocked and not self.arrived


class PathFollower:
    """Turns a path into `/cmd_vel`, with arrival, braking and stuck detection."""

    def __init__(
        self,
        *,
        speed: float = 0.18,
        lookahead: float = LOOKAHEAD_M,
        arrival: float = ARRIVAL_M,
        speed_max: float = SPEED_MAX,
    ) -> None:
        self.speed = min(float(speed), float(speed_max))
        self.speed_max = float(speed_max)
        self.lookahead = float(lookahead)
        self.arrival = float(arrival)
        self._best_remaining = math.inf
        self._best_at: Optional[float] = None

    def reset(self) -> None:
        """Call on a new path, so the stuck watchdog does not inherit the last one's history."""
        self._best_remaining = math.inf
        self._best_at = None

    def step(
        self,
        pose: Sequence[float],
        path: Optional[np.ndarray],
        *,
        scan: Optional[LaserScan] = None,
        now: Optional[float] = None,
    ) -> FollowResult:
        if path is None or len(path) == 0:
            return FollowResult(Command())
        pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
        here = np.asarray(pose[:2], dtype=np.float64)

        remaining = float(np.hypot(*(pts[-1] - here)))
        if remaining <= self.arrival:
            return FollowResult(Command(), arrived=True, target=pts[-1], remaining=remaining)

        target = carrot(pts, here, self.lookahead)
        delta = target - here
        distance = float(np.hypot(*delta))
        if distance < 1e-6:
            return FollowResult(Command(), target=target, remaining=remaining)

        speed = self.speed
        # Ease off over the last stretch so arrival is a stop, not an overshoot-and-return.
        speed = min(speed, max(0.04, self.speed * remaining / max(self.arrival * 3.0, 1e-6)))

        # Direction of travel, in the body frame. Worked out before the obstacle check,
        # because on a holonomic base the check has to look where the robot is *going* --
        # see `nearest_obstacle`.
        yaw = float(pose[2])
        c, s = math.cos(yaw), math.sin(yaw)
        heading = delta / distance
        travel = math.atan2(-heading[0] * s + heading[1] * c, heading[0] * c + heading[1] * s)

        if scan is not None:
            clearance = nearest_obstacle(scan, BRAKE_HALF_ANGLE, bearing=travel)
            if clearance <= STOP_M:
                # Something is in the way that the map did not know about. Stop and let the
                # caller replan; creeping forward on a stale path is how a base wedges
                # itself under a chair.
                #
                # The watchdog is armed *before* returning, and that is the whole point: a
                # robot held still by the brake is exactly what "not making progress"
                # means. Returning early without it left `_best_at` at None, so `is_stuck`
                # answered False forever, the explorer rerouted to the same goal every
                # tick, and the run neither blacklisted the goal nor ever finished.
                self._track_progress(remaining, now)
                return FollowResult(
                    Command(), blocked=True, target=target, remaining=remaining
                )
            if clearance < SLOW_M:
                scale = (clearance - STOP_M) / (SLOW_M - STOP_M)
                speed *= max(0.25, scale)

        # World-frame velocity, rotated into the body frame -- which is what /cmd_vel is.
        world = heading * speed
        vx = world[0] * c + world[1] * s
        vy = -world[0] * s + world[1] * c

        # Face the way we are travelling. Holonomic, so this is free -- it just keeps the
        # camera and the lidar's better-sampled forward arc pointed where the robot is going.
        wz = 0.0
        if speed >= MIN_HEADING_SPEED:
            wz = clamp(
                YAW_GAIN * wrap_angle(math.atan2(delta[1], delta[0]) - yaw),
                -TURN_MAX,
                TURN_MAX,
            )

        self._track_progress(remaining, now)
        return FollowResult(
            Command(vx=vx, vy=vy, wz=wz),
            target=target,
            remaining=remaining,
        )

    def is_stuck(self, now: Optional[float] = None) -> bool:
        """True when the robot has stopped making progress toward the goal."""
        if self._best_at is None or now is None:
            return False
        return (now - self._best_at) > STUCK_SECONDS

    def _track_progress(self, remaining: float, now: Optional[float]) -> None:
        if now is None:
            return
        if remaining < self._best_remaining - STUCK_DISTANCE_M or self._best_at is None:
            self._best_remaining = remaining
            self._best_at = now


def carrot(path: np.ndarray, here: np.ndarray, lookahead: float) -> np.ndarray:
    """The point `lookahead` metres along the path, measured from the closest point on it.

    Projecting onto the path rather than picking the nearest *waypoint* is what keeps the
    base from cutting corners when waypoints are far apart, which they are after
    `planner.simplify`.
    """
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 1:
        return pts[0]

    seg_index, t = _closest_on_path(pts, here)
    # Walk forward from the projection, spending `lookahead` metres of path.
    budget = lookahead
    i = seg_index
    point = pts[i] + t * (pts[i + 1] - pts[i])
    while i < pts.shape[0] - 1:
        seg = pts[i + 1] - point
        length = float(np.hypot(*seg))
        if length >= budget:
            return point + seg / max(length, 1e-9) * budget
        budget -= length
        i += 1
        point = pts[i]
    return pts[-1]


def _closest_on_path(pts: np.ndarray, here: np.ndarray) -> Tuple[int, float]:
    starts, ends = pts[:-1], pts[1:]
    seg = ends - starts
    length_sq = np.maximum((seg ** 2).sum(axis=1), 1e-12)
    t = np.clip(((here - starts) * seg).sum(axis=1) / length_sq, 0.0, 1.0)
    proj = starts + t[:, None] * seg
    d = np.linalg.norm(proj - here, axis=1)
    i = int(np.argmin(d))
    return i, float(t[i])


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scale_to_limits(command: Command, speed_max: float = SPEED_MAX) -> Command:
    """Shrink a command so its translation fits the envelope, keeping its direction.

    Clipping the axes independently would rotate the commanded direction, which turns a
    too-fast diagonal into a slow curve away from the path.
    """
    magnitude = math.hypot(command.vx, command.vy)
    if magnitude <= speed_max or magnitude < 1e-9:
        return Command(command.vx, command.vy, clamp(command.wz, -TURN_MAX, TURN_MAX))
    k = speed_max / magnitude
    return Command(command.vx * k, command.vy * k, clamp(command.wz, -TURN_MAX, TURN_MAX))
