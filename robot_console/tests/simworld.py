"""A house, a robot that moves in it, and a loop that drives exploration to completion.

`synthetic.py` gives a scan from a pose; this gives the missing half -- a pose that
responds to a `Command` -- so a whole exploration run can be executed offline. That is the
only way to make a claim about *coverage* testable: every other frontier test is a single
shot at a static half-seen grid, and a strategy that abandons a room still passes all of
them.

Deliberately kept honest about the things that actually broke exploration:

- **rooms behind doorways**, because the single 6x4 m `WALLS` room cannot express "the
  robot never went in there";
- **a narrow doorway**, because a 0.5 m blacklist radius sealing one is what left whole
  rooms unmapped;
- **walls that stop the base**, because a robot that tunnels through furniture maps a
  house no real run could.

No RNG, no wall clock: `now` is `tick * dt` and `raycast` is exact, so a failure is
reproducible from the seedless test alone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from synthetic import raycast

from robot_console.slam.grid import FREE, OccupancyGrid
from robot_console.slam.planner import CostMap
from robot_console.teleop import Command

# A 10 x 7 m flat: a hall across the bottom, three rooms off it, and a narrow doorway into
# the last one. Segments are (x0, y0, x1, y1).
HOUSE = np.array(
    [
        # outer shell
        (0.0, 0.0, 10.0, 0.0),
        (0.0, 7.0, 10.0, 7.0),
        (0.0, 0.0, 0.0, 7.0),
        (10.0, 0.0, 10.0, 7.0),
        # the wall separating the hall (y < 2) from the rooms above it, with three gaps:
        # 1.6-2.5 (west room), 4.6-5.5 (middle room), 7.8-8.5 (narrow, east room)
        (0.0, 2.0, 1.6, 2.0),
        (2.5, 2.0, 4.6, 2.0),
        (5.5, 2.0, 7.8, 2.0),
        (8.5, 2.0, 10.0, 2.0),
        # partitions between the three rooms
        (3.5, 2.0, 3.5, 7.0),
        (6.8, 2.0, 6.8, 7.0),
    ],
    dtype=np.float64,
)

# Room name -> (x0, y0, x1, y1), used to assert that each was actually entered.
ROOMS: Dict[str, Tuple[float, float, float, float]] = {
    "hall": (0.2, 0.2, 9.8, 1.8),
    "west": (0.2, 2.2, 3.3, 6.8),
    "middle": (3.7, 2.2, 6.6, 6.8),
    "east": (7.0, 2.2, 9.8, 6.8),
}

START = (1.0, 1.0, 0.0)

# The myAGV is 0.311 x 0.230 m; this is the circumscribed radius the planner assumes.
ROBOT_RADIUS_M = 0.20


def sealed_house() -> np.ndarray:
    """`HOUSE` with the narrow east doorway bricked up.

    Exploration must still *finish* -- and must not claim to have mapped the room it can
    no longer reach.
    """
    return np.vstack([HOUSE, np.array([(7.8, 2.0, 8.5, 2.0)])])


# ------------------------------------------------------------------ motion


def _segment_distance(point: np.ndarray, walls: np.ndarray) -> float:
    a, b = walls[:, :2], walls[:, 2:]
    ab = b - a
    t = np.clip(((point - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * ab - point, axis=1).min())


def step_pose(
    pose,
    command: Command,
    dt: float,
    *,
    walls: np.ndarray = HOUSE,
    radius: float = ROBOT_RADIUS_M,
) -> np.ndarray:
    """Integrate one `Command` in the body frame, refusing moves that hit a wall.

    Rejection rather than sliding: a base that slides along whatever it touches quietly
    rescues a planner that aims through walls, and the point of this harness is to notice
    when that happens.
    """
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    nx = x + (command.vx * c - command.vy * s) * dt
    ny = y + (command.vx * s + command.vy * c) * dt
    nyaw = math.atan2(math.sin(yaw + command.wz * dt), math.cos(yaw + command.wz * dt))

    # Test the whole swept step, not just where it ends up. Checking only the destination
    # lets a long step hop clean over a wall and land legally on the far side, which would
    # make this harness certify a planner that routes through walls.
    here = np.array([x, y])
    there = np.array([nx, ny])
    travelled = float(np.hypot(*(there - here)))
    samples = max(2, int(math.ceil(travelled / max(radius * 0.5, 1e-6))) + 1)
    for t in np.linspace(0.0, 1.0, samples):
        if _segment_distance(here + t * (there - here), walls) <= radius:
            return np.array([x, y, nyaw], dtype=np.float64)  # rotation is always allowed
    return np.array([nx, ny, nyaw], dtype=np.float64)


class World:
    """Ground truth: where the robot really is, and what it really sees."""

    def __init__(
        self,
        *,
        walls: np.ndarray = HOUSE,
        start=START,
        dt: float = 0.2,
        max_range: float = 5.0,
        radius: float = ROBOT_RADIUS_M,
    ) -> None:
        self.walls = np.asarray(walls, dtype=np.float64)
        self.pose = np.asarray(start, dtype=np.float64).copy()
        self.dt = float(dt)
        self.max_range = float(max_range)
        self.radius = float(radius)
        self.history: List[np.ndarray] = [self.pose.copy()]

    def scan(self):
        return raycast(*self.pose, walls=self.walls)

    def drive(self, command: Command) -> None:
        self.pose = step_pose(
            self.pose, command, self.dt, walls=self.walls, radius=self.radius
        )
        self.history.append(self.pose.copy())


# ------------------------------------------------------------------ ground truth


def reachable_mask(
    grid: OccupancyGrid,
    *,
    walls: np.ndarray = HOUSE,
    start=START,
    radius: float = ROBOT_RADIUS_M,
) -> np.ndarray:
    """Cells the base could actually stand in, flood-filled from `start`.

    Coverage has to be measured against this rather than the grid's bounding box: the box
    includes the outside of the house and the insides of walls, so a run that maps
    everything reachable still scores far below 100 % against it.
    """
    height, width = grid.data.shape
    ys, xs = np.mgrid[0:height, 0:width]
    centres = np.stack(
        [
            grid.origin[0] + (xs.ravel() + 0.5) * grid.resolution,
            grid.origin[1] + (ys.ravel() + 0.5) * grid.resolution,
        ],
        axis=1,
    )
    clear = np.array([_segment_distance(p, walls) > radius for p in centres])
    open_cells = clear.reshape(height, width)

    seed = grid.world_to_cell(np.asarray(start, dtype=np.float64)[None, :2])[0]
    if not (0 <= seed[0] < width and 0 <= seed[1] < height) or not open_cells[seed[1], seed[0]]:
        return np.zeros_like(open_cells)

    out = np.zeros_like(open_cells)
    stack = [(int(seed[0]), int(seed[1]))]
    out[seed[1], seed[0]] = True
    while stack:
        cx, cy = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and open_cells[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                stack.append((nx, ny))
    return out


def coverage(grid: OccupancyGrid, reachable: np.ndarray) -> float:
    """Fraction of reachable cells that are no longer unknown."""
    total = int(reachable.sum())
    if total == 0:
        return 1.0
    known = grid.classify() != 0
    return float((known & reachable).sum()) / total


def room_coverage(grid: OccupancyGrid, room: Tuple[float, float, float, float]) -> float:
    """Fraction of one room's floor that has been mapped.

    Worth asserting separately: a whole-house fraction hides a missed room behind the
    corridor and the rooms that *were* visited.
    """
    x0, y0, x1, y1 = room
    lo = grid.world_to_cell(np.array([[x0, y0]]))[0]
    hi = grid.world_to_cell(np.array([[x1, y1]]))[0]
    height, width = grid.data.shape
    x_lo, x_hi = max(0, lo[0]), min(width, hi[0] + 1)
    y_lo, y_hi = max(0, lo[1]), min(height, hi[1] + 1)
    if x_hi <= x_lo or y_hi <= y_lo:
        return 0.0
    patch = grid.classify()[y_lo:y_hi, x_lo:x_hi]
    return float((patch != 0).sum()) / patch.size


def visited(history: List[np.ndarray], room: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = room
    return any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in history)


def residual_frontiers(grid: OccupancyGrid, *, min_cells: int = 3) -> int:
    """Frontier clusters still on the map -- places the robot knew about and never went."""
    from robot_console.slam.frontier import frontier_clusters

    return len(frontier_clusters(grid, min_cells=min_cells)[1])


def reachable_residual_frontiers(
    grid: OccupancyGrid, start=START, *, min_cells: int = 3
) -> int:
    """Residual frontiers the robot could still have driven to.

    A sealed-off room leaves frontiers on the map forever; those are not a failure of the
    explorer, and counting them as one would make the sealed-house test unsatisfiable.
    """
    from robot_console.slam.frontier import rank_frontiers
    from robot_console.slam.planner import distance_field

    cost = CostMap(grid, allow_unknown=True)
    field = distance_field(cost, np.asarray(start, dtype=np.float64)[:2])
    return sum(1 for f in rank_frontiers(grid, field, min_cells=min_cells) if f.reachable)


# ------------------------------------------------------------------ the driver


def offline_session(
    *,
    resolution: float = 0.05,
    max_range: float = 5.0,
    speed: float = 0.25,
    **overrides,
):
    """A `_Session` with no socket, no window and no recorder.

    Same construction `tests/test_slam_cli.py` uses; scan matching is off so nothing
    reaches for the wall clock and the run stays deterministic.
    """
    from robot_console.slam.app import _Session
    from robot_console.slam.cli import SlamOptions
    from robot_console.slam.controller import PathFollower
    from robot_console.slam.mapview import MapView
    from robot_console.slam.pose import PoseTracker
    from robot_console.teleop import TeleopState

    options = SlamOptions(
        mode="explore", resolution=resolution, max_range=max_range,
        speed=speed, no_match=True, camera_window=False, **overrides,
    )
    grid = OccupancyGrid(resolution)
    return _Session(
        options, grid, PoseTracker(match_enabled=False),
        PathFollower(speed=speed), MapView(), None, TeleopState(speed=speed), None,
    )


class _Odom:
    """Perfect odometry, which is what `no_match=True` implies anyway."""

    __slots__ = ("x", "y", "yaw", "vx", "vy", "wz")

    def __init__(self, pose) -> None:
        self.x, self.y, self.yaw = float(pose[0]), float(pose[1]), float(pose[2])
        self.vx = self.vy = self.wz = 0.0


def run_exploration(
    *,
    walls: np.ndarray = HOUSE,
    start=START,
    resolution: float = 0.05,
    max_range: float = 5.0,
    dt: float = 0.2,
    max_ticks: int = 3000,
    **overrides,
) -> dict:
    """Drive a real `_Session` around `walls` until it says it is finished.

    Returns the grid, the path actually taken, why it stopped, and the worst single
    `decide` -- that last one because a strategy that maps beautifully but overruns the
    50 ms publish period is a robot that drives badly.
    """
    import time

    from robot_console.slam.scan import scan_points, transform_points

    world = World(walls=walls, start=start, dt=dt, max_range=max_range)
    session = offline_session(resolution=resolution, max_range=max_range, **overrides)

    worst_ms = 0.0
    reason: Optional[str] = None
    for tick in range(max_ticks):
        now = tick * dt
        session.tracker.update_odom(_Odom(world.pose))
        session.on_scan(world.scan(), now)

        started = time.perf_counter()
        command = session.decide(now)
        worst_ms = max(worst_ms, (time.perf_counter() - started) * 1000.0)

        if session.finished:
            reason = session.finished
            break
        world.drive(command)

    return {
        "grid": session.grid,
        "history": world.history,
        "reason": reason,
        "ticks": tick + 1,
        "seconds": (tick + 1) * dt,
        "worst_decide_ms": worst_ms,
        "session": session,
    }
