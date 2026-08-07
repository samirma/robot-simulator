"""Where to go next: the boundary between what has been mapped and what has not.

A frontier is a free cell adjacent to an unknown one. Drive to frontiers until there are
none and the map is complete by construction -- which is why exploration is posed this way
rather than as a coverage or lawnmower pattern that has no idea when it is finished.

Detection is morphology on the classified grid, so a house-sized map costs well under a
millisecond. The expensive question is not *where* the frontiers are but *which one to go
to*, and that is where this used to get it wrong in three separate ways:

- it ranked candidates by **straight-line** distance, so a frontier a metre away through a
  wall beat one three metres down an open corridor;
- it planned to only the best twelve, and returned "nothing left" when all twelve failed,
  which is a finished map and an unlucky shortlist reported identically;
- it aimed at the cluster **centroid**, which for a C-shaped or ring-shaped cluster sits in
  the wall the cluster wraps around, so the biggest frontiers were the likeliest to be
  discarded as unreachable.

All three are answered by one `planner.distance_field`: it gives true cost-to-come to every
cell at once, so every cluster can be scored, unreachable means unreachable rather than
unlucky, and the goal is picked as a real cell near the cluster instead of an average of
its coordinates.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from robot_console.slam.grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from robot_console.slam.planner import (
    CostMap,
    DistanceField,
    distance_field,
    path_length,
    plan,
)

# A frontier smaller than this is noise at the edge of the sensor's reach, or a gap behind
# a chair leg the robot cannot fit through anyway. It is a *starting* threshold: the
# explorer relaxes it before giving up, because a doorway seen edge-on at range is a two-
# or three-cell frontier and a permanent floor here makes a whole room invisible.
MIN_FRONTIER_CELLS = 6

# Utility is size / (distance + BIAS). BIAS in metres sets how much a big far frontier is
# worth against a small near one; at 2 m the robot finishes the room it is in before
# crossing the house, which produces far less backtracking.
DISTANCE_BIAS_M = 2.0

# How far short of the boundary to stop. Matched to `controller.STOP_M`: a goal any nearer
# sits inside the obstacle brake, so the robot arrives already braking, reports "blocked"
# rather than "arrived", and re-routes to a frontier it is standing on.
FRONTIER_STANDOFF_M = 0.35

# A goal that could not be reached is suppressed within this radius. It was 0.5 m, which is
# wider than half a 0.8 m doorway -- one strike against a goal in a doorway sealed the room
# behind it for the rest of the run.
BLACKLIST_RADIUS_M = 0.25

# Suppression decays. A frontier abandoned because someone walked past, or because the pose
# estimate was briefly bad, has to become available again or the map ends up with holes
# that no amount of driving can fill.
BLACKLIST_SECONDS = 45.0
BLACKLIST_STRIKES = 2

# How many TTLs a single-strike entry is remembered for. Suppression lapses after one TTL;
# the *tally* has to outlive that, or a frontier revisited every couple of minutes never
# accumulates a second strike and never becomes permanent.
FORGET_MULTIPLE = 10.0

# A goal nearer than this is one the robot has effectively already reached: the follower
# calls anything inside `controller.ARRIVAL_M` arrived, so issuing one just burns a tick.
# Comfortably above that threshold, so a goal is always worth the drive.
MIN_PROGRESS_M = 0.45

# The incumbent goal's utility is multiplied by this when it is re-scored. Without it the
# full re-selection every 0.6 m makes two comparable frontiers on opposite sides trade
# places as the robot moves, and it dithers between them instead of finishing either.
INCUMBENT_BONUS = 1.25

# An unknown region this size or smaller, fully enclosed by mapped space, is a sensor hole
# rather than an unexplored room -- a cell needs four free observations to count as FREE
# and only one hit to count as OCCUPIED, so grazing beams leave weakly-seen gaps behind.
POCKET_MAX_CELLS = 400


@dataclasses.dataclass(frozen=True)
class Frontier:
    # `centroid` stays the identity of a frontier: `mapview` draws it and `vslam` steers by
    # it. `goal` is where the robot is actually sent, which is a different point whenever
    # the centroid is not somewhere the base can stand.
    centroid: np.ndarray
    size: int
    distance: float = 0.0
    utility: float = 0.0
    goal: Optional[np.ndarray] = None
    reachable: bool = True

    def with_scores(self, distance: float, utility: float) -> "Frontier":
        return Frontier(
            self.centroid, self.size, distance, utility, self.goal, self.reachable
        )

    @property
    def target(self) -> np.ndarray:
        """Where to drive. The standoff goal when there is one, else the centroid."""
        return self.centroid if self.goal is None else self.goal


@dataclasses.dataclass(frozen=True)
class GoalChoice:
    """Why goal selection returned what it did.

    The distinction is the point. `choose_goal` used to answer `(None, None)` for a
    finished map, for a shortlist that all failed to plan, and for everything being
    suppressed -- and the explorer, unable to tell them apart, called all three "explored"
    and stopped with frontiers still on the map.
    """

    frontier: Optional[Frontier] = None
    path: Optional[np.ndarray] = None
    frontiers: List[Frontier] = dataclasses.field(default_factory=list)
    # "ok" | "finished" | "unreachable" | "blacklisted" | "no_path"
    reason: str = "finished"

    @property
    def found(self) -> bool:
        return self.frontier is not None and self.path is not None


class Blacklist:
    """Goals that failed, forgotten again after a while.

    Append-only suppression is why exploration used to leave whole rooms unmapped: a goal
    abandoned once stayed abandoned for the session, and the radius was wide enough that a
    single entry could cover a doorway. Here an entry has to earn permanence -- it expires
    on a timer unless it has failed `strikes` times.
    """

    def __init__(
        self,
        *,
        radius: float = BLACKLIST_RADIUS_M,
        ttl: float = BLACKLIST_SECONDS,
        strikes: int = BLACKLIST_STRIKES,
    ) -> None:
        self.radius = float(radius)
        self.ttl = float(ttl)
        self.strikes = int(strikes)
        self._entries: List[dict] = []

    def strike(self, point: Sequence[float], now: float) -> int:
        """Record a failure at `point`; returns how many it has now."""
        target = np.asarray(point, dtype=np.float64)[:2]
        for entry in self._entries:
            if float(np.hypot(*(entry["point"] - target))) < self.radius:
                entry["strikes"] += 1
                entry["at"] = now
                return int(entry["strikes"])
        self._entries.append({"point": target.copy(), "strikes": 1, "at": now})
        return 1

    def suppress(self, point: Sequence[float], now: float) -> None:
        """Retire `point` outright, rather than adding one strike.

        For failures that repetition cannot fix. Standing at a frontier's standoff point
        and finding the frontier still there is the case: the lidar has seen everything it
        will ever see from there, so coming back is guaranteed to achieve nothing. The
        radius is small enough that this retires an approach point, not a cluster -- the
        rest of a long frontier still gets its own goal cell next time round.
        """
        target = np.asarray(point, dtype=np.float64)[:2]
        for entry in self._entries:
            if float(np.hypot(*(entry["point"] - target))) < self.radius:
                entry["strikes"] = max(entry["strikes"], self.strikes)
                entry["at"] = now
                return
        self._entries.append(
            {"point": target.copy(), "strikes": self.strikes, "at": now}
        )

    def blocks(self, point: Sequence[float], now: float) -> bool:
        target = np.asarray(point, dtype=np.float64)[:2]
        for entry in self._entries:
            if float(np.hypot(*(entry["point"] - target))) >= self.radius:
                continue
            if entry["strikes"] >= self.strikes or now - entry["at"] < self.ttl:
                return True
        return False

    def expire(self, now: float) -> None:
        """Drop entries that have gone quiet, *keeping* their strike counts.

        Suppression lapsing and the tally being forgotten are different things, and
        conflating them makes the tally useless: a frontier against an outer wall is
        revisited every couple of minutes, so if the count reset each time the TTL lapsed
        it would sit at one for the whole run and never earn permanence. Entries are only
        really discarded once they have gone a long time without ever failing twice.
        """
        cutoff = self.ttl * FORGET_MULTIPLE
        self._entries = [
            e for e in self._entries
            if e["strikes"] >= self.strikes or now - e["at"] < cutoff
        ]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class _HardBlacklist:
    """Adapter for a bare sequence of points: suppression with no expiry.

    `vslam` keeps its own expiry and hands over a plain list, and so do the older tests.
    Treating that as permanent within the call is exactly the old behaviour.
    """

    def __init__(self, points: Sequence[Sequence[float]], radius: float) -> None:
        self.points = [np.asarray(p, dtype=np.float64)[:2] for p in points]
        self.radius = float(radius)

    def blocks(self, point: Sequence[float], _now: float) -> bool:
        target = np.asarray(point, dtype=np.float64)[:2]
        return any(float(np.hypot(*(p - target))) < self.radius for p in self.points)


def _as_blacklist(blacklist, radius: float = BLACKLIST_RADIUS_M):
    if blacklist is None:
        return None
    if hasattr(blacklist, "blocks"):
        return blacklist
    return _HardBlacklist(blacklist, radius)


# ------------------------------------------------------------------ detection


def frontier_mask(grid: OccupancyGrid, *, min_thickness: int = 1) -> np.ndarray:
    """Boolean mask of free cells that touch unknown space worth driving to.

    "Worth driving to" is doing real work here. Walls are recorded a cell thick, and a
    beam arriving at a grazing angle steps over cells rather than landing in every one,
    so a mapped wall is a *dashed* line of occupied cells with unknown slivers between
    them. Those slivers are free space's neighbours, so the naive test calls them
    frontiers -- and they are frontiers that can never be resolved, because what is behind
    them is behind a wall. Left in, they are what an explorer spends its endgame on:
    measured on the four-room house, the robot drove wall to wall for ninety seconds
    picking a new one each time and mapping nothing, until the stall guard stopped it.

    Eroding the unknown region first is the whole filter. A one-cell seam has no interior
    and vanishes; a room nobody has entered has plenty and survives untouched.
    """
    classes = grid.classify()
    free = (classes == FREE).astype(np.uint8)
    unknown = (classes == UNKNOWN).astype(np.uint8)
    if min_thickness > 0:
        # An *opening*, not a bare erosion: erode-then-dilate with the same kernel deletes
        # anything thinner than the kernel and leaves everything else where it was. Eroding
        # alone would pull every boundary inward by a cell and fragment the thin ring of
        # frontier that a single scan produces, which is not the same test at all.
        size = 2 * int(min_thickness) + 1
        unknown = cv2.morphologyEx(
            unknown, cv2.MORPH_OPEN, np.ones((size, size), np.uint8)
        )
    # A cell is a frontier if it is free and any 4-neighbour is unknown. Dilating the
    # unknown region and intersecting is the same test, done in one pass.
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    return (free & cv2.dilate(unknown, kernel)).astype(bool)


def frontier_clusters(
    grid: OccupancyGrid, *, min_cells: int = MIN_FRONTIER_CELLS
) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray]:
    """`(labels, kept_labels, centroids)` for the frontier components worth considering.

    Returns the label image rather than only centroids, because scoring needs each
    cluster's actual cells -- the centroid of a cluster that wraps a corner is not on the
    cluster.
    """
    mask = frontier_mask(grid).astype(np.uint8)
    if not mask.any():
        return np.zeros(mask.shape, np.int32), [], np.empty((0, 2)), np.empty((0, 5), int)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    kept = [
        i for i in range(1, count)  # label 0 is the background
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_cells
    ]
    return labels, kept, centroids, stats


def _padded_bounds(stat, pad: int, shape) -> Tuple[int, int, int, int]:
    x, y = int(stat[cv2.CC_STAT_LEFT]), int(stat[cv2.CC_STAT_TOP])
    w, h = int(stat[cv2.CC_STAT_WIDTH]), int(stat[cv2.CC_STAT_HEIGHT])
    return (
        max(0, y - pad), min(shape[0], y + h + pad),
        max(0, x - pad), min(shape[1], x + w + pad),
    )


def find_frontiers(
    grid: OccupancyGrid,
    *,
    min_cells: int = MIN_FRONTIER_CELLS,
) -> List[Frontier]:
    """Cluster the frontier cells and return one candidate per cluster, largest first."""
    labels, kept, centroids, stats = frontier_clusters(grid, min_cells=min_cells)
    out: List[Frontier] = []
    for i in kept:
        size = int((labels == i).sum())
        world = grid.origin + (centroids[i] + 0.5) * grid.resolution
        out.append(Frontier(world, size))
    out.sort(key=lambda f: f.size, reverse=True)
    return out


def unknown_pockets(
    grid: OccupancyGrid, *, max_cells: int = POCKET_MAX_CELLS
) -> List[Frontier]:
    """Small enclosed unknown regions: sensor holes rather than unexplored rooms.

    `classify` needs four free observations to promote a cell out of UNKNOWN but a single
    hit to call it OCCUPIED, so grazing-incidence and long-range beams leave weakly
    observed cells scattered *inside* rooms the robot has already driven through. Their
    frontier rings are only a few cells across and get filtered as speckle, so ordinary
    frontier exploration never resolves them. Measured on saved maps that is 58 pockets on
    one house and 219 on another.

    A genuinely unexplored room is large and touches the edge of the map; a sensor hole is
    small and enclosed. Both tests are applied.
    """
    classes = grid.classify()
    unknown = (classes == UNKNOWN).astype(np.uint8)
    if not unknown.any():
        return []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        unknown, connectivity=8
    )
    # A wall is recorded a cell thick and grazing beams skip cells, so a mapped wall has
    # unknown slivers along it -- small, enclosed, and indistinguishable from a sensor
    # hole by size alone. They are told apart by what surrounds them: a hole in the floor
    # is ringed by free space, a sliver in a wall is flanked by the wall. Without this the
    # explorer spends its endgame driving at walls trying to see through them.
    touching_wall = cv2.dilate(
        (classes == OCCUPIED).astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    height, width = classes.shape
    out: List[Frontier] = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > max_cells:
            continue
        if touching_wall[labels == i].any():
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue  # touches the edge of the map: unexplored, not a hole
        world = grid.origin + (centroids[i] + 0.5) * grid.resolution
        out.append(Frontier(world, area))
    out.sort(key=lambda f: f.size, reverse=True)
    return out


# ------------------------------------------------------------------ scoring


def _upsampled(field: DistanceField, shape: Tuple[int, int]) -> np.ndarray:
    """The coarse field at grid resolution, so cluster masks can index it directly.

    Cached on the field. The give-up ladder re-scores the same wavefront at several
    thresholds, and rebuilding a house-sized float array per rung was most of what pushed
    a replan tick past the publish period.
    """
    cached = getattr(field, "_upsampled", None)
    if cached is not None and cached.shape == shape:
        return cached
    if field.decimate == 1:
        coarse = field.dist
    else:
        coarse = np.repeat(
            np.repeat(field.dist, field.decimate, axis=0), field.decimate, axis=1
        )
    out = np.full(shape, np.inf, dtype=np.float32)
    h = min(shape[0], coarse.shape[0])
    w = min(shape[1], coarse.shape[1])
    out[:h, :w] = coarse[:h, :w]
    field._upsampled = out
    return out


def _too_close(
    field: DistanceField, grid: OccupancyGrid, shape: Tuple[int, int], min_progress: float
) -> np.ndarray:
    """Cells within `min_progress` of the wavefront's source. Cached alongside it."""
    key = (shape, round(float(min_progress), 6))
    if getattr(field, "_close_key", None) == key:
        return field._close_mask
    origin_cell = grid.world_to_cell(field.source[None, :])[0]
    yy, xx = np.ogrid[0:shape[0], 0:shape[1]]
    reach = max(1.0, min_progress / grid.resolution)
    mask = ((xx - origin_cell[0]) ** 2 + (yy - origin_cell[1]) ** 2) < reach ** 2
    field._close_key = key
    field._close_mask = mask
    return mask


def rank_frontiers(
    grid: OccupancyGrid,
    field: DistanceField,
    *,
    min_cells: int = MIN_FRONTIER_CELLS,
    blacklist=None,
    standoff: float = FRONTIER_STANDOFF_M,
    distance_bias: float = DISTANCE_BIAS_M,
    incumbent: Optional[Sequence[float]] = None,
    incumbent_bonus: float = INCUMBENT_BONUS,
    min_progress: float = MIN_PROGRESS_M,
    now: float = 0.0,
) -> List[Frontier]:
    """Score every cluster against the wavefront, best first.

    The goal for a cluster is the cheapest reachable cell in a `standoff`-wide collar
    around it, not its centroid. That one choice does three jobs: it never picks a point
    inside the wall a curved cluster wraps, it stops the robot short of the boundary
    instead of driving its centre onto it, and it is by construction somewhere the
    wavefront actually reached.
    """
    labels, kept, centroids, stats = frontier_clusters(grid, min_cells=min_cells)
    if not kept:
        return []

    suppress = _as_blacklist(blacklist)
    costs = _upsampled(field, labels.shape)
    collar = max(1, int(round(standoff / grid.resolution)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * collar + 1, 2 * collar + 1)
    )
    incumbent_xy = (
        None if incumbent is None else np.asarray(incumbent, dtype=np.float64)[:2]
    )

    # Cells the robot has effectively already reached. This has to be measured
    # geometrically rather than off the field: field values are scaled by the soft wall
    # penalty, so a cell 0.18 m away beside a doorway scores like one 0.6 m away in the
    # open, sails past the threshold, and is then handed to a follower that calls it
    # arrived on the tick it is issued -- a goal the robot re-picks and re-arrives at
    # forever without moving.
    too_close = _too_close(field, grid, labels.shape, min_progress)

    out: List[Frontier] = []
    for i in kept:
        centre = grid.origin + (centroids[i] + 0.5) * grid.resolution
        if suppress is not None and suppress.blocks(centre, now):
            continue

        # Work inside the cluster's own bounding box, padded by the collar. Dilating a
        # full house-sized array once per cluster is the difference between a few
        # milliseconds and tens of them when the map has grown and the thresholds have
        # been relaxed enough to admit seventy of them.
        y0, y1, x0, x1 = _padded_bounds(stats[i], collar, labels.shape)
        window = labels[y0:y1, x0:x1]
        cluster = (window == i).astype(np.uint8)
        size = int(cluster.sum())
        near = cv2.dilate(cluster, kernel).astype(bool)
        # A cell needs four free observations to classify as FREE but a single beam
        # endpoint is enough to be OCCUPIED, so early in a run the only solidly free space
        # is where beams converge -- right under the robot. The frontier is then a ring a
        # few centimetres out whose collar contains the robot's own cell, which is why the
        # near field has to be excluded rather than merely ranked last.
        candidates = np.where(
            near & ~too_close[y0:y1, x0:x1], costs[y0:y1, x0:x1], np.inf
        )
        flat = int(np.argmin(candidates))
        best = float(candidates.flat[flat])
        if not math.isfinite(best):
            out.append(Frontier(centre, size, math.inf, 0.0, None, False))
            continue

        wy, wx = np.unravel_index(flat, candidates.shape)
        iy, ix = wy + y0, wx + x0
        goal = grid.origin + (np.array([ix, iy], dtype=np.float64) + 0.5) * grid.resolution
        # The field is in cell-steps scaled by the soft cost; metres is what the utility's
        # distance bias is expressed in.
        metres = best * field.resolution
        utility = size / (metres + distance_bias)
        # Match the incumbent against the collar, not the cluster itself: the goal the
        # robot is currently driving to is a standoff short of the boundary, and a
        # centroid need not sit on its own cluster at all.
        iny, inx = (
            (-1, -1) if incumbent_xy is None
            else _clamp_cell(grid, incumbent_xy, labels.shape)
        )
        if incumbent_xy is not None and (
            y0 <= iny < y1 and x0 <= inx < x1 and near[iny - y0, inx - x0]
        ):
            utility *= incumbent_bonus
        out.append(Frontier(centre, size, metres, utility, goal, True))

    out.sort(key=lambda f: f.utility, reverse=True)
    return out


def _clamp_cell(
    grid: OccupancyGrid, point: np.ndarray, shape: Tuple[int, int]
) -> Tuple[int, int]:
    cell = grid.world_to_cell(point[None, :])[0]
    iy = int(np.clip(cell[1], 0, shape[0] - 1))
    ix = int(np.clip(cell[0], 0, shape[1] - 1))
    return iy, ix


# ------------------------------------------------------------------ selection


def survey(
    grid: OccupancyGrid,
    cost: CostMap,
    pose: Sequence[float],
    *,
    blacklist=None,
    min_cells: int = MIN_FRONTIER_CELLS,
    distance_bias: float = DISTANCE_BIAS_M,
    standoff: float = FRONTIER_STANDOFF_M,
    incumbent: Optional[Sequence[float]] = None,
    field: Optional[DistanceField] = None,
    min_progress: float = MIN_PROGRESS_M,
    now: float = 0.0,
) -> GoalChoice:
    """Pick the best reachable frontier and the path to it, and say why if there is none.

    One wavefront serves every candidate, so the caller can re-run this at a lower
    `min_cells` for the price of the scoring pass rather than another search -- which is
    what makes the explorer's give-up ladder affordable.
    """
    origin = np.asarray(pose[:2], dtype=np.float64)
    if field is None:
        field = distance_field(cost, origin)

    # Detect once at the loosest threshold, so "there are frontiers, just none big enough"
    # is distinguishable from "the map is finished".
    any_frontier = bool(frontier_clusters(grid, min_cells=1)[1])
    if not any_frontier:
        return GoalChoice(reason="finished")

    ranked = rank_frontiers(
        grid, field,
        min_cells=min_cells, blacklist=blacklist, standoff=standoff,
        distance_bias=distance_bias, incumbent=incumbent,
        min_progress=min_progress, now=now,
    )
    if not ranked:
        # Everything was either below `min_cells` or suppressed. Re-rank without the
        # blacklist to find out which, because the answers call for different recoveries.
        unsuppressed = rank_frontiers(
            grid, field, min_cells=min_cells, standoff=standoff,
            distance_bias=distance_bias, min_progress=min_progress, now=now,
        )
        return GoalChoice(reason="blacklisted" if unsuppressed else "unreachable")

    reachable = [f for f in ranked if f.reachable]
    if not reachable:
        return GoalChoice(frontiers=ranked, reason="unreachable")

    for candidate in reachable:
        path = plan(cost, origin, candidate.target)
        if path is None:
            continue
        # Re-score on the route actually found. The wavefront is computed on a decimated
        # grid, so it ranks well but is not the last word on length.
        travelled = path_length(path)
        scored = candidate.with_scores(
            travelled, candidate.size / (travelled + distance_bias)
        )
        return GoalChoice(scored, path, ranked, "ok")
    return GoalChoice(frontiers=ranked, reason="no_path")


def choose_goal(
    grid: OccupancyGrid,
    cost: CostMap,
    pose: Sequence[float],
    *,
    blacklist=None,
    min_cells: int = MIN_FRONTIER_CELLS,
    distance_bias: float = DISTANCE_BIAS_M,
    standoff: float = FRONTIER_STANDOFF_M,
    incumbent: Optional[Sequence[float]] = None,
    field: Optional[DistanceField] = None,
    now: float = 0.0,
    snap_to_frontier_cell: bool = True,
    max_candidates: Optional[int] = None,
):
    """`(frontier, path)`, or `(None, None)`. See `survey` for why there is none.

    `snap_to_frontier_cell` is accepted for callers that asked for it explicitly; goals are
    always a real cell near the cluster now, so it has nothing left to switch off.
    `max_candidates` is accepted and ignored -- capping the shortlist is what made a
    finished map and an unlucky one indistinguishable.
    """
    choice = survey(
        grid, cost, pose,
        blacklist=blacklist, min_cells=min_cells, distance_bias=distance_bias,
        standoff=standoff, incumbent=incumbent, field=field, now=now,
    )
    return (choice.frontier, choice.path) if choice.found else (None, None)


# ------------------------------------------------------------------ readout


def explored_area(grid: OccupancyGrid) -> float:
    """Square metres of the map that are no longer unknown."""
    return float((grid.classify() != UNKNOWN).sum()) * grid.resolution ** 2


def is_complete(grid: OccupancyGrid, *, min_cells: int = MIN_FRONTIER_CELLS) -> bool:
    return not find_frontiers(grid, min_cells=min_cells)


def approach_point(
    frontier: Sequence[float], pose: Sequence[float], standoff: float = 0.0
) -> np.ndarray:
    """A point `standoff` metres short of the frontier, along the line from `pose`.

    The geometric fallback for when a cluster has no reachable collar cell to aim at:
    driving onto a frontier cell means driving into the boundary of what is known, and a
    little standoff lets the lidar see past it without the base having to get there.
    """
    target = np.asarray(frontier, dtype=np.float64)[:2]
    origin = np.asarray(pose, dtype=np.float64)[:2]
    delta = target - origin
    d = float(np.hypot(*delta))
    if d <= standoff or d < 1e-6:
        return target
    return target - delta / d * standoff
