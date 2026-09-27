"""Deciding where to explore next, and when the map is finished (console spec §2.2).

Split out of `app.py` because it is the part with the interesting behaviour and none of
the reasons `app.py` is hard to test: no window, no socket, no clock. Everything here
takes `now` as an argument.

**Goals.** `explore` chases frontiers. A goal *progresses* when the robot comes
`GOAL_PROGRESS_M` (0.45 m) closer to it than its closest approach at the last progress,
or when `GOAL_PROGRESS_AREA_M2` (0.25 m²) of map is uncovered since that progress. A goal
is blacklisted after `GOAL_STALL_S` (90 s) without progress or `GOAL_TOTAL_S` (300 s) in
total, and when it is reached without uncovering any map. A goal reached without
uncovering any map is never chosen again; the ones that timed out may be, once, after
rung 3 clears them.

**The give-up ladder.** When no frontier qualifies, each rung is tried only if the one
before found nothing:

    1  frontiers >= 6 cells
    2  frontiers >= 3 cells
    3  once per run: clear timed-out blacklist entries, then frontiers >= 6 cells
    4  frontiers >= 1 cell
    5  unknown holes fully enclosed by observed cells
    6  once per run: one rotation on the spot, then back to rung 1

A goal from any rung restarts the ladder at rung 1. The run ends `explored` when rung 5
finds nothing and rung 6 has already run; `limit` (the app's, and `max_goals` here) is
the other ending. The per-goal deadline plus the two hard run limits bound the run.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import List, Optional, Sequence

import numpy as np

from robot_console.slam import frontier as frontier_mod
from robot_console.slam.frontier import BLACKLIST_RADIUS_M, Frontier
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import CostMap, distance_field, plan

# How far the robot may be from the last planned-against pose before the path is stale.
REPLAN_DISTANCE_M = 0.6
REPLAN_SECONDS = 3.0

# A freshly chosen goal is not reconsidered for this long, so two comparable frontiers
# do not trade places as the robot moves.
COMMIT_SECONDS = 2.0

# --- console spec §2.2 ---------------------------------------------------------------
GOAL_PROGRESS_M = 0.45
GOAL_PROGRESS_AREA_M2 = 0.25
GOAL_STALL_S = 90.0
GOAL_TOTAL_S = 300.0
MIN_CELLS_LARGE = 6
MIN_CELLS_SMALL = 3
MIN_CELLS_ANY = 1

# Rung 6: one full rotation on the spot.
SWEEP_RATE = 0.5
SWEEP_SECONDS = 2.0 * math.pi / SWEEP_RATE

DEFAULT_MAX_GOALS = 500

EXPLORED = "explored"
LIMIT = "limit"


class Rung(enum.IntEnum):
    LARGE = 1
    SMALL = 2
    FLUSH = 3
    ANY = 4
    HOLES = 5
    SWEEP = 6


class GoalBlacklist:
    """Goals ruled out, and why: `timeout` (rung 3 may clear) or `fruitless` (never)."""

    TIMEOUT = "timeout"
    FRUITLESS = "fruitless"

    def __init__(self, radius: float = BLACKLIST_RADIUS_M) -> None:
        self.radius = float(radius)
        self.entries: List[tuple] = []  # (point, kind)

    def add(self, points: Sequence[Sequence[float]], kind: str) -> None:
        for point in points:
            if point is not None:
                self.entries.append((np.asarray(point, dtype=np.float64)[:2].copy(), kind))

    def blocks(self, point: Sequence[float], _now: float = 0.0) -> bool:
        target = np.asarray(point, dtype=np.float64)[:2]
        return any(float(np.hypot(*(p - target))) < self.radius for p, _ in self.entries)

    def clear_timeouts(self) -> int:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e[1] != self.TIMEOUT]
        return before - len(self.entries)

    def count(self, kind: str) -> int:
        return sum(1 for _, k in self.entries if k == kind)

    def expire(self, _now: float) -> None:
        """Nothing expires: entries leave only through rung 3."""

    def __len__(self) -> int:
        return len(self.entries)


@dataclasses.dataclass
class GoalRecord:
    """One goal's history, kept across re-selection so its deadlines are cumulative."""

    point: np.ndarray
    centroid: np.ndarray
    rung: int
    active_s: float = 0.0
    since_progress_s: float = 0.0
    anchor_distance: float = math.inf
    closest: float = math.inf
    anchor_area: float = 0.0
    area_at_selection: float = 0.0
    last_seen: Optional[float] = None
    progress_events: int = 0

    def matches(self, point: Sequence[float], centroid: Sequence[float], radius: float) -> bool:
        p = np.asarray(point, dtype=np.float64)[:2]
        c = np.asarray(centroid, dtype=np.float64)[:2]
        return (float(np.hypot(*(self.point - p))) < radius
                or float(np.hypot(*(self.centroid - c))) < radius)


@dataclasses.dataclass(frozen=True)
class Decision:
    """What the explorer wants to happen next. `app.py` turns this into a `Command`."""

    goal: Optional[np.ndarray] = None
    path: Optional[np.ndarray] = None
    frontier: Optional[Frontier] = None
    frontiers: List[Frontier] = dataclasses.field(default_factory=list)
    note: str = ""
    spin: float = 0.0
    finished: Optional[str] = None
    rung: Optional[int] = None

    @property
    def driving(self) -> bool:
        return self.path is not None


class Explorer:
    def __init__(
        self,
        *,
        standoff: float = frontier_mod.FRONTIER_STANDOFF_M,
        distance_bias: float = frontier_mod.DISTANCE_BIAS_M,
        blacklist: Optional[GoalBlacklist] = None,
        commit: float = COMMIT_SECONDS,
        sweep_seconds: float = SWEEP_SECONDS,
        max_goals: int = DEFAULT_MAX_GOALS,
        stall_seconds: float = GOAL_STALL_S,
        total_seconds: float = GOAL_TOTAL_S,
    ) -> None:
        self.standoff = float(standoff)
        self.distance_bias = float(distance_bias)
        self.blacklist = blacklist if blacklist is not None else GoalBlacklist()
        self.commit = float(commit)
        self.sweep_seconds = float(sweep_seconds)
        self.max_goals = int(max_goals)
        self.stall_seconds = float(stall_seconds)
        self.total_seconds = float(total_seconds)

        self.goal: Optional[np.ndarray] = None
        self.path: Optional[np.ndarray] = None
        self.frontiers: List[Frontier] = []
        self.records: List[GoalRecord] = []
        self.current: Optional[GoalRecord] = None
        self.attempted = 0
        self.rung_log: List[int] = []   # the rung each accepted goal came from
        self.flushed = False
        self.swept = False
        self._sweep_until: Optional[float] = None
        self._planned_at: Optional[np.ndarray] = None
        self._planned_when = 0.0
        self._chosen_at = -math.inf
        self.finished: Optional[str] = None

    # ------------------------------------------------------------------ state

    def needs_replan(self, pose: Sequence[float], now: float) -> bool:
        if self.path is None or self._planned_at is None:
            return True
        if now - self._chosen_at < self.commit:
            return False
        if now - self._planned_when >= REPLAN_SECONDS:
            return True
        here = np.asarray(pose[:2], dtype=np.float64)
        return float(np.hypot(*(here - self._planned_at))) >= REPLAN_DISTANCE_M

    def sweeping(self, now: float) -> bool:
        return self._sweep_until is not None and now < self._sweep_until

    # ------------------------------------------------------------------ progress

    def track(self, grid: OccupancyGrid, pose: Sequence[float], now: float,
              area: Optional[float] = None) -> Optional[str]:
        """Account the current goal's time and progress. Call every tick.

        Returns "timeout" when the goal has just been blacklisted for 90 s without
        progress or 300 s in total (and dropped), else None.
        """
        rec = self.current
        if rec is None:
            return None
        if area is None:
            area = frontier_mod.explored_area(grid)
        if rec.last_seen is not None:
            dt = max(0.0, now - rec.last_seen)
            rec.active_s += dt
            rec.since_progress_s += dt
        rec.last_seen = now
        distance = float(np.hypot(*(np.asarray(pose[:2], dtype=np.float64) - rec.point)))
        rec.closest = min(rec.closest, distance)
        if distance <= rec.anchor_distance - GOAL_PROGRESS_M:
            self._progress(rec, area)
        elif area - rec.anchor_area >= GOAL_PROGRESS_AREA_M2:
            self._progress(rec, area)
        if rec.since_progress_s > self.stall_seconds or rec.active_s > self.total_seconds:
            self.blacklist.add([rec.point, rec.centroid], GoalBlacklist.TIMEOUT)
            self._drop()
            return "timeout"
        return None

    @staticmethod
    def _progress(rec: GoalRecord, area: float) -> None:
        rec.anchor_distance = rec.closest
        rec.anchor_area = area
        rec.since_progress_s = 0.0
        rec.progress_events += 1

    def _drop(self) -> None:
        if self.current is not None:
            self.current.last_seen = None
        self.current = None
        self.goal = None
        self.path = None
        self._planned_when = 0.0
        self._chosen_at = -math.inf

    # ------------------------------------------------------------------ events

    def on_arrived(self, grid: OccupancyGrid, now: float, area: Optional[float] = None) -> bool:
        """The goal was reached. Returns True if it was fruitless (and so blacklisted)."""
        rec = self.current
        fruitless = False
        if rec is not None:
            if area is None:
                area = frontier_mod.explored_area(grid)
            if area <= rec.area_at_selection + 1e-9:
                self.blacklist.add([rec.point, rec.centroid], GoalBlacklist.FRUITLESS)
                fruitless = True
        self._drop()
        return fruitless

    def on_blocked(self, now: float) -> None:
        """A local obstruction: redraw the route, keep the goal."""
        self.path = None
        self._planned_when = 0.0
        self._chosen_at = -math.inf

    on_stuck = on_blocked

    # ------------------------------------------------------------------ decision

    def replan(
        self, grid: OccupancyGrid, cost: CostMap, pose: Sequence[float], now: float,
        area: Optional[float] = None,
    ) -> Decision:
        if self.finished is not None:
            return Decision(finished=self.finished, frontiers=self.frontiers)
        origin = np.asarray(pose[:2], dtype=np.float64)
        self._planned_at = origin.copy()
        self._planned_when = now
        if area is None:
            area = frontier_mod.explored_area(grid)
        field = distance_field(cost, origin)

        for rung in Rung:
            if rung is Rung.FLUSH:
                if self.flushed:
                    continue
                self.flushed = True
                self.blacklist.clear_timeouts()
            if rung is Rung.SWEEP:
                if self.swept:
                    break
                self.swept = True
                self._sweep_until = now + self.sweep_seconds
                self._drop()
                return Decision(spin=SWEEP_RATE, note="one look round (rung 6)",
                                frontiers=self.frontiers, rung=int(rung))
            decision = self._try(rung, grid, cost, origin, field, now, area)
            if decision is not None:
                return decision
        return self._stop(EXPLORED, "no frontiers left")

    def _try(self, rung, grid, cost, origin, field, now, area) -> Optional[Decision]:
        if rung is Rung.HOLES:
            for pocket in frontier_mod.unknown_pockets(grid):
                if self.blacklist.blocks(pocket.centroid):
                    continue
                path = plan(cost, origin, pocket.centroid)
                if path is None:
                    continue
                self.frontiers = [pocket]
                return self._accept(pocket, path, now, rung, area, "closing a hole (rung 5)")
            return None
        min_cells = {Rung.LARGE: MIN_CELLS_LARGE, Rung.SMALL: MIN_CELLS_SMALL,
                     Rung.FLUSH: MIN_CELLS_LARGE, Rung.ANY: MIN_CELLS_ANY}[rung]
        choice = frontier_mod.survey(
            grid, cost, origin,
            blacklist=self.blacklist, min_cells=min_cells, field=field,
            standoff=self.standoff, distance_bias=self.distance_bias,
            incumbent=self.goal, now=now,
        )
        self.frontiers = choice.frontiers or self.frontiers
        if not choice.found:
            return None
        if self.blacklist.blocks(choice.frontier.target):
            return None
        note = "exploring" if rung is Rung.LARGE else f"looking harder (rung {int(rung)})"
        return self._accept(choice.frontier, choice.path, now, rung, area, note)

    def _accept(self, target: Frontier, path, now, rung, area, note) -> Decision:
        point = np.asarray(target.target, dtype=np.float64)[:2]
        centroid = np.asarray(target.centroid, dtype=np.float64)[:2]
        rec = self.current if (self.current is not None and self.current.matches(
            point, centroid, BLACKLIST_RADIUS_M)) else None
        if rec is None:
            rec = next((r for r in self.records
                        if r.matches(point, centroid, BLACKLIST_RADIUS_M)), None)
            if rec is None:
                if self.attempted >= self.max_goals:
                    return self._stop(LIMIT, f"{self.attempted} goals attempted")
                rec = GoalRecord(point=point.copy(), centroid=centroid.copy(), rung=int(rung))
                self.records.append(rec)
                self.attempted += 1
            if self.current is not None:
                self.current.last_seen = None
            rec.anchor_distance = min(rec.closest, float(np.hypot(*(point - np.asarray(
                self._planned_at if self._planned_at is not None else point)))))
            rec.anchor_area = area
            rec.area_at_selection = area
            rec.last_seen = now
            self.current = rec
            self.rung_log.append(int(rung))
        rec.point = point.copy()
        self.goal = point
        self.path = path
        self._chosen_at = now
        return Decision(goal=self.goal, path=path, frontier=target,
                        frontiers=self.frontiers, note=note, rung=int(rung))

    def _stop(self, reason: str, note: str) -> Decision:
        self._drop()
        self.finished = reason
        return Decision(note=note, finished=reason, frontiers=self.frontiers)

    def stop_for_limit(self, note: str) -> Decision:
        return self._stop(LIMIT, note)
