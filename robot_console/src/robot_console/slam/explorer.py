"""Deciding where to explore next, and when to admit the map is finished.

Split out of `app.py` because it is the part with the interesting behaviour and none of
the reasons `app.py` is hard to test: no window, no socket, no clock. Everything here
takes `now` as an argument.

The thing this module exists to get right is **giving up**. Frontier exploration is
"complete by construction" only if running out of frontiers really means the map is done,
and in practice it usually does not: the threshold that filters sensor speckle also hides
a doorway seen edge-on, a goal suppressed after one bad approach may be a room's only
entrance, and weakly-observed cells leave holes that raise frontiers too small to chase.
So instead of one give-up test there is a ladder, and each rung has to come back empty
before the next is tried:

    6 cells -> 3 cells -> flush the blacklist and retry at 6 -> 1 cell
            -> enclosed sensor holes -> one sweep on the spot -> finished

The blacklist flush deliberately sits *before* the one-cell rung: a suppressed 270-cell
room is far likelier to be genuinely reachable than a one-cell frontier is to be real, so
the cheap correct retry should come first.

All of it runs off a single `planner.distance_field`, which is what makes the ladder
affordable -- each extra rung is a scoring pass over an existing wavefront, not another
search.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import List, Optional, Sequence

import numpy as np

from robot_console.slam import frontier as frontier_mod
from robot_console.slam.frontier import Blacklist, Frontier
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.planner import CostMap, distance_field, plan

# How far the robot may be from the last planned-against pose before the path is stale.
REPLAN_DISTANCE_M = 0.6
REPLAN_SECONDS = 3.0

# A freshly chosen goal is not reconsidered for this long. Without it the full
# re-selection every 0.6 m lets two comparable frontiers trade places as the robot moves,
# and it dithers between them instead of finishing either.
COMMIT_SECONDS = 2.0

# One rotation on the spot, as a last look before declaring the map done. The X2 is a 360
# degree lidar so this reveals little geometry directly; what it buys is time for the map
# to settle after the thresholds have been relaxed. Capped at one per stage entry so it
# cannot become a spin loop.
SWEEP_SECONDS = 6.0
SWEEP_RATE = 0.6

# Progress watchdog. Preferred over a wall-clock `--timeout` because any timeout long
# enough for a real house is too long to bound a test, and any timeout short enough for a
# test truncates a working run. Growth in mapped area is the thing actually worth watching.
STALL_SECONDS = 90.0
STALL_AREA_M2 = 0.25


class Stage(enum.IntEnum):
    NORMAL = 0
    SMALL = 1
    FLUSH = 2
    TINY = 3
    POCKETS = 4
    SWEEP = 5
    DONE = 6


# (stage, min_cells) for the rungs that are ordinary frontier searches.
_FRONTIER_RUNGS = {
    Stage.NORMAL: None,   # None -> the explorer's configured min_cells
    Stage.SMALL: 3,
    Stage.FLUSH: None,
    Stage.TINY: 1,
}


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

    @property
    def driving(self) -> bool:
        return self.path is not None


class Explorer:
    def __init__(
        self,
        *,
        min_cells: int = frontier_mod.MIN_FRONTIER_CELLS,
        standoff: float = frontier_mod.FRONTIER_STANDOFF_M,
        distance_bias: float = frontier_mod.DISTANCE_BIAS_M,
        blacklist: Optional[Blacklist] = None,
        commit: float = COMMIT_SECONDS,
        sweep_seconds: float = SWEEP_SECONDS,
        stall_seconds: float = STALL_SECONDS,
    ) -> None:
        self.min_cells = int(min_cells)
        self.standoff = float(standoff)
        self.distance_bias = float(distance_bias)
        self.blacklist = blacklist if blacklist is not None else Blacklist()
        self.commit = float(commit)
        self.sweep_seconds = float(sweep_seconds)
        self.stall_seconds = float(stall_seconds)

        self.goal: Optional[np.ndarray] = None
        self.path: Optional[np.ndarray] = None
        self.frontiers: List[Frontier] = []
        self._stage = Stage.NORMAL
        self._flushed = False
        self._swept = False
        self._sweep_until: Optional[float] = None
        self._planned_at: Optional[np.ndarray] = None
        self._planned_when = 0.0
        self._chosen_at = -math.inf
        self._best_area = -math.inf
        self._best_area_at: Optional[float] = None
        self._retired = 0
        self._retired_seen = -1

    # ------------------------------------------------------------------ state

    @property
    def stage(self) -> Stage:
        return self._stage

    def reset(self) -> None:
        self.goal = None
        self.path = None
        self._planned_at = None
        self._planned_when = 0.0
        self._chosen_at = -math.inf
        self._stage = Stage.NORMAL

    def needs_replan(self, pose: Sequence[float], now: float) -> bool:
        if self.path is None or self._planned_at is None:
            return True
        # Inside the commitment window the current goal is kept even if something else
        # would now score better -- that is the whole point of it.
        if now - self._chosen_at < self.commit:
            return False
        if now - self._planned_when >= REPLAN_SECONDS:
            return True
        here = np.asarray(pose[:2], dtype=np.float64)
        return float(np.hypot(*(here - self._planned_at))) >= REPLAN_DISTANCE_M

    def sweeping(self, now: float) -> bool:
        return self._sweep_until is not None and now < self._sweep_until

    # ------------------------------------------------------------------ events

    def on_arrived(self, now: float) -> None:
        """Standing on a frontier is as far as that frontier can take us.

        Retiring it looks aggressive and is not. Normally arriving *resolves* the frontier
        -- the lidar sees past it, the cells classify, the cluster disappears -- so the
        strike lands on a point that will never be a candidate again and costs nothing.
        It matters in the case that does not resolve: a frontier against an outer wall,
        where the unknown space behind it can never be observed from anywhere. Without
        this the robot drives to the wall, arrives, finds the frontier still there, and
        goes back, forever. The retired radius is only a quarter of a metre, so a long
        frontier keeps its other approach points and the robot works along it.
        """
        if self.goal is not None:
            self.blacklist.suppress(self.goal, now)
            self._retired += 1
        self.path = None
        self._planned_when = 0.0
        self._chosen_at = -math.inf

    def on_blocked(self, now: float) -> None:
        """A local obstruction. Redraw the route, keep the goal.

        Suppressing the goal here would burn through every frontier in the house in a
        couple of seconds while the robot never moved.
        """
        self.path = None
        self._planned_when = 0.0
        self._chosen_at = -math.inf

    def on_stuck(self, now: float) -> int:
        """Real failure: time passed and the robot got no closer. One strike."""
        strikes = 0
        if self.goal is not None:
            strikes = self.blacklist.strike(self.goal, now)
        self.goal = None
        self.path = None
        self._planned_when = 0.0
        self._chosen_at = -math.inf
        return strikes

    # ------------------------------------------------------------------ decision

    def replan(
        self, grid: OccupancyGrid, cost: CostMap, pose: Sequence[float], now: float
    ) -> Decision:
        origin = np.asarray(pose[:2], dtype=np.float64)
        self.blacklist.expire(now)
        self._planned_at = origin.copy()
        self._planned_when = now

        if self._stalled(grid, now):
            return self._stop("stalled", "no new ground in %.0fs" % self.stall_seconds)

        # One wavefront, reused by every rung below.
        field = distance_field(cost, origin)

        while self._stage is not Stage.DONE:
            decision = self._try_stage(grid, cost, origin, field, now)
            if decision is not None:
                return decision
            self._stage = Stage(self._stage + 1)

        return self._stop("explored", "no frontiers left")

    def _try_stage(self, grid, cost, origin, field, now) -> Optional[Decision]:
        stage = self._stage
        if stage in _FRONTIER_RUNGS:
            if stage is Stage.FLUSH:
                if self._flushed:
                    return None
                self._flushed = True
                self.blacklist.clear()
            min_cells = _FRONTIER_RUNGS[stage] or self.min_cells
            return self._try_frontiers(grid, cost, origin, field, now, min_cells)
        if stage is Stage.POCKETS:
            return self._try_pockets(grid, cost, origin, now)
        if stage is Stage.SWEEP:
            return self._try_sweep(now)
        return None

    def _try_frontiers(self, grid, cost, origin, field, now, min_cells) -> Optional[Decision]:
        choice = frontier_mod.survey(
            grid, cost, origin,
            blacklist=self.blacklist, min_cells=min_cells, field=field,
            standoff=self.standoff, distance_bias=self.distance_bias,
            incumbent=self.goal, now=now,
        )
        self.frontiers = choice.frontiers or self.frontiers
        if not choice.found:
            return None
        return self._accept(choice.frontier, choice.path, now, min_cells)

    def _try_pockets(self, grid, cost, origin, now) -> Optional[Decision]:
        """Enclosed unknown regions: holes the sensor left, not rooms it never entered."""
        for pocket in frontier_mod.unknown_pockets(grid):
            if self.blacklist.blocks(pocket.centroid, now):
                continue
            path = plan(cost, origin, pocket.centroid)
            if path is None:
                continue
            # Retired on selection, not on arrival. A hole is worth one look: rays leave
            # the robot radially, so a cell directly beneath it gets barely any crossings
            # and driving onto a pocket frequently does not resolve it. Waiting for an
            # arrival that may never come leaves the robot cycling between the last few
            # holes until the stall guard stops it -- and this phase has to be finite,
            # because "finished" is on the other side of it.
            self.blacklist.suppress(pocket.centroid, now)
            self.frontiers = [pocket]
            return self._accept(pocket, path, now, note="closing a gap in the map")
        return None

    def _try_sweep(self, now: float) -> Optional[Decision]:
        if self._swept:
            return None
        if self._sweep_until is None:
            self._sweep_until = now + self.sweep_seconds
        if now < self._sweep_until:
            return Decision(spin=SWEEP_RATE, note="last look round", frontiers=self.frontiers)
        # The sweep is over; let the map settle by restarting the ladder once.
        self._swept = True
        self._sweep_until = None
        self._stage = Stage.NORMAL
        return Decision(note="sweep done", frontiers=self.frontiers)

    def _accept(self, target, path, now, min_cells=None, note=None) -> Decision:
        self.goal = np.asarray(target.target, dtype=np.float64)[:2]
        self.path = path
        self._chosen_at = now
        if note is None:
            note = (
                "exploring"
                if min_cells is None or min_cells >= self.min_cells
                else "looking harder (>=%d cells)" % min_cells
            )
        # Any rung that produces a goal proves the map is not finished, so the ladder
        # starts again from the top next time.
        self._stage = Stage.NORMAL
        return Decision(
            goal=self.goal, path=path, frontier=target,
            frontiers=self.frontiers, note=note,
        )

    def _stop(self, reason: str, note: str) -> Decision:
        self.goal = None
        self.path = None
        self._stage = Stage.DONE
        return Decision(note=note, finished=reason, frontiers=self.frontiers)

    def _stalled(self, grid: OccupancyGrid, now: float) -> bool:
        """Livelock guard: no new ground *and* no goal seen through, for a long time.

        Retiring a goal counts as progress even when the map does not grow, because it is
        monotonic and bounded -- every arrival permanently rules a point out, so a run
        doing nothing but that still terminates. Without it the guard fires on the endgame
        it is meant to protect: closing the last few sensor holes maps almost no new area
        by definition, and a house-sized run was being cut off part-way through.
        """
        area = frontier_mod.explored_area(grid)
        progressed = (
            self._best_area_at is None
            or area > self._best_area + STALL_AREA_M2
            or self._retired != self._retired_seen
        )
        if progressed:
            self._best_area = max(self._best_area, area)
            self._best_area_at = now
            self._retired_seen = self._retired
            return False
        return (now - self._best_area_at) > self.stall_seconds
