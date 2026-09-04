"""The success contract shared by both embodiments and the scorers.

This module implements ``CONTRACT.md`` section 5 — the only success definition
that counts — clause for clause, in the world frame:

1. horizontal distance apple centre to plate centre ``< 0.080`` m
2. apple centre ``z`` within ``0.040 +/- 0.015`` m
3. apple linear speed ``< 0.01`` m/s
4. all of the above held continuously for ``>= 1.0`` s of **simulated** time
5. 3-D displacement from the spawn point ``(0.30, 0.10, 0.020)`` ``> 0.25`` m

Clauses 1, 2, 3 and 5 are instantaneous and evaluated by
[`PlateGoal.is_placed`][robot_console.arm.success.PlateGoal.is_placed]. Clause 4 is
the *hold*, which spans steps: [`HoldTracker`][robot_console.arm.success.HoldTracker]
accumulates it live inside the embodiment and
[`final_hold`][robot_console.arm.success.final_hold] reconstructs the same quantity
offline from a recorded trajectory, so a scorer never needs the network.

The verdict itself is no longer computed here. It is computed from the overhead
camera by [`vision_success`][robot_console.arm.vision_success] and handed in, so
that what grades the episode is something the robot can see; the simulator used to
publish its own answer on ``/task_success`` and no longer does, because no camera
sees that topic and no real SO-101 publishes it.

What this module still owns is the *reference*: the same predicate evaluated on the
free-joint poses, recorded beside the camera's answer on every step and graded on by
nothing. Keeping it is the point rather than an oversight — a detector that has
quietly drifted is indistinguishable from a policy that got worse, and this column is
what tells those two apart afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: The task's instantaneous verdict, clauses 1, 3 and 5 as the overhead camera measures
#: them, with clause 2 enforced through the projection (see
#: [`vision_success`][robot_console.arm.vision_success]). This is the verdict that counts.
GEOMETRIC_SUCCESS_KEY = "apple_on_plate"
#: The same predicate computed from the free-joint poses instead of the camera. Nothing
#: grades on it. It is recorded so the camera verdict stays auditable against ground
#: truth: a detector that has silently drifted looks exactly like a policy that got worse,
#: and without this column there is no way to tell those apart after the fact.
REFERENCE_SUCCESS_KEY = "reference_apple_on_plate"
#: Pose-derived apple-to-plate distance, metres, beside the camera's own.
REFERENCE_DISTANCE_KEY = "reference_distance"
#: The apple's apparent radius over the radius it would have resting where it appears.
#: Recorded per step so the offline hold can apply the same airborne test the live one
#: does; see `vision_success.RESTING_RADIUS_RATIO`.
RADIUS_RATIO_KEY = "apple_radius_ratio"
#: Clause 4 satisfied as well — the full contract predicate (bool).
HELD_KEY = "apple_on_plate_held"
#: How long the instantaneous verdict has been continuously true, seconds.
HOLD_ELAPSED_KEY = "apple_on_plate_hold_s"
#: The apple's world position as ``[x, y, z]``, or None if unseen.
APPLE_POSITION_KEY = "apple_position"
#: The apple's linear speed in m/s, or None if the twist was not observed.
APPLE_SPEED_KEY = "apple_speed"
#: 3-D displacement from the contract spawn point, metres (inf if unseen).
DISPLACEMENT_KEY = "apple_displacement"
#: Simulated-time stamp of the apple-pose message this verdict was computed from.
#: ``None`` when no clock accompanied the pose; the hold then falls back to steps.
STAMP_KEY = "object_stamp_s"
#: Horizontal apple-to-plate distance in metres. Named ``distance`` so the core
#: ``min_distance_to_goal`` and ``reached_goal_state`` scorers work unmodified.
DISTANCE_KEY = "distance"

#: Clause 4. Simulated seconds the instantaneous predicate must hold.
HOLD_SECONDS = 1.0


@dataclass(frozen=True)
class PlateGoal:
    """The instantaneous half of ``CONTRACT.md`` section 5, in world coordinates.

    Every default is the contract's own number, and each is the same constant
    ``so_arm_mujoco/task_manager.py`` compares against. Nothing grades on this any
    more — the camera does — but it is what the recorded reference column is built
    from, so it has to keep matching the contract exactly or the audit it supports
    is worthless.

    The comparisons match the simulator's directions exactly: horizontal
    distance and speed are strict ``<``, the height band is inclusive
    (``abs(z - center_z) <= z_tolerance``), and displacement is strict ``>``.
    """

    #: Clause 1. Plate centre and the horizontal limit, plate radius 0.10 minus
    #: apple radius 0.020.
    center_xy: tuple[float, float] = (0.226, -0.226)
    radius: float = 0.080
    #: Clause 2. Plate top 0.020 plus apple radius 0.020, and the tolerance.
    center_z: float = 0.040
    z_tolerance: float = 0.015
    #: Clause 3. Above this speed the apple is passing through, not resting.
    max_speed: float = 0.01
    #: Clause 5. Spawn point and the travel the apple must show against it.
    spawn_xyz: tuple[float, float, float] = (0.30, 0.10, 0.020)
    min_displacement: float = 0.25

    @classmethod
    def for_apple(
        cls,
        *,
        apple_radius: float = 0.020,
        plate_top_z: float = 0.020,
        center_xy: tuple[float, float] = (0.226, -0.226),
        plate_radius: float = 0.10,
        spawn_xy: tuple[float, float] = (0.30, 0.10),
    ) -> "PlateGoal":
        """Re-derive the contract gate for a differently sized apple.

        Diagnostic only. With the contract's own numbers this returns exactly
        the defaults above; passing another radius rebuilds the same *formulae*
        the contract states (``plate_radius - apple_radius`` horizontally,
        ``plate_top_z + apple_radius`` vertically) so an offline run against a
        modified scene stays self-consistent. The reference must use the defaults,
        because the contract's own numbers are what it exists to represent.
        """
        return cls(
            center_xy=center_xy,
            radius=plate_radius - apple_radius,
            center_z=plate_top_z + apple_radius,
            spawn_xyz=(spawn_xy[0], spawn_xy[1], apple_radius),
        )

    def horizontal_distance(self, apple_xyz: Sequence[float]) -> float:
        """Return the xy distance from the apple to the plate centre."""
        return math.hypot(apple_xyz[0] - self.center_xy[0], apple_xyz[1] - self.center_xy[1])

    def displacement(self, apple_xyz: Sequence[float]) -> float:
        """Return the 3-D distance the apple has travelled from its spawn point."""
        return math.dist(tuple(apple_xyz[:3]), self.spawn_xyz)

    def is_placed(
        self, apple_xyz: Sequence[float] | None, speed: float | None = None
    ) -> bool:
        """Whether clauses 1, 2, 3 and 5 all hold for this single observation.

        An unobserved pose *or an unobserved speed* is not a placement. Clause 3
        requires the apple to be at rest, and a missing twist is no evidence
        that it is — treating it as passing would let a still-swinging apple
        score, which is exactly the failure mode the clause exists to catch.
        """
        if apple_xyz is None or speed is None:
            return False
        return (
            self.horizontal_distance(apple_xyz) < self.radius
            and abs(apple_xyz[2] - self.center_z) <= self.z_tolerance
            and speed < self.max_speed
            and self.displacement(apple_xyz) > self.min_displacement
        )

    def explain(self, apple_xyz: Sequence[float] | None, speed: float | None) -> str:
        """Return which clause fails first, in the simulator's own wording."""
        if apple_xyz is None:
            return "apple pose never observed"
        if speed is None:
            return "apple twist never observed (clause 3 needs a speed)"
        horizontal = self.horizontal_distance(apple_xyz)
        if horizontal >= self.radius:
            return f"horizontal {horizontal:.4f} >= {self.radius:.3f}"
        if abs(apple_xyz[2] - self.center_z) > self.z_tolerance:
            return f"z {apple_xyz[2]:.4f} outside {self.center_z:.3f} +/- {self.z_tolerance:.3f}"
        if speed >= self.max_speed:
            return f"speed {speed:.4f} >= {self.max_speed:.3f}"
        travelled = self.displacement(apple_xyz)
        if travelled <= self.min_displacement:
            return f"displacement {travelled:.4f} <= {self.min_displacement:.3f}"
        return "ok"


class HoldTracker:
    """Accumulate clause 4 — the continuous hold — across steps.

    Simulated time is preferred and is what the contract specifies; it comes
    from the stamp the free-joint plugin wrote on the pose message. When no
    stamp accompanies a pose the tracker degrades to counting steps at the
    declared control rate, which is coarser but never silently reports a hold
    that the timestamps would not support.
    """

    def __init__(self, hold_seconds: float = HOLD_SECONDS, *, control_hz: float = 10.0) -> None:
        if hold_seconds < 0:
            raise ValueError(f"hold_seconds must be non-negative, got {hold_seconds!r}")
        if not control_hz > 0:
            raise ValueError(f"control_hz must be > 0, got {control_hz!r}")
        self.hold_seconds = float(hold_seconds)
        self.control_hz = float(control_hz)
        #: Steps required when no simulated clock is available.
        self.fallback_steps = max(1, math.ceil(self.hold_seconds * self.control_hz))
        self.reset()

    def reset(self) -> None:
        """Forget any hold in progress. Called on every episode reset."""
        self._since: float | None = None
        self._steps = 0
        self.elapsed = 0.0

    def update(self, *, placed: bool, stamp: float | None) -> bool:
        """Fold one step in and return whether the hold is now satisfied."""
        if not placed:
            self._since = None
            self._steps = 0
            self.elapsed = 0.0
            return False
        self._steps += 1
        if stamp is None:
            # No clock: approximate the elapsed time from the control period.
            self.elapsed = (self._steps - 1) / self.control_hz
            return self._steps >= self.fallback_steps
        if self._since is None:
            self._since = stamp
        self.elapsed = max(0.0, stamp - self._since)
        return self.elapsed >= self.hold_seconds


def success_info(
    apple_xyz: Sequence[float] | None,
    *,
    goal: PlateGoal,
    placed: bool = False,
    distance: float | None = None,
    apple_speed: float | None = None,
    stamp: float | None = None,
    radius_ratio: float | None = None,
) -> dict[str, Any]:
    """Build the per-step ``StepResult.info`` payload both embodiments publish.

    Everything here is measured state or a function of measured state. The hold
    (clause 4) is *not* decided here — it belongs to the caller's
    [`HoldTracker`][robot_console.arm.success.HoldTracker], because it depends on
    the steps before this one.
    """
    return {
        APPLE_POSITION_KEY: None if apple_xyz is None else [float(v) for v in apple_xyz],
        APPLE_SPEED_KEY: None if apple_speed is None else float(apple_speed),
        # `distance` is the camera's, and is what the core distance scorers read.
        DISTANCE_KEY: float("inf") if distance is None else float(distance),
        DISPLACEMENT_KEY: (
            float("inf") if apple_xyz is None else float(goal.displacement(apple_xyz))
        ),
        GEOMETRIC_SUCCESS_KEY: bool(placed),
        REFERENCE_SUCCESS_KEY: goal.is_placed(apple_xyz, apple_speed),
        REFERENCE_DISTANCE_KEY: (
            None if apple_xyz is None else float(goal.horizontal_distance(apple_xyz))
        ),
        STAMP_KEY: None if stamp is None else float(stamp),
        RADIUS_RATIO_KEY: None if radius_ratio is None else float(radius_ratio),
    }


@dataclass(frozen=True)
class HoldSpan:
    """The trailing run of instantaneously-placed steps in a recorded trial."""

    steps: int
    #: Simulated seconds the run covers, or None when the steps carried no stamp.
    seconds: float | None

    def satisfied(self, *, hold_seconds: float = HOLD_SECONDS, fallback_steps: int = 3) -> bool:
        """Whether the run meets clause 4, by time when timed and by count when not."""
        if self.steps == 0:
            return False
        if self.seconds is None:
            return self.steps >= fallback_steps
        return self.seconds >= hold_seconds


def final_hold(infos: Sequence[Mapping[str, Any]]) -> HoldSpan:
    """Measure the trailing run of placed steps in a recorded trajectory.

    This is the offline half of clause 4 and it is deliberately independent of
    anything the embodiment wrote about the hold: it re-derives the run from the
    per-step geometric verdicts and their simulated stamps. A scorer reading a
    saved log therefore reaches the same answer the live run did, without
    trusting a boolean somebody else computed.
    """
    run: list[Mapping[str, Any]] = []
    for info in reversed(infos):
        if not bool(info.get(GEOMETRIC_SUCCESS_KEY)):
            break
        run.append(info)
    if not run:
        return HoldSpan(steps=0, seconds=None)
    run.reverse()
    # No height test here, matching the live verdict: clause 2 is not enforced from the
    # camera. `RADIUS_RATIO_KEY` is recorded on every step so an airborne placement can be
    # recognised after the fact, but it does not decide anything -- the two populations
    # overlap, and `vision_success.APPLE_RADIUS_M` carries the measurements that show it.
    first, last = run[0].get(STAMP_KEY), run[-1].get(STAMP_KEY)
    if first is None or last is None:
        return HoldSpan(steps=len(run), seconds=None)
    return HoldSpan(steps=len(run), seconds=max(0.0, float(last) - float(first)))


def any_step_succeeded(infos: Sequence[Mapping[str, Any]], key: str) -> bool:
    """Whether any recorded step reported a true verdict under ``key``."""
    return any(bool(info.get(key)) for info in infos)
