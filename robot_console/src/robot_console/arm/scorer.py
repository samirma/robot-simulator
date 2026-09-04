"""Scorers for the apple-on-plate task.

All three are pure readers of the recorded trajectory, per the framework's rule
that scoring must be reproducible from a saved log: the live polling of
the overhead frame and of the apple's pose happens in the embodiment's ``step``,
which writes its measurements into
[`StepResult`][inspect_robots.types.StepResult] ``info``.

``apple_on_plate`` re-derives ``CONTRACT.md`` section 5 clause 4 — the >= 1.0 s
hold — from those per-step measurements rather than reading back any boolean
the embodiment wrote about the hold. The live run and the offline scorer
therefore agree by construction, and a disagreement is a real signal rather
than a copied verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from inspect_robots.scene import Target
from inspect_robots.scorer import Score, Scorer

from robot_console.arm.success import (
    APPLE_SPEED_KEY,
    DISPLACEMENT_KEY,
    DISTANCE_KEY,
    GEOMETRIC_SUCCESS_KEY,
    HOLD_SECONDS,
    REFERENCE_SUCCESS_KEY,
    final_hold,
)

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord


def _infos(record: TrialRecord) -> list[Mapping[str, Any]]:
    return [step.result.info for step in record.steps]


@dataclass(frozen=True)
class _AppleOnPlate:
    """Success iff the apple ends the episode resting on the plate for >= 1.0 s.

    The trailing run of instantaneously-placed steps is measured in **simulated
    seconds** whenever the recorded steps carry the free-joint plugin's stamp,
    which is what the contract specifies and what the simulator's own
    ``task_manager`` times against. ``hold_steps`` is only the fallback for a
    trajectory recorded without stamps.

    Requiring a run at the *end* rather than "ever true" rejects an apple that
    was momentarily over the plate while still in the jaws, or that bounced
    across the plate and rolled off.
    """

    hold_seconds: float = HOLD_SECONDS
    hold_steps: int = 3
    name: str = "apple_on_plate"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        infos = _infos(record)
        if not infos:
            return Score(value=False, explanation="no steps recorded")
        span = final_hold(infos)
        held = span.satisfied(hold_seconds=self.hold_seconds, fallback_steps=self.hold_steps)
        ever = any(bool(info.get(GEOMETRIC_SUCCESS_KEY)) for info in infos)
        ref_ever = any(bool(info.get(REFERENCE_SUCCESS_KEY)) for info in infos)
        ref_seen = any(info.get(REFERENCE_SUCCESS_KEY) is not None for info in infos)
        final = infos[-1]
        timing = (
            f"{span.seconds:.3f} s of simulated time"
            if span.seconds is not None
            else f"{span.steps} steps (untimed; needed {self.hold_steps})"
        )
        explanation = (
            f"final apple position {final.get('apple_position')}, "
            f"plate distance {final.get(DISTANCE_KEY)!r} m, "
            f"speed {final.get(APPLE_SPEED_KEY)!r} m/s, "
            f"travel {final.get(DISPLACEMENT_KEY)!r} m; "
            f"held over the last {span.steps} step(s) = {timing}, "
            f"needed >= {self.hold_seconds:g} s; "
            f"placed_and_held={held} ever_placed={ever}; "
            + (
                f"pose reference ever_placed={ref_ever}"
                if ref_seen
                else "pose reference never observed"
            )
        )
        return Score(
            value=held,
            explanation=explanation,
            metadata={
                "ever_placed": ever,
                "hold_steps": span.steps,
                "hold_seconds": span.seconds,
                "reference_ever_placed": ref_ever if ref_seen else None,
                "final_distance_m": final.get(DISTANCE_KEY),
                "final_speed_mps": final.get(APPLE_SPEED_KEY),
                "final_displacement_m": final.get(DISPLACEMENT_KEY),
                "final_apple_position": final.get("apple_position"),
            },
        )


@dataclass(frozen=True)
class _ReferenceSuccess:
    """The same predicate on the free-joint poses, for auditing the camera.

    This does not grade the episode -- `apple_on_plate_success` does, from the overhead
    frame. This is the column you compare it against. A vision detector that has drifted
    (a re-styled kitchen, a differently lit scene, a plate that is no longer white)
    produces exactly the same run of failures as a policy that stopped working, and
    without a verdict computed a different way there is nothing to tell them apart.
    Disagreement here is a reason to look at the detector, not to trust this number over
    the camera's.
    """

    name: str = "reference_success"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        infos = _infos(record)
        seen = [info for info in infos if info.get(REFERENCE_SUCCESS_KEY) is not None]
        if not seen:
            return Score(value=False, explanation="no free-joint poses were observed")
        succeeded = any(bool(info.get(REFERENCE_SUCCESS_KEY)) for info in seen)
        graded = any(bool(info.get(GEOMETRIC_SUCCESS_KEY)) for info in infos)
        note = "" if succeeded == graded else "  *** disagrees with the camera verdict ***"
        return Score(
            value=succeeded,
            explanation=(
                f"poses seen on {len(seen)}/{len(infos)} steps, predicate true on "
                f"{sum(1 for i in seen if i.get(REFERENCE_SUCCESS_KEY))}{note}"
            ),
        )


@dataclass(frozen=True)
class _ApplePlateDistance:
    """Closest the apple got to the plate centre, horizontally, in metres."""

    name: str = "apple_plate_distance"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        distances = [
            float(info[DISTANCE_KEY]) for info in _infos(record) if DISTANCE_KEY in info.keys()
        ]
        finite = [value for value in distances if value != float("inf")]
        if not finite:
            return Score(value=float("inf"), explanation="apple pose never observed")
        return Score(
            value=min(finite),
            explanation=f"closest of {len(finite)} observations, final {distances[-1]:.4f} m",
        )


def apple_on_plate_success(
    hold_seconds: float = HOLD_SECONDS, hold_steps: int = 3
) -> Scorer:
    """Contract section 5 success: the apple rests on the plate for >= 1.0 s."""
    return _AppleOnPlate(hold_seconds=float(hold_seconds), hold_steps=int(hold_steps))


def reference_success() -> Scorer:
    """The pose-derived verdict, recorded so the camera's can be audited against it."""
    return _ReferenceSuccess()


def apple_plate_distance() -> Scorer:
    """Minimum horizontal apple-to-plate distance (lower is better)."""
    return _ApplePlateDistance()
