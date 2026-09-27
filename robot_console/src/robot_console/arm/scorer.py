"""The camera-verdict scorer, ``apple_on_plate`` -- the task's only scorer.

A pure reader of the recorded trajectory: the embodiment records every synchronized rig
pair with its calibration and joint state in each step's ``info["rig_samples"]``, and
[`vision_success.assess`][robot_console.arm.vision_success.assess] applies the pass
criterion to them. Nothing it reads is simulator-private; nothing on the wire answers the
question for it.

The framework's JSON log keeps a score's value and not its explanation, so the verdict's
reason and measurements are also written to the trial's metadata (``trial_metadata`` in
the log), which is where `verdict.py` reads the one-line detail from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from inspect_robots.scene import Target
from inspect_robots.scorer import Score, Scorer

from robot_console.arm.vision_success import RIG_SAMPLES_KEY, assess

if TYPE_CHECKING:
    from inspect_robots.rollout import TrialRecord

#: Where the verdict's detail lands in the trial metadata.
METADATA_KEY = "apple_on_plate"


def rig_samples(record: TrialRecord) -> list[Any]:
    """Every synchronized sample the embodiment recorded, in step order."""
    out: list[Any] = []
    for step in record.steps:
        info = getattr(step.result, "info", None) or {}
        out.extend(info.get(RIG_SAMPLES_KEY) or ())
    return out


@dataclass(frozen=True)
class _AppleOnPlate:
    name: str = "apple_on_plate"

    def __call__(self, record: TrialRecord, target: Target | None) -> Score:
        del target  # the verdict is measured, not read off the scene's constants
        verdict = assess(rig_samples(record))
        detail = verdict.as_dict()
        try:
            record.metadata[METADATA_KEY] = detail
        except (AttributeError, TypeError):
            pass
        return Score(value=verdict.passed, explanation=verdict.reason, metadata=detail)


def apple_on_plate() -> Scorer:
    """Pass iff the apple rests, released, on the plate for the episode's last 1.0 s."""
    return _AppleOnPlate()
