"""Per-step episode trace, written beside the eval log.

The eval log keeps three reduced scores per episode, which is enough to say
whether a run passed and nothing about *why*. Both eval scripts already collect
a per-step ``info`` dict in memory and then throw all but three aggregates away.

This keeps it. The question that motivated it: MolmoAct2 failures end with the
apple metres away on the floor, and two explanations fit that equally well from
the summary alone -- the arm hit it hard, or the arm nudged it and nothing in the
scene ever slowed it down. Those call for opposite fixes, and telling them apart
needs the apple's speed at the moment it first moves, which no artefact currently
records.

Purely observational: nothing here can influence an action, a score or a
termination, so adding it does not invalidate any measurement taken before it
existed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars and arrays into something ``json`` will accept."""
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_step_trace(
    log_dir: str | Path,
    infos: Sequence[Mapping[str, Any]],
    measured: Sequence[Any] = (),
    *,
    name: str = "steps.jsonl",
) -> Path | None:
    """Write one JSON object per step; return the path, or None if nothing to write.

    ``measured`` is the per-step joint vector, zipped in under ``joint_pos`` when
    its length matches. It is kept alongside rather than merged upstream because
    the joints come off the observation and the rest off ``StepResult.info``, and
    keeping the join here means neither eval script grows a second bookkeeping
    structure.
    """
    if not infos:
        return None
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    joints = list(measured) if len(measured) == len(infos) else [None] * len(infos)
    with path.open("w", encoding="utf-8") as handle:
        for step, (info, joint) in enumerate(zip(infos, joints, strict=True)):
            row: dict[str, Any] = {"step": step, **_jsonable(dict(info))}
            if joint is not None:
                row["joint_pos"] = _jsonable(joint)
            handle.write(json.dumps(row) + "\n")
    return path


def first_disturbance(
    infos: Sequence[Mapping[str, Any]], *, threshold_m: float = 0.001
) -> dict[str, Any] | None:
    """The step at which the apple first moves, and how fast it is going.

    This is the number the whole diagnosis turns on. An apple that leaves slowly
    and then keeps going was not hit hard -- it was never stopped, which points
    at the scene's contact model rather than at the policy. One that leaves fast
    points the other way.
    """
    for step, info in enumerate(infos):
        moved = info.get("apple_displacement")
        if moved is None or float(moved) <= threshold_m:
            continue
        speeds = [
            float(later["apple_speed"])
            for later in infos[step:]
            if later.get("apple_speed") is not None
        ]
        return {
            "step": step,
            "displacement_m": float(moved),
            "speed_at_first_move_mps": (
                float(info["apple_speed"]) if info.get("apple_speed") is not None else None
            ),
            "peak_speed_after_mps": max(speeds) if speeds else None,
            "apple_position": _jsonable(info.get("apple_position")),
        }
    return None
