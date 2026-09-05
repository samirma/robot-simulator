"""Grade one `inspect-robot run` directory: PASS, FAIL or ERROR, and why.

This used to be an eighteen-line Python heredoc inside `simulator/kitchen.sh` -- grading
logic living in a shell string in the *simulator's* tree, untested, and carrying a bug
nobody had hit yet: its `*.json` glob also matched inspect-robot's live snapshot
(`<task>_<id>.live.json`, status `started`), so a run directory holding both files could
have its verdict read from the half-written one. Grading is the console's business; it
lives here, it is stdlib-only so the base install can run it, and it is tested.

Two facts the shell learned the hard way and this module keeps:

* Scores are **floats** on the wire (`1.0`, not `1`). String-matching `"1"` quietly
  called a passing episode a failure -- measured on an episode that placed the apple
  13 mm from the plate centre and was reported FAIL.
* An episode that *errored* and one that ran and scored zero are different failures.
  Reporting both as "apple nan m from plate centre" hides the first behind the second.

The one-line CLI form is what `run_task.sh` reads (`read -r scored ref detail`):

    python -m robot_console.arm.verdict RUN_DIR
    PASS FAIL 0.0446 m from plate centre
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: The graded verdict is the camera's: `arm/vision_success.py` reads it off the overhead
#: frame the policy is handed. `reference_success` recomputes the same predicate from the
#: free-joint poses and grades nothing; it is the cross-check that separates "the policy
#: failed" from "the detector stopped seeing". Logs before 2026-09-04 carry its earlier
#: name, so both are read.
REFERENCE_KEYS = ("reference_success", "sim_task_success")
PASS_KEY = "apple_on_plate"
DISTANCE_KEY = "apple_plate_distance"


@dataclass(frozen=True)
class Verdict:
    outcome: str                  # PASS | FAIL | ERROR
    reference: str                # PASS | FAIL | n/a | ERROR -- the pose-based cross-check
    distance_m: float             # closest approach across the episode, nan if unknown
    detail: str                   # one line: the distance, or an error's first line
    instruction: str | None       # what the policy was told, as the log recorded it
    termination: tuple[str, ...]  # inspect-robot's termination reasons, if recorded
    log: str | None = None        # the file this was read from

    @property
    def line(self) -> str:
        """The shell-facing form: three leading tokens, then free text."""
        return f"{self.outcome} {self.reference} {self.detail}"


def find_log(run_dir: Path | str) -> Path | None:
    """The eval log in `run_dir`: the newest `*.json` that is neither the preflight's
    `scene_reset.json` nor inspect-robot's `*.live.json` snapshot."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    candidates = [
        p for p in run_dir.glob("*.json")
        if p.name != "scene_reset.json" and not p.name.endswith(".live.json")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _instruction_of(log: dict, sample: dict) -> str | None:
    for source in (sample, log.get("metadata") or {}, (log.get("task") or {}).get("metadata") or {}):
        text = source.get("instruction") if isinstance(source, dict) else None
        if text:
            return str(text)
    return None


def verdict_from_log(log: dict[str, Any], log_path: str | None = None) -> Verdict:
    """Pure: an inspect-robot eval log dict -> a `Verdict`."""
    samples = log.get("samples") or []
    sample = samples[0] if samples and isinstance(samples[0], dict) else {}
    instruction = _instruction_of(log, sample)
    termination = tuple(str(t) for t in (sample.get("termination_reasons") or ()))

    if log.get("status") != "success" or sample.get("error"):
        reason = str(sample.get("error") or log.get("error") or "errored").splitlines()[0]
        return Verdict("ERROR", "ERROR", math.nan, reason[:90], instruction, termination, log_path)

    epochs = sample.get("epochs") or []
    scores = epochs[0] if epochs and isinstance(epochs[0], dict) else {}
    outcome = "PASS" if float(scores.get(PASS_KEY) or 0.0) >= 1.0 else "FAIL"

    reference = "n/a"
    for key in REFERENCE_KEYS:
        if key in scores and scores[key] is not None:
            reference = "PASS" if float(scores[key]) >= 1.0 else "FAIL"
            break

    raw = scores.get(DISTANCE_KEY)
    distance = float(raw) if raw is not None else math.nan
    return Verdict(outcome, reference, distance, f"{distance:.4f} m from plate centre",
                   instruction, termination, log_path)


def grade(run_dir: Path | str) -> Verdict:
    """Find the log in `run_dir` and grade it; a directory with no log is an ERROR."""
    path = find_log(run_dir)
    if path is None:
        return Verdict("ERROR", "ERROR", math.nan, f"no eval log in {run_dir}", None, ())
    try:
        log = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return Verdict("ERROR", "ERROR", math.nan, f"unreadable log {path.name}: {exc}"[:90],
                       None, (), str(path))
    return verdict_from_log(log, str(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="an inspect-robot --log-dir directory")
    parser.add_argument("--field", default=None,
                        help="print one field of the verdict instead of the summary line")
    parser.add_argument("--json", action="store_true", help="print the whole verdict as JSON")
    args = parser.parse_args(argv)

    verdict = grade(args.run_dir)
    if args.json:
        out = asdict(verdict)
        out["distance_m"] = None if math.isnan(verdict.distance_m) else verdict.distance_m
        print(json.dumps(out))
    elif args.field:
        value = getattr(verdict, args.field, None)
        print("" if value is None else (" ".join(value) if isinstance(value, tuple) else value))
    else:
        print(verdict.line)
    # Always 0: the caller decides what a FAIL or an ERROR means for its exit status.
    return 0


if __name__ == "__main__":
    sys.exit(main())
