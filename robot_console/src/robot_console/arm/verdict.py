"""Read one episode's graded log: PASS, FAIL or ERROR, and why.

The grading itself is the camera-verdict scorer's (``scorer.apple_on_plate``), which
``inspect-robot`` runs at the end of the episode; this reads its score and the reason it
left in the trial metadata out of the framework's JSON log. Stdlib only, so the base
install can run it.

Two facts learned the hard way:

* Scores are **floats** in the log (`1.0`, not `1`).
* An episode that *errored* and one that ran and failed are different outcomes, and
  are reported differently.

The one-line CLI form is what `run_task.sh` reads (`read -r outcome detail`):

    python -m robot_console.arm.verdict RUN_DIR
    FAIL apple 0.113 m from the plate centre (gate 0.08) at t=41.20 s (...)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: The scorer's name, which is the key its score is logged under.
PASS_KEY = "apple_on_plate"


@dataclass(frozen=True)
class Verdict:
    outcome: str                  # PASS | FAIL | ERROR
    detail: str                   # one line: the scorer's reason, or an error's first line
    instruction: str | None       # what the policy was told, as the log recorded it
    termination: tuple[str, ...]  # inspect-robot's termination reasons, if recorded
    log: str | None = None        # the file this was read from

    @property
    def line(self) -> str:
        return f"{self.outcome} {self.detail}"


def find_log(run_dir: Path | str) -> Path | None:
    """The eval log in `run_dir`: the newest `*.json` that is not a `*.live.json` snapshot."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    candidates = [p for p in run_dir.glob("*.json") if not p.name.endswith(".live.json")]
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
    termination = tuple(str(t) for t in (sample.get("termination_reasons") or ()) if t)

    if log.get("status") != "success" or sample.get("error"):
        reason = str(sample.get("error") or log.get("error") or "errored").splitlines()[0]
        return Verdict("ERROR", reason[:160], instruction, termination, log_path)

    epochs = sample.get("epochs") or []
    scores = epochs[0] if epochs and isinstance(epochs[0], dict) else {}
    if scores.get(PASS_KEY) is None:
        return Verdict("ERROR", f"the log carries no {PASS_KEY} score", instruction,
                       termination, log_path)
    outcome = "PASS" if float(scores[PASS_KEY]) >= 1.0 else "FAIL"
    metadata = sample.get("trial_metadata") or []
    detail = ""
    if metadata and isinstance(metadata[0], dict):
        detail = str((metadata[0].get(PASS_KEY) or {}).get("reason") or "")
    return Verdict(outcome, detail or "(no reason recorded)", instruction, termination, log_path)


def grade(run_dir: Path | str) -> Verdict:
    """Find the log in `run_dir` and read it; a directory with no log is an ERROR."""
    path = find_log(run_dir)
    if path is None:
        return Verdict("ERROR", f"no eval log in {run_dir}", None, ())
    try:
        log = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return Verdict("ERROR", f"unreadable log {path.name}: {exc}"[:160], None, (), str(path))
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
        print(json.dumps(asdict(verdict)))
    elif args.field:
        value = getattr(verdict, args.field, None)
        print("" if value is None else (" ".join(value) if isinstance(value, tuple) else value))
    else:
        print(verdict.line)
    # Always 0: the caller decides what a FAIL or an ERROR means for its exit status.
    return 0


if __name__ == "__main__":
    sys.exit(main())
