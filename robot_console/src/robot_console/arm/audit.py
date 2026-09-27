"""Audit camera verdicts against simulator ground truth, offline (console spec §4).

The camera-verdict scorer grades an episode from public observations only. This audit is
the separate check that its verdicts are right: for recorded episodes it compares each
camera verdict with the verdict the same pass criterion gives on the simulator's own
record of where the apple, the plate and the fingers really were.

That truth never reaches the scorer, and never crosses the wire:

* the simulator writes it to a **local file**, and only when asked -- `kitchen.sh serve`
  appends to the path in `SIMULATOR_TRUTH_LOG` (see "Truth log" below); nothing about it
  is published, so the wire stays exactly the vendor interfaces plus the workspace
  extensions;
* this module reads that file and the episodes' eval logs **after** the episodes, from
  disk. It is not an `inspect_robots` entry point, the scorer never imports it, and a
  test holds both of those true.

    python -m robot_console.arm.audit --truth simulator/runs/truth.jsonl \
        runs/task/molmospaces/

Each argument is an episode's run directory (as `run_task.sh` writes them) or a directory
of them. The exit status is 0 when every audited episode agrees, 1 when any disagrees,
and 2 when none could be audited.

**Truth log.** JSON lines, one object each, in the order written:

* `{"kind": "reset", "wall": <unix s>, "stamp": <sim s>}` -- `/reset` has completed and
  its observations are out. Starts a segment; an episode is matched to the last reset
  before it started (`stats.started_at` in its eval log).
* `{"kind": "state", "stamp": <sim s>, "apple": [x, y, z], "plate": [x, y, z],
  "fingers": [[[x, y, z], [x, y, z]], ...]}` -- one per rig frame, `stamp` on the same
  clock as the rig's `header.stamp`; positions in the rig's root frame (`scene/worktop`,
  the frame `ros_settings.SCENE_CAMERAS` are written in): the apple's centre, the
  plate's centre and each finger's gripping segment (pad root to tip).

Other kinds (a header, say) are ignored.

**The criterion** is the scorer's, with its constants from `vision_success`: over the
last `HOLD_SECONDS` of the graded window the apple is within `MAX_HORIZONTAL_DIST_M` of
the plate centre horizontally and `Z_TOLERANCE_M` of `RESTING_Z_M`, no faster than
`MAX_SPEED_MPS` over `SPEED_BASELINE_S`, and every finger at least `FINGER_CLEARANCE_M`
from its centre. Truth that does not cover the hold is `UNKNOWN`, never a verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from robot_console.arm import verdict as verdict_mod
from robot_console.arm.vision_success import (
    FINGER_CLEARANCE_M,
    HOLD_SECONDS,
    MAX_HORIZONTAL_DIST_M,
    MAX_SAMPLE_GAP_S,
    MAX_SPEED_MPS,
    RESTING_Z_M,
    SPEED_BASELINE_S,
    WINDOW_KEY,
    Z_TOLERANCE_M,
)

#: The environment variable the simulator reads to write its truth log. Named here for
#: the operator's benefit; this console never sets or reads it.
TRUTH_ENV = "SIMULATOR_TRUTH_LOG"

EXIT_AGREE, EXIT_DISAGREE, EXIT_NOTHING = 0, 1, 2


# ------------------------------------------------------------------ the truth log


@dataclass(frozen=True)
class State:
    stamp: float
    apple: tuple[float, float, float]
    plate: tuple[float, float, float]
    fingers: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]


@dataclass
class Segment:
    """Everything the simulator recorded between one `/reset` and the next."""

    wall: float
    states: list[State] = field(default_factory=list)
    source: str = ""


def _point(value: Any) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in value)
    return (x, y, z)


def read_truth(paths: Iterable[Path]) -> list[Segment]:
    """Every segment in the truth logs, ordered by reset time. A state before any reset
    belongs to no episode and is dropped."""
    segments: list[Segment] = []
    for path in paths:
        current: Segment | None = None
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: not JSON ({exc})") from None
            kind = row.get("kind")
            if kind == "reset":
                current = Segment(float(row["wall"]), source=f"{path.name}:{n}")
                segments.append(current)
            elif kind == "state" and current is not None:
                current.states.append(State(
                    float(row["stamp"]), _point(row["apple"]), _point(row["plate"]),
                    tuple((_point(a), _point(b)) for a, b in row["fingers"])))
    for segment in segments:
        segment.states.sort(key=lambda s: s.stamp)
    segments.sort(key=lambda s: s.wall)
    return segments


def truth_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.jsonl")) if path.is_dir() else [path]


# ------------------------------------------------------------------ the criterion, on truth


def _segment_distance(p, a, b) -> float:
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    denom = sum(v * v for v in ab)
    t = 0.0 if denom == 0 else max(0.0, min(1.0, sum(ap[i] * ab[i] for i in range(3)) / denom))
    return math.dist(p, [a[i] + t * ab[i] for i in range(3)])


@dataclass
class TruthVerdict:
    outcome: str        # PASS | FAIL | UNKNOWN
    reason: str
    apple_at_end: tuple[float, float, float] | None = None
    plate_xy: tuple[float, float] | None = None


def judge(states: Sequence[State], end: float) -> TruthVerdict:
    """The pass criterion over truth, for a hold ending at `end` (sim seconds)."""
    start = end - HOLD_SECONDS
    hold = [s for s in states if start - 1e-9 <= s.stamp <= end + 1e-9]
    if not hold or hold[0].stamp - start > MAX_SAMPLE_GAP_S \
            or end - hold[-1].stamp > MAX_SAMPLE_GAP_S:
        return TruthVerdict("UNKNOWN", f"truth does not cover the hold {start:.2f}..{end:.2f} s")
    for a, b in zip(hold, hold[1:]):
        if b.stamp - a.stamp > MAX_SAMPLE_GAP_S:
            return TruthVerdict("UNKNOWN", f"truth has a {b.stamp - a.stamp:.2f} s gap at "
                                           f"t={b.stamp:.2f} s")
    last = hold[-1]
    out = TruthVerdict("PASS", "", last.apple, last.plate[:2])
    for s in hold:
        h = math.dist(s.apple[:2], s.plate[:2])
        dz = abs(s.apple[2] - RESTING_Z_M)
        clearance = min((_segment_distance(s.apple, a, b) for a, b in s.fingers),
                        default=math.inf)
        earlier = [e for e in states if s.stamp - e.stamp >= SPEED_BASELINE_S]
        if not earlier:
            return TruthVerdict("UNKNOWN", "truth starts too late to measure speed",
                                out.apple_at_end, out.plate_xy)
        e = earlier[-1]
        speed = math.dist(s.apple, e.apple) / (s.stamp - e.stamp)
        if clearance < FINGER_CLEARANCE_M:
            out.outcome, out.reason = "FAIL", f"a finger {clearance:.3f} m from the apple"
        elif h > MAX_HORIZONTAL_DIST_M:
            out.outcome, out.reason = "FAIL", f"apple {h:.3f} m from the plate centre"
        elif dz > Z_TOLERANCE_M:
            out.outcome, out.reason = "FAIL", f"apple centre at z={s.apple[2]:.3f} m"
        elif speed > MAX_SPEED_MPS:
            out.outcome, out.reason = "FAIL", f"apple moving at {speed:.4f} m/s"
        if out.outcome == "FAIL":
            out.reason += f" at t={s.stamp:.2f} s"
            return out
    out.reason = f"held {hold[-1].stamp - hold[0].stamp:.2f} s on the plate, released"
    return out


# ------------------------------------------------------------------ one episode


@dataclass
class Audit:
    run: str
    camera: str                     # PASS | FAIL | ERROR
    truth: str                      # PASS | FAIL | UNKNOWN
    agree: bool | None              # None when either side has no verdict
    camera_reason: str = ""
    truth_reason: str = ""
    apple_error_m: float | None = None
    plate_error_m: float | None = None
    note: str = ""

    @property
    def false_pass(self) -> bool:
        return self.camera == "PASS" and self.truth == "FAIL"


def _wall(text: Any) -> float | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _detail(log: Mapping[str, Any]) -> Mapping[str, Any]:
    samples = log.get("samples") or [{}]
    meta = (samples[0] or {}).get("trial_metadata") or [{}]
    return (meta[0] or {}).get(verdict_mod.PASS_KEY) or {}


def _segment_for(segments: Sequence[Segment], started: float | None) -> Segment | None:
    if started is None:
        return None
    before = [s for s in segments if s.wall <= started]
    return before[-1] if before else None


def audit_run(run_dir: Path, segments: Sequence[Segment]) -> Audit:
    """Compare one episode's camera verdict with the truth recorded for it."""
    camera = verdict_mod.grade(run_dir)
    audit = Audit(str(run_dir), camera.outcome, "UNKNOWN", None, camera.detail)
    if camera.log is None:
        audit.truth_reason = "no eval log"
        return audit
    log = json.loads(Path(camera.log).read_text(encoding="utf-8"))
    stats, detail = log.get("stats") or {}, _detail(log)
    started = _wall(stats.get("started_at")) or _wall((log.get("eval") or {}).get("created"))
    segment = _segment_for(segments, started)
    if segment is None:
        audit.truth_reason = "no /reset in the truth log before this episode started"
        return audit
    window = detail.get(WINDOW_KEY)
    if window:
        end = float(window[1])
    else:
        # No graded window (an errored episode, or a log from before it was recorded):
        # the segment's own end is the best estimate, and is said to be one.
        if not segment.states:
            audit.truth_reason = "the truth segment holds no state"
            return audit
        end = segment.states[-1].stamp
        audit.note = "hold end taken from the truth segment; the log records no window"
    truth = judge(segment.states, end)
    audit.truth, audit.truth_reason = truth.outcome, truth.reason
    if camera.outcome in ("PASS", "FAIL") and truth.outcome in ("PASS", "FAIL"):
        audit.agree = camera.outcome == truth.outcome
    final, plate = detail.get("final_position"), detail.get("plate_xy")
    if final and truth.apple_at_end:
        audit.apple_error_m = round(math.dist([float(v) for v in final], truth.apple_at_end), 4)
    if plate and truth.plate_xy:
        audit.plate_error_m = round(math.dist([float(v) for v in plate], truth.plate_xy), 4)
    return audit


def run_dirs(paths: Iterable[Path]) -> list[Path]:
    """Each argument, or the run directories inside it when it holds no log itself."""
    out: list[Path] = []
    for path in paths:
        if verdict_mod.find_log(path) is not None or not path.is_dir():
            out.append(path)
        else:
            out.extend(sorted(p for p in path.iterdir()
                              if p.is_dir() and verdict_mod.find_log(p) is not None))
    return out


# ------------------------------------------------------------------ the report


def summarise(audits: Sequence[Audit]) -> dict[str, int]:
    compared = [a for a in audits if a.agree is not None]
    return {
        "episodes": len(audits),
        "audited": len(compared),
        "agree": sum(a.agree for a in compared),
        "false_pass": sum(a.false_pass for a in compared),
        "false_fail": sum(a.camera == "FAIL" and a.truth == "PASS" for a in compared),
        "unaudited": len(audits) - len(compared),
    }


def format_report(audits: Sequence[Audit]) -> str:
    lines = []
    for a in audits:
        mark = {True: "agree", False: "DISAGREE", None: "unaudited"}[a.agree]
        errs = []
        if a.apple_error_m is not None:
            errs.append(f"apple err {a.apple_error_m:.4f} m")
        if a.plate_error_m is not None:
            errs.append(f"plate err {a.plate_error_m:.4f} m")
        lines.append(f"{mark:9s} camera {a.camera:5s} truth {a.truth:7s} {Path(a.run).name}"
                     + (f"  ({', '.join(errs)})" if errs else ""))
        if a.agree is not True:
            lines.append(f"          camera: {a.camera_reason}")
            lines.append(f"          truth:  {a.truth_reason}")
        if a.note:
            lines.append(f"          note:   {a.note}")
    s = summarise(audits)
    lines.append(f"{s['audited']}/{s['episodes']} audited: {s['agree']} agree, "
                 f"{s['false_pass']} false PASS, {s['false_fail']} false FAIL, "
                 f"{s['unaudited']} unaudited")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m robot_console.arm.audit",
        description="Compare camera verdicts with the simulator's ground-truth log, offline.")
    parser.add_argument("--truth", required=True, type=Path,
                        help=f"the simulator's truth log (written when {TRUTH_ENV} is set), "
                             "or a directory of *.jsonl logs")
    parser.add_argument("runs", nargs="+", type=Path,
                        help="episode run directories, or directories holding them")
    parser.add_argument("--json", action="store_true", help="print the audits as JSON")
    args = parser.parse_args(argv)

    files = truth_files(args.truth)
    missing = [str(p) for p in files if not p.is_file()]
    if missing or not files:
        parser.error(f"no truth log at {args.truth}")
    segments = read_truth(files)
    audits = [audit_run(run, segments) for run in run_dirs(args.runs)]
    if args.json:
        print(json.dumps({"audits": [asdict(a) for a in audits], "summary": summarise(audits)},
                         indent=2))
    else:
        print(format_report(audits))
    s = summarise(audits)
    if s["audited"] == 0:
        return EXIT_NOTHING
    return EXIT_AGREE if s["agree"] == s["audited"] else EXIT_DISAGREE


if __name__ == "__main__":
    sys.exit(main())
