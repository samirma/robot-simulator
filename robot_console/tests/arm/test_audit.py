"""The task-grading audit (console spec §4): camera verdicts against simulator truth.

Offline, end to end on fixtures: a rig episode rendered from known poses is graded by the
real camera-verdict scorer and written as an eval log, the same poses are written as the
simulator's truth log, and the audit compares the two. The truth never goes near the
scorer -- the last tests hold that to the source.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from robot_console.arm import audit
from robot_console.arm import vision_success as vs
from robot_console.arm.kinematics import finger_segments

from .rig_fixtures import AWAY, PLATE_XY, RESTING, episode, holding

STARTED = "2026-09-27T10:00:00+00:00"
STARTED_WALL = 1790503200.0   # STARTED as unix seconds


def _eval_log(samples: list[dict]) -> dict:
    """What `inspect-robot` writes for one graded episode, with the scorer's detail."""
    verdict = vs.assess(samples)
    return {
        "status": "success",
        "stats": {"started_at": STARTED},
        "samples": [{
            "epochs": [{"apple_on_plate": 1.0 if verdict.passed else 0.0}],
            "trial_metadata": [{"apple_on_plate": verdict.as_dict()}],
        }],
    }


def _run(tmp_path: Path, name: str, samples: list[dict]) -> Path:
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    (run / "apple-on-plate_x.json").write_text(json.dumps(_eval_log(samples)))
    return run


def _fingers(joints) -> list:
    return [[list(map(float, a)), list(map(float, b))]
            for a, b in finger_segments(list(joints[:5]), joints[5])]


def _truth(tmp_path: Path, apple_at, joints=AWAY, *, reset_wall=STARTED_WALL - 2.0,
           seconds: float = 3.0, stale_before: bool = True, name: str = "truth.jsonl") -> Path:
    """A truth log shaped like the simulator's: a reset, then one state per rig frame on
    the rig's clock (the fixture episodes stamp from 100.0 s at 10 Hz)."""
    lines = []
    if stale_before:  # an earlier episode's segment, which must not be matched
        lines.append({"kind": "reset", "wall": reset_wall - 600, "stamp": 0.0})
        lines += [{"kind": "state", "stamp": 100.0 + i / 10, "apple": [0.5, 0.5, 0.3],
                   "plate": [*PLATE_XY, 0.0], "fingers": _fingers(AWAY)} for i in range(31)]
    lines.append({"kind": "reset", "wall": reset_wall, "stamp": 99.5})
    for i in range(int(seconds * 10) + 1):
        t = i / 10
        lines.append({"kind": "state", "stamp": 100.0 + t, "apple": list(apple_at(t)),
                      "plate": [*PLATE_XY, 0.0], "fingers": _fingers(joints)})
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def _audit(tmp_path, run, truth) -> audit.Audit:
    return audit.audit_run(run, audit.read_truth([truth]))


def test_the_scorer_records_the_window_it_graded() -> None:
    verdict = vs.assess(episode(lambda t: RESTING))
    assert verdict.as_dict()[vs.WINDOW_KEY] == [100.0, 103.0]


def test_a_valid_placement_agrees_and_the_camera_error_is_measured(tmp_path) -> None:
    run = _run(tmp_path, "pass", episode(lambda t: RESTING))
    result = _audit(tmp_path, run, _truth(tmp_path, lambda t: RESTING))
    assert (result.camera, result.truth, result.agree) == ("PASS", "PASS", True)
    assert result.apple_error_m is not None and result.apple_error_m < 0.004
    assert result.plate_error_m is not None and result.plate_error_m < 0.01


def test_a_held_apple_is_a_fail_on_both_sides(tmp_path) -> None:
    grip = holding(RESTING)
    run = _run(tmp_path, "held", episode(lambda t: RESTING, joints_at=lambda t: grip))
    result = _audit(tmp_path, run, _truth(tmp_path, lambda t: RESTING, joints=grip))
    assert (result.camera, result.truth, result.agree) == ("FAIL", "FAIL", True)
    assert "finger" in result.truth_reason


def test_a_camera_pass_the_truth_contradicts_is_a_false_pass(tmp_path, capsys) -> None:
    """The cameras saw it on the plate; the simulator had it 0.1 m away (e.g. a drifted
    calibration). This is the failure the audit exists to catch."""
    run = _run(tmp_path, "wrong", episode(lambda t: RESTING))
    elsewhere = (PLATE_XY[0] + 0.10, PLATE_XY[1], 0.040)
    truth = _truth(tmp_path, lambda t: elsewhere)
    result = _audit(tmp_path, run, truth)
    assert (result.camera, result.truth, result.agree) == ("PASS", "FAIL", False)
    assert result.false_pass
    assert audit.main(["--truth", str(truth), str(run.parent)]) == audit.EXIT_DISAGREE
    out = capsys.readouterr().out
    assert "DISAGREE" in out and "1 false PASS" in out


def test_moving_apple_fails_on_truth_too(tmp_path) -> None:
    def moving(t):
        return (PLATE_XY[0] - 0.04 + 0.03 * t, PLATE_XY[1], 0.040)

    run = _run(tmp_path, "moving", episode(moving))
    result = _audit(tmp_path, run, _truth(tmp_path, moving))
    assert (result.camera, result.truth, result.agree) == ("FAIL", "FAIL", True)


def test_the_episode_is_matched_to_the_last_reset_before_it(tmp_path) -> None:
    """The stale segment's apple is off the plate; matching it would disagree."""
    run = _run(tmp_path, "pass", episode(lambda t: RESTING))
    result = _audit(tmp_path, run, _truth(tmp_path, lambda t: RESTING, stale_before=True))
    assert result.agree is True


def test_truth_that_misses_the_hold_is_unknown_not_a_verdict(tmp_path) -> None:
    run = _run(tmp_path, "pass", episode(lambda t: RESTING))
    truth = _truth(tmp_path, lambda t: RESTING, seconds=1.0)
    result = _audit(tmp_path, run, truth)
    assert (result.truth, result.agree) == ("UNKNOWN", None)
    assert audit.main(["--truth", str(truth), str(run)]) == audit.EXIT_NOTHING


def test_an_episode_with_no_reset_before_it_is_unaudited(tmp_path) -> None:
    run = _run(tmp_path, "pass", episode(lambda t: RESTING))
    truth = _truth(tmp_path, lambda t: RESTING, reset_wall=STARTED_WALL + 60,
                   stale_before=False)
    result = _audit(tmp_path, run, truth)
    assert result.agree is None and "no /reset" in result.truth_reason


def test_a_directory_of_runs_and_json_output(tmp_path, capsys) -> None:
    _run(tmp_path, "a", episode(lambda t: RESTING))
    truth = _truth(tmp_path, lambda t: RESTING)
    assert audit.main(["--truth", str(truth), "--json", str(tmp_path / "runs")]) == \
        audit.EXIT_AGREE
    report = json.loads(capsys.readouterr().out)
    assert report["summary"] == {"episodes": 1, "audited": 1, "agree": 1, "false_pass": 0,
                                 "false_fail": 0, "unaudited": 0}


def test_a_missing_truth_log_is_a_usage_error(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc:
        audit.main(["--truth", str(tmp_path / "nope.jsonl"), str(tmp_path)])
    assert exc.value.code == 2


# ------------------------------------------------------------------ truth stays out of grading


def test_the_scorer_side_never_reaches_the_audit_or_the_truth() -> None:
    from robot_console.arm import embodiment, molmoact, preflight, ros_settings, scorer, task

    import ast

    for module in (scorer, vs, embodiment, task, preflight, ros_settings, molmoact):
        source = inspect.getsource(module)
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported |= {f"{node.module}.{a.name}" for a in node.names}
        assert not any("audit" in name for name in imported), module.__name__
        assert audit.TRUTH_ENV not in source, module.__name__


def test_the_audit_is_not_a_framework_entry_point() -> None:
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert "arm.audit" not in pyproject
