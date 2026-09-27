"""`arm/verdict.py`: reading the camera-verdict scorer's grade back out of a log."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from robot_console.arm import verdict as v

REASON = "held 1.20 s: apple within 0.012 m of the plate centre"


def _log(*, status="success", apple=1.0, error=None, reason=REASON,
         instruction="put it on the plate") -> dict:
    sample = {"epochs": [{"apple_on_plate": apple}], "instruction": instruction,
              "termination_reasons": ["max_steps"],
              "trial_metadata": [{"apple_on_plate": {"reason": reason, "passed": apple >= 1}}]}
    if error:
        sample["error"] = error
    return {"status": status, "samples": [sample]}


def _write(path: Path, log: dict, *, mtime: float | None = None) -> Path:
    path.write_text(json.dumps(log))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_a_passing_episode_reads_pass_and_carries_the_scorers_reason() -> None:
    out = v.verdict_from_log(_log())
    assert out.outcome == "PASS"
    assert out.detail == REASON
    assert out.instruction == "put it on the plate"
    assert out.termination == ("max_steps",)
    assert out.line == f"PASS {REASON}"


def test_scores_are_floats_and_a_one_point_zero_passes() -> None:
    assert v.verdict_from_log(_log(apple=1.0)).outcome == "PASS"
    assert v.verdict_from_log(_log(apple=0.0, reason="no pose")).outcome == "FAIL"


def test_a_log_without_the_scorers_grade_is_an_error() -> None:
    log = _log()
    log["samples"][0]["epochs"] = [{"reference_success": 1.0}]
    assert v.verdict_from_log(log).outcome == "ERROR"


def test_a_failed_run_status_is_an_error_not_a_zero() -> None:
    assert v.verdict_from_log(_log(status="error")).outcome == "ERROR"


def test_a_sample_error_under_a_successful_status_is_still_an_error() -> None:
    out = v.verdict_from_log(_log(error="EmbodimentFault: no post-publish joint state\nmore"))
    assert out.outcome == "ERROR"
    assert out.detail == "EmbodimentFault: no post-publish joint state"


def test_find_log_ignores_the_live_snapshot(tmp_path: Path) -> None:
    _write(tmp_path / "apple-on-plate_abc.live.json", {"status": "started", "samples": []})
    assert v.find_log(tmp_path) is None
    assert v.grade(tmp_path).outcome == "ERROR"
    real = _write(tmp_path / "apple-on-plate_abc.json", _log())
    assert v.find_log(tmp_path) == real
    assert v.grade(tmp_path).outcome == "PASS"


def test_the_newest_log_wins_when_there_are_two(tmp_path: Path) -> None:
    now = time.time()
    _write(tmp_path / "old.json", _log(apple=0.0), mtime=now - 100)
    _write(tmp_path / "new.json", _log(apple=1.0), mtime=now)
    assert v.grade(tmp_path).outcome == "PASS"


def test_a_missing_directory_is_an_error_with_a_reason(tmp_path: Path) -> None:
    out = v.grade(tmp_path / "nowhere")
    assert out.outcome == "ERROR" and "no eval log" in out.detail


def test_the_cli_prints_the_outcome_then_free_text(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "log.json", _log(apple=0.0, reason="apple moving at 0.0300 m/s"))
    assert v.main([str(tmp_path)]) == 0
    outcome, detail = capsys.readouterr().out.strip().split(" ", 1)
    assert (outcome, detail) == ("FAIL", "apple moving at 0.0300 m/s")


def test_the_cli_can_print_one_field_and_json(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "log.json", _log())
    v.main([str(tmp_path), "--field", "instruction"])
    assert capsys.readouterr().out.strip() == "put it on the plate"
    v.main([str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["outcome"] == "PASS" and data["detail"] == REASON
