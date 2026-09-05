"""`arm/verdict.py`: the grading that used to be a shell heredoc, now testable."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from robot_console.arm import verdict as v


def _log(*, status="success", apple=1.0, reference=1.0, distance=0.0446, error=None,
         ref_key="reference_success", instruction="put it on the plate") -> dict:
    epoch = {"apple_on_plate": apple, "apple_plate_distance": distance}
    if reference is not None:
        epoch[ref_key] = reference
    sample = {"epochs": [epoch], "instruction": instruction, "termination_reasons": ["success"]}
    if error:
        sample["error"] = error
    return {"status": status, "samples": [sample]}


def _write(path: Path, log: dict, *, mtime: float | None = None) -> Path:
    path.write_text(json.dumps(log))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_a_passing_episode_reads_pass_and_carries_the_closest_approach() -> None:
    out = v.verdict_from_log(_log())
    assert (out.outcome, out.reference) == ("PASS", "PASS")
    assert out.distance_m == 0.0446
    assert out.detail == "0.0446 m from plate centre"
    assert out.instruction == "put it on the plate"
    assert out.termination == ("success",)
    assert out.line == "PASS PASS 0.0446 m from plate centre"


def test_scores_are_floats_and_a_one_point_zero_passes() -> None:
    # `"1"` string-matching once called this a failure; the score is 1.0 on the wire.
    assert v.verdict_from_log(_log(apple=1.0)).outcome == "PASS"
    assert v.verdict_from_log(_log(apple=0.0, distance=0.3137)).outcome == "FAIL"


def test_camera_and_pose_reference_can_disagree() -> None:
    # The camera verdict cannot see height; the pose reference can. Both are reported so
    # the disagreement is visible, and only the camera's is the grade.
    out = v.verdict_from_log(_log(apple=1.0, reference=0.0))
    assert (out.outcome, out.reference) == ("PASS", "FAIL")


def test_the_legacy_reference_key_still_reads() -> None:
    out = v.verdict_from_log(_log(ref_key="sim_task_success", reference=1.0))
    assert out.reference == "PASS"
    assert v.verdict_from_log(_log(reference=None)).reference == "n/a"


def test_a_failed_run_status_is_an_error_not_a_zero() -> None:
    out = v.verdict_from_log(_log(status="error"))
    assert (out.outcome, out.reference) == ("ERROR", "ERROR")
    assert math.isnan(out.distance_m)


def test_a_sample_error_under_a_successful_status_is_still_an_error() -> None:
    out = v.verdict_from_log(_log(error="EmbodimentFault: no post-publish joint state\nmore"))
    assert out.outcome == "ERROR"
    assert out.detail == "EmbodimentFault: no post-publish joint state"


def test_find_log_ignores_the_live_snapshot_and_the_preflight_record(tmp_path: Path) -> None:
    # The old heredoc's glob matched both of these; the snapshot has status `started`.
    _write(tmp_path / "apple-on-plate_abc.live.json", {"status": "started", "samples": []})
    _write(tmp_path / "scene_reset.json", {"url": "ws://x", "reset": True})
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


def test_the_cli_prints_three_leading_tokens_then_free_text(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "log.json", _log(apple=1.0, reference=0.0))
    assert v.main([str(tmp_path)]) == 0
    line = capsys.readouterr().out.strip()
    scored, ref, *detail = line.split()
    assert (scored, ref) == ("PASS", "FAIL")
    assert " ".join(detail) == "0.0446 m from plate centre"


def test_the_cli_can_print_one_field_and_json(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "log.json", _log())
    v.main([str(tmp_path), "--field", "instruction"])
    assert capsys.readouterr().out.strip() == "put it on the plate"
    v.main([str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["outcome"] == "PASS" and data["distance_m"] == 0.0446
