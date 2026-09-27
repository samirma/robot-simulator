"""The arm task's own constants: its scorer, step budget and instruction.

The console's copies of the rig mount poses, pass thresholds and start pose are held
equal to the simulator's task constants by the workspace tests in ``../tests/``
(console spec §4, Contract parity), which read both source trees; this file needs neither.
"""

from __future__ import annotations

import pytest

from robot_console.arm import ros_settings as rs
from robot_console.arm.task import (
    INSTRUCTION,
    MAX_POLICY_STEPS,
    apple_on_plate,
    instruction_warning,
    resolve_instruction,
)


def test_the_task_has_one_scorer_and_the_step_budget() -> None:
    task = apple_on_plate()
    assert list(task.scorer) == ["apple_on_plate"] or [getattr(s, "name", s) for s in
                                                       task.scorer] == ["apple_on_plate"]
    assert task.max_steps == MAX_POLICY_STEPS == 220
    assert task.scenes[0].instruction == INSTRUCTION


def test_a_custom_instruction_reaches_both_the_scene_and_the_metadata() -> None:
    task = apple_on_plate(instruction="put the apple in the bowl")
    assert task.scenes[0].instruction == "put the apple in the bowl"
    assert task.metadata["instruction"] == "put the apple in the bowl"


def test_the_instruction_warning_fires_only_off_the_default() -> None:
    assert instruction_warning(INSTRUCTION) is None
    warning = instruction_warning("put the apple in the bowl")
    assert warning is not None and "apple_on_plate" in warning


def test_the_default_instruction_is_the_specs() -> None:
    assert INSTRUCTION == ("Move the arm towards the red apple, grasp it, lift it up, and "
                           "place it on the white plate.")


def test_resolve_instruction_prefers_explicit_text_then_falls_back(tmp_path) -> None:
    assert resolve_instruction() == INSTRUCTION
    assert resolve_instruction("lift the apple") == "lift the apple"
    path = tmp_path / "prompt.txt"
    path.write_text("  from a file\n", encoding="utf-8")
    assert resolve_instruction(None, str(path)) == "from a file"
    with pytest.raises(ValueError):
        resolve_instruction("both", str(path))


def test_the_docs_describe_the_cameras_the_console_calibrates_with() -> None:
    from robot_console.arm.embodiment import _DOCS

    for name, (x, y, z) in rs.SCENE_CAMERA_POSES.items():
        assert f"'{name}' at ({x:.3f}, {y:.3f}, {z:.3f})" in _DOCS
