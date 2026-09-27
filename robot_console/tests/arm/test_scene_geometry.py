"""The console's copies of the task and rig constants must be the simulator's.

The rig mount poses are what the camera-verdict scorer triangulates through, and the
pass thresholds are what it gates on, so a drifted copy is a wrong verdict. The
simulator's task module imports MuJoCo, which the console deliberately lacks, so its
literal constants are read with ``ast`` rather than by importing it. Skips when no
sibling checkout is present.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from robot_console.arm import ros_settings as rs
from robot_console.arm import vision_success as vs
from robot_console.arm.task import (
    INSTRUCTION,
    MAX_POLICY_STEPS,
    START_ARM_QPOS,
    apple_on_plate,
    instruction_warning,
    resolve_instruction,
)

TASK_MODULE = (Path(__file__).resolve().parents[3] / "simulator" / "shared" / "tasks"
               / "apple_on_plate.py")


def _constants() -> dict:
    if not TASK_MODULE.exists():
        pytest.skip("sibling simulator checkout not present")
    out: dict = {}
    for node in ast.parse(TASK_MODULE.read_text()).body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        value = getattr(node, "value", None)
        for target in targets:
            if isinstance(target, ast.Name) and value is not None:
                try:
                    out[target.id] = ast.literal_eval(value)
                except ValueError:
                    pass
    return out


def test_the_rig_mount_poses_are_the_simulators() -> None:
    staged = {name: (tuple(pos), tuple(xy), fovy, tuple(res))
              for name, pos, xy, fovy, res in _constants()["SCENE_CAMERAS"]}
    assert rs.SCENE_CAMERAS == staged
    assert rs.SCENE_CAMERA_HZ == _constants()["SCENE_CAMERA_HZ"]


def test_the_pass_thresholds_are_the_simulators() -> None:
    c = _constants()
    assert vs.MAX_HORIZONTAL_DIST_M == c["MAX_HORIZONTAL_DIST"]
    assert vs.RESTING_Z_M == c["RESTING_Z"]
    assert vs.Z_TOLERANCE_M == c["Z_TOLERANCE"]
    assert vs.MAX_SPEED_MPS == c["MAX_SPEED"]
    assert vs.HOLD_SECONDS == c["SUSTAIN_SECONDS"]
    assert vs.APPLE_RADIUS_M == c["APPLE_RADIUS"]


def test_the_start_pose_matches_the_simulator() -> None:
    assert tuple(_constants()["START_ARM_QPOS"]) == START_ARM_QPOS


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
