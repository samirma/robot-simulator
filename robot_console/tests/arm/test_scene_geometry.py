"""Every module's copy of the scene geometry must agree, on both sides of the wire.

The plate has moved once already -- ``(0.40, -0.20)`` -> ``(0.226, -0.226)`` --
and four files each held their own literal copy of the old position, none of
which the other tests would have caught. These tests fail if any of them drifts
again. The two halves of the contract live in different projects and are
installed separately, so nothing but a test can hold them together.
"""

from __future__ import annotations

import math

import pytest

from robot_console.arm.success import PlateGoal
from robot_console.arm.task import (
    APPLE_RADIUS,
    APPLE_XYZ,
    INSTRUCTION,
    LAYOUTS,
    PLATE_XYZ,
    START_ARM_QPOS,
    apple_on_plate,
    instruction_warning,
    layout_of,
    resolve_instruction,
)

GOAL = PlateGoal()


def test_every_module_agrees_on_the_plate_centre() -> None:
    assert PLATE_XYZ[:2] == GOAL.center_xy


def test_every_module_agrees_on_the_apple_spawn() -> None:
    assert APPLE_XYZ == GOAL.spawn_xyz


def test_the_geometry_still_matches_the_simulator_scene() -> None:
    """The staged task is the source of truth; these constants only mirror it.

    When the simulator tree is checked out beside the console -- which is how anyone
    runs the task -- the numbers the console grades against are compared to the numbers
    the simulator actually stages.
    """
    task = _shared_task_module()
    if task is None:
        pytest.skip("sibling simulator checkout not present")

    assert task.APPLE_SPAWN == APPLE_XYZ, (
        "the simulator spawns the apple somewhere the console does not expect"
    )
    assert task.PLATE_CENTRE[:2] == PLATE_XYZ[:2], (
        "the simulator and the console disagree about where the plate is"
    )
    assert task.APPLE_RADIUS == APPLE_RADIUS
    # The success gate, clause by clause, on both sides of the wire.
    assert task.MAX_HORIZONTAL_DIST == GOAL.radius
    assert task.RESTING_Z == GOAL.center_z
    assert task.Z_TOLERANCE == GOAL.z_tolerance
    assert task.MAX_SPEED == GOAL.max_speed
    assert task.MIN_DISPLACEMENT == GOAL.min_displacement


def test_the_start_pose_matches_the_simulator() -> None:
    """The first state a VLA sees is the simulator's, and the console's record of it.

    The scripted plan's first waypoint used to be that record. It is gone, so the
    tuple lives in ``task.py`` and this holds it to the one the simulator applies before
    it snapshots the spawn state.
    """
    task = _shared_task_module()
    if task is None:
        pytest.skip("sibling simulator checkout not present")
    assert tuple(task.START_ARM_QPOS) == START_ARM_QPOS


def test_the_swapped_layout_is_the_standard_one_with_the_objects_exchanged() -> None:
    """Swapping means the plate sits at the apple's spawn and the apple at the plate's.

    Heights are the objects' own: the apple rests at its radius wherever it is, and the
    plate's gate height does not move with its position.
    """
    apple, plate = LAYOUTS["swapped"]
    assert apple[:2] == PLATE_XYZ[:2]
    assert apple[2] == APPLE_XYZ[2]
    assert plate[:2] == APPLE_XYZ[:2]
    assert plate[2] == PLATE_XYZ[2]
    assert LAYOUTS["standard"] == (APPLE_XYZ, PLATE_XYZ)


def test_layout_of_reads_the_layout_off_the_apple_and_nothing_else() -> None:
    assert layout_of(APPLE_XYZ) == "standard"
    assert layout_of((PLATE_XYZ[0], PLATE_XYZ[1], 0.024)) == "swapped"
    # A metre off the table is neither, not the nearer of the two.
    assert layout_of((1.30, 0.10, 0.02)) is None
    # The tolerance is the caller's: a native engine apple settles a few millimetres
    # off the contract point, which is inside the default and outside a strict one.
    assert layout_of((0.32, 0.10, 0.02)) == "standard"
    assert layout_of((0.32, 0.10, 0.02), tolerance=0.005) is None


def test_the_task_carries_the_layout_into_the_scene_it_grades() -> None:
    """``-T layout=swapped`` has to reach the scene's target, or the reference column
    grades a swapped world against the standard plate centre on every step."""
    swapped = apple_on_plate(layout="swapped")
    spec = swapped.scenes[0].target.spec
    apple, plate = LAYOUTS["swapped"]
    assert tuple(spec["apple_xyz"]) == apple
    assert tuple(spec["plate_center_xy"]) == plate[:2]
    assert swapped.metadata["layout"] == "swapped"

    standard = apple_on_plate()
    assert tuple(standard.scenes[0].target.spec["plate_center_xy"]) == PLATE_XYZ[:2]
    assert standard.metadata["layout"] == "standard"

    with pytest.raises(ValueError):
        apple_on_plate(layout="upside_down")


def test_the_embodiment_rebuilds_its_goal_from_the_scene() -> None:
    from robot_console.arm.embodiment import _goal_from_scene

    scene = apple_on_plate(layout="swapped").scenes[0]
    goal = _goal_from_scene(scene, PlateGoal())
    apple, plate = LAYOUTS["swapped"]
    assert goal.center_xy == plate[:2]
    assert goal.spawn_xyz == apple
    assert goal.radius == GOAL.radius
    assert goal.center_z == pytest.approx(GOAL.center_z)
    assert goal.z_tolerance == pytest.approx(GOAL.z_tolerance)
    # Displacement is measured from the swapped spawn, so a resting apple at the swapped
    # plate has travelled the same distance the standard layout requires.
    resting = (plate[0], plate[1], GOAL.center_z)
    assert goal.displacement(resting) == pytest.approx(
        math.hypot(APPLE_XYZ[0] - PLATE_XYZ[0], APPLE_XYZ[1] - PLATE_XYZ[1]), abs=1e-3
    )


def _shared_task_module():
    """Import the simulator's task module without importing the simulator."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "simulator" / "shared" / "tasks" / "apple_on_plate.py"
    )
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_shared_apple_on_plate", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        return None  # the task module needs mujoco; the console deliberately lacks it
    return module


def test_a_custom_instruction_reaches_both_the_scene_and_the_metadata() -> None:
    """``apple_on_plate`` ends in ``**_: Any``, which swallows a misspelled
    keyword without a word. The scene assertion catches the parameter being
    dropped; the metadata one catches the easier mistake of plumbing it to the
    scene while the recorded metadata keeps claiming the default, so the log
    would say the model was told something it was not.
    """
    task = apple_on_plate(instruction="put the apple in the bowl")
    assert task.scenes[0].instruction == "put the apple in the bowl"
    assert task.metadata["instruction"] == "put the apple in the bowl"

    default = apple_on_plate()
    assert default.scenes[0].instruction == INSTRUCTION
    assert default.metadata["instruction"] == INSTRUCTION


def test_the_instruction_warning_fires_only_off_the_default() -> None:
    """Silence on the benchmark text, and a warning that names the scorers
    otherwise -- a custom-goal run scoring 0 is a statement about apple-on-plate
    geometry, not about the goal that was typed.
    """
    assert instruction_warning(INSTRUCTION) is None
    warning = instruction_warning("put the apple in the bowl")
    assert warning is not None
    assert "apple_on_plate_success" in warning
    assert "put the apple in the bowl" in warning


def test_resolve_instruction_prefers_explicit_text_then_falls_back(tmp_path) -> None:
    assert resolve_instruction() == INSTRUCTION
    assert resolve_instruction("lift the apple") == "lift the apple"
    path = tmp_path / "prompt.txt"
    path.write_text("  from a file\n", encoding="utf-8")
    assert resolve_instruction(None, str(path)) == "from a file"
    with pytest.raises(ValueError):
        resolve_instruction("both", str(path))


def test_the_docs_describe_the_cameras_the_simulator_actually_stages() -> None:
    """The text a policy is told about the cameras must be the truth the simulator stages.

    Two assertions on purpose. The pose table can be right while the prose still carries
    a stale literal -- that is exactly how the overhead camera's old position survived in
    `_DOCS` for weeks after the simulator moved it -- so the formatted numbers are checked
    in the string itself, not only in the table it is built from.
    """
    from robot_console.arm.embodiment import _DOCS
    from robot_console.arm.ros_settings import SCENE_CAMERA_POSES

    task = _shared_task_module()
    if task is None:
        pytest.skip("sibling simulator checkout not present")
    staged = {name: tuple(pos) for name, pos, *_ in task.SCENE_CAMERAS}
    assert SCENE_CAMERA_POSES == staged
    for name, (x, y, z) in staged.items():
        assert f"'{name}' at ({x:.3f}, {y:.3f}, {z:.3f})" in _DOCS
