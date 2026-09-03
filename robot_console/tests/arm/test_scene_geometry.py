"""Every module's copy of the scene geometry must agree, and the scripted
release point must be a placement the contract's own gate accepts.

The plate has moved once already -- ``(0.40, -0.20)`` -> ``(0.226, -0.226)`` --
and four files each held their own literal copy of the old position, none of
which the other tests would have caught. These tests fail if any of them drifts
again, and they judge the release point against the gate rather than against a
remembered number.
"""

from __future__ import annotations

import math
import re

import pytest

from robot_console.arm.success import PlateGoal
from robot_console.arm.task import (
    APPLE_RADIUS,
    APPLE_XYZ,
    INSTRUCTION,
    PLATE_XYZ,
    apple_on_plate,
    instruction_warning,
    resolve_instruction,
)
from robot_console.arm.waypoints import PickPlaceConfig, build_plan

GOAL = PlateGoal()
CFG = PickPlaceConfig()
#: Where an apple's centre sits once it is resting on the plate.
RESTING_Z = 0.0204 + APPLE_RADIUS


def test_every_module_agrees_on_the_plate_centre() -> None:
    assert PLATE_XYZ[:2] == GOAL.center_xy
    assert CFG.plate_xyz == PLATE_XYZ


def test_every_module_agrees_on_the_apple_spawn() -> None:
    assert APPLE_XYZ == GOAL.spawn_xyz
    assert CFG.apple_xyz == APPLE_XYZ


def test_the_geometry_still_matches_the_simulator_scene() -> None:
    """The staged task is the source of truth; these constants only mirror it.

    The two halves of this contract live in different projects and are installed
    separately, so nothing but a test can hold them together. When the simulator tree is
    checked out beside the console -- which is how anyone runs the task -- the numbers
    the console plans against are compared to the numbers the simulator actually stages.
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

def _resting_at_release() -> list[float]:
    """Where the apple ends up if the release point is hit exactly."""
    return [CFG.release_xy[0], CFG.release_xy[1], RESTING_Z]


def test_the_release_point_places_the_apple_inside_the_gate() -> None:
    placed = _resting_at_release()
    assert GOAL.explain(placed, 0.0) == "ok"
    assert GOAL.horizontal_distance(placed) < GOAL.radius
    assert GOAL.displacement(placed) > GOAL.min_displacement


def test_the_release_point_is_biased_away_from_the_spawn() -> None:
    """Clause 5's slack is thinnest on the spawn side, so aim past the centre.

    An apple resting at the near edge of the clause-1 disc has travelled only
    0.2551 m against a 0.25 m gate. Biasing the release point away from the
    spawn widens the near-ward settling error the plan tolerates from 80 mm to
    105 mm; the cost is far-ward tolerance, 80 mm down to 55 mm.
    """
    dx = PLATE_XYZ[0] - APPLE_XYZ[0]
    dy = PLATE_XYZ[1] - APPLE_XYZ[1]
    norm = math.hypot(dx, dy)
    away = (dx / norm, dy / norm)
    offset = (CFG.release_xy[0] - PLATE_XYZ[0], CFG.release_xy[1] - PLATE_XYZ[1])

    along = offset[0] * away[0] + offset[1] * away[1]
    assert along > 0, "release point is on the spawn side of the plate centre"
    # Essentially all of the bias is along that axis, not sideways.
    assert math.hypot(*offset) == pytest.approx(along, abs=1e-3)
    # Big enough to matter, small enough to leave clause 1 most of its margin.
    assert 0.015 < along < 0.040


def test_the_whole_plan_still_solves_for_the_moved_plate() -> None:
    plan = build_plan()
    assert plan.worst_position_error < CFG.max_position_error


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
    # The waypoint policy never reads the text at all; say so rather than
    # letting the run look like it honoured the request.
    assert "SO101WaypointPolicy" in instruction_warning("x", policy_ignores_it=True)


def test_resolve_instruction_prefers_explicit_text_then_falls_back(tmp_path) -> None:
    assert resolve_instruction() == INSTRUCTION
    assert resolve_instruction("lift the apple") == "lift the apple"
    path = tmp_path / "prompt.txt"
    path.write_text("  from a file\n", encoding="utf-8")
    assert resolve_instruction(None, str(path)) == "from a file"
    with pytest.raises(ValueError):
        resolve_instruction("both", str(path))
