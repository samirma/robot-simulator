"""``CONTRACT.md`` section 5, clause by clause, plus the hold that spans steps.

The expected values are the same constants ``so_arm_mujoco/task_manager.py``
compares against, and the boundary tests pin the same comparison *directions*
the simulator uses, so the two verdicts cannot silently drift apart.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from robot_console.arm.success import (
    GEOMETRIC_SUCCESS_KEY,
    HOLD_SECONDS,
    STAMP_KEY,
    HoldTracker,
    PlateGoal,
    final_hold,
    success_info,
)

GOAL = PlateGoal()
#: An apple resting dead centre on the plate, at rest, having travelled.
RESTING = [0.226, -0.226, 0.0404]


def test_defaults_are_the_contract_numbers() -> None:
    assert GOAL.center_xy == (0.226, -0.226)
    assert GOAL.radius == 0.080
    assert GOAL.center_z == 0.040
    assert GOAL.z_tolerance == 0.015
    assert GOAL.max_speed == 0.01
    assert GOAL.spawn_xyz == (0.30, 0.10, 0.020)
    assert GOAL.min_displacement == 0.25
    assert HOLD_SECONDS == 1.0


def test_a_settled_apple_on_the_plate_passes() -> None:
    assert GOAL.is_placed(RESTING, 0.0)
    assert GOAL.explain(RESTING, 0.0) == "ok"


def test_clause_1_horizontal_distance_is_strict() -> None:
    just_inside = [0.226 + 0.0799, -0.226, 0.0404]
    just_outside = [0.226 + 0.0801, -0.226, 0.0404]
    assert GOAL.is_placed(just_inside, 0.0)
    assert not GOAL.is_placed(just_outside, 0.0)
    assert "horizontal" in GOAL.explain(just_outside, 0.0)


def test_clause_2_height_band_is_inclusive() -> None:
    assert GOAL.is_placed([0.226, -0.226, 0.055], 0.0)
    assert not GOAL.is_placed([0.226, -0.226, 0.0551], 0.0)
    assert "z " in GOAL.explain([0.226, -0.226, 0.0551], 0.0)


def test_clause_2_rejects_an_apple_still_held_above_the_plate() -> None:
    # The old height band ran to z = 0.095, which accepted a 55 mm hover.
    assert not GOAL.is_placed([0.226, -0.226, 0.090], 0.0)


def test_clause_3_requires_the_apple_to_be_at_rest() -> None:
    assert GOAL.is_placed(RESTING, 0.0099)
    assert not GOAL.is_placed(RESTING, 0.01)
    assert "speed" in GOAL.explain(RESTING, 0.5)


def test_clause_3_treats_an_unobserved_twist_as_not_at_rest() -> None:
    assert not GOAL.is_placed(RESTING, None)
    assert "twist" in GOAL.explain(RESTING, None)


def _on_plate_toward_spawn(offset: float) -> list[float]:
    """A resting point ``offset`` metres from the plate centre, towards the spawn."""
    dx = GOAL.spawn_xyz[0] - GOAL.center_xy[0]
    dy = GOAL.spawn_xyz[1] - GOAL.center_xy[1]
    norm = math.hypot(dx, dy)
    return [
        GOAL.center_xy[0] + offset * dx / norm,
        GOAL.center_xy[1] + offset * dy / norm,
        0.0404,
    ]


def test_clause_5_requires_the_apple_to_have_travelled() -> None:
    assert GOAL.displacement(RESTING) > 0.25
    # No point the moved plate's clause-1 disc admits can fail clause 5 any more
    # (see below), so the clause has to be exercised against a goal that asks
    # for more travel than this scene can supply.
    strict = replace(GOAL, min_displacement=0.40)
    assert not strict.is_placed(RESTING, 0.0)
    assert "displacement" in strict.explain(RESTING, 0.0)


def test_clause_1_now_implies_clause_5_at_the_moved_plate() -> None:
    """The plate move inverted the relationship between clauses 1 and 5.

    At ``(0.40, -0.20)`` the spawn was 0.3162 m from the plate centre, so the
    near part of the clause-1 disc had not travelled 0.25 m: the two clauses
    together were tighter than clause 1 alone, 0.065 m towards the spawn passed
    and 0.070 m did not, and the scripted release point had to be biased away
    from the spawn's side of the plate to stay clear of clause 5.

    At ``(0.226, -0.226)`` the spawn is 0.3343 m away and the whole disc clears
    clause 5. The near edge clears it by only 5.1 mm, which is why the release
    point is still biased -- but clause 1 is now the binding constraint
    everywhere, and clause 5 does not bite until 0.0851 m, which clause 1 has
    already rejected.
    """
    near_edge = _on_plate_toward_spawn(0.0799)
    assert GOAL.horizontal_distance(near_edge) < GOAL.radius
    assert GOAL.is_placed(near_edge, 0.0)

    # The whole clause-1 disc has travelled far enough, with 5.1 mm to spare.
    slack = GOAL.displacement(_on_plate_toward_spawn(GOAL.radius)) - GOAL.min_displacement
    assert slack == pytest.approx(0.0051, abs=2e-4)

    # Where clause 5 would start to bite, clause 1 already has.
    beyond = _on_plate_toward_spawn(0.0852)
    assert GOAL.displacement(beyond) < GOAL.min_displacement
    assert GOAL.horizontal_distance(beyond) > GOAL.radius


def test_success_info_reports_every_measured_quantity() -> None:
    info = success_info(RESTING, goal=GOAL, apple_speed=0.001, stamp=12.5, sim_success=True)
    assert info[GEOMETRIC_SUCCESS_KEY] is True
    assert info[STAMP_KEY] == 12.5
    assert info["apple_speed"] == 0.001
    assert math.isclose(info["distance"], 0.0, abs_tol=1e-12)
    assert info["apple_displacement"] > 0.25


def test_success_info_survives_an_unobserved_apple() -> None:
    info = success_info(None, goal=GOAL)
    assert info[GEOMETRIC_SUCCESS_KEY] is False
    assert info["apple_position"] is None
    assert info["distance"] == float("inf")


def test_hold_needs_a_full_second_of_simulated_time() -> None:
    tracker = HoldTracker(control_hz=10.0)
    stamps = [i * 0.1 for i in range(12)]
    verdicts = [tracker.update(placed=True, stamp=t) for t in stamps]
    # First placed step is t=0.0, so the hold completes at t=1.0, the 11th step.
    assert verdicts[:10] == [False] * 10
    assert verdicts[10] is True
    assert tracker.elapsed >= 1.0


def test_hold_restarts_when_the_apple_moves_again() -> None:
    tracker = HoldTracker(control_hz=10.0)
    for t in [0.0, 0.3, 0.6, 0.9]:
        assert tracker.update(placed=True, stamp=t) is False
    assert tracker.update(placed=False, stamp=1.0) is False
    assert tracker.elapsed == 0.0
    # The clock restarts from here, so 1.1 is not yet a second of holding.
    assert tracker.update(placed=True, stamp=1.1) is False
    assert tracker.update(placed=True, stamp=2.1) is True


def test_hold_falls_back_to_step_counting_without_a_clock() -> None:
    tracker = HoldTracker(control_hz=10.0)
    assert tracker.fallback_steps == 10
    verdicts = [tracker.update(placed=True, stamp=None) for _ in range(10)]
    assert verdicts[:9] == [False] * 9
    assert verdicts[9] is True


def test_final_hold_reads_the_trailing_run_back_out_of_a_log() -> None:
    infos = [
        {GEOMETRIC_SUCCESS_KEY: False, STAMP_KEY: 0.0},
        {GEOMETRIC_SUCCESS_KEY: True, STAMP_KEY: 0.1},  # a bounce, not the end
        {GEOMETRIC_SUCCESS_KEY: False, STAMP_KEY: 0.2},
        *[{GEOMETRIC_SUCCESS_KEY: True, STAMP_KEY: 0.3 + 0.1 * i} for i in range(12)],
    ]
    span = final_hold(infos)
    assert span.steps == 12
    assert span.seconds is not None and span.seconds >= 1.0
    assert span.satisfied()


def test_final_hold_rejects_a_single_true_final_step() -> None:
    infos = [
        {GEOMETRIC_SUCCESS_KEY: False, STAMP_KEY: 0.0},
        {GEOMETRIC_SUCCESS_KEY: True, STAMP_KEY: 0.1},
    ]
    span = final_hold(infos)
    assert span.steps == 1
    assert span.seconds == 0.0
    assert not span.satisfied()


def test_final_hold_counts_steps_when_the_log_carried_no_stamps() -> None:
    infos = [{GEOMETRIC_SUCCESS_KEY: True} for _ in range(4)]
    span = final_hold(infos)
    assert span.steps == 4
    assert span.seconds is None
    assert span.satisfied(fallback_steps=3)
    assert not span.satisfied(fallback_steps=5)


def test_for_apple_reproduces_the_contract_gate_at_the_contract_radius() -> None:
    assert PlateGoal.for_apple() == GOAL
