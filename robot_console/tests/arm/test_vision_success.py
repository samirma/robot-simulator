"""The camera-verdict scorer on recorded synchronized observations (spec §4, Task grading).

Only the valid released placement passes. The mandatory negatives -- an apple held by the
gripper at the target pose, an apple moving through the target, the wrong height, an
incomplete hold, missing frames and stale joint states -- all fail, and so does anything
the scorer cannot measure. The thresholds are duplicated here on purpose rather than
imported, so changing one is a change made twice.
"""

from __future__ import annotations

import math

import pytest
from inspect_robots import Action, Observation, StepResult
from inspect_robots.rollout import StepRecord, TrialRecord

from robot_console.arm import vision_success as vs
from robot_console.arm.scorer import apple_on_plate

from .rig_fixtures import AWAY, PLATE_XY, RESTING, episode, holding, sample

# The pass criterion, duplicated from the spec (§2.3) -- not imported.
GATE_M, RESTING_Z, Z_TOL, MAX_SPEED, HOLD_S, CLEARANCE_M = 0.08, 0.040, 0.015, 0.01, 1.0, 0.05


def test_the_thresholds_are_the_specs() -> None:
    assert (vs.MAX_HORIZONTAL_DIST_M, vs.RESTING_Z_M, vs.Z_TOLERANCE_M, vs.MAX_SPEED_MPS,
            vs.HOLD_SECONDS, vs.FINGER_CLEARANCE_M) == (GATE_M, RESTING_Z, Z_TOL, MAX_SPEED,
                                                       HOLD_S, CLEARANCE_M)
    assert 0 < vs.SYNC_TOLERANCE_S <= 0.05


def test_the_apple_is_triangulated_from_the_two_views() -> None:
    for truth in (RESTING, (0.30, 0.10, 0.020), (0.20, 0.05, 0.12)):
        m = vs.measure(sample(1.0, truth))
        assert m.problem == ""
        assert math.dist(m.apple, truth) < 0.004, (truth, m.apple)


def test_a_valid_released_placement_passes() -> None:
    verdict = vs.assess(episode(lambda t: RESTING))
    assert verdict.passed, verdict.reason
    assert verdict.hold_s >= HOLD_S
    assert verdict.horizontal_m < GATE_M and verdict.height_error_m < Z_TOL
    assert verdict.clearance_m >= CLEARANCE_M


def test_a_red_bowl_touching_the_apple_in_the_side_view_does_not_hide_it() -> None:
    """Measured on the MolmoSpaces kitchen: one red mask merged the apple with the bowl."""
    s = sample(1.0, RESTING, bowl=True)
    m = vs.measure(s)
    assert m.problem == "" and math.dist(m.apple, RESTING) < 0.004
    assert vs.assess(episode(lambda t: RESTING, bowl=True)).passed


def test_a_placement_that_arrives_and_then_rests_passes() -> None:
    def apple(t: float):
        if t < 1.0:   # carried in from the spawn, then resting for the last 2 s
            return (0.30 + (PLATE_XY[0] - 0.30) * t, 0.10 + (PLATE_XY[1] - 0.10) * t,
                    0.10 - 0.06 * t)
        return RESTING
    assert vs.assess(episode(apple)).passed


def test_an_apple_held_by_the_gripper_at_the_target_pose_fails() -> None:
    grip = holding(RESTING)
    verdict = vs.assess(episode(lambda t: RESTING, joints_at=lambda t: grip))
    assert not verdict.passed
    assert "not released" in verdict.reason


def test_an_apple_moving_through_the_target_fails() -> None:
    # 0.03 m/s straight across the plate centre: inside the gate the whole last second.
    def apple(t: float):
        return (PLATE_XY[0] - 0.045 + 0.03 * t, PLATE_XY[1], RESTING_Z)
    verdict = vs.assess(episode(apple))
    assert not verdict.passed
    assert "moving" in verdict.reason


def test_a_slow_creep_just_under_the_limit_still_passes() -> None:
    def apple(t: float):
        return (PLATE_XY[0] + 0.004 * t, PLATE_XY[1], RESTING_Z)
    assert vs.assess(episode(apple)).passed


def test_the_wrong_height_fails() -> None:
    verdict = vs.assess(episode(lambda t: (PLATE_XY[0], PLATE_XY[1], 0.075)))
    assert not verdict.passed
    assert "resting" in verdict.reason


def test_off_the_plate_fails() -> None:
    verdict = vs.assess(episode(lambda t: (PLATE_XY[0] + 0.12, PLATE_XY[1], RESTING_Z)))
    assert not verdict.passed and "plate centre" in verdict.reason


def test_an_incomplete_hold_fails() -> None:
    # On the plate for only the last 0.6 s; before that still held above it.
    def apple(t: float):
        return RESTING if t >= 2.4 else (PLATE_XY[0], PLATE_XY[1], 0.08)
    grip_until = holding((PLATE_XY[0], PLATE_XY[1], 0.08))
    verdict = vs.assess(episode(apple, joints_at=lambda t: AWAY if t >= 2.4 else grip_until))
    assert not verdict.passed


def test_an_episode_shorter_than_the_hold_fails() -> None:
    verdict = vs.assess(episode(lambda t: RESTING, seconds=0.6))
    assert not verdict.passed and "cover only" in verdict.reason


def test_missing_frames_in_the_hold_fail() -> None:
    samples = episode(lambda t: RESTING)
    del samples[-8:-4]   # a 0.5 s hole in the last second
    verdict = vs.assess(samples)
    assert not verdict.passed and "without a synchronized sample" in verdict.reason


def test_a_missing_side_frame_fails() -> None:
    samples = episode(lambda t: RESTING)
    samples[-3]["side"] = None
    assert not vs.assess(samples).passed


def test_unsynchronized_views_fail() -> None:
    samples = episode(lambda t: RESTING, side_offset=0.1)
    verdict = vs.assess(samples)
    assert not verdict.passed and "apart" in verdict.reason


def test_stale_joint_states_fail() -> None:
    verdict = vs.assess(episode(lambda t: RESTING, joint_offset=-0.4))
    assert not verdict.passed and "joint state" in verdict.reason


def test_missing_joint_state_fails() -> None:
    samples = episode(lambda t: RESTING)
    samples[-1]["joint_state"] = None
    assert not vs.assess(samples).passed


def test_missing_calibration_fails() -> None:
    verdict = vs.assess(episode(lambda t: RESTING, camera_info=None))
    assert not verdict.passed and ("calibration" in verdict.reason or "plate" in verdict.reason)


def test_a_camera_info_that_disagrees_with_the_rig_fails() -> None:
    samples = episode(lambda t: RESTING)
    for s in samples:
        s["camera_info"] = {**s["camera_info"],
                            "side": {**s["camera_info"]["side"], "width": 1280}}
    assert not vs.assess(samples).passed


def test_an_apple_the_cameras_cannot_see_fails() -> None:
    verdict = vs.assess(episode(lambda t: None))
    assert not verdict.passed and "segmented" in verdict.reason


def test_no_observations_fail() -> None:
    assert not vs.assess([]).passed


def test_trimmed_frames_before_the_hold_do_not_matter() -> None:
    samples = episode(lambda t: RESTING, seconds=6.0)
    for s in samples[:20]:
        s["overhead"]["data"] = s["side"]["data"] = None
    assert vs.assess(samples).passed


# ------------------------------------------------------------------ the scorer


def _record(samples: list[dict], per_step: int = 7) -> TrialRecord:
    record = TrialRecord(scene_id="apple-on-plate", epoch=0, seed=0)
    obs = Observation(images={}, state={})
    for t, i in enumerate(range(0, len(samples), per_step)):
        record.steps.append(StepRecord(
            t=t, observation=obs, action=Action(data=[0.0] * 6),
            result=StepResult(observation=obs, info={vs.RIG_SAMPLES_KEY: samples[i:i + per_step]})))
    return record


@pytest.mark.parametrize("passes", [True, False])
def test_the_registered_scorer_grades_the_recorded_trajectory(passes: bool) -> None:
    grip = holding(RESTING)
    samples = episode(lambda t: RESTING, joints_at=lambda t: AWAY if passes else grip)
    record = _record(samples)
    score = apple_on_plate()(record, None)
    assert bool(score.value) is passes
    assert record.metadata["apple_on_plate"]["passed"] is passes
    assert record.metadata["apple_on_plate"]["reason"] == score.explanation
