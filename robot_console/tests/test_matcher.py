"""Scan matching, and the pose tracker that decides when to run it."""

from __future__ import annotations

import math

import numpy as np
import pytest
from synthetic import mapped_room, raycast

from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.matcher import LikelihoodField, field_for, match
from robot_console.slam.pose import PoseTracker, compose, relative
from robot_console.slam.scan import scan_points


def body_points(pose):
    return scan_points(raycast(*pose), max_range=8.0, offset=(0.0, 0.0))


@pytest.fixture(scope="module")
def room():
    return mapped_room()


@pytest.fixture(scope="module")
def field(room):
    return LikelihoodField(room)


def test_a_perfect_prior_scores_high(field):
    pose = (2.0, 2.0, 0.4)
    assert match(field, body_points(pose), pose).score > 0.8


@pytest.mark.parametrize(
    "offset",
    [(0.10, 0.0, 0.0), (0.0, -0.12, 0.0), (0.0, 0.0, 0.08), (0.15, 0.12, -0.07)],
)
def test_a_perturbed_prior_is_pulled_back(field, offset):
    truth = np.array([2.0, 2.0, 0.4])
    result = match(field, body_points(truth), truth + np.array(offset))

    assert result.accepted
    prior_error = float(np.hypot(*np.asarray(offset[:2])))
    error = float(np.hypot(*(result.pose[:2] - truth[:2])))
    assert error < max(0.06, prior_error * 0.6), "the match must beat the prior it started from"
    assert abs(result.pose[2] - truth[2]) < 0.05


def test_a_scan_of_noise_is_rejected(field):
    rng = np.random.default_rng(7)
    noise = rng.uniform(-6.0, 6.0, size=(200, 2))
    result = match(field, noise, (2.0, 2.0, 0.0))
    assert not result.accepted
    assert tuple(result.pose) == (2.0, 2.0, 0.0), "a rejected match must leave the prior alone"


def test_too_few_beams_is_rejected(field):
    """Three points cannot constrain three degrees of freedom without fitting noise."""
    result = match(field, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), (2.0, 2.0, 0.0))
    assert not result.accepted


def test_an_empty_map_rejects_everything():
    grid = OccupancyGrid(0.05, width=64, height=64)
    result = match(LikelihoodField(grid), body_points((1.0, 1.0, 0.0)), (1.0, 1.0, 0.0))
    assert not result.accepted, "the first scan has nothing to match against"


def test_the_field_is_cached_until_the_map_changes(room):
    first = field_for(room, None)
    assert field_for(room, first) is first
    grown = room.copy()
    grown.integrate((1.0, 1.0, 0.0), np.array([[1.5, 1.0]]))
    assert field_for(grown, first) is not first


# ------------------------------------------------------------------ frame algebra


@pytest.mark.parametrize(
    "correction, odom",
    [((0, 0, 0), (1, 2, 0.3)), ((1, -2, 0.5), (0, 0, 0)), ((-3, 4, -1.1), (2, 1, 0.7))],
)
def test_relative_inverts_compose(correction, odom):
    target = compose(correction, odom)
    assert compose(relative(target, odom), odom) == pytest.approx(target)


def test_compose_rotates_the_odom_frame():
    assert compose((0.0, 0.0, math.pi / 2), (1.0, 0.0, 0.0)) == pytest.approx([0.0, 1.0, math.pi / 2])


# ------------------------------------------------------------------ PoseTracker


def test_the_pose_follows_odometry_when_uncorrected():
    tracker = PoseTracker()
    tracker.update_odom((1.0, 2.0, 0.3))
    assert tracker.pose == pytest.approx([1.0, 2.0, 0.3])


def test_seeding_places_the_robot_on_a_loaded_map():
    """A loaded map's frame has nothing to do with wherever the robot's driver booted."""
    tracker = PoseTracker()
    tracker.update_odom((5.0, 5.0, 0.0))
    tracker.seed((1.0, 1.0, math.pi))
    assert tracker.pose == pytest.approx([1.0, 1.0, math.pi])
    tracker.update_odom((6.0, 5.0, 0.0))
    assert tracker.pose == pytest.approx([0.0, 1.0, math.pi]), "moving +x in odom is -x here"


def test_a_keyframe_is_due_only_after_real_motion():
    tracker = PoseTracker(min_interval=0.0)
    tracker.update_odom((0.0, 0.0, 0.0))
    assert not tracker.keyframe_due(now=1.0)
    tracker.update_odom((0.05, 0.0, 0.0))
    assert not tracker.keyframe_due(now=2.0), "5 cm is one cell; nothing new to say"
    tracker.update_odom((0.30, 0.0, 0.0))
    assert tracker.keyframe_due(now=3.0)


def test_a_keyframe_is_due_after_turning_on_the_spot():
    tracker = PoseTracker(min_interval=0.0)
    tracker.update_odom((0.0, 0.0, 0.0))
    tracker.update_odom((0.0, 0.0, 0.5))
    assert tracker.keyframe_due(now=1.0)


def test_the_minimum_interval_is_enforced():
    """This is the budget that keeps matching off the 20 Hz command path."""
    tracker = PoseTracker(min_interval=1.0)
    tracker.update_odom((0.0, 0.0, 0.0))
    tracker.update_odom((5.0, 0.0, 0.0))
    assert not tracker.keyframe_due(now=0.5)
    assert tracker.keyframe_due(now=1.5)


def test_matching_can_be_switched_off_entirely():
    tracker = PoseTracker(match_enabled=False, min_interval=0.0)
    tracker.update_odom((0.0, 0.0, 0.0))
    tracker.update_odom((5.0, 0.0, 0.0))
    assert not tracker.keyframe_due(now=10.0)


def test_refine_corrects_a_drifted_estimate(room):
    tracker = PoseTracker(min_interval=0.0)
    truth = np.array([2.0, 2.0, 0.4])
    tracker.update_odom(truth + np.array([0.14, -0.10, 0.06]))

    before = float(np.hypot(*(tracker.pose[:2] - truth[:2])))
    result = tracker.refine(room, body_points(truth), now=1.0)
    after = float(np.hypot(*(tracker.pose[:2] - truth[:2])))

    assert result.accepted
    assert after < before
    assert tracker.stats.matches == 1 and tracker.stats.rejected == 0


def test_a_rejected_refine_leaves_the_estimate_alone(room):
    tracker = PoseTracker(min_interval=0.0)
    tracker.update_odom((2.0, 2.0, 0.0))
    rng = np.random.default_rng(3)
    tracker.refine(room, rng.uniform(-6, 6, size=(200, 2)), now=1.0)
    assert tracker.stats.rejected == 1
    assert tracker.pose == pytest.approx([2.0, 2.0, 0.0])


def test_correction_persists_across_odom_updates(room):
    tracker = PoseTracker(min_interval=0.0)
    truth = np.array([2.0, 2.0, 0.4])
    tracker.update_odom(truth + np.array([0.12, 0.0, 0.0]))
    tracker.refine(room, body_points(truth), now=1.0)
    corrected = tracker.pose.copy()

    tracker.update_odom(truth + np.array([0.12 + 0.5, 0.0, 0.0]))
    assert tracker.pose[0] == pytest.approx(corrected[0] + 0.5, abs=1e-9)
