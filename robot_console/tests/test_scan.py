"""LaserScan parsing and the base-frame projection."""

from __future__ import annotations

import math

import numpy as np
import pytest
from synthetic import raycast, scan_message

from robot_console.slam.scan import (
    DEFAULT_RANGE_MAX,
    DEFAULT_RANGE_MIN,
    LASER_OFFSET_X,
    LaserScan,
    nearest_obstacle,
    parse_scan,
    scan_points,
    transform_points,
)


def test_parse_scan_reads_a_rosbridge_message():
    scan = parse_scan(scan_message(1.0, 1.0))
    assert scan.count == 360
    assert scan.frame_id == "laser_frame"
    assert scan.stamp == pytest.approx(100.5)
    assert scan.angle_min == pytest.approx(-math.pi)
    assert scan.angle_increment == pytest.approx(2 * math.pi / 360)


def test_parse_scan_matches_the_hardware_defaults():
    """The X2's own launch parameters, so a message that omits them still reads sanely."""
    scan = parse_scan({"ranges": [1.0, 2.0]})
    assert scan.range_min == DEFAULT_RANGE_MIN == 0.1
    assert scan.range_max == DEFAULT_RANGE_MAX == 12.0


@pytest.mark.parametrize(
    "message",
    [
        {},
        {"ranges": None},
        {"ranges": "not a list"},
        {"ranges": [1.0, None, "x", 2.0]},
        None,
        [1, 2, 3],
    ],
)
def test_parse_scan_never_raises(message):
    """The loop that parses scans is the loop keeping the robot's command stream alive."""
    scan = parse_scan(message)
    assert isinstance(scan, LaserScan)
    assert np.isfinite(scan.ranges[scan.valid]).all()


def test_angle_increment_is_derived_when_missing():
    scan = parse_scan({"ranges": [1.0] * 5, "angle_min": -1.0, "angle_max": 1.0})
    assert scan.angle_increment == pytest.approx(0.5)


@pytest.mark.parametrize(
    "value, why",
    [
        (0.0, "ydlidar with invalid_range_is_inf:=false reports 0.0"),
        (13.0, "the simulator sends range_max + 1, because JSON has no inf"),
        (float("inf"), "a stock ROS driver reports inf"),
        (float("nan"), "corruption"),
        (0.05, "below range_min"),
    ],
)
def test_every_no_return_convention_is_rejected(value, why):
    scan = LaserScan(
        ranges=np.array([1.0, value, 2.0]), angle_min=-math.pi,
        angle_increment=0.1, range_min=0.1, range_max=12.0,
    )
    assert list(scan.valid) == [True, False, True], why
    assert scan_points(scan, offset=(0.0, 0.0)).shape[0] == 2


def test_scan_points_puts_a_forward_return_ahead_of_the_robot():
    scan = LaserScan(
        ranges=np.array([2.0]), angle_min=0.0, angle_increment=0.0,
        range_min=0.1, range_max=12.0,
    )
    (x, y), = scan_points(scan, offset=(0.0, 0.0))
    assert (x, y) == pytest.approx((2.0, 0.0))


def test_scan_points_applies_the_laser_mount_offset():
    """65 mm is more than a whole map cell; leaving it out smears every wall."""
    scan = LaserScan(
        ranges=np.array([2.0]), angle_min=0.0, angle_increment=0.0,
        range_min=0.1, range_max=12.0,
    )
    (x, y), = scan_points(scan)
    assert x == pytest.approx(2.0 + LASER_OFFSET_X)
    assert y == pytest.approx(0.0)
    assert LASER_OFFSET_X == 0.065, "must match myagv_active.launch's static transform"


def test_max_range_trims_long_returns():
    scan = raycast(1.0, 1.0)
    assert scan_points(scan, max_range=2.0).shape[0] < scan_points(scan).shape[0]
    assert np.all(np.linalg.norm(scan_points(scan, max_range=2.0, offset=(0.0, 0.0)), axis=1) <= 2.0)


def test_transform_points_rotates_then_translates():
    pts = np.array([[1.0, 0.0]])
    out = transform_points(pts, (2.0, 3.0, math.pi / 2))
    assert out[0] == pytest.approx([2.0, 4.0])


def test_transform_points_survives_an_empty_cloud():
    assert transform_points(np.empty((0, 2)), (1.0, 2.0, 0.3)).shape == (0, 2)


def test_a_scan_from_a_known_spot_reconstructs_the_room():
    """The whole pipeline: ray-cast, project, transform, and land on the walls."""
    from synthetic import wall_distance

    pose = (1.5, 2.0, 0.6)
    world = transform_points(scan_points(raycast(*pose), offset=(0.0, 0.0)), pose)
    assert wall_distance(world).max() < 1e-6


def test_nearest_obstacle_looks_forward_only():
    ranges = np.full(360, 5.0)
    ranges[180] = 0.5   # index 180 is bearing 0, straight ahead
    ranges[0] = 0.2     # index 0 is bearing -pi, behind
    scan = LaserScan(
        ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / 360,
        range_min=0.1, range_max=12.0,
    )
    assert nearest_obstacle(scan, math.radians(30)) == pytest.approx(0.5)


def test_nearest_obstacle_is_infinite_with_nothing_in_view():
    assert nearest_obstacle(LaserScan()) == math.inf


def test_nearest_obstacle_can_look_in_any_direction():
    """Holonomic: the useful question is what is in the way, not what is in front."""
    ranges = np.full(360, 5.0)
    ranges[270] = 0.4   # index 270 is bearing +pi/2, to the left
    scan = LaserScan(ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / 360,
                     range_min=0.1, range_max=12.0)
    assert nearest_obstacle(scan, math.radians(30), bearing=math.pi / 2) == pytest.approx(0.4)
    assert nearest_obstacle(scan, math.radians(30), bearing=0.0) == pytest.approx(5.0)


def test_nearest_obstacle_wraps_around_pi():
    ranges = np.full(360, 5.0)
    ranges[2] = 0.3     # just past -pi, i.e. behind
    scan = LaserScan(ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / 360,
                     range_min=0.1, range_max=12.0)
    assert nearest_obstacle(scan, math.radians(20), bearing=math.pi) == pytest.approx(0.3)
