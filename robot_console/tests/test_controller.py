"""The path follower: carrot chasing, braking, arrival and stuck detection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_console.slam.controller import (
    ARRIVAL_M,
    SLOW_M,
    STOP_M,
    PathFollower,
    carrot,
    scale_to_limits,
)
from robot_console.slam.scan import LaserScan
from robot_console.teleop import SPEED_MAX, TURN_MAX, Command

STRAIGHT = np.array([[0.0, 0.0], [5.0, 0.0]])


def clear_scan(distance: float = 10.0) -> LaserScan:
    return LaserScan(
        ranges=np.full(360, distance), angle_min=-math.pi,
        angle_increment=2 * math.pi / 360, range_min=0.1, range_max=12.0,
    )


def scan_with_obstacle_ahead(distance: float) -> LaserScan:
    ranges = np.full(360, 10.0)
    ranges[175:186] = distance   # a wedge around bearing 0
    return LaserScan(
        ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / 360,
        range_min=0.1, range_max=12.0,
    )


def test_the_command_points_down_the_path():
    result = PathFollower(speed=0.2).step((0.0, 0.0, 0.0), STRAIGHT)
    assert result.command.vx > 0
    assert result.command.vy == pytest.approx(0.0, abs=1e-9)
    assert not result.arrived


def test_a_sideways_goal_produces_a_real_strafe():
    """The base is Mecanum, so linear.y is a genuine output, not a rounding artefact."""
    result = PathFollower(speed=0.2).step((0.0, 0.0, 0.0), np.array([[0.0, 0.0], [0.0, 5.0]]))
    assert result.command.vy > 0.05
    assert result.command.wz > 0, "and it should still turn to face where it is going"


def test_the_command_is_in_the_body_frame():
    """Facing +y, a goal at +x world must come out as a negative strafe."""
    result = PathFollower(speed=0.2).step((0.0, 0.0, math.pi / 2), STRAIGHT)
    assert result.command.vx == pytest.approx(0.0, abs=1e-6)
    assert result.command.vy < -0.05


def test_arrival_stops_the_robot():
    result = PathFollower().step((5.0, 0.0, 0.0), STRAIGHT)
    assert result.arrived
    assert result.command.is_zero()


def test_arrival_uses_the_declared_tolerance():
    follower = PathFollower()
    assert not follower.step((5.0 - ARRIVAL_M - 0.05, 0.0, 0.0), STRAIGHT).arrived
    assert follower.step((5.0 - ARRIVAL_M + 0.05, 0.0, 0.0), STRAIGHT).arrived


def test_speed_eases_off_on_the_approach():
    follower = PathFollower(speed=0.25)
    far = follower.step((0.0, 0.0, 0.0), STRAIGHT).command
    near = follower.step((4.6, 0.0, 0.0), STRAIGHT).command
    assert math.hypot(near.vx, near.vy) < math.hypot(far.vx, far.vy)


def test_an_obstacle_inside_the_stop_distance_halts_and_asks_for_a_replan():
    result = PathFollower(speed=0.2).step(
        (0.0, 0.0, 0.0), STRAIGHT, scan=scan_with_obstacle_ahead(STOP_M - 0.05)
    )
    assert result.command.is_zero()
    assert result.blocked and result.should_replan


def test_an_obstacle_at_middle_distance_only_slows_it_down():
    follower = PathFollower(speed=0.25)
    fast = follower.step((0.0, 0.0, 0.0), STRAIGHT, scan=clear_scan()).command
    slow = follower.step(
        (0.0, 0.0, 0.0), STRAIGHT, scan=scan_with_obstacle_ahead((SLOW_M + STOP_M) / 2)
    ).command
    assert 0 < math.hypot(slow.vx, slow.vy) < math.hypot(fast.vx, fast.vy)


def test_an_obstacle_behind_is_ignored():
    ranges = np.full(360, 10.0)
    ranges[0] = 0.1   # bearing -pi, directly behind
    scan = LaserScan(ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / 360,
                     range_min=0.1, range_max=12.0)
    assert not PathFollower(speed=0.2).step((0.0, 0.0, 0.0), STRAIGHT, scan=scan).blocked


def test_no_path_means_no_motion():
    assert PathFollower().step((0.0, 0.0, 0.0), None).command.is_zero()
    assert PathFollower().step((0.0, 0.0, 0.0), np.empty((0, 2))).command.is_zero()


def test_the_command_stays_inside_the_hardware_envelope():
    follower = PathFollower(speed=SPEED_MAX)
    for pose in [(0, 0, 0), (1, 1, 2.0), (2, -1, -1.5), (4.9, 0.2, 3.0)]:
        c = scale_to_limits(follower.step(pose, STRAIGHT).command)
        assert math.hypot(c.vx, c.vy) <= SPEED_MAX + 1e-9
        assert abs(c.wz) <= TURN_MAX + 1e-9


def test_speed_is_capped_at_construction():
    assert PathFollower(speed=99.0).speed == SPEED_MAX


def test_stuck_detection_fires_only_after_a_stall():
    follower = PathFollower(speed=0.2)
    follower.step((0.0, 0.0, 0.0), STRAIGHT, now=0.0)
    follower.step((1.0, 0.0, 0.0), STRAIGHT, now=1.0)
    assert not follower.is_stuck(now=2.0)
    # No further progress for a long time.
    for t in range(3, 12):
        follower.step((1.0, 0.0, 0.0), STRAIGHT, now=float(t))
    assert follower.is_stuck(now=11.0)


def test_reset_clears_the_stuck_history():
    follower = PathFollower(speed=0.2)
    for t in range(0, 12):
        follower.step((1.0, 0.0, 0.0), STRAIGHT, now=float(t))
    assert follower.is_stuck(now=11.0)
    follower.reset()
    assert not follower.is_stuck(now=11.0)


# ------------------------------------------------------------------ carrot


def test_the_carrot_sits_a_lookahead_along_the_path():
    assert carrot(STRAIGHT, np.array([0.0, 0.0]), 1.0) == pytest.approx([1.0, 0.0])
    assert carrot(STRAIGHT, np.array([2.0, 0.0]), 1.5) == pytest.approx([3.5, 0.0])


def test_the_carrot_projects_onto_the_path_rather_than_snapping_to_a_waypoint():
    """Snapping to the nearest waypoint makes the base cut corners after simplify()."""
    assert carrot(STRAIGHT, np.array([2.0, 0.6]), 1.0) == pytest.approx([3.0, 0.0])


def test_the_carrot_stops_at_the_end_of_the_path():
    assert carrot(STRAIGHT, np.array([4.9, 0.0]), 5.0) == pytest.approx([5.0, 0.0])


def test_the_carrot_follows_a_corner():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    assert carrot(path, np.array([0.0, 0.0]), 3.0) == pytest.approx([2.0, 1.0])


def test_a_single_point_path_is_its_own_carrot():
    assert carrot(np.array([[1.0, 2.0]]), np.array([0.0, 0.0]), 1.0) == pytest.approx([1.0, 2.0])


# ------------------------------------------------------------------ limiting


def test_scaling_preserves_the_commanded_direction():
    """Clipping the axes separately would rotate a fast diagonal into a slow curve."""
    scaled = scale_to_limits(Command(vx=0.8, vy=0.6, wz=0.0), speed_max=0.28)
    assert math.hypot(scaled.vx, scaled.vy) == pytest.approx(0.28)
    assert scaled.vy / scaled.vx == pytest.approx(0.6 / 0.8)


def test_scaling_leaves_a_legal_command_alone():
    assert scale_to_limits(Command(vx=0.1, vy=0.0, wz=0.2)) == Command(vx=0.1, vy=0.0, wz=0.2)


def test_scaling_clamps_the_turn_rate():
    assert scale_to_limits(Command(wz=99.0)).wz == pytest.approx(TURN_MAX)


# ------------------------------------------------------- holonomic obstacle checking


def scan_with_obstacle_at(bearing_deg: float, distance: float) -> LaserScan:
    ranges = np.full(360, 10.0)
    i = int(round((math.radians(bearing_deg) + math.pi) / (2 * math.pi / 360))) % 360
    ranges[(i - 5) % 360: (i + 6) % 360 or 360] = distance
    return LaserScan(ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / 360,
                     range_min=0.1, range_max=12.0)


def test_an_obstacle_ahead_does_not_block_a_strafe_away_from_it():
    """The base is Mecanum. Braking for what is in front while driving sideways is what
    wedges it permanently anywhere something happens to sit ahead of it."""
    sideways = np.array([[0.0, 0.0], [0.0, 5.0]])   # goal is to the robot's left
    result = PathFollower(speed=0.2).step(
        (0.0, 0.0, 0.0), sideways, scan=scan_with_obstacle_at(0.0, STOP_M - 0.1)
    )
    assert not result.blocked
    assert result.command.vy > 0.05


def test_an_obstacle_in_the_direction_of_travel_does_block():
    sideways = np.array([[0.0, 0.0], [0.0, 5.0]])
    result = PathFollower(speed=0.2).step(
        (0.0, 0.0, 0.0), sideways, scan=scan_with_obstacle_at(90.0, STOP_M - 0.1)
    )
    assert result.blocked and result.command.is_zero()


def test_reversing_away_from_an_obstacle_is_allowed():
    backwards = np.array([[0.0, 0.0], [-5.0, 0.0]])
    result = PathFollower(speed=0.2).step(
        (0.0, 0.0, 0.0), backwards, scan=scan_with_obstacle_at(0.0, STOP_M - 0.1)
    )
    assert not result.blocked
    assert result.command.vx < -0.05


def test_the_brake_follows_the_robots_heading_not_the_world():
    """Facing +y, a goal at +y world is bearing 0 in the body frame."""
    forward_in_world_y = np.array([[0.0, 0.0], [0.0, 5.0]])
    result = PathFollower(speed=0.2).step(
        (0.0, 0.0, math.pi / 2), forward_in_world_y, scan=scan_with_obstacle_at(0.0, STOP_M - 0.1)
    )
    assert result.blocked
