import pytest

from robot_console.teleop import (
    HOLD_TIMEOUT,
    KEY_ESC,
    KEY_SPACE,
    SPEED_MAX,
    SPEED_MIN,
    SPEED_STEP,
    TURN_MAX,
    TURN_RATIO,
    Action,
    Command,
    TeleopState,
    action_for_key,
    clamp_speed,
    key_label,
)


@pytest.mark.parametrize(
    "key, expected",
    [
        (ord("w"), Action.FORWARD),
        (ord("W"), Action.FORWARD),
        (ord("s"), Action.BACK),
        (ord("S"), Action.BACK),
        (ord("a"), Action.STRAFE_LEFT),
        (ord("d"), Action.STRAFE_RIGHT),
        (ord("q"), Action.ROT_LEFT),
        (ord("e"), Action.ROT_RIGHT),
        (KEY_SPACE, Action.STOP),
        (KEY_ESC, Action.QUIT),
        (ord("+"), Action.FASTER),
        (ord("="), Action.FASTER),  # '+' needs shift on most layouts
        (ord("-"), Action.SLOWER),
        (ord("_"), Action.SLOWER),
        (ord("h"), Action.HELP),
        (ord("H"), Action.HELP),
        (ord("?"), Action.HELP),
        (-1, Action.NONE),  # waitKey timeout
        (ord("z"), Action.NONE),
        (ord("1"), Action.NONE),
    ],
)
def test_action_for_key(key, expected):
    assert action_for_key(key) is expected


def test_action_for_key_ignores_high_bits():
    # Some platforms return the key code with high bits set.
    assert action_for_key(0xFFFF00 | ord("w")) is Action.FORWARD


def test_key_label():
    assert key_label(ord("W")) == "w"
    assert key_label(KEY_SPACE) == "space"
    assert key_label(KEY_ESC) == "esc"
    assert key_label(ord("z")) is None
    assert key_label(-1) is None


def test_clamp_speed():
    assert clamp_speed(0.0) == SPEED_MIN
    assert clamp_speed(99.0) == SPEED_MAX
    assert clamp_speed(0.15) == pytest.approx(0.15)


def test_command_to_twist_shape():
    twist = Command(vx=0.15, vy=-0.05, wz=0.3).to_twist()
    assert twist == {
        "linear": {"x": 0.15, "y": -0.05, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.3},
    }


def test_command_is_zero():
    assert Command().is_zero()
    assert not Command(vx=0.01).is_zero()


def test_direction_replaces_rather_than_accumulates():
    # W then D must strafe, not drive diagonally.
    state = TeleopState()
    state.apply(Action.FORWARD)
    state.apply(Action.STRAFE_RIGHT)
    command = state.command()
    assert command.vx == 0.0
    assert command.vy == pytest.approx(-state.speed)


def test_motion_expires_when_the_key_stops_repeating():
    # The whole point: releasing the key must stop the robot.
    state = TeleopState(hold_timeout=0.6)
    state.apply(Action.FORWARD, now=100.0)
    assert not state.expire(now=100.3)
    assert state.command().vx > 0
    assert state.expire(now=100.7)
    assert state.command().is_zero()
    assert not state.is_moving
    assert state.running


def test_expire_only_fires_once_per_release():
    state = TeleopState(hold_timeout=0.6)
    state.apply(Action.FORWARD, now=0.0)
    assert state.expire(now=1.0)
    assert not state.expire(now=2.0)


def test_key_repeat_keeps_the_robot_moving():
    # macOS repeats at ~90 ms after a ~375 ms initial delay; the timeout must bridge
    # that gap or a held key would stutter.
    state = TeleopState(hold_timeout=0.6)
    now = 0.0
    state.apply(Action.FORWARD, now=now)
    for step in (0.375, 0.09, 0.09, 0.09, 0.09):
        now += step
        assert not state.expire(now), f"expired mid-hold at t={now}"
        state.apply(Action.FORWARD, now=now)
    assert state.command().vx > 0
    # Now let go.
    assert state.expire(now + 0.7)
    assert state.command().is_zero()


def test_held_since_tracks_a_continuous_hold():
    state = TeleopState(hold_timeout=0.6)
    state.apply(Action.FORWARD, now=10.0)
    state.apply(Action.FORWARD, now=10.1)
    assert state.held_since == 10.0
    # A different direction restarts the hold.
    state.apply(Action.STRAFE_LEFT, now=10.2)
    assert state.held_since == 10.2


def test_latch_mode_never_expires():
    state = TeleopState(hold_timeout=None)
    state.apply(Action.FORWARD, now=0.0)
    assert not state.expire(now=1000.0)
    assert state.command().vx > 0
    state.apply(Action.STOP, now=1000.0)
    assert state.command().is_zero()


def test_speed_keys_do_not_arm_motion():
    state = TeleopState(hold_timeout=0.6)
    state.apply(Action.FASTER, now=0.0)
    assert not state.is_moving
    assert not state.expire(now=5.0)


def test_help_toggles():
    state = TeleopState()
    assert state.show_help is True
    state.apply(Action.HELP)
    assert state.show_help is False
    state.apply(Action.HELP)
    assert state.show_help is True
    # Toggling hints must not touch the robot.
    assert state.command().is_zero()


def test_hold_timeout_clears_the_os_initial_repeat_delay():
    # macOS waits 375 ms before the first repeat (25 x 15 ms). A timeout at or below
    # that would make every held key stutter.
    assert HOLD_TIMEOUT > 0.375


def test_space_stops_but_keeps_running():
    state = TeleopState()
    state.apply(Action.FORWARD)
    state.apply(Action.STOP)
    assert state.command().is_zero()
    assert state.running


def test_quit_zeroes_and_stops():
    state = TeleopState()
    state.apply(Action.FORWARD)
    state.apply(Action.QUIT)
    assert state.command().is_zero()
    assert not state.running


def test_speed_saturates_at_the_hardware_cap():
    state = TeleopState()
    for _ in range(20):
        state.apply(Action.FASTER)
    assert state.speed == pytest.approx(SPEED_MAX)
    assert state.at_max_speed


def test_speed_floors():
    state = TeleopState()
    for _ in range(20):
        state.apply(Action.SLOWER)
    assert state.speed == pytest.approx(SPEED_MIN)
    assert state.at_min_speed


def test_speed_step():
    state = TeleopState(speed=0.15)
    state.apply(Action.FASTER)
    assert state.speed == pytest.approx(0.15 + SPEED_STEP)


def test_custom_max_speed_is_respected():
    state = TeleopState(speed=0.1, speed_max=0.6)
    for _ in range(50):
        state.apply(Action.FASTER)
    assert state.speed == pytest.approx(0.6)


def test_turn_rate_follows_speed():
    state = TeleopState(speed=0.25)
    state.apply(Action.ROT_LEFT)
    # The vendor teleop pairs speed 0.25 with turn 0.5, which is this ratio.
    assert state.command().wz == pytest.approx(0.5)


def test_turn_rate_is_capped():
    state = TeleopState(speed=2.0, speed_max=2.0)
    state.apply(Action.ROT_LEFT)
    assert state.command().wz == pytest.approx(TURN_MAX)
    assert 2.0 * TURN_RATIO > TURN_MAX  # the cap is what is being exercised


def test_ros_sign_conventions():
    # +x forward, +y left, +z counter-clockwise, matching myagv_teleop's bindings.
    state = TeleopState(speed=0.2)
    assert state.apply(Action.FORWARD) and state.command().vx > 0
    state.apply(Action.STRAFE_LEFT)
    assert state.command().vy > 0
    state.apply(Action.ROT_LEFT)
    assert state.command().wz > 0
    state.apply(Action.BACK)
    assert state.command().vx < 0
    state.apply(Action.STRAFE_RIGHT)
    assert state.command().vy < 0
    state.apply(Action.ROT_RIGHT)
    assert state.command().wz < 0


def test_none_action_leaves_state_alone():
    state = TeleopState()
    state.apply(Action.FORWARD)
    before = state.command()
    state.apply(Action.NONE)
    assert state.command() == before
