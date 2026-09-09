import pytest

from robot_console.teleop import (
    HEAD_STEP_MAX_S,
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
    HeadPose,
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


# --- the head, and the arrows that point it -------------------------------------------


@pytest.mark.parametrize(
    "key, expected",
    [
        (63232, Action.HEAD_UP), (63233, Action.HEAD_DOWN),      # Cocoa
        (63234, Action.HEAD_LEFT), (63235, Action.HEAD_RIGHT),
        (65362, Action.HEAD_UP), (65364, Action.HEAD_DOWN),      # GTK / Qt
        (65361, Action.HEAD_LEFT), (65363, Action.HEAD_RIGHT),
        (2490368, Action.HEAD_UP), (2621440, Action.HEAD_DOWN),  # Win32
        (2424832, Action.HEAD_LEFT), (2555904, Action.HEAD_RIGHT),
        (ord("0"), Action.HEAD_CENTRE),
    ],
)
def test_arrow_keys_decode_on_every_backend(key, expected):
    assert action_for_key(key) is expected


def test_an_arrow_is_never_mistaken_for_the_letter_in_its_low_byte():
    """The whole reason `action_for_key` looks at the full code before masking.

    GTK reports Left as 0xFF51, whose low byte is 0x51 -> 'Q' -> ROT_LEFT. Masked first,
    pressing left-arrow would turn the robot instead of the head -- a wrong action, not a
    missing one, which is the kind that gets blamed on the gait.
    """
    assert action_for_key(65361) is Action.HEAD_LEFT
    assert action_for_key(0xFF51 & 0xFF) is Action.ROT_LEFT  # the trap, when unmasked


def test_head_turns_while_a_key_is_held_and_stops_at_its_limit():
    head = HeadPose(pan_limit=0.5, tilt_limit=0.5, rate=1.0)
    assert head.apply(Action.HEAD_LEFT, 0.1)
    assert head.pan == pytest.approx(-0.1)
    for _ in range(20):
        head.apply(Action.HEAD_LEFT, 0.1)
    assert head.pan == pytest.approx(-0.5)
    # At the limit nothing moves, so nothing is published -- the point of the return value.
    assert not head.apply(Action.HEAD_LEFT, 0.1)


def test_head_signs_are_the_vendors_joint_signs_not_the_bases():
    """Measured off the compiled model: `head_pan`'s axis is [0, 0, -1], so **+pan looks
    right** -- the opposite of the base's `+z` counter-clockwise yaw. `head_tilt`'s is
    [0, -1, 0] and +tilt looks up. Written the intuitive way round (left positive, like
    `Q`), the left arrow pointed the camera right: a wrong direction, not a dead key.
    """
    head = HeadPose(pan_limit=2.0, tilt_limit=2.0, rate=1.0)
    head.apply(Action.HEAD_LEFT, 0.1)
    assert head.pan < 0
    head.apply(Action.HEAD_CENTRE, 0.0)
    head.apply(Action.HEAD_RIGHT, 0.1)
    assert head.pan > 0
    head.apply(Action.HEAD_CENTRE, 0.0)
    head.apply(Action.HEAD_UP, 0.1)
    assert head.tilt > 0
    head.apply(Action.HEAD_CENTRE, 0.0)
    head.apply(Action.HEAD_DOWN, 0.1)
    assert head.tilt < 0


def test_a_long_pause_does_not_swing_the_head_across_its_travel():
    """`dt` is wall time since the last arrow, so it has to be capped.

    Without the cap, an arrow pressed a minute after the last one would ask for 60 s of
    travel in a single step and slam the head into its stop.
    """
    head = HeadPose(pan_limit=2.0, tilt_limit=2.0, rate=1.0)
    head.apply(Action.HEAD_LEFT, 60.0)
    assert head.pan == pytest.approx(-HEAD_STEP_MAX_S)


def test_centring_reports_movement_only_when_there_was_some():
    head = HeadPose(pan_limit=1.0, tilt_limit=1.0, rate=1.0)
    assert not head.apply(Action.HEAD_CENTRE, 0.1)
    head.apply(Action.HEAD_UP, 0.1)
    assert head.apply(Action.HEAD_CENTRE, 0.1)
    assert (head.pan, head.tilt) == (0.0, 0.0)


def test_head_actions_do_not_drive_the_base():
    """The arrows point a camera; they must not arm motion or disarm what is armed."""
    state = TeleopState(speed=0.2)
    state.apply(Action.FORWARD)
    before = state.command()
    for action in (Action.HEAD_LEFT, Action.HEAD_UP, Action.HEAD_CENTRE):
        state.apply(action)
    assert state.command() == before
