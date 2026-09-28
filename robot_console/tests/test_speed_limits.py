"""Speed limits (console spec §4): for each robot, a `--speed` or `--max-speed` above its
hardware cap is refused, and no key sequence makes teleop publish a speed above the cap.

The caps are duplicated here from console spec §2.1 rather than imported, so a constant
drifting in the code fails a test instead of moving the limit.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time

import pytest

from robot_console.cli import SpeedLimitError, parse_args
from robot_console.robots import PROFILES
from robot_console.supervisor import Supervisor
from robot_console.teleop import Action, Command, TeleopState, within_caps

#: Console spec §2.1: each robot's hardware limit, m/s, and its rotation cap, rad/s.
CAPS = {"myagv": 0.28, "myagv_mycobot280": 0.28, "rosmaster_x3_plus": 0.70, "ainex": 0.20}
TURN_CAPS = {"rosmaster_x3_plus": 3.2, "ainex": 1.0}


def test_the_caps_are_the_specs() -> None:
    for robot, cap in CAPS.items():
        assert PROFILES[robot].speed_max == cap, robot
    for robot, cap in TURN_CAPS.items():
        assert PROFILES[robot].turn_max == cap, robot


# ------------------------------------------------------------------ the flags


@pytest.mark.parametrize("robot", sorted(CAPS))
@pytest.mark.parametrize("flag", ["--speed", "--max-speed"])
def test_a_speed_above_the_cap_is_refused_naming_the_limit(robot, flag, capsys) -> None:
    above = CAPS[robot] + 0.01
    with pytest.raises(SystemExit) as err:
        parse_args(["--robot", robot, flag, f"{above:g}"])
    assert err.value.code == 2
    message = capsys.readouterr().err
    assert f"{flag} {above:g} m/s is above" in message
    assert f"of {CAPS[robot]:g} m/s" in message


@pytest.mark.parametrize("robot", sorted(CAPS))
def test_the_cap_itself_and_anything_below_are_accepted(robot) -> None:
    cap = CAPS[robot]
    options = parse_args(["--robot", robot, "--speed", f"{cap:g}"])
    assert (options.speed, options.max_speed) == (pytest.approx(cap), pytest.approx(cap))
    lower = parse_args(["--robot", robot, "--max-speed", f"{cap / 2:g}"])
    assert lower.max_speed == pytest.approx(cap / 2) and lower.speed <= cap / 2


def test_max_speed_may_only_lower_the_cap_and_speed_must_fit_under_it(capsys) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--robot", "myagv", "--speed", "0.2", "--max-speed", "0.1"])
    assert "--speed 0.2 m/s is above --max-speed 0.1 m/s" in capsys.readouterr().err


def test_before_discovery_a_speed_no_robot_allows_is_refused_at_once(capsys) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--speed", "0.9"])
    assert "above every teleop robot's hardware limit" in capsys.readouterr().err


@pytest.mark.parametrize("robot", ["myagv", "myagv_mycobot280", "ainex"])
def test_the_robot_the_wire_names_refuses_what_it_cannot_do(robot) -> None:
    """0.5 m/s is within the X3 PLUS's limit, so it waits for discovery -- and is then
    refused, not clamped, for a robot whose limit is lower."""
    options = parse_args(["--speed", "0.5"])
    with pytest.raises(SpeedLimitError, match=f"of {CAPS[robot]:g} m/s"):
        options.resolved(robot, robot)
    assert parse_args(["--speed", "0.5"]).resolved(
        "rosmaster_x3_plus", "rosmaster_x3_plus").speed == pytest.approx(0.5)


# ------------------------------------------------------------------ the keys


@pytest.mark.parametrize("robot", sorted(CAPS))
def test_no_key_sequence_goes_above_the_cap(robot) -> None:
    profile = PROFILES[robot]
    options = parse_args(["--robot", robot])
    state = TeleopState(speed=options.speed, speed_max=options.max_speed,
                        speed_min=profile.speed_min, speed_step=profile.speed_step,
                        turn_ratio=profile.turn_ratio, turn_max=profile.turn_max)
    now = 0.0
    for action in [Action.FASTER] * 50 + [Action.FORWARD, Action.STRAFE_LEFT, Action.ROT_LEFT]:
        now += 0.05
        state.apply(action, now)
        command = state.command()
        assert math.hypot(command.vx, command.vy) <= CAPS[robot] + 1e-9
        assert abs(command.wz) <= profile.turn_max + 1e-9
    assert state.speed == pytest.approx(CAPS[robot])


def test_the_supervisor_scales_anything_above_the_caps_down() -> None:
    """The last line of defence: whatever the UI sends, the supervisor publishes at most
    the caps, in the same direction."""
    capped = within_caps(Command(vx=0.6, vy=0.8, wz=-9.0), speed_max=0.28, turn_max=1.0)
    assert math.hypot(capped.vx, capped.vy) == pytest.approx(0.28)
    assert capped.vx / capped.vy == pytest.approx(0.75)
    assert capped.wz == pytest.approx(-1.0)
    same = Command(vx=0.1, vy=0.0, wz=0.2)
    assert within_caps(same, 0.28, 1.0) == same

    class _Out:
        def send(self, _message) -> None:
            pass

    supervisor = Supervisor(None, out=_Out(), inp=iter(()), parent_pid=0,
                            speed_max=0.28, turn_max=1.0)
    supervisor.handle({"op": "cmd", "vx": 5.0, "vy": 0.0, "wz": 0.0})
    assert supervisor._desired.vx == pytest.approx(0.28)


@pytest.mark.parametrize("robot", ["myagv", "rosmaster_x3_plus", "ainex"])
def test_holding_plus_then_driving_never_publishes_above_the_cap(bridge, tmp_path, robot) -> None:
    """End to end: the real UI loop and supervisor against the fake bridge, `+` pressed
    far more often than the cap allows, then a held drive key."""
    from test_motion_safety import AINEX_TOPICS, DRIVER, MYAGV_TOPICS, X3_TOPICS, _env

    bridge.topics = dict({"myagv": MYAGV_TOPICS, "rosmaster_x3_plus": X3_TOPICS,
                          "ainex": AINEX_TOPICS}[robot])
    marker = tmp_path / "marker"
    cmd = [sys.executable, str(DRIVER), "--url", f"ws://127.0.0.1:{bridge.port}",
           "--event", "quit", "--marker", str(marker), "--drive-s", "1.5",
           "--pre-keys", "+" * 40, "--key", "w"]
    proc = subprocess.run(cmd, env=_env(), stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=60)
    assert marker.exists(), proc.stdout + proc.stderr
    if robot == "ainex":
        from robot_console.ainex_link import PERIOD_S, STRIDE_FACTOR

        sent = bridge.received_on("/ainex/walking/set_param")
        speeds = [STRIDE_FACTOR * m["x_move_amplitude"] / PERIOD_S for m in sent]
    else:
        sent = bridge.received_on(f"/{robot}/cmd_vel")
        speeds = [math.hypot(m["linear"]["x"], m["linear"]["y"]) for m in sent]
    assert speeds, "nothing was published"
    assert max(speeds) == pytest.approx(CAPS[robot], abs=1e-6), \
        "the drive reached the cap, and no further"
    assert max(speeds) <= CAPS[robot] + 1e-9
