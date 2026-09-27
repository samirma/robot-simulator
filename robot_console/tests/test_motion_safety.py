"""Motion safety (console spec §4): whatever happens to the UI, the robot gets stopped.

Each test runs the real teleop UI loop in a subprocess (`teleop_driver.py`), with the real
safety supervisor as *its* subprocess, against the in-process fake bridge. The UI drives
the robot, then something happens to it: it freezes, closes its IPC, quits normally,
raises, or is sent SIGINT, SIGTERM or SIGKILL. The fake bridge must then receive the
robot's stop command no later than `--safety-timeout` + 100 ms after the event, three
times in total, and nothing after it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import robot_console

HERE = Path(__file__).resolve().parent
SRC = Path(robot_console.__file__).resolve().parents[1]
DRIVER = HERE / "teleop_driver.py"
SAFETY_TIMEOUT = 0.25
SLACK = 0.10

MYAGV_TOPICS = {
    "/myagv/cmd_vel": "geometry_msgs/Twist",
    "/myagv/odom": "nav_msgs/Odometry",
    "/myagv/camera/image_raw/compressed": "sensor_msgs/CompressedImage",
}
AINEX_TOPICS = {
    "/ainex/walking/set_param": "ainex_interfaces/WalkingParam",
    "/ainex/walking/is_walking": "std_msgs/Bool",
    "/ainex/camera/image_raw/compressed": "sensor_msgs/CompressedImage",
}


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(HERE), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _drive(bridge, tmp_path, event, *, robot="myagv", latch=False, extra=()):
    marker = tmp_path / "marker"
    cmd = [sys.executable, str(DRIVER), "--url", f"ws://127.0.0.1:{bridge.port}",
           "--event", event, "--marker", str(marker),
           "--safety-timeout", str(SAFETY_TIMEOUT), *extra]
    if latch:
        cmd.append("--latch")
    proc = subprocess.Popen(cmd, env=_env(), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 30
    while not marker.exists() or not marker.read_text():
        if proc.poll() is not None or time.monotonic() > deadline:
            out = proc.communicate(timeout=5)[0]
            pytest.fail(f"driver never reached the event (exit {proc.returncode}):\n{out}")
        time.sleep(0.01)
    return proc, float(marker.read_text())


def _finish(proc, timeout=15.0) -> str:
    try:
        return proc.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.communicate()[0]


def _myagv_stops(bridge):
    """(time, twist) of every zero Twist after the last non-zero one, and whether any
    non-zero Twist was published at all."""
    timed = bridge.timed_on("/myagv/cmd_vel")

    def moving(msg):
        return any(abs(float(msg.get(part, {}).get(axis, 0.0))) > 1e-9
                   for part in ("linear", "angular") for axis in ("x", "y", "z"))

    last = max((i for i, (_, m) in enumerate(timed) if moving(m)), default=None)
    assert last is not None, "the UI never got the robot moving"
    return [t for t, _ in timed[last + 1:]]


def _ainex_stops(bridge):
    """Arrival times of each stop command (`enable_control` then `stop`) after the last
    `start`, asserting nothing else was called."""
    calls = bridge.timed_calls_on("/ainex/walking/command")
    commands = [args.get("command") for _, args in calls]
    assert "start" in commands, "the UI never got the robot walking"
    last = len(commands) - 1 - commands[::-1].index("start")
    after = calls[last + 1:]
    assert [a.get("command") for _, a in after] == ["enable_control", "stop"] * (len(after) // 2)
    assert len(after) % 2 == 0
    return [t for t, _ in after[0::2]]


def _check(stops, event_at):
    assert len(stops) == 3, f"expected three stops, got {len(stops)}"
    first = stops[0] - event_at
    assert first <= SAFETY_TIMEOUT + SLACK, f"first stop {first * 1000:.0f} ms after the event"
    gaps = [b - a for a, b in zip(stops, stops[1:])]
    assert all(0.03 <= g <= 0.15 for g in gaps), f"stops not ~50 ms apart: {gaps}"


@pytest.fixture
def myagv_bridge(bridge):
    bridge.topics = dict(MYAGV_TOPICS)
    return bridge


@pytest.fixture
def ainex_bridge(bridge):
    bridge.topics = dict(AINEX_TOPICS)
    return bridge


@pytest.mark.parametrize("latch", [False, True], ids=["held", "latched"])
def test_a_frozen_ui_is_stopped_by_the_heartbeat(myagv_bridge, tmp_path, latch):
    proc, event_at = _drive(myagv_bridge, tmp_path, "freeze", latch=latch)
    time.sleep(SAFETY_TIMEOUT + 1.0)
    try:
        _check(_myagv_stops(myagv_bridge), event_at)
    finally:
        proc.kill()
        _finish(proc)


def test_closing_the_ipc_stops_the_robot(myagv_bridge, tmp_path):
    proc, event_at = _drive(myagv_bridge, tmp_path, "close-ipc", latch=True)
    out = _finish(proc)
    assert proc.returncode != 0, out
    _check(_myagv_stops(myagv_bridge), event_at)


def test_a_normal_quit_stops_the_robot(myagv_bridge, tmp_path):
    proc, event_at = _drive(myagv_bridge, tmp_path, "quit", latch=True)
    out = _finish(proc)
    assert proc.returncode == 0, out
    _check(_myagv_stops(myagv_bridge), event_at)


def test_an_exception_in_the_ui_stops_the_robot(myagv_bridge, tmp_path):
    proc, event_at = _drive(myagv_bridge, tmp_path, "exception", latch=True)
    out = _finish(proc)
    assert proc.returncode != 0 and "scripted UI failure" in out, out
    _check(_myagv_stops(myagv_bridge), event_at)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGKILL],
                         ids=["SIGINT", "SIGTERM", "SIGKILL"])
def test_a_signal_to_the_ui_stops_the_robot(myagv_bridge, tmp_path, sig):
    proc, _ = _drive(myagv_bridge, tmp_path, "wait", latch=True)
    time.sleep(0.2)
    event_at = time.time()
    proc.send_signal(sig)
    _finish(proc)
    time.sleep(0.5)   # SIGKILL leaves the supervisor running until it has stopped
    _check(_myagv_stops(myagv_bridge), event_at)


@pytest.mark.parametrize("event", ["freeze", "quit"])
def test_the_ainex_gets_its_own_stop_command(ainex_bridge, tmp_path, event):
    """`enable_control` then `stop`, per robots_specs/ainex/ros.yml, three times."""
    proc, event_at = _drive(ainex_bridge, tmp_path, event, latch=True)
    time.sleep(SAFETY_TIMEOUT + 1.0)
    try:
        _check(_ainex_stops(ainex_bridge), event_at)
    finally:
        proc.kill()
        _finish(proc)


def test_nothing_moves_without_the_emergency_stop_confirmation(myagv_bridge, tmp_path):
    """A UI that never confirms the e-stop never starts a supervisor at all."""
    code = (
        "import sys; from robot_console.app import run, Frontend; "
        "from robot_console.cli import Options\n"
        "class F(Frontend):\n"
        "    def confirm_estop(self, prompt): return False\n"
        f"sys.exit(run(Options(url='ws://127.0.0.1:{myagv_bridge.port}').in_envelope(), F()))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], env=_env(), capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 2
    assert "emergency stop" in result.stderr
    assert myagv_bridge.received_on("/myagv/cmd_vel") == []


def test_the_supervisor_refuses_motion_until_enabled(myagv_bridge):
    """Defence in depth: a non-zero command before `enable` goes out as zero."""
    from robot_console.supervisor import SupervisedLink
    from robot_console.teleop import Command

    link = SupervisedLink(f"ws://127.0.0.1:{myagv_bridge.port}", robot="myagv",
                          namespace="myagv", python=sys.executable)
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = _env()["PYTHONPATH"]
    try:
        link.start()
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH")
        else:
            os.environ["PYTHONPATH"] = old
    try:
        for _ in range(10):
            link.publish_cmd_vel(Command(vx=0.2))
            time.sleep(0.02)
        assert myagv_bridge.wait_for_publish("/myagv/cmd_vel", 3)
        assert all(m["linear"]["x"] == 0.0 for m in myagv_bridge.received_on("/myagv/cmd_vel"))
        link.enable_motion()
        link.publish_cmd_vel(Command(vx=0.2))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            link.heartbeat()
            if any(m["linear"]["x"] == 0.2 for m in myagv_bridge.received_on("/myagv/cmd_vel")):
                break
            time.sleep(0.02)
        else:
            pytest.fail("an enabled command never reached the wire")
    finally:
        link.close()
    assert link.stopped_reason == "quit"
