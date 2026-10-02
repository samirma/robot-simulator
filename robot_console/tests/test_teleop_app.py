"""teleop.sh end to end against the fake rosbridge (console spec §2.1, §2.3, §4).

In-process tests drive the real pygame window (SDL dummy video driver) by posting key and
focus events into SDL's queue; subprocess tests cover signals, connection loss, relaunch and
refusals of the real ``python -m robot_console.teleop`` entry point.
"""

import os
import signal
import subprocess
import sys
import time

import pygame as pg
import pytest

from robot_console.profiles import load
from robot_console.teleop import TeleopApp
from wirespec import merge, wire_spec

MOBILE = ["myagv", "ainex", "rosmaster_x3_plus"]


class Script:
    """An event source that posts scheduled events into SDL's queue, then reads it."""

    def __init__(self, steps, fail_at=None):
        self.steps = sorted(steps, key=lambda s: s[0])
        self.n = 0
        self.fail_at = fail_at

    def __call__(self):
        self.n += 1
        if self.fail_at is not None and self.n >= self.fail_at:
            raise pg.error("keyboard device went away")
        while self.steps and self.steps[0][0] <= self.n:
            _, evs = self.steps.pop(0)
            for e in evs:
                pg.event.post(e)
        return pg.event.get()


def down(key):
    return pg.event.Event(pg.KEYDOWN, key=key, mod=0, unicode="", scancode=0)


def up(key):
    return pg.event.Event(pg.KEYUP, key=key, mod=0, unicode="", scancode=0)


def run_app(url, steps, robot=None, fail_at=None, max_seconds=20):
    app = TeleopApp(url, load(robot) if robot else None, None, events=Script(steps, fail_at),
                    max_seconds=max_seconds)
    return app.run(), app


def motion_publishes(srv, p):
    if p.teleop_base:
        return [o["msg"] for o in srv.published(p.teleop_base.topic)]
    return [o["msg"] for o in srv.published(p.teleop_walk.param_topic)]


def stop_ops(srv, p):
    if p.teleop_walk:
        return [c for c in srv.calls("/walking/command") if c["args"] == {"command": "stop"}]
    return [o for o in srv.published(p.stop[0].name) if o["msg"] == p.stop[0].msg]


@pytest.mark.parametrize("pid", MOBILE)
def test_hold_release_and_exit_through_the_window(fake, pid):
    p = load(pid)
    srv = fake(wire_spec(p))
    # no Enter: commands are enabled once the start-up stop was delivered
    steps = [(10, [down(pg.K_w)]), (11, [down(pg.K_w)]),   # a repeated key-down is not a fresh press
             (25, [up(pg.K_w)]),
             (30, [down(pg.K_ESCAPE)])]
    code, app = run_app(srv.url, steps)
    assert code == 0
    moves = motion_publishes(srv, p)
    sx = (p.teleop_base or p.teleop_walk).axes["x"].speed
    field = "linear" if p.teleop_base else None
    xs = [m["linear"]["x"] if field else m[p.teleop_walk.axes["x"].field] for m in moves]
    assert sx in xs and all(abs(x) <= (p.teleop_base or p.teleop_walk).axes["x"].limit for x in xs)
    # start-up stop, release stop and exit stop, in that order
    assert len(stop_ops(srv, p)) >= 3
    first_move = min(o["_t"] for o in srv.command_ops() if o.get("op") == "publish"
                     and o["topic"] == (p.teleop_base.topic if p.teleop_base else p.teleop_walk.param_topic)
                     and o["msg"] != (p.stop[0].msg if p.teleop_base else None))
    first_stop = min(o["_t"] for o in stop_ops(srv, p))
    assert first_stop < first_move, "the start-up stop precedes any motion"


@pytest.mark.parametrize("pid", ["myagv", "ainex"])
def test_focus_loss_stops(fake, pid):
    p = load(pid)
    srv = fake(wire_spec(p))
    steps = [(3, [down(pg.K_RETURN), up(pg.K_RETURN)]), (5, [down(pg.K_q)]),
             (12, [pg.event.Event(pg.WINDOWFOCUSLOST)]), (20, [down(pg.K_ESCAPE)])]
    code, app = run_app(srv.url, steps)
    assert code == 0
    assert any("lost keyboard focus" in m for m in app.core.messages + app.lines)


def test_input_loss_stops_and_exits_nonzero(fake):
    p = load("myagv")
    srv = fake(wire_spec(p))
    steps = [(3, [down(pg.K_RETURN), up(pg.K_RETURN)]), (5, [down(pg.K_w)])]
    code, app = run_app(srv.url, steps, fail_at=15)
    assert code == 5
    assert srv.published("/cmd_vel")[-1]["msg"] == p.stop[0].msg


def test_ainex_head_arrows(fake):
    p = load("ainex")
    srv = fake(wire_spec(p))
    steps = [(3, [down(pg.K_RETURN), up(pg.K_RETURN)]), (5, [down(pg.K_LEFT)]), (40, [up(pg.K_LEFT)]),
             (45, [down(pg.K_UP)]), (60, [up(pg.K_UP)]), (65, [down(pg.K_ESCAPE)])]
    code, app = run_app(srv.url, steps, robot="ainex")
    assert code == 0
    pans = [o["msg"]["position"] for o in srv.published("/head_pan_controller/command")]
    tilts = [o["msg"]["position"] for o in srv.published("/head_tilt_controller/command")]
    # Left turns the head left: a negative pan on the simulated model (axis -Z)
    assert pans and pans == sorted(pans, reverse=True) and min(pans) >= p.head["pan"].min
    assert min(pans) < -0.5
    assert tilts and max(tilts) <= p.head["tilt"].max
    # head keys never request the walking stop; only start-up and exit did
    assert len(stop_ops(srv, p)) == 2


def test_automatically_identified_arm_is_refused(fake):
    srv = fake(wire_spec(load("so101")))
    code, app = run_app(srv.url, [(2, [down(pg.K_ESCAPE)])])
    assert code == 2
    assert any("myagv, ainex, rosmaster_x3_plus" in line for line in app.lines)
    assert srv.command_ops() == []


def test_no_target_refuses_commands_and_shows_untied_cameras(fake):
    srv = fake(merge(wire_spec(load("myagv")), wire_spec(load("rosmaster_x3_plus"))))
    steps = [(2, [down(pg.K_RETURN), down(pg.K_w)]), (15, [down(pg.K_ESCAPE)])]
    code, app = run_app(srv.url, steps)
    assert code == 1
    assert "myagv" in app.reason and "rosmaster_x3_plus" in app.reason
    streams = list(app.cams.streams.values())
    assert [s.spec.topic for s in streams] == ["/camera/rgb/image_raw", "/usb_cam/image_raw"]
    assert not any(s.spec.tied for s in streams)
    assert srv.command_ops() == []


# ------------------------------------------------------------------ subprocess

def launch(url, *args):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    return subprocess.Popen([sys.executable, "-m", "robot_console.teleop", "--url", url, *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


def wait_for(pred, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("sig, code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
@pytest.mark.parametrize("pid", MOBILE)
def test_handled_signals_stop_and_exit(fake, pid, sig, code):
    p = load(pid)
    srv = fake(wire_spec(p))
    proc = launch(srv.url)
    assert wait_for(lambda: len(stop_ops(srv, p)) >= 1), "start-up stop after validation"
    time.sleep(0.3)
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == code, out
    assert len(stop_ops(srv, p)) == 2, out
    assert motion_publishes(srv, p) == [o["msg"] for o in stop_ops(srv, p)] if p.teleop_base else True


@pytest.mark.parametrize("pid", MOBILE)
def test_connection_loss_ends_teleop_nonzero_without_claiming_a_stop(fake, pid):
    p = load(pid)
    srv = fake(wire_spec(p))
    proc = launch(srv.url)
    assert wait_for(lambda: len(stop_ops(srv, p)) >= 1)
    srv.drop_all()
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 4, out
    assert "CONNECTION LOST" in out and "NOT guaranteed" in out
    assert "stop requested (teleop ending" not in out


def test_every_start_stops_once_without_resumed_motion(fake):
    p = load("rosmaster_x3_plus")
    srv = fake(wire_spec(p))
    for n in (1, 2):
        proc = launch(srv.url)
        assert wait_for(lambda: len(stop_ops(srv, p)) >= 2 * n - 1)
        time.sleep(0.5)
        assert all(m == p.stop[0].msg for m in motion_publishes(srv, p)), "no resumed motion"
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=20)
    assert len(stop_ops(srv, p)) == 4


@pytest.mark.parametrize("args, word", [
    (["--robot", "turtlebot"], "unknown robot id"),
    (["--robot", "so101"], "is an arm"),
    (["--robot", "mycobot280"], "is an arm"),
    (["--robot", "myagv_mycobot280"], "assembly"),
    (["--robot", "myagv", "--namespace", "robot1"], "no namespace override"),
])
def test_refusals(args, word):
    proc = launch("ws://127.0.0.1:1", *args)
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 2 and word in out, out
    if "arm" in word:
        assert "myagv, ainex, rosmaster_x3_plus" in out


def test_unreachable():
    proc = launch("ws://127.0.0.1:1")
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 3 and "unreachable" in out
