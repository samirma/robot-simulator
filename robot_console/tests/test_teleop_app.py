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
from robot_console.rosbridge import Rosbridge, TransportError
from robot_console.teleop import TeleopApp
from wirespec import merge, wire_spec

MOBILE = ["myagv", "ainex", "rosmaster_x3_plus"]
CONSOLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Script:
    """An event source that posts scheduled events into SDL's queue, then reads it."""

    def __init__(self, steps, fail_at=None, hooks=None):
        self.steps = sorted(steps, key=lambda s: s[0])
        self.n = 0
        self.fail_at = fail_at
        self.hooks = dict(hooks or {})   # step -> callable run when that step is reached
        self.posted = {}          # step -> when its events were posted (time.monotonic)

    def __call__(self):
        self.n += 1
        if self.fail_at is not None and self.n >= self.fail_at:
            raise pg.error("keyboard device went away")
        if self.n in self.hooks:
            self.hooks[self.n]()
        while self.steps and self.steps[0][0] <= self.n:
            step, evs = self.steps.pop(0)
            self.posted[step] = time.monotonic()
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


def stops_between(srv, p, script, a, b):
    """Stop ops the bridge received after step ``a``'s events were posted, before step ``b``'s."""
    return [o for o in stop_ops(srv, p) if script.posted[a] <= o["_t"] < script.posted[b]]


@pytest.mark.parametrize("pid", ["myagv", "ainex"])
def test_focus_loss_stops(fake, pid):
    p = load(pid)
    srv = fake(wire_spec(p))
    steps = [(3, [down(pg.K_RETURN), up(pg.K_RETURN)]), (5, [down(pg.K_q)]),
             (12, [pg.event.Event(pg.WINDOWFOCUSLOST)]), (20, [down(pg.K_ESCAPE)])]
    code, app = run_app(srv.url, steps)
    assert code == 0
    assert any("lost keyboard focus" in m for m in app.core.messages + app.lines)
    assert len(stops_between(srv, p, app._events, 12, 20)) == 1, "focus loss reached the wire as a stop"


def test_focus_loss_requests_stop_while_commands_are_disabled(fake):
    """After a failed start-up stop commands stay disabled, and focus loss still requests stop."""
    p = load("ainex")
    srv = fake(wire_spec(p))
    srv.set_service("/walking/command", "fail")
    steps = [(10, [pg.event.Event(pg.WINDOWFOCUSLOST)]), (20, [down(pg.K_ESCAPE)])]
    code, app = run_app(srv.url, steps, robot="ainex")
    assert code == 0 and not app.core.enabled
    assert any("STOP FAILED (start-up" in m for m in app.core.messages + app.lines)
    assert len(stops_between(srv, p, app._events, 10, 20)) == 1
    assert len(stop_ops(srv, p)) == 3, "start-up, focus loss and exit each requested the stop"


def test_enter_reenables_after_a_failed_startup_stop_with_fresh_input(fake):
    """Console spec §2.1: a failed start-up stop leaves commands disabled (keys are ignored and
    the limitation is shown); Enter re-enables them without checking that the robot stopped, and
    only a fresh key press moves the robot."""
    p = load("ainex")
    srv = fake(wire_spec(p))
    refused = []

    def walking_command(args):                    # the first stop (the start-up stop) is refused
        if args.get("command") == "stop" and not refused:
            refused.append(args)
            raise RuntimeError("stop refused")
        return {"result": True}
    srv.set_service("/walking/command", walking_command)
    steps = [(10, [down(pg.K_w)]), (14, [down(pg.K_RETURN), up(pg.K_RETURN)]),
             (20, [up(pg.K_w)]), (24, [down(pg.K_w)]), (34, [up(pg.K_w)]), (40, [down(pg.K_ESCAPE)])]
    code, app = run_app(srv.url, steps, robot="ainex")
    assert code == 0 and refused
    text = app.core.messages + app.lines
    assert any("STOP FAILED (start-up" in m and "does not check or confirm" in m for m in text), text
    script = app._events
    moves = [o for o in srv.published(p.teleop_walk.param_topic)]
    assert moves and min(o["_t"] for o in moves) >= script.posted[24], "no motion before Enter and a fresh press"
    starts = [c for c in srv.calls("/walking/command") if c["args"] == {"command": "start"}]
    assert len(starts) == 1 and starts[0]["_t"] >= script.posted[24]
    assert len(stop_ops(srv, p)) == 3, "start-up (refused), release and exit"


def test_cameras_show_live_then_stale_in_the_window(fake):
    """Console spec §2.3 in teleop: the profile camera is live while frames arrive and turns stale
    (its frame no longer drawn as live) once they stop."""
    p = load("myagv")
    srv = fake(wire_spec(p))
    topic = p.cameras[0].topic
    seen = {}
    script = Script([(90, [down(pg.K_ESCAPE)])])
    app = TeleopApp(srv.url, None, None, events=script, max_seconds=30)

    def look(step):
        s = app.cams.streams[topic]
        seen[step] = (s.state(), s.status_text(), s.spec.tied)
    script.hooks = {40: lambda: look(40), 41: lambda: srv.paused.add(topic), 89: lambda: look(89)}
    assert app.run() == 0
    assert seen[40][0] == "live" and seen[40][2], seen
    assert seen[89][0] == "stale" and seen[89][1].startswith("stale: last frame"), seen


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
    srv.set_service("/ros_robot_controller/bus_servo/get_position",     # head at 500 pulses = 0 rad
                    lambda args: {"success": True, "position": [{"id": i, "position": 500} for i in args["id"]]})
    steps = [(3, [down(pg.K_RETURN), up(pg.K_RETURN)]), (5, [down(pg.K_LEFT)]), (40, [up(pg.K_LEFT)]),
             (45, [down(pg.K_UP)]), (60, [up(pg.K_UP)]), (65, [down(pg.K_ESCAPE)])]
    code, app = run_app(srv.url, steps, robot="ainex")
    assert code == 0
    pans = [o["msg"]["position"] for o in srv.published("/head_pan_controller/command")]
    tilts = [o["msg"]["position"] for o in srv.published("/head_tilt_controller/command")]
    # Left turns the head left: a negative pan on the vendor model's -Z head_pan axis
    # (ainex.urdf.xacro#L786-L787; console spec §2.1 as amended 2026-10-02)
    assert pans and pans == sorted(pans, reverse=True) and min(pans) >= p.head["pan"].min
    assert min(pans) < -0.5
    assert tilts and max(tilts) <= p.head["tilt"].max
    # head keys never request the walking stop; only start-up and exit did
    assert len(stop_ops(srv, p)) == 2


def test_connection_lost_while_advertising_ends_with_status_4(fake, monkeypatch):
    """The socket drops after discovery, before the start-up stop: no traceback, status 4."""
    p = load("myagv")
    srv = fake(wire_spec(p))

    def advertise(self, topic, type):        # the socket drops as teleop advertises
        srv.drop_all()
        raise TransportError("send failed")
    monkeypatch.setattr(Rosbridge, "advertise", advertise)
    code, app = run_app(srv.url, [(30, [down(pg.K_ESCAPE)])])
    assert code == 4
    assert any("NOT guaranteed" in line for line in app.lines), app.lines
    assert app.lines[-1] == "teleop ends; relaunch to reconnect"
    assert stop_ops(srv, p) == [], "no stop is claimed or sent"


def test_launcher_header_describes_automatic_enable():
    """teleop.sh's header follows console spec §2.1 as amended 2026-10-02."""
    with open(os.path.join(CONSOLE, "teleop.sh")) as f:
        text = f.read()
    assert "Enter enable commands" not in text and "re-enable" in text


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


@pytest.mark.parametrize("sig, code", [(signal.SIGHUP, 129), (signal.SIGQUIT, 131)])
def test_other_handled_interruptions_stop_and_exit(fake, sig, code):
    """A closed terminal (SIGHUP) or Ctrl-\\ (SIGQUIT) is an interruption the process can handle
    (console spec §2.1): it clears intent and attempts the stop like SIGINT/SIGTERM."""
    p = load("myagv")
    srv = fake(wire_spec(p))
    proc = launch(srv.url)
    assert wait_for(lambda: len(stop_ops(srv, p)) >= 1)
    time.sleep(0.3)
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == code, out
    assert len(stop_ops(srv, p)) == 2, out


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
    (["--robot", "myagv_mycobot280"], "unknown robot id"),
    (["--robot", "so101"], "is an arm"),
    (["--robot", "mycobot280"], "is an arm"),
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
