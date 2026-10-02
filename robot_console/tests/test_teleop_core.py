"""Teleop motion lifecycle for each mobile profile (console spec §2.1, §4)."""

import dataclasses

import pytest

from robot_console.discovery import Target, Validation
from robot_console.profiles import load
from robot_console.rosbridge import ServiceError, TransportError
from robot_console.teleop_core import REENABLE_LIMITATION, TeleopCore

MOBILE = ["myagv", "ainex", "rosmaster_x3_plus"]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Rec:
    """A Sender that records, and can fail on demand."""

    def __init__(self):
        self.sent = []            # (op, name, payload)
        self.fail_publish = None  # predicate(name) -> exception or None
        self.fail_call = None

    def publish(self, topic, type, msg):
        if self.fail_publish and self.fail_publish(topic):
            raise self.fail_publish(topic)
        self.sent.append(("publish", topic, msg))

    def call(self, service, type, args):
        if self.fail_call and self.fail_call(service, args):
            raise self.fail_call(service, args)
        self.sent.append(("call", service, args))
        return {}

    def advertise(self, topic, type):
        pass

    def clear(self):
        self.sent.clear()


def make(pid, profile=None):
    p = profile or load(pid)
    rec, clk = Rec(), Clock()
    core = TeleopCore(Target(p, "", Validation()), rec, clock=clk)
    return core, rec, clk, p


def motion(rec, p):
    """The motion commands sent, as (x, y, yaw)."""
    out = []
    for op, name, msg in rec.sent:
        if p.teleop_base and op == "publish" and name == p.teleop_base.topic:
            out.append((msg["linear"]["x"], msg["linear"]["y"], msg["angular"]["z"]))
        if p.teleop_walk and op == "publish" and name == p.teleop_walk.param_topic:
            ax = p.teleop_walk.axes
            out.append((msg[ax["x"].field], msg[ax["y"].field], msg[ax["yaw"].field]))
    return out


def is_stop(p, entry):
    op, name, msg = entry
    if p.teleop_walk:
        return op == "call" and msg == {"command": "stop"}
    return op == "publish" and name == p.stop[0].name and msg == p.stop[0].msg


def stops(rec, p):
    return [e for e in rec.sent if is_stop(p, e)]


def speeds(p):
    ax = (p.teleop_base or p.teleop_walk).axes
    return ax["x"].speed, ax["y"].speed, ax["yaw"].speed


@pytest.mark.parametrize("pid", MOBILE)
def test_nothing_is_sent_before_enter(pid):
    core, rec, clk, p = make(pid)
    core.key_down("w")
    clk.t += 1
    core.tick()
    assert motion(rec, p) == [] and not core.enabled


@pytest.mark.parametrize("pid", MOBILE)
def test_startup_stop_after_validation_leaves_commands_disabled(pid):
    core, rec, clk, p = make(pid)
    out = core.startup_stop()
    assert out.ok and len(core.stop_outcomes) == 1 and not core.enabled
    assert len(rec.sent) == len(core.stop_ops()) and all(is_stop(p, e) or e[0] == "call" for e in rec.sent)
    core.key_down("w")
    assert len(rec.sent) == len(core.stop_ops()), "no resumed motion"


@pytest.mark.parametrize("pid", MOBILE)
@pytest.mark.parametrize("key, sign", [("w", (1, 0, 0)), ("s", (-1, 0, 0)), ("a", (0, 1, 0)),
                                       ("d", (0, -1, 0)), ("q", (0, 0, 1)), ("e", (0, 0, -1))])
def test_key_direction_and_recorded_speed(pid, key, sign):
    core, rec, clk, p = make(pid)
    core.enable()
    rec.clear()
    core.key_down(key)
    sx, sy, sw = speeds(p)
    assert motion(rec, p) == [(sign[0] * sx, sign[1] * sy, sign[2] * sw)]
    core.key_up(key)
    assert is_stop(p, rec.sent[-1]), "release of the last motion key requests the stop"


@pytest.mark.parametrize("pid", MOBILE)
def test_opposite_keys_cancel_and_combination_stays_within_limits(pid):
    core, rec, clk, p = make(pid)
    core.enable()
    for k in "wsaq":
        core.key_down(k)
    x, y, yaw = motion(rec, p)[-1]
    assert x == 0.0
    ax = (p.teleop_base or p.teleop_walk).axes
    assert abs(y) <= ax["y"].limit and abs(yaw) <= ax["yaw"].limit
    core.key_up("a")                       # not the last key: no stop yet
    assert not core.stop_outcomes


@pytest.mark.parametrize("pid", MOBILE)
def test_speeds_are_clamped_to_documented_limits(pid):
    p = load(pid)
    sec = p.teleop_base or p.teleop_walk
    fast = {k: dataclasses.replace(a, speed=a.limit * 5) for k, a in sec.axes.items()}
    p2 = dataclasses.replace(p, **{("teleop_base" if p.teleop_base else "teleop_walk"): dataclasses.replace(sec, axes=fast)})
    core, rec, clk, _ = make(pid, p2)
    core.enable()
    core.key_down("w")
    core.key_down("a")
    core.key_down("q")
    x, y, yaw = motion(rec, p2)[-1]
    assert (x, y, yaw) == (sec.axes["x"].limit, sec.axes["y"].limit, sec.axes["yaw"].limit)


@pytest.mark.parametrize("pid", MOBILE)
def test_held_key_republishes_and_repeat_is_not_a_fresh_press(pid):
    core, rec, clk, p = make(pid)
    core.enable()
    core.key_down("w")
    core.key_down("w")                     # key repeat: ignored
    assert len(motion(rec, p)) == 1
    rate = (p.teleop_base or p.teleop_walk).rate_hz
    for _ in range(5):
        clk.t += 1.0 / rate + 1e-3
        core.tick()
    assert len(motion(rec, p)) == 6


@pytest.mark.parametrize("pid", MOBILE)
@pytest.mark.parametrize("event", ["space", "focus", "input"])
def test_space_focus_and_input_loss_clear_intent_and_stop(pid, event):
    core, rec, clk, p = make(pid)
    core.enable()
    core.key_down("w")
    core.key_down("q")
    {"space": lambda: core.key_down("space"), "focus": core.focus_lost, "input": core.input_lost}[event]()
    assert is_stop(p, rec.sent[-1]) and core.held == []
    n = len(motion(rec, p))
    core.key_up("w")                       # the old press does not resume or re-stop motion
    clk.t += 5
    core.tick()
    assert len(motion(rec, p)) == n
    core.key_down("w")                     # a fresh press moves again
    assert len(motion(rec, p)) == n + 1


@pytest.mark.parametrize("pid", MOBILE)
def test_ordinary_exit_stops(pid):
    core, rec, clk, p = make(pid)
    core.enable()
    core.key_down("d")
    out = core.shutdown("Esc")
    assert out.ok and is_stop(p, rec.sent[-1]) and not core.enabled


def _fail_stop(core, rec, p, exc):
    if p.teleop_walk:
        rec.fail_call = lambda s, a: exc if a == {"command": "stop"} else None
    else:
        rec.fail_publish = lambda t: exc if t == p.stop[0].name else None


@pytest.mark.parametrize("pid", MOBILE)
def test_failed_stop_disables_until_enter_and_fresh_input(pid):
    core, rec, clk, p = make(pid)
    core.enable()
    core.key_down("w")
    exc = ServiceError("stop refused") if p.teleop_walk else TransportError("send failed")
    _fail_stop(core, rec, p, exc)
    out = core.key_up("w") or core.stop_outcomes[-1]
    assert not out.ok and not core.enabled
    assert REENABLE_LIMITATION in out.text
    rec.fail_publish = rec.fail_call = None
    n = len(motion(rec, p))
    core.key_down("w")                     # disabled: ignored
    assert len(motion(rec, p)) == n
    core.key_down("enter")
    assert core.enabled and core.held == [], "old intent is never resumed"
    clk.t += 5
    core.tick()
    assert len(motion(rec, p)) == n
    core.key_up("w")
    core.key_down("w")
    assert len(motion(rec, p)) == n + 1


def test_ainex_stop_call_timeout_and_false_result_are_failures():
    core, rec, clk, p = make("ainex")
    core.enable()
    core.key_down("w")
    rec.fail_call = lambda s, a: TimeoutError("no answer") if a == {"command": "stop"} else None
    core.key_up("w")
    assert not core.enabled and not core.stop_outcomes[-1].ok
    core2, rec2, _, _ = make("ainex")
    rec2.call = lambda s, t, a: {"result": False}
    out = core2.request_stop("test")
    assert not out.ok


def test_ainex_walk_start_then_param_updates():
    core, rec, clk, p = make("ainex")
    core.enable()
    rec.clear()
    core.key_down("w")
    kinds = [(op, name, msg.get("command") if op == "call" else None) for op, name, msg in rec.sent]
    assert kinds == [("publish", "/walking/set_param", None), ("call", "/walking/command", "start")]
    core.key_down("q")
    assert rec.sent[-1][:2] == ("publish", "/walking/set_param")
    core.key_up("q")
    core.key_up("w")
    assert rec.sent[-1] == ("call", "/walking/command", {"command": "stop"})


def test_ainex_enable_calls_documented_prerequisites():
    core, rec, clk, p = make("ainex")
    core.enable()
    assert [a for op, n, a in rec.sent if op == "call"] == [o.msg for o in p.teleop_walk.enable]


def test_ainex_head_keys_move_within_limits_and_never_stop_walking():
    core, rec, clk, p = make("ainex")
    core.enable()
    rec.clear()
    core.key_down("left")
    for _ in range(100):
        clk.t += 0.1
        core.tick()
    pans = [m["position"] for op, n, m in rec.sent if n == p.head["pan"].topic]
    # Left turns the head left, which is a negative pan on the simulated AiNex (axis -Z)
    assert pans and min(pans) == pytest.approx(p.head["pan"].min) and pans == sorted(pans, reverse=True)
    core.key_up("left")
    n = len(rec.sent)
    clk.t += 1
    core.tick()
    assert len(rec.sent) == n, "head target stops changing on release"
    assert not stops(rec, p), "head keys never request the walking stop"
    core.key_down("down")
    for _ in range(100):
        clk.t += 0.1
        core.tick()
    tilts = [m["position"] for op, n, m in rec.sent if n == p.head["tilt"].topic]
    assert min(tilts) == pytest.approx(p.head["tilt"].min)


@pytest.mark.parametrize("pid", MOBILE)
def test_connection_loss_claims_no_stop(pid):
    core, rec, clk, p = make(pid)
    core.enable()
    core.key_down("w")
    rec.clear()
    core.connection_lost("socket closed")
    assert rec.sent == [] and core.held == [] and not core.enabled
    assert "NOT guaranteed" in core.messages[-1]
    out = core.shutdown("exit")
    assert not out.ok and rec.sent == []
