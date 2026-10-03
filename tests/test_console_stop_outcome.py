"""Console spec §4: check stop outcomes where documented feedback exists, without making
status recovery a condition of re-enablement -- the console's teleop window against a
simulated robot's wire.

Opt in with the wires to drive (a started simulation with the robot spawned, as for the
evidence run), as a comma-separated `<id>=<url>` list; a robot not listed is skipped:

    simulator/kitchen.sh start                    # one terminal
    simulator/spawn.sh myagv                      # another: wire on ws://127.0.0.1:9090
    TELEOP_WIRES=myagv=ws://127.0.0.1:9090 robot_console/python.sh -m pytest tests/test_console_stop_outcome.py

The console's own `TeleopApp` runs in-process (SDL dummy video) on scripted keys: W held
for 1 s, released, Esc 2.5 s later. The commands are the console profile's. The outcome is
read on a connection of its own from the feedback the profile documents from the pinned
sources: `/odom` at rest after the release for a wheeled base, `/walking/is_walking` False
once the walking stop call has returned for the AiNex. Nothing is fed back to the console:
whatever the feedback says, it never gates the console (core tests re-enable without any).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

import pytest

WIRES = dict(p.strip().split("=", 1) for p in os.environ.get("TELEOP_WIRES", "").split(",") if "=" in p)
if not WIRES:
    pytest.skip("opt in with TELEOP_WIRES=<id>=ws://host:port[,...] (a running simulation with the "
                "robot spawned)", allow_module_level=True)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame as pg  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from robot_console.profiles import load  # noqa: E402
from robot_console.teleop import TeleopApp  # noqa: E402

#: The stop's documented feedback, by the console profiles' pinned sources:
#: myagv_odometry `/odom`, the ROSMASTER's ekf `/odom`, ainex_controller `/walking/is_walking`.
FEEDBACK = {"myagv": "/odom", "rosmaster_x3_plus": "/odom", "ainex": "/walking/is_walking"}
#: A held W moves a wheeled base (0.2 m/s for 1 s) at least this far, m.
MOVED_M = 0.05
#: At rest from 0.5 s after the release stop: /odom moves at most REST_M over the next 1 s and
#: reports at most REST_MPS -- the residual the robots record for their stop (myAGV 0.01 m/s
#: 0.5 s after the zero Twist, ROSMASTER 0.02 m after it), at the larger.
REST_M, REST_MPS = 0.02, 0.02


class Feedback:
    """One topic's messages with their arrival times, on a rosbridge connection of its own."""

    def __init__(self, url: str, topic: str, type_: str) -> None:
        self.msgs: list = []
        # entered as a context manager (the supported form); closed in close()
        self.ws = connect(url, open_timeout=5.0, close_timeout=1.0, max_size=None,
                          compression=None).__enter__()
        self.ws.send(json.dumps({"op": "subscribe", "topic": topic, "type": type_}))
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        try:
            for raw in self.ws:
                m = json.loads(raw)
                if m.get("op") == "publish":
                    self.msgs.append((time.monotonic(), m["msg"]))
        except ConnectionClosed:
            pass

    def at(self, t: float) -> dict:
        """The last message received by time t."""
        got = [m for at, m in self.msgs if at <= t]
        assert got, f"no feedback by t={t:.2f}"
        return got[-1]

    def close(self) -> None:
        self.ws.close()
        self._reader.join(2.0)


class Keys:
    """The window's event source: key events at offsets (s) from its first poll, made once
    the start-up stop was delivered and commands enabled. Records when each step was posted
    and when the window polled next -- by then its handling (a blocking stop call) returned."""

    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.t0 = None
        self.posted: dict = {}
        self.handled: dict = {}
        self._last = None

    def __call__(self) -> list:
        now = time.monotonic()
        self.t0 = self.t0 or now
        if self._last is not None:
            self.handled[self._last] = now
            self._last = None
        while self.steps and now - self.t0 >= self.steps[0][0]:
            _, name, events = self.steps.pop(0)
            for e in events:
                pg.event.post(e)
            self.posted[name] = now
            self._last = name
        return pg.event.get()


def key(kind, k):
    return pg.event.Event(kind, key=k, mod=0, unicode="", scancode=0)


def xy(odom: dict):
    p = odom["pose"]["pose"]["position"]
    return p["x"], p["y"]


@pytest.mark.parametrize("rid", ["myagv", "rosmaster_x3_plus", "ainex"])
def test_release_stop_outcome_on_documented_feedback(rid):
    if rid not in WIRES:
        pytest.skip(f"{rid} is not in TELEOP_WIRES")
    p = load(rid)
    topic = FEEDBACK[rid]
    # by kind: the AiNex also has a service of the same name (/walking/is_walking)
    e = next((x for x in p.endpoints if x.name == topic and x.kind == "topic"), None)
    assert e and e.direction == "out", f"the {rid} profile documents no {topic} topic"
    fb = Feedback(WIRES[rid], topic, e.type)
    keys = Keys([(0.5, "press", [key(pg.KEYDOWN, pg.K_w)]), (1.5, "release", [key(pg.KEYUP, pg.K_w)]),
                 (4.0, "exit", [key(pg.KEYDOWN, pg.K_ESCAPE)])])
    app = TeleopApp(WIRES[rid], p, None, events=keys, max_seconds=30)
    try:
        code = app.run()
    finally:
        fb.close()
    assert code == 0 and "release" in keys.handled and "exit" in keys.posted, app.lines
    t_press, t_release = keys.posted["press"], keys.posted["release"]
    t_stopped, t_exit = keys.handled["release"], keys.posted["exit"]
    if p.teleop_base:
        moved = math.dist(xy(fb.at(t_release)), xy(fb.at(t_press)))
        assert moved >= MOVED_M, f"{rid}: holding W moved the base {moved:.3f} m on {topic}"
        assert fb.msgs[-1][0] >= t_stopped + 1.5, f"{rid}: {topic} went quiet after the release"
        a, b = fb.at(t_stopped + 0.5), fb.at(t_stopped + 1.5)
        drift = math.dist(xy(a), xy(b))
        tw = b["twist"]["twist"]["linear"]
        speed = math.hypot(tw["x"], tw["y"])
        assert drift <= REST_M and speed <= REST_MPS, \
            f"{rid}: not at rest after the release stop: {drift:.3f} m over 1 s, {speed:.3f} m/s on {topic}"
    else:
        states = [(at, bool(m["data"])) for at, m in fb.msgs if at < t_exit]
        assert any(s for at, s in states if t_press <= at <= t_stopped), \
            f"{rid}: holding W did not start walking ({topic}: {states})"
        last_at, last = states[-1]
        assert last is False and t_release <= last_at <= t_stopped + 1.0, \
            f"{rid}: {topic} did not end False once the stop call returned ({states})"
    released = [o for o in app.core.stop_outcomes if "last motion key released" in o.text]
    assert released and released[0].ok, app.core.stop_outcomes
