"""Motion smoke runs through each robot's vendor interface, judged by its recorded
feedback or, where the interface publishes none, by the simulation's joint and pose
readings (control port). Tolerances come from the interface files."""

from __future__ import annotations

import math
import threading
import time

import protocol
import wirecheck
from rosbridge_client import Rosbridge


def tol(rid: str, fragment: str) -> dict:
    for t in wirecheck.interface(rid).get("tolerances", []):
        if fragment in t["figure"]:
            return t["tolerance"]
    raise KeyError(fragment)


def within(got: float, want: float, t: dict) -> bool:
    lim = max(abs(want) * t.get("relative", 0.0), t.get("absolute", 0.0))
    return abs(got - want) <= lim + 1e-9


def yaw_of(q) -> float:
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Latest:
    """The latest message of a topic, over rosbridge."""

    def __init__(self, rb: Rosbridge, topic: str, **kw):
        self.msg = None
        self.count = 0
        self.ev = threading.Event()
        rb.subscribe(topic, self._cb, **kw)

    def _cb(self, m):
        self.msg = m["msg"]
        self.count += 1
        self.ev.set()

    def wait(self, timeout=10.0):
        assert self.ev.wait(timeout), "no message"
        return self.msg


def twist(x=0.0, y=0.0, z=0.0):
    return {"linear": {"x": x, "y": y, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": z}}


def drive(port: int, rid: str, odom_topic: str = "/odom", duration: float = 1.25) -> list:
    """Forward, back, left, right and a turn, each followed by the recorded stop (a zero
    Twist); every displacement judged from odometry with the recorded tolerances.
    Returns the problems found."""
    problems = []
    lin_t = tol(rid, "displacement") if rid != "rosmaster_x3_plus" else \
        tol(rid, "translation displacement")
    yaw_t = tol(rid, "rotation") if rid != "rosmaster_x3_plus" else tol(rid, "yaw change")
    rb = Rosbridge("127.0.0.1", port)
    try:
        odom = Latest(rb, odom_topic)
        odom.wait()
        rb.advertise("/cmd_vel", "geometry_msgs/Twist")
        time.sleep(0.8)
        for name, (vx, vy, wz) in (("forward", (0.2, 0, 0)), ("back", (-0.2, 0, 0)),
                                   ("left", (0, 0.2, 0)), ("right", (0, -0.2, 0)),
                                   ("turn", (0, 0, 0.5))):
            p0 = odom.msg["pose"]["pose"]
            t0 = time.monotonic()
            rb.publish("/cmd_vel", twist(vx, vy, wz))
            time.sleep(max(0.0, duration if name != "turn" else 2.0))
            rb.publish("/cmd_vel", twist())
            t_run = time.monotonic() - t0
            time.sleep(1.2)
            p1 = odom.msg["pose"]["pose"]
            th0 = yaw_of(p0["orientation"])
            dx = p1["position"]["x"] - p0["position"]["x"]
            dy = p1["position"]["y"] - p0["position"]["y"]
            fwd = dx * math.cos(th0) + dy * math.sin(th0)
            lat = -dx * math.sin(th0) + dy * math.cos(th0)
            dth = wrap(yaw_of(p1["orientation"]) - th0)
            if name == "turn":
                if not within(dth, wz * t_run, yaw_t):
                    problems.append(f"turn: odom yaw {dth:.3f} rad, commanded {wz * t_run:.3f}")
            else:
                # the displacement vector, within the tolerance of the commanded one
                want = math.hypot(vx, vy) * t_run
                err = math.hypot(fwd - vx * t_run, lat - vy * t_run)
                if err > max(want * lin_t.get("relative", 0.0), lin_t.get("absolute", 0.0)):
                    problems.append(f"{name}: odom ({fwd:.3f}, {lat:.3f}) m, commanded "
                                    f"({vx * t_run:.3f}, {vy * t_run:.3f})")
            tw = odom.msg["twist"]["twist"]
            speed = math.hypot(tw["linear"]["x"], tw["linear"]["y"])
            if speed > 0.02 or abs(tw["angular"]["z"]) > 0.05:
                problems.append(f"{name}: not at rest after the stop ({speed:.3f} m/s)")
    finally:
        rb.close()
    return problems


def readings(sim_port: int, rid: str) -> dict:
    c = protocol.Client("127.0.0.1", sim_port)
    try:
        return c.call("readings", robot=rid)
    finally:
        c.close()


def joints(sim_port: int, rid: str) -> dict:
    return {k: v["position"] for k, v in readings(sim_port, rid)["joints"].items()}


def wait_joints(sim_port, rid, target: dict, tolerance: float, timeout: float) -> dict:
    t0 = time.monotonic()
    while True:
        j = joints(sim_port, rid)
        err = {k: abs(j[k] - v) for k, v in target.items()}
        if all(e <= tolerance for e in err.values()) or time.monotonic() - t0 > timeout:
            return err
        time.sleep(0.2)


def at_rest(sim_port, rid, joint_names, window=0.5, timeout=4.0) -> bool:
    """The joints come to rest (moving < 0.005 rad over `window`) within `timeout`."""
    t0 = time.monotonic()
    while True:
        a = joints(sim_port, rid)
        time.sleep(window)
        b = joints(sim_port, rid)
        if all(abs(a[k] - b[k]) < 0.005 for k in joint_names):
            return True
        if time.monotonic() - t0 > timeout:
            return False
