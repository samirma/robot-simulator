"""Motion smoke runs through each robot's vendor interface, judged by its recorded
feedback or, where the interface publishes none, by the simulation's joint and pose
readings (control port). Tolerances come from the interface files."""

from __future__ import annotations

import collections
import math
import re
import threading
import time

import protocol
import wirecheck
from rosbridge_client import Rosbridge

#: The drive's legs: forward, back, left, right and a turn in place ((vx, vy, wz), seconds).
#: Each leg's commanded displacement plus its recorded acceptance bound stays inside the
#: travel the floor placement guarantees (simulator spec §2.3: 0.5 m forward, 0.25 m back
#: and to each side): 0.25 m forward and back to the start, 0.20 m to each side.
LEGS = (("forward", (0.2, 0, 0), 1.25), ("back", (-0.2, 0, 0), 1.25),
        ("left", (0, 0.2, 0), 1.0), ("right", (0, -0.2, 0), 1.0), ("turn", (0, 0, 0.5), 2.0))


def tol_row(rid: str, fragment: str) -> dict:
    """The one recorded tolerance row whose figure contains `fragment`."""
    rows = [t for t in wirecheck.interface(rid).get("tolerances", []) if fragment in t["figure"]]
    if len(rows) != 1:
        raise KeyError(f"{rid}: {len(rows)} recorded tolerances match {fragment!r} "
                       f"({[t['figure'] for t in rows]})")
    return rows[0]


def tol(rid: str, fragment: str) -> dict:
    return tol_row(rid, fragment)["tolerance"]


def within(got: float, want: float, t: dict) -> bool:
    lim = max(abs(want) * t.get("relative", 0.0), t.get("absolute", 0.0))
    return abs(got - want) <= lim + 1e-9


def yaw_of(q) -> float:
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def stamp_of(msg: dict) -> float:
    """A message's header stamp, in s (ROS 1 secs/nsecs, ROS 2 sec/nanosec)."""
    s = msg["header"]["stamp"]
    return s.get("secs", s.get("sec", 0)) + s.get("nsecs", s.get("nanosec", 0)) * 1e-9


class Latest:
    """The latest message of a topic, over rosbridge, and the recent ones (`history`)."""

    def __init__(self, rb: Rosbridge, topic: str, **kw):
        self.msg = None
        self.count = 0
        self.history = collections.deque(maxlen=5000)
        self.ev = threading.Event()
        rb.subscribe(topic, self._cb, **kw)

    def _cb(self, m):
        self.msg = m["msg"]
        self.history.append(self.msg)
        self.count += 1
        self.ev.set()

    def wait(self, timeout=10.0):
        assert self.ev.wait(timeout), "no message"
        return self.msg

    def at(self, t: float) -> dict:
        """The last message stamped at or before wall time `t` (the first one if none is)."""
        msgs = list(self.history)
        before = [m for m in msgs if stamp_of(m) <= t]
        return max(before, key=stamp_of) if before else msgs[0]

    def since(self, t: float) -> list:
        """The messages stamped at or after wall time `t`."""
        return [m for m in list(self.history) if stamp_of(m) >= t]


def twist(x=0.0, y=0.0, z=0.0):
    return {"linear": {"x": x, "y": y, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": z}}


def drive_legs(port: int, rid: str, odom_topic: str = "/odom") -> dict:
    """The drive's legs (`LEGS`), each followed by the recorded stop (a zero Twist). Per leg:
    the command, how long it ran, the odometry displacement over it once settled (forward,
    lateral, yaw, in the frame it started in) and the motion left after the stop, as the
    recorded `residual` figure states it: a speed some time after the stop (unit m/s), or
    the distance travelled after the stop (unit m)."""
    res = tol_row(rid, "residual")
    unit = res["tolerance"].get("unit")
    after = None
    if unit == "m/s":
        m = re.search(r"([\d.]+) s after", res["figure"])
        if m is None:
            raise ValueError(f"{rid}: {res['figure']!r} names no time after the stop")
        after = float(m.group(1))
    legs = {}
    rb = Rosbridge("127.0.0.1", port)
    try:
        odom = Latest(rb, odom_topic)
        odom.wait()
        rb.advertise("/cmd_vel", "geometry_msgs/Twist")
        time.sleep(0.8)
        for name, (vx, vy, wz), seconds in LEGS:
            p0 = odom.msg["pose"]["pose"]
            t0 = time.monotonic()
            rb.publish("/cmd_vel", twist(vx, vy, wz))
            time.sleep(seconds)
            rb.publish("/cmd_vel", twist())
            t_stop = time.time()
            t_run = time.monotonic() - t0
            time.sleep(1.2)
            p1 = odom.msg["pose"]["pose"]
            th0 = yaw_of(p0["orientation"])
            dx = p1["position"]["x"] - p0["position"]["x"]
            dy = p1["position"]["y"] - p0["position"]["y"]
            if after is not None:
                tw = odom.at(t_stop + after)["twist"]["twist"]
                residual = math.hypot(tw["linear"]["x"], tw["linear"]["y"])
            else:
                ps = odom.at(t_stop)["pose"]["pose"]["position"]
                residual = math.hypot(p1["position"]["x"] - ps["x"], p1["position"]["y"] - ps["y"])
            legs[name] = {"cmd": (vx, vy, wz), "t_run": t_run,
                          "fwd": dx * math.cos(th0) + dy * math.sin(th0),
                          "lat": -dx * math.sin(th0) + dy * math.cos(th0),
                          "yaw": wrap(yaw_of(p1["orientation"]) - th0), "residual": residual}
    finally:
        rb.close()
    return legs


def drive_tolerances(rid: str) -> tuple:
    """(displacement, rotation, residual) tolerances of the robot's recorded drive."""
    yaw = tol(rid, "yaw change") if rid == "rosmaster_x3_plus" else tol(rid, "rotation")
    return tol(rid, "displacement"), yaw, tol_row(rid, "residual")


def drive(port: int, rid: str, odom_topic: str = "/odom") -> list:
    """The drive's legs judged from odometry with the recorded tolerances: each
    displacement against the commanded one, and what is left after the stop against the
    recorded residual figure. Returns the problems found."""
    lin_t, yaw_t, res = drive_tolerances(rid)
    problems = []
    for name, leg in drive_legs(port, rid, odom_topic).items():
        vx, vy, wz = leg["cmd"]
        t_run = leg["t_run"]
        if name == "turn":
            if not within(leg["yaw"], wz * t_run, yaw_t):
                problems.append(f"turn: odom yaw {leg['yaw']:.3f} rad, commanded {wz * t_run:.3f}")
        else:
            # the displacement vector, within the tolerance of the commanded one
            want = math.hypot(vx, vy) * t_run
            err = math.hypot(leg["fwd"] - vx * t_run, leg["lat"] - vy * t_run)
            if err > max(want * lin_t.get("relative", 0.0), lin_t.get("absolute", 0.0)):
                problems.append(f"{name}: odom ({leg['fwd']:.3f}, {leg['lat']:.3f}) m, commanded "
                                f"({vx * t_run:.3f}, {vy * t_run:.3f})")
        if leg["residual"] > res["tolerance"]["absolute"] + 1e-9:
            problems.append(f"{name}: {res['figure']} is {leg['residual']:.3f} "
                            f"{res['tolerance'].get('unit', '')}, recorded at most "
                            f"{res['tolerance']['absolute']}")
    return problems


def beyond(limit: float) -> float:
    """A command half as far again beyond a recorded maximum."""
    return limit + 0.5 * abs(limit) if limit else 0.5


def drive_limits(port: int, rid: str, odom_topic: str = "/odom") -> list:
    """Each limited Twist field of the recorded drive (`motions[drive].command.fields`)
    commanded beyond its maximum (`beyond`), briefly and then back the other way, so the
    run stays inside the travel the floor placement guarantees: the odometry never reports
    more than the recorded maximum, within the recorded displacement or rotation
    tolerance. Returns the problems found."""
    row = next(m for m in wirecheck.interface(rid)["motions"] if m["id"] == "drive")
    lin_t, yaw_t, _ = drive_tolerances(rid)
    problems = []
    rb = Rosbridge("127.0.0.1", port)
    try:
        odom = Latest(rb, odom_topic)
        odom.wait()
        rb.advertise("/cmd_vel", "geometry_msgs/Twist")
        time.sleep(0.8)
        for f in row["command"]["fields"]:
            if f.get("max") is None or f["field"] not in ("linear.x", "linear.y", "angular.z"):
                continue
            part, axis = f["field"].split(".")
            hi = float(f["max"])
            seconds = 0.2 if f["field"] == "linear.y" else 0.3
            peak = 0.0
            for sign in (1.0, -1.0):   # out, then back
                cmd = twist()
                cmd[part][axis] = sign * beyond(hi)
                t0 = time.time()
                rb.publish("/cmd_vel", cmd)
                time.sleep(seconds)
                rb.publish("/cmd_vel", twist())
                time.sleep(1.0)
                peak = max([peak] + [abs(m["twist"]["twist"][part][axis]) for m in odom.since(t0)])
            t = yaw_t if part == "angular" else lin_t
            if peak > hi + max(hi * t.get("relative", 0.0), t.get("absolute", 0.0)) + 1e-9:
                problems.append(f"{f['field']} commanded at {beyond(hi)}: odometry reports "
                                f"{peak:.3f}, recorded maximum {hi}")
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


def peak_joint(sim_port, rid, name: str, seconds: float) -> float:
    """The largest position a joint reaches over `seconds`."""
    t0 = time.monotonic()
    peak = -math.inf
    while time.monotonic() - t0 < seconds:
        peak = max(peak, joints(sim_port, rid)[name])
        time.sleep(0.1)
    return peak


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
