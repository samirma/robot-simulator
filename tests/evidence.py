#!/usr/bin/env python3
"""Workspace evidence (workspace spec §3): one case per recorded robot per engine.

    tests/evidence.sh                                   # every robot on both engines
    tests/evidence.sh --engine robocasa --robot ainex   # a subset (repeatable flags)
    tests/evidence.sh --index-only                      # rebuild evidences/index.{md,json}

A case starts the engine's default scene with `simulator/kitchen.sh start --engine <e>`,
spawns the robot with `simulator/spawn.sh` (mobile robots on the floor, arms on the
worktop), and records into `evidences/<engine>/<id>/`:

* `scene.png` -- an offscreen render of the simulation framing the robot in its scene;
* `cam__<topic>.png` -- one frame of every camera not marked `optional` in the robot's
  interface file, taken off the vendor wire through rosbridge and decoded;
* `smoke.json` + `motion_<name>_{before,after}.png` -- the motion smoke run through the
  vendor wire with each robot's recorded command, judged by recorded feedback (or, where
  the interface publishes none, by readings from the simulation's control port) against
  the tolerances recorded in `robots_specs/<id>/ros*.yml`;
* `fleet_<profile>.txt` -- `robot_console/python.sh -m robot_console.fleet --expect` for
  every console profile naming the robot;
* `case.json` -- robot, engine, scene, placement, ports, readiness line, timestamps, host
  load, per-item results and the verdict.

Independence (workspace spec §1.4): the smoke-run commands and bounds come from
`robots_specs/` (normative for the simulator); the console's verdict is its own exit
status. Nothing here imports either project; both are exercised as installed, over
rosbridge and the documented control port.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_io import (Latest, Rosbridge, SimPort, decode_image, png_bytes,  # noqa: E402
                         png_stats, wrap, yaw_of_quat_xyzw, yaw_of_wxyz)

REPO = Path(__file__).resolve().parent.parent
SPECS = REPO / "robots_specs"
SIM = REPO / "simulator"
CONSOLE = REPO / "robot_console"
ENGINES = ["molmospaces", "robocasa"]


# =========================================================================== registry


def registry() -> List[dict]:
    """Robots of the robot files robots_specs/<id>.md: id, name, kind."""
    out = []
    for p in sorted(SPECS.glob("*.md")):
        if p.name in ("high_level_spec.md", "SCHEMA.md"):
            continue
        text = p.read_text()
        m = re.search(r"\*\*Robot id:\*\*\s*`([^`]+)`", text)
        if not m:
            continue
        r = {"id": m.group(1), "name": text.splitlines()[0].lstrip("# ").strip()}
        k = re.search(r"\*\*Kind:\*\*\s*`([^`]+)`", text)
        r["kind"] = k.group(1) if k else None
        out.append(r)
    return out


def interface(rid: str) -> dict:
    for name in ("ros.yml", "ros2.yml"):
        p = SPECS / rid / name
        if p.exists():
            return yaml.safe_load(p.read_text())
    raise FileNotFoundError(f"no interface file for {rid}")


def tolerance(rid: str, fragment: str) -> dict:
    for t in interface(rid).get("tolerances") or []:
        if fragment in t["figure"]:
            return {"figure": t["figure"], "tolerance": t["tolerance"], "basis": t.get("basis"),
                    "source": f"robots_specs/{rid}/{'ros.yml' if (SPECS / rid / 'ros.yml').exists() else 'ros2.yml'} tolerances"}
    raise KeyError(f"{rid}: no tolerance '{fragment}'")


def bound_of(tol: dict, commanded: float) -> float:
    """The acceptance bound of a recorded tolerance for a commanded figure: the larger of
    its relative part (of |commanded|) and its absolute part."""
    t = tol["tolerance"]
    return max(abs(commanded) * t.get("relative", 0.0), t.get("absolute", 0.0))


def motion_row(rid: str, mid: str, joint: Optional[str] = None) -> dict:
    for m in interface(rid)["motions"]:
        if m["id"] == mid and (joint is None or m.get("joint") == joint):
            return m
    raise KeyError(f"{rid}: no motion {mid}")


def console_profiles() -> List[str]:
    """The console's packaged profile ids (read from its tree; the console judges)."""
    out = []
    for p in sorted((CONSOLE / "src" / "robot_console" / "profiles").glob("*.yaml")):
        m = re.search(r"^id:\s*(\S+)", p.read_text(), re.M)
        if m:
            out.append(m.group(1))
    return out


# =========================================================================== processes


def port_free(port: int) -> bool:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def wait_port_free(port: int, timeout: float = 30.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if port_free(port):
            return True
        time.sleep(0.3)
    return False


class Proc:
    """A child process (own process group) logging to a file."""

    def __init__(self, args: List[str], log: Path):
        self.args, self.log = args, log
        self.logf = open(log, "w")
        self.proc = subprocess.Popen(args, cwd=REPO, stdout=self.logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def text(self) -> str:
        try:
            return self.log.read_text(errors="replace")
        except FileNotFoundError:
            return ""

    def wait_for(self, prefix: str, timeout: float) -> str:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            for line in self.text().splitlines():
                if line.startswith(prefix):
                    return line
            if self.proc.poll() is not None:
                raise RuntimeError(f"{Path(self.args[0]).name} exited {self.proc.returncode}: "
                                   f"{self.text()[-2500:]}")
            time.sleep(0.5)
        raise TimeoutError(f"no '{prefix}' within {timeout:.0f} s: {self.text()[-2500:]}")

    def stop(self, timeout: float = 90.0) -> Optional[int]:
        """SIGINT to the process itself (its documented stop), SIGKILL to the group last."""
        if self.proc.poll() is None:
            try:
                os.kill(self.proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(15)
        self.logf.close()
        return self.proc.returncode


def host_load() -> dict:
    l1, l5, l15 = os.getloadavg()
    return {"load1": round(l1, 2), "load5": round(l5, 2), "load15": round(l15, 2),
            "ncpu": os.cpu_count()}


def wait_low_load(max_load: float, max_wait: float, say) -> dict:
    t0 = time.monotonic()
    while True:
        ld = host_load()
        if ld["load1"] <= max_load:
            ld["waited_s"] = round(time.monotonic() - t0, 1)
            return ld
        if time.monotonic() - t0 > max_wait:
            ld["waited_s"] = round(time.monotonic() - t0, 1)
            ld["note"] = f"load still above {max_load} after {max_wait:.0f} s; ran anyway"
            say(f"  load {ld['load1']} > {max_load} after {max_wait:.0f} s; running anyway")
            return ld
        say(f"  host load {ld['load1']} > {max_load}; waiting")
        time.sleep(20)


def docker_left(sim_port: int) -> List[str]:
    try:
        out = subprocess.run(["docker", "ps", "-a", "--filter", f"label=rsim.sim_port={sim_port}",
                              "--format", "{{.Names}}"], stdout=subprocess.PIPE, text=True,
                             timeout=30).stdout
        return [x for x in out.split() if x]
    except Exception as exc:
        return [f"docker ps failed: {exc}"]


def parse_ready(line: str) -> List[dict]:
    """Wires of a readiness line: `... wire(s): <owner> ws://h:p [ROS n distro]` or
    `base (<owner>) ws://... ; arm (<owner>) ws://...`."""
    wires = []
    tail = line.split("wire(s):", 1)[-1]
    for part in tail.split(";"):
        m = re.search(r"(?:(\w+) \((\w+)\)|(\w+)) ws://([\d.]+):(\d+)\s*\[([^\]]+)\]", part)
        if m:
            role = m.group(1) or "main"
            owner = m.group(2) or m.group(3)
            wires.append({"role": role, "owner": owner, "host": m.group(4),
                          "port": int(m.group(5)), "ros": m.group(6),
                          "url": f"ws://{m.group(4)}:{m.group(5)}"})
    return wires


# =========================================================================== rendering


def robot_bbox(rd: dict):
    pts = np.array([b["pos"] for b in rd["bodies"].values()])
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)), 0.25)


def choose_view(sp: SimPort, rid: str, mobile: bool, extra=()) -> dict:
    """A free-camera view showing the whole robot (and the bodies named in `extra`: an
    arm's worktop objects) with a clear line of sight and as much of its surroundings as
    possible: candidate eyes around the robot, each checked with a depth render (nothing
    nearer than the robot inside the robot's image window)."""
    rd = sp.call("readings", robot=rid)
    if extra:
        rd = dict(rd, bodies=dict(rd["bodies"], **sp.call("bodies", names=list(extra))["bodies"]))
    centre, size = robot_bbox(rd)
    yaw = math.degrees(yaw_of_wxyz(rd["base"]["quat"]))
    W, H, fovy = 160, 90, 45.0
    if mobile:
        dists = [max(1.3, 2.6 * size), max(1.0, 2.0 * size), max(0.8, 1.5 * size)]
        els = [-28.0]
    else:
        dists = [max(1.3, 2.6 * size), max(1.0, 2.0 * size), max(0.8, 1.5 * size)]
        els = [-25.0]
    best = None
    for dist in dists:
        for el in els:
            for daz in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
                view = {"lookat": [float(x) for x in centre], "distance": dist,
                        "azimuth": yaw + daz + 180.0, "elevation": el}
                r = sp.call("render", view=view, width=W, height=H, format="depth")
                d = np.frombuffer(r["_payload"], np.float32).reshape(H, W)
                half = math.degrees(math.atan2(0.6 * size, dist))
                ph = int(min(H / 2 - 1, max(2, half / fovy * H)))
                pw = int(min(W / 2 - 1, max(2, ph)))
                win = d[H // 2 - ph:H // 2 + ph, W // 2 - pw:W // 2 + pw]
                near = float((win < dist - 0.6 * size - 0.05).mean())
                if near > 0.03:
                    continue
                context = float(np.clip(d, 0, 8).mean())
                score = context - 0.4 * abs(daz) / 180.0
                if best is None or score > best[0]:
                    best = (score, view, near, context)
        if best is not None:
            break
    if best is None:
        return {"frame_robot": rid, "elevation": -30.0}
    return dict(best[1], _clear_fraction=round(1 - best[2], 3), _context_m=round(best[3], 2))


def render(sp: SimPort, view: dict, path: Path, width=1280, height=720) -> dict:
    v = {k: x for k, x in view.items() if not k.startswith("_")}
    r = sp.call("render", view=v, width=width, height=height, format="png")
    path.write_bytes(r["_payload"])
    st = png_stats(r["_payload"])
    return {"file": path.name, "view": v, "sim_time": r.get("sim_time"), **st}


# =========================================================================== motions


def twist(x=0.0, y=0.0, z=0.0) -> dict:
    return {"linear": {"x": x, "y": y, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": z}}


class Ctx:
    """Everything a motion needs."""

    def __init__(self, rid, sp: SimPort, wires, outdir: Path, view: dict, say):
        self.rid, self.sp, self.wires, self.out, self.view, self.say = rid, sp, wires, outdir, view, say

    def wire(self, owner: str) -> dict:
        for w in self.wires:
            if w["owner"] == owner:
                return w
        raise KeyError(owner)

    def readings(self) -> dict:
        return self.sp.call("readings", robot=self.rid)

    def joints(self) -> Dict[str, float]:
        return {k: v["position"] for k, v in self.readings()["joints"].items()}

    def shot(self, name: str, when: str) -> str:
        p = self.out / f"motion_{name}_{when}.png"
        render(self.sp, self.view, p)
        return p.name


def run_motion(ctx: Ctx, name: str, fn) -> dict:
    """One motion with its before/after renders; an exception fails the motion."""
    rec = {"name": name}
    try:
        rec["renders"] = {"before": ctx.shot(name, "before")}
        rec.update(fn())
        time.sleep(0.3)
        rec["renders"]["after"] = ctx.shot(name, "after")
    except Exception as exc:
        rec["pass"] = False
        rec.setdefault("problems", []).append(f"error: {exc!r}")
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec.setdefault("pass", not rec.get("problems"))
    ctx.say(f"    motion {name}: {'PASS' if rec['pass'] else 'FAIL'}"
            + ("" if rec["pass"] else f" {rec.get('problems')}"))
    return rec


# --------------------------------------------------------------------------- wheeled base

DRIVES = [("forward", 0.2, 0.0, 0.0, 1.25), ("back", -0.2, 0.0, 0.0, 1.25),
          ("left", 0.0, 0.2, 0.0, 1.25), ("right", 0.0, -0.2, 0.0, 1.25),
          ("turn", 0.0, 0.0, 0.5, 2.0)]


def stamp_of(msg: dict) -> float:
    st = msg["header"]["stamp"]
    return st.get("secs", st.get("sec", 0)) + st.get("nsecs", st.get("nanosec", 0)) * 1e-9


def pose_at(history, t: float):
    """The odometry pose at wall time t, linearly interpolated between the two messages
    whose header stamps bracket it (position; orientation of the nearer one)."""
    rows = sorted(((stamp_of(m), m) for _, m in history), key=lambda r: r[0])
    before = [r for r in rows if r[0] <= t]
    after = [r for r in rows if r[0] > t]
    if not before:
        r = after[0]
        return r[1]["pose"]["pose"], r[0] - t
    if not after:
        r = before[-1]
        return r[1]["pose"]["pose"], r[0] - t
    (ta, ma), (tb, mb) = before[-1], after[0]
    f = (t - ta) / (tb - ta) if tb > ta else 0.0
    pa, pb = ma["pose"]["pose"], mb["pose"]["pose"]
    pos = {k: pa["position"][k] + f * (pb["position"][k] - pa["position"][k]) for k in ("x", "y", "z")}
    near = pa if f < 0.5 else pb
    return {"position": pos, "orientation": near["orientation"]}, (ta if f < 0.5 else tb) - t


def drive_motions(ctx: Ctx, owner: str) -> List[dict]:
    """Forward, back, left, right and an in-place turn on the base wire, each a timed
    recorded /cmd_vel command followed by the recorded stop (zero Twist), confirmed by
    the recorded odometry. Travel: 0.25 m forward/back/side and back to the start."""
    w = ctx.wire(owner)
    row = motion_row(owner, "drive")
    stop_msg = row["stop"]["message"]
    rb = Rosbridge(w["url"])
    out = []
    try:
        odom_topic = "/odom"
        odom = Latest(rb, odom_topic)
        odom.keep = True
        odom.wait(20)
        extra = {}
        if owner == "rosmaster_x3_plus":
            extra["vel_raw"] = Latest(rb, "/vel_raw")
            extra["odom_raw"] = Latest(rb, "/odom_raw")
        if owner == "myagv":
            lin_tol = tolerance(owner, "drive displacement")
            yaw_tol = tolerance(owner, "drive rotation")
            rest_tol = tolerance(owner, "residual chassis speed")
        else:
            lin_tol = tolerance(owner, "translation displacement")
            yaw_tol = tolerance(owner, "yaw change")
            rest_tol = tolerance(owner, "residual motion after the stop")
        rb.advertise(row["command"]["name"], row["command"]["type"])
        time.sleep(1.0)
        # the stop once first: the robot starts from rest under the recorded stop
        rb.publish(row["stop"]["name"], stop_msg)
        time.sleep(0.5)
        for name, vx, vy, wz, dur in DRIVES:
            def one(name=name, vx=vx, vy=vy, wz=wz, dur=dur):
                problems = []
                p0 = odom.next()["pose"]["pose"]
                s0 = ctx.readings()["base"]
                cmd = twist(vx, vy, wz)
                t0 = time.monotonic()
                rb.publish(row["command"]["name"], cmd)
                time.sleep(dur)
                rb.publish(row["stop"]["name"], stop_msg)
                t_stop_wall = time.time()
                t_run = time.monotonic() - t0
                time.sleep(0.5)
                m05 = odom.next()
                vr05 = extra["vel_raw"].msg if "vel_raw" in extra else None
                time.sleep(1.0)
                m15 = odom.next()
                vr15 = extra["vel_raw"].msg if "vel_raw" in extra else None
                p1 = m15["pose"]["pose"]
                s1 = ctx.readings()["base"]
                # the odometry's own pose at the instant of the stop command, by its
                # header stamps (the wire's ROS time is the host's wall clock)
                p_stop, stop_lag = pose_at(odom.history, t_stop_wall)
                odom.history.clear()
                th0 = yaw_of_quat_xyzw(p0["orientation"])

                def local(pa, pb):
                    dx = pb["position"]["x"] - pa["position"]["x"]
                    dy = pb["position"]["y"] - pa["position"]["y"]
                    return (dx * math.cos(th0) + dy * math.sin(th0),
                            -dx * math.sin(th0) + dy * math.cos(th0),
                            wrap(yaw_of_quat_xyzw(pb["orientation"]) - yaw_of_quat_xyzw(pa["orientation"])))

                fwd, lat, dth = local(p0, p1)
                want = (vx * t_run, vy * t_run, wz * t_run)
                rec = {"wire": w["url"], "command": {"topic": row["command"]["name"],
                                                     "type": row["command"]["type"],
                                                     "message": cmd, "duration_s": round(t_run, 3)},
                       "stop": {"topic": row["stop"]["name"], "message": stop_msg},
                       "feedback": odom_topic,
                       "commanded": {"forward_m": round(want[0], 4), "left_m": round(want[1], 4),
                                     "yaw_rad": round(want[2], 4)},
                       "measured": {"forward_m": round(fwd, 4), "left_m": round(lat, 4),
                                    "yaw_rad": round(dth, 4)}}
                sy0 = yaw_of_wxyz(s0["quat"])
                sdx, sdy = s1["pos"][0] - s0["pos"][0], s1["pos"][1] - s0["pos"][1]
                rec["sim_readings_crosscheck"] = {
                    "forward_m": round(sdx * math.cos(sy0) + sdy * math.sin(sy0), 4),
                    "left_m": round(-sdx * math.sin(sy0) + sdy * math.cos(sy0), 4),
                    "yaw_rad": round(wrap(yaw_of_wxyz(s1["quat"]) - sy0), 4)}
                if name == "turn":
                    b = bound_of(yaw_tol, want[2])
                    err = abs(dth - want[2])
                    rec["bound"] = {"figure": yaw_tol["figure"], "tolerance": yaw_tol["tolerance"],
                                    "bound": round(b, 4), "unit": "rad",
                                    "rule": "|measured yaw - commanded yaw| <= max(relative*|commanded|, absolute)"}
                    trans = math.hypot(fwd, lat)
                    rec["measured"]["translation_m"] = round(trans, 4)
                    lb = bound_of(lin_tol, 0.0)
                    if err > b:
                        problems.append(f"yaw {dth:.3f} rad vs commanded {want[2]:.3f} (bound {b:.3f})")
                    if lb and trans > lb:
                        problems.append(f"in-place turn translated {trans:.3f} m (> {lb:.3f})")
                else:
                    want_d = math.hypot(want[0], want[1])
                    err = math.hypot(fwd - want[0], lat - want[1])
                    b = bound_of(lin_tol, want_d)
                    rec["bound"] = {"figure": lin_tol["figure"], "tolerance": lin_tol["tolerance"],
                                    "bound": round(b, 4), "unit": "m",
                                    "rule": "|measured - commanded displacement vector| <= max(relative*|commanded|, absolute)"}
                    if err > b:
                        problems.append(f"displacement ({fwd:.3f}, {lat:.3f}) m vs commanded "
                                        f"({want[0]:.3f}, {want[1]:.3f}) (error {err:.3f} > {b:.3f})")
                rec["error"] = round(err, 4)
                # at rest after the stop. Rotation is compared at the wheel rim: a yaw
                # (rate) times the recorded (lx + ly) is the rim travel (speed) it causes.
                kin = row.get("kinematics") or {}
                lxly = kin.get("lx_plus_ly") or (kin.get("lx", 0.0) + kin.get("ly", 0.0))
                rest = {}
                if owner == "myagv":
                    tw = m05["twist"]["twist"]
                    speed = math.hypot(tw["linear"]["x"], tw["linear"]["y"])
                    rim = abs(tw["angular"]["z"]) * lxly
                    lim = rest_tol["tolerance"]["absolute"]
                    a05, a15 = m05["pose"]["pose"], p1
                    dpos = math.hypot(a15["position"]["x"] - a05["position"]["x"],
                                      a15["position"]["y"] - a05["position"]["y"])
                    dyaw = abs(wrap(yaw_of_quat_xyzw(a15["orientation"]) - yaw_of_quat_xyzw(a05["orientation"])))
                    rest = {"figure": rest_tol["figure"], "tolerance": rest_tol["tolerance"],
                            "lx_plus_ly_m": lxly,
                            "odom_twist_0_5s": {"linear_speed_mps": round(speed, 4),
                                                "angular_z_radps": tw["angular"]["z"],
                                                "angular_at_rim_mps": round(rim, 4)},
                            "odom_change_0_5_to_1_5s": {"translation_m": round(dpos, 4),
                                                        "yaw_rad": round(dyaw, 4),
                                                        "yaw_at_rim_m": round(dyaw * lxly, 4)},
                            "rule": f"/odom twist 0.5 s after the zero Twist: linear speed and |angular.z|*(lx+ly) "
                                    f"<= {lim} m/s; then /odom still for 1 s: translation and |yaw|*(lx+ly) <= {lim} m"}
                    if speed > lim + 1e-9 or rim > lim + 1e-9:
                        problems.append(f"not at rest 0.5 s after the stop: {speed:.3f} m/s, "
                                        f"{tw['angular']['z']:.3f} rad/s")
                    if dpos > lim + 1e-9 or dyaw * lxly > lim + 1e-9:
                        problems.append(f"odometry still moving after the stop: {dpos:.4f} m, {dyaw:.4f} rad")
                else:
                    lim = rest_tol["tolerance"]["absolute"]
                    resid = math.hypot(p1["position"]["x"] - p_stop["position"]["x"],
                                       p1["position"]["y"] - p_stop["position"]["y"])
                    ryaw = abs(wrap(yaw_of_quat_xyzw(p1["orientation"]) - yaw_of_quat_xyzw(p_stop["orientation"])))
                    vr = vr15 or {}
                    rest = {"figure": rest_tol["figure"], "tolerance": rest_tol["tolerance"],
                            "lx_plus_ly_m": lxly,
                            "stop_pose_from": f"/odom interpolated at the stop's wall time by header stamp "
                                              f"(nearest stamp {stop_lag:+.3f} s)",
                            "travel_after_stop_m": round(resid, 4),
                            "yaw_after_stop_rad": round(ryaw, 4),
                            "yaw_after_stop_at_rim_m": round(ryaw * lxly, 4),
                            "vel_raw_1_5s_info": {"linear": vr.get("linear"), "angular": vr.get("angular")},
                            "rule": f"/odom from the zero Twist to 1.5 s after it: travel <= {lim} m and "
                                    f"|yaw|*(lx+ly) <= {lim} m"}
                    if resid > lim + 1e-9:
                        problems.append(f"travelled {resid:.3f} m after the stop (> {lim})")
                    if ryaw * lxly > lim + 1e-9:
                        problems.append(f"turned {ryaw:.3f} rad after the stop (rim {ryaw * lxly:.3f} m > {lim})")
                rec["rest"] = rest
                rec["problems"] = problems
                return rec
            out.append(run_motion(ctx, f"drive_{name}", one))
        rb.publish(row["stop"]["name"], stop_msg)
    finally:
        rb.close()
    return out


# --------------------------------------------------------------------------- joints helpers


def wait_settled(get, names, target: Dict[str, float], tol_of, timeout: float,
                 still: float, window: float = 0.5) -> dict:
    """Poll `get()` (name -> position) until every named joint is within its tolerance of
    the target and moved less than `still` over `window`; returns the last samples."""
    t0 = time.monotonic()
    prev = get()
    last = prev
    reached_at = None
    while True:
        time.sleep(window)
        cur = get()
        err = {k: abs(cur[k] - v) for k, v in target.items()}
        moved = {k: abs(cur[k] - prev[k]) for k in names}
        ok = all(err[k] <= tol_of(k) for k in target) and all(m < still for m in moved.values())
        if ok and reached_at is None:
            reached_at = time.monotonic() - t0
        last = {"positions": cur, "error": err, "moved_last_window": moved,
                "settled": ok, "t_s": round(time.monotonic() - t0, 2)}
        if ok or time.monotonic() - t0 > timeout:
            return last
        prev = cur


def rounded(d: dict, n=4) -> dict:
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- SO-101


def so101_motions(ctx: Ctx) -> List[dict]:
    w = ctx.wire("so101")
    arm = motion_row("so101", "arm")
    grip = motion_row("so101", "gripper")
    t_arm = tolerance("so101", "arm joint")
    t_grip = tolerance("so101", "gripper final")
    names = arm["command"]["example"]["joint_names"]
    target = arm["command"]["example"]["points"][0]["positions"]
    rb = Rosbridge(w["url"])
    out = []
    try:
        js = Latest(rb, "/joint_states")
        js.wait(20)

        def fb() -> Dict[str, float]:
            m = js.next()
            return dict(zip(m["name"], m["position"]))

        def fbv() -> Dict[str, float]:
            m = js.msg
            return dict(zip(m["name"], m.get("velocity") or [0.0] * len(m["name"])))

        rb.advertise(arm["command"]["name"], "trajectory_msgs/msg/JointTrajectory")
        time.sleep(1.0)

        def do_arm():
            ex = arm["command"]["example"]
            msg = {"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                   "joint_names": names,
                   "points": [{"positions": target, "velocities": [], "accelerations": [],
                               "effort": [], "time_from_start": ex["points"][0]["time_from_start"]}]}
            before = fb()
            rb.publish(arm["command"]["name"], msg)
            lim = t_arm["tolerance"]["absolute"]
            res = wait_settled(fb, names, dict(zip(names, target)), lambda k: lim, 12.0, still=0.002)
            vel = fbv()
            problems = []
            for k in names:
                if res["error"][k] > lim:
                    problems.append(f"{k}: {res['positions'][k]:.4f} vs {dict(zip(names, target))[k]} (> {lim})")
            vmax = max(abs(vel.get(k, 0.0)) for k in names)
            if vmax >= 0.01:
                problems.append(f"not at rest: |velocity| {vmax:.4f} rad/s")
            if not res["settled"]:
                problems.append("did not settle at the goal")
            sim = ctx.joints()
            return {"wire": w["url"], "command": {"topic": arm["command"]["name"],
                                                  "type": arm["command"]["type"], "message": msg},
                    "feedback": "/joint_states (joint_state_broadcaster: measured positions/velocities)",
                    "commanded": dict(zip(names, target)),
                    "before": rounded({k: before[k] for k in names}),
                    "measured": rounded({k: res["positions"][k] for k in names}),
                    "error": rounded(res["error"]),
                    "velocity_after": rounded({k: vel.get(k) for k in names}),
                    "sim_readings_crosscheck": rounded({k: sim[k] for k in names}),
                    "bound": {"figure": t_arm["figure"], "tolerance": t_arm["tolerance"],
                              "rule": "every joint within the absolute tolerance of the goal; at rest = "
                                      "|velocity| < 0.01 rad/s (end_state: stopped_velocity_tolerance)"},
                    "settle_time_s": res["t_s"], "problems": problems}

        def do_grip():
            ex = grip["command"]["example"]
            goal = {"command": {"name": ex["command"]["name"], "position": ex["command"]["position"],
                                "velocity": [], "effort": []}}
            want = float(ex["command"]["position"][0])
            before = fb()["gripper_joint"]
            res = rb.action(grip["command"]["name"], grip["command"]["type"], goal, timeout=30)
            lim = t_grip["tolerance"]["absolute"]
            st = wait_settled(fb, ["gripper_joint"], {"gripper_joint": want}, lambda k: lim, 8.0, still=0.002)
            vel = fbv().get("gripper_joint", 0.0)
            problems = []
            values = res.get("values") or {}
            if not res.get("result", False):
                problems.append(f"action result not successful: {res}")
            if st["error"]["gripper_joint"] > lim:
                problems.append(f"gripper_joint {st['positions']['gripper_joint']:.4f} vs {want} (> {lim})")
            if abs(vel) >= 0.01:
                problems.append(f"gripper not at rest ({vel:.4f} rad/s)")
            return {"wire": w["url"], "command": {"action": grip["command"]["name"],
                                                  "type": grip["command"]["type"], "goal": goal},
                    "feedback": "action result + /joint_states gripper_joint",
                    "commanded": {"gripper_joint": want}, "before": {"gripper_joint": round(before, 4)},
                    "action_result": values,
                    "measured": {"gripper_joint": round(st["positions"]["gripper_joint"], 4)},
                    "error": round(st["error"]["gripper_joint"], 4),
                    "velocity_after": vel,
                    "sim_readings_crosscheck": {"gripper_joint": round(ctx.joints()["gripper_joint"], 4)},
                    "bound": {"figure": t_grip["figure"], "tolerance": t_grip["tolerance"],
                              "rule": "action succeeds and gripper_joint within the absolute tolerance; at rest"},
                    "problems": problems}

        out.append(run_motion(ctx, "arm", do_arm))
        out.append(run_motion(ctx, "gripper", do_grip))
    finally:
        rb.close()
    return out


# --------------------------------------------------------------------------- myCobot 280


def mycobot_motions(ctx: Ctx) -> List[dict]:
    """Arm, then gripper, through /joint_states on the myCobot wire; judged by the
    simulation's joint readings (the boot publishes no measured feedback)."""
    w = ctx.wire("mycobot280")
    arm = motion_row("mycobot280", "arm")
    grip = motion_row("mycobot280", "gripper")
    t_arm = tolerance("mycobot280", "arm joint")
    t_grip = tolerance("mycobot280", "gripper reaches")
    ex = arm["command"]["example"]
    names, target = ex["name"], ex["position"]
    arm_names = names[:6]
    rb = Rosbridge(w["url"])
    out = []

    def jget():
        return ctx.joints()

    try:
        rb.advertise(arm["command"]["name"], "sensor_msgs/msg/JointState")
        time.sleep(1.0)

        def msg_of(pos):
            return {"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""},
                    "name": names, "position": pos, "velocity": [], "effort": []}

        def do_arm():
            before = jget()
            g_now = before["gripper_controller"]
            # the node truncates the gripper value int((p + 0.74) / 0.89 * 100): send the
            # current opening so that only the arm moves (interface note)
            pos = list(target[:6]) + [round(g_now, 4)]
            msg = msg_of(pos)
            rb.publish(arm["command"]["name"], msg)
            lim = t_arm["tolerance"]["absolute"]
            res = wait_settled(jget, arm_names, dict(zip(arm_names, target[:6])), lambda k: lim,
                               15.0, still=0.002)
            problems = [f"{k}: {res['positions'][k]:.4f} vs {v} (> {lim})"
                        for k, v in zip(arm_names, target[:6]) if res["error"][k] > lim]
            if not res["settled"]:
                problems.append("did not settle at the goal (still moving or outside the tolerance)")
            return {"wire": w["url"], "command": {"topic": arm["command"]["name"],
                                                  "type": arm["command"]["type"], "message": msg},
                    "feedback": "none on the wire (interface feedback: []); judged by sim-port readings",
                    "commanded": dict(zip(arm_names, target[:6])),
                    "before": rounded({k: before[k] for k in arm_names}),
                    "measured": rounded({k: res["positions"][k] for k in arm_names}),
                    "error": rounded({k: res["error"][k] for k in arm_names}),
                    "moved_last_0_5s": rounded(res["moved_last_window"], 5),
                    "bound": {"figure": t_arm["figure"], "tolerance": t_arm["tolerance"],
                              "rule": "each arm joint within the absolute tolerance; at rest = moved < 0.002 rad over 0.5 s"},
                    "settle_time_s": res["t_s"], "problems": problems}

        def do_grip():
            before = jget()
            want = -0.5 if before["gripper_controller"] > -0.2 else 0.15
            pos = [round(before[k], 4) for k in arm_names] + [want]
            msg = msg_of(pos)
            rb.publish(grip["command"]["name"], msg)
            lim = t_grip["tolerance"]["absolute"]
            res = wait_settled(jget, ["gripper_controller"], {"gripper_controller": want},
                               lambda k: lim, 8.0, still=0.002)
            problems = []
            if res["error"]["gripper_controller"] > lim:
                problems.append(f"gripper_controller {res['positions']['gripper_controller']:.4f} vs {want} (> {lim})")
            if not res["settled"]:
                problems.append("gripper did not settle")
            return {"wire": w["url"], "command": {"topic": grip["command"]["name"],
                                                  "type": grip["command"]["type"], "message": msg},
                    "feedback": "none on the wire (interface feedback: []); judged by sim-port readings",
                    "commanded": {"gripper_controller": want,
                                  "pymycobot_value": int((want + 0.74) / 0.89 * 100)},
                    "before": {"gripper_controller": round(before["gripper_controller"], 4)},
                    "measured": {"gripper_controller": round(res["positions"]["gripper_controller"], 4)},
                    "error": round(res["error"]["gripper_controller"], 4),
                    "bound": {"figure": t_grip["figure"], "tolerance": t_grip["tolerance"],
                              "rule": "gripper within the absolute tolerance; at rest = moved < 0.002 rad over 0.5 s"},
                    "settle_time_s": res["t_s"], "problems": problems}

        out.append(run_motion(ctx, "arm", do_arm))
        out.append(run_motion(ctx, "gripper", do_grip))
    finally:
        rb.close()
    return out


# --------------------------------------------------------------------------- ROSMASTER X3 PLUS


def rosmaster_arm_motions(ctx: Ctx) -> List[dict]:
    """/TargetAngle (yahboomcar_msgs/ArmJoint): the arm (all six servos, gripper held)
    and the gripper (single-servo form, the recorded example); judged by the measured
    servo angles of the /CurrentAngle service."""
    w = ctx.wire("rosmaster_x3_plus")
    arm = motion_row("rosmaster_x3_plus", "arm")
    grip = motion_row("rosmaster_x3_plus", "gripper")
    t_arm = tolerance("rosmaster_x3_plus", "arm joint")
    t_grip = tolerance("rosmaster_x3_plus", "gripper reached")
    rb = Rosbridge(w["url"])
    out = []
    names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint"]

    def measured() -> List[float]:
        r = rb.call("/CurrentAngle", {}, timeout=10)
        return [float(a) for a in r.get("angles", [])]

    def to_joint(i: int, deg: float) -> float:
        # recorded mappings: arm joint = servo deg - 90; gripper 30..180 deg -> -pi/2..0
        if i < 5:
            return math.radians(deg - 90.0)
        return (deg - 180.0) / 150.0 * (math.pi / 2)

    try:
        rb.advertise(arm["command"]["name"], arm["command"]["type"])
        time.sleep(1.0)

        def settle(want_deg: Dict[int, float], timeout: float):
            t0 = time.monotonic()
            prev = measured()
            while True:
                time.sleep(0.5)
                cur = measured()
                errs = {i: abs(to_joint(i, cur[i]) - to_joint(i, d)) for i, d in want_deg.items()}
                lims = {i: (t_grip if i == 5 else t_arm)["tolerance"]["absolute"] for i in want_deg}
                still = all(abs(cur[i] - prev[i]) < 0.5 for i in range(6))
                ok = all(errs[i] <= lims[i] for i in want_deg) and still
                if ok or time.monotonic() - t0 > timeout:
                    return cur, errs, lims, ok, round(time.monotonic() - t0, 2)
                prev = cur

        def do_arm():
            before = measured()
            ex = arm["command"]["example"]
            joints = list(ex["joints"][:5]) + [int(round(before[5]))]
            msg = {"id": 0, "run_time": ex["run_time"], "angle": 0.0, "joints": joints}
            rb.publish(arm["command"]["name"], msg)
            cur, errs, lims, ok, t = settle({i: joints[i] for i in range(5)}, 8.0)
            problems = [f"{names[i]}: servo {cur[i]:.1f} deg vs {joints[i]} (error {errs[i]:.4f} rad > {lims[i]})"
                        for i in range(5) if errs[i] > lims[i]]
            if not ok:
                problems.append("did not settle at the goal")
            sim = ctx.joints()
            return {"wire": w["url"], "command": {"topic": arm["command"]["name"],
                                                  "type": arm["command"]["type"], "message": msg},
                    "feedback": "/CurrentAngle (measured servo angles, deg)",
                    "commanded_servo_deg": joints[:5],
                    "commanded": {names[i]: round(to_joint(i, joints[i]), 4) for i in range(5)},
                    "before_servo_deg": before,
                    "measured_servo_deg": cur[:5],
                    "measured": {names[i]: round(to_joint(i, cur[i]), 4) for i in range(5)},
                    "error": {names[i]: round(errs[i], 4) for i in range(5)},
                    "sim_readings_crosscheck": {n: round(sim[n], 4) for n in names[:5]},
                    "bound": {"figure": t_arm["figure"], "tolerance": t_arm["tolerance"],
                              "rule": "each arm joint (servo deg - 90) within the absolute tolerance; "
                                      "at rest = measured angles change < 0.5 deg over 0.5 s"},
                    "settle_time_s": t, "problems": problems}

        def do_grip():
            before = measured()
            ex = dict(grip["command"]["example"])
            if abs(before[5] - ex["angle"]) < 20:   # already there: open instead
                ex["angle"] = 30.0
            msg = {"id": ex["id"], "run_time": ex["run_time"], "angle": float(ex["angle"]), "joints": []}
            rb.publish(grip["command"]["name"], msg)
            cur, errs, lims, ok, t = settle({5: msg["angle"]}, 6.0)
            problems = []
            if errs[5] > lims[5]:
                problems.append(f"grip servo {cur[5]:.1f} deg vs {msg['angle']} (error {errs[5]:.4f} rad > {lims[5]})")
            if not ok:
                problems.append("gripper did not settle")
            return {"wire": w["url"], "command": {"topic": grip["command"]["name"],
                                                  "type": grip["command"]["type"], "message": msg},
                    "feedback": "/CurrentAngle angles[5] (measured, deg)",
                    "commanded_servo_deg": msg["angle"],
                    "commanded": {"grip_joint": round(to_joint(5, msg["angle"]), 4)},
                    "before_servo_deg": before[5], "measured_servo_deg": cur[5],
                    "measured": {"grip_joint": round(to_joint(5, cur[5]), 4)},
                    "error": round(errs[5], 4),
                    "sim_readings_crosscheck": {"grip_joint": round(ctx.joints()["grip_joint"], 4)},
                    "bound": {"figure": t_grip["figure"], "tolerance": t_grip["tolerance"],
                              "rule": "grip_joint ((deg-180)/150*pi/2) within the absolute tolerance; at rest"},
                    "settle_time_s": t, "problems": problems}

        out.append(run_motion(ctx, "arm", do_arm))
        out.append(run_motion(ctx, "gripper", do_grip))
    finally:
        rb.close()
    return out


# --------------------------------------------------------------------------- AiNex


def ainex_motions(ctx: Ctx) -> List[dict]:
    w = ctx.wire("ainex")
    rb = Rosbridge(w["url"])
    out = []
    t_head = tolerance("ainex", "head joint")
    t_walk = tolerance("ainex", "walk displacement")
    t_drift = tolerance("ainex", "heading drift")
    t_ag = tolerance("ainex", "action group")

    def servo_positions(ids) -> Dict[int, int]:
        r = rb.call("/ros_robot_controller/bus_servo/get_position", {"id": list(ids)}, timeout=10)
        res = {}
        for p in r.get("position", []):
            res[int(p["id"])] = int(p["position"])
        return res

    def pulse_to_rad(p: float) -> float:
        # recorded mapping: pulse = 500 + position_deg * 1000 / 240
        return math.radians((p - 500.0) * 240.0 / 1000.0)

    try:
        # ---------------- head pan / tilt
        for joint, servo in (("head_pan", 23), ("head_tilt", 24)):
            row = motion_row("ainex", "head", joint)

            def do_head(row=row, joint=joint, servo=servo):
                ex = row["command"]["example"]
                rb.advertise(row["command"]["name"], row["command"]["type"])
                time.sleep(0.8)
                before = servo_positions([servo])[servo]
                rb.publish(row["command"]["name"], {"position": ex["position"], "duration": ex["duration"]})
                lim = t_head["tolerance"]["absolute"]
                t0 = time.monotonic()
                prev = before
                ok = False
                while time.monotonic() - t0 < ex["duration"] + 4.0:
                    time.sleep(0.25)
                    cur = servo_positions([servo])[servo]
                    err = abs(pulse_to_rad(cur) - ex["position"])
                    if err <= lim and abs(cur - prev) <= 1 and time.monotonic() - t0 >= ex["duration"]:
                        ok = True
                        break
                    prev = cur
                sim = ctx.joints()[joint]
                problems = []
                if err > lim:
                    problems.append(f"{joint}: {pulse_to_rad(cur):.4f} rad vs {ex['position']} (> {lim})")
                if not ok:
                    problems.append(f"{joint}: did not settle")
                return {"wire": w["url"], "command": {"topic": row["command"]["name"],
                                                      "type": row["command"]["type"],
                                                      "message": {"position": ex["position"], "duration": ex["duration"]}},
                        "feedback": f"/ros_robot_controller/bus_servo/get_position servo {servo} (pulses)",
                        "commanded": {joint: ex["position"]},
                        "before_pulse": before, "measured_pulse": cur,
                        "measured": {joint: round(pulse_to_rad(cur), 4)}, "error": round(err, 4),
                        "sim_readings_crosscheck": {joint: round(sim, 4)},
                        "bound": {"figure": t_head["figure"], "tolerance": t_head["tolerance"],
                                  "rule": "servo position (pulse-500)*0.24 deg within the absolute tolerance "
                                          "after the move time, then still (<= 1 pulse over 0.25 s)"},
                        "problems": problems}
            out.append(run_motion(ctx, joint, do_head))
        # back to the centre (not judged: restores the initial head pose)
        for joint in ("head_pan", "head_tilt"):
            row = motion_row("ainex", "head", joint)
            rb.publish(row["command"]["name"], {"position": 0.0, "duration": 0.5})
        time.sleep(1.5)

        # ---------------- walk + recorded stop
        walk = motion_row("ainex", "walk")

        def do_walk():
            ex = walk["command"]["example"]
            states = []
            rb.subscribe("/walking/is_walking", lambda m: states.append((time.monotonic(), bool(m["data"]))))
            prm = rb.call("/walking/get_param", {})["parameters"]
            prm = dict(prm)
            amp = ex["set_param"]["x_move_amplitude"]
            prm["x_move_amplitude"] = amp
            rb.advertise(walk["command"]["name"], walk["command"]["type"])
            time.sleep(0.8)
            r0 = ctx.readings()["base"]
            rb.publish(walk["command"]["name"], prm)
            time.sleep(0.4)
            t_start = time.monotonic()
            rb.call(ex["start"]["service"], ex["start"]["request"])
            time.sleep(4.0)
            t_stop_call = time.monotonic()
            rb.call(walk["stop"]["name"], walk["stop"]["message"], timeout=30)
            t_stop_ret = time.monotonic()
            r_stop = ctx.readings()["base"]
            time.sleep(1.0)
            r1 = ctx.readings()["base"]
            time.sleep(1.0)
            r2 = ctx.readings()["base"]
            rb.unsubscribe("/walking/is_walking")
            yaw0 = yaw_of_wxyz(r0["quat"])
            dx, dy = r1["pos"][0] - r0["pos"][0], r1["pos"][1] - r0["pos"][1]
            fwd = dx * math.cos(yaw0) + dy * math.sin(yaw0)
            lat = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
            drift = wrap(yaw_of_wxyz(r1["quat"]) - yaw0)
            # walking time: from the first True to the last False on /walking/is_walking
            t_true = next((t for t, s in states if s), None)
            t_false = next((t for t, s in reversed(states) if not s), None)
            walked = (t_false - t_true) if (t_true and t_false and t_false > t_true) else (t_stop_ret - t_start)
            period = float(prm.get("period_time", 400.0)) / 1000.0
            steps = walked / period
            commanded = amp * steps
            b = bound_of(t_walk, commanded)
            settle = math.hypot(r2["pos"][0] - r1["pos"][0], r2["pos"][1] - r1["pos"][1])
            problems = []
            if not (fwd > 0):
                problems.append(f"walked {fwd:.3f} m: wrong sign")
            if abs(fwd - commanded) > b:
                problems.append(f"forward {fwd:.3f} m vs commanded {commanded:.3f} m (bound {b:.3f})")
            if abs(drift) > t_drift["tolerance"]["absolute"]:
                problems.append(f"heading drift {drift:.3f} rad (> {t_drift['tolerance']['absolute']})")
            if abs(r1["pos"][2] - r0["pos"][2]) > 0.03:
                problems.append(f"body height changed {r1['pos'][2] - r0['pos'][2]:.3f} m (fell?)")
            if not states or states[-1][1]:
                problems.append(f"/walking/is_walking did not end False: {[s for _, s in states]}")
            if settle > 0.01:
                problems.append(f"body still moving after the stop: {settle:.4f} m in 1 s")
            return {"wire": w["url"],
                    "command": {"sequence": "get_param -> publish /walking/set_param -> /walking/command start",
                                "topic": walk["command"]["name"], "type": walk["command"]["type"],
                                "set_param": prm, "start": ex["start"], "walk_s": 4.0},
                    "stop": {"service": walk["stop"]["name"], "request": walk["stop"]["message"],
                             "returned_after_s": round(t_stop_ret - t_stop_call, 3)},
                    "feedback": "/walking/is_walking (state) + sim-port base pose (no odometry on the wire)",
                    "is_walking": [[round(t - t_start, 3), s] for t, s in states],
                    "walking_time_s": round(walked, 3), "gait_periods": round(steps, 2),
                    "commanded": {"forward_m": round(commanded, 4),
                                  "derivation": f"x_move_amplitude {amp} m per step x {steps:.2f} steps "
                                                f"(walking time / period_time {period} s)"},
                    "measured": {"forward_m": round(fwd, 4), "left_m": round(lat, 4),
                                 "heading_drift_rad": round(drift, 4),
                                 "height_change_m": round(r1["pos"][2] - r0["pos"][2], 4)},
                    "rest": {"is_walking_last": states[-1][1] if states else None,
                             "travel_1_to_2s_after_stop_m": round(settle, 4),
                             "travel_during_stop_call_m": round(math.hypot(r_stop["pos"][0] - r0["pos"][0], r_stop["pos"][1] - r0["pos"][1]) - math.hypot(dx, dy), 4)},
                    "bound": {"figure": t_walk["figure"], "tolerance": t_walk["tolerance"],
                              "bound": round(b, 4), "drift": t_drift["tolerance"],
                              "rule": "forward > 0 and |forward - commanded| <= relative*commanded; |heading drift| <= "
                                      "the drift tolerance; is_walking ends False; still (< 0.01 m over 1 s) after the stop"},
                    "problems": problems}
        out.append(run_motion(ctx, "walk", do_walk))

        # ---------------- action group
        ag = motion_row("ainex", "action_group")

        def do_action():
            group = "wave"
            arms = ["r_sho_pitch", "r_sho_roll", "r_el_pitch", "r_el_yaw",
                    "l_sho_pitch", "l_sho_roll", "l_el_pitch", "l_el_yaw", "head_pan", "head_tilt"]
            j0 = ctx.joints()
            arms = [a for a in arms if a in j0]
            rb.advertise(ag["command"]["name"], ag["command"]["type"])
            time.sleep(0.8)
            rb.publish(ag["command"]["name"], {"data": group})
            peak = {a: 0.0 for a in arms}
            t0 = time.monotonic()
            quiet_since = None
            prev = j0
            while time.monotonic() - t0 < 20:
                time.sleep(0.1)
                j = ctx.joints()
                for a in arms:
                    peak[a] = max(peak[a], abs(j[a] - j0[a]))
                moving = any(abs(j[a] - prev[a]) > 0.002 for a in arms)
                prev = j
                if max(peak.values()) > 0.2 and not moving:
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since > 1.5:
                        break
                else:
                    quiet_since = None
            j1 = ctx.joints()
            back = {a: abs(j1[a] - j0[a]) for a in arms}
            lim = t_ag["tolerance"]["absolute"]
            problems = []
            if max(peak.values()) < 0.2:
                problems.append(f"the action group did not move the arms (peak {max(peak.values()):.3f} rad)")
            bad = {a: round(v, 4) for a, v in back.items() if v > lim}
            if bad:
                problems.append(f"not back at the init pose: {bad} (> {lim})")
            return {"wire": w["url"], "command": {"topic": ag["command"]["name"], "type": ag["command"]["type"],
                                                  "message": {"data": group}},
                    "note": "the vendor .d6a files are in no pinned source (robots_specs ainex boot.data_files); "
                            "'wave' is one of the groups the simulator ships (simulator/README.md)",
                    "feedback": "none on the wire (interface feedback: []); judged by sim-port joint readings",
                    "peak_excursion_rad": rounded(peak), "final_error_vs_init_rad": rounded(back),
                    "duration_s": round(time.monotonic() - t0, 2),
                    "bound": {"figure": t_ag["figure"], "tolerance": t_ag["tolerance"],
                              "rule": "an arm joint moves > 0.2 rad (the group plays), then every arm/head joint "
                                      "returns within the absolute tolerance of the init pose and is still for 1.5 s"},
                    "problems": problems}
        out.append(run_motion(ctx, "action_group", do_action))
    finally:
        rb.close()
    return out


def smoke(ctx: Ctx) -> List[dict]:
    rid = ctx.rid
    if rid == "myagv":
        return drive_motions(ctx, "myagv")
    if rid == "rosmaster_x3_plus":
        return drive_motions(ctx, "rosmaster_x3_plus") + rosmaster_arm_motions(ctx)
    if rid == "so101":
        return so101_motions(ctx)
    if rid == "mycobot280":
        return mycobot_motions(ctx)
    if rid == "ainex":
        return ainex_motions(ctx)
    raise KeyError(f"no smoke run defined for {rid}")


# =========================================================================== cameras


def topic_file(topic: str) -> str:
    return "cam__" + topic.strip("/").replace("/", "__") + ".png"


def capture_cameras(wires: List[dict], outdir: Path, say) -> List[dict]:
    res = []
    for w in wires:
        iface = interface(w["owner"])
        cams = [c for c in (iface.get("sensors") or {}).get("cameras") or [] if not c.get("optional")]
        if not cams:
            continue
        rb = Rosbridge(w["url"])
        try:
            for c in cams:
                topic = c["image_topic"]
                rec = {"wire": w["url"], "owner": w["owner"], "camera": c["id"], "topic": topic,
                       "recorded": {"encoding": c["encoding"], "width": c["width"],
                                    "height": c["height"], "frame_id": c["frame_id"]}}
                box = []
                rb.subscribe(topic, lambda m: box.append((time.monotonic(), m)), throttle_rate=300)
                t0 = time.monotonic()
                while len(box) < 2 and time.monotonic() - t0 < 20:
                    time.sleep(0.05)
                rb.unsubscribe(topic)
                if not box:
                    rec.update({"pass": False, "problems": ["no image on the wire within 20 s"]})
                    res.append(rec)
                    say(f"    camera {topic}: FAIL (no image)")
                    continue
                msg = box[-1][1]
                try:
                    d = decode_image(msg)
                except Exception as exc:
                    rec.update({"pass": False, "problems": [f"decode failed: {exc!r}"]})
                    res.append(rec)
                    continue
                fn = topic_file(topic)
                (outdir / fn).write_bytes(png_bytes(d["rgb"]))
                info = d["info"]
                problems = []
                for k in ("encoding", "width", "height", "frame_id"):
                    if info.get(k) != c[k]:
                        problems.append(f"{k} {info.get(k)!r} != recorded {c[k]!r}")
                if len(box) < 2:
                    problems.append("only one frame within 20 s")
                if info["std"] < 2.0:
                    problems.append(f"picture is flat (std {info['std']})")
                rec.update({"file": fn, "frames_seen": len(box), "message": info,
                            "pass": not problems, "problems": problems})
                say(f"    camera {topic}: {'PASS' if not problems else 'FAIL ' + str(problems)} -> {fn}")
                res.append(rec)
        finally:
            rb.close()
    return res


# =========================================================================== fleet


def fleet_checks(rid: str, wires: List[dict], outdir: Path, say) -> List[dict]:
    res = []
    for pid in console_profiles():
        if pid != rid:
            continue
        owner = pid
        try:
            w = next(x for x in wires if x["owner"] == owner)
        except StopIteration:
            res.append({"profile": pid, "pass": False, "problems": [f"no wire for {owner}"]})
            continue
        cmd = [str(CONSOLE / "python.sh"), "-m", "robot_console.fleet", "--url", w["url"],
               "--expect", pid]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=300)
            text, code = p.stdout, p.returncode
        except subprocess.TimeoutExpired as exc:
            text, code = (exc.stdout or "") + "\n[timed out after 300 s]", None
        fn = f"fleet_{pid}.txt"
        rel = "robot_console/python.sh " + " ".join(cmd[1:])
        (outdir / fn).write_text(f"$ {rel}\n# started {dt.datetime.fromtimestamp(t0).isoformat(timespec='seconds')}"
                                 f", took {time.time() - t0:.1f} s\n{text}\n# exit code: {code}\n")
        ok = code == 0
        res.append({"profile": pid, "wire": w["url"], "command": rel, "exit_code": code,
                    "file": fn, "pass": ok, "summary": text.strip().splitlines()[-1] if text.strip() else ""})
        say(f"    fleet --expect {pid} on {w['url']}: exit {code}")
    return res


# =========================================================================== a case


def iso(t: Optional[float] = None) -> str:
    return dt.datetime.fromtimestamp(t or time.time()).astimezone().isoformat(timespec="seconds")


class RtfMonitor:
    """Polls the simulation's real-time factor (`hello.rtf`, the RTF over its last 10 s
    window) every second on its own control-port connection. A change of value marks a
    window that ended then; `low_windows(a, b)` lists the windows below 0.90 (the
    simulator's warning threshold) that overlap the wall-time interval [a, b]."""

    WINDOW_S = 10.0

    def __init__(self, port: int):
        self.sp = SimPort(port)
        self.samples: List[tuple] = []
        self.updates: List[tuple] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        last = None
        while not self._stop.is_set():
            try:
                rtf = float(self.sp.call("hello").get("rtf"))
            except Exception:
                return
            now = time.time()
            self.samples.append((now, rtf))
            if last is not None and rtf != last:
                self.updates.append((now, rtf))
            last = rtf
            self._stop.wait(1.0)

    def low_windows(self, a: float, b: float, threshold: float = 0.90) -> List[dict]:
        return [{"window_end": iso(t), "rtf": r} for t, r in self.updates
                if r < threshold and t > a and t - self.WINDOW_S - 1.0 < b]

    def close(self):
        self._stop.set()
        self._t.join(3)
        self.sp.close()


def run_case(engine: str, robot: dict, out_root: Path, args, say) -> dict:
    rid = robot["id"]
    mobile = robot["kind"] != "arm"
    placement = "floor" if mobile else "worktop"
    outdir = out_root / engine / rid
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    case = {"robot": rid, "name": robot["name"], "kind": robot["kind"], "engine": engine,
            "placement": placement, "started": iso(), "items": {}, "notes": []}
    say(f"== case {engine}/{rid} ({placement})")
    case["host_load_before"] = wait_low_load(args.max_load, args.max_wait, say)
    sim = spawn = None
    sp = rtfmon = None
    t_collect = None
    sim_log = outdir / "simulation.log"
    try:
        if not port_free(args.sim_port) or not port_free(args.port):
            raise RuntimeError(f"port {args.sim_port} or {args.port} is taken")
        sim = Proc([str(SIM / "kitchen.sh"), "start", "--engine", engine,
                    "--sim-port", str(args.sim_port)], sim_log)
        sim_line = sim.wait_for("simulation ready", 600)
        case["simulation_ready_line"] = sim_line
        case["simulation_command"] = f"simulator/kitchen.sh start --engine {engine} --sim-port {args.sim_port}"
        sp = SimPort(args.sim_port)
        rtfmon = RtfMonitor(args.sim_port)
        hello = sp.call("hello")
        case["scene"] = hello["scene"]
        case["engine_reported"] = hello["engine"]
        if hello["engine"] != engine:
            raise RuntimeError(f"simulation reports engine {hello['engine']}")
        spawn_cmd = [str(SIM / "spawn.sh"), rid, "--placement", placement,
                     "--sim-port", str(args.sim_port), "--port", str(args.port)]
        case["spawn_command"] = "simulator/spawn.sh " + " ".join(spawn_cmd[1:])
        spawn = Proc(spawn_cmd, outdir / "spawn.log")
        case["readiness_line"] = spawn.wait_for("spawn ready:", 900)
        case["ready_at"] = iso()
        wires = parse_ready(case["readiness_line"])
        case["wires"] = wires
        case["ports"] = {w["role"]: w["port"] for w in wires} | {"sim_port": args.sim_port}
        # let the spawn's recompile transient pass out of the RTF window before collecting
        time.sleep(args.settle)
        t_collect = [time.time(), None]
        row = next(r for r in sp.call("robots")["robots"] if r["id"] == rid)
        case["spawned"] = {k: row.get(k) for k in ("state", "placement", "xyz", "yaw", "ports",
                                                   "staged", "cleared")}
        case["rtf_at_start"] = sp.call("hello").get("rtf")
        if row.get("staged"):
            say(f"    worktop objects: {', '.join(row['staged'])}; cleared: "
                f"{', '.join(row.get('cleared') or []) or 'none'}")

        # scene picture (an arm's: with the objects staged around it in view)
        view = choose_view(sp, rid, mobile, extra=[o["body"] for o in (row.get("staged") or {}).values()])
        case["view"] = view
        sc = render(sp, view, outdir / "scene.png")
        case["items"]["scene"] = {"file": "scene.png", **sc,
                                  "pass": sc["width"] == 1280 and sc["height"] == 720 and sc["std"] > 5}
        say(f"    scene.png ({view.get('_context_m')} m context)")

        # cameras on the wire
        case["items"]["cameras"] = capture_cameras(wires, outdir, say)

        # the console's check
        case["host_load_fleet"] = host_load()
        case["items"]["fleet"] = fleet_checks(rid, wires, outdir, say)

        # the motion smoke run
        case["host_load_smoke"] = host_load()
        ctx = Ctx(rid, sp, wires, outdir, view, say)
        motions = smoke(ctx)
        sm = {"robot": rid, "engine": engine, "scene": case["scene"],
              "bounds_note": "Bounds are the tolerances recorded in robots_specs/<id>/ros*.yml; a recorded "
                             "{relative, absolute} pair bounds |measured - commanded| by "
                             "max(relative*|commanded|, absolute). Commands are each robot's recorded "
                             "command (its interface file's motions[].command and example) through the "
                             "vendor wire; feedback is the recorded feedback, or the simulation's "
                             "control-port readings where the interface records none.",
              "motions": motions}
        (outdir / "smoke.json").write_text(json.dumps(sm, indent=1, default=str))
        case["items"]["smoke"] = {"file": "smoke.json", "pass": all(m["pass"] for m in motions),
                                  "motions": {m["name"]: m["pass"] for m in motions}}
        case["rtf_at_end"] = sp.call("hello").get("rtf")
        t_collect[1] = time.time()
        # the RTF window covering the end of the collection, before the robot is removed
        time.sleep(RtfMonitor.WINDOW_S + 1.0)
    except Exception as exc:
        case["error"] = f"{exc!r}"
        case["traceback"] = traceback.format_exc()[-4000:]
        say(f"    ERROR {exc!r}")
    finally:
        if rtfmon is not None:
            if t_collect and t_collect[1] is None:
                t_collect[1] = time.time()
            if t_collect:
                case["collection"] = {"from": iso(t_collect[0]), "to": iso(t_collect[1])}
                case["rtf_low_windows_during_collection"] = rtfmon.low_windows(*t_collect)
            case["rtf_samples"] = [[round(t - rtfmon.samples[0][0], 1), r] for t, r in rtfmon.samples]
            rtfmon.close()
        if sp is not None:
            sp.close()
        if spawn is not None:
            case["spawn_exit_code"] = spawn.stop()
        if sim is not None:
            case["simulation_exit_code"] = sim.stop()
            wait_port_free(args.sim_port)
        case["containers_left"] = docker_left(args.sim_port)
    log = sim_log.read_text(errors="replace") if sim_log.exists() else ""
    case["rtf_warnings"] = [ln for ln in log.splitlines() if "real-time factor" in ln]
    case["host_load_after"] = host_load()
    case["finished"] = iso()
    it = case["items"]
    checks = {
        "scene": bool(it.get("scene", {}).get("pass")),
        "cameras": all(c.get("pass") for c in it.get("cameras", [])) and
                   len(it.get("cameras", [])) == sum(
                       len([c for c in (interface(w["owner"]).get("sensors") or {}).get("cameras") or []
                            if not c.get("optional")]) for w in case.get("wires", [])),
        "fleet": bool(it.get("fleet")) and all(f["pass"] for f in it.get("fleet", [])),
        "smoke": bool(it.get("smoke", {}).get("pass")),
        "spawn_clean_exit": case.get("spawn_exit_code") == 0 and not case["containers_left"],
    }
    if not mobile:
        # an arm on the worktop brings the six worktop objects (simulator spec §2.3)
        checks["worktop_objects"] = sorted((case.get("spawned") or {}).get("staged") or {}) == \
            sorted(["apple", "banana", "bowl", "lemon", "mug", "plate"])
    case["checks"] = checks
    case["pass"] = "error" not in case and all(checks.values())
    # evidence collected while the simulation ran below real time is retried (main loop)
    case["rtf_clean"] = not case.get("rtf_low_windows_during_collection")
    (outdir / "case.json").write_text(json.dumps(case, indent=1, default=str))
    say(f"== {engine}/{rid}: {'PASS' if case['pass'] else 'FAIL'} {checks}"
        + ("" if case["rtf_clean"] else f" (RTF below 0.90 during collection: "
                                         f"{case['rtf_low_windows_during_collection']})"))
    return case


# =========================================================================== index


def required_cameras(rid: str) -> List[str]:
    out = []
    for c in (interface(rid).get("sensors") or {}).get("cameras") or []:
        if not c.get("optional"):
            out.append(topic_file(c["image_topic"]))
    return out


def write_index(out_root: Path) -> dict:
    robots = {r["id"]: r for r in registry()}
    cases = []
    for engine in ENGINES:
        for rid in robots:
            p = out_root / engine / rid / "case.json"
            if not p.exists():
                cases.append({"engine": engine, "robot": rid, "pass": False, "missing": True})
                continue
            c = json.loads(p.read_text())
            d = f"{engine}/{rid}"
            hist = []
            hdir = out_root / engine / rid / "history"
            if hdir.exists():
                for h in sorted(hdir.glob("*/case.json")):
                    hc = json.loads(h.read_text())
                    hist.append({"dir": str(h.parent.relative_to(out_root)), "started": hc.get("started"),
                                 "pass": hc.get("pass"), "checks": hc.get("checks"),
                                 "rtf_clean": hc.get("rtf_clean"),
                                 "rtf_low_windows": hc.get("rtf_low_windows_during_collection"),
                                 "error": hc.get("error"), "host_load_before": hc.get("host_load_before"),
                                 "rtf_warnings": len(hc.get("rtf_warnings") or [])})
            cams = [x.get("file") for x in c["items"].get("cameras", []) if x.get("file")]
            motion_pics = sorted(x.name for x in (out_root / d).glob("motion_*.png"))
            cases.append({
                "engine": engine, "robot": rid, "name": c.get("name"), "kind": c.get("kind"),
                "scene": c.get("scene"), "placement": c.get("placement"), "pass": c.get("pass"),
                "checks": c.get("checks"), "error": c.get("error"),
                "dir": d, "scene_png": f"{d}/scene.png" if (out_root / d / "scene.png").exists() else None,
                "camera_pngs": [f"{d}/{x}" for x in cams],
                "required_camera_pngs": [f"{d}/{x}" for x in required_cameras(rid)],
                "motion_pngs": [f"{d}/{x}" for x in motion_pics],
                "smoke": f"{d}/smoke.json", "motions": c["items"].get("smoke", {}).get("motions"),
                "fleet": [{"profile": f["profile"], "exit_code": f.get("exit_code"), "pass": f["pass"],
                           "file": f"{d}/{f['file']}" if f.get("file") else None}
                          for f in c["items"].get("fleet", [])],
                "readiness_line": c.get("readiness_line"), "ports": c.get("ports"),
                "staged": sorted(((c.get("spawned") or {}).get("staged") or {})),
                "cleared": (c.get("spawned") or {}).get("cleared") or [],
                "started": c.get("started"), "finished": c.get("finished"),
                "host_load_before": c.get("host_load_before"),
                "rtf_warnings": len(c.get("rtf_warnings") or []),
                "rtf_clean": c.get("rtf_clean"),
                "rtf_low_windows": c.get("rtf_low_windows_during_collection"),
                "collection": c.get("collection"),
                "earlier_attempts": hist,
            })
    idx = {"generated": iso(), "generator": "tests/evidence.py",
           "robots": list(robots), "engines": ENGINES, "cases": cases,
           "all_pass": all(c.get("pass") for c in cases)}
    (out_root / "index.json").write_text(json.dumps(idx, indent=1, default=str))

    L = ["# Evidence index", "",
         f"Generated {idx['generated']} by `tests/evidence.py` (workspace spec §3). "
         "One case per robot of `robots_specs/high_level_spec.md` per engine, on the engine's "
         "default scene (`simulator/kitchen.sh start --engine <engine>`), spawned with "
         "`simulator/spawn.sh` (mobile robots on the floor, arms on the worktop).", "",
         f"**Overall: {'PASS' if idx['all_pass'] else 'FAIL'}** "
         f"({sum(1 for c in cases if c.get('pass'))}/{len(cases)} cases pass)", "",
         "| Engine | Robot | Scene | Placement | Result | Picture | Cameras | Smoke run | Console check | Attempts |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cases:
        if c.get("missing"):
            L.append(f"| {c['engine']} | `{c['robot']}` | | | **MISSING** | | | | | |")
            continue
        sc = (f"<a href=\"{c['scene_png']}\"><img src=\"{c['scene_png']}\" width=\"160\"></a> "
              f"[scene.png]({c['scene_png']})") if c["scene_png"] else "missing"
        cams = "<br>".join(f"[{Path(x).name}]({x})" for x in c["camera_pngs"]) or "none recorded"
        mot = c.get("motions") or {}
        sm = f"[smoke.json]({c['smoke']}): " + ", ".join(f"{k} {'pass' if v else '**FAIL**'}" for k, v in mot.items())
        fl = "<br>".join(f"[{f['profile']}]({f['file']}) exit {f['exit_code']}" for f in c["fleet"]) or "none"
        att = 1 + len(c["earlier_attempts"])
        res = "PASS" if c["pass"] else "**FAIL**"
        L.append(f"| {c['engine']} | `{c['robot']}` | {c['scene']} | {c['placement']} | {res} | {sc} | "
                 f"{cams} | {sm} | {fl} | {att} |")
    L += ["", "## How a case is judged", "",
          "* **Scene picture**: an offscreen 1280x720 render through the simulation's control port, "
          "from a viewpoint with a clear line of sight to the robot (checked with a depth render).",
          "* **Camera pictures**: one frame of every camera not marked `optional` in the robot's "
          "interface file (`robots_specs/<id>/ros*.yml`), taken off the vendor wire through rosbridge "
          "and decoded from its `sensor_msgs/Image` (depth colourised: near red, far blue, no return black); "
          "its encoding, size and frame id must be the recorded ones.",
          "* **Smoke run** (`smoke.json`): each motion uses the robot's recorded command (interface file "
          "`motions[].command` and its example) on the vendor wire; a drive or walk is followed by the "
          "recorded stop. A motion passes when the recorded feedback (or, where the interface records "
          "none, the control port's joint/pose readings) shows the commanded displacement within the "
          "tolerance recorded in the interface file -- a `{relative, absolute}` pair bounds "
          "|measured - commanded| by max(relative x |commanded|, absolute) -- and the robot is at rest "
          "afterwards. Each motion in `smoke.json` names its bound and rule. Base motions: 0.25 m "
          "forward/back/left/right at 0.2 m/s and a 1 rad turn at 0.5 rad/s, within the placement's "
          "guaranteed travel.",
          "* **Console check**: `robot_console/python.sh -m robot_console.fleet --url <wire> --expect <profile>` "
          "for every console profile naming the robot; exit 0 is the console's own verdict.",
          "* A case also requires `spawn.sh` to exit 0 on SIGINT with no container left.",
          "* **Worktop objects** (an arm on the worktop): the spawn staged the six worktop objects "
          "(apple, plate, bowl, mug, banana, lemon) around the arm (simulator spec §2.3); the scene "
          "picture frames them with the arm, and the case records them and any scene objects "
          "cleared for them.",
          "", "## Cases", ""]
    for c in cases:
        if c.get("missing"):
            continue
        L.append(f"### {c['engine']} / {c['robot']} — {'PASS' if c['pass'] else 'FAIL'}")
        L.append("")
        L.append(f"* Readiness: `{c['readiness_line']}`")
        if c.get("staged"):
            L.append(f"* Worktop objects staged: {', '.join(c['staged'])}; scene objects cleared: "
                     + (", ".join(f"`{n}`" for n in c["cleared"]) or "none"))
        low = c.get("rtf_low_windows") or []
        L.append(f"* Started {c['started']}, finished {c['finished']}; host load (1 min) before "
                 f"{(c.get('host_load_before') or {}).get('load1')}; real-time factor during the "
                 f"collection: " + ("never below 0.90" if c.get("rtf_clean") else
                                    "below 0.90 in " + ", ".join(f"{w['rtf']} (window ending {w['window_end']})" for w in low))
                 + f"; RTF warnings in the whole simulation log {c['rtf_warnings']}")
        L.append(f"* Checks: {c['checks']}" + (f"; error: `{c['error']}`" if c.get("error") else ""))
        L.append(f"* Details: [case.json]({c['dir']}/case.json), [smoke.json]({c['smoke']}), "
                 f"[spawn.log]({c['dir']}/spawn.log), [simulation.log]({c['dir']}/simulation.log)")
        for h in c["earlier_attempts"]:
            L.append(f"* Earlier attempt {h['started']}: {'PASS' if h['pass'] else 'FAIL'} "
                     f"(load {(h.get('host_load_before') or {}).get('load1')}, RTF warnings {h['rtf_warnings']}, "
                     f"RTF below 0.90 during collection: {h.get('rtf_low_windows') or 'no'}, "
                     f"checks {h['checks']}{', error ' + str(h['error']) if h.get('error') else ''}) — "
                     f"[{h['dir']}]({h['dir']}/case.json)")
        L.append("")
        pics = [c["scene_png"]] + c["camera_pngs"]
        L.append(" ".join(f"<img src=\"{p}\" width=\"240\">" for p in pics if p))
        L.append("")
        if c["motion_pngs"]:
            L.append("Motions (before / after):")
            L.append("")
            mp = c["motion_pngs"]
            names = sorted({Path(p).name[len("motion_"):].rsplit("_", 1)[0] for p in mp})
            for n in names:
                b = f"{c['dir']}/motion_{n}_before.png"
                a = f"{c['dir']}/motion_{n}_after.png"
                L.append(f"* `{n}`: <img src=\"{b}\" width=\"200\"> <img src=\"{a}\" width=\"200\">")
            L.append("")
    (out_root / "index.md").write_text("\n".join(L) + "\n")
    return idx


# =========================================================================== main


def run_case_keeping_history(engine: str, robot: dict, out_root: Path, args, say) -> dict:
    """run_case, keeping every earlier failing attempt of the case (and its own history)
    under `<case>/history/<started>/`: the index lists them beside the latest result."""
    case_dir = out_root / engine / robot["id"]
    staging = out_root / f".staging-{engine}-{robot['id']}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    if (case_dir / "history").exists():
        shutil.move(str(case_dir / "history"), str(staging / "history"))
    if (case_dir / "case.json").exists():
        old = json.loads((case_dir / "case.json").read_text())
        if not old.get("pass") or not old.get("rtf_clean", True):
            (staging / "history").mkdir(exist_ok=True)
            stamp = re.sub(r"[^0-9T+-]", "", old.get("started", "earlier"))
            shutil.move(str(case_dir), str(staging / "history" / stamp))
    case = run_case(engine, robot, out_root, args, say)
    if (staging / "history").exists():
        shutil.move(str(staging / "history"), str(case_dir / "history"))
    shutil.rmtree(staging, ignore_errors=True)
    return case


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the workspace evidence (workspace spec §3).")
    ap.add_argument("--engine", action="append", choices=ENGINES, help="engine(s); default both")
    ap.add_argument("--robot", action="append", help="robot id(s); default every recorded robot")
    ap.add_argument("--out", default=str(REPO / "evidences"), help="output directory")
    ap.add_argument("--sim-port", type=int, default=9080)
    ap.add_argument("--port", type=int, default=9090, help="wire port (an arm wire takes port+1)")
    ap.add_argument("--max-load", type=float, default=float(os.cpu_count() or 8) * 0.5,
                    help="wait before a case until the 1-min load average is at most this")
    ap.add_argument("--max-wait", type=float, default=1800.0, help="longest wait for low load, s")
    ap.add_argument("--retries", type=int, default=1,
                    help="re-run a failed case this many times (after waiting for low load)")
    ap.add_argument("--settle", type=float, default=12.0,
                    help="wait after the readiness line before collecting, s")
    ap.add_argument("--index-only", action="store_true", help="only rebuild the index")
    args = ap.parse_args(argv)
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(out_root / "evidence_run.log", "a")

    def say(msg: str):
        line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if args.index_only:
        idx = write_index(out_root)
        say(f"index: {sum(1 for c in idx['cases'] if c.get('pass'))}/{len(idx['cases'])} pass")
        return 0 if idx["all_pass"] else 1
    robots = registry()
    ids = [r["id"] for r in robots]
    for r in args.robot or []:
        if r not in ids:
            ap.error(f"unknown robot id {r!r}; recorded: {', '.join(ids)}")
    sel = [r for r in robots if not args.robot or r["id"] in args.robot]
    for engine in args.engine or ENGINES:
        for robot in sel:
            for attempt in range(1 + args.retries):
                case = run_case_keeping_history(engine, robot, out_root, args, say)
                if case["pass"] and case["rtf_clean"]:
                    break
                if attempt < args.retries:
                    say(f"  {engine}/{robot['id']} {'failed' if not case['pass'] else 'ran below real time'}; "
                        "re-running once the load is low")
                    time.sleep(30)
            write_index(out_root)
    idx = write_index(out_root)
    say(f"index: {sum(1 for c in idx['cases'] if c.get('pass'))}/{len(idx['cases'])} pass -> "
        f"{out_root / 'index.md'}")
    return 0 if idx["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
