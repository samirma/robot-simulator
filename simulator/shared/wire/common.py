"""Shared by every wire process inside a wire container (Python 3.8+, ROS 1 or ROS 2).

* the robot's interface file (ros.yml / ros2.yml), read through the read-only mount, with
  the rows the simulator serves (everything not marked `optional`);
* `SimLink`, a wire process's connection to the simulation's control port: it attaches
  with the spawn's token, sets actuator targets and receives the periodic state, camera
  and lidar streams it subscribes to. The simulation closes it when the robot is removed,
  and the process then exits.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
for p in (str(SHARED), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import protocol  # noqa: E402
import registry  # noqa: E402

import yaml  # noqa: E402  (python3-yaml in every wire image)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def owner() -> registry.Robot:
    return registry.get(env("RSIM_ROBOT"))


_iface_cache = {}


def interface(robot: registry.Robot = None) -> dict:
    robot = robot or owner()
    if robot.id not in _iface_cache:
        with open(robot.path(robot.ros_file), encoding="utf-8") as fh:
            _iface_cache[robot.id] = yaml.safe_load(fh)
    return _iface_cache[robot.id]


def served(rows):
    """The rows of a section the simulator serves: all but those marked optional."""
    return [r for r in (rows or []) if not r.get("optional")]


def topics_of(iface: dict, node: str, direction: str = None):
    out = []
    for t in served(iface.get("topics")):
        if node in (t.get("nodes") or []) and (direction is None or t.get("direction") == direction):
            out.append(t)
    return out


def services_of(iface: dict, node: str):
    return [s for s in served(iface.get("services")) if s.get("node") == node]


def actions_of(iface: dict, node: str):
    return [s for s in served(iface.get("actions")) if s.get("node") == node]


def params_of(iface: dict, node: str = None):
    rows = served(iface.get("parameters"))
    return [p for p in rows if node is None or p.get("node") == node]


def topic_row(iface: dict, name: str):
    for t in iface.get("topics") or []:
        if t.get("name") == name:
            return t
    return None


def param_value(robot: registry.Robot, row: dict):
    """A parameter's literal boot value; `{file: X}` is the text of that file in the
    robot's folder. `{generated: ...}` values are supplied by the robot's wire module."""
    v = row.get("value")
    if isinstance(v, dict) and "file" in v and len(v) <= 2:
        return (robot.folder_path / v["file"]).read_text(encoding="utf-8")
    return v


def rate_of(row: dict):
    r = row.get("rate")
    return float(r) if isinstance(r, (int, float)) else None


def rpy_to_quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)   # x, y, z, w


def yaw_of(q_wxyz):
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class SimLink:
    """One wire process's link to the simulation."""

    def __init__(self, role: str = None, on_lost=None):
        self.role = role or env("RSIM_ROLE", "main")
        self.lost = threading.Event()
        self._on_lost = on_lost
        self._subs = {}
        self._lock = threading.Lock()
        host, port = env("RSIM_SIM_HOST", "host.docker.internal"), int(env("RSIM_SIM_PORT", "9080"))
        last = None
        for _ in range(40):
            try:
                self.client = protocol.Client(host, port, timeout=5.0, on_event=self._event,
                                              on_close=self._closed)
                break
            except OSError as exc:
                last = exc
                time.sleep(0.25)
        else:
            raise SystemExit(f"cannot reach the simulation at {host}:{port}: {last}")
        res = self.client.call("wire", token=env("RSIM_TOKEN"), role=self.role)
        self.robot_id = res["robot"]
        self.describe = res["describe"]

    def _event(self, header, payload):
        ev = header.get("event")
        if ev == "sample":
            cb = self._subs.get(header.get("sub"))
            if cb is not None:
                try:
                    cb(header, payload)
                except Exception as exc:  # a bad sample never kills the link
                    print(f"wire: sample handler failed: {exc!r}", file=sys.stderr, flush=True)
        elif ev in ("removed", "shutdown"):
            self._closed()

    def _closed(self):
        if self.lost.is_set():
            return
        self.lost.set()
        if self._on_lost is not None:
            self._on_lost()

    def subscribe(self, stream: str, rate: float, callback, **params) -> int:
        res = self.client.call("subscribe", stream=stream, rate=rate, **params)
        self._subs[res["sub"]] = callback
        return res["sub"]

    def ctrl(self, values: dict) -> None:
        if values and not self.lost.is_set():
            try:
                self.client.notify("ctrl", values={k: float(v) for k, v in values.items()})
            except OSError:
                self._closed()

    def actuators(self):
        return {a["name"]: a for a in self.describe["actuators"]}

    def joints(self):
        return {j["name"]: j for j in self.describe["joints"]}


def exit_when_lost(link: SimLink, cleanup=None):
    """Watch the link: when the simulation removes the robot or ends, exit this process."""

    def watch():
        link.lost.wait()
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass
        os._exit(0)

    threading.Thread(target=watch, daemon=True, name="simlink-watch").start()
