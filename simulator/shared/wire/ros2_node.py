#!/usr/bin/env python3
"""Recorded ROS 2 nodes, served by the simulator: `ros2_node.py /<node> [/<node> ...]`.

The ROS 2 counterpart of `ros1_node.py`: each named node gets exactly the non-optional
topics, services and parameters the interface file records for it (names, types, frames,
QoS, periodic rates), and the robot wire module's behaviour for that node
(`robots/<id>.py`, `BEHAVIOURS[node]`) supplies what it does. All nodes of the process
share one executor.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402

import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node as RclNode  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,  # noqa: E402
                       ReliabilityPolicy)
from rosidl_runtime_py.utilities import get_message, get_service  # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from rcl_interfaces.msg import ParameterDescriptor  # noqa: E402

_link = None
_link_lock = threading.Lock()


def shared_link() -> common.SimLink:
    global _link
    with _link_lock:
        if _link is None:
            _link = common.SimLink()
            common.exit_when_lost(_link, rclpy.try_shutdown)
        return _link


def qos_of(row: dict) -> QoSProfile:
    q = row.get("qos") or {}
    depth = int(q.get("depth", 10) or 10)
    prof = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth)
    if str(q.get("reliability", "reliable")).lower() == "best_effort":
        prof.reliability = ReliabilityPolicy.BEST_EFFORT
    if str(q.get("durability", "volatile")).lower() == "transient_local" or row.get("latched"):
        prof.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return prof


def stamp_msg(t: float) -> TimeMsg:
    sec = int(t)
    return TimeMsg(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1000000000)


def _param_obj(name, value):
    if isinstance(value, list) and not value:
        return Parameter(name, Parameter.Type.STRING_ARRAY, [])
    if isinstance(value, list) and all(isinstance(v, bool) for v in value):
        return Parameter(name, Parameter.Type.BOOL_ARRAY, value)
    if isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                       for v in value) and any(isinstance(v, float) for v in value):
        return Parameter(name, Parameter.Type.DOUBLE_ARRAY, [float(v) for v in value])
    return Parameter(name, value=value)


class Node:
    dialect = "ros2"

    def __init__(self, name: str, module):
        self.name = name
        self.robot = common.owner()
        self.iface = common.interface(self.robot)
        ns, _, base = name.rpartition("/")
        self.rcl = RclNode(base, namespace=ns or "/")
        overrides = getattr(module, "param_values", lambda n: {})(name)
        for row in common.params_of(self.iface, name):
            value = overrides.get(row["name"], common.param_value(self.robot, row))
            p = _param_obj(row["name"], value)
            # a parameter the record notes as read-only (rclpy's QoS overrides, rviz2's
            # tf_buffer_cache_time_ms) is declared so: setting it is refused, as on the robot
            ro = str(row.get("notes") or "").lower().startswith("read-only")
            try:
                self.rcl.declare_parameter(p.name, p.value, ParameterDescriptor(read_only=ro))
            except Exception:
                self.rcl.declare_parameter(p.name, p.type_)
                self.rcl.set_parameters([p])
        self.pub, self.rows, self.types = {}, {}, {}
        for t in common.topics_of(self.iface, name, "out"):
            cls = get_message(t["type"])
            self.types[t["name"]], self.rows[t["name"]] = cls, t
            self.pub[t["name"]] = self.rcl.create_publisher(cls, t["name"], qos_of(t))
        behaviours = getattr(module, "BEHAVIOURS", {})
        self.behaviour = behaviours[name](self) if name in behaviours else None
        handled = set(getattr(self.behaviour, "publishes", ()))
        for t in common.topics_of(self.iface, name, "in"):
            cls = get_message(t["type"])
            self.types[t["name"]] = cls
            self.rcl.create_subscription(cls, t["name"], self._cb(t["name"]), qos_of(t))
        for s in common.services_of(self.iface, name):
            cls = get_service(s["type"])
            self.rcl.create_service(cls, s["name"], self._handler(s["name"], cls))
        for tname, row in self.rows.items():
            rate = common.rate_of(row)
            if rate and tname not in handled:
                self.rcl.create_timer(1.0 / rate, self._default_pub(tname))
            elif not rate and row.get("latched") and tname not in handled:
                self.publish(tname, self.new(tname))

    @property
    def link(self):
        return shared_link()

    def new(self, topic: str, stamp: float = None):
        msg = self.types[topic]()
        self.fill_header(msg, topic, stamp)
        return msg

    def fill_header(self, msg, topic, stamp=None):
        if hasattr(msg, "header"):
            msg.header.stamp = stamp_msg(stamp if stamp is not None else time.time())
            fid = (self.rows.get(topic) or {}).get("frame_id")
            if fid is not None:
                msg.header.frame_id = fid

    @staticmethod
    def time(t: float):
        return stamp_msg(t)

    def publish(self, topic, msg):
        self.pub[topic].publish(msg)

    def _default_pub(self, topic):
        return lambda: self.publish(topic, self.new(topic))

    def _cb(self, topic):
        def cb(msg):
            if self.behaviour is not None and hasattr(self.behaviour, "on_message"):
                self.behaviour.on_message(topic, msg)
        return cb

    def _handler(self, name, cls):
        def handle(req, res):
            if self.behaviour is not None and hasattr(self.behaviour, "on_service"):
                out = self.behaviour.on_service(name, req, res)
                if out is not None:
                    return out
            if hasattr(res, "success"):
                res.success = True
            return res
        return handle


def main():
    rclpy.init()
    robot = common.owner()
    module = importlib.import_module(f"robots.{robot.id}")
    nodes = [Node(n, module) for n in sys.argv[1:]]
    for n in nodes:
        if n.behaviour is not None and hasattr(n.behaviour, "start"):
            n.behaviour.start()
    ex = MultiThreadedExecutor(num_threads=4)
    for n in nodes:
        ex.add_node(n.rcl)
    try:
        ex.spin()
    except Exception:
        pass


if __name__ == "__main__":
    main()
