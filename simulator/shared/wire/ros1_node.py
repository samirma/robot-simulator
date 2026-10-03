#!/usr/bin/env python3
"""One recorded ROS 1 node, served by the simulator: `ros1_node.py /<node name>`.

The node's endpoints come from the interface file: every non-optional topic it publishes
or subscribes, every service it offers, with the recorded names, types, frames and rates,
and the topics the record lists it among the `internal_publishers` / `internal_subscribers`
of (endpoints of another node of the boot it uses), so discovery shows them as on the robot.
What the node *does* is the robot's wire module's behaviour for that node
(`robots/<id>.py`, `BEHAVIOURS[node]`): the simulated driver of the real hardware, fed by
and commanding the simulation. A periodic topic the behaviour does not publish itself is
published at its recorded rate with its recorded frame; a service it does not answer
returns its default (successful) response.
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

# rospy fixes its namespace when it is imported: a node in a namespace (/camera/camera)
# gets it through ROS_NAMESPACE, set before the import.
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1].rpartition("/")[0]:
    import os

    os.environ["ROS_NAMESPACE"] = sys.argv[1].rpartition("/")[0]

import rospy  # noqa: E402
import roslib.message  # noqa: E402


def msg_class(type_name: str):
    cls = roslib.message.get_message_class(type_name)
    if cls is None:
        raise SystemExit(f"message type {type_name} is not installed in this wire image")
    return cls


def srv_class(type_name: str):
    cls = roslib.message.get_service_class(type_name)
    if cls is None:
        raise SystemExit(f"service type {type_name} is not installed in this wire image")
    return cls


class Node:
    dialect = "ros1"

    def __init__(self, name: str):
        self.name = name
        self.robot = common.owner()
        self.iface = common.interface(self.robot)
        rospy.init_node(name.rpartition("/")[2], anonymous=False, disable_signals=True)
        if rospy.get_name() != name:
            raise SystemExit(f"node registered as {rospy.get_name()}, expected {name}")
        self._link = None
        self._link_lock = threading.Lock()
        self.pub = {}
        self.rows = {}
        self.types = {}
        for t in common.topics_of(self.iface, name, "out"):
            cls = msg_class(t["type"])
            self.types[t["name"]] = cls
            self.rows[t["name"]] = t
            self.pub[t["name"]] = rospy.Publisher(t["name"], cls, queue_size=10,
                                                  latch=bool(t.get("latched")))
        for t in common.served(self.iface.get("topics")):
            if name in (t.get("internal_publishers") or []) and t["name"] not in self.pub:
                cls = msg_class(t["type"])
                self.types[t["name"]] = cls
                self.pub[t["name"]] = rospy.Publisher(t["name"], cls, queue_size=10)
        module = importlib.import_module(f"robots.{self.robot.id}")
        behaviours = dict(getattr(module, "BEHAVIOURS", {}))
        row = next((n for n in self.iface.get("nodes", []) if n.get("name") == name), {})
        if name not in behaviours and row.get("package") == "tf" and \
                row.get("executable") == "static_transform_publisher":
            from robots._ros1_drivers import StaticTransformPublisher

            behaviours[name] = StaticTransformPublisher
        self.behaviour = behaviours[name](self) if name in behaviours else None
        handled = set(getattr(self.behaviour, "publishes", ()))
        internal_in = [t for t in common.served(self.iface.get("topics"))
                       if name in (t.get("internal_subscribers") or [])]
        for t in common.topics_of(self.iface, name, "in") + internal_in:
            cls = msg_class(t["type"])
            self.types[t["name"]] = cls
            rospy.Subscriber(t["name"], cls, self._on_msg, callback_args=t["name"],
                             queue_size=10)
        self.srv = {}
        for s in common.services_of(self.iface, name):
            cls = srv_class(s["type"])
            self.srv[s["name"]] = rospy.Service(s["name"], cls, self._handler(s["name"], cls))
        self.timers = []
        for tname, row in self.rows.items():
            rate = common.rate_of(row)
            if rate and tname not in handled:
                self.timers.append(rospy.Timer(rospy.Duration(1.0 / rate),
                                               self._default_pub(tname)))
            elif not rate and row.get("latched") and tname not in handled:
                self.publish(tname, self.new(tname))
        if self.behaviour is not None and hasattr(self.behaviour, "start"):
            self.behaviour.start()

    # ------------------------------------------------------------------ helpers

    @property
    def link(self) -> common.SimLink:
        with self._link_lock:
            if self._link is None:
                self._link = common.SimLink()
                common.exit_when_lost(self._link, lambda: rospy.signal_shutdown("removed"))
            return self._link

    def new(self, topic: str, stamp: float = None):
        msg = self.types[topic]()
        self.fill_header(msg, topic, stamp)
        return msg

    def fill_header(self, msg, topic, stamp=None):
        if hasattr(msg, "header"):
            msg.header.stamp = rospy.Time.from_sec(stamp if stamp is not None else time.time())
            fid = (self.rows.get(topic) or {}).get("frame_id")
            if fid is not None:
                msg.header.frame_id = fid

    @staticmethod
    def time(t: float):
        return rospy.Time.from_sec(t)

    def publish(self, topic: str, msg) -> None:
        self.pub[topic].publish(msg)

    def _default_pub(self, topic):
        def fire(event):
            self.publish(topic, self.new(topic))
        return fire

    def _on_msg(self, msg, topic):
        if self.behaviour is not None:
            fn = getattr(self.behaviour, "on_message", None)
            if fn is not None:
                fn(topic, msg)

    def _handler(self, name, cls):
        def handle(req):
            if self.behaviour is not None and hasattr(self.behaviour, "on_service"):
                res = self.behaviour.on_service(name, req)
                if res is not None:
                    return res
            res = cls._response_class()
            if hasattr(res, "success"):
                res.success = True
            return res
        return handle


def main():
    node = Node(sys.argv[1])
    rospy.spin()
    return node


if __name__ == "__main__":
    main()
