#!/usr/bin/env python3
"""A ROS 2 node presenting a typed interface from a JSON spec, for real-rosbridge tests.

spec = {"topics": [{"name", "type", "direction": "in"|"out", "rate_hz", "camera": bool,
                    "encoding"}],
        "services": [{"name", "type"}], "actions": [{"name", "type", "exec_s"}]}

"out" topics are published (default-constructed messages; cameras as small images) and
"in" topics subscribed. Every received message, service request and action event is
printed as one JSON line prefixed with ``WIRE `` so tests can read them from the logs.
"""

import importlib
import json
import sys
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosidl_runtime_py.convert import message_to_ordereddict


def load_type(spelling):
    pkg, kind, name = spelling.split("/")
    return getattr(importlib.import_module(f"{pkg}.{kind}"), name)


def log(**kw):
    print("WIRE " + json.dumps(kw, default=str), flush=True)


class Wire(Node):
    def __init__(self, spec):
        super().__init__("console_test_wire")
        self.group = ReentrantCallbackGroup()
        self.keep = []
        for t in spec.get("topics", []):
            cls = load_type(t["type"])
            if t.get("direction") == "in":
                self.keep.append(self.create_subscription(
                    cls, t["name"], lambda m, n=t["name"]: log(ev="msg", topic=n, msg=message_to_ordereddict(m)), 10))
            else:
                latched = t["name"] in ("/robot_description", "/tf_static")
                qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL) if latched else 10
                pub = self.create_publisher(cls, t["name"], qos)
                self.keep.append(pub)
                rate = t.get("rate_hz") or (1.0 if not latched else None)
                msg = cls()
                if t.get("camera"):
                    msg.height, msg.width, msg.encoding = 48, 64, t.get("encoding", "rgb8")
                    msg.step = msg.width * {"yuv422_yuy2": 2, "yuv422": 2, "mono8": 1, "rgba8": 4, "bgra8": 4}.get(msg.encoding, 3)
                    msg.data = bytes([(i * 7) % 256 for i in range(msg.step * msg.height)])
                    msg.header.frame_id = "camera"
                if latched:
                    pub.publish(msg)
                if rate:
                    self.keep.append(self.create_timer(1.0 / rate, lambda p=pub, m=msg: p.publish(m)))
        for s in spec.get("services", []):
            cls = load_type(s["type"])

            def handler(req, resp, n=s["name"]):
                log(ev="call", service=n, req=message_to_ordereddict(req))
                return resp
            self.keep.append(self.create_service(cls, s["name"], handler, callback_group=self.group))
        for a in spec.get("actions", []):
            cls = load_type(a["type"])
            exec_s = float(a.get("exec_s", 3.0))

            latest = {}

            # Like a ros2_control controller: a newer goal preempts the running one (here it
            # ends ABORTED; rclpy offers no cancel without a cancel request), and a running goal
            # publishes feedback periodically.
            def execute(handle, n=a["name"], cls=cls, exec_s=exec_s, latest=latest):
                latest["h"] = handle
                log(ev="goal_exec", action=n, goal=message_to_ordereddict(handle.request))
                end = time.monotonic() + exec_s
                next_fb = 0.0
                while time.monotonic() < end:
                    if handle.is_cancel_requested:
                        handle.canceled()
                        log(ev="goal_canceled", action=n)
                        return cls.Result()
                    if latest.get("h") is not handle:
                        handle.abort()
                        log(ev="goal_preempted", action=n)
                        return cls.Result()
                    if time.monotonic() >= next_fb:
                        handle.publish_feedback(cls.Feedback())
                        next_fb = time.monotonic() + 0.05
                    time.sleep(0.02)
                handle.succeed()
                log(ev="goal_succeeded", action=n)
                return cls.Result()

            def goal_cb(goal, n=a["name"]):
                log(ev="goal", action=n)
                return GoalResponse.ACCEPT

            def cancel_cb(handle, n=a["name"]):
                log(ev="cancel", action=n)
                return CancelResponse.ACCEPT
            self.keep.append(ActionServer(self, cls, a["name"], execute, goal_callback=goal_cb,
                                          cancel_callback=cancel_cb, callback_group=self.group))
        log(ev="ready")


def main():
    spec = json.load(open(sys.argv[1]))
    rclpy.init()
    node = Wire(spec)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
