#!/usr/bin/env python3
"""A ROS 2 node presenting a typed interface from a JSON spec, for real-rosbridge tests.

spec = {"topics": [{"name", "type", "direction": "in"|"out", "rate_hz", "camera": bool,
                    "encoding", "sample": message dict, "latched": bool, "qos": "best_effort"}],
        "services": [{"name", "type"}], "actions": [{"name", "type", "exec_s", "busy"}]}

"out" topics are published (default-constructed messages; cameras as small images) and
"in" topics subscribed. An action server handles a goal that arrives while one of its goals
runs as its row's "busy" says: "preempt" (the default) ends the running goal, "reject"
refuses the new goal (as the reBot's controller does), "parallel" runs it beside the others.
Every received message, service request and action event is printed as one JSON line
prefixed with ``WIRE `` so tests can read them from the logs.
"""

import importlib
import json
import sys
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields


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
                latched = t["name"] in ("/robot_description", "/tf_static") or bool(t.get("latched"))
                if latched:
                    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                elif t.get("qos") == "best_effort":      # sensor-data QoS, as the reBot's controller publishes its states
                    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
                else:
                    qos = 10
                pub = self.create_publisher(cls, t["name"], qos)
                self.keep.append(pub)
                rate = t.get("rate_hz") or (1.0 if not latched else None)
                msg = cls()
                if t.get("sample") is not None:           # a sensor topic: this message, not a default one
                    set_message_fields(msg, t["sample"])
                if t.get("camera"):
                    msg.height, msg.width, msg.encoding = 48, 64, t.get("encoding", "rgb8")
                    msg.step = msg.width * {"yuv422_yuy2": 2, "yuv422": 2, "mono8": 1, "rgba8": 4, "bgra8": 4, "16UC1": 2}.get(msg.encoding, 3)
                    if msg.encoding == "16UC1":           # aligned depth: little-endian millimetres, 0 = no depth, 70..500 mm
                        n = msg.width * msg.height
                        msg.data = b"".join((0 if i == 0 else 70 + round((i - 1) * 430 / (n - 2))).to_bytes(2, "little") for i in range(n))
                    else:
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
            busy = a.get("busy", "preempt")
            state = {"latest": None, "running": False, "lock": threading.Lock()}

            # A running goal publishes feedback periodically. "preempt": like a ros2_control
            # controller, a newer goal preempts the running one (here it ends ABORTED; rclpy offers
            # no cancel without a cancel request). "reject": like the reBot's controller, the server
            # is busy from accepting a goal until just before that goal's result
            # (reBotArmController_ROS2 ros_actions.py#L46-L59, #L228-L233), and refuses goals meanwhile.
            def execute(handle, n=a["name"], cls=cls, exec_s=exec_s, busy=busy, state=state):
                state["latest"] = handle
                log(ev="goal_exec", action=n, goal=message_to_ordereddict(handle.request))
                end = time.monotonic() + exec_s
                next_fb = 0.0
                outcome = "succeeded"
                while time.monotonic() < end:
                    if handle.is_cancel_requested:
                        outcome = "canceled"
                        break
                    if busy == "preempt" and state["latest"] is not handle:
                        outcome = "preempted"
                        break
                    if time.monotonic() >= next_fb:
                        handle.publish_feedback(cls.Feedback())
                        next_fb = time.monotonic() + 0.05
                    time.sleep(0.02)
                if busy == "reject":
                    with state["lock"]:
                        state["running"] = False
                {"succeeded": handle.succeed, "canceled": handle.canceled, "preempted": handle.abort}[outcome]()
                log(ev="goal_" + outcome, action=n)
                return cls.Result()

            def goal_cb(goal, n=a["name"], busy=busy, state=state):
                if busy == "reject":
                    with state["lock"]:
                        if state["running"]:
                            log(ev="goal_rejected", action=n)
                            return GoalResponse.REJECT
                        state["running"] = True
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
