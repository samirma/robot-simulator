#!/usr/bin/env python3
"""A ROS 1 node presenting a typed interface from a JSON spec, for real-rosbridge tests.

Same spec as ros2_wire.py (ROS 1 spellings). Received messages and service requests are
printed as ``WIRE {json}`` lines.
"""

import importlib
import json
import sys

import rospy


def load_type(spelling, kind="msg"):
    pkg, name = spelling.split("/")
    return getattr(importlib.import_module(f"{pkg}.{kind}"), name)


def to_dict(m):
    out = {}
    for slot in getattr(m, "__slots__", []):
        v = getattr(m, slot)
        out[slot] = to_dict(v) if hasattr(v, "__slots__") else (v if isinstance(v, (int, float, str, bool)) else str(v))
    return out


def log(**kw):
    print("WIRE " + json.dumps(kw, default=str), flush=True)


def main():
    spec = json.load(open(sys.argv[1]))
    rospy.init_node("console_test_wire")
    keep = []
    timers = []
    for t in spec.get("topics", []):
        cls = load_type(t["type"])
        if t.get("direction") == "in":
            keep.append(rospy.Subscriber(t["name"], cls, lambda m, n=t["name"]: log(ev="msg", topic=n, msg=to_dict(m))))
        else:
            latched = t["name"] in ("/tf_static",)
            pub = rospy.Publisher(t["name"], cls, queue_size=10, latch=latched)
            keep.append(pub)
            msg = cls()
            if t.get("camera"):
                msg.height, msg.width, msg.encoding = 48, 64, t.get("encoding", "rgb8")
                msg.step = msg.width * {"yuv422_yuy2": 2, "yuv422": 2, "mono8": 1, "rgba8": 4, "bgra8": 4}.get(msg.encoding, 3)
                msg.data = bytes([(i * 7) % 256 for i in range(msg.step * msg.height)])
            rate = t.get("rate_hz") or 1.0
            timers.append(rospy.Timer(rospy.Duration(1.0 / rate), lambda e, p=pub, m=msg: p.publish(m)))
    for s in spec.get("services", []):
        cls = load_type(s["type"], "srv")
        resp_cls = getattr(importlib.import_module(cls.__module__), cls.__name__ + "Response")

        def handler(req, n=s["name"], resp_cls=resp_cls):
            log(ev="call", service=n, req=to_dict(req))
            return resp_cls()
        keep.append(rospy.Service(s["name"], cls, handler))
    log(ev="ready")
    rospy.spin()


if __name__ == "__main__":
    main()
