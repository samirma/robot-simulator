#!/usr/bin/env python3
"""Measures topic rates on a ROS 2 wire from inside its container without deserialising
(raw subscriptions with the publishers' reliability, as the robot's own subscribers see them):

    rate_probe_ros2.py <seconds> <topic>=<type> [...]
      -> JSON {"window": s, "times": {topic: [receive times]}, "stamps": {topic: [header stamps]}}

Header stamps are read from the CDR bytes of message types whose first field is a
std_msgs/Header.
"""
import json
import os
import struct
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message

secs = float(sys.argv[1])
pairs = [a.split("=", 1) for a in sys.argv[2:]]
rclpy.init()
node = rclpy.create_node("_rsim_rate_probe")


def qos_for(topic):
    """Reliable when every publisher offers it (a best-effort reader drops samples under
    load that the robot's own reliable subscribers would receive), else best effort."""
    deadline = time.time() + 5.0
    infos = []
    while time.time() < deadline and not infos:
        infos = node.get_publishers_info_by_topic(topic)
        if not infos:
            time.sleep(0.1)
    reliable = bool(infos) and all(
        i.qos_profile.reliability == ReliabilityPolicy.RELIABLE for i in infos)
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=200,
                      reliability=ReliabilityPolicy.RELIABLE if reliable
                      else ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE)


times = {t: [] for t, _ in pairs}
stamps = {t: [] for t, _ in pairs}


def make_cb(t, header):
    def cb(raw):
        times[t].append(time.time())
        if header:
            s, ns = struct.unpack_from("<iI", raw, 4)
            stamps[t].append(s + ns * 1e-9)
    return cb


for t, typ in pairs:
    cls = get_message(typ)
    fields = list(cls.get_fields_and_field_types().items())
    header = bool(fields) and fields[0][1] == "std_msgs/Header"
    node.create_subscription(cls, t, make_cb(t, header), qos_for(t), raw=True)
ex = MultiThreadedExecutor(num_threads=4)
ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start()
time.sleep(float(os.environ.get("RSIM_PROBE_WARMUP", "1.5")))
for t in times:
    times[t].clear()
    stamps[t].clear()
t0 = time.time()
time.sleep(secs)
print(json.dumps({"window": time.time() - t0, "times": times, "stamps": stamps}))
rclpy.try_shutdown()
