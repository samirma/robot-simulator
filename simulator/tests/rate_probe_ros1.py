#!/usr/bin/env python3
"""Measures topic rates on a ROS 1 wire from inside its container, without deserialising
(rospy.AnyMsg), so large images and point clouds are counted as published:

    rate_probe_ros1.py <seconds> <topic> [<topic> ...]
      -> JSON {"window": s, "times": {topic: [receive times]}, "stamps": {topic: [header stamps]}}

Header stamps (acquisition times) are read straight from the serialised bytes of message
types whose first field is a std_msgs/Header.
"""
import json
import os
import struct
import sys
import time

import rosgraph
import roslib.message
import rospy

secs = float(sys.argv[1])
topics = sys.argv[2:]
rospy.init_node("rsim_rate_probe", anonymous=True, disable_signals=True)
master = rosgraph.Master("/rsim_rate_probe")
types = dict(master.getTopicTypes())
has_header = {}
for t in topics:
    cls = roslib.message.get_message_class(types.get(t, "")) if t in types else None
    has_header[t] = bool(cls and cls._slot_types and cls._slot_types[0] == "std_msgs/Header")
times = {t: [] for t in topics}
stamps = {t: [] for t in topics}


def cb(m, t):
    times[t].append(time.time())
    if has_header[t]:
        s, ns = struct.unpack_from("<II", m._buff, 4)
        stamps[t].append(s + ns * 1e-9)


for t in topics:
    rospy.Subscriber(t, rospy.AnyMsg, cb, callback_args=t, queue_size=200,
                     buff_size=64 * 1024 * 1024, tcp_nodelay=True)
time.sleep(float(os.environ.get("RSIM_PROBE_WARMUP", "1.0")))
for t in topics:
    times[t].clear()
    stamps[t].clear()
t0 = time.time()
time.sleep(secs)
print(json.dumps({"window": time.time() - t0, "times": times, "stamps": stamps}))
