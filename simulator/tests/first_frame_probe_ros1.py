#!/usr/bin/env python3
"""First-message latency of ROS 1 topics, from inside a wire container:

    first_frame_probe_ros1.py <topic> [<repeats>]
      -> JSON {"latency": [s, ...], "connect": [s, ...]}

For each repeat: a new subscriber is created, and the time to its connection with the
publisher (`connect`) and to its first message (`latency`) is measured; between repeats the
subscriber is gone for a second, so a publisher that works only while it is listened to
stops in between. Messages are not deserialised (rospy.AnyMsg)."""
import json
import sys
import threading
import time

import rospy

topic = sys.argv[1]
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rospy.init_node("rsim_first_frame_probe", anonymous=True, disable_signals=True)
out = {"latency": [], "connect": []}
for _ in range(repeats):
    got = threading.Event()
    t0 = time.time()
    sub = rospy.Subscriber(topic, rospy.AnyMsg, lambda _m: got.set(), queue_size=1,
                           buff_size=64 * 1024 * 1024, tcp_nodelay=True)
    while sub.get_num_connections() == 0 and time.time() - t0 < 10:
        time.sleep(0.001)
    out["connect"].append(time.time() - t0)
    got.wait(10)
    out["latency"].append(time.time() - t0 if got.is_set() else None)
    sub.unregister()
    time.sleep(1.0)
print(json.dumps(out), flush=True)
