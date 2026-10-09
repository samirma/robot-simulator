#!/usr/bin/env python3
"""Measures each transform's rate on /tf from inside a wire container (ROS 1 or ROS 2), as
a native subscriber on the wire's own graph sees it:

    tf_rate_probe.py ros1|ros2 <seconds>
      -> JSON {"window": s, "times": {"<parent> -> <child>": [receive times]},
               "stamps": {"<parent> -> <child>": [transform header stamps]}}

A message carrying several transforms (robot_state_publisher sends every moving joint in
one) counts once for each of them.
"""
import json
import os
import sys
import threading
import time

dialect, secs = sys.argv[1], float(sys.argv[2])
times, stamps = {}, {}
lock = threading.Lock()
recording = threading.Event()


def on_tf(msg, sec_of):
    now = time.time()
    if not recording.is_set():
        return
    with lock:
        for t in msg.transforms:
            key = f"{t.header.frame_id} -> {t.child_frame_id}"
            times.setdefault(key, []).append(now)
            stamps.setdefault(key, []).append(sec_of(t.header.stamp))


def settle_and_measure():
    time.sleep(float(os.environ.get("RSIM_PROBE_WARMUP", "1.5")))
    recording.set()
    t0 = time.time()
    time.sleep(secs)
    recording.clear()
    with lock:
        return {"window": time.time() - t0, "times": times, "stamps": stamps}


if dialect == "ros1":
    import rospy
    from tf2_msgs.msg import TFMessage

    rospy.init_node("rsim_tf_rate_probe", anonymous=True, disable_signals=True)
    rospy.Subscriber("/tf", TFMessage, lambda m: on_tf(m, lambda s: s.secs + s.nsecs * 1e-9),
                     queue_size=1000, tcp_nodelay=True)
    print(json.dumps(settle_and_measure()))
else:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = rclpy.create_node("_rsim_tf_rate_probe")
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)
    node.create_subscription(TFMessage, "/tf",
                             lambda m: on_tf(m, lambda s: s.sec + s.nanosec * 1e-9), qos)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    print(json.dumps(settle_and_measure()))
    rclpy.try_shutdown()
