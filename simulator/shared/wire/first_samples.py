#!/usr/bin/env python3
"""Waits, on the wire's own ROS graph, until every listed topic has delivered a message:
the supervisor's last readiness step (spec §2.3: a spawn succeeds only when its wire is
ready to accept its recorded commands *and provide its required outputs*).

    first_samples.py ros1 <timeout s> <topic> [<topic> ...]
    first_samples.py ros2 <timeout s> <topic>=<type> [...]
      -> JSON {"silent": [topics with no message within the timeout]}; exit 0 when none

Messages are not deserialised (ROS 1 `rospy.AnyMsg`, ROS 2 raw subscriptions), so large
images and point clouds cost nothing. The ROS 2 subscriber is best effort and volatile,
which matches every publisher's QoS; its node is hidden (a leading underscore), and the
ROS 1 one unregisters when it exits -- the probe is gone before the wire is ready.
"""

import json
import sys
import threading
import time


def ros1(topics, timeout):
    import rospy

    got = set()
    rospy.init_node("rsim_ready_probe", anonymous=True, disable_signals=True)
    subs = [rospy.Subscriber(t, rospy.AnyMsg, lambda _m, t=t: got.add(t), queue_size=1,
                             buff_size=64 * 1024 * 1024) for t in topics]
    deadline = time.monotonic() + timeout
    while got != set(topics) and time.monotonic() < deadline:
        time.sleep(0.1)
    for s in subs:
        s.unregister()
    rospy.signal_shutdown("done")
    return got


def ros2(pairs, timeout):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message

    got = set()
    rclpy.init()
    node = rclpy.create_node("_rsim_ready_probe")
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    for t, typ in pairs:
        node.create_subscription(get_message(typ), t, lambda _m, t=t: got.add(t), qos,
                                 raw=True)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    deadline = time.monotonic() + timeout
    while got != {t for t, _ in pairs} and time.monotonic() < deadline:
        time.sleep(0.1)
    ex.shutdown(timeout_sec=1.0)
    node.destroy_node()
    rclpy.try_shutdown()
    return got


def main() -> int:
    dialect, timeout, args = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    if dialect == "ros1":
        topics = list(args)
        got = ros1(topics, timeout)
    else:
        pairs = [a.split("=", 1) for a in args]
        topics = [t for t, _ in pairs]
        got = ros2(pairs, timeout)
    silent = [t for t in topics if t not in got]
    print(json.dumps({"silent": silent}), flush=True)
    return 1 if silent else 0


if __name__ == "__main__":
    sys.exit(main())
