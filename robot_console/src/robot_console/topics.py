"""The ROS contract, in one place.

These names and type strings must match
`simulator/shared/contracts/rosbridge_server.py` and the real
`elephantrobotics/myagv_ros` stack. They are ROS1 single-slash type strings
(`geometry_msgs/Twist`), not the ROS2 `geometry_msgs/msg/Twist` form.
"""

from __future__ import annotations

TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOM = "/odom"
TOPIC_CAMERA = "/camera/image_raw/compressed"
# The YDLidar X2's topic, published by ydlidar_ros_driver on the real robot and by the
# simulator's ray-cast stand-in. Nothing in the console consumes it since the mapping
# subsystem was removed, but it is part of the contract and both robots still emit it.
TOPIC_SCAN = "/scan"

TYPE_TWIST = "geometry_msgs/Twist"
TYPE_ODOM = "nav_msgs/Odometry"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
TYPE_LASER_SCAN = "sensor_msgs/LaserScan"

# The simulator's bridge normalises names itself, but stock rosbridge_suite does not:
# a relative `cmd_vel` there resolves against the node namespace and silently misses.
# Sending the absolute form keeps one client working against both.
def normalise(topic: str) -> str:
    """Return `topic` with a leading slash."""
    return topic if topic.startswith("/") else "/" + topic


# When several robots share one graph -- one rosbridge, one port -- each is given a
# namespace and these names are prefixed with it: `/myagv/cmd_vel`. That is ROS's own
# convention (`ROS_NAMESPACE` / `<group ns=>` in ROS 1, `-r __ns:=` in ROS 2), and it is
# why the constants above stay bare: they are what a *single* robot presents, which is
# what a real myAGV bringup is, and the prefix is applied to them rather than baked in.
#
# The rule is duplicated from `simulator/shared/contracts/namespace.py` rather than
# imported, because the console must install and run with no simulator checkout at all.
# `tests/arm/test_ros_contract.py` is what holds the two copies to the same answers.
def namespaced(topic: str, namespace: str) -> str:
    """Return `topic` under `namespace`. Empty namespace changes nothing.

    Idempotent, so a topic a caller already spelled out in full is never prefixed twice --
    that matters because `--namespace` and an explicit `--odom-topic` can both be given,
    and the explicit one has to win.
    """
    topic = normalise(topic)
    namespace = namespace.strip("/")
    if not namespace:
        return topic
    if topic == f"/{namespace}" or topic.startswith(f"/{namespace}/"):
        return topic
    return f"/{namespace}{topic}"
