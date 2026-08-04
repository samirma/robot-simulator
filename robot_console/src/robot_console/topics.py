"""The ROS contract, in one place.

These names and type strings must match `simulator/bridge/rosbridge_server.py` and the
real `elephantrobotics/myagv_ros` stack. They are ROS1 single-slash type strings
(`geometry_msgs/Twist`), not the ROS2 `geometry_msgs/msg/Twist` form.
"""

from __future__ import annotations

TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOM = "/odom"
TOPIC_CAMERA = "/camera/image_raw/compressed"

TYPE_TWIST = "geometry_msgs/Twist"
TYPE_ODOM = "nav_msgs/Odometry"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"

# The simulator's bridge normalises names itself, but stock rosbridge_suite does not:
# a relative `cmd_vel` there resolves against the node namespace and silently misses.
# Sending the absolute form keeps one client working against both.
def normalise(topic: str) -> str:
    """Return `topic` with a leading slash."""
    return topic if topic.startswith("/") else "/" + topic
