"""The myAGV's ROS contract, in one place.

The console's copy of `robots_specs/myagv/ros.yml`, the official interface of the real
`elephantrobotics/myagv_ros` stack, which the simulator's
`simulator/shared/ros_surfaces/myagv.py` presents too. `tests/test_myagv_contract.py`
holds the two copies to the same answers when the simulator is checked out alongside.

They are ROS 1 single-slash type strings (`geometry_msgs/Twist`), not the ROS 2
`geometry_msgs/msg/Twist` form.

**There is no command watchdog.** `myagv_odometry_node` clamps each `/cmd_vel` component
to [-1, 1] and re-sends the last command to the base every cycle with no timeout, so the
base keeps moving until a zero Twist arrives. Stopping the robot is the console's job.
"""

from __future__ import annotations

# --- myagv_odometry_node --------------------------------------------------------------
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOM = "/odom"
TOPIC_IMU = "/imu"
TOPIC_VOLTAGE = "/Voltage"
TOPIC_VOLTAGE_BACKUP = "/voltage_backup"
# --- robot description and transform tree ---------------------------------------------
# `robot_state_publisher` (base_footprint -> base_up, and a latched, empty `/tf_static`:
# the URDF has no fixed joint), three `static_transform_publisher`s re-publishing the
# camera, IMU and lidar mounts on `/tf`, and `robot_pose_ekf`, which is what broadcasts
# `odom -> base_footprint`. Nothing in this console consumes them except `slam/scan.py`,
# which applies the lidar mount itself.
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_TF = "/tf"
TOPIC_TF_STATIC = "/tf_static"
TOPIC_ODOM_COMBINED = "/robot_pose_ekf/odom_combined"
# --- ydlidar_lidar_publisher (YDLidar X2), in laser_frame ------------------------------
TOPIC_SCAN = "/scan"
TOPIC_POINT_CLOUD = "/point_cloud"
# --- camera (usb_cam), in camera_link ---------------------------------------------------
TOPIC_IMAGE_RAW = "/camera/image_raw"
TOPIC_CAMERA_INFO = "/camera/camera_info"
TOPIC_CAMERA = "/camera/image_raw/compressed"

#: The parameter carrying the robot's URDF, which is what a tf tree is read against.
PARAM_ROBOT_DESCRIPTION = "/robot_description"

TYPE_TWIST = "geometry_msgs/Twist"
TYPE_ODOM = "nav_msgs/Odometry"
TYPE_IMU = "sensor_msgs/Imu"
TYPE_FLOAT32 = "std_msgs/Float32"
TYPE_JOINT_STATE = "sensor_msgs/JointState"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
TYPE_IMAGE = "sensor_msgs/Image"
TYPE_CAMERA_INFO = "sensor_msgs/CameraInfo"
TYPE_LASER_SCAN = "sensor_msgs/LaserScan"
TYPE_POINT_CLOUD = "sensor_msgs/PointCloud"
TYPE_TF_MESSAGE = "tf2_msgs/TFMessage"

#: Every topic the myAGV presents, with its type.
CONTRACT_TOPICS: dict[str, str] = {
    TOPIC_CMD_VEL: TYPE_TWIST,
    TOPIC_ODOM: TYPE_ODOM,
    TOPIC_IMU: TYPE_IMU,
    TOPIC_VOLTAGE: TYPE_FLOAT32,
    TOPIC_VOLTAGE_BACKUP: TYPE_FLOAT32,
    TOPIC_JOINT_STATES: TYPE_JOINT_STATE,
    TOPIC_TF: TYPE_TF_MESSAGE,
    TOPIC_TF_STATIC: TYPE_TF_MESSAGE,
    TOPIC_ODOM_COMBINED: TYPE_ODOM,
    TOPIC_SCAN: TYPE_LASER_SCAN,
    TOPIC_POINT_CLOUD: TYPE_POINT_CLOUD,
    TOPIC_IMAGE_RAW: TYPE_IMAGE,
    TOPIC_CAMERA_INFO: TYPE_CAMERA_INFO,
    TOPIC_CAMERA: TYPE_COMPRESSED_IMAGE,
}

#: Every service, with its type.
CONTRACT_SERVICES: dict[str, str] = {
    "/stop_scan": "std_srvs/Empty",
    "/start_scan": "std_srvs/Empty",
    "/robot_pose_ekf/get_status": "robot_pose_ekf/GetStatus",
    "/camera/start_capture": "std_srvs/Empty",
    "/camera/stop_capture": "std_srvs/Empty",
    "/camera/set_camera_info": "sensor_msgs/SetCameraInfo",
}

#: Each `/cmd_vel` component is clamped to [-CMD_VEL_LIMIT, CMD_VEL_LIMIT].
CMD_VEL_LIMIT = 1.0

#: Frames, bare.
FRAME_ODOM = "odom"
FRAME_BASE = "base_footprint"
FRAME_LASER = "laser_frame"
FRAME_CAMERA = "camera_link"
FRAME_IMU = "imu_link"


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
