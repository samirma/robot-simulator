"""The ROSMASTER X3 PLUS's ROS interface: `robots_specs/rosmaster_x3_plus/ros.yml`, transcribed.

A ROS 1 (Noetic) graph: `roslaunch yahboomcar_nav laser_astrapro_bringup.launch` with
`ROBOT_TYPE=X3plus` -- the YDLIDAR TG launch, the Orbbec Astra Pro Plus launch under
`/camera`, and `yahboomcar_bringup/bringup.launch` (the `driver_node` on the STM32
expansion board, `odometry_publisher`, `imu_filter_madgwick`, `ekf_localization`,
`robot_state_publisher` and the gamepad's `joy_node` / `yahboom_joy`). Every name, type,
node, frame, parameter and periodic rate below is that file's; nothing is added and
nothing is left out. Node names are the file's, without its leading slash: they are
composed with the robot's namespace where they reach the wire, as every name is.

Behaviour, as the vendor's nodes behave:

* `/cmd_vel` goes to the board through `set_car_motion(vx, vy, wz)` and is held until the
  next command: **no timeout**. The board takes X3PLUS commands within v_x, v_y
  [-0.7, 0.7] m/s and v_z [-3.2, 3.2] rad/s (`CMD_VEL_LIMITS`), so a larger component is
  carried out at its limit. The stop command is a zero Twist (`STOP_COMMAND`).
* `/vel_raw` is the board's measured chassis velocity; `odometry_publisher` integrates it,
  scaled by its `linear_scale_x` / `linear_scale_y`, into `/odom_raw`. The board is
  simulated as reporting what those calibrated scales expect (measured velocity divided by
  them), so `/odom_raw` and the EKF's `/odom` track the simulated base, from the pose it
  started at, as a calibrated robot's do.
* The arm is six serial-bus servos. `/TargetAngle` moves one (`id`, `angle` deg,
  `run_time` ms) or all six (`joints`, deg); an out-of-range set of six is refused, as the
  board library refuses it, and `run_time` is clamped to [0, 2000] ms. Each accepted
  command is echoed twice on `/ArmAngleUpdate` -- carrying the joints as they were before
  it, as the driver's own code does. `/joint_states` reports the *commanded* angles
  (servo degrees minus 90, the gripper's 30..180 mapped to 0..90 first), and
  `/CurrentAngle` reads the servos back. At start-up the driver sends
  `STARTUP_JOINTS_DEG`, which is the pose the robot is spawned in (`REST_POSITIONS`).
* `/scan` is the TG launch's: -85..85 deg in frame `laser`, 0.01..50 m, invalid returns
  as infinity, one point every 360 / (sample_rate x 1000 / frequency) deg.
  `/stop_scan` stops `/scan` and `/point_cloud` until `/start_scan`.
* The Astra Pro Plus publishes colour (rgb8), depth (16UC1, millimetres, 0 outside the
  sensor's published 0.6..8 m working range) and IR (mono16) at 640x480 and 30 Hz, each
  with its `camera_info`, and the two point clouds (in the depth and colour optical
  frames) only while subscribed, as the driver does. `/camera/toggle_*` turns a stream
  off and on. The simulated IR image is the depth sensor's view in grey.

numpy and MuJoCo are imported inside the functions that need them, so the constants stay
plain Python: the workspace parity tests read them parsed, not imported.
"""

from __future__ import annotations

import math
import sys

from contracts.physical import MESH_DIMENSION_PCT, Figure, percent

# ------------------------------------------------------------------------------ nodes

NODE_DRIVER = "driver_node"
NODE_ODOMETRY = "odometry_publisher"
NODE_MADGWICK = "imu_filter_madgwick"
NODE_EKF = "ekf_localization"
NODE_RSP = "robot_state_publisher"
NODE_LASER_TF = "laser_link_to_laser"
NODE_JOY = "joy_node"
NODE_YAHBOOM_JOY = "yahboom_joy"
NODE_LIDAR = "ydlidar_lidar_publisher"
NODE_CAMERA = "camera/camera"
NODE_CAMERA_STATICS = ("camera/camera_base_link", "camera/camera_base_link1",
                       "camera/camera_base_link2", "camera/camera_base_link3")

# ------------------------------------------------------------------------------ topics

TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_RGB_LIGHT = "/RGBLight"
TOPIC_BUZZER = "/Buzzer"
TOPIC_TARGET_ANGLE = "/TargetAngle"
TOPIC_ARM_ANGLE_UPDATE = "/ArmAngleUpdate"
TOPIC_EDITION = "/edition"
TOPIC_VOLTAGE = "/voltage"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_VEL_RAW = "/vel_raw"
TOPIC_IMU_RAW = "/imu/data_raw"
TOPIC_MAG_RAW = "/mag/mag_raw"
TOPIC_PARAM_DESCRIPTIONS = "/driver_node/parameter_descriptions"
TOPIC_PARAM_UPDATES = "/driver_node/parameter_updates"
TOPIC_ODOM_RAW = "/odom_raw"
TOPIC_IMU = "/imu/data"
TOPIC_ODOM = "/odom"
TOPIC_DIAGNOSTICS = "/diagnostics"
TOPIC_TF = "/tf"
TOPIC_TF_STATIC = "/tf_static"
TOPIC_JOY = "/joy"
TOPIC_JOY_FEEDBACK = "/joy/set_feedback"
TOPIC_JOY_STATE = "/JoyState"
TOPIC_MOVE_BASE_CANCEL = "/move_base/cancel"
TOPIC_SCAN = "/scan"
TOPIC_POINT_CLOUD = "/point_cloud"
TOPIC_DEPTH_IMAGE = "/camera/depth/image_raw"
TOPIC_DEPTH_INFO = "/camera/depth/camera_info"
TOPIC_RGB_IMAGE = "/camera/rgb/image_raw"
TOPIC_RGB_INFO = "/camera/rgb/camera_info"
TOPIC_IR_IMAGE = "/camera/ir/image_raw"
TOPIC_IR_INFO = "/camera/ir/camera_info"
TOPIC_DEPTH_POINTS = "/camera/depth/points"
TOPIC_REGISTERED_POINTS = "/camera/depth_registered/points"

TYPE_TWIST = "geometry_msgs/Twist"
TYPE_INT32 = "std_msgs/Int32"
TYPE_BOOL = "std_msgs/Bool"
TYPE_FLOAT32 = "std_msgs/Float32"
TYPE_ARM_JOINT = "yahboomcar_msgs/ArmJoint"
TYPE_JOINT_STATE = "sensor_msgs/JointState"
TYPE_IMU = "sensor_msgs/Imu"
TYPE_MAG = "sensor_msgs/MagneticField"
TYPE_CONFIG_DESCRIPTION = "dynamic_reconfigure/ConfigDescription"
TYPE_CONFIG = "dynamic_reconfigure/Config"
TYPE_ODOM = "nav_msgs/Odometry"
TYPE_DIAGNOSTICS = "diagnostic_msgs/DiagnosticArray"
TYPE_TF_MESSAGE = "tf2_msgs/TFMessage"
TYPE_JOY = "sensor_msgs/Joy"
TYPE_JOY_FEEDBACK = "sensor_msgs/JoyFeedbackArray"
TYPE_GOAL_ID = "actionlib_msgs/GoalID"
TYPE_LASER_SCAN = "sensor_msgs/LaserScan"
TYPE_POINT_CLOUD = "sensor_msgs/PointCloud"
TYPE_IMAGE = "sensor_msgs/Image"
TYPE_CAMERA_INFO = "sensor_msgs/CameraInfo"
TYPE_POINT_CLOUD2 = "sensor_msgs/PointCloud2"

EVENT = "event"
LATCHED = "latched"
#: `/diagnostics` has no stated rate in the ROS file; robot_localization's diagnostic
#: updater sends once a second (`DIAGNOSTICS_HZ`).
UNVERIFIED = "unverified"

#: Every topic, as `(name, type, direction, node, rate_hz)`: one row per publishing or
#: subscribing node, as the ROS file lists them.
TOPICS: tuple[tuple[str, str, str, str, float | str], ...] = (
    (TOPIC_CMD_VEL, TYPE_TWIST, "in", NODE_DRIVER, EVENT),
    (TOPIC_RGB_LIGHT, TYPE_INT32, "in", NODE_DRIVER, EVENT),
    (TOPIC_BUZZER, TYPE_BOOL, "in", NODE_DRIVER, EVENT),
    (TOPIC_TARGET_ANGLE, TYPE_ARM_JOINT, "in", NODE_DRIVER, EVENT),
    (TOPIC_ARM_ANGLE_UPDATE, TYPE_ARM_JOINT, "out", NODE_DRIVER, EVENT),
    (TOPIC_EDITION, TYPE_FLOAT32, "out", NODE_DRIVER, 20.0),
    (TOPIC_VOLTAGE, TYPE_FLOAT32, "out", NODE_DRIVER, 20.0),
    (TOPIC_JOINT_STATES, TYPE_JOINT_STATE, "out", NODE_DRIVER, 20.0),
    (TOPIC_VEL_RAW, TYPE_TWIST, "out", NODE_DRIVER, 20.0),
    (TOPIC_IMU_RAW, TYPE_IMU, "out", NODE_DRIVER, 20.0),
    (TOPIC_MAG_RAW, TYPE_MAG, "out", NODE_DRIVER, 20.0),
    (TOPIC_PARAM_DESCRIPTIONS, TYPE_CONFIG_DESCRIPTION, "out", NODE_DRIVER, LATCHED),
    (TOPIC_PARAM_UPDATES, TYPE_CONFIG, "out", NODE_DRIVER, LATCHED),
    (TOPIC_ODOM_RAW, TYPE_ODOM, "out", NODE_ODOMETRY, 20.0),
    (TOPIC_IMU, TYPE_IMU, "out", NODE_MADGWICK, 20.0),
    (TOPIC_ODOM, TYPE_ODOM, "out", NODE_EKF, 20.0),
    (TOPIC_DIAGNOSTICS, TYPE_DIAGNOSTICS, "out", NODE_EKF, UNVERIFIED),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_EKF, 20.0),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_RSP, 20.0),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_LASER_TF, 33.333),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_CAMERA, 10.0),
    (TOPIC_TF_STATIC, TYPE_TF_MESSAGE, "out", NODE_RSP, LATCHED),
    (TOPIC_TF_STATIC, TYPE_TF_MESSAGE, "out", NODE_CAMERA_STATICS[0], LATCHED),
    (TOPIC_TF_STATIC, TYPE_TF_MESSAGE, "out", NODE_CAMERA_STATICS[1], LATCHED),
    (TOPIC_TF_STATIC, TYPE_TF_MESSAGE, "out", NODE_CAMERA_STATICS[2], LATCHED),
    (TOPIC_TF_STATIC, TYPE_TF_MESSAGE, "out", NODE_CAMERA_STATICS[3], LATCHED),
    (TOPIC_JOY, TYPE_JOY, "out", NODE_JOY, EVENT),
    (TOPIC_JOY_FEEDBACK, TYPE_JOY_FEEDBACK, "in", NODE_JOY, EVENT),
    (TOPIC_CMD_VEL, TYPE_TWIST, "out", NODE_YAHBOOM_JOY, EVENT),
    (TOPIC_BUZZER, TYPE_BOOL, "out", NODE_YAHBOOM_JOY, EVENT),
    (TOPIC_JOY_STATE, TYPE_BOOL, "out", NODE_YAHBOOM_JOY, EVENT),
    (TOPIC_RGB_LIGHT, TYPE_INT32, "out", NODE_YAHBOOM_JOY, EVENT),
    (TOPIC_TARGET_ANGLE, TYPE_ARM_JOINT, "out", NODE_YAHBOOM_JOY, EVENT),
    (TOPIC_ARM_ANGLE_UPDATE, TYPE_ARM_JOINT, "in", NODE_YAHBOOM_JOY, EVENT),
    (TOPIC_MOVE_BASE_CANCEL, TYPE_GOAL_ID, "out", NODE_YAHBOOM_JOY, EVENT),
    (TOPIC_SCAN, TYPE_LASER_SCAN, "out", NODE_LIDAR, 10.0),
    (TOPIC_POINT_CLOUD, TYPE_POINT_CLOUD, "out", NODE_LIDAR, 10.0),
    (TOPIC_DEPTH_IMAGE, TYPE_IMAGE, "out", NODE_CAMERA, 30.0),
    (TOPIC_DEPTH_INFO, TYPE_CAMERA_INFO, "out", NODE_CAMERA, 30.0),
    (TOPIC_RGB_IMAGE, TYPE_IMAGE, "out", NODE_CAMERA, 30.0),
    (TOPIC_RGB_INFO, TYPE_CAMERA_INFO, "out", NODE_CAMERA, 30.0),
    (TOPIC_IR_IMAGE, TYPE_IMAGE, "out", NODE_CAMERA, 30.0),
    (TOPIC_IR_INFO, TYPE_CAMERA_INFO, "out", NODE_CAMERA, 30.0),
    (TOPIC_DEPTH_POINTS, TYPE_POINT_CLOUD2, "out", NODE_CAMERA, 30.0),
    (TOPIC_REGISTERED_POINTS, TYPE_POINT_CLOUD2, "out", NODE_CAMERA, 30.0),
)


def rate_of(topic: str, node: str) -> float | str:
    """The declared rate of `topic` as `node` publishes it."""
    for name, _type, direction, owner, rate in TOPICS:
        if name == topic and owner == node and direction == "out":
            return rate
    raise KeyError(f"{topic} is not published by {node} on the ROSMASTER X3 PLUS")


# ---------------------------------------------------------------------------- services

SRV_EMPTY = "std_srvs/Empty"
SRV_SET_BOOL = "std_srvs/SetBool"
SRV_ROBOT_ARM_ARRAY = "yahboomcar_msgs/RobotArmArray"
SRV_RECONFIGURE = "dynamic_reconfigure/Reconfigure"
SRV_GET_INT32 = "orbbec_camera/GetInt32"
SRV_SET_INT32 = "orbbec_camera/SetInt32"
SRV_GET_BOOL = "orbbec_camera/GetBool"
SRV_GET_CAMERA_INFO = "orbbec_camera/GetCameraInfo"
SRV_GET_DEVICE_INFO = "orbbec_camera/GetDeviceInfo"
SRV_GET_STRING = "orbbec_camera/GetString"
SRV_GET_CAMERA_PARAMS = "orbbec_camera/GetCameraParams"
SRV_SET_STRING = "orbbec_camera/SetString"

SERVICE_CURRENT_ANGLE = "/CurrentAngle"
SERVICE_DRIVER_RECONFIGURE = "/driver_node/set_parameters"
SERVICE_STOP_SCAN = "/stop_scan"
SERVICE_START_SCAN = "/start_scan"

#: The Astra's image streams, as its driver names them in its services.
CAMERA_STREAMS = ("depth", "color", "ir")

#: Every service, as `(name, type, node)`. The camera node creates the same eleven for
#: each enabled stream (ros_service.cpp), then its device-wide ones.
SERVICES: tuple[tuple[str, str, str], ...] = (
    (SERVICE_CURRENT_ANGLE, SRV_ROBOT_ARM_ARRAY, NODE_DRIVER),
    (SERVICE_DRIVER_RECONFIGURE, SRV_RECONFIGURE, NODE_DRIVER),
    (SERVICE_STOP_SCAN, SRV_EMPTY, NODE_LIDAR),
    (SERVICE_START_SCAN, SRV_EMPTY, NODE_LIDAR),
    *((f"/camera/{verb}_{stream}_{what}", kind, NODE_CAMERA)
      for stream in CAMERA_STREAMS
      for verb, what, kind in (
          ("get", "exposure", SRV_GET_INT32), ("set", "exposure", SRV_SET_INT32),
          ("reset", "exposure", SRV_EMPTY), ("get", "gain", SRV_GET_INT32),
          ("set", "gain", SRV_SET_INT32), ("reset", "gain", SRV_EMPTY),
          ("set", "mirror", SRV_SET_BOOL), ("set", "auto_exposure", SRV_SET_BOOL),
          ("get", "auto_exposure", SRV_GET_BOOL))),
    *((f"/camera/toggle_{stream}", SRV_SET_BOOL, NODE_CAMERA) for stream in CAMERA_STREAMS),
    *((f"/camera/get_{stream}_camera_info", SRV_GET_CAMERA_INFO, NODE_CAMERA)
      for stream in CAMERA_STREAMS),
    ("/camera/get_auto_white_balance", SRV_GET_INT32, NODE_CAMERA),
    ("/camera/set_auto_white_balance", SRV_SET_INT32, NODE_CAMERA),
    ("/camera/get_white_balance", SRV_GET_INT32, NODE_CAMERA),
    ("/camera/set_white_balance", SRV_SET_INT32, NODE_CAMERA),
    ("/camera/reset_white_balance", SRV_EMPTY, NODE_CAMERA),
    ("/camera/set_fan_work_mode", SRV_SET_BOOL, NODE_CAMERA),
    ("/camera/set_floor", SRV_SET_BOOL, NODE_CAMERA),
    ("/camera/set_laser", SRV_SET_BOOL, NODE_CAMERA),
    ("/camera/set_ldp", SRV_SET_BOOL, NODE_CAMERA),
    ("/camera/get_ldp_status", SRV_GET_BOOL, NODE_CAMERA),
    ("/camera/get_device_info", SRV_GET_DEVICE_INFO, NODE_CAMERA),
    ("/camera/get_serial", SRV_GET_STRING, NODE_CAMERA),
    ("/camera/get_camera_params", SRV_GET_CAMERA_PARAMS, NODE_CAMERA),
    ("/camera/get_sdk_version", SRV_GET_STRING, NODE_CAMERA),
    ("/camera/get_device_type", SRV_GET_STRING, NODE_CAMERA),
    ("/camera/save_point_cloud", SRV_EMPTY, NODE_CAMERA),
    ("/camera/save_images", SRV_EMPTY, NODE_CAMERA),
    ("/camera/switch_ir_mode", SRV_SET_INT32, NODE_CAMERA),
    ("/camera/switch_ir", SRV_SET_STRING, NODE_CAMERA),
)

#: The ROS file's `actions: []`: the bringup starts no action server.
ACTIONS: tuple = ()

# -------------------------------------------------------------------------- parameters

PARAM_ROBOT_DESCRIPTION = "/robot_description"

#: The EKF's `odom0_config` / `imu0_config`, as robot_localization.yaml sets them (the
#: ROS file names the true entries: vx, vy, vyaw; roll, pitch, yaw, vroll, vpitch, vyaw).
_ODOM0_CONFIG = [False, False, False, False, False, False, True, True, False,
                 False, False, True, False, False, False]
_IMU0_CONFIG = [False, False, False, True, True, True, False, False, False,
                True, True, True, False, False, False]

#: Every other parameter, with the value the launches set.
PARAMETERS: dict[str, object] = {
    "/driver_node/xlinear_speed_limit": 0.7,
    "/driver_node/ylinear_speed_limit": 0.7,
    "/driver_node/angular_speed_limit": 3.2,
    "/driver_node/imu_link": "imu_link",
    "/driver_node/prefix": "",
    "/driver_node/Kp": 1.5,
    "/driver_node/Ki": 0.3,
    "/driver_node/Kd": 0.2,
    "/driver_node/linear_max": 0.4,
    "/driver_node/angular_max": 2.0,
    "/driver_node/linear_min": 0.0,
    "/driver_node/angular_min": 0.0,
    "/driver_node/joint1": 90,
    "/driver_node/joint2": 145,
    "/driver_node/joint3": 0,
    "/driver_node/joint4": 0,
    "/driver_node/joint5": 90,
    "/driver_node/joint6": 30,
    "/driver_node/SetArmjoint": False,
    "/odometry_publisher/odom_frame": "odom",
    "/odometry_publisher/base_footprint_frame": "base_footprint",
    "/odometry_publisher/linear_scale_x": 1.1,
    "/odometry_publisher/linear_scale_y": 0.95,
    "/imu_filter_madgwick/fixed_frame": "base_link",
    "/imu_filter_madgwick/use_mag": False,
    "/imu_filter_madgwick/publish_tf": False,
    "/imu_filter_madgwick/use_magnetic_field_msg": False,
    "/imu_filter_madgwick/world_frame": "enu",
    "/imu_filter_madgwick/orientation_stddev": 0.05,
    "/imu_filter_madgwick/angular_scale": 1.03,
    "/ekf_localization/odom_frame": "/odom",
    "/ekf_localization/world_frame": "/odom",
    "/ekf_localization/base_link_frame": "/base_footprint",
    "/ekf_localization/frequency": 20,
    "/ekf_localization/sensor_timeout": 0.1,
    "/ekf_localization/two_d_mode": True,
    "/ekf_localization/odom0": "/odom0",
    "/ekf_localization/odom0_config": _ODOM0_CONFIG,
    "/ekf_localization/odom0_differential": True,
    "/ekf_localization/imu0": "/imu0",
    "/ekf_localization/imu0_config": _IMU0_CONFIG,
    "/ekf_localization/imu0_differential": True,
    "/ekf_localization/imu0_remove_gravitational_acceleration": True,
    "/ekf_localization/use_control": False,
    "/use_sim_time": False,
    "/yahboom_joy/linear_speed_limit": 0.7,
    "/yahboom_joy/angular_speed_limit": 3.2,
    "/ydlidar_lidar_publisher/port": "/dev/ydlidar",
    "/ydlidar_lidar_publisher/frame_id": "laser",
    "/ydlidar_lidar_publisher/ignore_array": "",
    "/ydlidar_lidar_publisher/baudrate": 512000,
    "/ydlidar_lidar_publisher/lidar_type": 0,
    "/ydlidar_lidar_publisher/device_type": 0,
    "/ydlidar_lidar_publisher/sample_rate": 20,
    "/ydlidar_lidar_publisher/abnormal_check_count": 4,
    "/ydlidar_lidar_publisher/resolution_fixed": True,
    "/ydlidar_lidar_publisher/auto_reconnect": True,
    "/ydlidar_lidar_publisher/reversion": True,
    "/ydlidar_lidar_publisher/inverted": True,
    "/ydlidar_lidar_publisher/isSingleChannel": False,
    "/ydlidar_lidar_publisher/intensity": False,
    "/ydlidar_lidar_publisher/support_motor_dtr": False,
    "/ydlidar_lidar_publisher/invalid_range_is_inf": True,
    "/ydlidar_lidar_publisher/point_cloud_preservative": False,
    "/ydlidar_lidar_publisher/angle_min": -85.0,
    "/ydlidar_lidar_publisher/angle_max": 85.0,
    "/ydlidar_lidar_publisher/range_min": 0.01,
    "/ydlidar_lidar_publisher/range_max": 50.0,
    "/ydlidar_lidar_publisher/frequency": 10.0,
    "/camera/camera/camera_name": "camera",
    "/camera/camera/depth_registration": False,
    "/camera/camera/serial_number": "",
    "/camera/camera/usb_port": "",
    "/camera/camera/device_num": 1,
    "/camera/camera/vendor_id": "0x2bc5",
    "/camera/camera/product_id": "",
    "/camera/camera/enable_point_cloud": True,
    "/camera/camera/enable_colored_point_cloud": True,
    "/camera/camera/connection_delay": 100,
    "/camera/camera/color_width": 640,
    "/camera/camera/color_height": 480,
    "/camera/camera/color_fps": 30,
    "/camera/camera/enable_color": True,
    "/camera/camera/color_format": "MJPG",
    "/camera/camera/flip_color": False,
    "/camera/camera/enable_color_auto_exposure": True,
    "/camera/camera/depth_width": 640,
    "/camera/camera/depth_height": 480,
    "/camera/camera/depth_fps": 30,
    "/camera/camera/enable_depth": True,
    "/camera/camera/depth_format": "Y11",
    "/camera/camera/flip_depth": False,
    "/camera/camera/ir_width": 640,
    "/camera/camera/ir_height": 480,
    "/camera/camera/ir_fps": 30,
    "/camera/camera/enable_ir": True,
    "/camera/camera/ir_format": "Y10",
    "/camera/camera/flip_ir": False,
    "/camera/camera/enable_ir_auto_exposure": True,
    "/camera/camera/publish_tf": True,
    "/camera/camera/tf_publish_rate": 10.0,
    "/camera/camera/ir_info_uri": "",
    "/camera/camera/color_info_uri": "",
    "/camera/camera/log_level": "none",
    "/camera/camera/enable_d2c_viewer": False,
    "/camera/camera/enable_pipeline": True,
    "/camera/camera/enable_soft_filter": True,
}

# ------------------------------------------------------------------------------ frames

FRAME_ODOM = "odom"
FRAME_BASE = "base_footprint"
FRAME_IMU = "imu_link"
FRAME_LASER_LINK = "laser_link"
FRAME_LASER = "laser"
FRAME_CAMERA_LINK = "camera_link"
FRAME_JOINT_STATES = "joint_states"   # the driver's own header.frame_id on /joint_states
#: The Astra driver's frames (ob_camera_node.cpp: `<camera>_<stream>_frame` and
#: `..._optical_frame`), and astra_frames.launch's older `rgb` pair.
FRAME_DEPTH = "camera_depth_frame"
FRAME_COLOR = "camera_color_frame"
FRAME_IR = "camera_ir_frame"
FRAME_DEPTH_OPTICAL = "camera_depth_optical_frame"
FRAME_COLOR_OPTICAL = "camera_color_optical_frame"
FRAME_IR_OPTICAL = "camera_ir_optical_frame"
FRAME_RGB = "camera_rgb_frame"
FRAME_RGB_OPTICAL = "camera_rgb_optical_frame"

#: `laser_link_to_laser`: `0 0 0 6.28 0 0 /laser_link /laser 30` (x y z yaw pitch roll).
LASER_YAW = 6.28

#: The optical rotation both the driver and astra_frames.launch use: rpy (-pi/2, 0, -pi/2).
OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)

# ------------------------------------------------------------------------- behaviour

#: `set_car_motion`'s X3PLUS input range (Rosmaster_Lib.py#L558): v_x, v_y, v_z.
CMD_VEL_LIMITS = (0.7, 0.7, 3.2)

#: The stop command: a zero Twist on `/cmd_vel`. There is no timeout.
STOP_COMMAND = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}

#: `/joint_states` names, in the driver's order (also the ROS file's `joints`).
JOINTS = ("arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint")

#: The servo angle ranges the board library accepts (`set_uart_servo_angle_array`), deg.
SERVO_RANGES_DEG = ((0, 180), (0, 180), (0, 180), (0, 180), (0, 270), (0, 180))
#: The gripper servo's travel, and the joint span `/joint_states` maps it onto.
GRIPPER_SERVO_DEG = (30.0, 180.0)
GRIPPER_JOINT_DEG = (0.0, 90.0)
#: `set_uart_servo_angle_array`'s `run_time` clamp, ms.
RUN_TIME_MS = (0, 2000)

#: What the driver sends the arm at start-up (Mcnamu_X3plus.py#L53), servo degrees.
STARTUP_JOINTS_DEG = (90, 145, 0, 45, 90, 30)


def joint_positions(servo_deg) -> list[float]:
    """Servo degrees -> `/joint_states` positions (rad), exactly as the driver maps them."""
    deg = list(float(v) for v in servo_deg)
    lo, hi = GRIPPER_SERVO_DEG
    jlo, jhi = GRIPPER_JOINT_DEG
    g = min(max(deg[5], lo), hi)
    deg[5] = jlo + (g - lo) * (jhi - jlo) / (hi - lo)
    return [math.radians(v - 90.0) for v in deg]


def servo_degrees(positions) -> list[float]:
    """The inverse of `joint_positions`: joint radians -> servo degrees."""
    deg = [math.degrees(float(q)) + 90.0 for q in positions]
    lo, hi = GRIPPER_SERVO_DEG
    jlo, jhi = GRIPPER_JOINT_DEG
    deg[5] = lo + (deg[5] - jlo) * (hi - lo) / (jhi - jlo)
    return deg


#: The URDF's joint limits, which the compiled model's joints hold (the gripper's is
#: narrower than the servo's mapped travel: -1.54 rad against the mapping's -pi/2).
URDF_LIMITS = {"arm_joint1": (-1.5708, 1.5708), "arm_joint2": (-1.5708, 1.5708),
               "arm_joint3": (-1.5708, 1.5708), "arm_joint4": (-1.5708, 1.5708),
               "arm_joint5": (-1.5708, 3.14159), "grip_joint": (-1.54, 0.0)}


def _clamp_joint(name: str, q: float) -> float:
    lo, hi = URDF_LIMITS[name]
    return min(max(q, lo), hi)


#: The pose the robot is spawned in: the driver's start-up command, within the URDF.
REST_POSITIONS: dict[str, float] = {
    name: _clamp_joint(name, q)
    for name, q in zip(JOINTS, joint_positions(STARTUP_JOINTS_DEG))
}

#: `odometry_publisher`'s scales, which the simulated board's `/vel_raw` expects.
LINEAR_SCALE = (1.1, 0.95)

#: TG.launch's scan geometry.
SCAN_ANGLE_MIN = math.radians(-85.0)
SCAN_ANGLE_MAX = math.radians(85.0)
SCAN_RANGE_MIN = 0.01
SCAN_RANGE_MAX = 50.0
SCAN_INVALID = float("inf")
#: Points per revolution with `resolution_fixed`: sample_rate (kHz) over the frequency.
SCAN_POINTS_PER_TURN = round(20 * 1000 / 10.0)
SCAN_INCREMENT = 2.0 * math.pi / SCAN_POINTS_PER_TURN
SCAN_BEAMS = int((SCAN_ANGLE_MAX - SCAN_ANGLE_MIN) / SCAN_INCREMENT + 1)

#: The Astra Pro Plus at the launch's 640x480: published fields of view (RGB V 46.81 deg,
#: depth V 45.8 deg) and working range, 0.6..8 m.
CAMERA_SIZE = (640, 480)
RGB_FOVY_DEG = 46.81
DEPTH_FOVY_DEG = 45.8
DEPTH_RANGE = (0.6, 8.0)
#: IR is Y10: ten bits in a mono16 image.
IR_MAX = 1023

#: The expansion board's firmware (rosmaster_V3.5.1.hex; `get_version` gives H + L/10).
EDITION = 3.5
#: `/voltage`: a charged 3S pack. The simulated battery does not drain.
VOLTAGE = 12.3
GRAVITY = 9.80665
#: The magnetic field `/mag/mag_raw` reads, in the world (ENU), tesla: a mid-latitude
#: field. Content, like the camera images: the board's magnetometer points at whatever
#: field it is in.
EARTH_FIELD_ENU = (0.0, 2.2e-5, -4.2e-5)
DIAGNOSTICS_HZ = 1.0

#: The geom group the lidar's rays skip (MuJoCo has six); this robot's geoms are moved
#: into it on the lidar's private copy of the model.
RAY_HIDDEN_GROUP = 5

#: How often this surface is stepped. A multiple of every rate it keeps (20, 30, 10,
#: 33.3 Hz) at which each lands within a third of its own period, and the servo moves
#: stay smooth.
LOOP_HZ = 100.0

# --------------------------------------------------------------- physical figures

_PRODUCT = "https://category.yahboom.net/products/rosmaster-x3-plus"
_DIMENSIONS = ("https://cdn.shopify.com/s/files/1/0066/9686/1780/files/"
               "ROSMASTER_X3_PLUS_Yahboom2_08.jpg")
_PHOTOS = ("https://cdn.shopify.com/s/files/1/0066/9686/1780/files/"
           "ROSMASTER_X3_PLUS_Details_23.jpg")
_SENSORS = ("https://cdn.shopify.com/s/files/1/0066/9686/1780/files/"
            "ROSMASTER_X3_PLUS_Details_19.jpg")

#: The published figures (spec §3, real-robot fidelity), from the product page's own
#: images, each measured on the compiled model by `shared/tests/physical_figures_check.py`.
PHYSICAL_FIGURES: tuple[Figure, ...] = (
    Figure("width_m", "width over the wheels", 0.2456, "m",
           percent(0.2456, MESH_DIMENSION_PCT), _DIMENSIONS, "245.60mm",
           "y extent of the chassis mesh (base_link) in the base frame"),
    Figure("height_m", "overall height", 0.515, "m", percent(0.515, MESH_DIMENSION_PCT),
           _DIMENSIONS, "515mm",
           "top of the Astra camera over the floor, the drawing's overall height"),
    Figure("mass_kg", "weight after assembly", 4.35, "kg", 0.01, _PHOTOS,
           "Weight after assembly: about 4.35kg",
           "subtree mass of the base body; the URDF's links total 1.37 kg, so the chassis "
           "link carries the published remainder (shared/robot_models.py)"),
    Figure("max_speed_mps", "maximum commanded speed", 0.7, "m/s", percent(0.7, 3.0),
           "https://drive.google.com/file/d/1SRg1aD_u8kyxxjm4vp0cFYxu8Ddj2sXU",
           "X3PLUS: v_x=[-0.7, 0.7], v_y=[-0.7, 0.7], v_z=[-3.2, 3.2]",
           "planar speed the compiled base reaches under the largest /cmd_vel x the board "
           "takes (Rosmaster_Lib set_car_motion's X3PLUS range); 3% covers the position "
           "servo's tracking lag"),
    Figure("rgb_hfov_deg", "colour camera field of view (horizontal) at 640x480", 60.60,
           "deg", 1.0, _SENSORS, "H60.60° V46.81° @640*480",
           "horizontal field of view of the MJCF rgb_camera at 640x480, square pixels "
           "(its vertical one is the published 46.81)"),
)

GRAVITY_ENU = (0.0, 0.0, GRAVITY)


# ---------------------------------------------------------------------- message shapes


def _stamp(seq: int, frame_id: str, stamp_s: float) -> dict:
    return {"seq": int(seq),
            "stamp": {"secs": int(stamp_s), "nsecs": int((stamp_s % 1) * 1e9)},
            "frame_id": frame_id}


def _quat_msg(q) -> dict:
    return {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])}


def _yaw_quat(yaw: float):
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _vec(v) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _clamp(value, limit: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:
        return 0.0
    return max(-limit, min(limit, v))


def scan_bearings(beams: int = SCAN_BEAMS) -> list[float]:
    return [SCAN_ANGLE_MIN + i * SCAN_INCREMENT for i in range(beams)]


def intrinsics(fovy_deg: float, size=CAMERA_SIZE) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a pinhole with square pixels and vertical field `fovy_deg`."""
    width, height = size
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return f, f, (width - 1) / 2.0, (height - 1) / 2.0


def camera_info(seq: int, frame_id: str, stamp_s: float, fovy_deg: float) -> dict:
    """The driver's `convertToCameraInfo`: plumb_bob, the device's intrinsics."""
    fx, fy, cx, cy = intrinsics(fovy_deg)
    width, height = CAMERA_SIZE
    return {
        "header": _stamp(seq, frame_id, stamp_s), "height": height, "width": width,
        "distortion_model": "plumb_bob", "D": [0.0] * 5,
        "K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "binning_x": 0, "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }


# ------------------------------------------------------- dynamic_reconfigure, driver_node

#: PIDparam.cfg: (name, type, default, min, max, description).
PID_PARAMS = (
    ("Kp", "double", 1.5, 0.0, 10.0, "Kp in PID"),
    ("Ki", "double", 0.3, 0.0, 10.0, "Ki in PID"),
    ("Kd", "double", 0.2, 0.0, 10.0, "Kd in PID"),
    ("linear_max", "double", 0.4, 0.0, 1.0, "speed in limit"),
    ("angular_max", "double", 2.0, 0.0, 5.0, "speed in limit"),
    ("linear_min", "double", 0.0, 0.0, 1.0, "speed in limit"),
    ("angular_min", "double", 0.0, 0.0, 5.0, "speed in limit"),
    ("joint1", "int", 90, 0, 180, "joint1 in arm"),
    ("joint2", "int", 145, 0, 180, "joint2 in arm"),
    ("joint3", "int", 0, 0, 180, "joint3 in arm"),
    ("joint4", "int", 0, 0, 180, "joint4 in arm"),
    ("joint5", "int", 90, 0, 270, "joint5 in arm"),
    ("joint6", "int", 30, 30, 180, "joint6 in arm"),
    ("SetArmjoint", "bool", False, False, True, "SetArmjoint"),
)


def dr_config(values: dict) -> dict:
    """A dynamic_reconfigure/Config of `values`, in PIDparam.cfg's order."""
    out = {"bools": [], "ints": [], "strs": [], "doubles": [],
           "groups": [{"name": "Default", "state": True, "id": 0, "parent": 0}]}
    for name, kind, *_ in PID_PARAMS:
        key = {"bool": "bools", "int": "ints", "double": "doubles"}[kind]
        out[key].append({"name": name, "value": values[name]})
    return out


def dr_description() -> dict:
    """PIDparam.cfg as its dynamic_reconfigure server describes it."""
    return {
        "groups": [{"name": "Default", "type": "", "parent": 0, "id": 0, "parameters": [
            {"name": n, "type": k, "level": 0, "description": d, "edit_method": ""}
            for n, k, _dflt, _lo, _hi, d in PID_PARAMS]}],
        "max": dr_config({n: hi for n, _k, _d, _lo, hi, _x in PID_PARAMS}),
        "min": dr_config({n: lo for n, _k, _d, lo, _hi, _x in PID_PARAMS}),
        "dflt": dr_config({n: d for n, _k, d, _lo, _hi, _x in PID_PARAMS}),
    }


# ------------------------------------------------------------------------- the surface


class _Every:
    """A drift-free clock for one periodic stream (as `ros_surfaces.myagv._Every`)."""

    def __init__(self, hz: float, slack: float) -> None:
        self.period = 1.0 / float(hz)
        self.slack = slack
        self.next: float | None = None

    def ready(self, now: float) -> bool:
        if self.next is None or self.next > now + 2 * self.period:
            self.next = now
        return now + self.slack >= self.next

    def take(self, now: float) -> None:
        self.next += self.period
        if self.next < now - self.period:
            self.next = now + self.period

    def due(self, now: float) -> bool:
        if not self.ready(now):
            return False
        self.take(now)
        return True


class ServoArm:
    """The six serial-bus servos, as the board moves them: each to its target over the
    command's run time. `commanded` is the driver's `self.joints` (servo degrees)."""

    def __init__(self) -> None:
        self.commanded = [float(v) for v in STARTUP_JOINTS_DEG]
        start = [REST_POSITIONS[j] for j in JOINTS]
        self._from = list(start)
        self._to = list(start)
        self._t0 = 0.0
        self._duration = 0.0
        self._pending: list | None = None
        self.now = 0.0

    def command(self, servo_deg, run_time_ms: float) -> None:
        run = min(max(float(run_time_ms), RUN_TIME_MS[0]), RUN_TIME_MS[1]) / 1000.0
        target = [_clamp_joint(n, q) for n, q in zip(JOINTS, joint_positions(servo_deg))]
        self._pending = [self.target(self.now), target, run]

    def target(self, now: float) -> list[float]:
        if self._duration <= 0.0 or now >= self._t0 + self._duration:
            return list(self._to)
        a = (now - self._t0) / self._duration
        return [f + (t - f) * a for f, t in zip(self._from, self._to)]

    def step(self, now: float) -> list[float]:
        self.now = now
        if self._pending is not None:
            self._from, self._to, self._duration = self._pending
            self._t0 = now
            self._pending = None
        return self.target(now)


def attach_ros(bus, base, model, camera: str | None = None, *, jpeg_quality: int = 80,
               lidar: dict | None = None, scene_option=None, world_reset=None,
               prefix: str = ""):
    """Wire one ROSMASTER X3 PLUS onto `bus` and return the per-step callback.

    `base` is the planar base (`pose`, writable `ctrl`); `model`, with `prefix` its MJCF
    prefix, carries the arm joints, the lidar's `laser_link` and the Astra's two cameras
    (`rgb_camera`, `depth_camera`). With `model=None` the sensors publish nothing and the
    arm is a record of its targets, which is what a wire-only test wants. `camera` is
    accepted with the engines' common arguments; the Astra's cameras are resolved here.
    """
    import threading

    import numpy as np

    import robots_spec
    from contracts.rosbridge_server import b64text, odometry
    from contracts.tf import UrdfTree, rpy_to_quat, tf_message
    from mujoco_bridge import PlanarSetpoint, RenderWorker, laser_scan_ranges

    slack = 0.5 / LOOP_HZ
    urdf_text = robots_spec.urdf_path("rosmaster_x3_plus").read_text(encoding="utf-8")
    tree = UrdfTree(urdf_text)
    mimics = _mimics(urdf_text)
    lock = threading.Lock()

    # -- declarations -----------------------------------------------------------------
    for name, mtype, direction, node, _rate in TOPICS:
        if direction == "out" and name != TOPIC_TF_STATIC:
            bus.advertise(name, mtype, node=node)
    bus.set_param(PARAM_ROBOT_DESCRIPTION, urdf_text)
    for name, value in PARAMETERS.items():
        bus.set_param(name, value)

    # -- driver_node: commands --------------------------------------------------------
    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    arm = ServoArm()
    state = {"scanning": True, "config": {n: d for n, _k, d, *_ in PID_PARAMS},
             "streams": {s: True for s in CAMERA_STREAMS}, "rgb_light": 0,
             "buzzer": False, "camera_settings": {}}

    def on_cmd_vel(msg: dict) -> None:
        linear, angular = msg.get("linear") or {}, msg.get("angular") or {}
        lx, ly, lz = CMD_VEL_LIMITS
        with lock:
            command.update(vx=_clamp(linear.get("x", 0.0), lx),
                           vy=_clamp(linear.get("y", 0.0), ly),
                           wz=_clamp(angular.get("z", 0.0), lz))

    def publish_arm_update(msg: dict) -> None:
        for _ in range(2):
            bus.publish(TOPIC_ARM_ANGLE_UPDATE, msg, TYPE_ARM_JOINT, node=NODE_DRIVER)

    def on_target_angle(msg: dict) -> None:
        joints = list(msg.get("joints") or [])
        run_time = msg.get("run_time", 0) or 0
        # A malformed command (not six joints, or no servo 1..6) is dropped: on the robot it
        # raises inside the driver's callback, and its 20 Hz loop with it.
        with lock:
            before = list(arm.commanded)
            if joints:
                if len(joints) != 6:
                    return
                if all(lo <= float(v) <= hi for v, (lo, hi) in zip(joints, SERVO_RANGES_DEG)):
                    arm.command(joints, run_time)
                # Out of range, Rosmaster_Lib prints "angle set error" and sends nothing;
                # the driver records and echoes the command all the same.
                arm.commanded = [float(v) for v in joints]
                reply = {"id": 0, "run_time": 0, "angle": 0.0, "joints": before}
            else:
                sid = int(msg.get("id", 0) or 0)
                angle = float(msg.get("angle", 0.0) or 0.0)
                if not 1 <= sid <= 6:
                    return
                lo, hi = SERVO_RANGES_DEG[sid - 1]
                if lo <= angle <= hi:
                    target = list(arm.commanded)
                    target[sid - 1] = angle
                    arm.command(target, run_time)
                arm.commanded[sid - 1] = angle
                reply = {"id": sid, "run_time": 0, "angle": angle, "joints": []}
        publish_arm_update(reply)

    def on_rgb_light(msg: dict) -> None:
        state["rgb_light"] = int(msg.get("data", 0) or 0)

    def on_buzzer(msg: dict) -> None:
        state["buzzer"] = bool(msg.get("data", False))

    bus.on(TOPIC_CMD_VEL, on_cmd_vel, TYPE_TWIST, node=NODE_DRIVER)
    bus.on(TOPIC_RGB_LIGHT, on_rgb_light, TYPE_INT32, node=NODE_DRIVER)
    bus.on(TOPIC_BUZZER, on_buzzer, TYPE_BOOL, node=NODE_DRIVER)
    bus.on(TOPIC_TARGET_ANGLE, on_target_angle, TYPE_ARM_JOINT, node=NODE_DRIVER)
    bus.on(TOPIC_JOY_FEEDBACK, lambda _m: None, TYPE_JOY_FEEDBACK, node=NODE_JOY)
    bus.on(TOPIC_ARM_ANGLE_UPDATE, lambda _m: None, TYPE_ARM_JOINT, node=NODE_YAHBOOM_JOY)

    # -- driver_node: dynamic_reconfigure ------------------------------------------------
    bus.publish(TOPIC_PARAM_DESCRIPTIONS, dr_description(), TYPE_CONFIG_DESCRIPTION,
                latched=True, node=NODE_DRIVER)
    bus.publish(TOPIC_PARAM_UPDATES, dr_config(state["config"]), TYPE_CONFIG,
                latched=True, node=NODE_DRIVER)

    def reconfigure(args: dict) -> dict:
        config = (args or {}).get("config") or {}
        with lock:
            for key in ("bools", "ints", "doubles"):
                for item in config.get(key) or []:
                    name = item.get("name")
                    spec = next((p for p in PID_PARAMS if p[0] == name), None)
                    if spec is None:
                        continue
                    _n, kind, _d, lo, hi, _x = spec
                    value = item.get("value")
                    value = bool(value) if kind == "bool" else \
                        min(max((int if kind == "int" else float)(value), lo), hi)
                    state["config"][name] = value
            cfg = dict(state["config"])
            if cfg["SetArmjoint"]:
                arm.command([cfg[f"joint{i}"] for i in range(1, 7)], 1000)
        updated = dr_config(cfg)
        bus.publish(TOPIC_PARAM_UPDATES, updated, TYPE_CONFIG, latched=True, node=NODE_DRIVER)
        return {"config": updated}

    # -- services ---------------------------------------------------------------------------
    joint_addr: dict[str, int] = {}

    def current_angle(_args: dict) -> dict:
        data = last_data["data"]
        if data is None:
            positions = arm.target(arm.now)
        else:
            positions = [float(data.qpos[joint_addr[j]]) for j in JOINTS]
        return {"angles": [float(int(round(d))) for d in servo_degrees(positions)]}

    def setter(key: str, value):
        def handler(_args: dict) -> dict:
            state[key] = value
            return {}
        return handler

    def ob_ok(extra: dict | None = None) -> dict:
        return {"success": True, "message": "", **(extra or {})}

    def ob_set(key: str):
        def handler(args: dict) -> dict:
            state["camera_settings"][key] = (args or {}).get("data")
            return ob_ok()
        return handler

    def ob_get(key: str, default):
        def handler(_args: dict) -> dict:
            return ob_ok({"data": state["camera_settings"].get(key, default)})
        return handler

    def ob_toggle(stream: str):
        def handler(args: dict) -> dict:
            state["streams"][stream] = bool((args or {}).get("data", True))
            return ob_ok()
        return handler

    frame = bus.frame
    fovs = {"depth": DEPTH_FOVY_DEG, "color": RGB_FOVY_DEG, "ir": DEPTH_FOVY_DEG}
    optical = {"depth": FRAME_DEPTH_OPTICAL, "color": FRAME_COLOR_OPTICAL,
               "ir": FRAME_IR_OPTICAL}

    def ob_info(stream: str):
        def handler(_args: dict) -> dict:
            return ob_ok({"info": camera_info(0, frame(optical[stream]), bus_now(),
                                              fovs[stream])})
        return handler

    def ob_params(_args: dict) -> dict:
        fx, fy, cx, cy = intrinsics(DEPTH_FOVY_DEG)
        rfx, rfy, rcx, rcy = intrinsics(RGB_FOVY_DEG)
        return ob_ok({"l_intr_p": [fx, fy, cx, cy], "r_intr_p": [rfx, rfy, rcx, rcy],
                      "r2l_r": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], "r2l_t": [0.0, 0.0, 0.0]})

    def ob_device(_args: dict) -> dict:
        return ob_ok({"info": {"header": _stamp(0, "", bus_now()), "name": "Astra Pro Plus",
                               "vid": 0x2BC5, "pid": 0, "serial_number": "",
                               "firmware_version": "", "supported_min_sdk_version": "",
                               "hardware_version": ""}})

    def bus_now() -> float:
        return float(last_data["t"])

    handlers = {
        SERVICE_CURRENT_ANGLE: current_angle,
        SERVICE_DRIVER_RECONFIGURE: reconfigure,
        SERVICE_STOP_SCAN: setter("scanning", False),
        SERVICE_START_SCAN: setter("scanning", True),
        "/camera/get_auto_white_balance": ob_get("auto_white_balance", 1),
        "/camera/set_auto_white_balance": ob_set("auto_white_balance"),
        "/camera/get_white_balance": ob_get("white_balance", 4600),
        "/camera/set_white_balance": ob_set("white_balance"),
        "/camera/reset_white_balance": lambda _a: {},
        "/camera/get_ldp_status": ob_get("ldp", True),
        "/camera/get_device_info": ob_device,
        "/camera/get_serial": ob_get("serial", ""),
        "/camera/get_camera_params": ob_params,
        "/camera/get_sdk_version": ob_get("sdk_version", ""),
        "/camera/get_device_type": ob_get("device_type", "Astra Pro Plus"),
        "/camera/save_point_cloud": lambda _a: {},
        "/camera/save_images": lambda _a: {},
    }
    for name, stype, node in SERVICES:
        if name in handlers:
            handler = handlers[name]
        elif name.startswith("/camera/toggle_"):
            handler = ob_toggle(name.rsplit("_", 1)[1])
        elif name.endswith("_camera_info"):
            handler = ob_info(name[len("/camera/get_"):-len("_camera_info")])
        elif stype == SRV_EMPTY:
            handler = lambda _a: {}  # noqa: E731  (resets: nothing is persisted)
        elif stype in (SRV_SET_INT32, SRV_SET_STRING, SRV_SET_BOOL):
            key = name[len("/camera/"):].replace("set_", "", 1)
            handler = ob_set(key) if stype != SRV_SET_BOOL else \
                (lambda k: (lambda a: (state["camera_settings"].__setitem__(
                    k, bool((a or {}).get("data"))), ob_ok())[1]))(key)
        elif stype in (SRV_GET_INT32, SRV_GET_BOOL):
            key = name[len("/camera/"):].replace("get_", "", 1)
            handler = ob_get(key, 0 if stype == SRV_GET_INT32 else True)
        else:
            raise ValueError(f"no handler for {name}")
        bus.service(name, handler, stype, node=node)

    # -- the model -----------------------------------------------------------------------
    worker = depth_worker = None
    act_ids: dict[str, int] = {}
    laser_body = lidar_body = -1
    lidar_exclude = None
    ray_model = None
    ray_groups = np.array([g != RAY_HIDDEN_GROUP for g in range(6)], dtype=np.uint8)
    cam_rgb = cam_depth = None
    if model is not None:
        import mujoco

        for j in JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + j)
            if jid < 0 or aid < 0:
                raise ValueError(f"no servo joint {prefix}{j} in this model")
            joint_addr[j] = int(model.jnt_qposadr[jid])
            act_ids[j] = aid
        laser_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + FRAME_LASER_LINK)
        cam_rgb, cam_depth = prefix + "rgb_camera", prefix + "depth_camera"
        for cam in (cam_rgb, cam_depth):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                raise ValueError(f"no camera {cam!r} in this model")
        if laser_body < 0:
            raise ValueError(f"no body {prefix}{FRAME_LASER_LINK} in this model")
        if lidar is None:
            raise ValueError("the X3 PLUS's lidar is part of its interface; pass `lidar`")
        lidar_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar["body"])
        lidar_exclude = frozenset(lidar.get("exclude_bodies") or ())
        # The lidar ranges nothing of its own robot. Its origin sits inside its own
        # housing and the chassis around it, so skipping them hit by hit would re-cast most
        # beams through meshes of 10^5 triangles (100 ms a sweep): the rays are cast on a
        # copy of the model in which this robot's geoms are in a group the rays ignore.
        import copy

        ray_model = copy.copy(model)
        own = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in lidar_exclude]
        ray_model.geom_group[own] = RAY_HIDDEN_GROUP
        # One render thread per sensor: the colour and the depth view are rendered side
        # by side, as the device captures them, rather than one after the other.
        worker = RenderWorker(model, name=f"{bus.ns or 'rosmaster_x3_plus'} astra colour")
        depth_worker = RenderWorker(model, name=f"{bus.ns or 'rosmaster_x3_plus'} astra depth")

    # -- frames and fixed transforms --------------------------------------------------------
    frames = {n: frame(n) for n in (FRAME_ODOM, FRAME_BASE, FRAME_IMU, FRAME_LASER_LINK,
                                     FRAME_LASER, FRAME_CAMERA_LINK, FRAME_DEPTH, FRAME_COLOR,
                                     FRAME_IR, FRAME_DEPTH_OPTICAL, FRAME_COLOR_OPTICAL,
                                     FRAME_IR_OPTICAL, FRAME_RGB, FRAME_RGB_OPTICAL)}
    identity = (1.0, 0.0, 0.0, 0.0)
    q_optical = rpy_to_quat(*OPTICAL_RPY)
    laser_tf = (frames[FRAME_LASER_LINK], frames[FRAME_LASER], (0.0, 0.0, 0.0),
                _yaw_quat(LASER_YAW))
    # ob_camera_node's six, re-sent on /tf at tf_publish_rate (the simulated device's
    # colour and depth sensors share camera_link's origin).
    camera_dynamic = [
        (frames[FRAME_COLOR], frames[FRAME_COLOR_OPTICAL], (0.0, 0.0, 0.0), q_optical),
        (frames[FRAME_DEPTH], frames[FRAME_DEPTH_OPTICAL], (0.0, 0.0, 0.0), q_optical),
        (frames[FRAME_IR], frames[FRAME_IR_OPTICAL], (0.0, 0.0, 0.0), q_optical),
        (frames[FRAME_CAMERA_LINK], frames[FRAME_DEPTH], (0.0, 0.0, 0.0), identity),
        (frames[FRAME_CAMERA_LINK], frames[FRAME_COLOR], (0.0, 0.0, 0.0), identity),
        (frames[FRAME_CAMERA_LINK], frames[FRAME_IR], (0.0, 0.0, 0.0), identity),
    ]
    # astra_frames.launch's four tf2 static publishers, latched on /tf_static.
    camera_static = {
        NODE_CAMERA_STATICS[0]: (frames[FRAME_CAMERA_LINK], frames[FRAME_DEPTH],
                                 (0.0, 0.0, 0.0), identity),
        NODE_CAMERA_STATICS[1]: (frames[FRAME_CAMERA_LINK], frames[FRAME_RGB],
                                 (0.0, 0.0, 0.0), identity),
        NODE_CAMERA_STATICS[2]: (frames[FRAME_DEPTH], frames[FRAME_DEPTH_OPTICAL],
                                 (0.0, 0.0, 0.0), q_optical),
        NODE_CAMERA_STATICS[3]: (frames[FRAME_RGB], frames[FRAME_RGB_OPTICAL],
                                 (0.0, 0.0, 0.0), q_optical),
    }
    for node, entry in camera_static.items():
        bus.publish(TOPIC_TF_STATIC, tf_message([entry], stamp_s=0.0), TYPE_TF_MESSAGE,
                    latched=True, node=node)
    fixed = [(frame(p), frame(c), pos, quat) for p, c, pos, quat in tree.fixed()]
    bus.publish(TOPIC_TF_STATIC, tf_message(fixed, stamp_s=0.0), TYPE_TF_MESSAGE,
                latched=True, node=NODE_RSP)
    imu_mount = next(q for p, c, _pos, q in tree.fixed() if c == FRAME_IMU)

    clocks = {
        "driver": _Every(rate_of(TOPIC_VEL_RAW, NODE_DRIVER), slack),
        "ekf": _Every(rate_of(TOPIC_ODOM, NODE_EKF), slack),
        "laser_tf": _Every(rate_of(TOPIC_TF, NODE_LASER_TF), slack),
        "camera_tf": _Every(rate_of(TOPIC_TF, NODE_CAMERA), slack),
        "scan": _Every(rate_of(TOPIC_SCAN, NODE_LIDAR), slack),
        "camera": _Every(rate_of(TOPIC_RGB_IMAGE, NODE_CAMERA), slack),
        "diagnostics": _Every(DIAGNOSTICS_HZ, slack),
    }
    scan_time = 1.0 / rate_of(TOPIC_SCAN, NODE_LIDAR)
    setpoint = PlanarSetpoint()
    seqs: dict[str, int] = {}
    last_data = {"data": None, "t": 0.0}
    odom = {"origin": None, "last": None, "v": (0.0, 0.0, 0.0), "accel": (0.0, 0.0)}

    def seq(key: str) -> int:
        seqs[key] = seqs.get(key, 0) + 1
        return seqs[key]

    if world_reset is not None:
        def _forget() -> None:
            setpoint.reset()
            with lock:
                command.update(vx=0.0, vy=0.0, wz=0.0)
            odom.update(last=None, v=(0.0, 0.0, 0.0), accel=(0.0, 0.0))

        world_reset.on_reset(_forget)

    def publish_tf(entries, stamp: float, node: str, key: str) -> None:
        bus.publish(TOPIC_TF, tf_message(entries, stamp_s=stamp, seq=seq(key)),
                    TYPE_TF_MESSAGE, node=node)

    # -- the Astra ---------------------------------------------------------------------------
    width, height = CAMERA_SIZE

    encoders = {topic: _Encoder(f"{bus.ns or 'x3'} {topic}")
                for topic in (TOPIC_RGB_IMAGE, TOPIC_REGISTERED_POINTS, TOPIC_DEPTH_IMAGE,
                              TOPIC_IR_IMAGE, TOPIC_DEPTH_POINTS)}

    def image_msg(n, frame_id, stamp, encoding, step, raw) -> dict:
        return {"header": _stamp(n, frame_id, stamp), "height": height, "width": width,
                "encoding": encoding, "is_bigendian": 0, "step": step,
                "data": b64text(np.ascontiguousarray(raw))}

    rays: dict[float, tuple] = {}

    def cloud_msg(n, frame_id, stamp, depth, rgb, fovy) -> dict:
        """The driver's cloud: every pixel in range, in the optical frame, xyz (and the
        colour packed as a float `rgb`, bytes b, g, r, 0) on a 16- or 20-byte point."""
        if fovy not in rays:
            fx, fy, cx, cy = intrinsics(fovy)
            u, v = np.meshgrid((np.arange(width) - cx) / fx, (np.arange(height) - cy) / fy)
            rays[fovy] = (u.astype(np.float32), v.astype(np.float32))
        ru, rv = rays[fovy]
        valid = (depth >= DEPTH_RANGE[0]) & (depth <= DEPTH_RANGE[1])
        z = depth[valid]
        xyz = np.zeros((len(z), 4 if rgb is None else 5), dtype=np.float32)
        np.multiply(ru[valid], z, out=xyz[:, 0])
        np.multiply(rv[valid], z, out=xyz[:, 1])
        xyz[:, 2] = z
        fields = [{"name": n_, "offset": o, "datatype": 7, "count": 1}
                  for n_, o in (("x", 0), ("y", 4), ("z", 8))]
        if rgb is not None:
            bgr0 = np.zeros((height, width, 4), dtype=np.uint8)
            bgr0[..., 0], bgr0[..., 1], bgr0[..., 2] = rgb[..., 2], rgb[..., 1], rgb[..., 0]
            xyz[:, 4] = bgr0.view(np.float32).reshape(height, width)[valid]
            fields.append({"name": "rgb", "offset": 16, "datatype": 7, "count": 1})
        step = xyz.shape[1] * 4
        return {"header": _stamp(n, frame_id, stamp), "height": 1, "width": int(len(z)),
                "fields": fields, "is_bigendian": False, "point_step": step,
                "row_step": step * int(len(z)),
                "data": b64text(memoryview(xyz).cast("B")),
                "is_dense": True}

    def publish_camera(data, stamp: float) -> bool:
        streams = state["streams"]
        infos = ((TOPIC_RGB_INFO, "color"), (TOPIC_DEPTH_INFO, "depth"), (TOPIC_IR_INFO, "ir"))
        want = {t: bus.has_subscribers(t) for t in (TOPIC_RGB_IMAGE, TOPIC_DEPTH_IMAGE,
                                                    TOPIC_IR_IMAGE, TOPIC_DEPTH_POINTS,
                                                    TOPIC_REGISTERED_POINTS)}
        want[TOPIC_RGB_IMAGE] &= streams["color"]
        want[TOPIC_REGISTERED_POINTS] &= streams["color"] and streams["depth"]
        want[TOPIC_DEPTH_IMAGE] &= streams["depth"]
        want[TOPIC_DEPTH_POINTS] &= streams["depth"]
        want[TOPIC_IR_IMAGE] &= streams["ir"]
        rgb_job = want[TOPIC_RGB_IMAGE] or want[TOPIC_REGISTERED_POINTS]
        depth_job = want[TOPIC_DEPTH_IMAGE] or want[TOPIC_DEPTH_POINTS] or want[TOPIC_IR_IMAGE]
        if worker is not None and ((rgb_job and worker.busy)
                                   or (depth_job and depth_worker.busy)
                                   or any(e.backlog > 1 for e in encoders.values())):
            return False
        n = seq("camera")
        for topic, stream in infos:
            if streams[stream]:
                bus.publish(topic, camera_info(n, frames[optical[stream]], stamp, fovs[stream]),
                            TYPE_CAMERA_INFO, node=NODE_CAMERA)
        if worker is None or not (rgb_job or depth_job):
            return True

        # Each stream is encoded and published on its own thread, so a frame's clouds are
        # encoded while the next frame renders; one thread per topic keeps its order.
        def on_rgb(pixels, _s, depth) -> None:
            if want[TOPIC_RGB_IMAGE]:
                encoders[TOPIC_RGB_IMAGE].run(lambda: bus.publish(
                    TOPIC_RGB_IMAGE, image_msg(n, frames[FRAME_COLOR_OPTICAL], stamp, "rgb8",
                                               width * 3, pixels), TYPE_IMAGE, node=NODE_CAMERA))
            if want[TOPIC_REGISTERED_POINTS]:
                encoders[TOPIC_REGISTERED_POINTS].run(lambda: bus.publish(
                    TOPIC_REGISTERED_POINTS, cloud_msg(n, frames[FRAME_COLOR_OPTICAL], stamp,
                                                       depth, pixels, RGB_FOVY_DEG),
                    TYPE_POINT_CLOUD2, node=NODE_CAMERA))

        def on_depth(pixels, _s, depth) -> None:
            if want[TOPIC_DEPTH_IMAGE]:
                def depth_image() -> None:
                    valid = (depth >= DEPTH_RANGE[0]) & (depth <= DEPTH_RANGE[1])
                    mm = np.where(valid, np.rint(depth * 1000.0), 0).astype("<u2")
                    bus.publish(TOPIC_DEPTH_IMAGE, image_msg(n, frames[FRAME_DEPTH_OPTICAL],
                                                             stamp, "16UC1", width * 2, mm),
                                TYPE_IMAGE, node=NODE_CAMERA)
                encoders[TOPIC_DEPTH_IMAGE].run(depth_image)
            if want[TOPIC_IR_IMAGE]:
                def ir_image() -> None:
                    total = pixels.sum(axis=2, dtype=np.uint16)          # 0..765
                    ir = ((total.astype(np.uint32) * IR_MAX + 382) // 765).astype("<u2")
                    bus.publish(TOPIC_IR_IMAGE, image_msg(n, frames[FRAME_IR_OPTICAL], stamp,
                                                          "mono16", width * 2, ir),
                                TYPE_IMAGE, node=NODE_CAMERA)
                encoders[TOPIC_IR_IMAGE].run(ir_image)
            if want[TOPIC_DEPTH_POINTS]:
                encoders[TOPIC_DEPTH_POINTS].run(lambda: bus.publish(
                    TOPIC_DEPTH_POINTS, cloud_msg(n, frames[FRAME_DEPTH_OPTICAL], stamp, depth,
                                                  None, DEPTH_FOVY_DEG),
                    TYPE_POINT_CLOUD2, node=NODE_CAMERA))

        if rgb_job:
            worker.submit(data, stamp, [(cam_rgb, width, height, scene_option, on_rgb, True)])
        if depth_job:
            depth_worker.submit(data, stamp,
                                [(cam_depth, width, height, scene_option, on_depth, True)])
        return True

    # -- the loop ----------------------------------------------------------------------------
    rot_mount = _quat_matrix(imu_mount)

    def step(data):
        if data is None:
            if worker is not None:
                worker.close()
                depth_worker.close()
            for encoder in encoders.values():
                encoder.close()
            return
        now = stamp = float(getattr(data, "time", 0.0))
        last_data.update(data=data if model is not None else None, t=now)
        pose = base.pose
        x, y = float(pose[0, 3]), float(pose[1, 3])
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        with lock:
            vx, vy, wz = command["vx"], command["vy"], command["wz"]
        base.ctrl = setpoint.step(x, y, yaw, vx, vy, wz, 1.0 / LOOP_HZ)

        # The servos, as the board moves them.
        targets = arm.step(now)
        if model is not None:
            for j, q in zip(JOINTS, targets):
                data.ctrl[act_ids[j]] = q

        if odom["origin"] is None:
            odom["origin"] = (x, y, yaw)
        ox, oy, oyaw = odom["origin"]
        c0, s0 = math.cos(oyaw), math.sin(oyaw)
        px, py = c0 * (x - ox) + s0 * (y - oy), -s0 * (x - ox) + c0 * (y - oy)
        pyaw = math.atan2(math.sin(yaw - oyaw), math.cos(yaw - oyaw))

        # -- driver_node at 20 Hz: vel_raw, imu, mag, voltage, edition, joint_states ------
        if clocks["driver"].due(now):
            last = odom["last"]
            if last is not None and now > last[0]:
                h = now - last[0]
                wvx, wvy = (x - last[1]) / h, (y - last[2]) / h
                rate = math.atan2(math.sin(yaw - last[3]), math.cos(yaw - last[3])) / h
                c, s = math.cos(yaw), math.sin(yaw)
                bvx, bvy = c * wvx + s * wvy, -s * wvx + c * wvy
                pvx, pvy, _ = odom["v"]
                odom["accel"] = ((bvx - pvx) / h, (bvy - pvy) / h)
                odom["v"] = (bvx, bvy, rate)
            odom["last"] = (now, x, y, yaw)
            bvx, bvy, rate = odom["v"]
            ax, ay = odom["accel"]
            # Specific force in the base frame (gravity reaction up), then in imu_link.
            accel_imu = rot_mount.T @ np.array([ax, ay, GRAVITY])
            gyro_imu = rot_mount.T @ np.array([0.0, 0.0, rate])
            field_base = _rotz(-yaw) @ np.array(EARTH_FIELD_ENU)
            mag_imu = rot_mount.T @ field_base
            sx, sy = LINEAR_SCALE
            bus.publish(TOPIC_VEL_RAW, {"linear": _vec((bvx / sx, bvy / sy, 0.0)),
                                        "angular": _vec((0.0, 0.0, rate))},
                        TYPE_TWIST, node=NODE_DRIVER)
            imu_raw = {
                "header": _stamp(seq("imu_raw"), frames[FRAME_IMU], stamp),
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
                "orientation_covariance": [0.0] * 9,
                "angular_velocity": _vec(gyro_imu), "angular_velocity_covariance": [0.0] * 9,
                "linear_acceleration": _vec(accel_imu),
                "linear_acceleration_covariance": [0.0] * 9,
            }
            bus.publish(TOPIC_IMU_RAW, imu_raw, TYPE_IMU, node=NODE_DRIVER)
            bus.publish(TOPIC_MAG_RAW, {"header": _stamp(seq("mag"), frames[FRAME_IMU], stamp),
                                        "magnetic_field": _vec(mag_imu),
                                        "magnetic_field_covariance": [0.0] * 9},
                        TYPE_MAG, node=NODE_DRIVER)
            bus.publish(TOPIC_VOLTAGE, {"data": VOLTAGE}, TYPE_FLOAT32, node=NODE_DRIVER)
            bus.publish(TOPIC_EDITION, {"data": EDITION}, TYPE_FLOAT32, node=NODE_DRIVER)
            with lock:
                commanded = list(arm.commanded)
            positions = joint_positions(commanded)
            bus.publish(TOPIC_JOINT_STATES, {
                "header": _stamp(seq("joint_states"), FRAME_JOINT_STATES, stamp),
                "name": list(JOINTS), "position": positions, "velocity": [], "effort": [],
            }, TYPE_JOINT_STATE, node=NODE_DRIVER)
            # robot_state_publisher, on that joint state (mimic joints followed).
            named = dict(zip(JOINTS, positions))
            for joint, driver, mult, offset in mimics:
                if driver in named:
                    named[joint] = offset + mult * named[driver]
            moving = [(frame(p), frame(c), pos, q) for p, c, pos, q in tree.moving(named)]
            publish_tf(moving, stamp, NODE_RSP, "rsp")
            # odometry_publisher, on each /vel_raw: the calibrated velocity, integrated.
            bus.publish(TOPIC_ODOM_RAW, odometry(seq("odom_raw"), px, py, pyaw, bvx, bvy, rate,
                                                 frame_id=frames[FRAME_ODOM],
                                                 child_frame_id=frames[FRAME_BASE],
                                                 stamp_s=stamp), TYPE_ODOM, node=NODE_ODOMETRY)
            # imu_filter_madgwick, on each raw IMU message: orientation from the start.
            q_world = _quat_mul(_yaw_quat(pyaw), imu_mount)
            imu = dict(imu_raw, header=_stamp(seq("imu"), frames[FRAME_IMU], stamp),
                       orientation=_quat_msg(q_world),
                       orientation_covariance=[0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0,
                                               0.0025])
            bus.publish(TOPIC_IMU, imu, TYPE_IMU, node=NODE_MADGWICK)

        # -- ekf_localization at its 20 Hz ------------------------------------------------------
        if clocks["ekf"].due(now):
            bvx, bvy, rate = odom["v"]
            bus.publish(TOPIC_ODOM, odometry(seq("odom"), px, py, pyaw, bvx, bvy, rate,
                                             frame_id=frames[FRAME_ODOM],
                                             child_frame_id=frames[FRAME_BASE], stamp_s=stamp),
                        TYPE_ODOM, node=NODE_EKF)
            publish_tf([(frames[FRAME_ODOM], frames[FRAME_BASE], (px, py, 0.0),
                         _yaw_quat(pyaw))], stamp, NODE_EKF, "ekf")
        if clocks["diagnostics"].due(now):
            bus.publish(TOPIC_DIAGNOSTICS, {
                "header": _stamp(seq("diag"), "", stamp),
                "status": [{"level": 0, "name": "ekf_localization: Filter diagnostic updater",
                            "message": "", "hardware_id": "none", "values": []}],
            }, TYPE_DIAGNOSTICS, node=NODE_EKF)

        # -- the tf1 static publisher, and the camera node's transforms -----------------------
        if clocks["laser_tf"].due(now):
            publish_tf([laser_tf], stamp, NODE_LASER_TF, "laser_tf")
        if clocks["camera_tf"].due(now):
            publish_tf(camera_dynamic, stamp, NODE_CAMERA, "camera_tf")

        # -- ydlidar_lidar_publisher ------------------------------------------------------------
        if clocks["scan"].due(now) and state["scanning"] and model is not None:
            origin = np.array(data.xpos[laser_body])
            beams = SCAN_BEAMS
            raw = laser_scan_ranges(ray_model, data, origin, yaw + LASER_YAW, beams,
                                    SCAN_RANGE_MAX, bodyexclude=lidar_body,
                                    angle_min=SCAN_ANGLE_MIN,
                                    angle_max=SCAN_ANGLE_MIN + beams * SCAN_INCREMENT,
                                    geomgroup=ray_groups)
            ranges = [float(r) if SCAN_RANGE_MIN <= r <= SCAN_RANGE_MAX else SCAN_INVALID
                      for r in raw]
            n = seq("scan")
            header = _stamp(n, frames[FRAME_LASER], stamp)
            bus.publish(TOPIC_SCAN, {
                "header": header, "angle_min": SCAN_ANGLE_MIN, "angle_max": SCAN_ANGLE_MAX,
                "angle_increment": SCAN_INCREMENT,
                "time_increment": scan_time / SCAN_POINTS_PER_TURN, "scan_time": scan_time,
                "range_min": SCAN_RANGE_MIN, "range_max": SCAN_RANGE_MAX,
                "ranges": ranges, "intensities": [0.0] * beams,
            }, TYPE_LASER_SCAN, node=NODE_LIDAR)
            points, stamps = [], []
            for i, (b, r) in enumerate(zip(scan_bearings(beams), ranges)):
                if SCAN_RANGE_MIN <= r <= SCAN_RANGE_MAX:
                    points.append({"x": r * math.cos(b), "y": r * math.sin(b), "z": 0.0})
                    stamps.append(i * scan_time / SCAN_POINTS_PER_TURN)
            bus.publish(TOPIC_POINT_CLOUD, {
                "header": header, "points": points,
                "channels": [{"name": "intensities", "values": [0.0] * len(points)},
                             {"name": "stamps", "values": stamps}],
            }, TYPE_POINT_CLOUD, node=NODE_LIDAR)

        # -- the Astra ------------------------------------------------------------------------------
        if clocks["camera"].ready(now) and publish_camera(data, stamp):
            clocks["camera"].take(now)

    step.rate_hz = LOOP_HZ
    print(f"ROSMASTER X3 PLUS under namespace {bus.ns or '<bare>'}, stepped at {LOOP_HZ:g} Hz",
          file=sys.stderr)
    return step


class _Encoder:
    """One stream's encoding and publishing, on a thread of its own, in submission order.

    The encoders spend their time in numpy (the clouds, `b64text`), which releases the
    GIL, so several run side by side and beside the physics. `backlog` is how many frames
    wait; the camera clock keeps a frame due rather than letting it grow.
    """

    def __init__(self, name: str) -> None:
        import queue
        import threading

        self._queue: "queue.Queue" = queue.Queue()
        self.backlog = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def run(self, job) -> None:
        self.backlog += 1
        self._queue.put(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                job()
            except Exception as exc:  # a failed frame must not stop the stream
                print(f"x3 encoder: {exc!r}", file=sys.stderr)
            finally:
                self.backlog -= 1

    def close(self) -> None:
        self._queue.put(None)


def _mimics(urdf_text: str) -> list[tuple[str, str, float, float]]:
    import xml.etree.ElementTree as ET

    out = []
    for joint in ET.fromstring(urdf_text).findall("joint"):
        m = joint.find("mimic")
        if m is not None:
            out.append((joint.get("name"), m.get("joint"), float(m.get("multiplier", "1")),
                        float(m.get("offset", "0"))))
    return out


def _quat_matrix(q):
    import numpy as np

    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rotz(a: float):
    import numpy as np

    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)
