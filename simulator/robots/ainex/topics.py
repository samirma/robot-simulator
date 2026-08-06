"""The AiNex's ROS contract: topic names and type strings, in one place.

Transcribed from the vendor's own nodes -- `ainex_kinematics/scripts/ainex_controller.py`
registers every walking topic and service, and `ros_robot_controller_node.py` the bus
servo one. Same role `robot_console/topics.py` plays for the myAGV.

**There is no `/cmd_vel`, no `/odom` and no `/tf` here, and that is not an omission.** A
sweep of every `.py` and `.launch` in `Hiwonder/ainex` finds none of the three: the AiNex
is driven as a state machine (`/walking/command` plus a parameter block), not by a Twist,
and it publishes no wheel odometry because it has no wheels. Anyone arriving from the
myAGV will expect all three; they are absent on the real robot, so they are absent here.

Departures from the hardware are marked below and listed in `robots/README.md`.
"""

from __future__ import annotations

# --- console -> robot ------------------------------------------------------------------

TOPIC_SET_WALKING_PARAM = "/walking/set_param"
TOPIC_APP_WALKING_PARAM = "/app/set_walking_param"
TOPIC_APP_ACTION = "/app/set_action"
TOPIC_BUS_SERVO_SET = "/ros_robot_controller/bus_servo/set_position"
TOPIC_HEAD_PAN = "/head_pan_controller/command"
TOPIC_HEAD_TILT = "/head_tilt_controller/command"

# --- robot -> console ------------------------------------------------------------------

TOPIC_IS_WALKING = "/walking/is_walking"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_IMU = "/imu"
# `usb_cam` publishes /camera/image_raw and image_transport adds this companion topic
# alongside it, so this name is the hardware's. Raw would be ~920 KB a frame base64'd
# through a JSON websocket, which is why only the compressed one is served.
TOPIC_CAMERA = "/camera/image_raw/compressed"
# DEPARTURE: the real AiNex has no lidar of any kind. See robots/README.md.
TOPIC_SCAN = "/scan"

# --- services --------------------------------------------------------------------------

SRV_WALKING_COMMAND = "/walking/command"
SRV_GET_WALKING_PARAM = "/walking/get_param"
SRV_IS_WALKING = "/walking/is_walking"
SRV_INIT_POSE = "/walking/init_pose"

# --- type strings ----------------------------------------------------------------------
#
# ROS1 single-slash form, matching the rest of this simulator. The `ainex_interfaces` and
# `ros_robot_controller` packages are the vendor's own.

TYPE_WALKING_PARAM = "ainex_interfaces/WalkingParam"
TYPE_APP_WALKING_PARAM = "ainex_interfaces/AppWalkingParam"
TYPE_HEAD_STATE = "ainex_interfaces/HeadState"
TYPE_SET_BUS_SERVOS_POSITION = "ros_robot_controller/SetBusServosPosition"
TYPE_STRING = "std_msgs/String"
TYPE_BOOL = "std_msgs/Bool"
TYPE_JOINT_STATE = "sensor_msgs/JointState"
TYPE_IMU = "sensor_msgs/Imu"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
TYPE_LASER_SCAN = "sensor_msgs/LaserScan"

SRV_TYPE_SET_WALKING_COMMAND = "ainex_interfaces/SetWalkingCommand"
SRV_TYPE_GET_WALKING_PARAM = "ainex_interfaces/GetWalkingParam"
SRV_TYPE_GET_WALKING_STATE = "ainex_interfaces/GetWalkingState"
SRV_TYPE_EMPTY = "std_srvs/Empty"

# --- frames ----------------------------------------------------------------------------

FRAME_BASE = "base_link"
FRAME_IMU = "imu_link"
FRAME_CAMERA = "camera_link"
# DEPARTURE: invented along with the virtual lidar itself. The myAGV's `laser_frame` name
# is the YDLidar driver's; nothing on the AiNex names a laser frame, so this follows the
# same convention rather than inventing a second one.
FRAME_LASER = "laser_frame"

# The six strings `/walking/command` accepts, from ainex_controller.py's
# `walking_command_callback`. `enable`/`disable` gate the gait engine; `start`/`stop` run
# it; `enable_control`/`disable_control` gate whether the robot considers itself
# initialised at all, which is a separate axis the vendor's app layer uses.
WALKING_COMMANDS = (
    "enable", "disable", "start", "stop", "enable_control", "disable_control",
)
