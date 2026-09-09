"""The AiNex's ROS contract: topic names and type strings, in one place.

Transcribed from the vendor's own nodes -- `ainex_kinematics/scripts/ainex_controller.py`
registers every walking topic and service, and `ros_robot_controller_node.py` the bus
servo one. Same role `robot_console/topics.py` plays for the myAGV.

**There is no `/cmd_vel` and no `/odom` here, and that is not an omission.** A sweep of
every `.py` and `.launch` in `Hiwonder/ainex` finds neither: the AiNex is driven as a
state machine (`/walking/command` plus a parameter block), not by a Twist, and it
publishes no wheel odometry because it has no wheels. Anyone arriving from the myAGV will
expect both; they are absent on the real robot, so they are absent here.

**`/tf` and `robot_description` are a narrower departure than they look, and the
distinction is the vendor's own.** An exhaustive sweep of both `Hiwonder/ainex` (485
paths) and `UruBots/ainex-robot-code` (1299) finds `robot_state_publisher` in three
launch files -- `ainex_description/launch/display.launch`,
`ainex_gazebo/launch/position_controller.launch`, and `ainex_peripherals/launch/imu.launch`
inside a `debug` group that defaults false -- each loading `robot_description` from the
same `ainex.urdf.xacro` this simulator vendors. What the *physical* robot runs is
`ainex_bringup/service/start_app_node.service` -> `bringup.launch`, and no node in that
include closure broadcasts a transform or sets that parameter.

So a simulated AiNex with a tree matches the vendor's **own** Gazebo and RViz bringups,
off the vendor's own description; it differs from what the shipped robot exposes over
rosbridge at boot. Those are two claims and this file used to conflate them, asserting
"no `robot_state_publisher` and no `/tf` anywhere", which is false. `/tf_static` is
absent here as it is on the real robot: nothing in either repo names it, and the ROS 1
`tf` static publisher writes to `/tf`.

Two of these frames were already in this contract and named nothing: `/imu` stamps
`imu_link` and the camera stamps `camera_link`, and until the tree existed no client could
place either against the body.

Departures from the hardware are marked below and listed in `robots/README.md`.
"""

from __future__ import annotations

from pathlib import Path

# --- console -> robot ------------------------------------------------------------------

TOPIC_SET_WALKING_PARAM = "/walking/set_param"
TOPIC_APP_WALKING_PARAM = "/app/set_walking_param"
TOPIC_APP_ACTION = "/app/set_action"
TOPIC_BUS_SERVO_SET = "/ros_robot_controller/bus_servo/set_position"
TOPIC_HEAD_PAN = "/head_pan_controller/command"
TOPIC_HEAD_TILT = "/head_tilt_controller/command"

#: The 24 joints in servo-id order -- the order `/joint_states` lists them and the order
#: the vendor's `transmissions.xacro` declares them. Held here as a literal rather than
#: imported from `servos.py`, because the console keeps a copy of this table and its
#: contract test compares the two by name; a derived tuple would have nothing to compare.
JOINT_NAMES: tuple[str, ...] = (
    "l_ank_roll", "r_ank_roll", "l_ank_pitch", "r_ank_pitch", "l_knee", "r_knee",
    "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll", "l_hip_yaw", "r_hip_yaw",
    "l_sho_pitch", "r_sho_pitch", "l_sho_roll", "r_sho_roll", "l_el_pitch", "r_el_pitch",
    "l_el_yaw", "r_el_yaw", "l_gripper", "r_gripper", "head_pan", "head_tilt",
)
#: One position-command topic per joint, `/<joint>_controller/command`: the vendor's own
#: ros_control layout, from `ainex_gazebo/config/position_controller.yaml`, which declares
#: an `effort_controllers/JointPositionController` under exactly that name for each of the
#: 24. The head pair above are two of these; the other 22 are what lets a client drive an
#: individual arm or hand joint rather than the whole body through a bus-servo write.
JOINT_COMMAND_TOPICS: dict[str, str] = {j: f"/{j}_controller/command" for j in JOINT_NAMES}

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
#: `ros_robot_controller`'s read-back of the servo bus, in raw counts by id -- the
#: other half of `bus_servo/set_position`, and what the vendor's action-group editor
#: reads a pose off the robot with.
SRV_BUS_SERVO_GET = "/ros_robot_controller/bus_servo/get_position"

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
TYPE_FLOAT64 = "std_msgs/Float64"
TYPE_JOINT_STATE = "sensor_msgs/JointState"
TYPE_IMU = "sensor_msgs/Imu"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
TYPE_LASER_SCAN = "sensor_msgs/LaserScan"

SRV_TYPE_SET_WALKING_COMMAND = "ainex_interfaces/SetWalkingCommand"
SRV_TYPE_GET_WALKING_PARAM = "ainex_interfaces/GetWalkingParam"
SRV_TYPE_GET_WALKING_STATE = "ainex_interfaces/GetWalkingState"
SRV_TYPE_EMPTY = "std_srvs/Empty"
SRV_TYPE_GET_BUS_SERVOS_POSITION = "ros_robot_controller/GetBusServosPosition"


def joint_command_type(joint: str) -> str:
    """The message type a joint's `_controller/command` topic carries.

    Two dialects on one robot, both the vendor's. The head pair are driven through
    `ainex_interfaces/HeadState` (`{position, duration}`) by the real controller; every
    other joint's controller is the Gazebo `JointPositionController`, which takes a plain
    `std_msgs/Float64` (`{data}`). The surface accepts either shape on every topic, so a
    client that gets this wrong still moves the joint; the declared type is what
    `/rosapi/topics` reports, and it reports what the vendor's stack would.
    """
    return TYPE_HEAD_STATE if joint in ("head_pan", "head_tilt") else TYPE_FLOAT64


# --- frames ----------------------------------------------------------------------------

FRAME_BASE = "base_link"
FRAME_IMU = "imu_link"
FRAME_CAMERA = "camera_link"
#: The torso, which is where this robot's kinematic tree actually roots: the vendor's
#: `base_link` is jointless and MuJoCo merges it away, so `body_link` is the body that
#: exists and `base_link -> body_link` comes back from the description as a fixed joint.
TF_ROOT_BODY = "body_link"
TF_ROOT_FRAME = TF_ROOT_BODY
# DEPARTURE: invented along with the virtual lidar itself. The myAGV's `laser_frame` name
# is the YDLidar driver's; nothing on the AiNex names a laser frame, so this follows the
# same convention rather than inventing a second one.
FRAME_LASER = "laser_frame"

#: The vendor description, and the frames read off it. `ainex_description`'s URDF is what
#: `robot_description` carries; every link name below is the description's own, and the
#: compiled model uses the same names because the model is built from that file.
URDF_PATH = Path(__file__).resolve().parents[2] / "robots/ainex/urdf/ainex.urdf"
#: The 24 moving links, one per servo, plus the torso they hang off. Derived from
#: `JOINT_NAMES` rather than typed again: the vendor names each link after its joint, so
#: a link table typed by hand would be 24 more chances to disagree with the joint table
#: two dozen lines up -- and a frame that disagrees is a body drawn in the wrong place.
TF_FRAMES: dict[str, str] = {TF_ROOT_BODY: TF_ROOT_FRAME,
                             **{f"{j}_link": f"{j}_link" for j in JOINT_NAMES}}
#: The head camera, under the frame its images are already stamped with.
TF_CAMERAS: dict[str, str] = {"front_camera": FRAME_CAMERA}

# The six strings `/walking/command` accepts, from ainex_controller.py's
# `walking_command_callback`. `enable`/`disable` gate the gait engine; `start`/`stop` run
# it; `enable_control`/`disable_control` gate whether the robot considers itself
# initialised at all, which is a separate axis the vendor's app layer uses.
WALKING_COMMANDS = (
    "enable", "disable", "start", "stop", "enable_control", "disable_control",
)
