"""The AiNex's ROS contract, in one place -- the console's copy of what it uses.

The sibling of `topics.py`, which is the myAGV's. Two robots, two vendors, two entirely
unrelated topic sets: the myAGV takes a `geometry_msgs/Twist` and reports wheel odometry,
while a Hiwonder AiNex is commanded as a **walking state machine** and has no wheels to
report. There is deliberately no `/cmd_vel` and no `/odom` here, and a check that expected
them of this robot would be asking a biped to be a Mecanum base.

The authority is `robots_specs/ainex/ros.yml`, which the simulator transcribes in
`simulator/shared/ros_surfaces/ainex/topics.py`. This copy holds only the facts the
console consumes, duplicated rather than shared because this project must install and
run with no simulator checkout at all; the workspace parity tests (`tests/test_contract_parity.py` at the workspace root)
hold them equal.

Bare, like every other constant here: the namespace is applied where a name reaches the
wire (`namespaced()` in `topics.py`), because these are the record of what a *single*
robot's vendor stack presents.
"""

from __future__ import annotations

# --- what it accepts -----------------------------------------------------------------
#: The walking parameter block, and the action-group trigger.
TOPIC_SET_WALKING_PARAM = "/walking/set_param"
TOPIC_APP_ACTION = "/app/set_action"

#: The head's two controllers, `ainex_interfaces/HeadState` (`position` rad, `duration`
#: s), which the real `ainex_controller` sends to servos 23 and 24. The only per-joint
#: command topics the boot chain has: the other 22 exist only in the vendor's Gazebo
#: bringup.
TOPIC_HEAD_PAN = "/head_pan_controller/command"
TOPIC_HEAD_TILT = "/head_tilt_controller/command"

#: How far the head turns each way, radians. The **servo's** range, not a comfortable
#: viewing range: `joint_limits` in the simulator's `servos.py` takes the tighter of the
#: servo's 0..1000 counts and the URDF's uniform +/-2.09, and for these two -- `init` 500,
#: so symmetric -- the URDF wins on both sides. Held equal to that by the workspace
#: parity tests; the console clamps to it so a held arrow stops asking
#: for angles the robot will silently clamp anyway.
HEAD_PAN_LIMIT = 2.09
HEAD_TILT_LIMIT = 2.09

# --- what it reports -----------------------------------------------------------------
#: `std_msgs/Bool`, published on transitions only -- ask the service below for the state.
TOPIC_IS_WALKING = "/walking/is_walking"
#: The complementary filter's output: the only attitude (and so the only yaw) the robot
#: reports. It has no `/odom` and no `/tf`.
TOPIC_IMU = "/imu"
#: image_transport's compressed companion of usb_cam's `/camera/image_raw`.
TOPIC_CAMERA = "/camera/image_raw/compressed"

# --- services ------------------------------------------------------------------------
#: Walking is not a topic on this robot: the parameter block says *how* to walk and this
#: service says *whether* to.
SRV_WALKING_COMMAND = "/walking/command"
#: `state` true while the gait is moving.
SRV_IS_WALKING = "/walking/is_walking"
#: Servo positions (0-1000 counts) by id: there is no `/joint_states` on this robot.
SRV_BUS_SERVO_GET = "/ros_robot_controller/bus_servo/get_position"

#: The six strings `/walking/command` acts on, from `ainex_controller.py`'s
#: `walking_command_callback`. Two axes, not one: `enable`/`disable` gate the gait engine,
#: `start`/`stop` run it, and `enable_control`/`disable_control` gate whether the others do
#: anything at all -- every call answers `result: true` either way, so a command sent
#: while control is disabled is accepted and ignored, silently.
WALKING_COMMANDS: tuple[str, ...] = (
    "enable", "disable", "start", "stop", "enable_control", "disable_control",
)

#: ros.yml `joints`: servo ids 1..24 in this order.
JOINT_NAMES: tuple[str, ...] = (
    "l_ank_roll", "r_ank_roll", "l_ank_pitch", "r_ank_pitch", "l_knee", "r_knee",
    "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll", "l_hip_yaw", "r_hip_yaw",
    "l_sho_pitch", "r_sho_pitch", "l_sho_roll", "r_sho_roll", "l_el_pitch", "r_el_pitch",
    "l_el_yaw", "r_el_yaw", "l_gripper", "r_gripper", "head_pan", "head_tilt",
)


def servo_id(joint: str) -> int:
    """The bus servo id of a joint: its position in `JOINT_NAMES`, from 1."""
    return JOINT_NAMES.index(joint) + 1


#: `ainex_controller.py`'s pulse<->radian scale: 1000 counts over the servo's 240 degrees.
SERVO_TICKS_PER_RADIAN = 180.0 / 3.141592653589793 / 240.0 * 1000.0
#: The count the head servos read at zero radians (`servo_controller.yaml` `init`).
HEAD_SERVO_CENTRE = 500

# --- message and service types ---------------------------------------------------------
# ROS 1 single-slash strings: the AiNex's vendor stack is ROS 1, like the myAGV's and
# unlike the SO-101's. `ainex_interfaces` and `ros_robot_controller` are the vendor's own.
TYPE_WALKING_PARAM = "ainex_interfaces/WalkingParam"
TYPE_HEAD_STATE = "ainex_interfaces/HeadState"
TYPE_STRING = "std_msgs/String"
TYPE_BOOL = "std_msgs/Bool"
TYPE_IMU = "sensor_msgs/Imu"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
SRV_TYPE_SET_WALKING_COMMAND = "ainex_interfaces/SetWalkingCommand"
SRV_TYPE_GET_WALKING_STATE = "ainex_interfaces/GetWalkingState"
SRV_TYPE_GET_BUS_SERVOS_POSITION = "ros_robot_controller/GetBusServosPosition"

#: What a fleet check requires of an AiNex: the command topics a client drives it with
#: (the gait block, the action trigger and the two head controllers), the walking state,
#: the attitude and the camera on its head.
#:
#: Nothing the boot chain does not present: no `/joint_states`, no `/tf`, no `/scan`. The
#: vendor's own `display.launch` and Gazebo bringup run `robot_state_publisher`, but the
#: shipped robot's boot chain (`start_app_node.service` -> `bringup.launch`) does not, and
#: it is the boot chain a client meets over rosbridge.
CONTRACT_TOPICS: tuple[str, ...] = (
    TOPIC_SET_WALKING_PARAM,
    TOPIC_APP_ACTION,
    TOPIC_HEAD_PAN,
    TOPIC_HEAD_TILT,
    TOPIC_IS_WALKING,
    TOPIC_IMU,
    TOPIC_CAMERA,
)
