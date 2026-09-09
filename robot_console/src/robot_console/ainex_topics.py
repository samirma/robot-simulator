"""The AiNex's ROS contract, in one place -- the console's copy.

The sibling of `topics.py`, which is the myAGV's. Two robots, two vendors, two entirely
unrelated topic sets: the myAGV takes a `geometry_msgs/Twist` and reports wheel odometry,
while a Hiwonder AiNex is commanded as a **walking state machine** and has no wheels to
report. There is deliberately no `/cmd_vel` and no `/odom` here, and a check that expected
them of this robot would be asking a biped to be a Mecanum base.

These names mirror `Hiwonder/ainex` -- `ainex_kinematics/scripts/ainex_controller.py`
registers the walking topics and `ros_robot_controller_node.py` the bus servo one -- and
the simulator holds its own copy in `simulator/shared/ros_surfaces/ainex/topics.py`. The
two are duplicated rather than shared because this project must install and run with no
simulator checkout at all; `tests/arm/test_ros_contract.py` is what holds them equal.

Bare, like every other constant here: the namespace is applied where a name reaches the
wire (`namespaced()` in `topics.py`), because these are the record of what a *single*
robot's vendor stack presents.
"""

from __future__ import annotations

# --- what it accepts -----------------------------------------------------------------
#: The walking parameter block and the state machine's command topic. `/app/*` are the
#: vendor's own app-facing aliases, which the real controller registers alongside.
TOPIC_SET_WALKING_PARAM = "/walking/set_param"
TOPIC_APP_ACTION = "/app/set_action"
TOPIC_BUS_SERVO_SET = "/ros_robot_controller/bus_servo/set_position"

#: The 24 joints in servo-id order, and one position-command topic per joint --
#: `/<joint>_controller/command`, the vendor's own ros_control layout from
#: `ainex_gazebo/config/position_controller.yaml`. A literal, matching the simulator's
#: `topics.py` name for name; `tests/test_ainex_contract.py` holds the two equal.
JOINT_NAMES: tuple[str, ...] = (
    "l_ank_roll", "r_ank_roll", "l_ank_pitch", "r_ank_pitch", "l_knee", "r_knee",
    "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll", "l_hip_yaw", "r_hip_yaw",
    "l_sho_pitch", "r_sho_pitch", "l_sho_roll", "r_sho_roll", "l_el_pitch", "r_el_pitch",
    "l_el_yaw", "r_el_yaw", "l_gripper", "r_gripper", "head_pan", "head_tilt",
)
JOINT_COMMAND_TOPICS: tuple[str, ...] = tuple(f"/{j}_controller/command" for j in JOINT_NAMES)

#: The two of those the teleop arrows drive, named rather than indexed -- derived from the
#: table above so a rename cannot leave them pointing at a topic that is not there.
TOPIC_HEAD_PAN = f"/{JOINT_NAMES[JOINT_NAMES.index('head_pan')]}_controller/command"
TOPIC_HEAD_TILT = f"/{JOINT_NAMES[JOINT_NAMES.index('head_tilt')]}_controller/command"

#: How far the head turns each way, radians. The **servo's** range, not a comfortable
#: viewing range: `joint_limits` in the simulator's `servos.py` takes the tighter of the
#: servo's 0..1000 counts and the URDF's uniform +/-2.09, and for these two -- `init` 500,
#: so symmetric -- the URDF wins on both sides. Held equal to that by
#: `tests/test_ainex_contract.py`; the console clamps to it so a held arrow stops asking
#: for angles the robot will silently clamp anyway.
HEAD_PAN_LIMIT = 2.09
HEAD_TILT_LIMIT = 2.09

# --- what it reports -----------------------------------------------------------------
TOPIC_IS_WALKING = "/walking/is_walking"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_IMU = "/imu"
TOPIC_CAMERA = "/camera/image_raw/compressed"

# --- the gait state machine's service, and the strings it takes ------------------------
#: Walking is not a topic on this robot: the parameter block says *how* to walk and this
#: service says *whether* to. `ainex_link` calls it, so it belongs here with the rest of
#: the contract rather than being re-typed there -- a service name is exactly as easy to
#: get wrong as a topic name and, until this moved, nothing held it against the
#: simulator's copy.
SRV_WALKING_COMMAND = "/walking/command"

#: The six strings that service accepts, from `ainex_controller.py`'s
#: `walking_command_callback`. Two axes, not one: `enable`/`disable` gate the gait engine,
#: `start`/`stop` run it, and `enable_control`/`disable_control` gate whether the robot
#: considers itself initialised at all -- which the vendor's app layer sets first and
#: without which the other four are accepted and ignored, silently.
WALKING_COMMANDS: tuple[str, ...] = (
    "enable", "disable", "start", "stop", "enable_control", "disable_control",
)

# --- message and service types ---------------------------------------------------------
# ROS 1 single-slash strings: the AiNex's vendor stack is ROS 1, like the myAGV's and
# unlike the SO-101's. `ainex_interfaces` is the vendor's own package.
TYPE_WALKING_PARAM = "ainex_interfaces/WalkingParam"
TYPE_HEAD_STATE = "ainex_interfaces/HeadState"
TYPE_BOOL = "std_msgs/Bool"
TYPE_FLOAT64 = "std_msgs/Float64"
TYPE_JOINT_STATE = "sensor_msgs/JointState"
TYPE_IMU = "sensor_msgs/Imu"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
SRV_TYPE_SET_WALKING_COMMAND = "ainex_interfaces/SetWalkingCommand"

#: What a fleet check requires of an AiNex: the two command topics a client drives it
#: with, the two state topics it reports through, and the camera on its head.
#:
#: `/scan` is deliberately **not** here. The simulator publishes one, and says in its own
#: constants that both the topic and the lidar behind it are invented -- the real AiNex
#: has none. Requiring it would make this file assert a simulator's departure from the
#: hardware as though it were the hardware, which is the one thing these constants exist
#: not to do.
#:
#: `/tf` is absent for a narrower reason, worth stating because the myAGV's contract does
#: require it. The vendor's own `ainex_description/launch/display.launch` and
#: `ainex_gazebo/launch/position_controller.launch` both run `robot_state_publisher`, so
#: the AiNex's software does have a tree -- but the shipped robot's boot chain
#: (`start_app_node.service` -> `bringup.launch`) starts neither, and it is the boot chain
#: a client meets over rosbridge. The simulator gives every robot a tree; the myAGV's is
#: on its wire at boot and the AiNex's is not, so only the myAGV's is required here.
CONTRACT_TOPICS: tuple[str, ...] = (
    TOPIC_SET_WALKING_PARAM,
    TOPIC_APP_ACTION,
    TOPIC_IS_WALKING,
    TOPIC_JOINT_STATES,
    TOPIC_IMU,
    TOPIC_CAMERA,
    # Every joint individually commandable: the manufacturer's per-joint controllers
    # are part of what an AiNex presents, not an extra.
    *JOINT_COMMAND_TOPICS,
)
