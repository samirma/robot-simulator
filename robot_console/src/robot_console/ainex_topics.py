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

# --- what it reports -----------------------------------------------------------------
TOPIC_IS_WALKING = "/walking/is_walking"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_IMU = "/imu"
TOPIC_CAMERA = "/camera/image_raw/compressed"

#: What a fleet check requires of an AiNex: the two command topics a client drives it
#: with, the two state topics it reports through, and the camera on its head.
#:
#: `/scan` is deliberately **not** here. The simulator publishes one, and says in its own
#: constants that both the topic and the lidar behind it are invented -- the real AiNex
#: has none. Requiring it would make this file assert a simulator's departure from the
#: hardware as though it were the hardware, which is the one thing these constants exist
#: not to do.
CONTRACT_TOPICS: tuple[str, ...] = (
    TOPIC_SET_WALKING_PARAM,
    TOPIC_APP_ACTION,
    TOPIC_IS_WALKING,
    TOPIC_JOINT_STATES,
    TOPIC_IMU,
    TOPIC_CAMERA,
)
