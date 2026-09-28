"""The ROSMASTER X3 PLUS's ROS contract -- the console's copy of what it uses.

The authority is `robots_specs/rosmaster_x3_plus/ros.yml` (Yahboom's ROS 1 bringup,
`laser_astrapro_bringup.launch` with `ROBOT_TYPE=X3plus`), which the simulator transcribes
in `simulator/shared/ros_surfaces/rosmaster_x3_plus.py`. This copy holds only the facts
the console consumes; the workspace parity tests (`tests/test_contract_parity.py` at the
workspace root) hold them equal.

Like the myAGV it is a Mecanum base driven by a `geometry_msgs/Twist` on `/cmd_vel` with
**no command watchdog** -- the driver hands each command to the STM32 board, which keeps
executing it -- and it reports `/odom` (its EKF's) and a `/scan`. What tells it apart on a
wire is its own driver's arm topic, `/TargetAngle`. Its only colour stream is the Astra's
raw `sensor_msgs/Image`: the bringup publishes no compressed image.
"""

from __future__ import annotations

TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOM = "/odom"
TOPIC_SCAN = "/scan"
#: The Astra Pro Plus's colour image (`color/image_raw` remapped), rgb8, 640x480.
TOPIC_CAMERA = "/camera/rgb/image_raw"
#: The driver's arm command: what identifies this robot among `/cmd_vel` bases.
TOPIC_TARGET_ANGLE = "/TargetAngle"

TYPE_TWIST = "geometry_msgs/Twist"
TYPE_ODOM = "nav_msgs/Odometry"
TYPE_LASER_SCAN = "sensor_msgs/LaserScan"
TYPE_IMAGE = "sensor_msgs/Image"
TYPE_ARM_JOINT = "yahboomcar_msgs/ArmJoint"

#: `set_car_motion`'s X3PLUS input range: v_x, v_y (m/s) and v_z (rad/s).
CMD_VEL_LIMITS = (0.7, 0.7, 3.2)

#: The teleop envelope: Yahboom's own keyboard teleop starts at 0.2 m/s and 1.0 rad/s (a
#: turn ratio of 5), and its bringup sets the X3PLUS limits the board takes, 0.7 m/s and
#: 3.2 rad/s, as the cap.
SPEED_MIN = 0.05
SPEED_MAX = CMD_VEL_LIMITS[0]
SPEED_STEP = 0.05
SPEED_DEFAULT = 0.2
TURN_RATIO = 5.0
TURN_MAX = CMD_VEL_LIMITS[2]

#: What discovery needs of a wire to call a namespace an X3 PLUS: the signature and these.
CONTRACT_TOPICS: dict[str, str] = {
    TOPIC_CMD_VEL: TYPE_TWIST,
    TOPIC_ODOM: TYPE_ODOM,
    TOPIC_SCAN: TYPE_LASER_SCAN,
    TOPIC_CAMERA: TYPE_IMAGE,
    TOPIC_TARGET_ANGLE: TYPE_ARM_JOINT,
}
