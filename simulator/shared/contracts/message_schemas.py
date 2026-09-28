"""The message and service definitions the bridge answers `rosapi` schema queries from.

A real rosbridge ships `rosapi`, and `rosapi` answers `/rosapi/message_details` by reading
the `.msg` files installed on the robot. There is no ROS install here and nothing to read,
so the definitions live in this table -- **transcribed, not authored**. Every entry is
copied from the manufacturer's own definition file for that embodiment, or from the ROS
distribution's for the standard packages, and the block below records where each came
from. That is the same standing the vendor STLs have (`shared/robots/ainex/
PROVENANCE.md`): a definition that is not verbatim would be indistinguishable from the
vendor's afterwards, and a client generating code against it would be generating it
against a guess.

Two things the table has to get right that a naive one would not:

* **Both dialects resolve to one definition -- except where ROS 2 changed it.**
  `sensor_msgs/Imu` and `sensor_msgs/msg/Imu` are the same message; this graph
  deliberately carries ROS 1 names for the myAGV and the AiNex beside ROS 2 names for the
  SO-101, and a client may ask in either spelling. A handful of types are *not* the same
  message in the two distributions -- `std_msgs/Header` lost `seq` and stamps with
  `builtin_interfaces/Time` (`sec`/`nanosec`), `sensor_msgs/CameraInfo` lower-cased
  `D K R P` to `d k r p`, `JointTrajectoryPoint.time_from_start` became a
  `builtin_interfaces/Duration` -- and those have a second, ROS 2 entry in
  `ROS2_MESSAGES`. The dialect of a query is read off the name it was asked in: a
  `pkg/msg/Type` spelling, or a package this table holds only in ROS 2 form
  (`_ROS2_PACKAGES`), gets the ROS 2 definitions for itself and for everything it nests;
  a `pkg/Type` spelling gets ROS 1's, so the myAGV's `sensor_msgs/CameraInfo` keeps `K`.
* **`typedefs()` returns the transitive closure**, as `rosapi` does. Asking for
  `sensor_msgs/Imu` also returns `std_msgs/Header`, `geometry_msgs/Quaternion` and
  `geometry_msgs/Vector3`, in `rosapi`'s own `TypeDef` shape, or a tool walking the schema
  dead-ends at the first non-primitive field -- and Imu, Odometry and LaserScan are mostly
  non-primitive fields.

`fieldarraylen` follows `rosapi`: -1 for a scalar, 0 for a variable-length array, N for a
fixed one. Pure Python, no imports: `contracts/` must stay free of MuJoCo and of ROS.

PROVENANCE
----------
AiNex (Hiwonder) -- `UruBots/ainex-robot-code`, a real AiNex deployment, at
`ros_ws_src/ainex_interfaces/{msg,srv}` and
`ros_ws_src/ainex_driver/ros_robot_controller/{msg,srv}`:
    HeadState.msg, WalkingParam.msg, AppWalkingParam.msg, SetWalkingCommand.srv,
    GetWalkingParam.srv, GetWalkingState.srv, SetBusServosPosition.msg,
    BusServoPosition.msg, GetBusServosPosition.srv
    ...and the rest of its boot interface from `Hiwonder/ainex` at the revision
    `robots_specs/robots.yml` records, listed where they are defined (the `AiNex: the rest
    of its boot interface` block below).
myAGV (Elephant Robotics) -- `elephantrobotics/myagv_ros` at c71f3cc5 (branch
    `myagv_ros_2023Pi`): standard messages, plus the vendored `robot_pose_ekf`'s
    `srv/GetStatus.srv`. Its standard types not used by another member (std_msgs/Float32,
    geometry_msgs/Point32, sensor_msgs/ChannelFloat32, sensor_msgs/PointCloud) are in
    the `# --- myAGV` block; sensor_msgs/SetCameraInfo is the SO-101 block's, one
    definition that nests the CameraInfo of whichever dialect it is asked in.
SO-101 (ros2_control bringup, ROS 2 Jazzy) -- every interface its graph uses, from the
    upstream files at the revisions Jazzy's rosdistro releases (`jazzy/distribution.yaml`):
    `ros-controls/control_msgs` tag `5.10.0` (the `jazzy` branch head, and the release
    `ros2_controllers` 4.42.1 builds against), `control_msgs/`:
        msg/{JointTrajectoryControllerState, DynamicJointState, InterfaceValue,
        SpeedScalingFactor, JointTolerance, JointComponentTolerance}.msg,
        srv/QueryTrajectoryState.srv,
        action/{FollowJointTrajectory, ParallelGripperCommand}.action
    `ros-controls/ros2_control` tag `4.48.1`, `controller_manager_msgs/`:
        msg/{ControllerManagerActivity, NamedLifecycleState, ControllerState,
        ChainConnection, HardwareComponentState, HardwareInterface}.msg,
        srv/{ListControllers, ListControllerTypes, LoadController, ConfigureController,
        ReloadControllerLibraries, SwitchController, UnloadController, CleanupController,
        ListHardwareComponents, ListHardwareInterfaces, SetHardwareComponentState}.srv
    `pal-robotics/pal_statistics` tag `2.8.2` (Jazzy's release), `pal_statistics_msgs/msg/`:
        Statistic.msg, Statistics.msg, StatisticsNames.msg, StatisticsValues.msg
    `ros2/common_interfaces` tag `5.3.8` (Jazzy's release; byte-identical to the `jazzy`
    branch for every file here): diagnostic_msgs/msg/{DiagnosticArray, DiagnosticStatus,
        KeyValue}, trajectory_msgs/msg/{MultiDOFJointTrajectory,
        MultiDOFJointTrajectoryPoint, JointTrajectoryPoint}, sensor_msgs/msg/CameraInfo,
        sensor_msgs/srv/SetCameraInfo, std_srvs/srv/SetBool, std_msgs/msg/Header
    `ros2/rcl_interfaces` tag `2.0.4` (likewise identical to `jazzy`):
        lifecycle_msgs/msg/State.msg, builtin_interfaces/msg/{Time, Duration}.msg
    Constants are transcribed too, into `CONSTANTS`. The `mujoco_ros2_control_msgs`
    FreeJointState/FreeJointStateArray pair that used to be listed here is gone: the
    SO-101 no longer serves it.
Standard packages -- the ROS distributions' own files, identical between Noetic and
    Humble for every type here: std_msgs, std_srvs, geometry_msgs, nav_msgs, sensor_msgs,
    trajectory_msgs, builtin_interfaces -- except the few `ROS2_MESSAGES` holds twice.
"""

from __future__ import annotations

# rosapi's fieldarraylen convention.
SCALAR = -1
VARIABLE = 0

#: One field: (name, type, arraylen). Types are written ROS 1 style (`pkg/Type`) and
#: resolved through `canonical()`, so the table never needs both spellings.
Field = tuple[str, str, int]

PRIMITIVES = frozenset({
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "float32", "float64", "string", "time", "duration", "byte", "char",
})

#: ROS 1 -> ROS 2 renames for the few types whose *package* differs by dialect, so that
#: `canonical()` names them one way. Schema lookups for these no longer land here: a
#: ROS 2 query finds `builtin_interfaces/Time` in `ROS2_MESSAGES` (`sec`/`nanosec`), which
#: is what Jazzy's file says, rather than ROS 1's `secs`/`nsecs`.
_ALIASES = {
    "builtin_interfaces/Time": "std_msgs/Time",
    "builtin_interfaces/Duration": "std_msgs/Duration",
}


def canonical(type_name: str) -> str:
    """One spelling for a type: `pkg/Type`, with any `/msg/`, `/srv/` or `/action/` removed."""
    parts = type_name.split("/")
    if len(parts) == 3 and parts[1] in ("msg", "srv", "action"):
        parts = [parts[0], parts[2]]
    name = "/".join(parts)
    return _ALIASES.get(name, name)


def _h(*fields: Field) -> list[Field]:
    return list(fields)


# --- standard packages -----------------------------------------------------------------

_STD: dict[str, list[Field]] = {
    # ROS 1 `time`/`duration` are primitives on the wire; ROS 2 spells them as messages.
    # Both are answered from these, under both names (see _ALIASES).
    "std_msgs/Time": _h(("secs", "int32", SCALAR), ("nsecs", "int32", SCALAR)),
    "std_msgs/Duration": _h(("secs", "int32", SCALAR), ("nsecs", "int32", SCALAR)),
    "std_msgs/Header": _h(
        ("seq", "uint32", SCALAR), ("stamp", "time", SCALAR), ("frame_id", "string", SCALAR),
    ),
    "std_msgs/Bool": _h(("data", "bool", SCALAR)),
    "std_msgs/String": _h(("data", "string", SCALAR)),
    "std_msgs/Float64": _h(("data", "float64", SCALAR)),
    "std_msgs/MultiArrayDimension": _h(
        ("label", "string", SCALAR), ("size", "uint32", SCALAR), ("stride", "uint32", SCALAR),
    ),
    "std_msgs/MultiArrayLayout": _h(
        ("dim", "std_msgs/MultiArrayDimension", VARIABLE), ("data_offset", "uint32", SCALAR),
    ),
    "std_msgs/Float64MultiArray": _h(
        ("layout", "std_msgs/MultiArrayLayout", SCALAR), ("data", "float64", VARIABLE),
    ),
    "geometry_msgs/Vector3": _h(
        ("x", "float64", SCALAR), ("y", "float64", SCALAR), ("z", "float64", SCALAR),
    ),
    "geometry_msgs/Point": _h(
        ("x", "float64", SCALAR), ("y", "float64", SCALAR), ("z", "float64", SCALAR),
    ),
    "geometry_msgs/Quaternion": _h(
        ("x", "float64", SCALAR), ("y", "float64", SCALAR), ("z", "float64", SCALAR),
        ("w", "float64", SCALAR),
    ),
    "geometry_msgs/Pose": _h(
        ("position", "geometry_msgs/Point", SCALAR),
        ("orientation", "geometry_msgs/Quaternion", SCALAR),
    ),
    "geometry_msgs/PoseStamped": _h(
        ("header", "std_msgs/Header", SCALAR), ("pose", "geometry_msgs/Pose", SCALAR),
    ),
    "geometry_msgs/Twist": _h(
        ("linear", "geometry_msgs/Vector3", SCALAR), ("angular", "geometry_msgs/Vector3", SCALAR),
    ),
    "geometry_msgs/TwistStamped": _h(
        ("header", "std_msgs/Header", SCALAR), ("twist", "geometry_msgs/Twist", SCALAR),
    ),
    "geometry_msgs/Transform": _h(
        ("translation", "geometry_msgs/Vector3", SCALAR),
        ("rotation", "geometry_msgs/Quaternion", SCALAR),
    ),
    "geometry_msgs/TransformStamped": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("child_frame_id", "string", SCALAR),
        ("transform", "geometry_msgs/Transform", SCALAR),
    ),
    "tf2_msgs/TFMessage": _h(
        ("transforms", "geometry_msgs/TransformStamped", VARIABLE),
    ),
    "geometry_msgs/PoseWithCovariance": _h(
        ("pose", "geometry_msgs/Pose", SCALAR), ("covariance", "float64", 36),
    ),
    "geometry_msgs/TwistWithCovariance": _h(
        ("twist", "geometry_msgs/Twist", SCALAR), ("covariance", "float64", 36),
    ),
    "nav_msgs/Odometry": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("child_frame_id", "string", SCALAR),
        ("pose", "geometry_msgs/PoseWithCovariance", SCALAR),
        ("twist", "geometry_msgs/TwistWithCovariance", SCALAR),
    ),
    "sensor_msgs/CompressedImage": _h(
        ("header", "std_msgs/Header", SCALAR), ("format", "string", SCALAR),
        ("data", "uint8", VARIABLE),
    ),
    "sensor_msgs/Image": _h(
        ("header", "std_msgs/Header", SCALAR), ("height", "uint32", SCALAR),
        ("width", "uint32", SCALAR), ("encoding", "string", SCALAR),
        ("is_bigendian", "uint8", SCALAR), ("step", "uint32", SCALAR),
        ("data", "uint8", VARIABLE),
    ),
    "sensor_msgs/RegionOfInterest": _h(
        ("x_offset", "uint32", SCALAR), ("y_offset", "uint32", SCALAR),
        ("height", "uint32", SCALAR), ("width", "uint32", SCALAR),
        ("do_rectify", "bool", SCALAR),
    ),
    "sensor_msgs/CameraInfo": _h(
        ("header", "std_msgs/Header", SCALAR), ("height", "uint32", SCALAR),
        ("width", "uint32", SCALAR), ("distortion_model", "string", SCALAR),
        ("D", "float64", VARIABLE), ("K", "float64", 9), ("R", "float64", 9),
        ("P", "float64", 12), ("binning_x", "uint32", SCALAR), ("binning_y", "uint32", SCALAR),
        ("roi", "sensor_msgs/RegionOfInterest", SCALAR),
    ),
    "sensor_msgs/LaserScan": _h(
        ("header", "std_msgs/Header", SCALAR), ("angle_min", "float32", SCALAR),
        ("angle_max", "float32", SCALAR), ("angle_increment", "float32", SCALAR),
        ("time_increment", "float32", SCALAR), ("scan_time", "float32", SCALAR),
        ("range_min", "float32", SCALAR), ("range_max", "float32", SCALAR),
        ("ranges", "float32", VARIABLE), ("intensities", "float32", VARIABLE),
    ),
    "sensor_msgs/Imu": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("orientation", "geometry_msgs/Quaternion", SCALAR),
        ("orientation_covariance", "float64", 9),
        ("angular_velocity", "geometry_msgs/Vector3", SCALAR),
        ("angular_velocity_covariance", "float64", 9),
        ("linear_acceleration", "geometry_msgs/Vector3", SCALAR),
        ("linear_acceleration_covariance", "float64", 9),
    ),
    "sensor_msgs/JointState": _h(
        ("header", "std_msgs/Header", SCALAR), ("name", "string", VARIABLE),
        ("position", "float64", VARIABLE), ("velocity", "float64", VARIABLE),
        ("effort", "float64", VARIABLE),
    ),
    "trajectory_msgs/JointTrajectoryPoint": _h(
        ("positions", "float64", VARIABLE), ("velocities", "float64", VARIABLE),
        ("accelerations", "float64", VARIABLE), ("effort", "float64", VARIABLE),
        ("time_from_start", "duration", SCALAR),
    ),
    "trajectory_msgs/JointTrajectory": _h(
        ("header", "std_msgs/Header", SCALAR), ("joint_names", "string", VARIABLE),
        ("points", "trajectory_msgs/JointTrajectoryPoint", VARIABLE),
    ),
}

# --- AiNex: Hiwonder's ainex_interfaces and ros_robot_controller -------------------------

_AINEX: dict[str, list[Field]] = {
    "ainex_interfaces/HeadState": _h(
        ("position", "float64", SCALAR), ("duration", "float64", SCALAR),
    ),
    "ainex_interfaces/WalkingParam": _h(
        ("init_x_offset", "float32", SCALAR), ("init_y_offset", "float32", SCALAR),
        ("init_z_offset", "float32", SCALAR), ("init_roll_offset", "float32", SCALAR),
        ("init_pitch_offset", "float32", SCALAR), ("init_yaw_offset", "float32", SCALAR),
        ("period_time", "float32", SCALAR), ("dsp_ratio", "float32", SCALAR),
        ("step_fb_ratio", "float32", SCALAR), ("period_times", "uint32", SCALAR),
        ("x_move_amplitude", "float32", SCALAR), ("y_move_amplitude", "float32", SCALAR),
        ("z_move_amplitude", "float32", SCALAR), ("angle_move_amplitude", "float32", SCALAR),
        ("move_aim_on", "bool", SCALAR), ("arm_swing_gain", "float32", SCALAR),
        ("y_swap_amplitude", "float32", SCALAR), ("z_swap_amplitude", "float32", SCALAR),
        ("pelvis_offset", "float32", SCALAR), ("hip_pitch_offset", "float32", SCALAR),
        ("balance_enable", "bool", SCALAR), ("balance_hip_roll_gain", "float32", SCALAR),
        ("balance_knee_gain", "float32", SCALAR),
        ("balance_ankle_roll_gain", "float32", SCALAR),
        ("balance_ankle_pitch_gain", "float32", SCALAR),
    ),
    "ainex_interfaces/AppWalkingParam": _h(
        ("speed", "int16", SCALAR), ("height", "float64", SCALAR), ("x", "float64", SCALAR),
        ("y", "float64", SCALAR), ("angle", "float64", SCALAR),
    ),
    "ros_robot_controller/BusServoPosition": _h(
        ("id", "uint16", SCALAR), ("position", "uint16", SCALAR),
    ),
    "ros_robot_controller/SetBusServosPosition": _h(
        ("duration", "float64", SCALAR),
        ("position", "ros_robot_controller/BusServoPosition", VARIABLE),
    ),
}

# --- SO-101 ------------------------------------------------------------------------------
# Everything the SO-101's ROS 2 Jazzy graph uses that the standard block above does not
# already hold, transcribed from the files and revisions named in PROVENANCE. Unqualified
# field types in the upstream files (`JointTolerance[]` inside control_msgs) are written
# out with their package, which is how rosapi reports them.

_SO101: dict[str, list[Field]] = {
    # ros2/common_interfaces 5.3.8
    "trajectory_msgs/MultiDOFJointTrajectoryPoint": _h(
        ("transforms", "geometry_msgs/Transform", VARIABLE),
        ("velocities", "geometry_msgs/Twist", VARIABLE),
        ("accelerations", "geometry_msgs/Twist", VARIABLE),
        ("time_from_start", "builtin_interfaces/Duration", SCALAR),
    ),
    "trajectory_msgs/MultiDOFJointTrajectory": _h(
        ("header", "std_msgs/Header", SCALAR), ("joint_names", "string", VARIABLE),
        ("points", "trajectory_msgs/MultiDOFJointTrajectoryPoint", VARIABLE),
    ),
    "diagnostic_msgs/KeyValue": _h(("key", "string", SCALAR), ("value", "string", SCALAR)),
    "diagnostic_msgs/DiagnosticStatus": _h(
        ("level", "byte", SCALAR), ("name", "string", SCALAR),
        ("message", "string", SCALAR), ("hardware_id", "string", SCALAR),
        ("values", "diagnostic_msgs/KeyValue", VARIABLE),
    ),
    "diagnostic_msgs/DiagnosticArray": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("status", "diagnostic_msgs/DiagnosticStatus", VARIABLE),
    ),
    # ros2/rcl_interfaces 2.0.4
    "lifecycle_msgs/State": _h(("id", "uint8", SCALAR), ("label", "string", SCALAR)),
    # ros-controls/control_msgs 5.10.0
    "control_msgs/JointTolerance": _h(
        ("name", "string", SCALAR), ("position", "float64", SCALAR),
        ("velocity", "float64", SCALAR), ("acceleration", "float64", SCALAR),
    ),
    "control_msgs/JointComponentTolerance": _h(
        ("joint_name", "string", SCALAR), ("component", "uint16", SCALAR),
        ("position", "float64", SCALAR), ("velocity", "float64", SCALAR),
        ("acceleration", "float64", SCALAR),
    ),
    "control_msgs/InterfaceValue": _h(
        ("interface_names", "string", VARIABLE), ("values", "float64", VARIABLE),
    ),
    "control_msgs/DynamicJointState": _h(
        ("header", "std_msgs/Header", SCALAR), ("joint_names", "string", VARIABLE),
        ("interface_values", "control_msgs/InterfaceValue", VARIABLE),
    ),
    "control_msgs/SpeedScalingFactor": _h(("factor", "float64", SCALAR)),
    "control_msgs/JointTrajectoryControllerState": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("joint_names", "string", VARIABLE),
        ("reference", "trajectory_msgs/JointTrajectoryPoint", SCALAR),
        ("feedback", "trajectory_msgs/JointTrajectoryPoint", SCALAR),
        ("error", "trajectory_msgs/JointTrajectoryPoint", SCALAR),
        ("output", "trajectory_msgs/JointTrajectoryPoint", SCALAR),
        ("multi_dof_joint_names", "string", VARIABLE),
        ("multi_dof_reference", "trajectory_msgs/MultiDOFJointTrajectoryPoint", SCALAR),
        ("multi_dof_feedback", "trajectory_msgs/MultiDOFJointTrajectoryPoint", SCALAR),
        ("multi_dof_error", "trajectory_msgs/MultiDOFJointTrajectoryPoint", SCALAR),
        ("multi_dof_output", "trajectory_msgs/MultiDOFJointTrajectoryPoint", SCALAR),
        ("speed_scaling_factor", "float64", SCALAR),
    ),
    # ros-controls/ros2_control 4.48.1
    "controller_manager_msgs/NamedLifecycleState": _h(
        ("name", "string", SCALAR), ("state", "lifecycle_msgs/State", SCALAR),
    ),
    "controller_manager_msgs/ControllerManagerActivity": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("controllers", "controller_manager_msgs/NamedLifecycleState", VARIABLE),
        ("hardware_components", "controller_manager_msgs/NamedLifecycleState", VARIABLE),
    ),
    "controller_manager_msgs/ChainConnection": _h(
        ("name", "string", SCALAR), ("reference_interfaces", "string", VARIABLE),
    ),
    "controller_manager_msgs/ControllerState": _h(
        ("name", "string", SCALAR), ("state", "string", SCALAR), ("type", "string", SCALAR),
        ("is_async", "bool", SCALAR), ("update_rate", "uint16", SCALAR),
        ("claimed_interfaces", "string", VARIABLE),
        ("required_command_interfaces", "string", VARIABLE),
        ("required_state_interfaces", "string", VARIABLE),
        ("is_chainable", "bool", SCALAR), ("is_chained", "bool", SCALAR),
        ("exported_state_interfaces", "string", VARIABLE),
        ("reference_interfaces", "string", VARIABLE),
        ("chain_connections", "controller_manager_msgs/ChainConnection", VARIABLE),
    ),
    "controller_manager_msgs/HardwareInterface": _h(
        ("name", "string", SCALAR), ("data_type", "string", SCALAR),
        ("is_available", "bool", SCALAR), ("is_claimed", "bool", SCALAR),
    ),
    "controller_manager_msgs/HardwareComponentState": _h(
        ("name", "string", SCALAR), ("type", "string", SCALAR),
        ("is_async", "bool", SCALAR), ("rw_rate", "uint16", SCALAR),
        ("class_type", "string", SCALAR), ("plugin_name", "string", SCALAR),
        ("state", "lifecycle_msgs/State", SCALAR),
        ("command_interfaces", "controller_manager_msgs/HardwareInterface", VARIABLE),
        ("state_interfaces", "controller_manager_msgs/HardwareInterface", VARIABLE),
    ),
    # pal-robotics/pal_statistics 2.8.2
    "pal_statistics_msgs/Statistic": _h(("name", "string", SCALAR), ("value", "float64", SCALAR)),
    "pal_statistics_msgs/Statistics": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("statistics", "pal_statistics_msgs/Statistic", VARIABLE),
    ),
    "pal_statistics_msgs/StatisticsNames": _h(
        ("header", "std_msgs/Header", SCALAR), ("names", "string", VARIABLE),
        ("names_version", "uint32", SCALAR),
    ),
    "pal_statistics_msgs/StatisticsValues": _h(
        ("header", "std_msgs/Header", SCALAR), ("values", "float64", VARIABLE),
        ("names_version", "uint32", SCALAR),
    ),
}

#: The ROS 2 (Jazzy) definition of the types whose fields differ from ROS 1's, under the
#: same canonical name. Consulted only for a query in the ROS 2 dialect (see `_is_ros2`),
#: so a ROS 1 member asking in `pkg/Type` spelling still gets `seq` and `K`.
ROS2_MESSAGES: dict[str, list[Field]] = {
    "builtin_interfaces/Time": _h(("sec", "int32", SCALAR), ("nanosec", "uint32", SCALAR)),
    "builtin_interfaces/Duration": _h(
        ("sec", "int32", SCALAR), ("nanosec", "uint32", SCALAR),
    ),
    "std_msgs/Header": _h(
        ("stamp", "builtin_interfaces/Time", SCALAR), ("frame_id", "string", SCALAR),
    ),
    "sensor_msgs/CameraInfo": _h(
        ("header", "std_msgs/Header", SCALAR), ("height", "uint32", SCALAR),
        ("width", "uint32", SCALAR), ("distortion_model", "string", SCALAR),
        ("d", "float64", VARIABLE), ("k", "float64", 9), ("r", "float64", 9),
        ("p", "float64", 12), ("binning_x", "uint32", SCALAR), ("binning_y", "uint32", SCALAR),
        ("roi", "sensor_msgs/RegionOfInterest", SCALAR),
    ),
    "trajectory_msgs/JointTrajectoryPoint": _h(
        ("positions", "float64", VARIABLE), ("velocities", "float64", VARIABLE),
        ("accelerations", "float64", VARIABLE), ("effort", "float64", VARIABLE),
        ("time_from_start", "builtin_interfaces/Duration", SCALAR),
    ),
}

#: Packages this table holds in ROS 2 form only: a query naming one is ROS 2 whatever its
#: spelling, so `control_msgs/FollowJointTrajectory` nests a Header without `seq`.
_ROS2_PACKAGES = frozenset({
    "builtin_interfaces", "control_msgs", "controller_manager_msgs", "lifecycle_msgs",
    "pal_statistics_msgs",
})

_SO101_SERVICES: dict[str, tuple[list[Field], list[Field]]] = {
    # ros2/common_interfaces 5.3.8
    "std_srvs/SetBool": (
        _h(("data", "bool", SCALAR)),
        _h(("success", "bool", SCALAR), ("message", "string", SCALAR)),
    ),
    "sensor_msgs/SetCameraInfo": (
        _h(("camera_info", "sensor_msgs/CameraInfo", SCALAR)),
        _h(("success", "bool", SCALAR), ("status_message", "string", SCALAR)),
    ),
    # ros-controls/control_msgs 5.10.0
    "control_msgs/QueryTrajectoryState": (
        _h(("time", "builtin_interfaces/Time", SCALAR)),
        _h(("success", "bool", SCALAR), ("message", "string", SCALAR),
           ("name", "string", VARIABLE), ("position", "float64", VARIABLE),
           ("velocity", "float64", VARIABLE), ("acceleration", "float64", VARIABLE)),
    ),
    # ros-controls/ros2_control 4.48.1
    "controller_manager_msgs/ListControllers": (
        [], _h(("controller", "controller_manager_msgs/ControllerState", VARIABLE)),
    ),
    "controller_manager_msgs/ListControllerTypes": (
        [], _h(("types", "string", VARIABLE), ("base_classes", "string", VARIABLE)),
    ),
    "controller_manager_msgs/LoadController": (
        _h(("name", "string", SCALAR)), _h(("ok", "bool", SCALAR)),
    ),
    "controller_manager_msgs/ConfigureController": (
        _h(("name", "string", SCALAR)), _h(("ok", "bool", SCALAR)),
    ),
    "controller_manager_msgs/ReloadControllerLibraries": (
        _h(("force_kill", "bool", SCALAR)), _h(("ok", "bool", SCALAR)),
    ),
    "controller_manager_msgs/SwitchController": (
        _h(("activate_controllers", "string", VARIABLE),
           ("deactivate_controllers", "string", VARIABLE),
           ("strictness", "int32", SCALAR), ("activate_asap", "bool", SCALAR),
           ("timeout", "builtin_interfaces/Duration", SCALAR)),
        _h(("ok", "bool", SCALAR), ("message", "string", SCALAR)),
    ),
    "controller_manager_msgs/UnloadController": (
        _h(("name", "string", SCALAR)), _h(("ok", "bool", SCALAR)),
    ),
    "controller_manager_msgs/CleanupController": (
        _h(("name", "string", SCALAR)), _h(("ok", "bool", SCALAR)),
    ),
    "controller_manager_msgs/ListHardwareComponents": (
        [], _h(("component", "controller_manager_msgs/HardwareComponentState", VARIABLE)),
    ),
    "controller_manager_msgs/ListHardwareInterfaces": (
        [], _h(("command_interfaces", "controller_manager_msgs/HardwareInterface", VARIABLE),
               ("state_interfaces", "controller_manager_msgs/HardwareInterface", VARIABLE)),
    ),
    "controller_manager_msgs/SetHardwareComponentState": (
        _h(("name", "string", SCALAR), ("target_state", "lifecycle_msgs/State", SCALAR)),
        _h(("ok", "bool", SCALAR), ("state", "lifecycle_msgs/State", SCALAR)),
    ),
}

_SO101_ACTIONS: dict[str, tuple[list[Field], list[Field], list[Field]]] = {
    # ros-controls/control_msgs 5.10.0
    "control_msgs/FollowJointTrajectory": (
        _h(("trajectory", "trajectory_msgs/JointTrajectory", SCALAR),
           ("multi_dof_trajectory", "trajectory_msgs/MultiDOFJointTrajectory", SCALAR),
           ("path_tolerance", "control_msgs/JointTolerance", VARIABLE),
           ("component_path_tolerance", "control_msgs/JointComponentTolerance", VARIABLE),
           ("goal_tolerance", "control_msgs/JointTolerance", VARIABLE),
           ("component_goal_tolerance", "control_msgs/JointComponentTolerance", VARIABLE),
           ("goal_time_tolerance", "builtin_interfaces/Duration", SCALAR)),
        _h(("error_code", "int32", SCALAR), ("error_string", "string", SCALAR)),
        _h(("header", "std_msgs/Header", SCALAR), ("joint_names", "string", VARIABLE),
           ("desired", "trajectory_msgs/JointTrajectoryPoint", SCALAR),
           ("actual", "trajectory_msgs/JointTrajectoryPoint", SCALAR),
           ("error", "trajectory_msgs/JointTrajectoryPoint", SCALAR),
           ("multi_dof_joint_names", "string", VARIABLE),
           ("multi_dof_desired", "trajectory_msgs/MultiDOFJointTrajectoryPoint", SCALAR),
           ("multi_dof_actual", "trajectory_msgs/MultiDOFJointTrajectoryPoint", SCALAR),
           ("multi_dof_error", "trajectory_msgs/MultiDOFJointTrajectoryPoint", SCALAR)),
    ),
    "control_msgs/ParallelGripperCommand": (
        _h(("command", "sensor_msgs/JointState", SCALAR)),
        _h(("state", "sensor_msgs/JointState", SCALAR), ("stalled", "bool", SCALAR),
           ("reached_goal", "bool", SCALAR)),
        _h(("state", "sensor_msgs/JointState", SCALAR)),
    ),
}

#: Constants, as (type, name, value) in file order, keyed by the name `typedefs()` labels
#: the definition with: the canonical type for a message, `<Srv>Request`/`<Srv>Response`
#: for a service half, `<Action>_Goal`/`_Result`/`_Feedback` for an action part.
_SO101_CONSTANTS: dict[str, list[tuple[str, str, str]]] = {
    "diagnostic_msgs/DiagnosticStatus": [
        ("byte", "OK", "0"), ("byte", "WARN", "1"), ("byte", "ERROR", "2"),
        ("byte", "STALE", "3"),
    ],
    "lifecycle_msgs/State": [
        ("uint8", "PRIMARY_STATE_UNKNOWN", "0"), ("uint8", "PRIMARY_STATE_UNCONFIGURED", "1"),
        ("uint8", "PRIMARY_STATE_INACTIVE", "2"), ("uint8", "PRIMARY_STATE_ACTIVE", "3"),
        ("uint8", "PRIMARY_STATE_FINALIZED", "4"),
        ("uint8", "TRANSITION_STATE_CONFIGURING", "10"),
        ("uint8", "TRANSITION_STATE_CLEANINGUP", "11"),
        ("uint8", "TRANSITION_STATE_SHUTTINGDOWN", "12"),
        ("uint8", "TRANSITION_STATE_ACTIVATING", "13"),
        ("uint8", "TRANSITION_STATE_DEACTIVATING", "14"),
        ("uint8", "TRANSITION_STATE_ERRORPROCESSING", "15"),
    ],
    "control_msgs/JointComponentTolerance": [
        ("uint16", "X_AXIS", "1"), ("uint16", "Y_AXIS", "2"), ("uint16", "Z_AXIS", "3"),
        ("uint16", "TRANSLATION", "4"), ("uint16", "ROTATION", "5"),
    ],
    "controller_manager_msgs/SwitchControllerRequest": [
        ("int32", "BEST_EFFORT", "1"), ("int32", "STRICT", "2"), ("int32", "AUTO", "3"),
        ("int32", "FORCE_AUTO", "4"),
    ],
    "control_msgs/FollowJointTrajectory_Result": [
        ("int32", "SUCCESSFUL", "0"), ("int32", "INVALID_GOAL", "-1"),
        ("int32", "INVALID_JOINTS", "-2"), ("int32", "OLD_HEADER_TIMESTAMP", "-3"),
        ("int32", "PATH_TOLERANCE_VIOLATED", "-4"), ("int32", "GOAL_TOLERANCE_VIOLATED", "-5"),
    ],
}

# --- end SO-101 ----------------------------------------------------------------------------

# --- myAGV: its standard types no other member uses, and robot_pose_ekf's service ------

_MYAGV: dict[str, list[Field]] = {
    "std_msgs/Float32": _h(("data", "float32", SCALAR)),
    "geometry_msgs/Point32": _h(
        ("x", "float32", SCALAR), ("y", "float32", SCALAR), ("z", "float32", SCALAR),
    ),
    "sensor_msgs/ChannelFloat32": _h(
        ("name", "string", SCALAR), ("values", "float32", VARIABLE),
    ),
    "sensor_msgs/PointCloud": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("points", "geometry_msgs/Point32", VARIABLE),
        ("channels", "sensor_msgs/ChannelFloat32", VARIABLE),
    ),
}

_MYAGV_SERVICES: dict[str, tuple[list[Field], list[Field]]] = {
    "robot_pose_ekf/GetStatus": ([], _h(("status", "string", SCALAR))),
}

# --- end myAGV ---------------------------------------------------------------------------

# --- AiNex: the rest of its boot interface (robots_specs/ainex/ros.yml) ------------------
#
# Hiwonder/ainex at e8fe2a816797cf83054135160df5a82ec3596a69:
#   src/ainex_interfaces/{msg,srv}/  ColorsDetect, ColorDetect, ROI, LineROI, ObjectsInfo,
#       ObjectInfo, SetInt, SetPoint, SetFloat
#   src/ainex_driver/ros_robot_controller/{msg,srv}/  SetBusServoState, BusServoState,
#       SetPWMServoState, PWMServoState, GetBusServoCmd, GetPWMServoCmd, LedState,
#       BuzzerState, OLEDState, MotorsState, MotorState, RGBsState, RGBState, Sbus,
#       ButtonState, GetBusServoState, GetPWMServoState
#   src/third_party/ros-sensor_msgs_ext/msg/magnetometer.msg
# plus the standard ROS 1 Noetic definitions those topics and services use: std_msgs/UInt16,
# sensor_msgs/Joy, sensor_msgs/MagneticField; sensor_msgs/SetCameraInfo and std_srvs/SetBool
# are the SO-101 block's (identical text, and the dialect is the query's).

_AINEX_INTERFACE: dict[str, list[Field]] = {
    "ainex_interfaces/ROI": _h(
        ("y_min", "int32", SCALAR), ("y_max", "int32", SCALAR),
        ("x_min", "int32", SCALAR), ("x_max", "int32", SCALAR),
    ),
    "ainex_interfaces/LineROI": _h(
        ("up", "ainex_interfaces/ROI", SCALAR), ("center", "ainex_interfaces/ROI", SCALAR),
        ("down", "ainex_interfaces/ROI", SCALAR),
    ),
    "ainex_interfaces/ColorDetect": _h(
        ("color_name", "string", SCALAR), ("use_name", "bool", SCALAR),
        ("detect_type", "string", SCALAR), ("roi", "ainex_interfaces/ROI", SCALAR),
        ("line_roi", "ainex_interfaces/LineROI", SCALAR),
        ("image_process_size", "uint32", VARIABLE), ("lab_min", "int16", VARIABLE),
        ("lab_max", "int16", VARIABLE), ("min_area", "float64", SCALAR),
        ("max_area", "float64", SCALAR),
    ),
    "ainex_interfaces/ColorsDetect": _h(
        ("data", "ainex_interfaces/ColorDetect", VARIABLE),
    ),
    "ainex_interfaces/ObjectInfo": _h(
        ("label", "string", SCALAR), ("type", "string", SCALAR),
        ("width", "int32", SCALAR), ("height", "int32", SCALAR), ("x", "int32", SCALAR),
        ("y", "int32", SCALAR), ("radius", "int32", SCALAR), ("angle", "int32", SCALAR),
        ("left_point", "int32", VARIABLE), ("right_point", "int32", VARIABLE),
    ),
    "ainex_interfaces/ObjectsInfo": _h(
        ("data", "ainex_interfaces/ObjectInfo", VARIABLE),
    ),
    "ros_robot_controller/BusServoState": _h(
        ("present_id", "uint16", VARIABLE), ("target_id", "uint16", VARIABLE),
        ("position", "uint16", VARIABLE), ("offset", "int16", VARIABLE),
        ("voltage", "uint16", VARIABLE), ("temperature", "uint16", VARIABLE),
        ("position_limit", "uint16", VARIABLE), ("voltage_limit", "uint16", VARIABLE),
        ("max_temperature_limit", "uint16", VARIABLE), ("enable_torque", "uint16", VARIABLE),
        ("save_offset", "uint16", VARIABLE), ("stop", "uint16", VARIABLE),
    ),
    "ros_robot_controller/SetBusServoState": _h(
        ("state", "ros_robot_controller/BusServoState", VARIABLE),
        ("duration", "float64", SCALAR),
    ),
    "ros_robot_controller/PWMServoState": _h(
        ("id", "uint16", VARIABLE), ("position", "uint16", VARIABLE),
        ("offset", "int16", VARIABLE),
    ),
    "ros_robot_controller/SetPWMServoState": _h(
        ("state", "ros_robot_controller/PWMServoState", VARIABLE),
        ("duration", "float64", SCALAR),
    ),
    "ros_robot_controller/GetBusServoCmd": _h(
        ("id", "uint8", SCALAR), ("get_id", "uint8", SCALAR),
        ("get_position", "uint8", SCALAR), ("get_offset", "uint8", SCALAR),
        ("get_voltage", "uint8", SCALAR), ("get_temperature", "uint8", SCALAR),
        ("get_position_limit", "uint8", SCALAR), ("get_voltage_limit", "uint8", SCALAR),
        ("get_max_temperature_limit", "uint8", SCALAR), ("get_torque_state", "uint8", SCALAR),
    ),
    "ros_robot_controller/GetPWMServoCmd": _h(
        ("id", "uint8", SCALAR), ("get_position", "uint8", SCALAR),
        ("get_offset", "uint8", SCALAR),
    ),
    "ros_robot_controller/LedState": _h(
        ("id", "uint8", SCALAR), ("on_time", "float32", SCALAR),
        ("off_time", "float32", SCALAR), ("repeat", "uint16", SCALAR),
    ),
    "ros_robot_controller/BuzzerState": _h(
        ("freq", "uint16", SCALAR), ("on_time", "float32", SCALAR),
        ("off_time", "float32", SCALAR), ("repeat", "uint16", SCALAR),
    ),
    "ros_robot_controller/OLEDState": _h(
        ("index", "uint8", SCALAR), ("text", "string", SCALAR),
    ),
    "ros_robot_controller/MotorState": _h(
        ("id", "uint16", SCALAR), ("rps", "float64", SCALAR), ("duty", "int8", SCALAR),
    ),
    "ros_robot_controller/MotorsState": _h(
        ("data", "ros_robot_controller/MotorState", VARIABLE),
    ),
    "ros_robot_controller/RGBState": _h(
        ("id", "uint8", SCALAR), ("r", "uint8", SCALAR), ("g", "uint8", SCALAR),
        ("b", "uint8", SCALAR),
    ),
    "ros_robot_controller/RGBsState": _h(
        ("data", "ros_robot_controller/RGBState", VARIABLE),
    ),
    "ros_robot_controller/Sbus": _h(
        ("header", "std_msgs/Header", SCALAR), ("channel", "float32", VARIABLE),
    ),
    "ros_robot_controller/ButtonState": _h(
        ("id", "uint8", SCALAR), ("state", "uint8", SCALAR),
    ),
    "sensor_msgs_ext/magnetometer": _h(
        ("x", "float64", SCALAR), ("y", "float64", SCALAR), ("z", "float64", SCALAR),
    ),
    "std_msgs/UInt16": _h(("data", "uint16", SCALAR)),
    "sensor_msgs/Joy": _h(
        ("header", "std_msgs/Header", SCALAR), ("axes", "float32", VARIABLE),
        ("buttons", "int32", VARIABLE),
    ),
    "sensor_msgs/MagneticField": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("magnetic_field", "geometry_msgs/Vector3", SCALAR),
        ("magnetic_field_covariance", "float64", 9),
    ),
}

_AINEX_SERVICES: dict[str, tuple[list[Field], list[Field]]] = {
    "ainex_interfaces/SetInt": (
        _h(("data", "int64", SCALAR)),
        _h(("success", "bool", SCALAR), ("message", "string", SCALAR)),
    ),
    "ainex_interfaces/SetFloat": (
        _h(("data", "float64", SCALAR)),
        _h(("success", "bool", SCALAR), ("message", "string", SCALAR)),
    ),
    "ainex_interfaces/SetPoint": (
        _h(("data", "geometry_msgs/Point", SCALAR)),
        _h(("success", "bool", SCALAR), ("message", "string", SCALAR)),
    ),
    "ros_robot_controller/GetBusServoState": (
        _h(("cmd", "ros_robot_controller/GetBusServoCmd", VARIABLE)),
        _h(("success", "bool", SCALAR),
           ("state", "ros_robot_controller/BusServoState", VARIABLE)),
    ),
    "ros_robot_controller/GetPWMServoState": (
        _h(("cmd", "ros_robot_controller/GetPWMServoCmd", VARIABLE)),
        _h(("success", "bool", SCALAR),
           ("state", "ros_robot_controller/PWMServoState", VARIABLE)),
    ),
}

# --- end AiNex ----------------------------------------------------------------------------

# --- myAGV + myCobot 280: its navigation launch (robots_specs/myagv_mycobot280/ros.yml) ---
#
# The standard ROS 1 Noetic definitions its map_server, amcl and move_base use, from the
# distribution's own files: ros/common_msgs noetic-devel (nav_msgs, geometry_msgs,
# sensor_msgs, actionlib_msgs), ros-planning/navigation_msgs noetic-devel (move_base_msgs,
# map_msgs), ros/dynamic_reconfigure noetic-devel (msg/*, srv/Reconfigure.srv). The
# ROSMASTER X3 PLUS shares the dynamic_reconfigure and actionlib_msgs ones.

_NAV: dict[str, list[Field]] = {
    "nav_msgs/MapMetaData": _h(
        ("map_load_time", "time", SCALAR), ("resolution", "float32", SCALAR),
        ("width", "uint32", SCALAR), ("height", "uint32", SCALAR),
        ("origin", "geometry_msgs/Pose", SCALAR),
    ),
    "nav_msgs/OccupancyGrid": _h(
        ("header", "std_msgs/Header", SCALAR), ("info", "nav_msgs/MapMetaData", SCALAR),
        ("data", "int8", VARIABLE),
    ),
    "nav_msgs/Path": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("poses", "geometry_msgs/PoseStamped", VARIABLE),
    ),
    "geometry_msgs/PoseWithCovarianceStamped": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("pose", "geometry_msgs/PoseWithCovariance", SCALAR),
    ),
    "geometry_msgs/PoseArray": _h(
        ("header", "std_msgs/Header", SCALAR), ("poses", "geometry_msgs/Pose", VARIABLE),
    ),
    "geometry_msgs/Polygon": _h(("points", "geometry_msgs/Point32", VARIABLE)),
    "geometry_msgs/PolygonStamped": _h(
        ("header", "std_msgs/Header", SCALAR), ("polygon", "geometry_msgs/Polygon", SCALAR),
    ),
    "map_msgs/OccupancyGridUpdate": _h(
        ("header", "std_msgs/Header", SCALAR), ("x", "int32", SCALAR), ("y", "int32", SCALAR),
        ("width", "uint32", SCALAR), ("height", "uint32", SCALAR), ("data", "int8", VARIABLE),
    ),
    "actionlib_msgs/GoalID": _h(("stamp", "time", SCALAR), ("id", "string", SCALAR)),
    "actionlib_msgs/GoalStatus": _h(
        ("goal_id", "actionlib_msgs/GoalID", SCALAR), ("status", "uint8", SCALAR),
        ("text", "string", SCALAR),
    ),
    "actionlib_msgs/GoalStatusArray": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("status_list", "actionlib_msgs/GoalStatus", VARIABLE),
    ),
    "move_base_msgs/MoveBaseGoal": _h(("target_pose", "geometry_msgs/PoseStamped", SCALAR)),
    "move_base_msgs/MoveBaseResult": _h(),
    "move_base_msgs/MoveBaseFeedback": _h(
        ("base_position", "geometry_msgs/PoseStamped", SCALAR),
    ),
    "move_base_msgs/MoveBaseActionGoal": _h(
        ("header", "std_msgs/Header", SCALAR), ("goal_id", "actionlib_msgs/GoalID", SCALAR),
        ("goal", "move_base_msgs/MoveBaseGoal", SCALAR),
    ),
    "move_base_msgs/MoveBaseActionResult": _h(
        ("header", "std_msgs/Header", SCALAR), ("status", "actionlib_msgs/GoalStatus", SCALAR),
        ("result", "move_base_msgs/MoveBaseResult", SCALAR),
    ),
    "move_base_msgs/MoveBaseActionFeedback": _h(
        ("header", "std_msgs/Header", SCALAR), ("status", "actionlib_msgs/GoalStatus", SCALAR),
        ("feedback", "move_base_msgs/MoveBaseFeedback", SCALAR),
    ),
    "move_base_msgs/RecoveryStatus": _h(
        ("pose_stamped", "geometry_msgs/PoseStamped", SCALAR),
        ("current_recovery_number", "uint16", SCALAR),
        ("total_number_of_recoveries", "uint16", SCALAR),
        ("recovery_behavior_name", "string", SCALAR),
    ),
    "sensor_msgs/PointField": _h(
        ("name", "string", SCALAR), ("offset", "uint32", SCALAR), ("datatype", "uint8", SCALAR),
        ("count", "uint32", SCALAR),
    ),
    "sensor_msgs/PointCloud2": _h(
        ("header", "std_msgs/Header", SCALAR), ("height", "uint32", SCALAR),
        ("width", "uint32", SCALAR), ("fields", "sensor_msgs/PointField", VARIABLE),
        ("is_bigendian", "bool", SCALAR), ("point_step", "uint32", SCALAR),
        ("row_step", "uint32", SCALAR), ("data", "uint8", VARIABLE),
        ("is_dense", "bool", SCALAR),
    ),
    "dynamic_reconfigure/BoolParameter": _h(("name", "string", SCALAR), ("value", "bool", SCALAR)),
    "dynamic_reconfigure/IntParameter": _h(("name", "string", SCALAR), ("value", "int32", SCALAR)),
    "dynamic_reconfigure/StrParameter": _h(("name", "string", SCALAR), ("value", "string", SCALAR)),
    "dynamic_reconfigure/DoubleParameter": _h(
        ("name", "string", SCALAR), ("value", "float64", SCALAR),
    ),
    "dynamic_reconfigure/GroupState": _h(
        ("name", "string", SCALAR), ("state", "bool", SCALAR), ("id", "int32", SCALAR),
        ("parent", "int32", SCALAR),
    ),
    "dynamic_reconfigure/Config": _h(
        ("bools", "dynamic_reconfigure/BoolParameter", VARIABLE),
        ("ints", "dynamic_reconfigure/IntParameter", VARIABLE),
        ("strs", "dynamic_reconfigure/StrParameter", VARIABLE),
        ("doubles", "dynamic_reconfigure/DoubleParameter", VARIABLE),
        ("groups", "dynamic_reconfigure/GroupState", VARIABLE),
    ),
    "dynamic_reconfigure/ParamDescription": _h(
        ("name", "string", SCALAR), ("type", "string", SCALAR), ("level", "uint32", SCALAR),
        ("description", "string", SCALAR), ("edit_method", "string", SCALAR),
    ),
    "dynamic_reconfigure/Group": _h(
        ("name", "string", SCALAR), ("type", "string", SCALAR),
        ("parameters", "dynamic_reconfigure/ParamDescription", VARIABLE),
        ("parent", "int32", SCALAR), ("id", "int32", SCALAR),
    ),
    "dynamic_reconfigure/ConfigDescription": _h(
        ("groups", "dynamic_reconfigure/Group", VARIABLE),
        ("max", "dynamic_reconfigure/Config", SCALAR),
        ("min", "dynamic_reconfigure/Config", SCALAR),
        ("dflt", "dynamic_reconfigure/Config", SCALAR),
    ),
}

_NAV_CONSTANTS: dict[str, list[tuple[str, str, str]]] = {
    "actionlib_msgs/GoalStatus": [
        ("uint8", "PENDING", "0"), ("uint8", "ACTIVE", "1"), ("uint8", "PREEMPTED", "2"),
        ("uint8", "SUCCEEDED", "3"), ("uint8", "ABORTED", "4"), ("uint8", "REJECTED", "5"),
        ("uint8", "PREEMPTING", "6"), ("uint8", "RECALLING", "7"), ("uint8", "RECALLED", "8"),
        ("uint8", "LOST", "9"),
    ],
    "sensor_msgs/PointField": [
        ("uint8", "INT8", "1"), ("uint8", "UINT8", "2"), ("uint8", "INT16", "3"),
        ("uint8", "UINT16", "4"), ("uint8", "INT32", "5"), ("uint8", "UINT32", "6"),
        ("uint8", "FLOAT32", "7"), ("uint8", "FLOAT64", "8"),
    ],
    "nav_msgs/LoadMapResponse": [
        ("uint8", "RESULT_SUCCESS", "0"), ("uint8", "RESULT_MAP_DOES_NOT_EXIST", "1"),
        ("uint8", "RESULT_INVALID_MAP_DATA", "2"), ("uint8", "RESULT_INVALID_MAP_METADATA", "3"),
        ("uint8", "RESULT_UNDEFINED_FAILURE", "255"),
    ],
}

_NAV_SERVICES: dict[str, tuple[list[Field], list[Field]]] = {
    "nav_msgs/GetMap": ([], _h(("map", "nav_msgs/OccupancyGrid", SCALAR))),
    "nav_msgs/GetPlan": (
        _h(("start", "geometry_msgs/PoseStamped", SCALAR),
           ("goal", "geometry_msgs/PoseStamped", SCALAR), ("tolerance", "float32", SCALAR)),
        _h(("plan", "nav_msgs/Path", SCALAR)),
    ),
    "nav_msgs/LoadMap": (
        _h(("map_url", "string", SCALAR)),
        _h(("map", "nav_msgs/OccupancyGrid", SCALAR), ("result", "uint8", SCALAR)),
    ),
    "nav_msgs/SetMap": (
        _h(("map", "nav_msgs/OccupancyGrid", SCALAR),
           ("initial_pose", "geometry_msgs/PoseWithCovarianceStamped", SCALAR)),
        _h(("success", "bool", SCALAR)),
    ),
    "dynamic_reconfigure/Reconfigure": (
        _h(("config", "dynamic_reconfigure/Config", SCALAR)),
        _h(("config", "dynamic_reconfigure/Config", SCALAR)),
    ),
}

# --- end myAGV + myCobot 280 ---------------------------------------------------------------

# --- ROSMASTER X3 PLUS (robots_specs/rosmaster_x3_plus/ros.yml) ----------------------------
#
# Yahboom's ROS 1 workspace, ROSMASTER-X3Plus_ROS1_code.zip (the Drive archive robots.yml
# pins): yahboomcar_ws.zip!/yahboomcar_ws/src/yahboomcar_msgs/{msg/ArmJoint.msg,
# srv/RobotArmArray.srv}; software.zip!/software/library_ws/src/orbbec-ros-sdk (package
# orbbec_camera) {srv/GetBool, GetCameraInfo, GetCameraParams, GetDeviceInfo, GetInt32,
# GetString, SetInt32, SetString .srv, msg/DeviceInfo.msg}. Standard Noetic definitions
# from ros/common_msgs and ros/std_msgs: std_msgs/Int32, sensor_msgs/JoyFeedback(Array).
# Its std_srvs/SetBool is the SO-101 block's (identical text).

_X3: dict[str, list[Field]] = {
    "std_msgs/Int32": _h(("data", "int32", SCALAR)),
    "sensor_msgs/JoyFeedback": _h(
        ("type", "uint8", SCALAR), ("id", "uint8", SCALAR), ("intensity", "float32", SCALAR),
    ),
    "sensor_msgs/JoyFeedbackArray": _h(("array", "sensor_msgs/JoyFeedback", VARIABLE)),
    "yahboomcar_msgs/ArmJoint": _h(
        ("id", "int32", SCALAR), ("run_time", "int32", SCALAR), ("angle", "float32", SCALAR),
        ("joints", "float32", VARIABLE),
    ),
    "orbbec_camera/DeviceInfo": _h(
        ("header", "std_msgs/Header", SCALAR), ("name", "string", SCALAR),
        ("vid", "int32", SCALAR), ("pid", "int32", SCALAR),
        ("serial_number", "string", SCALAR), ("firmware_version", "string", SCALAR),
        ("supported_min_sdk_version", "string", SCALAR),
        ("hardware_version", "string", SCALAR),
    ),
}

_X3_CONSTANTS: dict[str, list[tuple[str, str, str]]] = {
    "sensor_msgs/JoyFeedback": [
        ("uint8", "TYPE_LED", "0"), ("uint8", "TYPE_RUMBLE", "1"), ("uint8", "TYPE_BUZZER", "2"),
    ],
}

_OB_RESULT = (("success", "bool", SCALAR), ("message", "string", SCALAR))

_X3_SERVICES: dict[str, tuple[list[Field], list[Field]]] = {
    "yahboomcar_msgs/RobotArmArray": (
        _h(("apply", "string", SCALAR)), _h(("angles", "float64", VARIABLE)),
    ),
    "orbbec_camera/GetBool": ([], _h(("data", "bool", SCALAR), *_OB_RESULT)),
    "orbbec_camera/GetCameraInfo": (
        [], _h(("info", "sensor_msgs/CameraInfo", SCALAR), *_OB_RESULT),
    ),
    "orbbec_camera/GetCameraParams": ([], _h(
        ("l_intr_p", "float32", 4), ("r_intr_p", "float32", 4), ("r2l_r", "float32", 9),
        ("r2l_t", "float32", 3), *_OB_RESULT)),
    "orbbec_camera/GetDeviceInfo": (
        [], _h(("info", "orbbec_camera/DeviceInfo", SCALAR), *_OB_RESULT),
    ),
    "orbbec_camera/GetInt32": ([], _h(("data", "int32", SCALAR), *_OB_RESULT)),
    "orbbec_camera/GetString": ([], _h(("data", "string", SCALAR), *_OB_RESULT)),
    "orbbec_camera/SetInt32": (_h(("data", "int32", SCALAR)), _h(*_OB_RESULT)),
    "orbbec_camera/SetString": (_h(("data", "string", SCALAR)), _h(*_OB_RESULT)),
}

# --- end ROSMASTER X3 PLUS -------------------------------------------------------------------

MESSAGES: dict[str, list[Field]] = {**_STD, **_AINEX, **_SO101, **_MYAGV, **_AINEX_INTERFACE,
                                    **_NAV, **_X3}

#: Constants per definition, as `rosapi`'s `constnames`/`constvalues` carry them. Keyed as
#: `_SO101_CONSTANTS` documents; a definition with none is simply absent.
CONSTANTS: dict[str, list[tuple[str, str, str]]] = {**_SO101_CONSTANTS, **_NAV_CONSTANTS,
                                                   **_X3_CONSTANTS}

#: Services: canonical name -> (request fields, response fields). `rosapi` names the two
#: halves `<Srv>Request` and `<Srv>Response`, and that is how `typedefs()` labels them.
SERVICES: dict[str, tuple[list[Field], list[Field]]] = {
    "std_srvs/Empty": ([], []),
    "std_srvs/Trigger": ([], _h(("success", "bool", SCALAR), ("message", "string", SCALAR))),
    "ainex_interfaces/SetWalkingCommand": (
        _h(("command", "string", SCALAR)), _h(("result", "bool", SCALAR)),
    ),
    "ainex_interfaces/GetWalkingParam": (
        _h(("get_param", "bool", SCALAR)),
        _h(("parameters", "ainex_interfaces/WalkingParam", SCALAR)),
    ),
    "ainex_interfaces/GetWalkingState": (
        [], _h(("state", "bool", SCALAR), ("message", "string", SCALAR)),
    ),
    "ros_robot_controller/GetBusServosPosition": (
        _h(("id", "uint8", VARIABLE)),
        _h(("success", "bool", SCALAR),
           ("position", "ros_robot_controller/BusServoPosition", VARIABLE)),
    ),
    **_SO101_SERVICES,
    **_MYAGV_SERVICES,
    **_AINEX_SERVICES,
    **_NAV_SERVICES,
    **_X3_SERVICES,
}


#: Actions: canonical name -> (goal fields, result fields, feedback fields). A surface
#: that serves one transcribes its `.action` file here, with its provenance in the block
#: above, exactly as for messages and services. `rosapi` labels the three parts
#: `<Action>_Goal`, `<Action>_Result` and `<Action>_Feedback`.
ACTIONS: dict[str, tuple[list[Field], list[Field], list[Field]]] = {**_SO101_ACTIONS}

_ACTION_PARTS = ("goal", "result", "feedback")


def _is_ros2(type_name: str) -> bool:
    """Whether a query is in the ROS 2 dialect: `pkg/msg/Type` spelling, or a ROS-2-only package."""
    parts = str(type_name).split("/")
    return (len(parts) == 3 and parts[1] in ("msg", "srv", "action")) \
        or parts[0] in _ROS2_PACKAGES


def _message(type_name: str, ros2: bool) -> tuple[str, list[Field]] | None:
    """(label, fields) for a message type in one dialect, or None if unknown."""
    if ros2:
        parts = str(type_name).split("/")
        plain = f"{parts[0]}/{parts[-1]}" if len(parts) == 3 else str(type_name)
        if plain in ROS2_MESSAGES:
            return plain, ROS2_MESSAGES[plain]
    name = canonical(type_name)
    fields = MESSAGES.get(name)
    return None if fields is None else (name, fields)


def fields_of(type_name: str) -> list[Field] | None:
    """The top-level fields of a message type, or None if it is not one this bridge knows."""
    hit = _message(type_name, _is_ros2(type_name))
    return None if hit is None else hit[1]


def _example(field_type: str, arraylen: int) -> str:
    """`rosapi`'s `examples` column: a literal for primitives, `{}` for nested, `[]` arrays."""
    if arraylen != SCALAR:
        return "[]"
    if field_type in ("string",):
        return ""
    if field_type == "bool":
        return "False"
    if field_type in PRIMITIVES:
        return "0" if field_type not in ("float32", "float64") else "0.0"
    return "{}"


def _typedef(type_name: str, fields: list[Field]) -> dict:
    consts = CONSTANTS.get(type_name, [])
    return {
        "type": type_name,
        "fieldnames": [f[0] for f in fields],
        "fieldtypes": [f[1] for f in fields],
        "fieldarraylen": [f[2] for f in fields],
        "examples": [_example(f[1], f[2]) for f in fields],
        "constnames": [c[1] for c in consts],
        "constvalues": [c[2] for c in consts],
    }


def _closure(type_name: str, fields: list[Field], ros2: bool = False) -> list[dict]:
    """This type's typedef followed by every nested type's, each once, depth first.

    `ros2` is the dialect of the query, and every nested type is looked up in it: a ROS 2
    CameraInfo nests a ROS 2 Header.
    """
    out: list[dict] = []
    seen: set[str] = set()

    def walk(name: str, flds: list[Field]) -> None:
        if name in seen:
            return
        seen.add(name)
        out.append(_typedef(name, flds))
        for _, ftype, _ in flds:
            if ftype in PRIMITIVES:
                continue
            hit = _message(ftype, ros2)
            if hit is not None:
                walk(*hit)

    walk(type_name, fields)
    return out


def typedefs(type_name: str) -> list[dict]:
    """`/rosapi/message_details`: the type and everything it nests, or `[]` if unknown."""
    ros2 = _is_ros2(type_name)
    hit = _message(type_name, ros2)
    return [] if hit is None else _closure(*hit, ros2=ros2)


def service_typedefs(service_type: str, half: str) -> list[dict]:
    """`/rosapi/service_{request,response}_details`, `half` being `request` or `response`."""
    name = canonical(service_type)
    pair = SERVICES.get(name)
    if pair is None:
        return []
    fields = pair[0] if half == "request" else pair[1]
    return _closure(f"{name}{half.capitalize()}", fields, ros2=_is_ros2(service_type))


def action_fields(action_type: str, part: str) -> list[Field] | None:
    """One part (`goal`, `result` or `feedback`) of an action's definition, or None."""
    triple = ACTIONS.get(canonical(action_type))
    if triple is None or part not in _ACTION_PARTS:
        return None
    return triple[_ACTION_PARTS.index(part)]


def action_typedefs(action_type: str, part: str) -> list[dict]:
    """`/rosapi/action_{goal,result,feedback}_details`, or `[]` if unknown."""
    fields = action_fields(action_type, part)
    if fields is None:
        return []
    return _closure(f"{canonical(action_type)}_{part.capitalize()}", fields,
                    ros2=_is_ros2(action_type))


def _field_line(field: Field) -> str:
    name, ftype, arraylen = field
    suffix = "" if arraylen == SCALAR else "[]" if arraylen == VARIABLE else f"[{arraylen}]"
    return f"{ftype}{suffix} {name}"


def _definition_lines(typedef: dict) -> str:
    """One definition's text from its typedef: constants first, then the fields."""
    consts = [f"{c[0]} {c[1]}={c[2]}" for c in CONSTANTS.get(typedef["type"], [])]
    fields = zip(typedef["fieldnames"], typedef["fieldtypes"], typedef["fieldarraylen"])
    return "\n".join(consts + [_field_line(f) for f in fields])


def definition_text(type_name: str) -> str:
    """A message's full definition text, as `/rosapi/topics_and_raw_types` answers it.

    `gendeps --cat`'s layout: the type's own fields, then each nested type once, after a
    line of 80 `=` and `MSG: <type>`. Empty for a type this table does not hold.
    """
    closure = typedefs(type_name)
    if not closure:
        return ""
    blocks = [_definition_lines(closure[0])]
    for typedef in closure[1:]:
        blocks.append("=" * 80 + f"\nMSG: {typedef['type']}\n" + _definition_lines(typedef))
    return "\n".join(blocks) + "\n"


def ros2_name(type_name: str, category: str) -> str:
    """`pkg/Type` -> `pkg/<category>/Type`; a name already in ROS 2 form is unchanged."""
    parts = str(type_name).split("/")
    if len(parts) == 2:
        return f"{parts[0]}/{category}/{parts[1]}"
    return str(type_name)


def interfaces() -> list[str]:
    """`/rosapi/interfaces`: every interface this table holds, in ROS 2 spelling."""
    return sorted(
        {ros2_name(t, "msg") for t in {**MESSAGES, **ROS2_MESSAGES}}
        | {ros2_name(t, "srv") for t in SERVICES}
        | {ros2_name(t, "action") for t in ACTIONS}
    )


def known_types() -> frozenset[str]:
    """Every canonical message, service and action type this table can answer for."""
    return (frozenset(MESSAGES) | frozenset(ROS2_MESSAGES) | frozenset(SERVICES)
            | frozenset(ACTIONS))
