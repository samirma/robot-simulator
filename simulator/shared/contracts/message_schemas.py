"""The message and service definitions the bridge answers `rosapi` schema queries from.

A real rosbridge ships `rosapi`, and `rosapi` answers `/rosapi/message_details` by reading
the `.msg` files installed on the robot. There is no ROS install here and nothing to read,
so the definitions live in this table -- **transcribed, not authored**. Every entry is
copied from the manufacturer's own definition file for that embodiment, or from the ROS
distribution's for the standard packages, and the block below records where each came
from. That is the same standing the vendored STLs have (`shared/robots/ainex/urdf/
PROVENANCE.md`): a definition that is not verbatim would be indistinguishable from the
vendor's afterwards, and a client generating code against it would be generating it
against a guess.

Two things the table has to get right that a naive one would not:

* **Both dialects resolve to one definition.** `sensor_msgs/Imu` and `sensor_msgs/msg/Imu`
  are the same message; this graph deliberately carries ROS 1 names for the myAGV and the
  AiNex beside ROS 2 names for the SO-101, and a client may ask in either spelling.
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
myAGV (Elephant Robotics) -- `elephantrobotics/myagv_ros`, branch `myagv_ros_2023Pi`: uses
    the standard messages only (Twist, Odometry, CompressedImage, LaserScan, Image,
    CameraInfo); no vendor package.
SO-101 (ros2_control bringup) -- standard ROS 2 messages, plus
    `ros-controls/mujoco_ros2_control` at `mujoco_ros2_control_msgs/msg/`:
    FreeJointStateArray.msg, FreeJointState.msg (the plugin the reference rig runs).
Standard packages -- the ROS distributions' own files, identical between Noetic and
    Humble for every type here: std_msgs, std_srvs, geometry_msgs, nav_msgs, sensor_msgs,
    trajectory_msgs, builtin_interfaces.
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

#: ROS 1 -> ROS 2 renames for the few types whose *package* differs by dialect, so that a
#: ROS 2 client asking for `builtin_interfaces/msg/Time` finds the same definition a ROS 1
#: client gets for `time`.
_ALIASES = {
    "builtin_interfaces/Time": "std_msgs/Time",
    "builtin_interfaces/Duration": "std_msgs/Duration",
}


def canonical(type_name: str) -> str:
    """One spelling for a type: `pkg/Type`, with any `/msg/` or `/srv/` segment removed."""
    parts = type_name.split("/")
    if len(parts) == 3 and parts[1] in ("msg", "srv"):
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

# --- SO-101: the reference rig's mujoco_ros2_control plugin ------------------------------

_SO101: dict[str, list[Field]] = {
    "mujoco_ros2_control_msgs/FreeJointState": _h(
        ("name", "string", SCALAR), ("pose", "geometry_msgs/PoseStamped", SCALAR),
        ("twist", "geometry_msgs/TwistStamped", SCALAR),
    ),
    "mujoco_ros2_control_msgs/FreeJointStateArray": _h(
        ("header", "std_msgs/Header", SCALAR),
        ("free_joints", "mujoco_ros2_control_msgs/FreeJointState", VARIABLE),
    ),
}

MESSAGES: dict[str, list[Field]] = {**_STD, **_AINEX, **_SO101}

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
}


def fields_of(type_name: str) -> list[Field] | None:
    """The top-level fields of a message type, or None if it is not one this bridge knows."""
    return MESSAGES.get(canonical(type_name))


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
    return {
        "type": type_name,
        "fieldnames": [f[0] for f in fields],
        "fieldtypes": [f[1] for f in fields],
        "fieldarraylen": [f[2] for f in fields],
        "examples": [_example(f[1], f[2]) for f in fields],
        "constnames": [],
        "constvalues": [],
    }


def _closure(type_name: str, fields: list[Field]) -> list[dict]:
    """This type's typedef followed by every nested type's, each once, depth first."""
    out: list[dict] = []
    seen: set[str] = set()

    def walk(name: str, flds: list[Field]) -> None:
        if name in seen:
            return
        seen.add(name)
        out.append(_typedef(name, flds))
        for _, ftype, _ in flds:
            nested = canonical(ftype)
            if nested in PRIMITIVES:
                continue
            body = MESSAGES.get(nested)
            if body is not None:
                walk(nested, body)

    walk(type_name, fields)
    return out


def typedefs(type_name: str) -> list[dict]:
    """`/rosapi/message_details`: the type and everything it nests, or `[]` if unknown."""
    name = canonical(type_name)
    fields = MESSAGES.get(name)
    return [] if fields is None else _closure(name, fields)


def service_typedefs(service_type: str, half: str) -> list[dict]:
    """`/rosapi/service_{request,response}_details`, `half` being `request` or `response`."""
    name = canonical(service_type)
    pair = SERVICES.get(name)
    if pair is None:
        return []
    fields = pair[0] if half == "request" else pair[1]
    return _closure(f"{name}{half.capitalize()}", fields)


def known_types() -> frozenset[str]:
    """Every canonical message and service type this table can answer for."""
    return frozenset(MESSAGES) | frozenset(SERVICES)
