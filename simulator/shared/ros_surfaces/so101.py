"""The SO-101's ROS 2 interface, transcribed from `robots_specs/so101/ros2.yml`.

The interface is the community bringup's (`so_arm101_description`'s
`controllers_bringup.launch.py hardware_type:=real`) plus `usb_cam` in `/wrist`:

    node /controller_manager           controller_manager services, /controller_manager/activity,
                                       /diagnostics, introspection_data/*, statistics/*
    node /joint_state_broadcaster      /joint_states, /dynamic_joint_states                 50 Hz
    node /joint_trajectory_controller  joint_trajectory (in), speed_scaling_input (in),
                                       controller_state 50 Hz, query_state,
                                       follow_joint_trajectory (action)
    node /gripper_controller           gripper_cmd (action, control_msgs/action/ParallelGripperCommand)
    node /robot_state_publisher        /robot_description (latched), /tf 20 Hz, /tf_static (latched)
    node /wrist/usb_cam                /wrist/image_raw, /wrist/camera_info,
                                       /wrist/image_raw/compressed 30 Hz, /wrist/set_capture,
                                       /wrist/usb_cam/set_camera_info

The `image_transport` plugin topics ros2.yml marks `unverified` (compressedDepth, theora,
zstd) are not served. Beside the vendor interface, the workspace-owned `/reset`
(`std_srvs/srv/Trigger`, node `/simulator`) and the worktop rig under `/scene`
(`scene.py`, whose names are defined here) are the only additions (spec §3).

Everything above is a bare vendor name; the fleet's `NamespacedBus` composes the robot's
namespace where a name reaches the wire. The constants are plain Python, because the
workspace parity tests (`tests/` at the repository root) read them as data, parsed rather
than imported, with no MuJoCo or numpy.
"""

from __future__ import annotations

import math
import threading
import time as _time
import xml.etree.ElementTree as ET

import numpy as np

from contracts.physical import Figure

# ------------------------------------------------------------------------- contract

#: `joints` in ros2.yml, in the bringup's order (servo ids 1-6).
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)
GRIPPER_JOINT = "gripper_joint"
JOINT_ORDER: tuple[str, ...] = (*ARM_JOINTS, GRIPPER_JOINT)

NODE_CONTROLLER_MANAGER = "/controller_manager"
NODE_JOINT_STATE_BROADCASTER = "/joint_state_broadcaster"
NODE_JOINT_TRAJECTORY_CONTROLLER = "/joint_trajectory_controller"
NODE_GRIPPER_CONTROLLER = "/gripper_controller"
NODE_ROBOT_STATE_PUBLISHER = "/robot_state_publisher"
NODE_WRIST_CAMERA = "/wrist/usb_cam"

TOPIC_ARM_COMMAND = "/joint_trajectory_controller/joint_trajectory"
TOPIC_CONTROLLER_STATE = "/joint_trajectory_controller/controller_state"
TOPIC_SPEED_SCALING = "/joint_trajectory_controller/speed_scaling_input"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_DYNAMIC_JOINT_STATES = "/dynamic_joint_states"
TOPIC_ROBOT_DESCRIPTION = "/robot_description"
TOPIC_TF = "/tf"
TOPIC_TF_STATIC = "/tf_static"
TOPIC_CM_ACTIVITY = "/controller_manager/activity"
TOPIC_DIAGNOSTICS = "/diagnostics"
TOPIC_INTROSPECTION_FULL = "/controller_manager/introspection_data/full"
TOPIC_INTROSPECTION_NAMES = "/controller_manager/introspection_data/names"
TOPIC_INTROSPECTION_VALUES = "/controller_manager/introspection_data/values"
TOPIC_STATISTICS_FULL = "/controller_manager/statistics/full"
TOPIC_STATISTICS_NAMES = "/controller_manager/statistics/names"
TOPIC_STATISTICS_VALUES = "/controller_manager/statistics/values"
TOPIC_WRIST_IMAGE = "/wrist/image_raw"
TOPIC_WRIST_CAMERA_INFO = "/wrist/camera_info"
TOPIC_WRIST_COMPRESSED = "/wrist/image_raw/compressed"

ACTION_FOLLOW_JOINT_TRAJECTORY = "/joint_trajectory_controller/follow_joint_trajectory"
ACTION_GRIPPER_COMMAND = "/gripper_controller/gripper_cmd"
SERVICE_QUERY_STATE = "/joint_trajectory_controller/query_state"
SERVICE_WRIST_SET_CAPTURE = "/wrist/set_capture"
SERVICE_WRIST_SET_CAMERA_INFO = "/wrist/usb_cam/set_camera_info"

#: `topics` in ros2.yml that are served: name -> (type, direction, rate_hz, node).
#: `rate_hz` is a number for a periodic topic, "latched" or "event" otherwise.
TOPICS: dict[str, tuple[str, str, float | str, str]] = {
    TOPIC_ARM_COMMAND: ("trajectory_msgs/msg/JointTrajectory", "in", "event",
                        NODE_JOINT_TRAJECTORY_CONTROLLER),
    TOPIC_CONTROLLER_STATE: ("control_msgs/msg/JointTrajectoryControllerState", "out", 50,
                             NODE_JOINT_TRAJECTORY_CONTROLLER),
    TOPIC_SPEED_SCALING: ("control_msgs/msg/SpeedScalingFactor", "in", "event",
                          NODE_JOINT_TRAJECTORY_CONTROLLER),
    TOPIC_JOINT_STATES: ("sensor_msgs/msg/JointState", "out", 50, NODE_JOINT_STATE_BROADCASTER),
    TOPIC_DYNAMIC_JOINT_STATES: ("control_msgs/msg/DynamicJointState", "out", 50,
                                 NODE_JOINT_STATE_BROADCASTER),
    TOPIC_ROBOT_DESCRIPTION: ("std_msgs/msg/String", "out", "latched", NODE_ROBOT_STATE_PUBLISHER),
    TOPIC_TF: ("tf2_msgs/msg/TFMessage", "out", 20, NODE_ROBOT_STATE_PUBLISHER),
    TOPIC_TF_STATIC: ("tf2_msgs/msg/TFMessage", "out", "latched", NODE_ROBOT_STATE_PUBLISHER),
    TOPIC_CM_ACTIVITY: ("controller_manager_msgs/msg/ControllerManagerActivity", "out",
                        "latched", NODE_CONTROLLER_MANAGER),
    TOPIC_DIAGNOSTICS: ("diagnostic_msgs/msg/DiagnosticArray", "out", 1, NODE_CONTROLLER_MANAGER),
    TOPIC_INTROSPECTION_FULL: ("pal_statistics_msgs/msg/Statistics", "out", 50,
                               NODE_CONTROLLER_MANAGER),
    TOPIC_INTROSPECTION_NAMES: ("pal_statistics_msgs/msg/StatisticsNames", "out", "event",
                                NODE_CONTROLLER_MANAGER),
    TOPIC_INTROSPECTION_VALUES: ("pal_statistics_msgs/msg/StatisticsValues", "out", 50,
                                 NODE_CONTROLLER_MANAGER),
    TOPIC_STATISTICS_FULL: ("pal_statistics_msgs/msg/Statistics", "out", 50,
                            NODE_CONTROLLER_MANAGER),
    TOPIC_STATISTICS_NAMES: ("pal_statistics_msgs/msg/StatisticsNames", "out", "event",
                             NODE_CONTROLLER_MANAGER),
    TOPIC_STATISTICS_VALUES: ("pal_statistics_msgs/msg/StatisticsValues", "out", 50,
                              NODE_CONTROLLER_MANAGER),
    TOPIC_WRIST_IMAGE: ("sensor_msgs/msg/Image", "out", 30, NODE_WRIST_CAMERA),
    TOPIC_WRIST_CAMERA_INFO: ("sensor_msgs/msg/CameraInfo", "out", 30, NODE_WRIST_CAMERA),
    TOPIC_WRIST_COMPRESSED: ("sensor_msgs/msg/CompressedImage", "out", 30, NODE_WRIST_CAMERA),
}

#: ros2.yml topics marked `unverified: true` that are optional `image_transport_plugins`
#: outputs, and are not served.
OMITTED_TOPICS: tuple[str, ...] = (
    "/wrist/image_raw/compressedDepth",
    "/wrist/image_raw/theora",
    "/wrist/image_raw/zstd",
)

#: `services` in ros2.yml: name -> (type, node).
SERVICES: dict[str, tuple[str, str]] = {
    **{f"/controller_manager/{name}": (f"controller_manager_msgs/srv/{stype}",
                                       NODE_CONTROLLER_MANAGER)
       for name, stype in (
           ("list_controllers", "ListControllers"),
           ("list_controller_types", "ListControllerTypes"),
           ("load_controller", "LoadController"),
           ("configure_controller", "ConfigureController"),
           ("reload_controller_libraries", "ReloadControllerLibraries"),
           ("switch_controller", "SwitchController"),
           ("unload_controller", "UnloadController"),
           ("cleanup_controller", "CleanupController"),
           ("list_hardware_components", "ListHardwareComponents"),
           ("list_hardware_interfaces", "ListHardwareInterfaces"),
           ("set_hardware_component_state", "SetHardwareComponentState"),
       )},
    SERVICE_QUERY_STATE: ("control_msgs/srv/QueryTrajectoryState",
                          NODE_JOINT_TRAJECTORY_CONTROLLER),
    SERVICE_WRIST_SET_CAPTURE: ("std_srvs/srv/SetBool", NODE_WRIST_CAMERA),
    SERVICE_WRIST_SET_CAMERA_INFO: ("sensor_msgs/srv/SetCameraInfo", NODE_WRIST_CAMERA),
}

#: `actions` in ros2.yml: name -> (type, node).
ACTIONS: dict[str, tuple[str, str]] = {
    ACTION_FOLLOW_JOINT_TRAJECTORY: ("control_msgs/action/FollowJointTrajectory",
                                     NODE_JOINT_TRAJECTORY_CONTROLLER),
    ACTION_GRIPPER_COMMAND: ("control_msgs/action/ParallelGripperCommand",
                             NODE_GRIPPER_CONTROLLER),
}

#: `parameters` in ros2.yml that are ROS parameters: (node, name) -> value.
#: `robot_description` is filled with the served description at attach time.
PARAMETERS: dict[tuple[str, str], object] = {
    (NODE_CONTROLLER_MANAGER, "update_rate"): 50,
    (NODE_CONTROLLER_MANAGER, "use_sim_time"): False,
    (NODE_CONTROLLER_MANAGER, "joint_trajectory_controller.type"):
        "joint_trajectory_controller/JointTrajectoryController",
    (NODE_CONTROLLER_MANAGER, "joint_state_broadcaster.type"):
        "joint_state_broadcaster/JointStateBroadcaster",
    (NODE_CONTROLLER_MANAGER, "forward_position_controller.type"):
        "position_controllers/JointGroupPositionController",
    (NODE_CONTROLLER_MANAGER, "gripper_controller.type"):
        "parallel_gripper_action_controller/GripperActionController",
    (NODE_JOINT_TRAJECTORY_CONTROLLER, "joints"): list(ARM_JOINTS),
    (NODE_JOINT_TRAJECTORY_CONTROLLER, "command_interfaces"): ["position"],
    (NODE_JOINT_TRAJECTORY_CONTROLLER, "state_interfaces"): ["position", "velocity"],
    (NODE_JOINT_TRAJECTORY_CONTROLLER, "allow_partial_joints_goal"): False,
    (NODE_JOINT_TRAJECTORY_CONTROLLER, "action_monitor_rate"): 20.0,
    (NODE_JOINT_TRAJECTORY_CONTROLLER, "constraints.goal_time"): 0.0,
    (NODE_JOINT_TRAJECTORY_CONTROLLER, "constraints.decelerate_on_cancel"): False,
    (NODE_GRIPPER_CONTROLLER, "joint"): GRIPPER_JOINT,
    (NODE_GRIPPER_CONTROLLER, "allow_stalling"): True,
    (NODE_GRIPPER_CONTROLLER, "goal_tolerance"): 0.01,
    (NODE_GRIPPER_CONTROLLER, "stall_velocity_threshold"): 0.001,
    (NODE_GRIPPER_CONTROLLER, "stall_timeout"): 1.0,
    (NODE_GRIPPER_CONTROLLER, "max_effort_interface"): "",
    (NODE_JOINT_STATE_BROADCASTER, "use_local_topics"): False,
    (NODE_JOINT_STATE_BROADCASTER, "publish_dynamic_joint_states"): True,
    (NODE_ROBOT_STATE_PUBLISHER, "robot_description"): None,
    (NODE_ROBOT_STATE_PUBLISHER, "publish_frequency"): 20.0,
    (NODE_ROBOT_STATE_PUBLISHER, "frame_prefix"): "",
    (NODE_WRIST_CAMERA, "video_device"): "/dev/video0",
    (NODE_WRIST_CAMERA, "framerate"): 30.0,
    (NODE_WRIST_CAMERA, "image_width"): 640,
    (NODE_WRIST_CAMERA, "image_height"): 480,
    (NODE_WRIST_CAMERA, "pixel_format"): "yuyv",
    (NODE_WRIST_CAMERA, "frame_id"): "default_cam",
    (NODE_WRIST_CAMERA, "camera_name"): "default_cam",
    (NODE_WRIST_CAMERA, "camera_info_url"): "",
    (NODE_WRIST_CAMERA, "io_method"): "mmap",
    (NODE_WRIST_CAMERA, "av_device_format"): "YUV422P",
    (NODE_WRIST_CAMERA, "brightness"): 50,
    (NODE_WRIST_CAMERA, "contrast"): -1,
    (NODE_WRIST_CAMERA, "saturation"): -1,
    (NODE_WRIST_CAMERA, "sharpness"): -1,
    (NODE_WRIST_CAMERA, "gain"): -1,
    (NODE_WRIST_CAMERA, "auto_white_balance"): True,
    (NODE_WRIST_CAMERA, "white_balance"): 4000,
    (NODE_WRIST_CAMERA, "autoexposure"): True,
    (NODE_WRIST_CAMERA, "exposure"): 100,
    (NODE_WRIST_CAMERA, "autofocus"): False,
    (NODE_WRIST_CAMERA, "focus"): -1,
}

UPDATE_RATE_HZ = 50.0          # /controller_manager update_rate
PUBLISH_FREQUENCY_HZ = 20.0    # /robot_state_publisher publish_frequency
ACTION_MONITOR_RATE_HZ = 20.0  # /joint_trajectory_controller action_monitor_rate
WRIST_FRAMERATE_HZ = 30.0      # /wrist/usb_cam framerate
DIAGNOSTICS_PERIOD_S = 1.0
WRIST_SIZE = (640, 480)        # image_width, image_height
WRIST_FRAME_ID = "default_cam"
#: The MJCF camera the wrist driver renders (a simulator addition on the `gripper` body).
WRIST_MJCF_CAMERA = "wrist_cam"
#: usb_cam 0.8.1 with pixel_format `yuyv` publishes YUYV bytes as `yuv422_yuy2`.
WRIST_ENCODING = "yuv422_yuy2"

#: The SO-101's published physical figures (spec §3, real-robot fidelity), each measured
#: on the compiled model by `shared/tests/physical_figures_check.py`. TheRobotStudio
#: publish no dimensions, masses or joint ranges for the arm beyond its URDF and MJCF,
#: which the same check holds the compiled model to joint for joint and body for body.
#:
#: Published but not a figure here: the follower's STS3215 stall torque, "16.5kg.cm at
#: 6V" for the 7.4V servo the bill of materials lists (the 12V one: 30 kg.cm). The
#: official MJCF's actuators take their motor parameters from the Open Duck Mini project
#: and limit every joint to 3.35 N m (34 kg.cm); they are the official model's, not a
#: simulator addition, so the model follows the MJCF and the conflict is reported instead.
PHYSICAL_FIGURES: tuple[Figure, ...] = (
    Figure("servo_joints", "joints, one STS3215 servo each", 6, "", 0,
           "https://huggingface.co/docs/lerobot/so101",
           "The follower arm uses 6x STS3215 motors with 1/345 gearing.",
           "hinge joints of the compiled arm, each driven by exactly one actuator"),
)

HARDWARE_COMPONENT = "SO_ARM101"
HARDWARE_PLUGIN = "feetech_ros2_driver/FeetechHardwareInterface"
USB_PORT = "/dev/LeRobotFollower"

#: Controllers the boot launch spawns (and activates), and their types.
BOOT_CONTROLLERS: dict[str, str] = {
    "joint_state_broadcaster": "joint_state_broadcaster/JointStateBroadcaster",
    "joint_trajectory_controller": "joint_trajectory_controller/JointTrajectoryController",
    "gripper_controller": "parallel_gripper_action_controller/GripperActionController",
}
#: Declared in ros2_controllers.yaml but not spawned at boot.
DECLARED_CONTROLLERS: dict[str, str] = {
    **BOOT_CONTROLLERS,
    "forward_position_controller": "position_controllers/JointGroupPositionController",
}
TOPIC_FORWARD_COMMANDS = "/forward_position_controller/commands"

#: The workspace-owned reset (spec §3), provided by node `/simulator`.
SERVICE_RESET = "/reset"
SRV_TYPE_TRIGGER = "std_srvs/srv/Trigger"

#: The gripper joint's zero. The bringup's description bakes the jaw's closed stop into
#: `gripper_joint`'s origin, so its 0 is the official MJCF/URDF `gripper` hinge at
#: -0.174533 rad: an exact offset between the two, never a rescale. The joint range on the
#: wire is the official hinge range shifted by it.
GRIPPER_OFFSET_RAD = 0.174533
GRIPPER_MJCF_RANGE = (-0.174533, 1.74533)
GRIPPER_RANGE = (GRIPPER_MJCF_RANGE[0] + GRIPPER_OFFSET_RAD,
                 GRIPPER_MJCF_RANGE[1] + GRIPPER_OFFSET_RAD)

#: Official URDF joint -> the bringup's joint name.
URDF_JOINTS = {name.removesuffix("_joint"): name for name in JOINT_ORDER}
URDF_GRIPPER = "gripper"

# ------------------------------------------------------------------ the worktop rig

#: The worktop's fixed camera rig, a member of its own under this namespace (spec §3).
SCENE_NAMESPACE = "scene"
#: rig topic -> (MJCF camera, width, height). Sizes and poses are the constants in
#: `shared/tasks/apple_on_plate.py` (`SCENE_CAMERAS`); these must agree with them.
SCENE_CAMERA_TOPICS: dict[str, tuple[str, int, int]] = {
    "/overhead/color/compressed": ("overhead", 640, 480),
    "/side/color/compressed": ("side", 640, 480),
}
SCENE_CAMERA_INFO_TOPICS: dict[str, str] = {
    "/overhead/color/compressed": "/overhead/color/camera_info",
    "/side/color/compressed": "/side/color/camera_info",
}
SCENE_TF_STATIC = "/tf_static"
#: The rig's root frame: the arm base frame the task's constants are written in.
SCENE_ROOT_FRAME = "worktop"
TYPE_SCENE_IMAGE = "sensor_msgs/msg/CompressedImage"
TYPE_SCENE_CAMERA_INFO = "sensor_msgs/msg/CameraInfo"


def to_wire_gripper(mjcf_rad: float) -> float:
    """Official `gripper` hinge angle -> `gripper_joint` position (rad)."""
    return float(mjcf_rad) + GRIPPER_OFFSET_RAD


def to_mjcf_gripper(wire_rad: float) -> float:
    """`gripper_joint` position (rad) -> official hinge angle, clamped to its range."""
    low, high = GRIPPER_MJCF_RANGE
    return float(min(max(float(wire_rad) - GRIPPER_OFFSET_RAD, low), high))


# ------------------------------------------------------------------ the description


def _rpy_matrix(roll: float, pitch: float, yaw: float):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _matrix_rpy(m) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -m[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        return math.atan2(m[2][1], m[2][2]), pitch, math.atan2(m[1][0], m[0][0])
    return math.atan2(-m[1][2], m[1][1]), pitch, 0.0


def _ros2_control_block() -> ET.Element:
    """`so_arm101.ros2_control.xacro` expanded for hardware_type:=real (ros2.yml)."""
    root = ET.Element("ros2_control", name=HARDWARE_COMPONENT, type="system")
    hardware = ET.SubElement(root, "hardware")
    ET.SubElement(hardware, "plugin").text = HARDWARE_PLUGIN
    ET.SubElement(hardware, "param", name="usb_port").text = USB_PORT
    for servo_id, name in enumerate(JOINT_ORDER, start=1):
        joint = ET.SubElement(root, "joint", name=name)
        ET.SubElement(joint, "param", name="id").text = str(servo_id)
        ET.SubElement(joint, "command_interface", name="position")
        ET.SubElement(joint, "state_interface", name="position")
        ET.SubElement(joint, "state_interface", name="velocity")
    return root


def bringup_description(official_urdf: str) -> str:
    """The description the SO-101 serves, from the official `so101_new_calib.urdf`.

    Geometry, links and limits are the official file's; what the bringup adds is applied:
    its joint names (`<joint>_joint`), the `world` root with a fixed `world_to_base_joint`
    (ros2.yml's `/tf_static`: world->base_link), the jaw's zero at the closed stop (the
    origin turned by `GRIPPER_OFFSET_RAD` about the joint axis, the limits shifted by the
    same, so every pose is unchanged), and the ros2_control block for real hardware.
    """
    root = ET.fromstring(official_urdf)
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name in URDF_JOINTS:
            joint.set("name", URDF_JOINTS[name])
    for joint in root.findall("joint"):
        if joint.get("name") != GRIPPER_JOINT:
            continue
        origin = joint.find("origin")
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        m = _rpy_matrix(*rpy)
        c, s = math.cos(-GRIPPER_OFFSET_RAD), math.sin(-GRIPPER_OFFSET_RAD)
        rz = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        turned = [[sum(m[i][k] * rz[k][j] for k in range(3)) for j in range(3)]
                  for i in range(3)]
        origin.set("rpy", " ".join(f"{v:.9g}" for v in _matrix_rpy(turned)))
        limit = joint.find("limit")
        limit.set("lower", f"{float(limit.get('lower')) + GRIPPER_OFFSET_RAD:.6g}")
        limit.set("upper", f"{float(limit.get('upper')) + GRIPPER_OFFSET_RAD:.6g}")
    world = ET.Element("link", name="world")
    fixed = ET.Element("joint", name="world_to_base_joint", type="fixed")
    ET.SubElement(fixed, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(fixed, "parent", link="world")
    ET.SubElement(fixed, "child", link="base_link")
    root.insert(0, fixed)
    root.insert(0, world)
    root.append(_ros2_control_block())
    ET.indent(root)
    return '<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n"


#: MJCF body -> the description's link, for checks that compare the two trees.
MJCF_BODY_LINKS = {
    "base": "base_link",
    "shoulder": "shoulder_link",
    "upper_arm": "upper_arm_link",
    "lower_arm": "lower_arm_link",
    "wrist": "wrist_link",
    "gripper": "gripper_link",
    "moving_jaw_so101_v1": "moving_jaw_so101_v1_link",
}


# ------------------------------------------------------------------ trajectories


def _dur(seconds: float) -> dict:
    sec = math.floor(seconds)
    return {"sec": int(sec), "nanosec": int(round((seconds - sec) * 1e9)) % 1_000_000_000}


def _seconds(stamp) -> float:
    if not isinstance(stamp, dict):
        return 0.0
    sec = stamp.get("sec", stamp.get("secs", 0)) or 0
    nsec = stamp.get("nanosec", stamp.get("nsecs", 0)) or 0
    return float(sec) + float(nsec) * 1e-9


def _point(positions=(), velocities=(), accelerations=(), tfs: float = 0.0) -> dict:
    return {"positions": [float(v) for v in positions],
            "velocities": [float(v) for v in velocities],
            "accelerations": [float(v) for v in accelerations],
            "effort": [], "time_from_start": _dur(max(tfs, 0.0))}


def validate_trajectory(msg: dict, now: float) -> str | None:
    """`JointTrajectoryController::validate_trajectory_msg` (4.42.1); the error, or None."""
    names = list(msg.get("joint_names") or [])
    points = list(msg.get("points") or [])
    if len(names) != len(ARM_JOINTS):
        return "Joints on incoming trajectory don't match the controller joints."
    if not names:
        return "Empty joint names on incoming trajectory."
    if not points:
        return "Empty trajectory received."
    for name in names:
        if name not in ARM_JOINTS:
            return f"Incoming joint {name} doesn't match the controller's joints."
    for i, v in enumerate((points[-1] or {}).get("velocities") or []):
        if abs(float(v)) > 1.1920929e-07:
            return (f"Velocity of last trajectory point of joint {names[i]} is not zero: "
                    f"{float(v):.15f}")
    start = _seconds((msg.get("header") or {}).get("stamp"))
    if start != 0.0:
        end = start + _seconds((points[-1] or {}).get("time_from_start"))
        if end < now:
            return (f"Received trajectory with non-zero start time ({start:f}) that ends "
                    f"in the past ({end:f})")
    previous = 0.0
    for i, point in enumerate(points):
        point = point or {}
        t = _seconds(point.get("time_from_start"))
        if i > 0 and t <= previous:
            return (f"Time between points {i - 1} and {i} is not strictly increasing, it is "
                    f"{previous:f} and {t:f} respectively")
        previous = t
        for field, allow_empty in (("positions", False), ("velocities", True),
                                   ("accelerations", True)):
            values = point.get(field) or []
            if (allow_empty and not values):
                continue
            if len(values) != len(names):
                return (f"Mismatch between joint_names size ({len(names)}) and {field} "
                        f"({len(values)}) at point #{i}.")
        if point.get("effort"):
            return ("Trajectories with effort fields are only supported for controllers "
                    "using the 'effort' command interface.")
    return None


class Trajectory:
    """A validated JointTrajectory in controller joint order, sampled as JTC samples it.

    Times are absolute simulated seconds. The segment before the first point starts from
    the state the controller measured when it first sampled (`set_point_before`), and the
    interpolation per segment is linear, cubic or quintic by what both ends carry.
    """

    def __init__(self, msg: dict | None, hold=None) -> None:
        self.holding = hold is not None
        if hold is not None:
            self.stamp = 0.0
            self.points = [(0.0, np.asarray(hold, dtype=float), np.zeros(len(hold)), None)]
        else:
            names = list(msg["joint_names"])
            order = [names.index(j) for j in ARM_JOINTS]
            self.stamp = _seconds((msg.get("header") or {}).get("stamp"))
            self.points = []
            for p in msg["points"]:
                pos = np.asarray(p["positions"], dtype=float)[order]
                vel = p.get("velocities") or []
                acc = p.get("accelerations") or []
                self.points.append((
                    _seconds(p.get("time_from_start")),
                    pos,
                    np.asarray(vel, dtype=float)[order] if vel else None,
                    np.asarray(acc, dtype=float)[order] if acc else None,
                ))
        self.start: float | None = None
        self.before = None

    def set_point_before(self, now: float, positions, velocities) -> None:
        self.start = self.stamp if self.stamp != 0.0 else now
        self.before = (now, np.asarray(positions, dtype=float),
                       np.asarray(velocities, dtype=float), None)

    @property
    def end(self) -> float:
        return (self.start or 0.0) + self.points[-1][0]

    def sample(self, t: float):
        """(ok, positions, velocities, accelerations, past_end); ok False before start."""
        n = len(ARM_JOINTS)
        if self.start is None or self.before is None:
            return False, None, None, None, False
        if t < self.before[0]:
            return False, None, None, None, False
        stops = [(self.before[0], self.before[1], self.before[2], self.before[3])]
        stops += [(self.start + tfs, pos, vel, acc) for tfs, pos, vel, acc in self.points]
        for (ta, pa, va, aa), (tb, pb, vb, ab) in zip(stops, stops[1:]):
            if t < tb:
                return (True, *_interpolate(ta, pa, va, aa, tb, pb, vb, ab, t), False)
        _, pos, _, _ = stops[-1]
        return True, pos.copy(), np.zeros(n), np.zeros(n), True


def _interpolate(ta, pa, va, aa, tb, pb, vb, ab, t):
    dt = tb - ta
    s = min(max(t - ta, 0.0), dt)
    n = len(pa)
    if dt <= 0:
        return pb.copy(), np.zeros(n), np.zeros(n)
    if va is None or vb is None:
        vel = (pb - pa) / dt
        return pa + vel * s, vel, np.zeros(n)
    if aa is None or ab is None:
        # cubic Hermite
        c0, c1 = pa, va
        c2 = (3 * (pb - pa) - (2 * va + vb) * dt) / dt ** 2
        c3 = (2 * (pa - pb) + (va + vb) * dt) / dt ** 3
        return (c0 + c1 * s + c2 * s ** 2 + c3 * s ** 3,
                c1 + 2 * c2 * s + 3 * c3 * s ** 2, 2 * c2 + 6 * c3 * s)
    c0, c1, c2 = pa, va, aa / 2
    T = dt
    c3 = (20 * (pb - pa) - (8 * vb + 12 * va) * T - (3 * aa - ab) * T ** 2) / (2 * T ** 3)
    c4 = (30 * (pa - pb) + (14 * vb + 16 * va) * T + (3 * aa - 2 * ab) * T ** 2) / (2 * T ** 4)
    c5 = (12 * (pb - pa) - 6 * (vb + va) * T - (aa - ab) * T ** 2) / (2 * T ** 5)
    return (c0 + c1 * s + c2 * s ** 2 + c3 * s ** 3 + c4 * s ** 4 + c5 * s ** 5,
            c1 + 2 * c2 * s + 3 * c3 * s ** 2 + 4 * c4 * s ** 3 + 5 * c5 * s ** 4,
            2 * c2 + 6 * c3 * s + 12 * c4 * s ** 2 + 20 * c5 * s ** 3)


# ------------------------------------------------------------------ the surface


def _header(frame_id: str, stamp_s: float) -> dict:
    return {"stamp": _dur(stamp_s), "frame_id": frame_id}


def _lifecycle(state: str) -> dict:
    ids = {"unknown": 0, "unconfigured": 1, "inactive": 2, "active": 3, "finalized": 4}
    return {"id": ids.get(state, 0), "label": state}


def _uncalibrated_camera_info(width: int, height: int) -> dict:
    """What camera_info_manager holds with no calibration URL: sizes, and zeros."""
    return {"header": _header("", 0.0), "height": height, "width": width,
            "distortion_model": "", "d": [], "k": [0.0] * 9, "r": [0.0] * 9,
            "p": [0.0] * 12, "binning_x": 0, "binning_y": 0,
            "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0,
                    "do_rectify": False}}


def rgb_to_yuyv(rgb) -> bytes:
    """RGB uint8 HxWx3 -> YUYV (YUV 4:2:2, BT.601 limited range) bytes."""
    img = np.asarray(rgb, dtype=np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 16 + 0.257 * r + 0.504 * g + 0.098 * b
    u = 128 - 0.148 * r - 0.291 * g + 0.439 * b
    v = 128 + 0.439 * r - 0.368 * g - 0.071 * b
    out = np.empty((img.shape[0], img.shape[1] * 2), dtype=np.float32)
    out[:, 0::4] = y[:, 0::2]
    out[:, 1::4] = (u[:, 0::2] + u[:, 1::2]) / 2
    out[:, 2::4] = y[:, 1::2]
    out[:, 3::4] = (v[:, 0::2] + v[:, 1::2]) / 2
    return np.clip(np.rint(out), 0, 255).astype(np.uint8).tobytes()


def attach_ros(bus, view, model, task=None, *, cameras=None, jpeg_quality: int = 70,
               control_hz: float = UPDATE_RATE_HZ, scene_option=None, world_reset=None,
               prefix: str = "", wrist: bool = True):
    """Present the SO-101's interface on `bus`; return the step the fleet drives.

    The step carries `rate_hz` = the controller manager's `update_rate`, so the fleet
    calls it at 50 Hz whatever `--control-hz` is; everything slower (tf, camera,
    diagnostics) is throttled inside it on simulated time. `view` is the engine's move
    groups (`arm` and `gripper`, each with `joint_pos`/`joint_vel`/writable `ctrl`).
    `cameras` is accepted for the engines' call shape and ignored: the wrist camera is
    part of the robot, and `wrist=False` exists only for tools that render nothing.
    """
    del cameras, control_hz
    import base64

    import robots_spec
    from contracts.rosbridge_server import SIMULATOR_NODE, TYPE_FLOAT64_MULTI_ARRAY
    from ros_surfaces.tf_stream import RobotStatePublisher, read_description

    groups = view if isinstance(view, dict) else {
        gid: view.get_move_group(gid) for gid in ("arm", "gripper")
    }
    arm, gripper = groups["arm"], groups["gripper"]
    n_arm = len(ARM_JOINTS)

    def measured():
        pos = np.asarray(arm.joint_pos, dtype=np.float64).reshape(-1)
        vel = np.asarray(arm.joint_vel, dtype=np.float64).reshape(-1)
        gpos = to_wire_gripper(float(np.asarray(gripper.joint_pos).reshape(-1)[0]))
        gvel = float(np.asarray(gripper.joint_vel).reshape(-1)[0])
        return pos, vel, gpos, gvel

    lock = threading.RLock()
    pos0, _, gpos0, _ = measured()

    # ---- controller manager state -------------------------------------------------
    st = {
        "hw_state": "active",
        "controllers": {name: "active" for name in BOOT_CONTROLLERS},
        "arm_cmd": pos0.copy(),          # the position command interfaces, wire units
        "gripper_cmd": gpos0,
        "traj": Trajectory(None, hold=pos0),
        "new_traj": None,
        "traj_time": None,
        "jtc_goal": None,                # (goal, tolerances)
        "gripper_goal": None,            # (goal, last_movement_time)
        "scaling": 1.0,
        "forward_cmd": None,
        "desired": None, "error": None,
        "capture": True,
        "camera_info": _uncalibrated_camera_info(*WRIST_SIZE),
        "activity_dirty": True,
        "forward_sub": False,
    }

    def active(name: str) -> bool:
        return st["controllers"].get(name) == "active" and st["hw_state"] == "active"

    # ---- robot_state_publisher -----------------------------------------------------
    description = bringup_description(read_description(robots_spec.urdf_path("so101")))
    rsp = RobotStatePublisher(
        bus, description, node=NODE_ROBOT_STATE_PUBLISHER,
        topic_description=TOPIC_ROBOT_DESCRIPTION,
        type_description=TOPICS[TOPIC_ROBOT_DESCRIPTION][0],
        param_description=None, publish_frequency=PUBLISH_FREQUENCY_HZ,
    )
    for (node, name), value in PARAMETERS.items():
        bus.set_param(f"{node}:{name}", description if value is None else value)

    # ---- declared publications (so discovery lists them before the first message) --
    for topic, (mtype, direction, _rate, node) in TOPICS.items():
        if direction == "out":
            bus.advertise(topic, mtype, node=node)

    # ---- joint_trajectory_controller -----------------------------------------------
    def on_trajectory(msg: dict) -> None:
        with lock:
            if validate_trajectory(msg, bus.server.now()) is not None:
                return
            if st["controllers"].get("joint_trajectory_controller") != "active":
                return
            st["new_traj"] = Trajectory(msg)

    def on_speed_scaling(msg: dict) -> None:
        factor = float((msg or {}).get("factor", 1.0))
        if factor >= 0:
            with lock:
                st["scaling"] = factor

    bus.on(TOPIC_ARM_COMMAND, on_trajectory, TOPICS[TOPIC_ARM_COMMAND][0],
           node=NODE_JOINT_TRAJECTORY_CONTROLLER)
    bus.on(TOPIC_SPEED_SCALING, on_speed_scaling, TOPICS[TOPIC_SPEED_SCALING][0],
           node=NODE_JOINT_TRAJECTORY_CONTROLLER)

    def hold_arm() -> None:
        pos, *_ = measured()
        st["new_traj"] = Trajectory(None, hold=pos)

    def tolerances(goal_args: dict) -> dict:
        # Defaults (no constraints set): goal position unchecked, stopped velocity 0.01.
        tol = {"path": {j: (0.0, 0.0) for j in ARM_JOINTS},
               "goal": {j: (0.0, 0.01) for j in ARM_JOINTS},
               "goal_time": 0.0}
        for key in ("path", "goal"):
            for entry in goal_args.get(f"{key}_tolerance") or []:
                name = entry.get("name")
                if name not in ARM_JOINTS:
                    continue
                pos_t, vel_t = tol[key][name]
                p, v = float(entry.get("position", 0.0)), float(entry.get("velocity", 0.0))
                pos_t = 0.0 if p < 0 else (p if p > 0 else pos_t)
                vel_t = 0.0 if v < 0 else (v if v > 0 else vel_t)
                tol[key][name] = (pos_t, vel_t)
        gtt = _seconds(goal_args.get("goal_time_tolerance"))
        if gtt:
            tol["goal_time"] = gtt
        return tol

    def jtc_accept(args: dict) -> bool:
        with lock:
            if st["controllers"].get("joint_trajectory_controller") != "active":
                return False
            return validate_trajectory(args.get("trajectory") or {},
                                       bus.server.now()) is None

    def jtc_execute(goal):
        with lock:
            previous = st["jtc_goal"]
            if previous is not None:
                previous[0].abort({"error_code": -1, "error_string":
                                   "Current goal preempted by new incoming action."})
            st["new_traj"] = Trajectory(goal.args["trajectory"])
            st["jtc_goal"] = (goal, tolerances(goal.args))
        while not goal.done:
            if goal.wait_for_cancel(0.02):
                with lock:
                    if st["jtc_goal"] is not None and st["jtc_goal"][0] is goal:
                        st["jtc_goal"] = None
                        hold_arm()
                return {"error_code": 0, "error_string": ""}
        return None

    bus.action(ACTION_FOLLOW_JOINT_TRAJECTORY, ACTIONS[ACTION_FOLLOW_JOINT_TRAJECTORY][0],
               jtc_execute, accept=jtc_accept, node=NODE_JOINT_TRAJECTORY_CONTROLLER)

    def query_state(args: dict) -> dict:
        with lock:
            response = {"success": False, "message": "", "name": list(ARM_JOINTS),
                        "position": [], "velocity": [], "acceleration": []}
            if st["controllers"].get("joint_trajectory_controller") != "active":
                return {**response, "name": []}
            pos, vel, *_ = measured()
            response.update(position=pos.tolist(), velocity=vel.tolist(),
                            acceleration=[0.0] * n_arm)
            traj = st["traj"]
            ok, p, v, a, past_end = traj.sample(_seconds(args.get("time")))
            if ok and not past_end:
                response.update(success=True, position=p.tolist(), velocity=v.tolist(),
                                acceleration=a.tolist())
            return response

    bus.service(SERVICE_QUERY_STATE, query_state, SERVICES[SERVICE_QUERY_STATE][0],
                node=NODE_JOINT_TRAJECTORY_CONTROLLER)

    # ---- gripper_controller ----------------------------------------------------------
    def gripper_accept(args: dict) -> bool:
        with lock:
            if st["controllers"].get("gripper_controller") != "active":
                return False
        return len(((args.get("command") or {}).get("position")) or []) == 1

    def gripper_execute(goal):
        with lock:
            previous = st["gripper_goal"]
            if previous is not None:
                previous[0].canceled({})
            st["gripper_cmd"] = float(goal.args["command"]["position"][0])
            st["gripper_goal"] = (goal, bus.server.now())
        while not goal.done:
            if goal.wait_for_cancel(0.02):
                with lock:
                    if st["gripper_goal"] is not None and st["gripper_goal"][0] is goal:
                        st["gripper_goal"] = None
                        st["gripper_cmd"] = measured()[2]
                return {}
        return None

    bus.action(ACTION_GRIPPER_COMMAND, ACTIONS[ACTION_GRIPPER_COMMAND][0], gripper_execute,
               accept=gripper_accept, node=NODE_GRIPPER_CONTROLLER)

    # ---- controller_manager services -----------------------------------------------
    interfaces_of = {
        "joint_trajectory_controller": [f"{j}/position" for j in ARM_JOINTS],
        "forward_position_controller": [f"{j}/position" for j in ARM_JOINTS],
        "gripper_controller": [f"{GRIPPER_JOINT}/position"],
        "joint_state_broadcaster": [],
    }
    state_interfaces_of = {
        "joint_trajectory_controller": [f"{j}/{i}" for j in ARM_JOINTS
                                        for i in ("position", "velocity")],
        "forward_position_controller": [],
        "gripper_controller": [f"{GRIPPER_JOINT}/position", f"{GRIPPER_JOINT}/velocity"],
        "joint_state_broadcaster": [f"{j}/{i}" for j in JOINT_ORDER
                                    for i in ("position", "velocity")],
    }

    def claimed() -> dict[str, str]:
        out = {}
        for name, state in st["controllers"].items():
            if state == "active":
                for iface in interfaces_of[name]:
                    out[iface] = name
        return out

    def list_controllers(_args: dict) -> dict:
        with lock:
            return {"controller": [{
                "name": name, "state": state, "type": DECLARED_CONTROLLERS[name],
                "is_async": False, "update_rate": int(UPDATE_RATE_HZ),
                "claimed_interfaces": interfaces_of[name] if state == "active" else [],
                "required_command_interfaces": interfaces_of[name],
                "required_state_interfaces": state_interfaces_of[name],
                "is_chainable": False, "is_chained": False,
                "exported_state_interfaces": [], "reference_interfaces": [],
                "chain_connections": [],
            } for name, state in st["controllers"].items()]}

    def list_controller_types(_args: dict) -> dict:
        types = sorted({*DECLARED_CONTROLLERS.values(),
                        "forward_command_controller/ForwardCommandController",
                        "position_controllers/JointGroupPositionController",
                        "velocity_controllers/JointGroupVelocityController",
                        "effort_controllers/JointGroupEffortController",
                        "joint_trajectory_controller/JointTrajectoryController",
                        "position_controllers/GripperActionController",
                        "admittance_controller/AdmittanceController",
                        "pid_controller/PidController"})
        base = {"pid_controller/PidController", "admittance_controller/AdmittanceController"}
        return {"types": types,
                "base_classes": ["controller_interface::ChainableControllerInterface"
                                 if t in base else "controller_interface::ControllerInterface"
                                 for t in types]}

    def set_controller(name: str, state: str | None) -> None:
        if state is None:
            st["controllers"].pop(name, None)
        else:
            st["controllers"][name] = state
        st["activity_dirty"] = True

    def load_controller(args: dict) -> dict:
        name = str(args.get("name", ""))
        with lock:
            if name in st["controllers"] or name not in DECLARED_CONTROLLERS:
                return {"ok": False}
            set_controller(name, "unconfigured")
        return {"ok": True}

    def configure_controller(args: dict) -> dict:
        name = str(args.get("name", ""))
        with lock:
            if st["controllers"].get(name) not in ("unconfigured", "inactive"):
                return {"ok": False}
            set_controller(name, "inactive")
            if name == "forward_position_controller" and not st["forward_sub"]:
                # Created in on_configure, so it exists from here on.
                st["forward_sub"] = True
                bus.on(TOPIC_FORWARD_COMMANDS, on_forward, TYPE_FLOAT64_MULTI_ARRAY,
                       node="/forward_position_controller")
        return {"ok": True}

    def on_forward(msg: dict) -> None:
        data = list((msg or {}).get("data") or [])
        with lock:
            if st["controllers"].get("forward_position_controller") == "active" \
                    and len(data) == n_arm:
                st["forward_cmd"] = np.asarray(data, dtype=float)

    def activate(name: str) -> None:
        set_controller(name, "active")
        if name == "joint_trajectory_controller":
            hold_arm()
        elif name == "gripper_controller":
            st["gripper_cmd"] = measured()[2]
        elif name == "forward_position_controller":
            st["forward_cmd"] = None

    def deactivate(name: str) -> None:
        set_controller(name, "inactive")
        if name == "joint_trajectory_controller" and st["jtc_goal"] is not None:
            st["jtc_goal"][0].abort({"error_code": -1, "error_string":
                                     "Current goal cancelled during deactivate transition."})
            st["jtc_goal"] = None
        if name == "gripper_controller" and st["gripper_goal"] is not None:
            st["gripper_goal"][0].abort({})
            st["gripper_goal"] = None

    def switch_controller(args: dict) -> dict:
        start = [str(n) for n in args.get("activate_controllers") or []]
        stop = [str(n) for n in args.get("deactivate_controllers") or []]
        strict = int(args.get("strictness", 2) or 2) != 1
        with lock:
            bad = [n for n in start if st["controllers"].get(n) not in ("inactive", "active")]
            bad += [n for n in stop if n not in st["controllers"]]
            if bad and strict:
                return {"ok": False, "message": f"Controllers not available: {bad}"}
            start = [n for n in start if n not in bad and st["controllers"][n] == "inactive"]
            stop = [n for n in stop if n not in bad and st["controllers"][n] == "active"]
            owners = {k: v for k, v in claimed().items() if v not in stop}
            conflicts = [n for n in start
                         if any(i in owners for i in interfaces_of[n])]
            if conflicts:
                if strict:
                    return {"ok": False, "message":
                            f"Resource conflict for controller(s): {conflicts}"}
                start = [n for n in start if n not in conflicts]
            for name in stop:
                deactivate(name)
            for name in start:
                activate(name)
        return {"ok": True, "message": ""}

    def unload_controller(args: dict) -> dict:
        name = str(args.get("name", ""))
        with lock:
            if st["controllers"].get(name) not in ("unconfigured", "inactive"):
                return {"ok": False}
            set_controller(name, None)
        return {"ok": True}

    def cleanup_controller(args: dict) -> dict:
        name = str(args.get("name", ""))
        with lock:
            if st["controllers"].get(name) != "inactive":
                return {"ok": False}
            set_controller(name, "unconfigured")
        return {"ok": True}

    def reload_controller_libraries(args: dict) -> dict:
        with lock:
            if st["controllers"] and not bool(args.get("force_kill", False)):
                return {"ok": False}
            for name in list(st["controllers"]):
                if st["controllers"][name] == "active":
                    deactivate(name)
                set_controller(name, None)
        return {"ok": True}

    def hw_interfaces():
        owners = claimed()
        available = st["hw_state"] == "active"
        cmd = [{"name": f"{j}/position", "data_type": "double", "is_available": available,
                "is_claimed": f"{j}/position" in owners} for j in JOINT_ORDER]
        state = [{"name": f"{j}/{i}", "data_type": "double",
                  "is_available": st["hw_state"] in ("active", "inactive"),
                  "is_claimed": False} for j in JOINT_ORDER for i in ("position", "velocity")]
        return cmd, state

    def list_hardware_components(_args: dict) -> dict:
        with lock:
            cmd, state = hw_interfaces()
            return {"component": [{
                "name": HARDWARE_COMPONENT, "type": "system", "is_async": False,
                "rw_rate": int(UPDATE_RATE_HZ), "class_type": HARDWARE_PLUGIN,
                "plugin_name": HARDWARE_PLUGIN, "state": _lifecycle(st["hw_state"]),
                "command_interfaces": cmd, "state_interfaces": state,
            }]}

    def list_hardware_interfaces(_args: dict) -> dict:
        with lock:
            cmd, state = hw_interfaces()
            return {"command_interfaces": cmd, "state_interfaces": state}

    def set_hardware_component_state(args: dict) -> dict:
        name = str(args.get("name", ""))
        target = args.get("target_state") or {}
        label = str(target.get("label", "")) or {1: "unconfigured", 2: "inactive",
                                                  3: "active"}.get(int(target.get("id", 0)), "")
        with lock:
            if name != HARDWARE_COMPONENT or label not in ("unconfigured", "inactive", "active"):
                return {"ok": False, "state": _lifecycle(st["hw_state"])}
            if label != st["hw_state"]:
                st["hw_state"] = label
                st["activity_dirty"] = True
                if label == "active":
                    # on_activate: the servos take their present position as the command.
                    pos, _, gpos, _ = measured()
                    st["arm_cmd"], st["gripper_cmd"] = pos.copy(), gpos
                    if st["controllers"].get("joint_trajectory_controller") == "active":
                        hold_arm()
            return {"ok": True, "state": _lifecycle(st["hw_state"])}

    for name, handler in (
        ("list_controllers", list_controllers),
        ("list_controller_types", list_controller_types),
        ("load_controller", load_controller),
        ("configure_controller", configure_controller),
        ("reload_controller_libraries", reload_controller_libraries),
        ("switch_controller", switch_controller),
        ("unload_controller", unload_controller),
        ("cleanup_controller", cleanup_controller),
        ("list_hardware_components", list_hardware_components),
        ("list_hardware_interfaces", list_hardware_interfaces),
        ("set_hardware_component_state", set_hardware_component_state),
    ):
        full = f"/controller_manager/{name}"
        bus.service(full, handler, SERVICES[full][0], node=NODE_CONTROLLER_MANAGER)

    # ---- wrist camera (usb_cam) ------------------------------------------------------
    def set_capture(args: dict) -> dict:
        on = bool(args.get("data", False))
        with lock:
            st["capture"] = on
        # usb_cam 0.8.1 sets only the message; `success` keeps its default.
        return {"success": False, "message": "Start Capturing" if on else "Stop Capturing"}

    def set_camera_info(args: dict) -> dict:
        info = args.get("camera_info") or {}
        with lock:
            merged = dict(_uncalibrated_camera_info(*WRIST_SIZE))
            merged.update({k: v for k, v in info.items() if k in merged})
            st["camera_info"] = merged
        return {"success": True, "status_message": ""}

    bus.service(SERVICE_WRIST_SET_CAPTURE, set_capture, SERVICES[SERVICE_WRIST_SET_CAPTURE][0],
                node=NODE_WRIST_CAMERA)
    bus.service(SERVICE_WRIST_SET_CAMERA_INFO, set_camera_info,
                SERVICES[SERVICE_WRIST_SET_CAMERA_INFO][0], node=NODE_WRIST_CAMERA)

    worker = None
    cam_name = f"{prefix}{WRIST_MJCF_CAMERA}"
    if wrist:
        import mujoco

        from mujoco_bridge import RenderWorker

        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) < 0:
            raise SystemExit(f"so101: the wrist camera {cam_name!r} is not in this model")
        # Rendered off the control loop, so the controller manager keeps its 50 Hz.
        worker = RenderWorker(model, name=f"{bus.ns} wrist camera")

    def camera_frame(rgb, stamp: float) -> None:
        """On the render worker's thread: one captured frame, as usb_cam sends it."""
        from mujoco_bridge import _encode_jpeg

        header = _header(bus.frame(WRIST_FRAME_ID), stamp)
        width, height = WRIST_SIZE
        if bus.has_subscribers(TOPIC_WRIST_IMAGE):
            bus.publish(TOPIC_WRIST_IMAGE, {
                "header": header, "height": height, "width": width,
                "encoding": WRIST_ENCODING, "is_bigendian": 0, "step": width * 2,
                "data": base64.b64encode(rgb_to_yuyv(rgb)).decode("ascii"),
            }, TOPICS[TOPIC_WRIST_IMAGE][0], node=NODE_WRIST_CAMERA)
        if bus.has_subscribers(TOPIC_WRIST_CAMERA_INFO):
            with lock:
                info = dict(st["camera_info"])
            info["header"] = header
            bus.publish(TOPIC_WRIST_CAMERA_INFO, info, TOPICS[TOPIC_WRIST_CAMERA_INFO][0],
                        node=NODE_WRIST_CAMERA)
        if bus.has_subscribers(TOPIC_WRIST_COMPRESSED):
            bus.publish(TOPIC_WRIST_COMPRESSED, {
                "header": header,
                "format": f"{WRIST_ENCODING}; jpeg compressed bgr8",
                "data": base64.b64encode(_encode_jpeg(rgb, jpeg_quality)).decode("ascii"),
            }, TOPICS[TOPIC_WRIST_COMPRESSED][0], node=NODE_WRIST_CAMERA)

    def publish_camera(data, stamp: float) -> bool:
        """Hand one frame to the worker. False when it could not take it (busy).

        Queued behind a frame still in progress: this step ticks at 50 Hz, so a 30 Hz
        frame is due 20 ms after the last one two ticks in five, and one render may take
        longer than that with the whole fleet's cameras on the GPU (see `RenderWorker`).
        """
        if worker is None or not any(bus.has_subscribers(t) for t in (
                TOPIC_WRIST_IMAGE, TOPIC_WRIST_COMPRESSED, TOPIC_WRIST_CAMERA_INFO)):
            return True
        return worker.submit(data, stamp, [(cam_name, WRIST_SIZE[0], WRIST_SIZE[1],
                                            scene_option, camera_frame)], queue=True)

    # ---- controller_manager introspection / statistics / activity ------------------
    stat_names = []
    for component in (f"{HARDWARE_COMPONENT}.stats/read_cycle", f"{HARDWARE_COMPONENT}.stats/write_cycle"):
        for kind in ("execution_time", "periodicity"):
            stat_names += [f"{component}/{kind}/{s}" for s in
                           ("max", "min", "average", "standard_deviation", "sample_count",
                            "current_value")]
    for name in BOOT_CONTROLLERS:
        for kind in ("execution_time", "periodicity"):
            stat_names += [f"{name}.stats/{kind}/{s}" for s in
                           ("max", "min", "average", "standard_deviation", "sample_count",
                            "current_value")]
    introspection_names = ([f"state_interface.{j}/{i}" for j in JOINT_ORDER
                            for i in ("position", "velocity")]
                           + [f"command_interface.{j}/position" for j in JOINT_ORDER])
    bus.publish(TOPIC_STATISTICS_NAMES, {"header": _header("", 0.0), "names": stat_names,
                                         "names_version": 1},
                TOPICS[TOPIC_STATISTICS_NAMES][0], latched=True, node=NODE_CONTROLLER_MANAGER)
    bus.publish(TOPIC_INTROSPECTION_NAMES, {"header": _header("", 0.0),
                                            "names": introspection_names, "names_version": 1},
                TOPICS[TOPIC_INTROSPECTION_NAMES][0], latched=True,
                node=NODE_CONTROLLER_MANAGER)
    loop_stats = {"last": None, "periods": [], "exec": []}

    def stat_values(period: float, exec_s: float) -> list[float]:
        periods = loop_stats["periods"][-500:] or [period]
        execs = loop_stats["exec"][-500:] or [exec_s]
        hz = [1.0 / p for p in periods if p > 0] or [UPDATE_RATE_HZ]
        us = [e * 1e6 for e in execs]

        def block(values, current):
            arr = np.asarray(values, dtype=float)
            return [float(arr.max()), float(arr.min()), float(arr.mean()), float(arr.std()),
                    float(arr.size), float(current)]

        per_kind = block(us, exec_s * 1e6) + block(hz, hz[-1])
        return per_kind * (2 + len(BOOT_CONTROLLERS))

    def publish_statistics(stamp: float, period: float, exec_s: float, pos, vel, gpos, gvel):
        header = _header("", stamp)
        stats_topics = (TOPIC_STATISTICS_FULL, TOPIC_STATISTICS_VALUES)
        if any(bus.has_subscribers(t) for t in stats_topics):
            values = stat_values(period, exec_s)
            if bus.has_subscribers(TOPIC_STATISTICS_FULL):
                bus.publish(TOPIC_STATISTICS_FULL, {"header": header, "statistics": [
                    {"name": n, "value": v} for n, v in zip(stat_names, values)]},
                    TOPICS[TOPIC_STATISTICS_FULL][0], node=NODE_CONTROLLER_MANAGER)
            if bus.has_subscribers(TOPIC_STATISTICS_VALUES):
                bus.publish(TOPIC_STATISTICS_VALUES, {"header": header, "values": values,
                                                      "names_version": 1},
                            TOPICS[TOPIC_STATISTICS_VALUES][0], node=NODE_CONTROLLER_MANAGER)
        intro_topics = (TOPIC_INTROSPECTION_FULL, TOPIC_INTROSPECTION_VALUES)
        if any(bus.has_subscribers(t) for t in intro_topics):
            state_vals = []
            for p, v in zip([*pos, gpos], [*vel, gvel]):
                state_vals += [float(p), float(v)]
            values = state_vals + [float(c) for c in st["arm_cmd"]] + [float(st["gripper_cmd"])]
            if bus.has_subscribers(TOPIC_INTROSPECTION_FULL):
                bus.publish(TOPIC_INTROSPECTION_FULL, {"header": header, "statistics": [
                    {"name": n, "value": v} for n, v in zip(introspection_names, values)]},
                    TOPICS[TOPIC_INTROSPECTION_FULL][0], node=NODE_CONTROLLER_MANAGER)
            if bus.has_subscribers(TOPIC_INTROSPECTION_VALUES):
                bus.publish(TOPIC_INTROSPECTION_VALUES, {"header": header, "values": values,
                                                         "names_version": 1},
                            TOPICS[TOPIC_INTROSPECTION_VALUES][0], node=NODE_CONTROLLER_MANAGER)

    def publish_activity(stamp: float) -> None:
        bus.publish(TOPIC_CM_ACTIVITY, {
            "header": _header("", stamp),
            "controllers": [{"name": n, "state": _lifecycle(s)}
                            for n, s in st["controllers"].items()],
            "hardware_components": [{"name": HARDWARE_COMPONENT,
                                     "state": _lifecycle(st["hw_state"])}],
        }, TOPICS[TOPIC_CM_ACTIVITY][0], latched=True, node=NODE_CONTROLLER_MANAGER)

    def publish_diagnostics(stamp: float) -> None:
        controllers = st["controllers"]
        all_active = all(s == "active" for s in controllers.values())
        periods = loop_stats["periods"][-50:]
        hz = (1.0 / float(np.mean(periods))) if periods else UPDATE_RATE_HZ
        kv = lambda k, v: {"key": k, "value": str(v)}  # noqa: E731
        status = [
            {"level": 0 if all_active else 1,
             "name": "controller_manager: Controllers Activity",
             "message": "All controllers are active" if all_active
             else "Not all controllers are active",
             "hardware_id": "ros2_control",
             "values": [kv(f"{n}.state", s) for n, s in controllers.items()]},
            {"level": 0 if st["hw_state"] == "active" else 1,
             "name": "controller_manager: Hardware Components Activity",
             "message": "All hardware components are active" if st["hw_state"] == "active"
             else "Not all hardware components are active",
             "hardware_id": "ros2_control",
             "values": [kv(f"{HARDWARE_COMPONENT}.state", st["hw_state"])]},
            {"level": 0, "name": "controller_manager: Controller Manager Activity",
             "message": "Controller Manager is running", "hardware_id": "ros2_control",
             "values": [kv("update_rate", int(UPDATE_RATE_HZ)),
                        kv("periodicity.average", f"{hz:.6f}")]},
        ]
        bus.publish(TOPIC_DIAGNOSTICS, {"header": _header("", stamp), "status": status},
                    TOPICS[TOPIC_DIAGNOSTICS][0], node=NODE_CONTROLLER_MANAGER)

    # ---- /reset (workspace-owned) ----------------------------------------------------
    reset_requested = threading.Event()
    reset_observed = threading.Event()
    spawn: dict = {}

    def do_reset(_args: dict) -> dict:
        reset_observed.clear()
        reset_requested.set()
        if not reset_observed.wait(timeout=10.0):
            return {"success": False,
                    "message": "reset requested but the simulation loop did not apply it"}
        return {"success": True, "message": "world reset"}

    bus.service(SERVICE_RESET, do_reset, SRV_TYPE_TRIGGER, node=SIMULATOR_NODE)

    def apply_reset(data) -> None:
        if task is not None:
            task.reset(data)
        elif spawn:
            data.qpos[:] = spawn["qpos"]
            data.qvel[:] = spawn["qvel"]
            data.ctrl[:] = spawn["ctrl"]
        import mujoco

        mujoco.mj_forward(model, data)
        with lock:
            # Aborts every outstanding goal (the fleet's world reset), then puts the
            # controller manager back to its boot state.
            if world_reset is not None:
                world_reset.fire()
            else:
                bus.abort_goals()
            st["jtc_goal"] = st["gripper_goal"] = None
            st["controllers"] = {name: "active" for name in BOOT_CONTROLLERS}
            st["hw_state"] = "active"
            st["scaling"] = 1.0
            st["forward_cmd"] = None
            st["capture"] = True
            st["activity_dirty"] = True
            pos, _, gpos, _ = measured()
            st["arm_cmd"], st["gripper_cmd"] = pos.copy(), gpos
            st["traj"] = Trajectory(None, hold=pos)
            st["traj"].set_point_before(float(data.time), pos, np.zeros(n_arm))
            st["new_traj"] = None
            st["traj_time"] = float(data.time)

    # ---- torque (hardware lifecycle) -------------------------------------------------
    import mujoco as _mj

    actuators = []
    for joint in (*[j.removesuffix("_joint") for j in ARM_JOINTS], URDF_GRIPPER):
        aid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{joint}")
        if aid >= 0:
            actuators.append(aid)
    gains = {a: (model.actuator_gainprm[a].copy(), model.actuator_biasprm[a].copy())
             for a in actuators}
    torque = {"on": True}

    def set_torque(on: bool) -> None:
        if on == torque["on"]:
            return
        torque["on"] = on
        for a in actuators:
            if on:
                model.actuator_gainprm[a][:] = gains[a][0]
                model.actuator_biasprm[a][:] = gains[a][1]
            else:
                model.actuator_gainprm[a][:] = 0.0
                model.actuator_biasprm[a][:] = 0.0

    # ---- the control loop ------------------------------------------------------------
    clock = {"last": None, "next_cam": None, "next_diag": None, "next_feedback": 0.0,
             "release": False}
    written: list = [None]

    def update_jtc(now: float, dt: float, pos, vel) -> None:
        if st["new_traj"] is not None:
            st["traj"], st["new_traj"] = st["new_traj"], None
            st["traj_time"] = None
        traj = st["traj"]
        if st["traj_time"] is None:
            traj.set_point_before(now, pos, vel)
            st["traj_time"] = now
        else:
            st["traj_time"] += dt * st["scaling"]
        t = st["traj_time"]
        ok, desired, dvel, dacc, past_end = traj.sample(t)
        ok_next, cmd_next, _, _, _ = traj.sample(t + 1.0 / UPDATE_RATE_HZ)
        if not ok:
            desired, dvel, dacc = pos.copy(), np.zeros(n_arm), np.zeros(n_arm)
        error_pos, error_vel = desired - pos, dvel - vel
        st["desired"] = (desired, dvel, dacc, t - (traj.start or t))
        st["error"] = (error_pos, error_vel)
        goal_entry = st["jtc_goal"]
        path_violated = outside_goal = False
        within_goal_time = True
        if goal_entry is not None and not traj.holding:
            tol = goal_entry[1]
            for i, joint in enumerate(ARM_JOINTS):
                if not past_end:
                    p_tol, v_tol = tol["path"][joint]
                    if (p_tol > 0 and abs(error_pos[i]) > p_tol) or \
                            (v_tol > 0 and abs(error_vel[i]) > v_tol):
                        path_violated = True
                else:
                    p_tol, v_tol = tol["goal"][joint]
                    if (p_tol > 0 and abs(error_pos[i]) > p_tol) or \
                            (v_tol > 0 and abs(error_vel[i]) > v_tol):
                        outside_goal = True
                        if tol["goal_time"] and t - traj.end > tol["goal_time"]:
                            within_goal_time = False
        if not path_violated and within_goal_time and ok_next:
            st["arm_cmd"] = np.asarray(cmd_next, dtype=float)
        if goal_entry is None:
            return
        goal = goal_entry[0]
        if goal.done:
            st["jtc_goal"] = None
            return
        if now >= clock["next_feedback"]:
            clock["next_feedback"] = now + 1.0 / ACTION_MONITOR_RATE_HZ
            goal.publish_feedback({
                "header": _header("", now), "joint_names": list(ARM_JOINTS),
                "desired": _point(desired, dvel, dacc, t - (traj.start or t)),
                "actual": _point(pos, vel, (), t - (traj.start or t)),
                "error": _point(error_pos, error_vel, ()),
                "multi_dof_joint_names": [],
                "multi_dof_desired": {"transforms": [], "velocities": [], "accelerations": [],
                                      "time_from_start": _dur(0.0)},
                "multi_dof_actual": {"transforms": [], "velocities": [], "accelerations": [],
                                     "time_from_start": _dur(0.0)},
                "multi_dof_error": {"transforms": [], "velocities": [], "accelerations": [],
                                    "time_from_start": _dur(0.0)},
            })
        if path_violated:
            st["jtc_goal"] = None
            goal.abort({"error_code": -4, "error_string": "Aborted due to path tolerance violation"})
            st["traj"], st["traj_time"] = Trajectory(None, hold=pos), None
        elif past_end and not outside_goal:
            st["jtc_goal"] = None
            goal.succeed({"error_code": 0, "error_string": "Goal successfully reached!"})
            last = traj.points[-1][1]
            st["traj"], st["traj_time"] = Trajectory(None, hold=last), None
        elif past_end and not within_goal_time:
            st["jtc_goal"] = None
            late = t - traj.end
            goal.abort({"error_code": -5, "error_string":
                        f"Aborted due to goal_time_tolerance exceeding by {late:f} seconds"})
            st["traj"], st["traj_time"] = Trajectory(None, hold=pos), None

    def update_gripper(now: float, gpos: float, gvel: float) -> None:
        entry = st["gripper_goal"]
        if entry is None:
            return
        goal, last_move = entry
        if goal.done:
            st["gripper_goal"] = None
            return
        result = {"state": {"header": _header("", 0.0), "name": [], "position": [gpos],
                            "velocity": [], "effort": [0.0]},
                  "stalled": False, "reached_goal": False}
        if abs(st["gripper_cmd"] - gpos) < 0.01:
            st["gripper_goal"] = None
            result["reached_goal"] = True
            goal.succeed(result)
        elif abs(gvel) > 0.001:
            st["gripper_goal"] = (goal, now)
        elif now - last_move > 1.0:
            st["gripper_goal"] = None
            result["stalled"] = True
            goal.succeed(result)  # allow_stalling: true

    def publish_states(stamp: float, pos, vel, gpos, gvel) -> None:
        names = list(JOINT_ORDER)
        positions = [*pos.tolist(), gpos]
        velocities = [*vel.tolist(), gvel]
        # The broadcaster's joint order is the resource manager's, which sorts by name.
        order = sorted(range(len(names)), key=lambda i: names[i])
        bus.publish(TOPIC_JOINT_STATES, {
            "header": _header("", stamp),
            "name": [names[i] for i in order],
            "position": [float(positions[i]) for i in order],
            "velocity": [float(velocities[i]) for i in order],
            # No effort interface: the broadcaster fills NaN, which rosbridge's JSON
            # carries as null.
            "effort": [None] * len(names),
        }, TOPICS[TOPIC_JOINT_STATES][0], node=NODE_JOINT_STATE_BROADCASTER)
        bus.publish(TOPIC_DYNAMIC_JOINT_STATES, {
            "header": _header("", stamp),
            "joint_names": [names[i] for i in order],
            "interface_values": [{"interface_names": ["position", "velocity"],
                                  "values": [float(positions[i]), float(velocities[i])]}
                                 for i in order],
        }, TOPICS[TOPIC_DYNAMIC_JOINT_STATES][0], node=NODE_JOINT_STATE_BROADCASTER)

    def publish_controller_state(stamp: float, pos, vel) -> None:
        desired, dvel, dacc, tfs = st["desired"] or (pos, np.zeros(n_arm), np.zeros(n_arm), 0.0)
        error_pos, error_vel = st["error"] or (np.zeros(n_arm), np.zeros(n_arm))
        empty_mdof = {"transforms": [], "velocities": [], "accelerations": [],
                      "time_from_start": _dur(0.0)}
        bus.publish(TOPIC_CONTROLLER_STATE, {
            "header": _header("", stamp), "joint_names": list(ARM_JOINTS),
            "reference": _point(desired, dvel, dacc, tfs),
            "feedback": _point(pos, vel, (), tfs),
            "error": _point(error_pos, error_vel, ()),
            "output": _point(st["arm_cmd"]),
            "multi_dof_joint_names": [],
            "multi_dof_reference": empty_mdof, "multi_dof_feedback": empty_mdof,
            "multi_dof_error": empty_mdof, "multi_dof_output": empty_mdof,
            "speed_scaling_factor": float(st["scaling"]),
        }, TOPICS[TOPIC_CONTROLLER_STATE][0], node=NODE_JOINT_TRAJECTORY_CONTROLLER)

    def step(data):
        if data is None:
            if worker is not None:
                worker.close()
            return
        started = _time.perf_counter()
        now = float(data.time)
        if not spawn:
            spawn.update(qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy())
        dt = (now - clock["last"]) if clock["last"] is not None else 1.0 / UPDATE_RATE_HZ
        period = dt if dt > 0 else 1.0 / UPDATE_RATE_HZ
        clock["last"] = now
        loop_stats["periods"].append(period)

        # The MuJoCo viewer's Control panel (`serve --mujoco`) writes ctrl on this thread;
        # a slider moved there is taken as a new position to hold, as a command would be.
        if written[0] is not None:
            now_ctrl = np.concatenate([np.asarray(arm.ctrl, dtype=float).reshape(-1),
                                       np.asarray(gripper.ctrl, dtype=float).reshape(-1)])
            moved = np.flatnonzero(np.abs(now_ctrl - written[0]) > 1e-9)
            if moved.size:
                with lock:
                    target = st["arm_cmd"].copy()
                    for i in moved:
                        if i < n_arm:
                            target[i] = now_ctrl[i]
                        else:
                            st["gripper_cmd"] = to_wire_gripper(now_ctrl[i])
                    st["new_traj"] = Trajectory(None, hold=target)

        release = False
        if clock["release"]:
            clock["release"] = False
            release = True
        if reset_requested.is_set():
            reset_requested.clear()
            apply_reset(data)
            clock["release"] = True       # released after every member has observed it
            clock["next_cam"] = None

        pos, vel, gpos, gvel = measured()
        with lock:
            if st["hw_state"] == "active":
                set_torque(True)
                if active("joint_trajectory_controller"):
                    update_jtc(now, dt, pos, vel)
                elif active("forward_position_controller") and st["forward_cmd"] is not None:
                    st["arm_cmd"] = st["forward_cmd"].copy()
                if active("gripper_controller"):
                    update_gripper(now, gpos, gvel)
                arm.ctrl = st["arm_cmd"].tolist()
                gripper.ctrl = [to_mjcf_gripper(st["gripper_cmd"])]
            else:
                set_torque(False)     # FeetechHardwareInterface::on_deactivate: torque off
            written[0] = np.concatenate([np.asarray(arm.ctrl, dtype=float).reshape(-1),
                                         np.asarray(gripper.ctrl, dtype=float).reshape(-1)])
            if st["controllers"].get("joint_state_broadcaster") == "active" \
                    and st["hw_state"] in ("active", "inactive"):
                publish_states(now, pos, vel, gpos, gvel)
                positions = {**dict(zip(ARM_JOINTS, pos.tolist())), GRIPPER_JOINT: gpos}
                rsp.publish(positions, now)
            if active("joint_trajectory_controller"):
                publish_controller_state(now, pos, vel)
            if st["activity_dirty"]:
                st["activity_dirty"] = False
                publish_activity(now)
            capture = st["capture"]

        if capture and (clock["next_cam"] is None or now >= clock["next_cam"] - 0.25 / WRIST_FRAMERATE_HZ):
            # A frame the worker cannot take yet stays due, and goes on the next tick.
            if publish_camera(data, now):
                nxt = (clock["next_cam"] or now) + 1.0 / WRIST_FRAMERATE_HZ
                clock["next_cam"] = nxt if nxt > now else now + 1.0 / WRIST_FRAMERATE_HZ
        if clock["next_diag"] is None or now >= clock["next_diag"]:
            clock["next_diag"] = now + DIAGNOSTICS_PERIOD_S
            with lock:
                publish_diagnostics(now)
        exec_s = _time.perf_counter() - started
        loop_stats["exec"].append(exec_s)
        if len(loop_stats["periods"]) > 1000:
            del loop_stats["periods"][:500], loop_stats["exec"][:500]
        with lock:
            publish_statistics(now, period, exec_s, pos, vel, gpos, gvel)
        if release:
            if worker is not None:
                worker.wait(0.2)      # the post-reset wrist frame is out too
            reset_observed.set()

    step.rate_hz = UPDATE_RATE_HZ
    return step


def serve_ros(port: int, view, model, task=None, *, cameras=None, jpeg_quality: int = 70,
              control_hz: float = UPDATE_RATE_HZ, scene_option=None, host: str = "0.0.0.0",
              namespace: str = "", prefix: str = ""):
    """The single-robot path: own a server on `port`, put one arm (and the rig) on it."""
    import sys

    from ros_surfaces import RobotFleet
    from ros_surfaces.scene import attach_scene_rig, probe_scene_cameras

    fleet = RobotFleet(port=port, host=host)
    fleet.attach(namespace, attach_ros, view=view, model=model, task=task,
                 jpeg_quality=jpeg_quality, scene_option=scene_option, prefix=prefix)
    rig = probe_scene_cameras(model, SCENE_CAMERA_TOPICS)
    if rig:
        fleet.attach(SCENE_NAMESPACE, attach_scene_rig, model=model, cameras=rig,
                     jpeg_quality=jpeg_quality, scene_option=scene_option)
    fleet.start()
    print(f"SO-101 on ws://{host}:{port} under namespace {namespace or '<bare>'}",
          file=sys.stderr)
    return fleet
