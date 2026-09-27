"""The myAGV's ROS interface: `robots_specs/myagv/ros.yml`, transcribed.

A ROS 1 (Noetic) graph: the boot launch `myagv_odometry/launch/myagv_active.launch` plus
the `usb_cam` node run beside it as `camera`. Every name, type, node, frame and periodic
rate below is that file's; nothing is added and nothing is left out.

    topic                              type                        dir  node                     Hz
    /cmd_vel                           geometry_msgs/Twist         in   myagv_odometry_node      event
    /odom                              nav_msgs/Odometry           out  myagv_odometry_node      100
    /imu                               sensor_msgs/Imu             out  myagv_odometry_node      100
    /Voltage                           std_msgs/Float32            out  myagv_odometry_node      100
    /voltage_backup                    std_msgs/Float32            out  myagv_odometry_node      100
    /joint_states                      sensor_msgs/JointState      out  joint_state_publisher    10
    /tf                                tf2_msgs/TFMessage          out  robot_state_publisher    10
    /tf_static                         tf2_msgs/TFMessage          out  robot_state_publisher    latched
    /tf                                tf2_msgs/TFMessage          out  base2camera_link         20
    /tf                                tf2_msgs/TFMessage          out  base2imu_link            20
    /tf                                tf2_msgs/TFMessage          out  base2laser_link          100
    /robot_pose_ekf/odom_combined      nav_msgs/Odometry           out  robot_pose_ekf           30
    /tf                                tf2_msgs/TFMessage          out  robot_pose_ekf           30
    /scan                              sensor_msgs/LaserScan       out  ydlidar_lidar_publisher  30
    /point_cloud                       sensor_msgs/PointCloud      out  ydlidar_lidar_publisher  30
    /camera/image_raw                  sensor_msgs/Image           out  camera                   30
    /camera/camera_info                sensor_msgs/CameraInfo      out  camera                   30
    /camera/image_raw/compressed       sensor_msgs/CompressedImage out  camera                   30

    service                            type                        node
    /stop_scan, /start_scan            std_srvs/Empty              ydlidar_lidar_publisher
    /robot_pose_ekf/get_status         robot_pose_ekf/GetStatus    robot_pose_ekf
    /camera/start_capture              std_srvs/Empty              camera
    /camera/stop_capture               std_srvs/Empty              camera
    /camera/set_camera_info            sensor_msgs/SetCameraInfo   camera

    frames   odom -> base_footprint (robot_pose_ekf), base_footprint -> base_up
             (robot_state_publisher), base_footprint -> camera_link / imu_link / laser_frame
             (the three static_transform_publishers, at the launch's offsets and periods)

Parameters are `PARAMETERS` below, plus `/robot_description` (the vendor URDF).

Behaviour, as the vendor's nodes behave:

* `/cmd_vel` clamps `linear.x`, `linear.y` and `angular.z` to [-1, 1] and the last
  command is held and executed every cycle, with **no timeout**. A client stops the base
  by publishing a zero Twist (`STOP_COMMAND`); nothing here stops it for them.
* `/odom` is `odom -> base_footprint` and broadcasts no transform; `robot_pose_ekf` owns
  `odom -> base_footprint` on `/tf` and publishes the fused pose on `odom_combined`.
* `/scan` is in `laser_frame`, which the launch turns a half-turn about z from
  `base_footprint`: the beam at 180 deg in the scan points along the base's +x. Ranges
  follow X2.launch: -180..180 deg, 0.1..12.0 m, invalid returns as 0.0, the
  `ignore_array` wedge -50..50 deg reported as 0.0, and `sample_rate` over the scan rate
  points per sweep (`SCAN_BEAMS`). `/stop_scan` stops `/scan` and
  `/point_cloud` until `/start_scan`.
* The camera publishes `camera_link` frames at 640x480, 30 Hz; `camera_info` is
  uncalibrated (all-zero intrinsics) until `/camera/set_camera_info` stores one.
  `/camera/stop_capture` stops all three camera topics until `/camera/start_capture`.
  The compressed stream encodes only while subscribed.

Names stay bare here; a `NamespacedBus` composes them with the robot's namespace where
they reach the wire (`contracts/namespace.py`).

This module is stdlib-only at import time -- numpy, MuJoCo and OpenCV are imported inside
the functions that need them -- so a contract test can load it by path.
"""

from __future__ import annotations

import base64
import math
import sys

# ------------------------------------------------------------------------------ nodes

NODE_ODOMETRY = "myagv_odometry_node"
NODE_ROBOT_STATE_PUBLISHER = "robot_state_publisher"
NODE_JOINT_STATE_PUBLISHER = "joint_state_publisher"
NODE_BASE2CAMERA = "base2camera_link"
NODE_BASE2IMU = "base2imu_link"
NODE_BASE2LASER = "base2laser_link"
NODE_EKF = "robot_pose_ekf"
NODE_LIDAR = "ydlidar_lidar_publisher"
NODE_CAMERA = "camera"

# ------------------------------------------------------------------------------ topics

TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOM = "/odom"
TOPIC_IMU = "/imu"
TOPIC_VOLTAGE = "/Voltage"
TOPIC_VOLTAGE_BACKUP = "/voltage_backup"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_TF = "/tf"
TOPIC_TF_STATIC = "/tf_static"
TOPIC_ODOM_COMBINED = "/robot_pose_ekf/odom_combined"
TOPIC_SCAN = "/scan"
TOPIC_POINT_CLOUD = "/point_cloud"
TOPIC_IMAGE_RAW = "/camera/image_raw"
TOPIC_CAMERA_INFO = "/camera/camera_info"
TOPIC_CAMERA = "/camera/image_raw/compressed"

TYPE_TWIST = "geometry_msgs/Twist"
TYPE_ODOM = "nav_msgs/Odometry"
TYPE_IMU = "sensor_msgs/Imu"
TYPE_FLOAT32 = "std_msgs/Float32"
TYPE_JOINT_STATE = "sensor_msgs/JointState"
TYPE_TF_MESSAGE = "tf2_msgs/TFMessage"
TYPE_LASER_SCAN = "sensor_msgs/LaserScan"
TYPE_POINT_CLOUD = "sensor_msgs/PointCloud"
TYPE_IMAGE = "sensor_msgs/Image"
TYPE_CAMERA_INFO = "sensor_msgs/CameraInfo"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"

#: A rate that is not periodic.
EVENT = "event"
LATCHED = "latched"

#: Every topic, as `(name, type, direction, node, rate_hz)`. `/tf` has one row per
#: publishing node, each with that node's own rate.
TOPICS: tuple[tuple[str, str, str, str, float | str], ...] = (
    (TOPIC_CMD_VEL, TYPE_TWIST, "in", NODE_ODOMETRY, EVENT),
    (TOPIC_ODOM, TYPE_ODOM, "out", NODE_ODOMETRY, 100.0),
    (TOPIC_IMU, TYPE_IMU, "out", NODE_ODOMETRY, 100.0),
    (TOPIC_VOLTAGE, TYPE_FLOAT32, "out", NODE_ODOMETRY, 100.0),
    (TOPIC_VOLTAGE_BACKUP, TYPE_FLOAT32, "out", NODE_ODOMETRY, 100.0),
    (TOPIC_JOINT_STATES, TYPE_JOINT_STATE, "out", NODE_JOINT_STATE_PUBLISHER, 10.0),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_ROBOT_STATE_PUBLISHER, 10.0),
    (TOPIC_TF_STATIC, TYPE_TF_MESSAGE, "out", NODE_ROBOT_STATE_PUBLISHER, LATCHED),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_BASE2CAMERA, 20.0),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_BASE2IMU, 20.0),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_BASE2LASER, 100.0),
    (TOPIC_ODOM_COMBINED, TYPE_ODOM, "out", NODE_EKF, 30.0),
    (TOPIC_TF, TYPE_TF_MESSAGE, "out", NODE_EKF, 30.0),
    (TOPIC_SCAN, TYPE_LASER_SCAN, "out", NODE_LIDAR, 30.0),
    (TOPIC_POINT_CLOUD, TYPE_POINT_CLOUD, "out", NODE_LIDAR, 30.0),
    (TOPIC_IMAGE_RAW, TYPE_IMAGE, "out", NODE_CAMERA, 30.0),
    (TOPIC_CAMERA_INFO, TYPE_CAMERA_INFO, "out", NODE_CAMERA, 30.0),
    (TOPIC_CAMERA, TYPE_COMPRESSED_IMAGE, "out", NODE_CAMERA, 30.0),
)


def rate_of(topic: str, node: str) -> float | str:
    """The declared rate of `topic` as `node` publishes it."""
    for name, _type, _dir, owner, rate in TOPICS:
        if name == topic and owner == node:
            return rate
    raise KeyError(f"{topic} is not published by {node} on the myAGV")


# ---------------------------------------------------------------------------- services

SERVICE_STOP_SCAN = "/stop_scan"
SERVICE_START_SCAN = "/start_scan"
SERVICE_EKF_STATUS = "/robot_pose_ekf/get_status"
SERVICE_START_CAPTURE = "/camera/start_capture"
SERVICE_STOP_CAPTURE = "/camera/stop_capture"
SERVICE_SET_CAMERA_INFO = "/camera/set_camera_info"

SRV_EMPTY = "std_srvs/Empty"
SRV_GET_STATUS = "robot_pose_ekf/GetStatus"
SRV_SET_CAMERA_INFO = "sensor_msgs/SetCameraInfo"

#: Every service, as `(name, type, node)`.
SERVICES: tuple[tuple[str, str, str], ...] = (
    (SERVICE_STOP_SCAN, SRV_EMPTY, NODE_LIDAR),
    (SERVICE_START_SCAN, SRV_EMPTY, NODE_LIDAR),
    (SERVICE_EKF_STATUS, SRV_GET_STATUS, NODE_EKF),
    (SERVICE_START_CAPTURE, SRV_EMPTY, NODE_CAMERA),
    (SERVICE_STOP_CAPTURE, SRV_EMPTY, NODE_CAMERA),
    (SERVICE_SET_CAMERA_INFO, SRV_SET_CAMERA_INFO, NODE_CAMERA),
)

# -------------------------------------------------------------------------- parameters

#: The vendor description, `myagv_urdf/urdf/myAGV.urdf`, is
#: `robots_specs/myagv/myAGV.urdf` (`robots_spec.urdf_path("myagv")`).
PARAM_ROBOT_DESCRIPTION = "/robot_description"

#: Every other parameter, with the value the launch sets.
PARAMETERS: dict[str, object] = {
    "/robot_pose_ekf/output_frame": "odom",
    "/robot_pose_ekf/base_footprint_frame": "base_footprint",
    "/robot_pose_ekf/freq": 30.0,
    "/robot_pose_ekf/sensor_timeout": 2.0,
    "/robot_pose_ekf/odom_used": True,
    "/robot_pose_ekf/odom_data": "odom",
    "/robot_pose_ekf/imu_used": True,
    "/robot_pose_ekf/vo_used": False,
    "/ydlidar_lidar_publisher/port": "/dev/ttyAMA0",
    "/ydlidar_lidar_publisher/frame_id": "laser_frame",
    "/ydlidar_lidar_publisher/ignore_array": "-50,50",
    "/ydlidar_lidar_publisher/baudrate": 115200,
    "/ydlidar_lidar_publisher/lidar_type": 1,
    "/ydlidar_lidar_publisher/device_type": 0,
    "/ydlidar_lidar_publisher/sample_rate": "3",
    "/ydlidar_lidar_publisher/abnormal_check_count": 4,
    "/ydlidar_lidar_publisher/resolution_fixed": True,
    "/ydlidar_lidar_publisher/auto_reconnect": True,
    "/ydlidar_lidar_publisher/reversion": False,
    "/ydlidar_lidar_publisher/inverted": True,
    "/ydlidar_lidar_publisher/isSingleChannel": True,
    "/ydlidar_lidar_publisher/intensity": False,
    "/ydlidar_lidar_publisher/support_motor_dtr": True,
    "/ydlidar_lidar_publisher/invalid_range_is_inf": False,
    "/ydlidar_lidar_publisher/point_cloud_preservative": False,
    "/ydlidar_lidar_publisher/angle_min": -180.0,
    "/ydlidar_lidar_publisher/angle_max": 180.0,
    "/ydlidar_lidar_publisher/range_min": 0.1,
    "/ydlidar_lidar_publisher/range_max": 12.0,
    "/ydlidar_lidar_publisher/frequency": 10.0,
    "/camera/camera_frame_id": "camera_link",
}

# ------------------------------------------------------------------------------ frames

FRAME_ODOM = "odom"
FRAME_BASE = "base_footprint"
FRAME_BASE_UP = "base_up"
FRAME_CAMERA = "camera_link"
FRAME_IMU = "imu_link"
FRAME_LASER = "laser_frame"

#: The URDF's one movable joint, `base_up` (continuous), held at its default by
#: `joint_state_publisher`.
JOINT_BASE_UP = "base_up"

#: The three `static_transform_publisher`s: node -> (parent, child, xyz, (yaw, pitch,
#: roll)), each re-published on `/tf` at its node's rate.
STATIC_TRANSFORMS: dict[str, tuple[str, str, tuple[float, float, float],
                                   tuple[float, float, float]]] = {
    NODE_BASE2CAMERA: (FRAME_BASE, FRAME_CAMERA, (0.13, 0.0, 0.131), (0.0, 0.0, 0.0)),
    NODE_BASE2IMU: (FRAME_BASE, FRAME_IMU, (0.0, 0.0, 0.0), (0.0, math.pi, math.pi)),
    NODE_BASE2LASER: (FRAME_BASE, FRAME_LASER, (0.065, 0.0, 0.08), (math.pi, 0.0, 0.0)),
}

# ------------------------------------------------------------------------- behaviour

#: `/cmd_vel` components are each clamped to [-limit, limit].
CMD_VEL_LIMIT = 1.0
#: The stop command: a zero Twist on `/cmd_vel`. There is no timeout.
STOP_COMMAND = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}

#: How often this surface is stepped: the fastest periodic topic it presents.
LOOP_HZ = max(r for *_, r in TOPICS if isinstance(r, float))

#: X2.launch's scan geometry.
SCAN_ANGLE_MIN = -math.pi
SCAN_ANGLE_MAX = math.pi
SCAN_RANGE_MIN = 0.1
SCAN_RANGE_MAX = 12.0
#: `ignore_array`: bearings in `laser_frame`, degrees, reported as 0.0.
SCAN_IGNORE_DEG = ((-50.0, 50.0),)
#: `invalid_range_is_inf: false`.
SCAN_INVALID = 0.0
#: Points per sweep: with `resolution_fixed`, the X2's `sample_rate` (3 kHz) over the
#: scan rate `/scan` is published at.
SCAN_BEAMS = round(int(PARAMETERS["/ydlidar_lidar_publisher/sample_rate"]) * 1000
                   / rate_of(TOPIC_SCAN, NODE_LIDAR))

#: usb_cam's defaults: 640x480, frames converted to rgb8.
CAMERA_SIZE = (640, 480)
CAMERA_ENCODING = "rgb8"
#: What `compressed_image_transport` writes for an rgb8 image encoded as JPEG.
COMPRESSED_FORMAT = "rgb8; jpeg compressed bgr8"

#: `/Voltage` and `/voltage_backup`, in volts. The simulated battery does not drain, and
#: carries no backup battery.
VOLTAGE = 12.0
VOLTAGE_BACKUP = 0.0

GRAVITY = 9.80665


# ---------------------------------------------------------------------- message shapes


def _stamp(seq: int, frame_id: str, stamp_s: float) -> dict:
    return {
        "seq": int(seq),
        "stamp": {"secs": int(stamp_s), "nsecs": int((stamp_s % 1) * 1e9)},
        "frame_id": frame_id,
    }


def _quat_msg(q) -> dict:
    """`(w, x, y, z)` as a geometry_msgs/Quaternion."""
    return {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])}


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _ypr_quat(yaw: float, pitch: float, roll: float) -> tuple[float, float, float, float]:
    """tf's `yaw pitch roll` as `(w, x, y, z)`."""
    from contracts.tf import rpy_to_quat

    return rpy_to_quat(roll, pitch, yaw)


def static_transforms() -> dict[str, tuple[str, str, tuple, tuple]]:
    """node -> (parent, child, xyz, quat `(w, x, y, z)`), bare frames."""
    return {
        node: (parent, child, xyz, _ypr_quat(*ypr))
        for node, (parent, child, xyz, ypr) in STATIC_TRANSFORMS.items()
    }


def tree(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, base_up: float = 0.0):
    """The whole transform tree at one pose, as `(parent, child, xyz, quat)`, bare frames."""
    entries = [
        (FRAME_ODOM, FRAME_BASE, (x, y, 0.0), _yaw_quat(yaw)),
        (FRAME_BASE, FRAME_BASE_UP, (0.0, 0.0, 0.0), _yaw_quat(base_up)),
    ]
    entries += list(static_transforms().values())
    return entries


def joint_states(seq: int, stamp_s: float) -> dict:
    """What `joint_state_publisher` sends for the URDF's one movable joint."""
    return {
        "header": _stamp(seq, "", stamp_s),
        "name": [JOINT_BASE_UP],
        "position": [0.0],
        "velocity": [],
        "effort": [],
    }


def uncalibrated_camera_info(seq: int, frame_id: str, stamp_s: float) -> dict:
    """`usb_cam` with an empty `camera_info_url`: size and frame only, zero intrinsics."""
    width, height = CAMERA_SIZE
    return {
        "header": _stamp(seq, frame_id, stamp_s),
        "height": height,
        "width": width,
        "distortion_model": "",
        "D": [],
        "K": [0.0] * 9,
        "R": [0.0] * 9,
        "P": [0.0] * 12,
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }


def scan_bearings(beams: int) -> list[float]:
    """Each beam's bearing in `laser_frame`, from `SCAN_ANGLE_MIN` to `SCAN_ANGLE_MAX`."""
    step = (SCAN_ANGLE_MAX - SCAN_ANGLE_MIN) / (beams - 1)
    return [SCAN_ANGLE_MIN + i * step for i in range(beams)]


def scan_ranges(model, data, x: float, y: float, z: float, yaw: float, beams: int,
                body: int = -1, exclude_bodies=None) -> list[float]:
    """One `/scan` sweep, in `laser_frame`, with X2.launch's range conventions applied.

    `x, y, z, yaw` are the base's pose. The laser sits at the `base2laser_link` offset and
    is turned by its yaw, so bearing `b` in the scan is `yaw + pi + b` in the world.
    Anything outside [range_min, range_max], and anything in the ignored wedge, is 0.0.
    """
    import numpy as np

    from mujoco_bridge import laser_scan_ranges

    _parent, _child, (lx, ly, lz), (lyaw, _p, _r) = STATIC_TRANSFORMS[NODE_BASE2LASER]
    c, s = math.cos(yaw), math.sin(yaw)
    origin = np.array([x + lx * c - ly * s, y + lx * s + ly * c, z + lz])
    step = (SCAN_ANGLE_MAX - SCAN_ANGLE_MIN) / (beams - 1)
    # `laser_scan_ranges` fans `beams` rays over [angle_min, angle_max) -- so one extra
    # step on the end makes the last ray land exactly on SCAN_ANGLE_MAX.
    raw = laser_scan_ranges(
        model, data, origin, yaw + lyaw, beams, SCAN_RANGE_MAX, bodyexclude=body,
        angle_min=SCAN_ANGLE_MIN, angle_max=SCAN_ANGLE_MAX + step,
        exclude_bodies=exclude_bodies,
    )
    out = []
    for bearing, r in zip(scan_bearings(beams), raw):
        deg = math.degrees(bearing)
        ignored = any(lo <= deg <= hi for lo, hi in SCAN_IGNORE_DEG)
        valid = SCAN_RANGE_MIN <= r <= SCAN_RANGE_MAX
        out.append(float(r) if valid and not ignored else SCAN_INVALID)
    return out


def laser_scan(seq: int, ranges, frame_id: str, stamp_s: float, scan_time: float) -> dict:
    beams = len(ranges)
    return {
        "header": _stamp(seq, frame_id, stamp_s),
        "angle_min": SCAN_ANGLE_MIN,
        "angle_max": SCAN_ANGLE_MAX,
        "angle_increment": (SCAN_ANGLE_MAX - SCAN_ANGLE_MIN) / (beams - 1),
        "time_increment": scan_time / beams,
        "scan_time": scan_time,
        "range_min": SCAN_RANGE_MIN,
        "range_max": SCAN_RANGE_MAX,
        "ranges": list(ranges),
        "intensities": [0.0] * beams,
    }


def point_cloud(seq: int, ranges, frame_id: str, stamp_s: float, scan_time: float) -> dict:
    """The same sweep as points in `laser_frame`, with intensity and stamp channels."""
    beams = len(ranges)
    points, intensities, stamps = [], [], []
    for i, (bearing, r) in enumerate(zip(scan_bearings(beams), ranges)):
        if r <= 0.0:
            continue
        points.append({"x": r * math.cos(bearing), "y": r * math.sin(bearing), "z": 0.0})
        intensities.append(0.0)
        stamps.append(i * scan_time / beams)
    return {
        "header": _stamp(seq, frame_id, stamp_s),
        "points": points,
        "channels": [{"name": "intensities", "values": intensities},
                     {"name": "stamps", "values": stamps}],
    }


def ekf_status(odom_count: int, imu_count: int, sent: int, prefix: str = "") -> str:
    """The shape of `robot_pose_ekf`'s own status text."""
    return (
        "Input:\n"
        f" * Odometry sensor\n   - is active\n   - received {odom_count} messages\n"
        f"   - listens to topic {prefix}{TOPIC_ODOM}\n"
        f" * IMU sensor\n   - is active\n   - received {imu_count} messages\n"
        f"   - listens to topic {prefix}{TOPIC_IMU}\n"
        " * Visual Odometry sensor\n   - is NOT used\n"
        "Output:\n"
        f" * Robot pose ekf filter\n   - is active\n   - sent {sent} messages\n"
        f"   - pulishes on topics {prefix}{TOPIC_ODOM_COMBINED} and {prefix}{TOPIC_TF}\n"
    )


def _clamp(value, limit: float = CMD_VEL_LIMIT) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(-limit, min(limit, v))


class _Every:
    """A drift-free clock for one periodic stream.

    Due times advance by exactly one period, so the long-run rate is the declared rate
    whatever the loop's jitter; `slack` lets a tick that lands a hair early still count.
    A stream that fell more than a period behind re-anchors instead of bursting.
    """

    def __init__(self, hz: float, slack: float) -> None:
        self.period = 1.0 / float(hz)
        self.slack = slack
        self.next: float | None = None

    def ready(self, now: float) -> bool:
        """Whether a tick is due, without taking it."""
        if self.next is None or self.next > now + 2 * self.period:
            self.next = now
        return now + self.slack >= self.next

    def take(self, now: float) -> None:
        """Take the due tick: the next is one period after this one was due."""
        self.next += self.period
        if self.next < now - self.period:
            self.next = now + self.period

    def due(self, now: float) -> bool:
        if not self.ready(now):
            return False
        self.take(now)
        return True


# ------------------------------------------------------------------------- the surface


def attach_ros(bus, base, model, camera: str | None, *, jpeg_quality: int = 80,
               lidar: dict | None = None, scene_option=None, world_reset=None,
               prefix: str = ""):
    """Wire one myAGV onto `bus` and return the per-step callback.

    `base` exposes a 4x4 `pose` and a writable `ctrl` triple (`mujoco_bridge.
    PlanarJointBase`, or MolmoSpaces' `HoloJointsRobotBaseGroup`). `camera` is the MJCF
    camera `/camera/*` renders, and `lidar` names the rays' own body and the bodies they
    skip (`{"body": name, "exclude_bodies": ids}`). With `model=None` the
    sensors publish nothing and everything else behaves as usual, which is what a
    wire-only test wants.

    The returned step carries `rate_hz`, the rate the fleet must call it at; call it with
    `MjData` each period and with `None` to close this robot's streams.

    `prefix`, the model's MJCF prefix, is accepted with the engines' common arguments and
    not needed: this robot's transform tree is its launch's and its URDF's, not the
    model's, and the camera and lidar bodies arrive already resolved.
    """
    from contracts.rosbridge_server import odometry
    from mujoco_bridge import PlanarSetpoint, RenderWorker

    import robots_spec

    slack = 0.5 / LOOP_HZ

    # -- declarations: every name, with its node, before the first message ----------
    for name, mtype, direction, node, _rate in TOPICS:
        if direction == "out":
            bus.advertise(name, mtype, node=node)
    bus.set_param(PARAM_ROBOT_DESCRIPTION,
                  robots_spec.urdf_path("myagv").read_text(encoding="utf-8"))
    for name, value in PARAMETERS.items():
        bus.set_param(name, value)

    # -- /cmd_vel: clamp, hold, no timeout ----------------------------------------
    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    def on_cmd_vel(msg: dict) -> None:
        linear = msg.get("linear") or {}
        angular = msg.get("angular") or {}
        command.update(vx=_clamp(linear.get("x", 0.0)), vy=_clamp(linear.get("y", 0.0)),
                       wz=_clamp(angular.get("z", 0.0)))

    bus.on(TOPIC_CMD_VEL, on_cmd_vel, TYPE_TWIST, node=NODE_ODOMETRY)

    # -- services -----------------------------------------------------------------
    state = {"scanning": True, "capturing": True, "camera_info": None,
             "odom_count": 0, "imu_count": 0, "ekf_sent": 0}

    def _setter(key: str, value: bool):
        def handler(_args: dict) -> dict:
            state[key] = value
            return {}
        return handler

    def set_camera_info(args: dict) -> dict:
        info = (args or {}).get("camera_info")
        if not isinstance(info, dict):
            return {"success": False, "status_message": "no camera_info in the request"}
        state["camera_info"] = dict(info)
        return {"success": True, "status_message": ""}

    handlers = {
        SERVICE_STOP_SCAN: _setter("scanning", False),
        SERVICE_START_SCAN: _setter("scanning", True),
        SERVICE_EKF_STATUS: lambda _args: {"status": ekf_status(
            state["odom_count"], state["imu_count"], state["ekf_sent"],
            prefix=bus.topic("/").rstrip("/"))},
        SERVICE_START_CAPTURE: _setter("capturing", True),
        SERVICE_STOP_CAPTURE: _setter("capturing", False),
        SERVICE_SET_CAMERA_INFO: set_camera_info,
    }
    for name, stype, node in SERVICES:
        bus.service(name, handlers[name], stype, node=node)

    # -- /tf_static: robot_state_publisher's, latched, and empty (no fixed joints) -
    bus.publish(TOPIC_TF_STATIC, {"transforms": []}, TYPE_TF_MESSAGE, latched=True,
                node=NODE_ROBOT_STATE_PUBLISHER)

    # -- the sensors ----------------------------------------------------------------
    worker = None
    lidar_body, lidar_exclude, beams = -1, None, SCAN_BEAMS
    if model is not None:
        import mujoco

        if camera is None:
            raise ValueError("the myAGV's camera is part of its interface; name an MJCF camera")
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError(f"no camera {camera!r} in this model")
        # Rendered off the control loop (see `RenderWorker`): a 640x480 frame is ~12 ms
        # of GL and readback, and on this thread it held every 100 Hz stream back by it,
        # and every other member on the port with them.
        worker = RenderWorker(model, name=f"{bus.ns or 'myagv'} camera")
        if lidar is None:
            raise ValueError("the myAGV's lidar is part of its interface; pass `lidar`")
        lidar_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lidar["body"])
        lidar_exclude = frozenset(lidar.get("exclude_bodies") or ())

    frame = bus.frame
    frames = {name: frame(name) for name in (FRAME_ODOM, FRAME_BASE, FRAME_BASE_UP,
                                             FRAME_CAMERA, FRAME_IMU, FRAME_LASER)}
    statics = {node: (frame(p), frame(c), xyz, q)
               for node, (p, c, xyz, q) in static_transforms().items()}
    clocks = {
        "odometry": _Every(rate_of(TOPIC_ODOM, NODE_ODOMETRY), slack),
        "joint_states": _Every(rate_of(TOPIC_JOINT_STATES, NODE_JOINT_STATE_PUBLISHER), slack),
        "ekf": _Every(rate_of(TOPIC_ODOM_COMBINED, NODE_EKF), slack),
        "scan": _Every(rate_of(TOPIC_SCAN, NODE_LIDAR), slack),
        "camera": _Every(rate_of(TOPIC_IMAGE_RAW, NODE_CAMERA), slack),
    }
    for node in STATIC_TRANSFORMS:
        clocks[node] = _Every(rate_of(TOPIC_TF, node), slack)
    scan_time = 1.0 / rate_of(TOPIC_SCAN, NODE_LIDAR)

    setpoint = PlanarSetpoint()
    # Per-publisher sequence numbers, as ROS 1 keeps them.
    seqs: dict[str, int] = {}

    def seq(key: str) -> int:
        seqs[key] = seqs.get(key, 0) + 1
        return seqs[key]

    if world_reset is not None:
        # `/reset` restores every joint and actuator target, this base's included; the
        # command and the integrated setpoint are the state it has to drop with them.
        def _forget() -> None:
            setpoint.reset()
            command.update(vx=0.0, vy=0.0, wz=0.0)

        world_reset.on_reset(_forget)

    last = {"t": None, "yaw": None, "v": None}

    def publish_tf(key: str, entries, stamp: float, node: str) -> None:
        from contracts.tf import tf_message

        bus.publish(TOPIC_TF, tf_message(entries, stamp_s=stamp, seq=seq(key)),
                    TYPE_TF_MESSAGE, node=node)

    def publish_camera(data, stamp: float) -> bool:
        """One camera tick. False when the render worker could not take the frame yet,
        which leaves the tick due for the next step rather than dropping it."""
        if not state["capturing"]:
            return True
        want_raw = bus.has_subscribers(TOPIC_IMAGE_RAW)
        want_jpeg = bus.has_subscribers(TOPIC_CAMERA)
        if worker is not None and (want_raw or want_jpeg) and worker.busy:
            return False
        n = seq("camera")
        info = state["camera_info"]
        if info is None:
            info_msg = uncalibrated_camera_info(n, frames[FRAME_CAMERA], stamp)
        else:
            info_msg = dict(info)
            info_msg["header"] = _stamp(n, frames[FRAME_CAMERA], stamp)
        if worker is None or not (want_raw or want_jpeg):
            encode_and_publish(n, stamp, None, False, False, info_msg)
            return True
        width, height = CAMERA_SIZE

        def done(pixels, _stamp_s) -> None:  # on the worker's thread
            encode_and_publish(n, stamp, pixels, want_raw, want_jpeg, info_msg)

        return worker.submit(data, stamp, [(camera, width, height, scene_option, done)])

    def encode_and_publish(n: int, stamp: float, pixels, want_raw: bool, want_jpeg: bool,
                           info_msg: dict) -> None:
        width, height = CAMERA_SIZE
        header = _stamp(n, frames[FRAME_CAMERA], stamp)
        if pixels is not None and want_raw:
            bus.publish(TOPIC_IMAGE_RAW, {
                "header": header,
                "height": height, "width": width, "encoding": CAMERA_ENCODING,
                "is_bigendian": 0, "step": width * 3,
                "data": base64.b64encode(pixels.tobytes()).decode("ascii"),
            }, TYPE_IMAGE, node=NODE_CAMERA)
        if pixels is not None and want_jpeg:
            import cv2

            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if ok:
                bus.publish(TOPIC_CAMERA, {
                    "header": header,
                    "format": COMPRESSED_FORMAT,
                    "data": base64.b64encode(buf.tobytes()).decode("ascii"),
                }, TYPE_COMPRESSED_IMAGE, node=NODE_CAMERA)
        bus.publish(TOPIC_CAMERA_INFO, info_msg, TYPE_CAMERA_INFO, node=NODE_CAMERA)

    def step(data):
        if data is None:
            if worker is not None:
                worker.close()
            return

        import numpy as np

        # Simulated time: what every stream is scheduled by and every header carries.
        now = stamp = float(getattr(data, "time", 0.0))
        pose = base.pose
        x, y, z = float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        vx, vy, wz = command["vx"], command["vy"], command["wz"]
        dt = 1.0 / LOOP_HZ
        base.ctrl = setpoint.step(x, y, yaw, vx, vy, wz, dt)

        # -- myagv_odometry_node: odom, imu, voltages -------------------------------
        if clocks["odometry"].due(now):
            t = now
            rate = 0.0
            accel = (0.0, 0.0)
            world_v = (vx * math.cos(yaw) - vy * math.sin(yaw),
                       vx * math.sin(yaw) + vy * math.cos(yaw))
            if last["t"] is not None and t > last["t"]:
                h = t - last["t"]
                rate = math.atan2(math.sin(yaw - last["yaw"]), math.cos(yaw - last["yaw"])) / h
                ax = (world_v[0] - last["v"][0]) / h
                ay = (world_v[1] - last["v"][1]) / h
                # World -> base: rotate by -yaw.
                accel = (ax * math.cos(yaw) + ay * math.sin(yaw),
                         -ax * math.sin(yaw) + ay * math.cos(yaw))
            last.update(t=t, yaw=yaw, v=world_v)

            bus.publish(TOPIC_ODOM, odometry(seq("odom"), x, y, yaw, vx, vy, wz,
                                             frame_id=frames[FRAME_ODOM],
                                             child_frame_id=frames[FRAME_BASE],
                                             stamp_s=stamp),
                        TYPE_ODOM, node=NODE_ODOMETRY)
            state["odom_count"] += 1
            # imu_link is base_footprint turned a half-turn about z (yaw 0, pitch pi,
            # roll pi), so the base's x and y read negated and z unchanged.
            bus.publish(TOPIC_IMU, {
                "header": _stamp(seq("imu"), frames[FRAME_IMU], stamp),
                "orientation": _quat_msg(_yaw_quat(yaw + math.pi)),
                "orientation_covariance": [0.0] * 9,
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": rate},
                "angular_velocity_covariance": [0.0] * 9,
                "linear_acceleration": {"x": -accel[0], "y": -accel[1], "z": GRAVITY},
                "linear_acceleration_covariance": [0.0] * 9,
            }, TYPE_IMU, node=NODE_ODOMETRY)
            state["imu_count"] += 1
            bus.publish(TOPIC_VOLTAGE, {"data": VOLTAGE}, TYPE_FLOAT32, node=NODE_ODOMETRY)
            bus.publish(TOPIC_VOLTAGE_BACKUP, {"data": VOLTAGE_BACKUP}, TYPE_FLOAT32,
                        node=NODE_ODOMETRY)

        # -- the three static_transform_publishers ------------------------------------
        for node, entry in statics.items():
            if clocks[node].due(now):
                publish_tf(node, [entry], stamp, node)

        # -- joint_state_publisher, and robot_state_publisher on each joint state ------
        if clocks["joint_states"].due(now):
            bus.publish(TOPIC_JOINT_STATES, joint_states(seq("joint_states"), stamp),
                        TYPE_JOINT_STATE, node=NODE_JOINT_STATE_PUBLISHER)
            publish_tf("rsp", [(frames[FRAME_BASE], frames[FRAME_BASE_UP], (0.0, 0.0, 0.0),
                                (1.0, 0.0, 0.0, 0.0))], stamp, NODE_ROBOT_STATE_PUBLISHER)

        # -- robot_pose_ekf ------------------------------------------------------------
        if clocks["ekf"].due(now):
            fused = odometry(seq("ekf"), x, y, yaw, vx, vy, wz,
                             frame_id=frames[FRAME_ODOM], child_frame_id=frames[FRAME_BASE],
                             stamp_s=stamp)
            bus.publish(TOPIC_ODOM_COMBINED, fused, TYPE_ODOM, node=NODE_EKF)
            publish_tf("ekf_tf", [(frames[FRAME_ODOM], frames[FRAME_BASE], (x, y, 0.0),
                                   _yaw_quat(yaw))], stamp, NODE_EKF)
            state["ekf_sent"] += 1

        # -- ydlidar_lidar_publisher ---------------------------------------------------
        if clocks["scan"].due(now) and state["scanning"] and model is not None:
            ranges = scan_ranges(model, data, x, y, z, yaw, beams, lidar_body, lidar_exclude)
            n = seq("scan")
            bus.publish(TOPIC_SCAN, laser_scan(n, ranges, frames[FRAME_LASER], stamp,
                                               scan_time), TYPE_LASER_SCAN, node=NODE_LIDAR)
            bus.publish(TOPIC_POINT_CLOUD, point_cloud(n, ranges, frames[FRAME_LASER], stamp,
                                                       scan_time),
                        TYPE_POINT_CLOUD, node=NODE_LIDAR)

        # -- camera ----------------------------------------------------------------------
        if clocks["camera"].ready(now) and publish_camera(data, stamp):
            clocks["camera"].take(now)

    step.rate_hz = LOOP_HZ
    print(f"myAGV under namespace {bus.ns or '<bare>'}, stepped at {LOOP_HZ:g} Hz",
          file=sys.stderr)
    return step


def serve_ros(port: int, base, model, camera: str | None, *, host: str = "0.0.0.0",
              namespace: str = "", **kwargs):
    """The single-robot path: own a server on `port`, put one myAGV on it, start it."""
    from ros_surfaces import RobotFleet

    fleet = RobotFleet(port=port, host=host)
    fleet.attach(namespace, attach_ros, base=base, model=model, camera=camera, **kwargs)
    fleet.start()
    return fleet
