"""The AiNex's periodic publications: each driver's loop, at its own rate.

On the robot these are separate processes with their own `rospy.Rate`: the board driver
polls at `freq` 100, `imu_calib` and the complementary filter follow it, `sensor` runs at
50, joy_node repeats at 20 and usb_cam delivers 30 frames a second. None of them is tied
to anyone's control loop, and here none of them is either: one thread per robot runs every
periodic topic on its own clock at the rate `topics.RATES_HZ` declares, whatever
`--control-hz` the engine steps its surfaces at.

What that thread publishes is a snapshot. The simulation thread owns MjData and is the
only thing that reads it (`surface.step`); it leaves the body's attitude and the latest
camera frame here, and the clocks below turn them into messages. So the IMU's *values*
update at the control rate and its *stream* runs at 100 Hz, which is how a 100 Hz driver
reads a sensor that changes more slowly than it is polled. The camera is the same: MuJoCo
can only render on the thread that steps it, so frames are rendered there, at most one per
control tick and at most 30 a second, and the 30 Hz stream carries the latest one.

Encoding a 640x480 `sensor_msgs/Image` is 1.2 MB of base64 a frame. Each image topic is
encoded only while a client is subscribed to it -- what a lazy image_transport publisher
and the rectify nodelet both do -- and at most once per rendered frame.
"""

from __future__ import annotations

import base64
import math
import sys
import threading
import time

import numpy as np

from . import topics

GRAVITY = 9.80665

#: `ainex_calibration/config/imu_calib.yaml`: the accelerometer calibration `apply_calib`
#: loads (`calib_file`). `apply_calib` computes corrected = SM * raw + bias, so the raw
#: reading a true acceleration produces is SM^-1 (a - bias).
IMU_CALIB_SM = np.array([
    [1.004811881631405, 0.005598202036732836, -0.004483863049147996],
    [-0.02164561883441042, 0.9990390056104881, 0.05252327041692323],
    [-0.01057121624399493, -0.04241740866065351, 1.000121897589685],
])
IMU_CALIB_BIAS = np.array([0.5600219843015065, 0.2545693087491484, -0.1404977336419296])

#: The covariances `ros_robot_controller_node.py::pub_imu_data` writes, carried unchanged
#: through `apply_calib` and the complementary filter.
ORIENTATION_COV = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
ANGULAR_VELOCITY_COV = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
LINEAR_ACCELERATION_COV = [0.0004, 0, 0, 0, 0.0004, 0, 0, 0, 0.004]
MAG_COV = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]

#: The simulated Earth field, in the world frame, tesla: horizontal along world +x and
#: dipping down. A nominal mid-latitude field; what matters is that it is fixed in the
#: world, so a magnetometer on a turning body sees it turn.
EARTH_FIELD_T = np.array([2.0e-5, 0.0, -4.4e-5])

#: joystick_control's axis and button maps, which fix the layout a `sensor_msgs/Joy` from
#: this gamepad has: eight axes, twenty-one buttons.
JOY_AXES = 8
JOY_BUTTONS = 21
#: joy_node stamps each message with the device it reads (`dev`).
JOY_FRAME = "/dev/input/js0"


def _stamp() -> dict:
    now = time.time()
    return {"secs": int(now), "nsecs": int((now % 1) * 1e9)}


def _header(seq: int, frame_id: str) -> dict:
    return {"seq": seq, "stamp": _stamp(), "frame_id": frame_id}


def _vec(v) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _body_rotation(yaw: float, pitch: float) -> np.ndarray:
    """World-from-body: yaw about world z, then the torso's lean about its own y."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rz @ ry


def _quaternion(yaw: float, pitch: float) -> dict:
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    return {"x": -sy * sp, "y": cy * sp, "z": sy * cp, "w": cy * cp}


class Attitude:
    """The torso's attitude, left by the simulation thread for the IMU clock to read.

    DEPARTURE: roll is identically zero and there is no linear acceleration beyond
    gravity, because the torso rides planar joints -- yaw, and the `base_pitch` lean, are
    the whole of its rotation.
    """

    __slots__ = ("yaw", "pitch", "wz", "wy")

    def __init__(self) -> None:
        self.yaw = self.pitch = self.wz = self.wy = 0.0

    def set(self, yaw: float, pitch: float, wz: float, wy: float) -> None:
        # One tuple assignment per field is atomic enough: a reader sees at worst one
        # field a tick newer than another.
        self.yaw, self.pitch, self.wz, self.wy = yaw, pitch, wz, wy


def imu_messages(att: Attitude, seq: int) -> dict[str, dict]:
    """The board's raw IMU and magnetometer, and the calibrated and filtered IMU.

    Built as the three nodes build them: the board fills acceleration and rate and leaves
    orientation zero; `apply_calib` corrects the acceleration; the complementary filter
    adds the orientation. The raw accelerometer reading is the calibration run backwards,
    so the corrected one is the true specific force.
    """
    rot = _body_rotation(att.yaw, att.pitch)
    accel = rot.T @ np.array([0.0, 0.0, GRAVITY])
    gyro = rot.T @ (np.array([0.0, 0.0, att.wz]) + att.wy * _yaw_y(att.yaw))
    raw_accel = np.linalg.solve(IMU_CALIB_SM, accel - IMU_CALIB_BIAS)
    field = rot.T @ EARTH_FIELD_T
    cal = np.array(topics.MAG_CALIBRATION).reshape(4, 4)
    raw_field = np.linalg.solve(cal, np.append(field, 1.0))[:3]

    header = _header(seq, topics.FRAME_IMU)
    raw = {
        "header": header,
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
        "orientation_covariance": ORIENTATION_COV,
        "angular_velocity": _vec(gyro),
        "angular_velocity_covariance": ANGULAR_VELOCITY_COV,
        "linear_acceleration": _vec(raw_accel),
        "linear_acceleration_covariance": LINEAR_ACCELERATION_COV,
    }
    corrected = dict(raw, linear_acceleration=_vec(accel))
    filtered = dict(corrected, orientation=_quaternion(att.yaw, att.pitch))
    return {
        topics.TOPIC_IMU_RAW: raw,
        topics.TOPIC_IMU_CORRECTED: corrected,
        topics.TOPIC_IMU: filtered,
        topics.TOPIC_MAG_RAW: _vec(raw_field),
        # The board driver stamps the calibrated field and leaves its frame_id empty.
        topics.TOPIC_MAG: {
            "header": {"seq": seq, "stamp": header["stamp"], "frame_id": ""},
            "magnetic_field": _vec(field),
            "magnetic_field_covariance": MAG_COV,
        },
    }


def _yaw_y(yaw: float) -> np.ndarray:
    """The lean axis in the world frame: the body's y after yaw alone."""
    return np.array([-math.sin(yaw), math.cos(yaw), 0.0])


def neutral_joy(seq: int) -> dict:
    """A gamepad at rest, as joy_node repeats it."""
    return {
        "header": _header(seq, JOY_FRAME),
        "axes": [0.0] * JOY_AXES,
        "buttons": [0] * JOY_BUTTONS,
    }


def has_subscriber(bus, topic: str) -> bool:
    """Whether any client is subscribed to `topic` under this bus's namespace.

    Reads the server's client table directly; the bridge has no public call for it. On a
    bridge that does not look like this one it answers True, which costs an encode and
    never a message.
    """
    server = bus.server
    name = bus.topic(topic)
    try:
        with server._lock:  # noqa: SLF001
            return any(name in c.subs for c in server._clients.values())  # noqa: SLF001
    except AttributeError:
        return True


class HeadCamera:
    """usb_cam on the head, plus the rectify nodelet beside it.

    `render` runs on the simulation thread; `publish` on the stream clock. The simulated
    lens has no distortion, so the rectified image is the raw one, as image_proc's rectify
    produces for a camera whose calibration has zero distortion.
    """

    def __init__(self, bus, model, camera: str | None, fovy_deg: float, jpeg_quality: int,
                 scene_option=None) -> None:
        import mujoco

        self._bus = bus
        self._camera = camera
        self._fovy = float(fovy_deg)
        self._jpeg_quality = int(jpeg_quality)
        self._scene_option = scene_option
        self._frame_id = bus.frame(topics.FRAME_CAMERA)
        self._period = 1.0 / topics.RATES_HZ[topics.TOPIC_CAMERA_RAW]
        self._next = 0.0
        self._frame: np.ndarray | None = None
        self._count = 0
        self._cache: dict[str, tuple[int, object]] = {}
        self._seq = 0
        self._renderer = None
        #: Called with each new frame's (count, rgb) from the stream thread -- the vision
        #: nodes, which subscribe to image_raw on the robot.
        self.frame_listeners: list = []
        if camera is not None:
            width, height = topics.CAMERA_SIZE
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            self._renderer = mujoco.Renderer(model, height, width)
        else:
            print(f"no head camera: {bus.topic(topics.TOPIC_CAMERA_RAW)} and its companions "
                  "will not be published", file=sys.stderr)

    def render(self, data) -> None:
        if self._renderer is None:
            return
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + self._period
        self._renderer.update_scene(data, camera=self._camera, scene_option=self._scene_option)
        frame = self._renderer.render().copy()
        self._frame, self._count = frame, self._count + 1

    def _encoded(self, kind: str, count: int, frame: np.ndarray, make):
        cached = self._cache.get(kind)
        if cached is not None and cached[0] == count:
            return cached[1]
        value = make(frame)
        self._cache[kind] = (count, value)
        return value

    def image_msg(self, seq: int, count: int, frame: np.ndarray) -> dict:
        width, height = topics.CAMERA_SIZE
        data = self._encoded("raw", count, frame,
                             lambda f: base64.b64encode(f.tobytes()).decode("ascii"))
        return {
            "header": _header(seq, self._frame_id),
            "height": height, "width": width, "encoding": topics.CAMERA_ENCODING,
            "is_bigendian": 0, "step": width * 3, "data": data,
        }

    def publish(self) -> None:
        frame, count = self._frame, self._count
        if frame is None:
            return
        from contracts.rosbridge_server import camera_info

        bus = self._bus
        self._seq += 1
        seq = self._seq
        width, height = topics.CAMERA_SIZE
        info = camera_info(seq, width, height, self._fovy, frame_id=self._frame_id)
        bus.publish(topics.TOPIC_CAMERA_INFO, info, topics.TYPE_CAMERA_INFO,
                    node=topics.NODE_CAMERA)
        if has_subscriber(bus, topics.TOPIC_CAMERA_RAW):
            bus.publish(topics.TOPIC_CAMERA_RAW, self.image_msg(seq, count, frame),
                        topics.TYPE_IMAGE, node=topics.NODE_CAMERA)
        if has_subscriber(bus, topics.TOPIC_CAMERA):
            jpeg = self._encoded("jpeg", count, frame, self._jpeg)
            if jpeg is not None:
                bus.publish(topics.TOPIC_CAMERA, {
                    "header": _header(seq, self._frame_id),
                    "format": topics.CAMERA_COMPRESSED_FORMAT,
                    "data": jpeg,
                }, topics.TYPE_COMPRESSED_IMAGE, node=topics.NODE_CAMERA)
        if has_subscriber(bus, topics.TOPIC_CAMERA_RECT):
            bus.publish(topics.TOPIC_CAMERA_RECT, self.image_msg(seq, count, frame),
                        topics.TYPE_IMAGE, node=topics.NODE_RECTIFY)
        for listener in self.frame_listeners:
            listener(seq, count, frame)

    def _jpeg(self, frame: np.ndarray):
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        except Exception as exc:  # noqa: BLE001 -- a bad frame must not stop the stream
            print(f"camera encode failed: {exc}", file=sys.stderr)
            return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class Clocks:
    """One thread running several fixed-rate jobs, each on its own schedule."""

    def __init__(self, name: str) -> None:
        self._jobs: list[list] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def every(self, hz: float, job) -> None:
        self._jobs.append([1.0 / hz, 0.0, job])

    def start(self) -> None:
        start = time.monotonic()
        for job in self._jobs:
            job[1] = start
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            for job in self._jobs:
                period, due, fn = job
                if now >= due:
                    try:
                        fn()
                    except Exception as exc:  # noqa: BLE001 -- one bad tick, not a dead clock
                        print(f"ainex stream: {exc!r}", file=sys.stderr)
                    # Keep the schedule rather than drifting by the work's own time; a
                    # stall longer than a period skips ahead instead of bursting to catch up.
                    due += period
                    job[1] = due if due > now else now + period
            wake = min(job[1] for job in self._jobs) if self._jobs else now + 0.1
            self._stop.wait(max(0.0, wake - time.monotonic()))
