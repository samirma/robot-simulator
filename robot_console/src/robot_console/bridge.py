"""The rosbridge client: `/cmd_vel` out, `/odom` and the camera in.

Everything network-facing is in `RobotLink`; `parse_odom` is pure so the quaternion
maths is testable without a server.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import threading
from typing import Callable, Mapping, Optional

import roslibpy

from robot_console.teleop import Command
from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TOPIC_SCAN,
    TYPE_COMPRESSED_IMAGE,
    TYPE_LASER_SCAN,
    TYPE_ODOM,
    TYPE_TWIST,
    normalise,
)


@dataclasses.dataclass(frozen=True)
class Odom:
    """The parts of `nav_msgs/Odometry` the console uses. Angles in radians."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    stamp: float = 0.0
    frame_id: str = ""
    child_frame_id: str = ""


def _f(mapping: object, *keys: str, default: float = 0.0) -> float:
    """Walk a nested dict, tolerating anything missing or the wrong shape."""
    node: object = mapping
    for key in keys:
        if not isinstance(node, Mapping):
            return default
        node = node.get(key)
    try:
        return float(node)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw from a quaternion, wrapped to (-pi, pi].

    The simulator only ever emits a yaw-only quaternion, for which this reduces to
    `2*atan2(z, w)`. A real myAGV's odometry has roll and pitch noise, so the general
    form is what actually works against hardware.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def parse_odom(msg: Mapping) -> Odom:
    """Pure `nav_msgs/Odometry` -> `Odom`. Missing fields default to zero."""
    pose = msg.get("pose") if isinstance(msg, Mapping) else None
    twist = msg.get("twist") if isinstance(msg, Mapping) else None
    header = msg.get("header") if isinstance(msg, Mapping) else None
    stamp_secs = _f(header, "stamp", "secs")
    stamp_nsecs = _f(header, "stamp", "nsecs")
    return Odom(
        x=_f(pose, "pose", "position", "x"),
        y=_f(pose, "pose", "position", "y"),
        yaw=quaternion_yaw(
            _f(pose, "pose", "orientation", "x"),
            _f(pose, "pose", "orientation", "y"),
            _f(pose, "pose", "orientation", "z"),
            _f(pose, "pose", "orientation", "w", default=1.0),
        ),
        vx=_f(twist, "twist", "linear", "x"),
        vy=_f(twist, "twist", "linear", "y"),
        wz=_f(twist, "twist", "angular", "z"),
        stamp=stamp_secs + stamp_nsecs * 1e-9,
        frame_id=str((header or {}).get("frame_id", "")) if isinstance(header, Mapping) else "",
        child_frame_id=str(msg.get("child_frame_id", "")) if isinstance(msg, Mapping) else "",
    )


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class RobotLink:
    """A rosbridge connection carrying the three topics of the myAGV contract."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        *,
        cmd_topic: str = TOPIC_CMD_VEL,
        odom_topic: str = TOPIC_ODOM,
        camera_topic: str = TOPIC_CAMERA,
        scan_topic: str = TOPIC_SCAN,
    ) -> None:
        self.host = host
        self.port = int(port)
        self._cmd_name = normalise(cmd_topic)
        self._odom_name = normalise(odom_topic)
        self._camera_name = normalise(camera_topic)
        self._scan_name = normalise(scan_topic)
        self._ros: Optional[roslibpy.Ros] = None
        self._cmd: Optional[roslibpy.Topic] = None
        self._odom: Optional[roslibpy.Topic] = None
        self._camera: Optional[roslibpy.Topic] = None
        self._scan: Optional[roslibpy.Topic] = None
        self._closed = False

    # ---------------------------------------------------------------- lifecycle

    def connect(self, timeout: float = 10.0) -> None:
        ros = roslibpy.Ros(host=self.host, port=self.port)
        ros.run(timeout=timeout)
        if not ros.is_connected:
            raise ConnectionError(f"could not connect to ws://{self.host}:{self.port}")
        self._ros = ros
        self._cmd = roslibpy.Topic(ros, self._cmd_name, TYPE_TWIST)
        # The bridge treats `advertise` as a no-op and never acks, so this is only for
        # the benefit of a real rosbridge_suite, which does want it before a publish.
        self._cmd.advertise()

    @property
    def is_connected(self) -> bool:
        return bool(self._ros and self._ros.is_connected)

    def __enter__(self) -> "RobotLink":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- topics

    def subscribe_odom(self, callback: Callable[[Odom], None]) -> None:
        if self._ros is None:
            raise RuntimeError("not connected")
        self._odom = roslibpy.Topic(self._ros, self._odom_name, TYPE_ODOM)
        self._odom.subscribe(lambda msg: callback(parse_odom(msg)))

    def subscribe_camera(self, callback: Callable[[dict], None]) -> None:
        if self._ros is None:
            raise RuntimeError("not connected")
        self._camera = roslibpy.Topic(self._ros, self._camera_name, TYPE_COMPRESSED_IMAGE)
        self._camera.subscribe(callback)

    def subscribe_scan(self, callback: Callable[[dict], None]) -> None:
        """Hand raw `sensor_msgs/LaserScan` dicts straight to `callback`.

        Raw, like the camera and unlike odom: a 360-beam scan at 10 Hz is real work to
        turn into points, and this runs on roslibpy's reactor thread, which also carries
        `/odom` and `/cmd_vel`. Park the dict and parse it in the main loop.
        """
        if self._ros is None:
            raise RuntimeError("not connected")
        self._scan = roslibpy.Topic(self._ros, self._scan_name, TYPE_LASER_SCAN)
        self._scan.subscribe(callback)

    def publish_cmd_vel(self, command: Command) -> None:
        if self._cmd is None or not self.is_connected:
            return
        self._cmd.publish(roslibpy.Message(command.to_twist()))

    def stop(self) -> None:
        """Publish an explicit zero Twist.

        The bridge's 0.5 s watchdog would stop the base anyway, but only after half a
        second of coasting. Saying stop is both faster and unambiguous in the log.
        """
        self.publish_cmd_vel(Command())

    def close(self, *, hard_exit_after: Optional[float] = 2.0) -> None:
        """Stop the robot and tear the connection down."""
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
        except Exception:
            pass
        for topic in (self._odom, self._camera, self._scan):
            try:
                if topic is not None:
                    topic.unsubscribe()
            except Exception:
                pass
        try:
            if self._cmd is not None:
                self._cmd.unadvertise()
        except Exception:
            pass

        # roslibpy drives a Twisted reactor on a background thread and its shutdown is
        # historically prone to hanging. A hang here means Esc does not quit, which is
        # worse than an abrupt exit -- and by this point the robot has already been
        # sent a zero Twist, so there is nothing left to lose.
        bail: Optional[threading.Timer] = None
        if hard_exit_after:
            bail = threading.Timer(hard_exit_after, lambda: os._exit(0))
            bail.daemon = True
            bail.start()
        try:
            if self._ros is not None:
                self._ros.close()
                self._ros.terminate()
        except Exception:
            pass
        finally:
            if bail is not None:
                bail.cancel()

    # ---------------------------------------------------------------- helpers

    def describe(self) -> str:
        return f"ws://{self.host}:{self.port}"


def quiet_roslibpy_logging() -> None:
    """Silence roslibpy/twisted chatter so it does not scribble over the console."""
    import logging

    for name in ("roslibpy", "twisted", "autobahn", "ws4py"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    # Twisted writes unhandled-deferred noise to stderr at shutdown on some versions.
    if not sys.warnoptions:
        import warnings

        warnings.filterwarnings("ignore", module="twisted")
