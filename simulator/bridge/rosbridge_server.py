#!/usr/bin/env python
"""A minimal rosbridge v2.0 server, so the simulator can be driven as a ROS robot.

The simulator presents the same topics `elephantrobotics/myagv_ros` does, which lets
one teleop client drive either the simulated or the real myAGV without changing a line:

    teleop -> robot    cmd_vel                          geometry_msgs/Twist
    robot  -> teleop   odom                             nav_msgs/Odometry
    robot  -> teleop   /camera/image_raw/compressed     sensor_msgs/CompressedImage

Why implement the protocol rather than use rospy: MuJoCo needs the Homebrew framework
Python (for `mjpython`) while ROS on macOS comes from conda, and reconciling the two
would mean migrating the whole molmospaces stack. rosbridge is plain JSON over a
websocket, so serving it in-process costs far less than that migration — and the real
robot runs the stock `ros-noetic-rosbridge-suite`, so the client is identical.

Implemented ops: advertise, unadvertise, publish, subscribe, unsubscribe. No services,
no TF, no params — anything else gets a `status` warning rather than an error.

Standalone, for protocol testing without the simulator:

    python bridge/rosbridge_server.py --port 9090 --echo
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Any, Callable

import websockets.sync.server as ws_server

log = logging.getLogger("rosbridge")

DEFAULT_PORT = 9090

# The topics the myAGV presents, so both ends agree on names and types.
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOM = "/odom"
TOPIC_CAMERA = "/camera/image_raw/compressed"
# The 2023 Pi AGV ships a YDLidar publishing /scan (its bring-up script is documented in
# robots/myagv/urdf/UPSTREAM_README.md), so a simulated one belongs on the same topic:
# anything consuming it works against either robot unchanged.
TOPIC_SCAN = "/scan"
TOPIC_DEPTH = "/camera/depth/image_raw"
TOPIC_CAMERA_INFO = "/camera/rgb/camera_info"

TYPE_TWIST = "geometry_msgs/Twist"
TYPE_ODOM = "nav_msgs/Odometry"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
TYPE_LASER_SCAN = "sensor_msgs/LaserScan"
TYPE_IMAGE = "sensor_msgs/Image"
TYPE_CAMERA_INFO = "sensor_msgs/CameraInfo"


def normalise(topic: str) -> str:
    """ROS resolves a relative name against the node namespace; we only have the root.

    myagv_ros advertises `cmd_vel` while clients usually ask for `/cmd_vel`; treating
    them as the same name avoids a silent no-op subscription.
    """
    return topic if topic.startswith("/") else "/" + topic


def header(seq: int, frame_id: str) -> dict:
    """A std_msgs/Header. rosbridge expects stamp split into secs/nsecs, not a float."""
    now = time.time()
    return {
        "seq": seq,
        "stamp": {"secs": int(now), "nsecs": int((now % 1) * 1e9)},
        "frame_id": frame_id,
    }


def compressed_image(seq: int, jpeg: bytes, frame_id: str = "camera") -> dict:
    """sensor_msgs/CompressedImage.

    `data` is a uint8[], which rosbridge transports **base64-encoded**, not as a JSON
    array of integers. Getting this wrong produces a message that looks valid but
    decodes to garbage on the client.
    """
    return {
        "header": header(seq, frame_id),
        "format": "jpeg",
        "data": base64.b64encode(jpeg).decode("ascii"),
    }


def laser_scan(
    seq: int,
    ranges,
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    *,
    range_min: float = 0.05,
    range_max: float = 8.0,
    scan_time: float = 0.05,
    frame_id: str = "laser",
) -> dict:
    """sensor_msgs/LaserScan.

    Unlike CompressedImage, `ranges` is a float32[] and goes over the wire as a plain JSON
    array -- rosbridge base64-encodes uint8[] only. An out-of-range return is `inf` in ROS,
    which JSON cannot express, so misses are sent as `range_max + 1`; every consumer
    already has to treat anything above range_max as "no return".
    """
    values = [float(r) for r in ranges]
    return {
        "header": header(seq, frame_id),
        "angle_min": float(angle_min),
        "angle_max": float(angle_max),
        "angle_increment": float(angle_increment),
        "time_increment": float(scan_time / max(len(values), 1)),
        "scan_time": float(scan_time),
        "range_min": float(range_min),
        "range_max": float(range_max),
        "ranges": values,
        "intensities": [],
    }


def image(seq: int, data: bytes, encoding: str, width: int, height: int,
          frame_id: str = "camera") -> dict:
    """sensor_msgs/Image. `data` is a uint8[], so base64 as for CompressedImage."""
    step = len(data) // height if height else 0
    return {
        "header": header(seq, frame_id),
        "height": int(height),
        "width": int(width),
        "encoding": encoding,
        "is_bigendian": 0,
        "step": int(step),
        "data": base64.b64encode(data).decode("ascii"),
    }


def camera_info(seq: int, width: int, height: int, fovy_deg: float,
                frame_id: str = "camera") -> dict:
    """sensor_msgs/CameraInfo derived from a MuJoCo camera's vertical FOV.

    MuJoCo specifies `fovy` in degrees over the image height, so fy follows from it and fx
    equals fy -- the renderer has square pixels and no distortion, which is why D is zeros.
    """
    import math

    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    return {
        "header": header(seq, frame_id),
        "height": int(height),
        "width": int(width),
        "distortion_model": "plumb_bob",
        "D": [0.0] * 5,
        "K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }


# The constant covariance matrices myagv_odometry_node publishes, copied from
# myagv_ros/myagv_odometry/src/myAGV.cpp. All-zero covariance is not neutral: a
# consumer such as robot_pose_ekf reads it as "infinitely certain" and weights this
# odometry against the IMU accordingly, so a simulated robot sending zeros would fuse
# differently from the real one. The 1e6 entries mark z, roll and pitch as unobserved,
# which is exactly what a planar base knows about them.
ODOM_POSE_COVARIANCE = [
    1e-9, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 1e-9, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e-9,
]
ODOM_TWIST_COVARIANCE = list(ODOM_POSE_COVARIANCE)


def odometry(seq: int, x: float, y: float, yaw: float, vx: float, vy: float, wz: float) -> dict:
    """nav_msgs/Odometry in the frames myagv_odometry uses (odom -> base_footprint)."""
    import math

    return {
        "header": header(seq, "odom"),
        "child_frame_id": "base_footprint",
        "pose": {
            "pose": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": math.sin(yaw / 2.0),
                    "w": math.cos(yaw / 2.0),
                },
            },
            "covariance": ODOM_POSE_COVARIANCE,
        },
        "twist": {
            "twist": {
                "linear": {"x": vx, "y": vy, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": wz},
            },
            "covariance": ODOM_TWIST_COVARIANCE,
        },
    }


class RosBridgeServer:
    """Serves the rosbridge protocol on a websocket, for one or more clients."""

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._server: ws_server.Server | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        self._lock = threading.Lock()
        # connection -> set of topics that connection subscribed to
        self._clients: dict[Any, set[str]] = {}
        # topic -> callback invoked when a client publishes to it
        self._handlers: dict[str, Callable[[dict], None]] = {}
        self._seq = 0

    # -- lifecycle ---------------------------------------------------------------

    def on(self, topic: str, callback: Callable[[dict], None]) -> None:
        """Register a handler for messages clients publish to `topic`."""
        self._handlers[normalise(topic)] = callback

    def start(self) -> None:
        self._server = ws_server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info("rosbridge listening on ws://%s:%d", self._host, self._port)

    def stop(self) -> None:
        self._shutdown.set()
        with self._lock:
            connections = list(self._clients)
            self._clients.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- publishing --------------------------------------------------------------

    def publish(self, topic: str, msg: dict) -> None:
        """Send a message to every client subscribed to `topic`."""
        topic = normalise(topic)
        frame = json.dumps({"op": "publish", "topic": topic, "msg": msg})
        with self._lock:
            targets = [c for c, topics in self._clients.items() if topic in topics]
        for conn in targets:
            try:
                conn.send(frame)
            except Exception:
                # A client that went away is dropped on its handler thread; losing a
                # frame here must not interrupt the simulation loop.
                pass

    # -- protocol ----------------------------------------------------------------

    def _handler(self, websocket) -> None:
        log.info("client connected from %s", websocket.remote_address)
        with self._lock:
            self._clients[websocket] = set()
        try:
            for raw in websocket:
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    self._status(websocket, "error", "message was not valid JSON")
                    continue
                self._dispatch(websocket, message)
        except Exception as exc:
            log.debug("client loop ended: %s", exc)
        finally:
            with self._lock:
                self._clients.pop(websocket, None)
            log.info("client disconnected")

    def _dispatch(self, websocket, message: dict) -> None:
        op = message.get("op")
        topic = normalise(message.get("topic", "")) if message.get("topic") else None

        if op == "subscribe":
            with self._lock:
                self._clients.setdefault(websocket, set()).add(topic)
            log.info("client subscribed to %s", topic)

        elif op == "unsubscribe":
            with self._lock:
                self._clients.get(websocket, set()).discard(topic)

        elif op == "advertise":
            # Nothing to allocate: handlers are registered by the simulator, and an
            # advertise for a topic nobody consumes is harmless.
            log.info("client advertised %s (%s)", topic, message.get("type"))

        elif op == "unadvertise":
            pass

        elif op == "publish":
            handler = self._handlers.get(topic)
            if handler is None:
                self._status(websocket, "warning", f"nothing is listening on {topic}")
                return
            try:
                handler(message.get("msg") or {})
            except Exception as exc:
                log.exception("handler for %s failed", topic)
                self._status(websocket, "error", f"handler for {topic} failed: {exc}")

        elif op in ("set_level", "status"):
            pass  # client-side logging controls; nothing to do

        else:
            self._status(websocket, "warning", f"unsupported op {op!r}")

    def _status(self, websocket, level: str, msg: str) -> None:
        try:
            websocket.send(json.dumps({"op": "status", "level": level, "msg": msg}))
        except Exception:
            pass
        if level != "info":
            log.warning("%s: %s", level, msg)


def main() -> int:
    """Standalone mode: echo whatever is published to cmd_vel, for protocol testing."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--echo", action="store_true", help="log every cmd_vel received")
    ap.add_argument("--odom-hz", type=float, default=10.0, dest="odom_hz")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    server = RosBridgeServer(args.host, args.port)

    latest = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    def on_cmd_vel(msg: dict) -> None:
        latest["vx"] = float(msg.get("linear", {}).get("x", 0.0))
        latest["vy"] = float(msg.get("linear", {}).get("y", 0.0))
        latest["wz"] = float(msg.get("angular", {}).get("z", 0.0))
        if args.echo:
            log.info("cmd_vel %s", latest)

    server.on(TOPIC_CMD_VEL, on_cmd_vel)
    server.start()

    # Dead-reckon the echoed velocity so a standalone client sees odom move.
    x = y = yaw = 0.0
    dt = 1.0 / args.odom_hz
    try:
        import math

        while True:
            time.sleep(dt)
            c, s = math.cos(yaw), math.sin(yaw)
            x += (latest["vx"] * c - latest["vy"] * s) * dt
            y += (latest["vx"] * s + latest["vy"] * c) * dt
            yaw += latest["wz"] * dt
            server.publish(
                TOPIC_ODOM,
                odometry(server.next_seq(), x, y, yaw, latest["vx"], latest["vy"], latest["wz"]),
            )
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
