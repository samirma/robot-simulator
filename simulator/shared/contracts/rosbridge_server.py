#!/usr/bin/env python
"""A minimal rosbridge v2.0 server, so the simulator can be driven as a ROS robot.

Each robot brings its own topic set, because each vendor's ROS interface is its own
thing: `robots/myagv/ros_surface.py` presents what `elephantrobotics/myagv_ros` does, and
`robots/ainex/ros_surface.py` what `Hiwonder/ainex` does. The two have **no topic in
common** -- the AiNex has no `/cmd_vel` and no `/odom` at all. The constants below are the
myAGV's, kept here because they were here first and several tools import them; a robot
whose contract differs declares its own (see `robots/ainex/topics.py`).

The myAGV contract, as an example of the shape:

    teleop -> robot    cmd_vel                          geometry_msgs/Twist
    robot  -> teleop   odom                             nav_msgs/Odometry
    robot  -> teleop   /camera/image_raw/compressed     sensor_msgs/CompressedImage

Why implement the protocol rather than use rospy: MuJoCo needs the Homebrew framework
Python (for `mjpython`) while ROS on macOS comes from conda, and reconciling the two
would mean migrating the whole molmospaces stack. rosbridge is plain JSON over a
websocket, so serving it in-process costs far less than that migration — and the real
robot runs the stock `ros-noetic-rosbridge-suite`, so the client is identical.

Implemented ops: advertise, unadvertise, publish, subscribe, unsubscribe, call_service.
`advertise_service` and `unadvertise_service` are accepted as no-ops — nothing here
consumes a client-provided service. TF and `robot_description` are here now: the
transforms are ordinary topics (`contracts/tf.py`, fed by `ros_surfaces/tf_stream.py`)
and the description is a parameter `rosapi` answers for. Anything else gets a `status`
warning rather than an error.

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


# The namespacing rule lives in `namespace.py`, which is stdlib-only so the console's
# cross-project contract test can load it by path and pin the rule itself. Re-exported
# here because every existing caller imports these two from this module.
try:
    from contracts.namespace import RobotNamespace, normalise, ns_frame, ns_topic
except ImportError:  # run as a script, with this directory on the path rather than shared/
    from namespace import RobotNamespace, normalise, ns_frame, ns_topic

__all_namespace__ = ("RobotNamespace", "normalise", "ns_frame", "ns_topic")


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


# --------------------------------------------------------------------- ROS 2 builders
#
# The myAGV builders above are ROS 1 shaped, because that is what the vendor stack is:
# single-slash type strings and a `secs`/`nsecs` stamp. The SO-101 arm is the other
# contract on this server -- ROS 2 Jazzy, `pkg/msg/Type`, and a `sec`/`nanosec` stamp --
# so it gets its own builders rather than a flag on these. Two robots, two vendor
# realities; a client that had to guess which dialect a stamp was in would be a worse
# thing than a little duplication.
#
# Every one of these takes an explicit `stamp_s`, and the arm surface passes **simulated**
# time (`MjData.time`), never the wall clock. That is load bearing in three places: the
# success predicate holds for >= 1.0 s *of simulated time*, the client refuses to start
# if simulated time is not advancing against the wall clock, and the offline scorer
# re-derives the hold from these stamps. Handing it `time.time()` makes all three agree
# on an answer that has nothing to do with the simulation.

TYPE_JOINT_STATE = "sensor_msgs/msg/JointState"
TYPE_JOINT_TRAJECTORY = "trajectory_msgs/msg/JointTrajectory"
TYPE_FLOAT64_MULTI_ARRAY = "std_msgs/msg/Float64MultiArray"
TYPE_BOOL = "std_msgs/msg/Bool"
TYPE_COMPRESSED_IMAGE_ROS2 = "sensor_msgs/msg/CompressedImage"
# The arm's `/reset` services answer `{success, message}` -- the shape of std_srvs/Trigger,
# which is what a `mujoco_ros2_control` reset service is on the reference rig.
SRV_TYPE_TRIGGER = "std_srvs/srv/Trigger"
# Namespaced by the publishing plugin in the reference rig, and the client's settings
# name it in full; it is a custom message, which over rosbridge JSON is just this shape.
TYPE_FREE_JOINT_STATE_ARRAY = "mujoco_ros2_control_msgs/msg/FreeJointStateArray"


def header_ros2(frame_id: str, stamp_s: float) -> dict:
    """A ROS 2 std_msgs/Header: `sec`/`nanosec`, and no `seq` field at all.

    ROS 2 dropped `seq` from Header. Sending one anyway is harmless over rosbridge JSON,
    but leaving it out is what a real ROS 2 publisher looks like on the wire, and this
    contract is supposed to be indistinguishable from one.
    """
    seconds = int(stamp_s)
    return {
        "stamp": {"sec": seconds, "nanosec": int(round((stamp_s - seconds) * 1e9))},
        "frame_id": frame_id,
    }


def compressed_image_ros2(seq: int, jpeg: bytes, stamp_s: float, frame_id: str = "camera") -> dict:
    """sensor_msgs/msg/CompressedImage. `data` is base64, and `format` must say jpeg.

    The client decodes base64 JPEG or PNG and nothing else, and it matches `format` on
    containing "jpeg" -- real `image_transport` sends the longer
    "rgb8; jpeg compressed bgr8", so both spellings have to keep working.
    """
    del seq  # ROS 2 headers carry no sequence number; kept for call-site symmetry.
    return {
        "header": header_ros2(frame_id, stamp_s),
        "format": "jpeg",
        "data": base64.b64encode(jpeg).decode("ascii"),
    }


def joint_state(names, positions, velocities, stamp_s: float, frame_id: str = "") -> dict:
    """sensor_msgs/msg/JointState, with names and values sorted by name.

    **Sorting is not cosmetic.** The reference ROS 2 rig's `joint_state_broadcaster`
    returns names alphabetically, which for this arm shares no index with the contract
    order -- `elbow_flex_joint` first, `shoulder_pan_joint` fourth. A client that read
    by position instead of by name would be wrong about every joint, and would look
    plausible while doing it. Sorting here means the console's real-hardware code path
    is exercised against the same hazard the real broadcaster presents, so the bug
    cannot hide until someone plugs in an arm.
    """
    order = sorted(range(len(names)), key=lambda i: names[i])
    return {
        "header": header_ros2(frame_id, stamp_s),
        "name": [names[i] for i in order],
        "position": [float(positions[i]) for i in order],
        "velocity": [float(velocities[i]) for i in order],
        "effort": [],
    }


def free_joint_state_array(entries, stamp_s: float, frame_id: str = "world") -> dict:
    """mujoco_ros2_control_msgs/msg/FreeJointStateArray.

    The field is `free_joints`, **not** `states`, and every consumer selects its body by
    name rather than by index -- so adding a body to this list is always safe and
    reordering it is always harmless. `entries` is an iterable of
    `(name, (x, y, z), (qw, qx, qy, qz), (vx, vy, vz), (wx, wy, wz))`.
    """
    free_joints = []
    for name, pos, quat, lin, ang in entries:
        stamped = header_ros2(frame_id, stamp_s)
        free_joints.append(
            {
                "name": name,
                "pose": {
                    "header": stamped,
                    "pose": {
                        "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                        "orientation": {
                            "w": float(quat[0]),
                            "x": float(quat[1]),
                            "y": float(quat[2]),
                            "z": float(quat[3]),
                        },
                    },
                },
                "twist": {
                    "header": stamped,
                    "twist": {
                        "linear": {"x": float(lin[0]), "y": float(lin[1]), "z": float(lin[2])},
                        "angular": {"x": float(ang[0]), "y": float(ang[1]), "z": float(ang[2])},
                    },
                },
            }
        )
    return {"header": header_ros2(frame_id, stamp_s), "free_joints": free_joints}


def laser_scan(
    seq: int,
    ranges,
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    *,
    range_min: float = 0.1,
    range_max: float = 12.0,
    scan_time: float = 0.1,
    frame_id: str = "laser_frame",
) -> dict:
    """sensor_msgs/LaserScan, matching what the 2023 Pi AGV's lidar publishes.

    The defaults are the YDLidar X2's, read off `ydlidar_ros_driver/launch/X2.launch` on
    the `myagv_ros_2023Pi` branch: `frame_id: laser_frame`, `range_min: 0.1`,
    `range_max: 12.0`, `frequency: 10.0` (hence the 0.1 s scan_time). A consumer written
    against the real robot sees the same numbers here.

    Unlike CompressedImage, `ranges` is a float32[] and goes over the wire as a plain JSON
    array -- rosbridge base64-encodes uint8[] only.

    Two deliberate departures from the X2, both of which a client must tolerate anyway:

    - Misses are sent as `range_max + 1`. In ROS they would be `inf`, which JSON cannot
      express; the real driver runs with `invalid_range_is_inf: false` and reports `0.0`.
      Anything outside [range_min, range_max] means "no return" under all three
      conventions, which is the test a client should be applying.
    - The X2 is launched with `ignore_array: "-50,50"`, a blind wedge where the chassis
      occludes it. That is not modelled: its orientation cannot be confirmed without the
      hardware, and guessing wrong would carve free space out of a real obstacle.
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


def odometry(seq: int, x: float, y: float, yaw: float, vx: float, vy: float, wz: float,
             frame_id: str = "odom", child_frame_id: str = "base_footprint") -> dict:
    """nav_msgs/Odometry in the frames myagv_odometry uses (odom -> base_footprint).

    The frames are arguments rather than literals because a fleet prefixes them: two bases
    on one graph both report `odom -> base_footprint` otherwise, and a consumer building a
    tf tree out of that gets one frame with two parents. `ns_frame` is what supplies the
    prefixed pair; the defaults are the single-robot contract.
    """
    import math

    return {
        "header": header(seq, frame_id),
        "child_frame_id": child_frame_id,
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
        # service name -> callback returning the response's `values`
        self._services: dict[str, Callable[[dict], dict]] = {}
        # topic -> ROS type string, learned from what has actually been published. This
        # is what `rosapi` answers from; see `serve_rosapi`.
        self._published_types: dict[str, str] = {}
        # topic -> ROS type string for what this server *accepts*, declared by `on`. Kept
        # apart from `_published_types` because the two are learned differently: a
        # publication announces its own type the first time it goes out, a subscription
        # has to be told.
        self._subscribed_types: dict[str, str] = {}
        # service name -> ROS service type, declared by `service`. The same gap `on` had:
        # a type is not needed to route a call, and without one `/rosapi/services` can
        # list names while `/rosapi/service_type` has nothing to answer with.
        self._service_types: dict[str, str] = {}
        # name (topic or service) -> the namespace that declared it, filled by
        # `NamespacedBus`. It is what lets `publishers`, `subscribers`, `nodes` and
        # `node_details` answer from something real: a namespace is one robot's surface,
        # which is what a vendor bringup runs as one node.
        self._owner: dict[str, str] = {}
        # parameter name -> value, for the parameters a real bringup would have set.
        # `robot_description` is the one that matters: `robot_state_publisher` reads the
        # URDF from it, and so does every client that draws a robot. This server used to
        # answer `get_param` with the empty string unconditionally, which is what left a
        # client able to see a robot's joint angles and unable to learn what its body is.
        self._params: dict[str, Any] = {}
        self._seq = 0

    # -- lifecycle ---------------------------------------------------------------

    def on(
        self, topic: str, callback: Callable[[dict], None], message_type: str | None = None
    ) -> None:
        """Register a handler for messages clients publish to `topic`.

        `message_type` is optional only because it is not needed to *route* a message --
        but pass it, because it is what makes the topic discoverable. A real `rosapi`
        lists a node's subscriptions alongside its publications, so a client can find out
        how to command a robot as well as how to observe it; with no type recorded here
        the command topics are invisible to discovery and only the published ones answer
        `/rosapi/topics`.
        """
        topic = normalise(topic)
        # Refuse rather than overwrite. This dict is one callback per topic, so before
        # this check a second robot attaching the same surface to the same server took
        # the first one's `/cmd_vel` away in silence -- the first robot then sat still
        # while both published `/odom` onto one topic, which reads as a physics bug and
        # is not one. Namespacing is what makes a fleet legal; a collision means the
        # namespaces were not applied, and that is worth failing on.
        if topic in self._handlers:
            raise ValueError(
                f"{topic} already has a handler on this server. Two robots sharing one "
                "bridge must each be namespaced (see ns_topic); without that they "
                "silently overwrite each other's command topics."
            )
        self._handlers[topic] = callback
        if message_type is not None:
            self._subscribed_types[topic] = message_type

    def service(
        self, name: str, callback: Callable[[dict], dict], service_type: str | None = None
    ) -> None:
        """Register a handler for `call_service` on `name`.

        The handler receives the request's `args` and returns the response's `values`.
        Raising is reported as `result: false` with the message in `values`, which is what
        rosbridge does for a service that threw -- a caller is blocked on the response, so
        it needs an answer either way.

        `service_type` is optional only because it is not needed to *route* a call -- but
        pass it, for the same reason `on` asks for a message type: it is what makes the
        service discoverable through `/rosapi/services` and `/rosapi/service_type`, and
        what `/rosapi/service_request_details` resolves the schema from.

        Like `on`, the handler runs on the calling client's reader thread rather than on
        the simulation thread, so it may only touch small shared state; never MjData.
        """
        name = normalise(name)
        if name in self._services:
            raise ValueError(f"service {name} is already registered on this server")
        self._services[name] = callback
        if service_type is not None:
            self._service_types[name] = service_type

    def claim(self, namespace: str, name: str) -> None:
        """Record that `namespace`'s surface declared `name` (a topic or a service)."""
        self._owner[normalise(name)] = namespace

    def set_param(self, name: str, value: Any) -> None:
        """Set a ROS parameter, as a bringup's launch file would.

        Refuses to overwrite for the same reason `on` does: two robots writing one
        `robot_description` means the namespaces were not applied, and the client would
        then draw both of them with whichever body won.
        """
        name = normalise(name)
        if name in self._params:
            raise ValueError(
                f"parameter {name} is already set on this server. Two robots sharing one "
                "bridge must each be namespaced (see ns_topic); without that they "
                "silently overwrite each other's description."
            )
        self._params[name] = value

    def serve_rosapi(self) -> None:
        """Answer the `rosapi` introspection queries a real rosbridge answers.

        Real rosbridge ships `rosapi` alongside it -- `rosbridge_websocket.launch` starts
        `rosapi_node` unconditionally, and a real AiNex's launch file does exactly that
        with every glob set to `[*]` -- so clients written against a real bridge assume
        all of it. This used to implement two queries, the two `live_cameras.html` made,
        and a client asking anything else got `no service`: not "unknown type", not an
        empty list, but a bridge that could not be asked what a topic's type was. That is
        the largest way a client could tell this simulator from the robot it claims to be
        indistinguishable from, and it was found by a client doing nothing unusual.

        Every answer below is true *of this simulator*. Where that differs from what the
        hardware would say, it is said here rather than papered over:

        * `topics`, `topic_type`, `topics_for_type`, `services`, `service_type` -- from the
          server's own tables. These *are* the contract, and contract tests already hold
          them equal to the console's copies.
        * `message_details`, `service_request_details`, `service_response_details` -- from
          `message_schemas`, transcribed from each manufacturer's definition files with
          provenance recorded there.
        * `publishers`, `subscribers`, `nodes`, `node_details` -- the **namespace** that
          declared the name. A real robot returns node names (`/ainex_controller`); there
          are no nodes here, and a namespace -- one robot's surface -- is the closest true
          statement. No client in this project uses node names.
        * `get_param_names`, `get_param` -- the parameters a surface actually set, which
          in practice means each robot's `robot_description`. They answered empty until
          the transform tree went in, and that was the other half of the same gap: a
          client could be told a frame's name and never what the body in it looks like.
          A real robot's parameter server holds a great deal more than one URDF per
          robot, and none of the rest is here -- a difference, stated.
        * `action_servers` -- empty, which is true: this simulator runs none by design
          (the gripper is a topic; see CLAUDE.md).
        * `get_ros_version` -- 1, the dialect of this rosapi surface itself. The graph
          carries **two dialects by design** (ROS 1 for the myAGV and AiNex, ROS 2 for the
          SO-101), so a client must read per-topic types from `topics` and never infer
          them from this.

        `topics` reports what has actually been published at least once -- not what was
        advertised, because on this server nothing advertises: publishers are the surface
        code, not clients -- plus the command topics the surface declared to `on`, which is
        the half a client needs to discover how to *drive* the robot rather than only how
        to watch it. `topics_for_type` deliberately answers from publications alone. Its
        one caller asks for everything publishing CompressedImage and subscribes to the
        answer, so folding subscriptions in could only ever hand it a topic to listen to
        that nothing sends.
        """
        from contracts import message_schemas as schemas

        # Idempotent: a fleet's owner calls this once, but a single-robot surface used to
        # call it for itself, and `service()` refuses a duplicate registration.
        if "/rosapi/topics" in self._services:
            return

        def _known_topics() -> dict[str, str]:
            return {**self._subscribed_types, **self._published_types}

        def _owners_of(name: str) -> list[str]:
            owner = self._owner.get(normalise(name))
            return [f"/{owner}"] if owner else []

        def topics(_args: dict) -> dict:
            known = _known_topics()
            names = sorted(known)
            return {"topics": names, "types": [known[n] for n in names]}

        def topics_for_type(args: dict) -> dict:
            wanted = args.get("type")
            return {"topics": sorted(n for n, t in self._published_types.items() if t == wanted)}

        def topic_type(args: dict) -> dict:
            return {"type": _known_topics().get(normalise(args.get("topic", "")), "")}

        def services(_args: dict) -> dict:
            return {"services": sorted(self._services)}

        def service_type(args: dict) -> dict:
            return {"type": self._service_types.get(normalise(args.get("service", "")), "")}

        def publishers(args: dict) -> dict:
            name = normalise(args.get("topic", ""))
            return {"publishers": _owners_of(name) if name in self._published_types else []}

        def subscribers(args: dict) -> dict:
            name = normalise(args.get("topic", ""))
            return {"subscribers": _owners_of(name) if name in self._subscribed_types else []}

        def nodes(_args: dict) -> dict:
            return {"nodes": sorted({f"/{ns}" for ns in self._owner.values()})}

        def node_details(args: dict) -> dict:
            node = str(args.get("node", "")).lstrip("/")
            mine = {n for n, ns in self._owner.items() if ns == node}
            return {
                "subscribing": sorted(n for n in mine if n in self._subscribed_types),
                "publishing": sorted(n for n in mine if n in self._published_types),
                "services": sorted(n for n in mine if n in self._services),
            }

        def message_details(args: dict) -> dict:
            return {"typedefs": schemas.typedefs(str(args.get("type", "")))}

        def service_request_details(args: dict) -> dict:
            return {"typedefs": schemas.service_typedefs(str(args.get("type", "")), "request")}

        def service_response_details(args: dict) -> dict:
            return {"typedefs": schemas.service_typedefs(str(args.get("type", "")), "response")}

        def get_param_names(_args: dict) -> dict:
            return {"names": sorted(self._params)}

        def get_param(args: dict) -> dict:
            """rosapi returns a parameter **JSON-encoded**, and a client decodes it.

            `rosapi/GetParam` declares `string value`, and the real node fills it with
            `json.dumps(rospy.get_param(...))` -- which is why roslibpy's `Param.get`
            runs the answer back through `json.loads`. Handing back the raw URDF instead
            would decode as a JSON syntax error at the client, so a robot's description
            would arrive as a parse failure rather than as a body.
            """
            name = normalise(str(args.get("name", "")))
            if name in self._params:
                return {"value": json.dumps(self._params[name])}
            # rosapi answers an unset parameter with the caller's own default, verbatim.
            return {"value": args.get("default", "")}

        def action_servers(_args: dict) -> dict:
            return {"action_servers": []}

        def get_ros_version(_args: dict) -> dict:
            return {"version": 1, "distro": ""}

        for name, handler, stype in (
            ("/rosapi/topics", topics, "rosapi/Topics"),
            ("/rosapi/topics_for_type", topics_for_type, "rosapi/TopicsForType"),
            ("/rosapi/topic_type", topic_type, "rosapi/TopicType"),
            ("/rosapi/services", services, "rosapi/Services"),
            ("/rosapi/service_type", service_type, "rosapi/ServiceType"),
            ("/rosapi/publishers", publishers, "rosapi/Publishers"),
            ("/rosapi/subscribers", subscribers, "rosapi/Subscribers"),
            ("/rosapi/nodes", nodes, "rosapi/Nodes"),
            ("/rosapi/node_details", node_details, "rosapi/NodeDetails"),
            ("/rosapi/message_details", message_details, "rosapi/MessageDetails"),
            ("/rosapi/service_request_details", service_request_details,
             "rosapi/ServiceRequestDetails"),
            ("/rosapi/service_response_details", service_response_details,
             "rosapi/ServiceResponseDetails"),
            ("/rosapi/get_param_names", get_param_names, "rosapi/GetParamNames"),
            ("/rosapi/get_param", get_param, "rosapi/GetParam"),
            ("/rosapi/action_servers", action_servers, "rosapi/GetActionServers"),
            ("/rosapi/get_ros_version", get_ros_version, "rosapi/GetROSVersion"),
        ):
            self.service(name, handler, stype)

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

    def publish(self, topic: str, msg: dict, message_type: str | None = None) -> None:
        """Send a message to every client subscribed to `topic`."""
        topic = normalise(topic)
        if message_type is not None:
            self._published_types.setdefault(normalise(topic), message_type)
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

        elif op == "call_service":
            # rosbridge names the field `service`, not `topic`. And unlike `publish` --
            # where this bridge deliberately ignores ids -- the reply MUST echo `id`: the
            # caller is blocked waiting on it, so dropping it hangs the client rather than
            # failing it, which is a much worse failure to debug.
            name = normalise(message.get("service", ""))
            call_id = message.get("id")
            handler = self._services.get(name)
            if handler is None:
                self._service_response(
                    websocket, name, call_id, False, {"message": f"no service {name}"}
                )
                return
            try:
                values = handler(message.get("args") or {}) or {}
            except Exception as exc:
                log.exception("service %s failed", name)
                self._service_response(
                    websocket, name, call_id, False, {"message": str(exc)}
                )
                return
            self._service_response(websocket, name, call_id, True, values)

        elif op in ("advertise_service", "unadvertise_service"):
            # A client offering a service of its own. Nothing here consumes one, and
            # roslibpy advertises eagerly, so this is a no-op like `advertise`.
            log.info(
                "client advertised service %s (%s)",
                message.get("service"),
                message.get("type"),
            )

        elif op in ("set_level", "status"):
            pass  # client-side logging controls; nothing to do

        else:
            self._status(websocket, "warning", f"unsupported op {op!r}")

    def _service_response(self, websocket, service: str, call_id, result: bool,
                          values: dict) -> None:
        frame = {"op": "service_response", "service": service,
                 "values": values, "result": result}
        if call_id is not None:
            frame["id"] = call_id
        try:
            websocket.send(json.dumps(frame))
        except Exception:
            pass

    def _status(self, websocket, level: str, msg: str) -> None:
        try:
            websocket.send(json.dumps({"op": "status", "level": level, "msg": msg}))
        except Exception:
            pass
        if level != "info":
            log.warning("%s: %s", level, msg)


class NamespacedBus:
    """One robot's view of a shared `RosBridgeServer`.

    A surface takes a bus instead of a server and otherwise keeps its bare topic
    constants: `bus.on(TOPIC_CMD_VEL, ...)` registers `/myagv/cmd_vel`. That is the whole
    of what a surface has to know about sharing a graph, which is the point -- the
    alternative was every surface composing prefixes at every call site, where one missed
    name is a topic silently landing in another robot's namespace.

    A bus deliberately does NOT expose `start`/`stop`. The server is owned by whatever
    built it; a surface that stopped it would take every other robot on the port down.
    """

    def __init__(self, server: "RosBridgeServer", namespace: str = "") -> None:
        self.server = server
        self.ns = RobotNamespace(namespace)
        self.published: list[str] = []
        self.subscribed: list[str] = []
        # Per-bus, not per-server. `header.seq` is a per-publisher counter on real ROS 1,
        # and the console reads it to line video up against the command log
        # (`robot_console/camera.py:header_seq`). Sharing one counter across robots makes
        # every robot's seq skip by however many messages its neighbours sent.
        self._seq = 0

    # -- naming ------------------------------------------------------------------

    def topic(self, topic: str) -> str:
        return self.ns.topic(topic)

    def frame(self, frame_id: str) -> str:
        return self.ns.frame(frame_id)

    # There used to be a `sibling(namespace)` here, for the arm's surface to reach the
    # worktop rig's namespace from inside its own loop. The rig is a fleet member now
    # (`ros_surfaces/scene.py`), with a bus of its own from `RobotFleet.bus`, so no
    # surface needs to speak for a name that is not its own any more.

    # -- the server's surface, namespaced ----------------------------------------

    # Every name that goes through here is claimed for this namespace on the server.
    # That is what `/rosapi/publishers`, `subscribers`, `nodes` and `node_details` answer
    # from, and it is recorded here rather than in each surface because this is the one
    # place every name of a robot's already passes.
    def on(self, topic: str, callback: Callable[[dict], None],
           message_type: str | None = None) -> None:
        name = self.ns.topic(topic)
        self.server.on(name, callback, message_type)
        self.server.claim(self.ns.name, name)
        self.subscribed.append(name)

    def service(self, name: str, callback: Callable[[dict], dict],
                service_type: str | None = None) -> None:
        full = self.ns.service(name)
        self.server.service(full, callback, service_type)
        self.server.claim(self.ns.name, full)

    def set_param(self, name: str, value: Any) -> None:
        """Set one of this robot's parameters, namespaced like everything else it has.

        `robot_description` is a relative name in ROS, so a bringup inside
        `<group ns="myagv">` puts it at `/myagv/robot_description` -- which is where a
        client that found the robot by namespace then looks for its body.
        """
        full = self.ns.topic(name)
        self.server.set_param(full, value)
        self.server.claim(self.ns.name, full)

    def publish(self, topic: str, msg: dict, message_type: str | None = None) -> None:
        name = self.ns.topic(topic)
        if message_type is not None and name not in self.published:
            self.published.append(name)
            self.server.claim(self.ns.name, name)
        self.server.publish(name, msg, message_type)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def client_count(self) -> int:
        """How many clients the whole server has -- not this robot's share.

        rosbridge does not tell a publisher who is subscribed to what, so per-robot is
        not a number this transport can produce. The AiNex surface uses it only to decide
        whether anyone is listening at all, which this answers correctly.
        """
        return self.server.client_count


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

    server.on(TOPIC_CMD_VEL, on_cmd_vel, TYPE_TWIST)
    # The same introspection surface the fleet serves, so protocol testing against this
    # standalone bridge sees what a client of the real thing sees.
    server.serve_rosapi()
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
