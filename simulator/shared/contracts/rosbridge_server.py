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

Accepted ops are exactly the spec's (§3, "Transport protocol"): `advertise`,
`unadvertise`, `publish`, `subscribe`, `unsubscribe`, `call_service`,
`advertise_service`, `unadvertise_service`, `set_level`, `status`, and the ROS 2 action
ops `advertise_action`, `unadvertise_action`, `send_action_goal`, `cancel_action_goal`
(answered with `action_feedback` and `action_result`, whose `status` is the
`action_msgs/GoalStatus` code; each transition also goes out on the action's hidden
`<action>/_action/status` topic). A client that advertised a service or action may also
send the replies that advertisement implies (`service_response`, `action_feedback`,
`action_result`). Anything else, and anything malformed, gets a `status` error.

Latched publishers deliver each publisher's last message once to each new subscriber,
and are never republished. A client's action goal ends only by its result, by
`cancel_action_goal`, or ABORTED by `/reset` (`abort_goals`); disconnecting leaves it
running. `serve_rosapi` answers the spec's 31 `rosapi` services.

Standalone, for protocol testing without the simulator:

    python bridge/rosbridge_server.py --port 9090 --echo
"""

from __future__ import annotations

import base64
import collections
import json
import logging
import socket
import threading
import time
import uuid
from typing import Any, Callable

import websockets.sync.server as ws_server

log = logging.getLogger("rosbridge")

DEFAULT_PORT = 9090

# A few of the myAGV's names, kept for the standalone echo server below. The myAGV's
# whole interface is `ros_surfaces/myagv.py`.
TOPIC_CMD_VEL = "/cmd_vel"
TOPIC_ODOM = "/odom"
TOPIC_CAMERA = "/camera/image_raw/compressed"
TOPIC_SCAN = "/scan"

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


def header(seq: int, frame_id: str, stamp_s: float | None = None) -> dict:
    """A std_msgs/Header. rosbridge expects stamp split into secs/nsecs, not a float.

    Every simulated member passes `stamp_s`, the simulated time (spec §3: one clock on
    the wire); the wall clock is only the default for a caller outside a simulation.
    """
    now = time.time() if stamp_s is None else float(stamp_s)
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
# The workspace-owned `/reset` (spec §3) is a std_srvs/srv/Trigger.
SRV_TYPE_TRIGGER = "std_srvs/srv/Trigger"


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
                frame_id: str = "camera", stamp_s: float | None = None) -> dict:
    """sensor_msgs/CameraInfo derived from a MuJoCo camera's vertical FOV.

    MuJoCo specifies `fovy` in degrees over the image height, so fy follows from it and fx
    equals fy -- the renderer has square pixels and no distortion, which is why D is zeros.
    """
    import math

    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    return {
        "header": header(seq, frame_id, stamp_s),
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
             frame_id: str = "odom", child_frame_id: str = "base_footprint",
             stamp_s: float | None = None) -> dict:
    """nav_msgs/Odometry in the frames myagv_odometry uses (odom -> base_footprint).

    The frames are arguments rather than literals because a fleet prefixes them: two bases
    on one graph both report `odom -> base_footprint` otherwise, and a consumer building a
    tf tree out of that gets one frame with two parents. `ns_frame` is what supplies the
    prefixed pair; the defaults are the single-robot contract.
    """
    import math

    return {
        "header": header(seq, frame_id, stamp_s),
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


# --------------------------------------------------------------------- the transport
#
# The protocol side. What a client may send is exactly the list in the spec (§3,
# "Transport protocol"); everything else, and anything malformed, earns a rosbridge
# `status` error rather than silence, because silence is what a client cannot debug.

#: The operations a client may send. The first ten are the spec's list; the four action
#: ops are its ROS 2 additions. `service_response`, `action_feedback` and `action_result`
#: are accepted only as the other half of a client's own `advertise_service` /
#: `advertise_action` -- a client that advertised a service has to be able to answer it.
CLIENT_OPS = frozenset({
    "advertise", "unadvertise", "publish", "subscribe", "unsubscribe",
    "call_service", "advertise_service", "unadvertise_service", "set_level", "status",
    "advertise_action", "unadvertise_action", "send_action_goal", "cancel_action_goal",
})
PROVIDER_REPLY_OPS = frozenset({"service_response", "action_feedback", "action_result"})

#: `set_level` thresholds. A client receives a status message when its level is at least
#: as verbose as the message's; `none` silences everything. rosbridge's default is `error`.
STATUS_LEVELS = {"none": 0, "error": 1, "warning": 2, "info": 3}
DEFAULT_STATUS_LEVEL = "error"


class GoalStatus:
    """`action_msgs/msg/GoalStatus`'s enumeration, which `action_result.status` carries."""

    UNKNOWN = 0
    ACCEPTED = 1
    EXECUTING = 2
    CANCELING = 3
    SUCCEEDED = 4
    CANCELED = 5
    ABORTED = 6

    TERMINAL = frozenset({SUCCEEDED, CANCELED, ABORTED})


class GlobalName(str):
    """A node name a `NamespacedBus` must take literally rather than compose.

    Node names are composed with the member's namespace like every other name (a ROS 2
    bringup under `-r __ns:=/so101` namespaces its nodes too). The exception is a node
    that is not the vendor's -- `/simulator`, which provides the workspace-owned `/reset`
    -- and wrapping it in this type is how a surface says so.
    """


#: The node that provides the workspace-owned `/reset` (spec §3).
SIMULATOR_NODE = GlobalName("/simulator")
#: The protocol-defined runtime nodes: `rosapi`'s services belong to `/rosapi`, and every
#: client's advertisement (and subscription) to `/rosbridge_websocket`.
ROSAPI_NODE = "/rosapi"
BRIDGE_NODE = "/rosbridge_websocket"
#: What `/rosapi/get_ros_version` answers. See `serve_rosapi` for why this is ROS 2.
ROS_VERSION = 2
ROS_DISTRO = "jazzy"


class _Client:
    """One websocket's protocol state, and the queue its frames leave through.

    **Nothing that publishes ever waits on a client.** A send on a websocket blocks once
    the socket's buffer is full, and a client that reads slowly -- a laptop on wifi
    subscribed to three cameras -- would otherwise block whichever thread published: the
    simulation loop itself, for most topics, and with it every robot's rate for every
    client on the port. So each client has a queue and a sender thread of its own. A
    published topic message is kept per topic, at most `TOPIC_QUEUE` deep, the oldest
    dropped first -- a slow subscriber to a periodic topic gets fewer, fresher messages,
    as a ROS subscriber with a short queue does. Everything else (service responses,
    action frames, statuses, a latched message delivered on subscribe) is queued whole
    and in order.

    Frames leave in two lanes, small before bulk: a 100 Hz odometry message must not wait
    behind a megabyte of raw image, which on a real graph travels on a connection of its
    own. Within a lane, and within a topic, order is kept.
    """

    #: Topic messages a client may have waiting per topic before the oldest is dropped.
    TOPIC_QUEUE = 10
    #: Frames at least this long go in the bulk lane.
    BULK_BYTES = 65536

    def __init__(self, websocket) -> None:
        self.ws = websocket
        self.level = DEFAULT_STATUS_LEVEL
        # topic -> subscription ids (None for a subscription that gave none)
        self.subs: dict[str, set] = {}
        # topic -> {"type": str, "ids": set, "latched": bool}
        self.adverts: dict[str, dict] = {}
        self.services: dict[str, str] = {}   # advertised service -> type
        self.actions: dict[str, str] = {}    # advertised action -> type
        # (action, client goal id) -> goal this client sent. Outlives the connection:
        # a client that disconnects leaves its goals running (spec §2.2).
        self.goals: dict[tuple[str, Any], "ActionGoal"] = {}
        self.open = True
        self.dropped = 0
        self._cond = threading.Condition()
        #: What to send next, per lane, in order: (None, frame) for a frame, (topic, None)
        #: for the oldest waiting message of `topic`. Lane 0 is small frames, 1 bulk.
        self._lanes = (collections.deque(), collections.deque())
        self._topics: dict[str, collections.deque] = {}
        self._sender = threading.Thread(target=self._drain, daemon=True,
                                        name="rosbridge-client-sender")
        self._sender.start()

    @property
    def latch_key(self) -> tuple:
        return ("client", id(self))

    def send(self, frame: str, topic: str | None = None) -> bool:
        """Queue `frame` for this client and return at once; False once it has gone.

        With `topic`, the frame is one message of that topic and may be dropped for a
        newer one if the client falls `TOPIC_QUEUE` messages behind on it.
        """
        if not self.open:
            return False
        lane = self._lanes[len(frame) >= self.BULK_BYTES]
        with self._cond:
            if topic is None:
                lane.append((None, frame))
            else:
                waiting = self._topics.setdefault(topic, collections.deque())
                if len(waiting) >= self.TOPIC_QUEUE:
                    waiting.popleft()
                    self.dropped += 1
                waiting.append(frame)
                lane.append((topic, None))
            self._cond.notify()
        return True

    def close(self) -> None:
        """Stop sending: whatever is still queued is dropped with the connection."""
        self.open = False
        with self._cond:
            for lane in self._lanes:
                lane.clear()
            self._topics.clear()
            self._cond.notify()

    def _drain(self) -> None:
        while True:
            with self._cond:
                while self.open and not (self._lanes[0] or self._lanes[1]):
                    self._cond.wait()
                if not self.open:
                    return
                topic, frame = (self._lanes[0] or self._lanes[1]).popleft()
                if topic is not None:
                    waiting = self._topics.get(topic)
                    if not waiting:
                        continue  # its message was dropped for a newer one, already sent
                    frame = waiting.popleft()
            try:
                self.ws.send(frame)
            except Exception:
                self.close()
                return


class ActionGoal:
    """One goal on one action, as the surface that executes it sees it.

    A surface's `execute(goal)` runs on a thread of its own, reads `goal.args`, may call
    `goal.publish_feedback(values)` as often as it likes, and returns the result's values.
    It should watch `goal.cancel_requested` (or block on `goal.wait_for_cancel(t)`): once
    a client's `cancel_action_goal` is accepted, returning ends the goal CANCELED rather
    than SUCCEEDED. It may also end the goal itself with `succeed`, `abort` or `canceled`,
    from any thread, in which case its return value is ignored.

    A goal aborted by `/reset` (`RosBridgeServer.abort_goals`) is finished at once: its
    result goes out ABORTED immediately, `cancel_requested` turns true so the executor can
    stop, and whatever the executor later returns is dropped.
    """

    def __init__(self, server: "RosBridgeServer", action: str, action_type: str,
                 args: dict, client: _Client | None, client_id, feedback: bool,
                 member: str | None) -> None:
        self._server = server
        self.uuid = uuid.uuid4().bytes
        self.id = self.uuid.hex()
        self.action = action
        self.action_type = action_type
        self.args = args
        self.member = member
        self._client = client
        self._client_id = client_id
        self._feedback = feedback
        self._status = GoalStatus.ACCEPTED
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self.stamp_s = server.now()
        # For a goal relayed to a client-provided action server: the id it was sent under.
        self.relay_id: str | None = None

    # -- what a surface reads ---------------------------------------------------------

    @property
    def status(self) -> int:
        return self._status

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait_for_cancel(self, timeout: float | None = None) -> bool:
        return self._cancel.wait(timeout)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the goal has a result. For tests and for callers that must know."""
        return self._done.wait(timeout)

    # -- what a surface does ----------------------------------------------------------

    def publish_feedback(self, values: dict) -> None:
        if self.done:
            return
        if self._feedback and self._client is not None:
            frame = {"op": "action_feedback", "action": self.action, "values": values}
            if self._client_id is not None:
                frame["id"] = self._client_id
            self._client.send(json.dumps(frame))

    def succeed(self, values: dict | None = None) -> bool:
        return self._finish(GoalStatus.SUCCEEDED, values or {})

    def abort(self, values: dict | None = None) -> bool:
        return self._finish(GoalStatus.ABORTED, values or {})

    def canceled(self, values: dict | None = None) -> bool:
        return self._finish(GoalStatus.CANCELED, values or {})

    # -- internals ----------------------------------------------------------------------

    def _set_status(self, status: int) -> None:
        with self._lock:
            if self.done or self._status == status:
                return
            self._status = status
        self._server._action_status_changed(self)

    def _finish(self, status: int, values, result: bool = True) -> bool:
        with self._lock:
            if self._done.is_set():
                return False
            self._status = status
            self._done.set()
        # rosbridge's success path: `result: true` with the terminal status, whatever it
        # is -- a CANCELED or ABORTED goal still has a (possibly empty) result message.
        frame = {"op": "action_result", "action": self.action, "values": values,
                 "status": int(status), "result": bool(result)}
        if self._client_id is not None:
            frame["id"] = self._client_id
        if self._client is not None:
            self._client.send(json.dumps(frame))
        self._server._goal_finished(self)
        return True


class _ActionServer:
    """A registered action: a surface's executor, or a client's advertisement."""

    def __init__(self, name: str, action_type: str, *, execute=None, cancel=None,
                 accept=None, node: str | None = None, member: str | None = None,
                 provider: _Client | None = None) -> None:
        self.name = name
        self.type = action_type
        self.execute = execute
        self.cancel = cancel
        self.accept = accept
        self.node = node
        self.member = member
        self.provider = provider


class RosBridgeServer:
    """Serves the rosbridge protocol on a websocket, for one or more clients.

    The surface-facing API -- what a robot's surface calls, usually through a
    `NamespacedBus`:

    * `publish(topic, msg, type, latched=False, publisher=node)` -- send to every
      subscribed client. `latched=True` keeps that publisher's last message and delivers
      it once to each new subscriber, as a latched / transient-local publisher does; it is
      never republished.
    * `advertise(topic, type, node=...)` -- declare a publication before its first message.
    * `on(topic, callback, type, node=...)` -- consume what clients publish.
    * `service(name, callback, type, node=...)` -- answer `call_service`.
    * `action(name, type, execute, cancel=None, node=..., member=...)` -- a ROS 2 action
      server; see `ActionGoal` for the executor's contract.
    * `abort_goals(member=None)` -- what `/reset` calls: every outstanding goal (of one
      member, or all) ends ABORTED now.
    * `set_param(name, value)`, `set_time(sim_seconds)`.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._server: ws_server.Server | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        # Guards every table below. Never held while sending to a socket.
        self._lock = threading.RLock()
        self._clients: dict[Any, _Client] = {}
        # topic -> callback invoked when a client publishes to it
        self._handlers: dict[str, Callable[[dict], None]] = {}
        # service name -> callback returning the response's `values`
        self._services: dict[str, Callable[[dict], dict]] = {}
        # topic -> type, for what the surfaces publish (declared by `advertise`, or learned
        # from the first `publish` that names a type).
        self._published_types: dict[str, str] = {}
        # topic -> type for what the surfaces consume, declared by `on`.
        self._subscribed_types: dict[str, str] = {}
        # service name -> type, declared by `service`.
        self._service_types: dict[str, str] = {}
        # Who provides what, as the node names a real graph would report.
        self._pub_nodes: dict[str, set[str]] = {}
        self._sub_nodes: dict[str, set[str]] = {}
        self._service_nodes: dict[str, str] = {}
        # name -> node, from the legacy `claim`; used only where no role recorded a node.
        self._fallback_node: dict[str, str] = {}
        # parameter name -> value
        self._params: dict[str, Any] = {}
        # topic -> {publisher key -> the frame it last published latched}
        self._latched: dict[str, dict[Any, str]] = {}
        # action name -> server (a surface's or a client's)
        self._actions: dict[str, _ActionServer] = {}
        # goal id -> every goal that has not finished
        self._goals: dict[str, ActionGoal] = {}
        # relay ids for calls and goals forwarded to a client-provided server
        self._relay_calls: dict[str, tuple[_Client, Any, str, _Client]] = {}
        self._relay_goals: dict[str, ActionGoal] = {}
        self._relay_counter = 0
        # client-provided services: name -> (client, type)
        self._client_services: dict[str, tuple[_Client, str]] = {}
        self._sim_time: float | None = None
        self._rosapi = False
        self._seq = 0

    # -- registration --------------------------------------------------------------

    def on(
        self, topic: str, callback: Callable[[dict], None], message_type: str | None = None,
        *, node: str | None = None,
    ) -> None:
        """Register a handler for messages clients publish to `topic`.

        `message_type` is optional only because it is not needed to *route* a message --
        but pass it, because it is what makes the topic discoverable through rosapi.
        Refuses a second handler rather than overwrite: two robots on one topic means the
        namespaces were not applied, and the first would stop responding in silence.
        """
        topic = normalise(topic)
        with self._lock:
            if topic in self._handlers:
                raise ValueError(
                    f"{topic} already has a handler on this server. Two robots sharing one "
                    "bridge must each be namespaced (see ns_topic); without that they "
                    "silently overwrite each other's command topics."
                )
            self._handlers[topic] = callback
            if message_type is not None:
                self._subscribed_types[topic] = message_type
            if node:
                self._sub_nodes.setdefault(topic, set()).add(node)

    def advertise(self, topic: str, message_type: str, *, node: str | None = None) -> None:
        """Declare a publication before its first message, so discovery lists it."""
        topic = normalise(topic)
        with self._lock:
            self._published_types.setdefault(topic, message_type)
            if node:
                self._pub_nodes.setdefault(topic, set()).add(node)

    def service(
        self, name: str, callback: Callable[[dict], dict], service_type: str | None = None,
        *, node: str | None = None,
    ) -> None:
        """Register a handler for `call_service` on `name`.

        The handler receives the request's `args` and returns the response's `values`;
        raising is reported as `result: false`. It runs on the calling client's reader
        thread, so it may only touch small shared state -- never MjData. One provider per
        name: a second registration raises (spec §3).
        """
        name = normalise(name)
        with self._lock:
            if name in self._services or name in self._actions:
                raise ValueError(f"service {name} is already registered on this server")
            self._services[name] = callback
            if service_type is not None:
                self._service_types[name] = service_type
            if node:
                self._service_nodes[name] = node

    def action(
        self, name: str, action_type: str, execute: Callable[[ActionGoal], dict | None],
        *, cancel: Callable[[ActionGoal], bool] | None = None, node: str | None = None,
        member: str | None = None, accept: Callable[[dict], bool] | None = None,
    ) -> None:
        """Register a ROS 2 action server on `name`.

        `execute(goal)` runs on a thread per goal and returns the result's values (see
        `ActionGoal`). `cancel(goal)` decides whether a client's cancel is accepted; by
        default every cancel is. `accept(args)` is the server's goal callback: False
        rejects the goal before it exists, which rosbridge reports as a failed
        `action_result` ("Action goal was rejected"). `member` is the namespace
        `abort_goals` selects by.
        """
        name = normalise(name)
        with self._lock:
            if name in self._actions or name in self._services:
                raise ValueError(f"action {name} is already registered on this server")
            self._actions[name] = _ActionServer(
                name, action_type, execute=execute, cancel=cancel, accept=accept,
                node=node, member=member,
            )

    def claim(self, namespace: str, name: str) -> None:
        """Record that `namespace`'s surface declared `name`.

        Kept for callers that predate per-role node names: the namespace, as a node,
        answers for `name` wherever no role recorded a node of its own.
        """
        if namespace:
            self._fallback_node[normalise(name)] = f"/{namespace.strip('/')}"

    def set_param(self, name: str, value: Any) -> None:
        """Set a ROS parameter, as a bringup's launch file would. Refuses to overwrite."""
        name = normalise(name)
        with self._lock:
            if name in self._params:
                raise ValueError(
                    f"parameter {name} is already set on this server. Two robots sharing one "
                    "bridge must each be namespaced (see ns_topic); without that they "
                    "silently overwrite each other's description."
                )
            self._params[name] = value

    # -- time ---------------------------------------------------------------------

    def set_time(self, sim_seconds: float) -> None:
        """The simulated clock `/rosapi/get_time` answers from. The fleet feeds it."""
        self._sim_time = float(sim_seconds)

    def now(self) -> float:
        """Simulated time once the loop has reported any; the wall clock before that."""
        return self._sim_time if self._sim_time is not None else time.time()

    # -- lifecycle ---------------------------------------------------------------

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
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()
            try:
                client.ws.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def has_subscribers(self, topic: str) -> bool:
        """Whether any client subscribes to `topic` -- so a surface can skip rendering a
        camera nobody is watching, which no client can observe."""
        topic = normalise(topic)
        with self._lock:
            return any(topic in c.subs for c in self._clients.values())

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- publishing --------------------------------------------------------------

    def publish(self, topic: str, msg: dict, message_type: str | None = None, *,
                latched: bool = False, publisher: str | None = None) -> None:
        """Send a message to every client subscribed to `topic`.

        `latched=True` makes this publisher latched: its last message (one per
        `publisher`) is delivered once to each client that subscribes later.
        """
        topic = normalise(topic)
        if message_type is not None and topic not in self._published_types:
            with self._lock:
                self._published_types.setdefault(topic, message_type)
        if publisher and publisher not in self._pub_nodes.get(topic, ()):
            with self._lock:
                self._pub_nodes.setdefault(topic, set()).add(publisher)
        frame = json.dumps({"op": "publish", "topic": topic, "msg": msg})
        self._deliver(topic, frame, (publisher or "") if latched else None)

    def _deliver(self, topic: str, frame: str, latch_key=None) -> None:
        with self._lock:
            if latch_key is not None:
                self._latched.setdefault(topic, {})[latch_key] = frame
            targets = [c for c in self._clients.values() if topic in c.subs]
        for client in targets:
            # Queued, never sent here: a slow or vanished client must not hold up the
            # thread publishing, which is usually the simulation loop.
            client.send(frame, topic)

    # -- actions, server side ------------------------------------------------------

    def abort_goals(self, member: str | None = None) -> int:
        """End every outstanding goal ABORTED, now. What `/reset` calls.

        `member` selects one member's actions by the namespace they were registered
        under; `None` aborts every goal on the server, including goals relayed to a
        client-provided action server (which is also told to cancel them). Returns how
        many goals were aborted.
        """
        with self._lock:
            goals = [g for g in self._goals.values()
                     if member is None or g.member == member]
        count = 0
        for goal in goals:
            if goal._finish(GoalStatus.ABORTED, {}):
                count += 1
            goal._cancel.set()
            if goal.relay_id is not None:
                spec = self._actions.get(goal.action)
                if spec is not None and spec.provider is not None:
                    spec.provider.send(json.dumps({"op": "cancel_action_goal",
                                                   "id": goal.relay_id,
                                                   "action": goal.action}))
        return count

    def _goal_finished(self, goal: ActionGoal) -> None:
        with self._lock:
            self._goals.pop(goal.id, None)
            if goal.relay_id is not None:
                self._relay_goals.pop(goal.relay_id, None)
        self._action_status_changed(goal)

    def _action_status_changed(self, goal: ActionGoal) -> None:
        """Publish the action's `GoalStatusArray` on its hidden `_action/status` topic.

        That is where a ROS 2 action server reports status, so a client that subscribes
        to it sees each transition. It is delivered to subscribers only: rosapi does not
        list an action's hidden topics.
        """
        with self._lock:
            goals = [g for g in self._goals.values() if g.action == goal.action]
        if goal not in goals:
            goals.append(goal)
        status_list = []
        for g in goals:
            sec = int(g.stamp_s)
            status_list.append({
                "goal_info": {
                    "goal_id": {"uuid": base64.b64encode(g.uuid).decode("ascii")},
                    "stamp": {"sec": sec, "nanosec": int(round((g.stamp_s - sec) * 1e9))},
                },
                "status": int(g.status),
            })
        topic = f"{goal.action}/_action/status"
        self._deliver(topic, json.dumps({"op": "publish", "topic": topic,
                                         "msg": {"status_list": status_list}}))

    def _run_goal(self, spec: _ActionServer, goal: ActionGoal) -> None:
        goal._set_status(GoalStatus.EXECUTING)
        try:
            values = spec.execute(goal)
        except Exception:
            log.exception("action %s failed", spec.name)
            goal.abort({})
            return
        if goal.done:
            return
        if goal.cancel_requested:
            goal.canceled(values or {})
        else:
            goal.succeed(values or {})

    # -- rosapi --------------------------------------------------------------------

    def _topic_type(self, topic: str) -> str | None:
        with self._lock:
            if topic in self._published_types:
                return self._published_types[topic]
            if topic in self._subscribed_types:
                return self._subscribed_types[topic]
            for client in self._clients.values():
                advert = client.adverts.get(topic)
                if advert is not None:
                    return advert["type"]
        return None

    def _known_topics(self) -> dict[str, str]:
        with self._lock:
            known: dict[str, str] = {}
            for client in self._clients.values():
                for topic, advert in client.adverts.items():
                    known.setdefault(topic, advert["type"])
            known.update(self._subscribed_types)
            known.update(self._published_types)
            return known

    def _publishers_of(self, topic: str) -> list[str]:
        with self._lock:
            nodes = set(self._pub_nodes.get(topic, ()))
            if not nodes and topic in self._published_types and topic in self._fallback_node:
                nodes.add(self._fallback_node[topic])
            if any(topic in c.adverts for c in self._clients.values()):
                nodes.add(BRIDGE_NODE)
        return sorted(nodes)

    def _subscribers_of(self, topic: str) -> list[str]:
        with self._lock:
            nodes = set(self._sub_nodes.get(topic, ()))
            if not nodes and topic in self._subscribed_types and topic in self._fallback_node:
                nodes.add(self._fallback_node[topic])
            if any(topic in c.subs for c in self._clients.values()) and \
                    topic in self._known_topics():
                nodes.add(BRIDGE_NODE)
        return sorted(nodes)

    def _service_table(self) -> dict[str, str]:
        """service name -> type, for every service on the graph (the surfaces' and clients')."""
        with self._lock:
            table = {name: self._service_types.get(name, "") for name in self._services}
            for name, (_client, stype) in self._client_services.items():
                table.setdefault(name, stype)
            return table

    def _service_node(self, name: str) -> str:
        with self._lock:
            if name in self._service_nodes:
                return self._service_nodes[name]
            if name in self._client_services:
                return BRIDGE_NODE
            if name in self._services:
                return self._fallback_node.get(name, "")
        return ""

    def _action_node(self, name: str) -> str:
        spec = self._actions.get(name)
        if spec is None:
            return ""
        if spec.provider is not None:
            return BRIDGE_NODE
        return spec.node or self._fallback_node.get(name, "")

    def serve_rosapi(self) -> None:
        """Answer the 31 `rosapi` services the spec lists (§3, "Discovery").

        Topic, service and action answers come from this server's own tables; message,
        service and action details from `message_schemas`, which records their provenance.
        Node answers name the node each surface declared for a name (composed with its
        namespace by `NamespacedBus`), `/rosapi` for these services and
        `/rosbridge_websocket` for everything a client advertised or subscribed to.
        Parameter values travel JSON-encoded, both ways.

        **The dialect of this rosapi is ROS 2 Jazzy**, although the graph carries ROS 1
        members beside ROS 2 ones. The transport is rosbridge 2.x (it speaks the ROS 2
        action ops) and six of the 31 services -- `get_ros_version` among them -- exist
        only in the ROS 2 `rosapi_node`, so the node answering them is that one:
        `get_ros_version` answers `{version: 2, distro: "jazzy"}` exactly as Jazzy's
        rosapi does from `ROS_VERSION`/`ROS_DISTRO`, and the services both nodes share
        answer in their ROS 2 shape (`get_time` as `builtin_interfaces/Time`,
        `get_param` with `successful`/`reason`, `default_value` accepted beside ROS 1's
        `default`). The two the ROS 2 node lacks, `service_host` and `search_param`,
        answer in the ROS 1 node's shape. A client must still read each topic's dialect
        from its type, never from the version.
        """
        from contracts import message_schemas as schemas

        with self._lock:
            if self._rosapi:
                return
            self._rosapi = True

        def topics(_args: dict) -> dict:
            known = self._known_topics()
            names = sorted(known)
            return {"topics": names, "types": [known[n] for n in names]}

        def topics_for_type(args: dict) -> dict:
            wanted = args.get("type")
            return {"topics": sorted(n for n, t in self._known_topics().items() if t == wanted)}

        def topics_and_raw_types(_args: dict) -> dict:
            known = self._known_topics()
            names = sorted(known)
            return {"topics": names, "types": [known[n] for n in names],
                    "typedefs_full_text": [schemas.definition_text(known[n]) for n in names]}

        def topic_type(args: dict) -> dict:
            return {"type": self._topic_type(normalise(str(args.get("topic", "")))) or ""}

        def services(_args: dict) -> dict:
            return {"services": sorted(self._service_table())}

        def services_for_type(args: dict) -> dict:
            wanted = args.get("type")
            return {"services": sorted(n for n, t in self._service_table().items()
                                       if t == wanted)}

        def service_type(args: dict) -> dict:
            return {"type": self._service_table().get(normalise(str(args.get("service", ""))), "")}

        def service_providers(args: dict) -> dict:
            node = self._service_node(normalise(str(args.get("service", ""))))
            return {"providers": [node] if node else []}

        def service_node(args: dict) -> dict:
            return {"node": self._service_node(normalise(str(args.get("service", ""))))}

        def service_host(args: dict) -> dict:
            name = normalise(str(args.get("service", "")))
            return {"host": socket.gethostname() if name in self._service_table() else ""}

        def publishers(args: dict) -> dict:
            return {"publishers": self._publishers_of(normalise(str(args.get("topic", ""))))}

        def subscribers(args: dict) -> dict:
            return {"subscribers": self._subscribers_of(normalise(str(args.get("topic", ""))))}

        def _all_nodes() -> set[str]:
            known = self._known_topics()
            nodes = {ROSAPI_NODE, BRIDGE_NODE}
            for topic in known:
                nodes.update(self._publishers_of(topic))
                nodes.update(self._subscribers_of(topic))
            for name in self._service_table():
                nodes.add(self._service_node(name))
            for name in list(self._actions):
                nodes.add(self._action_node(name))
            nodes.discard("")
            return nodes

        def nodes(_args: dict) -> dict:
            return {"nodes": sorted(_all_nodes())}

        def node_details(args: dict) -> dict:
            node = normalise(str(args.get("node", "")))
            known = self._known_topics()
            return {
                "subscribing": sorted(t for t in known if node in self._subscribers_of(t)),
                "publishing": sorted(t for t in known if node in self._publishers_of(t)),
                "services": sorted(s for s in self._service_table()
                                   if self._service_node(s) == node),
            }

        def action_servers(_args: dict) -> dict:
            with self._lock:
                return {"action_servers": sorted(self._actions)}

        def interfaces(_args: dict) -> dict:
            found = set(schemas.interfaces())
            found.update(schemas.ros2_name(t, "msg") for t in self._known_topics().values())
            found.update(schemas.ros2_name(t, "srv") for t in self._service_table().values() if t)
            with self._lock:
                found.update(schemas.ros2_name(a.type, "action") for a in self._actions.values())
            return {"interfaces": sorted(found)}

        def action_type(args: dict) -> dict:
            spec = self._actions.get(normalise(str(args.get("action", ""))))
            return {"type": spec.type if spec is not None else ""}

        def message_details(args: dict) -> dict:
            return {"typedefs": schemas.typedefs(str(args.get("type", "")))}

        def service_request_details(args: dict) -> dict:
            return {"typedefs": schemas.service_typedefs(str(args.get("type", "")), "request")}

        def service_response_details(args: dict) -> dict:
            return {"typedefs": schemas.service_typedefs(str(args.get("type", "")), "response")}

        def action_goal_details(args: dict) -> dict:
            return {"typedefs": schemas.action_typedefs(str(args.get("type", "")), "goal")}

        def action_result_details(args: dict) -> dict:
            return {"typedefs": schemas.action_typedefs(str(args.get("type", "")), "result")}

        def action_feedback_details(args: dict) -> dict:
            return {"typedefs": schemas.action_typedefs(str(args.get("type", "")), "feedback")}

        def _param_lookup(name: str) -> tuple[bool, Any]:
            """A name is a parameter, or a namespace of them (ROS's dict answer)."""
            with self._lock:
                if name in self._params:
                    return True, self._params[name]
                prefix = name.rstrip("/") + "/"
                children = {k: v for k, v in self._params.items() if k.startswith(prefix)}
            if not children:
                return False, None
            tree: dict = {}
            for key, value in children.items():
                node = tree
                parts = key[len(prefix):].split("/")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
            return True, tree

        def get_param_names(_args: dict) -> dict:
            with self._lock:
                return {"names": sorted(self._params)}

        def get_param(args: dict) -> dict:
            """The value JSON-encoded, as rosapi sends it; an unset one is the default, verbatim."""
            name = normalise(str(args.get("name", "")))
            found, value = _param_lookup(name)
            if found:
                return {"value": json.dumps(value), "successful": True, "reason": ""}
            default = args.get("default_value", args.get("default", ""))
            return {"value": default, "successful": False,
                    "reason": f"parameter {name} is not set"}

        def set_param(args: dict) -> dict:
            name = normalise(str(args.get("name", "")))
            try:
                value = json.loads(args.get("value", ""))
            except (TypeError, ValueError):
                return {"successful": False, "reason": "value is not JSON-encoded"}
            with self._lock:
                self._params[name] = value
            return {"successful": True, "reason": ""}

        def has_param(args: dict) -> dict:
            return {"exists": _param_lookup(normalise(str(args.get("name", ""))))[0]}

        def search_param(args: dict) -> dict:
            # rosapi searches up from its own namespace, which is the root: the answer is
            # the global name if the key's first segment is set there, else empty.
            key = str(args.get("name", "")).strip("/")
            first = key.split("/")[0] if key else ""
            if first and _param_lookup(f"/{first}")[0]:
                return {"global_name": f"/{key}"}
            return {"global_name": ""}

        def delete_param(args: dict) -> dict:
            name = normalise(str(args.get("name", "")))
            if name == "/":
                return {"successful": False, "reason": "the root namespace cannot be deleted"}
            with self._lock:
                prefix = name.rstrip("/") + "/"
                doomed = [k for k in self._params if k == name or k.startswith(prefix)]
                for key in doomed:
                    del self._params[key]
            if not doomed:
                return {"successful": False, "reason": f"parameter {name} is not set"}
            return {"successful": True, "reason": ""}

        def get_time(_args: dict) -> dict:
            now = self.now()
            sec = int(now)
            return {"time": {"sec": sec, "nanosec": int(round((now - sec) * 1e9))}}

        def get_ros_version(_args: dict) -> dict:
            return {"version": ROS_VERSION, "distro": ROS_DISTRO}

        for name, handler, stype in (
            # the ROS 1 rosapi_node's 25
            ("topics", topics, "Topics"),
            ("topics_for_type", topics_for_type, "TopicsForType"),
            ("topics_and_raw_types", topics_and_raw_types, "TopicsAndRawTypes"),
            ("topic_type", topic_type, "TopicType"),
            ("services", services, "Services"),
            ("services_for_type", services_for_type, "ServicesForType"),
            ("service_type", service_type, "ServiceType"),
            ("service_providers", service_providers, "ServiceProviders"),
            ("service_node", service_node, "ServiceNode"),
            ("service_host", service_host, "ServiceHost"),
            ("nodes", nodes, "Nodes"),
            ("node_details", node_details, "NodeDetails"),
            ("publishers", publishers, "Publishers"),
            ("subscribers", subscribers, "Subscribers"),
            ("action_servers", action_servers, "GetActionServers"),
            ("message_details", message_details, "MessageDetails"),
            ("service_request_details", service_request_details, "ServiceRequestDetails"),
            ("service_response_details", service_response_details, "ServiceResponseDetails"),
            ("get_param_names", get_param_names, "GetParamNames"),
            ("get_param", get_param, "GetParam"),
            ("set_param", set_param, "SetParam"),
            ("has_param", has_param, "HasParam"),
            ("search_param", search_param, "SearchParam"),
            ("delete_param", delete_param, "DeleteParam"),
            ("get_time", get_time, "GetTime"),
            # the ROS 2 rosapi_node's 6 more
            ("interfaces", interfaces, "Interfaces"),
            ("action_type", action_type, "ActionType"),
            ("action_goal_details", action_goal_details, "ActionGoalDetails"),
            ("action_result_details", action_result_details, "ActionResultDetails"),
            ("action_feedback_details", action_feedback_details, "ActionFeedbackDetails"),
            ("get_ros_version", get_ros_version, "GetROSVersion"),
        ):
            self.service(f"/rosapi/{name}", handler, f"rosapi_msgs/srv/{stype}",
                         node=ROSAPI_NODE)

    # -- protocol ----------------------------------------------------------------

    def _handler(self, websocket) -> None:
        log.info("client connected from %s", websocket.remote_address)
        client = _Client(websocket)
        with self._lock:
            self._clients[websocket] = client
        try:
            for raw in websocket:
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    self._status(client, "error", "message was not valid JSON")
                    continue
                try:
                    self._dispatch(client, message)
                except Exception as exc:  # a bug here must not end the connection
                    log.exception("dispatch failed")
                    self._status(client, "error", f"internal error: {exc}",
                                 message.get("id") if isinstance(message, dict) else None)
        except Exception as exc:
            log.debug("client loop ended: %s", exc)
        finally:
            self._disconnect(client)
            log.info("client disconnected")

    def _disconnect(self, client: _Client) -> None:
        """Drop everything the client advertised. Goals it *sent* keep running."""
        client.close()
        with self._lock:
            self._clients.pop(client.ws, None)
            for topic in client.adverts:
                self._latched.get(topic, {}).pop(client.latch_key, None)
            for name in client.services:
                if self._client_services.get(name, (None,))[0] is client:
                    del self._client_services[name]
            pending_calls = [(rid, call) for rid, call in self._relay_calls.items()
                             if call[3] is client]
            for rid, _ in pending_calls:
                del self._relay_calls[rid]
            relayed = []
            for name in client.actions:
                spec = self._actions.get(name)
                if spec is not None and spec.provider is client:
                    del self._actions[name]
                    relayed += [g for g in self._goals.values() if g.action == name]
        for _rid, (caller, call_id, service, _provider) in pending_calls:
            self._service_response(caller, service, call_id, False,
                                   f"service {service} went away")
        for goal in relayed:
            goal._finish(GoalStatus.ABORTED, {})

    def _dispatch(self, client: _Client, message) -> None:
        if not isinstance(message, dict):
            self._status(client, "error", "a rosbridge message is a JSON object")
            return
        op = message.get("op")
        mid = message.get("id")
        if not isinstance(op, str):
            self._status(client, "error", "message has no 'op'", mid)
            return
        if op not in CLIENT_OPS and op not in PROVIDER_REPLY_OPS:
            self._status(client, "error", f"unsupported op {op!r}", mid)
            return
        getattr(self, f"_op_{op}")(client, message, mid)

    def _fields(self, client: _Client, message: dict, mid, *names: str) -> bool:
        """Every named field is present and a non-empty string; otherwise a status error."""
        for name in names:
            value = message.get(name)
            if not isinstance(value, str) or not value:
                self._status(client, "error",
                             f"{message.get('op')}: field '{name}' must be a non-empty string",
                             mid)
                return False
        return True

    # topics

    def _op_subscribe(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "topic"):
            return
        topic = normalise(message["topic"])
        wanted = message.get("type")
        known = self._topic_type(topic)
        if wanted and known and schemas_canonical(wanted) != schemas_canonical(known):
            self._status(client, "error",
                         f"subscribe: {topic} is {known}, not {wanted}", mid)
            return
        with self._lock:
            new = topic not in client.subs
            client.subs.setdefault(topic, set()).add(mid)
            latched = list(self._latched.get(topic, {}).values()) if new else []
        for frame in latched:
            client.send(frame)
        log.info("client subscribed to %s", topic)

    def _op_unsubscribe(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "topic"):
            return
        topic = normalise(message["topic"])
        with self._lock:
            ids = client.subs.get(topic)
            if ids is None or (mid is not None and mid not in ids):
                missing = True
            else:
                missing = False
                if mid is None:
                    ids.clear()
                else:
                    ids.discard(mid)
                if not ids:
                    del client.subs[topic]
        if missing:
            self._status(client, "error", f"unsubscribe: not subscribed to {topic}", mid)

    def _op_advertise(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "topic", "type"):
            return
        self._client_advertise(client, normalise(message["topic"]), message, mid)

    def _client_advertise(self, client: _Client, topic: str, message: dict, mid) -> bool:
        mtype = message["type"]
        known = self._topic_type(topic)
        if known and schemas_canonical(known) != schemas_canonical(mtype):
            self._status(client, "error", f"advertise: {topic} is {known}, not {mtype}", mid)
            return False
        qos = message.get("qos")
        latched = bool(message.get("latch")) or (
            isinstance(qos, dict) and qos.get("durability") == "transient_local")
        with self._lock:
            advert = client.adverts.setdefault(
                topic, {"type": known or mtype, "ids": set(), "latched": latched})
            advert["ids"].add(mid)
        log.info("client advertised %s (%s)", topic, mtype)
        return True

    def _op_unadvertise(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "topic"):
            return
        topic = normalise(message["topic"])
        with self._lock:
            advert = client.adverts.get(topic)
            if advert is None or (mid is not None and mid not in advert["ids"]):
                missing = True
            else:
                missing = False
                if mid is None:
                    advert["ids"].clear()
                else:
                    advert["ids"].discard(mid)
                if not advert["ids"]:
                    del client.adverts[topic]
                    self._latched.get(topic, {}).pop(client.latch_key, None)
        if missing:
            self._status(client, "error", f"unadvertise: {topic} is not advertised", mid)

    def _op_publish(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "topic"):
            return
        msg = message.get("msg", {})
        if not isinstance(msg, dict):
            self._status(client, "error", "publish: 'msg' must be an object", mid)
            return
        topic = normalise(message["topic"])
        if topic not in client.adverts:
            # rosbridge advertises on first publish, typed from the message or the graph.
            mtype = message.get("type") or self._topic_type(topic)
            if not isinstance(mtype, str) or not mtype:
                self._status(client, "error",
                             f"publish: {topic} has no known type; advertise it first", mid)
                return
            if not self._client_advertise(client, topic, {**message, "type": mtype}, mid):
                return
        advert = client.adverts[topic]
        handler = self._handlers.get(topic)
        if handler is not None:
            try:
                handler(msg)
            except Exception as exc:
                log.exception("handler for %s failed", topic)
                self._status(client, "error", f"handler for {topic} failed: {exc}", mid)
        frame = json.dumps({"op": "publish", "topic": topic, "msg": msg})
        self._deliver(topic, frame, client.latch_key if advert["latched"] else None)

    # services

    def _op_call_service(self, client: _Client, message: dict, mid) -> None:
        # The reply MUST echo `id`: the caller is blocked on it, and dropping it hangs the
        # client rather than failing it.
        if not self._fields(client, message, mid, "service"):
            return
        name = normalise(message["service"])
        args = message.get("args") or {}
        handler = self._services.get(name)
        if handler is not None:
            if isinstance(args, list):
                args = self._args_from_list(name, args)
            if not isinstance(args, dict):
                self._service_response(client, name, mid, False,
                                       {"message": "args must be an object or a list"})
                return
            try:
                values = handler(args) or {}
            except Exception as exc:
                log.exception("service %s failed", name)
                self._service_response(client, name, mid, False, {"message": str(exc)})
                return
            self._service_response(client, name, mid, True, values)
            return
        with self._lock:
            provided = self._client_services.get(name)
            if provided is not None:
                self._relay_counter += 1
                relay_id = f"service_request:{name}:{self._relay_counter}"
                self._relay_calls[relay_id] = (client, mid, name, provided[0])
        if provided is None:
            self._service_response(client, name, mid, False, {"message": f"no service {name}"})
            return
        provided[0].send(json.dumps({"op": "call_service", "id": relay_id,
                                     "service": name, "args": args}))

    def _args_from_list(self, service: str, args: list):
        from contracts import message_schemas as schemas

        stype = self._service_types.get(service, "")
        pair = schemas.SERVICES.get(schemas.canonical(stype))
        if pair is None:
            return args
        return {field[0]: value for field, value in zip(pair[0], args)}

    def _op_advertise_service(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "service", "type"):
            return
        name = normalise(message["service"])
        with self._lock:
            owner = self._client_services.get(name)
            refused = name in self._services or (owner is not None and owner[0] is not client)
            if not refused:
                self._client_services[name] = (client, message["type"])
                client.services[name] = message["type"]
        if refused:
            self._status(client, "error",
                         f"advertise_service: {name} already has a provider", mid)

    def _op_unadvertise_service(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "service"):
            return
        name = normalise(message["service"])
        with self._lock:
            ok = client.services.pop(name, None) is not None
            if ok and self._client_services.get(name, (None,))[0] is client:
                del self._client_services[name]
        if not ok:
            self._status(client, "error", f"unadvertise_service: {name} is not advertised", mid)

    def _op_service_response(self, client: _Client, message: dict, mid) -> None:
        with self._lock:
            call = self._relay_calls.get(mid) if isinstance(mid, str) else None
            if call is not None and call[3] is client:
                del self._relay_calls[mid]
            else:
                call = None
        if call is None:
            self._status(client, "error",
                         f"service_response: no call of yours is pending as {mid!r}", mid)
            return
        caller, call_id, name, _provider = call
        values = message.get("values")
        self._service_response(caller, name, call_id, bool(message.get("result", True)),
                               values if values is not None else {})

    # actions

    def _op_advertise_action(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "action", "type"):
            return
        name = normalise(message["action"])
        with self._lock:
            spec = self._actions.get(name)
            refused = name in self._services or (
                spec is not None and spec.provider is not client)
            if not refused:
                self._actions[name] = _ActionServer(name, message["type"], provider=client,
                                                    node=BRIDGE_NODE)
                client.actions[name] = message["type"]
        if refused:
            self._status(client, "error", f"advertise_action: {name} already has a server", mid)

    def _op_unadvertise_action(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "action"):
            return
        name = normalise(message["action"])
        with self._lock:
            ok = client.actions.pop(name, None) is not None
            spec = self._actions.get(name)
            relayed = []
            if ok and spec is not None and spec.provider is client:
                del self._actions[name]
                relayed = [g for g in self._goals.values() if g.action == name]
        if not ok:
            self._status(client, "error", f"unadvertise_action: {name} is not advertised", mid)
        for goal in relayed:
            goal._finish(GoalStatus.ABORTED, {})

    def _op_send_action_goal(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "action", "action_type"):
            return
        name = normalise(message["action"])
        action_type = message["action_type"]
        args = message.get("args") or {}
        spec = self._actions.get(name)

        def fail(reason: str) -> None:
            # rosbridge's failure shape: result false, status UNKNOWN, the reason as values.
            frame = {"op": "action_result", "action": name, "values": reason,
                     "status": GoalStatus.UNKNOWN, "result": False}
            if mid is not None:
                frame["id"] = mid
            client.send(json.dumps(frame))

        if spec is None:
            fail(f"no action server for {name}")
            return
        if schemas_canonical(spec.type) != schemas_canonical(action_type):
            fail(f"{name} is {spec.type}, not {action_type}")
            return
        if isinstance(args, list):
            from contracts import message_schemas as schemas

            fields = schemas.action_fields(spec.type, "goal")
            args = {f[0]: v for f, v in zip(fields or [], args)} if fields else None
        if not isinstance(args, dict):
            fail("args must be an object, or a list for a known action type")
            return
        if spec.accept is not None:
            try:
                accepted = bool(spec.accept(args))
            except Exception:
                log.exception("goal callback for %s failed", name)
                accepted = False
            if not accepted:
                # rosbridge's own words for a goal the server's goal callback refused.
                fail("Action goal was rejected")
                return
        goal = ActionGoal(self, name, spec.type, args, client, mid,
                          bool(message.get("feedback", False)), spec.member)
        with self._lock:
            self._goals[goal.id] = goal
            client.goals[(name, mid)] = goal
            if spec.provider is not None:
                self._relay_counter += 1
                goal.relay_id = f"action_goal:{name}:{self._relay_counter}"
                self._relay_goals[goal.relay_id] = goal
        self._action_status_changed(goal)
        if spec.provider is not None:
            goal._set_status(GoalStatus.EXECUTING)
            spec.provider.send(json.dumps({
                "op": "send_action_goal", "id": goal.relay_id, "action": name,
                "action_type": spec.type, "args": args, "feedback": True,
            }))
            return
        threading.Thread(target=self._run_goal, args=(spec, goal), daemon=True,
                         name=f"action {name}").start()

    def _op_cancel_action_goal(self, client: _Client, message: dict, mid) -> None:
        if not self._fields(client, message, mid, "action"):
            return
        name = normalise(message["action"])
        with self._lock:
            goal = client.goals.get((name, mid))
        if goal is None or goal.done:
            self._status(client, "error",
                         f"cancel_action_goal: no goal {mid!r} of yours is running on {name}",
                         mid)
            return
        if goal.cancel_requested:
            self._status(client, "error", f"cancel_action_goal: {mid!r} is already canceling",
                         mid)
            return
        spec = self._actions.get(name)
        if spec is not None and spec.provider is not None:
            goal._cancel.set()
            goal._set_status(GoalStatus.CANCELING)
            spec.provider.send(json.dumps({"op": "cancel_action_goal", "id": goal.relay_id,
                                           "action": name}))
            return
        accept = True
        if spec is not None and spec.cancel is not None:
            try:
                accept = bool(spec.cancel(goal))
            except Exception:
                log.exception("cancel callback for %s failed", name)
                accept = False
        if not accept:
            self._status(client, "info", f"cancel of {mid!r} on {name} was rejected", mid)
            return
        goal._cancel.set()
        goal._set_status(GoalStatus.CANCELING)

    def _relayed_goal(self, client: _Client, message: dict, mid) -> ActionGoal | None:
        with self._lock:
            goal = self._relay_goals.get(mid) if isinstance(mid, str) else None
        spec = self._actions.get(goal.action) if goal is not None else None
        if goal is None or spec is None or spec.provider is not client:
            self._status(client, "error",
                         f"{message.get('op')}: no goal of yours is running as {mid!r}", mid)
            return None
        return goal

    def _op_action_feedback(self, client: _Client, message: dict, mid) -> None:
        goal = self._relayed_goal(client, message, mid)
        if goal is not None:
            goal.publish_feedback(message.get("values") or {})

    def _op_action_result(self, client: _Client, message: dict, mid) -> None:
        goal = self._relayed_goal(client, message, mid)
        if goal is None:
            return
        status = message.get("status")
        if status not in GoalStatus.TERMINAL:
            status = GoalStatus.SUCCEEDED if message.get("result", True) else GoalStatus.ABORTED
        values = message.get("values")
        goal._finish(int(status), values if values is not None else {},
                     bool(message.get("result", True)))

    # logging controls

    def _op_set_level(self, client: _Client, message: dict, mid) -> None:
        level = message.get("level")
        if level not in STATUS_LEVELS:
            self._status(client, "error",
                         f"set_level: level must be one of {sorted(STATUS_LEVELS)}", mid)
            return
        client.level = level

    def _op_status(self, client: _Client, message: dict, mid) -> None:
        # A client's own status report: rosbridge logs it and answers nothing.
        log.info("client status %s: %s", message.get("level"), message.get("msg"))

    # replies

    def _service_response(self, client: _Client, service: str, call_id, result: bool,
                          values) -> None:
        frame = {"op": "service_response", "service": service,
                 "values": values, "result": result}
        if call_id is not None:
            frame["id"] = call_id
        client.send(json.dumps(frame))

    def _status(self, client: _Client, level: str, msg: str, mid=None) -> None:
        if level != "info":
            log.warning("%s: %s", level, msg)
        if STATUS_LEVELS.get(client.level, 1) < STATUS_LEVELS.get(level, 1):
            return
        frame = {"op": "status", "level": level, "msg": msg}
        if mid is not None:
            frame["id"] = mid
        client.send(json.dumps(frame))


def schemas_canonical(type_name: str) -> str:
    """One spelling per type, so `sensor_msgs/Image` and `sensor_msgs/msg/Image` agree."""
    parts = str(type_name).split("/")
    if len(parts) == 3 and parts[1] in ("msg", "srv", "action"):
        parts = [parts[0], parts[2]]
    return "/".join(parts)


class NamespacedBus:
    """One robot's view of a shared `RosBridgeServer`.

    A surface takes a bus instead of a server and otherwise keeps its bare topic
    constants: `bus.on(TOPIC_CMD_VEL, ...)` registers `/myagv/cmd_vel`. That is the whole
    of what a surface has to know about sharing a graph -- the alternative was every
    surface composing prefixes at every call site, where one missed name is a topic
    silently landing in another robot's namespace.

    Every registering call takes an optional `node=`: the vendor node that provides the
    name in the member's source (`"robot_state_publisher"`), composed with the namespace
    like any other name, or a `GlobalName` (`SIMULATOR_NODE`) taken literally. It is what
    `/rosapi/publishers`, `subscribers`, `nodes`, `node_details` and `service_node`
    answer. Omitted, the namespace itself stands in as the node, as it always has.

    A bus deliberately does NOT expose `start`/`stop`. The server is owned by whatever
    built it; a surface that stopped it would take every other robot on the port down.
    """

    def __init__(self, server: "RosBridgeServer", namespace: str = "") -> None:
        self.server = server
        self.ns = RobotNamespace(namespace)
        self.published: list[str] = []
        self.subscribed: list[str] = []
        # Per-bus, not per-server: `header.seq` is a per-publisher counter on real ROS 1.
        self._seq = 0

    # -- naming ------------------------------------------------------------------

    def topic(self, topic: str) -> str:
        return self.ns.topic(topic)

    def frame(self, frame_id: str) -> str:
        return self.ns.frame(frame_id)

    def node(self, node: str | None = None) -> str | None:
        """The composed node name `node` stands for on the wire (see the class notes)."""
        if node is None:
            return f"/{self.ns.name.strip('/')}" if self.ns else None
        if isinstance(node, GlobalName):
            return normalise(str(node))
        return self.ns.topic(node)

    # -- the server's surface, namespaced ----------------------------------------

    def on(self, topic: str, callback: Callable[[dict], None],
           message_type: str | None = None, *, node: str | None = None) -> None:
        name = self.ns.topic(topic)
        self.server.on(name, callback, message_type, node=self.node(node))
        self.subscribed.append(name)

    def advertise(self, topic: str, message_type: str, *, node: str | None = None) -> None:
        name = self.ns.topic(topic)
        self.server.advertise(name, message_type, node=self.node(node))
        if name not in self.published:
            self.published.append(name)

    def service(self, name: str, callback: Callable[[dict], dict],
                service_type: str | None = None, *, node: str | None = None) -> None:
        self.server.service(self.ns.service(name), callback, service_type,
                            node=self.node(node))

    def action(self, name: str, action_type: str,
               execute: Callable[[ActionGoal], dict | None], *,
               cancel: Callable[[ActionGoal], bool] | None = None,
               node: str | None = None,
               accept: Callable[[dict], bool] | None = None) -> None:
        """A ROS 2 action server under this namespace; see `RosBridgeServer.action`."""
        self.server.action(self.ns.service(name), action_type, execute, cancel=cancel,
                           node=self.node(node), member=self.ns.name, accept=accept)

    def has_subscribers(self, topic: str) -> bool:
        return self.server.has_subscribers(self.ns.topic(topic))

    def abort_goals(self) -> int:
        """End this member's outstanding goals ABORTED -- what its `/reset` does."""
        return self.server.abort_goals(self.ns.name)

    def set_param(self, name: str, value: Any) -> None:
        """Set one of this robot's parameters, namespaced like everything else it has."""
        full = self.ns.topic(name)
        self.server.set_param(full, value)

    def publish(self, topic: str, msg: dict, message_type: str | None = None, *,
                latched: bool = False, node: str | None = None) -> None:
        name = self.ns.topic(topic)
        if message_type is not None and name not in self.published:
            self.published.append(name)
        self.server.publish(name, msg, message_type, latched=latched,
                            publisher=self.node(node))

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def client_count(self) -> int:
        """How many clients the whole server has -- not this robot's share."""
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
