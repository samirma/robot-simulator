"""Ask a rosbridge what is on it, check it against the console's contract, and time it.

A listening socket says nothing. Neither does a driving arm: an episode can run perfectly
while a second robot on the same port published nothing at all, because nothing the arm
does touches the base's topics. So the multi-robot claim needs its own check, and this is
it -- one `/rosapi/topics` call, compared against **the console's own contract
constants**, namespaced.

    python -m robot_console.fleet [--url ws://127.0.0.1:9090]   # validate what is there
    python -m robot_console.fleet --dump                        # print every topic, sorted
    python -m robot_console.fleet --rates [--gate]              # time every periodic topic

**Validation** discovers the members (`discovery.find_members`: typed signatures, so a
`/cmd_vel` that is not a `geometry_msgs/Twist` is reported rather than counted) and
checks each against its contract below: every topic its ROS file declares that the
console expects on the wire, with the declared type, and nothing under its namespace the
contract does not declare. The legacy `--arm/--base/--humanoid NS` flags still check
named expectations instead, which is what `run_task.sh` uses.

**Rates** is simulator spec §5's gate. It subscribes, unthrottled, to every periodic
topic of every member, discards a 5 s warm-up, then observes for the greater of 30 s and
five periods of the slowest declared topic. Each topic's expected Hz is the console's
`PERIODIC` table -- which `tests/test_fleet.py` holds equal to the `rate_hz` of every
periodic row in `robots_specs/<id>/ros*.yml` -- and it passes when the measured rate is
within +/-10% and no inter-message gap exceeds three expected periods. The real-time
factor is read off the messages' own header stamps against their arrival: its mean must
lie in [0.90, 1.10] and no rolling 10 s window may fall below 0.90. A member with no rate
declaration, a declared periodic topic missing from the wire, or a window shorter than
required fails rather than being skipped. The report always prints; only `--gate` turns
a failure into a non-zero exit.

Checking against our own constants rather than a list typed into a shell script is the
point: if the wire and the console disagree, a run was going to fail later and less
legibly. `--dump` and validation use roslibpy (a base dependency); `--rates` speaks the
websocket itself with the standard library, because it must read a few hundred MB/s of
camera frames while decoding only each message's first bytes -- a JSON parse of every
raw image would measure this process, not the wire.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import re
import socket
import statistics
import struct
import sys
import time
from typing import Iterable, Mapping, Optional
from urllib.parse import urlsplit

from robot_console.ainex_topics import CONTRACT_TOPICS as AINEX_CONTRACT_TOPICS
from robot_console.topics import CONTRACT_TOPICS, namespaced
from robot_console.wire import add_url_argument, parse_url

#: Exit codes, so a shell can tell "nothing there" from "the wrong thing is there".
EXIT_OK = 0
EXIT_MISSING = 1
EXIT_TRANSPORT = 2
#: `--rates --gate` measured something outside the gate (or could not measure it).
EXIT_RATE = 3


#: What a mobile base must present, from `topics.py` -- the myAGV contract
#: (`robots_specs/myagv/ros.yml`): every topic it lists.
BASE_TOPICS: tuple[str, ...] = tuple(CONTRACT_TOPICS)

#: What a humanoid must present, from `ainex_topics.py` -- the Hiwonder AiNex contract.
#: A kind of its own rather than a flavour of base: the AiNex is commanded as a walking
#: state machine and has no `/cmd_vel` or `/odom`.
HUMANOID_TOPICS: tuple[str, ...] = AINEX_CONTRACT_TOPICS


def arm_topics() -> tuple[str, ...]:
    """What an arm must present, from `arm/ros_settings.py` (imported lazily: the arm's
    kinematics come with it, and a console without them can still check a base)."""
    from robot_console.arm import ros_settings as rs

    return (rs.ARM_COMMAND_TOPIC, rs.JOINT_STATES_TOPIC, rs.TF_TOPIC, rs.TF_STATIC_TOPIC)


def scene_topics() -> tuple[str, ...]:
    """What the worktop's fixed camera rig presents, under its own namespace."""
    from robot_console.arm import ros_settings as rs

    return (rs.OVERHEAD_CAMERA_TOPIC, rs.SIDE_CAMERA_TOPIC)


# ------------------------------------------------------------------ the contract, timed


@dataclasses.dataclass(frozen=True)
class Periodic:
    """A periodic topic as its robot's ROS file declares it.

    `hz` is the rate the topic carries: the **sum** over every row of that name, because
    a name several nodes publish on (the myAGV's `/tf` has five publishers) carries all of
    them. `fastest_hz` is the fastest single publisher's rate, and it is what bounds a gap:
    the interleaving of five clocks makes the summed period meaningless between messages,
    while the fastest publisher alone promises one every `1 / fastest_hz`.
    """

    type: str
    hz: float
    fastest_hz: float


#: Every periodic `out` topic of each member kind, bare, from `robots_specs/<id>/ros*.yml`
#: (`rate_hz` a number) and, for the rig, the simulator spec §3 / `ros_settings`. The
#: rates are the ROS files' and nothing else's; `tests/test_fleet.py` reads those files
#: and fails on any difference, in either direction.
PERIODIC: dict[str, dict[str, Periodic]] = {
    "so101": {
        "/joint_trajectory_controller/controller_state":
            Periodic("control_msgs/msg/JointTrajectoryControllerState", 50, 50),
        "/joint_states": Periodic("sensor_msgs/msg/JointState", 50, 50),
        "/dynamic_joint_states": Periodic("control_msgs/msg/DynamicJointState", 50, 50),
        "/tf": Periodic("tf2_msgs/msg/TFMessage", 20, 20),
        "/diagnostics": Periodic("diagnostic_msgs/msg/DiagnosticArray", 1, 1),
        "/controller_manager/introspection_data/full":
            Periodic("pal_statistics_msgs/msg/Statistics", 50, 50),
        "/controller_manager/introspection_data/values":
            Periodic("pal_statistics_msgs/msg/StatisticsValues", 50, 50),
        "/controller_manager/statistics/full":
            Periodic("pal_statistics_msgs/msg/Statistics", 50, 50),
        "/controller_manager/statistics/values":
            Periodic("pal_statistics_msgs/msg/StatisticsValues", 50, 50),
        "/wrist/image_raw": Periodic("sensor_msgs/msg/Image", 30, 30),
        "/wrist/camera_info": Periodic("sensor_msgs/msg/CameraInfo", 30, 30),
        "/wrist/image_raw/compressed": Periodic("sensor_msgs/msg/CompressedImage", 30, 30),
        "/wrist/image_raw/compressedDepth":
            Periodic("sensor_msgs/msg/CompressedImage", 30, 30),
        "/wrist/image_raw/theora": Periodic("theora_image_transport/msg/Packet", 30, 30),
        "/wrist/image_raw/zstd": Periodic("sensor_msgs/msg/CompressedImage", 30, 30),
    },
    "myagv": {
        "/odom": Periodic("nav_msgs/Odometry", 100, 100),
        "/imu": Periodic("sensor_msgs/Imu", 100, 100),
        "/Voltage": Periodic("std_msgs/Float32", 100, 100),
        "/voltage_backup": Periodic("std_msgs/Float32", 100, 100),
        "/joint_states": Periodic("sensor_msgs/JointState", 10, 10),
        # joint_state_publisher 10 + robot_state_publisher 20 + two tf1 statics at 20 and
        # 100 (the camera's and the IMU's) + robot_pose_ekf 30, all on one name.
        "/tf": Periodic("tf2_msgs/TFMessage", 180, 100),
        "/robot_pose_ekf/odom_combined": Periodic("nav_msgs/Odometry", 30, 30),
        "/scan": Periodic("sensor_msgs/LaserScan", 30, 30),
        "/point_cloud": Periodic("sensor_msgs/PointCloud", 30, 30),
        "/camera/image_raw": Periodic("sensor_msgs/Image", 30, 30),
        "/camera/camera_info": Periodic("sensor_msgs/CameraInfo", 30, 30),
        "/camera/image_raw/compressed": Periodic("sensor_msgs/CompressedImage", 30, 30),
    },
    "ainex": {
        "/ros_robot_controller/imu_raw": Periodic("sensor_msgs/Imu", 100, 100),
        "/ros_robot_controller/mag_raw": Periodic("sensor_msgs_ext/magnetometer", 100, 100),
        "/ros_robot_controller/mag": Periodic("sensor_msgs/MagneticField", 100, 100),
        "/imu_corrected": Periodic("sensor_msgs/Imu", 100, 100),
        "/imu": Periodic("sensor_msgs/Imu", 100, 100),
        "/camera/image_raw": Periodic("sensor_msgs/Image", 30, 30),
        "/camera/camera_info": Periodic("sensor_msgs/CameraInfo", 30, 30),
        "/camera/image_raw/compressed": Periodic("sensor_msgs/CompressedImage", 30, 30),
        "/camera/image_rect_color": Periodic("sensor_msgs/Image", 30, 30),
        "/joy": Periodic("sensor_msgs/Joy", 20, 20),
        "/sensor/button/get_button_state": Periodic("std_msgs/Bool", 50, 50),
    },
    # The rig (simulator spec §3): `SCENE_CAMERA_HZ`, held equal to `ros_settings`'.
    "scene": {
        "/overhead/color/compressed": Periodic("sensor_msgs/msg/CompressedImage", 10, 10),
        "/overhead/color/camera_info": Periodic("sensor_msgs/msg/CameraInfo", 10, 10),
        "/side/color/compressed": Periodic("sensor_msgs/msg/CompressedImage", 10, 10),
        "/side/color/camera_info": Periodic("sensor_msgs/msg/CameraInfo", 10, 10),
    },
}

#: Every other topic each kind's ROS file declares (`event` or `latched`, either
#: direction), with its type. Together with `PERIODIC` this is the member's whole topic
#: surface, which is what lets "a topic on the wire nobody declared a rate for" be found.
APERIODIC: dict[str, dict[str, str]] = {
    "so101": {
        "/joint_trajectory_controller/joint_trajectory": "trajectory_msgs/msg/JointTrajectory",
        "/joint_trajectory_controller/speed_scaling_input":
            "control_msgs/msg/SpeedScalingFactor",
        "/robot_description": "std_msgs/msg/String",
        "/tf_static": "tf2_msgs/msg/TFMessage",
        "/controller_manager/activity": "controller_manager_msgs/msg/ControllerManagerActivity",
        "/controller_manager/introspection_data/names": "pal_statistics_msgs/msg/StatisticsNames",
        "/controller_manager/statistics/names": "pal_statistics_msgs/msg/StatisticsNames",
    },
    "myagv": {
        "/cmd_vel": "geometry_msgs/Twist",
        "/tf_static": "tf2_msgs/TFMessage",
    },
    "ainex": {
        "/walking/set_param": "ainex_interfaces/WalkingParam",
        "/app/set_walking_param": "ainex_interfaces/AppWalkingParam",
        "/app/set_action": "std_msgs/String",
        "/head_pan_controller/command": "ainex_interfaces/HeadState",
        "/head_tilt_controller/command": "ainex_interfaces/HeadState",
        "/walking/is_walking": "std_msgs/Bool",
        "/ros_robot_controller/bus_servo/set_position": "ros_robot_controller/SetBusServosPosition",
        "/ros_robot_controller/bus_servo/set_state": "ros_robot_controller/SetBusServoState",
        "/ros_robot_controller/pwm_servo/set_state": "ros_robot_controller/SetPWMServoState",
        "/ros_robot_controller/set_led": "ros_robot_controller/LedState",
        "/ros_robot_controller/set_buzzer": "ros_robot_controller/BuzzerState",
        "/ros_robot_controller/set_oled": "ros_robot_controller/OLEDState",
        "/ros_robot_controller/set_motor": "ros_robot_controller/MotorsState",
        "/ros_robot_controller/set_rgb": "ros_robot_controller/RGBsState",
        "/ros_robot_controller/set_motor_duty": "ros_robot_controller/MotorsState",
        "/ros_robot_controller/enable_reception": "std_msgs/Bool",
        "/ros_robot_controller/joy": "sensor_msgs/Joy",
        "/ros_robot_controller/sbus": "ros_robot_controller/Sbus",
        "/ros_robot_controller/button": "ros_robot_controller/ButtonState",
        "/ros_robot_controller/battery": "std_msgs/UInt16",
        "/sensor/led/set_led_state": "std_msgs/Bool",
        "/color_detection/update_detect": "ainex_interfaces/ColorsDetect",
        "/color_detection/image_result": "sensor_msgs/Image",
        "/face_detect/image_result": "sensor_msgs/Image",
        "/object/pixel_coords": "ainex_interfaces/ObjectsInfo",
        "/app/image_result": "sensor_msgs/Image",
    },
    "scene": {"/tf_static": "tf2_msgs/msg/TFMessage"},
}

#: Declared topics a conforming wire may leave out: the SO-101's optional
#: `image_transport_plugins` streams, which ros2.yml marks `unverified` because the
#: bringup only has them when those plugins happen to be installed. The simulator's
#: contract module leaves the same three out; they are neither required nor timed.
OPTIONAL: dict[str, frozenset[str]] = {
    "so101": frozenset({
        "/wrist/image_raw/compressedDepth",
        "/wrist/image_raw/theora",
        "/wrist/image_raw/zstd",
    }),
}

#: Simulator spec §5, as numbers.
WARMUP_S = 5.0
MIN_WINDOW_S = 30.0
PERIODS_IN_WINDOW = 5.0
HZ_TOLERANCE = 0.10
MAX_GAP_PERIODS = 3.0
RTF_MEAN_BOUNDS = (0.90, 1.10)
RTF_WINDOW_S = 10.0
RTF_WINDOW_FLOOR = 0.90
#: Where rolling RTF windows start, relative to each other.
RTF_WINDOW_STEP_S = 1.0


def required_window_s(expected: Mapping[str, Periodic]) -> float:
    """The greater of 30 s and five periods of the slowest declared topic."""
    slowest = min((p.hz for p in expected.values()), default=0.0)
    periods = PERIODS_IN_WINDOW / slowest if slowest > 0 else 0.0
    return max(MIN_WINDOW_S, periods)


# ------------------------------------------------------------------ validation


def missing_for(present: dict[str, str], namespace: str,
                expected: tuple[str, ...]) -> list[str]:
    """Which of `expected`, under `namespace`, the wire is not offering."""
    return [t for t in (namespaced(e, namespace) for e in expected) if t not in present]


def contract_of(kind: str) -> dict[str, str]:
    """Every topic the kind's contract requires on the wire, bare, with its type."""
    topics = {name: p.type for name, p in PERIODIC.get(kind, {}).items()}
    topics.update(APERIODIC.get(kind, {}))
    for name in OPTIONAL.get(kind, ()):
        topics.pop(name, None)
    return topics


def _declared(kind: str) -> set[str]:
    return set(PERIODIC.get(kind, {})) | set(APERIODIC.get(kind, {})) | set(OPTIONAL.get(kind, ()))


def _owned_by(topic: str, namespace: str, others: Iterable[str]) -> bool:
    """Is `topic` under `namespace`, and not under a longer member namespace inside it."""
    if not namespace or not topic.startswith(f"/{namespace}/"):
        return False
    return not any(o != namespace and o.startswith(namespace + "/")
                   and topic.startswith(f"/{o}/") for o in others)


def validate(present: Mapping[str, str]) -> tuple[list, list[str]]:
    """The discovered members and every way the wire departs from their contracts.

    Each problem is one line naming the wire name. A bare member (namespace '') is not
    checked for undeclared names: on a bare wire every name is at the root, the rig's
    and `rosapi`'s among them, and there is no namespace to attribute them by.
    """
    from robot_console.discovery import find_members

    members, problems = find_members(present)
    if not members:
        problems.append("no fleet member is on the wire (no typed signature topic found)")
    namespaces = [m.namespace for m in members]
    for member in members:
        if member.kind not in PERIODIC:
            problems.append(f"{member.describe()}: the console has no rate declaration "
                            f"for a {member.kind}")
        for bare, kind_type in sorted(contract_of(member.kind).items()):
            wire = namespaced(bare, member.namespace)
            if wire not in present:
                problems.append(f"{member.describe()}: {wire} is missing")
            elif present[wire] != kind_type:
                problems.append(f"{member.describe()}: {wire} is {present[wire] or 'untyped'}, "
                                f"not {kind_type}")
        declared = {namespaced(b, member.namespace) for b in _declared(member.kind)}
        for topic in sorted(present):
            if "/_action/" in topic or not _owned_by(topic, member.namespace, namespaces):
                continue
            if topic not in declared:
                problems.append(f"{member.describe()}: {topic} is not in its contract, so "
                                "it has no rate declaration")
    return members, problems


def expected_rates(members, present: Mapping[str, str]) -> dict[str, Periodic]:
    """`{wire topic: Periodic}` for every periodic topic the members must carry."""
    expected: dict[str, Periodic] = {}
    for member in members:
        optional = OPTIONAL.get(member.kind, frozenset())
        for bare, periodic in PERIODIC.get(member.kind, {}).items():
            wire = namespaced(bare, member.namespace)
            if bare in optional and wire not in present:
                continue
            expected[wire] = periodic
    return expected


def clocks_of(members, topics: Iterable[str]) -> dict[str, str]:
    """`{topic: member label}` -- which member publishes each topic, for the RTF breakdown."""
    namespaces = [m.namespace for m in members]
    out = {}
    for topic in topics:
        for member in members:
            if not member.namespace or _owned_by(topic, member.namespace, namespaces):
                out[topic] = member.namespace or member.kind
                if member.namespace:
                    break
    return out


# ------------------------------------------------------------------ roslibpy transport


def topics_from(client, timeout_s: float = 10.0) -> dict[str, str]:
    """`{topic: type}` from an already-connected roslibpy client."""
    import roslibpy

    service = roslibpy.Service(client, "/rosapi/topics", "rosapi/Topics")
    result = service.call(roslibpy.ServiceRequest(), timeout=timeout_s)
    return _topic_table(result)


def _topic_table(values: Mapping) -> dict[str, str]:
    names = list(values.get("topics") or [])
    types = list(values.get("types") or [])
    # `types` is positional against `names` and a real rosapi can return fewer of
    # them; pad rather than zip short, so a missing type never hides a topic.
    types += [""] * (len(names) - len(types))
    return dict(zip(names, types))


def list_topics(url: str, timeout_s: float = 10.0) -> dict[str, str]:
    """`{topic: type}` as `/rosapi/topics` reports it. Raises on transport failure.

    Closes the connection and deliberately does **not** terminate it. `close()` sends a
    websocket close; `terminate()` stops roslibpy's process-global Twisted reactor, which
    cannot be restarted -- and `discovery.discover` calls this and then hands the process
    to a `RobotLink` that has to connect afterwards.
    """
    import roslibpy

    host, port = parse_url(url)
    client = roslibpy.Ros(host=host, port=port)
    client.run(timeout=timeout_s)
    try:
        return topics_from(client, timeout_s)
    finally:
        client.close()


# ------------------------------------------------------------------ stdlib websocket


class WireError(ConnectionError):
    """The websocket failed or closed."""


class Wire:
    """A minimal RFC 6455 client: text frames out, frames in, heads kept.

    `recv(keep)` returns `(arrival, payload_head, complete)` for the next message, keeping
    only its first `keep` bytes (all of them with `keep=None`) and reading the rest into a
    scratch buffer. That is the whole reason this exists: a raw 640x480 image is 1.2 MB
    of base64 and the rate check needs its topic name and header stamp, which rosbridge
    writes in the first few hundred bytes.
    """

    def __init__(self, url: str, timeout_s: float = 10.0) -> None:
        parts = urlsplit(url)
        if parts.scheme != "ws":
            raise WireError(f"{url}: only ws:// is supported by the rate check")
        try:
            host, port = parse_url(url)
        except ValueError as exc:
            raise WireError(str(exc)) from None
        self._sock = socket.create_connection((host, port), timeout=timeout_s)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = parts.path or "/"
        request = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n")
        self._sock.sendall(request.encode("ascii"))
        self._buf = bytearray()
        while b"\r\n\r\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise WireError(f"{url} closed during the websocket handshake")
            self._buf += chunk
        head, _, rest = bytes(self._buf).partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise WireError(f"{url} refused the websocket upgrade: {head[:80]!r}")
        self._buf = bytearray(rest)
        self._scratch = bytearray(1 << 20)
        self._frame_timeout_s = timeout_s

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        self._sock.close()

    def send(self, message: dict) -> None:
        self._send_frame(0x1, json.dumps(message).encode("utf-8"))

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < 1 << 16:
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(header + mask + masked)

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(max(65536, n - len(self._buf)))
            if not chunk:
                raise WireError("the rosbridge closed the connection")
            self._buf += chunk
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def _read_payload(self, n: int, keep: Optional[int]) -> bytes:
        take = n if keep is None else min(n, keep)
        head = self._read_exact(take)
        left = n - take
        # Whatever is already buffered first, then straight off the socket into scratch.
        from_buf = min(left, len(self._buf))
        del self._buf[:from_buf]
        left -= from_buf
        view = memoryview(self._scratch)
        while left:
            got = self._sock.recv_into(view, min(left, len(self._scratch)))
            if not got:
                raise WireError("the rosbridge closed the connection")
            left -= got
        return head

    def recv(self, keep: Optional[int] = 1024, timeout_s: Optional[float] = None
             ) -> tuple[float, bytes]:
        """The next data message: `(monotonic arrival, its first `keep` bytes)`.

        `timeout_s` bounds the wait for a message to *start* (raising `socket.timeout`
        with nothing consumed); once one has, it is read to the end, so a timeout can
        never leave the stream half way through a frame.
        """
        head = b""
        while True:
            if not self._buf:
                self._sock.settimeout(timeout_s)
                chunk = self._sock.recv(65536)
                if not chunk:
                    raise WireError("the rosbridge closed the connection")
                self._buf += chunk
            self._sock.settimeout(self._frame_timeout_s)
            b0, b1 = self._read_exact(2)
            opcode, fin = b0 & 0x0F, bool(b0 & 0x80)
            n = b1 & 0x7F
            if n == 126:
                (n,) = struct.unpack("!H", self._read_exact(2))
            elif n == 127:
                (n,) = struct.unpack("!Q", self._read_exact(8))
            mask = self._read_exact(4) if b1 & 0x80 else None
            want = None if keep is None else max(0, keep - len(head))
            payload = self._read_payload(n, want)
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x8:
                raise WireError("the rosbridge closed the connection")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            head += payload
            if fin:
                return time.monotonic(), head

    def call(self, service: str, args: Optional[dict] = None, timeout_s: float = 10.0) -> dict:
        """Call a service and return its `values`, skipping any other traffic."""
        call_id = f"fleet:{service}:{time.monotonic_ns()}"
        self.send({"op": "call_service", "id": call_id, "service": service, "args": args or {}})
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{service} did not answer within {timeout_s} s")
            try:
                _, raw = self.recv(keep=None, timeout_s=left)
            except socket.timeout as exc:
                raise TimeoutError(f"{service} did not answer within {timeout_s} s") from exc
            try:
                message = json.loads(raw)
            except ValueError:
                continue
            if message.get("op") == "service_response" and message.get("id") == call_id:
                if message.get("result") is False:
                    raise WireError(f"{service} failed: {message.get('values')}")
                return message.get("values") or {}


# ------------------------------------------------------------------ observing


_TOPIC = re.compile(rb'"topic"\s*:\s*"([^"]+)"')
_STAMP = re.compile(rb'"stamp"\s*:\s*\{([^{}]*)\}')
_SECS = re.compile(rb'"(?:secs|sec)"\s*:\s*(-?\d+)')
_NSECS = re.compile(rb'"(?:nsecs|nanosec)"\s*:\s*(\d+)')


def parse_head(head: bytes) -> tuple[Optional[str], Optional[float]]:
    """`(topic, header stamp in s)` from the first bytes of a rosbridge `publish` frame.

    Both dialects' stamps: ROS 1 `{secs, nsecs}` and ROS 2 `{sec, nanosec}`. A message
    with no header in its head -- `std_msgs/Float32`, a `Bool` -- has no stamp, and a
    zero stamp is "unset", not the epoch; both come back as None.
    """
    if b'"publish"' not in head:
        return None, None
    topic_match = _TOPIC.search(head)
    topic = topic_match.group(1).decode("utf-8", "replace") if topic_match else None
    stamp = None
    stamp_match = _STAMP.search(head)
    if stamp_match:
        secs = _SECS.search(stamp_match.group(1))
        nsecs = _NSECS.search(stamp_match.group(1))
        if secs and nsecs:
            value = int(secs.group(1)) + int(nsecs.group(1)) * 1e-9
            stamp = value if value != 0 else None
    return topic, stamp


@dataclasses.dataclass
class Observation:
    """What arrived: per topic, `(arrival, stamp or None)` in arrival order."""

    start: float
    end: float
    arrivals: dict[str, list[tuple[float, Optional[float]]]]

    @property
    def window_s(self) -> float:
        return self.end - self.start


def observe(wire: Wire, expected: Mapping[str, Periodic], warmup_s: float, window_s: float,
            clock=time.monotonic) -> Observation:
    """Subscribe unthrottled to every expected topic, warm up, and record the window."""
    for topic, periodic in sorted(expected.items()):
        wire.send({"op": "subscribe", "id": f"fleet-rates:{topic}", "topic": topic,
                   "type": periodic.type, "throttle_rate": 0, "queue_length": 0})
    start = clock() + warmup_s
    end = start + window_s
    arrivals: dict[str, list[tuple[float, Optional[float]]]] = {t: [] for t in expected}
    while True:
        now = clock()
        if now >= end:
            break
        try:
            arrival, head = wire.recv(keep=768, timeout_s=max(0.05, end - now))
        except socket.timeout:
            continue
        if arrival < start:
            continue
        if arrival > end:
            break
        topic, stamp = parse_head(head)
        if topic in arrivals:
            arrivals[topic].append((arrival, stamp))
    for topic in sorted(expected):
        try:
            wire.send({"op": "unsubscribe", "id": f"fleet-rates:{topic}", "topic": topic})
        except OSError:
            break
    return Observation(start, end, arrivals)


# ------------------------------------------------------------------ judging


@dataclasses.dataclass(frozen=True)
class TopicRate:
    topic: str
    expected_hz: float
    measured_hz: float
    max_gap_s: float
    gap_limit_s: float
    count: int

    @property
    def ratio(self) -> float:
        return self.measured_hz / self.expected_hz if self.expected_hz else float("nan")

    @property
    def rate_ok(self) -> bool:
        return abs(self.ratio - 1.0) <= HZ_TOLERANCE + 1e-9

    @property
    def gap_ok(self) -> bool:
        return self.max_gap_s <= self.gap_limit_s + 1e-9

    @property
    def ok(self) -> bool:
        return self.rate_ok and self.gap_ok


@dataclasses.dataclass(frozen=True)
class ClockRtf:
    """The real-time factor as a set of topics' header stamps tell it."""

    name: str
    topics: int
    mean: Optional[float]
    worst_window: Optional[float]


@dataclasses.dataclass
class RateReport:
    rows: list[TopicRate]
    window_s: float
    required_window_s: float
    #: The whole fleet's factor, from every stamped topic: the one the gate judges.
    fleet: Optional[ClockRtf]
    #: The same measure over each member's own topics: a breakdown, not gated.
    members: list[ClockRtf]
    failures: list[str]

    @property
    def ok(self) -> bool:
        return not self.failures


def _topic_rtf(samples: list[tuple[float, float]]) -> Optional[float]:
    """Stamped seconds per wall second across `samples`, or None with too few."""
    if len(samples) < 2:
        return None
    (a0, s0), (a1, s1) = samples[0], samples[-1]
    if a1 - a0 <= 0:
        return None
    return (s1 - s0) / (a1 - a0)


def _median_rtf(stamped: Mapping[str, list[tuple[float, float]]], lo: float, hi: float
                ) -> Optional[float]:
    values = []
    for samples in stamped.values():
        rtf = _topic_rtf([s for s in samples if lo <= s[0] <= hi])
        if rtf is not None:
            values.append(rtf)
    return statistics.median(values) if values else None


def _clock_rtf(name: str, stamped: Mapping[str, list[tuple[float, float]]],
               observation: "Observation") -> ClockRtf:
    mean = _median_rtf(stamped, observation.start, observation.end)
    worst = None
    t = observation.start
    while t + RTF_WINDOW_S <= observation.end + 1e-9:
        value = _median_rtf(stamped, t, t + RTF_WINDOW_S)
        if value is not None:
            worst = value if worst is None else min(worst, value)
        t += RTF_WINDOW_STEP_S
    return ClockRtf(name, len(stamped), mean, worst)


def evaluate(expected: Mapping[str, Periodic], observation: "Observation",
             required_s: Optional[float] = None, problems: Iterable[str] = (),
             clocks: Optional[Mapping[str, str]] = None) -> RateReport:
    """Judge an observation against spec §5. Pure: everything it needs is passed in.

    The real-time factor is the fleet's: every member stamps the one simulated clock,
    which starts from zero, so every stamped topic of every member measures it. Over a
    span it is, for each stamped topic, the stamped seconds over the wall seconds between
    the first and last arrival in the span, and the median across topics, so one topic
    with a frozen stamp cannot speak for the fleet. Zero stamps are "unset" and ignored; a
    fleet with no stamped topic fails, since then nothing measured the clock.

    `clocks` maps each topic to its member, for a per-member breakdown of the same
    measure; the breakdown is reported, not gated.
    """
    required = required_window_s(expected) if required_s is None else required_s
    failures = list(problems)
    window = observation.window_s
    if window + 1e-6 < required:
        failures.append(f"insufficient observation window: {window:.1f} s observed, "
                        f"{required:.1f} s required")
    rows = []
    stamped: dict[str, list[tuple[float, float]]] = {}
    by_member: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for topic in sorted(expected):
        periodic = expected[topic]
        samples = [(a, s) for a, s in observation.arrivals.get(topic, [])
                   if observation.start <= a <= observation.end]
        times = [a for a, _ in samples]
        edges = [observation.start, *times, observation.end]
        max_gap = max(b - a for a, b in zip(edges, edges[1:]))
        rows.append(TopicRate(topic, periodic.hz, len(times) / window if window > 0 else 0.0,
                              max_gap, MAX_GAP_PERIODS / periodic.fastest_hz, len(times)))
        stamps = [(a, s) for a, s in samples if s is not None]
        if len(stamps) >= 2:
            stamped[topic] = stamps
            member = (clocks or {}).get(topic)
            if member is not None:
                by_member.setdefault(member, {})[topic] = stamps
    for row in rows:
        if not row.count:
            failures.append(f"{row.topic}: no message in {window:.1f} s")
        elif not row.rate_ok:
            failures.append(f"{row.topic}: {row.measured_hz:.2f} Hz against {row.expected_hz:g} "
                            f"Hz ({row.ratio:.2f}x, outside +/-{HZ_TOLERANCE:.0%})")
        if row.count and not row.gap_ok:
            failures.append(f"{row.topic}: a {row.max_gap_s * 1000:.0f} ms gap, over "
                            f"{MAX_GAP_PERIODS:g} periods ({row.gap_limit_s * 1000:.0f} ms)")
    fleet = _clock_rtf("fleet", stamped, observation) if stamped else None
    members = [_clock_rtf(name, by_member[name], observation) for name in sorted(by_member)]
    lo, hi = RTF_MEAN_BOUNDS
    if fleet is None:
        failures.append("real-time factor: no topic carried two stamped messages, so no "
                        "clock was observed")
    else:
        if fleet.mean is None or not lo <= fleet.mean <= hi:
            mean = "n/a" if fleet.mean is None else f"{fleet.mean:.3f}"
            failures.append(f"real-time factor: mean {mean}, outside [{lo}, {hi}]")
        if fleet.worst_window is None:
            failures.append(f"real-time factor: no full {RTF_WINDOW_S:g} s window observed")
        elif fleet.worst_window < RTF_WINDOW_FLOOR:
            failures.append(f"real-time factor: a {RTF_WINDOW_S:g} s window at "
                            f"{fleet.worst_window:.3f}, below {RTF_WINDOW_FLOOR}")
    return RateReport(rows, window, required, fleet, members, failures)


def format_report(report: RateReport, url: str) -> str:
    lines = [f"rates on {url}: {len(report.rows)} periodic topic(s), "
             f"{report.window_s:.1f} s observed ({report.required_window_s:.1f} s required) "
             f"after a {WARMUP_S:g} s warm-up",
             f"{'topic':56s} {'exp Hz':>7s} {'got Hz':>7s} {'ratio':>6s} {'max gap':>8s} "
             f"{'limit':>7s}"]
    for row in report.rows:
        flag = "" if row.ok and row.count else "  FAIL"
        lines.append(f"{row.topic:56s} {row.expected_hz:7g} {row.measured_hz:7.2f} "
                     f"{row.ratio:6.2f} {row.max_gap_s * 1000:6.0f}ms "
                     f"{row.gap_limit_s * 1000:5.0f}ms{flag}")
    lines.append(f"real-time factor (gate: mean {RTF_MEAN_BOUNDS[0]}-{RTF_MEAN_BOUNDS[1]}, "
                 f"every {RTF_WINDOW_S:g} s window >= {RTF_WINDOW_FLOOR}):")
    breakdown = report.members if len(report.members) > 1 else []
    for clock in ([report.fleet] if report.fleet else []) + breakdown:
        mean = "n/a" if clock.mean is None else f"{clock.mean:.3f}"
        worst = "n/a" if clock.worst_window is None else f"{clock.worst_window:.3f}"
        indent = "  " if clock is report.fleet else "    "
        lines.append(f"{indent}{clock.name:12s} mean {mean}  worst window {worst}  "
                     f"from {clock.topics} stamped topic(s)")
    if report.failures:
        lines.append(f"FAIL: {len(report.failures)} problem(s)")
        lines += [f"  {f}" for f in report.failures]
    else:
        lines.append("PASS")
    return "\n".join(lines)


def measure_rates(url: str, warmup_s: float = WARMUP_S, window_s: Optional[float] = None,
                  timeout_s: float = 10.0, required_s: Optional[float] = None) -> RateReport:
    """Discover, validate and time a live wire. Raises `WireError`/`OSError` on transport."""
    wire = Wire(url, timeout_s)
    try:
        present = _topic_table(wire.call("/rosapi/topics", timeout_s=timeout_s))
        members, problems = validate(present)
        expected = expected_rates(members, present)
        # A topic that is missing or mistyped is already a problem; timing it would only
        # add a second line for the same fault.
        measurable = {t: p for t, p in expected.items() if present.get(t) == p.type}
        required = required_window_s(expected) if required_s is None else required_s
        observation = observe(wire, measurable, warmup_s,
                              required if window_s is None else window_s)
        return evaluate(measurable, observation, required, problems,
                        clocks=clocks_of(members, measurable))
    finally:
        wire.close()


# ------------------------------------------------------------------ the command


def _legacy_check(args, present: dict[str, str]) -> int:
    missing: list[str] = []
    for namespace in args.base:
        missing += missing_for(present, namespace, BASE_TOPICS)
    for namespace in args.humanoid:
        missing += missing_for(present, namespace, HUMANOID_TOPICS)
    for namespace in args.arm:
        missing += missing_for(present, namespace, arm_topics())
    # The rig is the worktop's, staged for every robot that stands at one.
    if args.arm or args.humanoid:
        from robot_console.arm import ros_settings as rs

        missing += missing_for(present, rs.SCENE_NAMESPACE, scene_topics())
    if missing:
        print(f"{args.url} is missing {len(missing)} expected topic(s):")
        for topic in missing:
            print(f"  {topic}")
        print(f"it offers {len(present)}: {', '.join(sorted(present))}")
        return EXIT_MISSING
    robots = [f"arm {ns or '<bare>'}" for ns in args.arm]
    robots += [f"base {ns or '<bare>'}" for ns in args.base]
    robots += [f"humanoid {ns or '<bare>'}" for ns in args.humanoid]
    print(f"{args.url}: {len(present)} topics, all expected ones present ({'; '.join(robots)})")
    return EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m robot_console.fleet",
        description="Validate, list or time the fleet on a rosbridge wire.")
    add_url_argument(parser)
    parser.add_argument("--dump", action="store_true",
                        help="print every topic on the wire with its type, sorted, and exit 0; "
                             "two engines' dumps must be identical")
    parser.add_argument("--rates", action="store_true",
                        help="time every periodic topic and the real-time factor (spec §5)")
    parser.add_argument("--gate", action="store_true",
                        help="with --rates: exit non-zero when anything fails the gate")
    parser.add_argument("--window", type=float, default=None, metavar="S",
                        help="with --rates: observe for S seconds instead of the required "
                             "window; shorter than required fails")
    parser.add_argument("--warmup", type=float, default=WARMUP_S, help=argparse.SUPPRESS)
    parser.add_argument("--arm", action="append", default=[], metavar="NS",
                        help="legacy: an SO-101 is expected under NS (and the rig); repeatable")
    parser.add_argument("--base", action="append", default=[], metavar="NS",
                        help="legacy: a myAGV is expected under NS; repeatable")
    parser.add_argument("--humanoid", action="append", default=[], metavar="NS",
                        help="legacy: an AiNex is expected under NS (and the rig); repeatable")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.gate and not args.rates:
        parser.error("--gate only applies to --rates")

    if args.rates:
        try:
            report = measure_rates(args.url, args.warmup, args.window, args.timeout)
        except (OSError, TimeoutError, WireError) as exc:
            print(f"cannot reach rosbridge at {args.url}: {exc}")
            return EXIT_TRANSPORT
        print(format_report(report, args.url))
        return EXIT_RATE if (args.gate and not report.ok) else EXIT_OK

    try:
        present = list_topics(args.url, args.timeout)
    except Exception as exc:  # noqa: BLE001 - every transport failure means the same thing
        print(f"cannot reach rosbridge at {args.url}: {exc}")
        return EXIT_TRANSPORT

    if args.dump:
        for topic in sorted(present):
            print(f"{topic}\t{present[topic]}")
        return EXIT_OK
    if args.arm or args.base or args.humanoid:
        return _legacy_check(args, present)

    members, problems = validate(present)
    for member in members:
        print(f"  {member.describe()}")
    if problems:
        print(f"{args.url}: {len(problems)} problem(s) against the console's contract:")
        for line in problems:
            print(f"  {line}")
        return EXIT_MISSING
    print(f"{args.url}: {len(present)} topics; {len(members)} member(s), each presenting "
          "its whole contract")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
