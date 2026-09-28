"""Compatibility, live (console spec §4): the same checks against whatever serves the wire.

Discovery, command, stop and observation, run against a rosbridge the console did not
start -- either simulator engine, or physical hardware -- with no knowledge of which it
is. The checks are the console's own: `discovery` and `fleet` decide what is on the wire,
each robot is commanded and stopped through its official interface and its ROS file's
`stop_command`, and observations are decoded as the console decodes them. Nothing here
branches on an engine, and `test_engine_neutral.py` holds the console's source to the same
rule.

    cd simulator && ./kitchen.sh serve --robots so101,myagv,ainex --port 9971
    ROBOT_CONSOLE_LIVE_URL=ws://127.0.0.1:9971 .venv/bin/python -m pytest -m live \
        tests/test_compat_live.py

`ROBOT_CONSOLE_LIVE_ROBOTS=so101,myagv,ainex` additionally requires exactly those robots
(default: whatever is there, at least one). A member kind that is absent skips its checks.
The motion is small and bounded: the myAGV creeps 0.10 m/s for 1.5 s, the AiNex's gait is
started and stopped, and the SO-101's wrist flexes by 0.15 rad and is stopped half way.
Against hardware, arm an independent emergency stop first, as teleop requires.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import pytest

pytestmark = pytest.mark.live

URL = os.environ.get("ROBOT_CONSOLE_LIVE_URL", "ws://127.0.0.1:9090")
EXPECTED = tuple(r for r in os.environ.get("ROBOT_CONSOLE_LIVE_ROBOTS", "").split(",") if r)

#: How long an observation may take to arrive: the slowest periodic topic here is the
#: rig at 10 Hz, and a busy simulator warms up.
OBSERVE_S = 10.0


# ------------------------------------------------------------------ the wire


@pytest.fixture(scope="module")
def wire():
    """The console's own rosbridge client (stdlib websocket, no process-global reactor)."""
    from robot_console.arm.ros_client import ActionRosbridgeClient

    client = ActionRosbridgeClient(URL, connect_timeout_s=5.0, request_timeout_s=15.0)
    try:
        client.connect()
        client.call_service("/rosapi/topics")
    except Exception as exc:  # noqa: BLE001 - nothing listening is the usual case
        client.close()
        pytest.skip(f"no rosbridge with rosapi on {URL}: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def present(wire) -> dict[str, str]:
    from robot_console.arm.preflight import topics

    return topics(wire)


@pytest.fixture(scope="module")
def members(present):
    from robot_console.discovery import find_members

    found, wrong = find_members(present)
    assert wrong == [], f"signatures of the wrong type on {URL}: {wrong}"
    return found


def _member(members, kind: str):
    for m in members:
        if m.kind == kind:
            return m
    pytest.skip(f"no {kind} on {URL}")


_SUB = iter(range(1, 1_000_000))


def _observe(wire, topic: str, message_type: str, timeout: float = OBSERVE_S) -> dict:
    """Subscribe and return the next message on `topic` (the console's subscription)."""
    sub_id = f"compat:{topic}:{next(_SUB)}"
    before = wire.sequence(topic)
    wire.subscribe(topic, subscription_id=sub_id, message_type=message_type,
                   throttle_rate=0, queue_length=1)
    try:
        return dict(wire.wait_for_sample(topic, after_seq=before, timeout_s=timeout).msg)
    finally:
        wire.unsubscribe(topic, subscription_id=sub_id)


def _call(wire, service: str, args: dict | None = None) -> dict[str, Any]:
    return dict(wire.call_service(service, args or {}).values)


# ------------------------------------------------------------------ discovery


def test_discovery_finds_every_member_typed_and_nothing_wrong(present, members) -> None:
    from robot_console.discovery import MEMBER_SIGNATURES, RIG_KIND
    from robot_console.fleet import validate

    kinds = {m.kind for m in members}
    assert kinds, f"no fleet member on {URL}"
    if EXPECTED:
        assert kinds - {RIG_KIND} == set(EXPECTED)
        # The rig is the worktop's, staged for every robot that stands at one.
        assert (RIG_KIND in kinds) == bool({"so101", "ainex"} & set(EXPECTED))
    assert kinds <= {k for k, _t, _y in MEMBER_SIGNATURES} | {RIG_KIND}
    _found, problems = validate(present)
    assert problems == [], "\n".join(problems)


def test_teleop_discovery_picks_each_drivable_robot_unambiguously(present, members) -> None:
    from robot_console.discovery import COMPANIONS, DiscoveryError, discover_from, survey

    found, rejected = survey(present)
    assert rejected == [], [r.describe() for r in rejected]
    drivable = [m for m in members if m.kind in COMPANIONS]
    assert {(d.robot, d.namespace) for d in found} == {(m.kind, m.namespace) for m in drivable}
    for m in drivable:
        chosen = discover_from(present, m.kind, m.namespace)
        assert chosen.camera_topic in present
    if len(drivable) > 1:
        with pytest.raises(DiscoveryError):   # ambiguous until narrowed, listing them
            discover_from(present)


def test_the_arm_interface_is_the_official_one(wire, present, members) -> None:
    from robot_console.arm.preflight import check

    arm = _member(members, "so101")
    check(wire, arm.namespace)   # raises with what refuses or is wrong


# ------------------------------------------------------------------ observation


def test_every_member_observes(wire, present, members) -> None:
    from robot_console import ainex_topics as at
    from robot_console import topics as t
    from robot_console.arm import ros_settings as rs
    from robot_console.camera import decode_compressed_image as decode_compressed
    from robot_console.topics import namespaced

    seen = []
    for m in members:
        ns = m.namespace
        if m.kind == "myagv":
            odom = _observe(wire, namespaced(t.TOPIC_ODOM, ns), t.TYPE_ODOM)
            assert "pose" in odom and "twist" in odom
            cam = namespaced(t.TOPIC_CAMERA, ns)
            frame = decode_compressed(_observe(wire, cam, present[cam]))
        elif m.kind == "ainex":
            imu = _observe(wire, namespaced(at.TOPIC_IMU, ns), at.TYPE_IMU)
            assert "orientation" in imu
            cam = namespaced(at.TOPIC_CAMERA, ns)
            frame = decode_compressed(_observe(wire, cam, present[cam]))
        elif m.kind == "so101":
            joints = _observe(wire, namespaced(rs.JOINT_STATES_TOPIC, ns), rs.JOINT_STATES_TYPE)
            assert set(joints["name"]) == {*rs.ARM_JOINTS, rs.GRIPPER_JOINT}
            cam = namespaced(rs.WRIST_CAMERA_TOPIC, ns)
            frame = decode_compressed(_observe(wire, cam, rs.WRIST_CAMERA_TYPE))
            assert frame is not None and frame.shape[:2] == (rs.WRIST_CAMERA_HEIGHT,
                                                             rs.WRIST_CAMERA_WIDTH)
        else:  # the rig
            for name, (topic, width, height) in rs.CAMERA_SPECS.items():
                if name not in rs.SCENE_CAMERA_NAMES:
                    continue
                frame = decode_compressed(_observe(wire, rs.rig_topic(topic),
                                                   rs.OVERHEAD_CAMERA_TYPE))
                assert frame is not None and frame.shape[:2] == (height, width)
                info = _observe(wire, rs.rig_topic(rs.SCENE_CAMERA_INFO_TOPICS[name]),
                                rs.CAMERA_INFO_TYPE)
                assert (info["width"], info["height"]) == (width, height)
                assert info["k"][0] > 0, "an uncalibrated rig cannot be triangulated through"
        assert frame is not None, f"{m.describe()}: camera frame did not decode"
        seen.append(m.kind)
    assert seen


# ------------------------------------------------------------------ command and stop


def _twist(vx: float = 0.0) -> dict:
    return {"linear": {"x": vx, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}


def test_the_myagv_moves_on_command_and_stops_on_its_stop_command(wire, members) -> None:
    """ros.yml: no watchdog; a zero Twist on /cmd_vel is the stop."""
    from robot_console import topics as t
    from robot_console.topics import namespaced

    ns = _member(members, "myagv").namespace
    cmd, odom = namespaced(t.TOPIC_CMD_VEL, ns), namespaced(t.TOPIC_ODOM, ns)
    wire.advertise(cmd, message_type=t.TYPE_TWIST)
    sub = "compat:odom"
    wire.subscribe(odom, subscription_id=sub, message_type=t.TYPE_ODOM, throttle_rate=0)
    try:
        start = wire.wait_for_sample(odom, timeout_s=OBSERVE_S).msg["pose"]["pose"]["position"]
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            wire.publish(cmd, _twist(0.10))
            time.sleep(0.05)
        moving = wire.latest(odom).msg
        for _ in range(3):   # the supervisor's stop: three times, 50 ms apart
            wire.publish(cmd, _twist())
            time.sleep(0.05)
        time.sleep(1.0)
        stopped = wire.latest(odom).msg["pose"]["pose"]["position"]
        time.sleep(0.5)
        still = wire.latest(odom).msg["pose"]["pose"]["position"]
    finally:
        for _ in range(3):
            wire.publish(cmd, _twist())
        wire.unsubscribe(odom, subscription_id=sub)
        wire.unadvertise(cmd)
    moved = math.dist((start["x"], start["y"]), (stopped["x"], stopped["y"]))
    assert moved > 0.05, f"0.10 m/s for 1.5 s moved the base {moved:.3f} m"
    assert moving["twist"]["twist"]["linear"]["x"] > 0.03
    drift = math.dist((stopped["x"], stopped["y"]), (still["x"], still["y"]))
    assert drift < 0.005, f"after the stop command the base still moved {drift:.4f} m"


def test_the_ainex_gait_starts_on_command_and_stops_on_its_stop_command(wire, members) -> None:
    """ros.yml: `/walking/command` with `enable_control`, then `stop`."""
    from robot_console import ainex_topics as at
    from robot_console.topics import namespaced

    ns = _member(members, "ainex").namespace
    command, state = namespaced(at.SRV_WALKING_COMMAND, ns), namespaced(at.SRV_IS_WALKING, ns)

    def walking() -> bool:
        return bool(_call(wire, state).get("state"))

    try:
        for word in ("enable_control", "enable", "start"):
            assert _call(wire, command, {"command": word}).get("result") is True
        deadline = time.monotonic() + 3.0
        while not walking() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert walking(), "the gait never started"
    finally:
        for word in ("enable_control", "stop"):
            _call(wire, command, {"command": word})
    # `stop` returns once the step cycle in progress has ended.
    deadline = time.monotonic() + 2.0
    while walking() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not walking(), "the stop command did not stop the gait"


def test_the_arm_follows_a_goal_and_holds_where_a_cancel_stops_it(wire, members) -> None:
    """ros2.yml: no stop topic; cancelling the trajectory goal holds the current position."""
    from robot_console.arm import ros_settings as rs
    from robot_console.topics import namespaced

    ns = _member(members, "so101").namespace
    states = namespaced(rs.JOINT_STATES_TOPIC, ns)
    action = namespaced("/joint_trajectory_controller/follow_joint_trajectory", ns)
    sub = "compat:joints"
    wire.subscribe(states, subscription_id=sub, message_type=rs.JOINT_STATES_TYPE,
                   throttle_rate=0)

    def position(name: str) -> float:
        msg = wire.latest(states).msg
        return float(dict(zip(msg["name"], msg["position"]))[name])

    joint = "wrist_flex_joint"
    try:
        wire.wait_for_sample(states, timeout_s=OBSERVE_S)
        start = {n: position(n) for n in rs.ARM_JOINTS}
        target = dict(start)
        target[joint] = start[joint] + (0.15 if start[joint] < 1.0 else -0.15)
        goal = wire.send_goal(action, "control_msgs/action/FollowJointTrajectory", {
            "trajectory": {
                "joint_names": list(rs.ARM_JOINTS),
                "points": [{"positions": [target[n] for n in rs.ARM_JOINTS],
                            "velocities": [0.0] * len(rs.ARM_JOINTS),
                            "time_from_start": {"sec": 3, "nanosec": 0}}],
            },
        })
        time.sleep(1.2)
        assert abs(position(joint) - start[joint]) > 0.01, "the arm did not follow the goal"
        wire.cancel_goal(goal)
        goal.done.wait(3.0)
        time.sleep(0.5)
        held = position(joint)
        time.sleep(1.0)
        assert abs(position(joint) - held) < 0.01, "the arm kept moving after the cancel"
        assert abs(held - target[joint]) > 0.02, "the cancel did not stop the goal short"
    finally:
        wire.cancel_all()
        wire.unsubscribe(states, subscription_id=sub)
