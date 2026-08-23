"""Proof that the bytes roslibpy emits are the bytes the bridge accepts.

Everything roslibpy touches lives in this one module, behind module-scoped fixtures that
never call `terminate()`. roslibpy drives a process-global Twisted reactor that cannot be
restarted once terminated, so a link opened in another test module would either hang or
fail depending on ordering. Two connections coexist here -- a `RobotLink` and an
`AiNexLink`, since the two robots share no topic and so cannot share a fixture -- which
is fine: the reactor is shared, only shutting it down is fatal.
"""

import base64
import math
import time

import cv2
import numpy as np
import pytest

from robot_console.bridge import RobotLink
from robot_console.camera import LatestFrame, decode_compressed_image
from robot_console.teleop import Command
from robot_console.topics import TOPIC_CAMERA, TOPIC_CMD_VEL, TOPIC_ODOM, TOPIC_SCAN

from fake_bridge import FakeBridge


@pytest.fixture(scope="module")
def server():
    bridge = FakeBridge()
    bridge.start()
    yield bridge
    bridge.stop()


@pytest.fixture(scope="module")
def link(server):
    connection = RobotLink("127.0.0.1", server.port)
    connection.connect(timeout=10.0)
    yield connection
    # Deliberately no terminate(): letting the daemon reactor die at interpreter exit
    # avoids wedging any test that runs after this module.
    connection.close(hard_exit_after=None)


@pytest.fixture(scope="module")
def subscriptions(server, link):
    frames = LatestFrame()
    scans = LatestFrame()
    odoms = []
    link.subscribe_odom(odoms.append)
    link.subscribe_camera(frames.offer)
    link.subscribe_scan(scans.offer)
    assert server.wait_for_subscription(TOPIC_ODOM)
    assert server.wait_for_subscription(TOPIC_CAMERA)
    assert server.wait_for_subscription(TOPIC_SCAN)
    return frames, odoms, scans


def test_connects(link):
    assert link.is_connected


def test_subscribes_to_the_contract_topics(server, subscriptions):
    assert {TOPIC_ODOM, TOPIC_CAMERA, TOPIC_SCAN} <= server.subscriptions


def test_publishes_a_well_formed_twist(server, link):
    before = len(server.received_on(TOPIC_CMD_VEL))
    link.publish_cmd_vel(Command(vx=0.15, vy=-0.05, wz=0.3))
    assert server.wait_for_publish(TOPIC_CMD_VEL, before + 1)
    message = server.received_on(TOPIC_CMD_VEL)[-1]
    assert message == {
        "linear": {"x": 0.15, "y": -0.05, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.3},
    }


def test_stop_publishes_zeros(server, link):
    before = len(server.received_on(TOPIC_CMD_VEL))
    link.stop()
    assert server.wait_for_publish(TOPIC_CMD_VEL, before + 1)
    message = server.received_on(TOPIC_CMD_VEL)[-1]
    assert message["linear"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert message["angular"] == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_sustains_the_watchdog_rate(server, link):
    # The bridge stops the base after 0.5 s of silence, so anything at or below 10 Hz
    # would stutter. The console publishes at 20 Hz.
    before = len(server.received_on(TOPIC_CMD_VEL))
    started = time.monotonic()
    sent = 0
    while time.monotonic() - started < 1.0:
        link.publish_cmd_vel(Command(vx=0.1))
        sent += 1
        time.sleep(0.05)
    assert server.wait_for_publish(TOPIC_CMD_VEL, before + sent, timeout=5.0)
    received = len(server.received_on(TOPIC_CMD_VEL)) - before
    assert received / (time.monotonic() - started) >= 10.0


def test_receives_and_parses_odom(server, subscriptions):
    _, odoms, _ = subscriptions
    before = len(odoms)
    server.publish(
        TOPIC_ODOM,
        {
            "header": {"seq": 1, "stamp": {"secs": 3, "nsecs": 0}, "frame_id": "odom"},
            "child_frame_id": "base_footprint",
            "pose": {
                "pose": {
                    "position": {"x": 1.25, "y": -0.5, "z": 0.0},
                    "orientation": {"x": 0, "y": 0, "z": math.sin(0.5), "w": math.cos(0.5)},
                }
            },
            "twist": {"twist": {"linear": {"x": 0.15, "y": 0.0}, "angular": {"z": 0.2}}},
        },
    )
    deadline = time.monotonic() + 5.0
    while len(odoms) == before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(odoms) > before
    odom = odoms[-1]
    assert odom.x == pytest.approx(1.25)
    assert odom.yaw == pytest.approx(1.0)
    assert odom.child_frame_id == "base_footprint"


def test_receives_and_decodes_a_camera_frame(server, subscriptions):
    frames, _, _ = subscriptions
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    server.publish(
        TOPIC_CAMERA,
        {
            "header": {"seq": 9, "stamp": {"secs": 1, "nsecs": 0}, "frame_id": "camera"},
            "format": "jpeg",
            # rosbridge transports uint8[] base64-encoded, not as an int array.
            "data": base64.b64encode(buffer.tobytes()).decode("ascii"),
        },
    )
    deadline = time.monotonic() + 5.0
    taken = None
    while taken is None and time.monotonic() < deadline:
        taken = frames.take()
        time.sleep(0.02)
    assert taken is not None
    decoded = decode_compressed_image(taken[0])
    assert decoded is not None
    assert decoded.shape == image.shape


def ydlidar_scan_message(count=360):
    """A `sensor_msgs/LaserScan` shaped like the YDLidar X2's, CCW from -pi.

    Built inline rather than ray-cast from a scene: nothing in the console interprets
    a scan any more, so the only thing left worth pinning down is the wire encoding.
    """
    return {
        "header": {"seq": 3, "frame_id": "laser_frame"},
        "angle_min": -math.pi,
        "angle_max": math.pi,
        "angle_increment": 2 * math.pi / count,
        "range_min": 0.1,
        "range_max": 12.0,
        # One out-of-range return among the valid ones. The three invalid spellings a
        # client has to cope with are 0.0, inf and range_max + 1; this is the last.
        "ranges": [13.0] + [2.5] * (count - 1),
    }


def test_receives_a_laser_scan_as_a_plain_float_array(server, subscriptions):
    """A scan over the wire is a plain JSON float array -- rosbridge base64-encodes
    uint8[] only, so unlike CompressedImage this must not be decoded.

    Nothing in the console consumes `/scan` since the SLAM subsystem was removed, but
    the topic is still part of the ROS contract and the myAGV still publishes it. This
    is what keeps `RobotLink.subscribe_scan` honest.
    """
    _, _, scans = subscriptions
    message = ydlidar_scan_message()
    server.publish(TOPIC_SCAN, message)

    deadline = time.monotonic() + 5.0
    taken = None
    while taken is None and time.monotonic() < deadline:
        taken = scans.take()
        time.sleep(0.02)
    assert taken is not None

    scan = taken[0]
    ranges = scan["ranges"]
    assert not isinstance(ranges, str), "ranges is a float array, never base64"
    assert len(ranges) == len(message["ranges"])
    assert all(isinstance(r, (int, float)) for r in ranges)
    assert scan["header"]["frame_id"] == "laser_frame", "the YDLidar X2's own frame"
    assert scan["range_max"] == pytest.approx(12.0)
    assert scan["range_min"] == pytest.approx(0.1)
    # range_max + 1 is the simulator's spelling of "no return"; it must survive the wire
    # as-is rather than being clamped, so a client can still tell it from a real hit.
    valid = [r for r in ranges if scan["range_min"] <= r <= scan["range_max"]]
    assert len(valid) == len(ranges) - 1


# ---------------------------------------------------------------- the AiNex


@pytest.fixture(scope="module")
def walker(server):
    """A second connection, to the same fake bridge. See the module docstring."""
    from robot_console.ainex_link import AiNexLink

    connection = AiNexLink("127.0.0.1", server.port)
    connection.connect(timeout=10.0)
    yield connection
    connection.close(hard_exit_after=None)


def _await_calls(server, count, timeout=5.0):
    from robot_console.ainex_link import SRV_WALKING_COMMAND

    server.wait_for_service_call(SRV_WALKING_COMMAND, count, timeout=timeout)
    return [args.get("command") for args in server.calls_on(SRV_WALKING_COMMAND)]


def test_the_walking_link_connects(walker):
    assert walker.is_connected


def test_a_repeated_command_starts_the_gait_once(server, walker):
    """The vendor node restarts its gait phase on `start`, so a `start` per publish makes
    the robot shuffle in place instead of walking. At 20 Hz that is the whole difference
    between a robot that moves and one that does not."""
    from robot_console.ainex_link import PERIOD_S, STRIDE_FACTOR, TOPIC_SET_WALKING_PARAM
    from robot_console.teleop import Command

    params_before = len(server.received_on(TOPIC_SET_WALKING_PARAM))
    calls_before = len(_await_calls(server, 0, timeout=0.0))

    for _ in range(5):
        walker.publish_cmd_vel(Command(vx=0.1))
    assert server.wait_for_publish(TOPIC_SET_WALKING_PARAM, params_before + 1)
    assert _await_calls(server, calls_before + 1)[calls_before:] == ["start"]

    # Give any redundant traffic time to arrive before claiming there was none.
    time.sleep(0.3)
    assert len(server.received_on(TOPIC_SET_WALKING_PARAM)) == params_before + 1
    assert len(server.calls_on("/walking/command")) == calls_before + 1

    message = server.received_on(TOPIC_SET_WALKING_PARAM)[-1]
    assert message["x_move_amplitude"] == pytest.approx(0.1 * PERIOD_S / STRIDE_FACTOR)
    assert message["period_time"] == pytest.approx(PERIOD_S * 1000.0), "milliseconds on the wire"


def test_a_zero_command_stops_the_gait(server, walker):
    from robot_console.teleop import Command

    calls_before = len(_await_calls(server, 0, timeout=0.0))
    walker.publish_cmd_vel(Command())
    assert _await_calls(server, calls_before + 1)[calls_before:] == ["stop"]


def test_stop_is_unconditional(server, walker):
    """`publish_cmd_vel` deduplicates; `stop()` must not. It is the exit path, and a
    stale idea of "already stopped" is how a robot is left walking."""
    calls_before = len(_await_calls(server, 0, timeout=0.0))
    walker.stop()
    assert _await_calls(server, calls_before + 1)[calls_before:] == ["stop"]
