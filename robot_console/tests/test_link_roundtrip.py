"""Proof that the bytes roslibpy emits are the bytes the bridge accepts.

Everything roslibpy touches lives in this one module, behind a session-scoped fixture
holding exactly one connection. roslibpy drives a process-global Twisted reactor that
cannot be restarted once terminated, so a second `RobotLink` in another test would
either hang or fail depending on ordering.
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


def test_receives_and_parses_a_laser_scan(server, subscriptions):
    """A scan over the wire is a plain JSON float array -- rosbridge base64-encodes
    uint8[] only, so unlike CompressedImage this must not be decoded."""
    from synthetic import scan_message

    from robot_console.slam.scan import parse_scan, scan_points

    _, _, scans = subscriptions
    message = scan_message(1.5, 2.0, 0.3)
    server.publish(TOPIC_SCAN, message)

    deadline = time.monotonic() + 5.0
    taken = None
    while taken is None and time.monotonic() < deadline:
        taken = scans.take()
        time.sleep(0.02)
    assert taken is not None

    scan = parse_scan(taken[0])
    assert scan.count == len(message["ranges"])
    assert scan.frame_id == "laser_frame", "the YDLidar X2's own frame"
    assert scan.range_max == pytest.approx(12.0)
    assert scan.valid.sum() > 300
    assert scan_points(scan).shape[1] == 2
