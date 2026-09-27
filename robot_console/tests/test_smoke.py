"""`robot_console.smoke` against the fake bridge, made to move.

A thread integrates whatever Twist the fake bridge last received into `/myagv/odom` and
publishes a camera frame, so the check can pass offline. Then the stop-command guarantee:
every run ends with three zero Twists after the last motion.
"""

from __future__ import annotations

import base64
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import robot_console

SRC = str(Path(robot_console.__file__).resolve().parents[1])
TOPICS = {
    "/myagv/cmd_vel": "geometry_msgs/Twist",
    "/myagv/odom": "nav_msgs/Odometry",
    "/myagv/camera/image_raw/compressed": "sensor_msgs/CompressedImage",
}


class _Base(threading.Thread):
    """A holonomic base that does exactly what the last Twist says, scaled by `gain`."""

    def __init__(self, bridge, gain: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.bridge, self.gain = bridge, gain
        self.pose = [0.0, 0.0, 0.3]
        self.running = True
        ok, jpeg = cv2.imencode(".jpg", (np.random.default_rng(0).random((48, 64, 3)) * 255)
                                .astype(np.uint8))
        self.jpeg = base64.b64encode(jpeg.tobytes()).decode()

    def run(self) -> None:
        dt = 0.02
        while self.running:
            twists = self.bridge.received_on("/myagv/cmd_vel")
            twist = twists[-1] if twists else {}
            vx = float(twist.get("linear", {}).get("x", 0.0)) * self.gain
            vy = float(twist.get("linear", {}).get("y", 0.0)) * self.gain
            wz = float(twist.get("angular", {}).get("z", 0.0)) * self.gain
            x, y, yaw = self.pose
            c, s = math.cos(yaw), math.sin(yaw)
            self.pose = [x + (vx * c - vy * s) * dt, y + (vx * s + vy * c) * dt, yaw + wz * dt]
            x, y, yaw = self.pose
            self.bridge.publish("/myagv/odom", {
                "header": {"frame_id": "myagv/odom", "stamp": {"secs": 0, "nsecs": 0}},
                "child_frame_id": "myagv/base_footprint",
                "pose": {"pose": {"position": {"x": x, "y": y, "z": 0.0},
                                  "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2),
                                                  "w": math.cos(yaw / 2)}}},
                "twist": {"twist": {"linear": {"x": vx, "y": vy, "z": 0.0},
                                    "angular": {"x": 0.0, "y": 0.0, "z": wz}}},
            })
            self.bridge.publish("/myagv/camera/image_raw/compressed",
                                {"format": "jpeg", "data": self.jpeg})
            time.sleep(dt)


def _zero(msg) -> bool:
    return all(abs(float(msg.get(p, {}).get(a, 0.0))) < 1e-9
               for p in ("linear", "angular") for a in ("x", "y", "z"))


@pytest.fixture
def base(bridge, monkeypatch):
    bridge.topics = dict(TOPICS)
    monkeypatch.setenv("PYTHONPATH", SRC)
    robot = _Base(bridge)
    robot.start()
    yield robot
    robot.running = False


def _stops_after_motion(bridge):
    msgs = bridge.received_on("/myagv/cmd_vel")
    last = max(i for i, m in enumerate(msgs) if not _zero(m))
    return msgs[last + 1:]


def test_smoke_passes_on_a_base_that_moves(bridge, base):
    from robot_console.smoke import execute

    checks, code = execute(f"ws://127.0.0.1:{bridge.port}", quiet=True)
    assert code == 0, [(c.name, c.detail) for c in checks]
    assert [c.name for c in checks] == ["connect", "odom", "camera", "forward", "back",
                                        "sideways", "turn"]
    time.sleep(0.3)
    stops = _stops_after_motion(bridge)
    assert len(stops) >= 3 and all(_zero(m) for m in stops)


def test_smoke_fails_a_base_that_barely_moves_and_still_stops_it(bridge, base):
    from robot_console.smoke import execute

    base.gain = 0.2   # 0.06 m per straight leg, under the 0.10 m bar
    checks, code = execute(f"ws://127.0.0.1:{bridge.port}", quiet=True)
    assert code == 1
    failed = {c.name for c in checks if not c.ok}
    assert {"forward", "back", "turn"} <= failed
    time.sleep(0.3)
    assert all(_zero(m) for m in _stops_after_motion(bridge))


def test_smoke_takes_url_not_host():
    from robot_console.smoke import main

    with pytest.raises(SystemExit):
        main(["--host", "127.0.0.1"])
