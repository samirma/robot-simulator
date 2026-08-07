"""The myAGV's ROS contract: `cmd_vel` in, `odom` out.

This is what `elephantrobotics/myagv_ros` presents, so one client drives either the
simulated or the real myAGV without changing a line:

    console -> robot   /cmd_vel                        geometry_msgs/Twist
    robot -> console   /odom                           nav_msgs/Odometry
    robot -> console   /camera/image_raw/compressed    sensor_msgs/CompressedImage
    robot -> console   /scan                           sensor_msgs/LaserScan

It lives beside the robot rather than in `tools/spawn_robot.py` because the topic set
*is* part of a robot definition, and the AiNex proves the point: its contract shares not
one topic with this one. What the two do share -- renderers, the depth stream, the
ray-cast lidar, and the velocity-to-setpoint integration -- is in `spawn_robot.py` as
`SensorStreams` and `PlanarSetpoint`.

`lekiwi` uses this surface too: it is a holonomic base with a camera, which is the whole
of what this contract assumes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[2]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))


def serve_ros(port: int, view, model, camera: str | None, camera_size, jpeg_quality: int,
              control_hz: float, watchdog_s: float, scan: dict | None = None,
              depth: dict | None = None, extra: dict | None = None):
    """Present the robot on the myagv_ros topics and return a per-step callback.

    The base integrates the commanded `cmd_vel` here rather than in the client: that is
    what `cmd_vel` means, and it keeps the client identical for real hardware. See
    bridge/rosbridge_server.py for why the protocol is served in-process.
    """
    from bridge.rosbridge_server import (
        TOPIC_CAMERA,
        TOPIC_CAMERA_INFO,
        TOPIC_CMD_VEL,
        TOPIC_DEPTH,
        TOPIC_ODOM,
        TOPIC_SCAN,
        RosBridgeServer,
        odometry,
    )
    from tools.spawn_robot import PlanarSetpoint, SensorStreams, SensorTopics

    if "base" not in view.move_group_ids():
        raise SystemExit(
            "the myagv ROS surface needs a robot with a mobile base; "
            f"this one has move groups {view.move_group_ids()}"
        )
    base = view.get_move_group("base")

    server = RosBridgeServer(port=port)
    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "at": 0.0}

    def on_cmd_vel(msg: dict) -> None:
        linear = msg.get("linear") or {}
        angular = msg.get("angular") or {}
        command["vx"] = float(linear.get("x", 0.0))
        command["vy"] = float(linear.get("y", 0.0))
        command["wz"] = float(angular.get("z", 0.0))
        command["at"] = time.monotonic()

    server.on(TOPIC_CMD_VEL, on_cmd_vel)

    sensors = SensorStreams(
        server, model, camera, camera_size, jpeg_quality, scan, depth,
        SensorTopics(TOPIC_CAMERA, TOPIC_SCAN, TOPIC_DEPTH, TOPIC_CAMERA_INFO),
    )
    setpoint = PlanarSetpoint()

    server.start()
    print(
        f"ROS topics on ws://0.0.0.0:{port} "
        f"(sub {TOPIC_CMD_VEL}; pub {', '.join([TOPIC_ODOM, *sensors.published])})",
        file=sys.stderr,
    )

    dt = 1.0 / control_hz

    def step(data):
        if data is None:
            sensors.close()
            server.stop()
            return

        # Watchdog. myAGVSub.cpp latches the last cmd_vel and keeps executing it, so a
        # client that disconnects mid-command would leave the robot driving; stopping is
        # the behaviour we want even though it is not what the firmware does.
        vx, vy, wz = command["vx"], command["vy"], command["wz"]
        if command["at"] and time.monotonic() - command["at"] > watchdog_s:
            vx = vy = wz = 0.0

        pose = base.pose
        x, y = float(pose[0, 3]), float(pose[1, 3])
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))

        base.ctrl = setpoint.step(x, y, yaw, vx, vy, wz, dt)

        seq = server.next_seq()
        server.publish(TOPIC_ODOM, odometry(seq, x, y, yaw, vx, vy, wz))
        sensors.publish(data, seq, x, y, yaw)

    return step
