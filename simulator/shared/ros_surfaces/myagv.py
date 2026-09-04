"""The myAGV's ROS contract: `cmd_vel` in, `odom` out.

This is what `elephantrobotics/myagv_ros` presents, so one client drives either the
simulated or the real myAGV without changing a line:

    console -> robot   /cmd_vel                        geometry_msgs/Twist
    robot -> console   /odom                           nav_msgs/Odometry
    robot -> console   /camera/image_raw/compressed    sensor_msgs/CompressedImage
    robot -> console   /scan                           sensor_msgs/LaserScan

It lives in `shared/` rather than beside one engine's robot adapter because the topic
set *is* part of the robot definition, and every engine has to present the same one.
Two copies of this loop would be two chances for the engines to drift far enough apart
that a console could tell them apart -- which is the one thing the split forbids.

What it needs from an engine is small on purpose: a `base` object exposing a 4x4 `pose`
and a writable `ctrl` triple, which `mujoco_bridge.PlanarJointBase` builds from a raw
MuJoCo model, and MolmoSpaces' `HoloJointsRobotBaseGroup` already satisfies.

Nothing here is myAGV-specific beyond the topic names and the lidar mount: any holonomic
base with a camera is the whole of what this contract assumes, so a second one would
reuse this surface rather than grow its own.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def serve_ros(port: int, base, model, camera: str | None, camera_size, jpeg_quality: int,
              control_hz: float, watchdog_s: float, scan: dict | None = None,
              depth: dict | None = None, scene_option=None, host: str = "0.0.0.0"):
    """Present the robot on the myagv_ros topics and return a per-step callback.

    The base integrates the commanded `cmd_vel` here rather than in the client: that is
    what `cmd_vel` means, and it keeps the client identical for real hardware. See
    contracts/rosbridge_server.py for why the protocol is served in-process.

    Call the returned function with a `mujoco.MjData` each control period, and with
    `None` to shut the server down.
    """
    from contracts.rosbridge_server import (
        TOPIC_CAMERA,
        TOPIC_CAMERA_INFO,
        TOPIC_CMD_VEL,
        TOPIC_DEPTH,
        TOPIC_ODOM,
        TOPIC_SCAN,
        TYPE_TWIST,
        RosBridgeServer,
        odometry,
    )
    from mujoco_bridge import PlanarSetpoint, SensorStreams, SensorTopics

    server = RosBridgeServer(port=port)
    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "at": 0.0}

    def on_cmd_vel(msg: dict) -> None:
        linear = msg.get("linear") or {}
        angular = msg.get("angular") or {}
        command["vx"] = float(linear.get("x", 0.0))
        command["vy"] = float(linear.get("y", 0.0))
        command["wz"] = float(angular.get("z", 0.0))
        command["at"] = time.monotonic()

    server.on(TOPIC_CMD_VEL, on_cmd_vel, TYPE_TWIST)

    sensors = SensorStreams(
        server, model, camera, camera_size, jpeg_quality, scan, depth,
        SensorTopics(TOPIC_CAMERA, TOPIC_SCAN, TOPIC_DEPTH, TOPIC_CAMERA_INFO),
        scene_option=scene_option,
    )
    setpoint = PlanarSetpoint()

    server.start()
    print(
        f"ROS topics on ws://{host}:{port} "
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
