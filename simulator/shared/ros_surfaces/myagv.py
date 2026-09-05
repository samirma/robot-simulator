"""The myAGV's ROS contract: `cmd_vel` in, `odom` out.

This is what `elephantrobotics/myagv_ros` presents, so one client drives either the
simulated or the real myAGV without changing a line:

    console -> robot   /cmd_vel                        geometry_msgs/Twist
    robot -> console   /odom                           nav_msgs/Odometry
    robot -> console   /camera/image_raw/compressed    sensor_msgs/CompressedImage
    robot -> console   /scan                           sensor_msgs/LaserScan

Those names are the *bare* contract, and they stay bare here because they are the record
of what the vendor stack actually publishes. When several robots share one graph each one
gets a namespace and these become `/myagv/cmd_vel` and friends -- applied by the
`NamespacedBus` this surface is handed, never spelled out at a call site. See
`contracts/namespace.py`.

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


def attach_ros(bus, base, model, camera: str | None, camera_size, jpeg_quality: int,
               control_hz: float, watchdog_s: float, scan: dict | None = None,
               depth: dict | None = None, scene_option=None, camera_period: float = 0.0,
               world_reset=None):
    """Wire this robot onto an already-built bus and return a per-step callback.

    The base integrates the commanded `cmd_vel` here rather than in the client: that is
    what `cmd_vel` means, and it keeps the client identical for real hardware.

    Call the returned function with a `mujoco.MjData` each control period, and with
    `None` to close this robot's streams. It does **not** stop the server -- the fleet
    that owns the port does that, once, after every member has closed.
    """
    from contracts.rosbridge_server import (
        TOPIC_CAMERA,
        TOPIC_CAMERA_INFO,
        TOPIC_CMD_VEL,
        TOPIC_DEPTH,
        TOPIC_ODOM,
        TOPIC_SCAN,
        TYPE_ODOM,
        TYPE_TWIST,
        odometry,
    )
    from mujoco_bridge import PlanarSetpoint, SensorStreams, SensorTopics

    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "at": 0.0}

    def on_cmd_vel(msg: dict) -> None:
        linear = msg.get("linear") or {}
        angular = msg.get("angular") or {}
        command["vx"] = float(linear.get("x", 0.0))
        command["vy"] = float(linear.get("y", 0.0))
        command["wz"] = float(angular.get("z", 0.0))
        command["at"] = time.monotonic()

    bus.on(TOPIC_CMD_VEL, on_cmd_vel, TYPE_TWIST)

    sensors = SensorStreams(
        bus, model, camera, camera_size, jpeg_quality, scan, depth,
        SensorTopics(TOPIC_CAMERA, TOPIC_SCAN, TOPIC_DEPTH, TOPIC_CAMERA_INFO,
                     camera_frame=bus.frame("camera"), scan_frame=bus.frame("laser_frame")),
        scene_option=scene_option,
        camera_period=camera_period,
    )
    setpoint = PlanarSetpoint()
    odom_frame, base_frame = bus.frame("odom"), bus.frame("base_footprint")

    if world_reset is not None:
        # A world reset -- the arm's `/reset`, on a shared scene -- restores every joint
        # and every actuator target, this base's included. The integrated setpoint is the
        # one piece of state that survives it, so without this the base wakes up back at
        # spawn still holding the target it was driving toward and lunges for a pose it no
        # longer occupies. Reads as a physics glitch; is a latched controller.
        def _forget_latched_state() -> None:
            setpoint.reset()
            command.update(vx=0.0, vy=0.0, wz=0.0, at=0.0)

        world_reset.on_reset(_forget_latched_state)

    dt = 1.0 / control_hz

    def step(data):
        if data is None:
            sensors.close()
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

        seq = bus.next_seq()
        bus.publish(
            TOPIC_ODOM,
            odometry(seq, x, y, yaw, vx, vy, wz,
                     frame_id=odom_frame, child_frame_id=base_frame),
            TYPE_ODOM,
        )
        sensors.publish(data, seq, x, y, yaw)

    return step


def serve_ros(port: int, base, model, camera: str | None, camera_size, jpeg_quality: int,
              control_hz: float, watchdog_s: float, scan: dict | None = None,
              depth: dict | None = None, scene_option=None, host: str = "0.0.0.0",
              namespace: str = ""):
    """The single-robot path: own a server on `port`, put one myAGV on it, start it.

    Kept as a thin wrapper over `attach_ros` so the callers that only ever want one robot
    -- `run.sh view --robot myagv --ros-port 9090`, and the engine-side adapters in
    `molmospaces/robots/*/ros_surface.py` -- need no knowledge of fleets. `spawn_robot.py`
    builds a `RobotFleet` instead, because it may be asked for several robots at once.
    """
    from ros_surfaces import RobotFleet

    fleet = RobotFleet(port=port, host=host)
    fleet.attach(namespace, attach_ros, base=base, model=model, camera=camera,
                 camera_size=camera_size, jpeg_quality=jpeg_quality,
                 control_hz=control_hz, watchdog_s=watchdog_s, scan=scan, depth=depth,
                 scene_option=scene_option)
    fleet.start()
    print(f"myAGV on ws://{host}:{port} under namespace {namespace or '<bare>'}",
          file=sys.stderr)
    return fleet
