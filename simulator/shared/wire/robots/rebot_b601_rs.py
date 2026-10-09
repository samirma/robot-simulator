"""reBot Arm B601-RS (ROS 2 Jazzy): the wire of Seeed's authoritative boot,
`ros2 launch rebotarm_bringup bringup.launch.py model:=rs channel:=can0`.

Real, unchanged, from the pinned Seeed-Projects/reBotArmController_ROS2 revision (built
into the image's workspace /opt/rsim_ws, the SDK reBotArm_control_py at its pinned
revision in the workspace's third_party/ where the controller looks for it): the launch
file itself, /reBotArmController (rebotarmcontroller with the SDK's grouped control loop,
Pinocchio kinematics and gravity) and the stock /robot_state_publisher it starts, with the
robot_description the launch generates from the vendor URDF. Simulated: the hardware below
the SDK -- the CAN bus and the seven RobStride motors -- as a drop-in `motorbridge`
(`sim_libs/motorbridge.py`) put first on the launch's PYTHONPATH, so the vendor's
`from motorbridge import Controller, Mode, CallError` drives the simulated arm.

The boot's two `sudo ip link set can0 ...` steps configure the host's SocketCAN interface;
the simulation has no CAN interface, so they are not run.

The wrist camera is the approved community camera boot of Seeed's grasping guide
(`ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
depth_module.depth_profile:=640x360x30 depth_module.color_profile:=640x360x30`): the Intel
RealSense D405 as `/camera/camera` (`ros2_realsense.py`: the node's recorded streams, parameters
and static transforms, from the rendered model camera on the gripper), and the camera's mount
on `gripper_end` as the stock `tf2_ros` static_transform_publisher, `/camera_mount_tf`.
"""

from __future__ import annotations

from pathlib import Path

import common
from robots import _plan

LIBS = Path(__file__).resolve().parent / "sim_libs"
BOOT = "ros2 launch rebotarm_bringup bringup.launch.py model:=rs channel:=can0"


MOUNT_NODE = "/camera_mount_tf"


def mount_command(iface: dict) -> list:
    """`ros2 run tf2_ros static_transform_publisher` for the recorded mount transform (the tf
    row published by `/camera_mount_tf`), named as recorded."""
    row = next(r for r in common.served(iface.get("tf")) if r.get("publisher") == MOUNT_NODE)
    cmd = ["ros2", "run", "tf2_ros", "static_transform_publisher"]
    for flag, value in zip(("x", "y", "z"), row["xyz"]):
        cmd += [f"--{flag}", repr(float(value))]
    for flag, value in zip(("roll", "pitch", "yaw"), row["rpy"]):
        cmd += [f"--{flag}", repr(float(value))]
    return cmd + ["--frame-id", row["parent"], "--child-frame-id", row["child"],
                  "--ros-args", "-r", f"__node:={MOUNT_NODE.lstrip('/')}"]


def plan(robot, iface, describe):
    p = _plan.Plan()
    p.add("bringup", ["bash", "-c", f"PYTHONPATH={LIBS}:$PYTHONPATH exec {BOOT}"])
    p.add(MOUNT_NODE, mount_command(iface))
    p.add("/camera/camera", ["python3", "-u", str(_plan.WIRE / "ros2_realsense.py")])
    return p
