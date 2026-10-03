"""SO-101 (ROS 2 Jazzy): the approved community interface -- ros2_so_arm's
`controllers_bringup.launch.py` (hardware_type:=real) plus the usb_cam wrist camera.

Stock, unchanged: robot_state_publisher with the boot's robot_description, ros2_control's
ros2_control_node as /controller_manager with the recorded parameters, and the spawner
that loads and activates joint_state_broadcaster, joint_trajectory_controller and
gripper_controller (it exits once they are active, as on the robot; the wire is ready only
after it has exited with status 0, so the controllers' action servers accept goals). The
hardware plugin the boot names, feetech_ros2_driver/FeetechHardwareInterface, is the
simulator's stand-in of that name (wire/ros2_plugins/feetech_ros2_driver), which talks to
the simulated STS3215 bus (`so101_bus.py`) instead of a serial port. Simulated: /usb_cam,
the wrist camera.
"""

from __future__ import annotations

from pathlib import Path

import common
from robots import _plan
from robots._ros1_drivers import UsbCam
from robots.mycobot280 import rsp_params_file

WIRE = Path(__file__).resolve().parents[1]
CONTROLLER_NODES = ("/controller_manager", "/joint_trajectory_controller",
                    "/joint_state_broadcaster", "/gripper_controller")


def controllers_file(robot, iface) -> str:
    """The recorded parameters of controller_manager and its controllers, as one ROS 2
    parameter file (the bringup's ros2_controllers.yaml, as loaded)."""
    import yaml

    doc = {}
    for node in CONTROLLER_NODES:
        params = {}
        for row in common.params_of(iface, node):
            v = common.param_value(robot, row)
            if isinstance(v, list) and not v:
                continue  # an empty list is the node's own default; YAML cannot type it
            params[row["name"]] = v
        doc[node] = {"ros__parameters": params}
    path = "/tmp/so101_controllers.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh)
    return path


def plan(robot, iface, describe):
    p = _plan.Plan()
    _plan.build_overlay(p, [WIRE / "ros2_plugins" / "feetech_ros2_driver"], "jazzy")
    p.helpers.append(("so101-bus", ["python3", "-u", str(WIRE / "robots" / "so101_bus.py")]))
    rsp = rsp_params_file(robot, iface)
    p.add("/robot_state_publisher", ["ros2", "run", "robot_state_publisher",
                                     "robot_state_publisher", "--ros-args", "-r",
                                     "__node:=robot_state_publisher", "--params-file", rsp])
    p.add("/controller_manager", ["ros2", "run", "controller_manager", "ros2_control_node",
                                  "--ros-args", "-r", "__node:=controller_manager",
                                  "-r", "~/robot_description:=/robot_description",
                                  "--params-file", controllers_file(robot, iface)])
    p.add_oneshot("spawner", ["ros2", "run", "controller_manager", "spawner",
                              "joint_state_broadcaster", "joint_trajectory_controller",
                              "gripper_controller", "--controller-manager", "/controller_manager"])
    _plan.ros2_emulated(p, ["/usb_cam"])
    return p


class WristCamera(UsbCam):
    image_topic = "/image_raw"

    def on_service(self, name, req, res=None):
        if name == "/set_capture":
            self.capturing = bool(req.data)
            res.success = True
            res.message = ""
            return res
        return None


BEHAVIOURS = {"/usb_cam": WristCamera}
