"""AiNex (ROS 1 Noetic): the wire of `ainex_bringup/bringup.launch`.

Real, unchanged, from the pinned Hiwonder/ainex revision (built into the image at the
boot's own path /home/ubuntu/ros_ws): /ros_robot_controller (against the simulated board,
`sim_libs/ainex_board.py`, injected by `run_with_fakes.py`) and /ainex_controller -- the
vendor's gait engine and controller (walking_module.so, kinematics.so), walking with the
boot's own walking parameters, playing action groups, moving the head. Stock:
imu_calib (the vendor's copy), imu_complementary_filter, the camera's nodelet manager
with image_proc/rectify, web_video_server. Simulated: /camera (the USB head camera),
/joystick (an idle gamepad receiver), and the nodes that need the robot's own peripherals
or app stack (/sensor, /joystick_control, /color_detection, /face_detect, /app), which
present their recorded endpoints with default behaviour.
"""

from __future__ import annotations

from pathlib import Path

import common
from robots import _plan
from robots._ros1_drivers import UsbCam

HERE = Path(__file__).resolve().parent
WS = Path("/home/ubuntu/ros_ws")
KIN = WS / "src/ainex_driver/ainex_kinematics"
ACTIONS = Path("/home/ubuntu/software/ainex_controller/ActionGroups")


def plan(robot, iface, describe):
    import yaml

    p = _plan.Plan()
    p.overlays.append(str(WS / "devel/setup.bash"))
    ctl = KIN / "config/servo_controller.yaml"
    pose = KIN / "config/init_pose.yaml"
    if ctl.is_file():
        p.params["/ainex_controller/controllers"] = yaml.safe_load(ctl.read_text())["controllers"]
    # The init pose the boot applies (and every later return to it) is the `init_pose`
    # parameter served from ros.yml, whose arms are down at the sides. The action groups
    # are offsets from the pose in the vendor's file, so that file gets the same arms first.
    p.prepare.append(f"python3 {HERE / 'ainex_pose.py'} {pose}")
    p.prepare.append(f"python3 {HERE / 'ainex_actions.py'} {ctl} {pose} {ACTIONS}")
    fakes = HERE / "sim_libs" / "run_with_fakes.py"
    p.add("/ros_robot_controller", ["bash", "-c",
          f"exec python3 {fakes} ros_robot_controller.ros_robot_controller_sdk=ainex_board -- "
          "$(rospack find ros_robot_controller)/scripts/ros_robot_controller_node.py "
          "__name:=ros_robot_controller"])
    p.add("/ainex_controller", ["bash", "-c",
          "exec python3 $(rospack find ainex_kinematics)/scripts/ainex_controller.py "
          "__name:=ainex_controller"])
    p.add("/imu_calib", ["rosrun", "imu_calib", "apply_calib", "__name:=imu_calib",
                         "raw:=/ros_robot_controller/imu_raw", "corrected:=imu_corrected"])
    p.add("/imu_filter", ["rosrun", "imu_complementary_filter", "complementary_filter_node",
                          "__name:=imu_filter", "imu/data_raw:=imu_corrected", "imu/data:=imu",
                          "imu/mag:=/ros_robot_controller/mag"])
    p.add("/camera/manager", ["rosrun", "nodelet", "nodelet", "manager", "--no-bond",
                              "__name:=manager", "__ns:=/camera"])
    p.add("/camera/rectify_color", ["rosrun", "nodelet", "nodelet", "load", "image_proc/rectify",
                                    "manager", "--no-bond", "__name:=rectify_color",
                                    "__ns:=/camera", "camera:=camera",
                                    "image_mono:=/camera/image_raw",
                                    "camera_info:=/camera/camera_info",
                                    "image_rect:=/camera/image_rect_color"])
    p.add("/web_video_server", ["rosrun", "web_video_server", "web_video_server",
                                "__name:=web_video_server"])
    for node in ("/camera", "/joystick", "/sensor", "/joystick_control", "/color_detection",
                 "/face_detect", "/app"):
        _plan.ros1_emulated(p, node)
    return p


class HeadCamera(UsbCam):
    image_topic = "/camera/image_raw"


class Gamepad:
    """/joystick (joy_node with the gamepad receiver plugged in, the controller idle):
    /joy at autorepeat_rate 20 Hz with the neutral state, diagnostics at 1 Hz."""

    publishes = ("/joy", "/diagnostics")

    def __init__(self, node):
        self.node = node

    def start(self):
        import rospy
        from diagnostic_msgs.msg import DiagnosticStatus, KeyValue

        def joy(_):
            m = self.node.new("/joy")
            m.axes = [0.0] * 8
            m.buttons = [0] * 15
            self.node.publish("/joy", m)

        def diag(_):
            m = self.node.new("/diagnostics")
            st = DiagnosticStatus(level=0, name="joystick: Joystick Driver Status",
                                  message="OK", hardware_id="none")
            st.values = [KeyValue("topic", "/joy"), KeyValue("device", "/dev/input/js0"),
                         KeyValue("autorepeat rate (Hz)", "20.0")]
            m.status = [st]
            self.node.publish("/diagnostics", m)

        self._t1 = rospy.Timer(rospy.Duration(1.0 / 20.0), joy)
        self._t2 = rospy.Timer(rospy.Duration(1.0), diag)


BEHAVIOURS = {"/camera": HeadCamera, "/joystick": Gamepad}
