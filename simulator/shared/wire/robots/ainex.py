"""AiNex (ROS 1 Noetic): the wire of `ainex_bringup/bringup.launch`.

Real, unchanged, from the pinned Hiwonder/ainex revision (built into the image at the
boot's own path /home/ubuntu/ros_ws): /ros_robot_controller (against the simulated board,
`sim_libs/ainex_board.py`, injected by `run_with_fakes.py`) and /ainex_controller -- the
vendor's gait engine and controller (walking_module.so, kinematics.so), walking with the
boot's own walking parameters, playing action groups, moving the head. Stock:
imu_calib (the vendor's copy), imu_complementary_filter, the camera's nodelet manager
with image_proc/rectify, web_video_server. Simulated: /camera (the USB head camera),
/joystick (an idle gamepad receiver), and the nodes that need the robot's own peripherals
or app stack, with their recorded endpoints and behaviour: /sensor (the user button),
/color_detection and /face_detect, /app and /joystick_control (its endpoints; the gamepad
is idle).

Wire adaptations (the simulator's, not robot data; the record holds what the vendor nodes
do):

* /color_detection and /face_detect publish the undrawn camera frame on
  `<node>/image_result` (rgb8, header.frame_id the node's name, stamped when published as
  the vendor's cv2_image2ros does) between `<node>/enter` and `<node>/exit`, and never
  publish /object/pixel_coords: detection needs the LAB thresholds and models no pinned
  source holds, so nothing is ever detected.
* /app stays in its boot state `idle`: /app/set_running true is refused; false stops the
  gait (/walking/command `stop`, as GaitManager.stop does) and answers success false.
* Action groups: no .d6a file is in any pinned source, so the vendor's groups (left_shot,
  greet, ...) cannot be played; those names answer as the vendor controller answers an
  unknown name. The simulator writes its own groups `wave`, `raise_hands` and `nod`
  (`ainex_actions.py`) -- SIMULATOR ESTIMATES, not vendor data.
"""

from __future__ import annotations

import threading
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
    if ctl.is_file():
        p.params["/ainex_controller/controllers"] = yaml.safe_load(ctl.read_text())["controllers"]
    # the record's estimated action groups, as .d6a files where the vendor player reads them
    # (offsets from the recorded init pose, the init_pose parameter served from ros.yml)
    p.prepare.append(f"python3 {HERE / 'ainex_actions.py'} {ctl} {ACTIONS}")
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


class Sensor:
    """/sensor (sensor_node.py): /sensor/button/get_button_state every cycle of its loop
    while the button is enabled (it is at start), the button released -- the pin reads 1
    through its pull-up (ainex_sdk/button.py), so True; /sensor/button/enable switches it
    and answers (True, 'set_button_enable') as the node does."""

    TOPIC = "/sensor/button/get_button_state"
    publishes = (TOPIC,)

    def __init__(self, node):
        self.node = node
        self.enabled = True

    def start(self):
        import rospy

        def fire(_):
            if self.enabled:
                m = self.node.new(self.TOPIC)
                m.data = True
                self.node.publish(self.TOPIC, m)

        rate = common.rate_of(self.node.rows[self.TOPIC])
        self._timer = rospy.Timer(rospy.Duration(1.0 / rate), fire)

    def on_service(self, name, req):
        if name == "/sensor/button/enable":
            from std_srvs.srv import SetBoolResponse

            self.enabled = bool(req.data)
            return SetBoolResponse(success=True, message="set_button_enable")
        return None


class Detector:
    """/color_detection and /face_detect (color_detection_node.py, face_detect_node.py):
    after <node>/enter the node subscribes to the head camera (the /camera parameter's
    camera_name/image_topic) and publishes each frame on <node>/image_result, labelled with
    the node's name and stamped with its publish time (common.cv2_image2ros, as recorded),
    until <node>/exit. With detection started the vendor node draws what it finds; the LAB
    thresholds and models it needs are in no pinned source, so the frame goes out undrawn and
    nothing is reported as detected (no /object/pixel_coords)."""

    def __init__(self, node):
        self.node = node
        self.topic = node.name + "/image_result"
        self.publishes = (self.topic,)
        self.sub = None
        self.lock = threading.Lock()

    def on_service(self, name, req):
        import rospy
        from sensor_msgs.msg import Image
        from std_srvs.srv import EmptyResponse

        with self.lock:
            if name == self.node.name + "/enter" and self.sub is None:
                cam = rospy.get_param("/camera")
                self.sub = rospy.Subscriber(f"/{cam['camera_name']}/{cam['image_topic']}",
                                            Image, self._frame, queue_size=1)
            elif name == self.node.name + "/exit" and self.sub is not None:
                self.sub.unregister()
                self.sub = None
        return EmptyResponse()

    def _frame(self, msg):
        out = self.node.new(self.topic)       # stamped now: the recorded publish time
        out.header.frame_id = self.node.name.lstrip("/")
        out.height, out.width, out.encoding = msg.height, msg.width, "rgb8"
        out.is_bigendian, out.step, out.data = 0, msg.width * 3, msg.data
        self.node.publish(self.topic, out)


class App:
    """/app (app_node.py) in its boot state `idle`: /app/set_running false stops the gait
    as GaitManager.stop does (/walking/command `stop`) and answers success = the request's
    data, as the node does; set_running true is refused in `idle` (success false)."""

    def __init__(self, node):
        self.node = node

    def on_service(self, name, req):
        if name != "/app/set_running":
            return None
        import rospy
        from ainex_interfaces.srv import SetWalkingCommand
        from std_srvs.srv import SetBoolResponse

        if not req.data:
            rospy.ServiceProxy("/walking/command", SetWalkingCommand)("stop")
        return SetBoolResponse(success=False)


BEHAVIOURS = {"/camera": HeadCamera, "/joystick": Gamepad, "/sensor": Sensor,
              "/color_detection": Detector, "/face_detect": Detector, "/app": App}
