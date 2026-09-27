"""The AiNex's ROS contract: `robots_specs/ainex/ros.yml`, transcribed.

Every topic, service and parameter the vendor boot launch presents, with its type, the
node that provides it, which way it flows and, for a periodic topic, its rate. The tables
at the bottom (`TOPICS`, `SERVICES`, `PARAMETERS`) are the whole interface; the named
constants above them are the entries the surface's code refers to by name.
`contracts/test_ainex_contract.py` holds the tables equal to the ROS file, entry for entry.

Bare vendor names, as everywhere in `ros_surfaces/`: the namespace is composed where a
name reaches the wire (`NamespacedBus`), and a node name here is the vendor's node, which
the bus composes the same way.

What the robot does **not** present is as much a part of this as what it does:

* no `/cmd_vel` and no `/odom` -- the AiNex is driven as a walking state machine
  (`/walking/command` plus a parameter block) and has no wheels to report;
* no `/joint_states`, no `/tf` and no `robot_description` -- the boot chain
  (`start_app_node.service` -> `bringup.launch`) runs no `robot_state_publisher` and no
  joint state publisher; servo positions are read through
  `/ros_robot_controller/bus_servo/get_position`;
* no per-joint `/<joint>_controller/command` beyond the head pair -- the other 22 exist
  only in the vendor's Gazebo bringup (`gazebo_sim: true`), where `ainex_controller`
  publishes them rather than subscribing;
* no lidar.

The frames are the drivers' own: `imu_link` for the board IMU (`imu_frame` parameter) and
`camera` for the head camera (`camera_frame_id`).
"""

from __future__ import annotations

from typing import NamedTuple

# --- node names -------------------------------------------------------------------------

NODE_CONTROLLER = "ainex_controller"
NODE_BOARD = "ros_robot_controller"
NODE_IMU_CALIB = "imu_calib"
NODE_IMU_FILTER = "imu_filter"
NODE_CAMERA = "camera"
NODE_RECTIFY = "camera/rectify_color"
NODE_JOY = "joystick"
NODE_JOYSTICK_CONTROL = "joystick_control"
NODE_SENSOR = "sensor"
NODE_COLOR_DETECTION = "color_detection"
NODE_FACE_DETECT = "face_detect"
NODE_APP = "app"

# --- walking / kinematics (ainex_controller) -------------------------------------------

TOPIC_SET_WALKING_PARAM = "/walking/set_param"
TOPIC_APP_WALKING_PARAM = "/app/set_walking_param"
TOPIC_APP_ACTION = "/app/set_action"
TOPIC_HEAD_PAN = "/head_pan_controller/command"
TOPIC_HEAD_TILT = "/head_tilt_controller/command"
TOPIC_IS_WALKING = "/walking/is_walking"

SRV_WALKING_COMMAND = "/walking/command"
SRV_GET_WALKING_PARAM = "/walking/get_param"
SRV_IS_WALKING = "/walking/is_walking"
SRV_INIT_POSE = "/walking/init_pose"

# --- servo / board driver (ros_robot_controller) ---------------------------------------

TOPIC_BUS_SERVO_SET = "/ros_robot_controller/bus_servo/set_position"
TOPIC_BUS_SERVO_SET_STATE = "/ros_robot_controller/bus_servo/set_state"
TOPIC_PWM_SERVO_SET_STATE = "/ros_robot_controller/pwm_servo/set_state"
TOPIC_SET_LED = "/ros_robot_controller/set_led"
TOPIC_SET_BUZZER = "/ros_robot_controller/set_buzzer"
TOPIC_SET_OLED = "/ros_robot_controller/set_oled"
TOPIC_SET_MOTOR = "/ros_robot_controller/set_motor"
TOPIC_SET_RGB = "/ros_robot_controller/set_rgb"
TOPIC_SET_MOTOR_DUTY = "/ros_robot_controller/set_motor_duty"
TOPIC_ENABLE_RECEPTION = "/ros_robot_controller/enable_reception"
TOPIC_IMU_RAW = "/ros_robot_controller/imu_raw"
TOPIC_MAG_RAW = "/ros_robot_controller/mag_raw"
TOPIC_MAG = "/ros_robot_controller/mag"
TOPIC_BOARD_JOY = "/ros_robot_controller/joy"
TOPIC_SBUS = "/ros_robot_controller/sbus"
TOPIC_BOARD_BUTTON = "/ros_robot_controller/button"
TOPIC_BATTERY = "/ros_robot_controller/battery"

SRV_BUS_SERVO_GET = "/ros_robot_controller/bus_servo/get_position"
SRV_BUS_SERVO_GET_STATE = "/ros_robot_controller/bus_servo/get_state"
SRV_PWM_SERVO_GET_STATE = "/ros_robot_controller/pwm_servo/get_state"

# --- IMU pipeline (imu_calib apply_calib -> imu_complementary_filter) -------------------

TOPIC_IMU_CORRECTED = "/imu_corrected"
TOPIC_IMU = "/imu"

# --- camera (usb_cam node `camera`, image_proc rectify nodelet) -------------------------

TOPIC_CAMERA_RAW = "/camera/image_raw"
TOPIC_CAMERA_INFO = "/camera/camera_info"
TOPIC_CAMERA = "/camera/image_raw/compressed"
TOPIC_CAMERA_RECT = "/camera/image_rect_color"
SRV_SET_CAMERA_INFO = "/camera/set_camera_info"

# --- joystick (joy_node `joystick`, joystick_control) -----------------------------------

TOPIC_JOY = "/joy"

# --- onboard sensor node (`sensor`) -----------------------------------------------------

TOPIC_BUTTON_STATE = "/sensor/button/get_button_state"
TOPIC_SENSOR_LED = "/sensor/led/set_led_state"
SRV_BUTTON_ENABLE = "/sensor/button/enable"

# --- vision nodes and the app (ainex_app/launch/start.launch) --------------------------

TOPIC_UPDATE_DETECT = "/color_detection/update_detect"
TOPIC_COLOR_IMAGE_RESULT = "/color_detection/image_result"
TOPIC_FACE_IMAGE_RESULT = "/face_detect/image_result"
TOPIC_PIXEL_COORDS = "/object/pixel_coords"
TOPIC_APP_IMAGE_RESULT = "/app/image_result"

SRV_APP_ENTER = "/app/enter"
SRV_APP_SET_RUNNING = "/app/set_running"
SRV_APP_SET_TARGET_COLOR = "/app/set_target_color"
SRV_APP_GET_TARGET_COLOR = "/app/get_target_color"
SRV_APP_SET_THRESHOLD = "/app/set_threshold"
SRV_APP_HEARTBEAT = "/app/heartbeat"
SRV_COLOR_ENTER = "/color_detection/enter"
SRV_COLOR_EXIT = "/color_detection/exit"
SRV_COLOR_START = "/color_detection/start"
SRV_COLOR_STOP = "/color_detection/stop"
SRV_COLOR_UPDATE_LAB = "/color_detection/update_lab"
SRV_FACE_ENTER = "/face_detect/enter"
SRV_FACE_EXIT = "/face_detect/exit"
SRV_FACE_START = "/face_detect/start"
SRV_FACE_STOP = "/face_detect/stop"

# --- type strings (ROS 1 single-slash, like the rest of this robot's stack) ------------

TYPE_WALKING_PARAM = "ainex_interfaces/WalkingParam"
TYPE_APP_WALKING_PARAM = "ainex_interfaces/AppWalkingParam"
TYPE_HEAD_STATE = "ainex_interfaces/HeadState"
TYPE_COLORS_DETECT = "ainex_interfaces/ColorsDetect"
TYPE_OBJECTS_INFO = "ainex_interfaces/ObjectsInfo"
TYPE_SET_BUS_SERVOS_POSITION = "ros_robot_controller/SetBusServosPosition"
TYPE_SET_BUS_SERVO_STATE = "ros_robot_controller/SetBusServoState"
TYPE_SET_PWM_SERVO_STATE = "ros_robot_controller/SetPWMServoState"
TYPE_LED_STATE = "ros_robot_controller/LedState"
TYPE_BUZZER_STATE = "ros_robot_controller/BuzzerState"
TYPE_OLED_STATE = "ros_robot_controller/OLEDState"
TYPE_MOTORS_STATE = "ros_robot_controller/MotorsState"
TYPE_RGBS_STATE = "ros_robot_controller/RGBsState"
TYPE_SBUS = "ros_robot_controller/Sbus"
TYPE_BUTTON_STATE = "ros_robot_controller/ButtonState"
TYPE_MAGNETOMETER = "sensor_msgs_ext/magnetometer"
TYPE_STRING = "std_msgs/String"
TYPE_BOOL = "std_msgs/Bool"
TYPE_UINT16 = "std_msgs/UInt16"
TYPE_IMU = "sensor_msgs/Imu"
TYPE_MAGNETIC_FIELD = "sensor_msgs/MagneticField"
TYPE_JOY = "sensor_msgs/Joy"
TYPE_IMAGE = "sensor_msgs/Image"
TYPE_CAMERA_INFO = "sensor_msgs/CameraInfo"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"

SRV_TYPE_SET_WALKING_COMMAND = "ainex_interfaces/SetWalkingCommand"
SRV_TYPE_GET_WALKING_PARAM = "ainex_interfaces/GetWalkingParam"
SRV_TYPE_GET_WALKING_STATE = "ainex_interfaces/GetWalkingState"
SRV_TYPE_SET_INT = "ainex_interfaces/SetInt"
SRV_TYPE_SET_POINT = "ainex_interfaces/SetPoint"
SRV_TYPE_SET_FLOAT = "ainex_interfaces/SetFloat"
SRV_TYPE_GET_BUS_SERVOS_POSITION = "ros_robot_controller/GetBusServosPosition"
SRV_TYPE_GET_BUS_SERVO_STATE = "ros_robot_controller/GetBusServoState"
SRV_TYPE_GET_PWM_SERVO_STATE = "ros_robot_controller/GetPWMServoState"
SRV_TYPE_EMPTY = "std_srvs/Empty"
SRV_TYPE_SET_BOOL = "std_srvs/SetBool"
SRV_TYPE_TRIGGER = "std_srvs/Trigger"
SRV_TYPE_SET_CAMERA_INFO = "sensor_msgs/SetCameraInfo"

# --- frames -----------------------------------------------------------------------------

#: `imu_link`: the board driver's `imu_frame`, which `apply_calib` and the complementary
#: filter carry through unchanged.
FRAME_IMU = "imu_link"
#: `camera`: usb_cam's `camera_frame_id` (the launch sets it to the camera name).
FRAME_CAMERA = "camera"

# --- camera geometry (usb_cam.launch) ---------------------------------------------------

CAMERA_SIZE = (640, 480)
#: usb_cam captures `yuyv` and publishes it converted to `rgb8`.
CAMERA_ENCODING = "rgb8"
#: What image_transport's compressed plugin writes into `format` for an rgb8 source.
CAMERA_COMPRESSED_FORMAT = "rgb8; jpeg compressed bgr8"

# --- the joints -------------------------------------------------------------------------

#: ros.yml `joints`: servo ids 1..24 in order, as `ainex_controller.py` declares them.
JOINT_NAMES: tuple[str, ...] = (
    "l_ank_roll", "r_ank_roll", "l_ank_pitch", "r_ank_pitch", "l_knee", "r_knee",
    "l_hip_pitch", "r_hip_pitch", "l_hip_roll", "r_hip_roll", "l_hip_yaw", "r_hip_yaw",
    "l_sho_pitch", "r_sho_pitch", "l_sho_roll", "r_sho_roll", "l_el_pitch", "r_el_pitch",
    "l_el_yaw", "r_el_yaw", "l_gripper", "r_gripper", "head_pan", "head_tilt",
)

#: The six strings `/walking/command` acts on. `start`/`stop`/`enable`/`disable` act only
#: while control is enabled; `enable_control`/`disable_control` set that; every call,
#: known or not, answers `result: true`.
WALKING_COMMANDS = (
    "enable", "disable", "start", "stop", "enable_control", "disable_control",
)

#: `/app/enter` modes, by the integer the service takes.
APP_MODES = {
    0: "idle", 1: "control", 2: "kick_ball", 3: "color_detect", 4: "visual_patrol",
    5: "color_track", 6: "face_detect", 7: "fall_rise",
}


# --- the interface ----------------------------------------------------------------------

EVENT = "event"


class Topic(NamedTuple):
    name: str
    type: str
    direction: str      # "in": the robot subscribes; "out": it publishes
    node: str
    rate_hz: float | str  # Hz for a periodic publication, EVENT otherwise


class Service(NamedTuple):
    name: str
    type: str
    node: str


class Parameter(NamedTuple):
    name: str
    type: str
    value: object


TOPICS: tuple[Topic, ...] = (
    Topic(TOPIC_SET_WALKING_PARAM, TYPE_WALKING_PARAM, "in", NODE_CONTROLLER, EVENT),
    Topic(TOPIC_APP_WALKING_PARAM, TYPE_APP_WALKING_PARAM, "in", NODE_CONTROLLER, EVENT),
    Topic(TOPIC_APP_ACTION, TYPE_STRING, "in", NODE_CONTROLLER, EVENT),
    Topic(TOPIC_HEAD_PAN, TYPE_HEAD_STATE, "in", NODE_CONTROLLER, EVENT),
    Topic(TOPIC_HEAD_TILT, TYPE_HEAD_STATE, "in", NODE_CONTROLLER, EVENT),
    Topic(TOPIC_IS_WALKING, TYPE_BOOL, "out", NODE_CONTROLLER, EVENT),
    Topic(TOPIC_BUS_SERVO_SET, TYPE_SET_BUS_SERVOS_POSITION, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_BUS_SERVO_SET_STATE, TYPE_SET_BUS_SERVO_STATE, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_PWM_SERVO_SET_STATE, TYPE_SET_PWM_SERVO_STATE, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_SET_LED, TYPE_LED_STATE, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_SET_BUZZER, TYPE_BUZZER_STATE, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_SET_OLED, TYPE_OLED_STATE, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_SET_MOTOR, TYPE_MOTORS_STATE, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_SET_RGB, TYPE_RGBS_STATE, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_SET_MOTOR_DUTY, TYPE_MOTORS_STATE, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_ENABLE_RECEPTION, TYPE_BOOL, "in", NODE_BOARD, EVENT),
    Topic(TOPIC_IMU_RAW, TYPE_IMU, "out", NODE_BOARD, 100),
    Topic(TOPIC_MAG_RAW, TYPE_MAGNETOMETER, "out", NODE_BOARD, 100),
    Topic(TOPIC_MAG, TYPE_MAGNETIC_FIELD, "out", NODE_BOARD, 100),
    Topic(TOPIC_BOARD_JOY, TYPE_JOY, "out", NODE_BOARD, EVENT),
    Topic(TOPIC_SBUS, TYPE_SBUS, "out", NODE_BOARD, EVENT),
    Topic(TOPIC_BOARD_BUTTON, TYPE_BUTTON_STATE, "out", NODE_BOARD, EVENT),
    Topic(TOPIC_BATTERY, TYPE_UINT16, "out", NODE_BOARD, EVENT),
    Topic(TOPIC_IMU_CORRECTED, TYPE_IMU, "out", NODE_IMU_CALIB, 100),
    Topic(TOPIC_IMU, TYPE_IMU, "out", NODE_IMU_FILTER, 100),
    Topic(TOPIC_CAMERA_RAW, TYPE_IMAGE, "out", NODE_CAMERA, 30),
    Topic(TOPIC_CAMERA_INFO, TYPE_CAMERA_INFO, "out", NODE_CAMERA, 30),
    Topic(TOPIC_CAMERA, TYPE_COMPRESSED_IMAGE, "out", NODE_CAMERA, 30),
    Topic(TOPIC_CAMERA_RECT, TYPE_IMAGE, "out", NODE_RECTIFY, 30),
    Topic(TOPIC_JOY, TYPE_JOY, "out", NODE_JOY, 20),
    Topic(TOPIC_JOY, TYPE_JOY, "in", NODE_JOYSTICK_CONTROL, EVENT),
    Topic(TOPIC_BUTTON_STATE, TYPE_BOOL, "out", NODE_SENSOR, 50),
    Topic(TOPIC_SENSOR_LED, TYPE_BOOL, "in", NODE_SENSOR, EVENT),
    Topic(TOPIC_UPDATE_DETECT, TYPE_COLORS_DETECT, "in", NODE_COLOR_DETECTION, EVENT),
    Topic(TOPIC_COLOR_IMAGE_RESULT, TYPE_IMAGE, "out", NODE_COLOR_DETECTION, EVENT),
    Topic(TOPIC_FACE_IMAGE_RESULT, TYPE_IMAGE, "out", NODE_FACE_DETECT, EVENT),
    Topic(TOPIC_PIXEL_COORDS, TYPE_OBJECTS_INFO, "out", NODE_COLOR_DETECTION, EVENT),
    Topic(TOPIC_APP_IMAGE_RESULT, TYPE_IMAGE, "out", NODE_APP, EVENT),
)

SERVICES: tuple[Service, ...] = (
    Service(SRV_WALKING_COMMAND, SRV_TYPE_SET_WALKING_COMMAND, NODE_CONTROLLER),
    Service(SRV_GET_WALKING_PARAM, SRV_TYPE_GET_WALKING_PARAM, NODE_CONTROLLER),
    Service(SRV_IS_WALKING, SRV_TYPE_GET_WALKING_STATE, NODE_CONTROLLER),
    Service(SRV_INIT_POSE, SRV_TYPE_EMPTY, NODE_CONTROLLER),
    Service(SRV_BUS_SERVO_GET, SRV_TYPE_GET_BUS_SERVOS_POSITION, NODE_BOARD),
    Service(SRV_BUS_SERVO_GET_STATE, SRV_TYPE_GET_BUS_SERVO_STATE, NODE_BOARD),
    Service(SRV_PWM_SERVO_GET_STATE, SRV_TYPE_GET_PWM_SERVO_STATE, NODE_BOARD),
    Service(SRV_BUTTON_ENABLE, SRV_TYPE_SET_BOOL, NODE_SENSOR),
    Service(SRV_APP_ENTER, SRV_TYPE_SET_INT, NODE_APP),
    Service(SRV_APP_SET_RUNNING, SRV_TYPE_SET_BOOL, NODE_APP),
    Service(SRV_APP_SET_TARGET_COLOR, SRV_TYPE_SET_POINT, NODE_APP),
    Service(SRV_APP_GET_TARGET_COLOR, SRV_TYPE_TRIGGER, NODE_APP),
    Service(SRV_APP_SET_THRESHOLD, SRV_TYPE_SET_FLOAT, NODE_APP),
    Service(SRV_APP_HEARTBEAT, SRV_TYPE_SET_BOOL, NODE_APP),
    Service(SRV_COLOR_ENTER, SRV_TYPE_EMPTY, NODE_COLOR_DETECTION),
    Service(SRV_COLOR_EXIT, SRV_TYPE_EMPTY, NODE_COLOR_DETECTION),
    Service(SRV_COLOR_START, SRV_TYPE_EMPTY, NODE_COLOR_DETECTION),
    Service(SRV_COLOR_STOP, SRV_TYPE_EMPTY, NODE_COLOR_DETECTION),
    Service(SRV_COLOR_UPDATE_LAB, SRV_TYPE_EMPTY, NODE_COLOR_DETECTION),
    Service(SRV_FACE_ENTER, SRV_TYPE_EMPTY, NODE_FACE_DETECT),
    Service(SRV_FACE_EXIT, SRV_TYPE_EMPTY, NODE_FACE_DETECT),
    Service(SRV_FACE_START, SRV_TYPE_EMPTY, NODE_FACE_DETECT),
    Service(SRV_FACE_STOP, SRV_TYPE_EMPTY, NODE_FACE_DETECT),
    Service(SRV_SET_CAMERA_INFO, SRV_TYPE_SET_CAMERA_INFO, NODE_CAMERA),
)

#: Periodic publications, topic -> Hz. Each is published on its own clock at this rate,
#: whatever the engine's control rate is: these are the drivers' loop rates
#: (`ros_robot_controller` freq 100, `sensor` freq 50, joy_node autorepeat 20, usb_cam's
#: default 30 fps), and the IMU pipeline downstream follows the board.
RATES_HZ: dict[str, float] = {
    t.name: float(t.rate_hz) for t in TOPICS if t.direction == "out" and t.rate_hz != EVENT
}

#: `mag_calib.yaml`'s 4x4 magnetometer calibration (row-major), loaded into the board
#: driver's `calibration` parameter.
MAG_CALIBRATION: tuple[float, ...] = (
    0.0000011001, -0.0000000209, -0.0000000051, 0.0000663759,
    -0.0000000209, 0.0000009472, -0.0000000032, -0.0000612174,
    -0.0000000051, -0.0000000032, 0.0000011188, 0.0001825604,
    0.0, 0.0, 0.0, 1.0,
)

#: `color_track_pid.yaml`, loaded by `start.launch` at the root and as `/app/color_track`.
COLOR_TRACK_PID = {"pid1_p": 0.13, "pid1_i": 0.0, "pid1_d": 0.003,
                   "pid2_p": 0.13, "pid2_i": 0.0, "pid2_d": 0.003}


def _servos():
    """`servos.py`, whether this module was imported in its package or loaded by path.

    The console's parity test loads this file on its own, outside any package, because
    that project must run with no simulator checkout; `servos.py` is stdlib-only, so it is
    loaded beside it the same way.
    """
    try:
        from . import servos  # noqa: PLC0415
    except ImportError:
        import importlib.util  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location(
            "_ainex_topics_servos", Path(__file__).with_name("servos.py"))
        servos = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(servos)
    return servos


def _init_pose() -> dict:
    """`init_pose.yaml`, all 24 joints in radians."""
    init = _servos().INIT_POSE
    return {j: init[j] for j in JOINT_NAMES}


def _controllers() -> dict:
    """`servo_controller.yaml`: one JointPositionController per joint, min > max = flipped."""
    table = _servos().SERVOS
    out = {}
    for joint in JOINT_NAMES:
        sid, init, flipped = table[joint]
        out[f"{joint}_controller"] = {
            "type": "JointPositionController", "joint_name": joint,
            "joint_speed": 1.0, "port_id": 1,
            "servo": {"id": sid, "init": init,
                      "min": 1000 if flipped else 0, "max": 0 if flipped else 1000},
        }
    return out


PARAMETERS: tuple[Parameter, ...] = (
    Parameter("/ainex_controller/init_pose", "dict", _init_pose()),
    Parameter("/ainex_controller/controllers", "dict", _controllers()),
    Parameter("/ainex_controller/gazebo_sim", "bool", False),
    Parameter("/init_pose/init_finish", "bool", False),
    Parameter("/ros_robot_controller/freq", "int", 100),
    Parameter("/ros_robot_controller/imu_frame", "string", FRAME_IMU),
    Parameter("/ros_robot_controller/calibration", "double[16]", list(MAG_CALIBRATION)),
    Parameter("/ros_robot_controller/init_finish", "bool", True),
    Parameter("/camera/camera_name", "string", "camera"),
    Parameter("/camera/image_topic", "string", "image_raw"),
    Parameter("/camera/video_device", "string", "/dev/usb_cam"),
    Parameter("/camera/image_width", "int", CAMERA_SIZE[0]),
    Parameter("/camera/image_height", "int", CAMERA_SIZE[1]),
    Parameter("/camera/pixel_format", "string", "yuyv"),
    Parameter("/camera/camera_frame_id", "string", FRAME_CAMERA),
    Parameter("/camera/io_method", "string", "mmap"),
    Parameter("/camera/camera_info_url", "string",
              "file:///home/ubuntu/.ros/camera_info/head_camera.yaml"),
    Parameter("/camera/rectify_color/queue_size", "int", 10),
    Parameter("/joystick/dev", "string", "/dev/input/js0"),
    Parameter("/joystick/autorepeat_rate", "double", 20.0),
    Parameter("/joystick/coalesce_interval", "double", 0.05),
    Parameter("/imu_calib/calib_file", "string",
              "$(find ainex_calibration)/config/imu_calib.yaml"),
    Parameter("/imu_filter/use_mag", "bool", True),
    Parameter("/imu_filter/gain_acc", "double", 0.2),
    Parameter("/imu_filter/bias_alpha", "double", 0.2),
    Parameter("/imu_filter/do_bias_estimation", "bool", True),
    Parameter("/imu_filter/do_adaptive_gain", "bool", True),
    Parameter("/color_track", "dict", dict(COLOR_TRACK_PID)),
    Parameter("/app/color_track", "dict", dict(COLOR_TRACK_PID)),
    Parameter("/color_detection/debug", "bool", False),
    Parameter("/color_detection/enable_display", "bool", False),
    Parameter("/color_detection/enable_roi", "bool", True),
    Parameter("/face_detect/enable_display", "bool", False),
    Parameter("/face_detect/confidence", "double", 0.5),
)
