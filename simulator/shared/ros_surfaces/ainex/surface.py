"""The AiNex's ROS surface: exactly the interface `robots_specs/ainex/ros.yml` records.

Every name in `topics.TOPICS`, `topics.SERVICES` and `topics.PARAMETERS` is registered
here under the node that provides it on the robot, and nothing else is: no `/joint_states`,
no `/tf`, no `robot_description`, no lidar -- the boot chain presents none of them. See
`topics.py` for the whole table and what is deliberately absent.

The behaviour is `ainex_controller.py`'s, and the parts that surprise are the vendor's:

* `/walking/command` answers `result: true` to everything, and `start`/`stop`/`enable`/
  `disable` do nothing unless control is enabled (`init_pose_finish`), which only
  `enable_control` sets -- and `/app/set_action` and `/walking/init_pose` clear while they
  run and set again when they finish. `disable_control` is how the app freezes the gait.
* `stop` finishes the step cycle in progress and the call returns once the gait has halted;
  `disable` does the same and then blocks gait output until `enable` or `start`.
* `period_times` N walks N full cycles and then stops by itself.
* `/walking/is_walking` is published on transitions only.
* There is **no watchdog**. A client that disconnects mid-walk leaves the robot walking,
  exactly as the real gait engine does; the official way to stop it is `/walking/command`
  `stop` (see `stop_command` in the ROS file).

Locomotion is the planar base plus the animated gait described in `gait.py`: the walking
parameters are turned into a body-frame velocity by `gait.planar_velocity` and integrated
by `PlanarSetpoint`, while `gait.leg_joint_targets` drives the legs at a phase matched to
the distance covered.

Threading, which is the invariant that keeps a service call from corrupting the
simulation: subscribers and service handlers run on websocket reader threads and write
only to the `_Controller` below; the periodic streams run on their own clock thread and
read only snapshots (`streams.py`); `step(data)` is the only thing that touches MjData, on
the simulation thread.
"""

from __future__ import annotations

import base64
import math
import sys
import threading
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[2]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from ros_surfaces.ainex import gait, servos, streams, topics  # noqa: E402
from ros_surfaces.ainex.actions import (  # noqa: E402
    BASE_PITCH, ActionPlayer, load_action_dir, rest_pose,
)
from ros_surfaces.ainex.ground import GroundFollow  # noqa: E402
from ros_surfaces.ainex.nodes import App, JoystickControl, VisionNode  # noqa: E402

#: How long a blocking call waits for the simulation to act on it before giving up. The
#: vendor's calls block without limit; a simulation that is not stepping would then hang
#: the caller for ever, which helps no one.
BLOCK_TIMEOUT_S = 5.0

#: `walking_param.yaml`, the gait parameters `ainex_controller` starts with, in the
#: message's own units (ms, degrees, radians for the arm swing).
DEFAULT_WALKING_PARAM = {
    "init_x_offset": 0.0, "init_y_offset": -0.005, "init_z_offset": 0.025,
    "init_roll_offset": 0.0, "init_pitch_offset": 0.0, "init_yaw_offset": 0.0,
    "period_time": 400.0, "dsp_ratio": 0.2, "step_fb_ratio": 0.028, "period_times": 0,
    "x_move_amplitude": 0.0, "y_move_amplitude": 0.0, "z_move_amplitude": 0.02,
    "angle_move_amplitude": 0.0, "move_aim_on": False, "arm_swing_gain": 0.5,
    "y_swap_amplitude": 0.02, "z_swap_amplitude": 0.006, "pelvis_offset": 5.0,
    "hip_pitch_offset": 15.0,
    "balance_enable": False, "balance_hip_roll_gain": 0.0, "balance_knee_gain": 0.0,
    "balance_ankle_roll_gain": 0.0, "balance_ankle_pitch_gain": 0.0,
}

#: `ainex_controller.py`'s clamps on `/walking/set_param`, per field.
WALKING_PARAM_RANGES = {
    "init_z_offset": (0.015, 0.06),
    "x_move_amplitude": (-0.05, 0.05),
    "y_move_amplitude": (-0.05, 0.05),
    "z_move_amplitude": (0.0, 0.05),
    "angle_move_amplitude": (-10.0, 10.0),
    "y_swap_amplitude": (0.0, 0.05),
    "arm_swing_gain": (0.0, math.radians(60)),
}


def _gait_of(wire: dict) -> gait.WalkingParam:
    """The wire block in `gait`'s units: seconds, radians. Balance gains have no effect on
    a torso riding position-controlled planar joints, so they are held and not used."""
    return gait.WalkingParam(
        period_time=float(wire["period_time"]) / 1000.0,
        dsp_ratio=float(wire["dsp_ratio"]),
        x_amplitude=float(wire["x_move_amplitude"]),
        y_amplitude=float(wire["y_move_amplitude"]),
        angle_amplitude=math.radians(float(wire["angle_move_amplitude"])),
        body_height=float(wire["init_z_offset"]),
        step_height=float(wire["z_move_amplitude"]),
        y_swap=float(wire["y_swap_amplitude"]),
        z_swap=float(wire["z_swap_amplitude"]),
        arm_swing_gain=float(wire["arm_swing_gain"]),
        hip_pitch_offset=math.radians(float(wire["hip_pitch_offset"])),
        period_times=int(wire["period_times"]),
    ).clamped()


class _Controller:
    """`ainex_controller`'s state machine, written by handlers and run by `step`.

    Guarded by one condition variable. Deliberately small: no MjData, nothing whose
    lifetime the simulation owns.
    """

    def __init__(self) -> None:
        self.cond = threading.Condition()
        # The vendor's flags, under the vendor's names where it has them.
        self.walking_enable = True
        self.init_pose_finish = True     # "control enabled"
        self.running = False             # walking_module started and not yet finished
        self.stop_pending = False        # finish the current cycle, then halt
        self.moving = False              # `not self.stop`: the gait is moving joints
        self.wire = dict(DEFAULT_WALKING_PARAM)
        self.param = _gait_of(self.wire)
        self.count_step = 0
        # Requests the simulation thread acts on.
        self.pending_action: str | None = None
        self.init_requests = 0
        self.inits_done = 0
        self.servo_writes: dict[str, float] = {}

    # -- /walking/command ---------------------------------------------------------

    def command(self, command: str) -> bool:
        wait = False
        with self.cond:
            if self.init_pose_finish:
                if command == "start":
                    self.walking_enable = True
                    self.running = True
                    self.stop_pending = False
                elif command in ("stop", "disable"):
                    if self.running:
                        self.stop_pending = True
                    wait = True
                elif command == "enable":
                    self.walking_enable = True
            if command == "enable_control":
                self.init_pose_finish = True
            elif command == "disable_control":
                self.init_pose_finish = False
        if wait:
            self.wait_halted()
            if command == "disable":
                with self.cond:
                    self.walking_enable = False
        return True

    def wait_halted(self, timeout: float = BLOCK_TIMEOUT_S) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: not self.running and not self.moving,
                                      timeout=timeout)

    # -- parameters ---------------------------------------------------------------

    def set_walking_param(self, msg: dict) -> None:
        """`set_walking_param_callback`: every field from the message, clamped.

        A field the client left out arrives as the message default -- zero, or false --
        because rosbridge fills an incomplete message that way before the node sees it.
        The balance block is not read by the controller at all.
        """
        with self.cond:
            for key, default in DEFAULT_WALKING_PARAM.items():
                if key.startswith("balance_"):
                    continue
                value = msg.get(key, False if isinstance(default, bool) else 0)
                if key in WALKING_PARAM_RANGES:
                    lo, hi = WALKING_PARAM_RANGES[key]
                    value = min(max(float(value), lo), hi)
                self.wire[key] = value
            self.param = _gait_of(self.wire)

    def set_app_walking_param(self, msg: dict) -> None:
        param = gait.from_app_params(
            int(msg.get("speed", gait.APP_DEFAULT_SPEED)), float(msg.get("height", 0.025)),
            float(msg.get("x", 0.0)), float(msg.get("y", 0.0)), float(msg.get("angle", 0.0)),
        )
        with self.cond:
            self.wire.update(
                period_times=0, init_x_offset=0.0, init_z_offset=param.body_height,
                init_roll_offset=0.0, init_pitch_offset=0.0, init_yaw_offset=0.0,
                hip_pitch_offset=15.0, z_move_amplitude=param.step_height,
                pelvis_offset=5.0, move_aim_on=False, arm_swing_gain=0.5,
                period_time=param.period_time * 1000.0, dsp_ratio=param.dsp_ratio,
                x_move_amplitude=param.x_amplitude, y_move_amplitude=param.y_amplitude,
                angle_move_amplitude=math.degrees(param.angle_amplitude),
                y_swap_amplitude=param.y_swap, z_swap_amplitude=param.z_swap,
            )
            self.param = _gait_of(self.wire)

    def walking_param(self) -> dict:
        """`/walking/get_param`'s answer: the current block, `period_times` as 0."""
        with self.cond:
            wire = dict(self.wire)
        return {**wire, "period_times": 0}

    # -- init pose and action groups ----------------------------------------------

    def request_init_pose(self, block: bool = True) -> None:
        with self.cond:
            self.init_requests += 1
            ticket = self.init_requests
            if self.running:
                self.stop_pending = True
            if block:
                self.cond.wait_for(lambda: self.inits_done >= ticket, timeout=BLOCK_TIMEOUT_S)

    def request_action(self, name: str) -> None:
        with self.cond:
            self.pending_action = name
            if self.running:
                self.stop_pending = True

    def is_walking(self) -> bool:
        with self.cond:
            return self.moving

    def write_servos(self, writes: dict[str, float]) -> None:
        with self.cond:
            self.servo_writes.update(writes)


def attach_ros(bus, base, model, prefix: str, camera: str | None, jpeg_quality: int = 80,
               control_hz: float = 20.0, extra: dict | None = None, scene_option=None,
               world_reset=None):
    """Wire the AiNex onto an already-built bus and return a per-step callback.

    **What an engine must supply**, and nothing more:

    * `base`  -- `.pose`, a 4x4 whose `[0:2, 3]` is x/y and whose `[1,0]`,`[0,0]` give
      yaw, and a writable `.ctrl` taking `(x, y, yaw)`. MolmoSpaces' holonomic base group
      and `mujoco_bridge.PlanarJointBase` both satisfy it already.
    * `model` -- the compiled `mujoco.MjModel` for the whole scene.
    * `prefix` -- this robot's MJCF name prefix, such that every name in `servos.SERVOS`
      has both a joint and a position actuator under it.
    * `camera` -- the head camera's MJCF name, or None for no camera.

    Every rate is this robot's own, from `topics.RATES_HZ`; `control_hz` is only the
    controller's tick, which the gait integrates over. The returned step carries
    `rate_hz`, the head camera's 30 Hz: the fleet calls it that often so a new frame is
    rendered for every frame the camera publishes, and the controller ticks inside it on
    a clock of its own at `control_hz`.
    """
    import ainex_model
    from mujoco_bridge import PlanarSetpoint

    actions = load_action_dir(Path(extra["action_dir"]) if (extra or {}).get("action_dir") else None)
    print(f"action groups: {', '.join(sorted(actions)) or '(none)'}", file=sys.stderr)

    namespace = prefix or ""
    leg_geometry = _leg_geometry(model, namespace)
    joint_ids = {n: model.joint(f"{namespace}{n}").id for n in servos.SERVOS}
    actuator_ids = {n: model.actuator(f"{namespace}{n}").id for n in servos.SERVOS}
    qpos_adr = {n: model.jnt_qposadr[joint_ids[n]] for n in servos.SERVOS}
    # The torso's lean is a 25th channel beside the servos: it is what the vendor's hip
    # chain does to the body when it bends to pick something up, and with the base riding
    # the torso it has to be a joint of its own. Action groups author it. It is not a
    # servo, so no servo read-back reports it.
    actuator_ids[BASE_PITCH] = model.actuator(f"{namespace}base_pitch_act").id
    qpos_adr[BASE_PITCH] = model.jnt_qposadr[model.joint(f"{namespace}base_pitch").id]
    pitch_dof = model.jnt_dofadr[model.joint(f"{namespace}base_pitch").id]

    ctl = _Controller()
    # The MjData the step loop owns, for a service answered on a websocket thread. Read
    # only, one float per servo; a torn read here is a count one tick stale, not a crash.
    live: list = [None]
    by_id = {sid: name for name, (sid, _, _) in servos.SERVOS.items()}

    def node(n: str) -> dict:
        return {"node": n}

    def ignored(_msg: dict) -> None:
        """Board I/O with nothing in the simulation to drive: LEDs, buzzer, OLED, DC
        motors the AiNex does not have, PWM servos it does not carry, the serial link."""

    # ---------------------------------------------------------------- ainex_controller

    def on_head(joint: str):
        def handler(msg: dict) -> None:
            # HeadState {position, duration}: the vendor sends it to the servo as a
            # timed move; the simulated servo takes its own time to travel.
            ctl.write_servos({joint: servos.clamp(joint, float(msg.get("position", 0.0)))})
        return handler

    bus.on(topics.TOPIC_SET_WALKING_PARAM, ctl.set_walking_param, topics.TYPE_WALKING_PARAM,
           **node(topics.NODE_CONTROLLER))
    bus.on(topics.TOPIC_APP_WALKING_PARAM, ctl.set_app_walking_param,
           topics.TYPE_APP_WALKING_PARAM, **node(topics.NODE_CONTROLLER))
    bus.on(topics.TOPIC_APP_ACTION, lambda m: ctl.request_action(str(m.get("data", ""))),
           topics.TYPE_STRING, **node(topics.NODE_CONTROLLER))
    bus.on(topics.TOPIC_HEAD_PAN, on_head("head_pan"), topics.TYPE_HEAD_STATE,
           **node(topics.NODE_CONTROLLER))
    bus.on(topics.TOPIC_HEAD_TILT, on_head("head_tilt"), topics.TYPE_HEAD_STATE,
           **node(topics.NODE_CONTROLLER))
    bus.advertise(topics.TOPIC_IS_WALKING, topics.TYPE_BOOL, **node(topics.NODE_CONTROLLER))

    bus.service(topics.SRV_WALKING_COMMAND,
                lambda a: {"result": ctl.command(str(a.get("command", "")))},
                topics.SRV_TYPE_SET_WALKING_COMMAND, **node(topics.NODE_CONTROLLER))
    bus.service(topics.SRV_GET_WALKING_PARAM, lambda a: {"parameters": ctl.walking_param()},
                topics.SRV_TYPE_GET_WALKING_PARAM, **node(topics.NODE_CONTROLLER))
    bus.service(topics.SRV_IS_WALKING,
                lambda a: {"state": ctl.is_walking(), "message": "is_walking"},
                topics.SRV_TYPE_GET_WALKING_STATE, **node(topics.NODE_CONTROLLER))

    def init_pose(_args: dict) -> dict:
        ctl.request_init_pose(block=True)
        return {}

    bus.service(topics.SRV_INIT_POSE, init_pose, topics.SRV_TYPE_EMPTY,
                **node(topics.NODE_CONTROLLER))

    # ---------------------------------------------------------------- ros_robot_controller

    def on_bus_servo_set(msg: dict) -> None:
        """`SetBusServosPosition`: raw counts by servo id, through the vendor's own
        count<->radian table, so a count means here what it means on the robot."""
        writes = {}
        for entry in msg.get("position") or []:
            name = by_id.get(int(entry.get("id", -1)))
            if name is not None:
                writes[name] = servos.clamp(
                    name, servos.count_to_angle(name, float(entry.get("position", 500))))
        ctl.write_servos(writes)

    def on_bus_servo_state(msg: dict) -> None:
        """`SetBusServoState`: each field is `[flag, value...]` for servo `present_id[1]`.
        A position moves it and `stop` holds it where it is; id, offset, limit, torque and
        save-offset writes configure servo firmware the simulation does not have."""
        writes = {}
        data = live[0]
        for state in msg.get("state") or []:
            present = list(state.get("present_id") or [])
            if len(present) < 2 or not present[0]:
                continue
            name = by_id.get(int(present[1]))
            if name is None:
                continue
            position = list(state.get("position") or [])
            if len(position) >= 2 and position[0]:
                writes[name] = servos.clamp(name, servos.count_to_angle(name, float(position[1])))
            stop = list(state.get("stop") or [])
            if stop and stop[0] and data is not None:
                writes[name] = float(data.qpos[qpos_adr[name]])
        ctl.write_servos(writes)

    board_in = (
        (topics.TOPIC_BUS_SERVO_SET, on_bus_servo_set, topics.TYPE_SET_BUS_SERVOS_POSITION),
        (topics.TOPIC_BUS_SERVO_SET_STATE, on_bus_servo_state, topics.TYPE_SET_BUS_SERVO_STATE),
        (topics.TOPIC_PWM_SERVO_SET_STATE, ignored, topics.TYPE_SET_PWM_SERVO_STATE),
        (topics.TOPIC_SET_LED, ignored, topics.TYPE_LED_STATE),
        (topics.TOPIC_SET_BUZZER, ignored, topics.TYPE_BUZZER_STATE),
        (topics.TOPIC_SET_OLED, ignored, topics.TYPE_OLED_STATE),
        (topics.TOPIC_SET_MOTOR, ignored, topics.TYPE_MOTORS_STATE),
        (topics.TOPIC_SET_RGB, ignored, topics.TYPE_RGBS_STATE),
        (topics.TOPIC_SET_MOTOR_DUTY, ignored, topics.TYPE_MOTORS_STATE),
        (topics.TOPIC_ENABLE_RECEPTION, ignored, topics.TYPE_BOOL),
    )
    for name, handler, mtype in board_in:
        bus.on(name, handler, mtype, **node(topics.NODE_BOARD))
    # Published only when the board reports them, which a simulated board never does:
    # there is no gamepad on its receiver, no SBUS radio, no button press and no battery.
    for name, mtype in ((topics.TOPIC_BOARD_JOY, topics.TYPE_JOY),
                        (topics.TOPIC_SBUS, topics.TYPE_SBUS),
                        (topics.TOPIC_BOARD_BUTTON, topics.TYPE_BUTTON_STATE),
                        (topics.TOPIC_BATTERY, topics.TYPE_UINT16)):
        bus.advertise(name, mtype, **node(topics.NODE_BOARD))

    def _ids(value) -> list[int]:
        # uint8[] may arrive as a JSON list or, as rosbridge encodes it, base64.
        if isinstance(value, str):
            return list(base64.b64decode(value))
        return [int(i) for i in (value or [])]

    def count_of(name: str) -> int | None:
        data = live[0]
        return None if data is None else servos.angle_to_count(
            name, float(data.qpos[qpos_adr[name]]))

    def get_bus_servos_position(args: dict) -> dict:
        """Positions of the requested ids, as the driver reads them one by one."""
        out = []
        for sid in _ids(args.get("id")):
            name = by_id.get(sid)
            count = count_of(name) if name else None
            if count is not None:
                out.append({"id": sid, "position": count})
        return {"success": True, "position": out}

    def get_bus_servo_state(args: dict) -> dict:
        """What the simulated servo can truthfully report: its id, position and the
        firmware defaults the simulation assumes (no offset, the 0..1000 travel, torque
        on). Voltage and temperature are not simulated and come back empty, as the
        driver leaves a field it could not read."""
        out = []
        for cmd in args.get("cmd") or []:
            sid = int(cmd.get("id", 0))
            name = by_id.get(sid)
            state = {k: [] for k in ("present_id", "target_id", "position", "offset",
                                     "voltage", "temperature", "position_limit",
                                     "voltage_limit", "max_temperature_limit",
                                     "enable_torque", "save_offset", "stop")}
            if name is not None:
                if cmd.get("get_id"):
                    state["present_id"] = [sid]
                if cmd.get("get_position") and (count := count_of(name)) is not None:
                    state["position"] = [count]
                if cmd.get("get_offset"):
                    state["offset"] = [0]
                if cmd.get("get_position_limit"):
                    state["position_limit"] = [servos.COUNT_MIN, servos.COUNT_MAX]
                if cmd.get("get_torque_state"):
                    state["enable_torque"] = [1]
            out.append(state)
        return {"success": True, "state": out}

    def get_pwm_servo_state(args: dict) -> dict:
        # The AiNex carries no PWM servo, so every read comes back empty.
        return {"success": True,
                "state": [{"id": [], "position": [], "offset": []}
                          for _ in args.get("cmd") or []]}

    bus.service(topics.SRV_BUS_SERVO_GET, get_bus_servos_position,
                topics.SRV_TYPE_GET_BUS_SERVOS_POSITION, **node(topics.NODE_BOARD))
    bus.service(topics.SRV_BUS_SERVO_GET_STATE, get_bus_servo_state,
                topics.SRV_TYPE_GET_BUS_SERVO_STATE, **node(topics.NODE_BOARD))
    bus.service(topics.SRV_PWM_SERVO_GET_STATE, get_pwm_servo_state,
                topics.SRV_TYPE_GET_PWM_SERVO_STATE, **node(topics.NODE_BOARD))

    # ---------------------------------------------------------------- IMU pipeline

    for name, owner in ((topics.TOPIC_IMU_RAW, topics.NODE_BOARD),
                        (topics.TOPIC_IMU_CORRECTED, topics.NODE_IMU_CALIB),
                        (topics.TOPIC_IMU, topics.NODE_IMU_FILTER)):
        bus.advertise(name, topics.TYPE_IMU, **node(owner))
    bus.advertise(topics.TOPIC_MAG_RAW, topics.TYPE_MAGNETOMETER, **node(topics.NODE_BOARD))
    bus.advertise(topics.TOPIC_MAG, topics.TYPE_MAGNETIC_FIELD, **node(topics.NODE_BOARD))

    # ---------------------------------------------------------------- camera

    camera_out = (
        (topics.TOPIC_CAMERA_RAW, topics.TYPE_IMAGE, topics.NODE_CAMERA),
        (topics.TOPIC_CAMERA_INFO, topics.TYPE_CAMERA_INFO, topics.NODE_CAMERA),
        (topics.TOPIC_CAMERA, topics.TYPE_COMPRESSED_IMAGE, topics.NODE_CAMERA),
        (topics.TOPIC_CAMERA_RECT, topics.TYPE_IMAGE, topics.NODE_RECTIFY),
    )
    for name, mtype, owner in camera_out:
        bus.advertise(name, mtype, **node(owner))
    bus.service(topics.SRV_SET_CAMERA_INFO, lambda a: {"success": True, "status_message": ""},
                topics.SRV_TYPE_SET_CAMERA_INFO, **node(topics.NODE_CAMERA))
    head_camera = streams.HeadCamera(bus, model, camera, ainex_model.CAMERA_FOVY_DEG,
                                     jpeg_quality, scene_option)

    # ---------------------------------------------------------------- sensor node

    button = {"enabled": True, "seq": 0}

    def button_enable(args: dict) -> dict:
        button["enabled"] = bool(args.get("data", False))
        return {"success": True, "message": "set_button_enable"}

    bus.advertise(topics.TOPIC_BUTTON_STATE, topics.TYPE_BOOL, **node(topics.NODE_SENSOR))
    bus.on(topics.TOPIC_SENSOR_LED, ignored, topics.TYPE_BOOL, **node(topics.NODE_SENSOR))
    bus.service(topics.SRV_BUTTON_ENABLE, button_enable, topics.SRV_TYPE_SET_BOOL,
                **node(topics.NODE_SENSOR))

    # ---------------------------------------------------------------- the other nodes

    def write_head_counts(pan: int, tilt: int) -> None:
        ctl.write_servos({"head_pan": servos.count_to_angle("head_pan", pan),
                          "head_tilt": servos.count_to_angle("head_tilt", tilt)})

    color = VisionNode(bus, topics.NODE_COLOR_DETECTION, topics.TOPIC_COLOR_IMAGE_RESULT,
                       head_camera)
    face = VisionNode(bus, topics.NODE_FACE_DETECT, topics.TOPIC_FACE_IMAGE_RESULT,
                      head_camera, label="face")
    color.register(topics.SRV_COLOR_ENTER, topics.SRV_COLOR_EXIT, topics.SRV_COLOR_START,
                   topics.SRV_COLOR_STOP, update_lab=topics.SRV_COLOR_UPDATE_LAB,
                   update_detect=topics.TOPIC_UPDATE_DETECT)
    face.register(topics.SRV_FACE_ENTER, topics.SRV_FACE_EXIT, topics.SRV_FACE_START,
                  topics.SRV_FACE_STOP)
    # `/object/pixel_coords` has two publishers on the robot; both are declared.
    bus.advertise(topics.TOPIC_PIXEL_COORDS, topics.TYPE_OBJECTS_INFO,
                  **node(topics.NODE_COLOR_DETECTION))

    joystick = JoystickControl(ctl)
    bus.advertise(topics.TOPIC_JOY, topics.TYPE_JOY, **node(topics.NODE_JOY))
    bus.on(topics.TOPIC_JOY, joystick.on_joy, topics.TYPE_JOY,
           **node(topics.NODE_JOYSTICK_CONTROL))

    app = App(ctl, write_head_counts, color, face)
    bus.advertise(topics.TOPIC_APP_IMAGE_RESULT, topics.TYPE_IMAGE, **node(topics.NODE_APP))
    for name, handler, stype in (
        (topics.SRV_APP_ENTER, app.enter, topics.SRV_TYPE_SET_INT),
        (topics.SRV_APP_SET_RUNNING, app.set_running, topics.SRV_TYPE_SET_BOOL),
        (topics.SRV_APP_SET_TARGET_COLOR, app.set_target_color, topics.SRV_TYPE_SET_POINT),
        (topics.SRV_APP_GET_TARGET_COLOR, app.get_target_color, topics.SRV_TYPE_TRIGGER),
        (topics.SRV_APP_SET_THRESHOLD, app.set_threshold, topics.SRV_TYPE_SET_FLOAT),
        (topics.SRV_APP_HEARTBEAT, app.heartbeat, topics.SRV_TYPE_SET_BOOL),
    ):
        bus.service(name, handler, stype, **node(topics.NODE_APP))

    # ---------------------------------------------------------------- parameters

    for param in topics.PARAMETERS:
        bus.set_param(param.name, param.value)

    # ---------------------------------------------------------------- periodic streams

    attitude = streams.Attitude()
    seqs = {"imu": 0, "joy": 0}

    def publish_imu(stamp_s: float) -> None:
        seqs["imu"] += 1
        msgs = streams.imu_messages(attitude, seqs["imu"], stamp_s)
        owners = {topics.TOPIC_IMU_RAW: topics.NODE_BOARD,
                  topics.TOPIC_MAG_RAW: topics.NODE_BOARD, topics.TOPIC_MAG: topics.NODE_BOARD,
                  topics.TOPIC_IMU_CORRECTED: topics.NODE_IMU_CALIB,
                  topics.TOPIC_IMU: topics.NODE_IMU_FILTER}
        types = {topics.TOPIC_MAG_RAW: topics.TYPE_MAGNETOMETER,
                 topics.TOPIC_MAG: topics.TYPE_MAGNETIC_FIELD}
        for name, msg in msgs.items():
            bus.publish(name, msg, types.get(name, topics.TYPE_IMU), node=owners[name])

    def publish_joy(stamp_s: float) -> None:
        seqs["joy"] += 1
        bus.publish(topics.TOPIC_JOY, streams.neutral_joy(seqs["joy"], stamp_s), topics.TYPE_JOY,
                    node=topics.NODE_JOY)

    def publish_button(_stamp_s: float) -> None:
        # Published every cycle while enabled; the simulated user button is never pressed.
        if button["enabled"]:
            bus.publish(topics.TOPIC_BUTTON_STATE, {"data": False}, topics.TYPE_BOOL,
                        node=topics.NODE_SENSOR)

    clocks = streams.Clocks(f"ainex-streams{'-' + str(bus.ns) if bus.ns else ''}",
                            bus.server.now)
    rate = topics.RATES_HZ
    clocks.every(rate[topics.TOPIC_IMU], publish_imu)
    clocks.every(rate[topics.TOPIC_JOY], publish_joy)
    clocks.every(rate[topics.TOPIC_BUTTON_STATE], publish_button)
    clocks.every(rate[topics.TOPIC_CAMERA_RAW], head_camera.publish)

    setpoint = PlanarSetpoint()
    # The ground under the soles, solved every tick; see ground.py.
    ground = GroundFollow(model, namespace)

    if world_reset is not None:
        # A whole-world reset restores this robot's joints and actuator targets but not
        # the setpoint integrating its gait, so the torso would drive for a pose it no
        # longer occupies. The fall integrator too.
        world_reset.on_reset(setpoint.reset)
        world_reset.on_reset(ground.reset)

    print(f"ainex under namespace {bus.ns or '<bare>'}: {len(topics.TOPICS)} topics, "
          f"{len(topics.SERVICES)} services, {len(topics.PARAMETERS)} parameters "
          "(robots_specs/ainex/ros.yml)", file=sys.stderr)

    dt = 1.0 / control_hz
    phase = {"at": 0.0}
    player: dict[str, ActionPlayer | None] = {"at": None}
    rest = rest_pose()
    held = dict(rest)
    was = {"falling": False, "fall_started": 0.0, "moving": None, "started": False,
           "stepping": False}

    def publish_is_walking(moving: bool) -> None:
        bus.publish(topics.TOPIC_IS_WALKING, {"data": bool(moving)}, topics.TYPE_BOOL,
                    node=topics.NODE_CONTROLLER)

    def to_init_pose() -> None:
        # `move_to_init_pose`: the body to init_pose.yaml, the head left where it is.
        player["at"] = None
        phase["at"] = 0.0
        held.update({k: v for k, v in rest.items() if k not in servos.HEAD_JOINTS})

    control_period = 1.0 / control_hz
    due = {"at": None}
    if world_reset is not None:
        # The observations a `/reset` caller waits for come from the tick after it.
        world_reset.on_reset(lambda: due.update(at=None))

    def step(data):
        if data is None:
            clocks.stop()
            head_camera.close()
            return
        live[0] = data
        if not was["started"]:
            was["started"] = True
            clocks.start()
        # The controller on its own drift-free clock of simulated time; the camera on
        # every call.
        now = float(getattr(data, "time", 0.0))
        if due["at"] is not None and due["at"] > now + 2 * control_period:
            due["at"] = None  # the clock went back: re-anchor
        if due["at"] is None or now >= due["at"] - 0.5 / step.rate_hz:
            control(data)
            due["at"] = (now if due["at"] is None else due["at"]) + control_period
            if due["at"] < now - control_period:
                due["at"] = now + control_period
        head_camera.render(data)

    step.rate_hz = max(float(topics.RATES_HZ[topics.TOPIC_CAMERA_RAW]), float(control_hz))

    def control(data):
        with ctl.cond:
            writes = dict(ctl.servo_writes)
            ctl.servo_writes.clear()
            # An init pose or an action group waits for the gait to finish its cycle.
            if not ctl.running:
                if ctl.inits_done < ctl.init_requests:
                    to_init_pose()
                    ctl.inits_done = ctl.init_requests
                    ctl.walking_enable = True
                    ctl.init_pose_finish = True
                    ctl.cond.notify_all()
                if ctl.pending_action is not None:
                    name, ctl.pending_action = ctl.pending_action, None
                    # set_action_callback: walking disabled and control cleared while the
                    # group plays; an unknown name plays nothing and still ends in the
                    # init pose.
                    ctl.walking_enable = False
                    ctl.init_pose_finish = False
                    if name in actions:
                        current = {n: float(data.qpos[qpos_adr[n]]) for n in qpos_adr}
                        player["at"] = ActionPlayer(actions[name], current)
                    else:
                        print(f"unknown action group {name!r}", file=sys.stderr)
                        player["at"] = ActionPlayer([], {})
            advancing = (ctl.running and ctl.walking_enable and ctl.init_pose_finish
                         and player["at"] is None)
            param = ctl.param

        if player["at"] is not None:
            held.update(player["at"].step(dt))
            if player["at"].finished:
                to_init_pose()
                with ctl.cond:
                    ctl.walking_enable = True
                    ctl.init_pose_finish = True
            vx = vy = wz = 0.0
        elif advancing:
            vx, vy, wz = gait.planar_velocity(param)
            phase["at"] += dt / max(param.period_time, 1e-3)
            if phase["at"] >= 1.0:
                phase["at"] -= 1.0
                _cycle_done(ctl)
            with ctl.cond:
                halted = not ctl.running
            if halted:
                phase["at"] = 0.0
            held.update(gait.leg_joint_targets(param, phase["at"], leg_geometry))
            held.update(gait.arm_joint_targets(param, phase["at"]))
        else:
            vx = vy = wz = 0.0
            with ctl.cond:
                frozen = ctl.running
            if not frozen:
                # Ease the legs back to the rest pose rather than snapping.
                phase["at"] = 0.0
                for name in (*servos.LEG_JOINTS, BASE_PITCH):
                    held[name] += (rest[name] - held[name]) * min(4.0 * dt, 1.0)

        # Raw servo writes win over everything: they are a direct command to a servo.
        held.update(writes)
        for name, value in held.items():
            data.ctrl[actuator_ids[name]] = value

        pose = base.pose
        x, y = float(pose[0, 3]), float(pose[1, 3])
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        stepping = bool(vx or vy or wz)
        if was["stepping"] and not stepping:
            # The gait has halted (or been frozen), so the body stops where it is. The
            # setpoint leads a walking robot by up to TARGET_LEAD_M; kept, the torso would
            # coast that far after the controller already reported the walk finished.
            setpoint.hold()
        was["stepping"] = stepping
        base.ctrl = setpoint.step(x, y, yaw, vx, vy, wz, dt)

        # z is the ground's to decide, after the legs and the lean have been commanded.
        gs = ground.step(data, dt)
        if gs.falling and not was["falling"]:
            was["fall_started"] = float(data.time)
            print(f"ainex: no surface under either sole at ({x:.2f}, {y:.2f}) -- falling",
                  file=sys.stderr)
        elif was["falling"] and not gs.falling:
            print(f"ainex: landed on z {gs.surface_z:.4f} after "
                  f"{float(data.time) - was['fall_started']:.2f} s", file=sys.stderr)
        was["falling"] = gs.falling

        attitude.set(yaw, float(data.qpos[qpos_adr[BASE_PITCH]]), wz,
                     float(data.qvel[pitch_dof]))

        with ctl.cond:
            ctl.moving = ctl.running
            moving = ctl.moving
            ctl.cond.notify_all()
        if moving != was["moving"]:
            # On transitions only, as the controller publishes it -- including the False
            # its first loop reports.
            publish_is_walking(moving)
            was["moving"] = moving

    return step


def _cycle_done(ctl: _Controller) -> None:
    """One full gait cycle finished: a pending stop takes effect, `period_times` counts."""
    with ctl.cond:
        if ctl.stop_pending:
            ctl.running = ctl.stop_pending = False
            ctl.count_step = 0
        elif int(ctl.wire.get("period_times", 0)) != 0:
            ctl.count_step += 1
            if ctl.count_step >= int(ctl.wire["period_times"]):
                ctl.count_step = 0
                ctl.wire["period_times"] = 0
                ctl.param = _gait_of(ctl.wire)
                ctl.running = False
        else:
            ctl.count_step = 0
        # Halting and reporting it are one event, as `self.stop` is on the robot: a
        # blocked `stop` returns, and `/walking/is_walking` answers false, together.
        ctl.moving = ctl.running
        ctl.cond.notify_all()


def _leg_geometry(model, namespace: str) -> gait.LegGeometry:
    def link(child: str) -> float:
        return float(np.linalg.norm(model.body_pos[model.body(f"{namespace}{child}").id]))

    return gait.LegGeometry(thigh=link("l_knee_link"), shank=link("l_ank_pitch_link"))


def serve_ros(port: int, base, model, prefix: str, camera: str | None,
              jpeg_quality: int = 80, control_hz: float = 20.0, extra: dict | None = None,
              host: str = "0.0.0.0", namespace: str = ""):
    """The single-robot path: own a server on `port`, put one AiNex on it, start it."""
    from ros_surfaces import RobotFleet

    fleet = RobotFleet(port=port, host=host)
    fleet.attach(namespace, attach_ros, base=base, model=model, prefix=prefix,
                 camera=camera, jpeg_quality=jpeg_quality, control_hz=control_hz,
                 extra=extra)
    fleet.start()
    return fleet
