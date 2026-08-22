"""The AiNex's ROS contract: the manufacturer's own topics and services.

This is what `Hiwonder/ainex` presents -- `ainex_kinematics/scripts/ainex_controller.py`
registers every walking topic and service, and `ros_robot_controller_node.py` the bus
servo one. A client written against the real robot drives this one.

**No `/cmd_vel`, no `/odom`, no `/tf`.** The AiNex is commanded as a state machine
(`/walking/command` plus a parameter block), not by a Twist, and it publishes no wheel
odometry because it has no wheels. See `topics.py`.

Locomotion here is the planar base plus the animated gait described in `ainex.py`: the
walking parameters are turned into a body-frame velocity by `gait.planar_velocity` and
integrated by the same `PlanarSetpoint` the myAGV uses, while `gait.leg_joint_targets`
drives the legs at a phase matched to the distance covered.

Threading, which is the invariant that keeps a service call from corrupting the
simulation: every subscriber and service handler below runs on a **websocket reader
thread** and may only write to the small `_State` object. The `step(data)` callback is
the only thing that touches MjData, and it is called from the simulation thread.
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[2]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from robots.ainex import gait, servos, topics  # noqa: E402
from robots.ainex.actions import ActionPlayer, load_action_dir  # noqa: E402


class _State:
    """Everything the reader threads write and the simulation thread reads.

    Guarded by one lock. Deliberately small: no MjData, no numpy views into the model,
    nothing whose lifetime the simulation owns.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # The vendor's state machine. `walking_enable` is enable/disable; `stepping` is
        # start/stop; `initialised` is enable_control/disable_control, which the app layer
        # uses as a separate axis to gate whether the robot considers itself ready at all.
        self.walking_enable = True
        self.stepping = False
        self.initialised = True
        self.param = gait.WalkingParam()
        # Name of an action group to begin on the next tick, or None.
        self.pending_action: str | None = None
        # Raw bus-servo writes: joint name -> radians. Applied once, then cleared.
        self.servo_writes: dict[str, float] = {}
        self.init_pose_requested = False


def serve_ros(port: int, view, model, camera: str | None, camera_size, jpeg_quality: int,
              control_hz: float, watchdog_s: float, scan: dict | None = None,
              depth: dict | None = None, extra: dict | None = None):
    """Present the AiNex on its manufacturer's topics and return a per-step callback."""
    from contracts.rosbridge_server import RosBridgeServer, header
    from tools.spawn_robot import PlanarSetpoint, SensorStreams, SensorTopics

    for group in ("base", "legs"):
        if group not in view.move_group_ids():
            raise SystemExit(
                f"the ainex ROS surface needs a {group!r} move group; "
                f"this robot has {view.move_group_ids()}"
            )
    base = view.get_move_group("base")

    actions = load_action_dir(Path(extra["action_dir"]) if (extra or {}).get("action_dir") else None)
    print(f"action groups: {', '.join(sorted(actions)) or '(none)'}", file=sys.stderr)

    namespace = getattr(view, "_namespace", "") or ""
    leg_geometry = _leg_geometry(model, namespace)
    joint_ids = {n: model.joint(f"{namespace}{n}").id for n in servos.SERVOS}
    actuator_ids = {n: model.actuator(f"{namespace}{n}").id for n in servos.SERVOS}
    qpos_adr = {n: model.jnt_qposadr[joint_ids[n]] for n in servos.SERVOS}

    server = RosBridgeServer(port=port)
    state = _State()

    # ---------------------------------------------------------------- subscribers

    def on_app_walking_param(msg: dict) -> None:
        with state.lock:
            state.param = gait.from_app_params(
                int(msg.get("speed", gait.APP_DEFAULT_SPEED)),
                float(msg.get("height", 0.025)),
                float(msg.get("x", 0.0)),
                float(msg.get("y", 0.0)),
                float(msg.get("angle", 0.0)),
            )

    def on_walking_param(msg: dict) -> None:
        """The full ZMP parameter block.

        `period_time` is milliseconds on the wire and seconds in `gait`; the balance gains
        are accepted and dropped, because a torso on planar joints cannot tip and there is
        nothing for them to act on.
        """
        with state.lock:
            state.param = gait.WalkingParam(
                period_time=float(msg.get("period_time", 400.0)) / 1000.0,
                dsp_ratio=float(msg.get("dsp_ratio", 0.2)),
                x_amplitude=float(msg.get("x_move_amplitude", 0.0)),
                y_amplitude=float(msg.get("y_move_amplitude", 0.0)),
                angle_amplitude=math.radians(float(msg.get("angle_move_amplitude", 0.0))),
                body_height=float(msg.get("init_z_offset", 0.025)),
                step_height=float(msg.get("z_move_amplitude", 0.02)),
                y_swap=float(msg.get("y_swap_amplitude", 0.02)),
                z_swap=float(msg.get("z_swap_amplitude", 0.006)),
                arm_swing_gain=float(msg.get("arm_swing_gain", 0.5)),
                hip_pitch_offset=math.radians(float(msg.get("hip_pitch_offset", 15.0))),
                period_times=int(msg.get("period_times", 0)),
            ).clamped()

    def on_set_action(msg: dict) -> None:
        name = str(msg.get("data", ""))
        with state.lock:
            if name in actions:
                # The vendor stops walking before running an action group and returns to
                # the init pose afterwards; `step` does the same.
                state.pending_action = name
                state.stepping = False
            else:
                print(f"unknown action group {name!r}", file=sys.stderr)

    def on_bus_servo_set(msg: dict) -> None:
        """`ros_robot_controller/SetBusServosPosition`: raw counts, addressed by servo id.

        The whole point of keeping the vendor's count<->radian mapping (servos.py) is that
        this means here exactly what it means on the robot. `duration` is accepted and
        ignored: the position servo takes its own time to travel, as the real one does.
        """
        by_id = {sid: name for name, (sid, _, _) in servos.SERVOS.items()}
        with state.lock:
            for entry in msg.get("position") or []:
                name = by_id.get(int(entry.get("id", -1)))
                if name is None:
                    continue
                state.servo_writes[name] = servos.clamp(
                    name, servos.count_to_angle(name, float(entry.get("position", 500)))
                )

    def head_setter(joint: str):
        def handler(msg: dict) -> None:
            # ainex_interfaces/HeadState is {position, duration}; the Float64 form is the
            # vendor's Gazebo-only path, so accept `data` too rather than drop the message.
            value = msg.get("position", msg.get("data", 0.0))
            with state.lock:
                state.servo_writes[joint] = servos.clamp(joint, float(value))

        return handler

    server.on(topics.TOPIC_APP_WALKING_PARAM, on_app_walking_param)
    server.on(topics.TOPIC_SET_WALKING_PARAM, on_walking_param)
    server.on(topics.TOPIC_APP_ACTION, on_set_action)
    server.on(topics.TOPIC_BUS_SERVO_SET, on_bus_servo_set)
    server.on(topics.TOPIC_HEAD_PAN, head_setter("head_pan"))
    server.on(topics.TOPIC_HEAD_TILT, head_setter("head_tilt"))

    # ---------------------------------------------------------------- services

    def walking_command(args: dict) -> dict:
        """`SetWalkingCommand`: one of six strings -> {result}.

        Transcribed from ainex_controller.py::walking_command_callback, including that
        enable/disable/start/stop are gated on the robot being initialised while
        enable_control/disable_control are what set that flag.
        """
        command = str(args.get("command", ""))
        if command not in topics.WALKING_COMMANDS:
            return {"result": False}
        with state.lock:
            if state.initialised:
                if command == "start":
                    # `start` implies `enable` on the real robot: ainex_controller.py's
                    # start branch sets walking_enable = True itself before starting the
                    # walking module. Gating on a prior `enable` here made disable->start
                    # walk on hardware but not in sim.
                    state.walking_enable = True
                    state.stepping = True
                elif command == "stop":
                    state.stepping = False
                elif command == "enable":
                    state.walking_enable = True
                elif command == "disable":
                    # The vendor stops before disabling, so `disable` implies `stop`.
                    state.stepping = False
                    state.walking_enable = False
            if command == "enable_control":
                state.initialised = True
            elif command == "disable_control":
                state.initialised = False
        return {"result": True}

    def get_walking_param(args: dict) -> dict:
        with state.lock:
            param = state.param
        return {"parameters": _to_walking_param_msg(param)}

    def is_walking(args: dict) -> dict:
        with state.lock:
            walking = state.stepping
        return {"state": walking, "message": "is_walking"}

    def init_pose(args: dict) -> dict:
        with state.lock:
            state.stepping = False
            state.init_pose_requested = True
        return {}

    server.service(topics.SRV_WALKING_COMMAND, walking_command)
    server.service(topics.SRV_GET_WALKING_PARAM, get_walking_param)
    server.service(topics.SRV_IS_WALKING, is_walking)
    server.service(topics.SRV_INIT_POSE, init_pose)

    # ---------------------------------------------------------------- streams

    sensors = SensorStreams(
        server, model, camera, camera_size, jpeg_quality, scan, depth,
        SensorTopics(
            topics.TOPIC_CAMERA,
            topics.TOPIC_SCAN,
            "/camera/depth/image_raw",
            "/camera/rgb/camera_info",
        ),
    )
    setpoint = PlanarSetpoint()

    server.start()
    subscribed = [
        topics.TOPIC_APP_WALKING_PARAM, topics.TOPIC_SET_WALKING_PARAM,
        topics.TOPIC_APP_ACTION, topics.TOPIC_BUS_SERVO_SET,
        topics.TOPIC_HEAD_PAN, topics.TOPIC_HEAD_TILT,
    ]
    published = [topics.TOPIC_IS_WALKING, topics.TOPIC_JOINT_STATES, topics.TOPIC_IMU]
    print(
        f"ROS topics on ws://0.0.0.0:{port}\n"
        f"  sub {', '.join(subscribed)}\n"
        f"  srv {', '.join([topics.SRV_WALKING_COMMAND, topics.SRV_GET_WALKING_PARAM, topics.SRV_IS_WALKING, topics.SRV_INIT_POSE])}\n"
        f"  pub {', '.join(published + sensors.published)}",
        file=sys.stderr,
    )

    dt = 1.0 / control_hz
    phase = {"at": 0.0}
    player: dict[str, ActionPlayer | None] = {"at": None}
    # Whatever the limbs should hold when nothing else is driving them.
    held = dict(servos.INIT_POSE)

    def step(data):
        if data is None:
            sensors.close()
            server.stop()
            return

        with state.lock:
            param = state.param
            walking = state.stepping and state.walking_enable and state.initialised
            pending = state.pending_action
            state.pending_action = None
            writes = dict(state.servo_writes)
            state.servo_writes.clear()
            wants_init = state.init_pose_requested
            state.init_pose_requested = False
            clients = server.client_count

        # The myAGV's silence-based watchdog is wrong for a state machine: a correct
        # client sends nothing at all between `start` and `stop`. Losing the last client
        # is a far stronger signal, and unlike a timer it keeps /walking/is_walking
        # honest, because the machine really does leave the walking state.
        #
        # This is a SIM-ONLY safety net. The real robot has nothing like it -- its gait
        # engine keeps walking on zero traffic until told to stop -- so a client must
        # issue `stop` explicitly on every exit path and never lean on this.
        if clients == 0 and walking:
            with state.lock:
                state.stepping = False
            walking = False

        if wants_init:
            player["at"] = None
            held.update(servos.INIT_POSE)
            phase["at"] = 0.0

        if pending is not None:
            current = {
                n: float(data.qpos[qpos_adr[n]]) for n in servos.SERVOS
            }
            player["at"] = ActionPlayer(actions[pending], current)

        # An action group owns the whole body while it runs -- as on the real robot, which
        # stops walking, replays, and only then re-enables the gait.
        if player["at"] is not None:
            held.update(player["at"].step(dt))
            if player["at"].finished:
                player["at"] = None
            vx = vy = wz = 0.0
        elif walking:
            vx, vy, wz = gait.planar_velocity(param)
            phase["at"] = (phase["at"] + dt / max(param.period_time, 1e-3)) % 1.0
            held.update(gait.leg_joint_targets(param, phase["at"], leg_geometry))
            held.update(gait.arm_joint_targets(param, phase["at"]))
        else:
            vx = vy = wz = 0.0
            # Ease the legs back to the rest pose rather than snapping: `stop` on the real
            # robot finishes the step it is in.
            phase["at"] = 0.0
            for name in servos.LEG_JOINTS:
                held[name] += (servos.INIT_POSE[name] - held[name]) * min(4.0 * dt, 1.0)

        # Raw bus-servo writes win over everything: they are a direct command to a servo,
        # which is exactly what they are on the robot.
        held.update(writes)

        for name, value in held.items():
            data.ctrl[actuator_ids[name]] = value

        pose = base.pose
        x, y = float(pose[0, 3]), float(pose[1, 3])
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        base.ctrl = setpoint.step(x, y, yaw, vx, vy, wz, dt)

        seq = server.next_seq()
        positions = [float(data.qpos[qpos_adr[n]]) for n in servos.BY_ID]
        server.publish(
            topics.TOPIC_JOINT_STATES,
            {
                "header": header(seq, topics.FRAME_BASE),
                "name": list(servos.BY_ID),
                "position": positions,
                "velocity": [],
                "effort": [],
            },
        )
        server.publish(topics.TOPIC_IS_WALKING, {"data": bool(walking)})
        server.publish(topics.TOPIC_IMU, _imu_msg(seq, yaw, wz))
        sensors.publish(data, seq, x, y, yaw)

    return step


def _leg_geometry(model, namespace: str) -> gait.LegGeometry:
    def link(child: str) -> float:
        return float(np.linalg.norm(model.body_pos[model.body(f"{namespace}{child}").id]))

    return gait.LegGeometry(thigh=link("l_knee_link"), shank=link("l_ank_pitch_link"))


def _to_walking_param_msg(param: gait.WalkingParam) -> dict:
    """A `WalkingParam` back onto the wire, in the vendor's units and field names."""
    return {
        "init_x_offset": 0.0,
        "init_y_offset": 0.0,
        "init_z_offset": param.body_height,
        "init_roll_offset": 0.0,
        "init_pitch_offset": 0.0,
        "init_yaw_offset": 0.0,
        "period_time": param.period_time * 1000.0,  # seconds here, ms on the wire
        "dsp_ratio": param.dsp_ratio,
        "step_fb_ratio": 0.028,
        "period_times": param.period_times,
        "x_move_amplitude": param.x_amplitude,
        "y_move_amplitude": param.y_amplitude,
        "z_move_amplitude": param.step_height,
        "angle_move_amplitude": math.degrees(param.angle_amplitude),
        "move_aim_on": False,
        "arm_swing_gain": param.arm_swing_gain,
        "y_swap_amplitude": param.y_swap,
        "z_swap_amplitude": param.z_swap,
        "pelvis_offset": 5.0,
        "hip_pitch_offset": math.degrees(param.hip_pitch_offset),
        # Accepted on the wire and inert: the torso rides position-controlled planar
        # joints and cannot tip, so a balance gain has nothing to act on.
        "balance_enable": False,
        "balance_hip_roll_gain": 0.0,
        "balance_knee_gain": 0.0,
        "balance_ankle_roll_gain": 0.0,
        "balance_ankle_pitch_gain": 0.0,
    }


def _imu_msg(seq: int, yaw: float, wz: float) -> dict:
    """A `sensor_msgs/Imu` carrying real yaw and yaw rate.

    DEPARTURE, documented in robots/README.md: roll and pitch are identically zero,
    because the base has no roll or pitch degree of freedom. The topic exists so a client
    written against the hardware connects and reads a heading; it is not a substitute for
    the real 9-axis IMU. Covariance of -1 in the first element is the ROS convention for
    "this quantity is not reported", which is the honest thing to say about the rest.
    """
    from contracts.rosbridge_server import header as make_header

    return {
        "header": make_header(seq, topics.FRAME_IMU),
        "orientation": {
            "x": 0.0, "y": 0.0,
            "z": math.sin(yaw / 2.0), "w": math.cos(yaw / 2.0),
        },
        "orientation_covariance": [-1.0] + [0.0] * 8,
        "angular_velocity": {"x": 0.0, "y": 0.0, "z": wz},
        "angular_velocity_covariance": [-1.0] + [0.0] * 8,
        "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
        "linear_acceleration_covariance": [-1.0] + [0.0] * 8,
    }
