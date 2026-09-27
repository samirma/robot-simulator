"""The AiNex's other boot nodes, as far as they touch the robot: joystick_control, the
app node, and the two vision nodes' service surface.

Each one talks to `ainex_controller` the way it does on the robot -- through
`/walking/command`, `/walking/set_param`, `/walking/init_pose` and the head servos -- which
here means through the `_Controller` methods those names are served by. So a stick
release, an app mode change and a client's `stop` all end in the one `stop` the ROS
file's `stop_command` describes.

What is not simulated: the app's autonomous behaviours once a mode is running (kicking,
patrolling, tracking, fall recovery), and detection itself. The vision nodes present
their services and topics and publish their result frames, and report no detections.
"""

from __future__ import annotations

import math
import threading
import time

from . import topics

# joystick_control.py
AXES_MAP = ("lx", "ly", "rx", "ry", "r2", "l2", "hat_x", "hat_y")
BUTTON_MAP = ("cross", "circle", "", "square", "triangle", "", "l1", "r1", "l2", "r2",
              "select", "start", "", "l3", "r3", "", "hat_xl", "hat_xr", "hat_yu",
              "hat_yd", "")


class _GaitManager:
    """`ainex_kinematics/gait_manager.py`, the client library joystick_control and the app
    both walk through: a parameter block on `/walking/set_param`, then `start` once."""

    def __init__(self, ctl) -> None:
        self._ctl = ctl
        self.state = "enable"
        self.walking_param = ctl.walking_param()

    def get_gait_param(self) -> dict:
        p = self.walking_param
        return {
            "init_x_offset": p["init_x_offset"], "init_y_offset": p["init_y_offset"],
            "body_height": p["init_z_offset"], "init_roll_offset": p["init_roll_offset"],
            "init_pitch_offset": p["init_pitch_offset"],
            "init_yaw_offset": p["init_yaw_offset"], "hip_pitch_offset": p["hip_pitch_offset"],
            "step_fb_ratio": p["step_fb_ratio"], "step_height": p["z_move_amplitude"],
            "angle_move_amplitude": p["angle_move_amplitude"],
            "z_swap_amplitude": p["z_swap_amplitude"], "pelvis_offset": p["pelvis_offset"],
            "move_aim_on": p["move_aim_on"],
        }

    def update_param(self, step_velocity, x, y, angle, walking_param=None, arm_swap=30,
                     step_num=0) -> None:
        p = dict(self.walking_param)
        if walking_param is not None:
            p.update(
                init_x_offset=walking_param["init_x_offset"],
                init_y_offset=walking_param["init_y_offset"],
                init_z_offset=walking_param["body_height"],
                init_roll_offset=walking_param["init_roll_offset"],
                init_pitch_offset=walking_param["init_pitch_offset"],
                init_yaw_offset=walking_param["init_yaw_offset"],
                hip_pitch_offset=walking_param["hip_pitch_offset"],
                step_fb_ratio=walking_param["step_fb_ratio"],
                z_move_amplitude=walking_param["step_height"],
                z_swap_amplitude=walking_param["z_swap_amplitude"],
                pelvis_offset=walking_param["pelvis_offset"],
                move_aim_on=walking_param["move_aim_on"],
            )
        p.update(period_time=step_velocity[0], dsp_ratio=step_velocity[1],
                 y_swap_amplitude=step_velocity[2], x_move_amplitude=x, y_move_amplitude=y,
                 angle_move_amplitude=angle, arm_swing_gain=math.radians(arm_swap),
                 period_times=step_num)
        self.walking_param = p
        self._ctl.set_walking_param(p)

    def set_step(self, step_velocity, x, y, angle, walking_param=None, arm_swap=30,
                 step_num=0) -> None:
        self.update_param(step_velocity, x, y, angle, walking_param, arm_swap, step_num)
        if self.state != "walking":
            self.state = "walking"
            self.walking_param = self._ctl.walking_param()
            self._ctl.command("start")

    def stop(self) -> None:
        self._ctl.command("stop")
        self.state = "stop"


class JoystickControl:
    """joystick_control.py on `/joy`: the left stick walks, the right stick turns and sets
    the body height, `start` resets it, and letting go stops the gait."""

    def __init__(self, ctl) -> None:
        self._ctl = ctl
        self._gait: _GaitManager | None = None
        self._lock = threading.Lock()
        self.period_time = [400, 0.2, 0.02]
        self.x = self.y = self.angle = 0.0
        self.init_z_offset = 0.025
        self.status = "stop"
        self.update_param = False
        self.time_stamp_ry = 0.0
        self.last_axes = dict.fromkeys(AXES_MAP, 0.0)
        self.last_buttons = dict.fromkeys(BUTTON_MAP, 0)

    @property
    def gait(self) -> _GaitManager:
        if self._gait is None:
            self._gait = _GaitManager(self._ctl)
        return self._gait

    def on_joy(self, msg: dict) -> None:
        with self._lock:
            axes = dict(zip(AXES_MAP, [float(a) for a in msg.get("axes") or []]))
            buttons = dict(zip(BUTTON_MAP, [int(b) for b in msg.get("buttons") or []]))
            self._height(axes)
            if any(k != "ry" and self.last_axes.get(k) != v for k, v in axes.items()):
                self._axes(axes)
            if buttons.get("start", 0) and not self.last_buttons.get("start", 0):
                self._reset_height()
            self.last_axes = {**self.last_axes, **axes}
            self.last_buttons = {**self.last_buttons, **buttons}

    def _axes(self, axes: dict) -> None:
        self.x = self.y = self.angle = 0.0
        self.period_time = [400, 0.2, 0.02]
        ly, lx, rx = axes.get("ly", 0.0), axes.get("lx", 0.0), axes.get("rx", 0.0)
        if ly > 0.3:
            self.update_param, self.x = True, 0.013
        elif ly < -0.3:
            self.update_param, self.x = True, -0.013
        if lx > 0.3:
            self.period_time[2] = 0.025
            self.update_param, self.y = True, 0.015
        elif lx < -0.3:
            self.period_time[2] = 0.025
            self.update_param, self.y = True, -0.015
        if rx > 0.3:
            self.update_param, self.angle = True, 8
        elif rx < -0.3:
            self.update_param, self.angle = True, -8
        if self.update_param:
            param = self.gait.get_gait_param()
            param["body_height"] = self.init_z_offset
            self.gait.set_step(self.period_time, self.x, self.y, self.angle, param)
        if self.status == "stop" and self.update_param:
            self.status = "move"
        elif self.status == "move" and not self.update_param:
            self.status = "stop"
            self.gait.stop()
        self.update_param = False

    def _height(self, axes: dict) -> None:
        now = time.monotonic()
        if now <= self.time_stamp_ry:
            return
        update = False
        ry = axes.get("ry", 0.0)
        if ry < -0.5:
            update = True
            self.init_z_offset += 0.005
            if self.init_z_offset > 0.06:
                update, self.init_z_offset = False, 0.06
        elif ry > 0.5:
            update = True
            self.init_z_offset -= 0.005
            if self.init_z_offset < 0.025:
                update, self.init_z_offset = False, 0.025
        if update and not self.update_param:
            param = self.gait.get_gait_param()
            param["body_height"] = self.init_z_offset
            self.gait.update_param(self.period_time, self.x, self.y, self.angle, param)
            self.time_stamp_ry = now + 0.05

    def _reset_height(self) -> None:
        param = self.gait.get_gait_param()
        steps = int(abs(0.025 - self.init_z_offset) / 0.005)
        for _ in range(steps):
            self.init_z_offset += math.copysign(0.005, 0.025 - self.init_z_offset)
            param["body_height"] = self.init_z_offset
            self.gait.update_param(self.period_time, 0.0, 0.0, 0.0, param, step_num=1)


class VisionNode:
    """color_detection / face_detect: enter subscribes to the camera, start detects.

    Result frames go out on `image_result` once entered; `/object/pixel_coords` goes out
    per frame once started, with no detections, because detection is not simulated.
    """

    def __init__(self, bus, node: str, image_result: str, camera, label: str = "") -> None:
        self._bus = bus
        self._node = node
        self._image_result = image_result
        self._camera = camera
        self._label = label
        self.entered = False
        self.started = False
        self.detect: list = []
        bus.advertise(image_result, topics.TYPE_IMAGE, node=node)
        camera.frame_listeners.append(self._on_frame)

    def register(self, enter: str, exit_: str, start: str, stop: str, *,
                 update_lab: str | None = None, update_detect: str | None = None) -> None:
        bus, node = self._bus, self._node

        def set_(entered=None, started=None):
            def handler(_args: dict) -> dict:
                if entered is not None:
                    self.entered = entered
                    if not entered:
                        self.started = False
                if started is not None:
                    self.started = started
                return {}
            return handler

        bus.service(enter, set_(entered=True), topics.SRV_TYPE_EMPTY, node=node)
        bus.service(exit_, set_(entered=False), topics.SRV_TYPE_EMPTY, node=node)
        bus.service(start, set_(started=True), topics.SRV_TYPE_EMPTY, node=node)
        bus.service(stop, set_(started=False), topics.SRV_TYPE_EMPTY, node=node)
        if update_lab is not None:
            bus.service(update_lab, lambda a: {}, topics.SRV_TYPE_EMPTY, node=node)
        if update_detect is not None:
            bus.on(update_detect, lambda m: setattr(self, "detect", list(m.get("data") or [])),
                   topics.TYPE_COLORS_DETECT, node=node)

    def _on_frame(self, seq: int, count: int, frame) -> None:
        from .streams import has_subscriber

        if not self.entered:
            return
        if has_subscriber(self._bus, self._image_result):
            self._bus.publish(self._image_result, self._camera.image_msg(seq, count, frame),
                              topics.TYPE_IMAGE, node=self._node)
        if self.started:
            self._bus.publish(topics.TOPIC_PIXEL_COORDS, {"data": []},
                              topics.TYPE_OBJECTS_INFO, node=self._node)


#: app_node.py `head_init_pose`: (pan, tilt) servo counts each mode starts from.
HEAD_INIT = {"control": (500, 500), "visual_patrol": (500, 260), "color_detect": (500, 550),
             "color_track": (500, 500), "face_detect": (500, 630), "kick_ball": (500, 300),
             "fall_rise": (500, 500)}
#: Modes whose exit stops the gait and leaves colour detection (face_detect: face).
_DETECTING = {"visual_patrol", "color_detect", "color_track", "kick_ball", "face_detect"}


class App:
    """app_node.py's service surface and the mode machine behind `/app/enter`."""

    def __init__(self, ctl, write_head_counts, color: VisionNode, face: VisionNode) -> None:
        self._ctl = ctl
        self._head = write_head_counts
        self._color = color
        self._face = face
        self._gait: _GaitManager | None = None
        self._lock = threading.Lock()
        self.state = "idle"
        self.is_running = False
        self.threshold = 0.2
        self.target_color = None

    @property
    def gait(self) -> _GaitManager:
        if self._gait is None:
            self._gait = _GaitManager(self._ctl)
        return self._gait

    def _init_action(self, mode: str) -> None:
        """init_action: `/walking/init_pose`, then the head to the mode's start pose."""
        self._ctl.request_init_pose(block=True)
        pan, tilt = HEAD_INIT[mode]
        self._head(pan, tilt)

    def enter(self, args: dict) -> dict:
        mode = topics.APP_MODES.get(int(args.get("data", -1)))
        if mode is None:
            return {"success": True, "message": ""}
        with self._lock:
            old = self.state
            if old in _DETECTING:
                self.gait.stop()
                (self._face if old == "face_detect" else self._color).entered = False
            # prepare: the image subscriptions each mode needs.
            if mode == "face_detect":
                self._face.entered = True
            elif mode not in ("idle", "fall_rise"):
                self._color.entered = True
            self.is_running = False
            if mode == "idle":
                self._ctl.command("enable_control")
                self._color.entered = self._face.entered = False
            elif mode == "control":
                self._ctl.command("enable_control")
                self._init_action(mode)
            elif mode == "fall_rise":
                self._init_action(mode)
                self._ctl.wait_halted()
                self._ctl.command("disable_control")
            else:
                self._init_action(mode)
            self.state = mode
        return {"success": True, "message": ""}

    def set_running(self, args: dict) -> dict:
        run = bool(args.get("data", False))
        with self._lock:
            if not run:
                self.is_running = False
                self.gait.stop()
                if self.state == "color_detect":
                    self._color.started = False
                elif self.state == "face_detect":
                    self._face.started = False
            elif self.state == "idle":
                return {"success": False, "message": ""}
            elif self.state in HEAD_INIT:
                self._init_action(self.state)
                self.is_running = True
                if self.state == "color_detect":
                    self._color.started = True
                elif self.state == "face_detect":
                    self._face.started = True
        # The vendor answers with the request's own value: `false` for a stop.
        return {"success": run, "message": ""}

    def set_target_color(self, args: dict) -> dict:
        point = args.get("data") or {}
        with self._lock:
            if point.get("x") == -1 and point.get("y") == -1:
                self.target_color = None
            else:
                # The colour picker samples the camera over several frames; with no
                # detector here it never settles on a colour.
                self.state = "color_picker"
        return {"success": True, "message": ""}

    def get_target_color(self, _args: dict) -> dict:
        with self._lock:
            if self.target_color is None:
                return {"success": False, "message": ""}
            r, g, b = self.target_color
            return {"success": True, "message": f"{int(r)},{int(g)},{int(b)}"}

    def set_threshold(self, args: dict) -> dict:
        with self._lock:
            self.threshold = float(args.get("data", self.threshold))
        return {"success": True, "message": ""}

    def heartbeat(self, _args: dict) -> dict:
        # A 5 s timeout only logs on the robot; nothing moves.
        return {"success": True, "message": ""}
