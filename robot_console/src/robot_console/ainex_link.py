"""Driving a Hiwonder AiNex over rosbridge: a gait state machine behind a Twist-shaped API.

Reconstructed against the simulator's transcription of the vendor stack
(`simulator/robots/ainex/{topics,gait,ros_surface}.py`, themselves transcribed from
`Hiwonder/ainex`'s `ainex_controller.py`) -- the original console-side module was lost
before it was committed. Everything on the wire here matches that surface, which is the
same contract the real robot presents.

The AiNex has **no `/cmd_vel`, no `/odom`, no `/tf`**, and that is the vendor's design,
not an omission: walking is a parameterised gait, commanded by writing a parameter block
to `/walking/set_param` and calling the `/walking/command` service with `start`/`stop`.
This link's job is to make that look like `publish_cmd_vel(Command)` so the console keeps
one loop and one intent type for every robot.

The mapping is the gait model's, inverted. A gait cycle of period T advances the body
4A per cycle for a foot half-sweep A (`STRIDE_FACTOR`, calibrated against the
manufacturer's published walking speed), so v = 4A/T and the amplitude for a requested
velocity is A = vT/4. The envelope is small and honest about it: |A| <= 0.02 m and 10
degrees per cycle, which at the 400 ms default period caps the robot at 0.20 m/s and
~1.75 rad/s. Ask for more and you get the cap, exactly as the vendor's own app tiers do.

Two wire quirks, both inherited from the vendor message and easy to get backwards:
`period_time` is **milliseconds** on the wire (seconds everywhere in this file), and
`angle_move_amplitude` is **degrees** (radians everywhere else in the console).
"""

from __future__ import annotations

import math
import os
import threading
from typing import Callable, Optional

import roslibpy

from robot_console.teleop import Command
from robot_console.topics import TOPIC_CAMERA_INFO, TOPIC_DEPTH, normalise

# --- the vendor's contract (simulator/robots/ainex/topics.py) ---------------------------

TOPIC_SET_WALKING_PARAM = "/walking/set_param"
TOPIC_CAMERA = "/camera/image_raw/compressed"
SRV_WALKING_COMMAND = "/walking/command"

TYPE_WALKING_PARAM = "ainex_interfaces/WalkingParam"
TYPE_SET_WALKING_COMMAND = "ainex_interfaces/SetWalkingCommand"
TYPE_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
TYPE_IMAGE = "sensor_msgs/Image"
TYPE_CAMERA_INFO = "sensor_msgs/CameraInfo"

# --- the gait envelope (simulator/robots/ainex/gait.py, vendor walking_param.yaml) ------

# Body advance per gait cycle, in units of amplitude: v = STRIDE_FACTOR * A / T.
STRIDE_FACTOR = 4.0
PERIOD_S = 0.400                 # the vendor default; the app presets span 0.3-0.6
X_AMPLITUDE_MAX = 0.02           # m per half-sweep; gait_manager rejects more
Y_AMPLITUDE_MAX = 0.02
ANGLE_AMPLITUDE_MAX_DEG = 10.0

# The speed envelope the console exposes, derived from the caps above at PERIOD_S:
# 4 * 0.02 / 0.4 = 0.20 m/s flat out. Derived rather than copied from the myAGV, because
# this robot walks -- its "fast" is a quarter of the AGV's.
SPEED_MIN = 0.02
SPEED_MAX = 0.20
SPEED_STEP = 0.02
SPEED_DEFAULT = 0.10

# Turn scales with the speed setting like the myAGV's does, capped inside the envelope's
# 4 * 10 deg / 0.4 s ~= 1.75 rad/s. Kept well under it: a biped spinning at its gait
# limit is a biped mapping nothing.
TURN_RATIO = 4.0
TURN_MAX = 1.0


def walking_params(command: Command, *, period: float = PERIOD_S) -> dict:
    """The `ainex_interfaces/WalkingParam` wire message for one velocity intent.

    Inverts v = 4A/T per axis and clamps into the gait envelope. Wire units apply:
    milliseconds for the period, degrees for the turn amplitude. Non-motion fields are
    the vendor defaults from `walking_param.yaml`.
    """
    x = _clamp(command.vx * period / STRIDE_FACTOR, X_AMPLITUDE_MAX)
    y = _clamp(command.vy * period / STRIDE_FACTOR, Y_AMPLITUDE_MAX)
    angle = _clamp(
        math.degrees(command.wz) * period / STRIDE_FACTOR, ANGLE_AMPLITUDE_MAX_DEG
    )
    return {
        "period_time": period * 1000.0,
        "dsp_ratio": 0.2,
        "x_move_amplitude": x,
        "y_move_amplitude": y,
        "angle_move_amplitude": angle,
        "init_z_offset": 0.025,
        "z_move_amplitude": 0.02,
        "y_swap_amplitude": 0.02,
        "z_swap_amplitude": 0.006,
        "arm_swing_gain": 0.5,
        "hip_pitch_offset": 15.0,
        "period_times": 0,
    }


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class AiNexLink:
    """A rosbridge connection with the same shape as `bridge.RobotLink`.

    Same shape on purpose: the console apps call `connect` / `subscribe_camera` /
    `publish_cmd_vel` / `stop` / `close` and never ask which robot answered. What differs
    is under `publish_cmd_vel`: a Twist publisher is stateless, a gait is not, so this
    link tracks whether the robot is stepping and calls the `start`/`stop` service only
    on transitions -- 20 Hz of redundant `start` calls is how a service queue drowns.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        *,
        camera_topic: str = TOPIC_CAMERA,
        camera_info_topic: str = TOPIC_CAMERA_INFO,
        depth_topic: str = TOPIC_DEPTH,
    ) -> None:
        self.host = host
        self.port = int(port)
        self._camera_name = normalise(camera_topic)
        self._camera_info_name = normalise(camera_info_topic)
        self._depth_name = normalise(depth_topic)
        self._ros: Optional[roslibpy.Ros] = None
        self._param: Optional[roslibpy.Topic] = None
        self._command_srv: Optional[roslibpy.Service] = None
        self._camera: Optional[roslibpy.Topic] = None
        self._camera_info: Optional[roslibpy.Topic] = None
        self._depth: Optional[roslibpy.Topic] = None
        self._closed = False
        # Written from the main loop, read in close(); the lock is cheap honesty about
        # the reactor thread also being around.
        self._lock = threading.Lock()
        self._walking = False
        self._last_params: Optional[tuple] = None

    # ---------------------------------------------------------------- lifecycle

    def connect(self, timeout: float = 10.0) -> None:
        ros = roslibpy.Ros(host=self.host, port=self.port)
        ros.run(timeout=timeout)
        if not ros.is_connected:
            raise ConnectionError(f"could not connect to ws://{self.host}:{self.port}")
        self._ros = ros
        self._param = roslibpy.Topic(ros, TOPIC_SET_WALKING_PARAM, TYPE_WALKING_PARAM)
        self._param.advertise()
        self._command_srv = roslibpy.Service(ros, SRV_WALKING_COMMAND, TYPE_SET_WALKING_COMMAND)

    @property
    def is_connected(self) -> bool:
        return bool(self._ros and self._ros.is_connected)

    def __enter__(self) -> "AiNexLink":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- topics

    def subscribe_camera(self, callback: Callable[[dict], None]) -> None:
        if self._ros is None:
            raise RuntimeError("not connected")
        self._camera = roslibpy.Topic(self._ros, self._camera_name, TYPE_COMPRESSED_IMAGE)
        self._camera.subscribe(callback)

    def subscribe_camera_info(self, callback: Callable[[dict], None]) -> None:
        if self._ros is None:
            raise RuntimeError("not connected")
        self._camera_info = roslibpy.Topic(self._ros, self._camera_info_name, TYPE_CAMERA_INFO)
        self._camera_info.subscribe(callback)

    def subscribe_depth(self, callback: Callable[[dict], None]) -> None:
        if self._ros is None:
            raise RuntimeError("not connected")
        self._depth = roslibpy.Topic(self._ros, self._depth_name, TYPE_IMAGE)
        self._depth.subscribe(callback)

    def subscribe_odom(self, callback) -> None:
        """The AiNex publishes no odometry; asking for it is a caller bug, not a shrug."""
        raise RuntimeError(
            "the AiNex has no /odom -- its profile says has_odom=False, and callers "
            "must integrate their own commands instead"
        )

    # ---------------------------------------------------------------- driving

    def publish_cmd_vel(self, command: Command) -> None:
        """Fold a velocity intent into the gait state machine.

        The parameter block goes out only when it changes and the start/stop service is
        called only on motion transitions. Not an optimisation: the vendor node restarts
        its gait phase on `start`, so hammering it at the publish rate makes the robot
        shuffle in place instead of walking.
        """
        if self._param is None or not self.is_connected:
            return
        moving = not command.is_zero()
        params = walking_params(command)
        key = (
            round(params["x_move_amplitude"], 4),
            round(params["y_move_amplitude"], 4),
            round(params["angle_move_amplitude"], 2),
        )
        with self._lock:
            walking_before = self._walking
            params_before = self._last_params
            self._walking = moving
            self._last_params = key

        if moving and key != params_before:
            self._param.publish(roslibpy.Message(params))
        if moving and not walking_before:
            self._call_walking("start")
        elif not moving and walking_before:
            self._call_walking("stop")

    def stop(self) -> None:
        """Stop stepping, explicitly and regardless of believed state.

        Unconditional where `publish_cmd_vel` deduplicates: this is the exit path, and
        the cost of a redundant `stop` is nothing against the cost of a robot that keeps
        walking because the link's idea of "already stopped" was stale.
        """
        with self._lock:
            self._walking = False
        self._call_walking("stop")

    def _call_walking(self, command: str) -> None:
        if self._command_srv is None or not self.is_connected:
            return
        try:
            # Fire-and-forget: this runs on the main loop, which is also feeding the map
            # and (in explore) the frontier planner. Blocking on a service round trip
            # would stall both for however long the bridge feels like taking.
            self._command_srv.call(
                roslibpy.ServiceRequest({"command": command}),
                callback=lambda _result: None,
                errback=lambda _error: None,
            )
        except Exception:
            pass

    # ---------------------------------------------------------------- teardown

    def close(self, *, hard_exit_after: Optional[float] = 2.0) -> None:
        """Stop the gait and tear the connection down. Same escape hatch as RobotLink:
        Twisted's reactor cannot be restarted, and a close that hangs must not leave a
        walking robot walking."""
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
        except Exception:
            pass
        for topic in (self._camera, self._camera_info, self._depth):
            try:
                if topic is not None:
                    topic.unsubscribe()
            except Exception:
                pass
        try:
            if self._param is not None:
                self._param.unadvertise()
        except Exception:
            pass
        bail: Optional[threading.Timer] = None
        if hard_exit_after is not None:
            bail = threading.Timer(hard_exit_after, lambda: os._exit(0))
            bail.daemon = True
            bail.start()
        try:
            if self._ros is not None:
                self._ros.close()
                self._ros.terminate()
        except Exception:
            pass
        finally:
            if bail is not None:
                bail.cancel()

    # ---------------------------------------------------------------- helpers

    def describe(self) -> str:
        return f"ws://{self.host}:{self.port}"
