"""The ``so101_ros`` embodiment: an SO-101 and the worktop rig over rosbridge.

Subclasses the upstream
[`RosEmbodiment`][inspect_robots_ros.embodiment.RosEmbodiment], so joint-state freshness
gating, staleness bounds, camera decoding and the rate preflight stay the upstream
plugin's. What it adds is only what the SO-101's official interface
(``robots_specs/so101/ros2.yml``) and the arm task need:

* **Observations**: the rig's ``/scene/overhead/color/compressed`` and
  ``/scene/side/color/compressed``, the SO-101's wrist camera
  (``/wrist/image_raw/compressed``) and ``/joint_states``.
* **Commands**: the five arm joints as a ``JointTrajectory`` on
  ``/joint_trajectory_controller/joint_trajectory`` (with a header, as a ROS message has),
  and the jaw as a ``control_msgs/action/ParallelGripperCommand`` goal on
  ``/gripper_controller/gripper_cmd`` through a real action client
  (`ros_client.ActionRosbridgeClient`). Unfinished goals are cancelled on close.
* **Reset** through the composed ``/reset`` (``std_srvs/srv/Trigger``); a
  ``success: false`` answer is an error carrying the server's message.
* **Evidence for the scorer**: every synchronized overhead/side pair received, with both
  cameras' ``camera_info`` and the ``/joint_states`` message nearest its stamp, recorded
  in each step's ``info["rig_samples"]``. Nothing here grades: the camera-verdict scorer
  (``scorer.apple_on_plate``) reads these after the episode, and the episode ends when
  the policy says it is done or the step budget runs out -- never on a verdict.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
from inspect_robots import Action, Observation, Scene, StepResult
from inspect_robots_ros._client import TopicSample
from inspect_robots_ros._msgs import message_type
from inspect_robots_ros.embodiment import RosEmbodiment

from robot_console.arm.ros_client import STATUS_SUCCEEDED, ActionRosbridgeClient
from robot_console.arm.ros_settings import (
    CAMERA_INFO_TYPE,
    DEFAULT_VIEWS,
    GRIPPER_ACTION_TYPE,
    OVERHEAD_CAMERA_NAME,
    SCENE_CAMERA_POSES,
    SCENE_CAMERA_TILT_DEG,
    SIDE_CAMERA_NAME,
    RosSettings,
)
from robot_console.arm.vision_success import (
    HOLD_SECONDS,
    MAX_SAMPLE_GAP_S,
    RIG_SAMPLES_KEY,
    SPEED_BASELINE_S,
    SYNC_TOLERANCE_S,
)

#: Frames older than this, behind the newest recorded one, keep their stamps but drop
#: their pixels. The scorer grades the episode's last `HOLD_SECONDS` plus a speed
#: baseline, so this keeps several times that and bounds an episode's memory.
RETAIN_FRAMES_S = 4.0 * (HOLD_SECONDS + SPEED_BASELINE_S + MAX_SAMPLE_GAP_S)
#: Per-topic history, in messages: the rig at 10 Hz and joint state throttled to
#: 2 x control_hz, across the seconds a VLA may think between steps.
HISTORY_LENGTH = 400
#: A new gripper goal is sent when the commanded jaw moves by more than this, rad.
GRIPPER_RESEND_RAD = 1e-3

_CAMERA_INFO_SUBSCRIPTION = "so101-ros-camera-info-{}"


def _stamp_seconds(msg: Any) -> float | None:
    """A ROS 2 ``header.stamp`` as float seconds, or None."""
    header = msg.get("header") if isinstance(msg, Mapping) else None
    stamp = header.get("stamp") if isinstance(header, Mapping) else None
    if not isinstance(stamp, Mapping):
        return None
    try:
        return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
    except (KeyError, TypeError, ValueError):
        return None


def _nearest(samples: list[TopicSample], stamp: float) -> tuple[float, TopicSample] | None:
    best: tuple[float, TopicSample] | None = None
    for sample in samples:
        other = _stamp_seconds(sample.msg)
        if other is None:
            continue
        if best is None or abs(other - stamp) < abs(best[0] - stamp):
            best = (other, sample)
    return best


def _duration(seconds: float) -> dict[str, int]:
    whole = int(seconds)
    nano = round((seconds - whole) * 1e9)
    if nano >= 1_000_000_000:
        whole, nano = whole + 1, 0
    return {"sec": whole, "nanosec": nano}


ZERO_HEADER: dict[str, Any] = {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}


class SO101RosEmbodiment(RosEmbodiment):
    """The SO-101 over rosbridge, recording the rig evidence the scorer grades from."""

    def __init__(self, settings: RosSettings | None = None, **overrides: Any) -> None:
        self.settings = settings or RosSettings(**overrides)
        super().__init__(**self.settings.base_kwargs())
        s = self.settings
        self._rig_topics = {
            name: topic for name, (topic, _h, _w) in s.cameras().items()
            if name in (OVERHEAD_CAMERA_NAME, SIDE_CAMERA_NAME)
        }
        self._info_topics = s.camera_info_topics()
        # Replaced before it ever connects: construction is network-free.
        self._client = ActionRosbridgeClient(
            self.url,
            history_topics=(*self._rig_topics.values(), self.joint_states_topic),
            history_length=HISTORY_LENGTH,
            clock=self._clock,
            sleep=self._sleep,
        )
        self.info = dataclasses.replace(self.info, docs=_DOCS)
        self._info_subscribed = False
        self._gripper_goal = None
        self._gripper_sent: float | None = None
        self._last_recorded: float | None = None
        self._recorded: deque[dict[str, Any]] = deque()

    # -- lifecycle ---------------------------------------------------------------------
    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """``/reset``, refusing a ``success: false``; then fresh observations."""
        del seed
        self._instruction = scene.instruction
        self._ensure_initialized()
        response = self._client.call_service(self.reset_service)
        values = response.values if isinstance(response.values, Mapping) else {}
        if values.get("success") is False:
            raise RuntimeError(f"{self.reset_service} refused: {values.get('message') or ''}")
        self._client.clear_histories()
        self._last_recorded = None
        self._recorded.clear()
        self._gripper_goal, self._gripper_sent = None, None
        sequences = self._capture_sequences(self._all_topics())
        self._wait_for_sequences(sequences, self.obs_timeout_s, "obs_timeout_s")
        observation = self._assemble_observation()
        self._last_publish_time = None
        self._reset_count += 1
        # Evidence starts after the reset landed: nothing from the previous world.
        self._last_recorded = self._newest_rig_stamp()
        return observation

    def step(self, action: Action) -> StepResult:
        """Publish the arm trajectory and the jaw goal, then record the rig evidence."""
        data = np.asarray(action.data, dtype=np.float64)
        if data.shape != self.info.action_space.shape:
            raise ValueError(
                f"action has shape {data.shape}, expected {self.info.action_space.shape}")
        now = self._clock()
        if self._last_publish_time is not None:
            remaining = self._last_publish_time + (1.0 / self.control_hz) - now
            if remaining > 0:
                self._sleep(remaining)

        seq_at_publish = self._client.sequence(self.joint_states_topic)
        publish_time = self._clock()
        self._client.publish(self.command_topic, self._trajectory(data[: len(self.joints)]))
        self._last_publish_time = publish_time
        self._command_gripper(float(data[-1]))
        try:
            self._client.wait_for_sample(self.joint_states_topic, after_seq=seq_at_publish,
                                         timeout_s=self.fresh_obs_timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"no post-publish joint state within fresh_obs_timeout_s="
                f"{self.fresh_obs_timeout_s:g}s at control_hz={self.control_hz:g}") from exc
        observation = self._assemble_observation()
        return StepResult(observation=observation, reward=None, terminated=False,
                          truncated=False, info={RIG_SAMPLES_KEY: self._collect()})

    def close(self) -> None:
        """Cancel every unfinished gripper goal, then release the socket."""
        self._client.close()

    # -- commands ----------------------------------------------------------------------
    def _trajectory(self, arm: np.ndarray) -> dict[str, Any]:
        return {
            "header": dict(ZERO_HEADER),   # stamp 0: "start now"
            "joint_names": list(self.joints),
            "points": [{"positions": [float(v) for v in arm],
                        "time_from_start": _duration(1.0 / self.control_hz)}],
        }

    def _command_gripper(self, value: float) -> None:
        assert self.gripper_low is not None and self.gripper_high is not None
        target = float(min(max(value, self.gripper_low), self.gripper_high))
        goal = self._gripper_goal
        unchanged = self._gripper_sent is not None and abs(target - self._gripper_sent) <= \
            GRIPPER_RESEND_RAD
        still_good = goal is not None and (not goal.done.is_set()
                                          or goal.status == STATUS_SUCCEEDED)
        if unchanged and still_good:
            return
        self._gripper_goal = self._client.send_goal(
            self.gripper_topic, GRIPPER_ACTION_TYPE,
            {"command": {"header": dict(ZERO_HEADER), "name": [self.gripper_joint],
                         "position": [target], "velocity": [], "effort": []}})
        self._gripper_sent = target

    # -- evidence ----------------------------------------------------------------------
    def _newest_rig_stamp(self) -> float | None:
        stamps = [_stamp_seconds(s.msg) for s in
                  self._client.history(self._rig_topics[OVERHEAD_CAMERA_NAME])]
        stamps = [s for s in stamps if s is not None]
        return max(stamps) if stamps else None

    def _camera_infos(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, topic in self._info_topics.items():
            sample = self._client.latest(topic)
            msg = sample.msg if sample is not None else None
            out[name] = None if msg is None else {
                "k": list(msg.get("k") or []), "width": msg.get("width"),
                "height": msg.get("height"),
                "frame_id": (msg.get("header") or {}).get("frame_id"),
            }
        return out

    def _collect(self) -> list[dict[str, Any]]:
        """Every overhead frame since the last step, paired by stamp; see module doc."""
        overhead = self._client.history(self._rig_topics[OVERHEAD_CAMERA_NAME])
        side = self._client.history(self._rig_topics[SIDE_CAMERA_NAME])
        joints = self._client.history(self.joint_states_topic)
        side_newest = max((t for t in (_stamp_seconds(s.msg) for s in side) if t is not None),
                          default=None)
        infos = self._camera_infos()
        frames = sorted(((t, s) for t, s in ((_stamp_seconds(s.msg), s) for s in overhead)
                         if t is not None), key=lambda pair: pair[0])
        out: list[dict[str, Any]] = []
        for index, (stamp, sample) in enumerate(frames):
            if self._last_recorded is not None and stamp <= self._last_recorded:
                continue
            is_newest = index == len(frames) - 1
            # The side partner may still be in flight: wait for it unless a newer
            # overhead frame shows the rig has moved on.
            if is_newest and (side_newest is None or side_newest < stamp - SYNC_TOLERANCE_S):
                break
            record: dict[str, Any] = {
                "overhead": {"stamp": stamp, "format": sample.msg.get("format"),
                             "data": sample.msg.get("data")},
                "side": None, "joint_state": None, "camera_info": infos,
            }
            match = _nearest(side, stamp)
            if match is not None:
                record["side"] = {"stamp": match[0], "format": match[1].msg.get("format"),
                                  "data": match[1].msg.get("data")}
            joint = _nearest(joints, stamp)
            if joint is not None:
                record["joint_state"] = {"stamp": joint[0],
                                         "name": list(joint[1].msg.get("name") or []),
                                         "position": list(joint[1].msg.get("position") or [])}
            out.append(record)
            self._recorded.append(record)
            self._last_recorded = stamp
        self._trim()
        return out

    def _trim(self) -> None:
        if not self._recorded:
            return
        newest = self._recorded[-1]["overhead"]["stamp"]
        while self._recorded and self._recorded[0]["overhead"]["stamp"] < newest - RETAIN_FRAMES_S:
            old = self._recorded.popleft()
            for part in ("overhead", "side"):
                if old.get(part):
                    old[part]["data"] = None
            old["trimmed"] = True

    # -- transport ---------------------------------------------------------------------
    def _all_topics(self) -> tuple[str, ...]:
        """Reset waits for calibration like any other topic: no rig, no episode."""
        return (*super()._all_topics(), *self._info_topics.values())

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            self._client.connect()
        except Exception as exc:  # noqa: BLE001 - any failure means "no bridge here"
            raise ConnectionError(f"could not connect to rosbridge at {self.url}; start a "
                                  "simulator (simulator/kitchen.sh serve) first") from exc
        if not self._info_subscribed:
            for name, topic in self._info_topics.items():
                self._client.subscribe(
                    topic, subscription_id=_CAMERA_INFO_SUBSCRIPTION.format(name),
                    message_type=CAMERA_INFO_TYPE, throttle_rate=0, queue_length=1)
            self._info_subscribed = True
        # The base adapter would advertise the gripper as a Float64MultiArray *topic*.
        # It is an action, commanded by `_command_gripper`, so it is hidden while the base
        # initialises and restored after: it is what gives the action space its sixth
        # dimension and the observation its jaw state.
        gripper, self.gripper_topic = self.gripper_topic, None
        try:
            super()._ensure_initialized()
        finally:
            self.gripper_topic = gripper


def _camera_clause(name: str) -> str:
    x, y, z = SCENE_CAMERA_POSES[name]
    tilt = SCENE_CAMERA_TILT_DEG[name]
    where = "above the table looking down" if tilt > 30 else "low and near-horizontal"
    return f"'{name}' at ({x:.3f}, {y:.3f}, {z:.3f}), {where} at {tilt:g} degrees below horizontal"


# Built from `ros_settings` rather than typed, so it cannot go stale beside the table.
_DOCS = (
    "SO-101 behind rosbridge. Six absolute joint-position commands in the order "
    "shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper. Arm joints "
    "are radians within the MJCF limits; the gripper is gripper_joint in radians, 0 at the "
    "closed stop. The arm base is the origin, +x points across the table, and the table "
    "top is the z=0 plane. Two fixed 640x480 cameras watch the workspace: "
    + _camera_clause("overhead") + ", and " + _camera_clause("side")
    + "; a third, 'wrist', rides the gripper."
)


def so101_ros(**kwargs: Any) -> SO101RosEmbodiment:
    """Registry entry point; CLI ``-E key=value`` strings are coerced here."""
    numeric = {"control_hz", "obs_timeout_s", "staleness_s", "fresh_obs_timeout_s"}
    booleans = {"simulated"}
    fields = {f.name for f in dataclasses.fields(RosSettings)}
    settings: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key not in fields:
            raise ValueError(f"so101_ros takes no option {key!r}; known: {sorted(fields)}")
        if key in numeric:
            value = float(value)
        elif key in booleans and not isinstance(value, bool):
            value = str(value).lower() in ("1", "true", "yes")
        elif key == "views" and isinstance(value, str):
            value = tuple(p.strip() for p in value.split(",") if p.strip()) or DEFAULT_VIEWS
        settings[key] = value
    return SO101RosEmbodiment(RosSettings(**settings))
