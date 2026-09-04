"""The SO-101 embodiment: ``inspect-robots-ros`` plus this task's extras.

Subclasses the upstream
[`RosEmbodiment`][inspect_robots_ros.embodiment.RosEmbodiment] rather than
reimplementing it, so the arm command path, the gripper publish, joint-state
freshness gating, staleness bounds, camera decoding and the rate preflight all
stay the upstream plugin's. The whole six-dimensional action vector — five arm
joints on ``/joint_trajectory_controller/joint_trajectory`` and the jaw on
``/gripper_controller/commands`` — is plain topic publishing that the adapter
already does, so nothing here touches the transport.

Two things are added on top, both of which the upstream adapter cannot do
because they are task knowledge:

1. **Grading from the overhead frame.** The verdict is computed by
   [`vision_success`][robot_console.arm.vision_success] from the image in the
   observation this step just returned, and folded into the
   [`StepResult`][inspect_robots.types.StepResult]'s ``info`` that the scorers read
   back out of the log. ``/free_joint_publisher/free_joint_states`` is subscribed
   alongside it and recorded as a reference the camera can be audited against; it
   grades nothing.
2. **Termination on a *held* success.** ``CONTRACT.md`` section 5 clause 4
   requires the apple to be at rest on the plate for at least 1.0 s of
   simulated time. The episode therefore keeps stepping while the apple merely
   *touches down*, and terminates only once the hold is complete — see
   [`HoldTracker`][robot_console.arm.success.HoldTracker]. Ending at first contact
   would both contradict the contract's physics and, because the recorded
   trajectory would then contain a single true step, make the offline scorer
   read a genuine success back as a failure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from inspect_robots import Action, Observation, Scene, StepResult
from inspect_robots_ros.embodiment import RosEmbodiment

from robot_console.arm.ros_client import HeaderStampingClient
from robot_console.arm.ros_settings import (
    OVERHEAD_CAMERA_NAME,
    SCENE_CAMERA_POSES,
    SCENE_CAMERA_TILT_DEG,
    FREE_JOINT_STATES_TYPE,
    RosSettings,
)
from robot_console.arm.vision_success import VisionTracker
from robot_console.arm.success import (
    GEOMETRIC_SUCCESS_KEY,
    HELD_KEY,
    HOLD_ELAPSED_KEY,
    HOLD_SECONDS,
    STAMP_KEY,
    HoldTracker,
    PlateGoal,
    success_info,
)

_OBJECT_SUBSCRIPTION_ID = "robot-control-free-joint-states"

_CONNECT_HINT = (
    "could not connect to rosbridge at {url}. Start the simulator and its rosbridge "
    "with: ros2 launch so_arm_mujoco sim.launch.py (rosbridge_server is included on "
    "port 9090)"
)


class SO101RosEmbodiment(RosEmbodiment):
    """The simulated SO-101 over rosbridge, with apple-on-plate success polling."""

    def __init__(
        self,
        settings: RosSettings | None = None,
        *,
        goal: PlateGoal | None = None,
        terminate_on_success: bool = True,
        hold_seconds: float = HOLD_SECONDS,
        expose_measured: bool = False,
        **overrides: Any,
    ) -> None:
        self.settings = settings or RosSettings(**overrides)
        self.goal = goal or PlateGoal()
        self.terminate_on_success = bool(terminate_on_success)
        # Off by default, and deliberately so. When on, the polled apple state
        # is copied into ``observation.extra["measured"]`` so a *supervisor*
        # can gate phase transitions on measured world state (jaw centre
        # through FK against the apple pose topic) instead of on a model's
        # account of a camera frame -- the rule in CLAUDE.md. A policy that
        # reads it is no longer "the model alone" and must be reported as a
        # separate policy row; see ``molmoact_supervisor``.
        self.expose_measured = bool(expose_measured)
        super().__init__(**self.settings.base_kwargs())
        # Replace the transport before it ever connects: construction is
        # network-free, so swapping the client here costs nothing. The base
        # adapter builds arm trajectories without a ``header``, and this
        # simulator's JointTrajectoryController silently ignores those -- see
        # robot_console.arm.ros_client for the measurements.
        self._client = HeaderStampingClient(
            self.url,
            stamped_topics=(self.settings.command_topic,),
            clock=self._clock,
            sleep=self._sleep,
        )
        # No PRIVILEGED_SUCCESS: the verdict is read off the same overhead frame the
        # policy is given, so the grader has no view of the scene the policy lacks.
        self.info = _with_docs(self.info, docs=_DOCS)
        self._hold = HoldTracker(hold_seconds, control_hz=self.settings.control_hz)
        self._monitors_subscribed = False
        self._vision = VisionTracker(hold_seconds=hold_seconds)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Reset through the base adapter, then clear cached success state."""
        self._vision.reset()
        self._hold.reset()
        return super().reset(scene, seed=seed)

    def step(self, action: Action) -> StepResult:
        """Take one base-adapter step, then attach the polled success verdicts.

        ``terminated`` is deliberately *not* raised on the first step where the
        apple is instantaneously on the plate. It is raised once the hold is
        complete, so the recorded trajectory ends with a run of true steps long
        enough for the offline scorer to re-derive the same verdict.
        """
        result = super().step(action)
        info = self._poll_success(result.observation)
        held = self._hold.update(placed=bool(info[GEOMETRIC_SUCCESS_KEY]), stamp=info[STAMP_KEY])
        info[HELD_KEY] = held
        info[HOLD_ELAPSED_KEY] = round(self._hold.elapsed, 4)
        terminated = held and self.terminate_on_success
        observation = result.observation
        if self.expose_measured:
            # Reserved rollout keys (env_step, approvals, operator_messages)
            # are injected later by the rollout and never set here.
            observation = replace(
                observation, extra={**dict(observation.extra), "measured": dict(info)}
            )
        return StepResult(
            observation=observation,
            reward=1.0 if held else 0.0,
            terminated=terminated,
            termination_reason="success" if terminated else None,
            truncated=result.truncated,
            info=info,
        )

    # -- success polling ---------------------------------------------------
    def _poll_success(self, observation: Observation) -> dict[str, Any]:
        """Grade this step from the overhead frame, and record the pose as reference."""
        position, speed, stamp = self._apple_state()
        verdict = self._see(observation, stamp)
        return success_info(
            position,
            goal=self.goal,
            placed=verdict.placed if verdict is not None else False,
            distance=verdict.reading.distance_m if verdict is not None else None,
            apple_speed=speed,
            stamp=stamp,
        )

    def _see(self, observation: Observation, stamp: float | None):
        """Fold the overhead frame into the vision verdict, or None if there was none.

        The frame comes from the observation rather than a private subscription, so the
        grader is looking at *the* image the policy acted on -- not a second one fetched
        a beat later, which would let the two disagree about what the world looked like.
        Images arrive RGB and OpenCV wants BGR; the reversal is the whole conversion.
        """
        images = getattr(observation, "images", None) or {}
        frame = images.get(OVERHEAD_CAMERA_NAME)
        if frame is None:
            return None
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            return None
        return self._vision.update(np.ascontiguousarray(array[:, :, ::-1]), stamp)

    def _apple_state(self) -> tuple[np.ndarray | None, float | None, float | None]:
        """Return the apple's world position, linear speed and simulated stamp.

        The parse itself is ``apple_state_from``, at module level, so that the
        preflight in ``scripts/scene_reset.py`` reads the apple through exactly
        this code and cannot disagree with the episode it is clearing the way
        for.
        """
        sample = self._client.latest(self.settings.object_state_topic)
        if sample is None:
            return None, None, None
        return apple_state_from(sample.msg, self.settings.object_body)

    # -- transport ---------------------------------------------------------
    def _all_topics(self) -> tuple[str, ...]:
        """Include the monitor topics so reset waits for them like any other."""
        return (*super()._all_topics(), *self._monitor_topics())

    def _monitor_topics(self) -> tuple[str, ...]:
        return (self.settings.object_state_topic,)

    def _ensure_initialized(self) -> None:
        """Subscribe the monitor topics before the base adapter waits on them."""
        if self._initialized:
            return
        if not self._monitors_subscribed:
            try:
                self._client.connect()
            except Exception as exc:
                raise ConnectionError(_CONNECT_HINT.format(url=self.url)) from exc
            self._client.subscribe(
                self.settings.object_state_topic,
                subscription_id=_OBJECT_SUBSCRIPTION_ID,
                message_type=FREE_JOINT_STATES_TYPE,
                throttle_rate=max(1, round(1000.0 / self.settings.control_hz)),
                queue_length=1,
            )
            self._monitors_subscribed = True
        super()._ensure_initialized()
        self._sim_clock_preflight()

    def _sim_clock_preflight(self) -> None:
        """Refuse to start an episode against a simulator whose clock is stopped.

        Measures simulated time against wall time over one short window, using
        the ``/joint_states`` header stamps the simulator publishes under
        ``use_sim_time``. Everything still *publishes* when the clock stalls, so
        without this the run completes, scores zero and looks like a policy bug.
        """
        threshold = self.settings.min_real_time_factor
        if threshold <= 0:
            return
        window = 1.0
        first = self._joint_state_stamp()
        if first is None:
            return
        wall_start = self._clock()
        self._sleep(window)
        second = self._joint_state_stamp()
        if second is None:
            return
        wall = self._clock() - wall_start
        if wall <= 0:
            return
        factor = (second - first) / wall
        if factor < threshold:
            raise RuntimeError(
                f"simulated time is advancing at {factor:.4f}x wall clock, below "
                f"min_real_time_factor={threshold:g}. The arm's "
                f"JointTrajectoryController interpolates against this clock, so it "
                f"will hold its pose while the gripper still responds. Check that "
                f"the simulator is not paused "
                f"(/mujoco_ros2_control_node/set_pause) and restart it if needed; "
                f"'docker restart' is the reliable reset."
            )

    def _joint_state_stamp(self) -> float | None:
        """Simulated timestamp of the latest ``/joint_states`` message."""
        sample = self._client.latest(self.settings.joint_states_topic)
        if sample is None:
            return None
        return _stamp_seconds(sample.msg.get("header"))


def _nested(msg: Any, *keys: str) -> Any:
    """Walk a chain of mapping keys, returning None at the first miss."""
    current = msg
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _stamp_seconds(header: Any) -> float | None:
    """Convert a ROS 2 ``header.stamp`` to float seconds, or None if absent.

    ``task_manager`` runs with ``use_sim_time: true`` and reads the same stamp,
    so this is simulated time — which is what ``CONTRACT.md`` section 5 clause 4
    measures the hold in.
    """
    stamp = _nested(header, "stamp")
    if not isinstance(stamp, Mapping):
        return None
    try:
        return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
    except (KeyError, TypeError, ValueError):
        return None


def _position(entry: Mapping[str, Any]) -> np.ndarray | None:
    """Read ``pose.pose.position`` out of one ``FreeJointState`` entry."""
    position = _nested(entry, "pose", "pose", "position")
    if not isinstance(position, Mapping):
        return None
    try:
        return np.asarray(
            [float(position["x"]), float(position["y"]), float(position["z"])],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _linear_speed(entry: Mapping[str, Any]) -> float | None:
    """Read ``twist.twist.linear`` and return its magnitude, or None."""
    linear = _nested(entry, "twist", "twist", "linear")
    if not isinstance(linear, Mapping):
        return None
    try:
        return float(np.linalg.norm([float(linear["x"]), float(linear["y"]), float(linear["z"])]))
    except (KeyError, TypeError, ValueError):
        return None


def apple_state_from(
    msg: Mapping[str, Any], body: str
) -> tuple[np.ndarray | None, float | None, float | None]:
    """One body's world position, linear speed and stamp from a ``FreeJointStateArray``.

    Public because the preflight in ``scripts/scene_reset.py`` must read the
    apple the same way the episode does: a preflight that parsed the pose
    slightly differently could pass on a world the run then measures as
    something else, which is worse than no preflight. Two other copies of this
    walk already exist (``scripts/live_viewer.py`` and a JavaScript one in
    ``scripts/live_cameras.html``); this is the one the score depends on.

    Entries are matched **by body name** — the array's order is not guaranteed.
    Any component the message does not carry comes back as ``None`` rather than
    as a plausible-looking default.
    """
    entries = msg.get("free_joints")
    if not isinstance(entries, list):
        return None, None, None
    stamp = _stamp_seconds(msg.get("header"))
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("name") != body:
            continue
        if stamp is None:
            stamp = _stamp_seconds(_nested(entry, "pose", "header"))
        return _position(entry), _linear_speed(entry), stamp
    return None, None, stamp


def _camera_clause(name: str) -> str:
    x, y, z = SCENE_CAMERA_POSES[name]
    tilt = SCENE_CAMERA_TILT_DEG[name]
    where = "above the table looking down" if tilt > 30 else "low and near-horizontal"
    return f"'{name}' at ({x:.3f}, {y:.3f}, {z:.3f}), {where} at {tilt:g} degrees below horizontal"


# Built from `ros_settings.SCENE_CAMERA_POSES` rather than typed, because the typed version
# went stale: it kept the overhead camera's previous pose for weeks after the simulator
# moved it. A test pins the table to what the simulator actually stages.
_DOCS = (
    "Simulated SO-101 behind rosbridge. Six absolute joint-position commands in "
    "the order shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, "
    "gripper. Arm joints are radians with MJCF limits; the gripper is the jaw "
    "hinge in radians, 0 fully closed and 1 fully open (measured aperture 1.5 mm "
    "to 43 mm). The arm base is the world origin, +x points across the table, and "
    "the table top is the z=0 plane: a wood work surface carrying, besides the red "
    "apple and the white plate, a bowl, a mug, a banana and a lemon that are scenery "
    "and not part of the task. Two fixed 640x480 cameras watch the workspace: "
    + _camera_clause("overhead") + ", and " + _camera_clause("side") + "."
)


def _with_docs(info: Any, *, docs: str) -> Any:
    """Return a copy of an ``EmbodimentInfo`` carrying this embodiment's docs.

    It used to add ``PRIVILEGED_SUCCESS`` as well. That capability was an honest
    declaration while the verdict came off ``/task_success``: it told the framework the
    grader could see something the policy could not. The verdict is now read from the
    same overhead frame the policy is handed, so the declaration would be false.
    """
    import dataclasses

    return dataclasses.replace(
        info, capabilities=frozenset(info.capabilities), docs=docs
    )


def so101_ros(**kwargs: Any) -> SO101RosEmbodiment:
    """Registry entry point; CLI ``-E key=value`` strings are coerced here."""
    numeric = {
        "control_hz",
        "obs_timeout_s",
        "staleness_s",
        "hold_seconds",
        # How long a step waits for an observation newer than the command it just sent.
        # It defaults to 2/control_hz, which is a *rate* assumption dressed up as a
        # freshness one: a simulator rendering several cameras inside its physics loop
        # publishes slower than the control rate, and a VLA doing seconds of inference
        # between steps makes that worse. Exposing it means a scene can say how long it
        # is actually willing to wait instead of the caller having to lower control_hz.
        "fresh_obs_timeout_s",
    }
    integers = {"camera_width", "camera_height"}
    booleans = {"simulated", "terminate_on_success"}
    settings_fields = {field.name for field in _settings_fields()}
    settings_kwargs: dict[str, Any] = {}
    other: dict[str, Any] = {}
    for key, value in kwargs.items():
        coerced = value
        if key in numeric:
            coerced = float(value)
        elif key in integers:
            coerced = int(value)
        elif key in booleans:
            coerced = (
                value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
            )
        elif key in ("camera_topic", "reset_service") and value in ("", "none", "None"):
            # ``-E camera_topic=`` is how the CLI says "run without images".
            coerced = None
        if key in settings_fields:
            settings_kwargs[key] = coerced
        else:
            other[key] = coerced
    return SO101RosEmbodiment(RosSettings(**settings_kwargs), **other)


def _settings_fields() -> tuple[Any, ...]:
    import dataclasses

    return dataclasses.fields(RosSettings)
