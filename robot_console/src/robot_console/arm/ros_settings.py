"""ROS wiring for the simulated SO-101, in one place.

Every topic, service, message type and joint bound here is copied from
``simulator/INTERFACE.md``, which the simulator role wrote from real
``ros2 topic list -t`` / ``ros2 service list -t`` / ``ros2 control
list_controllers`` output captured **inside the running container**. Where
``CONTRACT.md`` and ``INTERFACE.md`` disagree, ``INTERFACE.md`` wins.

Sources, per item:

* joints and their limits — ``ros2_so_arm/so_arm101_description/mjcf/so_arm101.xml``
* arm and gripper controllers — ``so_arm_mujoco/config/ros2_controllers.yaml``,
  plus ``so_arm_mujoco/launch/sim.launch.py``. **Not**
  ``so_arm101_description/control/ros2_controllers.yaml``: that file is a
  pristine upstream clone and the simulator no longer loads it. The two differ
  in exactly the way that matters here — the shipped file makes
  ``gripper_controller`` a ``forward_command_controller/ForwardCommandController``
  rather than an action controller.
* ``/task_success`` and ``/reset`` — ``so_arm_mujoco/so_arm_mujoco/task_manager.py``
* ``/free_joint_publisher/free_joint_states`` and ``/mujoco_ros2_control_node/*``
  — ``mujoco_ros2_control_plugins`` and ``mujoco_ros2_control_msgs``, with the
  names as they actually resolve at runtime (see the namespacing note below)
* ``/overhead/color/compressed`` — the ``image_transport republish`` node in
  ``sim.launch.py``, whose output is remapped to this contract-facing name

**Plugin topics are namespaced by their key in
``so_arm_mujoco/config/mujoco_plugins.yaml``.** Plugins are constructed with
``get_node()->create_sub_node(plugin_name)``, so the ``free_joint_publisher``
key becomes a ROS sub-namespace and the topic is
``/free_joint_publisher/free_joint_states``, not ``/free_joint_states``. The
same rule applies to the camera plugin, whose *raw* image is on
``/camera_publisher/overhead/color``; only the republished compressed stream
carries the contract name. Renaming a key in that YAML renames the topics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot_console.arm.kinematics import ARM_JOINTS, GRIPPER_JOINT, JOINT_LIMITS

#: rosbridge websocket the simulator exposes.
DEFAULT_URL = "ws://127.0.0.1:9090"

#: ``joint_trajectory_controller`` in ``so_arm_mujoco/config/ros2_controllers.yaml``
#: drives exactly the five arm joints; the gripper is on a separate controller.
ARM_COMMAND_TOPIC = "/joint_trajectory_controller/joint_trajectory"

#: ``joint_state_broadcaster`` publishes all six joints — five arm plus the jaw
#: — on this single topic at ~165 Hz, per ``CONTRACT.md`` section 3 constraint 3.
#: Its ``name`` array is **alphabetically sorted**, not in contract joint order,
#: so every consumer must index it by name.
JOINT_STATES_TOPIC = "/joint_states"

#: ``gripper_controller`` is a ``forward_command_controller/ForwardCommandController``
#: on ``gripper_joint``. It subscribes to ``~/commands``, so the sixth command
#: dimension is a plain ``std_msgs/msg/Float64MultiArray`` publish with a single
#: element in ``data``. There is no action server and no action client is needed.
GRIPPER_COMMAND_TOPIC = "/gripper_controller/commands"
GRIPPER_COMMAND_TYPE = "std_msgs/msg/Float64MultiArray"

#: ``task_manager`` publishes its verdict here on every free-joint message.
TASK_SUCCESS_TOPIC = "/task_success"
TASK_SUCCESS_TYPE = "std_msgs/msg/Bool"

#: ``FreeJointStatePublisherPlugin`` publishes the pose and twist of every body
#: in its ``body_names`` -- the apple and, since 02 Sep 2026, the four dressing
#: bodies too; select ``APPLE_BODY`` by name, never by position. With
#: no ``frame_id`` parameter set in ``config/mujoco_plugins.yaml``, and an empty
#: ``header.frame_id`` on the wire, poses are in the **world** frame. The
#: message field is ``free_joints``, not ``states``.
FREE_JOINT_STATES_TOPIC = "/free_joint_publisher/free_joint_states"
FREE_JOINT_STATES_TYPE = "mujoco_ros2_control_msgs/msg/FreeJointStateArray"
APPLE_BODY = "apple"

#: The overhead camera. The MuJoCo ``CameraPlugin`` publishes a raw
#: ``sensor_msgs/msg/Image`` on ``/camera_publisher/overhead/color``; an
#: ``image_transport republish`` node compresses it and its output is remapped
#: to the contract name below. 5 Hz, 1280x720, JPEG payload.
OVERHEAD_CAMERA_NAME = "overhead"
OVERHEAD_CAMERA_TOPIC = "/overhead/color/compressed"
OVERHEAD_CAMERA_TYPE = "sensor_msgs/msg/CompressedImage"
#: Measured on the wire 2026-08-30: both views publish 640x480. The overhead
#: camera was 1280x720 when INTERFACE.md section 3 was written; the simulator
#: has since changed it. The upstream adapter validates the first frame against
#: these numbers and raises on a mismatch, so they must track the sim.
OVERHEAD_CAMERA_WIDTH = 640
OVERHEAD_CAMERA_HEIGHT = 480

#: A second static third-person view. MolmoAct2 is trained on two *different*
#: views and consumes them positionally, so duplicating the overhead frame is
#: off-distribution input, not a harmless stand-in. Confirm the topic against
#: ``ros2 topic list`` before relying on it -- it is owned by the simulator.
SIDE_CAMERA_NAME = "side"
SIDE_CAMERA_TOPIC = "/side/color/compressed"
SIDE_CAMERA_TYPE = "sensor_msgs/msg/CompressedImage"
SIDE_CAMERA_WIDTH = 640
SIDE_CAMERA_HEIGHT = 480

#: The eye-in-hand view, riding ``gripper_link``. ``CONTRACT.md`` section 3 has
#: listed this topic since the beginning; nothing published it until the camera
#: was added to the arm MJCF, so a comment here used to say it did not exist.
#:
#: **It is not published unless the simulator was started with
#: ``./start_sim.sh --wrist``.** The camera is declared but disabled by default,
#: because the MuJoCo plugin renders inside the physics loop and rate falls for
#: *every* camera when another one is enabled: measured 4.23/4.15 Hz with it off
#: against 2.94/2.98/2.97 Hz with it on. Selecting this view against a sim
#: started without the flag fails at reset with a missing-topic timeout, which is
#: the intended loud failure -- an eye-in-hand policy fed a stale or absent wrist
#: frame is worse than one that refuses to start.
#:
#: 256x256, not 640x480 like the two scene cameras. The consuming policies resize
#: to 224, so 256 makes that a downsample rather than an upsample, and the
#: smaller frame is why enabling it costs ~29% of the camera rate instead of the
#: ~50% the older four-camera measurements would predict.
WRIST_CAMERA_NAME = "wrist"
WRIST_CAMERA_TOPIC = "/wrist/color/compressed"
WRIST_CAMERA_TYPE = "sensor_msgs/msg/CompressedImage"
WRIST_CAMERA_WIDTH = 256
WRIST_CAMERA_HEIGHT = 256

# History, so the gap in the record is not mistaken for an oversight: a
# ``trainlow``/``trainhigh`` pair was defined here and used as the default
# views. It measured strictly worse for MolmoAct2 (0/5, apple travel 0.000 m)
# and was deleted from the simulator's scene on 2026-08-31, along with an
# unused ``policylow``/``policyhigh`` pair. Those topics no longer exist, so
# the constants are gone too rather than left defined and unsubscribable:
# ``settings_for_views`` must reject those names, not wire them.

#: Every camera this simulator publishes, as ``name -> (topic, width, height)``,
#: derived from the constants above so there is exactly one definition of each.
#:
#: This exists so a caller can say *which views it wants* and get the matching
#: subscriptions, instead of naming views in one place and wiring topics in
#: another. The two have drifted apart before, with a script hard-coding one
#: pair of view names while ``RosSettings`` subscribed to another: the script
#: and the policy then silently disagreed about what the model was looking at.
#:
#: Only names in here are wirable, so this map is also what stops
#: ``settings_for_views`` accepting a camera the simulator does not publish.
CAMERA_SPECS: dict[str, tuple[str, int, int]] = {
    OVERHEAD_CAMERA_NAME: (
        OVERHEAD_CAMERA_TOPIC,
        OVERHEAD_CAMERA_WIDTH,
        OVERHEAD_CAMERA_HEIGHT,
    ),
    SIDE_CAMERA_NAME: (SIDE_CAMERA_TOPIC, SIDE_CAMERA_WIDTH, SIDE_CAMERA_HEIGHT),
    #: Selectable, but only published when the sim was started with --wrist. It is
    #: deliberately NOT in ``RosSettings.extra_cameras``' default, so no existing
    #: eval changes behaviour by its presence here.
    WRIST_CAMERA_NAME: (WRIST_CAMERA_TOPIC, WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT),
}


#: ``mujoco_ros2_control``'s own reset, which restores the state captured at
#: startup. The services live on the node named ``mujoco_ros2_control_node``
#: (``mujoco_system_interface.cpp``), **not** ``ros2_control_node``.
#:
#: ``task_manager``'s ``/reset`` (``std_srvs/srv/Trigger``) forwards to this
#: same service with an empty keyframe and no state overrides, and additionally
#: clears ``task_manager``'s own hold timer. It does **not** teleport the apple:
#: ``task_manager`` has no ``set_free_joint_state`` client at all, which
#: ``CONTRACT.md`` section 6 forbids anyway. Either service is a valid
#: ``reset_service``; ``/reset`` is the one that also resets ``/task_success``.
RESET_WORLD_SERVICE = "/mujoco_ros2_control_node/reset_world"
TASK_MANAGER_RESET_SERVICE = "/reset"


@dataclass(frozen=True)
class RosSettings:
    """Everything the ROS embodiment needs to talk to this simulator."""

    url: str = DEFAULT_URL
    ros_version: int = 2
    joints: tuple[str, ...] = ARM_JOINTS
    joint_states_topic: str = JOINT_STATES_TOPIC
    command_topic: str = ARM_COMMAND_TOPIC
    command_type: str = "joint_trajectory"
    gripper_joint: str = GRIPPER_JOINT
    gripper_mode: str = "topic"
    gripper_topic: str = GRIPPER_COMMAND_TOPIC
    success_topic: str = TASK_SUCCESS_TOPIC
    object_state_topic: str = FREE_JOINT_STATES_TOPIC
    object_body: str = APPLE_BODY
    camera_name: str = OVERHEAD_CAMERA_NAME
    #: Set to ``None`` to run without images. That clears ``extra_cameras`` too
    #: (see ``__post_init__``): "no primary camera" means no subscriptions at
    #: all, so ``--no-camera`` and ``-E camera_topic=`` cannot leave slot 1
    #: alive. The eval evidence policy wants a screenshot for every model
    #: tested, so the default is the live camera.
    camera_topic: str | None = OVERHEAD_CAMERA_TOPIC
    camera_width: int = OVERHEAD_CAMERA_WIDTH
    camera_height: int = OVERHEAD_CAMERA_HEIGHT
    #: Further cameras as ``(name, topic, width, height)``, appended after the
    #: primary one in declaration order. Order is load-bearing for policies that
    #: take views positionally, so this is a tuple, not a mapping.
    #:
    #: Defaults to the only pair the simulator publishes: slot 0 ``overhead``
    #: (primary, above), slot 1 ``side``. This matches ``molmoact.DEFAULT_VIEWS``
    #: so the dataclass default and the eval scripts agree without coordination.
    extra_cameras: tuple[tuple[str, str, int, int], ...] = (
        (
            SIDE_CAMERA_NAME,
            SIDE_CAMERA_TOPIC,
            SIDE_CAMERA_WIDTH,
            SIDE_CAMERA_HEIGHT,
        ),
    )
    #: ``/reset`` rather than ``reset_world``: it forwards to the same service
    #: **and** clears ``task_manager``'s own hold timer, so ``/task_success``
    #: starts each episode from a known state instead of relying on the apple's
    #: return to spawn to break a stale hold.
    reset_service: str | None = TASK_MANAGER_RESET_SERVICE
    control_hz: float = 10.0
    #: Minimum simulated-time-per-wall-second the simulator must be running at
    #: before an episode may start. A stalled or heavily throttled sim clock is
    #: silent and looks exactly like a broken policy: the
    #: ``JointTrajectoryController`` interpolates its goal against the ROS clock,
    #: so with ``use_sim_time`` and a stopped clock it never advances past the
    #: trajectory's first point and the arm holds its pose -- while the gripper's
    #: ``ForwardCommandController``, which has no time dependence at all, keeps
    #: working. "Gripper moves, arm frozen" is that failure, not a policy fault.
    #: Set to 0 to skip the check.
    min_real_time_factor: float = 0.10
    obs_timeout_s: float = 10.0

    #: How long a step waits for an observation *newer* than the command it just sent.
    #: `None` keeps the adapter's own default of 2/control_hz, which is a **rate**
    #: assumption dressed up as a freshness one: it only holds if the robot publishes
    #: faster than the control loop runs. A simulator rendering several cameras inside
    #: its physics loop does not -- measured here at 5-9 Hz against a 10 Hz control rate
    #: -- and a VLA doing seconds of inference between steps widens the gap further. The
    #: symptom is an episode dying part-way through with "EmbodimentFault: no
    #: post-publish joint state within fresh_obs_timeout_s=0.2s". Raising it does not
    #: weaken the guarantee: a stale observation is still refused, it is just given time
    #: to arrive.
    fresh_obs_timeout_s: float | None = None
    staleness_s: float = 3.0
    simulated: bool = True
    name: str = "so101_ros"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``camera_topic=None`` is the one way callers say "run without images"
        # (``rosbridge_eval.py --no-camera``, ``wire_probe.py --no-camera``,
        # ``-E camera_topic=`` through the registry entry point). Since
        # ``extra_cameras`` carries a default, honouring only the primary would
        # leave slot 1 subscribed and the run would still block for
        # ``obs_timeout_s`` on a camera the caller asked not to have. So no
        # primary means no cameras at all.
        if self.camera_topic is None and self.extra_cameras:
            object.__setattr__(self, "extra_cameras", ())
        if self.gripper_mode not in ("topic", "none"):
            raise ValueError(
                f"gripper_mode must be 'topic' or 'none', got {self.gripper_mode!r}. "
                "The simulator's gripper_controller is a ForwardCommandController on "
                f"{GRIPPER_COMMAND_TOPIC}; there is no action server to drive."
            )

    @property
    def action_low(self) -> tuple[float, ...]:
        """Lower arm-joint command bounds, straight from the MJCF ranges."""
        return tuple(JOINT_LIMITS[name][0] for name in self.joints)

    @property
    def action_high(self) -> tuple[float, ...]:
        """Upper arm-joint command bounds, straight from the MJCF ranges."""
        return tuple(JOINT_LIMITS[name][1] for name in self.joints)

    def cameras(self) -> dict[str, tuple[str, int, int]]:
        """Camera map in the upstream adapter's ``name -> (topic, height, width)`` form.

        Insertion order is the declaration order, which is what a policy taking
        views positionally depends on.
        """
        out: dict[str, tuple[str, int, int]] = {}
        if self.camera_topic is not None:
            out[self.camera_name] = (self.camera_topic, self.camera_height, self.camera_width)
        for name, topic, width, height in self.extra_cameras:
            if name in out:
                raise ValueError(f"duplicate camera name {name!r}")
            out[name] = (topic, int(height), int(width))
        return out

    def base_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the upstream ``RosEmbodiment`` constructor.

        The gripper is declared to the base adapter so the action space stays
        six-dimensional and ``joint_pos`` folds in the measured jaw angle. The
        adapter resolves ``/joint_states`` **by joint name**, so the
        simulator's alphabetical ordering is handled there and the action
        vector keeps the contract order end to end.
        """
        gripper_low, gripper_high = JOINT_LIMITS[self.gripper_joint]
        kwargs: dict[str, Any] = {
            "url": self.url,
            "ros_version": self.ros_version,
            "joints": self.joints,
            "joint_states_topic": self.joint_states_topic,
            "command_topic": self.command_topic,
            "command_type": self.command_type,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "cameras": self.cameras(),
            "control_hz": self.control_hz,
            "reset_service": self.reset_service,
            "obs_timeout_s": self.obs_timeout_s,
            "fresh_obs_timeout_s": self.fresh_obs_timeout_s,
            "staleness_s": self.staleness_s,
            "simulated": self.simulated,
            "name": self.name,
        }
        if self.gripper_mode != "none":
            kwargs.update(
                gripper_topic=self.gripper_topic,
                gripper_joint=self.gripper_joint,
                gripper_low=gripper_low,
                gripper_high=gripper_high,
                gripper_closed_at="low",
                gripper_command_type="float64_multi_array",
            )
        kwargs.update(self.extra)
        return kwargs


def settings_for_views(
    views: tuple[str, ...] | list[str],
    *,
    topic_overrides: dict[str, str] | None = None,
    **kwargs: Any,
) -> RosSettings:
    """Subscribe to exactly ``views``, in order: slot 0 primary, the rest extra.

    Policies that consume views **positionally** (MolmoAct2 does; its
    ``camera_keys`` is ``[]``) need the subscription order to be the view order,
    so this derives both from one list rather than letting a script state the
    names and a dataclass default state the topics.

    Unknown names raise: a typo'd view is otherwise a missing-topic timeout at
    reset, several minutes later and with a less useful message.
    """
    names = tuple(views)
    if not names:
        raise ValueError("at least one view is required")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate view in {names!r}")
    overrides = dict(topic_overrides or {})
    unknown = [name for name in names if name not in CAMERA_SPECS]
    if unknown:
        raise ValueError(
            f"unknown camera view(s) {unknown}; this simulator publishes "
            f"{sorted(CAMERA_SPECS)}"
        )
    resolved = [
        (name, overrides.get(name, CAMERA_SPECS[name][0]), *CAMERA_SPECS[name][1:])
        for name in names
    ]
    primary = resolved[0]
    return RosSettings(
        camera_name=primary[0],
        camera_topic=primary[1],
        camera_width=primary[2],
        camera_height=primary[3],
        extra_cameras=tuple(resolved[1:]),
        **kwargs,
    )
