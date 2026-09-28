"""ROS wiring for the SO-101 and the worktop rig, in one place.

The names, types and joints are the SO-101's official ROS 2 interface as
``robots_specs/so101/ros2.yml`` records it (the community ``so_arm101_description``
bringup with ``hardware_type:=real``, plus ``usb_cam`` in ``/wrist``), and the
workspace-owned ``/reset`` and ``/scene`` rig from the simulator's spec §3. The
simulator's contract modules (``simulator/shared/ros_surfaces/so101.py`` and
``simulator/shared/tasks/apple_on_plate.py``) transcribe the same facts;
the workspace parity tests (``tests/test_contract_parity.py`` at the workspace root)
hold the two sides equal, because this project cannot import that one.

Everything the console consumes is here and nothing else: no topic outside the official
interface, the composed ``/reset`` and the rig is named in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot_console.arm.kinematics import ARM_JOINTS, GRIPPER_JOINT, JOINT_LIMITS
from robot_console.topics import namespaced

#: rosbridge websocket the console connects to by default.
DEFAULT_URL = "ws://127.0.0.1:9090"

# ------------------------------------------------------------------ the SO-101

#: ``joint_trajectory_controller`` drives exactly the five arm joints.
ARM_COMMAND_TOPIC = "/joint_trajectory_controller/joint_trajectory"
ARM_COMMAND_TYPE = "trajectory_msgs/msg/JointTrajectory"

#: ``joint_state_broadcaster`` publishes all six joints on this one topic. Its ``name``
#: array is **alphabetically sorted**, not in contract order, so every consumer indexes
#: it by name.
JOINT_STATES_TOPIC = "/joint_states"
JOINT_STATES_TYPE = "sensor_msgs/msg/JointState"

#: ``gripper_controller`` is a ``parallel_gripper_action_controller/GripperActionController``
#: on ``gripper_joint``: an action server, goal ``command.position[0]`` in radians.
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"
GRIPPER_ACTION_TYPE = "control_msgs/action/ParallelGripperCommand"

#: ``robot_state_publisher``'s tree. Nothing here consumes it -- ``kinematics.py`` does
#: its own FK -- but an arm that publishes no frames is one a standard client cannot draw,
#: so the wire check requires it.
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
TF_TYPE = "tf2_msgs/msg/TFMessage"

#: The eye-in-hand view: ``usb_cam`` in ``/wrist``, 640x480 at 30 Hz, its compressed
#: stream on ``image_transport``'s ``/wrist/image_raw/compressed``.
WRIST_CAMERA_NAME = "wrist"
WRIST_CAMERA_TOPIC = "/wrist/image_raw/compressed"
WRIST_CAMERA_TYPE = "sensor_msgs/msg/CompressedImage"
WRIST_CAMERA_WIDTH = 640
WRIST_CAMERA_HEIGHT = 480

#: The workspace-owned reset (spec §3): ``std_srvs/srv/Trigger`` provided by
#: ``/simulator``, composed with the SO-101's namespace. It restores the staged world and
#: the controllers, aborts outstanding goals, and answers once observations of the reset
#: world are out. It exists only in simulation, which is what makes it the thing the arm
#: task refuses a wire without.
RESET_SERVICE = "/reset"
RESET_SERVICE_TYPE = "std_srvs/srv/Trigger"

# ------------------------------------------------------------------ the worktop rig

#: The rig is not the robot's, so the robot's namespace is not its own: it publishes
#: under this one, with or without an arm bolted beside it.
SCENE_NAMESPACE = "scene"
#: The rig's root frame, ``scene/worktop``: the arm base frame its constants are in.
SCENE_ROOT_FRAME = "worktop"

OVERHEAD_CAMERA_NAME = "overhead"
OVERHEAD_CAMERA_TOPIC = "/overhead/color/compressed"
OVERHEAD_CAMERA_INFO_TOPIC = "/overhead/color/camera_info"
OVERHEAD_CAMERA_TYPE = "sensor_msgs/msg/CompressedImage"
OVERHEAD_CAMERA_WIDTH = 640
OVERHEAD_CAMERA_HEIGHT = 480

SIDE_CAMERA_NAME = "side"
SIDE_CAMERA_TOPIC = "/side/color/compressed"
SIDE_CAMERA_INFO_TOPIC = "/side/color/camera_info"
SIDE_CAMERA_TYPE = "sensor_msgs/msg/CompressedImage"
SIDE_CAMERA_WIDTH = 640
SIDE_CAMERA_HEIGHT = 480

CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"
#: The rig's calibrated transforms, under its own namespace.
SCENE_TF_STATIC_TOPIC = "/tf_static"

#: The rig's mount poses, **duplicated exactly** from the simulator task's
#: ``SCENE_CAMERAS`` (spec §3): name -> (position, MuJoCo ``xyaxes``, vertical field of
#: view in degrees, (width, height)), all in the arm base frame (``scene/worktop``).
#: ``xyaxes`` is image-right then image-up, as vectors in that frame. This is rig
#: calibration -- where the cameras are bolted -- not episode state, and it is what the
#: camera-verdict scorer triangulates through, so a stale copy here is a wrong verdict.
SCENE_CAMERAS: dict[str, tuple[tuple[float, float, float], tuple[float, ...], float,
                               tuple[int, int]]] = {
    OVERHEAD_CAMERA_NAME: (
        (0.677, 0.000, 0.888),
        (0.00000, 1.00000, 0.00000, -0.88295, 0.00000, 0.46947),
        45.0,
        (640, 480),
    ),
    SIDE_CAMERA_NAME: (
        (0.265, 1.110, 0.161),
        (-0.99892, -0.04646, 0.00000, 0.00456, -0.09815, 0.99516),
        45.0,
        (640, 480),
    ),
}
#: Views of the table above, as the prose a policy is told reads them.
SCENE_CAMERA_POSES: dict[str, tuple[float, float, float]] = {
    name: spec[0] for name, spec in SCENE_CAMERAS.items()
}
#: Down-tilt of each scene camera, degrees below horizontal, for the policy's docs.
SCENE_CAMERA_TILT_DEG: dict[str, float] = {"overhead": 62.0, "side": 5.6}
#: The rig's frame rate, the simulator task's ``SCENE_CAMERA_HZ``.
SCENE_CAMERA_HZ = 10.0

#: Which views belong to the rig rather than to the arm.
SCENE_CAMERA_NAMES: frozenset[str] = frozenset({OVERHEAD_CAMERA_NAME, SIDE_CAMERA_NAME})
#: rig view -> its ``camera_info`` topic, bare.
SCENE_CAMERA_INFO_TOPICS: dict[str, str] = {
    OVERHEAD_CAMERA_NAME: OVERHEAD_CAMERA_INFO_TOPIC,
    SIDE_CAMERA_NAME: SIDE_CAMERA_INFO_TOPIC,
}

#: Every camera the embodiment can subscribe, ``name -> (topic, width, height)``.
CAMERA_SPECS: dict[str, tuple[str, int, int]] = {
    OVERHEAD_CAMERA_NAME: (OVERHEAD_CAMERA_TOPIC, OVERHEAD_CAMERA_WIDTH, OVERHEAD_CAMERA_HEIGHT),
    SIDE_CAMERA_NAME: (SIDE_CAMERA_TOPIC, SIDE_CAMERA_WIDTH, SIDE_CAMERA_HEIGHT),
    WRIST_CAMERA_NAME: (WRIST_CAMERA_TOPIC, WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT),
}


def camera_topic(name: str, topic: str, namespace: str) -> str:
    """`topic` under whichever namespace owns that camera: the rig's, or the robot's."""
    return namespaced(topic, SCENE_NAMESPACE if name in SCENE_CAMERA_NAMES else namespace)


def rig_topic(topic: str) -> str:
    """A bare rig name as it reaches the wire, under ``/scene``."""
    return namespaced(topic, SCENE_NAMESPACE)


def arm_interface(namespace: str) -> dict[str, dict[str, str]]:
    """The SO-101's typed interface the arm task uses, composed with ``namespace``.

    ``{"topics": {name: type}, "services": {...}, "actions": {...}}``. The wire check
    compares this against what ``rosapi`` reports, type for type.
    """
    return {
        "topics": {
            namespaced(JOINT_STATES_TOPIC, namespace): JOINT_STATES_TYPE,
            namespaced(ARM_COMMAND_TOPIC, namespace): ARM_COMMAND_TYPE,
            namespaced(WRIST_CAMERA_TOPIC, namespace): WRIST_CAMERA_TYPE,
            namespaced(TF_TOPIC, namespace): TF_TYPE,
            namespaced(TF_STATIC_TOPIC, namespace): TF_TYPE,
        },
        "services": {namespaced(RESET_SERVICE, namespace): RESET_SERVICE_TYPE},
        "actions": {namespaced(GRIPPER_ACTION, namespace): GRIPPER_ACTION_TYPE},
    }


def namespaced_reset(namespace: str) -> str:
    """The composed ``/reset`` for the SO-101 under ``namespace``."""
    return namespaced(RESET_SERVICE, namespace)


def rig_interface() -> dict[str, str]:
    """The rig's typed topics, under ``/scene``."""
    return {
        rig_topic(OVERHEAD_CAMERA_TOPIC): OVERHEAD_CAMERA_TYPE,
        rig_topic(OVERHEAD_CAMERA_INFO_TOPIC): CAMERA_INFO_TYPE,
        rig_topic(SIDE_CAMERA_TOPIC): SIDE_CAMERA_TYPE,
        rig_topic(SIDE_CAMERA_INFO_TOPIC): CAMERA_INFO_TYPE,
        rig_topic(SCENE_TF_STATIC_TOPIC): TF_TYPE,
    }


#: The embodiment's default views, in slot order: the two rig views MolmoAct2 takes
#: positionally, then the wrist.
DEFAULT_VIEWS: tuple[str, ...] = (OVERHEAD_CAMERA_NAME, SIDE_CAMERA_NAME, WRIST_CAMERA_NAME)


@dataclass(frozen=True)
class RosSettings:
    """Everything the ``so101_ros`` embodiment needs to talk to an SO-101 and the rig."""

    url: str = DEFAULT_URL
    #: The SO-101's ROS namespace. The fields below stay bare -- they are the record of
    #: the official interface -- and the prefix is applied where a name reaches the wire
    #: (`topic()`, `cameras()`). `""` is the bare single-robot contract.
    namespace: str = "so101"
    ros_version: int = 2
    joints: tuple[str, ...] = ARM_JOINTS
    joint_states_topic: str = JOINT_STATES_TOPIC
    command_topic: str = ARM_COMMAND_TOPIC
    gripper_joint: str = GRIPPER_JOINT
    gripper_action: str = GRIPPER_ACTION
    #: Views to subscribe, in slot order. Order is load-bearing for a policy that takes
    #: views positionally, so this is a tuple, not a set. The two rig views are always
    #: subscribed whether listed or not: the scorer grades from them.
    views: tuple[str, ...] = DEFAULT_VIEWS
    #: The workspace-owned ``/reset``: world, controllers and task timers together.
    reset_service: str = RESET_SERVICE
    control_hz: float = 10.0
    obs_timeout_s: float = 10.0
    #: How long a step waits for joint state newer than the command it just sent. `None`
    #: keeps the adapter's 2/control_hz, a rate assumption that a simulator rendering
    #: cameras beside its physics, with a VLA thinking between steps, does not meet.
    fresh_obs_timeout_s: float | None = None
    staleness_s: float = 3.0
    simulated: bool = True
    name: str = "so101_ros"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(self.views)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate view in {names!r}")
        unknown = [name for name in names if name not in CAMERA_SPECS]
        if unknown:
            raise ValueError(f"unknown camera view(s) {unknown}; known: {sorted(CAMERA_SPECS)}")
        object.__setattr__(self, "views", names)

    @property
    def action_low(self) -> tuple[float, ...]:
        return tuple(JOINT_LIMITS[name][0] for name in self.joints)

    @property
    def action_high(self) -> tuple[float, ...]:
        return tuple(JOINT_LIMITS[name][1] for name in self.joints)

    def topic(self, name: str) -> str:
        """One of the SO-101's names as it goes on the wire. Idempotent."""
        return namespaced(name, self.namespace)

    def camera_views(self) -> tuple[str, ...]:
        """The subscribed views: `views`, then any rig view it left out."""
        extra = tuple(v for v in (OVERHEAD_CAMERA_NAME, SIDE_CAMERA_NAME) if v not in self.views)
        return (*self.views, *extra)

    def cameras(self) -> dict[str, tuple[str, int, int]]:
        """Camera map in the upstream adapter's ``name -> (topic, height, width)`` form."""
        out: dict[str, tuple[str, int, int]] = {}
        for name in self.camera_views():
            topic, width, height = CAMERA_SPECS[name]
            out[name] = (camera_topic(name, topic, self.namespace), int(height), int(width))
        return out

    def camera_info_topics(self) -> dict[str, str]:
        """rig view -> its composed ``camera_info`` topic."""
        return {name: rig_topic(topic) for name, topic in SCENE_CAMERA_INFO_TOPICS.items()}

    def base_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the upstream ``RosEmbodiment`` constructor.

        The gripper is declared so the action space stays six-dimensional and
        ``joint_pos`` folds in the measured jaw angle. It is commanded through the
        embodiment's own action client, never as a topic: see ``SO101RosEmbodiment``.
        """
        gripper_low, gripper_high = JOINT_LIMITS[self.gripper_joint]
        kwargs: dict[str, Any] = {
            "url": self.url,
            "ros_version": self.ros_version,
            "joints": self.joints,
            "joint_states_topic": self.topic(self.joint_states_topic),
            "command_topic": self.topic(self.command_topic),
            "command_type": "joint_trajectory",
            "action_low": self.action_low,
            "action_high": self.action_high,
            "cameras": self.cameras(),
            "control_hz": self.control_hz,
            "reset_service": self.topic(self.reset_service),
            "obs_timeout_s": self.obs_timeout_s,
            "fresh_obs_timeout_s": self.fresh_obs_timeout_s,
            # Every rig frame, not one per control period: the scorer needs the frames
            # between steps too, and a VLA's steps are seconds apart.
            "camera_throttle_ms": 0,
            "staleness_s": self.staleness_s,
            "simulated": self.simulated,
            "name": self.name,
            "gripper_topic": self.topic(self.gripper_action),
            "gripper_joint": self.gripper_joint,
            "gripper_low": gripper_low,
            "gripper_high": gripper_high,
            "gripper_closed_at": "low",
        }
        kwargs.update(self.extra)
        return kwargs
