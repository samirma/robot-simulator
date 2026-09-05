"""The console's names and the simulator's must be the same names.

Two projects, installed separately, that only ever meet on a websocket -- so nothing but
a test holds their halves of the contract together. These run against the simulator's own
source when it is checked out alongside, and skip when it is not, which is the same rule
`test_scene_geometry` uses for the scene numbers.

The one that matters most is the joint-name check. `/joint_states` comes back
alphabetically sorted, and for this arm the sorted order and the contract order share
*no* index -- so a positional read is wrong about every joint while looking entirely
plausible. Both sides have to agree that the sort happens and that names are the key.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from robot_console.arm.kinematics import ARM_JOINTS, GRIPPER_JOINT, JOINT_ORDER
from robot_console.topics import namespaced
from robot_console.arm.ros_settings import (
    ARM_COMMAND_TOPIC,
    FREE_JOINT_STATES_TOPIC,
    GRIPPER_COMMAND_TOPIC,
    JOINT_STATES_TOPIC,
    OVERHEAD_CAMERA_TOPIC,
    SIDE_CAMERA_TOPIC,
    TASK_MANAGER_RESET_SERVICE,
    WRIST_CAMERA_TOPIC,
)

_SIMULATOR = Path(__file__).resolve().parents[3] / "simulator" / "shared"
SURFACE = _SIMULATOR / "ros_surfaces" / "so101.py"
#: The simulator's copy of the namespacing rule. Deliberately a stdlib-only module on
#: that side, so it can be loaded here the same way the surface is.
NAMESPACE = _SIMULATOR / "contracts" / "namespace.py"


def _load(path: Path, name: str):
    if not path.exists():
        pytest.skip(f"sibling simulator checkout not present at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before it is executed, because `@dataclass` resolves its annotations
    # through `sys.modules[cls.__module__]` and raises on a module that is not there.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)  # stdlib and numpy only; no mujoco at import time
    except Exception:
        del sys.modules[name]
        raise
    return module


def _surface():
    return _load(SURFACE, "_so101_surface")


def _namespace():
    return _load(NAMESPACE, "_sim_namespace")


def test_every_topic_name_matches_the_simulators() -> None:
    s = _surface()
    assert s.TOPIC_ARM_COMMAND == ARM_COMMAND_TOPIC
    assert s.TOPIC_GRIPPER_COMMAND == GRIPPER_COMMAND_TOPIC
    assert s.TOPIC_JOINT_STATES == JOINT_STATES_TOPIC
    assert s.TOPIC_FREE_JOINT_STATES == FREE_JOINT_STATES_TOPIC
    assert s.SERVICE_RESET == TASK_MANAGER_RESET_SERVICE


def test_the_simulators_publish_no_success_topic() -> None:
    """Both engines' shared surface must not offer a task verdict on the wire.

    The verdict is inferred from the overhead camera by `arm.vision_success`. Publishing
    it as well would hand the grader a channel the policy cannot see, and the two would
    then be free to drift apart without anything noticing.
    """
    s = _surface()
    assert not hasattr(s, "TOPIC_TASK_SUCCESS")


def test_the_camera_topics_and_sizes_match() -> None:
    s = _surface()
    published = {**s.DEFAULT_CAMERAS, **s.WRIST_CAMERA}
    assert set(published) == {OVERHEAD_CAMERA_TOPIC, SIDE_CAMERA_TOPIC, WRIST_CAMERA_TOPIC}
    # The sizes are contract terms: the VLA's preprocessor stretches to 4:3 without
    # preserving aspect, and the wrist view is 256 so a policy resizing to 224
    # downsamples rather than upsamples.
    assert published[OVERHEAD_CAMERA_TOPIC][1:] == (640, 480)
    assert published[SIDE_CAMERA_TOPIC][1:] == (640, 480)
    assert published[WRIST_CAMERA_TOPIC][1:] == (256, 256)


def test_both_sides_agree_on_the_joint_names_and_their_order() -> None:
    s = _surface()
    assert s.ARM_JOINTS == ARM_JOINTS
    assert s.GRIPPER_JOINT == GRIPPER_JOINT
    assert s.JOINT_ORDER == JOINT_ORDER


def test_the_sorted_wire_order_shares_no_index_with_the_contract_order() -> None:
    """Which is why this is worth a test on both sides rather than a comment."""
    wire = sorted(JOINT_ORDER)
    assert wire != list(JOINT_ORDER)
    assert not any(a == b for a, b in zip(wire, JOINT_ORDER))


def test_the_gripper_map_is_an_offset_and_round_trips() -> None:
    s = _surface()
    for contract in (0.0, 0.25, 0.4, 0.5, 0.75, 1.0):
        mjcf = s.to_mjcf_gripper(contract)
        assert mjcf == pytest.approx(contract - s.GRIPPER_OFFSET_RAD)
        assert s.to_contract_gripper(mjcf) == pytest.approx(contract)


def test_the_gripper_map_clamps_to_the_contract_range() -> None:
    s = _surface()
    assert s.to_contract_gripper(5.0) == 1.0
    assert s.to_contract_gripper(-5.0) == 0.0


# --------------------------------------------------------------- namespacing


def test_both_sides_compose_a_namespaced_topic_the_same_way() -> None:
    """The two projects each own a copy of the rule; this is where they are held equal.

    The console cannot import the simulator -- it has to install and run with no
    simulator checkout at all -- so the composition rule is duplicated rather than
    shared. A duplicated rule that drifts is worse than no rule: the client would
    subscribe to `/so101/joint_states` while the simulator published `/so101//joint_states`
    or `/joint_states`, and the failure would be an empty observation, not an error.
    """
    sim = _namespace()
    for namespace in ("so101", "myagv", "", "robot_2"):
        for topic in (
            ARM_COMMAND_TOPIC,
            GRIPPER_COMMAND_TOPIC,
            JOINT_STATES_TOPIC,
            FREE_JOINT_STATES_TOPIC,
            OVERHEAD_CAMERA_TOPIC,
            SIDE_CAMERA_TOPIC,
            WRIST_CAMERA_TOPIC,
            TASK_MANAGER_RESET_SERVICE,
            "/cmd_vel",
            "/odom",
            "/scan",
        ):
            assert sim.ns_topic(namespace, topic) == namespaced(topic, namespace)


def test_both_sides_agree_that_an_empty_namespace_changes_nothing() -> None:
    """The bare vendor contract, which is what the tables in CLAUDE.md document."""
    sim = _namespace()
    for topic in (JOINT_STATES_TOPIC, "/cmd_vel", "/odom"):
        assert sim.ns_topic("", topic) == topic == namespaced(topic, "")


def test_the_simulator_prefixes_frames_without_a_leading_slash() -> None:
    """Topics are absolute graph paths; frame ids are tf names joined by `tf_prefix`.

    Getting these the same way round produces `/myagv/odom` as a *frame*, which no real
    stack emits and nothing will connect a tf tree to. Only the simulator composes
    frames -- the console reads them, in `smoke.py`, and this is what that check is
    checking against.
    """
    sim = _namespace()
    assert sim.ns_frame("myagv", "base_footprint") == "myagv/base_footprint"
    assert sim.ns_frame("myagv", "odom") == "myagv/odom"
    assert sim.ns_frame("", "odom") == "odom"
    # A JointState carries an empty frame on a real broadcaster, and namespacing must not
    # invent one -- that would be a difference from hardware, which is the one thing the
    # contract exists to avoid.
    assert sim.ns_frame("so101", "") == ""


def test_the_arm_settings_put_the_namespace_on_every_wire_name() -> None:
    """What the embodiment actually subscribes and publishes, end to end.

    The dataclass fields stay bare -- they are the transcript of `ros2 topic list -t`
    inside the reference container -- and the prefix is applied where they are handed to
    the adapter. So this checks the half that goes on the wire, and
    `test_ros_settings.py` checks the half that records the hardware.
    """
    from robot_console.arm.ros_settings import RosSettings

    settings = RosSettings(namespace="so101")
    kwargs = settings.base_kwargs()
    assert kwargs["joint_states_topic"] == "/so101/joint_states"
    assert kwargs["command_topic"] == "/so101/joint_trajectory_controller/joint_trajectory"
    assert kwargs["gripper_topic"] == "/so101/gripper_controller/commands"
    assert kwargs["reset_service"] == "/so101/reset"
    assert set(kwargs["cameras"]) == {"overhead", "side"}
    assert kwargs["cameras"]["overhead"][0] == "/so101/overhead/color/compressed"

    # Stripping the namespace back off must reproduce the container's own names exactly.
    bare = RosSettings(namespace="").base_kwargs()
    for key in ("joint_states_topic", "command_topic", "gripper_topic", "reset_service"):
        assert kwargs[key] == "/so101" + bare[key]
