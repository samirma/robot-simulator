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
from pathlib import Path

import pytest

from robot_console.arm.kinematics import ARM_JOINTS, GRIPPER_JOINT, JOINT_ORDER
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

SURFACE = (
    Path(__file__).resolve().parents[3]
    / "simulator" / "shared" / "ros_surfaces" / "so101.py"
)


def _surface():
    if not SURFACE.exists():
        pytest.skip(f"sibling simulator checkout not present at {SURFACE}")
    spec = importlib.util.spec_from_file_location("_so101_surface", SURFACE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # numpy only; no mujoco at import time
    return module


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
