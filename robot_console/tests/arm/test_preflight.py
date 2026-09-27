"""The arm task's wire checks: discovery, the typed interface, and refusing hardware."""

from __future__ import annotations

import pytest

from robot_console.arm import preflight as pf
from robot_console.arm import ros_settings as rs

from .fake_rig import FakeRig


def _wire(namespace: str = "so101"):
    arm = rs.arm_interface(namespace)
    topics = {**arm["topics"], **rs.rig_interface(), "/myagv/cmd_vel": "geometry_msgs/Twist"}
    return topics, dict(arm["services"]), dict(arm["actions"])


def test_a_complete_simulator_wire_passes() -> None:
    assert pf.interface_problems(*_wire(), "so101") == ([], [])


def test_a_wire_without_reset_is_refused_as_hardware() -> None:
    topics, services, actions = _wire()
    refuse, _ = pf.interface_problems(topics, {}, actions, "so101")
    assert refuse and "/so101/reset" in refuse[0]


def test_a_wire_without_the_rig_is_refused() -> None:
    topics, services, actions = _wire()
    del topics["/scene/side/color/camera_info"]
    refuse, _ = pf.interface_problems(topics, services, actions, "so101")
    assert any("rig" in r and "camera_info" in r for r in refuse)


def test_a_wrong_type_or_a_missing_action_is_a_mismatch() -> None:
    topics, services, actions = _wire()
    topics["/so101/joint_states"] = "sensor_msgs/JointState"
    wrong = pf.interface_problems(topics, services, {}, "so101")[1]
    assert any("/so101/joint_states" in w for w in wrong)
    assert any("gripper_cmd" in w for w in wrong)


def test_discovery_finds_the_one_so101() -> None:
    assert pf.discover_namespace(_wire("so101")[0]) == "so101"
    assert pf.discover_namespace(_wire("")[0]) == ""


def test_discovery_names_every_candidate_when_there_are_two() -> None:
    topics = {**_wire("arm_a")[0], **_wire("arm_b")[0]}
    with pytest.raises(pf.PreflightError, match="/arm_a, /arm_b"):
        pf.discover_namespace(topics)


def test_discovery_refuses_a_wire_with_no_so101() -> None:
    with pytest.raises(pf.PreflightError, match="no SO-101"):
        pf.discover_namespace({"/myagv/cmd_vel": "geometry_msgs/Twist"})


def test_reset_reports_the_servers_refusal() -> None:
    with FakeRig(reset_success=False) as rig:
        assert pf.main(["reset", "--url", rig.url, "--namespace", "so101"]) == pf.EXIT_RESET_REFUSED
    with FakeRig() as rig:
        assert pf.main(["reset", "--url", rig.url, "--namespace", "so101"]) == pf.EXIT_OK


def test_wait_sees_every_stream() -> None:
    with FakeRig() as rig:
        assert pf.main(["wait", "--url", rig.url, "--namespace", "so101", "--timeout", "5"]) == 0
    assert pf.main(["wait", "--url", "ws://127.0.0.1:1", "--namespace", "so101"]) == \
        pf.EXIT_TRANSPORT
