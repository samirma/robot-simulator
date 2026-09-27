"""``so101_ros`` over a rosbridge: commands, the gripper action, reset, and evidence."""

from __future__ import annotations

import time

import pytest
from inspect_robots import Action, Scene

from robot_console.arm.embodiment import SO101RosEmbodiment, so101_ros
from robot_console.arm.ros_settings import RosSettings
from robot_console.arm.vision_success import RIG_SAMPLES_KEY, SYNC_TOLERANCE_S, measure

from .fake_rig import FakeRig
from .rig_fixtures import AWAY

SCENE = Scene(id="apple-on-plate", instruction="place the apple")


def _embodiment(rig: FakeRig) -> SO101RosEmbodiment:
    return SO101RosEmbodiment(RosSettings(url=rig.url, namespace="so101", obs_timeout_s=5.0,
                                          fresh_obs_timeout_s=2.0))


def test_a_step_commands_the_arm_and_the_gripper_action_and_records_the_rig() -> None:
    with FakeRig() as rig:
        emb = _embodiment(rig)
        try:
            obs = emb.reset(SCENE)
            assert set(obs.images) == {"overhead", "side", "wrist"}
            assert rig.service_calls == ["/so101/reset"]
            time.sleep(0.4)
            result = emb.step(Action(data=[*AWAY[:5], 0.4]))
            result = emb.step(Action(data=[*AWAY[:5], 0.4]))
        finally:
            emb.close()
        arm = [m for t, m in rig.published
               if t == "/so101/joint_trajectory_controller/joint_trajectory"]
        assert arm and arm[0]["header"]["stamp"] == {"sec": 0, "nanosec": 0}
        assert arm[0]["joint_names"][0] == "shoulder_pan_joint"
        # One goal: the second step asked for the same jaw position.
        assert len(rig.goals) == 1
        goal = rig.goals[0]
        assert goal["action"] == "/so101/gripper_controller/gripper_cmd"
        assert goal["action_type"] == "control_msgs/action/ParallelGripperCommand"
        assert goal["args"]["command"]["position"] == [0.4]
        assert goal["args"]["command"]["name"] == ["gripper_joint"]
        # Nothing ever went out as a topic on the action's name.
        assert not any("gripper_controller" in t for t, _ in rig.published)
        # The unfinished goal was cancelled on close.
        assert [c["id"] for c in rig.cancels] == [goal["id"]]

    samples = [s for s in result.info[RIG_SAMPLES_KEY]]
    assert samples
    for s in samples:
        assert abs(s["side"]["stamp"] - s["overhead"]["stamp"]) <= SYNC_TOLERANCE_S
        assert abs(s["joint_state"]["stamp"] - s["overhead"]["stamp"]) <= SYNC_TOLERANCE_S
        assert measure(s).problem == ""


def test_a_finished_goal_is_not_cancelled() -> None:
    with FakeRig(finish_goals=True) as rig:
        emb = _embodiment(rig)
        try:
            emb.reset(SCENE)
            emb.step(Action(data=[*AWAY[:5], 0.9]))
            time.sleep(0.2)
        finally:
            emb.close()
        assert len(rig.goals) == 1 and rig.cancels == []


def test_a_refused_reset_is_an_error_with_the_servers_message() -> None:
    with FakeRig(reset_success=False) as rig:
        emb = _embodiment(rig)
        try:
            with pytest.raises(RuntimeError, match="engine busy"):
                emb.reset(SCENE)
        finally:
            emb.close()


def test_the_entry_point_coerces_cli_strings_and_refuses_unknown_options() -> None:
    emb = so101_ros(url="ws://127.0.0.1:1", namespace="", views="overhead,side",
                    control_hz="5", fresh_obs_timeout_s="2")
    assert emb.settings.views == ("overhead", "side")
    assert emb.settings.control_hz == 5.0
    assert emb.info.action_space.shape == (6,)
    with pytest.raises(ValueError, match="object_state_topic"):
        so101_ros(object_state_topic="/free_joint_publisher/free_joint_states")
