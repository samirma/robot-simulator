"""The arm trajectory must carry a ``header``, or this simulator ignores it.

``inspect_robots_ros`` builds ``JointTrajectory`` messages without one. Measured
against the running container, that message is accepted by rosbridge and then
does nothing: the ``JointTrajectoryController`` holds its pose and reports zero
error, while the gripper's ``ForwardCommandController`` -- which needs no header
-- keeps working. The episode therefore looks alive and scores zero.

Nothing offline can catch this: the MuJoCo embodiment never builds a
``JointTrajectory``. These tests stand in for that missing coverage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from inspect_robots import Action, Observation, StepResult
from inspect_robots_ros._msgs import build_joint_trajectory
from inspect_robots_ros.embodiment import RosEmbodiment

from robot_console.arm.embodiment import SO101RosEmbodiment
from robot_console.arm.kinematics import ARM_JOINTS
from robot_console.arm.ros_client import ZERO_HEADER, HeaderStampingClient
from robot_console.arm.ros_settings import RosSettings

# The bare contract names. The embodiment publishes them under its namespace, so the
# tests that go through a real embodiment use `settings.topic(...)` rather than these --
# and one of them exists precisely to catch the shim being keyed on the wrong one.
ARM_TOPIC = "/joint_trajectory_controller/joint_trajectory"
GRIPPER_TOPIC = "/gripper_controller/commands"


class _Recorder(HeaderStampingClient):
    """Capture publishes instead of sending them."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, msg: Any) -> None:  # type: ignore[override]
        if topic in self.stamped_topics and "header" not in msg:
            self.headers_added += 1
            msg = {"header": dict(ZERO_HEADER), **dict(msg)}
        self.sent.append((topic, dict(msg)))

    def latest(self, topic: str) -> None:  # type: ignore[override]
        """No socket, so no cached messages; success polling degrades to None."""
        return


def test_upstream_still_omits_the_header() -> None:
    """Pin the upstream behaviour this shim exists to correct.

    If this ever fails, ``inspect-robots-ros`` has started stamping its own
    trajectories and the shim can be deleted.
    """
    message = build_joint_trajectory(ARM_JOINTS, np.zeros(5), period_s=0.1, ros_version=2)
    assert "header" not in message
    assert set(message) == {"joint_names", "points"}


def test_the_shim_adds_a_zero_stamp_to_the_arm_topic() -> None:
    client = _Recorder("ws://unused", stamped_topics=(ARM_TOPIC,))
    client.publish(ARM_TOPIC, build_joint_trajectory(ARM_JOINTS, np.zeros(5), period_s=0.1, ros_version=2))
    topic, message = client.sent[-1]
    assert topic == ARM_TOPIC
    assert message["header"] == {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}
    # stamp 0 is ROS's "start now"; a real time would schedule it in the past.
    assert message["header"]["stamp"]["sec"] == 0
    assert client.headers_added == 1


def test_the_shim_leaves_other_topics_alone() -> None:
    client = _Recorder("ws://unused", stamped_topics=(ARM_TOPIC,))
    client.publish(GRIPPER_TOPIC, {"data": [0.5]})
    assert client.sent[-1] == (GRIPPER_TOPIC, {"data": [0.5]})
    assert client.headers_added == 0


def test_an_existing_header_is_not_overwritten() -> None:
    client = _Recorder("ws://unused", stamped_topics=(ARM_TOPIC,))
    supplied = {"stamp": {"sec": 7, "nanosec": 5}, "frame_id": "base"}
    client.publish(ARM_TOPIC, {"header": supplied, "joint_names": [], "points": []})
    assert client.sent[-1][1]["header"] == supplied
    assert client.headers_added == 0


def test_the_embodiment_installs_the_shim_on_the_arm_topic() -> None:
    embodiment = SO101RosEmbodiment(RosSettings())
    settings = embodiment.settings
    assert isinstance(embodiment._client, HeaderStampingClient)
    # The topic as it goes on the wire, not the bare contract name. Keyed on the bare
    # name under a namespace the shim matches nothing, every trajectory ships without a
    # header, and the controller ignores all of them in silence.
    assert embodiment._client.stamped_topics == (settings.topic(settings.command_topic),)
    assert embodiment._client.stamped_topics == ("/so101/joint_trajectory_controller"
                                                 "/joint_trajectory",)
    # ...and the bare wire still works, so the single-robot contract stays expressible.
    bare = SO101RosEmbodiment(RosSettings(namespace=""))
    assert bare._client.stamped_topics == (ARM_TOPIC,)


def test_a_step_puts_a_stamped_trajectory_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through the real embodiment, with only the socket replaced."""
    monkeypatch.setattr(
        RosEmbodiment,
        "step",
        lambda self, action: _publish_like_upstream(self, action),
    )
    embodiment = SO101RosEmbodiment(RosSettings())
    arm_topic = embodiment.settings.topic(embodiment.settings.command_topic)
    recorder = _Recorder(embodiment.url, stamped_topics=(arm_topic,))
    embodiment._client = recorder
    embodiment.step(Action(data=np.arange(6, dtype=np.float64) / 10.0))

    arm = [m for t, m in recorder.sent if t == arm_topic]
    assert arm, "no arm trajectory was published"
    assert "header" in arm[-1]
    assert arm[-1]["joint_names"] == list(ARM_JOINTS)
    assert len(arm[-1]["points"][0]["positions"]) == len(ARM_JOINTS)


def _publish_like_upstream(embodiment: Any, action: Action) -> StepResult:
    """The publishing half of the upstream step, without the socket waits."""
    data = np.asarray(action.data, dtype=np.float64)
    embodiment._client.publish(
        embodiment.command_topic,
        build_joint_trajectory(
            embodiment.joints,
            data[: len(embodiment.joints)],
            period_s=1.0 / embodiment.control_hz,
            ros_version=embodiment.ros_version,
        ),
    )
    return StepResult(observation=Observation())
