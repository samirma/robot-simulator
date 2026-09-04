"""The episode must not end at first contact, and the log must still score.

``terminate_on_success`` used to fire on the first step where the apple was
instantaneously on the plate. The scorer, meanwhile, required a run of true
steps at the end of the trajectory. Those two rules contradict each other: a
genuine placement produced exactly one true step and was recorded as a failure.
The contract settles it — section 5 clause 4 wants the apple *at rest for
>= 1.0 s* — so the episode holds and then terminates, and both tests below
follow one trajectory from the live embodiment through to the offline scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from inspect_robots import Action, Observation, StepResult
from inspect_robots_ros.embodiment import RosEmbodiment

import cv2

from robot_console.arm.embodiment import SO101RosEmbodiment
from robot_console.arm.ros_settings import OVERHEAD_CAMERA_NAME, RosSettings
from robot_console.arm.scorer import apple_on_plate_success
from robot_console.arm.success import GEOMETRIC_SUCCESS_KEY, HELD_KEY
from robot_console.arm.vision_success import GroundProjector, PLATE_RADIUS_M

RESTING = (0.226, -0.226, 0.0404)
SPAWN = (0.30, 0.10, 0.020)
PLATE_XY = (0.226, -0.226)

_PROJ = GroundProjector()


def _render_overhead(apple_xyz: tuple[float, float, float]) -> np.ndarray:
    """Draw the scene the overhead camera would see, with the apple at ``apple_xyz``.

    The verdict is computed from pixels now, so a test that fed the embodiment a pose
    would be testing nothing that runs in production. Rendering through the projector's
    own forward transform means these tests exercise the real detector -- the colour
    thresholds, the ellipse fit, the back-projection -- and would fail if any of them
    stopped agreeing with the geometry.
    """
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (60, 130, 180)                     # BGR: a wood-toned, clearly non-white ground
    rim = [
        _PROJ.project(
            PLATE_XY[0] + PLATE_RADIUS_M * np.cos(t),
            PLATE_XY[1] + PLATE_RADIUS_M * np.sin(t),
            0.020,
        )
        for t in np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ]
    cv2.fillPoly(img, [np.array(rim, np.int32)], (245, 245, 245))
    u, v = _PROJ.project(*apple_xyz)
    cv2.circle(img, (int(round(u)), int(round(v))), 9, (30, 30, 210), -1)
    return img[:, :, ::-1].copy()               # the embodiment is handed RGB


@dataclass
class _Sample:
    msg: dict[str, Any]
    stamp: float = 0.0
    seq: int = 0


class _ScriptedClient:
    """The two monitor topics, driven by a caller-set apple pose and clock."""

    def __init__(self) -> None:
        self.position = SPAWN
        self.speed = 0.0
        self.time = 0.0

    def latest(self, topic: str) -> _Sample | None:
        if topic == "/free_joint_publisher/free_joint_states":
            whole = int(self.time)
            header = {
                "stamp": {"sec": whole, "nanosec": round((self.time - whole) * 1e9)},
                "frame_id": "",
            }
            x, y, z = self.position
            return _Sample(
                {
                    "header": header,
                    "free_joints": [
                        {
                            "name": "apple",
                            "pose": {
                                "header": header,
                                "pose": {
                                    "position": {"x": x, "y": y, "z": z},
                                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                },
                            },
                            "twist": {
                                "header": header,
                                "twist": {
                                    "linear": {"x": self.speed, "y": 0.0, "z": 0.0},
                                    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                                },
                            },
                        }
                    ],
                }
            )
        return None


@pytest.fixture()
def embodiment(monkeypatch: pytest.MonkeyPatch) -> SO101RosEmbodiment:
    """A real embodiment whose transport and base step are replaced, nothing else."""
    monkeypatch.setattr(
        RosEmbodiment,
        "step",
        lambda self, action: StepResult(
            observation=Observation(
                images={OVERHEAD_CAMERA_NAME: _render_overhead(self._client.position)}
            ),
            truncated=False,
        ),
    )
    subject = SO101RosEmbodiment(RosSettings(control_hz=10.0))
    subject._client = _ScriptedClient()  # type: ignore[assignment]
    # One frame with the apple at its spawn point, discarded. Clause 5 asks how far the
    # apple has travelled, and the camera can only answer that against somewhere it has
    # actually seen it -- so an episode that opens with the apple already on the plate
    # has, correctly, no evidence any placement happened. Every real episode starts at
    # spawn; these tests have to as well, or they test a case that cannot occur.
    subject.step(Action(data=np.zeros(6)))
    return subject


def _run(subject: SO101RosEmbodiment, steps: int, dt: float = 0.1) -> list[StepResult]:
    """Step until termination or the budget runs out, as the rollout would."""
    action = Action(data=np.zeros(6))
    results: list[StepResult] = []
    for _ in range(steps):
        subject._client.time += dt  # type: ignore[attr-defined]
        result = subject.step(action)
        results.append(result)
        if result.terminated:
            break
    return results


def test_first_placement_does_not_end_the_episode(
    embodiment: SO101RosEmbodiment,
) -> None:
    embodiment._client.position = RESTING  # type: ignore[attr-defined]
    results = _run(embodiment, steps=3)
    assert [bool(r.info[GEOMETRIC_SUCCESS_KEY]) for r in results] == [True, True, True]
    assert [r.terminated for r in results] == [False, False, False]
    assert [r.info[HELD_KEY] for r in results] == [False, False, False]


def test_episode_ends_once_the_hold_is_complete(embodiment: SO101RosEmbodiment) -> None:
    embodiment._client.position = RESTING  # type: ignore[attr-defined]
    results = _run(embodiment, steps=40)
    assert results[-1].terminated
    assert results[-1].termination_reason == "success"
    assert results[-1].reward == 1.0
    # 1.0 s at 10 Hz: the first placed step starts the clock, the 11th completes it.
    assert len(results) == 11
    assert results[-1].info[HELD_KEY] is True
    assert results[-1].info["apple_on_plate_hold_s"] == pytest.approx(1.0)


def test_a_bounce_off_the_plate_restarts_the_hold(embodiment: SO101RosEmbodiment) -> None:
    client = embodiment._client  # type: ignore[attr-defined]
    client.position = RESTING
    _run(embodiment, steps=5)
    # A knock has to *move* the apple now. Setting a speed field alone was enough when
    # the verdict read the twist off a topic; the camera only knows the apple moved if
    # it is somewhere else, which is the honest version of the same test.
    client.speed = 0.5
    client.position = (RESTING[0] + 0.05, RESTING[1], RESTING[2])
    result = embodiment.step(Action(data=np.zeros(6)))
    assert result.info[GEOMETRIC_SUCCESS_KEY] is False
    assert result.info["apple_on_plate_hold_s"] == 0.0
    assert not result.terminated


# -- and the log the episode leaves behind -------------------------------------


@dataclass
class _Step:
    result: StepResult


@dataclass
class _Record:
    steps: list[_Step]


def _record(results: list[StepResult]) -> _Record:
    return _Record(steps=[_Step(result=r) for r in results])


def test_the_recorded_trajectory_scores_as_a_success(
    embodiment: SO101RosEmbodiment,
) -> None:
    embodiment._client.position = RESTING  # type: ignore[attr-defined]
    results = _run(embodiment, steps=40)
    score = apple_on_plate_success()(_record(results), None)
    assert score.value is True
    assert score.metadata["hold_seconds"] == pytest.approx(1.0)


def test_the_old_terminate_at_first_contact_trajectory_scores_as_a_failure(
    embodiment: SO101RosEmbodiment,
) -> None:
    """The bug, reproduced: end at first contact and the log cannot support a hold."""
    client = embodiment._client  # type: ignore[attr-defined]
    results = _run(embodiment, steps=4)
    client.position = RESTING
    results.append(embodiment.step(Action(data=np.zeros(6))))
    assert bool(results[-1].info[GEOMETRIC_SUCCESS_KEY]) is True
    score = apple_on_plate_success()(_record(results), None)
    assert score.value is False
    assert "ever_placed=True" in score.explanation

