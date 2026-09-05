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
from robot_console.arm.vision_success import (
    APPLE_RADIUS_M,
    GroundProjector,
    PLATE_RADIUS_M,
    RESTING_CENTRE_Z,
)

#: The red mask stops short of the apple's shaded limb, so the detected blob is reliably
#: smaller than the geometric silhouette -- 0.90 of it, measured over 4882 real frames.
#: The renderer below reproduces that, because the airborne test is calibrated against it:
#: a synthetic apple drawn at its full silhouette would read as 1.0 and be judged airborne
#: while sitting on the plate.
_MASK_SHORTFALL = 0.90

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
    # Drawn at the size the camera would actually see it, so height is encoded in the
    # image the way the detector reads it: an apple held above the plate images bigger.
    radius = _PROJ.expected_radius_px(*apple_xyz, APPLE_RADIUS_M) * _MASK_SHORTFALL
    cv2.circle(img, (int(round(u)), int(round(v))), max(int(round(radius)), 2),
               (30, 30, 210), -1)
    return img[:, :, ::-1].copy()               # the embodiment is handed RGB


@dataclass
class _Sample:
    msg: dict[str, Any]
    stamp: float = 0.0
    seq: int = 0


class _ScriptedClient:
    """The two monitor topics, driven by a caller-set apple pose and clock.

    The topic it answers to is passed in rather than written out, because the embodiment
    subscribes under its ROS namespace: a fake keyed on the bare contract name answers
    nothing, which looks exactly like a simulator that stopped publishing.
    """

    def __init__(self, object_state_topic: str) -> None:
        self.object_state_topic = object_state_topic
        self.position = SPAWN
        self.speed = 0.0
        self.time = 0.0

    def latest(self, topic: str) -> _Sample | None:
        if topic == self.object_state_topic:
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
    subject._client = _ScriptedClient(  # type: ignore[assignment]
        subject.settings.topic(subject.settings.object_state_topic)
    )
    # A few steps with the apple at its spawn point. Every real episode begins this way,
    # and the camera needs it twice over: clause 5 measures travel against somewhere it has
    # actually seen the apple, and the airborne test needs the apple's own apparent size
    # while it is definitely resting. Without them these tests would exercise a case that
    # cannot occur -- an episode whose first frame already shows the apple placed.
    subject._spawn_steps = [subject.step(Action(data=np.zeros(6))) for _ in range(3)]
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


def _record(subject: SO101RosEmbodiment, results: list[StepResult]) -> _Record:
    """The trajectory a rollout would have saved: the spawn steps are part of it."""
    return _Record(steps=[_Step(result=r) for r in [*subject._spawn_steps, *results]])


def test_the_recorded_trajectory_scores_as_a_success(
    embodiment: SO101RosEmbodiment,
) -> None:
    embodiment._client.position = RESTING  # type: ignore[attr-defined]
    results = _run(embodiment, steps=40)
    score = apple_on_plate_success()(_record(embodiment, results), None)
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
    score = apple_on_plate_success()(_record(embodiment, results), None)
    assert score.value is False
    assert "ever_placed=True" in score.explanation



def test_an_apple_held_above_the_plate_scores_and_this_is_known(
    embodiment: SO101RosEmbodiment,
) -> None:
    """A known, accepted gap, pinned here so it cannot be mistaken for an accident.

    Clause 2 -- the apple's height -- is not enforced from the camera. A single overhead
    view cannot separate an apple resting on the plate from one a stationary gripper is
    holding above it: it is inside the gate, it has travelled, and being held by an arm
    that has stopped it is *more* still than a real placement. Apparent size was measured
    as a discriminator and the two populations overlap; see `vision_success`.

    So this passes, deliberately. MolmoAct2 does exactly this in about half its episodes,
    which is the difference between it reading 3/6 and 1/6. The `reference_success` scorer
    sees the pose and does check height, and is what distinguishes the two.

    If a future change makes this test fail, height has started being enforced somewhere.
    That is good news, and this test should be inverted rather than deleted.
    """
    client = embodiment._client  # type: ignore[attr-defined]
    # Directly over the plate, but 40 mm above where a resting apple's centre sits.
    client.position = (RESTING[0], RESTING[1], RESTING_CENTRE_Z + 0.040)
    results = _run(embodiment, steps=40)
    assert results[-1].terminated
    assert results[-1].info[HELD_KEY] is True


def test_an_apple_resting_on_the_plate_scores(embodiment: SO101RosEmbodiment) -> None:
    """The case the verdict is actually for, lowered into place rather than parked."""
    client = embodiment._client  # type: ignore[attr-defined]
    client.position = (RESTING[0], RESTING[1], RESTING_CENTRE_Z + 0.040)
    _run(embodiment, steps=6)
    client.position = RESTING
    results = _run(embodiment, steps=40)
    assert results[-1].terminated
    assert results[-1].info[HELD_KEY] is True

