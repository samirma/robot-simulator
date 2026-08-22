"""LeRobot-style SO-101 driver backed by simulator network observations."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MOTOR_KEYS = tuple(f"{name}.pos" for name in MOTORS)
JOINT_LOW = (-110.0, -100.0, -96.83, -95.0, -157.21, 0.0)
JOINT_HIGH = (110.0, 100.0, 96.83, 95.0, 162.79, 100.0)
GRIPPER_CLOSED_RAD = -0.17453
GRIPPER_OPEN_RAD = 1.74533


def _gripper_to_percent(value: float) -> float:
    span = GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
    return float(np.clip((value - GRIPPER_CLOSED_RAD) * 100.0 / span, 0.0, 100.0))


def _percent_to_gripper(value: float) -> float:
    percent = float(np.clip(value, 0.0, 100.0))
    return GRIPPER_CLOSED_RAD + percent * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD) / 100.0


def _decode_front(obs: Mapping[str, Any]) -> np.ndarray:
    jpeg = obs.get("camera_jpeg")
    if jpeg is not None:
        encoded = np.asarray(jpeg, dtype=np.uint8).reshape(-1)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("simulator camera JPEG could not be decoded")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = obs.get("camera")
    if frame is None:
        raise RuntimeError("simulator observation has no camera image")
    return np.asarray(frame, dtype=np.uint8)


class SimulatorSO101:
    """Translate simulator observations and actions to the SOArm driver surface."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._observation: dict[str, Any] | None = None
        self._sequence = 0
        self._last_read_sequence = -1
        self._target_rad: np.ndarray | None = None
        self._closed = False

    def __call__(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        arm, gripper = self._sim_positions(obs)
        with self._condition:
            self._observation = obs
            self._sequence += 1
            if self._target_rad is None:
                self._target_rad = np.concatenate((arm, gripper))
            target = self._target_rad.copy()
            self._condition.notify_all()
        return {"arm": target[:5], "gripper": target[5:]}

    def get_observation(self) -> Mapping[str, Any]:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._closed
                or (self._observation is not None and self._sequence > self._last_read_sequence),
                timeout=30.0,
            )
            if not ready or self._observation is None:
                raise RuntimeError("timed out waiting for an SO-101 simulator observation")
            if self._closed:
                raise RuntimeError("SO-101 simulator connection is closed")
            obs = self._observation
            self._last_read_sequence = self._sequence

        arm, gripper = self._sim_positions(obs)
        values = np.concatenate((np.degrees(arm), [_gripper_to_percent(gripper[0])]))
        result: dict[str, Any] = dict(zip(MOTOR_KEYS, values, strict=True))
        result["front"] = _decode_front(obs)
        return result

    def send_action(self, action: Mapping[str, float]) -> Mapping[str, Any]:
        values = np.array([float(action[key]) for key in MOTOR_KEYS], dtype=np.float64)
        values = np.clip(values, JOINT_LOW, JOINT_HIGH)
        target = np.concatenate((np.radians(values[:5]), [_percent_to_gripper(values[5])]))
        with self._condition:
            if self._closed:
                raise RuntimeError("SO-101 simulator connection is closed")
            self._target_rad = target
        return dict(zip(MOTOR_KEYS, values, strict=True))

    def disconnect(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @staticmethod
    def _sim_positions(obs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        qpos = obs.get("qpos")
        if not isinstance(qpos, Mapping):
            raise TypeError("SO-101 simulator observation has no qpos mapping")
        arm = np.asarray(qpos.get("arm"), dtype=np.float64).reshape(-1)
        gripper = np.asarray(qpos.get("gripper"), dtype=np.float64).reshape(-1)
        if arm.shape != (5,) or gripper.shape != (1,):
            raise RuntimeError(
                f"expected SO-101 qpos shapes arm=(5,), gripper=(1,), got "
                f"arm={arm.shape}, gripper={gripper.shape}"
            )
        return arm, gripper
