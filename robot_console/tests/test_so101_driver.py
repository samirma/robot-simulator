import cv2
import numpy as np

from robot_console.so101_driver import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    JOINT_HIGH,
    MOTOR_KEYS,
    SimulatorSO101,
)


def observation(arm, gripper):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :320] = [20, 80, 180]
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    assert ok
    return {
        "qpos": {"arm": list(arm), "gripper": [gripper], "base": []},
        "camera_jpeg": jpeg.reshape(-1),
    }


def test_driver_translates_observation_to_soarm_surface():
    driver = SimulatorSO101()
    driver(observation([0.0, -0.5, 0.8, 0.4, 0.1], GRIPPER_OPEN_RAD))

    result = driver.get_observation()

    assert result["front"].shape == (480, 640, 3)
    assert result["gripper.pos"] == 100.0
    np.testing.assert_allclose(
        [result[key] for key in MOTOR_KEYS[:5]],
        np.degrees([0.0, -0.5, 0.8, 0.4, 0.1]),
    )


def test_driver_clamps_and_translates_action_to_simulator_units():
    driver = SimulatorSO101()
    driver(observation([0.0] * 5, GRIPPER_OPEN_RAD))

    accepted = driver.send_action(dict.fromkeys(MOTOR_KEYS, 1000.0))
    action = driver(observation([0.0] * 5, GRIPPER_CLOSED_RAD))

    np.testing.assert_allclose([accepted[key] for key in MOTOR_KEYS], JOINT_HIGH)
    np.testing.assert_allclose(action["arm"], np.radians(JOINT_HIGH[:5]))
    np.testing.assert_allclose(action["gripper"], [GRIPPER_OPEN_RAD])
