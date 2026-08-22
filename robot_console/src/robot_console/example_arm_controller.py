"""Worked custom controller for the generic simulator arm client.

Run with:

    robot-console-arm --controller robot_console.example_arm_controller:reach_forward
"""

from __future__ import annotations

import numpy as np

_home = None
_step = 0
RAMP_STEPS = 40
SHOULDER_SWEEP_RAD = 0.6


def reach_forward(obs: dict) -> dict:
    """Ramp the shoulder forward, then close the gripper after contact."""
    global _home, _step
    commanded = obs["actions/joint_pos"]
    arm = np.asarray(commanded["arm"], dtype=np.float64)
    gripper = np.asarray(commanded.get("gripper", [0.0]), dtype=np.float64)
    if _home is None:
        _home = arm.copy()

    target = _home.copy()
    target[0] = _home[0] + SHOULDER_SWEEP_RAD * min(1.0, _step / RAMP_STEPS)
    touching = obs.get("grasp_state_pickup_obj", {}).get("gripper", {}).get("touching", False)
    if touching:
        gripper = np.array([1.0])
    _step += 1
    return {"arm": target, "gripper": gripper}
