"""A worked example of a custom controller for the simulator.

Run it with:

    ./run.sh serve --controller bridge.example_controller:reach_forward
    ./run.sh bridge --robot droid          # in another terminal

A controller is just a callable taking the observation dict and returning an
action dict. It is called once per policy step, synchronously, off the event loop.

Useful observation keys (Franka; see README for the full list):

    obs["actions/joint_pos"]  {"arm": [7], "gripper": [1]}  last commanded action,
                                                            already the right width
    obs["qpos"]               {"base": [], "arm": [7], "gripper": [2]}  measured
    obs["tcp_pose"]           ndarray(7,)   end-effector pose, xyz + wxyz quat
    obs["exo_camera_1"]       ndarray(720, 1280, 3) uint8
    obs["wrist_camera"]       ndarray(720, 1280, 3) uint8
    obs["grasp_state_pickup_obj"]["gripper"]  {"touching": bool, "held": bool}
    obs["task"]               str, e.g. "Pick up the kitchen utensil"

Command the *commanded* width, not the measured one: the Franka gripper reports
two finger joints but takes a single actuator value.
"""

from __future__ import annotations

import numpy as np

_state: dict = {"home": None, "step": 0}

RAMP_STEPS = 40
SHOULDER_SWEEP_RAD = 0.6


def reach_forward(obs: dict) -> dict:
    """Ramp the shoulder joint forward, then close the gripper once in contact.

    Deliberately simple -- it demonstrates reading state, ramping a command, and
    reacting to a sensor, without pretending to solve the task.
    """
    commanded = obs["actions/joint_pos"]
    arm = np.asarray(commanded["arm"], dtype=np.float64)
    gripper = np.asarray(commanded.get("gripper", [0.0]), dtype=np.float64)

    if _state["home"] is None:
        _state["home"] = arm.copy()

    # Ease the shoulder to its target rather than commanding a step change, which
    # the position controller would chase with a large transient.
    ramp = min(1.0, _state["step"] / RAMP_STEPS)
    target = _state["home"].copy()
    target[0] = _state["home"][0] + SHOULDER_SWEEP_RAD * ramp

    touching = obs.get("grasp_state_pickup_obj", {}).get("gripper", {}).get("touching", False)
    if touching:
        gripper = np.array([1.0])  # close

    _state["step"] += 1
    return {"arm": target, "gripper": gripper}


def hold_still(obs: dict) -> dict:
    """Minimal controller: echo the last command back, so nothing moves."""
    commanded = obs["actions/joint_pos"]
    return {"arm": commanded["arm"], "gripper": commanded.get("gripper", [0.0])}
