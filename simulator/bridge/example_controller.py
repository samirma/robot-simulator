"""A worked example of a custom controller for the simulator.

Run it with:

    ./run.sh serve --controller bridge.example_controller:reach_forward
    ./run.sh view --robot so101 --control 127.0.0.1:8000   # in another terminal

A controller is just a callable taking the observation dict and returning an
action dict. It is called once per control step, synchronously, off the event loop.

Useful observation keys (see bridge/server.py for the wire protocol):

    obs["actions/joint_pos"]  {"arm": [...], "gripper": [...]}  last commanded
                                                                action, already the
                                                                right width
    obs["qpos"]               measured joint positions, keyed by move group
    obs["tcp_pose"]           ndarray(3,)   end-effector position, xyz
    obs["camera_jpeg"]        JPEG bytes of the robot's camera, when streaming

Command the *commanded* width, not the measured one: a gripper may report more
finger joints than it takes actuator values.
"""

from __future__ import annotations

import numpy as np

_state: dict = {"home": None, "step": 0}

RAMP_STEPS = 40
SHOULDER_SWEEP_RAD = 0.6


def reach_forward(obs: dict) -> dict:
    """Ramp the shoulder joint forward, then close the gripper once in contact.

    Deliberately simple -- it demonstrates reading state, ramping a command, and
    reacting to a sensor, without pretending to do anything useful.
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
