"""The AiNex's action groups on the wire (wire/robots/ainex_actions.py) are the robot
specification's estimated groups, `robots_specs/ainex/action_groups.yml` (named by
ros.yml's motions[action_group].command.estimated_groups), as offsets from the recorded
init pose -- the same groups, frame for frame, the simulator served from its own code
before the record held them (2026-10-03) -- converted to pulses by the servo map."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[3] / "simulator" / "shared"
for p in (str(SHARED), str(SHARED / "wire"), str(SHARED / "wire" / "robots")):
    if p not in sys.path:
        sys.path.insert(0, p)

import registry  # noqa: E402

#: the groups the simulator served before they moved into the record
SERVED_BEFORE = {
    "wave": [
        (600, {"r_sho_pitch": -1.2, "r_sho_roll": -1.3, "r_el_yaw": 0.0}),
        (400, {"r_sho_pitch": -1.2, "r_sho_roll": -1.3, "r_el_pitch": 0.6}),
        (400, {"r_sho_pitch": -1.2, "r_sho_roll": -1.3, "r_el_pitch": -0.2}),
        (400, {"r_sho_pitch": -1.2, "r_sho_roll": -1.3, "r_el_pitch": 0.6}),
        (400, {"r_sho_pitch": -1.2, "r_sho_roll": -1.3, "r_el_pitch": -0.2}),
        (400, {"r_sho_pitch": -1.2, "r_sho_roll": -1.3, "r_el_pitch": 0.6}),
        (600, {}),
    ],
    "raise_hands": [
        (800, {"r_sho_roll": -1.4, "l_sho_roll": 1.4}),
        (800, {"r_sho_roll": -2.4, "l_sho_roll": 2.4}),
        (800, {"r_sho_roll": -1.4, "l_sho_roll": 1.4}),
        (800, {}),
    ],
    "nod": [
        (400, {"head_tilt": -0.4}),
        (400, {"head_tilt": 0.1}),
        (400, {"head_tilt": -0.4}),
        (400, {}),
    ],
}
#: the init pose's arms the groups were written from (vendor legs, head and grippers; arms
#: relaxed down at the sides), which the recorded init pose holds
ARMS_DOWN = {"l_sho_pitch": 0.0, "l_sho_roll": -1.45, "l_el_pitch": 0.0, "l_el_yaw": 0.0,
             "r_sho_pitch": 0.0, "r_sho_roll": 1.45, "r_el_pitch": 0.0, "r_el_yaw": 0.0}


def test_the_groups_are_the_records_estimates(monkeypatch):
    monkeypatch.setenv("RSIM_ROBOT", "ainex")
    import ainex_actions

    robot = registry.get("ainex")
    assert ainex_actions.groups(robot) == SERVED_BEFORE
    pose = ainex_actions.init_pose(robot)
    assert {k: pose[k] for k in ARMS_DOWN} == ARMS_DOWN


def test_a_frame_becomes_pulses_by_the_servo_map(monkeypatch, tmp_path):
    """A joint the frame names is at init pose + offset, every other at its init angle;
    pulse = clamp(0, 1000, round(init + angle * sign * 1000 / 240 deg))."""
    monkeypatch.setenv("RSIM_ROBOT", "ainex")
    import ainex_actions

    ctl = {"head_tilt_controller": {"type": "JointPositionController", "joint_name": "head_tilt",
                                    "servo": {"id": 24, "init": 500, "min": 0, "max": 1000}},
           "r_sho_roll_controller": {"type": "JointPositionController", "joint_name": "r_sho_roll",
                                     "servo": {"id": 13, "init": 875, "min": 1000, "max": 0}}}
    servo = ainex_actions.servo_map(ctl)
    per_rad = 1000 / 240 * 180 / 3.1415926
    pulses = ainex_actions.frame_pulses(servo, {"head_tilt": 0.0, "r_sho_roll": 1.45},
                                        {"head_tilt": -0.4})
    assert pulses[23] == round(500 - 0.4 * per_rad)
    assert pulses[12] == max(0, min(1000, round(875 - 1.45 * per_rad)))
    assert pulses.count(500) == 22
    # the writer puts one row per frame, in order, with its time
    ctl_file = tmp_path / "servo_controller.yaml"
    import yaml

    ctl_file.write_text(yaml.safe_dump({"controllers": ctl}))
    ainex_actions.main(str(ctl_file), str(tmp_path / "out"))
    rows = sqlite3.connect(tmp_path / "out" / "nod.d6a").execute(
        "select * from ActionGroup").fetchall()
    assert [r[1] for r in rows] == [400, 400, 400, 400]
