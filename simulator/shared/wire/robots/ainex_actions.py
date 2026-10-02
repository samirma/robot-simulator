#!/usr/bin/env python3
"""Writes the AiNex action groups the simulator provides, as the vendor's `.d6a` files.

The boot plays `/home/ubuntu/software/ainex_controller/ActionGroups/<name>.d6a` (an SQLite
table `ActionGroup`: index, move time in ms, then the 24 servo pulses) on /app/set_action.
Those files ship on the robot's image and are in no pinned source, so the simulator
provides its own groups -- an ESTIMATE, documented in simulator/README.md, not the
vendor's data. Each frame is given as joint offsets (rad) from the boot's init pose and
converted to pulses with the controller's own servo map, so the groups stay within the
servo ranges and keep both feet planted (arm and head motions only):

* `wave`         right arm raised and waved three times, then lowered
* `raise_hands`  both arms raised overhead and lowered
* `nod`          head tilts down and up twice

    ainex_actions.py <servo_controller.yaml> <init_pose.yaml> <output dir>
"""

import math
import os
import sqlite3
import sys

import yaml

TICKS_PER_RAD = 180 / 3.1415926 / 240 * 1000

GROUPS = {
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


def main(ctl_file, pose_file, out_dir):
    with open(ctl_file) as fh:
        ctl = yaml.safe_load(fh)["controllers"]
    with open(pose_file) as fh:
        pose = yaml.safe_load(fh)["init_pose"]
    servo = {}
    for c in ctl.values():
        if c.get("type") != "JointPositionController" or "servo" not in c:
            continue
        s = c["servo"]
        sign = -1.0 if s["min"] > s["max"] else 1.0
        servo[c["joint_name"]] = (int(s["id"]), float(s["init"]), sign * TICKS_PER_RAD)
    os.makedirs(out_dir, exist_ok=True)
    for name, frames in GROUPS.items():
        path = os.path.join(out_dir, name + ".d6a")
        if os.path.exists(path):
            os.remove(path)
        db = sqlite3.connect(path)
        cols = ", ".join(f"Servo{i} INTEGER" for i in range(1, 25))
        db.execute(f"CREATE TABLE ActionGroup ([Index] INTEGER PRIMARY KEY, Time INTEGER, {cols})")
        for k, (ms, offs) in enumerate(frames, start=1):
            pulses = [500] * 24
            for joint, (sid, init, tpr) in servo.items():
                angle = pose.get(joint, 0.0) + offs.get(joint, 0.0)
                pulses[sid - 1] = max(0, min(1000, int(round(init + angle * tpr))))
            db.execute(f"INSERT INTO ActionGroup VALUES ({k}, {ms}, " +
                       ", ".join(str(p) for p in pulses) + ")")
        db.commit()
        db.close()


if __name__ == "__main__":
    main(*sys.argv[1:4])
