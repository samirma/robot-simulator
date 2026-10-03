#!/usr/bin/env python3
"""Writes the AiNex's estimated action groups as the vendor's `.d6a` files.

The boot plays `/home/ubuntu/software/ainex_controller/ActionGroups/<name>.d6a` (an SQLite
table `ActionGroup`: index, move time in ms, then the 24 servo pulses) on /app/set_action.
Those files ship on the robot's image and are in no pinned source, so no vendor group can
be played (the vendor's names answer as the controller answers an unknown name). The
groups served are the robot specification's ESTIMATES, `robots_specs/ainex/action_groups.yml`
(named by `motions[action_group].command.estimated_groups` of ros.yml; format in
robots_specs/SCHEMA.md, "Estimated action groups"): each frame is offsets (rad) from the
recorded init pose, the `/ainex_controller/init_pose` parameter, a joint it does not name
staying at its init angle, converted to pulses with the controller's own servo map
(`servo_controller.yaml`): clamp(0, 1000, round(init + angle * sign * 1000 / 240 deg)).

    ainex_actions.py <servo_controller.yaml> <output dir>
"""

import os
import sqlite3
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # wire/

import common  # noqa: E402

TICKS_PER_RAD = 180 / 3.1415926 / 240 * 1000     # ainex_controller's ENCODER_TICKS_PER_RADIAN


def groups(robot=None) -> dict:
    """{name: [(time ms, {joint: offset rad})]} of the record's estimated action groups."""
    robot = robot or common.owner()
    row = common.motion_row(common.interface(robot), "action_group")
    with open(robot.folder_path / row["command"]["estimated_groups"], encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return {name: [(int(f["time_ms"]), dict(f.get("offsets_rad") or {})) for f in g["frames"]]
            for name, g in doc["groups"].items()}


def init_pose(robot=None) -> dict:
    """The recorded init pose (rad by joint), the `/ainex_controller/init_pose` parameter."""
    return common.param(common.interface(robot or common.owner()), "/ainex_controller/init_pose")


def servo_map(controllers: dict) -> dict:
    """joint -> (servo id, init pulse, pulses per rad with the servo's direction)."""
    out = {}
    for c in controllers.values():
        if c.get("type") != "JointPositionController" or "servo" not in c:
            continue
        s = c["servo"]
        sign = -1.0 if s["min"] > s["max"] else 1.0
        out[c["joint_name"]] = (int(s["id"]), float(s["init"]), sign * TICKS_PER_RAD)
    return out


def frame_pulses(servo: dict, pose: dict, offsets: dict) -> list:
    """The 24 servo pulses of one frame."""
    pulses = [500] * 24
    for joint, (sid, init, tpr) in servo.items():
        angle = pose.get(joint, 0.0) + offsets.get(joint, 0.0)
        pulses[sid - 1] = max(0, min(1000, int(round(init + angle * tpr))))
    return pulses


def main(ctl_file, out_dir):
    with open(ctl_file) as fh:
        servo = servo_map(yaml.safe_load(fh)["controllers"])
    pose = init_pose()
    os.makedirs(out_dir, exist_ok=True)
    for name, frames in groups().items():
        path = os.path.join(out_dir, name + ".d6a")
        if os.path.exists(path):
            os.remove(path)
        db = sqlite3.connect(path)
        cols = ", ".join(f"Servo{i} INTEGER" for i in range(1, 25))
        db.execute(f"CREATE TABLE ActionGroup ([Index] INTEGER PRIMARY KEY, Time INTEGER, {cols})")
        for k, (ms, offs) in enumerate(frames, start=1):
            db.execute(f"INSERT INTO ActionGroup VALUES ({k}, {ms}, " +
                       ", ".join(str(p) for p in frame_pulses(servo, pose, offs)) + ")")
        db.commit()
        db.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
