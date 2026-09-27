"""The console's teleop robot ids are ids in `robots_specs/robots.yml`.

`--robot` takes a robot id from that file (console spec §2.1); the console keeps its own
copy as contract constants so it installs with no workspace around it. This reads the file
as text, with the standard library only -- the console's base dependencies are numpy,
OpenCV and roslibpy, and a YAML parser is not one of them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from robot_console.cli import build_parser
from robot_console.robots import AINEX, MYAGV, SLAM_ROBOT, STOP_COMMANDS, TELEOP_ROBOTS

ROBOTS_YML = Path(__file__).resolve().parents[2] / "robots_specs" / "robots.yml"


def _entries() -> dict:
    """`{id: {field: scalar}}` for the top-level scalars of each `robots:` entry."""
    if not ROBOTS_YML.exists():
        pytest.skip(f"no workspace checkout around the console ({ROBOTS_YML})")
    entries: dict = {}
    current = None
    for line in ROBOTS_YML.read_text().splitlines():
        start = re.match(r"^  - id:\s*(\S+)", line)
        if start:
            current = entries.setdefault(start.group(1), {})
            continue
        field = re.match(r"^    (\w+):\s*([^#\s][^#]*?)\s*(#.*)?$", line)
        if current is not None and field:
            current[field.group(1)] = field.group(2)
    return entries


def test_every_teleop_robot_is_a_robot_in_robots_yml() -> None:
    entries = _entries()
    for robot in TELEOP_ROBOTS:
        assert robot in entries, f"{robot} is not an id in {ROBOTS_YML}"


def test_the_ids_are_the_ones_the_spec_names() -> None:
    entries = _entries()
    assert entries[MYAGV]["name"] == "myAGV" and entries[MYAGV]["kind"] == "mobile_base"
    assert entries[AINEX]["name"] == "AiNex" and entries[AINEX]["kind"] == "humanoid"
    assert TELEOP_ROBOTS == (MYAGV, AINEX)
    assert SLAM_ROBOT == MYAGV


def test_each_teleop_robot_has_a_ros_file_with_a_stop_command() -> None:
    entries = _entries()
    for robot in TELEOP_ROBOTS:
        ros_file = ROBOTS_YML.parent / entries[robot]["ros"]
        assert re.search(r"^stop_command:", ros_file.read_text(), re.M), ros_file
        assert robot in STOP_COMMANDS


def test_the_robot_flag_takes_exactly_these_ids() -> None:
    action = next(a for a in build_parser()._actions if "--robot" in a.option_strings)
    assert tuple(action.choices) == TELEOP_ROBOTS
