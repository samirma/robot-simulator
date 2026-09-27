"""The console's teleop robot ids, as `--robot` takes them.

`--robot` takes a robot id from `robots_specs/robots.yml` (console spec §2.1); the console
keeps its own copy as contract constants so it installs with no workspace around it. That
the copy equals the file is a workspace test (`../tests/test_contract_parity.py`,
`DiscoveryAndIds`); this checks the console uses its copy consistently.
"""

from __future__ import annotations

from robot_console.cli import build_parser
from robot_console.robots import AINEX, MYAGV, SLAM_ROBOT, STOP_COMMANDS, TELEOP_ROBOTS


def test_the_robot_flag_takes_exactly_these_ids() -> None:
    action = next(a for a in build_parser()._actions if "--robot" in a.option_strings)
    assert tuple(action.choices) == TELEOP_ROBOTS


def test_teleop_drives_the_base_and_the_humanoid_and_slam_the_base() -> None:
    assert TELEOP_ROBOTS == (MYAGV, AINEX)
    assert SLAM_ROBOT == MYAGV


def test_every_teleop_robot_has_a_stop_command() -> None:
    assert set(STOP_COMMANDS) == set(TELEOP_ROBOTS)
