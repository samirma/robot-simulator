"""The console's teleop robot ids, as `--robot` takes them.

`--robot` takes a robot id from `robots_specs/robots.yml` (console spec §2.1); the console
keeps its own copy as contract constants so it installs with no workspace around it. That
the copy equals the file is a workspace test (`../tests/test_contract_parity.py`,
`DiscoveryAndIds`); this checks the console uses its copy consistently.
"""

from __future__ import annotations

from robot_console.cli import build_parser
from robot_console.robots import (
    AINEX, MYAGV, MYAGV_MYCOBOT280, ROSMASTER_X3_PLUS, SLAM_ROBOT, STOP_COMMANDS, TELEOP_ROBOTS,
    WHEELED_ROBOTS,
)


def test_the_robot_flag_takes_exactly_these_ids() -> None:
    action = next(a for a in build_parser()._actions if "--robot" in a.option_strings)
    assert tuple(action.choices) == TELEOP_ROBOTS


def test_teleop_drives_every_mobile_robot_and_slam_the_base() -> None:
    """Every robot but the fixed arm moves, so teleop drives all four; slam maps with the
    myAGV, and smoke drives the three /cmd_vel bases."""
    assert TELEOP_ROBOTS == (MYAGV, AINEX, MYAGV_MYCOBOT280, ROSMASTER_X3_PLUS)
    assert WHEELED_ROBOTS == (MYAGV, MYAGV_MYCOBOT280, ROSMASTER_X3_PLUS)
    assert "so101" not in TELEOP_ROBOTS
    assert SLAM_ROBOT == MYAGV


def test_every_teleop_robot_has_a_stop_command() -> None:
    assert set(STOP_COMMANDS) == set(TELEOP_ROBOTS)
