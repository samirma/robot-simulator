"""The console's myAGV names and the simulator's must be the same names.

`topics.py` is the console's copy of `robots_specs/myagv/ros.yml`; the simulator's is
`simulator/shared/ros_surfaces/myagv.py`, stdlib-only at import so it loads here by path.
Skips when the sibling simulator is not checked out, as the arm's and the AiNex's do.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from robot_console import topics
from robot_console.slam import scan

SURFACE = (Path(__file__).resolve().parents[2] / "simulator" / "shared" / "ros_surfaces"
           / "myagv.py")


@pytest.fixture(scope="module")
def sim():
    if not SURFACE.exists():
        pytest.skip(f"sibling simulator checkout not present at {SURFACE}")
    spec = importlib.util.spec_from_file_location("_sim_myagv", SURFACE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sim_myagv"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_sim_myagv", None)
    return module


def test_every_topic_and_type_matches(sim) -> None:
    assert {name: mtype for name, mtype, *_ in sim.TOPICS} == topics.CONTRACT_TOPICS


def test_every_service_and_type_matches(sim) -> None:
    assert {name: stype for name, stype, _node in sim.SERVICES} == topics.CONTRACT_SERVICES


def test_the_command_limit_and_stop_match(sim) -> None:
    assert sim.CMD_VEL_LIMIT == topics.CMD_VEL_LIMIT
    assert all(v == 0.0 for part in sim.STOP_COMMAND.values() for v in part.values())


def test_the_frames_match(sim) -> None:
    assert (sim.FRAME_ODOM, sim.FRAME_BASE, sim.FRAME_LASER, sim.FRAME_CAMERA,
            sim.FRAME_IMU) == (topics.FRAME_ODOM, topics.FRAME_BASE, topics.FRAME_LASER,
                               topics.FRAME_CAMERA, topics.FRAME_IMU)
    assert sim.PARAM_ROBOT_DESCRIPTION == topics.PARAM_ROBOT_DESCRIPTION


def test_the_lidar_mount_and_ranges_match_the_mapper(sim) -> None:
    """`slam/scan.py` projects scans with its own copy of the mount; a drift would put
    every wall in the wrong place with nothing failing."""
    parent, child, (x, y, _z), (yaw, _pitch, _roll) = sim.STATIC_TRANSFORMS[sim.NODE_BASE2LASER]
    assert (parent, child) == (topics.FRAME_BASE, topics.FRAME_LASER)
    assert (x, y) == (scan.LASER_OFFSET_X, scan.LASER_OFFSET_Y)
    assert math.isclose(yaw, scan.LASER_YAW)
    assert (sim.SCAN_RANGE_MIN, sim.SCAN_RANGE_MAX) == (scan.DEFAULT_RANGE_MIN,
                                                       scan.DEFAULT_RANGE_MAX)
