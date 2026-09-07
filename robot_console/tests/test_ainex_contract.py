"""The two projects' copies of the AiNex's topic names, held equal.

Same mechanism and same reason as `tests/arm/test_ros_contract.py`: the console cannot
import the simulator -- it has to install and run with no simulator checkout at all -- so
the vendor's names are written down on both sides, and a duplicated constant that drifts
is worse than no constant. The failure it prevents is silent: a fleet check asking for a
topic nobody publishes reports a robot missing that is right there.

Skips when the sibling simulator is not checked out, exactly as the arm's does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from robot_console import ainex_topics as ac

#: The simulator's copy. Stdlib-only on that side, so it loads here by path the same way
#: the SO-101 surface does.
SIM_TOPICS = (
    Path(__file__).resolve().parents[2]
    / "simulator" / "shared" / "ros_surfaces" / "ainex" / "topics.py"
)


def _sim():
    if not SIM_TOPICS.exists():
        pytest.skip(f"sibling simulator checkout not present at {SIM_TOPICS}")
    spec = importlib.util.spec_from_file_location("_ainex_sim_topics", SIM_TOPICS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ainex_sim_topics"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules["_ainex_sim_topics"]
        raise
    return module


def test_every_name_matches_the_simulators() -> None:
    s = _sim()
    assert ac.TOPIC_SET_WALKING_PARAM == s.TOPIC_SET_WALKING_PARAM
    assert ac.TOPIC_APP_ACTION == s.TOPIC_APP_ACTION
    assert ac.TOPIC_BUS_SERVO_SET == s.TOPIC_BUS_SERVO_SET
    assert ac.TOPIC_IS_WALKING == s.TOPIC_IS_WALKING
    assert ac.TOPIC_JOINT_STATES == s.TOPIC_JOINT_STATES
    assert ac.TOPIC_IMU == s.TOPIC_IMU
    assert ac.TOPIC_CAMERA == s.TOPIC_CAMERA


def test_the_contract_topics_are_all_names_the_simulator_knows() -> None:
    """Nothing may be required of the robot that the other side has never heard of."""
    s = _sim()
    known = {v for k, v in vars(s).items() if k.startswith("TOPIC_")}
    assert set(ac.CONTRACT_TOPICS) <= known


def test_a_humanoid_is_not_a_mobile_base() -> None:
    """The AiNex is commanded as a walking state machine and has no wheels.

    Checking it against the myAGV's contract demanded `/cmd_vel` and `/odom` of a biped,
    which is how `--robots so101,ainex` failed its fleet check with every topic it does
    present sitting on the wire.
    """
    from robot_console.topics import TOPIC_CMD_VEL, TOPIC_ODOM

    assert TOPIC_CMD_VEL not in ac.CONTRACT_TOPICS
    assert TOPIC_ODOM not in ac.CONTRACT_TOPICS


def test_the_invented_lidar_is_not_required() -> None:
    """The simulator publishes `/scan` for this robot and its own constants call that a
    departure -- the real AiNex has no lidar. Requiring it here would make the console
    assert a simulator's invention as though it were the hardware."""
    assert "/scan" not in ac.CONTRACT_TOPICS
