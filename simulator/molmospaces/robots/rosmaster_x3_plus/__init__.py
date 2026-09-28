"""ROSMASTER X3 PLUS support for MolmoSpaces: the myAGV's planar-base adapter on the X3
PLUS's generated model (see `robots/planar_base.py`)."""

from ..planar_base import MyAGVRobot as RosmasterX3PlusRobot
from ..planar_base import config_for

RosmasterX3PlusRobotConfig = config_for("rosmaster_x3_plus")

__all__ = ["RosmasterX3PlusRobot", "RosmasterX3PlusRobotConfig"]
