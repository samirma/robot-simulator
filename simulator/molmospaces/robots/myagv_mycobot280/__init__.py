"""myAGV + myCobot 280 support for MolmoSpaces: the myAGV's planar-base adapter on the
composite's generated model (see `robots/planar_base.py`)."""

from ..planar_base import MyAGVRobot as MyAGVMyCobot280Robot
from ..planar_base import config_for

MyAGVMyCobot280RobotConfig = config_for("myagv_mycobot280")

__all__ = ["MyAGVMyCobot280Robot", "MyAGVMyCobot280RobotConfig"]
