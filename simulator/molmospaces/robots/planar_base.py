"""MolmoSpaces adapters for the mobile robots that ride the myAGV's planar base.

The myAGV + myCobot 280 and the ROSMASTER X3 PLUS are driven exactly as the myAGV is:
three world-aligned virtual joints (`base_x`, `base_y`, `base_theta`) under a `base` body,
their arms held by position servos the shared ROS surfaces command. So this engine needs
nothing of its own for them beyond pointing the myAGV's robot class, view and config at
each robot's generated model (`shared/robots/<id>/model.xml`, `shared/robot_models.py`).
"""

from __future__ import annotations

from pathlib import Path

import robots_spec

from .myagv import MyAGVRobot, MyAGVRobotConfig


def config_for(robot_id: str) -> type[MyAGVRobotConfig]:
    """A `BaseRobotConfig` for `robot_id`'s planar-base model."""
    xml = robots_spec.model_xml(robot_id)

    class PlanarBaseRobotConfig(MyAGVRobotConfig):
        name: str = robot_id
        robot_xml_path: Path = Path(xml.name)
        robot_dir: Path = xml.parent

    PlanarBaseRobotConfig.__name__ = f"{robot_id}_config"
    return PlanarBaseRobotConfig


__all__ = ["MyAGVRobot", "config_for"]
