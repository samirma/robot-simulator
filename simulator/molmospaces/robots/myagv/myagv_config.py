"""Experiment config for the Elephant Robotics myAGV Pi 2023."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from mujoco import MjData

from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

from robots_spec import spec_dir

from .myagv import MyAGVRobot
from .myagv_view import MyAGVRobotView

# The model.xml + meshes are the engine-neutral spec, shared across simulator engines.
ROBOT_DIR = spec_dir("myagv")


class MyAGVCameraSystem(CameraSystemConfig):
    """Just the forward-facing chassis camera — that is all the hardware has.

    Resolution is deliberately modest: this feed is streamed frame-by-frame to the
    teleop client, and raw RGB at 640x480 is ~920 KB per frame.
    """

    img_resolution: tuple[int, int] = (640, 480)  # (width, height)
    cameras: list[Any] = [
        MjcfCameraConfig(
            name="front_camera",
            mjcf_name="front_camera",
            robot_namespace="robot_0/",
            fov=42.0,
        )
    ]


class MyAGVRobotConfig(BaseRobotConfig):
    robot_cls: type[MyAGVRobot] | None = MyAGVRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = MyAGVRobot
    robot_view_factory: RobotViewFactory | None = MyAGVRobotView
    robot_namespace: str = "robot_0/"
    name: str = "myagv"

    robot_xml_path: Path = Path("model.xml")
    robot_dir: Path = ROBOT_DIR

    # It drives on the floor; there is no pedestal to stand it on.
    base_size: list[float] | None = None

    # [x, y, theta] in world metres/radians.
    init_qpos: dict[str, Any] = {"base": np.array([0.0, 0.0, 0.0])}
    init_qpos_noise_range: dict[str, Any] | None = None

    # Relative by default: the command is added to the current pose each step, so a
    # blocked robot does not accumulate an ever-growing target, and releasing the keys
    # parks it (JointRelPosController has a zero-command deadband).
    command_mode: dict[str, str | None] = {"base": "holo_joint_rel_planar_position"}

    gravcomp: bool = False

    def model_post_init(self, __context):
        super().model_post_init(__context)
        mode = self.command_mode.get("base")
        assert mode in ("holo_joint_planar_position", "holo_joint_rel_planar_position"), (
            f"unsupported base command mode {mode!r}"
        )
