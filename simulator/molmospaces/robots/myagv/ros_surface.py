"""The myAGV's ROS contract, as this engine presents it.

The contract itself -- `cmd_vel` in, `odom`/camera/`/scan` out, on the vendor's topic
names -- lives in `simulator/shared/ros_surfaces/myagv.py`, because every engine has to
present exactly the same one and two copies of that loop would be two chances to drift.
What is left here is the MolmoSpaces-specific half: pulling the base move group out of a
`RobotView`, which is this engine's way of saying "the thing with a pose and a ctrl".

See the shared module for the topic table and the reasoning about the drive model.
"""

from __future__ import annotations

import sys
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[2]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))


def _base_of(view):
    """This engine's way of saying "the thing with a pose and a ctrl"."""
    if "base" not in view.move_group_ids():
        raise SystemExit(
            "the myagv ROS surface needs a robot with a mobile base; "
            f"this one has move groups {view.move_group_ids()}"
        )
    return view.get_move_group("base")


def attach_ros(bus, view, model, camera: str | None, camera_size, jpeg_quality: int,
               control_hz: float, watchdog_s: float, scan: dict | None = None,
               depth: dict | None = None, extra: dict | None = None, scene_option=None,
               camera_period: float = 0.0, world_reset=None):
    """Wire this engine's myAGV onto a bus, via the shared contract.

    `extra` is accepted and ignored: `tools/spawn_robot.py` passes the same bag to every
    ROS surface, and the AiNex is the one that reads it.
    """
    from ros_surfaces.myagv import attach_ros as _attach_ros

    return _attach_ros(
        bus,
        _base_of(view),
        model,
        camera,
        camera_size,
        jpeg_quality,
        control_hz,
        watchdog_s,
        scan=scan,
        depth=depth,
        scene_option=scene_option,
        camera_period=camera_period,
        world_reset=world_reset,
    )


def serve_ros(port: int, view, model, camera: str | None, camera_size, jpeg_quality: int,
              control_hz: float, watchdog_s: float, scan: dict | None = None,
              depth: dict | None = None, extra: dict | None = None,
              host: str = "0.0.0.0", namespace: str = ""):
    """The single-robot path, kept for callers that only ever want one robot."""
    from ros_surfaces.myagv import serve_ros as _serve_ros

    return _serve_ros(
        port,
        _base_of(view),
        model,
        camera,
        camera_size,
        jpeg_quality,
        control_hz,
        watchdog_s,
        scan=scan,
        depth=depth,
        host=host,
        namespace=namespace,
    )
