"""The myAGV's ROS interface, as this engine presents it.

The interface itself -- every topic, service, parameter, node, frame and rate of
`robots_specs/myagv/ros.yml` -- lives in `simulator/shared/ros_surfaces/myagv.py`,
because every engine has to present exactly the same one. What is left here is the
MolmoSpaces-specific half: pulling the base move group out of a `RobotView`, which is
this engine's way of saying "the thing with a pose and a ctrl".
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


def attach_ros(bus, view, model, camera: str | None, *, jpeg_quality: int = 80,
               lidar: dict | None = None, scene_option=None, world_reset=None,
               prefix: str = ""):
    """Wire this engine's myAGV onto a bus, via the shared surface."""
    from ros_surfaces.myagv import attach_ros as _attach_ros

    return _attach_ros(
        bus, _base_of(view), model, camera,
        jpeg_quality=jpeg_quality, lidar=lidar, scene_option=scene_option,
        world_reset=world_reset,
        # The MJCF prefix. Topics never see it -- see the note in `spawn_robot.py` about
        # the two prefixes being different things.
        prefix=prefix or getattr(view, "_namespace", "") or "",
    )


def serve_ros(port: int, view, model, camera: str | None, *, host: str = "0.0.0.0",
              namespace: str = "", **kwargs):
    """The single-robot path, kept for callers that only ever want one robot."""
    from ros_surfaces.myagv import serve_ros as _serve_ros

    return _serve_ros(port, _base_of(view), model, camera, host=host,
                      namespace=namespace, **kwargs)
