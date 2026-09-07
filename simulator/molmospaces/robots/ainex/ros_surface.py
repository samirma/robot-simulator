"""The AiNex's ROS contract, as this engine presents it.

The contract itself -- the vendor's walking topics, the bus servo one, the gait and the
action library -- lives in `simulator/shared/ros_surfaces/ainex/`, because every engine has
to present exactly the same one and two copies of that loop would be two chances to drift.
What is left here is the MolmoSpaces-specific half: pulling the base move group out of a
`RobotView`, and reading the MJCF prefix off it.

That prefix used to be taken inside the shared loop as `getattr(view, "_namespace", "")`,
a private attribute of a MolmoSpaces class. Reaching into one engine's object from code
that claims to belong to every engine is exactly what kept this robot inside one of them.

See the shared package for the topic table and the reasoning about the gait.
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
            "the ainex ROS surface needs a robot with a planar base; "
            f"this one has move groups {view.move_group_ids()}"
        )
    return view.get_move_group("base")


def attach_ros(bus, view, model, camera: str | None, camera_size, jpeg_quality: int,
               control_hz: float, watchdog_s: float, scan: dict | None = None,
               depth: dict | None = None, extra: dict | None = None, scene_option=None,
               camera_period: float = 0.0, world_reset=None):
    """Wire this engine's AiNex onto a bus, via the shared contract."""
    from ros_surfaces.ainex import attach_ros as shared_attach_ros

    return shared_attach_ros(
        bus,
        _base_of(view),
        model,
        getattr(view, "_namespace", "") or "",
        camera,
        camera_size,
        jpeg_quality,
        control_hz,
        watchdog_s,
        scan=scan,
        depth=depth,
        extra=extra,
        scene_option=scene_option,
        camera_period=camera_period,
        world_reset=world_reset,
    )


def serve_ros(port: int, view, model, camera: str | None, camera_size, jpeg_quality: int,
              control_hz: float, watchdog_s: float, scan: dict | None = None,
              depth: dict | None = None, extra: dict | None = None,
              host: str = "0.0.0.0", namespace: str = ""):
    """The single-robot path, in this engine's terms. `robots/ainex/test_ros.py` uses it."""
    from ros_surfaces.ainex import serve_ros as shared_serve_ros

    return shared_serve_ros(
        port,
        _base_of(view),
        model,
        getattr(view, "_namespace", "") or "",
        camera,
        camera_size,
        jpeg_quality,
        control_hz,
        watchdog_s,
        scan=scan,
        depth=depth,
        extra=extra,
        host=host,
        namespace=namespace,
    )
