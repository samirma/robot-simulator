#!/usr/bin/env python
"""Standalone check that the simulated lidar matches the myAGV's YDLidar X2.

The geometry here is the part that silently ruins a map rather than raising: a beam that
ranges the robot's own chassis, a fan cast from the base centre instead of the laser
mount, or a scan indexed clockwise. All three produce a plausible-looking /scan.

Built in a box world of known size so every expected range is arithmetic, not a fixture.

    python robots/myagv/test_scan.py [--scene /path/to/house.xml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The static transform myagv_active.launch publishes for base_footprint -> laser_frame.
OFFSET_X = 0.065
OFFSET_Z = 0.08
# ydlidar_ros_driver/launch/X2.launch.
RANGE_MIN = 0.1
RANGE_MAX = 12.0

ROOM = 3.0  # half-width of the test box, in metres

FAIL = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(label)


def box_world() -> mujoco.MjSpec:
    """A closed room centred on the origin: floor plus four walls at +/-ROOM."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(
        pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.55, 0.56, 0.58, 1]
    )
    for name, pos, size in (
        ("wall_xp", [ROOM, 0, 0.5], [0.05, ROOM, 0.5]),
        ("wall_xn", [-ROOM, 0, 0.5], [0.05, ROOM, 0.5]),
        ("wall_yp", [0, ROOM, 0.5], [ROOM, 0.05, 0.5]),
        ("wall_yn", [0, -ROOM, 0.5], [ROOM, 0.05, 0.5]),
    ):
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
            rgba=[0.8, 0.8, 0.82, 1],
        )
    return spec


def laser_origin(x: float, y: float, yaw: float) -> np.ndarray:
    return np.array([x + OFFSET_X * np.cos(yaw), y + OFFSET_X * np.sin(yaw), OFFSET_Z])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="house MJCF to attach into instead of a box")
    ap.add_argument("--beams", type=int, default=360)
    args = ap.parse_args()

    from robots.myagv import MyAGVRobot, MyAGVRobotConfig, MyAGVRobotView
    from tools.spawn_robot import laser_scan_ranges

    config = MyAGVRobotConfig()
    ns = config.robot_namespace

    spec = mujoco.MjSpec.from_file(args.scene) if args.scene else box_world()
    MyAGVRobot.add_robot_to_scene(
        config, spec, prefix=ns, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"compiled: {model.nbody} bodies, {model.ngeom} geoms")

    view = MyAGVRobotView(data, ns)
    base = view.get_move_group("base")

    root = f"{ns}{MyAGVRobot.robot_model_root_name()}"
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    check(f"root body {root!r} resolves", body >= 0, f"id {body}")

    origin_xy = np.zeros(2)
    if args.scene:
        from tools.spawn_robot import find_open_spot

        origin_xy, yaw0 = find_open_spot(args.scene)
        pose = np.eye(4)
        pose[:2, 3] = origin_xy
        base.pose = pose
        base.ctrl = np.array([origin_xy[0], origin_xy[1], 0.0])
        mujoco.mj_forward(model, data)
        print(f"starting from open floor at {np.round(origin_xy, 3)}")

    x, y, yaw = float(origin_xy[0]), float(origin_xy[1]), 0.0
    ranges = laser_scan_ranges(
        model, data, laser_origin(x, y, yaw), yaw, args.beams, RANGE_MAX, bodyexclude=body
    )

    print("\nself-occlusion:")
    # The chassis half-width is 0.115 m and the box reaches 0.1556 m fore and aft, so
    # anything ranging under 0.1 m from a laser sitting 65 mm ahead of centre is the robot.
    check("no beam ranges the robot itself", float(ranges.min()) >= RANGE_MIN,
          f"closest return {ranges.min():.4f} m")

    print("\nexcluding the robot is load-bearing:")
    # Without bodyexclude every beam should hit the chassis, which is the failure mode
    # the exclusion exists to prevent. If this ever stops happening, the check above has
    # become vacuous and the exclusion is no longer being tested by it.
    unshielded = laser_scan_ranges(
        model, data, laser_origin(x, y, yaw), yaw, 36, RANGE_MAX, bodyexclude=-1
    )
    check("without bodyexclude the chassis dominates",
          float(np.median(unshielded)) < 0.3, f"median {np.median(unshielded):.4f} m")

    if not args.scene:
        print("\ngeometry in the box world:")
        step = 2 * np.pi / args.beams
        # Beam index i points at angle -pi + i*step in the base frame (yaw = 0 here).
        def beam(deg: float) -> float:
            return float(ranges[int(round((np.radians(deg) + np.pi) / step)) % args.beams])

        # Walls are 0.05 m half-thickness slabs, so their inner faces are at ROOM - 0.05.
        face = ROOM - 0.05
        ahead = face - OFFSET_X       # laser sits 65 mm forward, so less room ahead...
        behind = face + OFFSET_X      # ...and more behind
        for label, deg, want in (
            ("forward (+x)", 0.0, ahead),
            ("left (+y)", 90.0, face),
            ("backward (-x)", 180.0, behind),
            ("right (-y)", -90.0, face),
        ):
            got = beam(deg)
            check(f"{label} ranges {want:.3f} m", abs(got - want) < 0.02, f"got {got:.4f} m")

        # The asymmetry above is the whole point: it only appears if the fan is cast from
        # the laser mount rather than the base centre.
        check("mount offset is applied", abs(beam(180.0) - beam(0.0) - 2 * OFFSET_X) < 0.02,
              f"back - front = {beam(180.0) - beam(0.0):.4f} m, want {2 * OFFSET_X:.4f}")

        print("\nbeam ordering (counter-clockwise from -pi):")
        # Rotating the robot +45 deg must move the near wall's bearing by -45 deg in the
        # scan. A clockwise fan would move it the other way and still look sane.
        pose = np.eye(4)
        base.pose = pose
        near = np.eye(4)
        near[:2, 3] = [face - 1.0, 0.0]  # 1 m from the +x wall
        base.pose = near
        mujoco.mj_forward(model, data)
        turned = laser_scan_ranges(
            model, data, laser_origin(face - 1.0, 0.0, np.radians(45.0)), np.radians(45.0),
            args.beams, RANGE_MAX, bodyexclude=body,
        )
        idx = int(np.argmin(turned))
        bearing = np.degrees(-np.pi + idx * step)
        check("a +45 deg yaw puts the near wall at -45 deg in the scan",
              abs(bearing + 45.0) < 3.0, f"bearing {bearing:.1f} deg")

    print("\nrange encoding:")
    misses = ranges[ranges > RANGE_MAX]
    check("misses are range_max + 1 (or there are none)",
          misses.size == 0 or np.allclose(misses, RANGE_MAX + 1.0),
          f"{misses.size} miss(es)")
    check("every value is finite", bool(np.isfinite(ranges).all()))
    check(f"{args.beams} beams returned", ranges.shape == (args.beams,), str(ranges.shape))

    print("\nRESULT:", "ok" if not FAIL else f"{len(FAIL)} check(s) failed: {FAIL}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
