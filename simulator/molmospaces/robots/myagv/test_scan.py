#!/usr/bin/env python
"""Standalone check that the simulated lidar is the myAGV's YDLidar X2, as its launch has it.

The geometry here is the part that silently ruins a map rather than raising: a beam that
ranges the robot's own chassis, a fan cast from the base centre instead of the laser
mount, a scan not in `laser_frame` (which the launch turns a half-turn from the base), or
a scan indexed clockwise. All of them produce a plausible-looking /scan.

Built in a box world of known size so every expected range is arithmetic, not a fixture.
The numbers are read from the contract (`ros_surfaces/myagv.py`), which transcribes
`robots_specs/myagv/ros.yml`.

    python robots/myagv/test_scan.py [--scene /path/to/house.xml]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))

from ros_surfaces import myagv as contract  # noqa: E402

_, _, (OFFSET_X, _, OFFSET_Z), (LASER_YAW, _, _) = contract.STATIC_TRANSFORMS[
    contract.NODE_BASE2LASER]
RANGE_MIN, RANGE_MAX = contract.SCAN_RANGE_MIN, contract.SCAN_RANGE_MAX

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="house MJCF to attach into instead of a box")
    ap.add_argument("--beams", type=int, default=360)
    args = ap.parse_args()

    from robots.myagv import MyAGVRobot, MyAGVRobotConfig, MyAGVRobotView
    from mujoco_bridge import laser_scan_ranges

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

        origin_xy, _yaw0 = find_open_spot(args.scene)
        pose = np.eye(4)
        pose[:2, 3] = origin_xy
        base.pose = pose
        base.ctrl = np.array([origin_xy[0], origin_xy[1], 0.0])
        mujoco.mj_forward(model, data)
        print(f"starting from open floor at {np.round(origin_xy, 3)}")

    x, y = float(origin_xy[0]), float(origin_xy[1])

    def sweep(px: float, py: float, yaw: float) -> np.ndarray:
        return np.array(contract.scan_ranges(model, data, px, py, 0.0, yaw, args.beams, body))

    ranges = sweep(x, y, 0.0)
    bearings = np.degrees(np.array(contract.scan_bearings(args.beams)))
    valid = ranges > 0.0

    print("\nself-occlusion:")
    check("no beam ranges the robot itself",
          float(ranges[valid].min()) >= RANGE_MIN if valid.any() else False,
          f"closest return {ranges[valid].min():.4f} m" if valid.any() else "no returns")

    print("\nexcluding the robot is load-bearing:")
    # Without bodyexclude every beam should hit the chassis, which is the failure mode
    # the exclusion exists to prevent. If this ever stops happening, the check above has
    # become vacuous and the exclusion is no longer being tested by it.
    laser = np.array([x + OFFSET_X, y, OFFSET_Z])
    unshielded = laser_scan_ranges(model, data, laser, 0.0, 36, RANGE_MAX, bodyexclude=-1)
    check("without bodyexclude the chassis dominates",
          float(np.median(unshielded)) < 0.3, f"median {np.median(unshielded):.4f} m")

    print("\nX2.launch conventions:")
    check(f"{args.beams} beams, from -180 to 180 deg",
          ranges.shape == (args.beams,) and abs(bearings[0] + 180) < 1e-9
          and abs(bearings[-1] - 180) < 1e-9, f"{bearings[0]:.1f}..{bearings[-1]:.1f}")
    wedge = (bearings >= -50.0) & (bearings <= 50.0)
    check("the ignore_array wedge -50..50 deg reads 0.0", bool(np.all(ranges[wedge] == 0.0)),
          f"{int(np.count_nonzero(ranges[wedge]))} non-zero")
    check("every other value is 0.0 or inside [range_min, range_max]",
          bool(np.all((ranges == 0.0) | ((ranges >= RANGE_MIN) & (ranges <= RANGE_MAX)))))

    if not args.scene:
        print("\ngeometry in the box world (laser_frame is the base turned a half-turn):")

        def beam(deg: float) -> float:
            return float(ranges[int(np.argmin(np.abs(bearings - deg)))])

        # Walls are 0.05 m half-thickness slabs, so their inner faces are at ROOM - 0.05.
        face = ROOM - 0.05
        ahead = face - OFFSET_X       # the laser sits 65 mm forward of the base centre
        for label, deg, want in (
            ("base +x is bearing 180", 180.0, ahead),
            ("base +x is bearing -180", -180.0, ahead),
            ("base +y (left) is bearing -90", -90.0, face),
            ("base -y (right) is bearing +90", 90.0, face),
        ):
            got = beam(deg)
            check(f"{label}: {want:.3f} m", abs(got - want) < 0.02, f"got {got:.4f} m")
        # Base -x is bearing 0, inside the blanked wedge; bearing -60 is just outside it,
        # base direction 120 deg, towards +y and -x.
        diag = beam(-60.0)
        want = min((face + OFFSET_X) / abs(math.cos(math.radians(120))),
                   face / abs(math.sin(math.radians(120))))
        check("bearing -60 (base 120 deg) ranges the nearer wall", abs(diag - want) < 0.03,
              f"got {diag:.4f} m, want {want:.4f} m")

        print("\nbeam ordering (counter-clockwise in laser_frame):")
        # Turned +45 deg, 1 m from the +x wall: the wall's normal is at base bearing -45,
        # which is laser bearing -45 + 180 = 135. A clockwise fan would put it at -135.
        near = np.eye(4)
        near[:2, 3] = [face - 1.0, 0.0]
        base.pose = near
        mujoco.mj_forward(model, data)
        turned = sweep(face - 1.0, 0.0, math.radians(45.0))
        masked = np.where(turned > 0.0, turned, np.inf)
        bearing = float(bearings[int(np.argmin(masked))])
        check("a +45 deg yaw puts the near wall at bearing 135 deg",
              abs(bearing - 135.0) < 3.0, f"bearing {bearing:.1f} deg")

    print("\nRESULT:", "ok" if not FAIL else f"{len(FAIL)} check(s) failed: {FAIL}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
