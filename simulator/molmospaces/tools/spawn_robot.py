#!/usr/bin/env python
"""Spawn robots into a MolmoSpaces house and render, view or serve them.

    python   tools/spawn_robot.py so101,myagv --scene <house.xml> --headless
    mjpython tools/spawn_robot.py ainex --scene <house.xml> --ros-port 0
    python   tools/spawn_robot.py so101 --scene <house.xml> --render out.png

The tool itself -- command line, placement, task staging, start-up reports, the ROS
fleet, rendering and the loop -- is `simulator/shared/spawn.py`, the same for both
engines. What is here is only what MolmoSpaces knows and the other engine does not:

* **the worktop** is the iTHOR/ProcTHOR surface at working height holding the task's
  categories (`TARGET`), chosen by `tools/scene_placement.py` from the house's own
  metadata, with the task robot mounted at its rim looking in (`place_arm_on_table`);
* **open floor** is read off the house's occupancy map (`find_robot_placement`);
* **a robot is grafted** through its MolmoSpaces adapter in `robots/<id>/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
# The shared layer: the spawn tool, the wire, the robot specs.
_SHARED = SIM_ROOT.parent / "shared"
if _SHARED.is_dir() and str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

import placement  # noqa: E402
import robots_spec  # noqa: E402
import spawn  # noqa: E402

# Engine-neutral helpers, re-exported: tools and robot self-tests here import them
# `from tools.spawn_robot`.
from mujoco_bridge import (  # noqa: E402,F401
    TARGET_LEAD_M,
    TARGET_LEAD_RAD,
    PlanarSetpoint,
    SensorStreams,
    SensorTopics,
    laser_scan_ranges,
)

# id -> (module, config class, robot class): this engine's adapter for each simulated
# robot, imported lazily so one robot does not require the others to be installed.
ADAPTERS = {
    "so101": ("robots.so101", "SO101RobotConfig", "SO101Robot"),
    "myagv": ("robots.myagv", "MyAGVRobotConfig", "MyAGVRobot"),
    "ainex": ("robots.ainex", "AiNexRobotConfig", "AiNexRobot"),
}

#: The object categories that rank the house's surfaces when the worktop is chosen: the
#: task's own pair. A surface holding both is the one the task is staged on.
TARGET = ("plate", "apple")

#: The annulus each worktop robot's mount search aims the worktop into. The SO-101's is
#: `placement.ARM_REACH`; the AiNex's is the same fraction of its measured 0.289 m claw
#: sweep (see shared/ainex_model.py).
MOUNT_REACH = {"so101": placement.ARM_REACH, "ainex": (0.11, 0.25)}

# Stand-in boxes the mount search predicts for a surface with nothing on it.
SPAWN_OBJECT_HALF = 0.025
#: The base footprint the mount search keeps on the worktop.
MOUNT_FOOTPRINT = 0.14


def load_robot(name: str):
    """(config class, robot class) of this engine's adapter for `name`."""
    try:
        robots_spec.check_simulated([name])
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if name not in ADAPTERS:
        raise SystemExit(f"robots.yml marks {name!r} simulated, but this engine has no "
                         "adapter for it in robots/")
    import importlib

    module_name, config_attr, robot_attr = ADAPTERS[name]
    module = importlib.import_module(module_name)
    return getattr(module, config_attr), getattr(module, robot_attr)


def find_open_spot(
    scene_path: str, clearance_band: tuple[float, float] = (0.05, 1.3)
) -> tuple[np.ndarray, float]:
    """Return (xy position, yaw) for the most open floor spot, facing the room centre.

    Obstacles are taken to be collision geoms whose centres sit in the height band a
    robot on its pedestal would sweep through; the spot maximising distance to the
    nearest such geom is chosen.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    obstacles = np.array(
        [
            data.geom_xpos[i][:2]
            for i in range(model.ngeom)
            if model.geom_contype[i] != 0 and clearance_band[0] < data.geom_xpos[i][2] < clearance_band[1]
        ]
    )
    if obstacles.size == 0:
        return np.zeros(2), 0.0

    lo, hi = obstacles.min(axis=0), obstacles.max(axis=0)
    centre = (lo + hi) / 2

    xs = np.arange(lo[0], hi[0], 0.1)
    ys = np.arange(lo[1], hi[1], 0.1)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    # distance from every candidate to its nearest obstacle
    clearances = np.linalg.norm(grid[:, None, :] - obstacles[None, :, :], axis=-1).min(axis=1)

    best = grid[int(np.argmax(clearances))]
    to_centre = centre - best
    yaw = float(np.arctan2(to_centre[1], to_centre[0]))
    print(
        f"placing robot at ({best[0]:.2f}, {best[1]:.2f}) "
        f"clearance {clearances.max():.2f} m, facing room centre (yaw {np.degrees(yaw):.0f} deg)",
        file=sys.stderr,
    )
    return best, yaw



def _report_wanted(mount, targets, want: tuple[str, ...], reach) -> None:
    """Say, per `--target` category, how far the nearest such object is from the mount.

    `--target` ranks surfaces; it cannot put an object within reach that is not there.
    So whether the plate the user asked for is actually inside the working annulus has
    to be read off the chosen mount, and it is printed rather than assumed. Measured on
    the house's own objects; the task's staged plate is reported separately.
    """
    if not want:
        return
    for category in want:
        best = None
        for t in targets:
            label = (t.object_category or t.object_name or "").lower()
            if category not in label:
                continue
            d = float(np.hypot(*(np.asarray(t.object_xyz[:2]) - np.asarray(mount.xy))))
            if best is None or d < best[0]:
                best = (d, t)
        if best is None:
            print(f"  {category}: not in this scene", file=sys.stderr)
            continue
        d, t = best
        where = "" if t.support_name == mount.target.support_name else \
            f", on another surface ({t.support_name.split('_')[0]})"
        status = "in reach" if reach[0] <= d <= reach[1] else "OUT of reach"
        print(f"  {category}: {d:.2f} m from the mount ({status} {reach[0]:.2f}-{reach[1]:.2f} m){where}",
              file=sys.stderr)


def place_arm_on_table(scene_path: str, model, data, robot: str, reach, n_spawn: int,
                       want: tuple[str, ...] = (), edge_bias: bool = True):
    """Choose the worktop and where on it the task robot stands: (mount, predicted objects).

    A surface holding the `want` categories ranks first. A house whose surfaces hold
    nothing graspable still has a worktop: the search is then run against `n_spawn`
    predicted stand-in objects, which are returned but never staged -- the task brings
    its own objects.
    """
    from tools.scene_placement import (
        GraspTarget,
        dynamic_clutter,
        find_grasp_targets,
        find_supports,
        find_tabletop_mount,
        load_scene_map,
        static_blockers,
    )

    thormap = load_scene_map(scene_path)
    targets = find_grasp_targets(scene_path, model, data, thormap=thormap)
    if want and targets:
        # `--target` asks for a surface holding particular things, which is how two
        # engines get set up around the *same* objects: iTHOR's island carries a Bowl
        # and an Apple, and so does the RoboCasa worktop, but only if both are told
        # which pair to build the scene around. Ranking rather than filtering, because
        # a surface that has one of the two is still better than one that has neither,
        # and dropping every other candidate would turn a near miss into an error.
        def _wanted(target) -> int:
            category = (target.object_category or target.object_name or "").lower()
            return sum(1 for w in want if w in category)

        on_surface: dict[str, int] = {}
        for target in targets:
            on_surface[target.support_name] = on_surface.get(target.support_name, 0) + _wanted(target)
        # Stable sort, so the original best-first order breaks ties.
        targets = sorted(
            targets,
            key=lambda t: (-_wanted(t), -on_surface.get(t.support_name, 0)),
        )
        matched = [t for t in targets if _wanted(t)]
        print(
            f"--target {','.join(want)}: {len(matched)} of {len(targets)} candidate objects match",
            file=sys.stderr,
        )
    if targets:
        # Most targets share a surface, so the blocker sweep -- which walks every geom in
        # the house -- is cached by the height it was taken at.
        blocker_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        fallback = None
        for target in targets:
            key = round(target.support_top_z, 2)
            if key not in blocker_cache:
                height = placement.ROBOT_HEIGHT.get(robot, 1.0)
                blocker_cache[key] = (
                    static_blockers(model, data, target.support_top_z, height=height),
                    dynamic_clutter(model, data, target.support_top_z, height=height),
                )
            mount = find_tabletop_mount(
                target,
                reach_range=reach,
                footprint=MOUNT_FOOTPRINT,
                blockers=blocker_cache[key][0],
                clutter=blocker_cache[key][1],
                body_radius=placement.ROBOT_RADIUS.get(robot),
                model=model,
                data=data,
                edge_bias=edge_bias,
            )
            # A surface with no cell both clear and in reach is not the right surface: the
            # next target is usually the same table seen from a different object, and after
            # that a different table entirely. Keep the best rejected one as a floor.
            if mount.clear and mount.n_in_reach:
                _report_wanted(mount, targets, want, reach)
                return mount, None
            if fallback is None or mount.n_in_reach > fallback.n_in_reach:
                fallback = mount
        print(
            f"no surface in {Path(scene_path).name} has room for {robot} clear of its "
            f"surroundings; using the roomiest spot found",
            file=sys.stderr,
        )
        _report_wanted(fallback, targets, want, reach)
        return fallback, None

    supports = find_supports(scene_path, model, data)
    if not supports:
        raise SystemExit(
            f"no work surface at arm height in {Path(scene_path).name}: nothing to mount "
            f"{robot} on, and a worktop robot in a scene with no worktop is a start-up error. "
            "Try another scene index."
        )
    if n_spawn <= 0:
        listing = "\n".join(
            f"  {s[0]} ({s[1] or '?'}) top z={s[2][2] + s[3][2] / 2:.2f} "
            f"{s[3][0]:.2f}x{s[3][1]:.2f} m"
            for s in supports[:8]
        )
        raise SystemExit(
            f"no graspable objects on any surface in {Path(scene_path).name}, and "
            f"--spawn-objects 0 forbids adding some. Candidate surfaces:\n{listing}"
        )

    s_name, s_cat, s_centre, s_dims, s_rects = supports[0]
    xy_min = s_centre[:2] - s_dims[:2] / 2
    xy_max = s_centre[:2] + s_dims[:2] / 2
    top_z = float(s_centre[2] + s_dims[2] / 2)
    print(
        f"no graspables in the scene; choosing the worktop {s_cat or s_name} "
        f"(top {top_z:.2f} m) against {n_spawn} predicted objects",
        file=sys.stderr,
    )

    # A spread of stand-in objects the mount can be chosen against.
    dims = np.array([SPAWN_OBJECT_HALF * 2] * 3)
    centre = (xy_min + xy_max) / 2
    half = np.maximum((xy_max - xy_min) / 2 - 0.12, 0.0)
    predicted = []
    for i in range(n_spawn):
        angle = 2 * np.pi * i / max(n_spawn, 1)
        offset = half * np.array([np.cos(angle), np.sin(angle)]) * 0.6
        predicted.append(
            np.array([centre[0] + offset[0], centre[1] + offset[1], top_z + SPAWN_OBJECT_HALF + 0.002])
        )

    target = GraspTarget(
        support_name=s_name,
        support_category=s_cat,
        support_top_z=top_z,
        object_name="spawned_object_0",
        object_category="spawned box",
        object_xyz=predicted[0],
        n_objects_on_support=len(predicted),
        reach_slack=0.0,
        support_xy_min=xy_min,
        support_xy_max=xy_max,
        objects_on_support=tuple((p, dims) for p in predicted),
        support_top_rects=s_rects,
        support_body_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s_name)),
    )
    mount = find_tabletop_mount(
        target,
        reach_range=reach,
        footprint=MOUNT_FOOTPRINT,
        blockers=static_blockers(model, data, top_z, height=placement.ROBOT_HEIGHT.get(robot, 1.0)),
        clutter=dynamic_clutter(model, data, top_z, height=placement.ROBOT_HEIGHT.get(robot, 1.0)),
        model=model,
        data=data,
        body_radius=placement.ROBOT_RADIUS.get(robot),
        edge_bias=edge_bias,
    )
    return mount, (xy_min, xy_max, top_z, n_spawn)



class MolmoSpacesEngine:
    """MolmoSpaces' half of `shared/spawn.py`: its houses, its worktop, its adapters."""

    name = "molmospaces"

    @staticmethod
    def add_scene_args(ap) -> None:
        ap.add_argument("--scene", required=True, metavar="XML",
                        help="the house MJCF, as tools/resolve_scene.py resolves "
                             "ithor:<n> / procthor:<n>")

    @staticmethod
    def load_scene(args):
        # Compiled bare once: the worktop and floor searches both want a model, and
        # neither the robots nor the task may be in it or they would be their own
        # nearest obstacles.
        spec = mujoco.MjSpec.from_file(args.scene)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return spawn.scene(spec, model, data, path=args.scene)

    @staticmethod
    def find_worktop(scene, robot: str) -> placement.Worktop:
        from tools.scene_placement import describe

        mount, _ = place_arm_on_table(scene.path, scene.model, scene.data, robot,
                                      MOUNT_REACH[robot], 3, want=TARGET)
        print(describe(mount), file=sys.stderr)
        return placement.Worktop(name=mount.target.support_name,
                                 xy=np.asarray(mount.xy, dtype=float),
                                 yaw=float(mount.yaw), z=float(mount.z))

    @staticmethod
    def floor_spot(scene, inst, keep_out):
        from tools.scene_placement import describe, find_robot_placement

        found = find_robot_placement(scene.path, model=scene.model, data=scene.data,
                                     prefer="floor", exclude=tuple(keep_out))
        print(describe(found), file=sys.stderr)
        return np.asarray(found.pos[:2], dtype=float), float(found.yaw)

    @staticmethod
    def attach(scene, inst) -> None:
        config_cls, robot_cls = load_robot(inst.name)
        config = config_cls()
        config.robot_namespace = inst.mjcf
        if inst.name == "so101":
            # No pedestal: the base body sits on the worktop, which is the task frame.
            config.base_size = None
        if inst.holonomic:
            # World-aligned slide joints: grafted over the origin at the height of what
            # it stands on (the adapter adds a legged robot's ride height), and
            # teleported to its stand once compiled.
            pos, quat = [0.0, 0.0, inst.surface_z], [1.0, 0.0, 0.0, 0.0]
        else:
            pos = [float(inst.xy[0]), float(inst.xy[1]), inst.mount_z]
            quat = [float(np.cos(inst.yaw / 2)), 0.0, 0.0, float(np.sin(inst.yaw / 2))]
        robot_cls.add_robot_to_scene(config, scene.spec, prefix=inst.mjcf, pos=pos, quat=quat)
        inst.engine["config"], inst.engine["cls"] = config, robot_cls

    @staticmethod
    def scene_option():
        return None


ENGINE = MolmoSpacesEngine()


def main(argv=None) -> int:
    return spawn.main(ENGINE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
