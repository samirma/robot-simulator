#!/usr/bin/env python
"""Spawn an out-of-tree robot into a house and view or render it.

Places the robot somewhere worth being, attaches it, and then either opens the passive
viewer (run under mjpython) or writes a PNG.

    mjpython tools/spawn_robot.py so101 --scene <house.xml>
    python   tools/spawn_robot.py so101 --scene <house.xml> --render out.png

Where "somewhere worth being" depends on what the robot is:

* An **arm** has no way of getting anywhere, so it is mounted *on* a table -- a surface at
  working height with graspable objects on it, chosen by `tools/scene_placement.py` -- on a
  short riser in place of its floor pedestal, positioned so those objects are inside its
  working annulus. If the house has a surface but nothing on it, objects are spawned onto
  it, because an arm facing an empty table is not a useful thing to have rendered.
* A **mobile base** is put on the most open floor the house has, which is where a robot
  that has to drive wants to start.

This is the "spawn and look at it" path. Nothing beyond the scene and the robot is
involved, so it works for robots that have no grasp library.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
# The wire bridge (contracts.*) and robot specs (robots_spec) live in simulator/shared.
_SHARED = SIM_ROOT.parent / "shared"
if _SHARED.is_dir() and str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

# name -> (module, config class, robot class), kept lazy so importing one robot does
# not require the others to be installed.
ROBOTS = {
    "so101": ("robots.so101", "SO101RobotConfig", "SO101Robot"),
    "myagv": ("robots.myagv", "MyAGVRobotConfig", "MyAGVRobot"),
    "rebot_b601": ("robots.rebot_b601", "B601RobotConfig", "B601Robot"),
    "ainex": ("robots.ainex", "AiNexRobotConfig", "AiNexRobot"),
}

# name -> (module, function) presenting that robot's ROS topics. Each robot owns its own
# contract, because the contract *is* part of the robot definition: the myAGV speaks
# `cmd_vel`/`odom` and the AiNex speaks `/walking/*` and shares not one topic with it.
# Putting them here would make this file the place every vendor's ROS interface accretes.
# What they share -- renderers, /scan, the velocity-to-setpoint integration -- is below.
ROS_SURFACES = {
    "myagv": ("robots.myagv.ros_surface", "attach_ros"),
    "ainex": ("robots.ainex.ros_surface", "attach_ros"),
    # The arm's surface lives in shared/ directly rather than behind an engine-side
    # adapter: unlike a mobile base, there is no "how does this engine name a base"
    # question for it to answer -- it needs the arm and gripper move groups, and every
    # engine spells those the same way because they come from the shared robot spec.
    "so101": ("ros_surfaces.so101", "attach_ros"),
}

# Robots whose ROS surface is an arm contract rather than a mobile-base one: no cmd_vel,
# no odometry, no lidar, and several cameras instead of one. They take a different
# serve_ros signature, which is the honest way to say the two contracts are unrelated.
ARM_ROS_SURFACES = {"so101"}

# name -> (module, staging function, arbiter class). A task brings its own objects and
# cameras and its own definition of done; see simulator/shared/tasks/.
TASKS = {
    "apple_on_plate": ("tasks.apple_on_plate", "stage", "AppleOnPlate"),
}

# Per-robot lidar defaults, since the YDLidar X2's mount is meaningless for a robot that
# does not carry one. Overridden by any explicit --scan-* flag.
SCAN_DEFAULTS = {
    # Transcribed from ydlidar_ros_driver/launch/X2.launch and the
    # base_footprint -> laser_frame transform in myagv_active.launch.
    "myagv": {"offset": (0.065, 0.08), "min_range": 0.1, "max_range": 12.0},
    # INVENTED, not transcribed: the AiNex has no lidar at all (see robots/README.md).
    # Mid-torso on a 0.46 m robot, centred -- above the leg swing, low enough to see the
    # edges of furniture. The range is cut to the room scale a robot walking at 0.2 m/s
    # actually operates in.
    "ainex": {"offset": (0.0, 0.20), "min_range": 0.1, "max_range": 8.0},
}

# Robots whose base is three virtual holonomic joints must be grafted in at the origin
# and then *driven* to their spawn pose, because the slide joints are world-aligned.
# Robots on a mocap mount are placed by the attach pos/quat instead.
# The AiNex is a biped, but its torso rides the same three virtual joints: its gait is
# animated over a planar base rather than balanced. See robots/ainex/ainex.py.
HOLONOMIC_BASE_ROBOTS = {"myagv", "ainex"}

# Arms: no base of their own, so they are bolted to a work surface rather than stood on
# the floor. A robot with both an arm and wheels places as a mobile base.
TABLETOP_ROBOTS = {"so101", "rebot_b601"}

# The annulus on that surface the arm can comfortably work in. The SO-101 is a ~0.4 m
# tabletop arm (robots/so101/so101_config.py:41); the B601 has 767 mm of reach
# (robots/rebot_b601/b601_config.py:32). Both are kept short of the full figure: the last
# few centimetres of reach are a straight-out arm with no usable orientation left.
ARM_REACH = {"so101": (0.15, 0.35), "rebot_b601": (0.25, 0.60)}

# Room the arm itself needs around its mount, and how far up it needs it. Not the same as
# the base it stands on: a counter is against a wall, and it is the arm -- not the riser --
# that hits it. The height bounds which geometry counts as in the way at all, so that a
# wall does and the ceiling above it does not.
ARM_BODY_RADIUS = {"so101": 0.20, "rebot_b601": 0.35}
ARM_BODY_HEIGHT = {"so101": 0.45, "rebot_b601": 0.90}

# Replaces the floor pedestal (`base_size`) once the arm stands on the table itself: just
# enough to read as a mount, not enough to matter to the workspace.
TABLETOP_RISER = [0.14, 0.14, 0.03]

# The velocity-to-setpoint integration, the ray-cast /scan and the sensor streams are
# engine-neutral -- they take a raw mujoco model/data and the shared contracts server --
# so they live in simulator/shared and are reused by every MuJoCo engine. Re-exported
# here because robots/*/ros_surface.py import them `from tools.spawn_robot`.
from mujoco_bridge import (  # noqa: E402
    TARGET_LEAD_M,
    TARGET_LEAD_RAD,
    PlanarSetpoint,
    SensorStreams,
    SensorTopics,
    laser_scan_ranges,
)

# Where the `--target` categories are put on the worktop for `shot`/`view`, as angles off
# the arm's facing direction. Two objects land symmetrically in front of it, which is the
# layout RoboCasa's own `--objects` sampler produces and therefore what makes the two
# engines' screenshots comparable.
TARGET_LAYOUT_DEG = (-30.0, 30.0, 0.0, -60.0, 60.0)
# A millimetre of daylight under a relocated object, so it settles onto the surface
# instead of starting the episode interpenetrating it.
TARGET_LAYOUT_LIFT = 0.001
# Loose scene objects this close to the mount are moved out of the arm's way. The same
# figure as `apple_on_plate.CLEAR_RADIUS`: the arm's workspace is its workspace whether a
# task owns it or not, and using a second number here would make the two paths disagree
# about what "in the way" means.
TARGET_CLEAR_RADIUS = 0.55
TARGET_CLEAR_Z_BAND = (-0.15, 0.45)
# How far under the scene a cleared object is parked; only has to be out of every frustum.
TARGET_SUNK_DEPTH = 50.0

# Spawned stand-in objects, when a surface has nothing on it worth reaching for.
SPAWN_OBJECT_HALF = 0.025
SPAWN_OBJECT_COLOURS = (
    [0.85, 0.25, 0.20, 1.0],
    [0.20, 0.60, 0.85, 1.0],
    [0.90, 0.75, 0.20, 1.0],
    [0.35, 0.70, 0.35, 1.0],
)


class _Instance:
    """One robot in the scene: what it is, where its names live, and how it is driven.

    Two namespaces, deliberately kept apart. `mjcf` (`robot_0/`) prefixes bodies, joints,
    actuators and cameras inside the compiled MuJoCo model, so two robots can coexist
    without a name collision. `ns` (`so101`) prefixes topics and services on the wire, so
    two robots can coexist on one ROS graph. Conflating them would put an engine's model
    layout onto the wire, where a client could see it.
    """

    __slots__ = ("name", "config", "robot_cls", "ns", "holonomic", "tabletop",
                 "pos", "yaw", "attach_pos", "view", "step")

    def __init__(self, name, config, robot_cls, ns, holonomic, tabletop):
        self.name, self.config, self.robot_cls = name, config, robot_cls
        self.ns, self.holonomic, self.tabletop = ns, holonomic, tabletop
        self.pos = self.yaw = self.attach_pos = None
        self.view = self.step = None

    @property
    def mjcf(self) -> str:
        return self.config.robot_namespace

    def __repr__(self) -> str:
        return f"<{self.name} mjcf={self.mjcf!r} ns={self.ns!r}>"


def load_robot(name: str):
    if name not in ROBOTS:
        raise SystemExit(f"unknown robot {name!r}; available: {', '.join(sorted(ROBOTS))}")
    module_name, config_attr, robot_attr = ROBOTS[name]
    import importlib

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


def add_tabletop_objects(spec, xy_min, xy_max, top_z: float, n: int) -> list[np.ndarray]:
    """Put `n` small graspable boxes on a surface, and return where they landed.

    Only runs when the house gave us a work surface with nothing on it. Primitive boxes
    rather than library assets on purpose: `install_uid` wants a network fetch and a uid
    known to be small enough to pick up, and this is the fallback path -- it should not be
    the one that needs the internet. Positioned from the surface we actually measured
    instead of a hardcoded z.
    """
    xy_min = np.asarray(xy_min, dtype=float)
    xy_max = np.asarray(xy_max, dtype=float)
    centre = (xy_min + xy_max) / 2
    # Keep them off the lip of the surface, and inside it even if it is a narrow shelf.
    half = np.maximum((xy_max - xy_min) / 2 - 0.12, 0.0)

    positions = []
    for i in range(n):
        angle = 2 * np.pi * i / max(n, 1)
        offset = half * np.array([np.cos(angle), np.sin(angle)]) * 0.6
        pos = np.array(
            [centre[0] + offset[0], centre[1] + offset[1], top_z + SPAWN_OBJECT_HALF + 0.002]
        )
        body = spec.worldbody.add_body(name=f"spawned_object_{i}", pos=pos.tolist())
        body.add_freejoint()
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[SPAWN_OBJECT_HALF] * 3,
            rgba=SPAWN_OBJECT_COLOURS[i % len(SPAWN_OBJECT_COLOURS)],
            mass=0.1,
        )
        positions.append(pos)
    return positions


def _homogeneous(pos, quat) -> np.ndarray:
    """A 4x4 from an MJCF body's parent-relative pos/quat."""
    out = np.eye(4)
    w, x, y, z = (float(v) for v in quat)
    out[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    out[:3, 3] = [float(v) for v in pos]
    return out


def _movable_bodies(spec) -> dict[str, tuple[object, np.ndarray, np.ndarray]]:
    """Every free-jointed scene body, by name -> (body, parent world 4x4, its own world 4x4).

    `MjsBody.pos` is **parent-relative** and a house's graspables sit a couple of levels
    down inside whatever surface holds them, so the parent transform is what any world-frame
    edit has to go through. Same walk as `apple_on_plate._clear_workspace`, and for the same
    reason: comparing raw body positions against world coordinates silently misses exactly
    the objects that are nested.
    """
    found: dict[str, tuple[object, np.ndarray, np.ndarray]] = {}

    def walk(body, parent: np.ndarray) -> None:
        here = parent @ _homogeneous(body.pos, body.quat)
        name = body.name or ""
        if name and not name.startswith(("task_", "robot_", "spawned_")):
            if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints):
                found[name] = (body, parent, here)
                return  # its children move with it
        for child in body.bodies:
            walk(child, here)

    for child in spec.worldbody.bodies:
        walk(child, np.eye(4))
    return found


def _world_bottom(model, data, name: str) -> float | None:
    """Lowest point of a body's whole subtree in the compiled scene, or None if unknown."""
    from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if root < 0:
        return None
    # MuJoCo orders bodies parent-before-child, so one forward pass closes the subtree.
    subtree = {root}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in subtree:
            subtree.add(b)
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in subtree]
    if not geoms:
        return None
    centre, dims = geom_aabb(model, data, geoms)
    return float(centre[2] - dims[2] / 2)


#: How far inside the worktop's own edge a relocated object has to land. An apple at the
#: lip rolls off the moment the arm brushes it, and the placement angle that reads best
#: relative to the arm regularly points straight at the counter's edge.
TARGET_EDGE_MARGIN = 0.10


def _spot_on_worktop(mount, radius: float, preferred_deg: float, taken: list) -> np.ndarray:
    """A point `radius` from the mount that is actually on the surface, near `preferred_deg`.

    The arm is now bolted at the rim (`find_tabletop_mount(edge_bias=True)`), so a fixed
    angle off its facing direction lands over the floor about as often as over the worktop.
    Sweep outwards from the preferred angle and take the first spot that is on the top face,
    clear of the edge, and not on top of an object already placed.
    """
    from tools.scene_placement import _rects_covering

    mount_xy = np.asarray(mount.xy, dtype=float)
    rects = mount.target.support_top_rects
    lo = np.asarray(mount.target.support_xy_min, dtype=float) + TARGET_EDGE_MARGIN
    hi = np.asarray(mount.target.support_xy_max, dtype=float) - TARGET_EDGE_MARGIN

    def ok(xy: np.ndarray) -> bool:
        if any(float(np.linalg.norm(xy - other)) < 0.14 for other in taken):
            return False
        if rects is not None and len(rects):
            return bool(_rects_covering(xy, rects, pad=-TARGET_EDGE_MARGIN).any())
        return bool(np.all(xy >= lo) and np.all(xy <= hi))

    offsets = [0.0] + [s * d for d in (15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 110.0, 130.0)
                       for s in (1.0, -1.0)]
    fallback = None
    for offset in offsets:
        angle = mount.yaw + np.radians(preferred_deg + offset)
        xy = mount_xy + radius * np.array([np.cos(angle), np.sin(angle)])
        if fallback is None:
            fallback = xy
        if ok(xy):
            return xy
    return fallback


def arrange_targets(spec, model, data, mount, want: tuple[str, ...], reach) -> tuple[list[str], list[str]]:
    """Put the `--target` categories on the mount surface and clear the rest off it.

    MolmoSpaces used to spawn nothing at all for `shot`/`view`: `--target` ranked the
    house's surfaces and whatever happened to be on the chosen one was what the arm got. In
    FloorPlan1 that is a book, a loaf and a tomato -- while the house's own two plates sit
    on a different counter entirely. RoboCasa's `--objects` meanwhile spawns exactly the
    requested pair from its registry, so the two engines' screenshots were not of the same
    scene and the comparison this script exists to make was not being made.

    So move the house's *own* apple and plate into the arm's working annulus, and sink the
    loose clutter that is in its way. Nothing is downloaded and nothing is invented: these
    are the scene's objects with the scene's meshes and grasp metadata, which is the whole
    reason they are worth reaching for -- an invented primitive is what
    `add_tabletop_objects` already does for houses that have nothing.

    Bodies are moved rather than deleted because deleting one out of a compiled iTHOR house
    takes its metadata references with it. Returns (placed, cleared) names.

    Not used when a task is staged: `apple_on_plate` brings its own measured apple and
    plate and clears the same workspace itself, and two apples is a different task.
    """
    movable = _movable_bodies(spec)
    if not movable:
        return [], []

    mount_xy = np.asarray(mount.xy, dtype=float)
    placed: list[str] = []
    used: set[str] = set()
    taken: list[np.ndarray] = []
    radius = float(np.clip(0.5 * (reach[0] + reach[1]), reach[0] + 0.04, reach[1] - 0.04))

    for i, category in enumerate(want):
        # The nearest instance of the category, so a house with two apples gives up the one
        # already on this counter rather than the one across the room.
        candidates = [
            (float(np.linalg.norm(world[:2, 3] - mount_xy)), name)
            for name, (_, _, world) in movable.items()
            if name not in used and category in name.split("_")[0].lower()
        ]
        if not candidates:
            print(f"  {category}: no movable one in this scene to place", file=sys.stderr)
            continue
        _, name = min(candidates)
        body, parent, world = movable[name]

        want_xy = _spot_on_worktop(
            mount, radius, TARGET_LAYOUT_DEG[i % len(TARGET_LAYOUT_DEG)], taken
        )
        taken.append(want_xy)

        bottom = _world_bottom(model, data, name)
        want_z = world[2, 3] if bottom is None else world[2, 3] + (
            mount.z + TARGET_LAYOUT_LIFT - bottom
        )

        delta = np.array([want_xy[0] - world[0, 3], want_xy[1] - world[1, 3], want_z - world[2, 3]])
        # Through the parent's rotation, since `body.pos` is expressed in it.
        body.pos = list(np.asarray(body.pos, dtype=float) + parent[:3, :3].T @ delta)
        used.add(name)
        placed.append(f"{category} ({np.linalg.norm(want_xy - mount_xy):.2f} m)")

    # Everything else loose in the arm's way goes under the floor. Static first, THEN
    # moved: a free body parked below the floor is still a free body and falls forever,
    # which wrecks the solver -- see `apple_on_plate._clear_workspace`.
    cleared: list[str] = []
    for name, (body, _, world) in movable.items():
        if name in used:
            continue
        local_z = float(world[2, 3]) - mount.z
        if float(np.linalg.norm(world[:2, 3] - mount_xy)) > TARGET_CLEAR_RADIUS:
            continue
        if not (TARGET_CLEAR_Z_BAND[0] < local_z < TARGET_CLEAR_Z_BAND[1]):
            continue
        for joint in list(body.joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(joint)
        body.pos = [body.pos[0], body.pos[1], body.pos[2] - TARGET_SUNK_DEPTH]
        cleared.append(name.split("_")[0])
    return placed, cleared


def _spec_subtree(body) -> list:
    """`body` and every body under it, in the uncompiled spec."""
    out = [body]
    for child in body.bodies:
        out.extend(_spec_subtree(child))
    return out


def adopt_native_objects(spec, model, data, mount_pos, yaw, task, *, swap: bool) -> str:
    """Hand the house's own apple and plate to the task, at the task's positions.

    The task can stage a measured YCB pair of its own; this is the other choice, the one
    where the objects are the engine's. Each is the nearest movable body of its category
    in the house, renamed to the task's body name so the arbiter, the free-joint topic
    and the workspace clearing all treat it as the task's, and moved to `object_poses`.

    Two adjustments make an iTHOR apple *equivalent* rather than merely present, and
    both were measured before they were written down:

    * **It is scaled to the contract's 40 mm.** FloorPlan1's apple is ~100 mm across,
      well past the ~70 mm the SO-101's jaw opens to; an LLM agent once spent five clean
      grasp cycles on a full-sized apple whose skin the fingers could only slide off. The
      mesh assets scale about the body origin and the child offset scales with them, so
      the fruit stays whole.
    * **It gets the task's contact block.** `condim 6` is what makes the declared rolling
      friction exist at all; the house's apple ships at 3, and a fruit nudged at 7 mm/s
      then coasts for half a minute. The plate keeps its own parameters: it is static in
      effect, and its 8-box rim already does what the task's 24-box one does.

    Returns a one-line report for the log.
    """
    movable = _movable_bodies(spec)
    transform = task.base_frame(mount_pos, yaw)
    poses = task.object_poses(swap)
    mount_xy = np.asarray(mount_pos[:2], dtype=float)
    mount_z = float(mount_pos[2])
    report = []
    used: set[str] = set()
    for category, body_name in (("apple", task.APPLE_BODY), ("plate", task.PLATE_BODY)):
        candidates = [
            (float(np.linalg.norm(world[:2, 3] - mount_xy)), name)
            for name, (_, _, world) in movable.items()
            if name not in used and category in name.split("_")[0].lower()
        ]
        if not candidates:
            raise SystemExit(
                f"--native-objects: this house has no movable {category} to adopt. Use "
                f"another scene, or --task-objects for the task's own pair."
            )
        _, name = min(candidates)
        body, parent, world = movable[name]
        used.add(name)

        bottom = _world_bottom(model, data, name)
        root_z = float(world[2, 3])
        lift = (root_z - bottom) if bottom is not None else 0.0

        scale = 1.0
        if category == "apple":
            half = _collision_half_extent(model, data, name)
            scale = float(task.APPLE_RADIUS / half) if half else 1.0
            subtree = _spec_subtree(body)
            # NOT `mesh.scale *= scale` on the existing assets. This spec has already been
            # compiled once (for the placement search), and MjSpec keeps the compiled mesh
            # data across compiles: the scale field updates, the geometry does not, and
            # the apple comes out full-sized with `mesh_scale` claiming otherwise --
            # measured, rbound 0.060 against 0.024 from a fresh spec. New assets under
            # new names are compiled fresh, so the geoms are repointed at scaled copies.
            by_name = {m.name: m for m in spec.meshes}
            copies: dict[str, str] = {}
            for b in subtree:
                for g in b.geoms:
                    if not g.meshname:
                        continue
                    if g.meshname not in copies:
                        source = by_name[g.meshname]
                        copy = spec.add_mesh(name=f"{source.name}_x{int(round(scale * 100))}")
                        copy.file = source.file
                        copy.scale = [float(s) * scale for s in source.scale]
                        copies[g.meshname] = copy.name
                    g.meshname = copies[g.meshname]
            for child in subtree[1:]:
                child.pos = [float(p) * scale for p in child.pos]
            colliders = [g for b in subtree for g in b.geoms if g.contype or g.conaffinity]
            for b in subtree:
                # The house's apple takes its mass from geom density, so at 0.4x it would
                # weigh 9 g; the contract's 20 g is what the grasp tuning was done against.
                if b.mass > 0:
                    ratio = task.APPLE_MASS / float(b.mass)
                    b.mass = task.APPLE_MASS
                    b.inertia = [float(i) * ratio * scale * scale for i in b.inertia]
            for g in colliders:
                g.mass = task.APPLE_MASS / len(colliders)
                for key, value in task.APPLE_CONTACT.items():
                    setattr(g, key, value)
            lift *= scale

        want = task._apply(transform, poses[category])
        want[2] = mount_z + lift + 0.002
        delta = np.array(want) - world[:3, 3]
        body.pos = list(np.asarray(body.pos, dtype=float) + parent[:3, :3].T @ delta)
        body.name = body_name
        report.append(
            f"{category} <- {name.split('_')[0]}"
            + (f" scaled x{scale:.2f}" if scale != 1.0 else "")
            + f" at ({want[0]:.2f}, {want[1]:.2f}, {want[2]:.3f})"
        )
    return "native objects: " + ", ".join(report)


def _collision_half_extent(model, data, name: str) -> float | None:
    """Largest horizontal half-extent of a body subtree's collision geoms, compiled."""
    from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if root < 0:
        return None
    subtree = {root}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in subtree:
            subtree.add(b)
    geoms = [
        g for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in subtree and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    if not geoms:
        return None
    _centre, dims = geom_aabb(model, data, geoms)
    return float(max(dims[0], dims[1]) / 2)


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
    """Choose the surface, guarantee something is on it, and return (mount, spawn_request).

    Detection has to happen against a compiled model, but spawning objects happens on the
    spec -- so this returns what to spawn rather than spawning it, and the caller adds the
    objects before its own compile. The target is then built from the positions we chose,
    never by re-running detection: `find_grasp_targets` keys off the scene metadata JSON,
    which knows nothing about bodies we just invented, so a re-detect would come back empty.
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
                height = ARM_BODY_HEIGHT.get(robot, 1.0)
                blocker_cache[key] = (
                    static_blockers(model, data, target.support_top_z, height=height),
                    dynamic_clutter(model, data, target.support_top_z, height=height),
                )
            mount = find_tabletop_mount(
                target,
                reach_range=reach,
                footprint=TABLETOP_RISER[0],
                blockers=blocker_cache[key][0],
                clutter=blocker_cache[key][1],
                body_radius=ARM_BODY_RADIUS.get(robot),
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
            f"{robot} on. Try another scene index, or pass --pos/--yaw to place it by hand."
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
        f"no graspables in the scene; spawning {n_spawn} onto "
        f"{s_cat or s_name} (top {top_z:.2f} m)",
        file=sys.stderr,
    )

    # Predict the layout `add_tabletop_objects` will produce so the mount can be chosen
    # against it; the caller then spawns exactly that.
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
        footprint=TABLETOP_RISER[0],
        blockers=static_blockers(model, data, top_z, height=ARM_BODY_HEIGHT.get(robot, 1.0)),
        clutter=dynamic_clutter(model, data, top_z, height=ARM_BODY_HEIGHT.get(robot, 1.0)),
        model=model,
        data=data,
        body_radius=ARM_BODY_RADIUS.get(robot),
        edge_bias=edge_bias,
    )
    return mount, (xy_min, xy_max, top_z, n_spawn)


def check_task_contacts(model, namespace: str, task) -> None:
    """Refuse to serve a task whose objects the gripper cannot physically touch.

    MuJoCo pairs two geoms only if `(contype_a & conaffinity_b) or (contype_b &
    conaffinity_a)`, and a robot loader that rewrites those bitmasks -- for its own
    contact filtering -- can leave the jaws and the task's apple on disjoint masks. The
    failure is perfectly silent: the arm executes every waypoint, the jaw closes to its
    commanded width straight through the object, and the episode scores zero looking
    exactly like a policy that missed by a centimetre. This was found the slow way; the
    check exists so it is found the fast way.
    """
    jaw_prefixes = (f"{namespace}fixed_jaw", f"{namespace}moving_jaw")
    jaw = [g for g in range(model.ngeom)
           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(jaw_prefixes)]
    objects = [g for g in range(model.ngeom)
               if model.geom_bodyid[g] in task.contact_bodies()
               and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not jaw or not objects:
        raise SystemExit(f"task contact check: found {len(jaw)} jaw geoms and "
                         f"{len(objects)} collidable task geoms; expected both non-empty")

    def pairs(a: int, b: int) -> bool:
        return bool((model.geom_contype[a] & model.geom_conaffinity[b])
                    or (model.geom_contype[b] & model.geom_conaffinity[a]))

    touchable = sum(1 for j in jaw for o in objects if pairs(j, o))
    print(
        f"task contacts: {len(jaw)} jaw geoms x {len(objects)} task geoms, "
        f"{touchable} pairs collide "
        f"(jaw contype/conaffinity {sorted({(int(model.geom_contype[j]), int(model.geom_conaffinity[j])) for j in jaw})}, "
        f"objects {sorted({(int(model.geom_contype[o]), int(model.geom_conaffinity[o])) for o in objects})})",
        file=sys.stderr,
    )
    if touchable == 0:
        raise SystemExit(
            "task contact check FAILED: no jaw geom can collide with any task object. "
            "The jaws would close straight through the apple and the run would score zero "
            "while looking like a near miss."
        )


def warn_on_penetration(model, data, namespace: str, depth: float = -0.001) -> None:
    """Complain if the robot was bolted down inside something.

    The arm hangs off a mocap body, so nothing pushes it back out and nothing falls over:
    a mount that clips a bowl just silently interpenetrates. This is the only thing that
    notices.
    """
    def is_robot(geom_id: int) -> bool:
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        return body.startswith(namespace)

    for c in range(data.ncon):
        con = data.contact[c]
        if con.dist > depth:
            continue
        g1, g2 = int(con.geom1), int(con.geom2)
        if is_robot(g1) == is_robot(g2):
            continue
        other = g2 if is_robot(g1) else g1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom {other}"
        print(
            f"warning: robot intersects {name} by {-con.dist * 1000:.0f} mm at its spawn pose",
            file=sys.stderr,
        )
        return


# `serve_control` lived here: a msgpack-numpy binary protocol on its own port, which was
# how the arm was driven before it moved onto ROS. It is gone rather than deprecated. Two
# transports for one robot means two things to keep in step and two ways for the engines
# to drift, and the arm's ROS surface above is what a real ros2_control bringup for this
# arm presents -- so it is also the one a client can point at hardware.


def _surface_kwargs(args, inst, model, task):
    """Everything one robot's surface needs, chosen by which contract it speaks.

    An arm contract and a mobile-base contract share no argument beyond the model: no
    cmd_vel, no odometry and no lidar on one; several cameras and a task on the other.
    Building the two bags here rather than inline keeps the fleet loop readable and keeps
    the asymmetry visible, which is the honest way to say the contracts are unrelated.
    """
    ns = inst.mjcf
    if inst.name in ARM_ROS_SURFACES:
        from ros_surfaces.so101 import DEFAULT_CAMERAS, WRIST_CAMERA

        cameras = dict(DEFAULT_CAMERAS)
        if args.wrist_camera:
            # The scene cameras sit on the worldbody under their own names; the wrist
            # camera rides the gripper and so carries the robot's MJCF prefix.
            cameras.update({t: (f"{ns}{n}", w, h) for t, (n, w, h) in WRIST_CAMERA.items()})
        return {
            "view": inst.view, "model": model, "task": task, "cameras": cameras,
            "jpeg_quality": args.jpeg_quality, "control_hz": args.control_hz,
        }

    camera = _pick_camera(args, model, ns)
    scan_cfg = None
    if not args.no_scan:
        defaults = SCAN_DEFAULTS.get(inst.name, SCAN_DEFAULTS["myagv"])
        offset = args.scan_offset or defaults["offset"]
        scan_cfg = {
            "beams": args.scan_beams,
            "max_range": args.scan_range or defaults["max_range"],
            "min_range": args.scan_min_range or defaults["min_range"],
            "offset_x": offset[0],
            "offset_z": offset[1],
            "period": 1.0 / max(args.scan_hz, 1e-3),
            # Rays start at the robot's own root body and must not range it.
            "body": f"{ns}{inst.robot_cls.robot_model_root_name()}",
            # A legged robot's limbs are separate bodies, and mj_ray takes only one
            # bodyexclude. Without this a torso-mounted scanner ranges its own thigh.
            # It excludes only *this* robot: another robot in the room is something the
            # lidar is supposed to see, and a fleet that could not see itself would map a
            # kitchen with a hole where its neighbour stands.
            "exclude_bodies": frozenset(
                i for i in range(model.nbody)
                if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
                and n.startswith(ns)
            ),
        }
    depth_cfg = None
    if not args.no_depth and camera is not None:
        fovy = float(model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)])
        depth_cfg = {
            "size": args.depth_size,
            "period": 1.0 / max(args.depth_hz, 1e-3),
            "max_range": args.depth_range,
            "fovy": fovy,
        }
    return {
        "view": inst.view, "model": model, "camera": camera,
        "camera_size": args.camera_size, "jpeg_quality": args.jpeg_quality,
        "control_hz": args.control_hz, "watchdog_s": args.watchdog,
        "scan": scan_cfg, "depth": depth_cfg,
        "camera_period": (1.0 / args.camera_hz) if args.camera_hz > 0 else 0.0,
        "extra": {"action_dir": args.action_dir},
    }


def _pick_camera(args, model, ns: str) -> str | None:
    """The MJCF camera a mobile base streams, resolved against its own prefix.

    `--camera` names one for every robot, which is only meaningful when there is one.
    With a fleet each base falls back to its own `front_camera`, which is what makes the
    flag's default behaviour right for both cases.
    """
    if args.camera is not None:
        if args.camera.lower() == "none":
            return None
        return args.camera
    for candidate in (f"{ns}front_camera", f"{ns}wrist_cam"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, candidate) >= 0:
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "robot",
        help=f"one of: {', '.join(sorted(ROBOTS))}. A comma-separated list spawns several "
             "into the same scene, sharing one --ros-port: `so101,myagv` mounts the arm on "
             "a work surface and puts the base on the floor. That is one ROS graph with a "
             "namespace per robot, which is what a real multi-robot bringup is.",
    )
    ap.add_argument(
        "--ros-namespace", default=None, dest="ros_namespace",
        help="comma-separated ROS namespaces, positionally matched to the robots. "
             "Defaults to each robot's own name, so `so101` is on /so101/joint_states. "
             "Pass an empty element for the bare, unnamespaced vendor contract -- "
             "`--ros-namespace ''` with one robot reproduces the single-robot wire "
             "exactly, which is what keeps that contract testable.",
    )
    ap.add_argument("--scene", required=True, help="house MJCF")
    ap.add_argument("--render", default=None, help="write a PNG instead of opening the viewer")
    ap.add_argument(
        "--no-reference-table", action="store_false", dest="reference_table",
        help="stage the task's objects on the engine's own worktop instead of on the "
             "reference work surface. Verified both ways, so this is a supported "
             "experiment rather than a fallback.",
    )
    ap.add_argument(
        "--swap-objects", action="store_true", dest="swap_objects",
        help="stage the plate at the apple's spawn and the apple where the plate was. "
             "The console reads which layout is on the wire off the apple's reset "
             "position, so nothing else has to be told.",
    )
    ap.add_argument(
        "--task-objects", action="store_false", dest="native_objects",
        help="stage the task's own measured YCB apple and plate instead of adopting the "
             "house's. Default is the house's: its nearest apple and plate, moved to the "
             "task's positions, the apple scaled to the contract's 40 mm and given the "
             "task's contact block (see adopt_native_objects).",
    )
    ap.add_argument(
        "--no-dressing", action="store_false", dest="dressing",
        help="stage only the apple and the plate, without the bowl/mug/banana/lemon the "
             "reference keeps on its table as distractors.",
    )
    ap.add_argument(
        "--reference-lighting", action="store_true", dest="reference_lighting",
        help="impose the reference scene's exposure on the kitchen. OFF by default, "
             "which is not what the photometry would suggest: this block takes an iTHOR "
             "kitchen's overhead frame from 5.7%% of pixels clipped to white to 3.1%%, "
             "against the reference scene's own 3.0%%. It also costs the VLA the task -- "
             "0/24 episodes with it against 6/18 without. Use it to compare exposure "
             "against the reference rig, not to run a policy. Its two extra lamps are a "
             "separate thing and are also off -- they took the same frame to 75.2%%.",
    )
    ap.add_argument(
        "--extra-lights", action="store_true", dest="extra_lights",
        help="also add the reference's two directional lamps. For a scene that renders "
             "too dark; on a normally-lit kitchen they blow the frame out.",
    )
    ap.add_argument(
        "--render-framing", action="store_true", dest="render_framing",
        help="with --render: report where the staged work surface's corners land in the "
             "frame, and what fraction of the frame is clipped to white. Framing is a "
             "property of the surface under a camera as much as of the camera, so this "
             "is how 'the cameras are in the right place' stops being an impression.",
    )
    ap.add_argument(
        "--render-camera", default=None, dest="render_camera", metavar="NAME",
        help="with --render: look through this named MJCF camera at its own declared "
             "resolution, instead of the free camera. How a task camera's framing gets "
             "checked without starting a server.",
    )
    ap.add_argument("--pos", type=float, nargs=2, default=None,
                    help="override xy placement; an arm keeps the height of the surface "
                         "it is mounted on")
    ap.add_argument("--yaw", type=float, default=None, help="override facing, in degrees")
    ap.add_argument("--target", default=None, metavar="CAT,CAT",
                    help="prefer a work surface holding these object categories, "
                         "e.g. 'bowl,apple' (substring match on the scene category)")
    ap.add_argument("--reach", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="arm working annulus on the table, in metres (default per robot)")
    ap.add_argument("--mount-centre", action="store_true", dest="mount_centre",
                    help="drop the rim tie-break when bolting down a tabletop arm. It is "
                         "only a tie-break now: the mount is chosen by how much of the "
                         "arm's forward workspace has worktop under it, and that already "
                         "puts the arm at an edge looking in, so on most surfaces this "
                         "flag changes nothing. Kept for comparing the two on one scene")
    ap.add_argument("--gripper", choices=("rest", "open", "closed"), default="rest",
                    help="override the gripper's rest pose, for looking at what it can reach")
    ap.add_argument("--spawn-objects", type=int, default=3, dest="spawn_objects",
                    metavar="N",
                    help="objects to put on the table if the scene has none; 0 to fail instead")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--distance", type=float, default=None,
                    help="camera distance (default: scaled to the robot size)")
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=-15.0)
    ap.add_argument("--timeout", type=float, default=None, help="auto-close the viewer after N s")
    ap.add_argument(
        "--headless", action="store_true", dest="headless",
        help="run the simulation and the --ros-port server WITHOUT opening the "
             "viewer (for automated checks and displayless hosts)",
    )
    ap.add_argument(
        "--host", default="0.0.0.0", dest="control_host",
        help="interface the --ros-port server binds (default: all interfaces)",
    )
    ap.add_argument(
        "--control-hz", type=float, default=20.0, dest="control_hz", help="control loop rate"
    )
    ap.add_argument(
        "--camera",
        default=None,
        help="MJCF camera to stream on a mobile base's ROS surface; defaults to the "
        "robot's front_camera if it has one. Use 'none' to disable. Arms ignore this: "
        "their camera set is part of their contract.",
    )
    ap.add_argument("--camera-size", type=int, nargs=2, default=[640, 480], dest="camera_size",
                    metavar=("W", "H"))
    ap.add_argument("--jpeg-quality", type=int, default=70, dest="jpeg_quality")
    ap.add_argument(
        "--task", default=None, choices=sorted(TASKS),
        help="stage a task into the scene: its objects, its cameras and its success "
             "predicate. Requires --ros-port, which is what publishes the verdict.",
    )
    ap.add_argument(
        "--wrist-camera", action="store_true", dest="wrist_camera",
        help="also stream the eye-in-hand view. Off by default because every enabled "
             "camera is rendered inside the physics loop, so the control rate falls as "
             "cameras are added and the cost lands on every client, not just the one "
             "that wanted the view.",
    )
    ap.add_argument(
        "--ros-port", type=int, default=None, dest="ros_port",
        metavar="PORT",
        help="present the robot on the myagv_ros topics via rosbridge (usually 9090). "
    )
    ap.add_argument(
        "--watchdog", type=float, default=0.5,
        help="stop the base if no cmd_vel arrives for this many seconds",
    )
    # The lidar defaults are the YDLidar X2's, from ydlidar_ros_driver/launch/X2.launch
    # and the base_footprint -> laser_frame transform in myagv_active.launch.
    ap.add_argument("--scan-beams", type=int, default=360, dest="scan_beams",
                    help="rays in the simulated lidar's 360 deg sweep, published on /scan. "
                         "The X2's 3 kHz sample rate at 10 Hz gives ~300; 360 is close and "
                         "keeps one beam per degree")
    # These three default per robot (SCAN_DEFAULTS): the X2's figures are the myAGV's
    # hardware, and the AiNex has no lidar for them to describe.
    ap.add_argument("--scan-range", type=float, default=None, dest="scan_range",
                    metavar="M", help="lidar maximum range in metres (myagv/X2: 12.0)")
    ap.add_argument("--scan-min-range", type=float, default=None, dest="scan_min_range",
                    metavar="M", help="lidar minimum range in metres (myagv/X2: 0.1)")
    ap.add_argument("--scan-offset", type=float, nargs=2, default=None,
                    dest="scan_offset", metavar=("X", "Z"),
                    help="laser origin ahead of and above the base; defaults to the "
                         "static transform in myagv_active.launch for the myAGV")
    ap.add_argument("--scan-hz", type=float, default=10.0, dest="scan_hz",
                    help="scan rate; the X2 spins at a fixed 10 Hz independent of the "
                         "control rate")
    ap.add_argument("--no-scan", action="store_true", dest="no_scan",
                    help="do not publish /scan")
    ap.add_argument("--action-dir", default=None, dest="action_dir", metavar="DIR",
                    help="AiNex only: directory of action groups for /app/set_action. "
                         "Reads Hiwonder's .d6a format, so this can point straight at a "
                         "real robot's ActionGroups directory; defaults to the small "
                         "in-tree set in robots/ainex/actions")
    ap.add_argument("--depth-hz", type=float, default=5.0, dest="depth_hz",
                    help="rate for the depth image; 640x480 float over a websocket is "
                         "1.2 MB a frame, so this is deliberately slower than the control rate")
    ap.add_argument("--depth-size", type=int, nargs=2, default=[320, 240], dest="depth_size",
                    metavar=("W", "H"))
    ap.add_argument("--depth-range", type=float, default=8.0, dest="depth_range",
                    metavar="M", help="depth beyond this reads as 'no return' (0 in 16UC1). "
                                      "Its own flag, not the lidar's: the two sensors have "
                                      "nothing to do with each other")
    ap.add_argument("--camera-hz", type=float, default=0.0, dest="camera_hz",
                    help="cap the colour camera's frame rate, independent of --control-hz. "
                         "0 (default) renders one frame per control tick, which is what "
                         "this has always done. The render is the dominant cost of a "
                         "second camera-bearing robot in the same physics loop -- measured "
                         "9.8 -> 5.7 Hz on the arm's own topics when a myAGV joins it -- so "
                         "this is the knob that buys it back.")
    ap.add_argument("--no-depth", action="store_true", dest="no_depth",
                    help="do not publish the depth image or camera info")
    args = ap.parse_args()

    from tools.scene_placement import apply_init_qpos, describe, find_robot_placement

    names = [n.strip() for n in args.robot.split(",") if n.strip()]
    if not names:
        raise SystemExit("no robot named")
    unknown = [n for n in names if n not in ROBOTS]
    if unknown:
        raise SystemExit(f"unknown robot(s) {unknown}; available: {', '.join(sorted(ROBOTS))}")

    # The ROS namespace defaults to the robot's own name, which is what makes a topic list
    # readable at a glance. Duplicates would collide on the wire exactly as bare names do,
    # so they are refused here rather than discovered as a lost `/cmd_vel` later.
    if args.ros_namespace is None:
        namespaces = list(names)
    else:
        namespaces = [n.strip() for n in args.ros_namespace.split(",")]
        if len(namespaces) != len(names):
            raise SystemExit(
                f"--ros-namespace has {len(namespaces)} entries for {len(names)} robot(s); "
                "they are matched positionally"
            )
    if len(set(namespaces)) != len(namespaces):
        raise SystemExit(f"--ros-namespace entries must be distinct, got {namespaces}")
    if len(names) > 1 and "" in namespaces:
        raise SystemExit(
            "the bare contract is only available for a single robot: with two robots "
            "unnamespaced they share /cmd_vel and /joint_states, and one silently wins"
        )

    instances = []
    for i, name in enumerate(names):
        config_cls, robot_cls = load_robot(name)
        config = config_cls()
        # The MJCF prefix and the ROS namespace are different things and stay different.
        # `robot_0/` prefixes bodies, joints, actuators and cameras inside one MuJoCo
        # model; `/so101` prefixes topics on the wire. Keeping the `robot_N/` shape also
        # keeps `apple_on_plate._clear_workspace`'s "never sink a body called robot_*"
        # rule working, which a prefix named after the robot would break.
        config.robot_namespace = f"robot_{i}/"
        instances.append(_Instance(
            name=name, config=config, robot_cls=robot_cls, ns=namespaces[i],
            holonomic=name in HOLONOMIC_BASE_ROBOTS,
            tabletop=name in TABLETOP_ROBOTS,
        ))

    # Everything below that still speaks of "the robot" means the first one: it is the one
    # the viewer frames and the one a single-robot invocation is about.
    primary = instances[0]
    config, robot_cls = primary.config, primary.robot_cls
    holonomic, tabletop = primary.holonomic, primary.tabletop

    # Compile the bare scene once: both the surface search and the floor search want a
    # model, and neither should pay for its own compile.
    spec = mujoco.MjSpec.from_file(args.scene)
    scene_model = spec.compile()
    scene_data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, scene_data)

    # Arms first, then mobile bases. The order is not cosmetic: an arm has no choice
    # about where it goes -- it needs a work surface at working height with something on
    # it -- while a base can start anywhere open, so the base is the one that gives way.
    # Placing the base first regularly picked the standing space in front of the very
    # counter the arm was about to be bolted to.
    keep_out: list[tuple[np.ndarray, float]] = []
    # Kept for the worktop pass below, which runs once the arm's mount is settled.
    arm_mount = None
    arm_want: tuple[str, ...] = ()
    arm_reach: tuple[float, float] = ARM_REACH["so101"]

    for inst in sorted(instances, key=lambda i: not i.tabletop):
        if inst.tabletop:
            reach = tuple(args.reach) if args.reach else ARM_REACH[inst.name]
            want = tuple(w.strip().lower() for w in (args.target or "").split(",") if w.strip())
            arm_reach, arm_want = reach, want
            mount, to_spawn = place_arm_on_table(
                args.scene, scene_model, scene_data, inst.name, reach, args.spawn_objects,
                want=want, edge_bias=not args.mount_centre,
            )
            if to_spawn is not None:
                add_tabletop_objects(spec, *to_spawn)
            pos = np.array(args.pos) if args.pos is not None else mount.xy
            yaw = np.radians(args.yaw) if args.yaw is not None else mount.yaw
            print(describe(mount), file=sys.stderr)
            if args.pos is not None or args.yaw is not None:
                # The surface is still what was chosen -- and what supplies the height -- so say
                # what actually got used rather than leaving the line above to be read as final.
                print(
                    f"overridden to ({pos[0]:.2f}, {pos[1]:.2f}, {mount.z:.2f}) "
                    f"yaw {np.degrees(yaw):.0f} deg",
                    file=sys.stderr,
                )
            # The pedestal was there to lift a tabletop arm off the floor. Standing on the
            # table, it would only bury the arm in the surface.
            inst.config.base_size = list(TABLETOP_RISER)
            inst.attach_pos = [float(pos[0]), float(pos[1]), float(mount.z)]
            # The pose actually used, override included -- the worktop pass below places
            # objects around where the arm *is*, not where the search wanted it.
            arm_mount = dataclasses.replace(
                mount, xy=np.asarray(pos[:2], dtype=float), yaw=float(yaw)
            )
            # An arm reaches out over the floor beside its counter, and the task stages a
            # 0.92 m slab there. Anything that drives must stay out of both.
            keep_out.append((np.asarray(pos[:2], dtype=float),
                             ARM_BODY_RADIUS.get(inst.name, 0.20) + reach[1]))
        else:
            placement = find_robot_placement(
                args.scene,
                model=scene_model,
                data=scene_data,
                pos=args.pos,
                yaw_deg=args.yaw,
                prefer="floor",
                exclude=tuple(keep_out),
            )
            pos, yaw = placement.pos, placement.yaw
            print(describe(placement), file=sys.stderr)
            inst.attach_pos = [float(pos[0]), float(pos[1]), 0.0]
            keep_out.append((np.asarray(pos[:2], dtype=float),
                             ARM_BODY_RADIUS.get(inst.name, 0.25)))
        inst.pos, inst.yaw = pos, yaw

    pos, yaw, attach_pos = primary.pos, primary.yaw, primary.attach_pos

    quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]

    # The robot a task is about is the arm, whichever position it was named in.
    arm_instance = next((i for i in instances if i.tabletop), primary)

    stage_task = None
    if args.task:
        module_name, stage_name, arbiter_name = TASKS[args.task]
        task_module = importlib.import_module(module_name)
        stage_task = (getattr(task_module, stage_name), getattr(task_module, arbiter_name))
        # The riser goes when a task is staged. It exists to read as a mount, but the
        # task's geometry is measured from the arm's base with the work surface at z = 0
        # -- which is exactly what the reference rig has, its base_link sitting on the
        # table. Three centimetres of pedestal would shift every waypoint height in the
        # scripted plan relative to the surface the objects actually rest on, and the
        # top-down grasp envelope has nothing like that much slack.
        arm_instance.config.base_size = None

    for inst in instances:
        inst_quat = [float(np.cos(inst.yaw / 2)), 0.0, 0.0, float(np.sin(inst.yaw / 2))]
        inst.robot_cls.add_robot_to_scene(
            inst.config,
            spec,
            prefix=inst.mjcf,
            # A holonomic base has to be grafted in at the origin; it is driven to its
            # spawn pose below instead.
            pos=[0.0, 0.0, 0.0] if inst.holonomic else inst.attach_pos,
            quat=[1.0, 0.0, 0.0, 0.0] if inst.holonomic else inst_quat,
        )

    # No task, so nothing else is going to furnish the worktop: put the `--target` pair on
    # it out of the house's own objects and clear the rest of the clutter out of the arm's
    # way. With a task staged this is skipped -- `apple_on_plate` brings its own measured
    # apple and plate and clears the same radius itself.
    if stage_task is None and arm_mount is not None and arm_want:
        placed, cleared = arrange_targets(
            spec, scene_model, scene_data, arm_mount, arm_want, arm_reach
        )
        if placed:
            print(f"--target {','.join(arm_want)}: placed {', '.join(placed)} on the worktop",
                  file=sys.stderr)
        if cleared:
            print(f"  cleared {len(cleared)} scene object(s) from the working area: "
                  f"{', '.join(cleared)}", file=sys.stderr)

    if stage_task is not None:
        # The house's own apple and plate, moved to the task's positions and handed to
        # the task under its body names. Before `stage()`, because it is the `task_`
        # prefix that keeps the workspace clearing from sinking them.
        if args.native_objects:
            adopted = adopt_native_objects(
                spec, scene_model, scene_data, arm_instance.attach_pos, arm_instance.yaw,
                task_module, swap=args.swap_objects,
            )
            print(f"task {args.task}: {adopted}", file=sys.stderr)
        # The arm's base body lands exactly at attach_pos now that the riser is gone, so
        # that pose *is* the task's frame origin.
        cleared = stage_task[0](
            spec, arm_instance.attach_pos, arm_instance.yaw,
            reference_table=args.reference_table,
            dressing=args.dressing,
            lighting=args.reference_lighting,
            extra_lights=args.extra_lights,
            swap=args.swap_objects,
            objects="engine" if args.native_objects else "task",
        )
        if cleared:
            print(f"task {args.task}: cleared {len(cleared)} scene object(s) from the "
                  f"working area: {', '.join(n.split('_')[0] for n in cleared)}", file=sys.stderr)

    model = spec.compile()
    data = mujoco.MjData(model)

    # Set the rest pose before anything is stepped. Both the joint state and the
    # actuator targets have to be set: these are position actuators, so leaving ctrl at
    # its default of 0 would make the robot immediately drive out of the pose.
    for inst in instances:
        inst.view = inst.config.robot_view_factory(data, inst.mjcf)
        apply_init_qpos(inst.view, inst.config)

        # Any number of grippers: the AiNex has two ("left_gripper", "right_gripper") where
        # every other robot here has exactly one called "gripper".
        if args.gripper != "rest":
            for group_id in inst.view.move_group_ids():
                if not group_id.endswith("gripper"):
                    continue
                gripper = inst.view.get_move_group(group_id)
                gripper.set_gripper_ctrl_open(args.gripper == "open")
                # ctrl alone only shows up once the sim runs; the joint has to be moved too
                # for a still render to show anything.
                gripper.joint_pos = gripper.ctrl

        if inst.holonomic:
            base_pose = np.eye(4)
            base_pose[:2, 3] = inst.pos[:2]
            c, sn = np.cos(inst.yaw), np.sin(inst.yaw)
            base_pose[:2, :2] = np.array([[c, -sn], [sn, c]])
            base = inst.view.get_move_group("base")
            base.pose = base_pose
            # Hold there rather than driving back to the origin on the first step.
            base.ctrl = np.array([inst.pos[0], inst.pos[1], inst.yaw], dtype=np.float64)

    view = primary.view

    mujoco.mj_forward(model, data)
    for inst in instances:
        warn_on_penetration(model, data, inst.mjcf)

    task = None
    if stage_task is not None:
        # Built after the rest pose is applied and mj_forward has run, because it
        # captures this state as the one /reset restores -- an arbiter constructed a few
        # lines earlier would snapshot an arm that had not been posed yet.
        #
        # The prefix is passed rather than inferred. The arbiter used to find the arm by
        # looking for the one body in the model whose name ends in `/base` -- which stops
        # working the moment a second robot is in the scene, because every robot's root is
        # called `base`. Naming the arm is both correct and cheaper.
        task = stage_task[1](model, data, prefix=arm_instance.mjcf)
        placed, reason = task.instantaneous(data)
        print(
            f"task {args.task}: staged; success predicate reads "
            f"{'TRUE (!)' if placed else reason} at spawn",
            file=sys.stderr,
        )
        print(f"task {args.task}: {task.reach_report(data, ARM_REACH[arm_instance.name])}",
              file=sys.stderr)
        check_task_contacts(model, arm_instance.mjcf, task)

    ns = config.robot_namespace
    # Robots on a mocap pedestal have a "mount" body; free-standing and holonomic ones
    # are their own root.
    anchor = "mount"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{ns}{anchor}") < 0:
        anchor = robot_cls.robot_model_root_name()
    anchor_pos = data.body(f"{ns}{anchor}").xpos.copy()
    print(
        f"{args.robot} in scene: {model.nbody} bodies, {model.ngeom} geoms, "
        f"{model.nu} actuators; at {np.round(anchor_pos, 3)}",
        file=sys.stderr,
    )
    # Frame the camera on the robot's own geometry rather than a fixed height, so a
    # 13 cm AGV and a 1 m arm on a pedestal both fill the view.
    robot_geoms = np.array(
        [
            data.geom_xpos[gid]
            for gid in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or "").startswith(ns)
        ]
    )
    if len(robot_geoms):
        lookat = (robot_geoms.min(axis=0) + robot_geoms.max(axis=0)) / 2
        robot_radius = max(float(np.linalg.norm(robot_geoms.max(axis=0) - robot_geoms.min(axis=0))) / 2, 0.2)
    else:
        lookat = anchor_pos + np.array([0.0, 0.0, 0.5])
        robot_radius = 0.5
    distance = args.distance if args.distance is not None else robot_radius * 3.5

    # Look at the robot from the side it is facing, so it is not hidden behind furniture.
    azimuth = args.azimuth if args.azimuth is not None else np.degrees(yaw) + 180.0

    if args.render:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        for gid in range(model.ngeom):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            if "ceiling" in name:
                model.geom_rgba[gid, 3] = 0.0
        if args.render_camera:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.render_camera)
            if cam_id < 0:
                declared = [model.camera(i).name for i in range(model.ncam)]
                raise SystemExit(f"--render-camera {args.render_camera!r}: not in model; "
                                 f"declared cameras: {declared}")
            width, height = (int(v) for v in model.cam_resolution[cam_id])
            if width <= 1 or height <= 1:
                # A camera with no `resolution` attribute compiles to 1x1 and renders
                # nothing useful; fall back to the requested size but say so.
                print(f"camera {args.render_camera!r} declares no resolution; using "
                      f"{args.width}x{args.height}", file=sys.stderr)
                width, height = args.width, args.height
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            with mujoco.Renderer(model, height, width) as renderer:
                renderer.update_scene(data, camera=args.render_camera)
                pixels = renderer.render()
        else:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.lookat[:] = lookat
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = args.elevation
            with mujoco.Renderer(model, args.height, args.width) as renderer:
                renderer.update_scene(data, camera=cam)
                pixels = renderer.render()
        if args.render_framing:
            from mujoco_bridge import camera_framing, clipped_fraction, report_slab_fit

            report_slab_fit(model, data)
            clipped = clipped_fraction(pixels)
            if args.render_camera and mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "task_table"
            ) >= 0:
                from mujoco_bridge import _slab_corners_world

                slab = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task_table")
                uv = camera_framing(model, data, args.render_camera,
                                    _slab_corners_world(model, data, slab))
                worst = max(max(abs(u), abs(v)) for u, v in uv)
                inside = sum(1 for u, v in uv if abs(u) <= 1.0 and abs(v) <= 1.0)
                corners = "  ".join(f"({u:+.3f},{v:+.3f})" for u, v in uv)
                print(f"framing {args.render_camera}: {inside}/4 table corners in frame, "
                      f"worst |normalised| {worst:.3f} (reference 0.930)", file=sys.stderr)
                print(f"  corners {corners}", file=sys.stderr)
            print(f"exposure: {clipped * 100:.1f}% of pixels clipped to white "
                  f"(reference 3.0%, and 41.6% before it was fixed)", file=sys.stderr)

        from PIL import Image

        Image.fromarray(pixels).save(args.render)
        print(f"wrote {args.render}", file=sys.stderr)
        return 0

    # Camera selection moved into `_pick_camera`, because it is per robot: each base
    # resolves `front_camera` against its own MJCF prefix, and a single `camera` variable
    # here would have handed the second robot the first one's view.

    if args.task and not args.ros_port and not args.render:
        raise SystemExit(
            "--task needs --ros-port: the task publishes its objects and cameras there, and "
            "staging one with nothing to publish it would silently score nothing."
        )

    if args.ros_port:
        missing = [i.name for i in instances if i.name not in ROS_SURFACES]
        if missing:
            raise SystemExit(
                f"--ros-port: no ROS surface for {missing}; "
                f"available: {', '.join(sorted(ROS_SURFACES))}"
            )

        # One server, one port, one graph -- and a namespace per robot. Two servers on two
        # ports would be two graphs: nothing could discover the fleet, and no one client
        # could drive it. See shared/contracts/namespace.py.
        from ros_surfaces import RobotFleet

        fleet = RobotFleet(port=args.ros_port, host=args.control_host)
        for inst in instances:
            module_name, func_name = ROS_SURFACES[inst.name]
            attach_ros = getattr(importlib.import_module(module_name), func_name)
            fleet.attach(inst.ns, attach_ros, **_surface_kwargs(args, inst, model, task))
        fleet.start()
        controller = fleet
    else:
        controller = None

    control_period = 1.0 / args.control_hz
    next_control = 0.0

    deadline = None if args.timeout is None else time.monotonic() + args.timeout

    from mujoco_bridge import run_sim_loop

    try:
        if args.headless:
            # No window: what an automated console-connectivity check and a displayless
            # host run. Same loop as the viewer path, which is the point -- the ROS
            # server behaves identically either way.
            run_sim_loop(model, data, controller, control_hz=args.control_hz,
                         deadline=deadline, label="headless loop")
        else:
            # Bound as a separate name: `import mujoco.viewer` here would make `mujoco` a
            # function-local and shadow the module-level import above.
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(model, data) as viewer:
                viewer.cam.lookat[:] = lookat
                viewer.cam.distance = distance
                viewer.cam.azimuth = azimuth
                viewer.cam.elevation = args.elevation
                run_sim_loop(model, data, controller, control_hz=args.control_hz,
                             deadline=deadline, viewer=viewer, label="viewer loop")
    finally:
        if controller is not None:
            controller(None)  # close
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
