#!/usr/bin/env python
"""Spawn a shared robot into a RoboCasa kitchen and view, render, or serve it.

    mjpython tools/spawn_robot.py myagv --layout 1 --style 1 --ros-port 9090
    python   tools/spawn_robot.py myagv --headless --ros-port 9090
    python   tools/spawn_robot.py so101 --ros-port 9091 --task apple_on_plate
    python   tools/spawn_robot.py myagv --render /tmp/kitchen.png

This is the RoboCasa half of the multi-engine split, and it mirrors
`molmospaces/tools/spawn_robot.py` deliberately: same subcommand surface, same flags,
and -- the part that matters -- the same wire contracts out of `simulator/shared/`, so
`robot_console` cannot tell which engine it is driving.

**RoboCasa is used here as a scene provider, not as a robot stack.** The kitchen is built
straight from `KitchenArena` with an empty robot list, which yields a 44-fixture, 825-geom
kitchen with zero actuators; the shared robot MJCF is then grafted into that spec and the
whole thing is stepped by plain MuJoCo. Going through `robosuite.make` instead would drag
in a robosuite robot (its own controller stack, action space and observation dict) that
would then have to be surgically removed from the compiled model, and it would put a
Panda in the middle of every map. The robots here are not robosuite robots and are not
pretending to be: the myAGV is a vendor ROS device and the SO-101 speaks the
ros2_control topic set, and both of those are the *hardware's* interface.

The RoboCasa-specific traps, which is most of what this file knows that its MolmoSpaces
counterpart does not:

* **Geom groups are inverted from the MolmoSpaces convention.** RoboCasa puts collision
  hulls in group 0 (painted in random semi-transparent colours -- 501 of them in layout 1)
  and the visual meshes in group 1. Rendering MuJoCo's default groups therefore streams a
  camera feed full of translucent red and green boxes. Everything that renders here goes
  through `visual_only()`.
* **Clearance has to be measured to geom surfaces, not geom centres.** A kitchen is four
  long wall boxes and a run of counters; the centre of a 5 m wall is metres away from a
  robot pressed against it, so a centre-distance search parks the robot in the wall. The
  search below uses world-space AABBs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
# The wire bridge (contracts.*), the MuJoCo helpers and the robot specs live in
# simulator/shared, which env.sh also puts on PYTHONPATH.
_SHARED = SIM_ROOT.parent / "shared"
if _SHARED.is_dir() and str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from mujoco_bridge import PlanarJointBase  # noqa: E402
from robots_spec import model_xml  # noqa: E402

# Robots this engine can spawn. Kept to the shared specs: anything here must also spawn
# in the MolmoSpaces engine, since that is what "the console cannot tell them apart"
# means in practice.
ROBOTS = ("myagv", "so101")

# name -> (module, function) presenting that robot's own vendor ROS topics. Only mobile
# bases have one; an arm is served over the control protocol instead.
ROS_SURFACES = {
    "myagv": ("ros_surfaces.myagv", "attach_ros"),
    # The same shared surface the other engine uses. That is the point: the arm's topic
    # set belongs to the arm, so a client cannot tell which engine is hosting it.
    "so101": ("ros_surfaces.so101", "attach_ros"),
}

# Robots whose ROS surface is an arm contract rather than a mobile-base one: no cmd_vel,
# no odometry, no lidar, and several cameras instead of one.
ARM_ROS_SURFACES = {"so101"}

# Tasks an engine can stage into its scene; see simulator/shared/tasks/.
TASKS = {"apple_on_plate": ("tasks.apple_on_plate", "stage", "AppleOnPlate")}

# Transcribed from ydlidar_ros_driver/launch/X2.launch and the base_footprint ->
# laser_frame transform in myagv_active.launch. Same numbers as the MolmoSpaces engine;
# a client that could measure a difference here would have found a regression.
SCAN_DEFAULTS = {"myagv": {"offset": (0.065, 0.08), "min_range": 0.1, "max_range": 12.0}}

# Robots grafted in at the origin and then *driven* to their spawn pose, because their
# base joints are world-aligned slides.
HOLONOMIC_BASE_ROBOTS = {"myagv"}
# Arms: no base of their own, so they are bolted to a work surface. In a kitchen that is
# a countertop rather than a table.
TABLETOP_ROBOTS = {"so101"}

# Footprint radius used when searching for somewhere to stand. The myAGV chassis is
# 311 x 230 mm, so its half-diagonal is 0.193 m; the margin is what keeps a spawn from
# touching a cabinet door it would then have to unstick itself from.
ROBOT_RADIUS = {"myagv": 0.193, "so101": 0.20}
SPAWN_MARGIN_M = 0.12

# The height band a driving robot sweeps through. The floor sits at z=0 and RoboCasa
# hangs wall cabinets from about 1.4 m, so anything between counts as in the way.
FLOOR_BAND = (0.02, 1.3)
# Counter height in a RoboCasa kitchen is ~0.90 m; the band is wide enough for the island
# and breakfast-bar variants without catching a wall cabinet.
COUNTER_BAND = (0.70, 1.15)

# The SO-101's working annulus, from molmospaces/robots/so101/so101_config.py:41. Short of
# its full ~0.4 m reach: the last few centimetres are a straight-out arm with no usable
# orientation left.
ARM_REACH = (0.15, 0.35)

# The 5 arm joints and the gripper, in MJCF order. Two move groups, because that is what
# the control protocol's clients (`robot_console.arm_client`) expect to be offered.
SO101_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
SO101_GRIPPER_JOINTS = ("gripper",)
SO101_TCP_BODY = "gripper"
# The elbow-up rest pose and near-open gripper from
# molmospaces/robots/so101/so101_config.py:28. Copied rather than imported -- that file
# is a MolmoSpaces adapter and importing it here would put molmo_spaces in this venv --
# but it is the same robot, so it starts in the same pose in both engines.
SO101_REST_QPOS = (0.0, -0.6, 1.0, 0.6, 0.0)
SO101_REST_GRIPPER = (1.2,)
# The base plate's meshes hang 2.4 mm below the body origin (see the `pos` on
# base_motor_holder_so101_v1 in the shared MJCF), so mounting the origin exactly on a
# countertop buries the plate in it. Lift by enough to clear that, not enough to float.
SO101_BASE_LIFT = 0.004


def visual_only() -> mujoco.MjvOption:
    """A scene option that shows RoboCasa's visual meshes and hides its collision hulls.

    Group 0 is collision and group 1 is visual in a RoboCasa kitchen -- the reverse of the
    MuJoCo-default reading, and the reverse of the shared robot MJCFs, which use group 2
    for visual and 3 for collision. Both robot and kitchen visuals survive this mask.
    """
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    opt.geomgroup[2] = 1
    return opt


def build_kitchen_arena(layout: int, style: int, seed: int):
    """Compile a RoboCasa kitchen, fixtures and all, with no robot in it.

    `ManipulationTask` with an empty robot list is the whole trick. The env class
    (`robocasa.environments.kitchen.Kitchen`) exists to place *task objects* and drive a
    robosuite robot; neither is wanted here, and skipping it is what leaves a scene with
    zero actuators for the shared robot to be the only thing in.
    """
    import robocasa  # noqa: F401  -- registers assets_root
    from robocasa.models.scenes.kitchen_arena import KitchenArena
    from robosuite.models.tasks import ManipulationTask

    arena = KitchenArena(layout_id=layout, style_id=style, rng=np.random.default_rng(seed))
    arena.set_origin([0, 0, 0])
    fixtures = [cfg["model"] for cfg in arena.get_fixture_cfgs()]
    print(f"layout {layout}, style {style}: {len(fixtures)} fixtures", file=sys.stderr)

    def compile_spec(extra_objects=()):
        task = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[],
            mujoco_objects=fixtures + list(extra_objects),
            enable_multiccd=True,
            enable_sleeping_islands=False,
        )
        # robosuite rewrites every mesh/texture `file` to an absolute path when it loads
        # the arena, so the XML is self-contained and needs no assets dict or meshdir.
        return mujoco.MjSpec.from_string(task.get_xml())

    return arena, compile_spec


def world_boxes(model, data, band: tuple[float, float]) -> np.ndarray:
    """World-space xy rectangles for the collision geoms inside a height band.

    Returns an (n, 4) array of [cx, cy, ex, ey]. MuJoCo keeps a local AABB per geom;
    rotating its extents by the absolute value of the geom's frame gives a conservative
    world-aligned box, which for a kitchen (everything axis-aligned) is exact.
    """
    boxes = []
    zlo, zhi = band
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        centre_local = model.geom_aabb[gid][:3]
        extent_local = model.geom_aabb[gid][3:]
        rot = data.geom_xmat[gid].reshape(3, 3)
        centre = data.geom_xpos[gid] + rot @ centre_local
        extent = np.abs(rot) @ extent_local
        if centre[2] + extent[2] < zlo or centre[2] - extent[2] > zhi:
            continue
        boxes.append([centre[0], centre[1], extent[0], extent[1]])
    return np.array(boxes) if boxes else np.zeros((0, 4))


def clearance_field(boxes: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Distance from every grid point to the nearest box *surface* (0 inside one)."""
    if len(boxes) == 0:
        return np.full(len(grid), np.inf)
    delta = np.abs(grid[:, None, :] - boxes[None, :, :2]) - boxes[None, :, 2:]
    return np.linalg.norm(np.maximum(delta, 0.0), axis=-1).min(axis=1)


def find_open_floor(model, data, radius: float, step: float = 0.05, keep_out=()):
    """Return (xy, yaw) for the roomiest patch of floor, facing the middle of the room.

    Facing the room centre rather than a fixed heading matters for a robot that starts by
    looking at what is in front of it: spawned nose-first into a cabinet, the first thing
    an explorer sees is 30 cm of door.
    """
    boxes = world_boxes(model, data, FLOOR_BAND)
    # Keep-outs are simply more obstacles. `clearance_field` already measures distance to
    # box surfaces, so where another robot is standing -- and the room it needs -- costs
    # nothing extra to express, and the "no patch clears N metres" failure below stays the
    # one failure this function has.
    if len(keep_out):
        extra = np.asarray(keep_out, dtype=float).reshape(-1, 4)
        boxes = extra if len(boxes) == 0 else np.vstack([boxes, extra])
    if len(boxes) == 0:
        return np.zeros(2), 0.0

    lo = (boxes[:, :2] - boxes[:, 2:]).min(axis=0)
    hi = (boxes[:, :2] + boxes[:, 2:]).max(axis=0)
    xs = np.arange(lo[0], hi[0] + step, step)
    ys = np.arange(lo[1], hi[1] + step, step)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    clear = clearance_field(boxes, grid)
    best_i = int(np.argmax(clear))
    best, best_clear = grid[best_i], float(clear[best_i])
    if best_clear < radius:
        raise SystemExit(
            f"no floor patch in this kitchen clears {radius:.2f} m (best {best_clear:.2f} m"
            + (f", with {len(keep_out)} robot keep-out(s) in the way" if len(keep_out) else "")
            + "); try another --layout, or place the robot by hand with --pos/--yaw"
        )

    # The centroid of the free space, not of the room: in an L-shaped kitchen the room
    # centre can be inside a counter run.
    free = grid[clear > radius]
    centre = free.mean(axis=0)
    to_centre = centre - best
    yaw = float(np.arctan2(to_centre[1], to_centre[0])) if np.linalg.norm(to_centre) > 1e-6 else 0.0
    print(
        f"placing robot at ({best[0]:.2f}, {best[1]:.2f}), {best_clear:.2f} m clear, "
        f"facing the open floor (yaw {np.degrees(yaw):.0f} deg)",
        file=sys.stderr,
    )
    return best, yaw


def counter_regions(arena) -> list[dict]:
    """The free worktop rectangles of every counter, in world coordinates.

    RoboCasa already knows this. `Counter.get_reset_regions()` returns the very regions
    the dataset uses to place task objects -- the worktop minus the sink cut-out, minus
    the hob -- so asking it is both correct and far less code than inferring worktops from
    collision AABBs. That inference is a trap worth naming: a sink basin's floor is "a
    flat surface at counter height" whose centre is a clean 0.22 m clear of anything, so
    it beats a real worktop on every geometric score, and the arm gets mounted in the
    sink. Called with `env=None`, which the `ref=None` path never touches.
    """
    from robocasa.models.fixtures.counter import Counter

    regions = []
    for name, fixture in arena.fixtures.items():
        if not isinstance(fixture, Counter):
            continue
        try:
            found = fixture.get_reset_regions(env=None)
        except Exception as exc:  # a fixture variant with a different region model
            print(f"  ({name}: no reset regions -- {exc})", file=sys.stderr)
            continue
        rot = float(getattr(fixture, "rot", 0.0) or 0.0)
        c, s = np.cos(rot), np.sin(rot)
        for region_name, region in found.items():
            ox, oy, oz = region["offset"]
            regions.append({
                "name": f"{name}/{region_name}",
                # The fixture frame is a yaw rotation about the fixture's world position.
                "centre": np.array([
                    fixture.pos[0] + c * ox - s * oy,
                    fixture.pos[1] + s * ox + c * oy,
                ]),
                "half": np.array(region["size"], dtype=float) / 2.0,
                "top_z": float(fixture.pos[2] + oz),
                "rot": rot,
            })
    return regions


def outward_direction(region: dict, floor: np.ndarray) -> np.ndarray:
    """Which way off this worktop is the room, rather than the wall behind it.

    Decided by looking, not by convention: a counter's two long faces are the wall side
    and the standing side, and whichever has floor a person could stand on is the one the
    arm should be working towards. Guessing from the fixture's rotation instead gets it
    backwards on any island or any counter the layout mirrored.
    """
    c, s = np.cos(region["rot"]), np.sin(region["rot"])
    axes = (np.array([c, s]), np.array([-s, c]))
    # The short axis is the counter's depth; "out" is across it, not along the run.
    short = int(np.argmin(region["half"]))
    direction = axes[short]
    probe = region["half"][short] + 0.6

    best, best_clear = direction, -np.inf
    for sign in (1.0, -1.0):
        point = (region["centre"] + sign * direction * probe).reshape(1, 2)
        clear = float(clearance_field(floor, point)[0])
        if clear > best_clear:
            best, best_clear = sign * direction, clear
    return best


def find_counter_mount(arena, model, data, radius: float, reach=ARM_REACH):
    """Return (xy, z, yaw, out) for an arm mounted at the back of the roomiest worktop.

    At the *back* on purpose. The arm's working annulus starts 0.15 m out, so an arm in
    the middle of a 0.6 m counter can only reach the front lip and the empty air past it;
    put it against the wall and the whole depth of the counter is inside its reach.
    """
    regions = counter_regions(arena)
    if not regions:
        raise SystemExit(
            "this kitchen has no counter with a free worktop region: nothing to mount the "
            "arm on. Try another --layout, or place it by hand with --pos/--yaw."
        )

    floor = world_boxes(model, data, FLOOR_BAND)
    # Roomiest first: the arm needs its own footprint plus somewhere to put the objects.
    usable = [r for r in regions if min(r["half"]) * 2 >= radius + reach[0]]
    if not usable:
        usable = regions
    region = max(usable, key=lambda r: float(r["half"][0] * r["half"][1]))

    out = outward_direction(region, floor)
    depth = float(min(region["half"]))
    # Sit against the back edge, inset by the arm's own footprint.
    xy = region["centre"] - out * max(depth - radius, 0.0)
    yaw = float(np.arctan2(out[1], out[0]))
    print(
        f"mounting arm on {region['name']} at ({xy[0]:.2f}, {xy[1]:.2f}, "
        f"{region['top_z']:.2f}), worktop {2 * region['half'][0]:.2f} x "
        f"{2 * region['half'][1]:.2f} m, facing the room (yaw {np.degrees(yaw):.0f} deg)",
        file=sys.stderr,
    )
    # The free worktop as the arm sees it, in its base frame: x forward from the base,
    # back edge `radius` behind it, front edge the region's full depth ahead of that; y
    # along the run. The native-object placer needs this to keep things on the counter.
    worktop = (-float(radius), 2.0 * depth - float(radius), float(max(region["half"])))
    return xy, region["top_z"] + SO101_BASE_LIFT, yaw, out, worktop


def make_kitchen_objects(categories, seed: int, scale: float = 1.0):
    """Build one RoboCasa object per category, from the assets already on disk.

    `sample_kitchen_object` is RoboCasa's own sampler, so "bowl" resolves to whichever
    bowl models this install actually has rather than a hardcoded path. Size-capped so
    the pair stays something a 0.4 m arm could plausibly work with.

    `scale` shrinks everything spawned. These objects exist to be *manipulated by the
    SO-101*, whose gripper opens roughly 7 cm at the tips -- a real-world-sized apple is
    at the edge of that span, and an episode against an ungraspable object fails no
    matter how well the policy does. This was learned the expensive way: an LLM agent
    spent five well-aimed grasp cycles on a full-sized apple whose surface the fingers
    could only slide off.
    """
    from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
    from robocasa.models.objects.objects import MJCFObject

    rng = np.random.default_rng(seed)
    objects = []
    for i, category in enumerate(categories):
        kwargs, info = sample_kitchen_object(
            groups=[category],
            rng=rng,
            obj_registries=("objaverse", "lightwheel"),
            max_size=(0.30, 0.30, 0.30),
            object_scale=scale if scale != 1.0 else None,
        )
        objects.append(MJCFObject(name=f"obj_{i}_{category}", **kwargs))
        print(f"  object {category}: {Path(info['mjcf_path']).parent.name}", file=sys.stderr)
    return objects


#: Which of RoboCasa's registry models stand in for the task's apple and plate. Chosen by
#: measuring every model's texture, not by eye: apple_10 is the reddest of the 22 apples
#: (mean RGB 151/12/8; the sampler's usual pick, apple_13, is 185/153/64 -- yellow, which
#: the instruction's "red apple" and the camera verdict's red-hue detector both miss), and
#: plate_4 is the only pure-white plate (255/255/255; plate_19, the usual pick, is a
#: 173-grey the detector's `val > 150` gate only just admits).
NATIVE_MODELS = {"apple": "apple_10", "plate": "plate_4"}

#: The dressing, from the registry: the task's own four (bowl, mug, banana, lemon --
#: what the reference rig keeps on its table) plus four more so the counter reads as a
#: kitchen rather than a rig. Nothing red: the instruction names *the red apple* and
#: the camera verdict finds it by hue, so a red cup is a second apple to both. Measured
#: on the textures, both registry cups are red (redness 88 and 136 on the scale that
#: puts apple_10 at 141), so there is no cup and a pear instead; the sampler's default
#: mug and bowl were red too, so those are pinned to the least red of their kind
#: (mug_7 at -41, bowl_11 at -19). `None` lets the sampler choose, at the engine's own
#: 0.7x graspable scale; `layout_native_dressing` keeps them from overlapping.
NATIVE_DRESSING: dict[str, str | None] = {
    "bowl": "bowl_11",
    "mug": "mug_7",
    "banana": None,
    "lemon": None,
    "orange": None,
    "pear": None,
    "bread": None,
    "kiwi": None,
}


def make_task_objects(task, *, swap: bool, dressing: bool = True, seed: int = 0) -> list:
    """RoboCasa's own apple, plate and dressing, sized for the task and named for it.

    The engine-native counterpart of the task's measured YCB set. The apple and plate
    are scaled from their registry bounding boxes to the task's radii -- the apple to
    the contract's 20 mm (apple_10 ships at ~65 mm, past the ~70 mm the jaw opens), the
    plate to 0.10 m. The dressing is sampled by category (`NATIVE_DRESSING`) at the
    engine's graspable scale; which model each category resolves to depends on `seed`.

    Bodies are renamed to the task's names in `adopt_task_objects`, after the kitchen
    spec is compiled around them; the apple's physics is replaced there too.
    """
    import xml.etree.ElementTree as ET

    import robocasa
    from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
    from robocasa.models.objects.objects import MJCFObject

    root = Path(robocasa.__file__).resolve().parent / "models" / "assets" / "objects" / "objaverse"
    radii = {"apple": task.APPLE_RADIUS, "plate": task.PLATE_RADIUS}
    objects = []
    for category, model_name in NATIVE_MODELS.items():
        path = root / category / model_name / "model.xml"
        bbox = ET.parse(path).getroot().find(".//geom[@name='reg_bbox']")
        half = max(float(v) for v in bbox.get("size").split()[:2])
        scale = radii[category] / half
        kwargs, _info = sample_kitchen_object(groups=str(path), object_scale=scale)
        objects.append(MJCFObject(name=f"task_{category}_native", **kwargs))
        print(f"  native {category}: {model_name} scaled x{scale:.2f} "
              f"(registry half-extent {half * 1000:.0f} mm -> {radii[category] * 1000:.0f} mm)",
              file=sys.stderr)
    if dressing:
        rng = np.random.default_rng(seed)
        for category, pinned in NATIVE_DRESSING.items():
            groups = str(root / category / pinned / "model.xml") if pinned else [category]
            kwargs, info = sample_kitchen_object(
                groups=groups, rng=rng, obj_registries=("objaverse", "lightwheel"),
                max_size=(0.30, 0.30, 0.30), object_scale=0.7,
            )
            objects.append(MJCFObject(name=f"task_{category}_native", **kwargs))
            print(f"  native {category}: {Path(info['mjcf_path']).parent.name}"
                  f"{' (pinned)' if pinned else ''}", file=sys.stderr)
    return objects


def layout_native_dressing(objects, poses, worktop, task) -> dict[str, np.ndarray]:
    """xy in the arm base frame for each free dressing object, none of them overlapping.

    The task's own dressing positions are the first choice for the four categories the
    reference rig has -- they are what the VLA's training scenes looked like -- but they
    were laid out for the *standard* layout: with the plate at the apple's spawn, the
    banana at (0.156, 0.156) reaches into the plate's footprint. So every spot is checked
    against what is already down (the apple, the plate, the arm's own footprint, and each
    object placed before it), against the worktop's edges, and against the straight line
    the apple travels along to the plate; the nearest free cell to the preferred spot
    wins. Extras prefer the far corners of the worktop, away from the working area.

    Each object's footprint is robosuite's own `horizontal_radius`, plus a margin. An
    object with no free cell is left out rather than squeezed in, and said so.
    """
    x_min, x_max, y_half = worktop
    margin = 0.02
    apple = np.asarray(poses["apple"][:2], dtype=float)
    plate = np.asarray(poses["plate"][:2], dtype=float)
    placed: list[tuple[np.ndarray, float]] = [
        (apple, task.APPLE_RADIUS),
        (plate, task.PLATE_RADIUS),
        (np.zeros(2), 0.16),  # the arm's base and its elbow room at rest
    ]
    preferred = {name: np.asarray(pos[:2], dtype=float) for name, pos, *_ in task.DRESSING}
    corners = [np.array([0.12, 0.45]), np.array([0.12, -0.45]),
               np.array([0.40, 0.45]), np.array([0.40, -0.45])]

    def off_the_carry_line(c: np.ndarray, r: float) -> bool:
        d = plate - apple
        t = float(np.clip(np.dot(c - apple, d) / max(float(np.dot(d, d)), 1e-9), 0.0, 1.0))
        return float(np.linalg.norm(c - (apple + t * d))) > r + 0.05

    xs = np.arange(x_min + 0.05, x_max - 0.03, 0.02)
    ys = np.arange(-y_half + 0.03, y_half - 0.03, 0.02)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    spots: dict[str, np.ndarray] = {}
    extra = 0
    for obj in objects:
        category = obj.name.split("_")[1]
        r = float(obj.horizontal_radius) + margin
        inside = (grid[:, 0] > x_min + r) & (grid[:, 0] < x_max - r) & (np.abs(grid[:, 1]) < y_half - r)
        free = np.array([
            inside[i]
            and all(float(np.linalg.norm(c - p)) > r + pr for p, pr in placed)
            and off_the_carry_line(c, r)
            for i, c in enumerate(grid)
        ], dtype=bool)
        if not free.any():
            print(f"  native {category}: no free spot on the worktop; left out", file=sys.stderr)
            continue
        if category in preferred:
            goal = preferred[category]
        else:
            goal = corners[extra % len(corners)]
            extra += 1
        candidates = grid[free]
        choice = candidates[int(np.argmin(np.linalg.norm(candidates - goal, axis=1)))]
        spots[obj.name] = choice
        placed.append((choice, r))
    return spots


def _spec_body(body, name: str):
    """Find a body by name anywhere under `body`, or None."""
    if body.name == name:
        return body
    for child in body.bodies:
        found = _spec_body(child, name)
        if found is not None:
            return found
    return None


def adopt_task_objects(spec: mujoco.MjSpec, objects: list, task, plate_world, worktop_z) -> None:
    """Rename the native objects' root bodies to the task's, and give them the task's physics.

    On the compiled-around spec, so the task's `stage(objects="engine")` finds
    `APPLE_BODY` and `PLATE_BODY` as top-level bodies and the arbiter reads them like
    its own. The registry meshes stay for looks; the physics is the task's, and both
    halves of that were forced by what the wire showed when the meshes were left alone:

    * **The apple's collision is one 20 mm sphere, not its hull pieces.** apple_10
      decomposes into 12 convex pieces, some 1 mm thin, and under the task's soft
      contact block they bounce: measured after a `/reset`, the apple was 2 cm off its
      spawn within half a second and its speed spiked to 0.4-1.1 m/s in bursts while
      nothing touched it, drifting 14 cm in 15 s -- so the preflight never saw it still.
      A sphere on a textured mesh is exactly how the task's own apple is built, and it
      is what the grasp tuning (`APPLE_CONTACT`, `grasp_gripper`) was measured against.
    * **The plate is static.** Free, it crept 1 cm across the counter in 12 s and the
      arm's start pose pressed into its rim by 4 mm; the task's own plate has no free
      joint either, and the free-joint topic publishes it with zero velocity just as
      the reference rig does. It is therefore placed here, on the spec, rather than by
      the post-compile qpos write the free objects get.
    """
    names = {"apple": task.APPLE_BODY, "plate": task.PLATE_BODY}
    for obj in objects:
        category = obj.name.split("_")[1]
        body = _spec_body(spec.worldbody, obj.root_body)
        if body is None:
            raise SystemExit(f"native {category}: root body {obj.root_body!r} not in the spec")
        # `task_bowl` and friends for the dressing: the same names the task's own
        # dressing carries, so the free-joint topic publishes them and the workspace
        # clearing leaves them alone.
        body.name = names.get(category, f"task_{category}")
        if category not in names:
            continue
        if category == "apple":
            stack = [body]
            while stack:
                b = stack.pop()
                stack.extend(b.bodies)
                for g in list(b.geoms):
                    if g.contype or g.conaffinity:
                        task.spec_delete(spec, g)  # this venv's MuJoCo differs from the other's
            sphere = body.add_geom(
                name=f"{task.APPLE_BODY}_geom",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[task.APPLE_RADIUS, 0.0, 0.0],
                mass=task.APPLE_MASS,
                rgba=[1.0, 0.0, 0.0, 0.0],
                group=0,  # collision; group 0 carries inertia here, see the task module
            )
            for key, value in task.APPLE_CONTACT.items():
                setattr(sphere, key, value)
        else:
            for joint in list(body.joints):
                task.spec_delete(spec, joint)
            body.pos = [
                float(plate_world[0]), float(plate_world[1]),
                float(worktop_z) - float(obj.bottom_offset[2]) + 0.002,
            ]


def object_layout(xy, yaw, out, reach, n: int) -> list[np.ndarray]:
    """Where to put `n` objects so an arm at `xy` facing `out` can reach all of them.

    Spread across an arc at the middle of the working annulus rather than in a line: the
    annulus is a ring, and objects strung out along the counter leave the ones at the ends
    reachable only with the arm fully extended.
    """
    radius = float(np.mean(reach))
    if n == 1:
        angles = [yaw]
    else:
        spread = np.radians(50.0)
        angles = np.linspace(yaw - spread, yaw + spread, n)
    return [xy + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]


def add_task_camera(spec: mujoco.MjSpec, eye, target, name: str = "task_camera") -> None:
    """A fixed scene camera at `eye` looking at `target`.

    MuJoCo cameras look along their frame's -z with +y up in the image, so the frame is
    built from the view direction: z away from the target, x level (perpendicular to
    both z and world-up), y completing the right-handed set.
    """
    eye = np.asarray(eye, dtype=np.float64)
    direction = np.asarray(target, dtype=np.float64) - eye
    z_cam = -direction / np.linalg.norm(direction)
    x_cam = np.cross([0.0, 0.0, 1.0], z_cam)
    norm = np.linalg.norm(x_cam)
    x_cam = np.array([1.0, 0.0, 0.0]) if norm < 1e-9 else x_cam / norm
    y_cam = np.cross(z_cam, x_cam)
    spec.worldbody.add_camera(
        name=name,
        pos=eye.tolist(),
        xyaxes=[*x_cam.tolist(), *y_cam.tolist()],
        fovy=50.0,
    )


def attach_robot(spec: mujoco.MjSpec, robot: str, prefix: str, pos, quat) -> None:
    """Graft a shared robot MJCF into the kitchen spec under `prefix`."""
    robot_spec = mujoco.MjSpec.from_file(str(model_xml(robot)))
    root = robot_spec.body("base")
    if root is None:
        raise SystemExit(f"no 'base' body in the shared {robot} spec")
    # The reference site MolmoSpaces' base group measures against. Harmless here, and it
    # keeps the two engines' compiled models the same shape.
    spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])
    spec.worldbody.add_frame(pos=list(pos), quat=list(quat)).attach_body(root, prefix, "")


class JointGroup:
    """One move group of an arm, straight off a raw MuJoCo model.

    The control protocol is defined by what it puts on the wire, not by MolmoSpaces'
    `RobotView`, so this exposes exactly what the shared arm ROS surface reads. Keeping
    it here rather than importing an engine's robot classes is the whole point of the
    split.
    """

    def __init__(self, model, data, prefix: str, joints, leaf_body: str | None = None,
                 frame_body: str | None = None):
        self._model, self._data = model, data
        self._qpos, self._qvel, self._act = [], [], []
        for name in joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{name}")
            if jid < 0 or aid < 0:
                raise SystemExit(f"joint/actuator {prefix}{name!r} missing from the model")
            self._qpos.append(int(model.jnt_qposadr[jid]))
            self._qvel.append(int(model.jnt_dofadr[jid]))
            self._act.append(aid)
        self._leaf = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{leaf_body}")
            if leaf_body
            else -1
        )
        self._frame = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{frame_body}")
            if frame_body
            else -1
        )

    @property
    def pose(self) -> np.ndarray:
        """The group's frame as a 4x4, for groups that have one."""
        pose = np.eye(4)
        if self._frame >= 0:
            pose[:3, :3] = self._data.xmat[self._frame].reshape(3, 3)
            pose[:3, 3] = self._data.xpos[self._frame]
        return pose

    @property
    def joint_pos(self) -> np.ndarray:
        return self._data.qpos[self._qpos].copy()

    @joint_pos.setter
    def joint_pos(self, value) -> None:
        self._data.qpos[self._qpos] = np.asarray(value, dtype=np.float64)

    @property
    def joint_vel(self) -> np.ndarray:
        return self._data.qvel[self._qvel].copy()

    @property
    def ctrl(self) -> np.ndarray:
        return self._data.ctrl[self._act].copy()

    @ctrl.setter
    def ctrl(self, value) -> None:
        self._data.ctrl[self._act] = np.asarray(value, dtype=np.float64)

    @property
    def tcp_pos(self) -> np.ndarray:
        return self._data.xpos[self._leaf].copy() if self._leaf >= 0 else np.zeros(3)


# `serve_control` lived here: a msgpack-numpy binary protocol on its own port, which was
# how the arm was driven before it moved onto ROS. Gone rather than deprecated -- two
# transports for one robot is two things to keep in step, and two ways for the engines to
# drift apart. See ../../shared/ros_surfaces/so101.py.


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


def warn_on_penetration(model, data, prefix: str, depth: float = -0.001) -> None:
    """Complain if the robot was placed inside a cabinet.

    A mobile base shoves itself free over the next few steps, which looks like the robot
    randomly driving off; an arm bolted to a counter just interpenetrates silently. Either
    way the spawn is what was wrong, so it is worth saying so at spawn time.
    """
    def is_robot(gid: int) -> bool:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
        return name.startswith(prefix)

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


class _Instance:
    """One robot in the kitchen. See the MolmoSpaces engine's twin for the reasoning.

    `mjcf` prefixes bodies, joints and actuators inside the compiled model; `ns` prefixes
    topics and services on the wire. They stay separate so an engine's model layout never
    reaches a client.
    """

    __slots__ = ("name", "mjcf", "ns", "holonomic", "tabletop",
                 "xy", "yaw", "out", "mount_z", "base", "groups")

    def __init__(self, name, mjcf, ns, holonomic, tabletop):
        self.name, self.mjcf, self.ns = name, mjcf, ns
        self.holonomic, self.tabletop = holonomic, tabletop
        self.xy = self.yaw = self.out = None
        self.mount_z = 0.0
        self.base, self.groups = None, {}

    def __repr__(self) -> str:
        return f"<{self.name} mjcf={self.mjcf!r} ns={self.ns!r}>"


def _surface_kwargs(args, inst, model, task, scene_option):
    """Everything one robot's surface needs, chosen by which contract it speaks.

    The twin of the MolmoSpaces engine's function of the same name. The two engines build
    the same two bags, which is what keeps their topic lists identical -- the property the
    whole multi-engine split exists to preserve.
    """
    prefix = inst.mjcf
    if inst.name in ARM_ROS_SURFACES:
        from ros_surfaces.so101 import DEFAULT_CAMERAS, WRIST_CAMERA

        cameras = dict(DEFAULT_CAMERAS)
        if args.wrist_camera:
            cameras.update({t: (f"{prefix}{n}", w, h) for t, (n, w, h) in WRIST_CAMERA.items()})
        return {
            "view": inst.groups, "model": model, "task": task, "cameras": cameras,
            "jpeg_quality": args.jpeg_quality, "control_hz": args.control_hz,
            "scene_option": scene_option,
        }

    camera = _pick_camera(args, model, prefix)
    scan_cfg = None
    if not args.no_scan:
        defaults = SCAN_DEFAULTS[inst.name]
        offset = args.scan_offset or defaults["offset"]
        scan_cfg = {
            "beams": args.scan_beams,
            "max_range": args.scan_range or defaults["max_range"],
            "min_range": args.scan_min_range or defaults["min_range"],
            "offset_x": offset[0],
            "offset_z": offset[1],
            "period": 1.0 / max(args.scan_hz, 1e-3),
            # Rays start inside the robot's own chassis and must not range it. Only its
            # own: another robot in the kitchen is something the lidar is meant to see.
            "body": f"{prefix}base",
            "exclude_bodies": frozenset(
                i for i in range(model.nbody)
                if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
                and n.startswith(prefix)
            ),
        }
    # Depth is not one of the four topics in the ROS contract and nothing in the
    # console reads it, but the MolmoSpaces engine publishes it -- and an engine a
    # client could tell apart by its topic list is the regression this split exists
    # to prevent. Same defaults, same flag to turn it off.
    depth_cfg = None
    if not args.no_depth and camera is not None:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        depth_cfg = {
            "size": args.depth_size,
            "period": 1.0 / max(args.depth_hz, 1e-3),
            "max_range": args.depth_range,
            "fovy": float(model.cam_fovy[cam_id]),
        }
    return {
        "base": inst.base, "model": model, "camera": camera,
        "camera_size": args.camera_size, "jpeg_quality": args.jpeg_quality,
        "control_hz": args.control_hz, "watchdog_s": args.watchdog,
        "scan": scan_cfg, "depth": depth_cfg, "scene_option": scene_option,
        "camera_period": (1.0 / args.camera_hz) if args.camera_hz > 0 else 0.0,
    }


def _pick_camera(args, model, prefix: str) -> str | None:
    """The MJCF camera a mobile base streams, resolved against its own prefix.

    `--camera` names one for every robot, which is only meaningful when there is one;
    with a fleet each base falls back to its own.
    """
    if args.camera is not None:
        if args.camera.lower() == "none":
            return None
        return args.camera
    for candidate in ("task_camera", f"{prefix}front_camera", f"{prefix}exo_camera",
                      f"{prefix}wrist_cam"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, candidate) >= 0:
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "robot",
        help=f"one of: {', '.join(ROBOTS)}. A comma-separated list spawns several into "
             "the same kitchen, sharing one --ros-port: `so101,myagv` mounts the arm on a "
             "worktop and puts the base on the floor. One ROS graph, a namespace per "
             "robot -- which is what a real multi-robot bringup is.",
    )
    ap.add_argument(
        "--ros-namespace", default=None, dest="ros_namespace",
        help="comma-separated ROS namespaces, positionally matched to the robots. "
             "Defaults to each robot's own name; pass an empty element for the bare, "
             "unnamespaced vendor contract.",
    )
    ap.add_argument("--layout", type=int, default=1, help="kitchen layout id (1-60)")
    ap.add_argument("--style", type=int, default=1, help="kitchen style id (1-60)")
    ap.add_argument("--seed", type=int, default=0, help="fixture-state RNG seed")
    ap.add_argument("--pos", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="override the spawn position, in metres")
    ap.add_argument("--yaw", type=float, default=None, help="override facing, in degrees")
    ap.add_argument("--objects", default=None, metavar="CAT,CAT",
                    help="RoboCasa object categories to put within the arm's reach, "
                         "e.g. 'bowl,apple'. See models/assets/objects/objaverse/")
    ap.add_argument("--object-scale", type=float, default=0.7, dest="object_scale",
                    help="scale for spawned objects (default %(default)s: sized for "
                         "the SO-101's ~7 cm gripper span rather than for realism)")
    ap.add_argument("--swap-objects", action="store_true", dest="swap_objects",
                    help="with --task: stage the plate at the apple's spawn and the apple "
                         "where the plate was. The console reads which layout is on the "
                         "wire off the apple's reset position, so nothing else is told.")
    ap.add_argument("--task-objects", action="store_false", dest="native_objects",
                    help="with --task: stage the task's own measured YCB apple and plate "
                         "instead of RoboCasa's. Default is RoboCasa's -- apple_10 and "
                         "plate_4 from its registry (NATIVE_MODELS says why those two), "
                         "scaled to the task's radii and with the task's contact block "
                         "on the apple.")
    ap.add_argument("--side-camera-mirror", action="store_true", dest="side_camera_mirror",
                    help="with --task: stage the side camera on the other side of the "
                         "worktop (reflected across the arm's x-z plane, looking +y). "
                         "In the swapped layout the reference side view has the plate "
                         "between it and the apple; from the other side the apple is "
                         "the near object.")
    ap.add_argument("--render", default=None, help="write a PNG instead of opening the viewer")
    ap.add_argument("--headless", action="store_true",
                    help="run the step and control loop with no window")
    ap.add_argument("--timeout", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--distance", type=float, default=None, help="camera distance for --render")
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=-20.0)

    ap.add_argument("--ros-port", type=int, default=None, dest="ros_port",
                    help="serve this robot's vendor ROS topics on PORT (mobile bases)")
    ap.add_argument(
        "--no-reference-table", action="store_false", dest="reference_table",
        help="stage the task's objects on the kitchen's own worktop instead of on the "
             "reference work surface.",
    )
    ap.add_argument("--no-dressing", action="store_false", dest="dressing",
                    help="stage only the apple and the plate, without the distractors.")
    ap.add_argument("--reference-lighting", action="store_true", dest="reference_lighting",
                    help="impose the reference scene's exposure on the kitchen. Off by "
                         "default: it matches the reference's photometry and costs the "
                         "VLA the task. See the MolmoSpaces engine's copy of this flag.")
    ap.add_argument("--extra-lights", action="store_true", dest="extra_lights",
                    help="also add the reference's two directional lamps (they blow out "
                         "a normally-lit kitchen; for a scene that renders too dark).")
    ap.add_argument(
        "--render-camera", default=None, dest="render_camera", metavar="NAME",
        help="with --render: look through this named MJCF camera at its own declared "
             "resolution, instead of the free camera.",
    )
    ap.add_argument(
        "--render-framing", action="store_true", dest="render_framing",
        help="with --render: report where the staged work surface's corners land in the "
             "frame, and what fraction of the frame is clipped to white.",
    )
    ap.add_argument("--control-host", "--host", default="0.0.0.0", dest="control_host",
                    help="interface the --ros-port server binds (default: all)")
    ap.add_argument(
        "--control-hz", type=float, default=20.0, dest="control_hz",
        help="control loop rate. Removed by accident with the --control* flags it was "
             "named after, which left three uses behind and made this engine exit in "
             "argparse before it ever built a kitchen.",
    )
    ap.add_argument(
        "--task", default=None, choices=sorted(TASKS),
        help="stage a task into the kitchen: its objects, its cameras and its success "
             "predicate. Requires --ros-port, which is what publishes the verdict.",
    )
    ap.add_argument(
        "--wrist-camera", action="store_true", dest="wrist_camera",
        help="also stream the eye-in-hand view (every camera costs control rate, so it "
             "is off unless asked for)",
    )
    ap.add_argument("--watchdog", type=float, default=0.5,
                    help="stop the base if no command arrives for this long")
    ap.add_argument("--camera", default=None, help="MJCF camera to stream, or 'none'")
    ap.add_argument("--camera-size", type=int, nargs=2, default=[640, 480], dest="camera_size")
    ap.add_argument("--jpeg-quality", type=int, default=70, dest="jpeg_quality")
    ap.add_argument("--scan-beams", type=int, default=360, dest="scan_beams")
    ap.add_argument("--scan-range", type=float, default=None, dest="scan_range")
    ap.add_argument("--scan-min-range", type=float, default=None, dest="scan_min_range")
    ap.add_argument("--scan-offset", type=float, nargs=2, default=None, metavar=("X", "Z"))
    ap.add_argument("--scan-hz", type=float, default=10.0, dest="scan_hz")
    ap.add_argument("--no-scan", action="store_true", dest="no_scan")
    ap.add_argument("--depth-hz", type=float, default=5.0, dest="depth_hz")
    ap.add_argument("--depth-size", type=int, nargs=2, default=[320, 240], dest="depth_size")
    ap.add_argument("--depth-range", type=float, default=8.0, dest="depth_range")
    ap.add_argument("--no-depth", action="store_true", dest="no_depth")
    ap.add_argument("--camera-hz", type=float, default=0.0, dest="camera_hz",
                    help="cap the colour camera's frame rate, independent of --control-hz. "
                         "0 (default) renders one frame per control tick. The render is "
                         "the dominant cost of a second camera-bearing robot in the same "
                         "physics loop; see the MolmoSpaces engine for the measurements.")
    args = ap.parse_args()

    if args.task and not args.ros_port and not args.render:
        raise SystemExit(
            "--task needs --ros-port: the task publishes its objects and cameras there, and "
            "staging one with nothing to publish it would silently score nothing."
        )


    names = [n.strip() for n in args.robot.split(",") if n.strip()]
    if not names:
        raise SystemExit("no robot named")
    unknown = [n for n in names if n not in ROBOTS]
    if unknown:
        raise SystemExit(f"unknown robot(s) {unknown}; available: {', '.join(ROBOTS)}")

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

    # `robot_N/` prefixes bodies inside the compiled model; the ROS namespace prefixes
    # topics on the wire. Different things, kept apart -- see the same note in the
    # MolmoSpaces engine and in shared/contracts/namespace.py.
    instances = [_Instance(name=n, mjcf=f"robot_{i}/", ns=namespaces[i],
                           holonomic=n in HOLONOMIC_BASE_ROBOTS,
                           tabletop=n in TABLETOP_ROBOTS)
                 for i, n in enumerate(names)]
    primary = instances[0]
    prefix = primary.mjcf
    arm_instance = next((i for i in instances if i.tabletop), primary)

    arena, compile_spec = build_kitchen_arena(args.layout, args.style, args.seed)

    # Compile the bare kitchen first: the placement search needs a model, and neither the
    # robot nor the objects may be in it or they would be their own nearest obstacles.
    spec = compile_spec()
    kitchen = spec.compile()
    kdata = mujoco.MjData(kitchen)
    mujoco.mj_forward(kitchen, kdata)

    # Arms first, then mobile bases. An arm has no say in where it goes -- it needs a
    # worktop at working height -- while a base can start anywhere open, so the base is
    # the one that gives way. Placed the other way round, the roomiest floor in a
    # RoboCasa kitchen is repeatedly the standing space in front of the very counter the
    # arm is about to be bolted to, and the task then stages a 0.92 m slab on top of it.
    keep_out: list[np.ndarray] = []
    # The arm's free worktop in its base frame, from `find_counter_mount`; None without an arm.
    arm_worktop = None

    for inst in sorted(instances, key=lambda i: not i.tabletop):
        if inst.tabletop:
            # No spawn margin for an arm: the margin is a driving allowance -- room to not be
            # touching a cabinet door you would then have to unstick yourself from -- and
            # adding it here would reject every 0.6 m-deep counter run in the dataset.
            xy, mount_z, yaw, out, arm_worktop = find_counter_mount(
                arena, kitchen, kdata, ROBOT_RADIUS[inst.name]
            )
            inst.mount_z = mount_z
            # The arm reaches out over the floor beside its counter and the task stages a
            # slab there; both are somewhere a base must not be. `clearance_field`
            # measures to box *surfaces*, so a half-extent box is the natural spelling.
            reach = ARM_REACH[1] + ROBOT_RADIUS[inst.name]
            keep_out.append(np.array([xy[0], xy[1], reach, reach], dtype=float))
        else:
            xy, yaw = find_open_floor(
                kitchen, kdata, ROBOT_RADIUS[inst.name] + SPAWN_MARGIN_M,
                keep_out=keep_out,
            )
            out = np.array([np.cos(yaw), np.sin(yaw)])
            r = ROBOT_RADIUS[inst.name] + SPAWN_MARGIN_M
            keep_out.append(np.array([xy[0], xy[1], r, r], dtype=float))
        if args.pos is not None:
            xy = np.array(args.pos, dtype=float)
        if args.yaw is not None:
            yaw = np.radians(args.yaw)
            out = np.array([np.cos(yaw), np.sin(yaw)])
        inst.xy, inst.yaw, inst.out = xy, yaw, out

    holonomic = primary.holonomic
    xy, yaw, out, mount_z = primary.xy, primary.yaw, primary.out, primary.mount_z

    # Objects go in before the compile, and their poses are written after it: a free body
    # is placed by its joint, which does not exist until the model is built.
    if args.objects and args.task:
        raise SystemExit(
            "--objects and --task are mutually exclusive. A task stages its own objects at "
            "measured positions; --objects samples RoboCasa's registry and drops them on an "
            "arc through the arm's reach, which for apple_on_plate means a second apple the "
            "jaw cannot close on and a bowl inside the plate's footprint."
        )
    categories = [c.strip() for c in (args.objects or "").split(",") if c.strip()]
    objects = make_kitchen_objects(categories, args.seed, args.object_scale) if categories else []
    # A task served on RoboCasa's own apple and plate: the registry models go into the
    # kitchen like any --objects pair, then take the task's body names so the task stages
    # around them rather than bringing its own. Placed at the task's poses below.
    task_objects: list = []
    task_module = None
    if args.task and args.native_objects:
        import importlib

        task_module = importlib.import_module(TASKS[args.task][0])
        task_objects = make_task_objects(
            task_module, swap=args.swap_objects, dressing=args.dressing, seed=args.seed,
        )
        objects = list(task_objects)
    if objects:
        spec = compile_spec(objects)
    if task_objects:
        # The task's poses in the arm base frame, mapped into the kitchen with the same
        # transform `stage()` will use for everything else it places. The plate goes in
        # here, static; the apple keeps its free joint and is placed after the compile.
        task_transform = task_module.base_frame(
            [float(arm_instance.xy[0]), float(arm_instance.xy[1]), arm_instance.mount_z],
            float(arm_instance.yaw),
        )
        task_poses = task_module.object_poses(args.swap_objects)
        adopt_task_objects(
            spec, task_objects, task_module,
            plate_world=task_module._apply(task_transform, task_poses["plate"]),
            worktop_z=mount_z - SO101_BASE_LIFT,
        )

    if args.robot in TABLETOP_ROBOTS:
        # The SO-101's own exo_camera sits behind-left of its base, which on a counter
        # mounted against a wall puts it inside the wall cabinets -- the streamed frame
        # is the black inside of a cupboard. A kitchen counter is not the tabletop that
        # camera was framed for, so add a scene-level camera and prefer it below.
        #
        # Steeply overhead rather than from the front, and that angle was bought with
        # failed episodes: a policy watching from the front cannot see alignment along
        # the camera's depth axis, and a base rotation projected into a front view is
        # not even monotonic -- an LLM agent "verified" the joint-0 image direction
        # early, was right, and was then betrayed by the same rule at a larger angle.
        # Looking down turns lateral alignment into something directly visible.
        # Skipped under --task: the task stages `overhead` and `side` at the poses a
        # policy was calibrated against, and they are what the ROS surface publishes.
        # A third camera here would only make the two engines' compiled models differ.
        centre = xy + out * 0.25
        if args.task:
            pass
        else:
            add_task_camera(
                spec,
                eye=[float(centre[0] + out[0] * 0.22),
                     float(centre[1] + out[1] * 0.22), mount_z + 0.85],
                target=[float(centre[0]), float(centre[1]), mount_z],
            )

    for inst in instances:
        inst_quat = [float(np.cos(inst.yaw / 2)), 0.0, 0.0, float(np.sin(inst.yaw / 2))]
        attach_robot(
            spec,
            inst.name,
            inst.mjcf,
            # A holonomic base is grafted in at the origin and driven to its spawn pose
            # below; its slide joints are world-aligned and mean nothing anywhere else.
            pos=([0.0, 0.0, 0.0] if inst.holonomic
                 else [float(inst.xy[0]), float(inst.xy[1]), inst.mount_z]),
            quat=[1.0, 0.0, 0.0, 0.0] if inst.holonomic else inst_quat,
        )

    stage_task = None
    if args.task:
        import importlib

        module_name, stage_name, arbiter_name = TASKS[args.task]
        task_module = importlib.import_module(module_name)
        stage_task = (getattr(task_module, stage_name), getattr(task_module, arbiter_name))
        # The arm's base body sits exactly at the worktop here -- this engine bolts it
        # straight into the worldbody with no riser -- so that pose is the task's frame
        # origin, with the work surface at z = 0 just as the geometry assumes.
        # The task's frame origin is the **arm base**, not the worktop -- that is the
        # frame the arbiter reports poses in (it finds the `base` body) and the frame the
        # console does its kinematics in. The work surface is at z = 0 of that frame
        # because the task *stages* it there, so handing this the worktop instead puts
        # the two 4 mm apart: measured, a resting apple read 0.0158 here against
        # MolmoSpaces' 0.0204, which is a silent shift of the success gate between
        # engines. The slab then sinks SO101_BASE_LIFT deeper into the counter, which
        # costs nothing -- it is static.
        stage_task[0](
            spec,
            [float(arm_instance.xy[0]), float(arm_instance.xy[1]), arm_instance.mount_z],
            float(arm_instance.yaw),
            reference_table=args.reference_table,
            # With native objects the dressing is RoboCasa's too (NATIVE_DRESSING),
            # spawned above; the task's YCB set would be a second bowl inside the first.
            dressing=args.dressing and not task_objects,
            lighting=args.reference_lighting,
            extra_lights=args.extra_lights,
            swap=args.swap_objects,
            objects="engine" if task_objects else "task",
            side_camera_mirror=args.side_camera_mirror,
        )

    model = spec.compile()
    data = mujoco.MjData(model)

    if task_objects:
        # Everything but the plate, which was made static and placed on the spec in
        # `adopt_task_objects` and so has no free joint to write. The apple goes to the
        # task's pose; the dressing wherever the placer found room.
        laid_out = layout_native_dressing(
            [o for o in task_objects if o.name.split("_")[1] not in ("apple", "plate")],
            task_poses, arm_worktop, task_module,
        )
        placeable, spots = [], []
        for o in task_objects:
            category = o.name.split("_")[1]
            if category == "apple":
                local = task_poses["apple"]
            elif o.name in laid_out:
                local = (float(laid_out[o.name][0]), float(laid_out[o.name][1]), 0.0)
            else:
                continue
            placeable.append(o)
            spots.append(np.asarray(task_module._apply(task_transform, local)[:2]))
    else:
        placeable = objects
        spots = object_layout(xy, yaw, out, ARM_REACH, len(objects))

    for obj, spot in zip(placeable, spots):
        # `bottom_offset` is how far the object's origin sits above its lowest point, so
        # subtracting it is what rests the object on the worktop instead of half in it.
        z = mount_z - SO101_BASE_LIFT - float(obj.bottom_offset[2]) + 0.002
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, obj.joints[0])
        if joint < 0:
            print(f"warning: no free joint for {obj.name}", file=sys.stderr)
            continue
        adr = model.jnt_qposadr[joint]
        data.qpos[adr:adr + 7] = [spot[0], spot[1], z, 1.0, 0.0, 0.0, 0.0]
        print(f"  placed {obj.name} at ({spot[0]:.2f}, {spot[1]:.2f}, {z:.2f})", file=sys.stderr)

    for inst in instances:
        if inst.holonomic:
            inst.base = PlanarJointBase(model, data, inst.mjcf)
            inst.base.teleport(float(inst.xy[0]), float(inst.xy[1]), float(inst.yaw))
        else:
            inst.groups = {
                # An empty group, and deliberately so. MolmoSpaces gives its SO-101 a `base`
                # move group -- the unactuated mocap mount the arm is bolted to, zero controls,
                # see robots/so101/so101_view.py:47 -- and advertises the mount pose as
                # `base_pose`. Here the arm is bolted straight into the worldbody instead, so
                # there is no mount body, but the *information* is the same and the client must
                # not be able to tell: a console that saw `base` on one engine and not the
                # other could identify which one it was talking to.
                "base": JointGroup(model, data, inst.mjcf, (), frame_body="base"),
                "arm": JointGroup(model, data, inst.mjcf, SO101_ARM_JOINTS, SO101_TCP_BODY),
                "gripper": JointGroup(model, data, inst.mjcf, SO101_GRIPPER_JOINTS),
            }
            # Both the joint state and the target: these are position actuators, so a ctrl
            # left at its default of 0 would make the arm snap out of the rest pose on step 1.
            for gid, rest in (("arm", SO101_REST_QPOS), ("gripper", SO101_REST_GRIPPER)):
                inst.groups[gid].joint_pos = rest
                inst.groups[gid].ctrl = rest

    base, groups = primary.base, primary.groups

    mujoco.mj_forward(model, data)
    for inst in instances:
        warn_on_penetration(model, data, inst.mjcf)

    task = None
    if stage_task is not None:
        # After the rest pose is applied: the arbiter snapshots this state as the one
        # /reset restores. The arm's prefix is passed rather than inferred -- every
        # robot's root body is called `base`, so the arbiter's "find the one body ending
        # in /base" rule resolves nothing once a second robot is in the kitchen.
        task = stage_task[1](model, data, prefix=arm_instance.mjcf)
        placed, reason = task.instantaneous(data)
        print(f"task {args.task}: staged; success predicate reads "
              f"{'TRUE (!)' if placed else reason} at spawn", file=sys.stderr)
        print(f"task {args.task}: {task.reach_report(data, ARM_REACH)}", file=sys.stderr)
        check_task_contacts(model, arm_instance.mjcf, task)
        from mujoco_bridge import report_slab_fit
        report_slab_fit(model, data)
    print(
        f"{args.robot} in kitchen: {model.nbody} bodies, {model.ngeom} geoms, "
        f"{model.nu} actuators",
        file=sys.stderr,
    )

    # Camera selection moved into `_pick_camera`, because it is per robot: each base
    # resolves `front_camera` against its own MJCF prefix, and one `camera` variable here
    # would have handed the second robot the first one's view. `task_camera` still comes
    # first there, for the same reason it did here.

    scene_option = visual_only()

    if args.render:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = [xy[0], xy[1], mount_z + 0.4]
        cam.distance = args.distance if args.distance is not None else 4.0
        cam.azimuth = args.azimuth if args.azimuth is not None else np.degrees(yaw) + 180.0
        cam.elevation = args.elevation
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
                renderer.update_scene(data, camera=args.render_camera, scene_option=scene_option)
                pixels = renderer.render()
        else:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, cam)
            cam.lookat[:] = [xy[0], xy[1], mount_z + 0.4]
            cam.distance = args.distance if args.distance is not None else 4.0
            cam.azimuth = args.azimuth if args.azimuth is not None else np.degrees(yaw) + 180.0
            cam.elevation = args.elevation
            with mujoco.Renderer(model, args.height, args.width) as renderer:
                renderer.update_scene(data, camera=cam, scene_option=scene_option)
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

    controller = None
    if args.ros_port:
        missing = [i.name for i in instances if i.name not in ROS_SURFACES]
        if missing:
            raise SystemExit(
                f"--ros-port: no ROS surface for {missing}; "
                f"available: {', '.join(sorted(ROS_SURFACES))}"
            )

        # One server, one port, one graph -- a namespace per robot. Identical to the
        # MolmoSpaces engine on purpose: a client must not be able to tell which engine
        # it is talking to, and the topic list is the first thing it would notice.
        import importlib

        from ros_surfaces import RobotFleet

        fleet = RobotFleet(port=args.ros_port, host=args.control_host)
        for inst in instances:
            module_name, func_name = ROS_SURFACES[inst.name]
            attach_ros = getattr(importlib.import_module(module_name), func_name)
            fleet.attach(inst.ns, attach_ros,
                         **_surface_kwargs(args, inst, model, task, scene_option))
        fleet.start()
        controller = fleet

    control_period = 1.0 / args.control_hz
    next_control = 0.0
    deadline = None if args.timeout is None else time.monotonic() + args.timeout

    from mujoco_bridge import run_sim_loop

    try:
        if args.headless:
            # No window: what a displayless host and an automated check run.
            run_sim_loop(model, data, controller, control_hz=args.control_hz,
                         deadline=deadline, label="headless loop")
        else:
            # Bound as a separate name: `import mujoco.viewer` here would shadow the
            # module-level `mujoco` with a function-local.
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(model, data) as viewer:
                # Without this the window is a kitchen full of RoboCasa's translucent
                # collision hulls: its geom groups are inverted, see `visual_only()`.
                viewer.opt.geomgroup[:] = scene_option.geomgroup
                viewer.cam.lookat[:] = [xy[0], xy[1], mount_z + 0.4]
                viewer.cam.distance = args.distance if args.distance is not None else 4.0
                viewer.cam.azimuth = (
                    args.azimuth if args.azimuth is not None else np.degrees(yaw) + 180.0
                )
                viewer.cam.elevation = args.elevation
                run_sim_loop(model, data, controller, control_hz=args.control_hz,
                             deadline=deadline, viewer=viewer, label="viewer loop")
    finally:
        if controller is not None:
            controller(None)  # close
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
