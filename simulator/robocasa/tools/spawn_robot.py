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
    "myagv": ("ros_surfaces.myagv", "serve_ros"),
    # The same shared surface the other engine uses. That is the point: the arm's topic
    # set belongs to the arm, so a client cannot tell which engine is hosting it.
    "so101": ("ros_surfaces.so101", "serve_ros"),
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


def find_open_floor(model, data, radius: float, step: float = 0.05):
    """Return (xy, yaw) for the roomiest patch of floor, facing the middle of the room.

    Facing the room centre rather than a fixed heading matters for a robot that starts by
    looking at what is in front of it: spawned nose-first into a cabinet, the first thing
    an explorer sees is 30 cm of door.
    """
    boxes = world_boxes(model, data, FLOOR_BAND)
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
            f"no floor patch in this kitchen clears {radius:.2f} m (best {best_clear:.2f} m); "
            "try another --layout, or place the robot by hand with --pos/--yaw"
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
    return xy, region["top_z"] + SO101_BASE_LIFT, yaw, out


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", choices=ROBOTS)
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
    ap.add_argument("--no-reference-lighting", action="store_false", dest="reference_lighting",
                    help="leave the scene's own lighting alone.")
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
    args = ap.parse_args()

    if args.task and not args.ros_port and not args.render:
        raise SystemExit(
            "--task needs --ros-port: the task's verdict goes out on /task_success, and "
            "staging one with nothing to publish it would silently score nothing."
        )


    prefix = "robot_0/"
    arena, compile_spec = build_kitchen_arena(args.layout, args.style, args.seed)

    # Compile the bare kitchen first: the placement search needs a model, and neither the
    # robot nor the objects may be in it or they would be their own nearest obstacles.
    spec = compile_spec()
    kitchen = spec.compile()
    kdata = mujoco.MjData(kitchen)
    mujoco.mj_forward(kitchen, kdata)

    holonomic = args.robot in HOLONOMIC_BASE_ROBOTS
    mount_z = 0.0
    out = np.array([1.0, 0.0])
    if args.robot in TABLETOP_ROBOTS:
        # No spawn margin for an arm: the margin is a driving allowance -- room to not be
        # touching a cabinet door you would then have to unstick yourself from -- and
        # adding it here would reject every 0.6 m-deep counter run in the dataset.
        xy, mount_z, yaw, out = find_counter_mount(arena, kitchen, kdata, ROBOT_RADIUS[args.robot])
    else:
        xy, yaw = find_open_floor(kitchen, kdata, ROBOT_RADIUS[args.robot] + SPAWN_MARGIN_M)
    if args.pos is not None:
        xy = np.array(args.pos, dtype=float)
    if args.yaw is not None:
        yaw = np.radians(args.yaw)
        out = np.array([np.cos(yaw), np.sin(yaw)])

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
    if objects:
        spec = compile_spec(objects)

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

    quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
    attach_robot(
        spec,
        args.robot,
        prefix,
        # A holonomic base is grafted in at the origin and driven to its spawn pose
        # below; its slide joints are world-aligned and mean nothing anywhere else.
        pos=[0.0, 0.0, 0.0] if holonomic else [float(xy[0]), float(xy[1]), mount_z],
        quat=[1.0, 0.0, 0.0, 0.0] if holonomic else quat,
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
            [float(xy[0]), float(xy[1]), mount_z],
            float(yaw),
            reference_table=args.reference_table,
            dressing=args.dressing,
            lighting=args.reference_lighting,
            extra_lights=args.extra_lights,
        )

    model = spec.compile()
    data = mujoco.MjData(model)

    for obj, spot in zip(objects, object_layout(xy, yaw, out, ARM_REACH, len(objects))):
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

    base = None
    groups = {}
    if holonomic:
        base = PlanarJointBase(model, data, prefix)
        base.teleport(float(xy[0]), float(xy[1]), float(yaw))
    else:
        groups = {
            # An empty group, and deliberately so. MolmoSpaces gives its SO-101 a `base`
            # move group -- the unactuated mocap mount the arm is bolted to, zero controls,
            # see robots/so101/so101_view.py:47 -- and advertises the mount pose as
            # `base_pose`. Here the arm is bolted straight into the worldbody instead, so
            # there is no mount body, but the *information* is the same and the client must
            # not be able to tell: a console that saw `base` on one engine and not the
            # other could identify which one it was talking to.
            "base": JointGroup(model, data, prefix, (), frame_body="base"),
            "arm": JointGroup(model, data, prefix, SO101_ARM_JOINTS, SO101_TCP_BODY),
            "gripper": JointGroup(model, data, prefix, SO101_GRIPPER_JOINTS),
        }
        # Both the joint state and the target: these are position actuators, so a ctrl
        # left at its default of 0 would make the arm snap out of the rest pose on step 1.
        for gid, rest in (("arm", SO101_REST_QPOS), ("gripper", SO101_REST_GRIPPER)):
            groups[gid].joint_pos = rest
            groups[gid].ctrl = rest

    mujoco.mj_forward(model, data)
    warn_on_penetration(model, data, prefix)

    task = None
    if stage_task is not None:
        # After the rest pose is applied: the arbiter snapshots this state as the one
        # /reset restores.
        task = stage_task[1](model, data)
        placed, reason = task.instantaneous(data)
        print(f"task {args.task}: staged; success predicate reads "
              f"{'TRUE (!)' if placed else reason} at spawn", file=sys.stderr)
        check_task_contacts(model, prefix, task)
        from mujoco_bridge import report_slab_fit
        report_slab_fit(model, data)
    print(
        f"{args.robot} in kitchen: {model.nbody} bodies, {model.ngeom} geoms, "
        f"{model.nu} actuators",
        file=sys.stderr,
    )

    camera = args.camera
    if camera is None:
        # task_camera first: it exists only when this script added it (tabletop robots),
        # and it is the one guaranteed to see the workspace rather than a cupboard.
        camera = next(
            (
                name
                for name in (
                    "task_camera",
                    f"{prefix}front_camera",
                    f"{prefix}exo_camera",
                    f"{prefix}wrist_cam",
                )
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
            ),
            None,
        )
    elif camera.lower() == "none":
        camera = None

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
    if args.ros_port and args.robot in ARM_ROS_SURFACES:
        import importlib

        from ros_surfaces.so101 import DEFAULT_CAMERAS, WRIST_CAMERA

        module_name, func_name = ROS_SURFACES[args.robot]
        serve_ros = getattr(importlib.import_module(module_name), func_name)
        cameras = dict(DEFAULT_CAMERAS)
        if args.wrist_camera:
            cameras.update({t: (f"{prefix}{n}", w, h) for t, (n, w, h) in WRIST_CAMERA.items()})
        controller = serve_ros(
            args.ros_port, groups, model, task,
            cameras=cameras,
            jpeg_quality=args.jpeg_quality,
            control_hz=args.control_hz,
            scene_option=scene_option,
            host=args.control_host,
        )
    elif args.ros_port:
        if args.robot not in ROS_SURFACES:
            raise SystemExit(
                f"--ros-port: no ROS surface for {args.robot!r}; "
                f"available: {', '.join(sorted(ROS_SURFACES))}"
            )
        import importlib

        module_name, func_name = ROS_SURFACES[args.robot]
        serve_ros = getattr(importlib.import_module(module_name), func_name)

        scan_cfg = None
        if not args.no_scan:
            defaults = SCAN_DEFAULTS[args.robot]
            offset = args.scan_offset or defaults["offset"]
            scan_cfg = {
                "beams": args.scan_beams,
                "max_range": args.scan_range or defaults["max_range"],
                "min_range": args.scan_min_range or defaults["min_range"],
                "offset_x": offset[0],
                "offset_z": offset[1],
                "period": 1.0 / max(args.scan_hz, 1e-3),
                # Rays start inside the robot's own chassis and must not range it.
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
        controller = serve_ros(
            args.ros_port, base, model, camera, args.camera_size, args.jpeg_quality,
            args.control_hz, args.watchdog, scan=scan_cfg, depth=depth_cfg,
            scene_option=scene_option,
        )
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
