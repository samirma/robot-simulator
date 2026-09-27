#!/usr/bin/env python
"""Spawn robots into a RoboCasa kitchen and render, view or serve them.

    python   tools/spawn_robot.py so101,myagv --layout 1 --style 1 --headless
    mjpython tools/spawn_robot.py ainex --ros-port 0
    python   tools/spawn_robot.py myagv --render /tmp/kitchen.png

The tool itself -- command line, placement, task staging, start-up reports, the ROS
fleet, rendering and the loop -- is `simulator/shared/spawn.py`, the same for both
engines, so `robot_console` cannot tell which one it is driving. What is here is only
what RoboCasa knows and the other engine does not.

**RoboCasa is used as a scene provider, not as a robot stack.** The kitchen is built
straight from `KitchenArena` with an empty robot list, which yields a kitchen of fixtures
with zero actuators; the shared robot models are grafted into that spec and the whole
thing is stepped by plain MuJoCo. Going through `robosuite.make` would drag in a robosuite
robot (its controller stack, action space and observation dict) and put a Panda in the
middle of every map.

The RoboCasa-specific traps:

* **Geom groups are inverted from the MolmoSpaces convention.** RoboCasa puts collision
  hulls in group 0 (painted in random semi-transparent colours) and the visual meshes in
  group 1, so everything that renders here goes through `visual_only()`.
* **Clearance has to be measured to geom surfaces, not geom centres.** A kitchen is four
  long wall boxes and a run of counters; the centre of a 5 m wall is metres away from a
  robot pressed against it. The floor search uses world-space AABBs.
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

# The height band a driving robot sweeps through. The floor sits at z=0 and RoboCasa
# hangs wall cabinets from about 1.4 m, so anything between counts as in the way.
FLOOR_BAND = (0.02, 1.3)
#: Room a floor robot keeps from the nearest fixture: what keeps a spawn from touching a
#: cabinet door it would then have to unstick itself from.
SPAWN_MARGIN_M = 0.12
ARM_REACH = placement.ARM_REACH


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
            + "); try another --layout"
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
    """Return (region name, xy, top z, yaw) for a robot at the back of the roomiest worktop.

    At the *back* on purpose. The arm's working annulus starts 0.15 m out, so an arm in
    the middle of a 0.6 m counter can only reach the front lip and the empty air past it;
    put it against the wall and the whole depth of the counter is inside its reach.
    """
    regions = counter_regions(arena)
    if not regions:
        raise SystemExit(
            "this kitchen has no counter with a free worktop region, and a worktop robot "
            "in a scene with no worktop is a start-up error. Try another --layout."
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
    return region["name"], xy, float(region["top_z"]), yaw



def robot_spec(robot: str):
    """The shared description of one robot, and the name of its root body.

    Two robots are a `model.xml` on disk. The AiNex is not: what its vendor ships is a
    URDF that needs a documented set of corrections before it is a usable model, and
    those are measurements taken off compiled models, so they live as a builder in
    `shared/ainex_model.py` rather than as a file that could drift from the code that
    wrote it. Either way the description is shared, which is the part that matters --
    both engines compile the same robot.
    """
    if robot == "ainex":
        import ainex_model

        return ainex_model.build_spec(), ainex_model.robot_model_root_name()
    return mujoco.MjSpec.from_file(str(robots_spec.model_xml(robot))), "base"


def attach_robot(spec: mujoco.MjSpec, robot: str, prefix: str, pos, quat) -> None:
    """Graft a shared robot into the kitchen spec under `prefix`."""
    robot_spec_, root_name = robot_spec(robot)
    root = robot_spec_.body(root_name)
    if root is None:
        raise SystemExit(f"no {root_name!r} body in the shared {robot} spec")
    # The reference site MolmoSpaces' base group measures against. Harmless here, and it
    # keeps the two engines' compiled models the same shape.
    spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])
    spec.worldbody.add_frame(pos=list(pos), quat=list(quat)).attach_body(root, prefix, "")



class RoboCasaEngine:
    """RoboCasa's half of `shared/spawn.py`: its kitchens, its counters, its graft."""

    name = "robocasa"

    @staticmethod
    def add_scene_args(ap) -> None:
        ap.add_argument("--layout", type=int, default=1, help="kitchen layout id (1-60)")
        ap.add_argument("--style", type=int, default=1, help="kitchen style id (1-60)")
        ap.add_argument("--seed", type=int, default=0, help="fixture-state RNG seed")

    @staticmethod
    def load_scene(args):
        # No RoboCasa object sampling: the task brings its own objects, and the
        # sampler's would be a second apple the jaw cannot close on.
        arena, compile_spec = build_kitchen_arena(args.layout, args.style, args.seed)
        spec = compile_spec()
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return spawn.scene(spec, model, data, arena=arena)

    @staticmethod
    def find_worktop(scene, robot: str) -> placement.Worktop:
        name, xy, top_z, yaw = find_counter_mount(scene.arena, scene.model, scene.data,
                                                  placement.ROBOT_RADIUS[robot])
        return placement.Worktop(name=name, xy=np.asarray(xy, dtype=float), yaw=yaw, z=top_z)

    @staticmethod
    def floor_spot(scene, inst, keep_out):
        # Keep-outs are more obstacles to `clearance_field`, which measures to box
        # surfaces: a circle is the box of its radius.
        boxes = [np.array([xy[0], xy[1], r, r], dtype=float) for xy, r in keep_out]
        return find_open_floor(scene.model, scene.data, inst.radius + SPAWN_MARGIN_M,
                               keep_out=boxes)

    @staticmethod
    def attach(scene, inst) -> None:
        if inst.holonomic:
            # World-aligned slide joints: grafted over the origin and teleported once
            # compiled. Its z is the surface it stands on plus, for a legged robot, the
            # ride height measured off its compiled model.
            z = inst.surface_z
            if inst.name == "ainex":
                import ainex_model

                z += float(ainex_model.ride_height(robot_spec(inst.name)[0]))
            pos, quat = [0.0, 0.0, z], [1.0, 0.0, 0.0, 0.0]
        else:
            pos = [float(inst.xy[0]), float(inst.xy[1]), inst.mount_z]
            quat = [float(np.cos(inst.yaw / 2)), 0.0, 0.0, float(np.sin(inst.yaw / 2))]
        attach_robot(scene.spec, inst.name, inst.mjcf, pos, quat)

    @staticmethod
    def scene_option():
        return visual_only()


ENGINE = RoboCasaEngine()


def main(argv=None) -> int:
    return spawn.main(ENGINE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
