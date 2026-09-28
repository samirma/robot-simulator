"""Where each robot of a fleet stands: on the worktop or on the floor, never in another.

Spec §4: one function stands a robot on the floor or the worktop. It is `stand` below,
and both engines' spawn tools reach it through `stand_fleet`. What an engine supplies is
only what its scene knows and this module cannot: which surface is the worktop and where
the task robot is bolted to it (`Engine.find_worktop`), and where there is open floor
(`Engine.floor_spot`). Everything that decides *around* those -- the order robots are
placed in, how far apart they stand, what a second worktop robot stands beside, how high
each robot's base sits over what it stands on -- is here, once.

The worktop is read by ray-casting the bare compiled scene (`SurfaceMap`), which is the
same question on every engine: is there a top face at the worktop's height under this
point, and room above it. RoboCasa's counter regions and MolmoSpaces' support rectangles
are each engine's way of *finding* the worktop; once found, it is just geometry.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

#: Every mobile robot (`robots_spec.mobile`: any kind but a fixed arm) is driven on three
#: world-aligned virtual joints (x, y, yaw): grafted over the origin and teleported to its
#: stand. The AiNex's torso rides the same three joints; its gait is animated over them
#: (see `ros_surfaces/ainex/gait.py`).

#: Footprint radius each robot needs kept clear around its base, metres. The myAGV's
#: chassis is 311 x 230 mm (half-diagonal 0.193 m). The SO-101's is the arm's own sweep
#: at rest over its base. The AiNex's is a margin over the 0.1711 m footprint measured off
#: the compiled model at its init pose (`shared/ainex_model.py`). The myAGV + myCobot 280
#: stands on the myAGV's chassis with its arm held upright over the deck, so its footprint
#: is the myAGV's; the ROSMASTER X3 PLUS's is its arm's reach at the pose its driver
#: powers up in, 0.262 m forward of its base, measured off the compiled model (its
#: 300 x 245.6 mm chassis alone would be 0.195 m).
ROBOT_RADIUS = {"myagv": 0.193, "so101": 0.20, "ainex": 0.19, "myagv_mycobot280": 0.193,
                "rosmaster_x3_plus": 0.265}
#: The radius of what a robot stands on -- its base plate, its feet, its wheels -- which
#: has to have its surface under all of it. Measured off the compiled models: the SO-101's
#: base body's half-diagonal, the AiNex's two soles at its init pose, the myAGV's chassis.
SUPPORT_RADIUS = {"so101": 0.10, "ainex": 0.115, "myagv": 0.193,
                  "myagv_mycobot280": 0.193, "rosmaster_x3_plus": 0.15}
#: How tall each robot stands over what it stands on: the headroom a worktop spot needs.
#: The AiNex's is its measured 0.4027 m, sole to crown, and a margin. The composite's is
#: its upright arm's top over the floor; the X3 PLUS's its published 515 mm.
ROBOT_HEIGHT = {"myagv": 0.30, "so101": 0.45, "ainex": 0.41, "myagv_mycobot280": 0.72,
                "rosmaster_x3_plus": 0.52}
#: How far above its surface a fixed robot's base body is grafted. The SO-101's base
#: plate meshes hang 2.4 mm below its body origin (the `pos` on base_motor_holder_so101_v1
#: in the shared MJCF), so an origin exactly on the surface buries the plate in it.
BASE_CLEARANCE = {"so101": 0.004}
#: The SO-101's working annulus on its worktop, which the engines' mount searches aim
#: the worktop into: short of the full ~0.4 m, where no usable orientation is left.
ARM_REACH = (0.15, 0.35)
#: Room between two robots' footprints, and between a robot and a staged object's.
GAP = 0.02
#: Room a floor robot leaves around another floor robot, for driving off without
#: scraping it.
FLOOR_MARGIN = 0.12
#: What a worktop robot claims of the floor beside its counter: its footprint and its
#: reach. Anything that drives must stay out of both.
def floor_keep_out_radius(name: str) -> float:
    return ROBOT_RADIUS[name] + ARM_REACH[1]


@dataclass
class Instance:
    """One robot in the scene.

    Two namespaces, deliberately kept apart. `mjcf` (`robot_0/`) prefixes bodies, joints,
    actuators and cameras inside the compiled model, so two robots coexist without a name
    collision; `ns` (`so101`) prefixes topics and services on the wire. Conflating them
    would put an engine's model layout onto the wire, where a client could see it.
    """

    name: str
    mjcf: str
    ns: str
    #: Where it stands, set by `stand`: xy, facing, the top face it stands on, and the
    #: height its base body is grafted at over that face.
    xy: np.ndarray | None = None
    yaw: float = 0.0
    surface_z: float = 0.0
    mount_z: float = 0.0
    on: str = ""
    #: Engine-neutral handles, bound after the compile: the arm's move groups, or the
    #: planar base of a holonomic robot.
    view: dict | None = None
    base: object = None
    #: Anything an engine keeps per robot (MolmoSpaces' robot config and class).
    engine: dict = field(default_factory=dict)

    @property
    def holonomic(self) -> bool:
        """Driven on the planar joints: every mobile robot in robots.yml."""
        import robots_spec

        return robots_spec.mobile(self.name)

    @property
    def radius(self) -> float:
        return ROBOT_RADIUS[self.name]

    def __repr__(self) -> str:
        return f"<{self.name} mjcf={self.mjcf!r} ns={self.ns!r}>"


@dataclass
class Worktop:
    """The surface the task is staged on, as an engine found it.

    `xy`/`yaw` are where the task robot stands and which way it faces, chosen by the
    engine's own search so the worktop is in front of it; `z` is the top face.
    """

    name: str
    xy: np.ndarray
    yaw: float
    z: float


def stand(inst: Instance, on: str, xy, yaw: float, surface_z: float) -> None:
    """Stand one robot at `xy`, facing `yaw`, on the floor or the worktop at `surface_z`.

    The one place a robot's height over its surface is decided: a fixed robot's base body
    at `surface_z` plus its own base clearance, a holonomic one grafted at the surface
    (its engine adds the legged robot's ride height, measured off its compiled model).
    """
    if on not in ("floor", "worktop"):
        raise ValueError(f"a robot stands on the floor or the worktop, not {on!r}")
    inst.on = on
    inst.xy = np.asarray(xy, dtype=float)[:2].copy()
    inst.yaw = float(yaw)
    inst.surface_z = float(surface_z)
    inst.mount_z = float(surface_z) + BASE_CLEARANCE.get(inst.name, 0.0)
    print(f"{inst.name}: stands on the {on} at ({inst.xy[0]:.2f}, {inst.xy[1]:.2f}, "
          f"{inst.surface_z:.3f}), facing {math.degrees(inst.yaw):.0f} deg", file=sys.stderr)


# ---------------------------------------------------------------- the worktop, as geometry


class SurfaceMap:
    """Answers "is the worktop underfoot here, with room above?" for arrays of points.

    Built once by casting rays over a grid around the worktop in the *bare* scene: down
    from just above the top face (a top face within `TOLERANCE` of it under the point
    means worktop), and up (how much headroom). Loose objects are lifted out of the way
    first -- the task clears the ones in its working area, and one standing on the
    counter is not the counter.
    """

    TOLERANCE = 0.015
    START = 0.05

    def __init__(self, model, data, worktop: Worktop, radius: float = 1.3,
                 step: float = 0.02) -> None:
        self.step = step
        self.z = worktop.z
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                scratch.qpos[model.jnt_qposadr[j] + 2] -= 1000.0
        mujoco.mj_kinematics(model, scratch)
        centre = np.asarray(worktop.xy, dtype=float)
        self.xs = np.arange(centre[0] - radius, centre[0] + radius + step, step)
        self.ys = np.arange(centre[1] - radius, centre[1] + radius + step, step)
        self.supported = np.zeros((len(self.xs), len(self.ys)), dtype=bool)
        self.headroom = np.zeros((len(self.xs), len(self.ys)))
        geomid = np.zeros(1, dtype=np.int32)
        down, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
        for i, x in enumerate(self.xs):
            for k, y in enumerate(self.ys):
                start = np.array([x, y, self.z + self.START])
                d = mujoco.mj_ray(model, scratch, start, down, None, 1, -1, geomid)
                self.supported[i, k] = geomid[0] >= 0 and abs(d - self.START) <= self.TOLERANCE
                if self.supported[i, k]:
                    lift = np.array([x, y, self.z + 0.01])
                    h = mujoco.mj_ray(model, scratch, lift, up, None, 1, -1, geomid)
                    self.headroom[i, k] = np.inf if geomid[0] < 0 else h

    def _index(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        ix = np.rint((points[:, 0] - self.xs[0]) / self.step).astype(int)
        iy = np.rint((points[:, 1] - self.ys[0]) / self.step).astype(int)
        inside = (ix >= 0) & (ix < len(self.xs)) & (iy >= 0) & (iy < len(self.ys))
        return np.clip(ix, 0, len(self.xs) - 1), np.clip(iy, 0, len(self.ys) - 1), inside

    def underfoot(self, points) -> np.ndarray:
        ix, iy, inside = self._index(points)
        return self.supported[ix, iy] & inside

    def room(self, points, height: float) -> np.ndarray:
        ix, iy, inside = self._index(points)
        return self.supported[ix, iy] & inside & (self.headroom[ix, iy] >= height)

    def footprint_fits(self, xy, radius: float, height: float) -> bool:
        ring = [np.asarray(xy)] + [
            np.asarray(xy) + radius * np.array([math.cos(a), math.sin(a)])
            for a in np.linspace(0.0, 2 * math.pi, 12, endpoint=False)
        ]
        return bool(self.underfoot(ring).all() and self.room([xy], height).all())


def walkable(surface: SurfaceMap, start, goal, stop_short: float) -> bool:
    """Whether a walking robot at `start` can walk over the worktop to within
    `stop_short` of `goal`: worktop underfoot all the way, so no gap, sink or hob."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    span = float(np.linalg.norm(goal - start)) - stop_short
    if span <= 0:
        return True
    direction = (goal - start) / max(float(np.linalg.norm(goal - start)), 1e-9)
    points = [start + direction * s for s in np.arange(0.0, span + 1e-9, surface.step)]
    return bool(surface.underfoot(points).all())


#: The six staged task objects, as (world xy, footprint radius): every worktop robot
#: stands clear of all of them, so none is ever left out of the task (spec §2.3).
TaskObjects = dict[str, tuple[np.ndarray, float]]
#: The ones a worktop robot must be able to get at (spec §2.3: "the apple and plate are
#: within reach"), and how close a walking robot's base has to get to one to act on it.
REACHED = ("apple", "plate")
WALK_STOP_SHORT = 0.12
#: How far ahead a second worktop robot's heading is judged by the walk it leaves.
WALK_LOOKAHEAD = 1.0


def blocks_sightline(xy, radius: float, base_z: float, height: float, sightlines) -> bool:
    """Whether a robot standing at `xy` would stand in any of `sightlines`: segments from
    a rig camera to what it must see, each (start xyz, end xyz) in the world."""
    xy = np.asarray(xy, dtype=float)
    for start, end in sightlines:
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        for t in np.linspace(0.0, 1.0, 60):
            point = start + t * (end - start)
            if (base_z <= point[2] <= base_z + height
                    and float(np.linalg.norm(point[:2] - xy)) < radius):
                return True
    return False


def _beside(inst: Instance, surface: SurfaceMap, placed: list[Instance],
            objects: TaskObjects, sightlines=()) -> tuple[np.ndarray, float]:
    """A spot on the worktop for a second worktop robot, near the task and in nobody's way.

    Clear of every robot already placed and of all six staged objects' footprints -- the
    task is staged whole, so the robot gives way, never a distractor -- the worktop under
    its whole footprint with headroom over it, and a walk over the worktop to the apple and
    the plate. Of those, one out of the rig cameras' lines of sight to them if there is
    one -- a robot standing there puts its back in the task's frame -- and then the one
    closest to the farther of the two. It faces towards the apple, which is what it would
    be sent for, along the longest walk the worktop leaves it.
    """
    r = inst.radius
    height = ROBOT_HEIGHT[inst.name]
    others = [(p.xy, p.radius) for p in placed]
    reached = [objects[name] for name in REACHED if name in objects]
    best, best_score = None, None
    for x in surface.xs[::2]:
        for y in surface.ys[::2]:
            xy = np.array([x, y])
            if any(float(np.linalg.norm(xy - pxy)) < r + pr + GAP for pxy, pr in others):
                continue
            if any(float(np.linalg.norm(xy - oxy)) < r + orad + GAP
                   for oxy, orad in objects.values()):
                continue
            if not surface.footprint_fits(xy, r, height):
                continue
            far = max(float(np.linalg.norm(xy - oxy)) for oxy, _ in reached)
            if best_score is not None and (False, far) >= best_score:
                continue  # cannot beat the best even out of every sightline
            blocks = blocks_sightline(xy, r, surface.z, height, sightlines)
            if best_score is not None and (blocks, far) >= best_score:
                continue
            if not all(walkable(surface, xy, oxy, orad + WALK_STOP_SHORT)
                       for oxy, orad in reached):
                continue
            best, best_score = xy, (blocks, far)
    if best is None:
        raise SystemExit(
            f"no room on the worktop for {inst.name} beside "
            f"{', '.join(p.name for p in placed)}: nowhere clear of them and of the six task "
            "objects has the worktop under its whole footprint and a walk to the apple and "
            "the plate")
    if best_score[0]:
        print(f"warning: {inst.name} stands in a rig camera's line of sight to the task: "
              "nowhere else on this worktop has room for it", file=sys.stderr)
    apple = objects.get("apple", next(iter(objects.values())))[0]
    to = np.asarray(apple) - best
    bearing = math.atan2(to[1], to[0])
    # Facing the apple, give or take: of the headings within 90 deg of it, the one with the
    # longest walk ahead over the worktop, so the first steps a client asks for do not take
    # it straight off an edge or into the robot beside it.
    blocked = others + [(oxy, orad) for oxy, orad in objects.values()]

    def run(heading: float) -> float:
        ahead = np.array([math.cos(heading), math.sin(heading)])
        for d in np.arange(surface.step, WALK_LOOKAHEAD + 1e-9, surface.step):
            point = best + d * ahead
            if (not surface.underfoot([point]).all()
                    or any(float(np.linalg.norm(point - xy)) < rad for xy, rad in blocked)):
                return d
        return WALK_LOOKAHEAD

    headings = [bearing + math.radians(k) for k in range(-90, 91, 15)]
    yaw = max(headings, key=lambda h: (round(run(h), 2), -abs(h - bearing)))
    return best, float(math.atan2(math.sin(yaw), math.cos(yaw)))


def stand_fleet(instances: list[Instance], *, task_robot: str | None,
                find_worktop: Callable[[str], Worktop],
                floor_spot: Callable[[Instance, list], tuple[np.ndarray, float]],
                surface_map: Callable[[Worktop], SurfaceMap],
                task_objects: Callable[[Instance], TaskObjects],
                sightlines: Callable[[Instance], list] = lambda lead: []) -> Worktop | None:
    """Stand every robot: the task robot where the engine bolts it, the other worktop
    robots beside it, then the floor robots clear of all of them.

    Worktop robots go first. A worktop robot has no say in where it goes -- it needs the
    worktop -- while a floor robot can start anywhere open, so the floor robot is the one
    that gives way; placed the other way round, the roomiest floor is regularly the
    standing space in front of the very counter the arm is about to be bolted to.
    """
    by_name = {i.name: i for i in instances}
    worktop = None
    placed: list[Instance] = []
    if task_robot is not None:
        lead = by_name[task_robot]
        worktop = find_worktop(lead.name)
        stand(lead, "worktop", worktop.xy, worktop.yaw, worktop.z)
        placed.append(lead)
        rest = [i for i in instances if i.name != lead.name and i.name in
                _worktop_names(instances)]
        if rest:
            surface = surface_map(worktop)
            objects = task_objects(lead)
            lines = sightlines(lead)
            for inst in rest:
                xy, yaw = _beside(inst, surface, placed, objects, lines)
                stand(inst, "worktop", xy, yaw, worktop.z)
                placed.append(inst)
    keep_out = [(p.xy, floor_keep_out_radius(p.name)) for p in placed]
    for inst in instances:
        if inst.on:
            continue
        xy, yaw = floor_spot(inst, keep_out)
        stand(inst, "floor", xy, yaw, 0.0)
        keep_out.append((inst.xy, inst.radius + FLOOR_MARGIN))
    return worktop


def _worktop_names(instances) -> set[str]:
    import robots_spec

    return {i.name for i in instances if robots_spec.placement(i.name) == "worktop"}


def overlaps(instances: list[Instance]) -> list[str]:
    """Every pair of robots whose footprints overlap, as sentences. Empty is correct."""
    found = []
    for a_i, a in enumerate(instances):
        for b in instances[a_i + 1:]:
            if a.xy is None or b.xy is None:
                continue
            d = float(np.linalg.norm(a.xy - b.xy))
            same_level = abs(a.surface_z - b.surface_z) < 0.05
            if same_level and d < a.radius + b.radius:
                found.append(f"{a.name} and {b.name} are {d:.2f} m apart, closer than "
                             f"their footprints ({a.radius + b.radius:.2f} m)")
    return found
