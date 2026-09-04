"""Pick a red apple off a work surface and place it on a white plate.

The task is grafted onto whatever kitchen an engine has already compiled, rather than
shipping its own room: the engines' whole point is that they provide real furnished
scenes, and a task that replaced the scene would make them bystanders. What it *does*
bring is the geometry and the contact physics, because those are the part that was
measured and the part a procedurally chosen object cannot supply.

Every number here is a contract term, and several of them look arbitrary until you know
what they cost to find:

* **`condim="6"` on the apple.** Rolling friction is the *third* entry of `friction` and
  it only exists at condim 6 -- at condim 3 or 4 the value is parsed and silently thrown
  away, so a declared 0.001 is really 0.000 and a fruit nudged at 7 mm/s coasts for half
  a minute. On the rig this was measured on, raising condim alone (no friction number
  changed) took episodes that ended with the apple on the floor from 12 of 16 to 0 of 12.
* **The plate has a rim made of 24 boxes.** MuJoCo has no torus. Without it the plate is
  a flat disc, and apples delivered to within a millimetre of its centre rolled straight
  off -- so the "at rest" clause of the success predicate never fired and a placement
  that looked perfect scored zero.
* **Visual meshes are not the collision geometry, and carry `density="0"`.** A textured
  mesh for looks, a primitive for physics. Swapping the primitives for mesh hulls would
  discard the grasp tuning, and a visual geom without `density="0"` silently adds mass:
  the apple weighs 0.081 kg instead of 0.020 kg, which is most of the way to unliftable.
* **Every camera declares `resolution`.** A `<camera>` without it renders zero-sized
  images and reports no error at all.

Coordinates are the **arm base frame** -- x across the work surface, the surface itself
at z = 0 -- and the staging maps them into the engine's world with the mount transform
it was given. That is also the frame the poses go out on, so the console's geometry is
identical whether the arm is bolted to a kitchen island or standing on a bare table.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

#: Visual meshes (YCB scans; see assets/LICENSES.md). Absolute paths on purpose: the
#: spec this is staged into has its own `meshdir` pointing at the engine's asset tree,
#: and a relative file here would be resolved against that and silently not found.
ASSETS = Path(__file__).resolve().parent / "assets" / "ycb"
#: The YCB apple is 79 mm across; scaled to ~40 mm to match the 20 mm collision sphere.
APPLE_MESH_SCALE = 0.50
#: The YCB plate, scaled so its rim sits on the 0.10 m collision cylinder.
PLATE_MESH_SCALE = 0.77
#: The scanned plate's underside is slightly domed; lifting the visual by this much sits
#: it on the collision cylinder's top face rather than clipping through it.
PLATE_MESH_Z = 0.0092

#: The dressing: everything else the reference scene keeps on its table, as
#: (name, position, yaw, mass, mesh scale). Arm base frame, metres, radians, kilograms.
#:
#: They are inside the arm's 0.330 m top-down grasp envelope *on purpose* -- mug at +79.5
#: degrees, banana at +45.0, lemon at -74.9, bearings the pick-and-place never sweeps --
#: so the policy sees the distractors its training data had rather than a bare slab.
#: The bowl sits at r = 0.42 and stays out of reach. The banana's yaw is a contract term,
#: not decoration: it is 0.19 m long and the closest object to the arm's path.
#:
#: The z values are **settled equilibria**, not derived from the meshes. Two mesh-derived
#: attempts on the reference rig failed in opposite directions: one put the objects
#: 8-85 mm inside the slab, and the solver's separation impulse launched the lemon
#: across the room; the next left all four hanging 7-54 mm in the air, visibly dropping
#: at every startup. These four are where MuJoCo puts each body after six seconds of
#: stepping.
DRESSING: tuple[tuple[str, tuple[float, float, float], float, float, float], ...] = (
    ("bowl", (0.14, -0.40, 0.0271), 0.0, 0.147, 1.0),
    ("mug", (0.05, 0.27, 0.0272), 0.0, 0.118, 1.0),
    ("banana", (0.156, 0.156, 0.0172), 0.785, 0.066, 1.0),
    ("lemon", (0.07, -0.26, 0.0294), 0.0, 0.029, 1.0),
)
#: Contact parameters shared by the dressing. `condim 6` is the whole point: rolling
#: friction is the third `friction` entry and does not exist below condim 6, and these
#: are round scanned meshes. Measured on the reference rig, raising the *table's*
#: friction changed their residual velocities by nothing to five decimal places until
#: condim was raised; at 6 the residual speeds fell by about two orders of magnitude.
DRESSING_CONDIM = 6
DRESSING_FRICTION = (1.0, 0.1, 0.02)

#: The reference work surface: a 0.92 x 0.92 m wood slab whose top face is exactly z = 0,
#: centred 0.20 m ahead of the arm. Body `table`, geom `table_top` in the reference MJCF.
#:
#: Why bring a table into a kitchen that already has a counter: the overhead camera's
#: framing is a property of the surface under it, not of the camera. On the reference rig
#: that view is this slab filling the frame, all four corners inside it with 16.8 px to
#: spare; on a marble island it was a diagonal counter with a third of the frame floor,
#: and the policy had never seen anything like it. The slab's 4 cm of thickness sinks
#: *below* z = 0 into the engine's counter, because the arm base already sits at counter
#: height and every contract height (`RESTING_Z`, the waypoint heights, the success gate)
#: is measured from that plane. Aprons and legs are left out -- there is a worktop. It is
#: static, which is what makes overhang past a counter edge harmless.
TABLE_CENTRE = (0.20, 0.0, 0.0)
TABLE_HALF = (0.46, 0.46, 0.02)
TABLE_GEOM_Z = -0.02
TABLE_FRICTION = (1.0, 0.005, 0.0001)
TABLE_TEXREPEAT = (3.0, 3.0)

#: Where the wood texture lives; `ASSETS` is the YCB tree beside it.
TEXTURES = Path(__file__).resolve().parent / "assets" / "textures"

#: Lighting and exposure, from the reference scene, applied on the host spec -- scene-global
#: like `impratio`. `shadowclip 0.15` because 0.3 gave visible shadow acne on this arm; the
#: headlight at 0.35/0.28/0.08 because the reference's previous 0.55/0.35 clipped 41.6 % of
#: rendered pixels to white and this brought it to 3.0 %.
#:
#: **Measured here, on the overhead frame of an iTHOR kitchen** (`--render-framing` reports
#: this number, so it is one command to re-check on another scene):
#:
#:     kitchen's own lighting, untouched          5.7 % clipped
#:     + this exposure block                      3.1 %   <- shipped
#:     + this block and the two lamps below      75.2 %
#:
#: So the exposure block carries over and **the lamps do not**, which is why they are
#: `extra_lights`, off by default. They exist in the reference to light a bare table in an
#: otherwise empty room; a furnished kitchen already has its own, and adding a 0.45 and a
#: 0.35 directional on top of them is what takes the frame from correctly exposed to
#: three-quarters white. Turn them on for a scene that renders too dark, not by default.
SHADOW_CLIP = 0.15
HEADLIGHT = {"diffuse": 0.35, "ambient": 0.28, "specular": 0.08}
#: (name, pos, dir, diffuse, castshadow) in the arm base frame.
LIGHTS = (
    ("task_key_light", (0.0, 0.0, 1.5), (0.0, 0.0, -1.0), 0.45, True),
    ("task_fill_light", (0.9, -0.6, 0.7), (-0.6, 0.45, -0.65), 0.35, False),
)

# ------------------------------------------------------------------ scene geometry

APPLE_BODY = "task_apple"
PLATE_BODY = "task_plate"

APPLE_SPAWN = (0.30, 0.10, 0.020)
APPLE_RADIUS = 0.020
APPLE_MASS = 0.020

#: Moved in from the (0.40, -0.20) the contract text still names. Top-down grasp reach
#: peaks at 0.330 m dead ahead and closes to 0.295 m at +/-80 deg -- far tighter than the
#: 0.48 m free-pitch reach -- so the old plate sat outside the envelope the release has
#: to be solved in. Same bearing class, no success clause changed.
PLATE_CENTRE = (0.226, -0.226, 0.0)
PLATE_RADIUS = 0.10
PLATE_HALF_HEIGHT = 0.0102
PLATE_TOP_Z = 2 * PLATE_HALF_HEIGHT  # 0.0204

#: Where an apple sitting on the plate has its centre: plate top + apple radius.
RESTING_Z = 0.040

#: The pose the episode starts from: five arm joints, radians, contract order.
#:
#: **`wrist_roll` is +1.62 and that number is doing real work.** A VLA conditions on the
#: measured joint state, and its processor bins that state into 256 buckets and clips
#: silently -- so a start pose outside the band the checkpoint was trained on is a
#: quiet, total failure of conditioning rather than a visible error. MolmoAct2-SO100_101
#: was trained with `wrist_roll` in -63.5..+42.9 degrees of *its* frame, which maps to
#: ours as `model = 90 - degrees(ours)`. The engine's stock rest pose has `wrist_roll`
#: at 0, i.e. model +90 -- 47 degrees outside the band, on every step of every episode.
#:
#: +1.62 rad maps to model -2.8, near the middle of that band (its trained median is
#: -11), and it is **geometrically identical** to the -1.52 the plan used to pick: the
#: two differ by pi, and a jaw rolled half a turn is the same level jaw. Measured jaw
#: tilt is 0.000 either way. So this costs nothing and buys in-distribution
#: conditioning; with it the whole start state sits inside the band on every channel
#: except the jaw, which is exactly at its q99 when fully open.
START_ARM_QPOS = (0.0, -0.6, 1.0, 0.6, 1.62)

#: Jaw fully open at the start, in contract units.
START_GRIPPER = 1.0

# ------------------------------------------------------------------ success predicate

MAX_HORIZONTAL_DIST = 0.080  # plate radius 0.10 - apple radius 0.020
Z_TOLERANCE = 0.015
MAX_SPEED = 0.01  # m/s: at rest, not passing through
SUSTAIN_SECONDS = 1.0  # of SIMULATED time
MIN_DISPLACEMENT = 0.25  # it genuinely travelled, rather than being nudged

# ------------------------------------------------------------------ cameras

#: (name, pos, xyaxes, fovy, resolution) in the arm base frame.
#: `overhead` looks down at 62 deg from 1.110 m, framing all four corners of a work
#: surface with 16.8 px of edge slack, with zero roll -- its right vector is pinned to
#: +y, so "left in the image" is "left in the world" and a policy's pixel claims
#: unproject without a sign trap. `side` is distinguished from it by *elevation*, not by
#: standing on the other side of the table: near-horizontal at 5.6 deg, which is the view
#: that resolves height and the one an overhead camera cannot supply.
SCENE_CAMERAS: tuple[tuple[str, tuple, tuple, float, tuple[int, int]], ...] = (
    (
        "overhead",
        (0.795, 0.000, 1.110),
        (0.00000, 1.00000, 0.00000, -0.88295, 0.00000, 0.46947),
        45.0,
        (640, 480),
    ),
    (
        "side",
        (0.28, 0.80, 0.13),
        (-0.99892, -0.04646, 0.00000, 0.00456, -0.09815, 0.99516),
        45.0,
        (640, 480),
    ),
)

# The conical ring that keeps a delivered apple on the plate, generated once and frozen
# so the geometry is reproducible: N boxes around a cone of half-angle alpha, inner lip
# at Z_IN and outer at Z_OUT. The well floor stays at PLATE_TOP_Z, so RESTING_Z -- and
# therefore the whole success predicate -- is untouched by the rim's existence.
_RIM_N = 24
_RIM_HALF_LEN = 0.022270
_RIM_WIDTH = 0.013966
_RIM_THICKNESS = 0.004000
_RIM_RADIUS = 0.083063
_RIM_Z = 0.025418
_RIM_ALPHA = 0.4014  # rad, 23.00 deg


def _rim_geoms():
    """Yield (name, pos, quat, size) for each box of the plate rim, in the plate frame."""
    for i in range(_RIM_N):
        phi = 2.0 * math.pi * i / _RIM_N
        # quat = Rz(phi) . Ry(-alpha), composed by hand to keep this dependency-free.
        cz, sz = math.cos(phi / 2), math.sin(phi / 2)
        cy, sy = math.cos(-_RIM_ALPHA / 2), math.sin(-_RIM_ALPHA / 2)
        quat = (cz * cy, -sz * sy, cz * sy, sz * cy)
        pos = (_RIM_RADIUS * math.cos(phi), _RIM_RADIUS * math.sin(phi), _RIM_Z)
        yield f"plate_rim_{i:02d}", pos, quat, (_RIM_HALF_LEN, _RIM_WIDTH, _RIM_THICKNESS)


# ------------------------------------------------------------------ frame helpers


def base_frame(pos, yaw: float) -> np.ndarray:
    """4x4 transform from the arm base frame into the engine's world frame."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return transform


def _apply(transform: np.ndarray, point) -> list[float]:
    out = transform @ np.array([*point, 1.0], dtype=np.float64)
    return [float(v) for v in out[:3]]


def _yaw_quat(yaw: float) -> list[float]:
    return [float(math.cos(yaw / 2)), 0.0, 0.0, float(math.sin(yaw / 2))]


def _quat_mul(a, b) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


# ------------------------------------------------------------------ staging


#: Scene objects whose origin lands within this horizontal distance of the arm base, in
#: the band from just under the work surface to a little above it, are moved out of the
#: way. The task's own objects reach 0.34 m; this leaves a margin around that.
#:
#: A task owns its workspace. On the rig these numbers came from, the arm has a bare
#: table and its dressing sits out at the edges; here the arm is mounted into a furnished
#: kitchen whose own apple lands 0.10 m from the task's spawn and whose book and bread sit
#: inside the working annulus. Two apples is not a harder task, it is a different one --
#: the policy is told to fetch "the red apple" and only one of them is the 40 mm sphere
#: the jaw can close on -- and clutter in the working area occludes the plate in the side
#: view. Leaving it in confounds a grounding test with a disambiguation test.
CLEAR_RADIUS = 0.55
CLEAR_Z_BAND = (-0.15, 0.45)
#: How far under the scene a cleared object is parked. Only has to be out of every
#: camera's frustum; it is static once moved, so the distance costs nothing.
SUNK_DEPTH = 50.0


def stage(
    spec,
    mount_pos,
    yaw: float,
    *,
    clear_radius: float = CLEAR_RADIUS,
    reference_table: bool = True,
    dressing: bool = True,
    lighting: bool = True,
    extra_lights: bool = False,
) -> list[str]:
    """Add the reference table -- slab, apple, plate, dressing, lights, cameras -- to a spec.

    `reference_table`, `dressing` and `lighting` are the three things a later experiment
    might want to take away one at a time; each is verified off as well as on.

    `mount_pos`/`yaw` locate the arm's base body in the engine's world, and everything
    below is placed relative to it, so the contract geometry survives being dropped into
    a kitchen that knows nothing about it.

    Loose scene objects inside `clear_radius` of the base are sunk under the floor first;
    see `CLEAR_RADIUS`. Only *movable* bodies are touched -- anything without a free joint
    is a fixture, and sinking a counter would take the arm's own work surface with it.
    They are moved rather than deleted because deleting a body out of a compiled iTHOR
    house takes its metadata references with it.
    """
    transform = base_frame(mount_pos, yaw)
    base_quat = _yaw_quat(yaw)

    # Solver settings are scene-global, so a task that needs them has to set them on the
    # host scene. Elliptic friction cones with impratio 50 are what let a soft-contact
    # grasp survive a carry: with the defaults the jaws close on the apple, hold it for
    # the settle, and lose it the moment the arm starts to lift -- measured here as
    # jaw/apple contacts going 3 -> 0 at lift-off with the apple not having moved at
    # all. On the reference rig the same pair took the worst-case aggressive carry from
    # 4.4-7.8 s of hold to 14.8 s, with half the cases never losing the apple.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = max(float(spec.option.impratio), 50.0)

    if lighting:
        _apply_visual(spec)

    cleared = _clear_workspace(spec, transform, clear_radius)

    _add_assets(spec)

    # ---- the slab ---------------------------------------------------------------
    # Before the objects, so a reader sees the surface built before what rests on it.
    if reference_table:
        table = spec.worldbody.add_body(
            name="task_table", pos=_apply(transform, TABLE_CENTRE), quat=base_quat
        )
        table.add_geom(
            name="task_table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(TABLE_HALF),
            pos=[0.0, 0.0, TABLE_GEOM_Z],
            material="task_wood_mat",
            friction=list(TABLE_FRICTION),
            group=2,  # static, so it needs no inertia; visible under both masks
        )

    # ---- apple ------------------------------------------------------------------
    # Top-level body: MuJoCo refuses a free joint on a nested one.
    apple = spec.worldbody.add_body(
        name=APPLE_BODY, pos=_apply(transform, APPLE_SPAWN), quat=base_quat
    )
    apple.add_freejoint(name=f"{APPLE_BODY}_joint")
    # Textured mesh for looks, primitive for physics. The mesh never collides and adds
    # no mass; the sphere is invisible and carries every tuned contact parameter.
    apple.add_geom(
        name=f"{APPLE_BODY}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="task_apple_vis",
        material="task_apple_mat",
        contype=0,
        conaffinity=0,
        density=0.0,
        # Visual geoms group 2, collision geoms group 0 -- and BOTH halves of that are
        # forced, by different engines, in opposite directions:
        #
        #   RoboCasa renders through `visual_only()`, a mask showing groups 1 and 2 only,
        #   because its own collision hulls are group 0. A visual mesh in group 0 is
        #   therefore invisible in every camera frame there -- silently: the wire stays
        #   healthy, the poses are right, and a scripted policy still passes because it
        #   never looks, while a VLA sees an empty worktop.
        #
        #   RoboCasa also sets `inertiagrouprange = [0, 0]`, so **only group-0 geoms
        #   contribute inertia**. A collision geom outside group 0 leaves its body with
        #   no mass at all, and the kitchen refuses to compile with "mass and inertia of
        #   moving bodies must be larger than mjMINVAL" -- an error naming no body, three
        #   bisections away from its cause.
        #
        # Group 2 visual + group 0 collision satisfies both, and is RoboCasa's own
        # convention. On MolmoSpaces (inertiagrouprange 0-5, default option showing
        # groups 0-2) it changes nothing: the colliders carry alpha 0 either way.
        group=2,
    )
    apple.add_geom(
        name=f"{APPLE_BODY}_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[APPLE_RADIUS, 0.0, 0.0],
        mass=APPLE_MASS,
        condim=6,
        friction=[2.0, 0.05, 0.001],
        solref=[0.03, 1.0],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        # Deliberately NO `priority`. mujoco_menagerie gives the SO-101's finger geoms
        # `priority="1"` together with its own contact tuning (solref 0.01, friction 1),
        # and MuJoCo resolves a contact between geoms of unequal priority with the higher
        # one's parameters alone -- so at the jaw it is the *jaw's* parameters that act,
        # and the softer solref above only governs the apple against the table and the
        # plate. That turns out to be what holds: measured offline over timestep
        # {0.002, 0.005} x {ramped, jumped} lift, the apple is lifted in all four cases
        # with the jaw's parameters in force and dropped in all four when the apple was
        # given a higher priority to force its own soft contact onto the fingers. The
        # reference rig's soft apple was tuned against a jaw with no contact tuning of
        # its own; here the jaw already has some, and it is the better half.
        rgba=[1.0, 0.0, 0.0, 0.0],
        group=0,  # collision; must be group 0 to carry inertia -- see above
    )

    # ---- plate ------------------------------------------------------------------
    plate = spec.worldbody.add_body(
        name=PLATE_BODY, pos=_apply(transform, PLATE_CENTRE), quat=base_quat
    )
    plate.add_geom(
        name=f"{PLATE_BODY}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="task_plate_vis",
        material="task_plate_mat",
        pos=[0.0, 0.0, PLATE_MESH_Z],
        contype=0,
        conaffinity=0,
        density=0.0,
        group=2,  # visual; see the apple's visual geom
    )
    plate.add_geom(
        name=f"{PLATE_BODY}_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[PLATE_RADIUS, PLATE_HALF_HEIGHT, 0.0],
        pos=[0.0, 0.0, PLATE_HALF_HEIGHT],
        condim=4,
        friction=[1.5, 0.05, 0.001],
        rgba=[1.0, 1.0, 1.0, 0.0],
        group=0,  # collision
    )
    for name, pos, quat, size in _rim_geoms():
        plate.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(size),
            pos=list(pos),
            quat=list(quat),
            condim=4,
            friction=[1.5, 0.05, 0.001],
            rgba=[1.0, 1.0, 1.0, 0.0],
            group=0,  # collision
        )

    # ---- dressing ---------------------------------------------------------------
    # Same visual/collision split as the apple and the plate, for the same reason: a
    # geom cannot be both visible under RoboCasa's render mask (groups 1-2) and
    # inertia-bearing under its `inertiagrouprange` of [0, 0]. Both geoms use the one
    # mesh asset, so this costs a convex hull and no extra file. The reference gets away
    # with a single mesh geom because its scene constrains neither.
    if dressing:
        for name, pos, obj_yaw, mass, scale in DRESSING:
            body = spec.worldbody.add_body(
                name=f"task_{name}",
                pos=_apply(transform, pos),
                quat=_quat_mul(base_quat, _yaw_quat(obj_yaw)),
            )
            body.add_freejoint(name=f"task_{name}_joint")
            body.add_geom(
                name=f"task_{name}_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"task_{name}_vis",
                material=f"task_{name}_mat",
                contype=0,
                conaffinity=0,
                density=0.0,
                group=2,
            )
            body.add_geom(
                name=f"task_{name}_geom",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"task_{name}_vis",
                mass=mass,
                condim=DRESSING_CONDIM,
                friction=list(DRESSING_FRICTION),
                rgba=[1.0, 1.0, 1.0, 0.0],
                group=0,
            )

    # ---- lights -----------------------------------------------------------------
    # Positions go through the transform like everything else; directions only rotate,
    # the same split the camera loop below makes for its axes.
    if extra_lights:
        rot = transform[:3, :3]
        for name, pos, direction, diffuse, castshadow in LIGHTS:
            light = spec.worldbody.add_light(name=name)
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            light.pos = _apply(transform, pos)
            light.dir = [float(v) for v in rot @ np.asarray(direction, dtype=np.float64)]
            light.diffuse = [diffuse] * 3
            light.castshadow = castshadow

    # ---- cameras ----------------------------------------------------------------
    for name, pos, xyaxes, fovy, resolution in SCENE_CAMERAS:
        right = np.asarray(xyaxes[:3], dtype=np.float64)
        up = np.asarray(xyaxes[3:], dtype=np.float64)
        rot = transform[:3, :3]
        camera = spec.worldbody.add_camera(
            name=name,
            pos=_apply(transform, pos),
            fovy=fovy,
        )
        camera.resolution = list(resolution)
        # xyaxes is not a spec field; set the equivalent orientation as a quaternion.
        camera.quat = _camera_quat(rot @ right, rot @ up)

    return cleared


def table_corners(transform: np.ndarray) -> list[list[float]]:
    """The slab's four top-face corners in the engine's world, for fit and framing checks.

    One definition, used by both checks, so they can never disagree about where it is.
    """
    cx, cy, cz = TABLE_CENTRE
    hx, hy, _ = TABLE_HALF
    return [
        _apply(transform, (cx + sx * hx, cy + sy * hy, cz))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]


def _apply_visual(spec) -> None:
    """The reference scene's exposure, on the host spec. See `SHADOW_CLIP`/`HEADLIGHT`.

    Not copied: `shadowsize 4096` (a 44-fixture kitchen would pay for it on every frame,
    where a bare table did not) and `offwidth/offheight` (`CameraStreams` already raises
    those per camera to exactly what it renders).
    """
    spec.visual.map.shadowclip = SHADOW_CLIP
    spec.visual.headlight.diffuse = [HEADLIGHT["diffuse"]] * 3
    spec.visual.headlight.ambient = [HEADLIGHT["ambient"]] * 3
    spec.visual.headlight.specular = [HEADLIGHT["specular"]] * 3


def _add_assets(spec) -> None:
    """Meshes, textures and materials for everything staged -- once per spec."""
    if any(m.name == "task_apple_vis" for m in spec.meshes):
        return

    # The dressing and the wood: a mesh, its scan's texture and a plain material each.
    # No specular/shininess on the dressing -- the reference sets none, and the apple's
    # 0.35/0.5 were tuned for a sphere, not scanned crockery.
    for name, _pos, _yaw, _mass, scale in DRESSING:
        mesh = spec.add_mesh(name=f"task_{name}_vis")
        mesh.file = str(ASSETS / name / "textured.obj")
        mesh.scale = [scale] * 3
        texture = spec.add_texture(name=f"task_{name}_tex")
        texture.type = mujoco.mjtTexture.mjTEXTURE_2D
        texture.file = str(ASSETS / name / "texture_map.png")
        material = spec.add_material(name=f"task_{name}_mat")
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"task_{name}_tex"

    wood = spec.add_texture(name="task_wood_tex")
    wood.type = mujoco.mjtTexture.mjTEXTURE_2D
    wood.file = str(TEXTURES / "light-wood.png")
    wood_mat = spec.add_material(name="task_wood_mat")
    wood_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "task_wood_tex"
    wood_mat.texrepeat = list(TABLE_TEXREPEAT)
    wood_mat.specular = 0.25
    wood_mat.shininess = 0.35
    apple_mesh = spec.add_mesh(name="task_apple_vis")
    apple_mesh.file = str(ASSETS / "apple" / "textured.obj")
    apple_mesh.scale = [APPLE_MESH_SCALE] * 3
    plate_mesh = spec.add_mesh(name="task_plate_vis")
    plate_mesh.file = str(ASSETS / "plate" / "textured.obj")
    plate_mesh.scale = [PLATE_MESH_SCALE] * 3

    texture = spec.add_texture(name="task_apple_tex")
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.file = str(ASSETS / "apple" / "texture_map.png")

    apple_mat = spec.add_material(name="task_apple_mat")
    apple_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "task_apple_tex"
    apple_mat.specular = 0.35
    apple_mat.shininess = 0.5
    # The YCB plate scan is dark red; the task wants a plain white ceramic plate, so the
    # geometry is used and the scan's texture is not.
    plate_mat = spec.add_material(name="task_plate_mat")
    plate_mat.rgba = [0.97, 0.97, 0.95, 1.0]
    plate_mat.specular = 0.55
    plate_mat.shininess = 0.75
    plate_mat.reflectance = 0.08


def _camera_quat(right: np.ndarray, up: np.ndarray) -> list[float]:
    """MuJoCo camera quaternion from world-frame right/up vectors.

    A MuJoCo camera looks down its own **-z**, with +x right and +y up, which is why the
    third column is the negated forward direction rather than the forward direction.
    """
    right = right / np.linalg.norm(right)
    up = up - (up @ right) * right
    up = up / np.linalg.norm(up)
    back = np.cross(right, up)
    rot = np.column_stack([right, up, back])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))
    return [float(v) for v in quat]


def _clear_workspace(spec, transform: np.ndarray, radius: float) -> list[str]:
    """Sink every loose scene body inside the task's working area. Returns their names.

    Walks the body tree accumulating parent transforms, because `MjsBody.pos` is
    **parent-relative** and a scene's graspables are usually nested a couple of levels
    down inside whatever surface holds them. Comparing those raw positions against a
    world-frame radius silently misses exactly the objects that are nested -- measured:
    it caught the scene's apple and bread and left a book standing in the plate's half of
    the frame.

    Staging happens before the compile, so this cannot ask MuJoCo for world positions;
    accumulating them here is the same arithmetic MuJoCo would do, and for objects
    resting on a surface the spawn pose is where they are.
    """
    inverse = np.linalg.inv(transform)
    moved: list[str] = []

    def origins(body, here: np.ndarray, out: list) -> None:
        """World origins of this body and every descendant."""
        out.append(here[:3, 3])
        for child in body.bodies:
            origins(child, here @ _homogeneous(child.pos, child.quat), out)

    def inside(point) -> bool:
        local = inverse @ np.array([*point, 1.0])
        return (
            float(np.hypot(local[0], local[1])) <= radius
            and CLEAR_Z_BAND[0] < local[2] < CLEAR_Z_BAND[1]
        )

    def walk(body, parent: np.ndarray) -> None:
        here = parent @ _homogeneous(body.pos, body.quat)
        name = body.name or ""
        movable = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints)
        if name and movable and not name.startswith(("task_", "robot_")):
            # Judged on the whole object, not just its origin. A scene's objects are
            # often several bodies, and the root's origin can sit well outside the
            # region the visible parts occupy -- measured: this rule leaves a book
            # standing next to the plate when the root is tested alone, because the
            # part in frame is a child two levels down.
            points: list = []
            origins(body, here, points)
            if any(inside(point) for point in points):
                # Static first, THEN moved. A free body parked below the floor is still
                # a free body: it falls, forever, accelerating -- and a scene full of
                # objects in permanent freefall wrecks the solver. Measured: the
                # simulated clock went *backwards* (-0.006x wall clock) and every
                # episode died in the embodiment's own sim-clock preflight. Dropping the
                # free joint makes it a static frame, which costs nothing to step.
                for joint in list(body.joints):
                    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                        spec.delete(joint)
                body.pos = [body.pos[0], body.pos[1], body.pos[2] - SUNK_DEPTH]
                moved.append(name)
                return  # its children go with it
        for child in body.bodies:
            walk(child, here)

    for child in spec.worldbody.bodies:
        walk(child, np.eye(4))
    return moved


def _homogeneous(pos, quat) -> np.ndarray:
    out = np.eye(4)
    w, x, y, z = (float(v) for v in quat)
    out[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    out[:3, 3] = [float(v) for v in pos]
    return out


# ------------------------------------------------------------------ the arbiter


class AppleOnPlate:
    """Measures the world and answers whether the apple is on the plate.

    It only ever *reads*. Teleporting the apple into place would satisfy every clause
    below and mean nothing, so there is deliberately no code path here that writes to a
    free joint outside `reset`, and `reset` restores the whole world to its spawn state
    rather than moving one body.

    The hold is the clause that does the real work. Clauses 1-3 and 5 are instantaneous
    and an apple bouncing across the plate satisfies all of them for a frame or two on
    its way off the far side; requiring them to hold continuously for a second of
    simulated time is what separates a placement from a fly-past.
    """

    def __init__(self, model, data) -> None:
        self._model = model
        self._apple = model.body(APPLE_BODY).id
        self._plate = model.body(PLATE_BODY).id
        self._qpos_adr = model.jnt_qposadr[model.body(APPLE_BODY).jntadr[0]]
        self._qvel_adr = model.jnt_dofadr[model.body(APPLE_BODY).jntadr[0]]

        # Everything whose pose goes out on the free-joint topic, discovered rather than
        # asserted: `stage()` can be asked for no dressing and no slab, and an arbiter
        # that insisted on six bodies would turn a supported flag into a crash. A body
        # with no free joint (the plate) publishes zero velocity, which is what the
        # reference does for it too.
        self._published: list[tuple[str, int, int | None]] = []
        for name in ("apple", "plate", *(d[0] for d in DRESSING)):
            body_id = _find_body(model, f"task_{name}")
            if body_id is None:
                continue
            jntadr = int(model.body(body_id).jntadr[0])
            dof = int(model.jnt_dofadr[jntadr]) if jntadr >= 0 else None
            self._published.append((name, body_id, dof))

        # The arm base frame, captured at spawn: the console works in contract
        # coordinates, so poses go out transformed into this frame rather than the
        # engine's world. On the reference rig the two are the same thing, which is
        # exactly the property that keeps the console identical across them.
        # Each engine namespaces the robot its own way -- MolmoSpaces prefixes every body
        # with `robot_0/`, RoboCasa uses its own prefix -- so the arm's root is found by
        # suffix rather than by a name one engine happens to use. Getting this wrong would
        # not raise: it would silently leave the transform as identity and publish every
        # pose in the engine's world frame, which for an arm mounted on a kitchen island
        # is a metre and a rotation away from what the console expects.
        base_id = _find_body(model, "base")
        self._world_from_base = np.eye(4)
        if base_id is not None:
            mujoco.mj_forward(model, data)
            self._world_from_base[:3, :3] = data.body(base_id).xmat.reshape(3, 3)
            self._world_from_base[:3, 3] = data.body(base_id).xpos
        else:
            raise SystemExit(
                "apple_on_plate: no arm root body found (looked for one named 'base' or "
                "'<prefix>base'). Every pose would otherwise go out in the wrong frame."
            )
        self._base_from_world = np.linalg.inv(self._world_from_base)

        self._apply_start_pose(data)

        self._spawn_qpos = data.qpos.copy()
        self._spawn_qvel = data.qvel.copy()
        # Actuator targets too, not just state: these are position servos, so restoring
        # qpos while leaving ctrl at whatever the last episode commanded would put the
        # arm back at rest and then immediately drive it away again.
        self._spawn_ctrl = data.ctrl.copy()
        self._holding_since: float | None = None

    def _apply_start_pose(self, data) -> None:
        """Put the arm in the task's start pose before anything is snapshotted.

        Both the joint state and the actuator target: these are position servos, so a
        ctrl left where the engine's own rest pose put it would drive the arm straight
        back out of this one on the first step.
        """
        from ros_surfaces.so101 import ARM_JOINTS, GRIPPER_OFFSET_RAD

        targets = list(zip((n.removesuffix("_joint") for n in ARM_JOINTS), START_ARM_QPOS))
        targets.append(("gripper", START_GRIPPER - GRIPPER_OFFSET_RAD))
        missing = []
        for mjcf, value in targets:
            jid = _find_joint(self._model, mjcf)
            if jid < 0:
                missing.append(mjcf)
                continue
            data.qpos[self._model.jnt_qposadr[jid]] = value
            # Actuators in this spec are named after the joint they drive, namespace
            # and all, so the joint's own name is the way to find it.
            joint_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if aid >= 0:
                data.ctrl[aid] = value
        if missing:
            raise SystemExit(
                f"apple_on_plate: could not find joint(s) {missing} to apply the start "
                "pose. Silently skipping them would start the arm out of the band the "
                "VLA was trained on, which fails invisibly."
            )
        mujoco.mj_forward(self._model, data)

    def contact_bodies(self) -> frozenset[int]:
        """Bodies the gripper must be able to touch for the task to be possible."""
        return frozenset({self._apple, self._plate})

    # -- geometry -------------------------------------------------------------------

    def _apple_state(self, data):
        pos_w = np.asarray(data.xpos[self._apple], dtype=np.float64)
        vel_w = np.asarray(data.qvel[self._qvel_adr : self._qvel_adr + 3], dtype=np.float64)
        rot = self._base_from_world[:3, :3]
        pos = rot @ (pos_w - self._world_from_base[:3, 3])
        return pos, rot @ vel_w

    def free_joint_entries(self, data):
        """The bodies whose poses go out on the free-joint topic, in the base frame."""
        rot = self._base_from_world[:3, :3]
        origin = self._world_from_base[:3, 3]
        for name, body_id, dof in self._published:
            pos = rot @ (np.asarray(data.xpos[body_id], dtype=np.float64) - origin)
            quat = np.asarray(data.xquat[body_id], dtype=np.float64)
            if dof is None:
                lin = ang = np.zeros(3)
            else:
                lin = rot @ np.asarray(data.qvel[dof : dof + 3])
                ang = rot @ np.asarray(data.qvel[dof + 3 : dof + 6])
            # `apple` and `bowl`, not `task_apple`: the console selects by the contract's
            # names and must not have to know how a scene spells them.
            yield name, pos, quat, lin, ang

    def instantaneous(self, data) -> tuple[bool, str]:
        """Clauses 1, 2, 3 and 5, with the reason the first failing one gives."""
        pos, vel = self._apple_state(data)
        horizontal = math.hypot(pos[0] - PLATE_CENTRE[0], pos[1] - PLATE_CENTRE[1])
        if horizontal >= MAX_HORIZONTAL_DIST:
            return False, f"horizontal {horizontal:.4f} >= {MAX_HORIZONTAL_DIST:.3f}"
        if abs(pos[2] - RESTING_Z) > Z_TOLERANCE:
            return False, f"z {pos[2]:.4f} outside {RESTING_Z:.3f} +/- {Z_TOLERANCE:.3f}"
        speed = float(np.linalg.norm(vel))
        if speed >= MAX_SPEED:
            return False, f"speed {speed:.4f} >= {MAX_SPEED:.3f}"
        displacement = float(np.linalg.norm(pos - np.asarray(APPLE_SPAWN)))
        if displacement <= MIN_DISPLACEMENT:
            return False, f"displacement {displacement:.4f} <= {MIN_DISPLACEMENT:.3f}"
        return True, "ok"

    def success(self, data, stamp: float) -> bool:
        """Clause 4: everything above, held continuously for >= 1.0 s of sim time."""
        ok, _reason = self.instantaneous(data)
        if not ok:
            self._holding_since = None
            return False
        if self._holding_since is None:
            self._holding_since = stamp
        return (stamp - self._holding_since) >= SUSTAIN_SECONDS

    # -- reset ----------------------------------------------------------------------

    def reset(self, data) -> None:
        """Restore the state captured at spawn, and forget any partial hold.

        World state persists between episodes and a finished one leaves the apple
        wherever it fell -- metres away, on the floor, in the bad cases. Inheriting that
        makes the next run score zero while looking like a policy failure.
        """
        if data is None:
            return
        # The snapshot is the whole qpos/qvel/ctrl, so the dressing comes back with
        # everything else -- there is nothing per-object to restore here, and a reader
        # looking for it should find this line instead.
        data.qpos[:] = self._spawn_qpos
        data.qvel[:] = self._spawn_qvel
        data.ctrl[:] = self._spawn_ctrl
        mujoco.mj_forward(self._model, data)
        self._holding_since = None


def _find_joint(model, suffix: str) -> int:
    """The id of the joint called `suffix`, whatever namespace an engine prefixed it."""
    exact = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, suffix)
    if exact >= 0:
        return exact
    matches = [
        i for i in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").endswith("/" + suffix)
    ]
    return matches[0] if len(matches) == 1 else -1


def _find_body(model, suffix: str) -> int | None:
    """The id of the body called `suffix`, whatever namespace an engine prefixed it with."""
    exact = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, suffix)
    if exact >= 0:
        return exact
    matches = [
        i for i in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("/" + suffix)
    ]
    return matches[0] if len(matches) == 1 else None
