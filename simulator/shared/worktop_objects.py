"""The objects staged on the worktop around a worktop robot (spec §2.3 *Worktop objects*).

A port of the reference project's apple-on-plate scene staging
(github.com/samirma/robot-simulator rev 34547ae, `simulator/shared/tasks/apple_on_plate.py`,
"REF" below): the same six YCB objects -- the apple and the plate, and the bowl, mug,
banana and lemon -- at the same poses in the robot's base frame, with the same physics and
visuals, and the same clearing of the scene's own loose objects from the working area.

* poses and constants: REF lines 47-136 and 231-249 (the plate's 24-box rim);
* the apple and plate: REF `_stage_task_objects` (415-516) -- the apple a 20 mm sphere
  carrying the tuned contact parameters under an invisible-to-physics textured mesh, the
  plate a fixed body (no free joint) of a cylinder and its rim;
* the distractors: REF's loop at lines 370-396 (a group-2 visual mesh and a group-0
  collision mesh of the same scan, condim 6);
* the assets: REF `_add_assets` (520-559);
* clearing: REF `_clear_workspace` (591-651) -- every loose (free-joint) scene object any
  part of which is within `CLEAR_RADIUS` of the base, in the band `CLEAR_Z_BAND` about the
  surface, loses its free joint and is parked `SUNK_DEPTH` under the scene.

Not ported: REF's solver options (elliptic cones, impratio, no-slip iterations), its rig
cameras, `/reset` and the truth log. Nothing here reaches a wire.

Unlike REF, which staged once into a scene before its only compile, this staging is added
to a running world and taken out again: `plan_clear` reads which objects to clear off the
world as it is now, `Staging.apply`/`undo` edit the world's spec (the world recompiles),
and `Staging.capture`/`restore_state` carry each cleared object's state at the moment it
was cleared back onto it when the robot is removed.

The YCB meshes are fetched by each engine's `run.sh setup` into `objects/ycb/` (see
`tools/fetch_objects.py`, verified against `objects/ycb.sha256`).
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

#: The YCB scans (objects/LICENSES.md), fetched by `run.sh setup`.
ASSETS = Path(__file__).resolve().parent / "objects" / "ycb"
#: REF 47-52: the apple mesh scaled to its 20 mm sphere, the plate's to its cylinder, and
#: the plate's visual lifted onto the cylinder's top face.
APPLE_MESH_SCALE = 0.50
PLATE_MESH_SCALE = 0.77
PLATE_MESH_Z = 0.0092

#: REF 72-77: (name, position, yaw, mass, mesh scale) in the robot's base frame.
DISTRACTORS = (
    ("bowl", (0.14, -0.40, 0.0271), 0.0, 0.147, 1.0),
    ("mug", (0.05, 0.27, 0.0272), 0.0, 0.118, 1.0),
    ("banana", (0.156, 0.156, 0.0172), 0.785, 0.066, 1.0),
    ("lemon", (0.07, -0.26, 0.0294), 0.0, 0.029, 1.0),
)
DISTRACTOR_CONDIM = 6
DISTRACTOR_FRICTION = (1.0, 0.1, 0.02)

APPLE_BODY = "task_apple"
PLATE_BODY = "task_plate"
APPLE_SPAWN = (0.30, 0.10, 0.020)
APPLE_RADIUS = 0.020
APPLE_MASS = 0.020
PLATE_CENTRE = (0.226, -0.226, 0.0)
PLATE_RADIUS = 0.10
PLATE_HALF_HEIGHT = 0.0102

#: REF 119-126: the six objects, in staging order, and where each is staged.
OBJECTS = ("apple", "plate", *(d[0] for d in DISTRACTORS))
OBJECT_POSES = {"apple": APPLE_SPAWN, "plate": PLATE_CENTRE, **{d[0]: d[1] for d in DISTRACTORS}}
#: Body name of each staged object.
BODIES = {name: f"task_{name}" for name in OBJECTS}
#: REF 133-136: each object's footprint radius on the worktop about its origin.
FOOTPRINT_RADIUS = {"apple": 0.021, "plate": 0.103, "bowl": 0.084, "mug": 0.069,
                    "banana": 0.108, "lemon": 0.032}

#: REF 231-237: the plate's rim, 24 boxes around a cone.
_RIM_N = 24
_RIM_HALF_LEN = 0.022270
_RIM_WIDTH = 0.013966
_RIM_THICKNESS = 0.004000
_RIM_RADIUS = 0.083063
_RIM_Z = 0.025418
_RIM_ALPHA = 0.4014

#: REF 298-302: the working area cleared of the scene's loose objects, and where they go.
CLEAR_RADIUS = 0.55
CLEAR_Z_BAND = (-0.15, 0.45)
SUNK_DEPTH = 50.0

#: REF shared/placement.py:55 (`BASE_CLEARANCE`): the objects' frame is the robot's base
#: frame, which REF grafted this far above the top face.
FRAME_ABOVE_SURFACE = 0.004


class AssetsMissing(RuntimeError):
    pass


def missing_assets() -> list[str]:
    out = []
    for name in OBJECTS:
        for f in ("textured.obj", "texture_map.png"):
            if not (ASSETS / name / f).is_file():
                out.append(f"objects/ycb/{name}/{f}")
    return out


def _rim_geoms():
    """REF 240-249: (name, pos, quat, size) of each rim box, in the plate frame."""
    for i in range(_RIM_N):
        phi = 2.0 * math.pi * i / _RIM_N
        cz, sz = math.cos(phi / 2), math.sin(phi / 2)
        cy, sy = math.cos(-_RIM_ALPHA / 2), math.sin(-_RIM_ALPHA / 2)
        quat = (cz * cy, -sz * sy, cz * sy, sz * cy)
        pos = (_RIM_RADIUS * math.cos(phi), _RIM_RADIUS * math.sin(phi), _RIM_Z)
        yield f"plate_rim_{i:02d}", pos, quat, (_RIM_HALF_LEN, _RIM_WIDTH, _RIM_THICKNESS)


def base_frame(pos, yaw: float) -> np.ndarray:
    """REF 255-261: base frame -> world."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    t = np.eye(4)
    t[:3, :3] = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    t[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return t


def _apply(transform, point) -> list[float]:
    out = transform @ np.array([*point, 1.0], dtype=np.float64)
    return [float(v) for v in out[:3]]


def _yaw_quat(yaw: float) -> list[float]:
    return [float(math.cos(yaw / 2)), 0.0, 0.0, float(math.sin(yaw / 2))]


def _quat_mul(a, b) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def frame_of(xy, surface_z: float, yaw: float):
    """The objects' frame for a robot whose base stands at `xy` on a top face at
    `surface_z`, facing `yaw`: (position, yaw)."""
    return [float(xy[0]), float(xy[1]), float(surface_z) + FRAME_ABOVE_SURFACE], float(yaw)


def poses(frame_pos, yaw: float) -> dict:
    """World pose {name: (pos, quat)} of each object as staged."""
    t = base_frame(frame_pos, yaw)
    bq = _yaw_quat(yaw)
    out = {"apple": (_apply(t, OBJECT_POSES["apple"]), bq),
           "plate": (_apply(t, OBJECT_POSES["plate"]), bq)}
    for name, pos, obj_yaw, _m, _s in DISTRACTORS:
        out[name] = (_apply(t, pos), _quat_mul(bq, _yaw_quat(obj_yaw)))
    return out


# ---------------------------------------------------------------- clearing


def _subtree(model, root: int) -> list[int]:
    out = []
    for b in range(root, model.nbody):
        p = b
        while p > 0 and p != root:
            p = model.body_parentid[p]
        if p == root:
            out.append(b)
    return out


def plan_clear(model, data, frame_pos, yaw: float, keep=()) -> list[str]:
    """REF `_clear_workspace` (591-651), read off the world as it is now: the names of the
    loose scene bodies (a free joint, not staged, not part of a robot -- `keep` lists the
    robots' body ids) any of whose body origins lies within `CLEAR_RADIUS` of the base,
    horizontally, in the band `CLEAR_Z_BAND` about it. A body inside a cleared one goes
    with it and is not listed."""
    inverse = np.linalg.inv(base_frame(frame_pos, yaw))
    keep = set(keep)
    moved: list[str] = []
    gone: set[int] = set()

    def inside(point) -> bool:
        local = inverse @ np.array([*point, 1.0])
        if not CLEAR_Z_BAND[0] < local[2] < CLEAR_Z_BAND[1]:
            return False
        return float(np.hypot(local[0], local[1])) <= CLEAR_RADIUS

    for b in range(1, model.nbody):        # parents come before their children
        if model.body_parentid[b] in gone:
            gone.add(b)                        # goes with its cleared parent
            continue
        if b in keep:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        adr, num = model.body_jntadr[b], model.body_jntnum[b]
        movable = any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(adr, adr + num))
        if not (name and movable) or name.startswith("task_"):
            continue
        if any(inside(data.xpos[k]) for k in _subtree(model, b)):
            moved.append(name)
            gone.add(b)
    return moved


def spec_delete(spec, element) -> None:
    """REF 578-588: MuJoCo 3.5 has `spec.delete(element)`, 3.3 `element.delete()`."""
    if hasattr(spec, "delete"):
        spec.delete(element)
    else:
        element.delete()


_JOINT_SKIP = {"id", "signature", "type", "name", "classname", "frame"}


def _joint_props(joint) -> dict:
    """The settable properties of a spec joint, to recreate it as it was."""
    out = {}
    for k in dir(joint):
        if k.startswith("_") or k in _JOINT_SKIP:
            continue
        try:
            v = getattr(joint, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = np.array(v).copy() if isinstance(v, np.ndarray) else v
    return out


# ---------------------------------------------------------------- staging


class Staging:
    """The six objects staged in a robot's base frame, and the scene objects cleared for
    them. `apply` and `undo` edit a spec; the world recompiles around them."""

    def __init__(self, frame_pos, yaw: float, cleared=()):
        self.frame_pos = [float(v) for v in frame_pos]
        self.yaw = float(yaw)
        self.cleared = list(cleared)
        self._bodies: list = []
        self._assets: list = []
        self._sunk: dict = {}          # name -> (original pos, [joint name, props])
        self._state: dict = {}         # name -> (qpos(7), qvel(6)) when cleared

    # -- reading
    def poses(self) -> dict:
        return poses(self.frame_pos, self.yaw)

    def describe(self) -> dict:
        return {"frame": {"pos": [round(v, 6) for v in self.frame_pos], "yaw": self.yaw},
                "staged": {n: {"pos": p, "quat": q} for n, (p, q) in self.poses().items()},
                "cleared": list(self.cleared)}

    # -- state of the cleared objects
    def capture(self, model, data) -> None:
        """Before `apply`: each cleared object's free-joint state now."""
        self._state = {}
        for name in self.cleared:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    a, v = model.jnt_qposadr[j], model.jnt_dofadr[j]
                    self._state[name] = (data.qpos[a:a + 7].copy(), data.qvel[v:v + 6].copy())

    def restore_state(self, model, data) -> None:
        """After `undo` has been compiled: each cleared object back where it was cleared
        from, moving as it was."""
        for name, (q, v) in self._state.items():
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                continue
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 7] = q
                    data.qvel[model.jnt_dofadr[j]:model.jnt_dofadr[j] + 6] = v

    def place_objects(self, model, data) -> None:
        """Put the six objects at their staged poses, at rest: a recompile carries a
        free body's state over by name, and these names were staged elsewhere before."""
        for name, (pos, quat) in self.poses().items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"task_{name}_joint")
            if j < 0:
                continue                      # the plate is fixed
            a, v = model.jnt_qposadr[j], model.jnt_dofadr[j]
            data.qpos[a:a + 7] = [*pos, *quat]
            data.qvel[v:v + 6] = 0.0

    # -- editing a spec
    def apply(self, spec, check_assets: bool = True) -> None:
        """Clear, then stage: REF `stage()` (305-412) without its solver options and rig
        cameras."""
        if check_assets:
            missing = missing_assets()
            if missing:
                raise AssetsMissing(f"the worktop objects' meshes are missing ({missing[0]} ...); "
                                    "run simulator/<engine>/run.sh setup")
        self._sunk = {}
        for name in self.cleared:
            body = spec.body(name)
            if body is None:
                continue
            joints = []
            for joint in list(body.joints):
                if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                    joints.append((joint.name, _joint_props(joint)))
                    spec_delete(spec, joint)
            self._sunk[name] = ([float(v) for v in body.pos], joints)
            body.pos = [body.pos[0], body.pos[1], body.pos[2] - SUNK_DEPTH]
        self.add_objects(spec)

    def add_objects(self, spec) -> None:
        """The six objects alone, at this staging's frame (no clearing)."""
        self._add_assets(spec)
        t = base_frame(self.frame_pos, self.yaw)
        bq = _yaw_quat(self.yaw)
        self._stage_apple_plate(spec, t, bq)
        for name, pos, obj_yaw, mass, _scale in DISTRACTORS:
            body = spec.worldbody.add_body(name=f"task_{name}", pos=_apply(t, pos),
                                           quat=_quat_mul(bq, _yaw_quat(obj_yaw)))
            body.add_freejoint(name=f"task_{name}_joint")
            body.add_geom(name=f"task_{name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=f"task_{name}_vis", material=f"task_{name}_mat",
                          contype=0, conaffinity=0, density=0.0, group=2)
            body.add_geom(name=f"task_{name}_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=f"task_{name}_vis", mass=mass, condim=DISTRACTOR_CONDIM,
                          friction=list(DISTRACTOR_FRICTION), rgba=[1.0, 1.0, 1.0, 0.0], group=0)
            self._bodies.append(body)

    def remove_objects(self, spec) -> None:
        """Take the six objects (and their assets) out of the spec, found by name so it
        also works on a copy of the spec they were staged into; the cleared scene objects
        stay cleared."""
        from robot_model import _delete

        for body in self._bodies:
            _delete(spec, body)
        for name in BODIES.values():
            body = spec.body(name)
            if body is not None:
                _delete(spec, body)
        self._bodies = []
        for el in self._assets:
            try:
                spec_delete(spec, el)
            except Exception:
                pass  # MuJoCo 3.3: an unused asset may stay in the spec
        self._assets = []
        for kind in ("mesh", "texture", "material"):
            for name in OBJECTS:
                for asset in (f"task_{name}_vis", f"task_{name}_tex", f"task_{name}_mat"):
                    el = getattr(spec, kind)(asset)
                    if el is not None:
                        try:
                            spec_delete(spec, el)
                        except Exception:
                            pass

    def undo(self, spec) -> None:
        """Take the staged objects out of the spec and put the cleared ones back."""
        self.remove_objects(spec)
        for name, (pos, joints) in self._sunk.items():
            body = spec.body(name)
            if body is None:
                continue
            body.pos = pos
            for jname, props in joints:
                j = body.add_freejoint()
                if jname:
                    j.name = jname
                for k, v in props.items():
                    try:
                        setattr(j, k, v)
                    except Exception:
                        pass
        self._sunk = {}

    def _stage_apple_plate(self, spec, t, bq) -> None:
        """REF `_stage_task_objects` (415-516)."""
        apple = spec.worldbody.add_body(name=APPLE_BODY, pos=_apply(t, OBJECT_POSES["apple"]),
                                        quat=bq)
        apple.add_freejoint(name=f"{APPLE_BODY}_joint")
        apple.add_geom(name=f"{APPLE_BODY}_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                       meshname="task_apple_vis", material="task_apple_mat", contype=0,
                       conaffinity=0, density=0.0, group=2)
        apple.add_geom(name=f"{APPLE_BODY}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[APPLE_RADIUS, 0.0, 0.0], mass=APPLE_MASS, condim=6,
                       friction=[2.0, 0.05, 0.001], solref=[0.03, 1.0],
                       solimp=[0.95, 0.99, 0.001, 0.5, 2.0], rgba=[1.0, 0.0, 0.0, 0.0], group=0)
        plate = spec.worldbody.add_body(name=PLATE_BODY, pos=_apply(t, OBJECT_POSES["plate"]),
                                        quat=bq)
        plate.add_geom(name=f"{PLATE_BODY}_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                       meshname="task_plate_vis", material="task_plate_mat",
                       pos=[0.0, 0.0, PLATE_MESH_Z], contype=0, conaffinity=0, density=0.0,
                       group=2)
        plate.add_geom(name=f"{PLATE_BODY}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[PLATE_RADIUS, PLATE_HALF_HEIGHT, 0.0], pos=[0.0, 0.0, PLATE_HALF_HEIGHT],
                       condim=4, friction=[1.5, 0.05, 0.001], rgba=[1.0, 1.0, 1.0, 0.0], group=0)
        for name, pos, quat, size in _rim_geoms():
            plate.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size),
                           pos=list(pos), quat=list(quat), condim=4, friction=[1.5, 0.05, 0.001],
                           rgba=[1.0, 1.0, 1.0, 0.0], group=0)
        self._bodies += [apple, plate]

    def _add_assets(self, spec) -> None:
        """REF `_add_assets` (520-559)."""
        add = self._assets.append
        for name, _pos, _yaw, _mass, scale in DISTRACTORS:
            mesh = spec.add_mesh(name=f"task_{name}_vis")
            mesh.file = str(ASSETS / name / "textured.obj")
            mesh.scale = [scale] * 3
            tex = spec.add_texture(name=f"task_{name}_tex")
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            tex.file = str(ASSETS / name / "texture_map.png")
            mat = spec.add_material(name=f"task_{name}_mat")
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"task_{name}_tex"
            add(mesh), add(tex), add(mat)
        apple_mesh = spec.add_mesh(name="task_apple_vis")
        apple_mesh.file = str(ASSETS / "apple" / "textured.obj")
        apple_mesh.scale = [APPLE_MESH_SCALE] * 3
        plate_mesh = spec.add_mesh(name="task_plate_vis")
        plate_mesh.file = str(ASSETS / "plate" / "textured.obj")
        plate_mesh.scale = [PLATE_MESH_SCALE] * 3
        tex = spec.add_texture(name="task_apple_tex")
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = str(ASSETS / "apple" / "texture_map.png")
        apple_mat = spec.add_material(name="task_apple_mat")
        apple_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "task_apple_tex"
        apple_mat.specular = 0.35
        apple_mat.shininess = 0.5
        plate_mat = spec.add_material(name="task_plate_mat")
        plate_mat.rgba = [0.97, 0.97, 0.95, 1.0]
        plate_mat.specular = 0.55
        plate_mat.shininess = 0.75
        plate_mat.reflectance = 0.08
        for el in (apple_mesh, plate_mesh, tex, apple_mat, plate_mat):
            add(el)
