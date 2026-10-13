"""The six worktop objects staged on the worktop (spec §2.3 *Worktop objects*).

An apple, a plate, a bowl, a mug, a tomato and an egg, each the selected engine's own
model -- with its own visuals and its own collision geometry (a hollow bowl, mug and
plate, built from the engine's convex pieces or primitives) -- staged at the poses the
reference project (github.com/samirma/robot-simulator rev 34547ae, "REF" below:
`simulator/shared/tasks/apple_on_plate.py` lines 72-126) gave its apple-on-plate objects
in its arm's base frame. REF's lemon pose holds the tomato and its banana pose (with the
banana's own 0.785 rad heading) the egg. Each object's body origin is the bottom centre
of its collision geometry (the model's own horizontal origin), so every object starts
`LIFT` above the top face and settles onto it.

One fixed model per object and engine (`MODELS`):

* **MolmoSpaces** -- THOR prefabs from the engine's object library
  (`assets/objects/thor/Kitchen Objects/<Name>/Prefabs/<Name>_<n>/`), in the variant
  MolmoSpaces' own scene builder uses (`molmo_spaces/housegen/builder.py`
  `_get_thor_model_path`): `<prefab>_mesh.xml` (convex-decomposition colliders), except
  types in `TYPES_TO_USE_ONLY_PRIM` (the plate), which take `<prefab>_prim.xml`
  (primitive colliders). THOR is Y-up; the model turns Y onto Z as MolmoSpaces' scenes
  do. The bowl, mug and plate are the prefabs FloorPlan1 (`ithor:1`) itself uses; the
  apple, tomato and egg are the smallest of their category, the only ones of a real size
  (THOR's apples are 8-13 cm, the ROSMASTER X3 Plus gripper opens to 85 mm). THOR gives
  no masses that fit these objects (MolmoSpaces' scenes derive densities per scene), so
  each gets a real-world mass (`mass`), spread over its colliders.
* **RoboCasa** -- RoboCasa's own object registry (`robocasa/models/objects/
  kitchen_objects.py`), objaverse models, built as `robocasa_objects` builds RoboCasa's
  counter objects (`MJCFObject` with the category's scale, density, friction, solref and
  solimp). `run.sh setup` installs these six models alone (`tools/fetch_assets.py`
  source `objs_worktop`) into `assets/objects/worktop/objaverse/`, out of the registry
  folders RoboCasa's counter-object draw reads, so that draw is unchanged.

The staging logic is REF's: `plan_clear` reads which loose scene objects to clear off the
world as it is now, `Staging.apply`/`undo` edit the world's spec (the world recompiles),
and `Staging.capture`/`restore_state` carry each cleared object's state at the moment it
was cleared back onto it when the robot is removed. Nothing here reaches a wire.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

import mjutil

SIM = Path(__file__).resolve().parents[1]

#: The six objects, in staging order.
OBJECTS = ("apple", "plate", "bowl", "mug", "tomato", "egg")
#: Body name of each staged object (the control port, placement and the evidence
#: harness name them).
BODIES = {name: f"task_{name}" for name in OBJECTS}
APPLE_BODY = BODIES["apple"]
PLATE_BODY = BODIES["plate"]

#: REF 72-77 and 119-126: (x, y, heading) of each object in the robot's base frame; the
#: tomato stands at REF's lemon pose, the egg at its banana pose.
REF_POSES = {
    "apple": (0.30, 0.10, 0.0),
    "plate": (0.226, -0.226, 0.0),
    "bowl": (0.14, -0.40, 0.0),
    "mug": (0.05, 0.27, 0.0),
    "tomato": (0.07, -0.26, 0.0),     # REF's lemon
    "egg": (0.156, 0.156, 0.785),     # REF's banana
}

#: REF shared/placement.py:55 (`BASE_CLEARANCE`): the objects' frame is the robot's base
#: frame, which REF grafted this far above the top face.
FRAME_ABOVE_SURFACE = 0.004
#: each object's lowest point starts this far above the top face, and settles onto it
LIFT = 0.001
#: each object's pose in the base frame: its bottom `LIFT` above the top face
OBJECT_POSES = {name: (x, y, LIFT - FRAME_ABOVE_SURFACE) for name, (x, y, _) in REF_POSES.items()}

_R = math.sqrt(0.5)
#: THOR's Y-up onto MuJoCo's Z-up, as MolmoSpaces' scenes place THOR prefabs: a half turn
#: about (0, 1, 1)/sqrt(2) (x -> -x, y -> z, z -> y)
THOR_UP = (0.0, 0.0, _R, _R)
#: an upright THOR egg laid on its side (a quarter turn about x after `THOR_UP`): an egg
#: stands still only lying down
THOR_EGG_DOWN = (0.0, 0.0, 0.0, 1.0)
_THOR = "objects/thor/Kitchen Objects"
_WORKTOP_OBJAVERSE = "objects/worktop/objaverse"

#: One fixed model per object and engine: `path` under the engine's assets root; THOR:
#: `quat` the model's orientation in the object's body and `mass` (kg, an estimate of the
#: real object's); RoboCasa: `category` (its registry entry, for scale and contact).
MODELS = {
    "molmospaces": {
        "apple": {"path": f"{_THOR}/Apple/Prefabs/Apple_29/Apple_29_mesh.xml", "mass": 0.10},
        "plate": {"path": f"{_THOR}/Plate/Prefabs/Plate_1/Plate_1_prim.xml", "mass": 0.40},
        "bowl": {"path": f"{_THOR}/Bowl/Prefabs/Bowl_1/Bowl_1_mesh.xml", "mass": 0.35},
        "mug": {"path": f"{_THOR}/Mug/Prefabs/Mug_1/Mug_1_mesh.xml", "mass": 0.30},
        "tomato": {"path": f"{_THOR}/Tomato/Prefabs/Tomato_3/Tomato_3_mesh.xml", "mass": 0.25},
        "egg": {"path": f"{_THOR}/Egg/Prefabs/Egg_25/Egg_25_mesh.xml", "mass": 0.055,
                "lay": True},
    },
    "robocasa": {
        "apple": {"path": f"{_WORKTOP_OBJAVERSE}/apple/apple_11/model.xml", "category": "apple"},
        "plate": {"path": f"{_WORKTOP_OBJAVERSE}/plate/plate_2/model.xml", "category": "plate"},
        "bowl": {"path": f"{_WORKTOP_OBJAVERSE}/bowl/bowl_2/model.xml", "category": "bowl"},
        "mug": {"path": f"{_WORKTOP_OBJAVERSE}/mug/mug_2/model.xml", "category": "mug"},
        "tomato": {"path": f"{_WORKTOP_OBJAVERSE}/tomato/tomato_1/model.xml",
                   "category": "tomato"},
        "egg": {"path": f"{_WORKTOP_OBJAVERSE}/egg/egg_6/model.xml", "category": "egg"},
    },
}

#: The apple is the object an arm grasps and sets on the plate: its colliders carry REF's
#: contact tuning for a gripper's hold (REF `_stage_task_objects`: condim 6, high sliding
#: friction, a soft contact).
APPLE_CONTACT = {"condim": 6, "friction": [2.0, 0.05, 0.001], "solref": [0.03, 1.0],
                 "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]}

#: REF 298-302: the working area cleared of the scene's loose objects, and where they go.
CLEAR_RADIUS = 0.55
CLEAR_Z_BAND = (-0.15, 0.45)
SUNK_DEPTH = 50.0

_engine: str | None = None


class AssetsMissing(RuntimeError):
    pass


def set_engine(name: str) -> None:
    """The engine whose models are staged (`simulation.py` sets it from `--engine`)."""
    if name not in MODELS:
        raise ValueError(f"unknown engine {name!r}")
    global _engine
    _engine = name


def engine() -> str:
    """The engine set, else `ENGINE_NAME` (each engine's `env.sh`), else the one whose
    packages this interpreter has."""
    if _engine:
        return _engine
    env = os.environ.get("ENGINE_NAME")
    if env in MODELS:
        return env
    return "robocasa" if importlib.util.find_spec("robocasa") else "molmospaces"


def assets_root(name: str | None = None) -> Path:
    name = name or engine()
    if name == "molmospaces":
        return Path(os.environ.get("MLSPACES_ASSETS_DIR", SIM / "molmospaces" / "assets"))
    return Path(os.environ.get("ROBOCASA_ASSETS_DIR", SIM / "robocasa" / "assets"))


def model_path(obj: str, name: str | None = None) -> Path:
    name = name or engine()
    return assets_root(name) / MODELS[name][obj]["path"]


def missing_assets(name: str | None = None) -> list[str]:
    name = name or engine()
    return [MODELS[name][o]["path"] for o in OBJECTS if not model_path(o, name).is_file()]


def _quat_mul(a, b) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def base_frame(pos, yaw: float) -> np.ndarray:
    """REF 255-261: base frame -> world."""
    t = np.eye(4)
    t[:3, :3] = mjutil.yaw_matrix(yaw)
    t[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return t


def _apply(transform, point) -> list[float]:
    out = transform @ np.array([*point, 1.0], dtype=np.float64)
    return [float(v) for v in out[:3]]


def frame_of(xy, surface_z: float, yaw: float):
    """The objects' frame for a robot whose base stands at `xy` on a top face at
    `surface_z`, facing `yaw`: (position, yaw)."""
    return [float(xy[0]), float(xy[1]), float(surface_z) + FRAME_ABOVE_SURFACE], float(yaw)


def poses(frame_pos, yaw: float) -> dict:
    """World pose {name: (pos, quat)} of each object's body as staged."""
    t = base_frame(frame_pos, yaw)
    bq = mjutil.yaw_quat(yaw)
    return {name: (_apply(t, OBJECT_POSES[name]), _quat_mul(bq, mjutil.yaw_quat(REF_POSES[name][2])))
            for name in OBJECTS}


# ---------------------------------------------------------------- the models


def _absolute_files(spec, base: Path) -> None:
    """Asset files relative to the model's own folder made absolute, so the model can be
    attached into a spec read from elsewhere."""
    meshdir = base / (spec.meshdir or "")
    texdir = base / (spec.texturedir or "")
    for kind, folder in (("meshes", meshdir), ("textures", texdir)):
        for el in getattr(spec, kind):
            if el.file and not os.path.isabs(el.file):
                el.file = str((folder / el.file).resolve())
    spec.meshdir = ""
    spec.texturedir = ""


def _collidable(geom) -> bool:
    return bool(geom.contype or geom.conaffinity)


def _all_geoms(body):
    yield from body.geoms
    for child in body.bodies:
        yield from _all_geoms(child)


def _thor_source(model: dict):
    """A THOR prefab as a spec: its root body without the prefab's free joint, visuals
    massless, the colliders' density set so the object weighs `mass`."""
    path = model_path_from(model, "molmospaces")
    spec = mujoco.MjSpec.from_file(str(path))
    _absolute_files(spec, path.parent)
    root = spec.worldbody.first_body()
    for j in list(root.joints):
        mjutil.spec_delete(spec, j)
    for g in _all_geoms(root):
        if not _collidable(g):
            g.density = 0.0
            g.mass = 0.0
    m = spec.compile()
    total = float(m.body_subtreemass[1])
    if total > 0:
        for g in _all_geoms(root):
            if _collidable(g):
                g.density = g.density * model["mass"] / total
    quat = list(THOR_UP)
    if model.get("lay"):
        quat = _quat_mul([_R, _R, 0.0, 0.0], quat)   # a quarter turn about x
    return spec, root.name, quat


def _robocasa_source(model: dict, name: str):
    """A RoboCasa registry object, built as `robocasa_objects` builds the counters' (an
    `MJCFObject` with its category's settings, in a robosuite task), as a spec: its root
    body without its free joint."""
    import contextlib

    from robocasa.models.objects import kitchen_object_utils
    from robocasa.models.objects.objects import MJCFObject
    from robosuite.models.arenas import EmptyArena
    from robosuite.models.tasks import ManipulationTask

    import scenes

    scenes.point_robocasa_assets()
    # the category's objaverse settings (an `ObjCat` once kitchen_object_utils has built
    # the library, the raw table's dict before)
    cat = kitchen_object_utils.OBJ_CATEGORIES[model["category"]]["objaverse"]
    if hasattr(cat, "get_mjcf_kwargs"):
        kwargs = cat.get_mjcf_kwargs()
    else:
        kwargs = {k: v for k, v in cat.items() if k in ("scale", "solimp", "solref", "density",
                                                          "friction", "priority")}
    path = model_path_from(model, "robocasa")
    obj = MJCFObject(name=f"wt_{name}", mjcf_path=str(path), **kwargs)
    with contextlib.redirect_stdout(sys.stderr):
        task = ManipulationTask(mujoco_arena=EmptyArena(), mujoco_robots=[],
                                mujoco_objects=[obj])
        spec = mujoco.MjSpec.from_string(task.get_xml())
    root = spec.body(obj.root_body)
    for j in list(root.joints):
        mjutil.spec_delete(spec, j)
    return spec, root.name, [1.0, 0.0, 0.0, 0.0]


def model_path_from(model: dict, name: str) -> Path:
    return assets_root(name) / model["path"]


_sources: dict = {}


def _source(name: str):
    """(spec, root body name, orientation quat, bottom offset) of the engine's model of
    object `name`, built once per process; callers attach from a copy."""
    eng = engine()
    key = (eng, name)
    if key not in _sources:
        model = MODELS[eng][name]
        if not model_path_from(model, eng).is_file():
            raise AssetsMissing(f"the worktop {name}'s model is missing "
                                f"({model_path_from(model, eng)}); run "
                                f"simulator/{eng}/run.sh setup")
        spec, root, quat = (_thor_source(model) if eng == "molmospaces"
                            else _robocasa_source(model, name))
        if name == "apple":
            for g in _all_geoms(spec.body(root)):
                if _collidable(g):
                    for k, v in APPLE_CONTACT.items():
                        setattr(g, k, v)
        _sources[key] = (spec, root, quat, _bottom(spec, root, quat))
    return _sources[key]


def _bottom(spec, root: str, quat) -> float:
    """The lowest point of the model's colliders, turned by `quat`, in its own frame."""
    probe = mujoco.MjSpec()
    frame = probe.worldbody.add_frame(quat=list(quat))
    frame.attach_body(spec.copy().body(root), "probe/", "")
    m = probe.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo = np.inf
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            lo = min(lo, float(v[:, 2].min()))
        else:
            glo, _ = mjutil.geom_aabbs(m, d)
            lo = min(lo, float(glo[g, 2]))
    return lo


# ---------------------------------------------------------------- clearing


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
        if any(inside(data.xpos[k]) for k in mjutil.subtree(model, b)):
            moved.append(name)
            gone.add(b)
    return moved


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


def staged_bodies(model) -> dict:
    """{body id: object name} of every body of the staged objects (each object's body
    and the model's bodies inside it)."""
    out = {}
    for name, body in BODIES.items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        for k in mjutil.subtree(model, b):
            out[k] = name
    return out


# ---------------------------------------------------------------- staging


class Staging:
    """The six objects staged in a robot's base frame, and the scene objects cleared for
    them. `apply` and `undo` edit a spec; the world recompiles around them."""

    def __init__(self, frame_pos, yaw: float, cleared=()):
        self.frame_pos = [float(v) for v in frame_pos]
        self.yaw = float(yaw)
        self.cleared = list(cleared)
        self._sunk: dict = {}          # name -> (original pos, [joint name, props])
        self._state: dict = {}         # name -> (qpos(7), qvel(6)) when cleared

    # -- reading
    def poses(self) -> dict:
        return poses(self.frame_pos, self.yaw)

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
                continue
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
                raise AssetsMissing(f"the worktop objects' models are missing ({missing[0]} ...); "
                                    f"run simulator/{engine()}/run.sh setup")
        self._sunk = {}
        for name in self.cleared:
            body = spec.body(name)
            if body is None:
                continue
            joints = []
            for joint in list(body.joints):
                if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                    joints.append((joint.name, _joint_props(joint)))
                    mjutil.spec_delete(spec, joint)
            self._sunk[name] = ([float(v) for v in body.pos], joints)
            body.pos = [body.pos[0], body.pos[1], body.pos[2] - SUNK_DEPTH]
        self.add_objects(spec)

    def add_objects(self, spec) -> None:
        """The six objects alone, at this staging's frame (no clearing): each a body with
        a free joint at its pose, the engine's model attached inside it, turned upright
        and lowered so the body origin is the model's lowest point."""
        for name, (pos, quat) in self.poses().items():
            src, root, mquat, bottom = _source(name)
            body = spec.worldbody.add_body(name=BODIES[name], pos=pos, quat=quat)
            body.add_freejoint(name=f"{BODIES[name]}_joint")
            frame = body.add_frame(pos=[0.0, 0.0, -bottom], quat=list(mquat))
            frame.attach_body(src.copy().body(root), f"{BODIES[name]}/", "")

    def remove_objects(self, spec) -> None:
        """Take the six objects (and their assets) out of the spec, found by name so it
        also works on a copy of the spec they were staged into; the cleared scene objects
        stay cleared."""
        for body in BODIES.values():
            el = spec.body(body)
            if el is not None:
                mjutil.spec_delete(spec, el)
        prefixes = tuple(f"{b}/" for b in BODIES.values())
        for kind in ("meshes", "materials", "textures"):
            for el in list(getattr(spec, kind)):
                if el.name.startswith(prefixes):
                    try:
                        mjutil.spec_delete(spec, el)
                    except Exception:
                        pass  # MuJoCo 3.3: an unused asset may stay in the spec

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


def main(argv=None) -> int:
    """`python worktop_objects.py check [<engine>]`: exit non-zero, naming them, when the
    engine's six models are not installed (each engine's `run.sh setup` runs it)."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("engine", nargs="?", default=None, choices=sorted(MODELS))
    args = ap.parse_args(argv)
    eng = args.engine or engine()
    missing = missing_assets(eng)
    if missing:
        print(f"the worktop objects' models are missing on {eng}: {', '.join(missing)}",
              file=sys.stderr)
        return 1
    for o in OBJECTS:
        print(f"{o}: {model_path(o, eng)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
