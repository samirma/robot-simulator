"""A recorded robot as a MuJoCo spec, engine-neutral (one loader for both engines).

The robot specification designates each robot's model: its interface file's
`model.mjcf` (the official MJCF, or the derived `model.xml` beside it), following the
conventions of `robots_specs/SCHEMA.md` -- a mobile robot's top body carries
`<freejoint name="root"/>`, every actuated joint has one actuator of the same name,
cameras are named after their image frame ids, lidars and IMUs are sites named after
their frame ids, and a `home` keyframe holds the boot's initial configuration.
Every robot is a single body with one interface (simulator spec §2.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

import mjutil
import registry

try:  # PyYAML is in both engine venvs
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class ModelError(RuntimeError):
    pass


def load_interface(robot: registry.Robot) -> dict:
    if yaml is None:
        raise ModelError("PyYAML is not installed in this venv; run run.sh setup")
    path = robot.path(robot.ros_file)
    if path is None or not path.is_file():
        raise ModelError(f"{robot.id}: interface file {robot.ros_file} is missing; "
                         "run simulator/<engine>/run.sh setup")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_file(robot: registry.Robot) -> Path:
    """The MJCF to load: the interface file's `model.mjcf`, else the registry's choice."""
    try:
        iface = load_interface(robot)
        name = (iface.get("model") or {}).get("mjcf")
        if name:
            p = robot.folder_path / name
            if p.is_file():
                return p
    except ModelError:
        pass
    p = robot.model_path()
    if p is None or not p.is_file():
        raise ModelError(f"{robot.id}: no MuJoCo model in {robot.folder}; run "
                         "simulator/<engine>/run.sh setup")
    return p


@dataclass
class RobotModel:
    robot: registry.Robot
    spec: mujoco.MjSpec
    root: str                           # top body name (unprefixed)
    floating: bool                      # has the free joint "root"
    home: dict = field(default_factory=dict)   # joint name (robot-local) -> qpos (hinge/slide)
    #: the free joint's home pose in the robot's own frame: (pos(3), quat(4))
    home_root: tuple = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    def root_qpos(self, xyz, yaw: float) -> np.ndarray:
        """The free joint's qpos for the robot's frame placed at xyz with heading yaw."""
        pos = np.asarray(xyz, float) + mjutil.yaw_matrix(yaw) @ np.asarray(self.home_root[0], float)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array(mjutil.yaw_quat(yaw)), np.asarray(self.home_root[1], float))
        return np.concatenate([pos, q])


def _home(spec: mujoco.MjSpec):
    """The `home` keyframe: hinge/slide joint positions by name, and the free joint's
    pose; then the keyframes are removed so the spec attaches into a scene of any size."""
    home: dict = {}
    root = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    keys = list(spec.keys)
    m = spec.compile()
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, kid)
    for j in range(m.njnt):
        a = m.jnt_qposadr[j]
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            home[m.joint(j).name] = float(d.qpos[a])
        elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            root = (tuple(float(x) for x in d.qpos[a:a + 3]),
                    tuple(float(x) for x in d.qpos[a + 3:a + 7]))
    for k in keys:
        mjutil.spec_delete(spec, k)
    return home, root


def _top_body(spec: mujoco.MjSpec):
    bodies = [b for b in spec.worldbody.bodies]
    if len(bodies) != 1:
        raise ModelError(f"expected one top body in the robot model, found "
                         f"{[b.name for b in bodies]}")
    return bodies[0]


def _has_freejoint(body) -> bool:
    return any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints) or \
        any(True for _ in getattr(body, "freejoints", []) or [])


def load(robot: registry.Robot) -> RobotModel:
    """The robot's spec, ready to attach into a scene."""
    missing = registry.missing_files(robot)
    if missing:
        raise ModelError(f"{robot.id}: required files are missing ({', '.join(missing[:6])}"
                         f"{' ...' if len(missing) > 6 else ''}); run "
                         "simulator/<engine>/run.sh setup")
    spec = mujoco.MjSpec.from_file(str(model_file(robot)))
    home, home_root = _home(spec)
    top = _top_body(spec)
    return RobotModel(robot, spec, top.name, _has_freejoint(top), home, home_root)


# ---------------------------------------------------------------- geometry at rest


@dataclass
class Shape:
    """The robot's collision geometry at its home pose, in the frame it is attached by
    (the robot's own model frame: its free joint at its home pose)."""

    lo: np.ndarray          # xyz lower corner of all collision geoms
    hi: np.ndarray          # xyz upper corner
    radius: float           # max horizontal distance of any collision point from root
    cameras: list           # [(name, pos(3), mat(3x3), fovy_deg, (w, h))] in root frame
    footprint: np.ndarray   # (n, 2) xy points of the support contacts (lowest geoms)
    boxes: np.ndarray = None  # (n, 2, 3) lower and upper corner of each collision geom


def shape(rm: RobotModel) -> Shape:
    """Compile the robot alone, put it at home, and measure it."""
    spec = rm.spec.copy() if hasattr(rm.spec, "copy") else rm.spec
    m = spec.compile()
    d = mujoco.MjData(m)
    for name, q in rm.home.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = q
    if rm.floating:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")
        a = m.jnt_qposadr[j]
        d.qpos[a:a + 7] = rm.root_qpos([0.0, 0.0, 0.0], 0.0)
    mujoco.mj_forward(m, d)
    coll = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    glo, ghi = mjutil.geom_aabbs(m, d)
    boxes = np.stack([glo[coll], ghi[coll]], axis=1)
    # the eight corners of every collision geom's box
    corner = np.array([[sx, sy, sz] for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)], bool)
    pts = np.where(corner[None], boxes[:, 1:2], boxes[:, 0:1]).reshape(-1, 3) \
        if len(boxes) else np.zeros((1, 3))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    low = pts[pts[:, 2] < lo[2] + 0.01][:, :2]
    cams = []
    for c in range(m.ncam):
        res = tuple(int(x) for x in m.cam_resolution[c]) if hasattr(m, "cam_resolution") else (640, 480)
        cams.append((m.camera(c).name, d.cam_xpos[c].copy(), d.cam_xmat[c].reshape(3, 3).copy(),
                     float(m.cam_fovy[c]), res))
    radius = float(np.max(np.hypot(pts[:, 0], pts[:, 1])))
    return Shape(lo, hi, radius, cams, low, boxes)
