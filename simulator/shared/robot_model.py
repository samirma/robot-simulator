"""A recorded robot as a MuJoCo spec, engine-neutral (one loader for both engines).

The robot specification designates each robot's model: its interface file's
`model.mjcf` (the official MJCF, or the derived `model.xml` beside it), following the
conventions of `robots_specs/SCHEMA.md` -- a mobile robot's top body carries
`<freejoint name="root"/>`, every actuated joint has one actuator of the same name,
cameras are named after their image frame ids, lidars and IMUs are sites named after
their frame ids, and a `home` keyframe holds the boot's initial configuration.

A composite (`base` + `arm`) is the base model with the arm model attached at the
recorded mounting transform, as one physical body; the arm's elements are prefixed
`arm/` inside the robot so the two components never collide on names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

import registry

try:  # PyYAML is in both engine venvs
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ARM_PREFIX = "arm/"


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


def _rpy_to_quat(rpy) -> list[float]:
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


@dataclass
class Component:
    """One interface-bearing part of a spawned robot (the robot itself, or base/arm)."""

    role: str              # "main", "base" or "arm"
    robot: registry.Robot  # whose interface file this component serves
    prefix: str            # name prefix of its elements inside the robot ("" or "arm/")


@dataclass
class RobotModel:
    robot: registry.Robot
    spec: mujoco.MjSpec
    root: str                           # top body name (unprefixed)
    floating: bool                      # has the free joint "root"
    components: list[Component]
    home: dict = field(default_factory=dict)   # joint name (robot-local) -> qpos (hinge/slide)
    #: the free joint's home pose in the robot's own frame: (pos(3), quat(4))
    home_root: tuple = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    def root_qpos(self, xyz, yaw: float) -> np.ndarray:
        """The free joint's qpos for the robot's frame placed at xyz with heading yaw."""
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pos = np.asarray(xyz, float) + R @ np.asarray(self.home_root[0], float)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]),
                           np.asarray(self.home_root[1], float))
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
        _delete(spec, k)
    return home, root


def _delete(spec: mujoco.MjSpec, element) -> None:
    if hasattr(spec, "delete"):
        spec.delete(element)
    elif isinstance(element, mujoco.MjsBody) and hasattr(spec, "detach_body"):
        spec.detach_body(element)
    else:  # MuJoCo 3.3: elements without a detach are removed through their own API
        element.delete() if hasattr(element, "delete") else None


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
    if robot.composite:
        base_r, arm_r = registry.get(robot.base), registry.get(robot.arm)
        base = mujoco.MjSpec.from_file(str(model_file(base_r)))
        arm = mujoco.MjSpec.from_file(str(model_file(arm_r)))
        home, home_root = _home(base)
        arm_home, _ = _home(arm)
        link, xyz, rpy = robot.mounting_transform()
        parent = base.body(link)
        if parent is None:
            raise ModelError(f"{robot.id}: mounting link {link!r} is not a body of the "
                             f"{base_r.id} model")
        arm_top = _top_body(arm)
        if getattr(robot, "mount_arm_link", arm_top.name) != arm_top.name:
            raise ModelError(f"{robot.id}: the mounting transform names arm link "
                             f"{robot.mount_arm_link!r}, but the {arm_r.id} model's root is "
                             f"{arm_top.name!r}")
        if _has_freejoint(arm_top):
            raise ModelError(f"{robot.id}: the arm model {arm_r.id} carries a free joint")
        frame = parent.add_frame(pos=list(xyz), quat=_rpy_to_quat(rpy))
        frame.attach_body(arm_top, ARM_PREFIX, "")
        home.update({ARM_PREFIX + k: v for k, v in arm_home.items()})
        top = _top_body(base)
        comps = [Component("base", base_r, ""), Component("arm", arm_r, ARM_PREFIX)]
        return RobotModel(robot, base, top.name, _has_freejoint(top), comps, home, home_root)
    spec = mujoco.MjSpec.from_file(str(model_file(robot)))
    home, home_root = _home(spec)
    top = _top_body(spec)
    return RobotModel(robot, spec, top.name, _has_freejoint(top),
                      [Component("main", robot, "")], home, home_root)


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
    R0 = np.eye(3)
    p0 = np.zeros(3)
    pts = []
    boxes = []
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g][:3]
        e = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g][3:]
        boxes.append((c - e, c + e))
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(R0.T @ (c + e * np.array([sx, sy, sz]) - p0))
    pts = np.array(pts) if pts else np.zeros((1, 3))
    boxes = np.array(boxes, float).reshape(-1, 2, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    low = pts[pts[:, 2] < lo[2] + 0.01][:, :2]
    cams = []
    for c in range(m.ncam):
        pos = R0.T @ (d.cam_xpos[c] - p0)
        mat = R0.T @ d.cam_xmat[c].reshape(3, 3)
        res = tuple(int(x) for x in m.cam_resolution[c]) if hasattr(m, "cam_resolution") else (640, 480)
        cams.append((m.camera(c).name, pos, mat, float(m.cam_fovy[c]), res))
    radius = float(np.max(np.hypot(pts[:, 0], pts[:, 1])))
    return Shape(lo, hi, radius, cams, low, boxes)
