"""The compiled robots against their descriptions and their published figures (spec §3, §5).

Run from an engine's venv, with the engine named:

    molmospaces/.venv/bin/python shared/tests/physical_figures_check.py --engine molmospaces
    robocasa/.venv/bin/python    shared/tests/physical_figures_check.py --engine robocasa

Each simulated robot alone in the engine's default scene, built the way `kitchen.sh serve`
builds it (the shared spawn tool, grafted by the engine's own adapter), compiled, and held
to two things:

* **its descriptions** -- the compiled model reproduces the geometry, joints and limits of
  the URDF and MJCF in `robots_specs/<id>/`: every joint's axis and limits, and every
  moving link's pose over sampled configurations by forward kinematics composed straight
  from the URDF's joint origins (never through MuJoCo), plus, where an official MJCF
  exists, every body frame, inertial and joint of it exactly. Where the compiled model
  departs, the departure is named here with the higher source that orders it (spec §3's
  precedence), or the check fails;
* **the manufacturer's published figures** -- every `PHYSICAL_FIGURES` entry of the
  robot's contract module, measured on the compiled model and held to its tolerance.

It prints one table row per figure: published value, source, model value, tolerance.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import placement_check as pc  # noqa: E402  (puts shared/ on the path)

import robots_spec  # noqa: E402
from mujoco_bridge import PlanarSetpoint  # noqa: E402
from ros_surfaces import myagv as myagv_contract  # noqa: E402
from ros_surfaces import so101 as so101_contract  # noqa: E402
from ros_surfaces.ainex import gait, servos  # noqa: E402
from ros_surfaces.ainex import topics as ainex_contract  # noqa: E402

check = pc.check

CONTRACTS = {"myagv": myagv_contract, "so101": so101_contract, "ainex": ainex_contract}

#: The root link of each robot's URDF, and the compiled body that is its frame.
ROOTS = {"so101": ("base_link", "base"), "myagv": ("base_footprint", "base"),
         "ainex": ("body_link", "body_link")}

#: URDF link -> compiled body, where the names differ (the official SO-101 files' own
#: pairing; the AiNex's MJCF is imported from its URDF and keeps the link names).
LINK_BODIES = {
    "so101": {"base_link": "base", "shoulder_link": "shoulder",
              "upper_arm_link": "upper_arm", "lower_arm_link": "lower_arm",
              "wrist_link": "wrist", "gripper_link": "gripper",
              "moving_jaw_so101_v1_link": "moving_jaw_so101_v1"},
    # The chassis and its top deck are one body; see RIGID_URDF_JOINTS.
    "myagv": {"base_footprint": "base", "base_up": "base"},
}

#: URDF joints the compiled model holds rigid, and why that is the robot. Anything else
#: missing from the compiled model fails.
RIGID_URDF_JOINTS = {
    "myagv": {
        "base_up": "the vendor's visualisation-only `continuous` joint between the chassis "
                   "and its top deck, at an identity origin; nothing drives it and the boot "
                   "launch's joint_state_publisher publishes it at 0, so the deck is part of "
                   "the base body at that angle",
    },
}

#: FK agreement with the URDF: the URDF writes its origins to 5-6 significant digits.
URDF_POS_TOL = 1e-4  # m
URDF_ROT_TOL = 1e-3  # rad
#: Agreement with an official MJCF the compiled model is built from: exact but for the
#: float rounding of a re-serialised file.
MJCF_TOL = 1e-9


# ------------------------------------------------------------------------------ helpers


def _floats(text: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


def _rpy(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis, angle: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


class Urdf:
    """The URDF's joints, and its forward kinematics composed from their origins."""

    def __init__(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        self.links = [link.get("name") for link in root.findall("link")]
        self.joints = {}
        for j in root.findall("joint"):
            origin = j.find("origin")
            limit = j.find("limit")
            axis = j.find("axis")
            self.joints[j.get("name")] = {
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": _floats(origin.get("xyz") if origin is not None else None),
                "rpy": _floats(origin.get("rpy") if origin is not None else None),
                "axis": _floats(axis.get("xyz") if axis is not None else None, "1 0 0"),
                "limit": None if limit is None or j.get("type") != "revolute" else
                (float(limit.get("lower")), float(limit.get("upper"))),
            }
        self.masses = {}
        for link in root.findall("link"):
            mass = link.find("inertial/mass")
            self.masses[link.get("name")] = float(mass.get("value")) if mass is not None else 0.0

    @property
    def moving(self) -> dict:
        return {n: j for n, j in self.joints.items() if j["type"] in ("revolute", "continuous")}

    def fk(self, q: dict[str, float], base: str) -> dict[str, np.ndarray]:
        """4x4 of every link in the frame of link `base`."""
        by_child = {j["child"]: (n, j) for n, j in self.joints.items()}
        poses: dict[str, np.ndarray] = {}

        def pose(link: str) -> np.ndarray:
            if link in poses:
                return poses[link]
            if link not in by_child:
                poses[link] = np.eye(4)
                return poses[link]
            name, j = by_child[link]
            t = np.eye(4)
            t[:3, 3] = j["xyz"]
            t[:3, :3] = _rpy(j["rpy"])
            if j["type"] in ("revolute", "continuous"):
                m = np.eye(4)
                m[:3, :3] = _axis_angle(j["axis"], q.get(name, 0.0))
                t = t @ m
            poses[link] = pose(j["parent"]) @ t
            return poses[link]

        for link in self.links:
            pose(link)
        inv = np.linalg.inv(poses[base])
        return {link: inv @ p for link, p in poses.items()}


def _frame(data, body: int) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = data.xmat[body].reshape(3, 3)
    t[:3, 3] = data.xpos[body]
    return t


def _angle(r: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


def _visual_extent(model, data, bodies, frame_body: int, meshes=None) -> np.ndarray:
    """(x, y, z) extent of the visual meshes of `bodies`, in `frame_body`'s frame."""
    rot, origin = data.xmat[frame_body].reshape(3, 3), data.xpos[frame_body]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies or model.geom_group[g] != 2 \
                or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        if meshes is not None and not any(model.mesh(mesh).name.endswith(m) for m in meshes):
            continue
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr:adr + num] @ data.geom_xmat[g].reshape(3, 3).T \
            + data.geom_xpos[g]
        local = (verts - origin) @ rot
        lo, hi = np.minimum(lo, local.min(0)), np.maximum(hi, local.max(0))
    return hi - lo


def _drive(world, inst, vx: float, vy: float, seconds: float = 2.0) -> float:
    """Planar speed the compiled base reaches under a held body-frame velocity.

    Contacts are off for the drive: the figure is the base's, and a kitchen's furniture
    within a metre would otherwise measure the furniture. The last second is measured.
    """
    model, data = world.model, world.data
    flags = model.opt.disableflags
    model.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_CONTACT
    try:
        setpoint = PlanarSetpoint()
        dt = 0.01
        per = max(1, round(dt / model.opt.timestep))
        steps = int(seconds / dt)
        start = None
        for i in range(steps):
            x, y, yaw = (float(v) for v in inst.base.xytheta)
            inst.base.ctrl = setpoint.step(x, y, yaw, vx, vy, 0.0, dt)
            for _ in range(per):
                mujoco.mj_step(model, data)
            if i == steps - int(1.0 / dt) - 1:
                start = (np.array(inst.base.xytheta[:2], dtype=float), float(data.time))
        end = np.array(inst.base.xytheta[:2], dtype=float)
        return float(np.linalg.norm(end - start[0]) / (float(data.time) - start[1]))
    finally:
        model.opt.disableflags = flags


def _diagonal_fov_deg(fovy_deg: float, size) -> float:
    width, height = size
    half = math.tan(math.radians(fovy_deg) / 2.0) * math.hypot(width, height) / height
    return math.degrees(2.0 * math.atan(half))


# ------------------------------------------------------------------ the descriptions


def check_urdf(label: str, robot: str, model, data, pre: str, expected_limits) -> None:
    """Joints, limits and link poses against the URDF, by independent FK."""
    urdf = Urdf(robots_spec.urdf_path(robot))
    root_link, root_body = ROOTS[robot]
    names = LINK_BODIES.get(robot, {})
    rigid = RIGID_URDF_JOINTS.get(robot, {})
    body_of = {}
    for link in urdf.links:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pre + names.get(link, link))
        if bid >= 0:
            body_of[link] = bid

    moving = {}
    for name, j in urdf.moving.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + name)
        if jid < 0:
            ok = name in rigid and np.allclose(j["xyz"], 0) and np.allclose(j["rpy"], 0)
            check(f"{label}: URDF joint {name} held rigid at its identity origin", ok,
                  rigid.get(name, "missing from the compiled model, with no reason recorded"))
            continue
        moving[name] = jid
        body = model.jnt_bodyid[jid]
        check(f"{label}: joint {name} sits on link {j['child']}",
              body_of.get(j["child"]) == body, model.body(body).name)
        axis = model.jnt_axis[jid] / np.linalg.norm(model.jnt_axis[jid])
        want = j["axis"] / np.linalg.norm(j["axis"])
        check(f"{label}: joint {name} axis", np.allclose(axis, want, atol=1e-9),
              f"{np.round(axis, 6)} vs URDF {want}")
        check(f"{label}: joint {name} at its link's origin",
              np.allclose(model.jnt_pos[jid], 0.0, atol=1e-9), str(model.jnt_pos[jid]))
        lower, upper = (float(v) for v in model.jnt_range[jid])
        urdf_lo, urdf_hi = j["limit"] if j["limit"] else (-math.inf, math.inf)
        want_lo, want_hi = expected_limits.get(name, (urdf_lo, urdf_hi))
        within = urdf_lo - 1e-9 <= want_lo and want_hi <= urdf_hi + 1e-9
        same = abs(lower - want_lo) <= 5e-5 and abs(upper - want_hi) <= 5e-5
        narrowed = abs(want_lo - urdf_lo) > 1e-9 or abs(want_hi - urdf_hi) > 1e-9
        why = "" if not narrowed else \
            f" (narrowed from the URDF's [{urdf_lo:g}, {urdf_hi:g}] by the vendor's servo " \
            "table, servo_controller.yaml, which spec §3 ranks above the URDF)"
        check(f"{label}: joint {name} limits",
              bool(model.jnt_limited[jid]) == (j["limit"] is not None or name in expected_limits)
              and same and within,
              f"[{lower:.5f}, {upper:.5f}] vs [{want_lo:.5f}, {want_hi:.5f}]{why}")

    # Link poses, relative to the root, over sampled configurations.
    rng = np.random.default_rng(20260927)
    worst_p = worst_r = 0.0
    root = body_of[root_link]
    for trial in range(12):
        q = {}
        for name, jid in moving.items():
            lo, hi = model.jnt_range[jid] if model.jnt_limited[jid] else (-math.pi, math.pi)
            q[name] = 0.0 if trial == 0 else float(rng.uniform(lo, hi))
            data.qpos[model.jnt_qposadr[jid]] = q[name]
        mujoco.mj_kinematics(model, data)
        ref = urdf.fk(q, root_link)
        inv = np.linalg.inv(_frame(data, root))
        for link, bid in body_of.items():
            got = inv @ _frame(data, bid)
            worst_p = max(worst_p, float(np.linalg.norm(got[:3, 3] - ref[link][:3, 3])))
            worst_r = max(worst_r, _angle(ref[link][:3, :3].T @ got[:3, :3]))
    check(f"{label}: {len(body_of)} link frames against URDF FK over 12 configurations",
          worst_p <= URDF_POS_TOL and worst_r <= URDF_ROT_TOL,
          f"worst {worst_p * 1e3:.4f} mm, {math.degrees(worst_r):.4f} deg")

    bodies = [body_of[link] for link in urdf.links if link in body_of]
    compiled = float(model.body_subtreemass[root])
    declared = sum(urdf.masses.values())
    if robot != "myagv":  # the myAGV's URDF declares no inertials at all
        check(f"{label}: mass is the URDF's", abs(compiled - declared) <= 1e-9 or
              len(bodies) == 0, f"{compiled:.6f} vs {declared:.6f} kg")


def check_mjcf(label: str, robot: str, model, pre: str) -> None:
    """Every body, inertial and joint of the official MJCF, exactly."""
    ref = mujoco.MjModel.from_xml_path(str(robots_spec.mjcf_path(robot)))
    worst = 0.0
    bodies = 0
    for i in range(1, ref.nbody):
        name = ref.body(i).name
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pre + name)
        check(f"{label}: official body {name} present", k >= 0)
        if k < 0:
            continue
        bodies += 1
        fields = ["body_mass", "body_inertia", "body_ipos", "body_iquat"]
        if ref.body_parentid[i] != 0:  # the root's pose is its placement, not the file's
            fields += ["body_pos", "body_quat"]
            parent = ref.body(ref.body_parentid[i]).name
            check(f"{label}: official body {name} under {parent}",
                  model.body(model.body_parentid[k]).name == pre + parent)
        for field in fields:
            worst = max(worst, float(np.abs(getattr(model, field)[k] - getattr(ref, field)[i]).max()))
    for i in range(ref.njnt):
        name = ref.joint(i).name
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + name)
        check(f"{label}: official joint {name} present", k >= 0)
        if k < 0:
            continue
        for field in ("jnt_axis", "jnt_pos", "jnt_range"):
            worst = max(worst, float(np.abs(getattr(model, field)[k] - getattr(ref, field)[i]).max()))
        check(f"{label}: official joint {name} type and limited flag",
              model.jnt_type[k] == ref.jnt_type[i] and model.jnt_limited[k] == ref.jnt_limited[i])
    check(f"{label}: {bodies} bodies and {ref.njnt} joints equal the official MJCF",
          worst <= MJCF_TOL, f"worst difference {worst:.1e}")


# --------------------------------------------------------------------- the figures


def measure(robot: str, world, inst) -> dict[str, float]:
    model, data = world.model, world.data
    pre = inst.mjcf
    ours = {b for b in range(model.nbody) if model.body(b).name.startswith(pre)}
    out: dict[str, float] = {}
    if robot == "myagv":
        base = model.body(f"{pre}base").id
        mujoco.mj_forward(model, data)
        out["length_m"], out["width_m"], out["height_m"] = _visual_extent(
            model, data, ours, base, meshes=("myagv_base",))
        out["mass_kg"] = float(model.body_subtreemass[base])
        cam = model.camera(pick_camera(model, pre)).id
        out["camera_fov_deg"] = _diagonal_fov_deg(float(model.cam_fovy[cam]),
                                                  myagv_contract.CAMERA_SIZE)
        limit = myagv_contract.CMD_VEL_LIMIT
        speeds = [_drive(world, inst, *myagv_contract.limit_speed(vx, vy))
                  for vx, vy in ((limit, 0.0), (limit, limit))]
        out["max_speed_mps"] = max(speeds)
    elif robot == "so101":
        joints = [j for j in range(model.njnt) if model.joint(j).name.startswith(pre)
                  and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        driven = [sum(1 for a in range(model.nu) if model.actuator_trnid[a, 0] == j
                      and model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT)
                  for j in joints]
        out["servo_joints"] = float(len(joints)) if all(n == 1 for n in driven) else -1.0
    elif robot == "ainex":
        torso = model.body(f"{pre}body_link").id
        out["mass_kg"] = float(model.body_subtreemass[torso])
        servo_joints = [model.joint(pre + n).id for n in servos.SERVOS
                        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + n) >= 0]
        actuated = [j for j in servo_joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                    and any(model.actuator_trnid[a, 0] == j for a in range(model.nu))]
        out["dof"] = float(len(actuated))
        out["servo_travel_deg"] = max(math.degrees(float(model.jnt_range[j][1] - model.jnt_range[j][0]))
                                      for j in servo_joints)
        saved = data.qpos.copy()
        for j in servo_joints:
            data.qpos[model.jnt_qposadr[j]] = 0.0
        # Arms hanging: each shoulder roll a quarter turn down (the vendor's own home pose
        # has them at +/-1.293 rad, the same direction).
        for side in ("l", "r"):
            name = f"{side}_sho_roll"
            down = -math.copysign(math.pi / 2, servos.INIT_POSE[name])
            data.qpos[model.jnt_qposadr[model.joint(pre + name).id]] = down
        mujoco.mj_forward(model, data)
        out["depth_m"], out["width_m"], out["height_m"] = _visual_extent(model, data, ours, torso)
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        param = gait.WalkingParam(period_time=0.400,
                                  x_amplitude=gait.X_AMPLITUDE_RANGE[1]).clamped()
        vx, vy, _ = gait.planar_velocity(param)
        out["walking_speed_mps"] = _drive(world, inst, vx, vy)
    return out


def pick_camera(model, pre: str) -> str:
    import spawn

    return spawn.pick_camera(model, pre)


def check_figures(label: str, robot: str, world, inst, rows: list) -> None:
    measured = measure(robot, world, inst)
    for fig in CONTRACTS[robot].PHYSICAL_FIGURES:
        got = measured.get(fig.key)
        ok = got is not None and fig.within(got)
        check(f"{label}: {fig.label} {fig.value:g} {fig.unit}".rstrip(), ok,
              "not measured" if got is None else
              f"model {got:.4g} {fig.unit}, tolerance +/-{fig.tolerance:.3g}")
        rows.append((robot, fig, got, ok))


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=("molmospaces", "robocasa"))
    ap.add_argument("--robot", default=None, help="one id (default: every simulated robot)")
    args = ap.parse_args()
    engine = pc.load_engine(args.engine)
    flags = pc.scene_flags(args.engine)
    rows: list = []
    for robot in [args.robot] if args.robot else robots_spec.simulated_ids():
        label = f"{args.engine} {robot}"
        world, _log = pc.build(engine, flags, robot)
        inst = world.instances[0]
        print(f"{label}:")
        model, data = world.model, world.data
        limits = servos.all_limits() if robot == "ainex" else {}
        saved = data.qpos.copy()
        check_urdf(label, robot, model, data, inst.mjcf, limits)
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        if robots_spec.robot(robot).mjcf is not None:
            check_mjcf(label, robot, model, inst.mjcf)
        check_figures(label, robot, world, inst, rows)
    print()
    print(f"{'robot':6} {'figure':34} {'published':>11} {'model':>11} {'tolerance':>10}  source")
    for robot, fig, got, ok in rows:
        shown = "-" if got is None else f"{got:.4g}"
        print(f"{robot:6} {fig.label[:34]:34} {fig.value:>9.4g} {fig.unit:<2}"
              f"{shown:>9} {fig.unit:<2}{fig.tolerance:>8.3g}  {'' if ok else 'FAIL '}{fig.source}")
    print("all checks passed" if not pc.failures else f"FAILED: {len(pc.failures)} check(s)")
    return 0 if not pc.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
