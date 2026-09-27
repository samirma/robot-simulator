#!/usr/bin/env python
"""Generate model.xml from TheRobotStudio's official `so101_new_calib.xml`.

The official MJCF (`robots_specs/so101/`, as `robots_specs/robots.yml` records it) is the
model: every body, inertial, joint, visual geom, site, default and actuator in model.xml
is the official file's, value for value. The generator only *adds* what simulation needs
and never moves a frame, a joint or a limit:

1. **Collision.** The official file collides with convex hulls of its visual meshes, and
   a hull of a jaw fills the gap between the fingers, so nothing could be grasped. They
   are replaced by mujoco_menagerie's `robotstudio_so101` collision set (Apache 2.0, see
   shared/robots/so101/LICENSE), transcribed below with its three gripper-part meshes
   (shared/robots/so101/assets) and its `collision_gripper*` contact classes: boxes in
   group 3 and the part meshes in group 4, none of them visible. The jaw-tip spheres
   `fixed_jaw_sph_tip2` / `moving_jaw_sph_tip2` are named because the TCP below is
   measured from them, and the engines find the jaw geoms by the `*_jaw` prefix.
2. **TCP site `tcp`** (group 3), which MolmoSpaces' SO101RobotView resolves by name. Its
   convention is +z = approach with the fingers opening along y; the official
   `gripperframe` uses +x as the approach and sits on the fixed-jaw tip, so a separate
   site is placed at the grasp centre. Both axes are *measured* from the model: approach
   = gripper body origin -> jaw-tip midpoint, finger axis = fixed tip -> moving tip, at
   NOMINAL_GRIP (~4 cm between the tips). The jaw is hinged, not parallel, so the finger
   axis is only perpendicular to the approach at one opening; it is ~8 deg off here.
3. **Wrist camera `wrist_cam`**, directly in the official `gripper` body at menagerie's
   pose (menagerie hangs it off a `camera_mount` body with an identity pose, whose mount
   mesh is not official geometry and is not added). The intrinsics are a 640x480 capture,
   usb_cam's default mode, rather than menagerie's 16:9 sensor: same 5.76 mm sensor width
   and 3.6 mm focal length, 4:3 height, so pixels stay square (fx = fy = 400 px).

`<option>` is the official file's (none, so MuJoCo's defaults): the engines graft the
robot's bodies into their own scene, whose solver settings are what actually step it.

`--check` compiles the official file and model.xml and asserts the above -- bodies,
inertials, joints, official sites, actuators and visual geoms identical, nothing added
but collision geoms, `tcp` and `wrist_cam` -- and compares forward kinematics against the
official URDF at random configurations. Generation runs it at the end.

Run:  python robots/so101/make_model.py [--check]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "shared"))
import robots_spec  # noqa: E402

OFFICIAL = robots_spec.mjcf_path("so101")
OFFICIAL_URDF = robots_spec.urdf_path("so101")
OFFICIAL_MESHES = robots_spec.mesh_dirs("so101")[0]
SPEC = robots_spec.model_dir("so101")
OUT = robots_spec.model_xml("so101")
#: model.xml's `meshdir`: the gripper collision parts only; the official meshes are
#: referenced in robots_specs/ relative to it.
EXTRA_MESHES = SPEC / "assets"

# Gripper joint angle at which the finger axis is measured. ~0.04 m between jaw tips.
NOMINAL_GRIP = 0.3
FIXED_TIP = "fixed_jaw_sph_tip2"
MOVING_TIP = "moving_jaw_sph_tip2"
GRIPPER_BODY = "gripper"

# --- Transcribed from mujoco_menagerie robotstudio_so101/so101.xml (Apache 2.0) -----------
# Vendored verbatim as shared/robots/so101/so101.xml until 2026-09-27. Only collision is
# taken from it; its body frames are the official ones, so every geom lands where
# menagerie put it. Its `collision` class (group 3, condim 3) is the official `collision`
# class plus MuJoCo's default condim, so the boxes use the official class unchanged.

COLLISION_CLASSES = """
<default class="collision_gripper">
  <geom group="3" condim="6" friction="1 5e-3 5e-4" solref="0.01 1" priority="1" rgba="1.0 0 0 1.0" mass="0"/>
</default>
<default class="collision_gripper_mesh">
  <geom group="4" condim="6" friction="1 5e-3 5e-4" solref="0.01 1" priority="1" rgba="1.0 0 0 1.0" mass="0"/>
</default>
"""

COLLISION_MESHES = """
<mesh maxhullvert="64" name="wrist_roll_follower_so101_gripper_part0_v1" file="wrist_roll_follower_so101_gripper_part0_v1.stl"/>
<mesh maxhullvert="64" name="moving_jaw_so101_gripper_part0_v1" file="moving_jaw_so101_gripper_part0_v1.stl"/>
<mesh maxhullvert="64" name="moving_jaw_so101_gripper_part1_v1" file="moving_jaw_so101_gripper_part1_v1.stl"/>
"""

#: Per official body, the collision geoms menagerie gives it. `camera_mount`'s two boxes
#: are not here: they are the collision of a mount that is not official geometry.
COLLISION = {
    "shoulder": """
<geom type="box" class="collision" pos="-0.030399 0.000422 -0.0387" quat="1 1 1 -1" size="0.023 0.015 0.01"/>
<geom type="box" class="collision" pos="-0.025 0 0.0" quat="1 1 -1 1" size="0.038 0.025 0.02"/>
""",
    "upper_arm": """
<geom type="box" class="collision" pos="-0.06 0 0.02" quat="0 -1 1 0" size="0.01 0.07 0.03"/>
<geom type="box" class="collision" pos="-0.12 -0.014 0.0182" quat="0 1 0 0" size="0.01 0.02 0.015"/>
""",
    "lower_arm": """
<geom type="box" class="collision" pos="-0.05 0 0.0182" quat="0 1 0 0" size="0.07 0.01 0.03"/>
<geom type="box" class="collision" pos="-0.125 0.005 0.018" quat="0 -1 0 0" size="0.023 0.013 0.018"/>
""",
    "wrist": """
<geom type="box" class="collision" pos="0 -0.0424 0.0256" quat="1 1 1 -1" size="0.029 0.015 0.018"/>
<geom type="box" class="collision" pos="0 -0.02 0.0191" quat="1 -1 -1 -1" size="0.027 0.01 0.03"/>
""",
    "gripper": """
<geom type="box" class="collision" pos="0 -0.000218 0.00095" quat="0 1 0 0" size="0.01 0.01 0.018"/>
<geom name="fixed_jaw_box1" class="collision_gripper" type="box" size="0.0325 0.015 0.015" pos="-0.0025 0 -0.022"/>
<geom name="fixed_jaw_box2" class="collision_gripper" type="box" size="0.01 0.015 0.005" pos="-0.024 0 -0.04"/>
<geom name="fixed_jaw_sph_tip1" class="collision_gripper" type="sphere" size="0.00075" pos="-0.0081 0 -0.101"/>
<geom name="fixed_jaw_sph_tip2" class="collision_gripper" type="sphere" size="0.00075" pos="-0.0081 0.0035 -0.0975"/>
<geom name="fixed_jaw_sph_tip3" class="collision_gripper" type="sphere" size="0.00075" pos="-0.0081 -0.0035 -0.0975"/>
<geom type="mesh" class="collision_gripper_mesh" pos="0 -0.000218 0.00095" quat="0 1 0 0" mesh="wrist_roll_follower_so101_gripper_part0_v1"/>
<geom name="fixed_jaw_box3" class="collision_gripper" type="capsule" size="0.0011 0.002" pos="-0.009 0 -0.103" euler="1.57 0 0"/>
<geom name="fixed_jaw_box4" class="collision_gripper" type="box" size="0.001 0.004 0.004" pos="-0.009 0 -0.0982"/>
<geom name="fixed_jaw_box5" class="collision_gripper" type="box" size="0.001 0.005 0.006" pos="-0.0108 0 -0.0905"/>
<geom name="fixed_jaw_box6" class="collision_gripper" type="box" size="0.001 0.009 0.008" pos="-0.0125 0 -0.0727"/>
<geom name="fixed_jaw_box7" class="collision_gripper" type="box" size="0.001 0.01 0.008" pos="-0.0143 0 -0.053"/>
""",
    "moving_jaw_so101_v1": """
<geom type="mesh" class="collision_gripper_mesh" pos="0 0 0.0189" quat="1 0 0 0" mesh="moving_jaw_so101_gripper_part0_v1"/>
<geom type="mesh" class="collision_gripper_mesh" pos="0 0 0.0189" quat="1 0 0 0" mesh="moving_jaw_so101_gripper_part1_v1"/>
<geom name="moving_jaw_box1" class="collision_gripper" type="box" size="0.01 0.01 0.015" pos="-0.0 -0.013 0.019"/>
<geom name="moving_jaw_sph_tip1" class="collision_gripper" type="sphere" size="0.00075" pos="-0.012 -0.078 0.0192"/>
<geom name="moving_jaw_sph_tip2" class="collision_gripper" type="sphere" size="0.00075" pos="-0.012 -0.0745 0.0225"/>
<geom name="moving_jaw_sph_tip3" class="collision_gripper" type="sphere" size="0.00075" pos="-0.012 -0.0745 0.0155"/>
<geom name="moving_jaw_box2" class="collision_gripper" type="box" size="0.001 0.004 0.004" pos="-0.0113 -0.076 0.01875"/>
<geom name="moving_jaw_box3" class="collision_gripper" type="box" size="0.001 0.005 0.006" pos="-0.0093 -0.067 0.01875"/>
""",
}

#: menagerie's pose (its camera_mount body sits at the gripper origin), with a 4:3 sensor.
WRIST_CAM = (
    '<camera name="wrist_cam" mode="fixed" pos="0.0 0.055 -0.045" euler="-0.57 0 0" '
    'resolution="640 480" sensorsize="0.00576 0.00432" focal="0.0036 0.0036"/>'
)
# ---------------------------------------------------------------------------------------

ADDED_SITES = {"tcp"}
ADDED_CAMERAS = {"wrist_cam"}

HEADER = """
  GENERATED by molmospaces/robots/so101/make_model.py - do not edit by hand.

  Source: robots_specs/so101/so101_new_calib.xml, TheRobotStudio's official SO-101 MJCF
  (onshape-to-robot export). Every body, inertial, joint, visual geom, site, default and
  actuator below is that file's, unchanged; its meshes are loaded from
  robots_specs/so101/assets. Changes, none of which moves a frame, joint or limit:

  - official collision (convex hulls of the visual meshes) replaced by
    mujoco_menagerie's robotstudio_so101 collision primitives and gripper-part meshes
    (assets/ here, Apache 2.0, see LICENSE), groups 3 and 4, with its
    collision_gripper / collision_gripper_mesh classes;
  - site `tcp` (group 3) in the gripper body, MolmoSpaces' TCP convention;
  - camera `wrist_cam` in the gripper body at menagerie's pose, 640x480.

  make_model.py's check mode verifies all of this against the official MJCF and URDF.
"""


def _parse(text: str) -> ET.Element:
    return ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def _fragment(text: str) -> list[ET.Element]:
    return list(_parse(f"<x>{text}</x>"))


def _append_after_last(parent: ET.Element, tag: str, new: list[ET.Element]) -> None:
    """Insert `new` after `parent`'s last `tag` child, so geoms stay before child bodies."""
    children = list(parent)
    index = max((i for i, c in enumerate(children) if c.tag == tag), default=-1) + 1
    for offset, element in enumerate(new):
        parent.insert(index + offset, element)


def _body(root: ET.Element, name: str) -> ET.Element:
    found = root.find(f".//worldbody//body[@name='{name}']")
    if found is None:
        raise SystemExit(f"no body {name!r} in {OFFICIAL}; the official model changed?")
    return found


def build(tcp: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> ET.Element:
    """The official model plus the additions; `tcp` None leaves the TCP site out."""
    root = _parse(OFFICIAL.read_text())

    # Official meshes, referenced where robots_specs/ keeps them. A mesh's default name
    # is its file's stem and the geoms refer to it by that, so the name is made explicit.
    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        name = mesh.get("file")
        if not (OFFICIAL_MESHES / name).is_file():
            raise SystemExit(f"mesh {name} missing from {OFFICIAL_MESHES}; "
                             "run ./fetch_robot_assets.sh so101")
        mesh.set("name", mesh.get("name", Path(name).stem))
        mesh.set("file", os.path.relpath(OFFICIAL_MESHES / name, EXTRA_MESHES))
    extra = _fragment(COLLISION_MESHES)
    for mesh in extra:
        if not (EXTRA_MESHES / mesh.get("file")).is_file():
            raise SystemExit(f"collision mesh {mesh.get('file')} missing from {EXTRA_MESHES}")
    _append_after_last(asset, "mesh", extra)

    # Contact classes for the gripper, inside the official robot class.
    robot_class = root.find("default/default[@class='so101_new_calib']")
    robot_class.extend(_fragment(COLLISION_CLASSES))

    # Swap the official hull collision for the primitives.
    for body in root.iter("body"):
        for geom in [g for g in body.findall("geom") if g.get("class") == "collision"]:
            body.remove(geom)
    for name, geoms in COLLISION.items():
        _append_after_last(_body(root, name), "geom", _fragment(geoms))

    gripper = _body(root, GRIPPER_BODY)
    additions = [_parse(WRIST_CAM)]
    if tcp is not None:
        fmt = lambda v: " ".join(f"{c:.6g}" for c in v)  # noqa: E731
        pos, x_axis, y_axis = tcp
        additions.insert(0, ET.Comment(
            " MolmoSpaces TCP: +z = approach, +y = finger-opening axis, at the grasp centre."
            " Measured by make_model.py. "))
        additions.insert(1, ET.Element("site", {
            "group": "3", "name": "tcp", "pos": fmt(pos), "xyaxes": f"{fmt(x_axis)} {fmt(y_axis)}"}))
    _append_after_last(gripper, "site", additions)

    root.insert(0, ET.Comment(HEADER))
    return root


def to_xml(root: ET.Element) -> str:
    root = copy.deepcopy(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def load_model(xml: str) -> mujoco.MjModel:
    """Compile `xml` as if it sat in SPEC, where its relative mesh paths hold."""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=SPEC, delete=False) as f:
        f.write(xml)
    try:
        return mujoco.MjModel.from_xml_path(f.name)
    finally:
        os.unlink(f.name)


def compute_tcp_frame(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (pos, x_axis, y_axis, perpendicularity_error_deg) in the gripper body frame."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = NOMINAL_GRIP
    mujoco.mj_forward(model, data)

    fixed_tip = data.geom_xpos[model.geom(FIXED_TIP).id]
    moving_tip = data.geom_xpos[model.geom(MOVING_TIP).id]
    body_id = model.body(GRIPPER_BODY).id
    body_rot = data.xmat[body_id].reshape(3, 3)
    body_pos = data.xpos[body_id]

    grasp_centre = (fixed_tip + moving_tip) / 2
    approach = grasp_centre - body_pos
    approach /= np.linalg.norm(approach)
    finger = fixed_tip - moving_tip
    finger /= np.linalg.norm(finger)
    error_deg = abs(90.0 - np.degrees(np.arccos(abs(float(approach @ finger)))))

    # z = approach; y = finger axis with any approach component projected out.
    z_axis = approach
    y_axis = finger - (finger @ z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rot_world = np.column_stack([x_axis, y_axis, z_axis])
    assert np.isclose(np.linalg.det(rot_world), 1.0), "TCP frame is not right-handed"

    rot_local = body_rot.T @ rot_world
    pos_local = body_rot.T @ (grasp_centre - body_pos)
    return pos_local, rot_local[:, 0], rot_local[:, 1], error_deg


# --- verification -------------------------------------------------------------------------

def _names(model, objtype, count) -> list[str]:
    return [mujoco.mj_id2name(model, objtype, i) or "" for i in range(count)]


def _rpy(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis, angle) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def urdf_fk(urdf: ET.Element, q: dict[str, float]) -> dict[str, np.ndarray]:
    """World 4x4 of every URDF link, composing joint origins and rotations directly."""
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    poses: dict[str, np.ndarray] = {}

    def pose(link: str) -> np.ndarray:
        if link in poses:
            return poses[link]
        if link not in joints:
            poses[link] = np.eye(4)
            return poses[link]
        j = joints[link]
        origin = j.find("origin")
        t = np.eye(4)
        t[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        t[:3, :3] = _rpy([float(v) for v in origin.get("rpy", "0 0 0").split()])
        if j.get("type") in ("revolute", "continuous"):
            axis = [float(v) for v in j.find("axis").get("xyz").split()]
            m = np.eye(4)
            m[:3, :3] = _axis_angle(axis, q.get(j.get("name"), 0.0))
            t = t @ m
        poses[link] = pose(j.find("parent").get("link")) @ t
        return poses[link]

    for link in urdf.findall("link"):
        pose(link.get("name"))
    return poses


#: MJCF body -> URDF link, the official files' own pairing.
URDF_LINKS = {
    "base": "base_link", "shoulder": "shoulder_link", "upper_arm": "upper_arm_link",
    "lower_arm": "lower_arm_link", "wrist": "wrist_link", "gripper": "gripper_link",
    "moving_jaw_so101_v1": "moving_jaw_so101_v1_link",
}


def check(model_xml: Path = OUT, seed: int = 20260927, samples: int = 16) -> int:
    """Assert model.xml is the official model plus the additions; print the deviations."""
    ref = mujoco.MjModel.from_xml_path(str(OFFICIAL))
    got = mujoco.MjModel.from_xml_path(str(model_xml))
    exact = dict(rtol=0, atol=1e-12)
    B, J, G, S, A, C = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
                        mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE,
                        mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_CAMERA)

    # Bodies: the same set, nothing added, identical frames and inertials.
    ref_bodies, got_bodies = _names(ref, B, ref.nbody), _names(got, B, got.nbody)
    assert ref_bodies == got_bodies, f"bodies differ: {ref_bodies} vs {got_bodies}"
    for i, name in enumerate(ref_bodies):
        k = got.body(name).id
        assert ref_bodies[ref.body_parentid[i]] == got_bodies[got.body_parentid[k]], name
        for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_allclose(getattr(got, field)[k], getattr(ref, field)[i], **exact,
                                       err_msg=f"{field} of body {name}")

    # Joints: name, type, axis, range, body, and the official dynamics.
    ref_joints, got_joints = _names(ref, J, ref.njnt), _names(got, J, got.njnt)
    assert ref_joints == got_joints, f"joints differ: {ref_joints} vs {got_joints}"
    for i, name in enumerate(ref_joints):
        k = got.joint(name).id
        assert ref.jnt_type[i] == got.jnt_type[k], name
        assert ref_bodies[ref.jnt_bodyid[i]] == got_bodies[got.jnt_bodyid[k]], name
        for field in ("jnt_axis", "jnt_pos", "jnt_range", "jnt_limited"):
            np.testing.assert_allclose(getattr(got, field)[k], getattr(ref, field)[i], **exact,
                                       err_msg=f"{field} of joint {name}")
        dof_r, dof_g = ref.jnt_dofadr[i], got.jnt_dofadr[k]
        for field in ("dof_damping", "dof_armature", "dof_frictionloss"):
            np.testing.assert_allclose(getattr(got, field)[dof_g], getattr(ref, field)[dof_r], **exact,
                                       err_msg=f"{field} of joint {name}")

    # Sites: every official one unchanged; the only addition is `tcp`.
    ref_sites, got_sites = _names(ref, S, ref.nsite), _names(got, S, got.nsite)
    assert set(got_sites) - set(ref_sites) == ADDED_SITES, f"sites: {got_sites}"
    for i, name in enumerate(ref_sites):
        k = got.site(name).id
        assert ref_bodies[ref.site_bodyid[i]] == got_bodies[got.site_bodyid[k]], name
        np.testing.assert_allclose(got.site_pos[k], ref.site_pos[i], **exact, err_msg=f"site {name} pos")
        np.testing.assert_allclose(got.site_quat[k], ref.site_quat[i], **exact, err_msg=f"site {name} quat")

    # Actuators: name, joint, class-derived gains, ctrlrange, forcerange.
    ref_acts, got_acts = _names(ref, A, ref.nu), _names(got, A, got.nu)
    assert ref_acts == got_acts, f"actuators differ: {ref_acts} vs {got_acts}"
    for i, name in enumerate(ref_acts):
        k = got.actuator(name).id
        assert ref_joints[ref.actuator_trnid[i, 0]] == got_joints[got.actuator_trnid[k, 0]], name
        for field in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm",
                      "actuator_biasprm", "actuator_gear", "actuator_dyntype"):
            np.testing.assert_allclose(getattr(got, field)[k], getattr(ref, field)[i], **exact,
                                       err_msg=f"{field} of actuator {name}")

    # Visual geoms (group 2): the official set, same mesh, pose and material, per body.
    def visuals(m):
        mats = _names(m, mujoco.mjtObj.mjOBJ_MATERIAL, m.nmat)
        meshes = _names(m, mujoco.mjtObj.mjOBJ_MESH, m.nmesh)
        bodies = _names(m, B, m.nbody)
        return sorted(
            (bodies[m.geom_bodyid[g]], meshes[m.geom_dataid[g]], mats[m.geom_matid[g]],
             tuple(np.round(m.geom_pos[g], 12)), tuple(np.round(m.geom_quat[g], 12)),
             int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
            for g in range(m.ngeom) if m.geom_group[g] <= 2)
    assert visuals(ref) == visuals(got), "visual geoms differ from the official ones"
    # ...which also says every other geom is collision in group 3 or 4, hidden.
    assert all(got.geom_group[g] in (3, 4) for g in range(got.ngeom) if got.geom_group[g] > 2)

    # Cameras: only `wrist_cam`, and a square-pixel 640x480.
    cams = set(_names(got, C, got.ncam))
    assert cams - set(_names(ref, C, ref.ncam)) == ADDED_CAMERAS, f"cameras: {cams}"
    cam = got.camera("wrist_cam").id
    res, sensor, focal = got.cam_resolution[cam], got.cam_sensorsize[cam], got.cam_intrinsic[cam]
    assert tuple(res) == (640, 480), res
    fx, fy = focal[0] / sensor[0] * res[0], focal[1] / sensor[1] * res[1]
    assert abs(fx - fy) < 1e-3, (fx, fy)  # sensorsize is stored as float32

    # Solver options: the official file's.
    for field in ("timestep", "integrator", "cone", "impratio", "iterations", "noslip_iterations"):
        assert getattr(ref.opt, field) == getattr(got.opt, field), f"option {field}"

    # Forward kinematics against the official URDF, and against the official MJCF.
    urdf = ET.parse(OFFICIAL_URDF).getroot()
    rng = np.random.default_rng(seed)
    d_ref, d_got = mujoco.MjData(ref), mujoco.MjData(got)
    worst_urdf = [0.0, 0.0]
    worst_mjcf = [0.0, 0.0]
    gf_urdf = gf_angle = 0.0
    for trial in range(samples):
        q = {} if trial == 0 else {
            name: float(rng.uniform(*ref.jnt_range[ref.joint(name).id])) for name in ref_joints}
        for m, d in ((ref, d_ref), (got, d_got)):
            for name, v in q.items():
                d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
            mujoco.mj_kinematics(m, d)
        links = urdf_fk(urdf, q)
        for body, link in URDF_LINKS.items():
            i, k = ref.body(body).id, got.body(body).id
            rot_g = d_got.xmat[k].reshape(3, 3)
            dp = np.linalg.norm(links[link][:3, 3] - d_got.xpos[k])
            da = np.arccos(np.clip((np.trace(links[link][:3, :3].T @ rot_g) - 1) / 2, -1, 1))
            worst_urdf = [max(worst_urdf[0], dp), max(worst_urdf[1], da)]
            dp = np.linalg.norm(d_ref.xpos[i] - d_got.xpos[k])
            da = np.linalg.norm(d_ref.xmat[i] - d_got.xmat[k])
            worst_mjcf = [max(worst_mjcf[0], dp), max(worst_mjcf[1], da)]
        site = got.site("gripperframe").id
        gf_urdf = max(gf_urdf, np.linalg.norm(links["gripper_frame_link"][:3, 3] - d_got.site_xpos[site]))
        rot = links["gripper_frame_link"][:3, :3].T @ d_got.site_xmat[site].reshape(3, 3)
        gf_angle = max(gf_angle, np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    print(f"check: {len(ref_bodies) - 1} bodies, {len(ref_joints)} joints, {len(ref_sites)} sites, "
          f"{len(ref_acts)} actuators, {len(visuals(ref))} visual geoms identical to {OFFICIAL.name}")
    print(f"check: FK vs official MJCF over {samples} configurations: "
          f"max {worst_mjcf[0]:.2e} m, {worst_mjcf[1]:.2e} (rotation-matrix norm)")
    print(f"check: FK vs official URDF over {samples} configurations, {len(URDF_LINKS)} links: "
          f"max {worst_urdf[0]:.2e} m, {worst_urdf[1]:.2e} rad; "
          f"gripper_frame_link vs site gripperframe {gf_urdf:.2e} m")
    # Not asserted: the two official files disagree on this frame's orientation. The
    # URDF's gripper_frame_joint turns it pi about y, the MJCF's site pi/2 about y.
    print(f"check: note: official URDF gripper_frame_link and MJCF gripperframe differ in "
          f"orientation by {np.degrees(gf_angle):.1f} deg (the two official files disagree)")
    assert worst_mjcf[0] < 1e-12 and worst_mjcf[1] < 1e-12, worst_mjcf
    assert worst_urdf[0] < 1e-4 and worst_urdf[1] < 1e-3, worst_urdf
    assert gf_urdf < 1e-4, gf_urdf
    print("check: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify model.xml; do not regenerate it")
    args = ap.parse_args()
    if args.check:
        return check()

    pos, x_axis, y_axis, error_deg = compute_tcp_frame(load_model(to_xml(build(None))))
    fmt = lambda v: " ".join(f"{c:.6g}" for c in v)  # noqa: E731
    print(f"TCP at {fmt(pos)} (gripper frame); finger axis is {error_deg:.1f} deg off perpendicular")
    OUT.write_text(to_xml(build((pos, x_axis, y_axis))))
    print(f"wrote {OUT}")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
