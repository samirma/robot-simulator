"""The transform tree against the description it claims to be a tree of.

Standalone, in the style of the other checks here, runnable under either engine's venv:

    molmospaces/.venv/bin/python shared/tests/tf_frames_check.py
    robocasa/.venv/bin/python shared/tests/tf_frames_check.py

Every robot on this graph publishes `/tf` off its compiled MuJoCo model and
`robot_description` out of its vendor URDF. Those are two independent statements about
the same robot, and if they disagree a client draws a body whose links are in the wrong
places -- with nothing erroring anywhere, because each half is internally consistent.
This is the check that they agree, and it is deliberately made a *different* way than
`TransformTree` makes it: the tree reads MuJoCo's `xpos`/`xmat`, and this reads the
URDF's own joint origins with an independent rpy-to-quaternion conversion. That is the
rule this project learned the hard way and wrote down -- **a check of a measurement must
not share its method** -- applied before rather than after.

Four claims, per robot:

* every frame the tree publishes is a link the description declares, or a camera frame
  the contract names on purpose (a camera driver names its own frame; `wrist` is not a
  URDF link and is not meant to be);
* the tree is connected: one root, and every parent frame is somebody's child or that
  root;
* each moving link's transform, at the zero pose, equals the description's joint origin;
* the AiNex's camera, which this simulator deliberately moves from the torso to the head,
  still lands where the vendor's own `camera` joint puts it -- the round trip through
  `mujoco_bridge.camera_link_pose` back to the number in the URDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

SHARED = Path(__file__).resolve().parents[1]
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import ainex_model  # noqa: E402
import robots_spec  # noqa: E402
from contracts.tf import rpy_to_quat, urdf_fixed_joints, urdf_links  # noqa: E402
from mujoco_bridge import TransformTree, camera_link_pose  # noqa: E402
from ros_surfaces import myagv as myagv_surface  # noqa: E402
from ros_surfaces import so101 as so101_surface  # noqa: E402
from ros_surfaces.ainex import topics as ainex_topics  # noqa: E402

#: Angular agreement to 1e-5 rather than exactly. The SO-101's MJCF writes its body
#: quaternions as decimal text, so a quaternion read back off the compiled model differs
#: from the URDF's rpy in the sixth decimal -- measured 2e-6, and a tolerance tighter than
#: the file format is a check that fails for a reason no one can act on.
TOL = 1e-5

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def urdf_joint_origins(text: str) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """`child link -> (parent link, xyz, quat)` for every joint, fixed or not."""
    import xml.etree.ElementTree as ET

    out = {}
    for joint in ET.fromstring(text).findall("joint"):
        origin = joint.find("origin")
        xyz = (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
        rpy = (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
        out[joint.find("child").get("link")] = (
            joint.find("parent").get("link"),
            np.array([float(v) for v in xyz]),
            np.array(rpy_to_quat(*(float(v) for v in rpy))),
        )
    return out


def quat_error(a, b) -> float:
    """Distance between two quaternions, up to the sign that names the same rotation."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def check_robot(label: str, model, tree, urdf_text: str, camera_frames: set[str]) -> None:
    print(f"{label}")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    links = set(urdf_links(urdf_text))
    stray = [f for f in tree.frames if f not in links and f not in camera_frames]
    check("every frame is a link of the description", not stray, str(stray))

    entries = tree.static() + tree.dynamic(data)
    children = {child for _, child, _, _ in entries}
    roots = {parent for parent, _, _, _ in entries} - children
    check("the tree has exactly one root", len(roots) == 1, str(sorted(roots)))
    check("no frame has two parents",
          len(children) == len(entries), f"{len(children)} children in {len(entries)} edges")

    origins = urdf_joint_origins(urdf_text)
    worst_pos = worst_quat = 0.0
    compared = 0
    for parent, child, pos, quat in entries:
        if child not in origins or child in camera_frames:
            continue
        urdf_parent, urdf_pos, urdf_quat = origins[child]
        if urdf_parent != parent:
            check(f"{child} hangs off the link the description says",
                  False, f"{parent} vs {urdf_parent}")
            continue
        compared += 1
        worst_pos = max(worst_pos, float(np.linalg.norm(np.asarray(pos) - urdf_pos)))
        worst_quat = max(worst_quat, quat_error(quat, urdf_quat))
    check(f"all {compared} link transforms match the description at the zero pose",
          compared > 0 and worst_pos < TOL and worst_quat < TOL,
          f"worst dpos {worst_pos:.2e} m, dquat {worst_quat:.2e}")


class _Entries:
    """A fixed list of transforms in `TransformTree`'s shape, for `check_robot`."""

    def __init__(self, entries) -> None:
        self._entries = list(entries)

    @property
    def frames(self) -> tuple[str, ...]:
        return tuple(sorted({f for p, c, _, _ in self._entries for f in (p, c)}))

    def static(self):
        return list(self._entries)

    def dynamic(self, _data):
        return []


def check_myagv() -> None:
    """The myAGV's tree is its launch's and its URDF's, not the model's: `odom ->
    base_footprint` from robot_pose_ekf, `base_up` from robot_state_publisher, and the three
    `static_transform_publisher` mounts. So it is checked against the description, and the
    sensors the model simulates are checked against it -- MuJoCo must render and range
    from where the tree says the camera and the lidar are."""
    model = mujoco.MjModel.from_xml_path(str(robots_spec.model_xml("myagv")))
    text = robots_spec.urdf_path("myagv").read_text()
    entries = [e for e in myagv_surface.tree() if e[0] != myagv_surface.FRAME_ODOM]
    # camera_link, imu_link and laser_frame are no more links of `myagv_urdf` than the
    # AiNex's lidar is of `ainex_description`: on the real robot each is a
    # `static_transform_publisher` line in `myagv_active.launch`, not a joint.
    launch_frames = {child for _, child, _, _ in myagv_surface.static_transforms().values()}
    check_robot("myagv", model, _Entries(entries), text, launch_frames)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    pos, quat = camera_link_pose(model, cam)
    _, _, want_pos, want_quat = myagv_surface.static_transforms()[
        myagv_surface.NODE_BASE2CAMERA]
    check("the rendered camera sits at camera_link",
          float(np.linalg.norm(pos - np.asarray(want_pos))) < TOL
          and quat_error(quat, want_quat) < TOL,
          f"model {np.round(pos, 4)} vs launch {want_pos}")


def check_so101() -> None:
    """The SO-101's tree is `robot_state_publisher`'s: the served description plus joint
    states. So it is checked the other way round from the two above -- the published tree,
    composed from the description at random joint positions, against MuJoCo's own forward
    kinematics of the compiled model at the same positions."""
    print("so101")
    import random

    from contracts.tf import UrdfTree

    model = mujoco.MjModel.from_xml_path(str(robots_spec.model_xml("so101")))
    data = mujoco.MjData(model)
    served = so101_surface.bringup_description(robots_spec.urdf_path("so101").read_text())
    tree = UrdfTree(served)

    links = set(tree.links)
    entries = tree.fixed() + tree.moving({j: 0.0 for j in so101_surface.JOINT_ORDER})
    children = {child for _, child, _, _ in entries}
    roots = {parent for parent, _, _, _ in entries} - children
    check("the tree has exactly one root, `world`", roots == {"world"}, str(sorted(roots)))
    check("every frame is a link of the description",
          {p for p, *_ in entries} | children <= links)
    check("every joint the state names moves a link",
          len(tree.moving({j: 0.0 for j in so101_surface.JOINT_ORDER})) == 6)

    rng = random.Random(7)
    worst_pos = worst_quat = 0.0
    for _ in range(50):
        wire = {}
        for joint in so101_surface.ARM_JOINTS:
            jid = model.joint(joint.removesuffix("_joint")).id
            low, high = model.jnt_range[jid]
            wire[joint] = rng.uniform(low, high)
        low, high = so101_surface.GRIPPER_RANGE
        wire[so101_surface.GRIPPER_JOINT] = rng.uniform(low, high)
        mujoco.mj_resetData(model, data)
        for joint, value in wire.items():
            mjcf = joint.removesuffix("_joint")
            if joint == so101_surface.GRIPPER_JOINT:
                value = so101_surface.to_mjcf_gripper(value)
            data.qpos[model.jnt_qposadr[model.joint(mjcf).id]] = value
        mujoco.mj_forward(model, data)
        poses = tree.link_poses(wire)
        base = model.body("base").id
        base_rot = data.xmat[base].reshape(3, 3)
        for body, link in so101_surface.MJCF_BODY_LINKS.items():
            bid = model.body(body).id
            pos = base_rot.T @ (data.xpos[bid] - data.xpos[base])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(
                base_rot.T @ data.xmat[bid].reshape(3, 3)).flatten())
            upos, uquat = poses[link]
            worst_pos = max(worst_pos, float(np.linalg.norm(pos - np.asarray(upos))))
            worst_quat = max(worst_quat, quat_error(quat, uquat))
    # The URDF writes its angles rounded (1.5708 for pi/2), so it agrees with the MJCF's
    # quaternions to about 1e-5 rad per joint, which compounds down the chain.
    check("all 7 links, at 50 random joint states, sit where MuJoCo puts them",
          worst_pos < 2e-4 and worst_quat < 2e-4,
          f"worst dpos {worst_pos:.2e} m, dquat {worst_quat:.2e}")


def check_ainex() -> None:
    model = ainex_model.build_spec().compile()
    text = robots_spec.urdf_path("ainex").read_text()
    statics = [e for e in urdf_fixed_joints(text) if e[1] != ainex_topics.FRAME_CAMERA]
    tree = TransformTree(
        model, root_body=ainex_topics.TF_ROOT_BODY, frames=ainex_topics.TF_FRAMES,
        cameras=ainex_topics.TF_CAMERAS,
        extra_static=statics + [(ainex_topics.TF_ROOT_FRAME, ainex_topics.FRAME_LASER,
                                 (0.0, 0.0, 0.20), (1, 0, 0, 0))],
    )
    check_robot("ainex", model, tree, text, {ainex_topics.FRAME_CAMERA,
                                             ainex_topics.FRAME_LASER})

    # The camera is the one frame this simulator knowingly moves: the vendor bolts it to
    # the torso and `ainex_model` step 4 puts it on the head, because that is where the
    # hardware's 2-DOF camera is. The pose must still be the vendor's -- so compose the
    # published head-relative frame back into the torso and compare with the URDF.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, ainex_model.CAMERA_NAME)
    pos, quat = camera_link_pose(model, cam)
    head = int(model.cam_bodyid[cam])
    torso = model.body(ainex_model.TORSO_BODY).id
    head_rot, torso_rot = data.xmat[head].reshape(3, 3), data.xmat[torso].reshape(3, 3)
    in_torso = torso_rot.T @ (data.xpos[head] + head_rot @ pos - data.xpos[torso])
    vendor = {c: (p, xyz) for p, c, xyz, _ in urdf_fixed_joints(text)}
    _, vendor_xyz = vendor[ainex_topics.FRAME_CAMERA]
    error = float(np.linalg.norm(in_torso - np.asarray(vendor_xyz)))
    check("the head-mounted camera frame is still at the vendor's torso offset",
          error < TOL, f"{error:.2e} m from {tuple(round(v, 4) for v in vendor_xyz)}")
    link_rot = np.zeros(9)
    mujoco.mju_quat2Mat(link_rot, quat)
    upright = float(np.linalg.norm(head_rot @ link_rot.reshape(3, 3) - torso_rot))
    check("...and points the way a URDF camera link does, not the way MuJoCo's does",
          upright < TOL, f"{upright:.2e} off the torso's own axes")


def main() -> int:
    print("transform tree vs description\n")
    check_myagv()
    check_so101()
    check_ainex()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
