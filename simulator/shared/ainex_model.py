"""The AiNex's MuJoCo model, built from the vendor URDF -- shared by every engine.

`shared/robots/ainex/` holds what Hiwonder ship: a flattened URDF and its meshes. What
they ship is not directly usable, and the gap is measured rather than guessed -- real
per-joint limits from the servo table, a holonomic base the URDF has no joint for,
collision cut back to the torso and the hands, the camera moved to the head where the
hardware actually carries it, and position actuators gained off each joint's own inertia.
Every one of those is commented where it happens.

This lives here, and not in an engine, for the reason every robot's description does: the
two engines must compile the *same* robot or a client could tell them apart. It is a
builder rather than a generated `model.xml` because the corrections are measurements taken
off the compiled model -- the torso hull is sized from one, the camera pose is read back
from another -- so a static file would be a second description that could drift from the
code that produced it.

Deliberately free of any engine import: `mujoco`, `numpy`, and the vendor servo table.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjSpec

import robots_spec
from ros_surfaces.ainex import servos
from ros_surfaces.ainex.gait import LegGeometry

#: The vendor description, in the shared spec tree every engine reads.
URDF_PATH = robots_spec.spec_dir("ainex") / "urdf" / "ainex.urdf"

# The torso. `base_link` sits above it in the URDF but carries no joint and no geometry,
# so MuJoCo merges it into the worldbody on import -- and would merge `body_link` too, for
# the same reason, if the virtual joints below did not give it a DoF. See PROVENANCE.md:
# compiling the vendor file untouched yields five disconnected root bodies.
TORSO_BODY = "body_link"

# Where the 25 STLs live, relative to the URDF. See `_load_robot_spec`.

# Where the 25 STLs live, relative to the URDF. See `_load_robot_spec`.
MESHDIR = "meshes"

# The vendor URDF bolts the camera here; step 4 moves it to the head.

# The vendor URDF bolts the camera here; step 4 moves it to the head.
CAMERA_PARENT_URDF = "body_link"

HEAD_TILT_BODY = "head_tilt_link"

CAMERA_LINK = "camera_link"

CAMERA_NAME = "front_camera"
# A generic 640x480 USB module. Hiwonder publish no intrinsics -- the product page says
# 120 degrees diagonal and the wiki 170, which cannot both be right -- so this is an
# assumption, kept in one place. `camera_info` on the wire is derived from it.

# A generic 640x480 USB module. Hiwonder publish no intrinsics -- the product page says
# 120 degrees diagonal and the wiki 170, which cannot both be right -- so this is an
# assumption, kept in one place. `camera_info` on the wire is derived from it.
CAMERA_FOVY_DEG = 58.0

# Bodies that keep their collision geometry. Everything else is switched off and replaced
# by a single torso hull; see step 3.

# Bodies that keep their collision geometry. Everything else is switched off and replaced
# by a single torso hull; see step 3.
HAND_BODIES = {"l_gripper_link", "r_gripper_link"}

# (closed, open) claw angle per hand, in radians.
#
# Measured by sweeping each gripper joint over its full servo travel and tracking the gap
# between the claw tip and the fixed jaw it closes against; see `_add_gripper_frames`. The
# claw swings through an arc, so the gap peaks mid-range and closes at *both* ends -- only
# one of those ends is a grasp and the other is the claw swung past the hand, which is why
# these cannot be read off the joint range.
#
# The joint keeps its full servo range regardless: a bus-servo command has to mean on this
# robot what it means on the real one, so these bound `set_gripper_ctrl_open`, not the
# joint. The two hands are not exact mirrors because the vendor meshes are not.

# (closed, open) claw angle per hand, in radians.
#
# Measured by sweeping each gripper joint over its full servo travel and tracking the gap
# between the claw tip and the fixed jaw it closes against; see `_add_gripper_frames`. The
# claw swings through an arc, so the gap peaks mid-range and closes at *both* ends -- only
# one of those ends is a grasp and the other is the claw swung past the hand, which is why
# these cannot be read off the joint range.
#
# The joint keeps its full servo range regardless: a bus-servo command has to mean on this
# robot what it means on the real one, so these bound `set_gripper_ctrl_open`, not the
# joint. The two hands are not exact mirrors because the vendor meshes are not.
GRIPPER_ANGLES = {"l": (-1.810, 0.690), "r": (1.660, -0.520)}

# Keep the torso hull clear of the floor. The base has no vertical DoF so it cannot fall.

# Keep the torso hull clear of the floor. The base has no vertical DoF so it cannot fall.
TORSO_CLEARANCE = 0.005

# Rated stall torque of the servos, N.m at 11.1 V, from Hiwonder's servo pages: the
# HX-35H/HX-35HM are quoted at 35 kg.cm and the HX-12H at 12 kg.cm.
#
# Which model sits on which joint is NOT published -- servo_controller.yaml lists ids, not
# part numbers -- so this mapping is an assumption: the small HX-12H on the grippers and
# head, where there is nothing to hold up, and the HX-35H everywhere else. It matters less
# than it looks, because on a 2.35 kg robot the stability bound below dominates the
# stiffness one for almost every joint.

# Rated stall torque of the servos, N.m at 11.1 V, from Hiwonder's servo pages: the
# HX-35H/HX-35HM are quoted at 35 kg.cm and the HX-12H at 12 kg.cm.
#
# Which model sits on which joint is NOT published -- servo_controller.yaml lists ids, not
# part numbers -- so this mapping is an assumption: the small HX-12H on the grippers and
# head, where there is nothing to hold up, and the HX-35H everywhere else. It matters less
# than it looks, because on a 2.35 kg robot the stability bound below dominates the
# stiffness one for almost every joint.
TORQUE_LARGE = 3.43

TORQUE_SMALL = 1.18

SMALL_SERVO_JOINTS = frozenset(servos.GRIPPER_JOINTS + servos.HEAD_JOINTS)

# The b601 lesson, at a smaller scale. Two competing requirements:
#   * stiffness -- the joint should command full torque at STIFFNESS_ERROR of error;
#   * stability -- with an explicit integrator, kv*dt/I < 2, i.e. kp <= ~I/dt^2.
# The AiNex's ankle and head links are of order 1e-4 kg.m^2, so a kv that looks sensible
# beside the hip makes them oscillate instead of hold. Hence kp is clamped against the
# measured inertia and kv follows from critical damping.


# The b601 lesson, at a smaller scale. Two competing requirements:
#   * stiffness -- the joint should command full torque at STIFFNESS_ERROR of error;
#   * stability -- with an explicit integrator, kv*dt/I < 2, i.e. kp <= ~I/dt^2.
# The AiNex's ankle and head links are of order 1e-4 kg.m^2, so a kv that looks sensible
# beside the hip makes them oscillate instead of hold. Hence kp is clamped against the
# measured inertia and kv follows from critical damping.
STIFFNESS_ERROR = 0.08  # rad of error at which the joint commands full torque

STABILITY_MARGIN = 0.4

SIM_TIMESTEP = 0.002

MIN_INERTIA = 1e-6


def robot_model_root_name() -> str:
    return TORSO_BODY


def build_spec(urdf_path: Path | None = None) -> MjSpec:
    """The AiNex model: the vendor URDF plus every documented correction below.

    Both engines call this, and it is the one description of this robot in the repo. It
    used to be a MolmoSpaces `_load_robot_spec` hook, which put the whole of it -- the
    real joint limits, the collision surgery, the measured actuator gains, the camera the
    vendor bolts to the wrong link -- inside one engine, where the other could not reach
    it. Nothing here needs an engine: `mujoco`, `numpy` and the vendor servo table.
    """
    spec = MjSpec.from_file(str(urdf_path or URDF_PATH))

    # MuJoCo strips the directory from URDF mesh filenames, so `l_knee_link.STL` is
    # looked up in meshdir and nowhere else -- see robots/URDF.md, where this is the
    # first of the two import behaviours that bite every time. Relative to the URDF's
    # own directory, which is where MuJoCo resolves it from.
    spec.meshdir = MESHDIR

    # Keep visual-only geoms. MuJoCo's URDF importer defaults `discardvisual` to true,
    # which throws away every geom that can neither collide nor be picked -- and step 3
    # below turns almost the whole robot non-colliding on purpose. Leave this at the
    # default and the AiNex compiles down to the two hand meshes and a hull: a robot
    # that drives correctly and renders as nothing. It costs one line and is invisible
    # until you look at a render, so `test_attach.py` counts geoms.
    #
    # A robot loaded from an MJCF needs none of this: the flag only defaults on for
    # URDF, so the same collision surgery there costs nothing.
    spec.compiler.discardvisual = False

    torso = spec.body(TORSO_BODY)
    if torso is None:
        raise ValueError(f"{TORSO_BODY!r} not found; vendor description changed?")

    # 0. Gravity compensation on every body.
    #
    #    The real AiNex sags: it is 2.35 kg on hobby servos. But the gait here is
    #    cosmetic and there is no balance controller, so a visibly drooping knee reads
    #    as a broken simulation rather than as fidelity, and every limb pose --
    #    including the replayed grasps -- would land somewhere other than commanded.
    #    A documented departure; see robots/README.md.
    for body in spec.bodies:
        body.gravcomp = 1.0

    # 1. Real joint limits, from the servo table.
    #
    #    The URDF gives all 24 joints the same +/-2.09, which is the servo's full
    #    240-degree travel about a centred zero rather than a per-joint calibration.
    #    servos.joint_limits() intersects that with what the servo can actually reach
    #    given the joint's own `init` count -- which differs from 500 on exactly four
    #    joints, and those four come out asymmetric. See servos.py.
    for name, (lower, upper) in servos.all_limits().items():
        joint = spec.joint(name)
        if joint is None:
            raise ValueError(f"joint {name!r} missing; vendor description changed?")
        joint.limited = True
        joint.range = [lower, upper]

    # 2. The virtual holonomic base, in the (x, y, theta) order
    #    HoloJointsRobotBaseGroup requires.
    #
    #    Deleting first is unconditional so that this works against a source that
    #    floats the torso on a freejoint as well as against the vendor URDF, which
    #    carries no joint here at all. A body may hold at most 6 DoF, so a freejoint
    #    and these three cannot coexist.
    #
    #    On this robot the joints do something extra: without a DoF, `body_link` is
    #    jointless and MuJoCo merges it into the worldbody exactly as it merges
    #    `base_link`, leaving five disconnected root bodies and dropping 0.743 kg out
    #    of the tree. Adding them is what makes the torso a body at all.
    for joint in list(torso.joints):
        spec.delete(joint)
    torso.add_site(name="base_site", pos=[0, 0, 0], group=3)
    torso.add_joint(
        name="base_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], damping=5
    )
    torso.add_joint(
        name="base_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], damping=5
    )
    torso.add_joint(
        name="base_theta", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.5
    )

    # 3. Collision: one torso hull, and the hands.
    #
    #    Nothing about locomotion comes from foot contact -- the torso rides
    #    world-aligned position actuators -- so colliding feet grip the floor at
    #    default friction and fight the servo, which shows up as a base that
    #    undershoots and picks up yaw it was never commanded. The feet are decorative
    #    in precisely the sense the myAGV's wheels are.
    #
    #    The hands keep their colliders, because a replayed `clamp_left` has to be able
    #    to touch something. That is the whole point of the arms being real.
    for geom in list(spec.geoms):
        body_name = geom.parent.name if geom.parent is not None else ""
        if body_name not in HAND_BODIES:
            geom.contype = 0
            geom.conaffinity = 0
        # ...and into the shared robots' render convention while we are here: group 2 is
        # visual, group 3 is collision. MuJoCo's URDF importer leaves every mesh in
        # group 0, which is invisible in a RoboCasa kitchen -- that engine renders
        # through a mask showing groups 1-2, because *its* collision hulls are the ones
        # in group 0. The robot spawned, served, and rendered as nothing at all: the
        # kitchen was there, the counter was there, and the AiNex standing on it was not.
        # Group 2 is visible under both engines' masks and under MuJoCo's default.
        geom.group = 2

    extent = _measure(spec)
    torso.add_geom(
        name="torso_hull",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[
            extent["dx"] / 2,
            extent["dy"] / 2,
            max((extent["z_max"] - extent["z_min"] - TORSO_CLEARANCE) / 2, 0.01),
        ],
        pos=[
            extent["cx"],
            extent["cy"],
            (extent["z_min"] + TORSO_CLEARANCE + extent["z_max"]) / 2 - extent["torso_z"],
        ],
        group=3,
        rgba=[1, 0, 0, 0.0],
        mass=0.0,
        contype=1,
        conaffinity=1,
    )

    # 4. Move the camera from the torso to the head.
    #
    #    The vendor URDF fixes camera_link to body_link. That is wrong about the
    #    hardware -- Hiwonder's own README calls it a "2-DOF HD camera" and it sits on
    #    the pan/tilt head -- and harmless in RViz, where nobody looks through it. Here
    #    the camera IS the sensor, so leaving it on the torso would make
    #    /head_pan_controller/command, colour tracking and every look-at behaviour
    #    untestable: panning the head would not move the view.
    #
    #    This is a correction *toward* the hardware, not a departure from it.
    #
    #    The pose is measured rather than retyped: compile with the head at zero, read
    #    where the vendor's camera actually ends up, and express that in the head's
    #    frame. So the neutral view matches the vendor description exactly and only its
    #    behaviour under head motion changes.
    _reparent_camera(spec, urdf_path)

    # 5. Grasp markers and TCP frames, placed by measurement.
    _add_gripper_frames(spec)

    # 6. Position actuators: 24 joints plus the 3 base axes.
    inertias = _joint_inertias(spec)
    for name in servos.SERVOS:
        torque = TORQUE_SMALL if name in SMALL_SERVO_JOINTS else TORQUE_LARGE
        _add_joint_actuator(spec, name, torque, inertias[name])

    mass, izz = _measure_base(spec)
    _add_base_actuator(spec, "base_x_act", "base_x", kp=600.0 * mass, inertia=mass)
    _add_base_actuator(spec, "base_y_act", "base_y", kp=600.0 * mass, inertia=mass)
    _add_base_actuator(
        spec, "base_theta_act", "base_theta", kp=2000.0 * izz, inertia=izz
    )
    return spec


def _reparent_camera(spec: MjSpec, urdf_path=None) -> None:
    """Attach `front_camera` to the head, at the pose the vendor gives the torso one.

    The offset is read back out of the URDF rather than measured off the compiled
    model, because `camera_link` does not survive the import: it carries no joint and
    no geometry, so MuJoCo merges it into `body_link` exactly as it merges `base_link`
    (see PROVENANCE.md). Parsing the vendor file keeps the number sourced rather than
    retyped -- change the description and this follows.
    """
    camera_in_torso, camera_rpy = _urdf_camera_origin(urdf_path)
    if any(camera_rpy):
        # The vendor's camera joint has rpy 0 0 0. If that ever changes, the frame
        # construction below is no longer just the torso's axes and must be revisited.
        raise ValueError(f"camera joint gained a rotation {camera_rpy}; revisit this")

    model = spec.copy().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    torso = model.body(TORSO_BODY)
    head = model.body(HEAD_TILT_BODY)
    torso_rot = data.xmat[torso.id].reshape(3, 3)
    head_rot = data.xmat[head.id].reshape(3, 3)

    camera_world = data.xpos[torso.id] + torso_rot @ camera_in_torso
    offset = head_rot.T @ (camera_world - data.xpos[head.id])
    # Orientation of the camera link relative to the head, at the zero head pose.
    relative = head_rot.T @ torso_rot

    # MuJoCo cameras look down their own -z with +y up; the URDF link is the usual
    # robot convention of +x forward, +y left, +z up. Build the frame from those axes
    # rather than hardcoding a quaternion, so it survives a convention change upstream.
    forward, left, up = relative[:, 0], relative[:, 1], relative[:, 2]
    rotation = np.column_stack([-left, up, -forward])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).flatten())

    head_body = spec.body(HEAD_TILT_BODY)
    if head_body is None:
        raise ValueError(f"{HEAD_TILT_BODY!r} not found; vendor description changed?")
    camera = head_body.add_camera()
    camera.name = CAMERA_NAME
    camera.pos = offset.tolist()
    camera.quat = quat.tolist()
    camera.fovy = CAMERA_FOVY_DEG


def _urdf_camera_origin(urdf_path: Path | None = None) -> tuple[np.ndarray, tuple[float, float, float]]:
    """The `camera` fixed joint's origin, straight out of the vendor URDF."""
    urdf = Path(urdf_path or URDF_PATH)
    match = re.search(
        r'name="camera"\s*type="fixed">\s*<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)"\s*/>'
        r'\s*<parent\s+link="([^"]+)"',
        urdf.read_text(),
        re.S,
    )
    if match is None:
        raise ValueError(f"no `camera` fixed joint in {urdf}; description changed?")
    if match.group(3).strip() != CAMERA_PARENT_URDF:
        raise ValueError(
            f"camera is parented to {match.group(3)!r}, not {CAMERA_PARENT_URDF!r}; "
            "the defect this works around may have been fixed upstream"
        )
    xyz = np.array([float(v) for v in match.group(1).split()])
    rpy = tuple(float(v) for v in match.group(2).split())
    return xyz, rpy


def _add_gripper_frames(spec: MjSpec) -> None:
    """Claw-tip and fixed-jaw marker geoms, plus a TCP site, for each hand.

    Each hand is a **single hinged claw**: `{l,r}_gripper_link` is the only moving
    part, and it closes against the fixed structure of `{l,r}_el_yaw_link`. So unlike
    a parallel-jaw gripper there is no symmetric pair of fingers to measure between --
    `inter_finger_dist` is claw tip to fixed jaw, which is why the two markers live on
    different bodies.

    Both points are measured from the meshes rather than typed in: the claw tip is the
    vertex furthest out along the hand's own axis, and the fixed-jaw point is whatever
    that tip comes closest to when the claw is shut. The two hands' meshes are not
    exact mirrors of each other in the vendor CAD (957 vertices against 985), so each
    side is measured separately rather than one being negated.
    """
    model = spec.copy().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for side, outward in (("l", 1.0), ("r", -1.0)):
        claw_body = model.body(f"{side}_gripper_link")
        hand_body = model.body(f"{side}_el_yaw_link")
        claw_pts = _mesh_points(model, claw_body.id)
        hand_pts = _mesh_points(model, hand_body.id)

        tip = claw_pts[np.argmax(outward * claw_pts[:, 1])]
        # Only the jaw end of the forearm can oppose the claw; the rest is the shaft.
        jaw_pts = hand_pts[outward * hand_pts[:, 1] > 0.06]

        closed, _ = GRIPPER_ANGLES[side]
        joint = model.joint(f"{side}_gripper")
        data.qpos[model.jnt_qposadr[joint.id]] = closed
        mujoco.mj_forward(model, data)
        tip_world = data.xmat[claw_body.id].reshape(3, 3) @ tip + data.xpos[claw_body.id]
        hand_rot, hand_pos = (
            data.xmat[hand_body.id].reshape(3, 3),
            data.xpos[hand_body.id],
        )
        # numpy raises overflow/invalid/divide-by-zero from this `matmul` under one
        # engine's MuJoCo (3.3.1) and not the other's (3.5.0), on float32 mesh vertices
        # against a rotation. Measured before silencing: every input is finite, the
        # product is finite, and everything this file computes comes out identical to
        # six decimals under both -- mass, ride height, hull, all four claw frames, the
        # camera pose. The flags are the BLAS path's, not the arithmetic's. Left alone
        # they print on every start of one engine and not the other, which is the worst
        # way for a warning to behave: the next one that means something reads as more
        # of the same.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            jaw_world = jaw_pts @ hand_rot.T + hand_pos
        palm = jaw_pts[int(np.argmin(np.linalg.norm(jaw_world - tip_world, axis=1)))]
        data.qpos[model.jnt_qposadr[joint.id]] = 0.0

        claw = spec.body(f"{side}_gripper_link")
        hand = spec.body(f"{side}_el_yaw_link")
        for body, name, pos in (
            (claw, f"{side}_claw_tip", tip),
            (hand, f"{side}_claw_palm", palm),
        ):
            body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.003, 0, 0],
                pos=np.asarray(pos).tolist(),
                contype=0,
                conaffinity=0,
                group=3,
                rgba=[1, 0, 0, 0.4],
                mass=0.0,
            )
        # TCP on the *fixed* hand, so it does not swing with the claw. Midway between
        # the two markers is where a grasped object actually sits.
        hand.add_site(
            name=f"{side}_tcp", pos=((np.asarray(palm) + tip) / 2.0).tolist(), group=3
        )


def _mesh_points(model, body_id: int) -> np.ndarray:
    """Vertices of a body's mesh geom, in the body's own frame.

    The first *mesh* on the body, not the first geom: a marker sphere or a primitive hull
    has `geom_dataid == -1`, and indexing `mesh_vertadr` with that reads the last mesh in
    the model instead of raising. The measurement then comes out of another link's
    vertices -- which is not hypothetical. It showed up as `overflow encountered in
    matmul` under one engine's MuJoCo and silently correct numbers under the other's,
    which is the two engines compiling different robots: the thing the shared spec exists
    to make impossible.
    """
    geoms = [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] == body_id
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    if not geoms:
        raise ValueError(f"body {body_id} has no mesh geom to measure")
    geom = geoms[0]
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    return model.mesh_vert[start : start + count] + model.geom_pos[geom]


def _measure(spec: MjSpec) -> dict[str, float]:
    """Torso extent and the standing height, from a throwaway compile at the rest pose.

    Measured with the robot in the vendor's `init_pose`, because that -- not the URDF's
    all-zeros straight-legged pose -- is what it actually stands in.
    """
    model = spec.copy().compile()
    data = mujoco.MjData(model)
    for name in servos.INIT_POSE:
        joint = model.joint(name)
        data.qpos[model.jnt_qposadr[joint.id]] = servos.INIT_POSE[name]
    mujoco.mj_forward(model, data)

    torso_id = model.body(TORSO_BODY).id
    torso_pos = data.xpos[torso_id]

    # The hull spans the torso alone; a raised arm must not inflate it.
    torso_pts = np.array(
        [data.geom_xpos[g] for g in range(model.ngeom) if model.geom_bodyid[g] == torso_id]
    )
    if not len(torso_pts):
        torso_pts = torso_pos.reshape(1, 3)
    lo, hi = torso_pts.min(axis=0), torso_pts.max(axis=0)

    # Lowest point of anything, so the robot can be stood on the floor. Measured from
    # the mesh vertices, not from `geom_rbound`: that is a bounding-sphere radius, and
    # on a foot mesh it sits several centimetres below the sole, which would hang the
    # robot in the air by that much.
    # See the note on the same guard in _add_gripper_frames: spurious FPE flags from
    # numpy under one engine's MuJoCo and not the other's, with every input and the
    # product measured finite and every number this file returns identical under both.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        foot_z = min(
            float(
                (_mesh_points(model, bid) @ data.xmat[bid].reshape(3, 3).T
                 + data.xpos[bid])[:, 2].min()
            )
            for bid in range(1, model.nbody)
            if any(
                model.geom_bodyid[g] == bid and model.geom_dataid[g] >= 0
                for g in range(model.ngeom)
            )
        )
    return {
        "dx": max(float(hi[0] - lo[0]), 0.06),
        "dy": max(float(hi[1] - lo[1]), 0.06),
        "cx": float((hi[0] + lo[0]) / 2 - torso_pos[0]),
        "cy": float((hi[1] + lo[1]) / 2 - torso_pos[1]),
        "z_min": float(lo[2]),
        "z_max": float(hi[2]),
        "torso_z": float(torso_pos[2]),
        "foot_z": foot_z,
    }


def ride_height(spec: MjSpec) -> float:
    """How far to lift the robot so it stands on the floor in its rest pose."""
    return -_measure(spec)["foot_z"]


def leg_geometry(model, namespace: str = "") -> LegGeometry:
    """Thigh and shank lengths, read off the compiled model.

    `gait.py` is pure and takes these as data, so the gait stays correct if the vendor
    description changes rather than silently walking on the wrong-length legs. Takes a
    namespace because by the time anything asks, the robot has been attached into a
    scene under its prefix.
    """
    def link(child: str) -> float:
        body = model.body(f"{namespace}{child}")
        return float(np.linalg.norm(model.body_pos[body.id]))

    return LegGeometry(thigh=link("l_knee_link"), shank=link("l_ank_pitch_link"))


def _joint_inertias(spec: MjSpec) -> dict[str, float]:
    """Effective inertia per joint, from `dof_M0` on a throwaway compile."""
    model = spec.copy().compile()
    return {
        name: max(
            float(model.dof_M0[model.jnt_dofadr[model.joint(name).id]]), MIN_INERTIA
        )
        for name in servos.SERVOS
    }


def _measure_base(spec: MjSpec) -> tuple[float, float]:
    """Total mass and yaw inertia about the torso, from a throwaway compile."""
    model = spec.copy().compile()
    mass = float(model.body_mass.sum())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin = data.xpos[model.body(TORSO_BODY).id][:2]
    izz = 0.0
    for bid in range(1, model.nbody):
        r = data.xipos[bid][:2] - origin
        izz += float(model.body_inertia[bid][2] + model.body_mass[bid] * float(r @ r))
    return mass, max(izz, 1e-4)


def _add_joint_actuator(spec: MjSpec, joint: str, torque: float, inertia: float) -> None:
    wanted = torque / STIFFNESS_ERROR
    stable = STABILITY_MARGIN * inertia / (SIM_TIMESTEP**2)
    kp = min(wanted, stable)
    kv = 2.0 * float(np.sqrt(kp * inertia))  # critically damped

    act = spec.add_actuator()
    act.name = joint
    act.target = joint
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm[0] = kp
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kv
    lower, upper = servos.joint_limits(joint)
    act.ctrlrange = np.array([lower, upper])
    act.ctrllimited = True


def _add_base_actuator(spec: MjSpec, name: str, joint: str, kp: float, inertia: float
                       ) -> None:
    act = spec.add_actuator()
    act.name = name
    act.target = joint
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm[0] = kp
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    kv = 2.0 * float(np.sqrt(kp * inertia))
    act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kv
    act.ctrlrange = np.array([-25.0, 25.0])
