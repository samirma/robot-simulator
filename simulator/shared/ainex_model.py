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

import math
import re
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjSpec

import robots_spec
from mujoco_bridge import is_loose
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

#: The two bodies whose soles touch the ground -- the vendor's ankle-roll links, which
#: carry the foot plates. Defined here because the model owns its own body names;
#: `ros_surfaces/ainex/ground.py` imports these rather than keeping a second list, since
#: a robot standing on one name and ground-following another is a robot in the air.
FEET = ("l_ank_roll_link", "r_ank_roll_link")
#: The feet's `conaffinity` as built. It exists at build time only so the compiler gives
#: the foot meshes a convex hull -- MuJoCo builds one only for meshes some collidable geom
#: uses, and a foot compiled at 0/0 can never collide however the masks are set later.
#: Which bit it is does not matter: `enable_foot_contacts` re-points it against the scene
#: it is grafted into. Until that runs the feet touch nothing, because nothing else in
#: either engine's scenes carries this bit in its `contype`.
FOOT_CONTACT_BIT = 2

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

# Keep the torso hull clear of the surface the feet stand on. The hull is the one part of
# the body that collides with the world, and a crouch -- legs folded, torso lowered by the
# ground-follow -- brings it within centimetres of the worktop; seated on it, the hull
# would carry the robot's weight against the z actuator instead of the soles.
TORSO_CLEARANCE = 0.005

# How far the torso may lean about its lateral axis (`base_pitch`), radians: a little
# back, most of the way to horizontal forwards. This is the lean the real robot's hip
# chain produces when it bends to pick something off the ground -- with the base riding
# the torso, folding the hips alone lifts the legs, so the lean is a joint of its own,
# authored by action groups (`crawl_left`/`crawl_right`) and zero everywhere else.
BASE_PITCH_RANGE = (-0.2, 0.9)

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

    # 2. The virtual base: (x, y, theta) first, in the order HoloJointsRobotBaseGroup
    #    requires, then z and pitch.
    #
    #    Deleting first is unconditional so that this works against a source that
    #    floats the torso on a freejoint as well as against the vendor URDF, which
    #    carries no joint here at all. A body may hold at most 6 DoF, so a freejoint
    #    and these cannot coexist.
    #
    #    On this robot the joints do something extra: without a DoF, `body_link` is
    #    jointless and MuJoCo merges it into the worldbody exactly as it merges
    #    `base_link`, leaving five disconnected root bodies and dropping 0.743 kg out
    #    of the tree. Adding them is what makes the torso a body at all.
    #
    #    **The order is load-bearing.** MuJoCo applies a body's joints in sequence and a
    #    hinge rotates the axes of every joint after it. z after theta is still world z,
    #    because yaw leaves the vertical alone; pitch after theta is about the torso's
    #    own lateral axis, so a leaning robot leans along its heading. Pitch before z
    #    would tilt the axis the ground-follow drives, and the robot would slide
    #    forward every time it crouched.
    #
    #    z is not a freedom the robot has, it is one the *ground* has: nothing commands
    #    it directly. `ros_surfaces/ainex/ground.py` solves it every control tick so the
    #    stance sole sits on whatever a ray-cast finds beneath it, and integrates a fall
    #    when it finds nothing. Until it existed the torso rode at its graft height for
    #    ever -- walk it off the worktop and it hung in the air over the floor, and no
    #    number anywhere said so.
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
    torso.add_joint(
        name="base_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], damping=5
    )
    pitch = torso.add_joint(
        name="base_pitch", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.5
    )
    pitch.limited = True
    pitch.range = list(BASE_PITCH_RANGE)

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
    #
    #    The feet are the third case, and they have to be settled *here* rather than at
    #    spawn: MuJoCo builds a mesh's convex-hull graph only for meshes that some
    #    collidable geom uses, so a foot compiled at contype/conaffinity 0 comes out with
    #    `mesh_graphadr == -1` and can never collide with anything afterwards, whatever
    #    the masks are set to. Measured: with the masks flipped post-compile the foot geom
    #    passed 17 mm from a 20 mm apple, both rbounds overlapping, and MuJoCo reported
    #    zero contacts. So they are made collidable at build time and given `conaffinity`
    #    alone -- see `enable_foot_contacts` for why that side and not `contype`.
    for geom in list(spec.geoms):
        body_name = geom.parent.name if geom.parent is not None else ""
        if body_name in FEET:
            geom.contype = 0
            geom.conaffinity = FOOT_CONTACT_BIT
        elif body_name not in HAND_BODIES:
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

    # 6. Position actuators: 24 joints plus the 5 base axes.
    inertias = _joint_inertias(spec)
    for name in servos.SERVOS:
        torque = TORQUE_SMALL if name in SMALL_SERVO_JOINTS else TORQUE_LARGE
        _add_joint_actuator(spec, name, torque, inertias[name])

    mass, izz, iyy = _measure_base(spec)
    _add_base_actuator(spec, "base_x_act", "base_x", kp=600.0 * mass, inertia=mass)
    _add_base_actuator(spec, "base_y_act", "base_y", kp=600.0 * mass, inertia=mass)
    _add_base_actuator(
        spec, "base_theta_act", "base_theta", kp=2000.0 * izz, inertia=izz
    )
    _add_base_actuator(spec, "base_z_act", "base_z", kp=600.0 * mass, inertia=mass)
    _add_base_actuator(
        spec, "base_pitch_act", "base_pitch", kp=2000.0 * iyy, inertia=iyy,
        ctrlrange=BASE_PITCH_RANGE,
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

    **The geom's rotation counts, not just its offset.** Every one of this URDF's 25 mesh
    geoms carries a non-identity `geom_quat` -- the vendor orients each link's mesh inside
    its body -- so vertices offset by `geom_pos` alone are in no frame at all. Dropping it
    put `ride_height` 42.7 mm too high and stood the robot that far off the worktop on
    both engines, and it could not be caught by measuring the soles afterwards because
    that measurement went through here too: the graft and its check were wrong by the same
    amount and agreed with each other perfectly.
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
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
    return model.mesh_vert[start : start + count] @ rot.reshape(3, 3).T \
        + model.geom_pos[geom]


def _measure(spec: MjSpec, lean: float = 0.0) -> dict[str, float]:
    """Torso extent and the standing height, from a throwaway compile at the rest pose.

    Measured with the robot in the vendor's `init_pose`, because that -- not the URDF's
    all-zeros straight-legged pose -- is what it actually stands in.

    `lean` is the torso's `base_pitch`, and the two things this returns want different
    values of it. The hull's `dx`/`dy`/`cx`/`cy` are extents along *world* axes, so they
    are only the torso's own box while the torso is upright; measure them leaning and the
    box is a skewed shadow of one. `foot_z` is the opposite: the robot stands leaning (see
    `stance_lean`), so the height it stands at is the height it has when it does.
    """
    model = spec.copy().compile()
    data = mujoco.MjData(model)
    for name in servos.INIT_POSE:
        joint = model.joint(name)
        data.qpos[model.jnt_qposadr[joint.id]] = servos.INIT_POSE[name]
    pitch = model.joint("base_pitch")
    data.qpos[model.jnt_qposadr[pitch.id]] = lean
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


def stance_lean(spec: MjSpec) -> float:
    """The `base_pitch` that puts the soles flat in the vendor's `init_pose`, radians.

    The vendor's init pose is not a straight-legged stand: its left leg reads
    hip_pitch -0.828, knee +1.192, ank_pitch +0.625, and on the URDF's axes those sum to
    **-14.95 degrees** rather than to zero. On the real robot that difference is their
    `hip_pitch_offset`, and it leans the *torso* forward over feet that stay flat -- the
    hips carry the body, so a rotation left over in the chain shows up above the ankle.

    Our torso is bolted to planar joints, so with `base_pitch` at zero the same 14.95
    degrees landed on the feet instead: both soles tilted toe-up, the toe 37.6 mm off the
    surface, and the robot balanced on two heel corners. `gait.py` predicted exactly this
    ("the robot would walk on its heels") and corrects it only for the walking gait, which
    solves its own flat-sole IK. Standing had nothing.

    Nothing here catches that on its own, and that is the point of measuring rather than
    typing the number: the feet do not collide, so a heel-stand neither falls nor
    complains, and `ride_height`, `sole_z`, `report_sole_contact` and the attach test all
    report a perfect 0.00 mm gap because they all measure the same single lowest vertex --
    which *is* on the surface. It is the heel.

    So: read the foot's own world orientation off the compiled model at `init_pose` and
    return the lean that cancels it. Derived from the pose, so a vendor pose that changes
    cannot silently put the robot back on its heels, and measured through MuJoCo's
    `xmat` -- the independent frame, not the mesh arithmetic `_mesh_points` does.
    """
    model = spec.copy().compile()
    data = mujoco.MjData(model)
    for name in servos.INIT_POSE:
        joint = model.joint(name)
        data.qpos[model.jnt_qposadr[joint.id]] = servos.INIT_POSE[name]
    mujoco.mj_forward(model, data)

    # Each sole's tilt about the lateral axis: the angle of its own z-axis from world up.
    # The two are mirror images and must agree; averaging them says so, and a leg table
    # that made them differ would show up in the assertion below rather than as a robot
    # standing on one heel.
    tilts = []
    for foot in FEET:
        rot = data.xmat[model.body(foot).id].reshape(3, 3)
        tilts.append(math.atan2(rot[0, 2], rot[2, 2]))
    if abs(tilts[0] - tilts[1]) > math.radians(1.0):
        raise ValueError(
            f"the two soles disagree about their tilt ({math.degrees(tilts[0]):+.2f} vs "
            f"{math.degrees(tilts[1]):+.2f} deg); the init pose is not symmetric"
        )
    lean = -sum(tilts) / len(tilts)
    return float(min(max(lean, BASE_PITCH_RANGE[0]), BASE_PITCH_RANGE[1]))


def ride_height(spec: MjSpec) -> float:
    """How far to lift the robot so it stands on the floor in its rest pose.

    In the pose it actually holds, which leans: see `stance_lean`. Measured upright this
    was the height of a heel corner, 0.2114 m, with the rest of the sole above it.
    """
    return -_measure(spec, lean=stance_lean(spec))["foot_z"]


def stand(model, data, namespace: str = "") -> None:
    """Put a grafted robot into the pose it stands in: the vendor's init pose, leaning.

    Both engines call this straight after the graft, and it is one function rather than a
    loop in each because the two must stand this robot up identically -- a client that
    could tell the engines apart by the pose of a robot's legs is the invariant broken.

    Joint *and* control: these are position actuators, so a ctrl left at zero snaps all 24
    limbs out of the pose on the first step. `base_pitch` is included and is the reason
    this exists: `ride_height` is measured with the torso leaning (see `stance_lean`), so
    a robot grafted at that height and left upright stands 9 mm into the surface until the
    ROS surface's first tick -- and for ever in a run that serves no surface at all.
    """
    from ros_surfaces.ainex.actions import BASE_PITCH, rest_pose  # noqa: PLC0415

    for joint, angle in rest_pose().items():
        # Every servo's actuator is named after its joint; the base axes carry `_act`,
        # because they are the model's own and not the vendor's.
        actuator = f"{joint}_act" if joint == BASE_PITCH else joint
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{namespace}{joint}")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{namespace}{actuator}")
        if jid < 0 or aid < 0:
            raise ValueError(
                f"ainex joint/actuator {namespace}{joint!r} missing from the model"
            )
        data.qpos[model.jnt_qposadr[jid]] = angle
        data.ctrl[aid] = angle
    mujoco.mj_forward(model, data)


def enable_foot_contacts(model, namespace: str = "") -> int:
    """Let the feet meet what is lying on the floor, and nothing else. Returns the bit used.

    Every geom but the hands and the torso hull was made non-colliding in `build_spec`
    because contact with the *ground* fights the base: the torso rides world-aligned
    position actuators, so a foot gripping the floor at default friction shows up as a base
    that undershoots and picks up yaw it was never commanded. That reasoning is about the
    floor and was applied to everything, which left a robot that walks straight through an
    apple without touching it -- and, with the ground probe fixed, over it without touching
    it. Neither is interaction.

    So the feet get a collision class of their own: they meet **loose** bodies
    (`mujoco_bridge.is_loose` -- anything on a free joint) and pass through the world.

    1. `FOOT_BIT` is the lowest bit set in no geom's `conaffinity` -- excluding the feet,
       whose build-time `FOOT_CONTACT_BIT` is what this may be about to move. It is
       computed rather than chosen because a scene can already be using it: iTHOR writes
       `conaffinity` 7 and 15, so on FloorPlan1 the first free bit is 16, not 2.
    2. The feet take that bit as their whole `conaffinity`, with `contype` 0.
    3. Loose geoms get the bit added to `contype`; every other geom gets it cleared.
    4. The robot's own geoms are otherwise left alone -- the hands and the torso hull are
       1/1, and `FOOT_BIT` is never bit 1, so a foot cannot collide with its own robot.

    **The bit lives on the feet's `conaffinity` and the scene's `contype`, not the other
    way round, and that asymmetry is the safety.** A contact needs the bit in one side's
    `contype`; iTHOR's `contype` values are only 0, 1 and 8, so a robot grafted into a
    scene by some path that never calls this function has inert feet rather than feet
    snagging on 892 kitchen geoms. Mirrored, the untagged case is the dangerous one.
    """
    feet = {f"{namespace}{f}" for f in FEET}

    def body_of(gid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[gid])) or ""

    conaffinity_used = 0
    for gid in range(model.ngeom):
        if body_of(gid) not in feet:
            conaffinity_used |= int(model.geom_conaffinity[gid])
    bit = 1
    while bit & conaffinity_used:
        bit <<= 1
        if bit > 0x40000000:
            raise RuntimeError(
                "no collision bit left for the AiNex's feet: every usable conaffinity bit "
                f"is taken by this scene (union 0b{conaffinity_used:b})"
            )

    for gid in range(model.ngeom):
        name = body_of(gid)
        if name in feet:
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = bit
            continue
        if namespace and name.startswith(namespace):
            continue  # the robot's own hands and hull keep the contacts they were given
        if is_loose(model, int(model.geom_bodyid[gid])):
            model.geom_contype[gid] |= bit
        else:
            model.geom_contype[gid] &= ~bit

    # `body_contype`/`body_conaffinity` are the OR over a body's geoms, computed once at
    # compile and used to prune **whole bodies** in broadphase -- so a geom mask edited
    # afterwards is never reached and the contact silently does not exist. Measured: with
    # the geom masks correct and the apple placed at the exact centroid of a foot mesh,
    # MuJoCo reported the apple touching the table and nothing else.
    for bid in range(model.nbody):
        adr, num = int(model.body_geomadr[bid]), int(model.body_geomnum[bid])
        contype = conaffinity = 0
        for gid in range(adr, adr + num):
            contype |= int(model.geom_contype[gid])
            conaffinity |= int(model.geom_conaffinity[gid])
        model.body_contype[bid] = contype
        model.body_conaffinity[bid] = conaffinity
    return bit


def sole_z(model, data, namespace: str = "") -> float:
    """World z of the robot's lowest point, off the compiled scene it was grafted into.

    The check on `ride_height`: that number is measured on a throwaway compile of the
    robot alone, and grafting it onto a surface is a separate arithmetic that has been
    wrong -- an engine handing it the *arm's* mount height, which carries the SO-101's
    base-plate clearance, stands this robot a few millimetres off the worktop. Nothing
    catches that on its own, because the feet are deliberately non-colliding (see the
    collision surgery in `build_spec`): a floating AiNex neither falls nor complains, and
    at 4 mm it reads as a rendering artefact rather than as a placement bug.

    Measured from mesh vertices, and for the same reason `_measure` is: `geom_rbound` is
    a bounding-sphere radius and on a foot mesh sits centimetres below the sole.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lowest = None
        for bid in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if namespace and not name.startswith(namespace):
                continue
            if not any(model.geom_bodyid[g] == bid
                       and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                       for g in range(model.ngeom)):
                continue
            pts = _mesh_points(model, bid)
            z = float((pts @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid])[:, 2].min())
            lowest = z if lowest is None else min(lowest, z)
    if lowest is None:
        raise ValueError(f"no mesh geoms under namespace {namespace!r} to measure")
    return lowest


#: How far off the surface the soles may sit before an engine says so. A tenth of a
#: millimetre: the graft is exact arithmetic, so anything above rounding is a real error
#: in it rather than tolerance to be absorbed.
SOLE_TOLERANCE = 1e-4


class LowestPoint:
    """The lowest mesh vertex of chosen bodies, cheaply, every control tick.

    `sole_z` above scans every body and geom by name on each call, which is fine once at
    spawn and not fine 10 times a second. This caches, per mesh geom under `namespace`,
    the geom id and its vertices, and per call does one small matmul per geom through
    MuJoCo's own `geom_xmat`/`geom_xpos` -- the independent frame the sole check insists
    on, and the one `ride_height` deliberately does not share.
    """

    def __init__(self, model, namespace: str = "", bodies: tuple[str, ...] | None = None):
        self._model = model
        self._geoms: list[tuple[int, int, np.ndarray]] = []
        for g in range(model.ngeom):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            bid = int(model.geom_bodyid[g])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if namespace and not name.startswith(namespace):
                continue
            if bodies is not None and name not in bodies:
                continue
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            self._geoms.append((g, bid, np.array(model.mesh_vert[start:start + count])))
        if not self._geoms:
            raise ValueError(f"no mesh geoms to measure under {namespace!r} for {bodies}")

    def per_body(self, data) -> dict[int, np.ndarray]:
        """body id -> world xyz of that body's lowest mesh vertex."""
        out: dict[int, np.ndarray] = {}
        for g, bid, verts in self._geoms:
            world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            low = world[int(np.argmin(world[:, 2]))]
            if bid not in out or low[2] < out[bid][2]:
                out[bid] = low
        return out

    def lowest(self, data) -> np.ndarray:
        return min(self.per_body(data).values(), key=lambda p: p[2])


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


def _measure_base(spec: MjSpec) -> tuple[float, float, float]:
    """Total mass, and the yaw and pitch inertias about the torso, from a throwaway compile.

    Both inertias by the same parallel-axis approximation: each body's own principal
    inertia about that axis plus its mass times its squared distance from the torso's
    axis. Pitch is about the torso's y through its origin, so the distance is in x-z.
    """
    model = spec.copy().compile()
    mass = float(model.body_mass.sum())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin = data.xpos[model.body(TORSO_BODY).id]
    izz = iyy = 0.0
    for bid in range(1, model.nbody):
        r = data.xipos[bid] - origin
        izz += float(model.body_inertia[bid][2] + model.body_mass[bid] * float(r[:2] @ r[:2]))
        iyy += float(model.body_inertia[bid][1] + model.body_mass[bid] * float(r[[0, 2]] @ r[[0, 2]]))
    return mass, max(izz, 1e-4), max(iyy, 1e-4)


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


def _add_base_actuator(spec: MjSpec, name: str, joint: str, kp: float, inertia: float,
                       ctrlrange=(-25.0, 25.0)) -> None:
    act = spec.add_actuator()
    act.name = name
    act.target = joint
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm[0] = kp
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    kv = 2.0 * float(np.sqrt(kp * inertia))
    act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kv
    act.ctrlrange = np.array(ctrlrange, dtype=np.float64)
