"""Forward and inverse kinematics for the SO-101 arm, in NumPy only.

The link constants below are transcribed from TheRobotStudio's official
``so101_new_calib.xml`` (``robots_specs/so101/``), which the simulator's
``simulator/shared/robots/so101/model.xml`` reproduces body for body: each body's
fixed ``pos``/``quat`` relative to its parent (quaternions as MuJoCo normalises
them), and every joint's rotation about its own local ``z`` axis.
``tests/test_kinematics.py`` re-derives the same poses from MuJoCo and asserts
they agree, so a transcription slip fails loudly rather than producing a
plausible-but-wrong trajectory.

Deliberately MuJoCo-free: the policy that ships against the real rosbridge
robot imports this module, and that path must not need a simulator installed.
``mujoco`` is therefore a *test-only* dependency, in the ``kinematics`` extra
rather than ``dev`` -- ``import mujoco`` failing inside ``robot_console/.venv``
is a deliberate property of this project, so ``tests/test_kinematics.py`` skips
when it is absent and CI installs the extra to run it.

**The TCP is the official ``gripperframe`` site, and it moved on 2026-09-27.** Until
then the model was mujoco_menagerie's, which carries that site 19.9 mm further along
the gripper body's x (= the jaw axis): at the zero pose it landed at ``(0.391361,
-0.000979, 0.246345)``, where the official one lands at ``(0.391362, -0.000011,
0.226469)`` -- within 0.3 mm of the sibling ~/robot checkout's ``(0.391356,
-0.000305, 0.226343)``, which uses the official description too. Everything the
console aims with was kept on the same physical point by moving its offset rather
than its target: ``JAW_CENTER_OFFSET`` and ``REACH_POINT_OFFSET`` below each grew by
exactly that 19.9 mm, so ``jaw_center`` and the preflight's reach gate land where
they did before to 0.4 um. A constant copied from before that date is 20 mm out --
twice the jaw's capture window -- while every number still looks plausible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

#: Arm joints in the order used everywhere: actions, ``joint_pos``, IK.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)
GRIPPER_JOINT = "gripper_joint"

#: Full 6-DoF command order (five arm joints then the gripper).
JOINT_ORDER: tuple[str, ...] = (*ARM_JOINTS, GRIPPER_JOINT)

#: Commanded joint ranges, radians. **These are this MJCF's own limits, not the narrower
#: ones the sibling rig's description declares** -- and the difference is not cosmetic.
#:
#: `ros2_so_arm` narrows `wrist_flex` to 1.6 and `wrist_roll` to 2.3, below what the
#: mechanism allows (1.658063 and 2.7438473). The sibling rig accepts the resulting clip
#: as a known cost. It does not have to be paid here, because this arm keeps the
#: mechanism's full ranges -- and the clip lands exactly where a VLA
#: needs the room: mapped into our frame, MolmoAct2's action band reaches +1.630 on
#: `wrist_flex` and **+2.715 on `wrist_roll`**, so a 2.3 limit silently truncates the
#: most-rolled 24% of what the checkpoint can ask for.
#:
#: One deliberate exception since the model became the official `so101_new_calib.xml`
#: (2026-09-27): its `wrist_roll` runs -2.7438473..**+2.8412063**, asymmetric, where
#: mujoco_menagerie's ran +/-2.7438473. The table keeps the symmetric band, which is
#: inside the joint's range and is what `level_jaw_roll`'s sweep and every figure here
#: were measured on; widening it would move both.
#:
#: (An earlier version of this comment claimed the narrower band was "the one MolmoAct2's
#: calibration was fitted against". That was wrong, and measuring the checkpoint's own
#: action quantiles is what showed it.)
#:
#: The gripper entry is the jaw hinge, normalised: the MJCF hinge runs
#: -0.174533..1.745329 rad and the contract runs 0..1, an exact offset of +0.174533 that
#: `shared/ros_surfaces/so101.py` applies in both directions. Verified by sweeping the
#: jaw: tip separation is 7.01/20.91/38.42/55.80 mm at contract g = 0.00/0.24/0.48/0.72,
#: against the measured curve `gap(g) = -23.24g^2 + 92.54g - 0.84` mm the grasp tuning
#: was derived from, which predicts 20.86/38.29/55.79. Agreement to ~0.1 mm, so every
#: gripper constant in `waypoints` transfers unchanged.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan_joint": (-1.9198621771937616, 1.9198621771937634),
    "shoulder_lift_joint": (-1.7453292519943224, 1.7453292519943366),
    "elbow_flex_joint": (-1.69, 1.54),
    "wrist_flex_joint": (-1.658063, 1.658063),
    "wrist_roll_joint": (-2.7438473, 2.7438473),
    "gripper_joint": (0.0, 1.0),
}

#: (parent-relative translation, parent-relative quaternion wxyz) per moving link.
_LINKS: tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...] = (
    ((0.0388353, -8.97657e-09, 0.0624), (3.56167e-16, 1.22818e-15, -1.0, -4.14635e-16)),
    ((-0.0303992, -0.0182778, -0.0542), (0.5, -0.5, -0.5, -0.5)),
    (
        (-0.11257, -0.028, 1.73763e-16),
        (0.7071067811865475, -5.98613e-17, -2.58051e-17, 0.7071067811865475),
    ),
    (
        (-0.1349, 0.0052, 3.62355e-17),
        (0.7071067811865475, 9.58722e-16, -7.51313e-16, -0.7071067811865475),
    ),
    (
        (5.55112e-17, -0.0611, 0.0181),
        (0.017209108230571014, -0.017209108230571014, 0.7068973380865913, 0.7068973380865913),
    ),
)

#: The official ``gripperframe`` site inside the ``gripper`` body — the tool centre
#: point, at the fixed jaw's tip. It was menagerie's, 19.9 mm further along the body x,
#: until 2026-09-27; see the module docstring. ``tcp`` (the other site in this model)
#: is MolmoSpaces' own convention with +z as the approach axis, which is a
#: different frame again -- this module wants ``gripperframe``, because
#: `jaw_axis` reads the jaw-opening direction off the site's +z.
_TCP_POS = (-0.0079, -0.000218121, -0.0981274)
_TCP_QUAT = (0.7071067811865475, 0.0, 0.7071067811865475, 0.0)


def quat_to_mat(quat: tuple[float, float, float, float]) -> Vec:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _homogeneous(pos: tuple[float, float, float], quat: tuple[float, float, float, float]) -> Vec:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_mat(quat)
    transform[:3, 3] = pos
    return transform


_LINK_TRANSFORMS: tuple[Vec, ...] = tuple(_homogeneous(pos, quat) for pos, quat in _LINKS)
_TCP_TRANSFORM: Vec = _homogeneous(_TCP_POS, _TCP_QUAT)


def _rot_z(angle: float) -> Vec:
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(
        [[cos, -sin, 0.0, 0.0], [sin, cos, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def gripper_body_pose(joints: npt.ArrayLike) -> Vec:
    """Return the 4x4 base-frame pose of the ``gripper`` body for five arm angles."""
    angles = np.asarray(joints, dtype=np.float64).reshape(-1)
    if angles.size != len(ARM_JOINTS):
        raise ValueError(f"fk expects {len(ARM_JOINTS)} arm angles, got {angles.size}")
    transform = np.eye(4, dtype=np.float64)
    for link, angle in zip(_LINK_TRANSFORMS, angles, strict=True):
        transform = transform @ link @ _rot_z(float(angle))
    return transform


def fk(joints: npt.ArrayLike) -> Vec:
    """Return the 4x4 world pose of the tool centre point for five arm angles."""
    return gripper_body_pose(joints) @ _TCP_TRANSFORM


# ------------------------------------------------------------------ the two fingers
#
# The camera-verdict scorer establishes *release* by placing both fingers in the rig
# frame and measuring how far each is from the apple centre. The moving jaw is its own
# body, hinged on `gripper` inside the `gripper` body -- ``so101_new_calib.xml``'s
# ``moving_jaw_so101_v1`` pos/quat below -- so its pose needs the jaw angle as well.

#: `gripper_joint` on the wire is the official ``gripper`` hinge shifted so its closed
#: stop reads 0: hinge = wire - this. An exact offset, never a rescale (the simulator's
#: ``ros_surfaces/so101.py`` applies the same number in both directions).
GRIPPER_OFFSET_RAD = 0.174533

_MOVING_JAW_TRANSFORM: Vec = _homogeneous(
    (0.0202, 0.0188, -0.0234), (0.707107, 0.707107, -1.85362e-08, 1.85362e-08)
)

#: Each finger as a segment, root of its gripping pad to its tip, in its own body's
#: frame: the fixed finger in ``gripper``, the moving one in ``moving_jaw_so101_v1``. The
#: tips are the fingertip collision spheres' centres in the simulator's model; the roots
#: are where each pad starts. With these the closed jaw's tips sit 4 mm apart and open to
#: 40 mm at a wire position of 0.48 -- the measured aperture curve (38.4 mm) to within
#: the tip spheres' own size.
FIXED_FINGER: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (-0.0143, 0.0, -0.053),
    (-0.0081, 0.0, -0.101),
)
MOVING_FINGER: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (-0.0093, -0.045, 0.0189),
    (-0.012, -0.078, 0.0192),
)


def _apply(transform: Vec, point) -> Vec:
    return np.asarray(transform[:3, :3] @ np.asarray(point, dtype=np.float64)
                      + transform[:3, 3], dtype=np.float64)


def finger_segments(arm_joints: npt.ArrayLike, gripper_joint: float) -> tuple[tuple[Vec, Vec], ...]:
    """Both fingers as ``(root, tip)`` segments in the arm base frame.

    ``gripper_joint`` is the wire position, radians, 0 at the closed stop.
    """
    body = gripper_body_pose(arm_joints)
    jaw = body @ _MOVING_JAW_TRANSFORM @ _rot_z(float(gripper_joint) - GRIPPER_OFFSET_RAD)
    return (
        (_apply(body, FIXED_FINGER[0]), _apply(body, FIXED_FINGER[1])),
        (_apply(jaw, MOVING_FINGER[0]), _apply(jaw, MOVING_FINGER[1])),
    )


def point_segment_distance(point: npt.ArrayLike, a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Shortest distance from ``point`` to the segment ``a``-``b``."""
    p, a, b = (np.asarray(v, dtype=np.float64) for v in (point, a, b))
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom <= 0.0 else min(1.0, max(0.0, float((p - a) @ ab) / denom))
    return float(np.linalg.norm(p - (a + t * ab)))


def tcp_position(joints: npt.ArrayLike, *, offset: float = 0.0) -> Vec:
    """Return the world xyz of the tool centre point, or of the point ``offset``
    along `jaw_axis` from it."""
    pose = fk(joints)
    return np.asarray(pose[:3, 3] + offset * pose[:3, 2], dtype=np.float64)


def clamp_to_limits(joints: npt.ArrayLike, names: tuple[str, ...] = ARM_JOINTS) -> Vec:
    """Clamp each angle into its MJCF joint range."""
    angles = np.asarray(joints, dtype=np.float64).reshape(-1).copy()
    if angles.size != len(names):
        raise ValueError(f"expected {len(names)} angles for {names}, got {angles.size}")
    for index, name in enumerate(names):
        low, high = JOINT_LIMITS[name]
        angles[index] = min(max(float(angles[index]), low), high)
    return angles


def position_jacobian(joints: npt.ArrayLike, *, eps: float = 1e-6, offset: float = 0.0) -> Vec:
    """Return the 3x5 finite-difference Jacobian of TCP position wrt arm angles.

    ``offset`` is `tcp_position`'s: the point that far along the jaw axis instead.

    Finite differences rather than an analytic Jacobian: the chain is short, the
    IK below is a damped least-squares loop that tolerates the ~1e-9 error, and
    this cannot drift out of sync with ``fk``.
    """
    angles = np.asarray(joints, dtype=np.float64).reshape(-1)
    base = tcp_position(angles, offset=offset)
    jac = np.zeros((3, angles.size), dtype=np.float64)
    for index in range(angles.size):
        probe = angles.copy()
        probe[index] += eps
        jac[:, index] = (tcp_position(probe, offset=offset) - base) / eps
    return jac


def approach_pitch(pose: npt.ArrayLike) -> float:
    """Return the world elevation of a TCP pose's ``+x`` axis, in radians.

    ``+x`` is the direction the jaws point (the ``gripperframe`` site's own
    quaternion rotates it onto the link axis the gripper extends along), so
    ``-pi/2`` is a straight-down top grasp and ``0`` is a horizontal reach.
    """
    matrix = np.asarray(pose, dtype=np.float64)
    return float(np.arcsin(np.clip(matrix[2, 0], -1.0, 1.0)))


def jaw_axis(pose: npt.ArrayLike) -> Vec:
    """Return the world direction the jaws open along for a TCP pose.

    The ``gripperframe`` site's ``+z`` separates the fixed finger from the
    moving jaw, so a grasp of an object standing on a table wants this axis as
    close to horizontal as the wrist roll can make it — otherwise one finger is
    driven down into the table (or into the object) while the other passes over.
    """
    return np.asarray(pose, dtype=np.float64)[:3, 2]


#: How far menagerie's ``gripperframe`` site sat from the official one along the jaw
#: axis: 0.012 - (-0.0079) in the gripper body's x, which is the site's +z. (Its y and
#: z differed by 0.12 and 0.4 um of rounding, ignored here.) The preflight's reach gate
#: was calibrated as IK of that point and still solves for it, through
#: ``ik_position(..., offset=REACH_POINT_OFFSET)``, so its verdicts did not move when
#: the TCP did.
REACH_POINT_OFFSET = 0.0199

#: Distance from the ``gripperframe`` site to the centre of the gap the apple sits
#: in, along the jaw axis (the site's +z, which points from the fixed finger towards
#: the moving jaw). The site does not sit in the middle of the gap, so a pose that
#: puts the site on an object drives a finger into it instead of straddling it.
#:
#: **Measured functionally, not geometrically, and the distinction cost a miss.** The
#: sweep below was run against menagerie's site, `REACH_POINT_OFFSET` further along
#: this axis, so every offset in it is that much larger from the official site: the
#: 0.002 it found is 0.0219 here, which is also the reference arm's 0.0218 from what
#: is the same site to 0.3 mm. A first attempt instead measured the midpoint of the
#: two jaw-*tip* spheres at the open jaw (0.0174 then) -- a plausible proxy that is
#: 15 mm off, because the tips are not the contact faces. Sweeping the offset through
#: an offline pick (plan solved with each value, arm driven to `close`, apple dropped
#: at spawn, jaw closed to 0.50, arm lifted; offsets from menagerie's site):
#:
#:     offset (m)      0.000  0.002  0.004  0.006  0.008  0.010  0.0174
#:     jaw after close 0.500  0.500  0.500  0.515  0.542  0.570  0.670
#:     apple lifted    yes    yes    yes    no     no     no     no
#:
#: A jaw that stops above 0.50 is one pressing the apple against a finger rather than
#: closing on it. The working window is 4 mm wide and 0.002 is its centre, which is
#: also the transposed reference value -- so the two derivations agree and the tip
#: measurement was simply the wrong thing to measure.
JAW_CENTER_OFFSET = REACH_POINT_OFFSET + 0.002


def jaw_center(joints: npt.ArrayLike, *, offset: float = JAW_CENTER_OFFSET) -> Vec:
    """Return the world point midway between the jaws for a set of arm angles."""
    return tcp_position(joints, offset=offset)


def level_jaw_roll(
    joints: npt.ArrayLike,
    *,
    prefer: float | None = None,
    samples: int = 721,
    tilt_tolerance: float = 0.05,
) -> float:
    """Return the ``wrist_roll_joint`` angle that makes the jaw axis most level.

    ``wrist_roll_joint`` rotates the tool about its own approach axis, so it
    barely moves the TCP; sweeping it and picking the flattest jaw axis is both
    exact enough and cheaper than deriving the closed form.

    A level jaw axis has two solutions half a turn apart. ``prefer`` selects the
    one nearest a previous roll among all candidates within ``tilt_tolerance`` of
    the flattest, which is what stops the wrist flipping 180 degrees between
    consecutive waypoints and dropping whatever it is holding.
    """
    angles = np.asarray(joints, dtype=np.float64).reshape(-1).copy()
    low, high = JOINT_LIMITS["wrist_roll_joint"]
    candidates = np.linspace(low, high, samples)
    tilts = np.empty(samples, dtype=np.float64)
    for index, candidate in enumerate(candidates):
        angles[4] = candidate
        tilts[index] = abs(float(jaw_axis(fk(angles))[2]))
    flattest = float(tilts.min())
    acceptable = candidates[tilts <= flattest + tilt_tolerance]
    if prefer is None:
        return float(acceptable[int(np.argmin(tilts[tilts <= flattest + tilt_tolerance]))])
    return float(acceptable[int(np.argmin(np.abs(acceptable - float(prefer))))])


@dataclass(frozen=True)
class IKResult:
    """Outcome of one IK solve: the angles reached and the residual error."""

    joints: Vec
    position_error: float
    converged: bool


def _solve(
    goal: Vec,
    angles: Vec,
    *,
    pitch: float | None,
    pitch_weight: float,
    roll: float,
    tolerance: float,
    max_iterations: int,
    damping: float,
    offset: float,
) -> tuple[Vec, float, float]:
    """Run damped least squares from one seed, returning the best iterate seen.

    "Best" is lexicographic: any iterate that meets the position tolerance beats
    one that does not, and among those the smaller pitch error wins. Tracking
    the best iterate rather than the last matters because the pitch term can
    drag a converged position solution off the target late in the loop.
    """
    best = angles.copy()
    best_position = float("inf")
    best_pitch = float("inf")
    for _ in range(max_iterations):
        pose = fk(angles)
        position_error = goal - (pose[:3, 3] + offset * pose[:3, 2])
        position_norm = float(np.linalg.norm(position_error))
        pitch_error = 0.0 if pitch is None else pitch - approach_pitch(pose)
        if (position_norm <= tolerance, -abs(pitch_error), -position_norm) > (
            best_position <= tolerance,
            -best_pitch,
            -best_position,
        ):
            best, best_position, best_pitch = angles.copy(), position_norm, abs(pitch_error)
        if position_norm < tolerance and abs(pitch_error) < 1e-3:
            break

        rows = [position_jacobian(angles, offset=offset)]
        errors = [position_error]
        if pitch is not None:
            jac_pitch = np.zeros((1, len(ARM_JOINTS)), dtype=np.float64)
            current = approach_pitch(pose)
            for index in range(len(ARM_JOINTS)):
                probe = angles.copy()
                probe[index] += 1e-6
                jac_pitch[0, index] = (approach_pitch(fk(probe)) - current) / 1e-6
            rows.append(pitch_weight * jac_pitch)
            errors.append(np.asarray([pitch_weight * pitch_error]))
        jac = np.vstack(rows)
        residual = np.concatenate(errors)
        step = jac.T @ np.linalg.solve(jac @ jac.T + (damping**2) * np.eye(jac.shape[0]), residual)
        # Cap the per-iteration joint change so a large early residual cannot
        # throw the seed across the workspace and into a different IK branch.
        norm = float(np.linalg.norm(step))
        if norm > 0.2:
            step *= 0.2 / norm
        angles = clamp_to_limits(angles + step)
        angles[4] = roll
    return best, best_position, best_pitch


#: Extra seeds tried when the caller's seed lands in a poor IK branch. The arm
#: is 5-DoF with a mirrored pan axis, so elbow-up and elbow-down solutions are
#: far apart in joint space and a single seed regularly misses the good one.
_FALLBACK_SEEDS: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.3, -0.2, 0.9, 0.0),
    (0.0, -0.4, 0.6, 0.4, 0.0),
    (0.0, 0.9, -1.1, 1.4, 0.0),
    (0.0, 0.6, -0.6, 1.0, 0.0),
)


def ik_position(
    target: npt.ArrayLike,
    *,
    seed: npt.ArrayLike | None = None,
    pitch: float | None = None,
    pitch_weight: float = 0.35,
    roll: float = 0.0,
    tolerance: float = 1e-4,
    max_iterations: int = 400,
    damping: float = 0.05,
    offset: float = 0.0,
) -> IKResult:
    """Solve for arm angles putting the TCP at ``target``.

    ``offset`` solves for the point that far along `jaw_axis` from the TCP instead
    (`tcp_position`'s ``offset``); the preflight passes `REACH_POINT_OFFSET`.

    ``pitch`` optionally requests a tool approach angle in radians — see
    [`approach_pitch`][robot_console.arm.kinematics.approach_pitch]. It is a soft
    objective because the 5-DoF arm cannot always hit an exact position *and*
    orientation, and a reachable near-vertical grasp beats an unreachable exact
    one. ``roll`` pins ``wrist_roll_joint``, which barely moves the TCP and
    would otherwise wander.

    The caller's ``seed`` is tried first so consecutive waypoints stay on one IK
    branch; fixed fallback seeds are tried only if it fails to reach
    ``tolerance``, and the first branch that does wins.
    """
    goal = np.asarray(target, dtype=np.float64).reshape(3)
    seeds: list[Vec] = []
    if seed is not None:
        seeds.append(np.asarray(seed, dtype=np.float64).reshape(-1)[: len(ARM_JOINTS)].copy())
    seeds.extend(np.asarray(candidate, dtype=np.float64) for candidate in _FALLBACK_SEEDS)

    best: Vec | None = None
    best_position = float("inf")
    best_pitch = float("inf")
    for candidate in seeds:
        start = clamp_to_limits(candidate)
        start[4] = roll
        angles, position_error, pitch_error = _solve(
            goal,
            start,
            pitch=pitch,
            pitch_weight=pitch_weight,
            roll=roll,
            tolerance=tolerance,
            max_iterations=max_iterations,
            damping=damping,
            offset=offset,
        )
        if (position_error <= tolerance, -pitch_error, -position_error) > (
            best_position <= tolerance,
            -best_pitch,
            -best_position,
        ):
            best, best_position, best_pitch = angles, position_error, pitch_error
        if position_error <= tolerance and pitch_error < 5e-3:
            break

    assert best is not None  # seeds is never empty
    return IKResult(joints=best, position_error=best_position, converged=best_position <= tolerance)
