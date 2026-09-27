"""Whether the SO-101 can take a staged object: a top grasp, solved on the compiled model.

Spec §2.3: at start-up, with the SO-101 served, `serve` reports for each staged object
whether a top grasp at its position has an inverse-kinematics solution within the joint
ranges of the compiled model. A distance from the base against a reach annulus answered a
different question -- an object 0.32 m out dead ahead and one 0.32 m out at -45 degrees
are the same distance and not the same grasp -- so this solves the grasp itself.

**A top grasp** is the jaws pointing straight down at the object: the `gripperframe`
site's `+x` (the direction the jaws extend along, as the console's `approach_pitch`
reads it) along world `-z`, with the point midway between the jaws -- `JAW_CENTRE_OFFSET`
along the site's `+z`, the console's `JAW_CENTER_OFFSET` -- on the object's centre.
Solved by damped least squares over the five arm joints, clamped to the compiled model's
`jnt_range` at every iteration, from several seeds; a solution is one within
`POSITION_TOLERANCE` of the point and `TILT_TOLERANCE_DEG` of vertical.

Engine-neutral: a compiled `MjModel` and the robot's MJCF prefix are all it reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TCP_SITE = "gripperframe"
#: From `gripperframe` along its `+z` to the middle of the jaw gap: the console's
#: `kinematics.JAW_CENTER_OFFSET` (0.0199 + 0.002).
JAW_CENTRE_OFFSET = 0.0219
POSITION_TOLERANCE = 0.005
TILT_TOLERANCE_DEG = 10.0

#: Starting points: the rest pose, and elbow-up/elbow-down reaches the 5-DoF arm's
#: branches are far apart in joint space.
_SEEDS = (
    (0.0, 0.0, -1.5708, 1.0008, 0.0),
    (0.0, 0.3, -0.2, 0.9, 0.0),
    (0.0, -0.4, 0.6, 0.4, 0.0),
    (0.0, 0.9, -1.1, 1.4, 0.0),
    (0.0, 0.6, -0.6, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.2, 0.0),
)


@dataclass(frozen=True)
class Grasp:
    """One solve: whether it is a solution, and how close the best iterate came."""

    solved: bool
    position_error: float
    tilt_deg: float
    joints: tuple[float, ...]


def top_grasp(model, data, prefix: str, target) -> Grasp:
    """Solve a top grasp at world point `target` for the arm under MJCF `prefix`."""
    joints = [model.joint(f"{prefix}{j}").id for j in ARM_JOINTS]
    qadr = np.array([model.jnt_qposadr[j] for j in joints])
    dadr = np.array([model.jnt_dofadr[j] for j in joints])
    lo = np.array([model.jnt_range[j][0] for j in joints])
    hi = np.array([model.jnt_range[j][1] for j in joints])
    site = model.site(f"{prefix}{TCP_SITE}").id
    body = int(model.site_bodyid[site])
    goal = np.asarray(target, dtype=float).reshape(3)
    down = np.array([0.0, 0.0, -1.0])

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.mocap_pos[:] = data.mocap_pos
    scratch.mocap_quat[:] = data.mocap_quat
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    def evaluate(q):
        scratch.qpos[qadr] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        rot = scratch.site_xmat[site].reshape(3, 3)
        point = scratch.site_xpos[site] + JAW_CENTRE_OFFSET * rot[:, 2]
        return point, rot[:, 0]

    best = None
    for seed in _SEEDS:
        q = np.clip(np.asarray(seed, dtype=float), lo, hi)
        for _ in range(300):
            point, approach = evaluate(q)
            e_pos = goal - point
            e_rot = np.cross(approach, down)
            tilt = math.degrees(math.acos(float(np.clip(approach @ down, -1.0, 1.0))))
            err = float(np.linalg.norm(e_pos))
            score = (err <= POSITION_TOLERANCE and tilt <= TILT_TOLERANCE_DEG, -err - tilt / 1000)
            if best is None or score > best[0]:
                best = (score, Grasp(score[0], err, tilt, tuple(float(v) for v in q)))
            if err < 5e-4 and tilt < 0.5:
                break
            mujoco.mj_jac(model, scratch, jacp, jacr, point, body)
            jac = np.vstack([jacp[:, dadr], 0.3 * jacr[:, dadr]])
            residual = np.concatenate([e_pos, 0.3 * e_rot])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.03 ** 2 * np.eye(6), residual)
            norm = float(np.linalg.norm(step))
            if norm > 0.2:
                step *= 0.2 / norm
            q = np.clip(q + step, lo, hi)
        if best[0][0]:
            break
    return best[1]


def report(model, data, prefix: str, objects: dict[str, object]) -> list[str]:
    """One line per staged object: does a top grasp at its position solve?"""
    lines = []
    for name, target in objects.items():
        g = top_grasp(model, data, prefix, target)
        verdict = "in reach" if g.solved else "OUT of reach"
        lines.append(
            f"{name}: {verdict} -- top grasp IK "
            f"{'solved' if g.solved else 'has no solution'} within the compiled joint "
            f"ranges (best: {g.position_error * 1000:.1f} mm off, "
            f"{g.tilt_deg:.1f} deg from vertical)")
    return lines
