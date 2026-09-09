"""Author the AiNex's `crawl_left` / `crawl_right` action groups by measurement.

The real robot picks a block off the ground with a recorded servo trajectory the vendor
calls `crawl_left`/`crawl_right` (`visual_patrol_pick_up_node.py` runs `walk_ready` ->
`hand_back` -> `crawl_*` -> `place_block`). Those `.d6a` files are unlicensed vendor pose
data and are not shipped here; these are the simulator's stand-ins, and they are *solved*
rather than typed, because the geometry that decides whether a claw reaches the surface
was measured and is unforgiving: standing upright the claw tip is 0.379 m above the sole,
and with the legs folded flat and the torso level it is still 0.072 m up. What brings it
to the surface is the torso leaning forward -- which on the real robot the hip chain does
over planted feet, and which here is the `base_pitch` joint (`ainex_model`, step 2).

Kinematic sweep, in the robot's own frame: the ground-follow keeps the stance sole on the
surface at run time, so a pose's claw height *over its own lowest sole* is its claw
height over the worktop. The sweep folds the legs, leans the torso, and moves one arm
inside `servos.joint_limits`, scoring each pose by how close the TCP lands to a target
just above the surface in front of the feet, and rejecting anything that would put the
torso hull or a limb other than the hands into the surface.

    python shared/tools/author_ainex_crawl.py [--write]

Prints the measured poses; `--write` emits the two YAML files into `action_groups/`.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

SHARED = Path(__file__).resolve().parents[1]
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import ainex_model  # noqa: E402
from ros_surfaces.ainex import servos  # noqa: E402
from ros_surfaces.ainex.actions import ACTION_DIR, BASE_PITCH, rest_pose  # noqa: E402

#: Where the TCP should land, in the base frame with the surface at z = 0: ahead of the
#: feet, to the side of the grasping hand, one apple radius up. The apple is 20 mm.
TARGET = {"l": (0.06, 0.06, 0.020), "r": (0.06, -0.06, 0.020)}
#: Pre-grasp: the same, lifted, so the claw comes down on the object rather than into it.
PRE_LIFT = 0.04
#: How far a limb other than the hands may dip below the surface. Slightly negative: the
#: real robot crawls on its knees, the limb meshes do not collide, and measured, the only
#: poses that bring the claw to the surface have the knees and the torso's underside on
#: it. Insisting on 15 mm of air under every limb left the claw 66 mm up.
LIMB_CLEARANCE = -0.005
#: The torso hull is the one part of the body that collides with the world, so it must
#: stay above the surface -- but only just. It is `TORSO_CLEARANCE` above the torso's
#: lowest mesh point by construction.
HULL_CLEARANCE = 0.002

FEET = ("l_ank_roll_link", "r_ank_roll_link")
HANDS = ("l_gripper_link", "r_gripper_link")
#: The vendor's names for the two groups, as its pick-up demo runs them.
NAMES = {"l": "left", "r": "right"}


class Model:
    def __init__(self) -> None:
        self.spec = ainex_model.build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in servos.SERVOS}
        self.qadr[BASE_PITCH] = m.jnt_qposadr[m.joint("base_pitch").id]
        self.hull = m.geom("torso_hull").id
        # Mesh geoms by body, for the sole and the clearance checks.
        self.meshes: dict[str, list[tuple[int, np.ndarray]]] = {}
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[g]
            s, c = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            self.meshes.setdefault(m.body(m.geom_bodyid[g]).name, []).append(
                (g, np.array(m.mesh_vert[s:s + c]))
            )

    def pose(self, pose: dict[str, float]) -> None:
        for n, v in pose.items():
            self.data.qpos[self.qadr[n]] = v
        mujoco.mj_forward(self.model, self.data)

    def lowest(self, bodies) -> float:
        d = self.data
        return min(
            float((verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
            for b in bodies for g, verts in self.meshes.get(b, [])
        )

    def hull_low(self) -> float:
        d, m = self.data, self.model
        r = d.geom_xmat[self.hull].reshape(3, 3)
        return float(d.geom_xpos[self.hull][2] - np.abs(r[2]) @ m.geom_size[self.hull])

    def tcp(self, side: str) -> np.ndarray:
        return np.array(self.data.site_xpos[self.model.site(f"{side}_tcp").id])

    def claw_down(self, side: str) -> bool:
        d, m = self.data, self.model
        return float(d.geom_xpos[m.geom(f"{side}_claw_palm").id][2]) > float(
            d.geom_xpos[m.geom(f"{side}_claw_tip").id][2]
        )


def clamp(name: str, value: float) -> float:
    lo, hi = servos.joint_limits(name)
    return float(min(max(value, lo), hi))


def legs(hip: float, knee: float, ank: float) -> dict[str, float]:
    """Both legs folded the same amount; the right leg's signs are the vendor's mirror."""
    return {
        "l_hip_pitch": clamp("l_hip_pitch", hip), "r_hip_pitch": clamp("r_hip_pitch", -hip),
        "l_knee": clamp("l_knee", knee), "r_knee": clamp("r_knee", -knee),
        "l_ank_pitch": clamp("l_ank_pitch", ank), "r_ank_pitch": clamp("r_ank_pitch", -ank),
    }


def arm(side: str, sp: float, sr: float, ep: float, ey: float) -> dict[str, float]:
    return {
        f"{side}_sho_pitch": clamp(f"{side}_sho_pitch", sp),
        f"{side}_sho_roll": clamp(f"{side}_sho_roll", sr),
        f"{side}_el_pitch": clamp(f"{side}_el_pitch", ep),
        f"{side}_el_yaw": clamp(f"{side}_el_yaw", ey),
    }


#: How many body configurations (lean + leg fold) get the full arm sweep: the ones that
#: bring the grasping shoulder lowest over the sole, since that is what bounds the reach.
BODY_SHORTLIST = 24


def solve(mdl: Model, side: str, target_z: float, body_grid, arm_grid):
    """The pose whose TCP lands nearest (TARGET x, y, target_z) over the sole."""
    tx, ty, _ = TARGET[side]
    best = (np.inf, None, None)
    limbs = [b for b in mdl.meshes if b not in FEET and b not in HANDS and b != "body_link"]
    shoulder = mdl.model.body(f"{side}_sho_roll_link").id

    # Stage 1: the lean and the fold alone decide the hull, the legs and how low the
    # shoulder gets; rank them and spend the arm sweep only on the lowest few.
    ranked = []
    for pitch, hip, knee, ank in body_grid:
        base = {**rest_pose(), BASE_PITCH: pitch, **legs(hip, knee, ank)}
        mdl.pose(base)
        sole = mdl.lowest(FEET)
        if mdl.hull_low() - sole < HULL_CLEARANCE:
            continue
        if mdl.lowest([b for b in limbs if not b.startswith(side)]) - sole < LIMB_CLEARANCE:
            continue
        ranked.append((float(mdl.data.xpos[shoulder][2] - sole), base))
    ranked.sort(key=lambda r: r[0])

    for _, base in ranked[:BODY_SHORTLIST]:
        mdl.pose(base)
        sole = mdl.lowest(FEET)
        for sp, sr, ep, ey in arm_grid:
            pose = {**base, **arm(side, sp, sr, ep, ey)}
            mdl.pose(pose)
            tcp = mdl.tcp(side)
            h = float(tcp[2] - sole)
            err = float(np.hypot(np.hypot(tcp[0] - tx, tcp[1] - ty), (h - target_z) * 2.0))
            if err >= best[0] or not mdl.claw_down(side):
                continue
            if mdl.lowest([b for b in limbs if b.startswith(side) and "gripper" not in b]) \
                    - sole < LIMB_CLEARANCE:
                continue
            best = (err, pose, (float(tcp[0]), float(tcp[1]), h))
    return best


def grids(side: str):
    s = 1.0 if side == "l" else -1.0
    body = list(itertools.product(
        np.linspace(0.20, 0.90, 8),           # base_pitch, up to the model's limit
        np.linspace(0.2, 1.6, 8),             # hip pitch (left sign)
        np.linspace(0.6, 2.0, 8),             # knee
        np.linspace(-0.9, 0.9, 7),            # ankle pitch
    ))
    arms = list(itertools.product(
        s * np.linspace(-1.5, 1.5, 9),        # sho_pitch
        s * np.linspace(-0.6, 1.6, 9),        # sho_roll
        s * np.linspace(-1.6, 1.6, 9),        # el_pitch
        s * np.array([-1.926, -1.2, -0.6, 0.0]),  # el_yaw
    ))
    return body, arms


def refine(mdl: Model, side: str, target_z: float, pose: dict[str, float]):
    """A second, local sweep around a coarse solution."""
    lo, hi = ainex_model.BASE_PITCH_RANGE
    keys = [f"{side}_sho_pitch", f"{side}_sho_roll", f"{side}_el_pitch", f"{side}_el_yaw"]
    body = list(itertools.product(
        np.clip(pose[BASE_PITCH] + np.linspace(-0.06, 0.06, 5), lo, hi),
        pose["l_hip_pitch"] + np.linspace(-0.12, 0.12, 5),
        pose["l_knee"] + np.linspace(-0.12, 0.12, 5),
        pose["l_ank_pitch"] + np.linspace(-0.15, 0.15, 5),
    ))
    arms = list(itertools.product(
        pose[keys[0]] + np.linspace(-0.2, 0.2, 5),
        pose[keys[1]] + np.linspace(-0.15, 0.15, 5),
        pose[keys[2]] + np.linspace(-0.2, 0.2, 5),
        pose[keys[3]] + np.linspace(-0.3, 0.3, 5),
    ))
    return solve(mdl, side, target_z, body, arms)


def author(mdl: Model, side: str):
    body, arms = grids(side)
    err, grasp, at = solve(mdl, side, TARGET[side][2], body, arms)
    if grasp is None:
        raise SystemExit(f"{side}: no pose reaches the surface inside the limits")
    err, grasp, at = refine(mdl, side, TARGET[side][2], grasp)
    # Pre-grasp: the *same* body, with the arm re-solved over its full range for the
    # lifted target, so the claw comes down onto the object rather than sweeping into it.
    same_body = [(grasp[BASE_PITCH], grasp["l_hip_pitch"], grasp["l_knee"], grasp["l_ank_pitch"])]
    _, pre, pre_at = solve(mdl, side, TARGET[side][2] + PRE_LIFT, same_body, arms)
    return grasp, at, pre, pre_at


def emit(side: str, grasp, at, pre, pre_at) -> str:
    closed, opened = ainex_model.GRIPPER_ANGLES[side]
    hand = f"{side}_gripper"
    body_keys = ["l_hip_pitch", "r_hip_pitch", "l_knee", "r_knee", "l_ank_pitch", "r_ank_pitch"]
    arm_keys = [f"{side}_sho_pitch", f"{side}_sho_roll", f"{side}_el_pitch", f"{side}_el_yaw"]

    def servo_map(pose, keys):
        return "{" + ", ".join(f"{k}: {pose[k]:.3f}" for k in keys) + "}"

    stand_arm = {k: rest_pose()[k] for k in arm_keys}
    stand_legs = {k: rest_pose()[k] for k in body_keys}
    return f"""# Bend down and close the {'left' if side == 'l' else 'right'} claw on whatever is on the
# surface in front of the feet -- the vendor's `crawl_{NAMES[side]}`, which its pick-up demo runs
# between `hand_back` and `place_block`. Solved by shared/tools/author_ainex_crawl.py, not
# typed: measured in the robot's frame with the sole on the surface, the TCP arrives
# {at[2] * 1000:.0f} mm above the surface, {at[0] * 1000:.0f} mm ahead of the base and {at[1] * 1000:+.0f} mm
# to the side, with a torso lean of {grasp[BASE_PITCH]:.2f} rad. Standing upright it is
# 379 mm up and cannot get lower than 72 mm without the lean; see README.md.
#
# `base_pitch` is the lean -- the 25th channel beside the servos. The ground-follow keeps
# the stance sole on the surface while the legs fold under it, which is what lets the
# body come down.
#
# `reach` is where the TCP lands, in the base frame with the surface at z = 0: what an
# object has to be at for this group to close on it. shared/tests/ainex_grasp_check.py
# reads it to put the apple there.
reach: {{x: {at[0]:.3f}, y: {at[1]:.3f}, z: {at[2]:.3f}}}
frames:
  - duration: 0.5          # open the claw first, so it arrives ready
    servos: {{{hand}: {opened:.3f}}}
  - duration: 1.0          # lean and fold: the body comes down over the feet
    base_pitch: {grasp[BASE_PITCH]:.3f}
    servos: {servo_map(grasp, body_keys)}
  - duration: 0.7          # arm to just above the target
    servos: {servo_map(pre, arm_keys)}
  - duration: 0.6          # down onto it
    servos: {servo_map(grasp, arm_keys)}
  - duration: 0.5          # close
    servos: {{{hand}: {closed:.3f}}}
  - duration: 0.6          # lift, holding
    servos: {servo_map(pre, arm_keys)}
  - duration: 1.0          # stand back up, still holding
    base_pitch: 0.0
    servos: {servo_map({**stand_legs, **stand_arm}, body_keys + arm_keys)}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="write the YAML files")
    args = ap.parse_args()
    mdl = Model()
    for side in ("l", "r"):
        grasp, at, pre, pre_at = author(mdl, side)
        print(f"crawl_{NAMES[side]}: TCP {at[2] * 1000:.1f} mm over the sole, "
              f"{at[0] * 1000:.0f} mm ahead, {at[1] * 1000:+.0f} mm across; "
              f"pre-grasp {pre_at[2] * 1000:.1f} mm; lean {grasp[BASE_PITCH]:.2f} rad")
        text = emit(side, grasp, at, pre, pre_at)
        if args.write:
            path = ACTION_DIR / f"crawl_{NAMES[side]}.yaml"
            path.write_text(text)
            print(f"  wrote {path}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
