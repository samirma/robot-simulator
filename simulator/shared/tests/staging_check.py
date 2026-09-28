"""`apple_on_plate` is the fixed scene spec §2.3 describes, staged whole, and nothing else.

Run from an engine's venv, with the engine named:

    molmospaces/.venv/bin/python shared/tests/staging_check.py --engine molmospaces
    robocasa/.venv/bin/python    shared/tests/staging_check.py --engine robocasa

For each worktop robot alone and for all of them together, it builds the world
`kitchen.sh serve` builds -- the engine's default scene -- through the shared spawn tool,
without serving it, and checks:

* exactly the six task objects are staged -- a red apple, a white plate, a bowl, a mug, a
  banana and a lemon -- each at its constant pose (`apple_on_plate.OBJECT_POSES`, in the
  task robot's base frame), whichever worktop robots stand beside the task: another
  worktop robot is stood clear of all six rather than any being left out for it;
* the fixed overhead and side cameras are staged, and each frames every task object;
* nothing else is added: every body, light and camera of the compiled model is the
  scene's, a robot's, a task object or one of the rig's two cameras, and the scene's
  lighting is its own;
* the SO-101 is within reach of the apple and the plate -- a top grasp at each solves
  within the compiled model's joint ranges (`reach.top_grasp`) -- and no two task objects
  overlap;
* no loose scene object is left among them: every movable scene body in the working
  area, or where another worktop robot stands, has been moved out of view.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import placement_check as pc  # noqa: E402  (puts shared/ on the path)

import reach  # noqa: E402
import robots_spec  # noqa: E402
from mujoco_bridge import camera_framing  # noqa: E402
from tasks import apple_on_plate as task  # noqa: E402

check = pc.check

#: Position tolerance against the constants, metres: the model is compiled, not stepped.
POSE_TOLERANCE = 1e-6


def mesh_points(model, data, body: int) -> np.ndarray:
    """World xyz of every vertex of a task object's visual mesh: what the cameras see."""
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        out.append(model.mesh_vert[adr:adr + num] @ data.geom_xmat[g].reshape(3, 3).T
                   + data.geom_xpos[g])
    return np.vstack(out)


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two objects' footprints (convex hulls of their points) overlap on the
    worktop: some edge direction of one or the other separates them if they do not."""
    from scipy.spatial import ConvexHull

    hulls = [p[ConvexHull(p[:, :2]).vertices][:, :2] for p in (a, b)]
    for hull in hulls:
        edges = np.roll(hull, -1, axis=0) - hull
        for normal in np.stack([-edges[:, 1], edges[:, 0]], axis=1):
            pa, pb = hulls[0] @ normal, hulls[1] @ normal
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def names(model, kind, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, kind, i) or "" for i in range(count)]


def examine(label: str, world) -> None:
    print(f"{label}:")
    model, data, scene = world.model, world.data, world.scene
    check(f"{label}: the task is staged", world.task is not None)
    if world.task is None:
        return
    check(f"{label}: no placement or start-up problem", not world.problems,
          "; ".join(world.problems))
    prefixes = tuple(inst.mjcf for inst in world.instances)

    # ---- exactly the six and the rig, and nothing else ------------------------------
    bodies = names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    staged = sorted(b for b in bodies if b.startswith("task_"))
    want = sorted(f"task_{n}" for n in task.TASK_OBJECTS)
    check(f"{label}: exactly the six task objects are staged", staged == want,
          f"staged {staged}")
    kept = [b for b in bodies if not b.startswith(prefixes) and not b.startswith("task_")]
    check(f"{label}: no body is added beyond the robots and the six task objects",
          len(kept) == scene.model.nbody,
          f"{len(kept)} other bodies against the bare scene's {scene.model.nbody}")
    lights = [n for n in names(model, mujoco.mjtObj.mjOBJ_LIGHT, model.nlight)
              if not n.startswith(prefixes)]
    check(f"{label}: no light is added", len(lights) == scene.model.nlight,
          f"{len(lights)} against the bare scene's {scene.model.nlight}")
    cameras = sorted(n for n in names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
                     if not n.startswith(prefixes))
    bare = sorted(names(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, scene.model.ncam))
    rig = sorted(c[0] for c in task.SCENE_CAMERAS)
    check(f"{label}: the rig's overhead and side cameras are the only cameras added",
          cameras == sorted(bare + rig), f"{sorted(set(cameras) - set(bare))} added")
    headlight, bare_light = model.vis.headlight, scene.model.vis.headlight
    lit = (np.allclose(headlight.ambient, bare_light.ambient)
           and np.allclose(headlight.diffuse, bare_light.diffuse)
           and np.allclose(headlight.specular, bare_light.specular)
           and math.isclose(model.vis.map.shadowclip, scene.model.vis.map.shadowclip))
    check(f"{label}: the scene's lighting is its own", lit)

    # ---- where the constants say ----------------------------------------------------
    # The frame the task was staged in: the task robot's stand, at its mount height. Not
    # the arbiter's, which is the robot's root body once compiled -- for the AiNex a
    # leaning torso at its ride height.
    lead = next(i for i in world.instances if i.name == world.staging.task_robot)
    to_base = np.linalg.inv(task.base_frame([lead.xy[0], lead.xy[1], lead.mount_z], lead.yaw))
    rot, origin = to_base[:3, :3], to_base[:3, 3]
    yaws = {"apple": 0.0, "plate": 0.0, **{d[0]: d[2] for d in task.DISTRACTORS}}
    ids = {n: model.body(f"task_{n}").id for n in task.TASK_OBJECTS}
    for name in task.TASK_OBJECTS:
        local = rot @ data.xpos[ids[name]] + origin
        err = float(np.linalg.norm(local - np.asarray(task.OBJECT_POSES[name])))
        axis = rot @ data.xmat[ids[name]].reshape(3, 3)[:, 0]
        turn = math.atan2(axis[1], axis[0]) - yaws[name]
        turn = abs(math.atan2(math.sin(turn), math.cos(turn)))
        check(f"{label}: the {name} is at its constant pose",
              err <= POSE_TOLERANCE and turn <= 1e-6,
              f"{err * 1000:.3f} mm, {math.degrees(turn):.3f} deg off")

    # ---- reach, footprints, framing -------------------------------------------------
    arm = next((inst for inst in world.instances if inst.name == "so101"), None)
    if arm is not None:
        for name in ("apple", "plate"):
            g = reach.top_grasp(model, data, arm.mjcf, data.xpos[ids[name]])
            check(f"{label}: the {name} is within the SO-101's reach", g.solved,
                  f"top grasp {g.position_error * 1000:.1f} mm, {g.tilt_deg:.1f} deg off")
    shapes = {n: mesh_points(model, data, ids[n]) for n in task.TASK_OBJECTS}
    others = [inst for inst in world.instances
              if inst.on == "worktop" and inst.name != world.staging.task_robot]
    for name, points in shapes.items():
        spread = float(np.max(np.linalg.norm(points[:, :2] - data.xpos[ids[name]][:2], axis=1)))
        check(f"{label}: FOOTPRINT_RADIUS bounds the {name}",
              spread <= task.FOOTPRINT_RADIUS[name], f"{spread:.4f} m against "
                                                     f"{task.FOOTPRINT_RADIUS[name]:.3f}")
        for cam in rig:
            worst = max(max(abs(u), abs(v)) for u, v in camera_framing(model, data, cam, points))
            check(f"{label}: the {cam} camera frames the {name}", worst <= 1.0,
                  f"worst |normalised| {worst:.3f}")
        for inst in others:
            d = float(np.min(np.linalg.norm(points[:, :2] - inst.xy, axis=1)))
            check(f"{label}: {inst.name} stands clear of the {name}", d >= inst.radius,
                  f"{d:.3f} m from its centre, footprint {inst.radius:.3f}")
    for a, b in itertools.combinations(task.TASK_OBJECTS, 2):
        check(f"{label}: the {a} and the {b} do not overlap", not overlap(shapes[a], shapes[b]))

    # ---- no loose scene object among them -------------------------------------------
    among = []
    for b in range(1, model.nbody):
        if bodies[b].startswith(prefixes) or bodies[b].startswith("task_"):
            continue
        jnt = int(model.body_jntadr[b])
        if jnt < 0 or model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        local = rot @ data.xpos[b] + origin
        band = task.CLEAR_Z_BAND[0] < local[2] < task.CLEAR_Z_BAND[1]
        near = math.hypot(local[0], local[1]) <= task.CLEAR_RADIUS or any(
            float(np.linalg.norm(data.xpos[b][:2] - i.xy)) <= i.radius for i in others)
        if band and near:
            among.append(bodies[b])
    check(f"{label}: no loose scene object is left among the task objects", not among,
          ", ".join(among))


def run(engine_name: str) -> int:
    """Every worktop robot alone, then all of them together; 0 when every check passes."""
    engine = pc.load_engine(engine_name)
    flags = pc.scene_flags(engine_name)
    worktop = list(robots_spec.worktop_ids())
    fleets = [*worktop] + ([",".join(worktop)] if len(worktop) > 1 else [])
    for robots in fleets:
        world, _log = pc.build(engine, flags, robots)
        examine(f"{engine_name} {robots}", world)
    print("all checks passed" if not pc.failures else f"FAILED: {len(pc.failures)} check(s)")
    return 0 if not pc.failures else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=("molmospaces", "robocasa"))
    return run(ap.parse_args().engine)


if __name__ == "__main__":
    raise SystemExit(main())
