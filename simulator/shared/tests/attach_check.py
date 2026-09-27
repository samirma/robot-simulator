"""Each simulated robot's model attaches to the engine's scene, compiles and moves (spec §5).

The engine-parametrized counterpart of MolmoSpaces' `robots/<id>/test_attach.py`, run for
each engine from its own venv:

    molmospaces/.venv/bin/python shared/tests/attach_check.py --engine molmospaces
    robocasa/.venv/bin/python    shared/tests/attach_check.py --engine robocasa

Each robot alone in the engine's default scene, through the shared spawn tool, grafted by
the engine's own adapter (`Engine.attach`), compiled, bound and stepped:

* the SO-101's arm tracks a joint target and its jaw closes and opens;
* the myAGV's planar base drives to a commanded pose;
* the AiNex stands on its surface and its head servo tracks a target;

and nothing goes NaN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import placement_check as pc  # noqa: E402  (puts shared/ on the path)

import ainex_model  # noqa: E402
import robots_spec  # noqa: E402

check = pc.check


def settle(model, data, seconds: float) -> None:
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def so101(world, inst) -> None:
    model, data = world.model, world.data
    arm, gripper = inst.view["arm"], inst.view["gripper"]
    target = np.array([0.6, -0.4, 0.8, 0.3, 0.2])
    arm.ctrl = target
    settle(model, data, 3.0)
    err = float(np.abs(arm.joint_pos - target).max())
    check(f"{world_label}: so101 arm tracks a joint target", err < 0.05, f"max err {err:.4f} rad")
    gripper.ctrl = [-0.1]
    settle(model, data, 1.0)
    closed = float(gripper.joint_pos[0])
    gripper.ctrl = [1.2]
    settle(model, data, 1.0)
    opened = float(gripper.joint_pos[0])
    check(f"{world_label}: so101 jaw closes and opens", closed < 0.1 and opened > 1.0,
          f"closed {closed:.3f}, open {opened:.3f} rad")


def myagv(world, inst) -> None:
    model, data = world.model, world.data
    x, y, yaw = inst.base.xytheta
    # Towards the most open side, read off the bare scene: a house's open floor is often
    # a metre across, and "drove into the island and stopped" is correct behaviour.
    geomid = np.zeros(1, dtype=np.int32)
    heading = max(np.linspace(0.0, 2 * np.pi, 16, endpoint=False), key=lambda a: mujoco.mj_ray(
        world.scene.model, world.scene.data, np.array([x, y, 0.08]),
        np.array([np.cos(a), np.sin(a), 0.0]), None, 1, -1, geomid) % 1e6)
    goal = (x + 0.2 * np.cos(heading), y + 0.2 * np.sin(heading), yaw + 0.3)
    inst.base.ctrl = goal
    settle(model, data, 3.0)
    err = np.abs(inst.base.xytheta - np.array(goal))
    check(f"{world_label}: myagv base drives to a commanded pose",
          err[:2].max() < 0.02 and err[2] < 0.05, f"err {np.round(err, 4)}")


def ainex(world, inst) -> None:
    model, data = world.model, world.data
    gap = ainex_model.sole_z(model, data, inst.mjcf) - inst.surface_z
    check(f"{world_label}: ainex soles on its surface", abs(gap) <= ainex_model.SOLE_TOLERANCE,
          f"gap {gap * 1000:+.2f} mm")
    act = model.actuator(f"{inst.mjcf}head_pan").id
    joint = model.joint(f"{inst.mjcf}head_pan").id
    data.ctrl[act] = 0.5
    settle(model, data, 2.0)
    pan = float(data.qpos[model.jnt_qposadr[joint]])
    check(f"{world_label}: ainex head servo tracks a target", abs(pan - 0.5) < 0.05,
          f"head_pan {pan:.3f} rad")


world_label = ""


def main() -> int:
    global world_label
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=("molmospaces", "robocasa"))
    ap.add_argument("--robot", default=None, help="one id (default: every simulated robot)")
    args = ap.parse_args()
    engine = pc.load_engine(args.engine)
    flags = pc.scene_flags(args.engine)
    for robot in [args.robot] if args.robot else robots_spec.simulated_ids():
        world_label = f"{args.engine} {robot}"
        world, _log = pc.build(engine, flags, robot)
        print(f"{world_label}:")
        check(f"{world_label}: attaches and compiles",
              world.model.nu > 0 and not world.problems, "; ".join(world.problems))
        inst = world.instances[0]
        {"so101": so101, "myagv": myagv, "ainex": ainex}[robot](world, inst)
        check(f"{world_label}: the state stays finite",
              bool(np.isfinite(world.data.qpos).all() and np.isfinite(world.data.qvel).all()))
    print("all checks passed" if not pc.failures else f"FAILED: {len(pc.failures)} check(s)")
    return 0 if not pc.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
