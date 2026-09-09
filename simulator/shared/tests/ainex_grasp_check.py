"""Does `crawl_left` / `crawl_right` actually touch the task's apple?

A standalone check in the style of `test_attach.py`, runnable under either engine's venv:

    molmospaces/.venv/bin/python shared/tests/ainex_grasp_check.py
    robocasa/.venv/bin/python shared/tests/ainex_grasp_check.py

It builds the smallest scene that can answer the question -- a floor as the worktop, the
task's own apple and plate staged by `apple_on_plate.stage`, and the AiNex grafted at
its ride height -- teleports the base so the apple sits where the group's claw is solved
to land, replays the group through `ActionPlayer` with `GroundFollow` running each tick
exactly as the ROS surface does, and asserts two things the wire cannot show: that a
geom on the grasping hand made contact with the apple, and that the apple moved.

"Interact with the objects in the scene" is this, measured; a claw that closes on air
reads its commanded width either way.
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
from mujoco_bridge import PlanarJointBase  # noqa: E402
from ros_surfaces.ainex import servos  # noqa: E402
from ros_surfaces.ainex.actions import (  # noqa: E402
    ACTION_DIR, BASE_PITCH, ActionPlayer, load_action_dir, rest_pose,
)
from ros_surfaces.ainex.ground import GroundFollow  # noqa: E402
from tasks import apple_on_plate  # noqa: E402


def reach_of(side: str) -> tuple[float, float, float]:
    """Where the group's TCP lands, as the authoring script measured and recorded it."""
    import yaml

    doc = yaml.safe_load((ACTION_DIR / f"crawl_{NAMES[side]}.yaml").read_text())
    r = doc["reach"]
    return float(r["x"]), float(r["y"]), float(r["z"])

NS = "robot_0/"
CONTROL_DT = 0.1
MOVED_M = 0.005
#: The vendor's names for the two groups.
NAMES = {"l": "left", "r": "right"}

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" - {detail}" if detail else ""))
    failures += 0 if ok else 1


def build():
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5, 5, 0.1], pos=[0, 0, 0])
    # The task's objects at their contract positions in a base frame at the origin.
    apple_on_plate.stage(spec, [0.0, 0.0, 0.0], 0.0, reference_table=False,
                         dressing=False)
    robot = ainex_model.build_spec()
    ride = ainex_model.ride_height(ainex_model.build_spec())
    spec.worldbody.add_frame(pos=[0, 0, ride]).attach_body(robot.body("body_link"), NS, "")
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def run(side: str) -> None:
    model, data = build()
    base = PlanarJointBase(model, data, NS, body="body_link")
    apple = model.body(apple_on_plate.APPLE_BODY).id
    apple_pos = np.array(data.xpos[apple]) if data.xpos[apple].any() else None
    mujoco.mj_forward(model, data)
    apple_pos = np.array(data.xpos[apple])
    # Stand the robot so the apple sits where the group's claw lands.
    tx, ty, _ = reach_of(side)
    base.teleport(float(apple_pos[0] - tx), float(apple_pos[1] - ty), 0.0)
    act = {n: model.actuator(f"{NS}{n}").id for n in servos.SERVOS}
    act[BASE_PITCH] = model.actuator(f"{NS}base_pitch_act").id
    for n, a in rest_pose().items():
        if n != BASE_PITCH:
            data.qpos[model.jnt_qposadr[model.joint(f"{NS}{n}").id]] = a
        data.ctrl[act[n]] = a
    mujoco.mj_forward(model, data)
    ground = GroundFollow(model, NS)
    steps = int(CONTROL_DT / model.opt.timestep)
    for _ in range(20):
        ground.step(data, CONTROL_DT)
        for _ in range(steps):
            mujoco.mj_step(model, data)
    start = np.array(data.xpos[apple])

    hand = model.body(f"{NS}{side}_gripper_link").id
    hand_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == hand}
    apple_bodies = {
        b for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("task_apple")
    }

    player = ActionPlayer(load_action_dir()[f"crawl_{NAMES[side]}"], rest_pose())
    touched_at = None
    t = 0.0
    lowest_tcp = np.inf
    while not player.finished:
        pose = player.step(CONTROL_DT)
        for n, v in pose.items():
            data.ctrl[act[n]] = v
        ground.step(data, CONTROL_DT)
        for _ in range(steps):
            mujoco.mj_step(model, data)
            for i in range(data.ncon):
                c = data.contact[i]
                bodies = {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
                geoms = {int(c.geom1), int(c.geom2)}
                if (geoms & hand_geoms) and (bodies & apple_bodies) and touched_at is None:
                    touched_at = t
        t += CONTROL_DT
        tcp = float(data.site_xpos[model.site(f"{NS}{side}_tcp").id][2])
        lowest_tcp = min(lowest_tcp, tcp)
    moved = float(np.linalg.norm(np.array(data.xpos[apple])[:2] - start[:2]))
    check(f"crawl_{NAMES[side]}: claw reaches the surface",
          lowest_tcp < 0.045, f"lowest TCP {lowest_tcp * 1000:.0f} mm over the floor")
    check(f"crawl_{NAMES[side]}: {side}_gripper_link touches the apple", touched_at is not None,
          f"first contact at t={touched_at:.1f} s" if touched_at is not None else "never")
    check(f"crawl_{NAMES[side]}: the apple moves", moved > MOVED_M, f"{moved * 1000:.1f} mm")


def main() -> int:
    print("grasp:")
    for side in ("l", "r"):
        run(side)
    print("all checks passed" if not failures else f"FAILED: {failures} check(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
