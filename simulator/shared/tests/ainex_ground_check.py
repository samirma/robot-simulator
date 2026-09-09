"""The ground-follow, on a plane: stands, crouches, walks off the edge and lands.

Standalone, in the style of `test_attach.py`, runnable under either engine's venv:

    molmospaces/.venv/bin/python shared/tests/ainex_ground_check.py
    robocasa/.venv/bin/python shared/tests/ainex_ground_check.py

Three claims `ros_surfaces/ainex/ground.py` makes, each measured against a 2 m worktop
box at z = 0.5 with a floor at z = 0: the soles sit on the surface it finds; folding the
legs lowers the torso with the sole still on it; and a robot teleported past the edge
falls and comes to rest on the floor at its ride height. The last one is what was missing
before the z axis existed -- a robot that walked off a worktop hung in the air where it
left, and no number said so.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

SHARED = Path(__file__).resolve().parents[1]
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import ainex_model  # noqa: E402
from mujoco_bridge import PlanarJointBase, PlanarSetpoint  # noqa: E402
from tasks import apple_on_plate  # noqa: E402
from ros_surfaces.ainex import gait, servos  # noqa: E402
from ros_surfaces.ainex.actions import BASE_PITCH, rest_pose  # noqa: E402
from ros_surfaces.ainex.ground import GroundFollow  # noqa: E402

NS = "robot_0/"
PLANE_Z = 0.5
CONTROL_DT = 0.1
#: Where the loose apple waits until a check wants it, and its radius -- the task's own.
APPLE_PARK = 0.75
APPLE_R = apple_on_plate.APPLE_RADIUS
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" - {detail}" if detail else ""))
    failures += 0 if ok else 1


def main() -> int:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[10, 10, 0.1], pos=[0, 0, 0])
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[1.0, 1.0, 0.05], pos=[0, 0, PLANE_Z - 0.05])
    # The task's own apple, parked out of the way of everything above and moved under a
    # sole by the two checks that want it. Free-jointed and carrying `APPLE_CONTACT`,
    # because both of the things being checked turn on exactly that: `is_loose` reads the
    # free joint, and a kick that has no rolling friction to work against is not a kick.
    apple = spec.worldbody.add_body(name="loose_apple",
                                    pos=[APPLE_PARK, 0.0, PLANE_Z + APPLE_R])
    apple.add_freejoint(name="loose_apple_joint")
    apple.add_geom(name="loose_apple_col", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=[APPLE_R, 0.0, 0.0], group=0, **apple_on_plate.APPLE_CONTACT)
    ride = ainex_model.ride_height(ainex_model.build_spec())
    spec.worldbody.add_frame(pos=[0, 0, PLANE_Z + ride]).attach_body(
        ainex_model.build_spec().body("body_link"), NS, ""
    )
    model = spec.compile()
    # Exactly what both engines do after the graft, and the checks below are the reason
    # it is here rather than in each engine's spawn alone.
    foot_bit = ainex_model.enable_foot_contacts(model, NS)
    data = mujoco.MjData(model)
    base = PlanarJointBase(model, data, NS, body="body_link")
    base.teleport(0.0, 0.0, 0.0)
    # The torso's lean is driven like any other joint here, because it is one: the rest
    # pose the surface holds includes it, and holding the servos without it stands the
    # robot upright at a ride height measured for a leaning one.
    act = {n: model.actuator(f"{NS}{n}").id for n in servos.SERVOS}
    act[BASE_PITCH] = model.actuator(f"{NS}{BASE_PITCH}_act").id
    rest = rest_pose()
    for n, a in rest.items():
        data.qpos[model.jnt_qposadr[model.joint(f"{NS}{n}").id]] = a
        data.ctrl[act[n]] = a
    mujoco.mj_forward(model, data)
    ground = GroundFollow(model, NS)
    steps = int(CONTROL_DT / model.opt.timestep)

    def tick(n: int):
        last = None
        for _ in range(n):
            last = ground.step(data, CONTROL_DT)
            for _ in range(steps):
                mujoco.mj_step(model, data)
        return last

    def sole() -> float:
        return float(ainex_model.sole_z(model, data, NS))

    def torso() -> float:
        return float(data.xpos[model.body(f"{NS}body_link").id][2])

    apple_adr = int(model.jnt_qposadr[model.joint("loose_apple_joint").id])
    apple_dof = int(model.jnt_dofadr[model.joint("loose_apple_joint").id])

    def put_apple(_model, _data, x: float, y: float) -> None:
        """Park the apple resting on the worktop at (x, y), dead still."""
        _data.qpos[apple_adr:apple_adr + 3] = [x, y, PLANE_Z + APPLE_R]
        _data.qpos[apple_adr + 3:apple_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        _data.qvel[apple_dof:apple_dof + 6] = 0.0
        mujoco.mj_forward(_model, _data)

    def apple_xy(_model, _data):
        return np.array(_data.qpos[apple_adr:apple_adr + 2], dtype=float)

    def apple_z(_model, _data) -> float:
        return float(_data.qpos[apple_adr + 2])

    setpoint = PlanarSetpoint()
    leg_geometry = ainex_model.leg_geometry(model, NS)
    phase = [0.0]

    def walk(ticks: int) -> None:
        """Drive the real gait forward, the way `ros_surfaces/ainex/surface.py` does."""
        param = gait.WalkingParam(x_amplitude=0.02).clamped()
        vx, vy, wz = gait.planar_velocity(param)
        setpoint.reset()
        held = dict(rest)
        for _ in range(ticks):
            phase[0] = (phase[0] + CONTROL_DT / param.period_time) % 1.0
            held.update(gait.leg_joint_targets(param, phase[0], leg_geometry))
            held.update(gait.arm_joint_targets(param, phase[0]))
            for n, a in held.items():
                data.ctrl[act[n]] = a
            pose = base.pose
            base.ctrl = setpoint.step(float(pose[0, 3]), float(pose[1, 3]),
                                      math.atan2(pose[1, 0], pose[0, 0]),
                                      vx, vy, wz, CONTROL_DT)
            tick(1)

    print("ground-follow:")
    s = tick(30)
    check("standing, the soles sit on the worktop", abs(sole() - PLANE_Z) < 1e-3 and s.supported,
          f"gap {(sole() - PLANE_Z) * 1000:+.2f} mm")

    # ...on the whole sole, not on a heel corner. The gap above is the lowest single
    # vertex and was a perfect 0.00 mm while the robot stood 14.95 degrees toe-up with
    # its toes 36 mm in the air -- the check and the placement shared a method, so they
    # agreed with each other and with nothing else. Tilt is read off MuJoCo's own frames.
    tilt = max(abs(math.degrees(math.atan2(rot[0, 2], rot[2, 2])))
               for rot in (data.xmat[model.body(f"{NS}{f}").id].reshape(3, 3)
                           for f in ainex_model.FEET))
    check("and they are flat on it, not standing on their heels", tilt < 1.0,
          f"worst sole tilt {tilt:.2f} deg")

    # ...and it stays there, which the single sample above cannot tell you. The probe used
    # to step 1 mm past its own foot geoms, which cleared the worktop's top face and
    # restarted the ray inside the box: the surface came back as the box's *underside*, the
    # robot read itself as 100 mm in the air, fell, and climbed back over the next four
    # ticks -- a permanent 69 mm bob with a period of exactly 5 control ticks. `tick(30)`
    # lands on the one phase of that cycle where the sole is right, so every check above
    # passed at "gap +0.02 mm" while the robot juddered. Sampling consecutive ticks is a
    # different method, not a longer settle.
    heights, unsupported = [], 0
    for _ in range(40):
        s = tick(1)
        heights.append(sole())
        unsupported += 0 if s.supported else 1
    span = (max(heights) - min(heights)) * 1000
    check("and it holds that height tick after tick, not on average",
          span < 0.5 and unsupported == 0,
          f"sole span {span:.4f} mm over 40 ticks, {unsupported} of them unsupported")

    low = gait.leg_joint_targets(gait.WalkingParam(body_height=0.06), 0.0,
                                 ainex_model.leg_geometry(model, NS))
    for n, a in low.items():
        data.ctrl[act[n]] = a
    t0 = torso()
    tick(30)
    check("folding the legs lowers the torso with the sole still on the worktop",
          t0 - torso() > 0.02 and abs(sole() - PLANE_Z) < 1e-3,
          f"torso -{(t0 - torso()) * 1000:.1f} mm, gap {(sole() - PLANE_Z) * 1000:+.2f} mm")
    for n, a in rest.items():
        data.ctrl[act[n]] = a
    tick(20)

    # An apple is not a floor. The probe used to accept any geom that was not the robot,
    # so the tick a sole's probe point crossed the task's 20 mm apple the surface came
    # back as its crown -- measured 0.5392 against the worktop's 0.5000 -- and the robot
    # was lifted 39 mm onto it while the apple did not move by a millimetre. The apple is
    # placed under the sole rather than walked into here, because the probe is a single
    # point per foot and the crossing is a graze: over one walk-past the closest a sole
    # came to it was 20.6 mm, so a check that walked at it would pass by luck.
    print("\nloose objects:")
    base.teleport(0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    tick(10)
    under = min(ground._feet.per_body(data).values(), key=lambda p: p[2])
    put_apple(model, data, under[0], under[1])
    crown = apple_z(model, data) + APPLE_R
    highest = -1e9
    for _ in range(10):
        tick(1)
        highest = max(highest, sole())
    check("an apple under a sole is not a surface to stand on",
          abs(highest - PLANE_Z) < 1e-3,
          f"highest sole z {highest:.4f} against the worktop's {PLANE_Z:.4f}, "
          f"with the apple's crown at {crown:.4f}")

    # ...and the feet do touch it, which is the other half of the same complaint: with
    # every geom but the hands at contype 0 the robot walked through the apple without
    # disturbing it. `enable_foot_contacts` gives the feet a class that meets loose bodies
    # and passes through the world, so this moves it and the standing checks above still
    # hold -- the feet still do not grip the worktop.
    start = 0.30
    put_apple(model, data, start, 0.0)
    base.teleport(start - 0.35, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    before = apple_xy(model, data)
    walk(45)
    moved = float(np.linalg.norm(apple_xy(model, data) - before)) * 1000
    check("and walking into one kicks it rather than passing through",
          moved > 20.0, f"apple moved {moved:.1f} mm (contype bit {foot_bit})")
    for n, a in rest.items():
        data.ctrl[act[n]] = a
    put_apple(model, data, APPLE_PARK, 0.0)
    tick(10)

    print()
    base.teleport(3.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    fell = landed = False
    for i in range(60):
        s = tick(1)
        fell = fell or s.falling
        if fell and s.supported:
            landed = True
            break
    tick(15)
    check("past the edge it falls", fell)
    check("and lands on the floor at its ride height",
          landed and abs(sole()) < 1e-3 and abs(torso() - ride) < 1e-3,
          f"sole z {sole():+.4f}, torso z {torso():.4f} (ride {ride:.4f})")

    print("all checks passed" if not failures else f"FAILED: {failures} check(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
