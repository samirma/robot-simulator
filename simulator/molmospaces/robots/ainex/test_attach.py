#!/usr/bin/env python
"""Standalone check that the AiNex attaches, stands, drives and grasps.

Runs standalone, so a failure points at the robot definition.

    python robots/ainex/test_attach.py [--scene /path/to/house.xml] [--render out.png]

Most of what is checked here fails *silently* if it regresses -- a camera that stops
tracking the head still renders, a servo table that fails to apply still compiles, a gait
whose stance foot skates still walks. Those are the checks worth having.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ainex_model  # noqa: E402  (shared/, on the path via env.sh)
from ros_surfaces.ainex.actions import BASE_PITCH, rest_pose  # noqa: E402

FAIL = []

# Hiwonder publish 415 mm for the assembled robot. Ours measures a little more because
# the vendor figure is presumably taken in a different pose; a wide band still catches a
# model that is mis-scaled or standing on the wrong part of itself.
HEIGHT_RANGE = (0.38, 0.50)
# The band a standing robot's hands sweep through, sampled through every frame of every
# group that does not bend down. Measured: `greet`'s swing passes 0.183 m on its way, so
# the floor of the band is below the 0.25 m the end poses sit at. The crawl groups are
# the exception and are checked separately -- they exist to leave this band.
REACH_RANGE = (0.15, 0.45)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(label)


def drive_to(model, data, view, target, steps: int = 4000) -> np.ndarray:
    base = view.get_move_group("base")
    base.ctrl = np.asarray(target, dtype=np.float64)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return np.asarray(base.joint_pos, dtype=np.float64).copy()


def hold(model, data, pose: dict[str, float], ns: str, steps: int = 400) -> None:
    """Command a whole-body joint pose and let it settle.

    The 24 servo actuators are named after their joints; the torso's lean is the one
    channel whose actuator is not (`base_pitch_act`), because it is a base axis and
    those all carry the `_act` suffix.
    """
    for name, value in pose.items():
        actuator = "base_pitch_act" if name == BASE_PITCH else name
        data.ctrl[model.actuator(f"{ns}{actuator}").id] = value
    for _ in range(steps):
        mujoco.mj_step(model, data)


def mesh_world_z(model, data, body_names) -> list[float]:
    """World z of every mesh vertex on the named bodies.

    Vertices rather than `geom_rbound`, which is a bounding-sphere radius and sits several
    centimetres below a foot's sole -- enough to make a robot standing correctly look like
    it is hovering.

    Through each GEOM's own world frame (`geom_xpos`/`geom_xmat`), not the body's. This
    used to add `geom_pos` and apply the body frame, which drops `geom_quat` -- and every
    mesh geom on this robot carries one, because MuJoCo folds a mesh's principal-axes
    re-orientation into it at compile. That was the same shortcut `ainex_model._mesh_points`
    took, so this check agreed with the builder to the millimetre while both were 42.7 mm
    wrong and the robot stood that far off every surface. A witness that shares the
    subject's method is not a witness; MuJoCo's own frames are the independent one.
    """
    out: list[float] = []
    for name in body_names:
        bid = model.body(name).id
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_dataid[g] < 0:
                continue
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            pts = model.mesh_vert[start : start + count]
            world = pts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            out.extend(world[:, 2].tolist())
    return out


def lowest_point(model, data, body_names) -> float:
    return min(mesh_world_z(model, data, body_names))


def meshed_bodies(model, ns: str) -> list[str]:
    """The robot's own meshed bodies. Scoped by prefix: in a house the scene contributes
    thousands of geoms, and a check that swept them all would measure the building."""
    return [
        name
        for i in range(1, model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)) and name.startswith(ns)
        and any(model.geom_bodyid[g] == i and model.geom_dataid[g] >= 0
                for g in range(model.ngeom))
    ]


def robot_geoms(model, ns: str) -> list[int]:
    return [
        g
        for g in range(model.ngeom)
        if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]))
        and n.startswith(ns)
    ]


def main() -> int:  # noqa: PLR0915 -- a checklist reads better in one piece
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None, help="house MJCF to attach into")
    ap.add_argument("--render", default=None)
    args = ap.parse_args()

    from robots.ainex import AiNexRobot, AiNexRobotConfig, AiNexRobotView
    from ros_surfaces.ainex import gait, servos
    from ros_surfaces.ainex.actions import ActionPlayer, load_action_dir
    from robots.ainex.ainex import GRIPPER_ANGLES

    config = AiNexRobotConfig()
    ns = config.robot_namespace

    # ---------------------------------------------------------------- pure modules
    print("servo table:")
    ids = sorted(v[0] for v in servos.SERVOS.values())
    check("24 servos, ids 1..24 exactly once", ids == list(range(1, 25)), f"{len(ids)} ids")
    check(
        "count<->radian round-trips",
        all(
            abs(servos.count_to_angle(n, servos.angle_to_count(n, 0.3)) - 0.3) < 5e-3
            for n in servos.SERVOS
        ),
    )
    check(
        "zero radians maps to each joint's own init count",
        all(servos.angle_to_count(n, 0.0) == servos.SERVOS[n][1] for n in servos.SERVOS),
    )
    # The four joints whose `init` is not 500 are the only externally visible sign that
    # the vendor servo table was applied at all. If servos.py silently stopped being used,
    # every joint would keep the URDF's uniform placeholder and nothing else would notice.
    limit = servos.URDF_PLACEHOLDER_LIMIT
    asymmetric = sorted(n for n in servos.SERVOS if servos.joint_limits(n) != (-limit, limit))
    check(
        "exactly l/r knee and l/r sho_pitch are asymmetric",
        asymmetric == ["l_knee", "l_sho_pitch", "r_knee", "r_sho_pitch"],
        str(asymmetric),
    )
    check(
        "the two flipped servos are the sho_pitch pair",
        sorted(n for n in servos.SERVOS if servos.SERVOS[n][2]) == ["l_sho_pitch", "r_sho_pitch"],
    )
    check(
        "init pose is inside every limit",
        all(
            servos.joint_limits(n)[0] <= v <= servos.joint_limits(n)[1]
            for n, v in servos.INIT_POSE.items()
        ),
    )

    print("\ngait:")
    speed = gait.planar_velocity(gait.WalkingParam(x_amplitude=0.02, period_time=0.400))[0]
    check(
        "envelope-max speed matches the published 0.21 m/s",
        abs(speed - 0.21) / 0.21 < 0.10,
        f"{speed:.3f} m/s",
    )
    check(
        "app speed 4 is the fastest tier",
        gait.from_app_params(4, 0.025, 1, 0, 0).period_time
        < gait.from_app_params(1, 0.025, 1, 0, 0).period_time,
    )
    actions = load_action_dir()

    def _limits(joint: str) -> tuple[float, float]:
        # The torso's lean is the one channel that is not a servo; its range is the
        # model's (ainex_model.BASE_PITCH_RANGE), not the servo table's.
        return ainex_model.BASE_PITCH_RANGE if joint == BASE_PITCH else servos.joint_limits(joint)

    out_of_range = [
        (name, j)
        for name, frames in actions.items()
        for f in frames
        for j, v in f.angles.items()
        if not _limits(j)[0] - 1e-9 <= v <= _limits(j)[1] + 1e-9
    ]
    check(f"{len(actions)} action groups load, all frames in range", not out_of_range,
          str(out_of_range[:3]))

    # ---------------------------------------------------------------- the model
    spec = mujoco.MjSpec.from_file(args.scene) if args.scene else mujoco.MjSpec()
    if args.scene is None:
        spec.worldbody.add_light(
            pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.55, 0.56, 0.58, 1]
        )

    AiNexRobot.add_robot_to_scene(
        config, spec, prefix=ns, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"\ncompiled: {model.nbody} bodies, {model.ngeom} geoms, {model.nu} actuators")

    view = AiNexRobotView(data, ns)
    base = view.get_move_group("base")

    print("\nstructure:")
    check(
        "move groups",
        sorted(view.move_group_ids())
        == ["base", "head", "left_arm", "left_gripper", "legs", "right_arm", "right_gripper"],
        str(sorted(view.move_group_ids())),
    )
    check("3 base actuators", base.n_actuators == 3, f"{base.n_actuators}")
    check("29 actuators total (24 joints + 5 base)", model.nu == 29, f"{model.nu}")
    # The torso's joints in order: x, y, theta are the move group; z is the ground's
    # and pitch is the lean. The order is load-bearing -- a hinge rotates the axes of
    # every joint after it, so pitch before z would tilt the axis the ground-follow
    # drives (ainex_model, step 2).
    torso = model.body(f"{ns}body_link")
    torso_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(torso.jntadr[0]) + i)
        for i in range(int(torso.jntnum[0]))
    ]
    check("torso joints in order x, y, theta, z, pitch",
          torso_joints == [f"{ns}base_{a}" for a in ("x", "y", "theta", "z", "pitch")],
          str(torso_joints))
    # MuJoCo merges a jointless URDF root into the worldbody, and here it would merge two
    # -- leaving five disconnected root bodies and 0.743 kg outside the tree. The virtual
    # joints are what stop that; see PROVENANCE.md.
    roots = [
        name
        for i in range(model.nbody)
        if model.body_parentid[i] == 0 and i != 0
        and (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
        and name.startswith(ns)
    ]
    check("exactly one root body", len(roots) == 1, str(roots))
    check(
        "total mass matches the vendor's 2.347 kg",
        abs(float(model.body_subtreemass[model.body(f"{ns}body_link").id]) - 2.3475) < 0.01,
        f"{float(model.body_subtreemass[model.body(f'{ns}body_link').id]):.4f} kg",
    )
    # discardvisual defaults on for URDF, so making the robot non-colliding deletes its
    # meshes unless ainex.py turns it off. The robot would drive fine and render as
    # nothing at all.
    mesh_geoms = sum(1 for g in robot_geoms(model, ns) if model.geom_dataid[g] >= 0)
    check("all 25 link meshes survived the collision surgery", mesh_geoms >= 25,
          f"{mesh_geoms} mesh geoms")
    # Only the robot's own geoms; the scene's floor and furniture collide too, obviously.
    # Three classes, and the split is the point: the torso hull and the hands meet the
    # world (contype 1), the two feet meet only what `enable_foot_contacts` tags -- they
    # carry a conaffinity and no contype, so an untagged scene leaves them inert -- and
    # everything else is decorative. The feet must be collidable *here*, at compile: MuJoCo
    # builds a mesh's convex hull only for meshes some collidable geom uses, so a foot
    # compiled at 0/0 can never collide however the masks are set afterwards.
    def body_of(g):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))

    feet = {f"{ns}{f}" for f in ainex_model.FEET}
    world_colliding = [g for g in robot_geoms(model, ns)
                       if model.geom_contype[g] and body_of(g) not in feet]
    foot_colliding = [g for g in robot_geoms(model, ns)
                      if body_of(g) in feet and model.geom_conaffinity[g]]
    inert = [g for g in robot_geoms(model, ns)
             if not (model.geom_contype[g] or model.geom_conaffinity[g])]
    check("only the torso hull and the two hands meet the world",
          len(world_colliding) == 3,
          f"{len(world_colliding)}: "
          f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in world_colliding]}")
    check("and the two feet meet only what is tagged for them",
          len(foot_colliding) == 2
          and all(model.geom_contype[g] == 0 for g in foot_colliding),
          f"{len(foot_colliding)} foot geoms, conaffinity "
          f"{sorted({int(model.geom_conaffinity[g]) for g in foot_colliding})}, contype "
          f"{sorted({int(model.geom_contype[g]) for g in foot_colliding})}")
    check("and everything else is decorative", len(inert) >= 25, f"{len(inert)} inert geoms")
    check("world site present",
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{ns}world") >= 0)

    # The house origin is usually inside furniture -- in FloorPlan1 it is inside the
    # kitchen island -- and a robot embedded in geometry cannot stand, hold a pose or turn
    # its head, all of which read as broken actuators. Move to open floor before measuring
    # anything physical. Same reason robots/myagv/test_attach.py does it.
    origin = np.zeros(2)
    heading = np.array([1.0, 0.0])
    reach, strafe_reach = 1.0, 0.5
    if args.scene:
        from tools.spawn_robot import find_open_spot

        origin, yaw = find_open_spot(args.scene)
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        reach, strafe_reach = 0.5, 0.25
        start = np.eye(4)
        start[:2, 3] = origin
        base.pose = start
        base.ctrl = np.array([origin[0], origin[1], 0.0])
        mujoco.mj_forward(model, data)
        print(f"  placed on open floor at {np.round(origin, 3)}")

    # ---------------------------------------------------------------- the camera
    # The vendor URDF bolts the camera to the torso. Panning the head would then not move
    # the view -- which a render cannot show, so only this check catches a regression.
    print("\ncamera (reparented from the torso to the head):")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{ns}front_camera")
    check("front_camera present", cam_id >= 0)
    parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[cam_id])
    check("camera rides head_tilt_link", parent == f"{ns}head_tilt_link", str(parent))

    # The rest pose, torso lean included: the vendor's init pose leans the body forward
    # over flat feet, and holding the servos alone puts that 15 degrees on the soles
    # instead -- see `ainex_model.stance_lean` and the flatness check further down.
    rest = rest_pose()
    hold(model, data, rest, ns)
    forward_0 = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2].copy()
    hold(model, data, {**rest, "head_pan": 0.5}, ns)
    forward_1 = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2].copy()
    turned = math.atan2(forward_1[1], forward_1[0]) - math.atan2(forward_0[1], forward_0[0])
    # The vendor gives head_pan the axis `0 0 -1`, as it does every joint, so a positive
    # command yaws the head clockwise. Expect the sign the URDF implies rather than the
    # one that feels natural -- getting this "right" would mean disagreeing with the robot.
    check("panning the head turns the view by the commanded angle",
          abs(turned + 0.5) < math.radians(4.0),
          f"{math.degrees(turned):+.2f} deg for a +0.5 rad command on a -z axis")
    hold(model, data, rest, ns)  # put the head back before anything else is measured

    # ---------------------------------------------------------------- standing
    print("\nstanding:")
    hold(model, data, rest, ns)
    feet = [f"{ns}{f}" for f in ainex_model.FEET]
    ground = lowest_point(model, data, feet)
    check("feet rest on the floor", abs(ground) < 0.01, f"lowest foot point z={ground:+.4f}")

    # On the whole sole, and this needed saying separately: the check above is the lowest
    # single vertex, and it passed at -0.0000 while the robot stood 14.95 degrees toe-up
    # on two heel corners with its toes 37.6 mm in the air. A tilt is what tells a sole
    # resting from a sole touching, and neither the gap nor `ride_height` -- which is
    # measured the same way -- can see the difference.
    tilts = [
        math.degrees(math.atan2(*(lambda r: (r[0, 2], r[2, 2]))(
            data.xmat[model.body(f).id].reshape(3, 3))))
        for f in feet
    ]
    check("and flat on it, not on their heels", max(abs(t) for t in tilts) < 1.0,
          "sole tilt " + ", ".join(f"{t:+.2f}" for t in tilts) + " deg")

    # Heel *and* toe on the floor, measured from the mesh rather than from the body frame
    # -- an independent witness to the tilt above, which is a frame reading.
    for foot in feet:
        zs = mesh_world_z(model, data, [foot])
        check(f"{foot.split('/')[-1]}: the whole sole is down",
              max(zs) - min(zs) < 0.045 and min(zs) - ground < 0.002,
              f"sole spans {(max(zs) - min(zs)) * 1000:.1f} mm, lowest "
              f"{(min(zs) - ground) * 1000:+.1f} mm over the floor")

    top = max(mesh_world_z(model, data, meshed_bodies(model, ns)))
    check(f"standing height in {HEIGHT_RANGE}", HEIGHT_RANGE[0] <= top - ground <= HEIGHT_RANGE[1],
          f"{top - ground:.3f} m")

    # With gravcomp on and 24 small servos, this is what catches a kp clamped too soft by
    # the stability margin.
    before = np.array([data.qpos[model.jnt_qposadr[model.joint(f"{ns}{n}").id]]
                       for n in servos.SERVOS])
    for _ in range(3000):
        mujoco.mj_step(model, data)
    after = np.array([data.qpos[model.jnt_qposadr[model.joint(f"{ns}{n}").id]]
                      for n in servos.SERVOS])
    drift = float(np.abs(after - before).max())
    check("rest pose holds over 3000 steps", drift < 0.01, f"max drift {drift:.5f} rad")

    # ---------------------------------------------------------------- driving
    print("\ndriving:")
    left = np.array([-heading[1], heading[0]])
    fwd = origin + heading * reach
    strafed = fwd + left * strafe_reach

    reached = drive_to(model, data, view, [fwd[0], fwd[1], 0.0])
    check(f"walk {reach:.2f} m forward", np.linalg.norm(reached[:2] - fwd) < 0.02,
          f"reached {np.round(reached[:2], 3)}")
    reached = drive_to(model, data, view, [strafed[0], strafed[1], 0.0])
    check(f"sidestep {strafe_reach:.2f} m", np.linalg.norm(reached[:2] - strafed) < 0.02,
          f"reached {np.round(reached[:2], 3)}")
    check("heading held while sidestepping", abs(reached[2]) < np.radians(1.0),
          f"theta={np.degrees(reached[2]):.3f} deg")
    reached = drive_to(model, data, view, [strafed[0], strafed[1], np.pi / 2])
    check("turn to 90 deg", abs(reached[2] - np.pi / 2) < np.radians(1.0),
          f"theta={np.degrees(reached[2]):.3f} deg")
    check("position held while turning", np.linalg.norm(reached[:2] - strafed) < 0.02,
          f"xy={np.round(reached[:2], 4)}")

    pose = base.pose
    check("pose matches joints",
          np.allclose(pose[:2, 3], reached[:2], atol=1e-6)
          and abs(np.arctan2(pose[1, 0], pose[0, 0]) - reached[2]) < 1e-6)
    target = np.eye(4)
    target[:2, 3] = [2.0, -1.0]
    base.pose = target
    mujoco.mj_forward(model, data)
    check("pose setter teleports",
          np.allclose(np.asarray(base.joint_pos)[:2], [2.0, -1.0], atol=1e-9))

    # ---------------------------------------------------------------- the gait itself
    print("\ngait animation:")
    geom = AiNexRobot.leg_geometry(model, ns)
    check("leg lengths read off the model", 0.05 < geom.thigh < 0.15 and 0.05 < geom.shank < 0.15,
          f"thigh={geom.thigh:.4f} shank={geom.shank:.4f}")

    param = gait.WalkingParam(x_amplitude=0.02, period_time=0.400, step_height=0.03)
    first, last = gait.leg_joint_targets(param, 0.0, geom), gait.leg_joint_targets(param, 1.0, geom)
    check("gait is periodic", all(abs(first[k] - last[k]) < 1e-12 for k in first))

    # The property the 2-link IK exists for: while a foot is planted, the base advances by
    # exactly the distance the foot travels backwards through the body frame. Hand-tuned
    # sinusoids match only at one operating point.
    vx = gait.planar_velocity(param)[0]
    drift = []
    for i in range(200):
        phase = i / 400.0  # the left leg's stance half
        targets = gait.leg_joint_targets(param, phase, geom)
        hip = gait.PITCH_SIGN["l_hip_pitch"] * targets["l_hip_pitch"]
        knee = gait.PITCH_SIGN["l_knee"] * targets["l_knee"]
        foot_x = geom.thigh * math.sin(hip) + geom.shank * math.sin(hip + knee)
        drift.append(foot_x + vx * phase * param.period_time)
    check("stance foot does not skate", (max(drift) - min(drift)) < 0.002,
          f"{(max(drift) - min(drift)) * 1000:.3f} mm over one stance")

    lifts = []
    for i in range(400):
        targets = gait.leg_joint_targets(param, i / 400.0, geom)
        hip = gait.PITCH_SIGN["l_hip_pitch"] * targets["l_hip_pitch"]
        knee = gait.PITCH_SIGN["l_knee"] * targets["l_knee"]
        lifts.append(-(geom.thigh * math.cos(hip) + geom.shank * math.cos(hip + knee)))
    check("swing foot lifts by step_height",
          abs((max(lifts) - min(lifts)) - param.step_height) < 0.002,
          f"{(max(lifts) - min(lifts)) * 1000:.1f} mm for {param.step_height * 1000:.0f} mm")

    # ---------------------------------------------------------------- grasping
    print("\ngrasping:")
    for side, group in (("l", "left_gripper"), ("r", "right_gripper")):
        gripper = view.get_move_group(group)
        closed_a, open_a = GRIPPER_ANGLES[side]
        hold(model, data, {**rest, f"{side}_gripper": open_a}, ns)
        wide = gripper.inter_finger_dist
        hold(model, data, {**rest, f"{side}_gripper": closed_a}, ns)
        shut = gripper.inter_finger_dist
        check(f"{group} opens and closes", wide - shut > 0.03,
              f"open {wide * 1000:.1f} mm, closed {shut * 1000:.1f} mm")

    # Every reach an action group commands must land in the band the arms can cover.
    # Tip height over the lowest sole, sampled through each group's whole replay. In the
    # robot's own frame, which is what the ground-follow makes true over the worktop at
    # run time -- so this is the claw's height over the surface it stands on.
    heights: list[tuple[str, str, float]] = []
    for name, frames in actions.items():
        player = ActionPlayer(frames, rest_pose())
        while not player.finished:
            pose_now = player.step(0.05)
            hold(model, data, pose_now, ns, steps=1)
            mujoco.mj_forward(model, data)
            floor = lowest_point(model, data, feet)
            for site in ("l_tcp", "r_tcp"):
                heights.append(
                    (name, site, float(data.site_xpos[model.site(f"{ns}{site}").id][2] - floor))
                )
    # Two bands. The crawl groups exist to bring a claw to the surface -- the vendor's
    # `crawl_left`/`crawl_right` bend the robot down over its feet -- so for them the
    # grasping hand must dip to apple height, where every other group's hands stay in
    # the standing band.
    crawl = {"crawl_left": "l_tcp", "crawl_right": "r_tcp"}
    outside = sorted({
        (n, round(h, 3)) for n, _, h in heights
        if n not in crawl and not REACH_RANGE[0] <= h <= REACH_RANGE[1]
    })
    check(f"every other action group's hands stay within {REACH_RANGE} m of the floor",
          not outside, str(outside[:3]))
    for group, site in crawl.items():
        lowest = min((h for n, s, h in heights if n == group and s == site), default=math.inf)
        check(f"{group}'s claw dips to the surface", lowest < 0.045,
              f"lowest {site} {lowest * 1000:.0f} mm over the sole")

    if args.render:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 960)
        hold(model, data, rest, ns)
        renderer = mujoco.Renderer(model, 960, 1280)
        renderer.update_scene(data)
        import imageio.v3 as iio

        iio.imwrite(args.render, renderer.render())
        renderer.close()
        print(f"\nwrote {args.render}")

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
