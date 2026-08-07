#!/usr/bin/env python
"""Standalone check that the AiNex attaches, stands, drives and grasps.

Runs without the datagen pipeline, so a failure points at the robot definition rather
than at task sampling.

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

FAIL = []

# Hiwonder publish 415 mm for the assembled robot. Ours measures a little more because
# the vendor figure is presumably taken in a different pose; a wide band still catches a
# model that is mis-scaled or standing on the wrong part of itself.
HEIGHT_RANGE = (0.38, 0.50)
# The reach band the arms actually cover. See robots/ainex/actions/README.md: the torso
# cannot pitch, so the hands never get near the floor.
REACH_RANGE = (0.20, 0.45)


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
    """Command a whole-body joint pose and let it settle."""
    for name, value in pose.items():
        data.ctrl[model.actuator(f"{ns}{name}").id] = value
    for _ in range(steps):
        mujoco.mj_step(model, data)


def mesh_world_z(model, data, body_names) -> list[float]:
    """World z of every mesh vertex on the named bodies.

    Vertices rather than `geom_rbound`, which is a bounding-sphere radius and sits several
    centimetres below a foot's sole -- enough to make a robot standing correctly look like
    it is hovering.
    """
    out: list[float] = []
    for name in body_names:
        bid = model.body(name).id
        geoms = [g for g in range(model.ngeom)
                 if model.geom_bodyid[g] == bid and model.geom_dataid[g] >= 0]
        if not geoms:
            continue
        mesh = model.geom_dataid[geoms[0]]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        pts = model.mesh_vert[start : start + count] + model.geom_pos[geoms[0]]
        world = pts @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
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
    from robots.ainex import gait, servos
    from robots.ainex.actions import ActionPlayer, load_action_dir
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
    out_of_range = [
        (name, j)
        for name, frames in actions.items()
        for f in frames
        for j, v in f.angles.items()
        if not servos.joint_limits(j)[0] - 1e-9 <= v <= servos.joint_limits(j)[1] + 1e-9
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
    check("27 actuators total (24 joints + 3 base)", model.nu == 27, f"{model.nu}")
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
    colliding = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in robot_geoms(model, ns)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    check("only the torso hull and the two hands collide", len(colliding) == 3,
          f"{len(colliding)}: {colliding}")
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

    rest = dict(servos.INIT_POSE)
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
    feet = [f"{ns}l_ank_roll_link", f"{ns}r_ank_roll_link"]
    ground = lowest_point(model, data, feet)
    check("feet rest on the floor", abs(ground) < 0.01, f"lowest foot point z={ground:+.4f}")

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
    heights = []
    for name, frames in actions.items():
        player = ActionPlayer(frames, servos.INIT_POSE)
        while not player.finished:
            pose_now = player.step(0.05)
            hold(model, data, pose_now, ns, steps=1)
        mujoco.mj_forward(model, data)
        floor = lowest_point(model, data, feet)
        for site in ("l_tcp", "r_tcp"):
            heights.append(
                (name, float(data.site_xpos[model.site(f"{ns}{site}").id][2] - floor))
            )
    outside = [(n, round(h, 3)) for n, h in heights if not REACH_RANGE[0] <= h <= REACH_RANGE[1]]
    check(f"every action group's hands stay within {REACH_RANGE} m of the floor",
          not outside, str(outside[:3]))

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
