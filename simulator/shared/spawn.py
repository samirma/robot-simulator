"""The spawn tool both engines run: parse, place, stage, compile, then serve.

Each engine's `tools/spawn_robot.py` is an `Engine` -- how its scene is loaded, where
its worktop and its open floor are, how a robot is grafted into it -- handed to `main`
here. Everything else is one implementation (spec §4, "shared logic exists once"): the
command line, which namespaces and what staging a fleet gets (`serve_args`), where each
robot stands (`placement.stand_fleet`), the task (`tasks/apple_on_plate`), binding each
robot's joints after the compile, the start-up reports, the ROS fleet, the viewer and the
loop. Two copies of any of those were two chances for the engines to drift apart in a
way a client could see.

    python <engine>/tools/spawn_robot.py so101,myagv <scene flags> [--headless] [...]
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import numpy as np

import ainex_model
import placement
import reach
import robots_spec
import serve_args
from mujoco_bridge import PlanarJointBase
from placement import Instance, SurfaceMap, Worktop

#: Where the wire is, when nobody says otherwise: `contracts.rosbridge_server.DEFAULT_PORT`,
#: named here so `--help` works before the contracts package is imported.
DEFAULT_ROS_PORT = 9090
#: The interface the rosbridge server binds: all of them, as the real robots' bridges do.
ROS_HOST = "0.0.0.0"
#: The loop's tick for a member with no periodic rate of its own (the AiNex controller);
#: every published rate is its robot's own, from its ROS file.
CONTROL_HZ = 10.0
#: JPEG quality of every compressed camera image the fleet publishes.
JPEG_QUALITY = 70

#: The task every worktop fleet is staged with (spec §2.3).
TASK = "apple_on_plate"

#: name -> (module, function) presenting that robot's vendor ROS interface. All of them
#: are shared: an engine supplies the robot, never its interface.
ROS_SURFACES = {
    "so101": ("ros_surfaces.so101", "attach_ros"),
    "myagv": ("ros_surfaces.myagv", "attach_ros"),
    "ainex": ("ros_surfaces.ainex", "attach_ros"),
    "myagv_mycobot280": ("ros_surfaces.myagv_mycobot280", "attach_ros"),
    "rosmaster_x3_plus": ("ros_surfaces.rosmaster_x3_plus", "attach_ros"),
}

#: The SO-101's arm joints and gripper in MJCF order, and the upright rest pose with the
#: jaw near open that both engines start it in.
SO101_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
SO101_GRIPPER_JOINTS = ("gripper",)
SO101_REST_QPOS = (0.0, 0.0, -1.5708, 1.0008, -1.5221)
SO101_REST_GRIPPER = (1.2,)


def build_parser(engine) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{engine.name}/tools/spawn_robot.py",
        description=f"Spawn robots into a {engine.name} scene and serve them.")
    ap.add_argument(
        "robot",
        help=f"comma-separated ids of simulated robots in robots_specs/robots.yml "
             f"({', '.join(robots_spec.simulated_ids())}); they share one scene, one port "
             "and one ROS graph, each under its own namespace")
    ap.add_argument(
        "--ros-namespace", default=None, dest="ros_namespace", metavar="NS",
        help="put a lone robot under NS instead of its id; '' serves the bare vendor "
             "interface")
    engine.add_scene_args(ap)
    ap.add_argument("--ros-port", type=int, default=DEFAULT_ROS_PORT, dest="ros_port",
                    metavar="PORT", help="serve the fleet on rosbridge at PORT (default "
                                         "%(default)s; 0 serves nothing)")
    ap.add_argument("--headless", action="store_true",
                    help="run the loop without a window")
    return ap


# ---------------------------------------------------------------- the robots, once compiled


class JointGroup:
    """One move group of an arm, straight off a raw MuJoCo model: what the SO-101's
    shared ROS surface reads (`joint_pos`, `joint_vel`) and writes (`ctrl`)."""

    def __init__(self, model, data, prefix: str, joints) -> None:
        self._data = data
        self._qpos, self._qvel, self._act = [], [], []
        for name in joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{name}")
            if jid < 0 or aid < 0:
                raise SystemExit(f"joint/actuator {prefix}{name!r} missing from the model")
            self._qpos.append(int(model.jnt_qposadr[jid]))
            self._qvel.append(int(model.jnt_dofadr[jid]))
            self._act.append(aid)

    @property
    def joint_pos(self) -> np.ndarray:
        return self._data.qpos[self._qpos].copy()

    @joint_pos.setter
    def joint_pos(self, value) -> None:
        self._data.qpos[self._qpos] = np.asarray(value, dtype=np.float64)

    @property
    def joint_vel(self) -> np.ndarray:
        return self._data.qvel[self._qvel].copy()

    @property
    def ctrl(self) -> np.ndarray:
        return self._data.ctrl[self._act].copy()

    @ctrl.setter
    def ctrl(self, value) -> None:
        self._data.ctrl[self._act] = np.asarray(value, dtype=np.float64)


def root_body(name: str) -> str:
    """The robot's root body: the AiNex roots at the torso its vendor URDF does."""
    return ainex_model.robot_model_root_name() if name == "ainex" else "base"


def rest_positions(name: str) -> dict[str, float]:
    """A mobile robot's arm and gripper joints at the pose it stands in: its contract
    module's `REST_POSITIONS` (joint -> rad), where it has an arm the base carries."""
    module = importlib.import_module(ROS_SURFACES[name][0])
    return dict(getattr(module, "REST_POSITIONS", {}))


def hold(model, data, prefix: str, positions: dict[str, float]) -> None:
    """Put each named joint at its position, with its position servo's target there too,
    and every joint an equality couples to one (a URDF `<mimic>`) where the coupling says."""
    for name, q in positions.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
        if jid < 0 or aid < 0:
            raise SystemExit(f"joint/actuator {prefix}{name!r} missing from the model")
        data.qpos[model.jnt_qposadr[jid]] = q
        data.ctrl[aid] = q
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or model.eq_obj2id[e] < 0:
            continue
        j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j1) or "").startswith(prefix):
            continue
        c = model.eq_data[e]
        x = float(data.qpos[model.jnt_qposadr[j2]])
        data.qpos[model.jnt_qposadr[j1]] = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 \
            + c[4] * x ** 4


def bind(model, data, instances) -> None:
    """Give each robot its engine-neutral handles and put it in its rest pose.

    Both joint state and actuator target: these are position actuators, so a ctrl left
    at 0 would drive every robot out of its pose on the first step.
    """
    for inst in instances:
        if inst.holonomic:
            inst.base = PlanarJointBase(
                model, data, inst.mjcf,
                body=root_body(inst.name) if inst.name == "ainex" else None)
            inst.base.teleport(float(inst.xy[0]), float(inst.xy[1]), float(inst.yaw))
            if inst.name == "ainex":
                try:
                    # The vendor's init pose, torso leaning -- the one function both
                    # engines stand this robot up with.
                    ainex_model.stand(model, data, inst.mjcf)
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
                # ...and its feet meet what lies on the surface it stands on.
                ainex_model.enable_foot_contacts(model, inst.mjcf)
            else:
                hold(model, data, inst.mjcf, rest_positions(inst.name))
        elif inst.name == "so101":
            inst.view = {"arm": JointGroup(model, data, inst.mjcf, SO101_ARM_JOINTS),
                         "gripper": JointGroup(model, data, inst.mjcf, SO101_GRIPPER_JOINTS)}
            for gid, rest in (("arm", SO101_REST_QPOS), ("gripper", SO101_REST_GRIPPER)):
                inst.view[gid].joint_pos = rest
                inst.view[gid].ctrl = rest
        else:
            raise SystemExit(f"no binding for {inst.name!r}")


# ---------------------------------------------------------------- start-up checks


def gripper_bodies(robot: str) -> tuple[str, ...]:
    """The bodies carrying a robot's gripper geoms: the AiNex's hands. Empty for the
    SO-101, whose jaw geoms are found by their MJCF names instead."""
    return tuple(sorted(ainex_model.HAND_BODIES)) if robot == "ainex" else ()


def check_task_contacts(model, namespace: str, task, hand_bodies=()) -> None:
    """Refuse to serve a task whose objects the gripper cannot physically touch.

    MuJoCo pairs two geoms only if `(contype_a & conaffinity_b) or (contype_b &
    conaffinity_a)`, and a loader that rewrites those bitmasks can leave the jaws and the
    apple on disjoint masks. The failure is silent: the jaw closes straight through the
    object and the episode scores zero looking like a near miss.
    """
    if hand_bodies:
        wanted = {f"{namespace}{b}" for b in hand_bodies}
        jaw = [g for g in range(model.ngeom)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     model.geom_bodyid[g]) or "") in wanted]
    else:
        prefixes = (f"{namespace}fixed_jaw", f"{namespace}moving_jaw")
        jaw = [g for g in range(model.ngeom)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefixes)]
    objects = [g for g in range(model.ngeom)
               if model.geom_bodyid[g] in task.contact_bodies()
               and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not jaw or not objects:
        raise SystemExit(f"task contact check: found {len(jaw)} gripper geoms and "
                         f"{len(objects)} collidable task geoms; expected both non-empty")

    def pairs(a: int, b: int) -> bool:
        return bool((model.geom_contype[a] & model.geom_conaffinity[b])
                    or (model.geom_contype[b] & model.geom_conaffinity[a]))

    touchable = sum(1 for j in jaw for o in objects if pairs(j, o))
    print(f"task contacts: {len(jaw)} gripper geoms x {len(objects)} task geoms, "
          f"{touchable} pairs collide", file=sys.stderr)
    if touchable == 0:
        raise SystemExit("task contact check FAILED: no gripper geom can collide with any "
                         "task object; the gripper would close straight through the apple")


def penetrations(model, data, prefix: str, depth: float = -0.001) -> list[str]:
    """Contacts at least 1 mm deep between this robot and anything that is not it."""
    def is_robot(geom_id: int) -> bool:
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        return body.startswith(prefix)

    found = []
    for c in range(data.ncon):
        con = data.contact[c]
        if con.dist > depth:
            continue
        g1, g2 = int(con.geom1), int(con.geom2)
        if is_robot(g1) == is_robot(g2):
            continue
        other = g2 if is_robot(g1) else g1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom {other}"
        found.append(f"intersects {name} by {-con.dist * 1000:.0f} mm")
    return found


def report_sole_contact(model, data, instances) -> list[str]:
    """Where a legged robot's soles are against the surface it stands on.

    The feet do not collide with the world, on purpose (colliding feet fight the planar
    actuators), so a graft that leaves the robot hovering produces no fall and no
    warning: it simply looks like a robot in the air. Returns the problems found.
    """
    problems = []
    for inst in instances:
        if inst.name != "ainex":
            continue
        gap = ainex_model.sole_z(model, data, inst.mjcf) - inst.surface_z
        if abs(gap) <= ainex_model.SOLE_TOLERANCE:
            print(f"{inst.name}: soles on the {inst.on} at z {inst.surface_z:.4f} "
                  f"(gap {gap * 1000:+.2f} mm)", file=sys.stderr)
        else:
            problems.append(f"{inst.name} soles are {gap * 1000:+.1f} mm from the {inst.on} "
                            f"at z {inst.surface_z:.4f}")
    return problems


# ---------------------------------------------------------------- the world


@dataclass
class World:
    """A compiled scene with its fleet placed and the task (if any) staged."""

    model: object
    data: object
    instances: list
    staging: serve_args.Staging
    task: object          # the arbiter, or None
    scene: object         # the engine's bare scene: its compiled model before any robot
    worktop: Worktop | None
    scene_option: object
    problems: list        # placement and start-up problems found; empty when correct


def build_world(engine, args) -> World:
    try:
        names = robots_spec.check_simulated(args.robot)
        namespaces = serve_args.fleet_namespaces(names, args.ros_namespace)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    plan = serve_args.staging(names)
    # `robot_N/` prefixes bodies inside the compiled model; the namespace prefixes names
    # on the wire. Keeping the `robot_N/` shape also keeps the task's "never clear a body
    # called robot_*" rule working.
    instances = [Instance(n, f"robot_{i}/", namespaces[i]) for i, n in enumerate(names)]
    by_name = {i.name: i for i in instances}

    scene = engine.load_scene(args)
    task_mod = importlib.import_module(f"tasks.{TASK}") if plan.task else None

    def task_objects(lead: Instance):
        """All six task objects' footprints in the world: what every other worktop robot
        stands clear of, so the task is always staged whole (spec §2.3)."""
        transform = task_mod.base_frame([lead.xy[0], lead.xy[1], lead.mount_z], lead.yaw)
        return {name: (np.asarray(task_mod._apply(transform, task_mod.OBJECT_POSES[name])[:2]),
                       task_mod.FOOTPRINT_RADIUS[name])
                for name in task_mod.TASK_OBJECTS}

    def sightlines(lead: Instance):
        """Each rig camera to the apple and the plate: what a second worktop robot must
        not stand in, or the rig films its back instead of the task."""
        transform = task_mod.base_frame([lead.xy[0], lead.xy[1], lead.mount_z], lead.yaw)
        return [(task_mod._apply(transform, cam[1]),
                 task_mod._apply(transform, task_mod.OBJECT_POSES[obj]))
                for cam in task_mod.SCENE_CAMERAS for obj in ("apple", "plate")]

    worktop = placement.stand_fleet(
        instances, task_robot=plan.task_robot,
        find_worktop=lambda robot: engine.find_worktop(scene, robot),
        floor_spot=lambda inst, keep_out: engine.floor_spot(scene, inst, keep_out),
        surface_map=lambda w: SurfaceMap(scene.model, scene.data, w),
        task_objects=task_objects, sightlines=sightlines,
    )
    problems = placement.overlaps(instances)

    import robot_models

    for inst in instances:
        # A model built from COLLADA or oversized meshes needs its converted meshes on
        # disk first; a checkout that has not run setup since gets them here, once.
        robot_models.ensure(inst.name)
        engine.attach(scene, inst)
    # The AiNex's actuator gains assume an implicit integrator; with Euler its 24 servos
    # on ~1e-4 kg.m^2 links go NaN. Set only when it is present, because it changes the
    # physics of everything else in the scene.
    if "ainex" in by_name:
        scene.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    lead = by_name.get(plan.task_robot) if plan.task else None
    if lead is not None:
        # The task's frame is the task robot's base body, with the worktop at z = 0.
        # The other worktop robots already stand clear of all six task objects; loose
        # scene objects around them are cleared as they are in the working area.
        cleared = task_mod.stage(
            scene.spec, [float(lead.xy[0]), float(lead.xy[1]), lead.mount_z], lead.yaw,
            keep_clear=[(i.xy, i.radius) for i in instances
                        if i.on == "worktop" and i is not lead],
        )
        if cleared:
            print(f"task {TASK}: cleared {len(cleared)} scene object(s) from the working "
                  f"area: {', '.join(n.split('_')[0] for n in cleared)}", file=sys.stderr)

    model = scene.spec.compile()
    data = mujoco.MjData(model)
    bind(model, data, instances)
    mujoco.mj_forward(model, data)
    for inst in instances:
        for found in penetrations(model, data, inst.mjcf):
            problems.append(f"{inst.name} {found} at its spawn pose")

    arbiter = None
    if lead is not None:
        # After the rest pose: the arbiter snapshots this state as the one /reset
        # restores. The prefix and root are passed rather than inferred -- several
        # robots root at a body called `base`.
        arbiter = task_mod.AppleOnPlate(model, data, prefix=lead.mjcf,
                                        root=root_body(lead.name),
                                        start_pose=lead.name == "so101")
        placed, reason = arbiter.instantaneous(data)
        print(f"task {TASK}: staged in front of {lead.name}; success predicate reads "
              f"{'TRUE (!)' if placed else reason} at spawn", file=sys.stderr)
        print(f"task {TASK}: {arbiter.layout_report(data)}", file=sys.stderr)
        check_task_contacts(model, lead.mjcf, arbiter, hand_bodies=gripper_bodies(lead.name))
        report_reach(model, data, instances, arbiter, scene, worktop)
    problems += report_sole_contact(model, data, instances)
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    print(f"{','.join(names)} in {engine.name} scene: {model.nbody} bodies, "
          f"{model.ngeom} geoms, {model.nu} actuators", file=sys.stderr)
    return World(model, data, instances, plan, arbiter, scene, worktop,
                 engine.scene_option(), problems)


def report_reach(model, data, instances, arbiter, scene, worktop) -> None:
    """Whether each worktop robot can get at the apple and the plate (spec §2.3).

    The SO-101 by solving a top grasp at each on the compiled model. A robot that walks
    (the AiNex) by whether it can walk over the worktop to each.
    """
    objects = arbiter.object_positions(data)
    task_mod = importlib.import_module(f"tasks.{TASK}")
    radii = {"apple": task_mod.APPLE_RADIUS, "plate": task_mod.PLATE_RADIUS}
    walkers = [i for i in instances if i.on == "worktop" and i.name != "so101"]
    surface = SurfaceMap(scene.model, scene.data, worktop) if walkers else None
    for inst in instances:
        if inst.on != "worktop":
            continue
        if inst.name == "so101":
            for line in reach.report(model, data, inst.mjcf, objects):
                print(f"reach so101 {line}", file=sys.stderr)
            continue
        for name, xyz in objects.items():
            d = float(np.linalg.norm(np.asarray(xyz[:2]) - inst.xy))
            ok = placement.walkable(surface, inst.xy, xyz[:2],
                                    radii[name] + placement.WALK_STOP_SHORT)
            print(f"reach {inst.name} {name}: {'in reach' if ok else 'OUT of reach'} -- "
                  f"{d:.2f} m away, {'a walk over the worktop' if ok else 'no clear walk'}",
                  file=sys.stderr)


# ---------------------------------------------------------------- serving and viewing


def pick_camera(model, prefix: str) -> str | None:
    """A robot's own camera, resolved against its own MJCF prefix."""
    for candidate in (f"{prefix}front_camera", f"{prefix}wrist_cam", f"{prefix}rgb_camera"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, candidate) >= 0:
            return candidate
    return None


def surface_kwargs(inst, world) -> dict:
    """Everything one robot's surface needs, by which interface it presents."""
    model, prefix = world.model, inst.mjcf
    if inst.name == "so101":
        return {"view": inst.view, "model": model, "task": world.task, "wrist": True,
                "jpeg_quality": JPEG_QUALITY, "control_hz": CONTROL_HZ,
                "scene_option": world.scene_option, "prefix": prefix}
    camera = pick_camera(model, prefix)
    if inst.holonomic and inst.name != "ainex":
        # The planar-base robots: the myAGV, and the two mobile manipulators that ride
        # the same base. Each surface resolves its own cameras against `prefix`.
        return {
            "base": inst.base, "model": model, "camera": camera,
            "jpeg_quality": JPEG_QUALITY, "scene_option": world.scene_option,
            "lidar": {
                # Rays start at the robot's own root and range nothing of its own; a
                # neighbour is something to see.
                "body": f"{prefix}base",
                "exclude_bodies": frozenset(
                    i for i in range(model.nbody)
                    if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
                    and n.startswith(prefix)),
            },
            "prefix": prefix,
        }
    if inst.name == "ainex":
        return {"base": inst.base, "model": model, "camera": camera,
                "jpeg_quality": JPEG_QUALITY, "control_hz": CONTROL_HZ,
                "scene_option": world.scene_option, "prefix": prefix}
    raise SystemExit(f"no ROS surface arguments for {inst.name!r}")


def build_fleet(args, world):
    """One server, one port, one graph -- and a namespace per robot."""
    from ros_surfaces import RobotFleet
    from ros_surfaces.scene import (
        SCENE_CAMERA_TOPICS, SCENE_NAMESPACE, attach_scene_rig, probe_scene_cameras,
    )

    fleet = RobotFleet(port=args.ros_port, host=ROS_HOST, default_hz=CONTROL_HZ)
    for inst in world.instances:
        module_name, func_name = ROS_SURFACES[inst.name]
        attach = getattr(importlib.import_module(module_name), func_name)
        fleet.attach(inst.ns, attach, **surface_kwargs(inst, world))
    if world.staging.rig:
        # The worktop's rig, under its own namespace and after the robots so they step
        # first: it watches the task, not any robot.
        rig = probe_scene_cameras(world.model, SCENE_CAMERA_TOPICS)
        if not rig:
            raise SystemExit("the task is staged but its rig cameras are not in the model")
        # The truth the console's offline audit reads, to a local file and only when
        # SIMULATOR_TRUTH_LOG asks (tasks/truth_log.py); nothing of it reaches the wire.
        from tasks.truth_log import from_environment

        arm = next((inst.mjcf for inst in world.instances if inst.name == "so101"), None)
        truth = from_environment(world.model, world.task, arm_prefix=arm)
        if truth is not None:
            fleet.world_reset.on_observed(truth.reset)
        fleet.attach(SCENE_NAMESPACE, attach_scene_rig, model=world.model, cameras=rig,
                     jpeg_quality=JPEG_QUALITY, scene_option=world.scene_option,
                     truth=truth)
    fleet.start()
    return fleet


def framing(world):
    """(lookat, distance, azimuth) of a free camera on the first robot."""
    model, data = world.model, world.data
    first = world.instances[0]
    points = np.array([
        data.geom_xpos[g] for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
            ).startswith(first.mjcf)])
    if len(points):
        lookat = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) / 2, 0.2)
    else:
        lookat = np.array([first.xy[0], first.xy[1], first.surface_z + 0.3])
        radius = 0.5
    return lookat, radius * 4.0, math.degrees(first.yaw) + 180.0


def run(args, world) -> int:
    from mujoco_bridge import run_sim_loop

    controller = build_fleet(args, world) if args.ros_port else None
    loop_hz = (controller.rate_hz if controller is not None else None) or CONTROL_HZ
    try:
        if args.headless:
            run_sim_loop(world.model, world.data, controller, control_hz=loop_hz,
                         label="headless loop")
        else:
            # Bound as a separate name: `import mujoco.viewer` here would make `mujoco` a
            # function-local and shadow the module import.
            from mujoco import viewer as mj_viewer

            lookat, distance, azimuth = framing(world)
            with mj_viewer.launch_passive(world.model, world.data) as viewer:
                if world.scene_option is not None:
                    viewer.opt.geomgroup[:] = world.scene_option.geomgroup
                viewer.cam.lookat[:] = lookat
                viewer.cam.distance = distance
                viewer.cam.azimuth = azimuth
                viewer.cam.elevation = -20.0
                run_sim_loop(world.model, world.data, controller, control_hz=loop_hz,
                             viewer=viewer, label="viewer loop")
    finally:
        if controller is not None:
            controller(None)
    return 0


def main(engine, argv=None) -> int:
    args = build_parser(engine).parse_args(argv)
    return run(args, build_world(engine, args))


def scene(spec, model, data, **extra) -> SimpleNamespace:
    """What `Engine.load_scene` returns: the spec robots are grafted into, and that spec
    compiled bare (no robot, no task) with its data, which placement searches."""
    return SimpleNamespace(spec=spec, model=model, data=data, **extra)
