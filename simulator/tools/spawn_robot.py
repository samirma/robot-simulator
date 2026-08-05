#!/usr/bin/env python
"""Spawn an out-of-tree robot into a house and view or render it.

Places the robot somewhere worth being, attaches it, and then either opens the passive
viewer (run under mjpython) or writes a PNG.

    mjpython tools/spawn_robot.py so101 --scene <house.xml>
    python   tools/spawn_robot.py so101 --scene <house.xml> --render out.png

Where "somewhere worth being" depends on what the robot is:

* An **arm** has no way of getting anywhere, so it is mounted *on* a table -- a surface at
  working height with graspable objects on it, chosen by `tools/scene_placement.py` -- on a
  short riser in place of its floor pedestal, positioned so those objects are inside its
  working annulus. If the house has a surface but nothing on it, objects are spawned onto
  it, because an arm facing an empty table is not a useful thing to have rendered.
* A **mobile base** is put on the most open floor the house has, which is where a robot
  that has to drive wants to start.

This is the "spawn and look at it" path. It deliberately does not involve the task
sampler or a scripted policy, so it works for robots that have no grasp library.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

# name -> (module, config class, robot class), kept lazy so importing one robot does
# not require the others to be installed.
ROBOTS = {
    "so101": ("robots.so101", "SO101RobotConfig", "SO101Robot"),
    "myagv": ("robots.myagv", "MyAGVRobotConfig", "MyAGVRobot"),
    "lekiwi": ("robots.lekiwi", "LeKiwiRobotConfig", "LeKiwiRobot"),
    "rebot_b601": ("robots.rebot_b601", "B601RobotConfig", "B601Robot"),
}

# Robots whose base is three virtual holonomic joints must be grafted in at the origin
# and then *driven* to their spawn pose, because the slide joints are world-aligned.
# Robots on a mocap mount are placed by the attach pos/quat instead.
HOLONOMIC_BASE_ROBOTS = {"myagv", "lekiwi"}

# Arms: no base of their own, so they are bolted to a work surface rather than stood on
# the floor. (lekiwi has an arm too, but it also has wheels -- it places as a mobile base.)
TABLETOP_ROBOTS = {"so101", "rebot_b601"}

# The annulus on that surface the arm can comfortably work in. The SO-101 is a ~0.4 m
# tabletop arm (robots/so101/so101_config.py:41); the B601 has 767 mm of reach
# (robots/rebot_b601/b601_config.py:32). Both are kept short of the full figure: the last
# few centimetres of reach are a straight-out arm with no usable orientation left.
ARM_REACH = {"so101": (0.15, 0.35), "rebot_b601": (0.25, 0.60)}

# Room the arm itself needs around its mount, and how far up it needs it. Not the same as
# the base it stands on: a counter is against a wall, and it is the arm -- not the riser --
# that hits it. The height bounds which geometry counts as in the way at all, so that a
# wall does and the ceiling above it does not.
ARM_BODY_RADIUS = {"so101": 0.20, "rebot_b601": 0.35}
ARM_BODY_HEIGHT = {"so101": 0.45, "rebot_b601": 0.90}

# Replaces the floor pedestal (`base_size`) once the arm stands on the table itself: just
# enough to read as a mount, not enough to matter to the workspace.
TABLETOP_RISER = [0.14, 0.14, 0.03]

# How far the position setpoint of a velocity-driven base may run ahead of the robot.
# Large enough that the servo is always pulling at full effort, small enough that a robot
# stopped by a wall has only centimetres of wind-up to release when it turns away.
TARGET_LEAD_M = 0.12
# Tighter than the linear lead: yaw is the axis with the least inertia (~0.05 kg m2), and
# a setpoint far ahead of the robot yanks it round hard enough to shove the base sideways
# through the position servo -- which shows up as translation during a pure rotation.
TARGET_LEAD_RAD = 0.15

# Spawned stand-in objects, when a surface has nothing on it worth reaching for.
SPAWN_OBJECT_HALF = 0.025
SPAWN_OBJECT_COLOURS = (
    [0.85, 0.25, 0.20, 1.0],
    [0.20, 0.60, 0.85, 1.0],
    [0.90, 0.75, 0.20, 1.0],
    [0.35, 0.70, 0.35, 1.0],
)


def load_robot(name: str):
    if name not in ROBOTS:
        raise SystemExit(f"unknown robot {name!r}; available: {', '.join(sorted(ROBOTS))}")
    module_name, config_attr, robot_attr = ROBOTS[name]
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, config_attr), getattr(module, robot_attr)


def find_open_spot(
    scene_path: str, clearance_band: tuple[float, float] = (0.05, 1.3)
) -> tuple[np.ndarray, float]:
    """Return (xy position, yaw) for the most open floor spot, facing the room centre.

    Obstacles are taken to be collision geoms whose centres sit in the height band a
    robot on its pedestal would sweep through; the spot maximising distance to the
    nearest such geom is chosen.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    obstacles = np.array(
        [
            data.geom_xpos[i][:2]
            for i in range(model.ngeom)
            if model.geom_contype[i] != 0 and clearance_band[0] < data.geom_xpos[i][2] < clearance_band[1]
        ]
    )
    if obstacles.size == 0:
        return np.zeros(2), 0.0

    lo, hi = obstacles.min(axis=0), obstacles.max(axis=0)
    centre = (lo + hi) / 2

    xs = np.arange(lo[0], hi[0], 0.1)
    ys = np.arange(lo[1], hi[1], 0.1)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    # distance from every candidate to its nearest obstacle
    clearances = np.linalg.norm(grid[:, None, :] - obstacles[None, :, :], axis=-1).min(axis=1)

    best = grid[int(np.argmax(clearances))]
    to_centre = centre - best
    yaw = float(np.arctan2(to_centre[1], to_centre[0]))
    print(
        f"placing robot at ({best[0]:.2f}, {best[1]:.2f}) "
        f"clearance {clearances.max():.2f} m, facing room centre (yaw {np.degrees(yaw):.0f} deg)",
        file=sys.stderr,
    )
    return best, yaw


def add_tabletop_objects(spec, xy_min, xy_max, top_z: float, n: int) -> list[np.ndarray]:
    """Put `n` small graspable boxes on a surface, and return where they landed.

    Only runs when the house gave us a work surface with nothing on it. Primitive boxes
    rather than library assets on purpose: `install_uid` wants a network fetch and a uid
    known to be small enough to pick up, and this is the fallback path -- it should not be
    the one that needs the internet. Shaped after
    `molmo_spaces/tasks/task_scene_utils.py::add_pickup_target`, but positioned from the
    surface we actually measured instead of a hardcoded z.
    """
    xy_min = np.asarray(xy_min, dtype=float)
    xy_max = np.asarray(xy_max, dtype=float)
    centre = (xy_min + xy_max) / 2
    # Keep them off the lip of the surface, and inside it even if it is a narrow shelf.
    half = np.maximum((xy_max - xy_min) / 2 - 0.12, 0.0)

    positions = []
    for i in range(n):
        angle = 2 * np.pi * i / max(n, 1)
        offset = half * np.array([np.cos(angle), np.sin(angle)]) * 0.6
        pos = np.array(
            [centre[0] + offset[0], centre[1] + offset[1], top_z + SPAWN_OBJECT_HALF + 0.002]
        )
        body = spec.worldbody.add_body(name=f"spawned_object_{i}", pos=pos.tolist())
        body.add_freejoint()
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[SPAWN_OBJECT_HALF] * 3,
            rgba=SPAWN_OBJECT_COLOURS[i % len(SPAWN_OBJECT_COLOURS)],
            mass=0.1,
        )
        positions.append(pos)
    return positions


def place_arm_on_table(scene_path: str, model, data, robot: str, reach, n_spawn: int):
    """Choose the surface, guarantee something is on it, and return (mount, spawn_request).

    Detection has to happen against a compiled model, but spawning objects happens on the
    spec -- so this returns what to spawn rather than spawning it, and the caller adds the
    objects before its own compile. The target is then built from the positions we chose,
    never by re-running detection: `find_grasp_targets` keys off the scene metadata JSON,
    which knows nothing about bodies we just invented, so a re-detect would come back empty.
    """
    from tools.scene_placement import (
        GraspTarget,
        dynamic_clutter,
        find_grasp_targets,
        find_supports,
        find_tabletop_mount,
        load_scene_map,
        static_blockers,
    )

    thormap = load_scene_map(scene_path)
    targets = find_grasp_targets(scene_path, model, data, thormap=thormap)
    if targets:
        # Most targets share a surface, so the blocker sweep -- which walks every geom in
        # the house -- is cached by the height it was taken at.
        blocker_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        fallback = None
        for target in targets:
            key = round(target.support_top_z, 2)
            if key not in blocker_cache:
                height = ARM_BODY_HEIGHT.get(robot, 1.0)
                blocker_cache[key] = (
                    static_blockers(model, data, target.support_top_z, height=height),
                    dynamic_clutter(model, data, target.support_top_z, height=height),
                )
            mount = find_tabletop_mount(
                target,
                reach_range=reach,
                footprint=TABLETOP_RISER[0],
                blockers=blocker_cache[key][0],
                clutter=blocker_cache[key][1],
                body_radius=ARM_BODY_RADIUS.get(robot),
                model=model,
                data=data,
            )
            # A surface with no cell both clear and in reach is not the right surface: the
            # next target is usually the same table seen from a different object, and after
            # that a different table entirely. Keep the best rejected one as a floor.
            if mount.clear and mount.n_in_reach:
                return mount, None
            if fallback is None or mount.n_in_reach > fallback.n_in_reach:
                fallback = mount
        print(
            f"no surface in {Path(scene_path).name} has room for {robot} clear of its "
            f"surroundings; using the roomiest spot found",
            file=sys.stderr,
        )
        return fallback, None

    supports = find_supports(scene_path, model, data)
    if not supports:
        raise SystemExit(
            f"no work surface at arm height in {Path(scene_path).name}: nothing to mount "
            f"{robot} on. Try another scene index, or pass --pos/--yaw to place it by hand."
        )
    if n_spawn <= 0:
        listing = "\n".join(
            f"  {s[0]} ({s[1] or '?'}) top z={s[2][2] + s[3][2] / 2:.2f} "
            f"{s[3][0]:.2f}x{s[3][1]:.2f} m"
            for s in supports[:8]
        )
        raise SystemExit(
            f"no graspable objects on any surface in {Path(scene_path).name}, and "
            f"--spawn-objects 0 forbids adding some. Candidate surfaces:\n{listing}"
        )

    s_name, s_cat, s_centre, s_dims, s_rects = supports[0]
    xy_min = s_centre[:2] - s_dims[:2] / 2
    xy_max = s_centre[:2] + s_dims[:2] / 2
    top_z = float(s_centre[2] + s_dims[2] / 2)
    print(
        f"no graspables in the scene; spawning {n_spawn} onto "
        f"{s_cat or s_name} (top {top_z:.2f} m)",
        file=sys.stderr,
    )

    # Predict the layout `add_tabletop_objects` will produce so the mount can be chosen
    # against it; the caller then spawns exactly that.
    dims = np.array([SPAWN_OBJECT_HALF * 2] * 3)
    centre = (xy_min + xy_max) / 2
    half = np.maximum((xy_max - xy_min) / 2 - 0.12, 0.0)
    predicted = []
    for i in range(n_spawn):
        angle = 2 * np.pi * i / max(n_spawn, 1)
        offset = half * np.array([np.cos(angle), np.sin(angle)]) * 0.6
        predicted.append(
            np.array([centre[0] + offset[0], centre[1] + offset[1], top_z + SPAWN_OBJECT_HALF + 0.002])
        )

    target = GraspTarget(
        support_name=s_name,
        support_category=s_cat,
        support_top_z=top_z,
        object_name="spawned_object_0",
        object_category="spawned box",
        object_xyz=predicted[0],
        n_objects_on_support=len(predicted),
        reach_slack=0.0,
        support_xy_min=xy_min,
        support_xy_max=xy_max,
        objects_on_support=tuple((p, dims) for p in predicted),
        support_top_rects=s_rects,
        support_body_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s_name)),
    )
    mount = find_tabletop_mount(
        target,
        reach_range=reach,
        footprint=TABLETOP_RISER[0],
        blockers=static_blockers(model, data, top_z, height=ARM_BODY_HEIGHT.get(robot, 1.0)),
        clutter=dynamic_clutter(model, data, top_z, height=ARM_BODY_HEIGHT.get(robot, 1.0)),
        model=model,
        data=data,
        body_radius=ARM_BODY_RADIUS.get(robot),
    )
    return mount, (xy_min, xy_max, top_z, n_spawn)


def warn_on_penetration(model, data, namespace: str, depth: float = -0.001) -> None:
    """Complain if the robot was bolted down inside something.

    The arm hangs off a mocap body, so nothing pushes it back out and nothing falls over:
    a mount that clips a bowl just silently interpenetrates. This is the only thing that
    notices.
    """
    def is_robot(geom_id: int) -> bool:
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        return body.startswith(namespace)

    for c in range(data.ncon):
        con = data.contact[c]
        if con.dist > depth:
            continue
        g1, g2 = int(con.geom1), int(con.geom2)
        if is_robot(g1) == is_robot(g2):
            continue
        other = g2 if is_robot(g1) else g1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom {other}"
        print(
            f"warning: robot intersects {name} by {-con.dist * 1000:.0f} mm at its spawn pose",
            file=sys.stderr,
        )
        return


def connect_control(
    endpoint: str, view, model, camera: str | None, camera_size, jpeg_quality: int,
    connect_timeout: float = 180.0,
):
    """Connect to a bridge control server and return a step callback.

    Speaks the same msgpack-numpy protocol as bridge/server.py: send an observation,
    receive a dict of `{move_group: target}`, write those into the move-group ctrl.
    Calling the returned function with None closes the connection.

    Robot-agnostic: whatever move groups the RobotView exposes are sent and accepted,
    so this drives an arm ("arm"/"gripper") or a mobile base ("base") equally.

    If `camera` names an MJCF camera, each observation also carries a `camera` frame.
    Frames are JPEG-encoded rather than sent raw: 640x480x3 raw is ~920 KB, which at
    20 Hz is 18 MB/s and enough to stall the control loop.

    This is the lightweight path for a robot spawned straight into a scene; the
    datagen-pipeline equivalent is bridge/run_bridge_sim.py.
    """
    import msgpack_numpy
    import websockets.sync.client as ws_client

    uri = endpoint if endpoint.startswith("ws") else f"ws://{endpoint}"

    # Wait for the control server rather than dying if it is not up yet. `open_timeout`
    # covers the handshake, not a refused connection, so without this loop whichever of
    # the two processes starts first simply exits — and the simulator takes ~20 s to load
    # a scene, so it is usually the one left waiting. Same retry the upstream
    # WebsocketPolicy uses.
    deadline = time.monotonic() + connect_timeout
    attempt = 0
    while True:
        try:
            conn = ws_client.connect(uri, compression=None, max_size=None, open_timeout=30)
            break
        except (ConnectionRefusedError, OSError) as exc:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"no control server at {uri} after {connect_timeout:.0f}s. "
                    "Start one with: cd ../robot_console && ./bin/teleop.sh"
                ) from exc
            if attempt == 0:
                print(f"waiting for a control server at {uri} ...", file=sys.stderr)
            attempt += 1
            time.sleep(2.0)

    metadata = msgpack_numpy.unpackb(conn.recv(timeout=10))
    print(f"control server: {metadata}", file=sys.stderr)

    groups = {gid: view.get_move_group(gid) for gid in view.move_group_ids()}

    renderer = None
    if camera is not None:
        width, height = camera_size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        renderer = mujoco.Renderer(model, height, width)
        print(f"streaming camera {camera!r} at {width}x{height}", file=sys.stderr)

    def encode(frame):
        try:
            import cv2

            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if ok:
                return {"camera_jpeg": np.asarray(buf).reshape(-1)}
        except Exception:
            pass
        # Fall back to the raw array rather than dropping the feed entirely.
        return {"camera": frame}

    def step(data):
        if data is None:
            if renderer is not None:
                renderer.close()
            conn.close()
            return

        obs = {
            "qpos": {gid: np.asarray(g.joint_pos).tolist() for gid, g in groups.items()},
            "qvel": {gid: np.asarray(g.joint_vel).tolist() for gid, g in groups.items()},
            "actions/joint_pos": {
                gid: np.asarray(g.ctrl).tolist() for gid, g in groups.items()
            },
            "task": "",
        }
        if "arm" in groups:
            obs["tcp_pose"] = groups["arm"].leaf_frame_to_world[:3, 3]
        if "base" in groups:
            pose = groups["base"].pose
            obs["base_pose"] = np.array(
                [pose[0, 3], pose[1, 3], float(np.arctan2(pose[1, 0], pose[0, 0]))]
            )
        if renderer is not None:
            renderer.update_scene(data, camera=camera)
            obs.update(encode(renderer.render()))

        conn.send(msgpack_numpy.packb(obs))
        reply = conn.recv(timeout=30)
        if isinstance(reply, str):
            raise RuntimeError(f"control server error:\n{reply}")
        action = msgpack_numpy.unpackb(reply)
        for gid, target in action.items():
            if gid in groups:
                groups[gid].ctrl = np.asarray(target, dtype=np.float64)

    return step


def laser_scan_ranges(
    model,
    data,
    origin: np.ndarray,
    yaw: float,
    beams: int,
    max_range: float,
    bodyexclude: int = -1,
    angle_min: float = -np.pi,
    angle_max: float = np.pi,
) -> np.ndarray:
    """A 2D laser scan by ray casting, counter-clockwise from `angle_min`.

    `origin` is the *laser* origin, not the base origin -- see serve_ros for the mount
    offset. Angles are measured in the base frame, so `yaw + angle_min` is the first beam.

    The real 2023 Pi AGV carries a YDLidar X2; this stands in for it. `mj_ray` is the only
    per-step ranging primitive MuJoCo offers -- rangefinder sensors would mean regenerating
    `model.xml`, and depth rendering costs an order of magnitude more per beam. Same call
    shape as tools/scene_placement.py::_standing_on.

    On the beam ordering, which is easy to get backwards: the X2 is launched with
    `inverted: true` because it is mounted upside down, and `myagv_active.launch` then
    publishes `base_footprint -> laser_frame` with a roll of pi. Those two mirrors cancel,
    so in the base frame the published scan runs counter-clockwise from -pi -- which is
    what this function produces. Do not "fix" one without the other.

    Misses come back as `max_range + 1`; see bridge.rosbridge_server.laser_scan for why
    that rather than the X2's 0.0.
    """
    angles = yaw + np.linspace(angle_min, angle_max, beams, endpoint=False)
    geomid = np.zeros(1, dtype=np.int32)
    ranges = np.full(beams, max_range + 1.0)
    origin = np.ascontiguousarray(origin, dtype=np.float64)

    for i, a in enumerate(angles):
        vec = np.array([np.cos(a), np.sin(a), 0.0])
        dist = mujoco.mj_ray(model, data, origin, vec, None, 1, bodyexclude, geomid)
        if geomid[0] >= 0 and 0.0 <= dist <= max_range:
            ranges[i] = dist
    return ranges


def serve_ros(port: int, view, model, camera: str | None, camera_size, jpeg_quality: int,
              control_hz: float, watchdog_s: float, scan: dict | None = None,
              depth: dict | None = None):
    """Present the robot on the myagv_ros topics and return a per-step callback.

    Subscribes `cmd_vel` and publishes `odom` plus a compressed camera image, so a ROS
    client drives the simulated robot exactly as it drives the real one. See
    bridge/rosbridge_server.py for why the protocol is served in-process.

    The base integrates the commanded velocity here rather than in the client: that is
    what `cmd_vel` means, and it keeps the client identical for real hardware.
    """
    sys.path.insert(0, str(SIM_ROOT))
    from bridge.rosbridge_server import (
        TOPIC_CAMERA,
        TOPIC_CAMERA_INFO,
        TOPIC_CMD_VEL,
        TOPIC_DEPTH,
        TOPIC_ODOM,
        TOPIC_SCAN,
        RosBridgeServer,
        camera_info,
        compressed_image,
        image,
        laser_scan,
        odometry,
    )

    if "base" not in view.move_group_ids():
        raise SystemExit(
            "--ros-port needs a robot with a mobile base (myagv, lekiwi); "
            f"this one has move groups {view.move_group_ids()}"
        )
    base = view.get_move_group("base")

    renderer = None
    if camera is not None:
        width, height = camera_size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        renderer = mujoco.Renderer(model, height, width)

    # A second renderer, because a MuJoCo renderer is either in depth mode or not, and
    # toggling it per frame would fight the colour stream sharing the same object.
    depth_renderer = None
    if depth is not None and camera is not None:
        dw, dh = depth["size"]
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, dw)
        model.vis.global_.offheight = max(model.vis.global_.offheight, dh)
        depth_renderer = mujoco.Renderer(model, dh, dw)
        depth_renderer.enable_depth_rendering()

    server = RosBridgeServer(port=port)
    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "at": 0.0}

    def on_cmd_vel(msg: dict) -> None:
        linear = msg.get("linear") or {}
        angular = msg.get("angular") or {}
        command["vx"] = float(linear.get("x", 0.0))
        command["vy"] = float(linear.get("y", 0.0))
        command["wz"] = float(angular.get("z", 0.0))
        command["at"] = time.monotonic()

    server.on(TOPIC_CMD_VEL, on_cmd_vel)
    server.start()
    published = [TOPIC_ODOM]
    if renderer is not None:
        published.append(TOPIC_CAMERA)
    if scan is not None:
        published.append(TOPIC_SCAN)
    if depth_renderer is not None:
        published += [TOPIC_DEPTH, TOPIC_CAMERA_INFO]
    print(
        f"ROS topics on ws://0.0.0.0:{port} "
        f"(sub {TOPIC_CMD_VEL}; pub {', '.join(published)})",
        file=sys.stderr,
    )

    dt = 1.0 / control_hz
    # The lidar rides on the robot's root body, which is also the one body its own rays
    # must ignore -- the myAGV's only collision geom is the chassis box on that body.
    scan_body = -1
    if scan is not None:
        scan_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, scan["body"]
        )
    scan_clock = {"next": 0.0}
    depth_clock = {"next": 0.0}
    target = {"at": None}

    def step(data):
        if data is None:
            if renderer is not None:
                renderer.close()
            if depth_renderer is not None:
                depth_renderer.close()
            server.stop()
            return

        # Watchdog. myAGVSub.cpp latches the last cmd_vel and keeps executing it, so a
        # client that disconnects mid-command would leave the robot driving; stopping is
        # the behaviour we want even though it is not what the firmware does.
        vx, vy, wz = command["vx"], command["vy"], command["wz"]
        if command["at"] and time.monotonic() - command["at"] > watchdog_s:
            vx = vy = wz = 0.0

        pose = base.pose
        x, y = float(pose[0, 3]), float(pose[1, 3])
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))

        # Body-frame velocity -> world-frame target. The target is carried forward from
        # step to step rather than re-derived from the measured pose each time: these are
        # position actuators, so re-basing on the measurement left the setpoint only ever
        # one increment (14 mm at 0.28 m/s) ahead of the robot, and the servo settled at
        # roughly a sixth of the commanded speed. Integrating the target instead makes
        # cmd_vel mean what it says.
        c, s = np.cos(yaw), np.sin(yaw)
        if target["at"] is None:
            target["at"] = np.array([x, y, yaw])
        target["at"] = target["at"] + np.array(
            [(vx * c - vy * s) * dt, (vx * s + vy * c) * dt, wz * dt]
        )
        # ...but never more than a short lead ahead of where the robot actually is, so a
        # robot held up by a wall stops advancing instead of winding up a setpoint it will
        # lunge to the moment it comes free.
        lag = target["at"][:2] - np.array([x, y])
        dist = float(np.linalg.norm(lag))
        if dist > TARGET_LEAD_M:
            target["at"][:2] = np.array([x, y]) + lag / dist * TARGET_LEAD_M
        yaw_lag = float(np.arctan2(np.sin(target["at"][2] - yaw), np.cos(target["at"][2] - yaw)))
        if abs(yaw_lag) > TARGET_LEAD_RAD:
            target["at"][2] = yaw + np.sign(yaw_lag) * TARGET_LEAD_RAD
        base.ctrl = target["at"]

        seq = server.next_seq()
        server.publish(TOPIC_ODOM, odometry(seq, x, y, yaw, vx, vy, wz))

        if scan is not None and time.monotonic() >= scan_clock["next"]:
            # The X2 spins at a fixed 10 Hz regardless of how fast anything else runs, so
            # /scan gets its own clock rather than riding the control rate. A client that
            # assumes one scan per cmd_vel would break on the real robot.
            scan_clock["next"] = time.monotonic() + scan["period"]
            # myagv_active.launch mounts the laser 65 mm ahead of base_footprint and 80 mm
            # up. Casting from the base centre instead would shift every reading by that
            # much, which is a whole map cell at the 5 cm resolution gmapping uses.
            ranges = laser_scan_ranges(
                model,
                data,
                np.array(
                    [
                        x + scan["offset_x"] * np.cos(yaw),
                        y + scan["offset_x"] * np.sin(yaw),
                        scan["offset_z"],
                    ]
                ),
                yaw,
                scan["beams"],
                scan["max_range"],
                bodyexclude=scan_body,
            )
            step_angle = 2 * np.pi / scan["beams"]
            server.publish(
                TOPIC_SCAN,
                laser_scan(
                    seq,
                    ranges,
                    -np.pi,
                    np.pi - step_angle,
                    step_angle,
                    range_min=scan["min_range"],
                    range_max=scan["max_range"],
                    scan_time=scan["period"],
                ),
            )

        if depth_renderer is not None and time.monotonic() >= depth_clock["next"]:
            depth_clock["next"] = time.monotonic() + depth["period"]
            depth_renderer.update_scene(data, camera=camera)
            # Metres to millimetres in uint16: 640x480 float32 is 1.2 MB a frame, which a
            # JSON websocket will not carry at any useful rate. Anything beyond the sensor
            # range becomes 0, which is what "no return" means in a 16UC1 depth image.
            metres = depth_renderer.render()
            mm = np.where(
                np.isfinite(metres) & (metres < depth["max_range"]),
                metres * 1000.0,
                0.0,
            ).astype(np.uint16)
            dw, dh = depth["size"]
            server.publish(TOPIC_DEPTH, image(seq, mm.tobytes(), "16UC1", dw, dh))
            server.publish(TOPIC_CAMERA_INFO, camera_info(seq, dw, dh, depth["fovy"]))

        if renderer is not None:
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            try:
                import cv2

                ok, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if ok:
                    server.publish(TOPIC_CAMERA, compressed_image(seq, buf.tobytes()))
            except Exception as exc:
                print(f"camera encode failed: {exc}", file=sys.stderr)

    return step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", help=f"one of: {', '.join(sorted(ROBOTS))}")
    ap.add_argument("--scene", required=True, help="house MJCF")
    ap.add_argument("--render", default=None, help="write a PNG instead of opening the viewer")
    ap.add_argument("--pos", type=float, nargs=2, default=None,
                    help="override xy placement; an arm keeps the height of the surface "
                         "it is mounted on")
    ap.add_argument("--yaw", type=float, default=None, help="override facing, in degrees")
    ap.add_argument("--reach", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="arm working annulus on the table, in metres (default per robot)")
    ap.add_argument("--gripper", choices=("rest", "open", "closed"), default="rest",
                    help="override the gripper's rest pose, for looking at what it can reach")
    ap.add_argument("--spawn-objects", type=int, default=3, dest="spawn_objects",
                    metavar="N",
                    help="objects to put on the table if the scene has none; 0 to fail instead")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--distance", type=float, default=None,
                    help="camera distance (default: scaled to the robot size)")
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=-15.0)
    ap.add_argument("--timeout", type=float, default=None, help="auto-close the viewer after N s")
    ap.add_argument(
        "--control",
        default=None,
        metavar="HOST:PORT",
        help="drive the robot from an external control server (see bridge/server.py)",
    )
    ap.add_argument(
        "--control-hz", type=float, default=20.0, dest="control_hz", help="policy rate"
    )
    ap.add_argument(
        "--camera",
        default=None,
        help="MJCF camera to stream with --control; defaults to the robot's front_camera "
        "if it has one. Use 'none' to disable.",
    )
    ap.add_argument("--camera-size", type=int, nargs=2, default=[640, 480], dest="camera_size",
                    metavar=("W", "H"))
    ap.add_argument("--jpeg-quality", type=int, default=70, dest="jpeg_quality")
    ap.add_argument(
        "--connect-timeout", type=float, default=180.0, dest="connect_timeout",
        help="seconds to wait for the control server before giving up",
    )
    ap.add_argument(
        "--ros-port", type=int, default=None, dest="ros_port",
        metavar="PORT",
        help="present the robot on the myagv_ros topics via rosbridge (usually 9090). "
        "Mutually exclusive with --control.",
    )
    ap.add_argument(
        "--watchdog", type=float, default=0.5,
        help="stop the base if no cmd_vel arrives for this many seconds",
    )
    # The lidar defaults are the YDLidar X2's, from ydlidar_ros_driver/launch/X2.launch
    # and the base_footprint -> laser_frame transform in myagv_active.launch.
    ap.add_argument("--scan-beams", type=int, default=360, dest="scan_beams",
                    help="rays in the simulated lidar's 360 deg sweep, published on /scan. "
                         "The X2's 3 kHz sample rate at 10 Hz gives ~300; 360 is close and "
                         "keeps one beam per degree")
    ap.add_argument("--scan-range", type=float, default=12.0, dest="scan_range",
                    metavar="M", help="lidar maximum range in metres (X2: 12.0)")
    ap.add_argument("--scan-min-range", type=float, default=0.1, dest="scan_min_range",
                    metavar="M", help="lidar minimum range in metres (X2: 0.1)")
    ap.add_argument("--scan-offset", type=float, nargs=2, default=[0.065, 0.08],
                    dest="scan_offset", metavar=("X", "Z"),
                    help="laser origin ahead of and above base_footprint, matching the "
                         "static transform in myagv_active.launch")
    ap.add_argument("--scan-hz", type=float, default=10.0, dest="scan_hz",
                    help="scan rate; the X2 spins at a fixed 10 Hz independent of the "
                         "control rate")
    ap.add_argument("--no-scan", action="store_true", dest="no_scan",
                    help="do not publish /scan")
    ap.add_argument("--depth-hz", type=float, default=5.0, dest="depth_hz",
                    help="rate for the depth image; 640x480 float over a websocket is "
                         "1.2 MB a frame, so this is deliberately slower than the control rate")
    ap.add_argument("--depth-size", type=int, nargs=2, default=[320, 240], dest="depth_size",
                    metavar=("W", "H"))
    ap.add_argument("--depth-range", type=float, default=8.0, dest="depth_range",
                    metavar="M", help="depth beyond this reads as 'no return' (0 in 16UC1). "
                                      "Its own flag, not the lidar's: the two sensors have "
                                      "nothing to do with each other")
    ap.add_argument("--no-depth", action="store_true", dest="no_depth",
                    help="do not publish the depth image or camera info")
    args = ap.parse_args()

    from tools.scene_placement import apply_init_qpos, describe, find_robot_placement

    config_cls, robot_cls = load_robot(args.robot)
    config = config_cls()

    holonomic = args.robot in HOLONOMIC_BASE_ROBOTS
    tabletop = args.robot in TABLETOP_ROBOTS

    # Compile the bare scene once: both the surface search and the floor search want a
    # model, and neither should pay for its own compile.
    spec = mujoco.MjSpec.from_file(args.scene)
    scene_model = spec.compile()
    scene_data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, scene_data)

    if tabletop:
        reach = tuple(args.reach) if args.reach else ARM_REACH[args.robot]
        mount, to_spawn = place_arm_on_table(
            args.scene, scene_model, scene_data, args.robot, reach, args.spawn_objects
        )
        if to_spawn is not None:
            add_tabletop_objects(spec, *to_spawn)
        pos = np.array(args.pos) if args.pos is not None else mount.xy
        yaw = np.radians(args.yaw) if args.yaw is not None else mount.yaw
        print(describe(mount), file=sys.stderr)
        if args.pos is not None or args.yaw is not None:
            # The surface is still what was chosen -- and what supplies the height -- so say
            # what actually got used rather than leaving the line above to be read as final.
            print(
                f"overridden to ({pos[0]:.2f}, {pos[1]:.2f}, {mount.z:.2f}) "
                f"yaw {np.degrees(yaw):.0f} deg",
                file=sys.stderr,
            )
        # The pedestal was there to lift a tabletop arm off the floor. Standing on the
        # table, it would only bury the arm in the surface.
        config.base_size = list(TABLETOP_RISER)
        attach_pos = [float(pos[0]), float(pos[1]), float(mount.z)]
    else:
        placement = find_robot_placement(
            args.scene,
            model=scene_model,
            data=scene_data,
            pos=args.pos,
            yaw_deg=args.yaw,
            prefer="floor",
        )
        pos, yaw = placement.pos, placement.yaw
        print(describe(placement), file=sys.stderr)
        attach_pos = [float(pos[0]), float(pos[1]), 0.0]

    quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]

    robot_cls.add_robot_to_scene(
        config,
        spec,
        prefix=config.robot_namespace,
        # A holonomic base has to be grafted in at the origin; it is driven to its
        # spawn pose below instead.
        pos=[0.0, 0.0, 0.0] if holonomic else attach_pos,
        quat=[1.0, 0.0, 0.0, 0.0] if holonomic else quat,
    )
    model = spec.compile()
    data = mujoco.MjData(model)

    # Set the rest pose before anything is stepped. Both the joint state and the
    # actuator targets have to be set: these are position actuators, so leaving ctrl at
    # its default of 0 would make the robot immediately drive out of the pose.
    view = config.robot_view_factory(data, config.robot_namespace)
    apply_init_qpos(view, config)

    if args.gripper != "rest" and "gripper" in view.move_group_ids():
        gripper = view.get_move_group("gripper")
        gripper.set_gripper_ctrl_open(args.gripper == "open")
        # ctrl alone only shows up once the sim runs; the joint has to be moved too for a
        # still render to show anything.
        gripper.joint_pos = gripper.ctrl

    if holonomic:
        base_pose = np.eye(4)
        base_pose[:2, 3] = pos[:2]
        base_pose[:2, :2] = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        )
        base = view.get_move_group("base")
        base.pose = base_pose
        # Hold there rather than driving back to the origin on the first step.
        base.ctrl = np.array([pos[0], pos[1], yaw], dtype=np.float64)

    mujoco.mj_forward(model, data)
    warn_on_penetration(model, data, config.robot_namespace)

    ns = config.robot_namespace
    # Robots on a mocap pedestal have a "mount" body; free-standing and holonomic ones
    # are their own root.
    anchor = "mount"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{ns}{anchor}") < 0:
        anchor = robot_cls.robot_model_root_name()
    anchor_pos = data.body(f"{ns}{anchor}").xpos.copy()
    print(
        f"{args.robot} in scene: {model.nbody} bodies, {model.ngeom} geoms, "
        f"{model.nu} actuators; at {np.round(anchor_pos, 3)}",
        file=sys.stderr,
    )
    # Frame the camera on the robot's own geometry rather than a fixed height, so a
    # 13 cm AGV and a 1 m arm on a pedestal both fill the view.
    robot_geoms = np.array(
        [
            data.geom_xpos[gid]
            for gid in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or "").startswith(ns)
        ]
    )
    if len(robot_geoms):
        lookat = (robot_geoms.min(axis=0) + robot_geoms.max(axis=0)) / 2
        robot_radius = max(float(np.linalg.norm(robot_geoms.max(axis=0) - robot_geoms.min(axis=0))) / 2, 0.2)
    else:
        lookat = anchor_pos + np.array([0.0, 0.0, 0.5])
        robot_radius = 0.5
    distance = args.distance if args.distance is not None else robot_radius * 3.5

    # Look at the robot from the side it is facing, so it is not hidden behind furniture.
    azimuth = args.azimuth if args.azimuth is not None else np.degrees(yaw) + 180.0

    if args.render:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        for gid in range(model.ngeom):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            if "ceiling" in name:
                model.geom_rgba[gid, 3] = 0.0
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = args.elevation
        with mujoco.Renderer(model, args.height, args.width) as renderer:
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
        from PIL import Image

        Image.fromarray(pixels).save(args.render)
        print(f"wrote {args.render}", file=sys.stderr)
        return 0

    camera = args.camera
    if camera is None:
        # Default to the robot's own forward camera when it has one.
        default_cam = f"{ns}front_camera"
        camera = default_cam if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, default_cam) >= 0 else None
    elif camera.lower() == "none":
        camera = None

    if args.ros_port and args.control:
        raise SystemExit("use either --ros-port or --control, not both")

    if args.ros_port:
        scan_cfg = None
        if not args.no_scan:
            scan_cfg = {
                "beams": args.scan_beams,
                "max_range": args.scan_range,
                "min_range": args.scan_min_range,
                "offset_x": args.scan_offset[0],
                "offset_z": args.scan_offset[1],
                "period": 1.0 / max(args.scan_hz, 1e-3),
                # Rays start at the robot's own root body and must not range it.
                "body": f"{ns}{robot_cls.robot_model_root_name()}",
            }
        depth_cfg = None
        if not args.no_depth and camera is not None:
            fovy = float(model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)])
            depth_cfg = {
                "size": args.depth_size,
                "period": 1.0 / max(args.depth_hz, 1e-3),
                "max_range": args.depth_range,
                "fovy": fovy,
            }
        controller = serve_ros(
            args.ros_port, view, model, camera, args.camera_size, args.jpeg_quality,
            args.control_hz, args.watchdog, scan=scan_cfg, depth=depth_cfg,
        )
    elif args.control:
        controller = connect_control(
            args.control, view, model, camera, args.camera_size, args.jpeg_quality,
            args.connect_timeout,
        )
    else:
        controller = None
    control_period = 1.0 / args.control_hz
    next_control = 0.0

    # Bound as a separate name: `import mujoco.viewer` here would make `mujoco` a
    # function-local and shadow the module-level import above.
    from mujoco import viewer as mj_viewer

    deadline = None if args.timeout is None else time.monotonic() + args.timeout
    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = distance
        viewer.cam.azimuth = azimuth
        viewer.cam.elevation = args.elevation
        while viewer.is_running():
            step_start = time.time()
            now = time.monotonic()
            if controller is not None and now >= next_control:
                controller(data)
                next_control = now + control_period
            mujoco.mj_step(model, data)
            viewer.sync()
            if deadline is not None and time.monotonic() > deadline:
                break
            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)
    if controller is not None:
        controller(None)  # close
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
