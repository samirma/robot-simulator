#!/usr/bin/env python3
"""Spawn a shared robot into a RoboCasa kitchen and present the real robot's interface.

RoboCasa is robosuite's kitchen suite: `robosuite.make(...)` builds a whole task with a
robosuite-native robot, controllers and a reward. We want none of that -- we want the
*kitchen* as a MuJoCo scene, one of `simulator/shared`'s robots dropped into it, and the
exact same wire contract the console already speaks to MolmoSpaces and to the real robot.

So this bypasses the RL env: it builds the kitchen with RoboCasa's own arena + fixture
merger (`ManipulationTask` with an empty robot list), takes the composed MJCF, and grafts
the shared robot in with `mujoco.MjSpec.attach`. From there it is a plain MuJoCo step
loop feeding the shared bridge -- identical in every observable way to the MolmoSpaces
path, which is the whole point: `robot_console` cannot tell the engines apart.

    ./run.sh view --robot myagv --scene kitchen:1 --ros-port 9090
    ./run.sh view --robot so101 --scene kitchen:3/5 --control 127.0.0.1:8000
    ./run.sh view --robot myagv --scene empty --ros-port 9090 --headless

`--scene`:
    kitchen:<layout>[/<style>]   a RoboCasa kitchen (layouts 1-60, styles 1-60)
    empty                        the bare kitchen room shell (no fixtures; always loads)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
_SHARED = SIM_ROOT.parent / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

# name -> (root body, camera, is a holonomic mobile base). These are the shared model's
# own names; MjSpec.attach prefixes them, so the runtime looks them up as PREFIX+name.
ROBOTS = {
    "so101": {"root": "base", "camera": "exo_camera", "mobile": False},
    "myagv": {"root": "base", "camera": "front_camera", "mobile": True},
}
PREFIX = "robot_"

# The YDLidar X2 mount and limits, same numbers the MolmoSpaces engine and the real
# myagv_active.launch use: 65 mm forward, 80 mm up, 0.1-12 m.
SCAN_DEFAULTS = {"offset_x": 0.065, "offset_z": 0.08, "min_range": 0.1, "max_range": 12.0}

# so101 is a bench arm with a fixed base; float it at counter height so it is not buried
# in the floor. myagv drives on the floor from the attach frame.
SO101_MOUNT_Z = 0.90


# --------------------------------------------------------------------- world building

def _parse_scene(scene: str) -> tuple[str, int, int]:
    """('kitchen', layout, style) or ('empty', 0, 0)."""
    s = (scene or "kitchen:1").strip()
    if s.startswith("empty"):
        return ("empty", 0, 0)
    body = s.split(":", 1)[1] if ":" in s else s
    layout_s, _, style_s = body.partition("/")
    layout = int(layout_s) if layout_s else 1
    style = int(style_s) if style_s else 1
    return ("kitchen", layout, style)


def build_world(scene: str, robot: str, seed: int) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compose the kitchen and the shared robot into one compiled MuJoCo model."""
    import robocasa
    from robosuite.utils.mjcf_utils import xml_path_completion

    kind, layout, style = _parse_scene(scene)

    if kind == "empty":
        empty = xml_path_completion(
            "arenas/empty_kitchen_arena.xml", root=robocasa.models.assets_root
        )
        kspec = mujoco.MjSpec.from_file(empty)
        print(f"loaded empty kitchen room shell", file=sys.stderr)
    else:
        # RoboCasa's own arena + fixture merger. An empty robot list gives us the kitchen
        # without a robosuite robot; we attach our own below.
        from robocasa.models.scenes.kitchen_arena import KitchenArena
        from robosuite.models.tasks import ManipulationTask

        rng = np.random.default_rng(seed)
        arena = KitchenArena(layout_id=layout, style_id=style, rng=rng)
        arena.set_origin([0, 0, 0])
        fixtures = {c["name"]: c["model"] for c in arena.get_fixture_cfgs()}
        task = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=[], mujoco_objects=list(fixtures.values())
        )
        kspec = mujoco.MjSpec.from_string(task.get_xml())
        print(
            f"built RoboCasa kitchen layout={layout} style={style} "
            f"({len(fixtures)} fixtures)",
            file=sys.stderr,
        )

    spec_dir = _SHARED / "robots" / robot
    rspec = mujoco.MjSpec.from_file(str(spec_dir / "model.xml"))

    frame = kspec.worldbody.add_frame()
    frame.pos = [0.0, 0.0, SO101_MOUNT_Z if robot == "so101" else 0.0]
    kspec.attach(rspec, prefix=PREFIX, frame=frame)

    model = kspec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# --------------------------------------------------------------------- name lookups

def _jnt_qposadr(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise SystemExit(f"joint {name!r} not found in composed model")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _act_id(model, name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise SystemExit(f"actuator {name!r} not found in composed model")
    return int(aid)


# --------------------------------------------------------------------- myagv ROS surface

def serve_ros(port: int, model, data, camera: str, camera_size, jpeg_quality: int,
              control_hz: float, watchdog_s: float, scan: dict | None):
    """Present the myAGV on its vendor ROS topics over the shared rosbridge.

    Same contract as MolmoSpaces `robots/myagv/ros_surface.py`, but sourcing pose and
    sensors from a raw mujoco model/data rather than a MolmoSpaces RobotView -- the wire
    side is the shared code, so the console is identical.
    """
    from contracts.rosbridge_server import (
        TOPIC_CAMERA, TOPIC_CAMERA_INFO, TOPIC_CMD_VEL, TOPIC_DEPTH,
        TOPIC_ODOM, TOPIC_SCAN, RosBridgeServer, odometry,
    )
    from mujoco_bridge import PlanarSetpoint, SensorStreams, SensorTopics

    x_q, _ = _jnt_qposadr(model, PREFIX + "base_x")
    y_q, _ = _jnt_qposadr(model, PREFIX + "base_y")
    t_q, _ = _jnt_qposadr(model, PREFIX + "base_theta")
    x_a = _act_id(model, PREFIX + "base_x_act")
    y_a = _act_id(model, PREFIX + "base_y_act")
    t_a = _act_id(model, PREFIX + "base_theta_act")

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

    sensors = SensorStreams(
        server, model, camera, camera_size, jpeg_quality, scan, None,
        SensorTopics(TOPIC_CAMERA, TOPIC_SCAN, TOPIC_DEPTH, TOPIC_CAMERA_INFO),
    )
    setpoint = PlanarSetpoint()

    server.start()
    print(
        f"ROS topics on ws://0.0.0.0:{port} "
        f"(sub {TOPIC_CMD_VEL}; pub {', '.join([TOPIC_ODOM, *sensors.published])})",
        file=sys.stderr,
    )

    dt = 1.0 / control_hz

    def step(d):
        if d is None:
            sensors.close()
            server.stop()
            return

        vx, vy, wz = command["vx"], command["vy"], command["wz"]
        # The real myAGV latches the last cmd_vel forever; the sim stops on silence, which
        # is the behaviour a disconnecting client wants. See CLAUDE.md on the watchdog.
        if command["at"] and time.monotonic() - command["at"] > watchdog_s:
            vx = vy = wz = 0.0

        x = float(d.qpos[x_q])
        y = float(d.qpos[y_q])
        yaw = float(d.qpos[t_q])

        target = setpoint.step(x, y, yaw, vx, vy, wz, dt)
        d.ctrl[x_a], d.ctrl[y_a], d.ctrl[t_a] = target

        seq = server.next_seq()
        server.publish(TOPIC_ODOM, odometry(seq, x, y, yaw, vx, vy, wz))
        sensors.publish(d, seq, x, y, yaw)

    return step


# --------------------------------------------------------------------- so101 control

def serve_control(host: str, port: int, model, data, camera: str, camera_size,
                  jpeg_quality: int, robot_name: str):
    """Serve the generic msgpack-numpy control protocol for an arm.

    Same observation/action shape as MolmoSpaces `serve_control` (protocol
    `molmospaces-control-v1`), so `robot_console`'s arm client and SO-101 driver drive a
    RoboCasa-hosted arm without changing a line. Move groups are `arm` (5 joints) and
    `gripper` (1), matching the shared so101 model.
    """
    from contracts.control_server import ControlServer

    arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    groups = {"arm": arm_joints, "gripper": ["gripper"]}
    qadr = {g: [_jnt_qposadr(model, PREFIX + j) for j in js] for g, js in groups.items()}
    acts = {g: [_act_id(model, PREFIX + j) for j in js] for g, js in groups.items()}

    server = ControlServer(
        host, port,
        metadata={
            "model_name": robot_name,
            "protocol": "molmospaces-control-v1",
            "move_groups": {g: len(js) for g, js in groups.items()},
        },
    )
    server.start()
    print(f"control server listening on ws://{host}:{port}", file=sys.stderr)

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
                ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if ok:
                return {"camera_jpeg": np.asarray(buf).reshape(-1)}
        except Exception:
            pass
        return {"camera": frame}

    def step(d):
        if d is None:
            if renderer is not None:
                renderer.close()
            server.stop()
            return

        obs = {
            "qpos": {g: [float(d.qpos[q]) for q, _ in qadr[g]] for g in groups},
            "qvel": {g: [float(d.qvel[v]) for _, v in qadr[g]] for g in groups},
            "actions/joint_pos": {g: [float(d.ctrl[a]) for a in acts[g]] for g in groups},
        }
        if renderer is not None:
            renderer.update_scene(d, camera=camera)
            obs.update(encode(renderer.render()))

        action = server.publish(obs)
        if action is None:
            return
        for g, target in action.items():
            if g in acts:
                for a, val in zip(acts[g], np.asarray(target, dtype=np.float64).ravel()):
                    d.ctrl[a] = float(val)

    return step


# --------------------------------------------------------------------- run loop

def run(model, data, step_cb, headless: bool, camera: str | None, control_hz: float):
    """Step the sim, calling step_cb(data) at `control_hz`. Publishes a final stop.

    The sim integrates at `model.opt.timestep`; the wire surface only needs the control
    rate (20 Hz, matching the console's publish rate and MolmoSpaces). So each control
    tick advances several physics substeps, exactly as a real 500 Hz motor loop sits under
    a 20 Hz command stream -- streaming odom/camera every physics step would flood a JSON
    websocket. /scan is self-clocked to 10 Hz inside SensorStreams regardless.

    The real myAGV has no watchdog and latches the last command, so the wire surface must
    get its final zero on *every* exit path -- Ctrl-C, `kill`, or the viewer closing. We
    translate SIGINT/SIGTERM into KeyboardInterrupt so the one `finally` covers them all.
    """
    import signal

    timestep = model.opt.timestep
    period = 1.0 / control_hz
    substeps = max(1, round(period / timestep))

    def _raise(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise)
    signal.signal(signal.SIGINT, _raise)

    def tick():
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        step_cb(data)

    if headless:
        try:
            while True:
                t0 = time.monotonic()
                tick()
                slack = period - (time.monotonic() - t0)
                if slack > 0:
                    time.sleep(slack)
        except KeyboardInterrupt:
            pass
        finally:
            step_cb(None)
        return

    import mujoco.viewer as _mjviewer

    try:
        with _mjviewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.monotonic()
                tick()
                viewer.sync()
                slack = period - (time.monotonic() - t0)
                if slack > 0:
                    time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        step_cb(None)


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--scene", default="kitchen:1",
                    help="kitchen:<layout>[/<style>] or 'empty'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true",
                    help="run the sim + servers with no viewer window")
    ap.add_argument("--ros-port", type=int, default=None, dest="ros_port",
                    help="present the myAGV ROS topics on this port")
    ap.add_argument("--control", default=None,
                    help="present the arm control server on HOST:PORT")
    ap.add_argument("--control-hz", type=float, default=20.0, dest="control_hz")
    ap.add_argument("--watchdog", type=float, default=0.5)
    ap.add_argument("--camera-size", type=int, nargs=2, default=[640, 480],
                    dest="camera_size")
    ap.add_argument("--jpeg-quality", type=int, default=70, dest="jpeg_quality")
    ap.add_argument("--scan-beams", type=int, default=360, dest="scan_beams")
    ap.add_argument("--scan-range", type=float, default=None, dest="scan_range")
    ap.add_argument("--no-scan", action="store_true", dest="no_scan")
    args = ap.parse_args()

    info = ROBOTS[args.robot]
    model, data = build_world(args.scene, args.robot, args.seed)
    camera = PREFIX + info["camera"]

    if args.ros_port is not None and args.control is not None:
        raise SystemExit("--ros-port and --control are mutually exclusive")

    if args.ros_port is not None:
        if not info["mobile"]:
            raise SystemExit(f"--ros-port needs a mobile base; {args.robot} has none")
        scan = None
        if not args.no_scan:
            scan = {
                "beams": args.scan_beams,
                "max_range": args.scan_range or SCAN_DEFAULTS["max_range"],
                "min_range": SCAN_DEFAULTS["min_range"],
                "offset_x": SCAN_DEFAULTS["offset_x"],
                "offset_z": SCAN_DEFAULTS["offset_z"],
                "period": 0.1,
                "body": PREFIX + info["root"],
            }
        step_cb = serve_ros(
            args.ros_port, model, data, camera, args.camera_size, args.jpeg_quality,
            args.control_hz, args.watchdog, scan,
        )
    elif args.control is not None:
        host, _, port_s = args.control.rpartition(":")
        if not host or not port_s:
            raise SystemExit("--control expects HOST:PORT, e.g. 127.0.0.1:8000")
        step_cb = serve_control(
            host, int(port_s), model, data, camera, args.camera_size,
            args.jpeg_quality, args.robot,
        )
    else:
        # No wire surface: just drive the viewer so a user can look at the scene.
        def step_cb(d):
            return

    run(model, data, step_cb, args.headless, camera, args.control_hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
