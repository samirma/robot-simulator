#!/usr/bin/env python3
"""Spawn a shared robot into a CoppeliaSim scene and present the real robot's interface.

CoppeliaSim is a very different engine from the MuJoCo ones, but the whole point of this
workspace is that `robot_console` cannot tell them apart: every engine sources pose and
sensors its own way and then feeds the *same* shared wire bridge (`contracts.*`), which
speaks the real hardware's contract -- the myAGV's vendor ROS topics over rosbridge, and
the SO-101's msgpack-numpy control protocol. So this file is the CoppeliaSim half of that
adapter: it drives the CoppeliaSim ZMQ remote API and hands what it reads to the shared
servers, exactly as `robocasa/tools/spawn_robot.py` does for a MuJoCo model.

    ./run.sh view --robot myagv --scene room:6 --ros-port 9090
    ./run.sh view --robot so101 --scene empty --control 127.0.0.1:8000

`--scene`:
    room[:<size_m>]   a square walled room of the given inner size (default 6 m); the
                      myAGV's laser sees these walls. Always available.
    empty             no walls (an so101 bench arm does not need any).

CoppeliaSim must be launched into the macOS GUI (Aqua) session -- a plain `exec` of the
binary from a non-GUI shell exits immediately because it cannot reach the WindowServer, so
this launches it with `open`, which routes it through launchd's GUI session. That is also
why there is no true headless mode here; `--headless` is best effort.

Design notes that look odd without the constraint behind them:

- The myAGV base is driven *kinematically* (integrate `/cmd_vel` into a world pose and
  `setObjectPose`), matching the idealized holonomic base the MuJoCo engines use -- the
  real myAGV is Mecanum, so `linear.y` is a genuine strafe. Odom twist reports the
  commanded velocity (post-watchdog), the same choice the MuJoCo `serve_ros` makes.
- The laser scan is computed analytically against the room walls rather than from a ring
  of proximity sensors: it is deterministic, cheap over the remote API, and reproduces the
  YDLidar X2 contract exactly (360 beams CCW from -pi, 0.1-12 m, misses as range_max+1).
- so101 joints are imported from the shared URDF, then switched to *kinematic* mode so a
  commanded joint position is tracked exactly -- a position-controlled arm, which is what
  the control contract assumes.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
_SHARED = SIM_ROOT.parent / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

# name -> role. so101 is a fixed bench arm; myagv is a holonomic mobile base.
ROBOTS = {
    "so101": {"mobile": False},
    "myagv": {"mobile": True},
}

# The YDLidar X2 mount and limits, the same numbers the MuJoCo engines and the real
# myagv_active.launch use: 65 mm forward, 80 mm up, 0.1-12 m, 10 Hz.
SCAN_DEFAULTS = {"offset_x": 0.065, "offset_z": 0.08, "min_range": 0.1, "max_range": 12.0}

# The SO-101 arm joints in kinematic order, plus its single-DOF gripper. These are the
# shared URDF's own link/joint names, so the control metadata matches the MuJoCo engines.
SO101_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SO101_GRIPPER = ["gripper"]


# --------------------------------------------------------------------- launch / connect

def _coppelia_pids() -> set[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "coppeliaSim -G"], capture_output=True, text=True
        ).stdout
    except Exception:  # noqa: BLE001
        return set()
    return {int(p) for p in out.split() if p.strip().isdigit()}


def launch_coppelia(app: str, port: int, headless: bool) -> int | None:
    """Launch CoppeliaSim into the GUI session with `open`; return its new pid if found."""
    before = _coppelia_pids()
    args = ["open", "-n", app, "--args", f"-GzmqRemoteApi.rpcPort={port}"]
    if headless:
        # Best effort: -h needs a WindowServer and will exit on a pure headless host.
        args.insert(args.index("--args") + 1, "-h")
    subprocess.run(args, check=True)
    for _ in range(60):
        new = _coppelia_pids() - before
        if new:
            return sorted(new)[0]
        time.sleep(0.5)
    return None


def connect(port: int, timeout: float = 60.0):
    """Wait for the ZMQ remote API server and return (client, sim)."""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            client = RemoteAPIClient(port=port)
            sim = client.require("sim")
            sim.getInt32Param(sim.intparam_program_version)
            return client, sim
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0)
    raise SystemExit(f"could not reach CoppeliaSim ZMQ server on {port}: {last!r}")


# --------------------------------------------------------------------- scene building

def _parse_scene(scene: str) -> tuple[str, float]:
    s = (scene or "room").strip()
    if s.startswith("empty"):
        return ("empty", 0.0)
    body = s.split(":", 1)[1] if ":" in s else ""
    size = float(body) if body else 6.0
    return ("room", size)


def build_room(sim, size: float) -> list[tuple[float, float, float, float]]:
    """Create four walls of a square room of inner side `size`; return wall segments.

    Segments are returned as (x1, y1, x2, y2) of the walls' inner faces in the world
    plane, which is all the analytic laser needs.
    """
    half = size / 2.0
    th, h = 0.05, 0.6

    def wall(sx, sy, x, y):
        s = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [sx, sy, h], 0)
        sim.setObjectPosition(s, -1, [x, y, h / 2.0])
        sim.setObjectInt32Param(s, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(s, sim.shapeintparam_respondable, 1)
        sim.setShapeColor(s, None, sim.colorcomponent_ambient_diffuse, [0.8, 0.8, 0.85])

    wall(size + th, th, 0.0, half)
    wall(size + th, th, 0.0, -half)
    wall(th, size + th, half, 0.0)
    wall(th, size + th, -half, 0.0)
    # Inner faces as 2D segments.
    return [
        (-half, half, half, half),
        (-half, -half, half, -half),
        (half, -half, half, half),
        (-half, -half, -half, half),
    ]


def _add_camera(sim, parent: int, pos, ori, width: int, height: int, fovy_deg: float):
    """Create an explicit-handling RGB vision sensor as a child of `parent`."""
    intp = [width, height, 0, 0]
    floatp = [0.01, 20.0, math.radians(fovy_deg), 0.1, 0.1, 0.1, 0, 0, 0, 0, 0]
    vs = sim.createVisionSensor(1, intp, floatp)  # bit0 = explicit handling
    sim.setObjectParent(vs, parent, True)
    sim.setObjectPosition(vs, parent, list(pos))
    sim.setObjectOrientation(vs, parent, list(ori))
    return vs


def _grab_camera_jpeg(sim, vs: int, jpeg_quality: int):
    """Handle the sensor and return (jpeg_bytes, width, height) or None."""
    try:
        import cv2

        sim.handleVisionSensor(vs)
        buf, res = sim.getVisionSensorImg(vs)
        w, h = int(res[0]), int(res[1])
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        arr = np.flipud(arr)  # CoppeliaSim images are bottom-up (OpenGL origin)
        ok, enc = cv2.imencode(
            ".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if ok:
            return bytes(enc), w, h
    except Exception:  # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------- analytic laser

def _ray_segment(ox, oy, dx, dy, seg, max_range):
    """Distance from (ox,oy) along unit (dx,dy) to segment seg, or None."""
    x1, y1, x2, y2 = seg
    ex, ey = x2 - x1, y2 - y1
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
    if t < 0 or t > max_range:
        return None
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
    if u < 0.0 or u > 1.0:
        return None
    return t


def scan_ranges(x, y, yaw, segments, beams, min_range, max_range):
    """A YDLidar-style scan from the laser origin against the room wall segments."""
    ranges = []
    for i in range(beams):
        ang = -math.pi + (2.0 * math.pi) * i / beams
        a = yaw + ang
        dx, dy = math.cos(a), math.sin(a)
        best = math.inf
        for seg in segments:
            d = _ray_segment(x, y, dx, dy, seg, max_range)
            if d is not None and d < best:
                best = d
        if min_range <= best <= max_range:
            ranges.append(best)
        else:
            ranges.append(max_range + 1.0)  # miss; see laser_scan() docstring
    return ranges


# --------------------------------------------------------------------- myagv ROS surface

def serve_ros(sim, port, scene_kind, scene_size, camera_size, jpeg_quality,
              control_hz, watchdog_s, scan_cfg):
    """Present the myAGV on its vendor ROS topics over the shared rosbridge."""
    from contracts.rosbridge_server import (
        TOPIC_CAMERA, TOPIC_CMD_VEL, TOPIC_ODOM, TOPIC_SCAN,
        RosBridgeServer, compressed_image, laser_scan, odometry,
    )

    width, height = camera_size

    base = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.30, 0.30, 0.20], 0)
    sim.setObjectAlias(base, "myagv_base")
    sim.setObjectPosition(base, -1, [0.0, 0.0, 0.10])
    sim.setObjectInt32Param(base, sim.shapeintparam_static, 1)
    sim.setShapeColor(base, None, sim.colorcomponent_ambient_diffuse, [0.1, 0.3, 0.8])
    cam = _add_camera(
        sim, base, [0.16, 0.0, 0.06], [-math.pi / 2, 0.0, -math.pi / 2],
        width, height, 60.0,
    )

    segments = []
    if scene_kind == "room":
        segments = build_room(sim, scene_size)

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

    published = [TOPIC_ODOM, TOPIC_CAMERA]
    if scan_cfg:
        published.append(TOPIC_SCAN)
    print(
        f"ROS topics on ws://0.0.0.0:{port} "
        f"(sub {TOPIC_CMD_VEL}; pub {', '.join(published)})",
        file=sys.stderr,
    )

    sim.startSimulation()
    dt = 1.0 / control_hz
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    last_scan = {"t": 0.0}

    def step(alive: bool):
        if not alive:
            server.publish(TOPIC_ODOM, odometry(server.next_seq(), pose["x"], pose["y"],
                                                pose["yaw"], 0.0, 0.0, 0.0))
            server.stop()
            sim.stopSimulation()
            return

        vx, vy, wz = command["vx"], command["vy"], command["wz"]
        # The real myAGV latches the last cmd_vel forever; the sim stops on silence, which
        # is what a disconnecting client wants. See CLAUDE.md on the watchdog.
        if command["at"] and time.monotonic() - command["at"] > watchdog_s:
            vx = vy = wz = 0.0

        # Integrate the body-frame command into a world pose (holonomic; +x fwd, +y left).
        pose["yaw"] += wz * dt
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        pose["x"] += (vx * c - vy * s) * dt
        pose["y"] += (vx * s + vy * c) * dt
        sim.setObjectPose(
            base, -1,
            [pose["x"], pose["y"], 0.10,
             0.0, 0.0, math.sin(pose["yaw"] / 2), math.cos(pose["yaw"] / 2)],
        )
        sim.step()

        seq = server.next_seq()
        server.publish(TOPIC_ODOM, odometry(seq, pose["x"], pose["y"], pose["yaw"],
                                            vx, vy, wz))

        grabbed = _grab_camera_jpeg(sim, cam, jpeg_quality)
        if grabbed is not None:
            jpeg, _, _ = grabbed
            server.publish(TOPIC_CAMERA, compressed_image(seq, jpeg))

        if scan_cfg:
            now = time.monotonic()
            if now - last_scan["t"] >= scan_cfg["period"]:
                last_scan["t"] = now
                lx = pose["x"] + scan_cfg["offset_x"] * c
                ly = pose["y"] + scan_cfg["offset_x"] * s
                ranges = scan_ranges(
                    lx, ly, pose["yaw"], segments, scan_cfg["beams"],
                    scan_cfg["min_range"], scan_cfg["max_range"],
                )
                n = len(ranges)
                server.publish(TOPIC_SCAN, laser_scan(
                    seq, ranges, -math.pi, math.pi, 2 * math.pi / n,
                    range_min=scan_cfg["min_range"], range_max=scan_cfg["max_range"],
                    scan_time=scan_cfg["period"],
                ))

    return step


# --------------------------------------------------------------------- so101 control

def serve_control(sim, client, host, port, camera_size, jpeg_quality, robot_name):
    """Serve the generic msgpack-numpy control protocol for the SO-101 arm."""
    from contracts.control_server import ControlServer

    width, height = camera_size

    # Fix the arm to the world by making every shape it imports static, so nothing falls;
    # kinematic joints then track a commanded position exactly.
    urdf = client.require("simURDF")
    urdf_path = _SHARED / "robots" / robot_name / "urdf" / "so101_new_calib.urdf"

    def shapes() -> set[int]:
        out, i = set(), 0
        while True:
            h = sim.getObjects(i, sim.object_shape_type)
            if h == -1:
                break
            out.add(h)
            i += 1
        return out

    before = shapes()
    try:
        urdf.importFile(str(urdf_path))
    except Exception:  # noqa: BLE001 -- import succeeds but the wrapper reports a stale handle
        pass
    for h in shapes() - before:
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)

    groups = {"arm": SO101_ARM, "gripper": SO101_GRIPPER}
    handles = {}
    for js in groups.values():
        for name in js:
            h = sim.getObject("/" + name)
            sim.setJointMode(h, sim.jointmode_kinematic, 0)
            handles[name] = h

    # A wrist camera on the last arm link.
    wrist_cam = _add_camera(
        sim, handles["wrist_roll"], [0.0, 0.0, 0.05], [0.0, 0.0, 0.0],
        width, height, 55.0,
    )

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

    sim.startSimulation()
    targets = {name: sim.getJointPosition(h) for name, h in handles.items()}

    def step(alive: bool):
        if not alive:
            server.stop()
            sim.stopSimulation()
            return

        for name, h in handles.items():
            sim.setJointPosition(h, targets[name])
        sim.step()

        obs = {
            "qpos": {g: [sim.getJointPosition(handles[n]) for n in js]
                     for g, js in groups.items()},
            "qvel": {g: [sim.getJointVelocity(handles[n]) for n in js]
                     for g, js in groups.items()},
            "actions/joint_pos": {g: [targets[n] for n in js]
                                  for g, js in groups.items()},
        }
        grabbed = _grab_camera_jpeg(sim, wrist_cam, jpeg_quality)
        if grabbed is not None:
            jpeg, _, _ = grabbed
            obs["camera_jpeg"] = np.frombuffer(jpeg, dtype=np.uint8)

        action = server.publish(obs)
        if action:
            for g, target in action.items():
                if g in groups:
                    vals = np.asarray(target, dtype=np.float64).ravel()
                    for name, val in zip(groups[g], vals):
                        targets[name] = float(val)

    return step


# --------------------------------------------------------------------- run loop

def run(sim, step_cb, control_hz):
    """Step the sim at `control_hz`, publishing a final stop on any exit path."""
    def _raise(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise)
    signal.signal(signal.SIGINT, _raise)

    period = 1.0 / control_hz
    try:
        while True:
            t0 = time.monotonic()
            step_cb(True)
            slack = period - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            step_cb(False)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS))
    ap.add_argument("--scene", default="room", help="room[:<size_m>] or 'empty'")
    ap.add_argument("--headless", action="store_true",
                    help="best effort; CoppeliaSim needs a GUI session on macOS")
    ap.add_argument("--gui", action="store_true",
                    help="force the GUI window (the default; accepted for symmetry)")
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
    ap.add_argument("--zmq-port", type=int,
                    default=int(os.environ.get("COPSIM_ZMQ_PORT", "23000")),
                    dest="zmq_port")
    ap.add_argument("--app", default=os.environ.get("COPSIM_APP", ""),
                    help="path to CoppeliaSim.app (defaults to $COPSIM_APP)")
    ap.add_argument("--attach", action="store_true",
                    help="connect to an already-running CoppeliaSim instead of launching")
    args = ap.parse_args()

    if args.ros_port is not None and args.control is not None:
        raise SystemExit("--ros-port and --control are mutually exclusive")
    info = ROBOTS[args.robot]

    app_pid = None
    if not args.attach:
        if not args.app:
            raise SystemExit("set $COPSIM_APP or pass --app /path/to/CoppeliaSim.app")
        app_pid = launch_coppelia(args.app, args.zmq_port, args.headless)
        print(f"launched CoppeliaSim (pid {app_pid}) on ZMQ {args.zmq_port}",
              file=sys.stderr)

    client, sim = connect(args.zmq_port)
    sim.setStepping(True)
    scene_kind, scene_size = _parse_scene(args.scene)

    try:
        if args.ros_port is not None:
            if not info["mobile"]:
                raise SystemExit(f"--ros-port needs a mobile base; {args.robot} has none")
            scan_cfg = None
            if not args.no_scan:
                scan_cfg = {
                    "beams": args.scan_beams,
                    "min_range": SCAN_DEFAULTS["min_range"],
                    "max_range": args.scan_range or SCAN_DEFAULTS["max_range"],
                    "offset_x": SCAN_DEFAULTS["offset_x"],
                    "offset_z": SCAN_DEFAULTS["offset_z"],
                    "period": 0.1,
                }
            step_cb = serve_ros(
                sim, args.ros_port, scene_kind, scene_size, args.camera_size,
                args.jpeg_quality, args.control_hz, args.watchdog, scan_cfg,
            )
        elif args.control is not None:
            host, _, port_s = args.control.rpartition(":")
            if not host or not port_s:
                raise SystemExit("--control expects HOST:PORT, e.g. 127.0.0.1:8000")
            step_cb = serve_control(
                sim, client, host, int(port_s), args.camera_size, args.jpeg_quality,
                args.robot,
            )
        else:
            if scene_kind == "room":
                build_room(sim, scene_size)
            sim.startSimulation()

            def step_cb(alive: bool):
                if alive:
                    sim.step()
                else:
                    sim.stopSimulation()

        run(sim, step_cb, args.control_hz)
    finally:
        if app_pid is not None:
            try:
                os.kill(app_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
