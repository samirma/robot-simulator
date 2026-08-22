#!/usr/bin/env python
"""Render a robot-camera trajectory through a house, with ground-truth poses.

Footage for the visual-SLAM go/no-go gate. `robot_console`'s stella engine has to be
shown to initialize, track and *close a loop* on simulator imagery before any of the
protocol work around it is worth doing, and that check wants frames on disk rather than
a live rosbridge: a gate you can re-run against the same bytes after every change is the
only kind that can tell a binding regression from a scene that was always too textureless.

Runs headless under plain `python` -- `mujoco.Renderer` renders offscreen, so unlike
`tools/view_scene.py` there is no viewer and no `mjpython`.

    python tools/capture_trajectory.py --scene assets/scenes/procthor-10k-train/train_40.xml \
        --out /tmp/gate0

Pass the scene by its `assets/` path, not its realpath -- meshes resolve relatively
through the symlink tree (see CLAUDE.md).

## The trajectory is out-and-back, on purpose

A loop closure needs the robot to *return somewhere it has already been*, and needs the
return views to differ enough from the outbound ones to be a real re-recognition rather
than a replay of identical frames. Driving out along a heading and back along a parallel
line offset sideways gives both, in a way that needs no path planner and no per-scene
waypoints: the offset is what makes it a loop closure test instead of a frame-hash test.
Each leg is bracketed by an in-place rotation, which is what seeds enough features to
initialize monocular SLAM before any translation is asked of it.

Progress is watched rather than assumed. The base is position-controlled, so driving it
into a wall is a stall, not a crash -- the leg simply ends early and the return still
starts from wherever it actually got to.

## Frames

Ground-truth poses are written in **OpenCV camera axes** (+x right, +y down, +z forward),
because that is what the engine protocol and `robot_console/vslam/pointmap.py` use.
MuJoCo's camera looks down **-z** with +y up, so the recorded matrix carries a
`diag(1, -1, -1)` flip. Getting this backwards produces a trajectory that is a plausible
mirror of the truth and an ATE that looks like drift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# MuJoCo camera axes -> OpenCV camera axes. Applied on the right of the world<-camera
# rotation, so it re-expresses the camera's own axes without touching its position.
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])

# A stall is "commanded further than it went, repeatedly". One slow tick is noise --
# the base accelerates from rest, and a fresh command always lags for a moment. A second
# of grace at the default capture rate: by the time this fires the base really has met a
# wall, which is a legitimate end to a leg rather than a failure.
_STALL_TICKS = 15
_STALL_PROGRESS_M = 0.002

# How far ahead of the base's *actual* yaw a command may sit. Same value and same reason
# as `TARGET_LEAD_RAD` in tools/spawn_robot.py: the base is a position servo, so an
# integrated setpoint runs away from a base that geometry is holding up, and the base then
# lunges the moment it comes free. Unobstructed the clamp costs nothing -- measured on
# train_40, a commanded 0.6 rad/s is achieved at 0.588 rad/s for every lead from 0.15 to
# 2.5 rad -- so it is purely a guard for the blocked case.
TARGET_LEAD_RAD = 0.15

# A circuit that cannot make progress must still end. `travelled` accumulates the base's
# own movement, and a robot wedged in a corner jitters by a millimetre a tick -- enough to
# reach any distance eventually, at 15 frames a second of identical footage. Measured: a
# wedged 25 m circuit ran to 2224 s of sim time, 33k frames and 39 GB before it "finished".
_CIRCUIT_TIME_BUDGET = 6.0


def camera_intrinsics(model, cam_id: int, width: int, height: int) -> dict:
    """Pinhole intrinsics from a MuJoCo camera's vertical FOV.

    Square pixels, principal point at the geometric centre: `(width - 1) / 2`, matching
    `robot_console/vslam/app.py::_camera_model` rather than `width / 2`. Half a pixel of
    disagreement is not worth a class of bug that only shows up as a slow bias.
    """
    fovy = float(model.cam_fovy[cam_id])
    fy = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return {
        "fx": fy,
        "fy": fy,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "width": int(width),
        "height": int(height),
        "fovy_deg": fovy,
    }


def camera_pose_cv(data, cam_id: int) -> np.ndarray:
    """The camera's ground-truth cam->world 4x4, in OpenCV axes."""
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ _MJ_TO_CV
    T[:3, 3] = data.cam_xpos[cam_id]
    return T


def best_heading(model, data, ns: str, origin, *, robot_radius: float, max_range: float,
                 height: float):
    """Yaw of the widest clear corridor from `origin`, and how far it runs.

    `find_open_spot` picks a *spot* with room around it and faces the room centre, which
    is not the same as a direction with room to drive: in a furnished house the room
    centre is usually where the table is. Casting a scan and choosing a direction settles
    it from the geometry instead of from a heuristic.

    Scored on the *minimum* range across a cone as wide as the base, not on a single
    beam, so a gap between two chair legs does not read as a corridor.

    Cast at **camera** height, not base height. Scanning at the base's 8 cm finds
    corridors that run under kitchen islands and table tops -- clear for the chassis,
    a wall to the camera 6 cm higher. Measured: a direction the base-height scan called
    4.00 m clear left the camera 1.22 m from an overhang.
    """
    from tools.spawn_robot import laser_scan_ranges

    beams = 360
    ranges = laser_scan_ranges(
        model, data,
        np.array([origin[0], origin[1], height]),
        0.0, beams, max_range,
        exclude_bodies=frozenset(
            i for i in range(model.nbody)
            if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
            and n.startswith(ns)
        ),
    )
    ranges = np.where(ranges > max_range, max_range, ranges)
    # Half-width of the cone that must stay clear, in beams: the base has to fit through.
    half = max(1, int(round(math.degrees(math.atan2(robot_radius, 1.0)))))
    scores = np.array([
        ranges[np.arange(i - half, i + half + 1) % beams].min() for i in range(beams)
    ])
    best = int(np.argmax(scores))
    # Beams run counter-clockwise from -pi in the base frame; yaw 0 was passed above.
    return -math.pi + best * (2.0 * math.pi / beams), float(scores[best])


def build_scene(scene: str, robot: str):
    """Compile the house with the robot attached, placed on open floor.

    Mirrors `robots/myagv/test_attach.py` rather than `tools/spawn_robot.py`: this needs
    the holonomic-base placement and nothing else that script does around arms, grasps
    and viewer framing.
    """
    if robot != "myagv":
        raise SystemExit(f"--robot {robot!r}: only myagv is wired up here")
    from robots.myagv import MyAGVRobot, MyAGVRobotConfig, MyAGVRobotView
    from tools.spawn_robot import find_open_spot

    config = MyAGVRobotConfig()
    spec = mujoco.MjSpec.from_file(scene)
    MyAGVRobot.add_robot_to_scene(
        config, spec, prefix=config.robot_namespace,
        pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    view = MyAGVRobotView(data, config.robot_namespace)
    base = view.get_move_group("base")

    origin, yaw = find_open_spot(scene)
    start = np.eye(4)
    start[:2, 3] = origin
    start[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    base.pose = start
    base.ctrl = np.array([origin[0], origin[1], yaw], dtype=np.float64)
    mujoco.mj_forward(model, data)
    return model, data, config, base, np.array([origin[0], origin[1], yaw])


class Capture:
    """Steps the sim, renders on a fixed clock, and writes frames plus ground truth."""

    def __init__(self, model, data, base, cam_id: int, out: Path, *,
                 hz: float, size, jpeg_quality: int, want_depth: bool,
                 exclude_bodies=frozenset()) -> None:
        self.model, self.data, self.base, self.cam_id = model, data, base, cam_id
        self.exclude_bodies = exclude_bodies
        self.out = out
        self.period = 1.0 / hz
        self.jpeg_quality = jpeg_quality
        width, height = size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height, width)
        # A MuJoCo renderer is either in depth mode or it is not, and toggling it per
        # frame would fight the colour stream -- the same reason spawn_robot.py keeps two.
        self.depth_renderer = None
        if want_depth:
            self.depth_renderer = mujoco.Renderer(model, height, width)
            self.depth_renderer.enable_depth_rendering()
        self.frames_dir = out / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if want_depth:
            self.depth_dir = out / "depth"
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.poses = (out / "poses.jsonl").open("w")
        self.n = 0
        self._next_capture = 0.0

    def _grab(self) -> None:
        cam = self.cam_id
        self.renderer.update_scene(self.data, camera=cam)
        rgb = self.renderer.render()
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("jpeg encode failed")
        (self.frames_dir / f"{self.n:06d}.jpg").write_bytes(buf.tobytes())

        if self.depth_renderer is not None:
            self.depth_renderer.update_scene(self.data, camera=cam)
            np.save(self.depth_dir / f"{self.n:06d}.npy",
                    self.depth_renderer.render().astype(np.float32))

        pose = np.asarray(self.base.joint_pos, dtype=np.float64)
        self.poses.write(json.dumps({
            "i": self.n,
            "t": float(self.data.time),
            "T_wc": [float(v) for v in camera_pose_cv(self.data, cam).ravel()],
            "base": [float(pose[0]), float(pose[1]), float(pose[2])],
        }) + "\n")
        self.n += 1

    def camera_clearance(self, *, bearing: float = 0.0,
                         half_deg: float = 12.0, beams: int = 25) -> float:
        """Nearest surface in front of the *camera*, in metres.

        Measured from the camera, not the base, and at the camera's own height, because
        what ends a leg here is the camera going blind rather than the base colliding.
        The two differ by more than the 0.1356 m mount offset: a scan at base height
        passes under an overhang the camera is staring straight into, which is exactly
        how a run whose base never collided still spent 214 frames at a median scene
        depth of 0.23 m with no ORB features at all.
        """
        from tools.spawn_robot import laser_scan_ranges

        origin = self.data.cam_xpos[self.cam_id].copy()
        # MuJoCo cameras look down -z; the world-frame heading is that axis, flattened.
        forward = -self.data.cam_xmat[self.cam_id].reshape(3, 3)[:, 2]
        yaw = math.atan2(forward[1], forward[0]) + bearing
        half = math.radians(half_deg)
        ranges = laser_scan_ranges(
            self.model, self.data, origin, yaw, beams, 10.0,
            angle_min=-half, angle_max=half, exclude_bodies=self.exclude_bodies,
        )
        return float(np.min(np.where(ranges > 10.0, 10.0, ranges)))

    def run_until(self, t_end: float) -> None:
        """Step physics to `t_end` of sim time, capturing on the frame clock."""
        while self.data.time < t_end:
            mujoco.mj_step(self.model, self.data)
            if self.data.time >= self._next_capture:
                self._grab()
                self._next_capture += self.period

    def close(self) -> None:
        self.poses.close()
        self.renderer.close()
        if self.depth_renderer is not None:
            self.depth_renderer.close()


def drive(cap: Capture, target_xy, yaw: float, *, speed: float,
          standoff: float = 0.0) -> np.ndarray:
    """Advance the commanded pose toward `target_xy` at `speed`, capturing as it goes.

    Commands a moving setpoint rather than the endpoint: the base is position-controlled,
    so commanding the far end would make it accelerate to its limit and arrive with the
    inter-frame motion far too large for a feature tracker to follow.

    `standoff` ends the leg once the camera is that close to something, which is what the
    real console does -- `slam/controller.py`'s obstacle brake and `planner.py`'s
    robot-radius inflation both keep it off surfaces. Driving until the base physically
    stalls instead, as this did originally, parks the camera against a cabinet door and
    measures a tracking failure the robot would never actually have.
    """
    here = np.asarray(cap.base.joint_pos, dtype=np.float64)[:2].copy()
    target = np.asarray(target_xy, dtype=np.float64)
    total = float(np.linalg.norm(target - here))
    if total < 1e-6:
        return here
    direction = (target - here) / total
    travelled = 0.0
    stalled = 0
    last = here.copy()
    while travelled < total and stalled < _STALL_TICKS:
        if standoff > 0.0 and cap.camera_clearance() < standoff:
            break
        travelled = min(total, travelled + speed * cap.period)
        step = here + direction * travelled
        cap.base.ctrl = np.array([step[0], step[1], yaw], dtype=np.float64)
        cap.run_until(cap.data.time + cap.period)
        now = np.asarray(cap.base.joint_pos, dtype=np.float64)[:2]
        stalled = stalled + 1 if np.linalg.norm(now - last) < _STALL_PROGRESS_M else 0
        last = now.copy()
    return np.asarray(cap.base.joint_pos, dtype=np.float64)[:2].copy()


def circuit(cap: Capture, yaw: float, *, speed: float, standoff: float, turn_rate: float,
            distance: float, start_xy) -> dict:
    """Follow the room's perimeter for `distance` metres, counting returns to start.

    Loop closure is the whole reason for preferring a SLAM system with a global optimizer,
    and it cannot be tested by an out-and-back: a 6 m there-and-back accumulates ~14 mm of
    drift, which leaves a closure nothing to correct and makes "no closure" indistinguishable
    from "no closure needed". A circuit driven several times over builds real drift *and*
    re-observes the same places, which is the situation a loop closure exists for.

    Reactive rather than planned: drive until the camera closes on something, then rotate
    toward whichever side is more open and carry on. That needs no path planner and no
    per-scene waypoints, and it is roughly what the console's own obstacle brake plus
    frontier explorer end up doing anyway.
    """
    travelled = 0.0
    laps = 0
    away = False
    start_xy = np.asarray(start_xy, dtype=np.float64)
    stalled = 0
    # Ideal time for `distance` at `speed`, times a generous factor for turns and
    # obstacles. Past it the circuit is not slow, it is stuck.
    deadline = cap.data.time + _CIRCUIT_TIME_BUDGET * distance / max(speed, 1e-6)

    while travelled < distance:
        if cap.data.time > deadline:
            print(f"  circuit gave up after {travelled:.1f} m of {distance:.1f} m: "
                  f"no progress in {_CIRCUIT_TIME_BUDGET:.0f}x the time it should take",
                  file=sys.stderr)
            break
        if cap.camera_clearance() < standoff:
            # Turn toward the more open side. Sampling both beats always turning one way,
            # which in a rectangular room walks into the same corner every lap.
            direction = 1.0 if (cap.camera_clearance(bearing=math.pi / 2)
                                >= cap.camera_clearance(bearing=-math.pi / 2)) else -1.0
            turned = 0.0
            was = float(np.asarray(cap.base.joint_pos, dtype=np.float64)[2])
            while turned < 2.0 * math.pi:
                if cap.camera_clearance() > standoff * 1.5:
                    break
                # Integrated, but with the lag clamped, exactly as `PlanarSetpoint` does
                # on the ROS path: an unclamped command runs away from a base held up by
                # geometry, and `turned` then counts rotation that never happened -- which
                # is how a wedged robot decides it has already looked everywhere.
                yaw += direction * turn_rate * cap.period
                actual = float(np.asarray(cap.base.joint_pos, dtype=np.float64)[2])
                lag = math.atan2(math.sin(yaw - actual), math.cos(yaw - actual))
                if abs(lag) > TARGET_LEAD_RAD:
                    yaw = actual + math.copysign(TARGET_LEAD_RAD, lag)
                here = np.asarray(cap.base.joint_pos, dtype=np.float64)[:2]
                cap.base.ctrl = np.array([here[0], here[1], yaw], dtype=np.float64)
                cap.run_until(cap.data.time + cap.period)
                now = float(np.asarray(cap.base.joint_pos, dtype=np.float64)[2])
                turned += abs(now - was)
                was = now
            continue

        # Command a setpoint just ahead of where the base actually is, re-read every tick.
        # Integrating the commanded position instead lets it run away from a base that is
        # being held up by geometry, and the robot then teleports when it comes free.
        here = np.asarray(cap.base.joint_pos, dtype=np.float64)[:2]
        step = speed * cap.period
        target = here + step * np.array([math.cos(yaw), math.sin(yaw)])
        cap.base.ctrl = np.array([target[0], target[1], yaw], dtype=np.float64)
        cap.run_until(cap.data.time + cap.period)

        moved = float(np.linalg.norm(
            np.asarray(cap.base.joint_pos, dtype=np.float64)[:2] - here))
        travelled += moved
        stalled = stalled + 1 if moved < _STALL_PROGRESS_M else 0
        if stalled >= _STALL_TICKS:
            # Wedged on something the forward cone did not see -- a chair leg off-axis.
            # Turn rather than give up: a stalled circuit is a short capture, not a bug.
            yaw += math.pi / 2
            stalled = 0

        distance_from_start = float(np.linalg.norm(
            np.asarray(cap.base.joint_pos, dtype=np.float64)[:2] - start_xy))
        if distance_from_start > 1.5:
            away = True
        elif away and distance_from_start < 0.7:
            laps += 1
            away = False
            print(f"  lap {laps} at {travelled:.1f} m", file=sys.stderr)

    return {"travelled": travelled, "laps": laps, "yaw": yaw}


def spin(cap: Capture, xy, yaw0: float, *, turns: float = 1.0, rate: float = 0.6) -> float:
    """Rotate in place through `turns` revolutions at `rate` rad/s, capturing throughout.

    Returns the yaw actually left on the command, which is `yaw0 + 2*pi*turns` and *not*
    `yaw0`: the base's yaw actuator is a continuous joint, so its commanded position
    accumulates. Handing the caller the accumulated value is the whole point -- commanding
    the wrapped angle afterwards makes the base unwind a full revolution while it drives,
    which reads as a stall and leaves the trajectory metres short.
    """
    swept = 0.0
    total = 2.0 * math.pi * turns
    while swept < total:
        swept = min(total, swept + rate * cap.period)
        cap.base.ctrl = np.array([xy[0], xy[1], yaw0 + swept], dtype=np.float64)
        cap.run_until(cap.data.time + cap.period)
    return yaw0 + total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True, help="house MJCF, by its assets/ path")
    ap.add_argument("--robot", default="myagv")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--hz", type=float, default=15.0, help="capture rate in sim time")
    ap.add_argument("--speed", type=float, default=0.25,
                    help="drive speed in m/s (the myAGV teleop default)")
    ap.add_argument("--reach", type=float, default=3.0,
                    help="how far the outbound leg tries to go, in metres; clipped to "
                         "whatever the ray cast says is actually clear")
    ap.add_argument("--robot-radius", type=float, default=0.25, dest="robot_radius",
                    help="half-width the drive corridor has to clear, in metres")
    ap.add_argument("--circuit", type=float, default=0.0, metavar="METRES",
                    help="after the bootstrap, follow the room perimeter for this many "
                         "metres instead of driving out and back. This is how loop "
                         "closure gets tested: an out-and-back accumulates too little "
                         "drift for a closure to have anything to correct")
    ap.add_argument("--turn-rate", type=float, default=0.4, dest="turn_rate",
                    help="in-place turn rate during a circuit, rad/s. Slower keeps more "
                         "overlap between consecutive frames, which is what a feature "
                         "tracker needs to survive a corner")
    ap.add_argument("--standoff", type=float, default=0.6,
                    help="end a drive leg once the camera is this close to a surface, in "
                         "metres, mirroring the console's obstacle brake. 0 disables it "
                         "and drives until the base stalls -- which parks the camera "
                         "against furniture and measures a failure the robot would never "
                         "actually have")
    ap.add_argument("--offset", type=float, default=0.4,
                    help="sideways offset of the return leg; this is what makes the "
                         "revisit a loop closure rather than a replay")
    ap.add_argument("--strafe", type=float, default=0.0,
                    help="sideways bootstrap, in metres, driven before anything else "
                         "(0 disables). Pure *forward* translation is the degenerate "
                         "case for monocular initialization -- the epipole sits in the "
                         "image and the essential matrix is ill-conditioned -- so a "
                         "holonomic base should bootstrap sideways, where the parallax "
                         "is perpendicular to the optical axis and maximal")
    ap.add_argument("--spin-turns", type=float, default=1.0, dest="spin_turns",
                    help="revolutions in each in-place spin; 0 disables them. Pure "
                         "rotation is the classic monocular-SLAM killer -- no parallax, "
                         "and features leave a narrow FOV fast -- so being able to take "
                         "the spins out is how you tell a tracking failure caused by "
                         "rotation from one caused by the scene")
    ap.add_argument("--spin-rate", type=float, default=0.6, dest="spin_rate",
                    help="spin angular rate in rad/s; slower means more overlap between "
                         "consecutive frames")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    ap.add_argument("--jpeg-quality", type=int, default=85, dest="jpeg_quality",
                    help="higher than the sim's streaming default: this is the input to "
                         "a feature detector, not a preview")
    ap.add_argument("--no-depth", action="store_true", dest="no_depth",
                    help="skip the depth stream (it is what makes RGB-D testable later)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, data, config, base, start = build_scene(args.scene, args.robot)
    ns = config.robot_namespace
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{ns}front_camera")
    if cam_id < 0:
        raise SystemExit(f"{args.robot} has no {ns}front_camera")

    width, height = args.size
    info = camera_intrinsics(model, cam_id, width, height)
    (out / "camera_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"scene {Path(args.scene).name}, start {np.round(start, 3)}, "
          f"fx={info['fx']:.1f} at {width}x{height}", file=sys.stderr)

    cap = Capture(model, data, base, cam_id, out, hz=args.hz, size=args.size,
                  jpeg_quality=args.jpeg_quality, want_depth=not args.no_depth,
                  exclude_bodies=frozenset(
                      i for i in range(model.nbody)
                      if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
                      and n.startswith(ns)
                  ))

    origin = start[:2].copy()
    yaw, clear = best_heading(model, data, ns, origin,
                              robot_radius=args.robot_radius, max_range=args.reach + 2.0,
                              height=float(data.cam_xpos[cam_id][2]))
    reach = max(0.0, min(args.reach, clear - args.standoff))
    print(f"driving along yaw {math.degrees(yaw):.0f} deg, {clear:.2f} m clear "
          f"-> reach {reach:.2f} m", file=sys.stderr)
    if reach < 1.0:
        print("warning: under a metre of clear run; monocular initialization needs "
              "parallax and may not get enough here", file=sys.stderr)
    # Face the direction of travel: the camera is fixed to the base, so mapping what it
    # is driving into means turning first.
    base.ctrl = np.array([origin[0], origin[1], yaw], dtype=np.float64)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-heading[1], heading[0]])

    # Seed features before asking for any translation: monocular initialization needs
    # parallax, and a tracker that has never seen the scene cannot get it from the first
    # metre of a straight line. `yaw` accumulates across spins -- see spin()'s docstring.
    def maybe_spin(xy, yaw0: float) -> float:
        if args.spin_turns <= 0.0:
            return yaw0
        return spin(cap, xy, yaw0, turns=args.spin_turns, rate=args.spin_rate)

    # Sideways first, holding heading: this is the motion that actually initializes a
    # monocular tracker. Out and back, so the run still starts where it means to.
    if args.strafe > 0.0:
        drive(cap, origin + left * args.strafe, yaw, speed=args.speed)
        drive(cap, origin, yaw, speed=args.speed)

    yaw = maybe_spin(origin, yaw)

    if args.circuit > 0.0:
        result = circuit(cap, yaw, speed=args.speed, standoff=args.standoff,
                         turn_rate=args.turn_rate, distance=args.circuit,
                         start_xy=origin)
        print(f"circuit: {result['travelled']:.1f} m, {result['laps']} return(s) to start",
              file=sys.stderr)
        cap.close()
        print(f"wrote {cap.n} frames to {out}", file=sys.stderr)
        return 0

    far = drive(cap, origin + heading * reach, yaw, speed=args.speed,
                standoff=args.standoff)
    print(f"outbound reached {np.round(far, 3)} "
          f"({np.linalg.norm(far - origin):.2f} m of {reach:.2f}), "
          f"camera clearance {cap.camera_clearance():.2f} m", file=sys.stderr)
    yaw = maybe_spin(far, yaw)
    drive(cap, far + left * args.offset, yaw, speed=args.speed)
    back = drive(cap, origin + left * args.offset, yaw, speed=args.speed)
    print(f"return reached {np.round(back, 3)} "
          f"({np.linalg.norm(back - origin):.2f} m from start)", file=sys.stderr)
    maybe_spin(back, yaw)

    cap.close()
    print(f"wrote {cap.n} frames to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
