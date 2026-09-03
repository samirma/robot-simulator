"""Engine-neutral MuJoCo → wire-contract helpers, shared by every MuJoCo engine.

Any simulator that ends up with a raw `mujoco.MjModel` / `mujoco.MjData` for a scene
containing one of the shared robots can host the *real* robot's interface with the
pieces here:

- `PlanarSetpoint`  — integrate a body-frame `cmd_vel` into a world-frame position
  setpoint for a holonomic base (the myAGV drive model).
- `laser_scan_ranges` — a YDLidar-X2-shaped 2D scan by ray casting.
- `SensorStreams` — camera (JPEG), optional depth, and `/scan`, published on the shared
  rosbridge server.
- `SensorTopics` — the per-robot topic names those streams use.

They take a raw `model`/`data` and a `contracts` server, so they are identical whether
the MuJoCo model came from MolmoSpaces, RoboCasa/robosuite, or a hand-built scene. The
constants and the beam-ordering reasoning were proven against the real 2023 Pi myAGV;
see the long comments for the traps. This module deliberately depends only on `mujoco`,
`numpy`, and `simulator/shared/contracts` — never on any one engine.
"""

from __future__ import annotations

import sys
import time

import mujoco
import numpy as np

# How far the position setpoint of a velocity-driven base may run ahead of the robot.
# Large enough that the servo is always pulling at full effort, small enough that a robot
# stopped by a wall has only centimetres of wind-up to release when it turns away.
TARGET_LEAD_M = 0.12
# Tighter than the linear lead: yaw is the axis with the least inertia (~0.05 kg m2), and
# a setpoint far ahead of the robot yanks it round hard enough to shove the base sideways
# through the position servo -- which shows up as translation during a pure rotation.
TARGET_LEAD_RAD = 0.15


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
    exclude_bodies: frozenset[int] | None = None,
) -> np.ndarray:
    """A 2D laser scan by ray casting, counter-clockwise from `angle_min`.

    `origin` is the *laser* origin, not the base origin -- callers add the mount offset.
    Angles are measured in the base frame, so `yaw + angle_min` is the first beam.

    The real 2023 Pi AGV carries a YDLidar X2; this stands in for it. `mj_ray` is the only
    per-step ranging primitive MuJoCo offers -- rangefinder sensors would mean regenerating
    the model, and depth rendering costs an order of magnitude more per beam.

    On the beam ordering, which is easy to get backwards: the X2 is launched with
    `inverted: true` because it is mounted upside down, and `myagv_active.launch` then
    publishes `base_footprint -> laser_frame` with a roll of pi. Those two mirrors cancel,
    so in the base frame the published scan runs counter-clockwise from -pi -- which is
    what this function produces. Do not "fix" one without the other.

    Misses come back as `max_range + 1`; see contracts.rosbridge_server.laser_scan for why
    that rather than the X2's 0.0.

    `exclude_bodies` exists for legged robots. `mj_ray` takes a single `bodyexclude`,
    which is enough when all of a robot's geometry hangs off its root body -- the myAGV's
    chassis box does -- but a biped's separate limb bodies would otherwise be ranged at a
    few centimetres on most sweeps. Each beam that lands on an excluded body is re-cast
    from just past the hit.
    """
    angles = yaw + np.linspace(angle_min, angle_max, beams, endpoint=False)
    geomid = np.zeros(1, dtype=np.int32)
    ranges = np.full(beams, max_range + 1.0)
    origin = np.ascontiguousarray(origin, dtype=np.float64)
    exclude = exclude_bodies or frozenset()
    # Enough to clear a limb; unbounded re-casting would turn a bad pose into a hang.
    max_recast = 4
    nudge = 1e-3

    for i, a in enumerate(angles):
        vec = np.array([np.cos(a), np.sin(a), 0.0])
        start = origin
        travelled = 0.0
        for _ in range(max_recast + 1):
            dist = mujoco.mj_ray(model, data, start, vec, None, 1, bodyexclude, geomid)
            if geomid[0] < 0 or dist < 0.0:
                break
            total = travelled + dist
            if model.geom_bodyid[geomid[0]] in exclude:
                travelled = total + nudge
                start = np.ascontiguousarray(origin + vec * travelled, dtype=np.float64)
                continue
            if total <= max_range:
                ranges[i] = total
            break
    return ranges


class SensorTopics:
    """Topic names for the streams every ROS surface shares.

    A frozen-in-all-but-name record rather than a dataclass so this module keeps its
    single dependency-free import list. Defaults are the myAGV's, which is where these
    names came from.
    """

    __slots__ = ("camera", "scan", "depth", "camera_info")

    def __init__(self, camera: str, scan: str, depth: str, camera_info: str) -> None:
        self.camera, self.scan = camera, scan
        self.depth, self.camera_info = depth, camera_info


class PlanarSetpoint:
    """Integrate a body-frame velocity into a world-frame position setpoint.

    These are position actuators, and re-deriving the setpoint from the measured pose
    every step left it only ever one increment (14 mm at 0.28 m/s) ahead of a robot that
    was chasing it -- the base settled at roughly a **sixth** of the commanded speed.
    Integrating the target instead makes a commanded velocity mean what it says.

    The lead clamp keeps the property that made the old version tempting: a robot held up
    by a wall stops advancing its target rather than winding up a lunge it releases the
    moment it comes free. Yaw gets the tighter lead because it is the axis with the least
    inertia (~0.05 kg m2 on the myAGV), and a setpoint far ahead of the robot yanks it
    round hard enough to shove the base sideways through the position servo -- which reads
    as translation during a pure rotation.
    """

    def __init__(self, lead_m: float = TARGET_LEAD_M, lead_rad: float = TARGET_LEAD_RAD):
        self._target: np.ndarray | None = None
        self._lead_m = lead_m
        self._lead_rad = lead_rad

    def reset(self) -> None:
        self._target = None

    def step(self, x: float, y: float, yaw: float,
             vx: float, vy: float, wz: float, dt: float) -> np.ndarray:
        """Advance the setpoint by one control period and return [x, y, yaw]."""
        if self._target is None:
            self._target = np.array([x, y, yaw])

        c, s = np.cos(yaw), np.sin(yaw)
        self._target = self._target + np.array(
            [(vx * c - vy * s) * dt, (vx * s + vy * c) * dt, wz * dt]
        )

        lag = self._target[:2] - np.array([x, y])
        dist = float(np.linalg.norm(lag))
        if dist > self._lead_m:
            self._target[:2] = np.array([x, y]) + lag / dist * self._lead_m
        yaw_lag = float(
            np.arctan2(np.sin(self._target[2] - yaw), np.cos(self._target[2] - yaw))
        )
        if abs(yaw_lag) > self._lead_rad:
            self._target[2] = yaw + np.sign(yaw_lag) * self._lead_rad
        return self._target


class SensorStreams:
    """Camera, depth, camera_info and /scan -- shared by every robot's ROS surface.

    Everything here is a property of the scene and of where the sensors are mounted, not
    of the robot's control contract, which is why two robots with entirely disjoint topic
    sets still share it. The topic *names* are passed in, since those are the per-robot
    part.
    """

    def __init__(self, server, model, camera: str | None, camera_size,
                 jpeg_quality: int, scan: dict | None, depth: dict | None,
                 topics: SensorTopics, scene_option: "mujoco.MjvOption | None" = None) -> None:
        self._server = server
        self._model = model
        self._camera = camera
        self._jpeg_quality = jpeg_quality
        self._scan = scan
        self._depth = depth
        self._topics = topics
        self._scan_next = 0.0
        self._depth_next = 0.0
        # Engines whose scenes carry debug-only geometry (RoboCasa's collision geoms,
        # painted in random semi-transparent colours) pass a scene_option to keep it out
        # of the camera stream; None renders whatever MuJoCo's defaults show.
        self._scene_option = scene_option

        self._renderer = None
        if camera is not None:
            width, height = camera_size
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            self._renderer = mujoco.Renderer(model, height, width)

        # A second renderer, because a MuJoCo renderer is either in depth mode or not and
        # toggling it per frame would fight the colour stream sharing the same object.
        self._depth_renderer = None
        if depth is not None and camera is not None:
            dw, dh = depth["size"]
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, dw)
            model.vis.global_.offheight = max(model.vis.global_.offheight, dh)
            self._depth_renderer = mujoco.Renderer(model, dh, dw)
            self._depth_renderer.enable_depth_rendering()

        # The rays must not range the robot itself. One `bodyexclude` covers a robot whose
        # geometry hangs off a single root body; `exclude_bodies` covers the rest.
        self._scan_body = -1
        self._scan_exclude: frozenset[int] = frozenset()
        if scan is not None:
            self._scan_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, scan["body"]
            )
            self._scan_exclude = frozenset(scan.get("exclude_bodies") or ())

    @property
    def published(self) -> list[str]:
        out = []
        if self._renderer is not None:
            out.append(self._topics.camera)
        if self._scan is not None:
            out.append(self._topics.scan)
        if self._depth_renderer is not None:
            out += [self._topics.depth, self._topics.camera_info]
        return out

    def publish(self, data, seq: int, x: float, y: float, yaw: float) -> None:
        from contracts.rosbridge_server import (
            camera_info,
            compressed_image,
            image,
            laser_scan,
        )

        now = time.monotonic()

        if self._scan is not None and now >= self._scan_next:
            # A real lidar spins at a fixed rate regardless of how fast anything else
            # runs, so /scan gets its own clock rather than riding the control rate. A
            # client that assumed one scan per command would break on real hardware.
            self._scan_next = now + self._scan["period"]
            scan = self._scan
            ranges = laser_scan_ranges(
                self._model,
                data,
                np.array([
                    x + scan["offset_x"] * np.cos(yaw),
                    y + scan["offset_x"] * np.sin(yaw),
                    scan["offset_z"],
                ]),
                yaw,
                scan["beams"],
                scan["max_range"],
                bodyexclude=self._scan_body,
                exclude_bodies=self._scan_exclude,
            )
            step_angle = 2 * np.pi / scan["beams"]
            self._server.publish(
                self._topics.scan,
                laser_scan(
                    seq, ranges, -np.pi, np.pi - step_angle, step_angle,
                    range_min=scan["min_range"], range_max=scan["max_range"],
                    scan_time=scan["period"],
                ),
            )

        if self._depth_renderer is not None and now >= self._depth_next:
            self._depth_next = now + self._depth["period"]
            self._depth_renderer.update_scene(
                data, camera=self._camera, scene_option=self._scene_option
            )
            # Metres to millimetres in uint16: 640x480 float32 is 1.2 MB a frame, which a
            # JSON websocket will not carry at any useful rate. Anything beyond the sensor
            # range becomes 0, which is what "no return" means in a 16UC1 depth image.
            metres = self._depth_renderer.render()
            mm = np.where(
                np.isfinite(metres) & (metres < self._depth["max_range"]),
                metres * 1000.0, 0.0,
            ).astype(np.uint16)
            dw, dh = self._depth["size"]
            self._server.publish(self._topics.depth, image(seq, mm.tobytes(), "16UC1", dw, dh))
            self._server.publish(
                self._topics.camera_info, camera_info(seq, dw, dh, self._depth["fovy"])
            )

        if self._renderer is not None:
            self._renderer.update_scene(
                data, camera=self._camera, scene_option=self._scene_option
            )
            frame = self._renderer.render()
            try:
                import cv2

                ok, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                )
                if ok:
                    self._server.publish(
                        self._topics.camera, compressed_image(seq, buf.tobytes())
                    )
            except Exception as exc:
                print(f"camera encode failed: {exc}", file=sys.stderr)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        if self._depth_renderer is not None:
            self._depth_renderer.close()


class PlanarJointBase:
    """A holonomic base as three world-aligned joints, straight off a raw MuJoCo model.

    The mobile robots here are driven by a virtual (slide-x, slide-y, hinge-z) trio and
    matching position actuators rather than by simulated Mecanum contacts; see
    `shared/robots/myagv/model.xml`. An engine built on MolmoSpaces gets the same three
    numbers through its `HoloJointsRobotBaseGroup` move group, so the ROS surfaces are
    written against this two-property interface (`pose`, `ctrl`) and work with either.

    Reading the joints rather than the body transform is exact *because* the robot is
    grafted in at the origin with identity rotation -- which the holonomic spawn path
    guarantees, since world-aligned slide joints mean nothing anywhere else. The
    constructor checks that rather than trusting it.
    """

    __slots__ = ("_data", "_qpos", "_ctrl", "_body")

    AXES = ("x", "y", "theta")

    def __init__(self, model, data, prefix: str = "", root: str = "base") -> None:
        self._data = data
        self._qpos = []
        self._ctrl = []
        for axis in self.AXES:
            joint = f"{prefix}{root}_{axis}"
            actuator = f"{joint}_act"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            if jid < 0 or aid < 0:
                raise ValueError(
                    f"no planar base in this model: expected joint {joint!r} and "
                    f"actuator {actuator!r}"
                )
            self._qpos.append(int(model.jnt_qposadr[jid]))
            self._ctrl.append(aid)

        self._body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{root}")
        if self._body < 0:
            raise ValueError(f"no body named {prefix}{root!r}")
        # The joints are the pose only if the body's own frame is the world frame.
        offset = model.body_pos[self._body]
        quat = model.body_quat[self._body]
        if not (np.allclose(offset, 0.0, atol=1e-9) and np.allclose(quat, [1, 0, 0, 0], atol=1e-9)):
            raise ValueError(
                f"{prefix}{root} is attached at pos={offset} quat={quat}, not at the "
                "origin with identity rotation; its world-aligned slide joints would no "
                "longer mean world x/y. Attach at the origin and set the pose instead."
            )

    @property
    def xytheta(self) -> np.ndarray:
        return np.array([float(self._data.qpos[i]) for i in self._qpos])

    @property
    def pose(self) -> np.ndarray:
        """The base pose as a 4x4, the shape every ROS surface reads."""
        x, y, yaw = self.xytheta
        c, s = np.cos(yaw), np.sin(yaw)
        pose = np.eye(4)
        pose[:2, :2] = [[c, -s], [s, c]]
        pose[0, 3], pose[1, 3] = x, y
        return pose

    @property
    def ctrl(self) -> np.ndarray:
        return np.array([float(self._data.ctrl[i]) for i in self._ctrl])

    @ctrl.setter
    def ctrl(self, target) -> None:
        target = np.asarray(target, dtype=np.float64)
        for i, value in zip(self._ctrl, target):
            self._data.ctrl[i] = value

    def teleport(self, x: float, y: float, yaw: float) -> None:
        """Put the base somewhere at spawn time, holding the target there.

        Both halves are needed: writing only the joints makes the robot drive straight
        back to the actuators' default target of 0 on the first step, and writing only
        the target makes it drive there from the origin through whatever is in between.
        """
        for i, value in zip(self._qpos, (x, y, yaw)):
            self._data.qpos[i] = value
        self.ctrl = (x, y, yaw)


class CameraStreams:
    """Several named MJCF cameras, JPEG-encoded onto a rosbridge server.

    `SensorStreams` above is the mobile-base shape: one camera, plus depth and a lidar.
    An arm needs the opposite -- no scan, no depth, and *more than one* colour view,
    because a VLA policy consumes views positionally and a single frame gives it nothing
    to triangulate with. Rather than grow `SensorStreams` a list-shaped camera argument
    that only one caller would pass, this is its sibling, and both stay easy to read.

    One `mujoco.Renderer` per distinct frame size, shared by every camera at that size:
    a Renderer is bound to a width and height at construction, but not to a camera, so
    three 640x480 views cost one renderer and one offscreen buffer, not three.

    **Render cost lands on the physics loop, not on the client.** Every enabled camera
    is rendered inside the step callback, so the achievable control rate falls as
    cameras are added -- measured on the equivalent rig at 4.1 Hz with two and ~2.1 Hz
    with four, against a 10 Hz control loop. That is why the caller passes an explicit
    list instead of the surface enabling everything the model declares, and why the
    wrist view is opt-in.
    """

    def __init__(self, model, cameras, jpeg_quality: int = 70, scene_option=None) -> None:
        """`cameras` is an ordered mapping of topic -> (mjcf camera name, width, height)."""
        self._model = model
        self._jpeg_quality = jpeg_quality
        self._scene_option = scene_option
        self._cameras: list[tuple[str, str, int, int]] = []
        self._renderers: dict[tuple[int, int], "mujoco.Renderer"] = {}

        for topic, (name, width, height) in dict(cameras).items():
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                declared = [model.camera(i).name for i in range(model.ncam)]
                raise SystemExit(
                    f"camera {name!r} is not in this model; it declares {declared}. "
                    "A camera that is missing here would otherwise become a topic that "
                    "advertises fine and never publishes, and the client would find out "
                    "as a reset timeout minutes later."
                )
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            self._cameras.append((topic, name, width, height))

        for _topic, _name, width, height in self._cameras:
            self._renderers.setdefault((width, height), None)
        for size in list(self._renderers):
            width, height = size
            self._renderers[size] = mujoco.Renderer(model, height, width)

    @property
    def published(self) -> list[str]:
        return [topic for topic, _name, _w, _h in self._cameras]

    def publish(self, server, data, seq: int, stamp_s: float) -> None:
        from contracts.rosbridge_server import TYPE_COMPRESSED_IMAGE_ROS2, compressed_image_ros2

        for topic, name, width, height in self._cameras:
            renderer = self._renderers[(width, height)]
            if self._scene_option is not None:
                renderer.update_scene(data, camera=name, scene_option=self._scene_option)
            else:
                renderer.update_scene(data, camera=name)
            frame = renderer.render()
            server.publish(
                topic,
                compressed_image_ros2(seq, _encode_jpeg(frame, self._jpeg_quality), stamp_s,
                                      frame_id=name),
                TYPE_COMPRESSED_IMAGE_ROS2,
            )

    def close(self) -> None:
        for renderer in self._renderers.values():
            if renderer is not None:
                renderer.close()
        self._renderers.clear()


def _encode_jpeg(frame, quality: int) -> bytes:
    """RGB uint8 array -> JPEG bytes.

    cv2 when it is available (it is, in every engine venv, via the console's own
    dependency set) and Pillow otherwise, so a headless box without OpenCV still
    streams rather than failing at the first frame.
    """
    try:
        import cv2

        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return bytes(buf)
    except ImportError:
        import io

        from PIL import Image

        out = io.BytesIO()
        Image.fromarray(frame).save(out, format="JPEG", quality=quality)
        return out.getvalue()
