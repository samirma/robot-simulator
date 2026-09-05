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

    __slots__ = ("camera", "scan", "depth", "camera_info", "camera_frame", "scan_frame")

    def __init__(self, camera: str, scan: str, depth: str, camera_info: str,
                 camera_frame: str = "camera", scan_frame: str = "laser_frame") -> None:
        self.camera, self.scan = camera, scan
        self.depth, self.camera_info = depth, camera_info
        # Frames, not topics, and they carry the namespace *without* a leading slash --
        # see contracts/namespace.py. Two bases on one graph both reporting `laser_frame`
        # give a tf tree one frame with two parents.
        self.camera_frame, self.scan_frame = camera_frame, scan_frame


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
                 topics: SensorTopics, scene_option: "mujoco.MjvOption | None" = None,
                 camera_period: float = 0.0) -> None:
        self._server = server
        self._model = model
        self._camera = camera
        self._jpeg_quality = jpeg_quality
        self._scan = scan
        self._depth = depth
        self._topics = topics
        self._scan_next = 0.0
        self._depth_next = 0.0
        # The colour camera gets its own clock, like `/scan` and depth beside it. It is
        # the one stream here that used to render once per control tick, which made it
        # both less like real hardware -- a camera has a frame rate of its own -- and the
        # thing that couples one robot's cost to another's control rate. Measured on an
        # iTHOR kitchen at 10 Hz control: the SO-101 alone publishes at 9.8 Hz, and adding
        # a myAGV takes it to 5.7 Hz; disabling only the AGV's colour camera puts it back
        # to 8.4 Hz, while dropping the AGV's lidar entirely is worth just 1.0 Hz. The
        # render is the cost, so this is the knob that moves it.
        #
        # 0 keeps the old behaviour -- one frame per control tick -- so nothing changes
        # for a caller that does not ask.
        self._camera_period = float(camera_period)
        self._camera_next = 0.0
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
            TYPE_CAMERA_INFO,
            TYPE_COMPRESSED_IMAGE,
            TYPE_IMAGE,
            TYPE_LASER_SCAN,
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
                    scan_time=scan["period"], frame_id=self._topics.scan_frame,
                ),
                TYPE_LASER_SCAN,
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
            self._server.publish(
                self._topics.depth,
                image(seq, mm.tobytes(), "16UC1", dw, dh, frame_id=self._topics.camera_frame),
                TYPE_IMAGE,
            )
            self._server.publish(
                self._topics.camera_info,
                camera_info(seq, dw, dh, self._depth["fovy"],
                            frame_id=self._topics.camera_frame),
                TYPE_CAMERA_INFO,
            )

        if self._renderer is not None and now >= self._camera_next:
            self._camera_next = now + self._camera_period
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
                        self._topics.camera,
                        compressed_image(seq, buf.tobytes(),
                                         frame_id=self._topics.camera_frame),
                        TYPE_COMPRESSED_IMAGE,
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

    def __init__(self, model, cameras, jpeg_quality: int = 70, scene_option=None,
                 frame_of=None) -> None:
        """`cameras` is an ordered mapping of topic -> (mjcf camera name, width, height).

        `frame_of` maps a contract camera name to the `frame_id` to publish, which is how
        a namespace reaches the frame. The frame is derived from the **topic**, not from
        the MJCF camera name: those differ, and using the MJCF name shipped the engine's
        own body prefix onto the wire -- the wrist view went out as `frame_id`
        `robot_0/wrist`, which is an engine detail a client is not supposed to be able to
        see, let alone one it could tell the two engines apart by.
        """
        self._model = model
        self._jpeg_quality = jpeg_quality
        self._scene_option = scene_option
        self._frame_of = frame_of if frame_of is not None else (lambda name: name)
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
                                      frame_id=self._frame_of(_camera_frame(topic))),
                TYPE_COMPRESSED_IMAGE_ROS2,
            )

    def close(self) -> None:
        for renderer in self._renderers.values():
            if renderer is not None:
                renderer.close()
        self._renderers.clear()


def _camera_frame(topic: str) -> str:
    """`/overhead/color/compressed` -> `overhead`; the contract's name for that view.

    `image_transport republish` appends `/color/compressed` to the camera's own name, so
    stripping that suffix recovers it -- the same rule `simulator/live_cameras.html` uses
    to label a stream it discovered.
    """
    name = topic.strip("/")
    for suffix in ("/color/compressed", "/image_raw/compressed", "/compressed"):
        if name.endswith(suffix.strip("/")) and len(name) > len(suffix.strip("/")):
            return name[: -len(suffix)]
    return name


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


def camera_framing(model, data, camera: str, points) -> list[tuple[float, float]]:
    """Normalised image coordinates of world points through a named MJCF camera.

    `(0, 0)` is the frame centre and `|u|, |v| <= 1` is in frame, so one number -- the
    worst `|normalised|` over a set of points -- says whether a view actually contains
    what it is supposed to. That matters because a camera can be at exactly the right
    pose and still frame the wrong thing: framing is a property of the *surface* under
    the camera as much as of the camera, and the reference rig's overhead view scores
    0.930 on its table's four corners where a plausible-looking predecessor scored 3.135
    with all four corners outside the frame.

    Call after `mj_forward`. MuJoCo cameras look down their own **-z** with +x right and
    +y up, which is why the depth term is negated.
    """
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam < 0:
        raise ValueError(f"no camera {camera!r} in this model")
    eye = np.asarray(data.cam_xpos[cam], dtype=np.float64)
    rot = np.asarray(data.cam_xmat[cam], dtype=np.float64).reshape(3, 3)
    half = np.tan(np.radians(float(model.cam_fovy[cam])) / 2.0)
    width, height = (int(v) for v in model.cam_resolution[cam])
    aspect = (width / height) if height > 0 else 1.0

    out = []
    for point in points:
        local = rot.T @ (np.asarray(point, dtype=np.float64) - eye)
        depth = -local[2]
        if depth <= 1e-9:  # behind the camera; no meaningful projection
            out.append((float("inf"), float("inf")))
            continue
        out.append((float(local[0] / depth / (half * aspect)), float(local[1] / depth / half)))
    return out


def clipped_fraction(pixels, threshold: int = 255) -> float:
    """Share of pixels at full brightness -- the exposure number, not an impression.

    A scene lit for one room and then dropped into another is the common way a camera
    stops showing what a policy was trained on, and it is not obvious by eye until it is
    extreme. The reference rig tuned its headlight against exactly this measure: 41.6 %
    of pixels clipped before, 3.0 % after.
    """
    array = np.asarray(pixels)
    if array.size == 0:
        return 0.0
    return float((array >= threshold).sum()) / float(array.size)


def report_slab_fit(model, data, body: str = "task_table", ignore_prefixes=("task_", "robot_")) -> None:
    """Say where a staged work surface lands inside a scene that knows nothing about it.

    Two measurements, because a slab fits badly in two unrelated ways and each is
    invisible to the other's test. **Contacts** catch a corner driven into a movable
    obstacle. **Rays** catch a corner hanging over the edge of a counter with air beneath
    it, which produces no contact at all and is equally worth knowing about before an
    object is placed near it.

    Known limit of the contact half: MuJoCo generates no contacts between two static
    geoms, and both this slab and a kitchen's fixtures are static -- so a slab
    intersecting a wall reads as 0 mm here. The ray drops are what actually carry the
    report; the contact scan only catches the movable case. Fixing that properly means
    an explicit box-vs-geom sweep, which is more machinery than a placement hint needs.

    Reports; never raises. A static slab overhanging a worktop is harmless -- the objects
    sit on top of it either way -- and turning an unlucky kitchen layout into a hard
    failure would make it look like a broken build. The summary line is printed
    unconditionally, so "no warning" is distinguishable from "the check did not run".
    """
    slab = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if slab < 0:
        return
    slab_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == slab}

    def ours(geom: int) -> bool:
        """Is this geom the task's or the robot's, rather than the scene's?"""
        names = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) or "",
        )
        return any(n.startswith(ignore_prefixes) for n in names)

    worst, culprit = 0.0, ""
    for i in range(data.ncon):
        pair = (data.contact[i].geom1, data.contact[i].geom2)
        hit = set(pair) & slab_geoms
        if not hit:
            continue
        other = pair[0] if pair[1] in slab_geoms else pair[1]
        if other in slab_geoms or ours(other):
            continue
        depth = -data.contact[i].dist
        if depth > worst:
            worst, culprit = depth, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or "?"

    # `mj_ray`'s geomgroup mask takes 1 = consider; excluding our own geoms would need a
    # bodyexclude, so cast from just above each corner and ignore hits on the slab itself.
    drops: list[float] = []
    geomid = np.zeros(1, dtype=np.int32)
    down = np.array([0.0, 0.0, -1.0])
    for corner in _slab_corners_world(model, data, slab):
        start = np.asarray(corner, dtype=np.float64) + np.array([0.0, 0.0, 0.01])
        dist = mujoco.mj_ray(model, data, start, down, None, 1, slab, geomid)
        drops.append(float(dist) if dist >= 0 else float("inf"))

    supported = sum(1 for d in drops if d < 0.05)
    worst_drop = max(drops)
    drop_text = "unsupported (no surface below)" if not np.isfinite(worst_drop) else f"{worst_drop:.2f} m"
    print(
        f"table fit: {supported}/4 corners resting on something, worst gap {drop_text}; "
        f"penetration {worst * 1000:.0f} mm" + (f" into {culprit}" if culprit else ""),
        file=sys.stderr,
    )


def _slab_corners_world(model, data, slab: int):
    """The slab's four top-face corners in world coordinates, from its compiled pose."""
    from tasks.apple_on_plate import TABLE_HALF  # noqa: PLC0415 -- engine-side import

    rot = np.asarray(data.xmat[slab], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(data.xpos[slab], dtype=np.float64)
    hx, hy, _ = TABLE_HALF
    return [
        origin + rot @ np.array([sx * hx, sy * hy, 0.0])
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]


def run_sim_loop(model, data, controller, *, control_hz: float, deadline=None,
                 viewer=None, sync_hz: float = 60.0, label: str = "sim loop") -> None:
    """Step `model` pinned to the wall clock, feeding `controller` at `control_hz`.

    One loop for every way an engine can be run -- headless or with a window, either
    engine -- because there used to be three of these and two were wrong in the same way.

    **The physics is pinned to the wall clock**: each pass steps until `data.time` has
    caught up with elapsed real time, so rendering cost lands on the *camera* rate and
    never on simulated seconds per real second. The naive one-step-then-sleep loop let
    three cameras drag the simulation to 0.66x real time, and a policy that moves a fixed
    angle per wall-clock tick then moves 50 % faster in simulated time than it was tuned
    for -- which for a grasp tuned to 0.06 rad/step against a measured failure above 0.08
    is the difference between lifting the apple and leaving it on the table. A machine
    that genuinely cannot keep up is told so, rather than quietly slowed down.

    **`viewer.sync()` runs at `sync_hz`, not once per physics step.** At a 2 ms timestep
    that would be 500 syncs a second against a 60 Hz display, and the surplus was half of
    what made the windowed path slower than the headless one.

    `controller`, `mj_step` and `sync()` all run on **this one thread**, which is what
    both the thread-safe `/reset` handoff in `ros_surfaces/so101.py` and `launch_passive`
    require. Do not move any of them onto another.
    """
    control_period = 1.0 / control_hz
    sync_period = 1.0 / sync_hz if sync_hz > 0 else None
    next_control = 0.0
    next_sync = 0.0
    wall_start = time.monotonic()
    sim_start = float(data.time)
    max_catchup = int(0.25 / model.opt.timestep)  # cap a stall at a quarter second
    behind_since = None

    try:
        while viewer is None or viewer.is_running():
            now = time.monotonic()
            if controller is not None and now >= next_control:
                controller(data)
                next_control = now + control_period

            target_time = sim_start + (time.monotonic() - wall_start)
            steps = 0
            while data.time < target_time and steps < max_catchup:
                mujoco.mj_step(model, data)
                steps += 1

            if steps >= max_catchup:
                # Fell more than the cap behind: rebase rather than chase forever.
                if behind_since is None:
                    behind_since = now
                elif now - behind_since > 5.0:
                    print(f"{label} cannot keep real time on this machine (physics + "
                          "cameras take longer than the wall clock); reduce cameras or "
                          "--control-hz", file=sys.stderr)
                    behind_since = now
                wall_start = time.monotonic()
                sim_start = float(data.time)
            else:
                behind_since = None

            if viewer is not None and sync_period is not None:
                now = time.monotonic()
                if now >= next_sync:
                    viewer.sync()
                    next_sync = now + sync_period

            if deadline is not None and time.monotonic() > deadline:
                break

            slack = (target_time + model.opt.timestep) - (
                sim_start + (time.monotonic() - wall_start)
            )
            if slack > 0:
                time.sleep(min(slack, control_period / 4))
    except KeyboardInterrupt:
        pass
