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


def is_loose(model, bodyid: int) -> bool:
    """Is this body a thing lying in the world, rather than part of the world?

    Loose means its weld root hangs off the world on a **free joint**: an apple, a mug, a
    bowl, a plate -- something a robot pushes and does not stand on. Everything else is
    furniture, including a cabinet door and an oven drawer, which really are standable.

    The obvious test, "is this body welded to the world" (`body_weldid == 0`), is the wrong
    one and fails in a way that only a real scene shows: iTHOR hangs every cabinet door on
    a hinge and every oven drawer on a slide, so on FloorPlan1 it calls **1608 of 2116
    geoms** movable and a robot obeying it would fall through a third of the kitchen. The
    free joint is what separates the two, and it puts the island the AiNex stands on
    (`standardislandheight_...`, weld root 0) on the same side as the floor while leaving
    the 37 things actually lying about -- apple, bowl, cup, bread, bottle, book, pan,
    knife -- on the other.
    """
    root = int(model.body_weldid[bodyid])
    if root == 0:
        return False
    adr, num = int(model.body_jntadr[root]), int(model.body_jntnum[root])
    return any(
        model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(adr, adr + num)
    )


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
    geomgroup=None,
) -> np.ndarray:
    """A 2D laser scan by ray casting, counter-clockwise from `angle_min`.

    `origin` is the *laser* origin, not the base origin -- callers add the mount offset.
    Angles are measured in the base frame, so `yaw + angle_min` is the first beam.

    The real 2023 Pi AGV carries a YDLidar X2; this stands in for it. Ray casting
    (`mj_multiRay`, and `mj_ray` for re-casts) is the per-step ranging MuJoCo offers --
    rangefinder sensors would mean regenerating the model, and depth rendering costs an
    order of magnitude more per beam.

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
    vecs = np.zeros((beams, 3))
    vecs[:, 0] = np.cos(angles)
    vecs[:, 1] = np.sin(angles)
    origin = np.ascontiguousarray(origin, dtype=np.float64)
    exclude = exclude_bodies or frozenset()

    # Every beam in one call: `mj_multiRay` shares the per-geom culling across the fan.
    # On the iTHOR kitchen a 360-beam sweep went from 5.5 ms as separate `mj_ray` calls
    # to 2.1 ms, with identical ranges -- at 30 Hz that was ~6 % of the simulation thread.
    geomids = np.full(beams, -1, dtype=np.int32)
    dists = np.full(beams, -1.0)
    _multi_ray(model, data, origin, vecs, bodyexclude, geomids, dists, max_range, geomgroup)
    ranges = np.where((geomids >= 0) & (dists >= 0.0) & (dists <= max_range),
                      dists, max_range + 1.0)

    if exclude:
        # A beam whose first hit is one of the excluded bodies is re-cast from just past
        # that hit, one beam at a time; only these few take the slow path.
        geomid = np.zeros(1, dtype=np.int32)
        max_recast = 4   # enough to clear a limb; unbounded would turn a bad pose into a hang
        nudge = 1e-3
        hit_bodies = model.geom_bodyid[np.maximum(geomids, 0)]
        for i in np.flatnonzero(geomids >= 0):
            if int(hit_bodies[i]) not in exclude:
                continue
            ranges[i] = max_range + 1.0
            vec = vecs[i]
            travelled = float(dists[i]) + nudge
            for _ in range(max_recast):
                start = np.ascontiguousarray(origin + vec * travelled, dtype=np.float64)
                dist = mujoco.mj_ray(model, data, start, vec, None, 1, bodyexclude, geomid)
                if geomid[0] < 0 or dist < 0.0:
                    break
                total = travelled + dist
                if model.geom_bodyid[geomid[0]] in exclude:
                    travelled = total + nudge
                    continue
                if total <= max_range:
                    ranges[i] = total
                break
    return ranges


def _multi_ray(model, data, origin, vecs, bodyexclude, geomids, dists, cutoff,
               geomgroup=None) -> None:
    """`mj_multiRay`, across the bindings' two signatures (3.5 added `normal`)."""
    flat = np.ascontiguousarray(vecs.reshape(-1), dtype=np.float64)
    n = len(geomids)
    if _MULTIRAY_HAS_NORMAL:
        mujoco.mj_multiRay(model, data, origin, flat, geomgroup, 1, bodyexclude, geomids, dists,
                           None, n, float(cutoff))
    else:
        mujoco.mj_multiRay(model, data, origin, flat, geomgroup, 1, bodyexclude, geomids, dists,
                           n, float(cutoff))


_MULTIRAY_HAS_NORMAL = "normal" in (mujoco.mj_multiRay.__doc__ or "").split(")")[0]


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
        #: The measured heading, unwrapped. The yaw hinge is continuous and callers read
        #: the heading off a rotation matrix, in (-pi, pi]: a target set from the wrapped
        #: reading after the robot had turned past pi -- on every stop, and whenever the
        #: lead clamp engaged -- was a whole turn away from the hinge, and the servo spun
        #: the robot round to reach it (a 69 deg left turn from 128 deg measured 124).
        self._heading: float | None = None

    def reset(self) -> None:
        """Forget everything: the world was reset, and the hinge with it."""
        self._target = None
        self._heading = None

    def hold(self) -> None:
        """Drop the target (the robot stops where it is), keeping track of its heading."""
        self._target = None

    def step(self, x: float, y: float, yaw: float,
             vx: float, vy: float, wz: float, dt: float) -> np.ndarray:
        """Advance the setpoint by one control period and return [x, y, yaw]."""
        if self._heading is None:
            self._heading = float(yaw)
        else:
            self._heading += float(np.arctan2(np.sin(yaw - self._heading),
                                              np.cos(yaw - self._heading)))
        yaw = self._heading
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
        yaw_lag = float(self._target[2] - yaw)
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
        else:
            # Three topics of the vendor contract leave the wire together here -- colour,
            # depth and camera_info, since depth renders through the same camera -- and
            # the only other trace is that `published` below gets shorter. A client that
            # expects a mobile base's camera (the console's own fleet check does, because
            # a real myAGV always publishes one) then reports it missing, which is right
            # but reads as a fault in the simulator rather than as something that was
            # asked for. So say it was asked for.
            # Named as they would have gone on the wire, not as the bare constants: two
            # bases on one graph would otherwise print the same sentence twice with
            # nothing to say which robot lost its camera.
            on_wire = getattr(server, "topic", lambda t: t)
            print(
                f"no colour camera on {on_wire(topics.camera)}: that topic, "
                f"{on_wire(topics.depth)} and {on_wire(topics.camera_info)} "
                "will not be published",
                file=sys.stderr,
            )

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

    def __init__(self, model, data, prefix: str = "", root: str = "base",
                 body: str | None = None) -> None:
        """`root` names the three joints (`base_x`...); `body` is the body they move.

        The two are the same on a wheeled base and are not on a legged one: the AiNex
        carries its planar joints on `body_link`, the torso the vendor URDF roots at.
        Conflating them is a robot that reports a base pose from whichever body happened
        to be called `base`.
        """
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

        body = body or root
        self._body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{body}")
        if self._body < 0:
            raise ValueError(f"no body named {prefix}{body!r}")
        # The joints are the pose only if the body's own frame is the world frame in x,
        # y and rotation. A pure z offset is safe and is how a legged robot is grafted:
        # its root is a torso standing a ride height above whatever it stands on, and the
        # slide joints are x/y only, so lifting cannot rotate or shift the axes the base
        # drives along. Anything else and a commanded +x is no longer world +x.
        offset = model.body_pos[self._body]
        quat = model.body_quat[self._body]
        if not (np.allclose(offset[:2], 0.0, atol=1e-9)
                and np.allclose(quat, [1, 0, 0, 0], atol=1e-9)):
            raise ValueError(
                f"{prefix}{body} is attached at pos={offset} quat={quat}; its "
                "world-aligned slide joints would no longer mean world x/y. Attach it "
                "over the origin -- a z offset is fine -- and set the pose instead."
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


class RenderWorker:
    """Renders cameras on a thread of its own, off the physics loop.

    A camera's cost is the GL render and readback (about 12 ms for a 640x480 view of an
    iTHOR kitchen, 17-20 ms with the whole fleet's cameras sharing the GPU), and paid on
    the loop thread it caps every topic on the port at the camera's pace. Here the loop
    thread only copies `MjData` into the worker's buffer (`submit`, which never blocks)
    and the worker renders, then hands each image to its job's callback -- which
    publishes, from the worker thread. The renderers are made on that thread, so their GL
    contexts belong to it. Verified on macOS under both `MUJOCO_GL=glfw` and `cgl`:
    physics and rendering overlap (MuJoCo releases the GIL).

    A worker still busy with the last frame refuses a submit, and the caller tries again
    on its next tick -- unless the submit passes `queue=True`, which parks that state in a
    second buffer (one deep) and renders it the moment the current frame is out. That is
    for a camera whose caller ticks slower than a frame takes: the SO-101's wrist camera
    is scheduled from the 50 Hz controller manager, so a 30 Hz stream needs frames 20 ms
    apart two ticks in every five, and a frame that takes 23 ms under a full fleet turned
    every such pair into 40 ms -- 26.7 Hz on the wire, outside the rate gate. Queued, the
    frame keeps the state and stamp of the tick it was due on and goes out a few
    milliseconds late, as a real camera's frames do; the long-run rate is the caller's.
    """

    def __init__(self, model, name: str = "render") -> None:
        import threading

        self._model = model
        self._copy = mujoco.MjData(model)
        self._spare = mujoco.MjData(model)
        self._jobs: list = []
        self._stamp = 0.0
        #: `(state, jobs, stamp)` of the one frame queued behind the current one.
        self._pending: tuple | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    @staticmethod
    def _capture(copy, data) -> None:
        # The state a render needs, not the whole arena: MuJoCo 3.3.1's bindings (the
        # RoboCasa venv) have no `mj_copyData`, and the kinematics are recomputed from
        # these on the worker thread, identically on both engines.
        copy.qpos[:] = data.qpos
        copy.qvel[:] = data.qvel
        copy.act[:] = data.act
        copy.mocap_pos[:] = data.mocap_pos
        copy.mocap_quat[:] = data.mocap_quat
        copy.time = data.time

    def submit(self, data, stamp_s: float, jobs, *, queue: bool = False) -> bool:
        """`jobs`: `(camera, width, height, scene_option, callback(rgb, stamp_s))`, or with a
        sixth element True, `callback(rgb, stamp_s, depth)` from one render (`render_rgbd`).

        True when the frame was taken: started now, or with `queue` parked behind the
        frame in progress. False when the worker cannot take it (busy, and either not
        queueing or with a frame already queued); the caller keeps it due.
        """
        if not jobs:
            return False
        with self._lock:
            if self.busy:
                if not queue or self._pending is not None:
                    return False
                self._capture(self._spare, data)
                self._pending = (self._spare, list(jobs), float(stamp_s))
                return True
            self._capture(self._copy, data)
            self._jobs, self._stamp = list(jobs), float(stamp_s)
            self._idle.clear()
            self._wake.set()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        """Until the frame in progress and any queued behind it are out."""
        return self._idle.wait(timeout)

    def _run(self) -> None:
        renderers: dict = {}
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                break
            while True:
                self._render(renderers)
                with self._lock:
                    if self._pending is None or self._stop:
                        self._jobs, self._pending = [], None
                        self._idle.set()
                        break
                    state, self._jobs, self._stamp = self._pending
                    self._pending = None
                    self._spare, self._copy = self._copy, state
        for renderer in renderers.values():
            renderer.close()

    def _render(self, renderers: dict) -> None:
        try:
            mujoco.mj_kinematics(self._model, self._copy)
            mujoco.mj_comPos(self._model, self._copy)
            mujoco.mj_camlight(self._model, self._copy)
        except Exception as exc:
            print(f"render worker: kinematics: {exc!r}", file=sys.stderr)
        for job in self._jobs:
            camera, width, height, scene_option, callback = job[:5]
            with_depth = len(job) > 5 and bool(job[5])
            try:
                renderer = renderers.get((width, height))
                if renderer is None:
                    vis = self._model.vis.global_
                    vis.offwidth = max(vis.offwidth, width)
                    vis.offheight = max(vis.offheight, height)
                    renderer = renderers[(width, height)] = mujoco.Renderer(
                        self._model, height, width)
                if scene_option is not None:
                    renderer.update_scene(self._copy, camera=camera,
                                          scene_option=scene_option)
                else:
                    renderer.update_scene(self._copy, camera=camera)
                if with_depth:
                    rgb, depth = render_rgbd(renderer, self._model)
                    callback(rgb, self._stamp, depth)
                else:
                    callback(renderer.render(), self._stamp)
            except Exception as exc:  # a failed frame must not kill the camera
                print(f"render worker: {camera}: {exc!r}", file=sys.stderr)

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2.0)


def render_rgbd(renderer, model) -> tuple[np.ndarray, np.ndarray]:
    """One render of `renderer`'s scene read back twice: colour (H, W, 3 uint8) and metric
    depth along the optical axis (H, W float32, metres), from the same frame.

    `mujoco.Renderer` reads one buffer per render, and a depth camera beside a colour
    one would pay the GL render twice. This is its own readback with both buffers, and its
    own reversed-Z conversion (`mujoco/renderer.py`, identical in 3.3 and 3.5).
    """
    width, height = renderer.width, renderer.height
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    raw = np.empty((height, width), dtype=np.float32)
    if renderer._gl_context:
        renderer._gl_context.make_current()
    mujoco.mjr_render(renderer._rect, renderer._scene, renderer._mjr_context)
    mujoco.mjr_readPixels(rgb, raw, renderer._rect, renderer._mjr_context)
    extent = model.stat.extent
    zfar = np.float32(model.vis.map.zfar * extent)
    znear = np.float32(model.vis.map.znear * extent)
    c_coef = -(zfar + znear) / (zfar - znear)
    d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)
    c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
    d_coef = np.float32(-0.5) * d_coef
    depth = (d_coef / (raw.astype(np.float64) + c_coef)).astype(np.float32)
    return np.ascontiguousarray(rgb[::-1]), np.ascontiguousarray(depth[::-1])


def _camera_frame(topic: str) -> str:
    """`/overhead/color/compressed` -> `overhead`; the contract's name for that view.

    `image_transport republish` appends `/color/compressed` to the camera's own name, so
    stripping that suffix recovers it -- the same rule a client uses to label a stream it
    discovered.
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
    what it is supposed to: the staging check holds the rig to framing every task object
    (spec §2.3) this way. A camera can be at exactly the right pose and still frame the
    wrong thing.

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


def run_sim_loop(model, data, controller, *, control_hz: float,
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

    **It warns when the real-time factor over a 10 s window falls below 0.90** (spec §4):
    simulated seconds over wall seconds, over consecutive `RTF_WINDOW_S` windows of wall
    clock, the measure the rate gate holds a run to.
    """
    # Hand the GIL back to this thread quickly. Every camera encodes and serialises on a
    # thread of its own, and at the default 5 ms switch interval this loop, returning from
    # an `mj_step` that released the GIL, could wait that long to get it back: a
    # so101,myagv,ainex kitchen measured 0.95-0.98 real time at 5 ms and 1.00 at 0.5 ms.
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(min(switch_interval, SIM_LOOP_SWITCH_INTERVAL_S))
    control_period = 1.0 / control_hz
    sync_period = 1.0 / sync_hz if sync_hz > 0 else None
    next_control = 0.0
    next_sync = 0.0
    wall_start = time.monotonic()
    sim_start = float(data.time)
    max_catchup = int(0.25 / model.opt.timestep)  # cap a stall at a quarter second
    behind_since = None
    rtf_wall, rtf_sim = wall_start, sim_start
    # The simulated clock the wire is stamped and scheduled with, moved on every physics
    # step rather than only when the controller runs (see `RobotFleet.advance_clock`).
    advance_clock = getattr(controller, "advance_clock", None)

    try:
        while viewer is None or viewer.is_running():
            now = time.monotonic()
            if controller is not None and now >= next_control:
                controller(data)
                # Drift-free: the next tick is one period after this one was *due*, not
                # after it ran, so the long-run rate is `control_hz` rather than a little
                # under it. A tick or two late is made up on the next passes; a loop
                # further behind than that re-anchors instead of bursting.
                next_control += control_period
                if next_control < now - 2 * control_period:
                    next_control = now + control_period

            target_time = sim_start + (time.monotonic() - wall_start)
            steps = 0
            while data.time < target_time and steps < max_catchup:
                mujoco.mj_step(model, data)
                steps += 1
                if advance_clock is not None:
                    advance_clock(float(data.time))
                # The controller keeps its cadence through a catch-up: a pass that
                # stepped physics until caught up after a slow tick (a camera frame)
                # would otherwise push the next tick back by the whole catch-up, and a
                # fast member's rate would fall with every frame rendered. Physics
                # resumes catching up on the next pass.
                if controller is not None and time.monotonic() >= next_control:
                    break

            if steps >= max_catchup or target_time - data.time > 0.25:
                # Fell more than the cap behind: rebase rather than chase forever.
                if behind_since is None:
                    behind_since = now
                elif now - behind_since > 5.0:
                    print(f"{label} cannot keep real time on this machine (physics + "
                          "cameras take longer than the wall clock)", file=sys.stderr)
                    behind_since = now
                wall_start = time.monotonic()
                sim_start = float(data.time)
            else:
                behind_since = None

            now = time.monotonic()
            if now - rtf_wall >= RTF_WINDOW_S:
                rtf = (float(data.time) - rtf_sim) / (now - rtf_wall)
                if rtf < RTF_WARN_BELOW:
                    print(f"warning: {label} ran at {rtf:.2f}x real time over the last "
                          f"{now - rtf_wall:.0f} s (below {RTF_WARN_BELOW:.2f}); every "
                          "topic's rate falls with it", file=sys.stderr)
                rtf_wall, rtf_sim = now, float(data.time)

            if viewer is not None and sync_period is not None:
                now = time.monotonic()
                if now >= next_sync:
                    viewer.sync()
                    next_sync = now + sync_period

            # Sleep only while the physics is ahead of the wall clock. Measured from where
            # the physics *is*, not from the target this pass aimed at: a pass that broke
            # off its catch-up for a controller tick is behind, and sleeping the rest of a
            # timestep there held a three-robot fleet at 0.92 real time (0.97-0.99 without).
            now = time.monotonic()
            slack = (float(data.time) - sim_start) - (now - wall_start)
            if controller is not None:
                slack = min(slack, next_control - now)
            if slack > 0:
                time.sleep(min(slack, control_period / 4))
    except KeyboardInterrupt:
        pass
    finally:
        sys.setswitchinterval(switch_interval)


#: The GIL switch interval while `run_sim_loop` runs; see there.
SIM_LOOP_SWITCH_INTERVAL_S = 0.0005
#: `run_sim_loop` warns when simulated over wall time across this window falls below
#: `RTF_WARN_BELOW` (spec §4).
RTF_WINDOW_S = 10.0
RTF_WARN_BELOW = 0.90


# ------------------------------------------------------------------ the transform tree


def _rel_pose(model, data, child: int, parent: int):
    """`child`'s pose in `parent`'s frame, as `(pos, quat)` with quat `(w, x, y, z)`.

    Read off `xpos`/`xmat` -- MuJoCo's own forward kinematics -- rather than composed out
    of joint angles. Composing them here would be a second kinematics implementation to
    keep in step with the first, and this project already has one lesson about a check
    that shared its measurement's method.
    """
    del model  # signature symmetry with the rest of this module; data carries it all
    parent_rot = data.xmat[parent].reshape(3, 3)
    pos = parent_rot.T @ (data.xpos[child] - data.xpos[parent])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(
        quat, np.ascontiguousarray(parent_rot.T @ data.xmat[child].reshape(3, 3)).flatten()
    )
    return pos, quat


def camera_link_pose(model, cam_id: int):
    """A MuJoCo camera's pose in its parent body, in the URDF link convention.

    MuJoCo cameras look down their own `-z` with `+y` up; a URDF camera link is the usual
    robot convention of `+x` forward, `+y` left, `+z` up. `ainex_model._reparent_camera`
    builds the camera's MuJoCo frame out of the link's axes that way round, and this is
    the exact inverse -- so a camera frame published here lands where the description says
    the camera is, and `shared/tests/tf_frames_check.py` holds the round trip to the
    vendor's own numbers.

    Publishing the MuJoCo frame raw instead is the failure that looks like a working
    system: every transform resolves, nothing errors, and the camera is drawn on its side.
    """
    cam_rot = np.zeros(9)
    mujoco.mju_quat2Mat(cam_rot, model.cam_quat[cam_id])
    cam_rot = cam_rot.reshape(3, 3)
    link_rot = np.column_stack([-cam_rot[:, 2], -cam_rot[:, 0], cam_rot[:, 1]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(link_rot).flatten())
    return np.asarray(model.cam_pos[cam_id], dtype=np.float64), quat


class TransformTree:
    """One robot's `/tf` and `/tf_static` content, read off the compiled model.

    A robot's frames come from three sources here, each the most honest one available for
    what it covers:

    * **the model's bodies**, for every link the arm or the legs actually move. MuJoCo's
      forward kinematics is what the simulation is stepping, so a transform read off
      `xpos`/`xmat` cannot disagree with the robot in the viewer.
    * **the model's cameras**, for the frames images are stamped with. The AiNex's camera
      is on the head in this model and on the torso in the vendor description (see
      `ainex_model` step 4), and it is the model that is right about the hardware -- so
      the frame follows the model, converted back to the link convention.
    * **the description's fixed joints**, passed in as `extra_static`, for links MuJoCo
      merged away. A fixed-jointed link carries no body, so `imu_link` and
      `gripper_frame_link` exist nowhere in the compiled model; a real
      `robot_state_publisher` reads those out of the URDF too.

    Frames are returned **bare**, without a namespace: `ros_surfaces/tf_stream.py` applies
    `bus.frame()` at the one point they reach the wire, exactly as topic names are handled.

    `frames` maps an MJCF body name -- with the engine's `robot_0/` prefix already
    stripped -- to the name the contract gives that frame. It is a map rather than the
    identity because the two genuinely differ on the SO-101 (`shoulder` in the official
    MJCF, `shoulder_link` in the official URDF a client renders from), and because a body
    the description does not have must not reach the wire at all: publishing one (the
    model used to carry menagerie's `camera_mount`) would leak this engine's model layout
    into a client's tf tree, which is the rule that already keeps a camera's `frame_id`
    off its MJCF camera name.

    The root's own transform is never published. A robot's root has no parent inside the
    robot: the myAGV's comes from its odometry (`odom -> base_footprint`, which is what a
    real `myagv_odometry_node` publishes), and the SO-101 and the AiNex have none at all,
    which is what a real bringup of either presents.
    """

    def __init__(self, model, *, root_body: str, frames: dict[str, str], prefix: str = "",
                 cameras: dict[str, str] | None = None, extra_static=()) -> None:
        self._model = model
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
        if root < 0:
            raise ValueError(f"tf: no body {root_body!r} in this model")
        self._root = root

        # body id -> frame, for the bodies this contract names. Anything else is skipped,
        # and a skipped body's children attach to its nearest *named* ancestor rather than
        # vanishing with it: that is what keeps the tree connected when a model carries a
        # body the description does not.
        self._frame_of: dict[int, str] = {}
        for body in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if not name.startswith(prefix):
                continue
            bare = name[len(prefix):]
            if bare in frames and self._descends_from_root(body):
                self._frame_of[body] = frames[bare]
        if root not in self._frame_of:
            raise ValueError(f"tf: {root_body!r} is the root but `frames` does not name it")

        static: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self._moving: list[tuple[int, int, str, str]] = []
        for body, frame in sorted(self._frame_of.items()):
            if body == root:
                continue
            parent = self._named_ancestor(body)
            if parent is None:
                continue
            if parent == model.body_parentid[body] and model.body_jntnum[body] == 0:
                # Welded straight to its named parent: the relative transform is in the
                # model itself, so it needs no MjData and can never change.
                static.append(
                    (self._frame_of[parent], frame,
                     np.asarray(model.body_pos[body], dtype=np.float64),
                     np.asarray(model.body_quat[body], dtype=np.float64))
                )
            else:
                self._moving.append((body, parent, self._frame_of[parent], frame))

        for cam_name, frame in (cameras or {}).items():
            cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{prefix}{cam_name}")
            if cam < 0:
                continue  # an engine that did not compile this camera; not an error
            parent = int(model.cam_bodyid[cam])
            if parent not in self._frame_of:
                continue  # a camera on a body this contract does not name
            pos, quat = camera_link_pose(model, cam)
            static.append((self._frame_of[parent], frame, pos, quat))

        for parent_frame, child_frame, pos, quat in extra_static:
            static.append((parent_frame, child_frame,
                           np.asarray(pos, dtype=np.float64),
                           np.asarray(quat, dtype=np.float64)))
        self._static = static

    def _descends_from_root(self, body: int) -> bool:
        while body > 0:
            if body == self._root:
                return True
            body = int(self._model.body_parentid[body])
        return body == self._root

    def _named_ancestor(self, body: int) -> int | None:
        parent = int(self._model.body_parentid[body])
        while parent > 0 and parent not in self._frame_of:
            parent = int(self._model.body_parentid[parent])
        return parent if parent in self._frame_of else None

    @property
    def root_frame(self) -> str:
        return self._frame_of[self._root]

    @property
    def frames(self) -> tuple[str, ...]:
        """Every frame this tree mentions, parents included. For tests and reports."""
        names = {self.root_frame}
        for parent, child, _, _ in self._static:
            names.update((parent, child))
        for _, _, parent, child in self._moving:
            names.update((parent, child))
        return tuple(sorted(names))

    def static(self):
        """The fixed transforms, computed once, as `(parent, child, pos, quat)`."""
        return list(self._static)

    def dynamic(self, data):
        """The transforms a joint can change, this tick."""
        return [
            (parent_frame, frame) + _rel_pose(self._model, data, body, parent)
            for body, parent, parent_frame, frame in self._moving
        ]
