"""Rendering and sampling: the simulation's cameras, lidars, joint and pose readings.

All rendering happens here, in the simulation process (spec §2.2). A small pool of render
threads, each with its own offscreen OpenGL context and MuJoCo render context, renders
robot cameras for their wires and scene views for the control port. The scene is copied
into the thread's `MjvScene` under the world lock; the GPU work happens outside it.

Samples are stamped with the wall-clock time at which the simulation state was read
(their acquisition time), never with the time they are sent.
"""

from __future__ import annotations

import io
import math
import queue
import threading
from dataclasses import dataclass

import mujoco
import numpy as np

import mjutil

MAX_GEOM = 20000
# Largest near clipping plane of a model (robot) camera, metres. MuJoCo puts the near plane
# at vis.map.znear x stat.extent, and the extent of a composed household world is tens of
# metres, which put it several centimetres in front of the lens and cut away whatever is
# that close (the SO-101 wrist camera lost its own gripper jaws). A real camera has no near
# plane; 1 cm keeps everything a robot camera sees of its own body.
CAMERA_NEAR = 0.01


def clamp_near(scn, near: float) -> None:
    """Move the scene cameras' near plane to at most `near`, keeping the field of view (the
    frustum's sides are given at the near plane, so they scale with it)."""
    for eye in range(2):
        c = scn.camera[eye]
        if c.orthographic or c.frustum_near <= near:
            continue
        s = near / c.frustum_near
        c.frustum_near = near
        c.frustum_top *= s
        c.frustum_bottom *= s
        c.frustum_center *= s
        c.frustum_width *= s


def free_camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def camera_from_view(view: dict, model) -> tuple[mujoco.MjvCamera, float | None]:
    """An MjvCamera from a control-port viewpoint:

    * `{"camera": "<model camera name>"}` -- a camera of the model (robot cameras are
      `<robot id>/<frame id>`);
    * `{"lookat": [x,y,z], "distance": d, "azimuth": deg, "elevation": deg}`;
    * `{"pos": [x,y,z], "target": [x,y,z]}` -- eye and point looked at.

    Returns the camera and an optional vertical field of view (degrees) override."""
    fovy = view.get("fovy")
    if "camera" in view:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view["camera"])
        if cid < 0:
            raise KeyError(f"no camera {view['camera']!r}")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cid
        return cam, None
    if "pos" in view:
        pos = np.asarray(view["pos"], float)
        target = np.asarray(view.get("target", view.get("lookat")), float)
        v = target - pos
        dist = float(np.linalg.norm(v)) or 1e-3
        az = math.degrees(math.atan2(v[1], v[0]))
        el = math.degrees(math.asin(max(-1.0, min(1.0, v[2] / dist))))
        return free_camera(target, dist, az, el), fovy
    if "lookat" in view:
        return free_camera(view["lookat"], view.get("distance", 3.0),
                           view.get("azimuth", 90.0), view.get("elevation", -30.0)), fovy
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    return cam, fovy


@dataclass
class RenderJob:
    camera: dict                 # viewpoint (see camera_from_view)
    width: int
    height: int
    rgb: bool = True
    depth: bool = False
    callback: object = None      # fn(result dict | Exception)


class RenderThread(threading.Thread):
    def __init__(self, pool, index: int):
        super().__init__(name=f"render-{index}", daemon=True)
        self.pool = pool
        self.world = pool.world
        self.ctx = None
        self.con = None
        self.scn = None
        self.version = -1
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0   # textures, not island colours

    def _ensure(self, model, version):
        if version == self.version:
            return model
        from mujoco import gl_context

        import copy

        w, h = self.pool.max_size
        # The offscreen buffer is sized for this pool (a camera stream's own resolution,
        # or the largest control-port render); set on a copy so the world's model is
        # never touched from here.
        sized = copy.copy(model)
        sized.vis.global_.offwidth, sized.vis.global_.offheight = w, h
        if self.ctx is None:
            self.ctx = gl_context.GLContext(w, h)
        self.ctx.make_current()
        if self.con is not None:
            self.con.free()
        self.con = mujoco.MjrContext(sized, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.con)
        self.con.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        self.scn = mujoco.MjvScene(model, maxgeom=MAX_GEOM)
        self.version = version
        self._model = model
        return model

    def run(self):
        while True:
            job = self.pool.jobs.get()
            if job is None:
                try:
                    if self.con is not None:
                        self.ctx.make_current()
                        self.con.free()
                    if self.ctx is not None:
                        self.ctx.free()
                except Exception:
                    pass
                return
            try:
                result = self.render(job)
            except Exception as exc:  # reported to the requester
                result = exc
            try:
                job.callback(result)
            except Exception:
                pass

    def render(self, job: RenderJob) -> dict:
        # The world's configuration now (acquisition time), rebuilt on this thread's own
        # MjData: the physics thread only copies a few arrays.
        snap = self.world.snapshot()
        model = self._ensure(snap["model"], snap["version"])
        self.ctx.make_current()
        if not hasattr(self, "mirror"):
            from world import Mirror

            self.mirror = Mirror()
        model, data = self.mirror.update(snap)
        stamp, simtime = snap["stamp"], snap["time"]
        cam, fovy = camera_from_view(job.camera, model)
        gg = self.world.scene.geomgroup
        for i in range(6):
            self.opt.geomgroup[i] = int(gg[i]) if i < len(gg) else 0
        mujoco.mjv_updateScene(model, data, self.opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self.scn)
        if cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            clamp_near(self.scn, CAMERA_NEAR)
        if fovy is not None:  # a viewpoint's own vertical field of view
            for eye in range(2):
                c = self.scn.camera[eye]
                c.frustum_top = c.frustum_near * math.tan(math.radians(float(fovy)) / 2)
                c.frustum_bottom = -c.frustum_top
        rect = mujoco.MjrRect(0, 0, job.width, job.height)
        mujoco.mjr_render(rect, self.scn, self.con)
        rgb = np.empty((job.height, job.width, 3), np.uint8) if job.rgb else None
        depth = np.empty((job.height, job.width), np.float32) if job.depth else None
        mujoco.mjr_readPixels(rgb, depth, rect, self.con)
        out = {"stamp": stamp, "sim_time": simtime}
        if rgb is not None:
            out["rgb"] = np.flipud(rgb)
        if depth is not None:
            # the planes this frame was rendered with (vis.map.znear/zfar x stat.extent,
            # or the clamped near plane of a model camera)
            znear = np.float32(self.scn.camera[0].frustum_near)
            zfar = np.float32(self.scn.camera[0].frustum_far)
            c = -(zfar + znear) / (zfar - znear)
            dd = -(np.float32(2) * zfar * znear) / (zfar - znear)
            c = np.float32(-0.5) * c - np.float32(0.5)
            dd = np.float32(-0.5) * dd
            out["depth"] = np.flipud((dd / (depth.astype(np.float64) + c)).astype(np.float32))
        return out


class RenderPool:
    def __init__(self, world, threads: int = 2, max_size=(1920, 1080)):
        self.world = world
        self.max_size = max_size
        self.jobs: queue.Queue = queue.Queue()
        self.threads = [RenderThread(self, i) for i in range(threads)]
        for t in self.threads:
            t.start()

    def submit(self, job: RenderJob) -> None:
        self.jobs.put(job)

    def close(self) -> None:
        for _ in self.threads:
            self.jobs.put(None)

    def render_sync(self, job: RenderJob, timeout: float = 30.0) -> dict:
        box: queue.Queue = queue.Queue(1)
        job.callback = box.put
        self.submit(job)
        res = box.get(timeout=timeout)
        if isinstance(res, Exception):
            raise res
        return res


def png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- readings


def joint_readings(world, inst) -> dict:
    """{joint: [position, velocity, effort]} of the robot's hinge and slide joints, by
    their names in the robot's model. Effort is the actuator force at the joint."""
    m, d = world.model, world.data
    out = {}
    for name, j in world.robot_elements(inst, mujoco.mjtObj.mjOBJ_JOINT):
        if m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        out[name] = [float(d.qpos[qa]), float(d.qvel[va]), float(d.qfrc_actuator[va])]
    return out


def body_pose(world, body_id: int) -> dict:
    d = world.data
    return {"pos": [float(x) for x in d.xpos[body_id]],
            "quat": [float(x) for x in d.xquat[body_id]]}


def base_reading(world, inst) -> dict:
    """World pose of the robot's top body, its world linear velocity and its angular
    velocity in its own frame."""
    m, d = world.model, world.data
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, inst.prefix + inst.rm.root)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
    lvel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, lvel, 1)
    out = body_pose(world, b)
    out["linvel_world"] = [float(x) for x in vel[3:]]
    out["angvel_world"] = [float(x) for x in vel[:3]]
    out["linvel_local"] = [float(x) for x in lvel[3:]]
    out["angvel_local"] = [float(x) for x in lvel[:3]]
    return out


def site_readings(world, inst, names, imu: bool = False) -> dict:
    """Pose (and, for IMUs, angular velocity and specific force in the site frame)."""
    m, d = world.model, world.data
    out = {}
    need_acc = imu
    if need_acc:
        mujoco.mj_rnePostConstraint(m, d)
    for n in names:
        s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, inst.prefix + n)
        if s < 0:
            continue
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.site_xmat[s])
        row = {"pos": [float(x) for x in d.site_xpos[s]], "quat": [float(x) for x in q]}
        if imu:
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, s, v, 1)
            a = np.zeros(6)
            mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_SITE, s, a, 1)
            row["gyro"] = [float(x) for x in v[:3]]
            row["accel"] = [float(x) for x in a[3:]]
        out[n] = row
    return out


def lidar_scan(m, d, inst, site: str, angle_min: float, angle_max: float, samples: int,
               range_min: float, range_max: float, own=None) -> np.ndarray:
    """Planar ranges from the lidar site's frame (x forward, z up), counter-clockwise
    from angle_min, hitting everything but the robot's own bodies; inf beyond range.
    `m, d` may be a Mirror of the world (the scan then runs off the physics thread)."""
    s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, inst.prefix + site)
    if s < 0:
        raise KeyError(f"no lidar site {site!r}")
    R = d.site_xmat[s].reshape(3, 3)
    origin = d.site_xpos[s].copy()
    ang = angle_min + np.arange(samples) * ((angle_max - angle_min) / max(samples - 1, 1)) \
        if samples > 1 else np.array([angle_min])
    local = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    vec = (local @ R.T).astype(np.float64)
    geomid = np.zeros(samples, np.int32)
    dist = np.zeros(samples, np.float64)
    # Exclude this robot's own geometry: rays start at its root body's tree. mj_multiRay
    # excludes one body; own-body hits are filtered by walking past them.
    if own is None:
        own = mjutil.subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                                  inst.prefix + inst.rm.root))
    group = np.array([1, 1, 1, 1, 1, 1], np.uint8)
    try:  # MuJoCo >= 3.5 adds a `normal` output before nray
        mujoco.mj_multiRay(m, d, origin, vec.flatten(), group, 1, -1, geomid, dist, None,
                           samples, range_max + 0.5)
    except TypeError:
        mujoco.mj_multiRay(m, d, origin, vec.flatten(), group, 1, -1, geomid, dist, samples,
                           range_max + 0.5)
    for i in np.nonzero(geomid >= 0)[0]:
        guard = 0
        o = origin.copy()
        acc = 0.0
        while geomid[i] >= 0 and m.geom_bodyid[geomid[i]] in own and guard < 8:
            step = dist[i] + 1e-4
            acc += step
            o = o + vec[i] * step
            gid = np.zeros(1, np.int32)
            dd = mujoco.mj_ray(m, d, o, vec[i], group, 1, int(m.geom_bodyid[geomid[i]]), gid)
            geomid[i], dist[i] = gid[0], dd
            guard += 1
        if geomid[i] >= 0:
            dist[i] += acc
    ranges = np.where(geomid >= 0, dist, np.inf).astype(np.float32)
    ranges[(ranges < range_min)] = np.inf
    ranges[(ranges > range_max)] = np.inf
    return ranges
