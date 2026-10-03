"""The one function that stands a robot on the floor or on the worktop (spec §2.3, §4).

Both engines use it unchanged. Its checks are measured off the running world by ray
casting, so they need nothing an engine knows that the other does not:

* **Surface maps.** A grid of vertical rays, cast from above the room, gives the height of
  the first surface below. Geometry that lies wholly above the height band of interest
  (ceilings, wall cabinets, lamps) is left out of a map, so a map answers "what would a
  body of this height standing on this surface run into".
* **The floor** is the most common surface height near the lowest one. **The worktop**
  is the surface the reference project's survey ranks first (`worktop_survey.py`); a
  scene with none has no worktop.
* **A worktop robot** (`worktop_survey.MOUNT`: the arms) is tried first where the
  scene's six worktop objects are staged (`World.stage_scene`), which then stay
  untouched; if it does not fit there, at the survey's spots in their order -- the
  reference project's own choice first -- with the six objects (`worktop_objects.py`)
  staged around it instead and the scene's loose objects in its working area cleared.
  Every requirement below but travel still holds at the spot taken, with clearance judged
  on the robot's own collision geometry (not on boxes around it or its parts) and
  non-interpenetration also measured for the parts welded to the world (a fixed base,
  the staged plate), which make no contacts with fixed geometry; a staged object must
  stand on the surface without intersecting anything.
* **Any other placement** is a spot and heading tried in a deterministic order -- most
  clearance first; headings facing the longest run of open surface first -- until every
  applicable requirement holds:
  - support: the robot's support footprint lies wholly on the surface;
  - clearance: nothing but the surface lies inside the robot's footprint up to its
    height (loose objects and earlier robots included);
  - travel (every kind but `arm`, on the floor only: on the worktop a mobile robot has
    no travel requirement, so a top too small to walk on still takes it): the footprint swept 0.5 m forward, 0.25 m back and
    to each side, and the circle it sweeps turning in place, stay on the surface, clear
    of obstacles and of its unsupported edges;
  - camera clearance: a 7 x 5 grid of rays evenly spanning each camera's image frustum
    has no hit within 0.8 m of the lens except on the surface the robot stands on, its
    own body and the objects staged with it;
  - non-interpenetration: the robot compiled into the world at that pose (with its
    staging) has no contact deeper than 1 mm with anything but its support.
  The result depends only on the scene, the placement, the robot id and the world as it
  is (the robots already present, where the loose objects are). If nothing satisfies
  every requirement, the spawn is refused with a diagnostic naming what failed.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

import mjutil
import robot_model
import worktop_objects
import worktop_survey

RES = 0.03                     # surface-map cell (m)
LEVEL_TOL = 0.01               # neighbouring cells of one level surface differ by <= this
STATIC_CUTOFF = 1.15           # the static map holds fixed geometry starting this far up
CLEAR_ABOVE = 0.012            # something > this above the surface is an obstacle
STAGED_SUPPORT_TOL = 0.015     # a staged object's support is this close to the surface
EDGE_MARGIN = 0.03             # footprints keep this far from unsupported edges
CAMERA_CLEARANCE = 0.8
CAMERA_GRID = (7, 5)
TRAVEL = {"forward": 0.5, "back": 0.25, "left": 0.25, "right": 0.25}
PENETRATION = 0.001
MAX_SPOTS = 400
MAX_TRIALS = 6
MAX_WORKTOP_SPOTS = 600
MAX_WORKTOP_TRIALS = 40        # trial compiles of a worktop robot with its staging


class Refused(RuntimeError):
    """No placement satisfies every requirement; the message says why."""


NO_WORKTOP = ("no fixed counter or table top between 0.35 m and 1.30 m high for the "
              "worktop survey to stand a robot on")


@dataclass
class SurfaceMap:
    x0: float
    y0: float
    res: float
    height: np.ndarray        # (nx, ny) first-hit height, -inf where nothing was hit
    geom: np.ndarray          # (nx, ny) geom id of the first hit, -1 where none

    def cell(self, x, y):
        i = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        j = np.floor((np.asarray(y) - self.y0) / self.res).astype(int)
        return i, j

    def centre(self, i, j):
        return self.x0 + (np.asarray(i) + 0.5) * self.res, self.y0 + (np.asarray(j) + 0.5) * self.res

    def sample(self, pts: np.ndarray):
        """Heights and geoms at xy points (outside the map: -inf, -1)."""
        i, j = self.cell(pts[:, 0], pts[:, 1])
        ok = (i >= 0) & (j >= 0) & (i < self.height.shape[0]) & (j < self.height.shape[1])
        h = np.full(len(pts), -np.inf)
        g = np.full(len(pts), -1, int)
        h[ok] = self.height[i[ok], j[ok]]
        g[ok] = self.geom[i[ok], j[ok]]
        return h, g


@dataclass
class Placement:
    xyz: np.ndarray
    yaw: float
    surface: str
    surface_z: float
    support_geoms: tuple
    report: list = field(default_factory=list)
    #: the objects staged with a worktop robot and the scene objects cleared for them
    #: (`worktop_objects.Staging`), or None
    staging: object = None
    #: the robot stands where the scene's own worktop objects were staged (`World.stage_scene`)
    at_scene_objects: bool = False


# ---------------------------------------------------------------- geometry helpers


def _static(model) -> np.ndarray:
    return model.body_weldid[model.geom_bodyid] == 0


def _collidable(model) -> np.ndarray:
    return (model.geom_contype != 0) | (model.geom_conaffinity != 0)


PLANE_PAD = 2.0   # m of an unbounded floor plane mapped around the scene's other geometry


def scene_bounds(model, data) -> tuple:
    """The xy region the surface maps cover: every finite collidable geom, and a floor
    plane's own extent (an infinite plane: PLANE_PAD around everything else)."""
    lo, hi = mjutil.geom_aabbs(model, data)
    coll = _collidable(model)
    fin = np.isfinite(lo[:, 0]) & coll
    planes = coll & (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if fin.any():
        x0, y0 = float(lo[fin, 0].min()), float(lo[fin, 1].min())
        x1, y1 = float(hi[fin, 0].max()), float(hi[fin, 1].max())
    else:
        x0 = y0 = x1 = y1 = 0.0
    for g in np.nonzero(planes)[0]:
        sx, sy = model.geom_size[g][:2]
        c = data.geom_xpos[g]
        if sx > 0 and sy > 0:
            ex = np.abs(data.geom_xmat[g].reshape(3, 3)[:2, :2]) @ np.array([sx, sy])
            x0, x1 = min(x0, c[0] - ex[0]), max(x1, c[0] + ex[0])
            y0, y1 = min(y0, c[1] - ex[1]), max(y1, c[1] + ex[1])
        else:
            x0, y0, x1, y1 = x0 - PLANE_PAD, y0 - PLANE_PAD, x1 + PLANE_PAD, y1 + PLANE_PAD
    if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
        return (-3.0, -3.0, 3.0, 3.0)
    return (x0, y0, x1, y1)


def surface_map(model, data, include: np.ndarray, bounds, res: float = RES) -> SurfaceMap:
    """Heights of the first `include`d geom below each cell centre, rays from above."""
    m2 = copy.copy(model)
    m2.geom_group[:] = np.where(include, 0, 5)
    group = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    lo, hi = mjutil.geom_aabbs(model, data)
    top = float(np.max(hi[include & np.isfinite(hi[:, 2]), 2])) + 0.2 if include.any() else 3.0
    x0, y0, x1, y1 = bounds
    nx = max(1, int(math.ceil((x1 - x0) / res)))
    ny = max(1, int(math.ceil((y1 - y0) / res)))
    height = np.full((nx, ny), -np.inf)
    geom = np.full((nx, ny), -1, int)
    gid = np.zeros(1, np.int32)
    down = np.array([0.0, 0.0, -1.0])
    p = np.zeros(3)
    p[2] = top
    for i in range(nx):
        p[0] = x0 + (i + 0.5) * res
        for j in range(ny):
            p[1] = y0 + (j + 0.5) * res
            dist = mujoco.mj_ray(m2, data, p, down, group, 1, -1, gid)
            if dist >= 0:
                height[i, j] = top - dist
                geom[i, j] = gid[0]
    return SurfaceMap(x0, y0, res, height, geom)


class SceneSurfaces:
    """Floor level, the worktop and the static surface map of a scene, measured once
    on the bare scene (fixed geometry does not move). `facts` is the scene's survey input
    (`scenes.Scene.survey_facts`); without it the scene is surveyed by geometry alone."""

    def __init__(self, model, data, geomgroup=(1, 1, 1, 0, 0, 0), facts=None):
        # the bare scene, kept for the survey and for support rays
        self.model = model
        self.data = copy.copy(data)
        self.facts = facts or worktop_survey.Facts("generic", self.model, self.data)
        self._spots: dict = {}
        self._worktop_map = None
        self.bounds = scene_bounds(model, data)
        lo, _ = mjutil.geom_aabbs(model, data)
        coll, static = _collidable(model), _static(model)
        base = float(np.min(lo[coll & static, 2])) if (coll & static).any() else 0.0
        # First pass: find the floor under everything that starts within 1.5 m of the
        # lowest fixed geometry.
        # A surface only counts where its collision top is also a top the scene shows:
        # an invisible collider (e.g. a ground slab beyond a house's walls) is not a
        # floor to stand a robot on.
        shown = np.array([int(g) < len(geomgroup) and bool(geomgroup[int(g)])
                          for g in model.geom_group]) & (model.geom_rgba[:, 3] > 0.01)

        def maps(cutoff):
            cmap = surface_map(model, data, coll & static & (lo[:, 2] <= cutoff), self.bounds)
            vmap = surface_map(model, data, shown & static & (lo[:, 2] <= cutoff), self.bounds)
            with np.errstate(invalid="ignore"):
                ok = np.isfinite(cmap.height) & np.isfinite(vmap.height) & \
                    (np.abs(cmap.height - vmap.height) <= 0.03)
            return cmap, ok

        smap, ok = maps(base + 1.5)
        h = smap.height[ok]
        if h.size == 0:
            raise Refused("the scene has no surface to stand on")
        low = h[h <= h.min() + 0.3]
        vals, counts = np.unique(np.round(low, 2), return_counts=True)
        self.floor_z = float(vals[int(np.argmax(counts))])
        fsel = np.abs(h - self.floor_z) <= LEVEL_TOL
        self.floor_z = float(np.median(h[fsel]))
        # Second pass: fixed geometry up to just above table height. Cells whose top is
        # not shown become unsupported.
        self.static_map, ok = maps(self.floor_z + STATIC_CUTOFF)
        self.static_map.height[~ok] = -np.inf
        with np.errstate(all="ignore"):
            self.worktop = worktop_survey.worktop(self.facts)

    def spots(self, robot_id: str) -> list:
        """The survey's worktop spots for this robot, best first (cached: they depend on
        the bare scene and the robot only)."""
        if robot_id not in self._spots:
            with np.errstate(all="ignore"):
                self._spots[robot_id] = worktop_survey.candidates(self.facts, robot_id)
        return self._spots[robot_id]

    def worktop_map(self) -> SurfaceMap:
        """The worktop's support map, on the static map's grid: the first fixed collidable
        surface under each cell, cast from just above the worktop, over the worktop and
        0.6 m around it (-inf elsewhere)."""
        if self._worktop_map is None:
            wt, sm = self.worktop, self.static_map
            m, d = self.model, self.data
            caster = RayCaster(m, d, _collidable(m) & _static(m))
            height = np.full(sm.height.shape, -np.inf)
            geom = np.full(sm.height.shape, -1, int)
            lo = wt.rects[:, :2].min(axis=0) - 0.6
            hi = wt.rects[:, 2:4].max(axis=0) + 0.6
            i0, j0 = (int(v) for v in sm.cell(lo[0], lo[1]))
            i1, j1 = (int(v) for v in sm.cell(hi[0], hi[1]))
            down = np.array([0.0, 0.0, -1.0])
            for i in range(max(i0, 0), min(i1 + 1, sm.height.shape[0])):
                for j in range(max(j0, 0), min(j1 + 1, sm.height.shape[1])):
                    x, y = sm.centre(i, j)
                    p = np.array([x, y, wt.z + 0.05])
                    dist, g = caster.cast(p, down)
                    if g >= 0:
                        height[i, j] = p[2] - dist
                        geom[i, j] = g
            self._worktop_map = SurfaceMap(sm.x0, sm.y0, sm.res, height, geom)
        return self._worktop_map


# ---------------------------------------------------------------- the placement function


def _rot(yaw):
    return mjutil.yaw_matrix(yaw)[:2, :2]


def _rect_points(lo, hi, step=RES / 2):
    xs = np.arange(lo[0], hi[0] + 1e-9, step)
    ys = np.arange(lo[1], hi[1] + 1e-9, step)
    if xs.size == 0 or xs[-1] < hi[0] - 1e-6:
        xs = np.append(xs, hi[0])
    if ys.size == 0 or ys[-1] < hi[1] - 1e-6:
        ys = np.append(ys, hi[1])
    g = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    return g


def _disc_points(r, step=RES / 2):
    g = _rect_points((-r, -r), (r, r), step)
    return g[np.hypot(g[:, 0], g[:, 1]) <= r + 1e-9]


class Context:
    """What one placement decision reads off the world (under its lock)."""

    def __init__(self, world, surfaces: SceneSurfaces, rm, shape: robot_model.Shape,
                 mobile: bool):
        self.world = world
        self.surfaces = surfaces
        self.rm = rm
        self.shape = shape
        self.mobile = mobile
        with world.lock:
            self.model = world.model
            self.data = copy.copy(world.data)
            self.robot_worktop = [r.id for r in world.robots.values() if r.placement == "worktop"]
        self.height = max(float(shape.hi[2] - shape.lo[2]), 0.05)

    def full_map(self, surface_z: float) -> SurfaceMap:
        """Everything collidable -- fixed, loose objects and earlier robots -- that
        reaches into the band from the surface up to the robot's height."""
        m = self.model
        lo, _ = mjutil.geom_aabbs(m, self.data)
        inc = _collidable(m) & (lo[:, 2] <= surface_z + self.height + 0.03)
        return surface_map(m, self.data, inc, self.surfaces.bounds)


def _supported_free(fmap: SurfaceMap, smap: SurfaceMap, z: float, pts: np.ndarray,
                    support_geoms=None):
    """(supported, free) masks for world xy points on a surface at height z."""
    hs, gs = smap.sample(pts)
    hf, _ = fmap.sample(pts)
    supported = np.abs(hs - z) <= LEVEL_TOL
    if support_geoms is not None:
        supported &= np.isin(gs, list(support_geoms)) | (np.abs(hs - z) <= LEVEL_TOL)
    free = hf <= z + CLEAR_ABOVE
    return supported, free


class RayCaster:
    """Rays that see only the `accept`ed geoms and look through every other one -- also
    where an accepted geom's face coincides with a skipped one's (a collision box under
    its own visual mesh), which stepping `mj_ray` past the skipped hit would miss."""

    def __init__(self, m, d, accept, aabbs=None, region=None):
        """`aabbs`: the world AABBs (lo, hi) of every geom, when already computed;
        `region`: (lo, hi) corners outside which no ray of this caster will look."""
        self.m, self.d = m, d
        lo, hi = aabbs if aabbs is not None else mjutil.geom_aabbs(m, d)
        accept = np.asarray(accept, bool)
        if region is not None:
            accept = accept & np.all(lo <= region[1], axis=1) & np.all(hi >= region[0], axis=1)
        self.ids = np.nonzero(accept)[0]
        self.lo, self.hi = lo[self.ids], hi[self.ids]

    def cast(self, origin, direction):
        """(distance, geom) of the nearest accepted geom along the ray, or (-1, -1)."""
        p = np.asarray(origin, float)
        v = np.asarray(direction, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / v
            t1, t2 = (self.lo - p) * inv, (self.hi - p) * inv
        # a zero direction component: inside the slab or not, for any t
        par = v == 0
        tmin = np.where(par, np.where((p >= self.lo) & (p <= self.hi), -np.inf, np.inf),
                        np.minimum(t1, t2))
        tmax = np.where(par, np.where((p >= self.lo) & (p <= self.hi), np.inf, -np.inf),
                        np.maximum(t1, t2))
        near, far = tmin.max(axis=1), tmax.min(axis=1)
        cand = np.nonzero((far >= np.maximum(near, 0.0)))[0]
        best, best_g = np.inf, -1
        m, d = self.m, self.d
        for k in cand[np.argsort(near[cand])]:
            if near[k] > best:
                break
            g = int(self.ids[k])
            t = int(m.geom_type[g])
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                dist = mujoco.mj_rayMesh(m, d, g, p, v)
            elif t == mujoco.mjtGeom.mjGEOM_HFIELD:
                dist = mujoco.mj_rayHfield(m, d, g, p, v)
            elif t in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE,
                       mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                       mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX):
                dist = mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], p, v, t)
            else:
                continue
            if 0 <= dist < best:
                best, best_g = float(dist), g
        return (best, best_g) if best_g >= 0 else (-1.0, -1)


def _camera_rays(ctx: Context, xyz, yaw, surface_z, support_geoms, skip=frozenset()):
    """Hits within CAMERA_CLEARANCE of each camera lens: [(camera, geom name, dist)].
    `skip`: geoms the rays look through (objects about to be cleared from the spot)."""
    m, d = ctx.model, ctx.data
    R = mjutil.yaw_matrix(yaw)
    group = np.array([1, 1, 1, 1, 1, 1], np.uint8)
    gid = np.zeros(1, np.int32)
    caster = None
    if skip:
        accept = np.ones(m.ngeom, bool)
        accept[list(skip)] = False
        caster = RayCaster(m, d, accept)
    hits = []
    for name, cpos, cmat, fovy, res in ctx.shape.cameras:
        eye = np.asarray(xyz) + R @ cpos
        cm = R @ cmat
        w, h = (res if res and res[0] > 0 else (640, 480))
        tan_y = math.tan(math.radians(fovy) / 2)
        tan_x = tan_y * w / h
        nc, nr = CAMERA_GRID
        for a in np.linspace(-1, 1, nc):
            for b in np.linspace(-1, 1, nr):
                # MuJoCo camera: looks along -z, x right, y up.
                v = cm @ np.array([a * tan_x, b * tan_y, -1.0])
                v /= np.linalg.norm(v)
                if caster is None:
                    dist = mujoco.mj_ray(m, d, eye, v, group, 1, -1, gid)
                    g = int(gid[0])
                else:
                    dist, g = caster.cast(eye, v)
                if 0 <= dist <= CAMERA_CLEARANCE:
                    p = eye + v * dist
                    if g in support_geoms or abs(p[2] - surface_z) <= 0.015:
                        continue  # the surface it stands on
                    hits.append((name, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                                 or f"geom {g}", round(float(dist), 3)))
    return hits


def _travel_points(shape, radius):
    lo, hi = shape.lo[:2] - EDGE_MARGIN, shape.hi[:2] + EDGE_MARGIN
    parts = [_rect_points((lo[0], lo[1]), (hi[0] + TRAVEL["forward"], hi[1])),
             _rect_points((lo[0] - TRAVEL["back"], lo[1]), (hi[0], hi[1])),
             _rect_points((lo[0], lo[1] - TRAVEL["right"]), (hi[0], hi[1])),
             _rect_points((lo[0], lo[1]), (hi[0], hi[1] + TRAVEL["left"])),
             _disc_points(radius + EDGE_MARGIN)]
    return np.concatenate(parts)


def _open_run(fmap, smap, z, xy, yaw, limit=3.0):
    """How far open, supported surface runs from xy along the heading."""
    d = np.array([math.cos(yaw), math.sin(yaw)])
    n = int(limit / (RES / 2))
    pts = xy[None, :] + np.outer(np.arange(1, n + 1) * (RES / 2), d)
    sup, free = _supported_free(fmap, smap, z, pts)
    ok = sup & free
    bad = np.nonzero(~ok)[0]
    return (bad[0] if bad.size else n) * (RES / 2)


def _penetrations(world, rm, inst_prefix, xyz, yaw, support_geoms, surface_z, staging=None,
                  displace=None):
    """Contacts deeper than 1 mm between the robot, compiled in at the pose, and
    anything but its support; with a staging, also the staged objects' contacts deeper
    than 1 mm and any staged object not standing on the surface. The world itself is not
    changed: this is a trial compile of a copy."""
    spec = world.spec.copy()
    rspec = rm.spec.copy()
    from types import SimpleNamespace

    if displace is not None:
        worktop_objects.Staging(displace.frame_pos, displace.yaw).remove_objects(spec)
    if staging is not None:
        worktop_objects.Staging(staging.frame_pos, staging.yaw, staging.cleared).apply(spec)
    world._contact_bits(rspec, SimpleNamespace(info={}))
    frame = spec.worldbody.add_frame(pos=[float(v) for v in xyz],
                                     quat=mjutil.yaw_quat(yaw))
    frame.attach_body(rspec.body(rm.root), inst_prefix, "")
    m = spec.compile()
    d = mujoco.MjData(m)
    # carry the current state of everything already in the world
    with world.lock:
        wm, wd = world.model, world.data
        for j in range(wm.njnt):
            name = wm.joint(j).name
            if not name:
                continue
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if k < 0:
                continue
            n = {0: 7, 1: 4, 2: 1, 3: 1}[int(wm.jnt_type[j])]
            d.qpos[m.jnt_qposadr[k]:m.jnt_qposadr[k] + n] = wd.qpos[wm.jnt_qposadr[j]:wm.jnt_qposadr[j] + n]
    if staging is not None:
        # staged anew (moved, when the scene's own objects were displaced): at their poses
        worktop_objects.Staging(staging.frame_pos, staging.yaw).place_objects(m, d)
    for name, q in rm.home.items():
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, inst_prefix + name)
        if k >= 0:
            d.qpos[m.jnt_qposadr[k]] = q
    if rm.floating:
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, inst_prefix + "root")
        a = m.jnt_qposadr[k]
        d.qpos[a:a + 7] = rm.root_qpos(xyz, yaw)
    mujoco.mj_forward(m, d)
    own = mjutil.subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, inst_prefix + rm.root))
    support_names = {mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                     for g in support_geoms}
    staged = {}
    if staging is not None:
        for name, body in worktop_objects.BODIES.items():
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            if b >= 0:
                staged[b] = name
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
    found = []
    for c in range(d.ncon):
        con = d.contact[c]
        if con.dist > -PENETRATION:
            continue
        g1, g2 = int(con.geom1), int(con.geom2)
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        a, b = b1 in own, b2 in own
        if a != b:
            other = g2 if a else g1
            oname = gname(other)
            if oname in support_names or (abs(con.pos[2] - surface_z) <= 0.01
                                          and abs(con.frame[2]) > 0.9):
                continue  # intended support contact
            found.append(f"{oname} by {-con.dist * 1000:.1f} mm")
        elif (b1 in staged or b2 in staged) and b1 != b2:
            s, other = (b1, g2) if b1 in staged else (b2, g1)
            found.append(f"staged {staged[s]} into {gname(other)} by {-con.dist * 1000:.1f} mm")
    # Bodies welded to the world -- a fixed arm's base, the staged plate -- make no
    # contacts with the scene's fixed geometry, so their overlaps are measured directly.
    coll = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    welded = [g for g in range(m.ngeom) if coll[g] and m.body_weldid[m.geom_bodyid[g]] == 0
              and (m.geom_bodyid[g] in own or m.geom_bodyid[g] in staged)]
    if welded:
        lo, hi = mjutil.geom_aabbs(m, d)
        mine = own | set(staged)
        others = np.array([g for g in range(m.ngeom) if coll[g] and m.geom_bodyid[g] not in mine])
        fromto = np.zeros(6)
        for g in welded:
            near = others[np.all(lo[others] <= hi[g] + 0.002, axis=1)
                          & np.all(hi[others] >= lo[g] - 0.002, axis=1)] if len(others) else []
            for o in near:
                dist = mujoco.mj_geomDistance(m, d, int(g), int(o), 0.01, fromto)
                if dist > -PENETRATION:
                    continue
                oname = gname(int(o))
                zc = (fromto[2] + fromto[5]) / 2
                if oname in support_names or abs(zc - surface_z) <= 0.01:
                    continue  # intended support contact
                who = f"staged {staged[m.geom_bodyid[g]]}" if m.geom_bodyid[g] in staged else gname(g)
                found.append(f"{who} into {oname} by {-dist * 1000:.1f} mm")
    if staged:
        # every staged object stands on the surface: straight down from its origin, past
        # the staged objects and the robot, the first thing is the top face it stands on
        accept = np.ones(m.ngeom, bool)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in staged or m.geom_bodyid[g] in own or \
                    (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                accept[g] = False
        down = np.array([0.0, 0.0, -1.0])
        caster = RayCaster(m, d, accept)
        for b, name in staged.items():
            dist, g = caster.cast(d.xpos[b], down)
            top = d.xpos[b][2] - dist if g >= 0 else -np.inf
            if abs(top - surface_z) > STAGED_SUPPORT_TOL:
                found.append(f"staged {name} unsupported (the surface under it is at "
                             f"{top:.3f} m, not {surface_z:.3f} m)")
    return found


def _subtree_geoms(m, bodies) -> set:
    """Geoms of these bodies and every body inside them."""
    inside = set().union(*(mjutil.subtree(m, int(b)) for b in bodies))
    return {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in inside}


def _box_clearance(m, d, aabbs, coll, skip, boxes, xyz, yaw, surface_z):
    """The name of a collidable geom (other than `skip` and the surface at `surface_z`)
    inside one of the robot's part boxes (robot frame, `boxes`) with the robot at
    `xyz`/`yaw`, or None: vertical rays through each box, down from its top and up from
    its bottom, report anything they meet within it."""
    if boxes is None or not len(boxes):
        return None
    R = mjutil.yaw_matrix(yaw)
    corners = np.array([[sx, sy, sz] for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
    world = []
    for lo, hi in boxes:
        pts = lo + corners * (hi - lo)
        w = pts @ R.T + xyz
        world.append((w.min(axis=0), w.max(axis=0), lo, hi))
    region = (np.min([b[0] for b in world], axis=0) - 0.01,
              np.max([b[1] for b in world], axis=0) + 0.01)
    accept = coll.copy()
    if skip:
        accept[list(skip)] = False
    caster = RayCaster(m, d, accept, aabbs=aabbs, region=region)
    if not len(caster.ids):
        return None
    down, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
    for wlo, whi, lo, hi in world:
        bottom = max(float(wlo[2]), surface_z + CLEAR_ABOVE)
        top = float(whi[2])
        if top <= bottom:
            continue
        local = _rect_points(lo[:2], hi[:2])
        for p in local @ R[:2, :2].T + xyz[:2]:
            dist, g = caster.cast(np.array([p[0], p[1], top]), down)
            if g >= 0 and top - dist > bottom:
                return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
            dist, g = caster.cast(np.array([p[0], p[1], bottom]), up)
            if g >= 0 and bottom + dist < top:
                return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
    return None


def _place_on_spots(world, surfaces, ctx, rm, shape, spots, inst_prefix) -> Placement:
    """`_place_on_spots_at`, first where the scene's own worktop objects are staged and,
    if the arm does not fit there, at its survey spots with the objects moved to it."""
    scene = world.scene_staging
    if scene is None:
        return _place_on_spots_at(world, surfaces, ctx, rm, shape, spots, inst_prefix)
    try:
        return _place_on_spots_at(world, surfaces, ctx, rm, shape, spots, inst_prefix,
                                  fixed=scene)
    except Refused:
        return _place_on_spots_at(world, surfaces, ctx, rm, shape, spots, inst_prefix,
                                  displace=scene)


def _place_on_spots_at(world, surfaces, ctx, rm, shape, spots, inst_prefix, fixed=None,
                       displace=None) -> Placement:
    """A worktop robot at the survey's spots, in order, with its objects staged: the first
    spot where the robot's base is supported by the top face, its cameras are clear, and
    the trial compile (robot and staging) finds no interpenetration of its own collision
    geometry and every staged object standing on the surface."""
    m, d = ctx.model, ctx.data
    rid = rm.robot.id
    robots = set()
    for inst in list(world.robots.values()):
        robots |= world.robot_body_ids(inst)
    fp = shape.footprint
    support_local = _rect_points(fp.min(axis=0), fp.max(axis=0))
    static_rays = RayCaster(surfaces.model, surfaces.data,
                            _collidable(surfaces.model) & _static(surfaces.model))
    down = np.array([0.0, 0.0, -1.0])
    reasons: dict[str, int] = {}
    first_detail: dict[str, str] = {}

    def fail(kind, detail=""):
        reasons[kind] = reasons.get(kind, 0) + 1
        first_detail.setdefault(kind, detail)

    # With the worktop objects part of the scene (`World.stage_scene`), the arm first
    # stands where they are: the one spot, and nothing to stage or clear. `displace` is the
    # scene's objects when the arm stands elsewhere instead, and they move to it. Either
    # way they are the arm's own worktop objects, never a camera obstacle (spec §2.3).
    objects: set = set()
    if world.scene_staging is not None:
        objects = _subtree_geoms(m, [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                                     for b in worktop_objects.BODIES.values()])
    if fixed is not None:
        spots = [worktop_survey.Spot(
            xy=np.array(fixed.frame_pos[:2], float),
            z=fixed.frame_pos[2] - worktop_objects.FRAME_ABOVE_SURFACE, yaw=fixed.yaw,
            surface=spots[0].surface if spots else "worktop",
            why="where the scene's worktop objects are staged")]
    trials = 0
    for spot in spots[:MAX_WORKTOP_SPOTS]:
        xy, z, yaw = np.asarray(spot.xy, float), float(spot.z), float(spot.yaw)
        where = f"({xy[0]:.2f}, {xy[1]:.2f})"
        R = _rot(yaw)
        # support: the fixed top face at z under the whole base
        support, ok = set(), True
        for p in xy + support_local @ R.T:
            dist, g = static_rays.cast(np.array([p[0], p[1], z + 0.05]), down)
            if g < 0 or abs(z + 0.05 - dist - z) > LEVEL_TOL:
                ok = False
                break
            support.add(g)
        if not ok:
            fail("support", f"footprint unsupported at {where}")
            continue
        frame_pos, frame_yaw = worktop_objects.frame_of(xy, z, yaw)
        cleared = ([] if fixed is not None
                   else worktop_objects.plan_clear(m, d, frame_pos, frame_yaw, keep=robots))
        # the cameras look past the six objects and what is about to be cleared; clearance
        # is judged on the robot's own collision geometry, by the trial compile below
        # (spec §2.3), not on boxes around its parts, which take the room beside a part
        skip = objects | _subtree_geoms(m, [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                                            for n in cleared])
        xyz = np.array([xy[0], xy[1], z - shape.lo[2] + 0.0005])
        hits = _camera_rays(ctx, xyz, yaw, z, support, skip)
        if hits:
            fail("camera clearance", f"camera {hits[0][0]} sees {hits[0][1]} at "
                 f"{hits[0][2]:.2f} m from {where}")
            continue
        staging = (None if fixed is not None
                   else worktop_objects.Staging(frame_pos, frame_yaw, cleared))
        trials += 1
        pen = _penetrations(world, rm, inst_prefix, xyz, yaw, support, z, staging, displace)
        if pen:
            fail("interpenetration", f"at {where}: " + pen[0])
            if trials >= MAX_WORKTOP_TRIALS:
                break
            continue
        report = (f"worktop spot ({xy[0]:.3f}, {xy[1]:.3f}) z {z:.3f} on {spot.surface}, "
                  f"heading {math.degrees(yaw):.1f} deg ({spot.why}); "
                  + (f"the scene's {len(worktop_objects.OBJECTS)} worktop objects are there"
                     if fixed is not None else
                     f"staged {len(worktop_objects.OBJECTS)} objects, cleared "
                     f"{', '.join(cleared) if cleared else 'nothing'}"))
        return Placement(xyz, yaw, "worktop", z, tuple(sorted(support)), report=[report],
                         staging=staging, at_scene_objects=fixed is not None)
    detail = "; ".join(f"{k}: {n} candidate(s) failed, e.g. {first_detail[k]}"
                       for k, n in sorted(reasons.items(), key=lambda kv: -kv[1]))
    if set(reasons) <= {"interpenetration", "support"}:
        raise Refused(f"placement refused: no clear spot on the worktop for {rid} "
                      f"({detail or 'no candidate'})")
    raise Refused(f"placement refused: no worktop placement of {rid} satisfies support, "
                  f"clearance and camera clearance ({detail or 'no candidate'})")


def place(world, surfaces: SceneSurfaces, rm, placement: str, mobile: bool,
          inst_prefix: str, spots=None) -> Placement:
    """Where and facing which way the robot stands, or `Refused` saying why not. `spots`
    replaces the survey's worktop spots for a robot that has a mount (scene start-up
    tries one candidate spot at a time)."""
    # numpy on macOS Accelerate raises spurious floating-point warnings in small matmuls
    with np.errstate(all="ignore"):
        return _place(world, surfaces, rm, placement, mobile, inst_prefix, spots)


def _place(world, surfaces, rm, placement, mobile, inst_prefix, spots=None) -> Placement:
    shape = robot_model.shape(rm)
    ctx = Context(world, surfaces, rm, shape, mobile)
    smap = surfaces.static_map
    if placement == "worktop":
        wt = surfaces.worktop
        if wt is None:
            raise Refused("placement refused: this scene has no worktop (" + NO_WORKTOP + ")")
        if ctx.robot_worktop:
            raise Refused(f"placement refused: the worktop already holds "
                          f"{', '.join(ctx.robot_worktop)}")
        if spots is None:
            spots = surfaces.spots(rm.robot.id)
        if spots:
            return _place_on_spots(world, surfaces, ctx, rm, shape, spots, inst_prefix)
        # a robot the survey has no mount for (a mobile robot): the general search below,
        # over the worktop's own top face
        z = wt.z
        smap = surfaces.worktop_map()
        region = np.isfinite(smap.height) & (np.abs(smap.height - z) <= LEVEL_TOL)
        ci, cj = np.nonzero(region)
        xs, ys = smap.centre(ci, cj)
        on_top = np.array([wt.contains((x, y)) for x, y in zip(xs, ys)], bool)
        region[ci[~on_top], cj[~on_top]] = False
        support = frozenset(int(g) for g in smap.geom[region] if g >= 0)
    elif placement == "floor":
        z = surfaces.floor_z
        region = np.isfinite(smap.height) & (np.abs(smap.height - z) <= LEVEL_TOL)
        support = frozenset(int(g) for g in smap.geom[region] if g >= 0)
    else:
        raise Refused(f"unknown placement {placement!r} (worktop or floor)")

    fmap = ctx.full_map(z)
    # cells of the surface that are clear right now
    free = region & (fmap.height <= z + CLEAR_ABOVE)
    if not free.any():
        if placement == "worktop":
            raise Refused("placement refused: no clear spot on the worktop -- it is "
                          "covered by objects")
        raise Refused("placement refused: no clear floor")
    # clearance of each free cell to the nearest non-free cell (distance transform)
    from scipy.ndimage import distance_transform_edt

    padded = np.pad(free, 1, constant_values=False)
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * smap.res
    ii, jj = np.nonzero(free)
    order = sorted(range(len(ii)), key=lambda k: (-round(float(clearance[ii[k], jj[k]]), 3),
                                                  int(ii[k]), int(jj[k])))
    order = order[:MAX_SPOTS]

    foot_lo, foot_hi = shape.lo[:2], shape.hi[:2]
    if mobile:
        support_pts_local = _rect_points(foot_lo, foot_hi)
    else:
        # an arm stands on its base: the lowest geoms carry it
        fp = shape.footprint
        support_pts_local = _rect_points(fp.min(axis=0), fp.max(axis=0))
    body_pts_local = _rect_points(foot_lo, foot_hi)
    body_aabbs = mjutil.geom_aabbs(ctx.model, ctx.data) if placement == "worktop" else None
    body_coll = _collidable(ctx.model) if placement == "worktop" else None
    # a worktop crowded with the scene's objects needs more trial compiles than the floor
    max_trials = MAX_WORKTOP_TRIALS if placement == "worktop" else MAX_TRIALS
    # the scene's worktop objects are not a camera obstacle for a robot on the worktop
    camera_skip = frozenset()
    if placement == "worktop" and world.scene_staging is not None:
        m = ctx.model
        camera_skip = frozenset(_subtree_geoms(
            m, [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in worktop_objects.BODIES.values()]))
    # travel is guaranteed on the floor only: on the worktop a mobile robot is placed on
    # a clear, supported spot, whatever room the top leaves it to move in
    check_travel = mobile and placement == "floor"
    travel_local = _travel_points(shape, shape.radius) if check_travel else None
    min_needed = (math.hypot(*(foot_hi - foot_lo)) / 2) if not mobile else shape.radius

    reasons: dict[str, int] = {}
    first_detail: dict[str, str] = {}

    def fail(kind, detail=""):
        reasons[kind] = reasons.get(kind, 0) + 1
        first_detail.setdefault(kind, detail)

    trials = 0
    headings = [k * math.pi / 8 for k in range(16)]
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if clearance[i, j] + smap.res < min(min_needed, 0.05):
            fail("support", "no clear patch large enough for the footprint")
            continue
        xy = np.array(smap.centre(i, j), float)
        ranked = sorted(headings, key=lambda yw: (-round(_open_run(fmap, smap, z, xy, yw), 2), yw))
        for yaw in ranked[:8]:
            Rm = _rot(yaw)
            sp = xy + support_pts_local @ Rm.T
            sup, _ = _supported_free(fmap, smap, z, sp)
            if not sup.all():
                fail("support", f"footprint unsupported at ({xy[0]:.2f}, {xy[1]:.2f})")
                continue
            bp = xy + body_pts_local @ Rm.T
            _, fr = _supported_free(fmap, smap, z, bp)
            if not fr.all():
                fail("interpenetration", f"obstacle inside the footprint at ({xy[0]:.2f}, {xy[1]:.2f})")
                continue
            if check_travel:
                tp = xy + travel_local @ Rm.T
                s2, f2 = _supported_free(fmap, smap, z, tp)
                if not s2.all():
                    fail("travel support", "travel paths leave the supported surface "
                         f"at ({xy[0]:.2f}, {xy[1]:.2f})")
                    continue
                if not f2.all():
                    fail("travel clearance", f"an obstacle is on the travel paths at "
                         f"({xy[0]:.2f}, {xy[1]:.2f})")
                    continue
            xyz = np.array([xy[0], xy[1], z - shape.lo[2] + 0.0005])
            if placement == "worktop":
                # the boxes of the robot's parts against what stands on the top (the scene's
                # objects reach beyond the footprint rectangle's cells): no trial compile
                # is spent on a spot a part of the robot already runs into
                blocked = _box_clearance(ctx.model, ctx.data, body_aabbs, body_coll, (),
                                         shape.boxes, xyz, yaw, z)
                if blocked:
                    fail("interpenetration", f"{blocked} inside the robot at "
                         f"({xy[0]:.2f}, {xy[1]:.2f})")
                    continue
            hits = _camera_rays(ctx, xyz, yaw, z, support, camera_skip)
            if hits:
                fail("camera clearance", f"camera {hits[0][0]} sees {hits[0][1]} at "
                     f"{hits[0][2]:.2f} m")
                continue
            trials += 1
            pen = _penetrations(world, rm, inst_prefix, xyz, yaw, support, z)
            if pen:
                fail("interpenetration", "intersects " + pen[0])
                if trials >= max_trials:
                    break
                continue
            return Placement(xyz, float(yaw), placement, z, tuple(sorted(support)),
                             report=[f"{placement} spot ({xy[0]:.3f}, {xy[1]:.3f}) z "
                                     f"{z:.3f}, heading {math.degrees(yaw):.1f} deg, "
                                     f"clearance {clearance[i, j]:.2f} m"])
        if trials >= max_trials:
            break
    detail = "; ".join(f"{k}: {n} candidate(s) failed, e.g. {first_detail[k]}"
                       for k, n in sorted(reasons.items(), key=lambda kv: -kv[1]))
    if placement == "worktop" and set(reasons) <= {"interpenetration", "support"}:
        raise Refused("placement refused: no clear spot on the worktop for "
                      f"{rm.robot.id} ({detail or 'every spot is taken'})")
    raise Refused(f"placement refused: no {placement} placement of {rm.robot.id} satisfies "
                  f"support, clearance, travel and camera clearance ({detail or 'no candidate'})")
