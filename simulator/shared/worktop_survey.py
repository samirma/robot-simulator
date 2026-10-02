"""The worktop survey: which surface is the worktop, and the spots on it a worktop robot
is tried at, best first (spec §2.2, §2.3).

These are the reference project's own algorithms (github.com/samirma/robot-simulator rev
34547ae, "REF" below), ported with their constants so the robot stands where the
reference stood it:

* **iTHOR / ProcTHOR, and the `test` scene** -- the tabletop survey of
  `simulator/molmospaces/tools/scene_placement.py` (`find_grasp_targets`, `find_supports`,
  `find_tabletop_mount` and their helpers, lines 119-756) driven as
  `simulator/molmospaces/tools/spawn_robot.py:170-321` (`place_arm_on_table`) drove it:
  surfaces holding the task's own pair of categories (`TARGET`) rank first, and the arm
  stands at the rim of the chosen surface, looking in, with as much of its forward
  workspace over the worktop as any heading gives. A scene with no graspable objects (the
  `test` scene, and RoboCasa's fallback below) is surveyed against `N_SPAWN` predicted
  stand-in objects on its largest table-height top, exactly as REF did.
* **RoboCasa** -- `simulator/robocasa/tools/spawn_robot.py:104-279` (`world_boxes`,
  `clearance_field`, `counter_regions`, `outward_direction`, `find_counter_mount`): the
  arm stands against the back edge of the roomiest counter region RoboCasa itself
  reports, facing the room. After that spot come REF's tabletop coverage candidates over
  the counter tops (what the myCobot 280 stands on: on RoboCasa its back-edge spot
  intersects the wall cabinet above the counter).

What a scene knows and this module cannot -- the THOR metadata, occupancy map and type
sets, RoboCasa's counter regions -- comes in as `Facts` (`scenes.Scene.survey_facts`). The
survey only ranks spots; `placement.place` tries them in this order and keeps every
safety check of spec §2.3. Candidate 0 is REF's choice.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

#: The object categories that rank the house's surfaces: the task's own pair (REF
#: molmospaces/tools/spawn_robot.py:63).
TARGET = ("plate", "apple")
#: Predicted stand-in objects a surface with nothing on it is surveyed against (REF
#: spawn_robot.py:352, `place_arm_on_table(..., 3, ...)`), and their half size (:71).
N_SPAWN = 3
SPAWN_OBJECT_HALF = 0.025

# REF scene_placement.py:40-43.
DEFAULT_SUPPORT_Z_RANGE = (0.35, 1.30)
DEFAULT_MAX_GRASP_WIDTH = 0.30
# REF scene_placement.py:482-493.
_WORKSPACE_BEARINGS = 9
_WORKSPACE_RADII = 4
WORKSPACE_MARGIN = 0.10
_YAW_STEPS = 24
_COVERAGE_QUANTUM = 0.01
# REF robocasa/tools/spawn_robot.py:51.
FLOOR_BAND = (0.02, 1.3)


@dataclass(frozen=True)
class Mount:
    """How a worktop robot is surveyed for: its working annulus (`reach`), the radius kept
    clear around it and on the surface (`radius`), its height over the surface
    (`height`), the base footprint kept on the worktop (`footprint`) and how far out its
    forward workspace must have worktop under it (`workspace`)."""

    reach: tuple
    radius: float
    height: float
    footprint: float = 0.14
    workspace: float | None = None

    @property
    def workspace_radius(self) -> float:
        return self.workspace if self.workspace is not None else self.reach[1] + WORKSPACE_MARGIN


#: The worktop robots. so101: REF's constants (shared/placement.py:40-58,
#: molmospaces/tools/spawn_robot.py:68-73): reach `ARM_REACH` (0.15, 0.35), radius 0.20,
#: height 0.45, footprint 0.14, workspace 0.35 + 0.10. REF had no standalone myCobot 280;
#: its reach follows REF's rule -- the SO-101's annulus is 0.375-0.875 of its ~0.40 m
#: maximum reach, the AiNex's (0.11, 0.25) the same fraction of its measured 0.289 m --
#: applied to the myCobot 280's measured maximum reach, 0.3605 m (the farthest horizontal
#: distance of any of its body origins from the base axis over its joint ranges, measured
#: off the compiled model on 2026-10-01), and pinned: (0.14, 0.32). Its height is its
#: model's 0.468 m upright stack and a margin.
MOUNT = {
    "so101": Mount(reach=(0.15, 0.35), radius=0.20, height=0.45),
    "mycobot280": Mount(reach=(0.14, 0.32), radius=0.20, height=0.50),
}


@dataclass
class Facts:
    """What the survey reads off a scene. `model`/`data` are the bare scene, compiled and
    forwarded once at load (fixed geometry does not move; loose objects are where the
    scene put them)."""

    kind: str                                  # "thor", "robocasa" or "generic"
    model: object
    data: object
    metadata: dict = field(default_factory=dict)      # THOR object metadata, by body name
    thormap: object = None                            # THOR occupancy map (get_free_points)
    pickup_types: tuple = ()
    receptacle_types: tuple = ()
    regions: list | None = None                       # RoboCasa counter reset regions
    geom_aabb: object = None                          # (model, data, geoms) -> (centre, dims)
    descendant_bodies: object = None
    descendant_geoms: object = None

    def __post_init__(self):
        self.geom_aabb = self.geom_aabb or geom_aabb
        self.descendant_bodies = self.descendant_bodies or descendant_bodies
        self.descendant_geoms = self.descendant_geoms or descendant_geoms


@dataclass
class Surface:
    """A top face: its height and the xy rectangles (lo_x, lo_y, hi_x, hi_y, top) it is
    made of."""

    name: str
    z: float
    rects: np.ndarray

    def contains(self, xy, pad: float = 0.0) -> bool:
        return bool(_rects_covering(np.asarray(xy, float), self.rects, pad=pad).any())

    def describe(self) -> dict:
        r = self.rects
        area = float(np.sum((r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])))
        lo, hi = r[:, :2].min(axis=0), r[:, 2:4].max(axis=0)
        w = (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
        c = np.array([np.sum(w * (r[:, 0] + r[:, 2]) / 2), np.sum(w * (r[:, 1] + r[:, 3]) / 2)]) \
            / max(float(w.sum()), 1e-12)
        return {"name": self.name, "z": round(self.z, 5), "area_m2": round(area, 3),
                "centroid": [round(float(v), 3) for v in c],
                "bounds": [round(float(v), 3) for v in (*lo, *hi)]}


@dataclass
class Spot:
    """One candidate: the robot's base at `xy` on a top face at `z`, facing `yaw`."""

    xy: np.ndarray
    z: float
    yaw: float
    surface: str
    why: str


# ---------------------------------------------------------------- molmo_spaces helpers
# Ports of molmo_spaces/utils/mj_model_and_data_utils.py (`descendant_bodies`,
# `descendant_geoms`, `geom_aabb`, `mesh_aabb`), used where molmo_spaces is not installed
# (the RoboCasa venv) and on the `test` scene of both engines, so that scene is surveyed
# by the same code on either engine. iTHOR/ProcTHOR scenes use molmo_spaces' own.


def descendant_bodies(model, body_id: int) -> set:
    if body_id == 0:
        return set(range(model.nbody))
    out = {body_id}
    for bid in np.where(model.body_parentid == body_id)[0]:
        out.update(descendant_bodies(model, int(bid)))
    return out


def descendant_geoms(model, body_id: int, visible_only: bool = True) -> list:
    bodies = np.array(list(descendant_bodies(model, body_id)))
    mask = np.any(model.geom_bodyid.reshape(1, -1) == bodies.reshape(-1, 1), axis=0)
    geoms = np.where(mask)[0]
    if visible_only:
        geoms = geoms[model.geom_group[geoms] < 3]
    return geoms.tolist()


def _mesh_aabb(model, data, g):
    mesh = model.geom_dataid[g]
    a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rel = np.eye(4)
    rm = np.zeros(9)
    mujoco.mju_quat2Mat(rm, model.geom_quat[g])
    rel[:3, :3], rel[:3, 3] = rm.reshape(3, 3), model.geom_pos[g]
    body = np.eye(4)
    b = model.geom_bodyid[g]
    body[:3, :3], body[:3, 3] = data.xmat[b].reshape(3, 3), data.xpos[b]
    pose = body @ rel
    v = model.mesh_vert[a:a + n] @ pose[:3, :3].T + pose[:3, 3]
    lo, hi = v.min(axis=0), v.max(axis=0)
    return (lo + hi) / 2, hi - lo


def geom_aabb(model, data, geom_ids):
    if not len(geom_ids):
        return np.zeros(3), np.zeros(3)
    corners = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
    verts = []
    for g in geom_ids:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            c, s = _mesh_aabb(model, data, g)
            verts.append(c + corners * s / 2)
        else:
            rot = data.geom_xmat[g].reshape(3, 3)
            aabb = model.geom_aabb[g]
            verts.append((aabb[:3] + corners * aabb[3:]) @ rot.T + data.geom_xpos[g])
    verts = np.concatenate(verts, axis=0)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return (lo + hi) / 2, hi - lo


# ---------------------------------------------------------------- REF scene_placement.py


@dataclass(frozen=True)
class GraspTarget:
    """REF scene_placement.py:46-68."""

    support_name: str
    support_category: str
    support_top_z: float
    object_name: str
    object_category: str
    object_xyz: np.ndarray
    n_objects_on_support: int
    reach_slack: float
    support_xy_min: np.ndarray = None  # type: ignore[assignment]
    support_xy_max: np.ndarray = None  # type: ignore[assignment]
    objects_on_support: tuple = ()
    support_top_rects: np.ndarray = None  # type: ignore[assignment]
    support_body_id: int = -1


def _has_free_joint(model, body_ids) -> bool:
    for bid in body_ids:
        adr, num = model.body_jntadr[bid], model.body_jntnum[bid]
        for j in range(adr, adr + num):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return True
    return False


def _classify_bodies(facts: Facts, *, max_grasp_width=DEFAULT_MAX_GRASP_WIDTH,
                     support_z_range=DEFAULT_SUPPORT_Z_RANGE):
    """REF scene_placement.py:138-190: (graspables, supports) among the top-level bodies
    (the metadata's objects when the scene has metadata)."""
    model, data = facts.model, facts.data
    metadata = facts.metadata or {}
    names = list(metadata) if metadata else [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        for b in range(1, model.nbody) if model.body_parentid[b] == 0]
    graspables, supports = [], []
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        geoms = facts.descendant_geoms(model, bid)
        if not geoms:
            continue
        centre, dims = facts.geom_aabb(model, data, geoms)
        category = metadata.get(name, {}).get("category", "")
        entry = (name, category, centre, dims)
        if _has_free_joint(model, facts.descendant_bodies(model, bid)):
            if max(dims[:2]) < max_grasp_width and dims[2] < 0.5:
                graspables.append(entry)
        else:
            top_z = centre[2] + dims[2] / 2
            if support_z_range[0] < top_z < support_z_range[1]:
                supports.append((*entry, _top_face_rects(facts, geoms, top_z)))
    return graspables, supports


def _top_face_rects(facts: Facts, geoms, top_z: float, tol: float = 0.03) -> np.ndarray:
    """REF scene_placement.py:193-203."""
    rows = []
    for g in geoms:
        c, d = facts.geom_aabb(facts.model, facts.data, [g])
        top = c[2] + d[2] / 2
        if top > top_z - tol:
            rows.append([c[0] - d[0] / 2, c[1] - d[1] / 2, c[0] + d[0] / 2, c[1] + d[1] / 2, top])
    return np.array(rows, dtype=float).reshape(-1, 5)


def _rects_covering(xy, rects, pad: float = 0.0) -> np.ndarray:
    """REF scene_placement.py:206-213."""
    return ((xy[0] > rects[:, 0] - pad) & (xy[0] < rects[:, 2] + pad)
            & (xy[1] > rects[:, 1] - pad) & (xy[1] < rects[:, 3] + pad))


def _rests_on(centre, dims, rects, tol: float = 0.06) -> bool:
    """REF scene_placement.py:216-228."""
    covering = _rects_covering(centre[:2], rects, pad=0.05)
    if not covering.any():
        return False
    bottom = centre[2] - dims[2] / 2
    tops = rects[covering, 4]
    return bool(np.any((tops - tol <= bottom) & (bottom <= tops + 0.12)))


def find_supports(facts: Facts, *, support_z_range=DEFAULT_SUPPORT_Z_RANGE) -> list:
    """REF scene_placement.py:231-246: table-height static bodies, biggest top first."""
    _, supports = _classify_bodies(facts, support_z_range=support_z_range)
    supports.sort(key=lambda s: -float(s[3][0] * s[3][1]))
    return supports


def find_grasp_targets(facts: Facts, *, max_grasp_width=DEFAULT_MAX_GRASP_WIDTH,
                       support_z_range=DEFAULT_SUPPORT_Z_RANGE) -> list:
    """REF scene_placement.py:249-343: (surface, object on it) pairs, best first."""
    model = facts.model
    graspables, supports = _classify_bodies(facts, max_grasp_width=max_grasp_width,
                                            support_z_range=support_z_range)
    if not (graspables and supports):
        return []
    free_xy = None
    if facts.thormap is not None:
        pts = facts.thormap.get_free_points()
        free_xy = pts[:, :2] if len(pts) else None
    targets = []
    for s_name, s_cat, s_centre, s_dims, s_rects in supports:
        top_z = s_centre[2] + s_dims[2] / 2
        if not len(s_rects):
            continue
        on_it = [g for g in graspables if _rests_on(g[2], g[3], s_rects)]
        if not on_it:
            continue
        lo = s_rects[:, :2].min(axis=0)
        hi = s_rects[:, 2:4].max(axis=0)
        on_it_geometry = tuple((g[2].copy(), g[3].copy()) for g in on_it)
        for o_name, o_cat, o_centre, _ in on_it:
            slack = 0.0
            if free_xy is not None:
                slack = float(np.linalg.norm(free_xy - o_centre[:2], axis=1).min())
            targets.append(GraspTarget(
                support_name=s_name, support_category=s_cat, support_top_z=float(top_z),
                object_name=o_name, object_category=o_cat, object_xyz=o_centre.copy(),
                n_objects_on_support=len(on_it), reach_slack=slack, support_xy_min=lo,
                support_xy_max=hi, objects_on_support=on_it_geometry, support_top_rects=s_rects,
                support_body_id=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s_name))))
    receptacles, pickups = set(facts.receptacle_types), set(facts.pickup_types)
    targets.sort(key=lambda t: (round(t.reach_slack, 2), -t.n_objects_on_support,
                                t.support_category not in receptacles,
                                t.object_category not in pickups, t.object_name))
    return targets


def static_blockers(facts: Facts, above_z: float, margin: float = 0.02,
                    height: float = 1.0) -> np.ndarray:
    """REF scene_placement.py:370-411: xy rectangles of the static geometry standing in the
    band a robot on a surface at `above_z` occupies."""
    model, data = facts.model, facts.data
    rows = []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0:
            continue
        if _has_free_joint(model, [int(model.body_rootid[model.geom_bodyid[g]])]):
            continue
        centre, dims = facts.geom_aabb(model, data, [g])
        top, bottom = centre[2] + dims[2] / 2, centre[2] - dims[2] / 2
        if top > above_z + margin and bottom < above_z + height:
            rows.append([centre[0] - dims[0] / 2, centre[1] - dims[1] / 2,
                         centre[0] + dims[0] / 2, centre[1] + dims[1] / 2])
    return np.array(rows, dtype=float).reshape(-1, 4)


def _standing_on(model, data, cells, top_z: float, body_id: int, radius: float) -> np.ndarray:
    """REF scene_placement.py:414-453."""
    root = int(model.body_rootid[body_id]) if body_id >= 0 else -1
    if root < 0:
        return np.ones(len(cells), dtype=bool)
    geomid = np.zeros(1, dtype=np.int32)
    down = np.array([0.0, 0.0, -1.0])
    probes = [np.zeros(2)]
    if radius > 0:
        probes += [radius * np.array([np.cos(a), np.sin(a)])
                   for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    mask = np.ones(len(cells), dtype=bool)
    for i, cell in enumerate(cells):
        for probe in probes:
            start = np.array([cell[0] + probe[0], cell[1] + probe[1], top_z + 0.05])
            dist = mujoco.mj_ray(model, data, start, down, None, 1, -1, geomid)
            if (geomid[0] < 0 or dist > 0.15
                    or int(model.body_rootid[model.geom_bodyid[geomid[0]]]) != root):
                mask[i] = False
                break
    return mask


def dynamic_clutter(facts: Facts, above_z: float, height: float = 1.0) -> np.ndarray:
    """REF scene_placement.py:456-477: (x, y, radius) of loose objects in the band."""
    model, data = facts.model, facts.data
    rows = []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0:
            continue
        if not _has_free_joint(model, [int(model.body_rootid[model.geom_bodyid[g]])]):
            continue
        centre, dims = facts.geom_aabb(model, data, [g])
        top, bottom = centre[2] + dims[2] / 2, centre[2] - dims[2] / 2
        if top > above_z + 0.02 and bottom < above_z + height:
            rows.append([centre[0], centre[1], float(max(dims[:2])) / 2])
    return np.array(rows, dtype=float).reshape(-1, 3)


def _workspace_offsets(reach_range, radius: float) -> np.ndarray:
    """REF scene_placement.py:496-507."""
    radii = np.linspace(reach_range[0], radius, _WORKSPACE_RADII)
    bearings = np.linspace(-np.pi / 2, np.pi / 2, _WORKSPACE_BEARINGS)
    r, b = np.meshgrid(radii, bearings, indexing="ij")
    return np.stack([r * np.cos(b), r * np.sin(b)], axis=-1).reshape(-1, 2)


def _support_lookup(model, data, rects, top_z, body_id, lo, hi, step):
    """REF scene_placement.py:510-540."""
    xs = np.arange(lo[0], hi[0] + step, step)
    ys = np.arange(lo[1], hi[1] + step, step)
    points = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    if model is not None and data is not None and body_id >= 0:
        flat = _standing_on(model, data, points, top_z, body_id, radius=0.0)
    elif rects is not None and len(rects):
        flat = np.array([_rects_covering(p, rects).any() for p in points], dtype=bool)
    else:
        flat = np.ones(len(points), dtype=bool)
    grid = flat.reshape(len(xs), len(ys))

    def on_surface(query):
        ix = np.rint((query[..., 0] - xs[0]) / step).astype(int)
        iy = np.rint((query[..., 1] - ys[0]) / step).astype(int)
        inside = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
        return grid[np.clip(ix, 0, len(xs) - 1), np.clip(iy, 0, len(ys) - 1)] & inside

    return on_surface


@dataclass
class TabletopMount:
    """REF scene_placement.py:346-367, with every candidate cell in REF's order (`ranked`:
    (xy, yaw) pairs, `ranked[0]` being REF's choice)."""

    xy: np.ndarray
    z: float
    yaw: float
    target: GraspTarget
    n_in_reach: int
    clearance: float
    clear: bool = True
    coverage: float = 1.0
    ranked: list = field(default_factory=list)


def find_tabletop_mount(target: GraspTarget, *, reach_range, footprint: float,
                        obstacle_clearance: float = 0.08, step: float = 0.05,
                        blockers=None, body_radius=None, clutter=None, model=None, data=None,
                        edge_bias: bool = True, workspace_radius=None) -> TabletopMount:
    """REF scene_placement.py:543-756, unchanged but for keeping the whole ranking: REF
    took `order[0]`; the cells after it, in the same lexsort order, are the next
    candidates."""
    obj_xy = np.asarray(target.object_xyz[:2], dtype=float)
    lo = np.asarray(target.support_xy_min, dtype=float)
    hi = np.asarray(target.support_xy_max, dtype=float)

    inset = footprint / 2 + 0.02
    lo_in, hi_in = lo + inset, hi - inset
    for axis in (0, 1):
        if lo_in[axis] > hi_in[axis]:
            mid = (lo[axis] + hi[axis]) / 2
            lo_in[axis] = hi_in[axis] = mid

    xs = np.arange(lo_in[0], hi_in[0] + 1e-9, step)
    ys = np.arange(lo_in[1], hi_in[1] + 1e-9, step)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    clear = True
    rects = target.support_top_rects
    if model is not None and data is not None and target.support_body_id >= 0:
        for margin in (body_radius or inset, inset):
            standing = _standing_on(model, data, grid, target.support_top_z,
                                    target.support_body_id, margin)
            if standing.any():
                grid = grid[standing]
                break
            clear = False
        else:
            clear = False
    elif rects is not None and len(rects):
        for margin in (body_radius or inset, inset):
            on_face = np.array([_rects_covering(cell, rects, pad=-margin).any() for cell in grid],
                               dtype=bool)
            if on_face.any():
                grid = grid[on_face]
                break
            clear = False

    others_xy = np.array([o[0][:2] for o in target.objects_on_support], dtype=float).reshape(-1, 2)
    others_half = np.array([float(max(o[1][:2])) / 2 for o in target.objects_on_support],
                           dtype=float).reshape(-1)
    if clutter is not None and len(clutter):
        avoid_xy = np.vstack([others_xy, clutter[:, :2]])
        avoid_half = np.concatenate([others_half, clutter[:, 2]])
    else:
        avoid_xy, avoid_half = others_xy, others_half

    if len(avoid_xy):
        gaps = np.linalg.norm(grid[:, None, :] - avoid_xy[None, :, :], axis=-1) - avoid_half[None, :]
        clearance = gaps.min(axis=1)
    else:
        clearance = np.full(len(grid), np.inf)

    d_target = np.linalg.norm(grid - obj_xy, axis=1)
    free = clearance > obstacle_clearance

    if blockers is not None and len(blockers):
        pad = footprint / 2 if body_radius is None else body_radius
        inside = ((grid[:, None, 0] > blockers[None, :, 0] - pad)
                  & (grid[:, None, 0] < blockers[None, :, 2] + pad)
                  & (grid[:, None, 1] > blockers[None, :, 1] - pad)
                  & (grid[:, None, 1] < blockers[None, :, 3] + pad))
        free &= ~inside.any(axis=1)

    wide = (0.5 * reach_range[0], 1.3 * reach_range[1])
    for annulus in (reach_range, wide):
        keep = free & (d_target > annulus[0]) & (d_target < annulus[1])
        if keep.any():
            break
    else:
        clear = False
        keep = free if free.any() else np.ones(len(grid), dtype=bool)

    idx = np.where(keep)[0]
    in_reach = (np.linalg.norm(others_xy[None, :, :] - grid[idx][:, None, :], axis=-1)
                if len(others_xy) else np.zeros((len(idx), 0)))
    n_in_reach = ((in_reach > reach_range[0]) & (in_reach < reach_range[1])).sum(axis=1)
    to_edge = np.minimum(grid - lo_in, hi_in - grid).min(axis=1)

    cells = grid[idx]
    if workspace_radius is None:
        workspace_radius = reach_range[1] + WORKSPACE_MARGIN
    offsets = _workspace_offsets(reach_range, workspace_radius)
    headings = np.linspace(-np.pi, np.pi, _YAW_STEPS, endpoint=False)
    pad = float(np.abs(offsets).max()) + step
    on_surface = _support_lookup(model, data, rects, target.support_top_z,
                                 target.support_body_id, cells.min(axis=0) - pad,
                                 cells.max(axis=0) + pad, step)
    cos, sin = np.cos(headings), np.sin(headings)
    rotated = np.stack([np.outer(cos, offsets[:, 0]) - np.outer(sin, offsets[:, 1]),
                        np.outer(sin, offsets[:, 0]) + np.outer(cos, offsets[:, 1])], axis=-1)
    coverage = on_surface(cells[:, None, None, :] + rotated[None, :, :, :]).mean(axis=2)
    to_target = obj_xy[None, :] - cells
    target_yaw = np.arctan2(to_target[:, 1], to_target[:, 0])
    align = np.cos(headings[None, :] - target_yaw[:, None])
    score = np.round(coverage / _COVERAGE_QUANTUM) + 0.4 * (align + 1.0) / 2.0
    best_yaw = score.argmax(axis=1)
    cell_rows = np.arange(len(cells))
    cell_cov = np.round(coverage[cell_rows, best_yaw] / _COVERAGE_QUANTUM)

    reaching = (n_in_reach > 0).any()
    pool = cell_rows[n_in_reach > 0] if reaching else cell_rows

    def ordered(rows):
        if not len(rows):
            return rows
        if edge_bias:
            o = np.lexsort((-clearance[idx][rows], to_edge[idx][rows], -n_in_reach[rows],
                            -cell_cov[rows]))
        else:
            o = np.lexsort((-clearance[idx][rows], -n_in_reach[rows], -cell_cov[rows]))
        return rows[o]

    first = ordered(pool)
    rest = ordered(cell_rows[n_in_reach == 0]) if reaching else np.zeros(0, int)
    chosen = first[0]
    best = idx[chosen]
    ranked = [(grid[idx[c]].copy(), float(headings[best_yaw[c]])) for c in (*first, *rest)]
    return TabletopMount(xy=grid[best].copy(), z=float(target.support_top_z),
                         yaw=float(headings[best_yaw[chosen]]), target=target,
                         n_in_reach=int(n_in_reach[chosen]),
                         clearance=float(min(clearance[best], 9.99)), clear=clear,
                         coverage=float(coverage[chosen, best_yaw[chosen]]), ranked=ranked)


def _predicted_target(facts: Facts, support, n_spawn: int = N_SPAWN, body_id=None) -> GraspTarget:
    """REF spawn_robot.py:273-309: a support with nothing on it, surveyed against `n_spawn`
    stand-in objects spread around its centre."""
    s_name, s_cat, s_centre, s_dims, s_rects = support
    xy_min = s_centre[:2] - s_dims[:2] / 2
    xy_max = s_centre[:2] + s_dims[:2] / 2
    top_z = float(s_centre[2] + s_dims[2] / 2)
    dims = np.array([SPAWN_OBJECT_HALF * 2] * 3)
    centre = (xy_min + xy_max) / 2
    half = np.maximum((xy_max - xy_min) / 2 - 0.12, 0.0)
    predicted = []
    for i in range(n_spawn):
        angle = 2 * np.pi * i / max(n_spawn, 1)
        offset = half * np.array([np.cos(angle), np.sin(angle)]) * 0.6
        predicted.append(np.array([centre[0] + offset[0], centre[1] + offset[1],
                                   top_z + SPAWN_OBJECT_HALF + 0.002]))
    if body_id is None:
        body_id = int(mujoco.mj_name2id(facts.model, mujoco.mjtObj.mjOBJ_BODY, s_name))
    return GraspTarget(support_name=s_name, support_category=s_cat, support_top_z=top_z,
                       object_name="spawned_object_0", object_category="spawned box",
                       object_xyz=predicted[0], n_objects_on_support=len(predicted),
                       reach_slack=0.0, support_xy_min=xy_min, support_xy_max=xy_max,
                       objects_on_support=tuple((p, dims) for p in predicted),
                       support_top_rects=s_rects, support_body_id=body_id)


def _ranked_targets(facts: Facts, want=TARGET) -> list:
    """REF spawn_robot.py:189-214: the grasp targets, those of the `want` categories (and
    the surfaces holding most of them) first, the original order breaking ties."""
    targets = find_grasp_targets(facts)
    if want and targets:
        def wanted(t) -> int:
            category = (t.object_category or t.object_name or "").lower()
            return sum(1 for w in want if w in category)

        on_surface: dict = {}
        for t in targets:
            on_surface[t.support_name] = on_surface.get(t.support_name, 0) + wanted(t)
        targets = sorted(targets, key=lambda t: (-wanted(t), -on_surface.get(t.support_name, 0)))
    return targets


def _mount_of(facts: Facts, target: GraspTarget, mount: Mount, cache: dict, edge_bias=True,
              whole_robot: bool = True):
    key = round(target.support_top_z, 2)
    if key not in cache:
        cache[key] = (static_blockers(facts, target.support_top_z, height=mount.height),
                      dynamic_clutter(facts, target.support_top_z, height=mount.height))
    use_model = target.support_body_id >= 0
    return find_tabletop_mount(target, reach_range=mount.reach, footprint=mount.footprint,
                               blockers=cache[key][0], clutter=cache[key][1],
                               body_radius=mount.radius if whole_robot else None,
                               model=facts.model if use_model else None,
                               data=facts.data if use_model else None, edge_bias=edge_bias,
                               workspace_radius=mount.workspace_radius)


def tabletop_candidates(facts: Facts, mount: Mount, supports=None) -> list[Spot]:
    """REF `place_arm_on_table` (spawn_robot.py:170-321), as a ranking: the mount REF
    returned first, then the rest of its surface's cells, then the other targets' surfaces
    (those REF accepted -- clear and with objects in reach -- before those it would have
    kept only as its fallback, the fallback ordered by objects in reach), then every other
    table-height support surveyed against predicted stand-ins. After all of those, the
    same search once more with REF's margins for the robot's base footprint alone
    (`find_tabletop_mount` with no `body_radius`: cells nearer an edge or a wall cabinet
    than the whole robot's radius) -- spots REF would only have reached as its last
    resort, which the placement's own checks then judge on the robot's real geometry."""
    cache: dict = {}
    mounts = []
    for whole in (True, False):
        accepted, rejected = [], []
        targets = _ranked_targets(facts) if supports is None else []
        for t in targets:
            m = _mount_of(facts, t, mount, cache, whole_robot=whole)
            (accepted if (m.clear and m.n_in_reach) else rejected).append(m)
        # REF's fallback is the rejected mount with most objects in reach, first found winning
        rejected = sorted(rejected, key=lambda m: -m.n_in_reach)
        mounts += accepted + rejected
        sups = find_supports(facts) if supports is None else supports
        if not targets:
            # REF: nothing graspable anywhere -- survey the biggest support (and here, the
            # others after it) against predicted stand-in objects.
            for s in sups:
                body = s[5] if len(s) > 5 else None
                mounts.append(_mount_of(facts, _predicted_target(facts, s[:5], body_id=body),
                                        mount, cache, whole_robot=whole))
    out = []
    for m in mounts:
        t = m.target
        why = (f"{t.support_category or t.support_name.split('_')[0]}, chosen for "
               f"{t.object_category or t.object_name.split('_')[0]}")
        for xy, yaw in m.ranked:
            out.append(Spot(np.asarray(xy, float), float(t.support_top_z), float(yaw),
                            t.support_name, why))
    return _unique(out)


def _unique(spots: list[Spot]) -> list[Spot]:
    seen, out = set(), []
    for s in spots:
        k = (round(float(s.xy[0]), 6), round(float(s.xy[1]), 6), round(s.z, 6), round(s.yaw, 6))
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


# ---------------------------------------------------------------- REF robocasa spawn_robot.py


def world_boxes(model, data, band) -> np.ndarray:
    """REF robocasa/tools/spawn_robot.py:104-124: [cx, cy, ex, ey] of the collision geoms
    inside a height band."""
    boxes = []
    zlo, zhi = band
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        rot = data.geom_xmat[gid].reshape(3, 3)
        centre = data.geom_xpos[gid] + rot @ model.geom_aabb[gid][:3]
        extent = np.abs(rot) @ model.geom_aabb[gid][3:]
        if centre[2] + extent[2] < zlo or centre[2] - extent[2] > zhi:
            continue
        boxes.append([centre[0], centre[1], extent[0], extent[1]])
    return np.array(boxes) if boxes else np.zeros((0, 4))


def clearance_field(boxes, grid) -> np.ndarray:
    """REF robocasa/tools/spawn_robot.py:127-132."""
    if len(boxes) == 0:
        return np.full(len(grid), np.inf)
    delta = np.abs(grid[:, None, :] - boxes[None, :, :2]) - boxes[None, :, 2:]
    return np.linalg.norm(np.maximum(delta, 0.0), axis=-1).min(axis=1)


def outward_direction(region: dict, floor) -> np.ndarray:
    """REF robocasa/tools/spawn_robot.py:223-244."""
    c, s = np.cos(region["rot"]), np.sin(region["rot"])
    axes = (np.array([c, s]), np.array([-s, c]))
    short = int(np.argmin(region["half"]))
    direction = axes[short]
    probe = region["half"][short] + 0.6
    best, best_clear = direction, -np.inf
    for sign in (1.0, -1.0):
        point = (region["centre"] + sign * direction * probe).reshape(1, 2)
        clear = float(clearance_field(floor, point)[0])
        if clear > best_clear:
            best, best_clear = sign * direction, clear
    return best


def _usable_region(regions, radius, reach):
    """REF robocasa/tools/spawn_robot.py:262-266: the roomiest region deep enough."""
    usable = [r for r in regions if min(r["half"]) * 2 >= radius + reach[0]]
    if not usable:
        usable = regions
    return max(usable, key=lambda r: float(r["half"][0] * r["half"][1]))


def find_counter_mount(facts: Facts, radius: float, reach) -> Spot:
    """REF robocasa/tools/spawn_robot.py:247-279: against the back edge of the roomiest
    counter region, inset by the robot's radius, facing the room."""
    region = _usable_region(facts.regions, radius, reach)
    floor = world_boxes(facts.model, facts.data, FLOOR_BAND)
    out = outward_direction(region, floor)
    depth = float(min(region["half"]))
    xy = region["centre"] - out * max(depth - radius, 0.0)
    yaw = float(np.arctan2(out[1], out[0]))
    return Spot(np.asarray(xy, float), float(region["top_z"]), yaw, region["name"],
                "back edge of the roomiest counter, facing the room")


def region_rect(region: dict) -> np.ndarray:
    """A counter region's top as (lo_x, lo_y, hi_x, hi_y, top)."""
    c, s = abs(math.cos(region["rot"])), abs(math.sin(region["rot"]))
    hx = c * region["half"][0] + s * region["half"][1]
    hy = s * region["half"][0] + c * region["half"][1]
    cx, cy = region["centre"]
    return np.array([[cx - hx, cy - hy, cx + hx, cy + hy, region["top_z"]]])


def front_edge_candidates(facts: Facts, mount: Mount) -> list[Spot]:
    """Not REF: the mirror of REF's back-edge rule, the last resort for an arm too tall for
    the space under a wall cabinet. Against the *front* edge of each counter region
    (roomiest first), inset by half the base footprint, facing the wall -- so the whole
    depth of the counter, under the cabinet, is in front of it while the arm itself stays
    out from under it -- at steps of 0.05 m along the counter, its middle first."""
    floor = world_boxes(facts.model, facts.data, FLOOR_BAND)
    out = []
    for r in sorted(facts.regions, key=lambda r: -float(r["half"][0] * r["half"][1])):
        outward = outward_direction(r, floor)
        short = int(np.argmin(r["half"]))
        depth, length = float(r["half"][short]), float(r["half"][1 - short])
        c, s = np.cos(r["rot"]), np.sin(r["rot"])
        along = (np.array([c, s]), np.array([-s, c]))[1 - short]
        front = r["centre"] + outward * max(depth - mount.footprint / 2, 0.0)
        yaw = float(np.arctan2(-outward[1], -outward[0]))
        n = int(max(length - 0.1, 0.0) // 0.05)
        for k in sorted(range(-n, n + 1), key=lambda k: (abs(k), -k)):
            out.append(Spot(front + along * 0.05 * k, float(r["top_z"]), yaw, r["name"],
                            "front edge of a counter, facing the wall"))
    return out


def counter_candidates(facts: Facts, mount: Mount) -> list[Spot]:
    """REF's RoboCasa mount first, then REF's tabletop coverage search over every counter
    region (largest first), each surveyed against predicted stand-in objects, then the
    front-edge spots (`front_edge_candidates`)."""
    out = [find_counter_mount(facts, mount.radius, mount.reach)]
    supports = []
    for r in sorted(facts.regions, key=lambda r: -float(r["half"][0] * r["half"][1])):
        rect = region_rect(r)
        centre = np.array([(rect[0, 0] + rect[0, 2]) / 2, (rect[0, 1] + rect[0, 3]) / 2,
                           r["top_z"] - 0.01])
        dims = np.array([rect[0, 2] - rect[0, 0], rect[0, 3] - rect[0, 1], 0.02])
        supports.append((r["name"], "counter", centre, dims, rect, -1))
    out += tabletop_candidates(facts, mount, supports=supports)
    out += front_edge_candidates(facts, mount)
    return _unique(out)


# ---------------------------------------------------------------- entry points


def worktop(facts: Facts) -> Surface | None:
    """The scene's worktop: the surface the survey ranks first, whatever robot stands on
    it (REF: the support of the first grasp target after the `TARGET` ranking; with nothing
    graspable, the biggest table-height top; on RoboCasa, the roomiest counter region)."""
    if facts.kind == "robocasa" and facts.regions:
        region = _usable_region(facts.regions, MOUNT["so101"].radius, MOUNT["so101"].reach)
        return Surface(region["name"], float(region["top_z"]), region_rect(region))
    targets = _ranked_targets(facts)
    if targets:
        t = targets[0]
        return Surface(t.support_name, float(t.support_top_z), np.asarray(t.support_top_rects))
    supports = find_supports(facts)
    if supports:
        s = supports[0]
        return Surface(s[0], float(s[2][2] + s[3][2] / 2), np.asarray(s[4]))
    return None


def candidates(facts: Facts, robot_id: str) -> list[Spot]:
    """The spots `robot_id` is tried at on the worktop, best first (empty for a robot with
    no `MOUNT`: it is placed on the worktop's surface by the general search)."""
    mount = MOUNT.get(robot_id)
    if mount is None:
        return []
    if facts.kind == "robocasa" and facts.regions:
        return counter_candidates(facts, mount)
    return tabletop_candidates(facts, mount)
