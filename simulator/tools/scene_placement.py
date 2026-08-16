#!/usr/bin/env python
"""Where to put a robot in a house, and what to point it at.

Replaces the geom-centroid heuristic this file's predecessor used, which had no notion
of where the floor was: it treated every collision geom as a single point at its centre
and searched the bounding box of that point cloud. On procthor-10k:42 it chose (0.00,
0.04) -- outside the house entirely -- and on ithor:1 and holodeck-objaverse:0 it chose
spots on the floor but with under 0.35 m of clearance, i.e. wedged against furniture.

Instead, use the occupancy map every scene already ships as `<scene>_map.png`, and pick
a spot in an annulus around an object the robot could actually reach for -- the same
shape as upstream's `MujocoEnv.place_robot_near` (molmo_spaces/env/env.py:761), lifted
out of the env because `tools/` deliberately does not depend on the upstream env
stack.

    from tools.scene_placement import find_robot_placement
    p = find_robot_placement(scene_xml, agent_radius=0.35, reach_range=(0.40, 0.80))
    p.pos, p.yaw, p.target.object_category
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

# Widening annuli, tried in order. The first is the robot's own comfortable reach; the
# others trade posture for a spot that exists at all, which beats refusing to place.
FALLBACK_ANNULI = ((0.35, 1.00), (0.30, 1.40))

# A support has to be worth standing at: table height, not floor and not overhead.
DEFAULT_SUPPORT_Z_RANGE = (0.35, 1.30)

# Anything wider than this in its long horizontal axis is furniture, not a payload.
DEFAULT_MAX_GRASP_WIDTH = 0.30


@dataclass(frozen=True)
class GraspTarget:
    """A surface worth standing at, and the object on it worth reaching for."""

    support_name: str
    support_category: str
    support_top_z: float
    object_name: str
    object_category: str
    object_xyz: np.ndarray
    n_objects_on_support: int
    # Distance from the object to the nearest free floor cell. Small means a robot can
    # actually get to it; large means it is walled in behind other furniture.
    reach_slack: float
    # The top face as an xy rectangle, and everything resting on it as (xyz, dims) pairs.
    # Both fall out of the classification pass; an arm that bolts down *on* the surface
    # needs them to find a clear patch, so they are kept rather than discarded.
    support_xy_min: np.ndarray = None  # type: ignore[assignment]
    support_xy_max: np.ndarray = None  # type: ignore[assignment]
    objects_on_support: tuple = ()
    # Rows of (lo_x, lo_y, hi_x, hi_y, top_z): the geoms making up the top face itself.
    support_top_rects: np.ndarray = None  # type: ignore[assignment]
    support_body_id: int = -1


@dataclass(frozen=True)
class Placement:
    """Where to stand. `pos` is the xy of the robot's floor footprint, never a z.

    Keeping z out of this is what lets one placement serve every base convention: a
    mocap pedestal supplies its own height, and a holonomic base is grafted at the
    origin and driven to the pose through its world-aligned slide joints.
    """

    pos: np.ndarray
    yaw: float
    target: GraspTarget | None
    source: str  # override | table | open-floor | no-map-fallback
    clearance: float


def load_scene_map(scene_path: str | Path, agent_radius: float | None = 0.35):
    """Load the precomputed occupancy map beside a scene, or None if there is not one.

    Mirrors `MujocoEnv.get_thormap` (molmo_spaces/env/env.py:690): iTHOR maps are
    single-channel, ProcTHOR/Holodeck maps carry a room id in the blue channel, and the
    `_ceiling` variant of a scene shares the base scene's map.

    `agent_radius` is passed through to `load`, which erodes the free space by that
    radius -- that dilation is the only thing keeping a robot out of the walls, so pass
    the real footprint. Pass None to ask the raw question "is this on the floor at all".

    Deliberately does not fall back to `from_mj_model_path`: building a map renders
    depth views and takes 10-30 s, which is not something a viewer launch should do
    silently.
    """
    from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

    scene_path = Path(str(scene_path).replace("_ceiling", ""))
    text = str(scene_path)
    if "ithor" in text:
        cls = iTHORMap
    elif "procthor" in text or "holodeck" in text:
        cls = ProcTHORMap
    else:
        return None

    png = scene_path.parent / f"{scene_path.stem}_map.png"
    if not png.is_file():
        return None
    return cls.load(path=png.as_posix(), agent_radius=agent_radius)


def _has_free_joint(model: mujoco.MjModel, body_ids) -> bool:
    for bid in body_ids:
        adr, num = model.body_jntadr[bid], model.body_jntnum[bid]
        for j in range(adr, adr + num):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return True
    return False


def _compiled(scene_path, model, data):
    """Compile the scene if the caller did not already have a model/data pair."""
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    return model, data


def _classify_bodies(
    scene_path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    max_grasp_width: float = DEFAULT_MAX_GRASP_WIDTH,
    support_z_range: tuple[float, float] = DEFAULT_SUPPORT_Z_RANGE,
) -> tuple[list, list]:
    """Split the scene's top-level bodies into (graspables, supports).

    Each entry is `(name, category, centre, dims)` from the aggregated geom AABB; supports
    carry a fifth field, the xy rectangles of the geoms forming their actual top face, as
    rows of (lo_x, lo_y, hi_x, hi_y, top_z).

    The body AABB alone is not enough to say what is on a surface: FloorPlan1's counter run
    wraps around its island, so the counter's bounding box contains the island, everything
    standing on the island, and a good deal of open floor. The top faces are the part of
    that box that is really a work surface.
    """
    from molmo_spaces.utils.mj_model_and_data_utils import descendant_bodies, descendant_geoms, geom_aabb
    from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

    metadata = (get_scene_metadata(scene_path) or {}).get("objects", {})

    # Without metadata every top-level body is a candidate; the geometry test below is
    # what actually decides, so this degrades rather than fails.
    names = list(metadata) if metadata else [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        for b in range(1, model.nbody)
        if model.body_parentid[b] == 0
    ]

    graspables, supports = [], []
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        geoms = descendant_geoms(model, bid)
        if not geoms:
            continue
        centre, dims = geom_aabb(model, data, geoms)
        category = metadata.get(name, {}).get("category", "")
        entry = (name, category, centre, dims)

        if _has_free_joint(model, descendant_bodies(model, bid)):
            if max(dims[:2]) < max_grasp_width and dims[2] < 0.5:
                graspables.append(entry)
        else:
            top_z = centre[2] + dims[2] / 2
            if support_z_range[0] < top_z < support_z_range[1]:
                supports.append((*entry, _top_face_rects(model, data, geoms, top_z)))

    return graspables, supports


def _top_face_rects(model, data, geoms, top_z: float, tol: float = 0.03) -> np.ndarray:
    """The xy rectangles of the geoms that reach the body's top, as (lo_x, lo_y, hi_x, hi_y, top)."""
    from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb

    rows = []
    for g in geoms:
        c, d = geom_aabb(model, data, [g])
        top = c[2] + d[2] / 2
        if top > top_z - tol:
            rows.append([c[0] - d[0] / 2, c[1] - d[1] / 2, c[0] + d[0] / 2, c[1] + d[1] / 2, top])
    return np.array(rows, dtype=float).reshape(-1, 5)


def _rects_covering(xy: np.ndarray, rects: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Boolean mask over `rects` of those whose xy footprint contains `xy`."""
    return (
        (xy[0] > rects[:, 0] - pad)
        & (xy[0] < rects[:, 2] + pad)
        & (xy[1] > rects[:, 1] - pad)
        & (xy[1] < rects[:, 3] + pad)
    )


def _rests_on(centre: np.ndarray, dims: np.ndarray, rects: np.ndarray, tol: float = 0.06) -> bool:
    """True if an object sits over one of `rects` with its bottom at that face's height.

    The height test is per-rect rather than against the body's overall top, which is what
    keeps a counter from claiming the objects on the island it wraps around: those are 5 cm
    higher, over a face that belongs to something else.
    """
    covering = _rects_covering(centre[:2], rects, pad=0.05)
    if not covering.any():
        return False
    bottom = centre[2] - dims[2] / 2
    tops = rects[covering, 4]
    return bool(np.any((tops - tol <= bottom) & (bottom <= tops + 0.12)))


def find_supports(
    scene_path: str | Path,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    *,
    support_z_range: tuple[float, float] = DEFAULT_SUPPORT_Z_RANGE,
) -> list:
    """Table-height static bodies, biggest top face first, whether or not anything is on them.

    This is the bare-scene counterpart to `find_grasp_targets`: a surface worth putting a
    robot -- or an object -- on, in a house where nothing happens to be lying around.
    """
    model, data = _compiled(scene_path, model, data)
    _, supports = _classify_bodies(scene_path, model, data, support_z_range=support_z_range)
    supports.sort(key=lambda s: -float(s[3][0] * s[3][1]))
    return supports


def find_grasp_targets(
    scene_path: str | Path,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    *,
    max_grasp_width: float = DEFAULT_MAX_GRASP_WIDTH,
    support_z_range: tuple[float, float] = DEFAULT_SUPPORT_Z_RANGE,
    thormap=None,
) -> list[GraspTarget]:
    """Find (surface, object-on-it) pairs, best first.

    Detection is geometric, not metadata-driven, because the annotations do not survive
    contact with the four datasets: iTHOR's best kitchen counter has category
    "StandardCounterHeightWidth_c3ccfdb2", 26 of holodeck's 61 objects have category ""
    outright, and a "Receptacle" site -- which is supposed to mark a placement surface --
    fires on cups and toasters. So an object is graspable if it has a free joint and
    fits a hand, a support is a static body at working height with graspables resting on
    it, and the categories are only ever a tiebreaker.
    """
    from molmo_spaces.utils.constants.object_constants import (
        ALL_PICKUP_TYPES_THOR,
        RECEPTACLE_TYPES_THOR,
    )

    model, data = _compiled(scene_path, model, data)
    graspables, supports = _classify_bodies(
        scene_path,
        model,
        data,
        max_grasp_width=max_grasp_width,
        support_z_range=support_z_range,
    )

    if not (graspables and supports):
        return []

    free_xy = None
    if thormap is not None:
        pts = thormap.get_free_points()
        free_xy = pts[:, :2] if len(pts) else None

    targets = []
    for s_name, s_cat, s_centre, s_dims, s_rects in supports:
        top_z = s_centre[2] + s_dims[2] / 2
        if not len(s_rects):
            continue

        # An object is on this surface if it sits over one of its top faces and its bottom
        # is at that face's height -- not merely somewhere inside the body's bounding box,
        # which for a counter that wraps a kitchen means "anywhere in the room".
        on_it = [g for g in graspables if _rests_on(g[2], g[3], s_rects)]
        if not on_it:
            continue
        # The reachable surface is the union of the faces things are actually resting on.
        lo = s_rects[:, :2].min(axis=0)
        hi = s_rects[:, 2:4].max(axis=0)

        on_it_geometry = tuple((g[2].copy(), g[3].copy()) for g in on_it)
        for o_name, o_cat, o_centre, _ in on_it:
            slack = 0.0
            if free_xy is not None:
                slack = float(np.linalg.norm(free_xy - o_centre[:2], axis=1).min())
            targets.append(
                GraspTarget(
                    support_name=s_name,
                    support_category=s_cat,
                    support_top_z=float(top_z),
                    object_name=o_name,
                    object_category=o_cat,
                    object_xyz=o_centre.copy(),
                    n_objects_on_support=len(on_it),
                    reach_slack=slack,
                    support_xy_min=lo,
                    support_xy_max=hi,
                    objects_on_support=on_it_geometry,
                    support_top_rects=s_rects,
                    support_body_id=int(
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s_name)
                    ),
                )
            )

    # Reachability first, then a busy surface, and only then the annotations -- so a
    # correctly-labelled DiningTable the robot cannot get to still loses to an unlabelled
    # counter it can stand at.
    targets.sort(
        key=lambda t: (
            round(t.reach_slack, 2),
            -t.n_objects_on_support,
            t.support_category not in RECEPTACLE_TYPES_THOR,
            t.object_category not in ALL_PICKUP_TYPES_THOR,
            t.object_name,
        )
    )
    return targets


@dataclass(frozen=True)
class TabletopMount:
    """Where to bolt an arm down *on* a surface, rather than beside it.

    Unlike `Placement` this does carry a z: the whole point is that the robot's base sits
    at the height of the surface it works on, and that height came from the same
    measurement that chose the spot.
    """

    xy: np.ndarray
    z: float
    yaw: float
    target: GraspTarget
    n_in_reach: int
    clearance: float
    # False when no cell on this surface was both clear and in reach and the spot is a
    # last resort -- a 0.6 m counter against a wall has no room for a 0.77 m arm, and the
    # caller's job is then to try the next surface, not to bolt the arm into the wall.
    clear: bool = True


def static_blockers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    above_z: float,
    margin: float = 0.02,
    height: float = 1.0,
) -> np.ndarray:
    """xy rectangles of the static geometry standing on a surface, as (lo_x, lo_y, hi_x, hi_y).

    `height` is how far up the robot reaches from the surface; only geometry overlapping
    that band counts. Without it the ceiling and every light fixture in the house qualify
    as obstacles, and nowhere is free.

    A support is identified by the AABB of its whole body, and in a real kitchen those
    overlap: FloorPlan1's counter run and its island share a bounding box, so the clearest
    "spot on the counter" is a spot inside the island. Nothing about the surface itself
    says so -- only the geometry standing on it does.

    Only *static* geometry counts. The loose objects on the surface are the whole reason
    the robot is there, they are handled by the clearance test in `find_tabletop_mount`,
    and giving each of them a robot-sized berth would rule out every cell of a busy table.
    """
    from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb

    rows = []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0:
            continue
        if _has_free_joint(model, [int(model.body_rootid[model.geom_bodyid[g]])]):
            continue
        centre, dims = geom_aabb(model, data, [g])
        top, bottom = centre[2] + dims[2] / 2, centre[2] - dims[2] / 2
        if top > above_z + margin and bottom < above_z + height:
            rows.append(
                [
                    centre[0] - dims[0] / 2,
                    centre[1] - dims[1] / 2,
                    centre[0] + dims[0] / 2,
                    centre[1] + dims[1] / 2,
                ]
            )
    return np.array(rows, dtype=float).reshape(-1, 4)


def _standing_on(
    model, data, cells: np.ndarray, top_z: float, body_id: int, radius: float
) -> np.ndarray:
    """Mask of cells where the support is directly underfoot, out to `radius`.

    Rectangles cannot answer this. MuJoCo's AABB for the L-shaped counter in FloorPlan1 is
    a 3.4 m square covering the whole kitchen including the open floor in the crook of the
    L, so "inside the top face" is true of places with nothing under them at all. A ray
    straight down asks the question the rectangle only approximates.
    """
    root = int(model.body_rootid[body_id]) if body_id >= 0 else -1
    if root < 0:
        return np.ones(len(cells), dtype=bool)

    geomid = np.zeros(1, dtype=np.int32)
    down = np.array([0.0, 0.0, -1.0])
    # The centre plus four points at arm's length: the whole robot has to be over the
    # surface, not just the middle of its base.
    probes = [np.zeros(2)] + [
        radius * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ]

    mask = np.ones(len(cells), dtype=bool)
    for i, cell in enumerate(cells):
        for probe in probes:
            start = np.array([cell[0] + probe[0], cell[1] + probe[1], top_z + 0.05])
            dist = mujoco.mj_ray(model, data, start, down, None, 1, -1, geomid)
            if (
                geomid[0] < 0
                or dist > 0.15
                or int(model.body_rootid[model.geom_bodyid[geomid[0]]]) != root
            ):
                mask[i] = False
                break
    return mask


def dynamic_clutter(
    model: mujoco.MjModel, data: mujoco.MjData, above_z: float, height: float = 1.0
) -> np.ndarray:
    """Rows of (x, y, radius) for the loose objects standing in a surface's working band.

    The counterpart to `static_blockers`. `GraspTarget.objects_on_support` only knows about
    the objects the surface owns, and a mount that clears all ten of those can still be
    placed on top of the frying pan that belongs to the hob next to it.
    """
    from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb

    rows = []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0:
            continue
        if not _has_free_joint(model, [int(model.body_rootid[model.geom_bodyid[g]])]):
            continue
        centre, dims = geom_aabb(model, data, [g])
        top, bottom = centre[2] + dims[2] / 2, centre[2] - dims[2] / 2
        if top > above_z + 0.02 and bottom < above_z + height:
            rows.append([centre[0], centre[1], float(max(dims[:2])) / 2])
    return np.array(rows, dtype=float).reshape(-1, 3)


def find_tabletop_mount(
    target: GraspTarget,
    *,
    reach_range: tuple[float, float],
    footprint: float,
    obstacle_clearance: float = 0.08,
    step: float = 0.05,
    blockers: np.ndarray | None = None,
    body_radius: float | None = None,
    clutter: np.ndarray | None = None,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
) -> TabletopMount:
    """Pick a clear patch of `target`'s surface within reach of what is on it.

    Candidates are grid cells on the top face, inset by half the robot's footprint so the
    base does not overhang the edge, minus anything sitting too close to an object already
    there. Among what survives, prefer the cell with the most elbow room, and break ties by
    how much of the surface's clutter falls inside the arm's working annulus -- so the arm
    faces the busy side of the table rather than an empty corner of it.
    """
    obj_xy = np.asarray(target.object_xyz[:2], dtype=float)
    lo = np.asarray(target.support_xy_min, dtype=float)
    hi = np.asarray(target.support_xy_max, dtype=float)

    inset = footprint / 2 + 0.02
    lo_in, hi_in = lo + inset, hi - inset
    # A surface narrower than the base leaves nothing to inset; fall back to its centre
    # line in that axis rather than an empty grid.
    for axis in (0, 1):
        if lo_in[axis] > hi_in[axis]:
            mid = (lo[axis] + hi[axis]) / 2
            lo_in[axis] = hi_in[axis] = mid

    xs = np.arange(lo_in[0], hi_in[0] + 1e-9, step)
    ys = np.arange(lo_in[1], hi_in[1] + 1e-9, step)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    # Only cells over the top face itself. The union rectangle of an L-shaped counter
    # includes the notch, which is open floor, and a mocap-mounted arm placed there would
    # float in mid-air without anything objecting.
    clear = True
    rects = target.support_top_rects
    if model is not None and data is not None and target.support_body_id >= 0:
        for margin in (body_radius or inset, inset):
            standing = _standing_on(
                model, data, grid, target.support_top_z, target.support_body_id, margin
            )
            if standing.any():
                grid = grid[standing]
                break
            clear = False
        else:
            clear = False
    elif rects is not None and len(rects):
        # Prefer cells the whole robot stands over, not just its base: an arm bolted to the
        # lip of a counter is on the surface by the letter of the test and hanging off it by
        # every other measure. A 0.6 m counter cannot hold a 0.77 m arm that way, so falling
        # back to the base footprint is reported as not-clear and the caller tries the next
        # surface -- usually a kitchen island, which can.
        for margin in (body_radius or inset, inset):
            on_face = np.array(
                [_rects_covering(cell, rects, pad=-margin).any() for cell in grid], dtype=bool
            )
            if on_face.any():
                grid = grid[on_face]
                break
            clear = False

    others_xy = np.array([o[0][:2] for o in target.objects_on_support], dtype=float).reshape(-1, 2)
    # Each object's own half-extent, so a mug and a cereal box are not given the same berth.
    others_half = np.array(
        [float(max(o[1][:2])) / 2 for o in target.objects_on_support], dtype=float
    ).reshape(-1)

    # Everything to keep off: this surface's own objects, plus any loose thing standing in
    # the same band that belongs to a neighbour.
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
        # Grow the blocker rather than shrinking the candidate to a point: a cell is out if
        # the robot would overlap anything taller. Pad by the *robot*, not by its base --
        # what hits the wall behind a counter is the arm, standing well clear of it.
        inside = (
            (grid[:, None, 0] > blockers[None, :, 0] - pad)
            & (grid[:, None, 0] < blockers[None, :, 2] + pad)
            & (grid[:, None, 1] > blockers[None, :, 1] - pad)
            & (grid[:, None, 1] < blockers[None, :, 3] + pad)
        )
        free &= ~inside.any(axis=1)

    wide = (0.5 * reach_range[0], 1.3 * reach_range[1])
    for annulus in (reach_range, wide):
        keep = free & (d_target > annulus[0]) & (d_target < annulus[1])
        if keep.any():
            break
    else:
        # Nothing both clear and in reach: hand back the least-bad cell, flagged, so the
        # caller can move on to another surface.
        clear = False
        keep = free if free.any() else np.ones(len(grid), dtype=bool)

    idx = np.where(keep)[0]
    in_reach = (
        np.linalg.norm(others_xy[None, :, :] - grid[idx][:, None, :], axis=-1)
        if len(others_xy)
        else np.zeros((len(idx), 0))
    )
    n_in_reach = ((in_reach > reach_range[0]) & (in_reach < reach_range[1])).sum(axis=1)

    order = np.lexsort((-clearance[idx], -n_in_reach))
    best = idx[order[0]]

    to_target = obj_xy - grid[best]
    yaw = float(np.arctan2(to_target[1], to_target[0]))
    return TabletopMount(
        xy=grid[best].copy(),
        z=float(target.support_top_z),
        yaw=yaw,
        target=target,
        n_in_reach=int(n_in_reach[order[0]]),
        clearance=float(min(clearance[best], 9.99)),
        clear=clear,
    )


def _spot_in_annulus(free_xy: np.ndarray, centre: np.ndarray, annulus: tuple[float, float]):
    """The free cell closest to the middle of the annulus, or None if it is empty."""
    d = np.linalg.norm(free_xy - centre, axis=1)
    keep = (d > annulus[0]) & (d < annulus[1])
    if not keep.any():
        return None
    mid = (annulus[0] + annulus[1]) / 2
    idx = np.where(keep)[0]
    return free_xy[idx[np.argmin(np.abs(d[idx] - mid))]]


def _open_floor_spot(free_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """The free cell furthest from any non-free cell. Inside the house by construction.

    This is what the old heuristic was reaching for. Subsampled because the map is
    ~200 px/m and the pairwise distance matrix is otherwise enormous.
    """
    step = max(1, len(free_xy) // 4000)
    sample = free_xy[::step]
    lo, hi = sample.min(axis=0), sample.max(axis=0)
    xs = np.arange(lo[0], hi[0] + 1e-6, 0.1)
    ys = np.arange(lo[1], hi[1] + 1e-6, 0.1)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    # Keep only grid points that are themselves free, then take the one deepest inside.
    d_free = np.linalg.norm(grid[:, None, :] - sample[None, :, :], axis=-1).min(axis=1)
    inside = grid[d_free < 0.15]
    if not len(inside):
        centre = sample.mean(axis=0)
        return centre, 0.0
    centre = sample.mean(axis=0)
    d_edge = np.linalg.norm(inside[:, None, :] - sample[None, :, :], axis=-1)
    best = inside[int(np.argmax(np.sort(d_edge, axis=1)[:, min(50, d_edge.shape[1] - 1)]))]
    to_centre = centre - best
    return best, float(np.arctan2(to_centre[1], to_centre[0]))


def _no_map_fallback(model, data) -> tuple[np.ndarray, float]:
    """Last resort when a scene ships no occupancy map.

    Still geom-based, but with the two bugs that made the old version pick spots outside
    the house fixed: candidates are clamped to the extent of the *floor* geoms rather
    than to the bounding box of all obstacles, and obstacles are no longer filtered to
    centre-z < 1.3 m, which had been quietly excluding every wall (a 2.7 m wall's centre
    sits at 1.35 m).
    """
    floor_xy, obstacle_xy = [], []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0:
            continue
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        pos = data.geom_xpos[g]
        if "floor" in name or "room" in name:
            floor_xy.append(pos[:2])
        elif pos[2] > 0.05:
            obstacle_xy.append(pos[:2])

    if not floor_xy:
        return np.zeros(2), 0.0
    floor_xy = np.array(floor_xy)
    lo, hi = floor_xy.min(axis=0), floor_xy.max(axis=0)
    xs = np.arange(lo[0], hi[0] + 1e-6, 0.1)
    ys = np.arange(lo[1], hi[1] + 1e-6, 0.1)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    if not obstacle_xy:
        centre = (lo + hi) / 2
        return centre, 0.0
    obstacle_xy = np.array(obstacle_xy)
    d = np.linalg.norm(grid[:, None, :] - obstacle_xy[None, :, :], axis=-1).min(axis=1)
    best = grid[int(np.argmax(d))]
    centre = (lo + hi) / 2
    to_centre = centre - best
    return best, float(np.arctan2(to_centre[1], to_centre[0]))


def find_robot_placement(
    scene_path: str | Path,
    *,
    agent_radius: float = 0.35,
    reach_range: tuple[float, float] = (0.40, 0.80),
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    pos: np.ndarray | list[float] | None = None,
    yaw_deg: float | None = None,
    max_targets: int = 12,
    prefer: str = "table",
) -> Placement:
    """Choose where the robot should stand and which way it should face.

    Overrides compose rather than short-circuit: giving `pos` alone still selects a
    target and faces it *from the position you gave*. The old code computed the yaw from
    the auto-chosen spot even when you had overridden the position, so `--pos` without
    `--yaw` pointed the robot at nothing in particular.

    `prefer="floor"` skips target selection entirely and asks only for the most open spot
    inside the house. That is what a mobile base wants: it has nothing to reach for, and
    an annulus hugging a table is the one place a robot that has to drive does not want
    to start.
    """
    if pos is not None and yaw_deg is not None:
        return Placement(np.asarray(pos, dtype=float), float(np.radians(yaw_deg)), None, "override", 0.0)

    model, data = _compiled(scene_path, model, data)

    thormap = load_scene_map(scene_path, agent_radius=agent_radius)

    if thormap is None:
        spot, yaw = _no_map_fallback(model, data)
        if pos is not None:
            spot = np.asarray(pos, dtype=float)
        return Placement(spot, float(np.radians(yaw_deg)) if yaw_deg is not None else yaw,
                         None, "no-map-fallback", 0.0)

    free_pts = thormap.get_free_points()
    free_xy = free_pts[:, :2] if len(free_pts) else np.zeros((0, 2))
    targets = (
        find_grasp_targets(scene_path, model, data, thormap=thormap)[:max_targets]
        if prefer != "floor"
        else []
    )

    for target in targets:
        obj_xy = target.object_xyz[:2]
        for annulus in (reach_range, *FALLBACK_ANNULI):
            spot = obj_xy if pos is None else np.asarray(pos, dtype=float)
            if pos is None:
                spot = _spot_in_annulus(free_xy, obj_xy, annulus)
                if spot is None:
                    continue
            to_target = obj_xy - spot
            yaw = float(np.arctan2(to_target[1], to_target[0]))
            if yaw_deg is not None:
                yaw = float(np.radians(yaw_deg))
            return Placement(
                np.asarray(spot, dtype=float), yaw, target, "table",
                float(np.linalg.norm(obj_xy - spot)),
            )

    if len(free_xy):
        spot, yaw = _open_floor_spot(free_xy)
    else:
        spot, yaw = _no_map_fallback(model, data)
    if pos is not None:
        spot = np.asarray(pos, dtype=float)
    if yaw_deg is not None:
        yaw = float(np.radians(yaw_deg))
    return Placement(spot, yaw, None, "open-floor" if len(free_xy) else "no-map-fallback", 0.0)


def apply_init_qpos(view, config) -> None:
    """Write a robot's rest pose into both joint state and actuator targets.

    Both, because these are position actuators: leaving ctrl at its default of 0 makes
    the robot immediately drive out of the pose it was just placed in.
    """
    for group_id, qpos in (config.init_qpos or {}).items():
        if qpos is None or len(qpos) == 0 or group_id not in view.move_group_ids():
            continue
        group = view.get_move_group(group_id)
        group.joint_pos = qpos
        group.ctrl = qpos


def describe(placement: Placement | TabletopMount) -> str:
    """One line for a log: where, why, and what it is looking at."""
    if isinstance(placement, TabletopMount):
        t = placement.target
        return (
            f"mounting robot on {t.support_category or t.support_name.split('_')[0]} at "
            f"({placement.xy[0]:.2f}, {placement.xy[1]:.2f}, {placement.z:.2f}) "
            f"yaw {np.degrees(placement.yaw):.0f} deg, "
            f"facing {t.object_category or t.object_name.split('_')[0]} "
            f"({placement.n_in_reach}/{t.n_objects_on_support} objects in reach, "
            f"{placement.clearance:.2f} m clear)"
        )
    where = f"({placement.pos[0]:.2f}, {placement.pos[1]:.2f}) yaw {np.degrees(placement.yaw):.0f} deg"
    if placement.target is None:
        return f"placing robot at {where} [{placement.source}]"
    t = placement.target
    return (
        f"placing robot at {where} [{placement.source}] "
        f"{placement.clearance:.2f} m from {t.object_category or t.object_name.split('_')[0]} "
        f"on {t.support_category or t.support_name.split('_')[0]} "
        f"(top {t.support_top_z:.2f} m, {t.n_objects_on_support} objects)"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene")
    ap.add_argument("--agent-radius", type=float, default=0.35)
    ap.add_argument("--reach", type=float, nargs=2, default=[0.40, 0.80])
    ap.add_argument("--targets", action="store_true", help="list every target, not just the choice")
    args = ap.parse_args()

    if args.targets:
        for t in find_grasp_targets(args.scene)[:20]:
            print(f"{t.reach_slack:5.2f}  {t.n_objects_on_support:2d}  "
                  f"{t.support_category or '?':28.28} {t.object_category or '?':20.20} "
                  f"z={t.object_xyz[2]:.2f}")
    p = find_robot_placement(args.scene, agent_radius=args.agent_radius, reach_range=tuple(args.reach))
    print(describe(p))
