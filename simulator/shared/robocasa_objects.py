"""RoboCasa's own kitchen objects on the counters of a kitchen scene (spec §2.1, §2.2).

RoboCasa's bare `Kitchen` scene holds fixtures only; its loose objects come from its task
environments, which sample them from its object library (`sample_kitchen_object`) and set
them on counters. The simulator's RoboCasa scenes keep no robosuite robot or environment, so
this does the same thing without them: it draws graspable objects from the same library
with the same sampler, and stands each on a free spot of a counter top (RoboCasa's own
`Counter.get_reset_regions()`, via `scenes.counter_regions`), upright, not touching another
object. The draw is a deterministic function of the layout and style.

Each stays where it is set (spec §2.1, amended 2026-10-03): the scene is solved by
constraint island (`scenes.solve_by_island`), and an object that does not stand still
upright -- a dish brush or whisk that tips over off its end, a marshmallow that keeps
rolling -- is drawn again (`stands_still`: set alone on a flat top the way the counters get
it, under the scene's solver).

Which registries are drawn from follows what is installed: the Lightwheel set is part of
`run.sh setup`; objaverse and aigen are used when `run.sh assets` fetched them.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np

#: objects per square metre of counter top, and the most put in one scene
PER_SQUARE_METRE = 8.0
MAX_OBJECTS = 24
#: nothing bigger than this (x, y, z, metres): a countertop item, not a pot rack
MAX_SIZE = (0.30, 0.30, 0.30)
#: clearance kept between objects and from a region's edge (metres)
GAP = 0.02
#: attempts at a free spot per object
TRIES = 40
#: lift above the counter top at which an object is released (it settles by itself)
DROP = 0.004
#: the stand-still test: physics time an object is let settle, then watched (s); at rest it
#: is tilted less than MAX_TILT (rad) from upright and moves less than MAX_CREEP (m) watched
SETTLE_S, WATCH_S = 1.0, 2.0
MAX_TILT, MAX_CREEP = 0.17, 0.002

#: stand-still results by (model file, scale), for the life of the process
_stands_still: dict = {}


def registries() -> tuple:
    """The object registries whose assets are installed, in RoboCasa's naming."""
    import robocasa.models

    root = Path(robocasa.models.assets_root) / "objects"
    have = []
    for name, folder in (("objaverse", "objaverse"), ("aigen", "aigen_objs"),
                         ("lightwheel", "lightwheel")):
        if (root / folder).is_dir() and any((root / folder).iterdir()):
            have.append(name)
    return tuple(have)


def _free(spot, radius, taken) -> bool:
    return all(np.hypot(spot[0] - x, spot[1] - y) >= radius + r + GAP for x, y, r in taken)


def stands_still(kwargs: dict) -> bool:
    """Whether the object `kwargs` (from `sample_kitchen_object`) stands still upright when
    set on a counter: released `DROP` above a flat top, alone, under the scene's solver
    (`scenes.solve_by_island`), it has tilted less than `MAX_TILT` after `SETTLE_S` and moves
    less than `MAX_CREEP` in the `WATCH_S` after. Deterministic, and remembered per model."""
    key = (kwargs["mjcf_path"], tuple(np.atleast_1d(kwargs.get("scale", 1.0)).tolist()))
    if key not in _stands_still:
        _stands_still[key] = _stand_still_test(kwargs)
    return _stands_still[key]


def _stand_still_test(kwargs: dict) -> bool:
    import mujoco
    from robocasa.models.objects.objects import MJCFObject
    from robosuite.models.arenas import EmptyArena
    from robosuite.models.tasks import ManipulationTask

    import scenes

    top = 1.0
    obj = MJCFObject(name="stand_test", **kwargs)
    obj.set_pos((0.0, 0.0, top - float(obj.bottom_offset[2]) + DROP))
    obj.set_euler((0.0, 0.0, 0.0))
    with contextlib.redirect_stdout(sys.stderr):
        task = ManipulationTask(mujoco_arena=EmptyArena(), mujoco_robots=[],
                                mujoco_objects=[obj], enable_multiccd=True,
                                enable_sleeping_islands=False)
        spec = mujoco.MjSpec.from_string(task.get_xml())
    # a counter top: a box with MuJoCo's default contact, as RoboCasa's counter tops have
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.02],
                            pos=[0.0, 0.0, top - 0.02])
    scenes.solve_by_island(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
    up = d.xmat[body].reshape(3, 3)[:, 2].copy()
    mujoco.mj_step(m, d, int(SETTLE_S / m.opt.timestep))
    settled = d.xpos[body].copy()
    mujoco.mj_step(m, d, int(WATCH_S / m.opt.timestep))
    tilt = float(np.arccos(np.clip(d.xmat[body].reshape(3, 3)[:, 2] @ up, -1.0, 1.0)))
    creep = float(np.linalg.norm(d.xpos[body] - settled))
    return tilt < MAX_TILT and creep < MAX_CREEP


def populate(arena, regions, seed: int):
    """[(MJCFObject, (x, y, z), yaw)] for the counters' `regions` (`scenes.counter_regions`)."""
    import importlib

    from robocasa.models.objects import kitchen_object_utils, kitchen_objects
    from robocasa.models.objects.objects import MJCFObject

    # RoboCasa lists its object library once, at import, from its default asset path; this
    # engine keeps the assets in its own tree (`ROBOCASA_ASSETS_DIR`), so list it again
    # (the raw category table first: building the library rewrites it in place)
    importlib.reload(kitchen_objects)
    kitchen_object_utils = importlib.reload(kitchen_object_utils)
    sample_kitchen_object = kitchen_object_utils.sample_kitchen_object
    regs = registries()
    if not regs:
        print("  (no RoboCasa object assets installed: run simulator/robocasa/run.sh setup)",
              file=sys.stderr)
        return []
    rng = np.random.default_rng(seed)
    area = sum(4.0 * float(r["half"][0] * r["half"][1]) for r in regions)
    count = int(min(MAX_OBJECTS, round(area * PER_SQUARE_METRE)))
    # regions in proportion to their area
    weights = np.array([float(r["half"][0] * r["half"][1]) for r in regions])
    weights = weights / weights.sum() if weights.sum() > 0 else None
    placed, taken = [], {i: [] for i in range(len(regions))}
    # the fixtures RoboCasa set on the counters (`scenes.place_fixtures`: a toaster, a knife
    # block...) are kept clear of as well, as RoboCasa's own object placement does
    fixed = [(float(pos[0]), float(pos[1]), 0.5 * float(np.hypot(fxtr.size[0], fxtr.size[1])))
             for pos, _quat, fxtr in getattr(arena, "fixture_placements", {}).values()]
    for k in range(count):
        for _ in range(TRIES):
            idx = int(rng.choice(len(regions), p=weights))
            reg = regions[idx]
            kwargs, _info = sample_kitchen_object(
                "all", graspable=True, rng=rng, obj_registries=regs, max_size=MAX_SIZE)
            if not stands_still(kwargs):
                continue
            obj = MJCFObject(name=f"rc_obj_{k}", **kwargs)
            radius = float(obj.horizontal_radius)
            half = np.asarray(reg["half"], float) - radius - GAP
            if (half <= 0).any():
                continue
            local = rng.uniform(-half, half)
            c, s = np.cos(reg["rot"]), np.sin(reg["rot"])
            xy = (float(reg["centre"][0] + c * local[0] - s * local[1]),
                  float(reg["centre"][1] + s * local[0] + c * local[1]))
            # clear of every object already stood, on this counter or a neighbouring one
            # (regions of adjacent counters can meet)
            if not _free(xy, radius, fixed + [t for ts in taken.values() for t in ts]):
                continue
            z = float(reg["top_z"]) - float(obj.bottom_offset[2]) + DROP
            yaw = float(rng.uniform(-np.pi, np.pi))
            placed.append((obj, (xy[0], xy[1], z), yaw))
            taken[idx].append((xy[0], xy[1], radius))
            break
    return placed
