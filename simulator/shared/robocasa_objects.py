"""RoboCasa's own kitchen objects on the counters of a kitchen scene (spec §2.1, §2.2).

RoboCasa's bare `Kitchen` scene holds fixtures only; its loose objects come from its task
environments, which sample them from its object library (`sample_kitchen_object`) and set
them on counters. The simulator's RoboCasa scenes keep no robosuite robot or environment, so
this does the same thing without them: it draws graspable objects from the same library
with the same sampler, and stands each on a free spot of a counter top (RoboCasa's own
`Counter.get_reset_regions()`, via `scenes.counter_regions`), upright, not touching another
object. The draw is a deterministic function of the layout and style.

Which registries are drawn from follows what is installed: the Lightwheel set is part of
`run.sh setup`; objaverse and aigen are used when `run.sh assets` fetched them.
"""

from __future__ import annotations

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
    for k in range(count):
        for _ in range(TRIES):
            idx = int(rng.choice(len(regions), p=weights))
            reg = regions[idx]
            kwargs, _info = sample_kitchen_object(
                "all", graspable=True, rng=rng, obj_registries=regs, max_size=MAX_SIZE)
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
            if not _free(xy, radius, [t for ts in taken.values() for t in ts]):
                continue
            z = float(reg["top_z"]) - float(obj.bottom_offset[2]) + DROP
            yaw = float(rng.uniform(-np.pi, np.pi))
            placed.append((obj, (xy[0], xy[1], z), yaw))
            taken[idx].append((xy[0], xy[1], radius))
            break
    return placed
