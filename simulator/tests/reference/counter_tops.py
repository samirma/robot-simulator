#!/usr/bin/env python
"""The counter tops of one RoboCasa kitchen, from RoboCasa's own API alone (not this
project's code), as JSON for `integration/test_robocasa_objects.py`: every `Counter`
fixture's `get_reset_regions()` -- the worktop minus its sink and hob cut-outs, what
RoboCasa places task objects on -- in world coordinates.

    <robocasa venv python> counter_tops.py <layout> <style>

Run it in the engine's environment (`simulator/robocasa/env.sh`). The kitchen is built as
RoboCasa builds it: `KitchenArena(layout, style)` with its origin at the world's.

Output: [{"name", "centre": [x, y], "half": [hx, hy], "top_z", "rot"}] -- a rectangle of
half-extents `half` about `centre`, turned `rot` about z, at height `top_z`.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys

import numpy as np


def main() -> int:
    layout, style = int(sys.argv[1]), int(sys.argv[2])
    with contextlib.redirect_stdout(sys.stderr):
        import robocasa.models

        robocasa.models.assets_root = os.environ["ROBOCASA_ASSETS_DIR"]
        from robocasa.models.fixtures.counter import Counter
        from robocasa.models.scenes.kitchen_arena import KitchenArena

        arena = KitchenArena(layout_id=layout, style_id=style, rng=np.random.default_rng(0))
        arena.set_origin([0, 0, 0])
        tops = []
        for name, fixture in arena.fixtures.items():
            if not isinstance(fixture, Counter):
                continue
            rot = float(getattr(fixture, "rot", 0.0) or 0.0)
            c, s = math.cos(rot), math.sin(rot)
            for region, r in fixture.get_reset_regions(env=None).items():
                ox, oy, oz = (float(v) for v in r["offset"])
                tops.append({"name": f"{name}/{region}",
                             "centre": [float(fixture.pos[0]) + c * ox - s * oy,
                                        float(fixture.pos[1]) + s * ox + c * oy],
                             "half": [float(v) / 2.0 for v in r["size"][:2]],
                             "top_z": float(fixture.pos[2]) + oz, "rot": rot})
    print(json.dumps(tops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
