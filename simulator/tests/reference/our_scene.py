#!/usr/bin/env python
"""Build one engine's scene the way `kitchen.sh start` does -- `Simulation.load()`, the six
worktop objects staged with it -- with no physics stepped and no control port, and print
what the scene-parity check compares with the reference (`scene_views.describe`, the
objects cleared at start) as JSON; with `--views DIR`, also its renders.

    <engine venv python> our_scene.py molmospaces|robocasa --scene <s> [--views DIR]

Run it in the engine's environment (`simulator/<engine>/env.sh`), as `run.sh start` runs
the simulation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene_views  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=("molmospaces", "robocasa"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--views", type=Path, default=None)
    args = ap.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        import simulation
        import world

        world.World.start = lambda self: None        # the scene as loaded: nothing stepped
        sim = simulation.Simulation(argparse.Namespace(engine=args.engine, scene=args.scene,
                                                       sim_port=0, mujoco=False))
        sim.listener = socket.socket()                 # no control port: nothing accepted
        sim.listener.close()
        sim.load()
        w = sim.world
        out = scene_views.describe(w.model, w.data)
        out["cleared"] = list(w.scene_staging.cleared) if w.scene_staging is not None else []
        # RoboCasa: the fixtures its own fixture placement set on others (spec §2.1)
        arena = w.scene.extra.get("arena")
        out["placed_fixtures"] = sorted(getattr(arena, "fixture_placements", None) or [])
        if args.views is not None:
            out["views"] = scene_views.render(w.model, w.data, args.engine, args.views)
    print(json.dumps(out), flush=True)
    sys.stderr.flush()
    os._exit(0)                                        # the render threads hold GL contexts


if __name__ == "__main__":
    main()
