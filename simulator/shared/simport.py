#!/usr/bin/env python3
"""A command-line client of the simulation's private control port (stdlib only).

    python3 simulator/shared/simport.py [--sim-port 9080] hello
    python3 simulator/shared/simport.py robots
    python3 simulator/shared/simport.py scene
    python3 simulator/shared/simport.py readings <robot id>
    python3 simulator/shared/simport.py render --out scene.png [--width 1280 --height 720]
            [--robot <id>]                                  # frame a spawned robot
            [--lookat x y z --distance d --azimuth deg --elevation deg]
            [--pos x y z --target x y z [--fovy deg]]
            [--camera <robot id>/<camera>]                  # a model camera

With `--robot` and no other viewpoint the simulation frames that robot: looking at it
from in front (30 degrees above, or --elevation), turning around it until the line of
sight is clear. Output is JSON on stdout (the render writes a PNG).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocol  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sim-port", type=int, default=protocol.DEFAULT_SIM_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("hello", "robots", "scene"):
        sub.add_parser(name)
    r = sub.add_parser("readings")
    r.add_argument("robot")
    rd = sub.add_parser("render")
    rd.add_argument("--out", required=True)
    rd.add_argument("--width", type=int, default=1280)
    rd.add_argument("--height", type=int, default=720)
    rd.add_argument("--robot")
    rd.add_argument("--camera")
    rd.add_argument("--lookat", type=float, nargs=3)
    rd.add_argument("--distance", type=float, default=3.0)
    rd.add_argument("--azimuth", type=float, default=90.0)
    rd.add_argument("--elevation", type=float, default=-30.0)
    rd.add_argument("--pos", type=float, nargs=3)
    rd.add_argument("--target", type=float, nargs=3)
    rd.add_argument("--fovy", type=float)
    args = ap.parse_args(argv)

    c = protocol.Client(args.host, args.sim_port)
    try:
        if args.cmd in ("hello", "robots", "scene"):
            out = c.call(args.cmd)
        elif args.cmd == "readings":
            out = c.call("readings", robot=args.robot)
        else:
            if args.camera:
                view = {"camera": args.camera}
            elif args.pos:
                view = {"pos": args.pos, "target": args.target or args.lookat}
            elif args.lookat:
                view = {"lookat": args.lookat, "distance": args.distance,
                        "azimuth": args.azimuth, "elevation": args.elevation}
            elif args.robot:
                view = {"frame_robot": args.robot, "elevation": args.elevation}
            else:
                view = {}
            if args.fovy:
                view["fovy"] = args.fovy
            res = c.call("render", view=view, width=args.width, height=args.height,
                         format="png")
            with open(args.out, "wb") as fh:
                fh.write(res.pop("_payload"))
            out = dict(res, out=args.out, view=view)
        out.pop("id", None)
        print(json.dumps(out, indent=1))
    except protocol.RemoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
