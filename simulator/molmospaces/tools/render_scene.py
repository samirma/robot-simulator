#!/usr/bin/env python
"""Render one or more offscreen views of a scene to PNG.

Useful for looking at a house without an interactive session (headless runs,
remote shells, or when the OS denies screen-capture permission).

    python tools/render_scene.py <scene.xml> --out house.png
"""

import argparse
import sys

import mujoco
import numpy as np


def scene_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    """Centre and radius of the static geometry, used to frame the free camera."""
    pos = data.geom_xpos
    if pos.size == 0:
        return np.zeros(3), 3.0
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2
    return centre, max(radius, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mjcf")
    ap.add_argument("--out", default="scene.png")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-30.0)
    ap.add_argument(
        "--distance-scale",
        type=float,
        default=1.6,
        dest="distance_scale",
        help="camera distance as a multiple of the scene radius",
    )
    ap.add_argument(
        "--settle",
        type=int,
        default=100,
        help="physics steps to run before rendering, so objects rest naturally",
    )
    ap.add_argument(
        "--hide",
        default="ceiling",
        help="comma-separated geom-name substrings to make invisible; "
        "defaults to the ceiling so the interior is visible. Pass '' to keep everything.",
    )
    ap.add_argument("--lookat-z", type=float, default=None, dest="lookat_z")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    # Scenes declare a modest offscreen framebuffer (1280x720 for iTHOR); the
    # Renderer refuses any request larger than it, so raise it to what we need.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

    # Houses are sealed, so an exterior camera just sees the roof. Making the
    # ceiling transparent gives the usual dollhouse view of the interior.
    patterns = [p.strip().lower() for p in args.hide.split(",") if p.strip()]
    if patterns:
        hidden = 0
        for gid in range(model.ngeom):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            if any(p in name for p in patterns):
                model.geom_rgba[gid, 3] = 0.0
                hidden += 1
        print(f"hid {hidden} geoms matching {patterns}", file=sys.stderr)

    data = mujoco.MjData(model)
    for _ in range(args.settle):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    centre, radius = scene_bounds(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = centre
    if args.lookat_z is not None:
        cam.lookat[2] = args.lookat_z
    cam.distance = radius * args.distance_scale
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    with mujoco.Renderer(model, args.height, args.width) as renderer:
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required to write PNGs", file=sys.stderr)
        return 1

    Image.fromarray(pixels).save(args.out)
    print(
        f"{args.out}  ({args.width}x{args.height}, mean luminance "
        f"{pixels.mean():.1f}, camera az={args.azimuth} el={args.elevation} "
        f"dist={cam.distance:.1f}m)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
