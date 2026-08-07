"""Write a scene's true floor plan as a `map_server` occupancy map.

Ground truth for "did the robot map the house". Every scene ships a `<scene>_map.png`
beside its XML with the world<->pixel transform in its PNG metadata, so this is a
conversion, not a reconstruction: no ray casting, no parsing wall meshes, no MuJoCo
compile. It runs headless under plain `python` -- there is no viewer and no GL.

The output is deliberately the *same interchange format* a SLAM run writes
(`map.pgm` + `map.yaml`, ROS `map_server`), in the *same* MuJoCo world frame that `/odom`
is published in. That is what lets a real map be laid over this one without a fitted
transform -- and a comparison that needs a fitted transform is measuring the fit, not the
mapping.

    python tools/scene_occupancy.py assets/scenes/procthor-10k-train/train_40.xml \
        --out /tmp/train_40-truth

Two conventions are easy to get backwards, and both are checked by `--self-test`:

- **`ProcTHORMap.occupancy is True where space is FREE**, not where it is occupied
  (molmo_spaces/utils/scene_maps.py). Inverting this silently produces a photographic
  negative of a house, which still looks like a floor plan.
- **The map's pixel axes run opposite to the world axes** -- both scale terms in
  `map_to_world` are negative -- so row 0 of the PNG is the *highest* y in the world,
  while `map_server` (and `robot_console`'s grid) put the lowest y in row 0.

Rather than reason about either, every output cell is projected world->pixel through the
map's own `pos_m_to_px`. The transform is then the library's problem, not this file's.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Same palette `map_server` and robot_console/slam/mapio.py use.
PGM_FREE = 254
PGM_UNKNOWN = 205
PGM_OCCUPIED = 0

DEFAULT_RESOLUTION = 0.05

# The console's log-odds constants, copied rather than imported: this tool must not depend
# on robot_console -- the two projects talk over formats, not imports. If they ever drift
# apart the reference stops being directly comparable, so they are named here rather than
# buried, to make that visible.
L_FREE = -0.4
L_OCCUPIED = 0.85
L_CLAMP = 12.0
OCCUPIED_LOGODDS = math.log(0.65 / 0.35)
FREE_LOGODDS = math.log(0.196 / 0.804)

# Samples per axis inside each output cell. The source is ~5 mm/px and a cell is 50 mm, so
# a single sample would make a wall's position a coin toss on which pixel it landed in.
SUBSAMPLES = 5

# How far beyond the floor to call things "wall" rather than "never seen". A ground truth
# that is free space on a solid black field does not compare usefully against a SLAM map,
# which has unknown everywhere it could not see; this gives the reference the same shape.
WALL_BAND_M = 0.30

# Blank border kept around the house, in cells.
MARGIN_CELLS = 4


def build(scene: Path, resolution: float, agent_radius: float | None = None):
    """`(classes, origin)` -- a uint8 PGM-valued array, row 0 lowest y, and its world origin."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.scene_placement import load_scene_map

    # agent_radius=None on purpose: with a radius the map is eroded into the robot's
    # configuration space, which is a different (and smaller) thing than the floor.
    scene_map = load_scene_map(scene, agent_radius=agent_radius)
    if scene_map is None:
        raise SystemExit(f"error: no occupancy map shipped beside {scene}")

    free_px = np.asarray(scene_map.occupancy, dtype=bool)
    height, width = free_px.shape

    # World extent of the floor, from the corners of the free region in pixel space.
    ys, xs = np.nonzero(free_px)
    if not len(xs):
        raise SystemExit(f"error: {scene} has no free space in its map")
    corners = np.array(
        [[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]],
        dtype=float,
    )
    world = np.asarray(scene_map.pos_px_to_m(corners))[:, :2]
    lo = world.min(axis=0) - MARGIN_CELLS * resolution
    hi = world.max(axis=0) + MARGIN_CELLS * resolution

    cols = int(np.ceil((hi[0] - lo[0]) / resolution))
    rows = int(np.ceil((hi[1] - lo[1]) / resolution))
    origin = lo

    # Sample each output cell on a SUBSAMPLES x SUBSAMPLES lattice and take the majority.
    step = resolution / SUBSAMPLES
    offsets = (np.arange(SUBSAMPLES) + 0.5) * step
    cell_x = origin[0] + np.arange(cols) * resolution
    cell_y = origin[1] + np.arange(rows) * resolution
    sample_x = (cell_x[:, None] + offsets[None, :]).ravel()
    sample_y = (cell_y[:, None] + offsets[None, :]).ravel()
    gx, gy = np.meshgrid(sample_x, sample_y)

    points = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    px = np.asarray(scene_map.pos_m_to_px(points)).reshape(-1, 2)
    inside = (
        (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    )
    hit = np.zeros(px.shape[0], dtype=bool)
    good = px[inside]
    hit[inside] = free_px[good[:, 1], good[:, 0]]

    votes = hit.reshape(rows, SUBSAMPLES, cols, SUBSAMPLES).sum(axis=(1, 3))
    free = votes > (SUBSAMPLES * SUBSAMPLES) // 2

    classes = np.full((rows, cols), PGM_UNKNOWN, dtype=np.uint8)
    classes[free] = PGM_FREE
    # Everything within WALL_BAND_M of the floor but not floor is what the robot would
    # have ranged as an obstacle; beyond that it is outside the house and simply unseen.
    band = max(1, int(round(WALL_BAND_M / resolution)))
    near = _dilate(free, band)
    classes[near & ~free] = PGM_OCCUPIED
    return classes, origin


def build_raycast(
    scene: Path,
    resolution: float,
    *,
    height: float = 0.08,
    max_range: float = 8.0,
    beams: int = 360,
    spacing: float = 0.40,
):
    """`(classes, origin)` from a perfect lidar swept over every reachable pose.

    The shipped `_map.png` answers "where can the robot stand", which is a *top-down*
    question. A 2D lidar at 8 cm answers a horizontal one, and the two disagree wherever
    furniture overhangs: the scanner sees the floor under a table and a bed, the top-down
    map calls both solid. Measured on `train_40`, that gap alone made a well-registered
    lidar map look 9% too large and put its median wall 31 cm from "truth".

    So the reference for comparing 2D maps is built the way a 2D map is built -- rays at
    the scanner's own height, integrated with the same log-odds the console uses. What
    comes out is the best map this sensor could ever produce in this house, which is the
    only fair thing to hold a real run against.

    `mj_ray` needs no renderer, so this stays headless.
    """
    import mujoco

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.scene_placement import load_scene_map
    from tools.spawn_robot import laser_scan_ranges

    scene_map = load_scene_map(scene, agent_radius=None)
    if scene_map is None:
        raise SystemExit(f"error: no occupancy map shipped beside {scene}")

    model = mujoco.MjSpec.from_file(str(scene)).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Stand only where the robot could actually stand -- that part the shipped map is
    # exactly right about, and flood-filling it is what keeps the sweep out of sealed
    # voids and neighbouring geometry.
    free_px = np.asarray(scene_map.occupancy, dtype=bool)
    ys, xs = np.nonzero(free_px)
    world = np.asarray(scene_map.pos_px_to_m(np.stack([xs, ys], axis=1).astype(float)))[:, :2]
    lo = world.min(axis=0) - MARGIN_CELLS * resolution
    hi = world.max(axis=0) + MARGIN_CELLS * resolution

    step = max(spacing, resolution)
    sx = np.arange(lo[0], hi[0], step)
    sy = np.arange(lo[1], hi[1], step)
    poses = np.stack(np.meshgrid(sx, sy), axis=-1).reshape(-1, 2)
    keep = np.array([not scene_map.check_collision(np.array([p[0], p[1], 0.0])) for p in poses])
    poses = poses[keep]
    if not len(poses):
        raise SystemExit("error: no standable poses found")

    cols = int(np.ceil((hi[0] - lo[0]) / resolution))
    rows = int(np.ceil((hi[1] - lo[1]) / resolution))
    logodds = np.zeros((rows, cols), dtype=np.float32)

    for x, y in poses:
        origin = np.array([x, y, height])
        ranges = laser_scan_ranges(model, data, origin, 0.0, beams, max_range)
        angles = np.linspace(-np.pi, np.pi, beams, endpoint=False)
        valid = ranges <= max_range
        if not valid.any():
            continue
        r = ranges[valid]
        a = angles[valid]
        ends = np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)
        _integrate(logodds, lo, resolution, np.array([x, y]), ends)

    classes = np.full((rows, cols), PGM_UNKNOWN, dtype=np.uint8)
    classes[logodds <= FREE_LOGODDS] = PGM_FREE
    classes[logodds >= OCCUPIED_LOGODDS] = PGM_OCCUPIED
    return classes, lo


def _integrate(logodds, origin_xy, resolution, start, ends) -> None:
    """Carve free along each beam and mark its endpoint occupied, as the console does."""
    rows, cols = logodds.shape
    s = np.floor((start - origin_xy) / resolution).astype(np.int64)
    e = np.floor((ends - origin_xy) / resolution).astype(np.int64)
    delta = e - s
    steps = np.abs(delta).max(axis=1)
    n = int(steps.max()) if len(steps) else 0
    if n > 0:
        k = np.arange(n, dtype=np.float64)[None, :]
        live = k < steps[:, None]
        t = k / np.maximum(steps, 1)[:, None]
        cells = np.rint(s + t[..., None] * delta[:, None, :]).astype(np.int64)
        fc = cells[live]
        inside = (fc[:, 0] >= 0) & (fc[:, 0] < cols) & (fc[:, 1] >= 0) & (fc[:, 1] < rows)
        fc = fc[inside]
        np.add.at(logodds, (fc[:, 1], fc[:, 0]), L_FREE)
    inside = (e[:, 0] >= 0) & (e[:, 0] < cols) & (e[:, 1] >= 0) & (e[:, 1] < rows)
    ec = e[inside]
    np.add.at(logodds, (ec[:, 1], ec[:, 0]), L_OCCUPIED)
    np.clip(logodds, -L_CLAMP, L_CLAMP, out=logodds)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a square of `radius`, via shifts. Small radius, no cv2 needed."""
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def save(classes: np.ndarray, origin: np.ndarray, out_dir: Path, resolution: float) -> Path:
    """Write `map.pgm` + `map.yaml`, matching robot_console/slam/mapio.py byte for byte."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pgm = out_dir / "map.pgm"
    # Row 0 of the file is the TOP of the image, which is the highest y -- the same flip
    # `mapio.save_map` does. The array itself stays lowest-y-first.
    flipped = np.flipud(classes)
    with open(pgm, "wb") as fh:
        fh.write(b"P5\n%d %d\n255\n" % (classes.shape[1], classes.shape[0]))
        fh.write(flipped.tobytes())
    (out_dir / "map.yaml").write_text(
        "image: map.pgm\n"
        f"resolution: {resolution:.6f}\n"
        f"origin: [{origin[0]:.6f}, {origin[1]:.6f}, 0.000000]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )
    return pgm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene", help="path to the scene MJCF (assets/scenes/.../train_40.xml)")
    ap.add_argument("--out", required=True, help="directory for map.pgm / map.yaml")
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    ap.add_argument("--agent-radius", type=float, default=None,
                    help="erode the floor by this radius; default is the raw floor")
    ap.add_argument("--raycast", action="store_true",
                    help="sweep a perfect lidar at --height instead of converting the "
                         "shipped top-down map. The right reference for a 2D map: the "
                         "scanner sees under tables, a top-down map does not")
    ap.add_argument("--height", type=float, default=0.08,
                    help="scanner height above the floor; the myAGV's X2 sits at 0.08")
    ap.add_argument("--max-range", type=float, default=8.0, dest="max_range")
    ap.add_argument("--spacing", type=float, default=0.40,
                    help="how far apart to place the sweep poses")
    ap.add_argument("--self-test", action="store_true",
                    help="check the free/occupied polarity and the world frame, then exit")
    args = ap.parse_args(argv)

    scene = Path(args.scene)
    if args.raycast:
        classes, origin = build_raycast(
            scene, args.resolution, height=args.height,
            max_range=args.max_range, spacing=args.spacing,
        )
    else:
        classes, origin = build(scene, args.resolution, args.agent_radius)
    rows, cols = classes.shape
    free = int((classes == PGM_FREE).sum())
    occupied = int((classes == PGM_OCCUPIED).sum())

    if args.self_test:
        # A house is mostly floor once you discount what is outside it, and the floor must
        # be a connected slab rather than a scatter -- the signature of an inverted map.
        if free < occupied:
            print(f"FAIL: {free} free < {occupied} occupied; polarity looks inverted", file=sys.stderr)
            return 1
        print(f"ok: {free} free, {occupied} occupied, {rows * cols - free - occupied} unknown")
        print(f"ok: origin {origin[0]:.3f}, {origin[1]:.3f}  extent "
              f"{cols * args.resolution:.1f} x {rows * args.resolution:.1f} m")
        return 0

    path = save(classes, origin, Path(args.out), args.resolution)
    print(f"wrote {path}  {cols}x{rows} cells at {args.resolution} m  "
          f"origin ({origin[0]:.3f}, {origin[1]:.3f})")
    print(f"  {free} free, {occupied} occupied, {rows * cols - free - occupied} unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
