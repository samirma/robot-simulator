"""Lay several `map_server` maps side by side in one world window.

A figure script, not console runtime -- hence `tools/` rather than the package, and hence
no dependency beyond the three the console already has.

The point of insisting on a *shared* world window is that these maps are supposed to be
in the same frame already. Ground truth comes out of the scene's own occupancy image, and
both SLAM maps come out of runs whose `/odom` is published in that same MuJoCo frame, so
nothing here fits or registers anything. If the panels do not line up, that is a finding
about the maps, not a reason to nudge them into agreement -- so `--check` reports the
overlap and says nothing about "aligning".

    python tools/render_maps.py truth=runs/gt lidar=runs/a visual=runs/b --out cmp.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

FREE, UNKNOWN, OCCUPIED = 254, 205, 0

COL_FREE = (250, 250, 250)
COL_UNKNOWN = (150, 150, 150)
COL_OCCUPIED = (35, 35, 35)
COL_TRUTH = (60, 60, 235)     # ground-truth wall outline, drawn over the SLAM panels

ZOOM = 5
LABEL_H = 78


class Map:
    """One `map_server` pair: the classified array plus where it sits in the world."""

    def __init__(self, path: Path) -> None:
        self.path = path
        meta = _read_yaml(path / "map.yaml")
        self.resolution = float(meta["resolution"])
        origin = str(meta["origin"]).strip("[] ").split(",")
        self.origin = np.array([float(origin[0]), float(origin[1])])
        # Row 0 of a PGM is the top of the image, i.e. the *highest* y. Flip on the way in
        # so this array is lowest-y-first, like the grid that produced it.
        self.data = np.flipud(_read_pgm(path / (meta.get("image") or "map.pgm")))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def bounds(self) -> Tuple[float, float, float, float]:
        h, w = self.data.shape
        return (
            self.origin[0], self.origin[1],
            self.origin[0] + w * self.resolution,
            self.origin[1] + h * self.resolution,
        )

    def sample(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Classify world points, UNKNOWN outside the map's own extent."""
        h, w = self.data.shape
        ix = np.floor((xs - self.origin[0]) / self.resolution).astype(np.int64)
        iy = np.floor((ys - self.origin[1]) / self.resolution).astype(np.int64)
        out = np.full(ix.shape, UNKNOWN, dtype=np.uint8)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        out[inside] = self.data[iy[inside], ix[inside]]
        return out


def metrics(sample: np.ndarray, truth: np.ndarray, resolution: float) -> Dict[str, float]:
    """How well one map agrees with the truth, on a shared lattice.

    Coverage and alignment are different failures and get separate numbers. A map can
    cover the whole house while sitting 30 cm to the left, and a map can be perfectly
    registered while having seen a third of it -- reporting one number for both hides
    whichever is the real problem.
    """
    true_free = truth == FREE
    true_wall = truth == OCCUPIED
    seen = sample != UNKNOWN
    free = sample == FREE
    wall = sample == OCCUPIED

    out: Dict[str, float] = {}
    out["coverage"] = _ratio((seen & true_free).sum(), true_free.sum())
    out["free_iou"] = _ratio((free & true_free).sum(), (free | true_free).sum())
    # Free space claimed where a wall really is: the dangerous direction, because a
    # planner will happily route through it.
    out["free_on_wall"] = _ratio((free & true_wall).sum(), free.sum())

    # Wall displacement: distance from each mapped wall cell to the nearest true wall.
    if true_wall.any() and wall.any():
        dist = cv2.distanceTransform(
            (~true_wall).astype(np.uint8), cv2.DIST_L2, 5
        ) * resolution
        errors = dist[wall]
        out["wall_median_m"] = float(np.median(errors))
        out["wall_p90_m"] = float(np.percentile(errors, 90))
    else:
        out["wall_median_m"] = float("nan")
        out["wall_p90_m"] = float("nan")
    return out


def _ratio(numerator, denominator) -> float:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator else float("nan")


def _read_yaml(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line and not line.strip().startswith("#"):
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def _read_pgm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if not raw.startswith(b"P5"):
        raise ValueError(f"{path}: not a binary PGM")
    fields: List[int] = []
    i = 2
    while len(fields) < 3:
        while i < len(raw) and raw[i : i + 1].isspace():
            i += 1
        if raw[i : i + 1] == b"#":
            while raw[i : i + 1] not in (b"\n", b""):
                i += 1
            continue
        start = i
        while i < len(raw) and not raw[i : i + 1].isspace():
            i += 1
        fields.append(int(raw[start:i]))
    i += 1
    width, height, _maxval = fields
    return np.frombuffer(raw[i : i + width * height], np.uint8).reshape(height, width)


def render(
    maps: List[Tuple[str, Map]],
    truth: Optional[Map],
    out_path: Path,
    zoom: int = ZOOM,
) -> Dict[str, Dict[str, float]]:
    lo_x = min(m.bounds()[0] for _, m in maps)
    lo_y = min(m.bounds()[1] for _, m in maps)
    hi_x = max(m.bounds()[2] for _, m in maps)
    hi_y = max(m.bounds()[3] for _, m in maps)
    resolution = min(m.resolution for _, m in maps)

    cols = int(math.ceil((hi_x - lo_x) / resolution))
    rows = int(math.ceil((hi_y - lo_y) / resolution))
    xs = lo_x + (np.arange(cols) + 0.5) * resolution
    ys = lo_y + (np.arange(rows) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys)

    truth_classified = None
    truth_edge = None
    report: Dict[str, Dict[str, float]] = {}
    if truth is not None:
        classified = truth.sample(gx.ravel(), gy.ravel()).reshape(rows, cols)
        truth_classified = classified
        # The *outline* of the true walls, not the walls themselves. Drawn filled, the
        # reference covers exactly the thing a reader is trying to check it against.
        walls = (classified == OCCUPIED).astype(np.uint8)
        eroded = cv2.erode(walls, np.ones((3, 3), np.uint8))
        truth_edge = (walls - eroded).astype(bool)

    panels = []
    for title, m in maps:
        classified = m.sample(gx.ravel(), gy.ravel()).reshape(rows, cols)
        img = np.zeros((rows, cols, 3), np.uint8)
        img[classified == UNKNOWN] = COL_UNKNOWN
        img[classified == FREE] = COL_FREE
        img[classified == OCCUPIED] = COL_OCCUPIED
        big = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        # Ground-truth walls over every panel including its own, so a reader can see the
        # registration rather than take it on trust.
        if truth_edge is not None:
            mask = cv2.resize(
                truth_edge.astype(np.uint8), (big.shape[1], big.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            big[mask] = COL_TRUTH
        big = cv2.flip(big, 0)   # +y up

        sub = ""
        if truth_classified is not None:
            m = metrics(classified, truth_classified, resolution)
            report[title] = m
            area = float((classified == FREE).sum()) * resolution * resolution
            sub = (f"{100 * m['coverage']:.0f}% of the floor mapped   {area:.1f} m2 free"
                   f"   walls off by {100 * m['wall_median_m']:.0f} cm (median)")
        panels.append(_caption(big, title, sub))

    height = max(p.shape[0] for p in panels)
    width = max(p.shape[1] for p in panels)
    gap = np.full((height, 18, 3), 255, np.uint8)
    row: List[np.ndarray] = []
    for i, p in enumerate(panels):
        if i:
            row.append(gap)
        canvas = np.full((height, width, 3), 255, np.uint8)
        canvas[: p.shape[0], : p.shape[1]] = p
        row.append(canvas)
    sheet = np.hstack(row)

    legend = np.full((46, sheet.shape[1], 3), 255, np.uint8)
    for i, (text, colour) in enumerate([
        ("ground-truth wall", COL_TRUTH), ("mapped free", (200, 200, 200)),
        ("mapped wall", COL_OCCUPIED), ("never seen", COL_UNKNOWN),
    ]):
        x = 14 + i * 250
        cv2.rectangle(legend, (x, 16), (x + 24, 32), colour, -1)
        cv2.putText(legend, text, (x + 32, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([sheet, legend]))
    return report


def _caption(img: np.ndarray, title: str, sub: str) -> np.ndarray:
    pad = np.full((LABEL_H, img.shape[1], 3), 255, np.uint8)
    out = np.vstack([pad, img])
    cv2.putText(out, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (90, 90, 90), 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("maps", nargs="+", metavar="TITLE=DIR",
                    help="map directories, in the order they should appear")
    ap.add_argument("--truth", metavar="TITLE",
                    help="which of them is ground truth; its walls overlay every panel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=int, default=ZOOM)
    args = ap.parse_args(argv)

    maps: List[Tuple[str, Map]] = []
    for spec in args.maps:
        title, _, path = spec.partition("=")
        if not path:
            title, path = Path(spec).name, spec
        maps.append((title, Map(Path(path))))

    truth = None
    if args.truth:
        truth = dict(maps).get(args.truth)
        if truth is None:
            raise SystemExit(f"error: --truth {args.truth} is not one of the maps given")

    for title, m in maps:
        x0, y0, x1, y1 = m.bounds()
        print(f"{title:10s} {m.shape[1]}x{m.shape[0]} @ {m.resolution} m  "
              f"world x {x0:.2f}..{x1:.2f}  y {y0:.2f}..{y1:.2f}")
    report = render(maps, truth, Path(args.out), zoom=args.zoom)
    if report:
        print()
        print(f"{'map':14s} {'coverage':>9s} {'free IoU':>9s} {'free-on-wall':>13s} "
              f"{'wall median':>12s} {'wall p90':>9s}")
        for title, m in report.items():
            print(f"{title:14s} {100*m['coverage']:8.1f}% {100*m['free_iou']:8.1f}% "
                  f"{100*m['free_on_wall']:12.1f}% {100*m['wall_median_m']:10.0f} cm "
                  f"{100*m['wall_p90_m']:7.0f} cm")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
