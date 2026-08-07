"""Saving and loading maps in the format the myAGV's own navigation stack reads.

`myagv_navigation`'s `navigation_active.launch` feeds `map_server` a `.pgm`/`.yaml` pair,
so writing that pair means a map built in the simulator can be handed to the real robot
and vice versa. It is also lossy -- three grey levels out of a continuous log-odds field
-- so a `.npz` sidecar carries the raw values, letting a map be reloaded and *kept
building* rather than only navigated.

The yaml is written and read by hand. Adding PyYAML for eleven lines of flat key-value
would put a fourth dependency in a project whose whole point is that it needs three.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from robot_console.slam.grid import (
    FREE_LOGODDS,
    OCCUPIED_LOGODDS,
    OccupancyGrid,
)

MAP_NAME = "map"
PGM_NAME = f"{MAP_NAME}.pgm"
YAML_NAME = f"{MAP_NAME}.yaml"
NPZ_NAME = f"{MAP_NAME}.npz"

# map_server's palette: 0 occupied, 254 free, 205 unknown ("negate: 0" semantics).
PGM_OCCUPIED = 0
PGM_FREE = 254
PGM_UNKNOWN = 205

OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.196

PathLike = Union[str, os.PathLike]


def _resolve(path: PathLike) -> Tuple[Path, Path, Path]:
    """Accept a directory, a `map.yaml`, or a stem, and return (yaml, pgm, npz).

    An extensionless path is genuinely ambiguous -- `runs/house` is a directory holding
    `map.yaml`, while `runs/house/map` is a stem. Disambiguated by looking: an existing
    directory is a directory, and a path with a `.yaml` beside it is a stem. Neither
    (a fresh save) falls back to the directory reading, which is what `--out` means.
    """
    p = Path(path)
    if p.suffix in (".yaml", ".pgm", ".npz"):
        stem, parent = p.stem, p.parent
    elif p.suffix:
        stem, parent = p.name, p.parent
    elif not p.is_dir() and p.with_suffix(".yaml").exists():
        stem, parent = p.name, p.parent
    else:
        stem, parent = MAP_NAME, p
    return parent / f"{stem}.yaml", parent / f"{stem}.pgm", parent / f"{stem}.npz"


def to_pgm_image(grid: OccupancyGrid) -> np.ndarray:
    """The grid as a map_server greyscale image.

    Row order is flipped: a pgm's first row is the **top** of the image, which is the
    highest y, while `grid.data` row 0 is the lowest y at `origin`.
    """
    img = np.full(grid.data.shape, PGM_UNKNOWN, dtype=np.uint8)
    img[grid.data <= FREE_LOGODDS] = PGM_FREE
    img[grid.data >= OCCUPIED_LOGODDS] = PGM_OCCUPIED
    return np.flipud(img)


def crop_to_content(grid: OccupancyGrid, *, border_cells: int = 4) -> OccupancyGrid:
    """A copy trimmed to the cells that carry evidence, `origin` moved to match.

    The map grows in chunks and never shrinks, so a finished map carries a wide margin of
    cells no ray ever reached. Left in, that margin is most of the file, pushes the actual
    rooms into a corner of the image, and makes every consumer pay for empty space.
    map_server maps are conventionally trimmed; this makes ours look like one.
    """
    known = np.argwhere(grid.data != 0.0)
    if known.size == 0:
        return grid.copy()
    (y0, x0), (y1, x1) = known.min(axis=0), known.max(axis=0) + 1
    y0 = max(0, y0 - border_cells)
    x0 = max(0, x0 - border_cells)
    y1 = min(grid.height, y1 + border_cells)
    x1 = min(grid.width, x1 + border_cells)
    return OccupancyGrid(
        grid.resolution,
        # Cropping from the low corner moves cell (0, 0), so the origin moves with it.
        origin=grid.origin + np.array([x0, y0]) * grid.resolution,
        data=grid.data[y0:y1, x0:x1].copy(),
    )


def save_map(grid: OccupancyGrid, path: PathLike, *, sidecar: bool = True,
             crop: bool = True) -> Path:
    """Write `map.pgm` + `map.yaml` (+ `map.npz`). Returns the yaml path."""
    yaml_path, pgm_path, npz_path = _resolve(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if crop:
        grid = crop_to_content(grid)

    if not cv2.imwrite(str(pgm_path), to_pgm_image(grid)):
        raise OSError(f"could not write {pgm_path}")

    yaml_path.write_text(
        # Written in map_server's own field order so a diff against a map produced on the
        # robot shows content differences rather than reordering.
        f"image: {pgm_path.name}\n"
        f"resolution: {grid.resolution:.6f}\n"
        f"origin: [{grid.origin[0]:.6f}, {grid.origin[1]:.6f}, 0.000000]\n"
        f"negate: 0\n"
        f"occupied_thresh: {OCCUPIED_THRESH}\n"
        f"free_thresh: {FREE_THRESH}\n"
    )

    if sidecar:
        np.savez_compressed(
            npz_path,
            data=grid.data,
            resolution=np.float64(grid.resolution),
            origin=grid.origin,
        )
    return yaml_path


def load_map(path: PathLike) -> OccupancyGrid:
    """Load a map, preferring the `.npz` sidecar when it matches the yaml.

    Without the sidecar the log-odds are reconstructed from the three grey levels at a
    confidence just past each threshold: enough to plan and localize against, deliberately
    not enough to outvote a live observation, since a loaded map is a claim about the past.
    """
    yaml_path, pgm_path, npz_path = _resolve(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"no map yaml at {yaml_path}")
    meta = read_yaml(yaml_path)
    resolution = float(meta["resolution"])
    origin = np.asarray(meta["origin"][:2], dtype=np.float64)

    if npz_path.exists():
        with np.load(npz_path) as bundle:
            data = np.asarray(bundle["data"], dtype=np.float32)
            # The yaml is the authority on geometry: if someone edited it, or swapped the
            # pgm, the sidecar is stale and its raw values must not be trusted.
            if (
                abs(float(bundle["resolution"]) - resolution) < 1e-9
                and np.allclose(np.asarray(bundle["origin"], dtype=np.float64), origin)
            ):
                return OccupancyGrid(resolution, origin=origin, data=data)

    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"no map image at {pgm_path}")
    if int(meta.get("negate", 0)):
        img = 255 - img
    img = np.flipud(img)
    data = np.zeros(img.shape, dtype=np.float32)
    data[img <= PGM_OCCUPIED + 10] = OCCUPIED_LOGODDS * 2.0
    data[img >= PGM_FREE - 10] = FREE_LOGODDS * 2.0
    return OccupancyGrid(resolution, origin=origin, data=data)


def read_yaml(path: PathLike) -> dict:
    """A deliberately tiny reader for map_server's flat `key: value` yaml.

    Handles exactly what map_server writes: scalars, quoted strings and one-line lists.
    Anything else raises rather than being silently misread.
    """
    meta: dict = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"cannot parse map yaml line: {raw!r}")
        key, _, value = line.partition(":")
        value = value.strip()
        # An indented line or an empty value means nesting, which this reader does not
        # do. Raising is the point: quietly reading `origin:` followed by an indented
        # block as the string "" would put the map at the wrong place on the floor.
        if not value or raw[:1].isspace():
            raise ValueError(
                f"map yaml line {raw!r} is nested or has no value; this reader handles "
                "only the flat key/value form map_server writes"
            )
        meta[key.strip()] = _scalar(value)
    for required in ("image", "resolution", "origin"):
        if required not in meta:
            raise ValueError(f"map yaml is missing {required!r}")
    return meta


def _scalar(text: str):
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(part.strip()) for part in inner.split(",")] if inner else []
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text):
        return float(text) if any(c in text for c in ".eE") else int(text)
    return text


def describe(path: PathLike) -> Optional[str]:
    """A one-line summary for the console, or None if there is no map there."""
    try:
        grid = load_map(path)
    except (OSError, ValueError):
        return None
    classes = grid.classify()
    known = int((classes != 0).sum())
    area = known * grid.resolution ** 2
    return (
        f"{grid.width}x{grid.height} cells @ {grid.resolution:.3f} m, "
        f"origin ({grid.origin[0]:.2f}, {grid.origin[1]:.2f}), {area:.1f} m^2 mapped"
    )
