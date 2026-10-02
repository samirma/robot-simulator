"""Visual parts the model generators use, from the files robots_specs/tools/fetch_meshes.py
derives (read it for the conversion itself).

* `dae_parts(robot_dir, dae_rel)`: a COLLADA visual as one OBJ per material,
  `derived_meshes/<dae_rel>.m<k>.obj`, with that material's own diffuse colour, opacity and
  texture image from `derived_meshes/<dae_rel>.mtl`.
* `stl_parts(robot_dir, stl_rel)`: an STL over MuJoCo's 200000-face limit as the lossless
  parts `derived_meshes/<stl_rel>.part<k>.stl`; an STL under the limit is itself.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

STL_FACE_LIMIT = 200000


def dae_parts(robot_dir: Path, dae_rel: str) -> list[dict]:
    mtl = robot_dir / "derived_meshes" / f"{dae_rel}.mtl"
    parts, cur = [], None
    for line in mtl.read_text().splitlines():
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if f[0] == "newmtl":
            k = int(f[1][1:])
            cur = {"k": k, "obj": f"derived_meshes/{dae_rel}.m{k}.obj", "rgb": (1.0, 1.0, 1.0),
                   "alpha": 1.0, "texture": None}
            parts.append(cur)
        elif f[0] == "Kd":
            cur["rgb"] = tuple(float(v) for v in f[1:4])
        elif f[0] == "d":
            cur["alpha"] = float(f[1])
        elif f[0] == "map_Kd":
            # relative to the MTL; make it relative to the robot folder
            tex = (mtl.parent / f[1]).resolve().relative_to(robot_dir.resolve())
            cur["texture"] = str(tex)
    for p in parts:
        p["rgba"] = (*p["rgb"], p["alpha"])
    return parts


def stl_parts(robot_dir: Path, stl_rel: str) -> list[str]:
    data = (robot_dir / stl_rel).open("rb").read(84)
    (n,) = struct.unpack("<I", data[80:84])
    if n <= STL_FACE_LIMIT:
        return [stl_rel]
    return [f"derived_meshes/{stl_rel}.part{k}.stl" for k in range(math.ceil(n / STL_FACE_LIMIT))]
