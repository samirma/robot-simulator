#!/usr/bin/env python3
"""Fetch the robot meshes robots_specs/ references but git does not commit.

    python3 robots_specs/tools/fetch_meshes.py               every robot
    python3 robots_specs/tools/fetch_meshes.py so101 ainex   only these (a composite id
                                                             brings both components)
    python3 robots_specs/tools/fetch_meshes.py --check       verify only, fetch nothing

`robots_specs/meshes.sha256` lists every file (`<sha256>  <path from the repo root>`).
Each one comes from its robot's pinned source (SOURCES below, the revisions recorded in
robots_specs/high_level_spec.md): a sparse, blobless git checkout at the pinned commit,
or, for the ROSMASTER X3 PLUS, range reads of the one nested zip it needs from Yahboom's
pinned Google Drive archive. MuJoCo reads no COLLADA and no STL over 200000 faces, so
some files are also derived, deterministically and with the standard library only, under
`<id>/derived_meshes/` (same relative path); the derived files are in the manifest too:

* `<rel>.stl` from `<rel>.dae`: every triangle, one binary STL (collision geometry);
* `<rel>.dae.mtl` + `<rel>.dae.m<k>.obj` from `<rel>.dae`: the visual mesh split into one
  OBJ per COLLADA material (positions, the file's normals and texture coordinates), and
  one MTL holding each material's own diffuse colour, opacity and texture image;
* `<rel>.STL.part<k>.stl` from an STL over MuJoCo's face limit: its triangle records
  copied verbatim, in order, into parts of at most 200000 faces (lossless).

Every file is written to a staging directory first, checked against the manifest, and only
then moved into place. A mismatch is refused (non-zero exit) and leaves no unverified file
behind; a file already in place that does not match is removed. Files already matching are
left alone, so re-running is a no-op. Sources are cached under
${XDG_CACHE_HOME:-~/.cache}/robot-simulator. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SPECS = Path(__file__).resolve().parent.parent          # robots_specs/
ROOT = SPECS.parent                                     # repository root
MANIFEST = SPECS / "meshes.sha256"
CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "robot-simulator"

#: Pinned sources. Revisions are those of robots_specs/high_level_spec.md (a test checks).
GIT = {
    "SO-ARM100": ("https://github.com/TheRobotStudio/SO-ARM100",
                  "aec17bbc256d1a7342d53aaa4950595d4c30b40d"),
    "myagv_ros": ("https://github.com/elephantrobotics/myagv_ros",
                  "c71f3cc574e5ed1973a925238eabe88662cfa701"),
    "ainex": ("https://github.com/Hiwonder/ainex",
              "e8fe2a816797cf83054135160df5a82ec3596a69"),
    "mycobot_ros2": ("https://github.com/elephantrobotics/mycobot_ros2",
                     "d42ff61a78122c79246623391540d75738b03b23"),
}
DRIVE = {
    "yahboom": {
        "drive_id": "1SRg1aD_u8kyxxjm4vp0cFYxu8Ddj2sXU",   # ROSMASTER-X3Plus_ROS1_code.zip
        "outer_bytes": 6343438429,
        "member": "yahboomcar_ws.zip",
        "member_sha256": "13d752e04cba3e34116912c3903bbe194e5ca5185c304036fb0ddedb59937162",
    },
}
#: (robot id, local prefix under robots_specs/, source key, upstream prefix)
SOURCES = [
    ("so101", "so101/assets/", "SO-ARM100", "Simulation/SO101/assets/"),
    ("myagv", "myagv/urdf/", "myagv_ros", "myagv_urdf/urdf/"),
    ("ainex", "ainex/meshes/", "ainex", "src/ainex_simulations/ainex_description/meshes/"),
    ("mycobot280", "mycobot280/urdf/", "mycobot_ros2", "mycobot_description/urdf/"),
    ("rosmaster_x3_plus", "rosmaster_x3_plus/meshes/", "yahboom",
     "yahboomcar_ws/src/yahboomcar_description/meshes/"),
]
COMPOSITES = {"myagv_mycobot280": ("myagv", "mycobot280")}
DERIVED_DIR = "derived_meshes"


def log(msg: str) -> None:
    print(f">> {msg}", flush=True)


class Refused(SystemExit):
    def __init__(self, msg: str):
        print(f"error: {msg}", file=sys.stderr, flush=True)
        super().__init__(1)


def sha256(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def read_manifest() -> dict[str, str]:
    """{path under robots_specs/: sha256} from meshes.sha256 (paths there are repo-relative)."""
    if not MANIFEST.is_file():
        raise Refused(f"missing {MANIFEST}")
    out = {}
    for n, line in enumerate(MANIFEST.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        digest, path = line.split(None, 1)
        path = path.strip()
        if not path.startswith("robots_specs/") or len(digest) != 64:
            raise Refused(f"meshes.sha256 line {n} is malformed: {line!r}")
        out[path[len("robots_specs/"):]] = digest
    return out


def robot_of(rel: str) -> str:
    return rel.split("/", 1)[0]


def origin_of(rel: str):
    """('fetch', source key, upstream path) or ('derive', source under robots_specs/, kind)."""
    rid, rest = rel.split("/", 1)
    if rest.startswith(DERIVED_DIR + "/"):
        sub = rest[len(DERIVED_DIR) + 1:]
        m = re.match(r"(.+\.dae)\.(?:m\d+\.obj|mtl)$", sub)
        if m:
            return ("derive", f"{rid}/{m.group(1)}", "visual")
        m = re.match(r"(.+\.stl)\.part\d+\.stl$", sub, re.I)
        if m:
            return ("derive", f"{rid}/{m.group(1)}", "split")
        if sub.endswith(".stl"):
            return ("derive", f"{rid}/{sub[:-len('.stl')]}.dae", "stl")
        raise Refused(f"{rel}: not a derived file name fetch_meshes.py knows")
    for _rid, local, key, up in SOURCES:
        if rel.startswith(local):
            return ("fetch", key, up + rel[len(local):])
    raise Refused(f"{rel} is in meshes.sha256 but no source in fetch_meshes.py provides it")


# ------------------------------------------------------------------ sources


def git(*args: str, cwd: Path) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise Refused(f"git {' '.join(args)} failed in {cwd}: {r.stderr.strip()}")
    return r.stdout


def fetch_git(key: str, wanted: dict[str, str], stage: Path) -> None:
    """Copy upstream paths {upstream: staged rel} from the pinned commit into stage."""
    url, rev = GIT[key]
    src = CACHE / "src" / key
    if not (src / ".git").is_dir():
        src.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=src)
        git("remote", "add", "origin", url, cwd=src)
    git("config", "core.sparseCheckout", "true", cwd=src)
    patterns = sorted({"/" + up for up in wanted})
    old = set()
    info = src / ".git" / "info" / "sparse-checkout"
    if info.is_file():
        old = {l.strip() for l in info.read_text().splitlines() if l.strip()}
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text("\n".join(sorted(old | set(patterns))) + "\n")
    have = subprocess.run(["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=src,
                          capture_output=True).returncode == 0
    if not have:
        log(f"fetching {key} @ {rev[:12]}")
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", rev, cwd=src)
    git("checkout", "-q", "--detach", rev, cwd=src)
    git("read-tree", "-mu", "HEAD", cwd=src)   # apply the widened sparse patterns
    head = git("rev-parse", "HEAD", cwd=src).strip()
    if head != rev:
        raise Refused(f"{key}: checked out {head}, expected {rev}")
    for up, rel in wanted.items():
        f = src / up
        if not f.is_file():
            raise Refused(f"{key} @ {rev[:12]} has no {up}")
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)


def fetch_drive(key: str, wanted: dict[str, str], stage: Path) -> None:
    d = DRIVE[key]
    tool = Path(__file__).resolve().parent / "fetch_drive_zip_members.py"
    tmp = stage / f".{key}"
    cmd = [sys.executable, str(tool), "--drive-id", d["drive_id"],
           "--outer-bytes", str(d["outer_bytes"]), "--member", d["member"],
           "--member-sha256", d["member_sha256"],
           "--cache", str(CACHE / "drive" / d["drive_id"] / d["member"]),
           "--dest", str(tmp), *wanted]
    if subprocess.run(cmd).returncode:
        raise Refused(
            f"cannot fetch from the {key} archive. Alternatively download "
            f"https://drive.google.com/file/d/{d['drive_id']} by hand, take {d['member']} out of "
            f"it and put it at {CACHE / 'drive' / d['drive_id'] / d['member']} (it is verified "
            f"against sha256 {d['member_sha256']}), then run this again")
    for up, rel in wanted.items():
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp / up, dst)


# ------------------------------------------------------------------ COLLADA -> STL


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _eye():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def _node_matrix(node, ns):
    m = _eye()
    for child in node:
        tag = child.tag.replace(ns, "")
        vals = [float(v) for v in (child.text or "").split()]
        if tag == "matrix":
            t = [vals[r * 4:(r + 1) * 4] for r in range(4)]
        elif tag == "translate":
            t = _eye()
            for r in range(3):
                t[r][3] = vals[r]
        elif tag == "scale":
            t = _eye()
            for r in range(3):
                t[r][r] = vals[r]
        elif tag == "rotate":
            x, y, z, ang = vals[0], vals[1], vals[2], math.radians(vals[3])
            n = math.sqrt(x * x + y * y + z * z) or 1.0
            x, y, z = x / n, y / n, z / n
            c, s, k = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
            t = [[c + x * x * k, x * y * k - z * s, x * z * k + y * s, 0.0],
                 [y * x * k + z * s, c + y * y * k, y * z * k - x * s, 0.0],
                 [z * x * k - y * s, z * y * k + x * s, c + z * z * k, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]
        else:
            continue
        m = _mat_mul(m, t)
    return m


def dae_triangles(path: Path) -> list[tuple[tuple[float, ...], ...]]:
    """Every triangle a COLLADA file's visual scene instantiates, in metres (the file's
    `<unit meter>` applied; all pinned files are Z_UP). Walks nodes, `instance_node`s
    into `library_nodes`, and composes matrix/translate/rotate/scale. Reads
    `<triangles>`, `<polylist>` and `<polygons>` (polygons fanned into triangles)."""
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    up = root.find(f"{ns}asset/{ns}up_axis")
    if up is not None and (up.text or "").strip() not in ("Z_UP", ""):
        raise Refused(f"{path.name}: up_axis {up.text} is not handled")
    unit = root.find(f"{ns}asset/{ns}unit")
    scale = float(unit.get("meter", "1")) if unit is not None else 1.0
    by_id = {el.get("id"): el for el in root.iter() if el.get("id")}

    def positions(source_id):
        src = by_id[source_id.lstrip("#")]
        if src.tag.replace(ns, "") == "vertices":
            pos = next(i for i in src.findall(f"{ns}input") if i.get("semantic") == "POSITION")
            return positions(pos.get("source"))
        arr = [float(v) for v in (src.find(f"{ns}float_array").text or "").split()]
        acc = src.find(f"{ns}technique_common/{ns}accessor")
        stride = int(acc.get("stride", "3")) if acc is not None else 3
        return [tuple(arr[i:i + 3]) for i in range(0, len(arr) - stride + 1, stride)]

    cache = {}

    def geometry(gid):
        if gid in cache:
            return cache[gid]
        mesh = by_id[gid].find(f"{ns}mesh")
        parts = []
        for prim in (mesh if mesh is not None else []):
            kind = prim.tag.replace(ns, "")
            if kind not in ("triangles", "polylist", "polygons"):
                continue
            inputs = prim.findall(f"{ns}input")
            stride = max(int(i.get("offset", "0")) for i in inputs) + 1
            vert = next(i for i in inputs if i.get("semantic") == "VERTEX")
            off = int(vert.get("offset", "0"))
            pos = positions(vert.get("source"))
            if kind == "triangles":
                p = prim.find(f"{ns}p")
                if p is None or not (p.text or "").strip():
                    continue
                flat = [int(v) for v in p.text.split()]
                ids = flat[off::stride]
                tris = [tuple(ids[i:i + 3]) for i in range(0, len(ids) - 2, 3)]
            else:
                if kind == "polylist":
                    counts = [int(c) for c in prim.find(f"{ns}vcount").text.split()]
                    flat = [int(v) for v in prim.find(f"{ns}p").text.split()]
                    polys, at = [], 0
                    for c in counts:
                        polys.append(flat[at * stride:(at + c) * stride])
                        at += c
                else:
                    polys = [[int(v) for v in p.text.split()]
                             for p in prim.findall(f"{ns}p") if (p.text or "").strip()]
                tris = []
                for poly in polys:
                    ids = poly[off::stride]
                    tris += [(ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1)]
            parts.append((pos, tris))
        cache[gid] = parts
        return parts

    out = []

    def walk(node, parent, depth=0):
        if depth > 32:
            raise Refused(f"{path.name}: node graph nests deeper than 32 levels")
        m = _mat_mul(parent, _node_matrix(node, ns))
        for child in node:
            tag = child.tag.replace(ns, "")
            if tag == "node":
                walk(child, m, depth + 1)
            elif tag == "instance_node":
                walk(by_id[child.get("url").lstrip("#")], m, depth + 1)
            elif tag == "instance_geometry":
                for pos, tris in geometry(child.get("url").lstrip("#")):
                    xf = [tuple(scale * (m[r][0] * x + m[r][1] * y + m[r][2] * z + m[r][3])
                                for r in range(3)) for (x, y, z) in pos]
                    out.extend((xf[a], xf[b], xf[c]) for a, b, c in tris)

    ref = root.find(f"{ns}scene/{ns}instance_visual_scene")
    scene = by_id[ref.get("url").lstrip("#")]
    for node in scene.findall(f"{ns}node"):
        walk(node, _eye())
    if not out:
        raise Refused(f"{path}: no triangles in the visual scene")
    return out


def write_stl(path: Path, tris, source_name: str) -> None:
    """Binary STL, byte-for-byte deterministic for a given input."""
    header = f"derived from {source_name} by robots_specs/tools/fetch_meshes.py".encode()[:80]
    with open(path, "wb") as f:
        f.write(header.ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - a[i] for i in range(3)]
            n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
            ln = math.sqrt(sum(x * x for x in n))
            n = [x / ln for x in n] if ln > 0 else [0.0, 0.0, 0.0]
            f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


# ------------------------------------------------------------------ visual parts


def _num(v: float) -> str:
    s = f"{v:.9g}"
    return "0" if s in ("-0", "0") else s


def dae_visual(path: Path):
    """The visual scene of a COLLADA file grouped by material, in order of first use:
    [(material, [(corner, corner, corner), ...])] with corner = (position, normal|None,
    uv|None) in metres; material = {"id", "name", "rgba", "texture"} resolved through the
    instance_geometry's bind_material, the material's effect (diffuse colour or texture
    image, transparency). A symbol bound to no defined material gets white, no texture."""
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    up = root.find(f"{ns}asset/{ns}up_axis")
    if up is not None and (up.text or "").strip() not in ("Z_UP", ""):
        raise Refused(f"{path.name}: up_axis {up.text} is not handled")
    unit = root.find(f"{ns}asset/{ns}unit")
    scale = float(unit.get("meter", "1")) if unit is not None else 1.0
    by_id = {el.get("id"): el for el in root.iter() if el.get("id")}

    def arr(source_id, want):
        src = by_id[source_id.lstrip("#")]
        if src.tag.replace(ns, "") == "vertices":
            pos = next(i for i in src.findall(f"{ns}input") if i.get("semantic") == "POSITION")
            return arr(pos.get("source"), want)
        vals = [float(v) for v in (src.find(f"{ns}float_array").text or "").split()]
        acc = src.find(f"{ns}technique_common/{ns}accessor")
        stride = int(acc.get("stride", str(want))) if acc is not None else want
        return [tuple(vals[i:i + want]) for i in range(0, len(vals) - stride + 1, stride)]

    def material(symbol, bindings):
        target = bindings.get(symbol)
        mat = by_id.get(target.lstrip("#")) if target else None
        info = {"id": (target or symbol or "").lstrip("#"), "name": "", "rgba": (1.0, 1.0, 1.0, 1.0),
                "texture": None}
        if mat is None or mat.tag.replace(ns, "") != "material":
            return info
        info["name"] = mat.get("name") or ""
        eff = by_id[mat.find(f"{ns}instance_effect").get("url").lstrip("#")]
        diff = eff.find(f".//{ns}diffuse")
        rgba = [1.0, 1.0, 1.0, 1.0]
        if diff is not None and diff.find(f"{ns}color") is not None:
            rgba = [float(v) for v in diff.find(f"{ns}color").text.split()][:4]
        elif diff is not None and diff.find(f"{ns}texture") is not None:
            sampler = diff.find(f"{ns}texture").get("texture")
            params = {p.get("sid"): p for p in eff.iter(f"{ns}newparam")}
            img_id = sampler
            if sampler in params:
                src = params[sampler].find(f".//{ns}source")
                surf = params.get(src.text) if src is not None else None
                init = surf.find(f".//{ns}init_from") if surf is not None else None
                img_id = init.text if init is not None else sampler
            img = by_id.get(img_id)
            if img is not None:
                info["texture"] = img.find(f".//{ns}init_from").text.strip()
        tr = eff.find(f".//{ns}transparency/{ns}float")
        if tr is not None and len(rgba) == 4:
            rgba[3] = rgba[3] * float(tr.text)   # A_ONE opaque convention
        info["rgba"] = tuple(rgba)
        return info

    groups: dict[str, tuple[dict, list]] = {}

    def add_prims(gid, m, bindings):
        mesh = by_id[gid].find(f"{ns}mesh")
        rot = [[m[r][c] for c in range(3)] for r in range(3)]
        for prim in (mesh if mesh is not None else []):
            kind = prim.tag.replace(ns, "")
            if kind not in ("triangles", "polylist", "polygons"):
                continue
            inputs = prim.findall(f"{ns}input")
            stride = max(int(i.get("offset", "0")) for i in inputs) + 1
            sem = {}
            for i in inputs:
                sem.setdefault(i.get("semantic"), i)
            vi = sem["VERTEX"]
            pos = arr(vi.get("source"), 3)
            nrm = arr(sem["NORMAL"].get("source"), 3) if "NORMAL" in sem else None
            uvs = arr(sem["TEXCOORD"].get("source"), 2) if "TEXCOORD" in sem else None
            offs = (int(vi.get("offset", "0")),
                    int(sem["NORMAL"].get("offset", "0")) if nrm else None,
                    int(sem["TEXCOORD"].get("offset", "0")) if uvs else None)
            if kind == "triangles":
                p = prim.find(f"{ns}p")
                if p is None or not (p.text or "").strip():
                    continue
                flat = [int(v) for v in p.text.split()]
                polys = [flat[i:i + 3 * stride] for i in range(0, len(flat) - 3 * stride + 1, 3 * stride)]
            elif kind == "polylist":
                counts = [int(c) for c in prim.find(f"{ns}vcount").text.split()]
                flat = [int(v) for v in prim.find(f"{ns}p").text.split()]
                polys, at = [], 0
                for c in counts:
                    polys.append(flat[at * stride:(at + c) * stride])
                    at += c
            else:
                polys = [[int(v) for v in p.text.split()]
                         for p in prim.findall(f"{ns}p") if (p.text or "").strip()]
            info = material(prim.get("material"), bindings)
            key = info["id"]
            if key not in groups:
                groups[key] = (info, [])
            tris = groups[key][1]

            def corner(idx):
                x, y, z = pos[idx[offs[0]]]
                P = tuple(scale * (m[r][0] * x + m[r][1] * y + m[r][2] * z + m[r][3]) for r in range(3))
                N = None
                if nrm:
                    a, b, c = nrm[idx[offs[1]]]
                    n = [rot[r][0] * a + rot[r][1] * b + rot[r][2] * c for r in range(3)]
                    ln = math.sqrt(sum(v * v for v in n)) or 1.0
                    N = tuple(v / ln for v in n)
                T = uvs[idx[offs[2]]] if uvs else None
                return (P, N, T)

            for poly in polys:
                cs = [corner(poly[k * stride:(k + 1) * stride]) for k in range(len(poly) // stride)]
                tris.extend((cs[0], cs[k], cs[k + 1]) for k in range(1, len(cs) - 1))

    def walk(node, parent, depth=0):
        if depth > 32:
            raise Refused(f"{path.name}: node graph nests deeper than 32 levels")
        m = _mat_mul(parent, _node_matrix(node, ns))
        for child in node:
            tag = child.tag.replace(ns, "")
            if tag == "node":
                walk(child, m, depth + 1)
            elif tag == "instance_node":
                walk(by_id[child.get("url").lstrip("#")], m, depth + 1)
            elif tag == "instance_geometry":
                bindings = {im.get("symbol"): im.get("target")
                            for im in child.iter(f"{ns}instance_material")}
                add_prims(child.get("url").lstrip("#"), m, bindings)

    ref = root.find(f"{ns}scene/{ns}instance_visual_scene")
    for node in by_id[ref.get("url").lstrip("#")].findall(f"{ns}node"):
        walk(node, _eye())
    out = [g for g in groups.values() if g[1]]
    if not out:
        raise Refused(f"{path}: no triangles in the visual scene")
    return out


def write_visual(dae: Path, dae_rel: str, out_dir: Path) -> list[Path]:
    """`<name>.dae.mtl` and `<name>.dae.m<k>.obj` for every material of `dae` into out_dir.
    Texture paths in the MTL are relative to the MTL, pointing at the fetched image."""
    name = dae.name
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = dae_visual(dae)
    depth = len(Path(dae_rel).parts) - 1          # dae_rel is <id>/<rel>; mtl is under derived_meshes/
    up = "../" * depth
    mtl = [f"# derived from {name} by robots_specs/tools/fetch_meshes.py: one material per part"]
    written = []
    for k, (info, tris) in enumerate(parts):
        r, g, b, a = info["rgba"]
        mtl += ["", f"# COLLADA material {info['id']!r} {info['name']!r}", f"newmtl m{k}",
                f"Kd {_num(r)} {_num(g)} {_num(b)}", f"d {_num(a)}"]
        textured = info["texture"] is not None
        if textured:
            rel_dir = str(Path(dae_rel).parent.relative_to(Path(dae_rel).parts[0]))
            mtl.append(f"map_Kd {up}{rel_dir}/{info['texture']}")
        verts, vidx, faces = [], {}, []
        has_n = all(c[1] is not None for t in tris for c in t)
        for t in tris:
            face = []
            for P, N, T in t:
                key = (P, N if has_n else None, T if textured else None)
                if key not in vidx:
                    vidx[key] = len(verts) + 1
                    verts.append(key)
                face.append(vidx[key])
            faces.append(face)
        lines = [f"# derived from {name} by robots_specs/tools/fetch_meshes.py (material m{k})",
                 f"mtllib {name}.mtl", f"usemtl m{k}"]
        lines += [f"v {_num(P[0])} {_num(P[1])} {_num(P[2])}" for P, _, _ in verts]
        if textured:
            lines += [f"vt {_num(T[0] if T else 0.0)} {_num(T[1] if T else 0.0)}" for _, _, T in verts]
        if has_n:
            lines += [f"vn {_num(N[0])} {_num(N[1])} {_num(N[2])}" for _, N, _ in verts]
        if textured and has_n:
            fmt = lambda i: f"{i}/{i}/{i}"
        elif textured:
            fmt = lambda i: f"{i}/{i}"
        elif has_n:
            fmt = lambda i: f"{i}//{i}"
        else:
            fmt = str
        lines += ["f " + " ".join(fmt(i) for i in f) for f in faces]
        p = out_dir / f"{name}.m{k}.obj"
        p.write_text("\n".join(lines) + "\n")
        written.append(p)
    p = out_dir / f"{name}.mtl"
    p.write_text("\n".join(mtl) + "\n")
    written.append(p)
    return written


STL_FACE_LIMIT = 200000


def write_split(stl: Path, out_dir: Path) -> list[Path]:
    """`<name>.part<k>.stl`: the binary STL's triangle records copied verbatim, in order,
    at most STL_FACE_LIMIT per part (lossless; MuJoCo refuses larger STLs)."""
    data = stl.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        raise Refused(f"{stl}: ASCII STL splitting is not implemented")
    (n,) = struct.unpack("<I", data[80:84])
    if len(data) < 84 + 50 * n:
        raise Refused(f"{stl}: truncated STL")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, start in enumerate(range(0, n, STL_FACE_LIMIT)):
        cnt = min(STL_FACE_LIMIT, n - start)
        header = f"part {k} of {stl.name} (faces {start}..{start + cnt - 1})".encode()[:80]
        p = out_dir / f"{stl.name}.part{k}.stl"
        p.write_bytes(header.ljust(80, b"\0") + struct.pack("<I", cnt)
                      + data[84 + 50 * start:84 + 50 * (start + cnt)])
        written.append(p)
    return written


def derive(src_rel: str, kind: str, stage: Path) -> None:
    """Write every file derived from src_rel (under robots_specs/) of this kind into stage."""
    rid, rest = src_rel.split("/", 1)
    out_dir = stage / rid / DERIVED_DIR / Path(rest).parent
    src = SPECS / src_rel
    if kind == "stl":
        out_dir.mkdir(parents=True, exist_ok=True)
        write_stl(out_dir / (Path(rest).stem + ".stl"), dae_triangles(src), src.name)
    elif kind == "visual":
        write_visual(src, src_rel, out_dir)
    elif kind == "split":
        write_split(src, out_dir)


# ------------------------------------------------------------------ main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robots", nargs="*", help="robot ids (default: every robot)")
    ap.add_argument("--check", action="store_true", help="verify only; fetch nothing")
    a = ap.parse_args(argv)

    manifest = read_manifest()
    known = sorted({robot_of(p) for p in manifest})
    wanted_ids = set()
    for r in a.robots:
        if r in COMPOSITES:
            wanted_ids.update(COMPOSITES[r])
        elif r in known:
            wanted_ids.add(r)
        else:
            raise Refused(f"unknown robot {r!r}; known: {', '.join(known + sorted(COMPOSITES))}")
    entries = {p: d for p, d in manifest.items() if not wanted_ids or robot_of(p) in wanted_ids}
    origins = {p: origin_of(p) for p in entries}

    bad = [p for p, d in entries.items() if sha256(SPECS / p) != d]
    if a.check:
        for p in bad:
            print(f"missing or mismatched: robots_specs/{p}", file=sys.stderr)
        print(f">> {len(entries) - len(bad)}/{len(entries)} file(s) match robots_specs/meshes.sha256")
        return 1 if bad else 0

    for p in bad:                       # never leave an unverified file in place
        if (SPECS / p).exists():
            log(f"removing mismatched robots_specs/{p}")
            (SPECS / p).unlink()
    if not bad:
        log(f"{len(entries)} file(s) up to date")
        return 0

    failed: list[str] = []
    stage = Path(tempfile.mkdtemp(prefix=".fetch-", dir=SPECS))
    try:
        # A derived STL needs its verified .dae, fetched here if it is not in place.
        need = set(bad)
        for p in bad:
            o = origins[p]
            if o[0] == "derive":
                if o[1] not in manifest:
                    raise Refused(f"{p} derives from {o[1]}, which meshes.sha256 does not list")
                if sha256(SPECS / o[1]) != manifest[o[1]]:
                    need.add(o[1])
                    origins.setdefault(o[1], origin_of(o[1]))
        by_source: dict[str, dict[str, str]] = {}
        for p in sorted(need):
            o = origins[p]
            if o[0] == "fetch":
                by_source.setdefault(o[1], {})[o[2]] = p
        # One source at a time: fetch into the stage, verify, and only then move in. A
        # source that fails or mismatches keeps nothing; the others still complete.
        for key, files in by_source.items():
            log(f"{key}: fetching {len(files)} file(s)")
            try:
                (fetch_drive if key in DRIVE else fetch_git)(key, files, stage)
            except SystemExit:
                failed.append(f"{key}: fetch failed (see above)")
                continue
            wrong = [p for p in files.values() if sha256(stage / p) != manifest[p]]
            for p in wrong:
                print(f"error: robots_specs/{p}: sha256 {sha256(stage / p)} does not match "
                      f"meshes.sha256 ({manifest[p]})", file=sys.stderr)
            if wrong:
                failed.append(f"{key}: {len(wrong)} file(s) do not match meshes.sha256; none kept")
                continue
            for p in files.values():
                (SPECS / p).parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / p, SPECS / p)
        done_sources = set()
        for p in sorted(need):
            if origins[p][0] != "derive":
                continue
            _, src, kind = origins[p]
            if sha256(SPECS / src) != manifest[src]:
                failed.append(f"robots_specs/{p}: not derived, its source {src} is not verified")
                continue
            if (src, kind) not in done_sources:
                derive(src, kind, stage)
                done_sources.add((src, kind))
            if sha256(stage / p) != manifest[p]:
                failed.append(f"robots_specs/{p}: derived sha256 {sha256(stage / p)} does "
                              f"not match meshes.sha256 ({manifest[p]}); not kept")
                continue
            (SPECS / p).parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / p, SPECS / p)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    still = [p for p, d in entries.items() if sha256(SPECS / p) != d]
    if failed or still:
        per_robot: dict[str, int] = {}
        for p in still:
            per_robot[robot_of(p)] = per_robot.get(robot_of(p), 0) + 1
        raise Refused("\n  ".join(["robot meshes are incomplete; no unverified file was kept:"]
                                  + failed + [f"robots_specs/{r}/: {n} file(s) missing"
                                              for r, n in sorted(per_robot.items())]))
    log(f"{len(entries)} file(s) verified against robots_specs/meshes.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
