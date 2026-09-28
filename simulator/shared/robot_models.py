"""Build the engine-loadable models of the robots with no usable official MJCF.

    python robot_models.py [--check] [<id> ...]     # every robot built here, or the named

`shared/robots/<id>/model.xml` is written from the robot's URDF (and, for the myAGV +
myCobot 280, the myAGV's own generated model) in `robots_specs/<id>/`: every link a body at
its URDF joint origin, every movable joint with the URDF's axis and limits, every visual
mesh at its URDF origin, every URDF inertial. What the simulator adds is only what
simulation needs and the URDF does not say (spec §2.2, robot models):

* **a planar base** -- the same world-aligned `base_x` / `base_y` / `base_theta` trio and
  position actuators the myAGV rides (`shared/robots/myagv/model.xml`), standing in for
  the Mecanum drive both robots have;
* **position servos** on the arm and gripper joints, and each URDF `<mimic>` as a joint
  equality with its multiplier;
* **collision geometry**: boxes fitted to the chassis (lifted clear of the floor, as the
  myAGV's is, so the planar base does not scrape it) and the convex hulls of the arm and
  sensor meshes. Robot collision geoms have `conaffinity="0"`: they meet the scene, and
  never one another;
* **cameras** where the robot's ROS interface puts its camera frames.

Meshes MuJoCo reads directly (the X3 PLUS's STLs) are referenced in `robots_specs/`. The
myCobot 280 Pi and adaptive gripper meshes are COLLADA, which MuJoCo does not read: they
are converted here, by a standard-library COLLADA reader, to OBJ files under
`shared/robots/myagv_mycobot280/assets/` -- generated, never committed, and rebuilt by
`ensure` whenever one is missing (both engines' `run.sh setup` run this script, and the
spawn tool calls `ensure` before it loads a model). `model.xml` itself is small, committed
and deterministic: `--check` regenerates it in memory and fails on any difference.

numpy and the standard library only: both engines' venvs run it, and the RoboCasa venv
has neither trimesh nor pycollada.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import robots_spec  # noqa: E402

#: The robots whose `model.xml` this module writes.
BUILT = ("myagv_mycobot280", "rosmaster_x3_plus")

#: Keep a chassis collision box this far over the floor (the myAGV's GROUND_CLEARANCE):
#: the base has no vertical freedom, and a box on the floor would drag against it.
GROUND_CLEARANCE = 0.005

# ------------------------------------------------------------------------ geometry


def rpy_to_quat(r: float, p: float, y: float) -> tuple[float, float, float, float]:
    """URDF `rpy` (fixed-axis roll, pitch, yaw) as `(w, x, y, z)`."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _fmt(values) -> str:
    """Numbers as the model writes them: 9 significant digits, `-0` folded to `0`."""
    out = []
    for v in values:
        s = f"{float(v):.9g}"
        out.append("0" if s in ("-0", "0") else s)
    return " ".join(out)


def _floats(text: str | None, default: str = "0 0 0") -> list[float]:
    return [float(v) for v in (text or default).split()]


def read_stl(path: Path) -> np.ndarray:
    """An STL's triangle corners, (3N, 3), ASCII or binary."""
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        return np.array([[float(x) for x in line.split()[1:4]]
                         for line in data.decode("ascii", "replace").splitlines()
                         if line.strip().startswith("vertex")])
    (n,) = struct.unpack("<I", data[80:84])
    rows = np.frombuffer(data[84:84 + 50 * n],
                         dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rows["v"].reshape(-1, 3).astype(np.float64)


# ------------------------------------------------------------------------ COLLADA


def _matrix_of(node, ns: str) -> np.ndarray:
    """A COLLADA `<node>`'s own transform, its elements composed in document order."""
    m = np.eye(4)
    for child in node:
        tag = child.tag.replace(ns, "")
        vals = [float(v) for v in (child.text or "").split()]
        if tag == "matrix":
            m = m @ np.array(vals).reshape(4, 4)
        elif tag == "translate":
            t = np.eye(4)
            t[:3, 3] = vals
            m = m @ t
        elif tag == "scale":
            m = m @ np.diag(vals + [1.0])
        elif tag == "rotate":
            axis, angle = np.array(vals[:3]), math.radians(vals[3])
            axis = axis / (np.linalg.norm(axis) or 1.0)
            x, y, z = axis
            c, s, k = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
            r = np.eye(4)
            r[:3, :3] = [[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                         [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                         [z * x * k - y * s, z * y * k + x * s, c + z * z * k]]
            m = m @ r
    return m


def read_dae(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Every triangle a COLLADA file's visual scene instantiates: `(vertices, faces)` in
    metres (the file's `<unit meter>` applied), vertices shared.

    Walks the scene graph itself, `instance_node`s into `library_nodes` included (the
    myCobot's joint5.dae instantiates its geometry only through those), composing each
    node's `matrix`/`translate`/`rotate`/`scale`. `<triangles>`, `<polylist>` and
    `<polygons>` are read; a polygon is fanned into triangles.
    """
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    unit = root.find(f"{ns}asset/{ns}unit")
    scale = float(unit.get("meter", "1")) if unit is not None else 1.0
    by_id = {el.get("id"): el for el in root.iter() if el.get("id")}

    def source_array(source_id: str) -> np.ndarray:
        src = by_id[source_id.lstrip("#")]
        if src.tag.replace(ns, "") == "vertices":
            pos = next(i for i in src.findall(f"{ns}input") if i.get("semantic") == "POSITION")
            return source_array(pos.get("source"))
        arr = src.find(f"{ns}float_array")
        acc = src.find(f"{ns}technique_common/{ns}accessor")
        stride = int(acc.get("stride", "3")) if acc is not None else 3
        values = np.array((arr.text or "").split(), dtype=np.float64)
        return values.reshape(-1, stride)[:, :3]

    cache: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    def geometry(geom_id: str):
        if geom_id in cache:
            return cache[geom_id]
        mesh = by_id[geom_id].find(f"{ns}mesh")
        parts = []
        for prim in mesh:
            kind = prim.tag.replace(ns, "")
            if kind not in ("triangles", "polylist", "polygons"):
                continue
            inputs = prim.findall(f"{ns}input")
            stride = max(int(i.get("offset", "0")) for i in inputs) + 1
            vert = next(i for i in inputs if i.get("semantic") == "VERTEX")
            offset = int(vert.get("offset", "0"))
            positions = source_array(vert.get("source"))
            if kind == "triangles":
                p = prim.find(f"{ns}p")
                if p is None or not (p.text or "").strip():
                    continue
                idx = np.array(p.text.split(), dtype=np.int64).reshape(-1, stride)[:, offset]
                faces = idx.reshape(-1, 3)
            else:
                if kind == "polylist":
                    counts = [int(c) for c in prim.find(f"{ns}vcount").text.split()]
                    flat = np.array(prim.find(f"{ns}p").text.split(), dtype=np.int64)
                    polys, at = [], 0
                    for c in counts:
                        polys.append(flat[at * stride:(at + c) * stride])
                        at += c
                else:
                    polys = [np.array(p.text.split(), dtype=np.int64)
                             for p in prim.findall(f"{ns}p") if (p.text or "").strip()]
                tris = []
                for poly in polys:
                    ids = poly.reshape(-1, stride)[:, offset]
                    tris += [(ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1)]
                if not tris:
                    continue
                faces = np.array(tris, dtype=np.int64)
            parts.append((positions, faces))
        cache[geom_id] = parts
        return parts

    verts, faces, count = [], [], 0

    def walk(node, parent: np.ndarray, depth: int = 0) -> None:
        if depth > 32:
            raise ValueError(f"{path.name}: node graph nests deeper than 32 levels")
        m = parent @ _matrix_of(node, ns)
        nonlocal count
        for child in node:
            tag = child.tag.replace(ns, "")
            if tag == "node":
                walk(child, m, depth + 1)
            elif tag == "instance_node":
                walk(by_id[child.get("url").lstrip("#")], m, depth + 1)
            elif tag == "instance_geometry":
                for positions, f in geometry(child.get("url").lstrip("#")):
                    homo = np.c_[positions, np.ones(len(positions))]
                    verts.append((homo @ m.T)[:, :3])
                    faces.append(f + count)
                    count += len(positions)

    scene_ref = root.find(f"{ns}scene/{ns}instance_visual_scene")
    scene = by_id[scene_ref.get("url").lstrip("#")]
    for node in scene.findall(f"{ns}node"):
        walk(node, np.eye(4))
    if not verts:
        raise ValueError(f"{path}: no triangles in the visual scene")
    v = np.vstack(verts) * scale
    f = np.vstack(faces)
    # Drop the vertices no face uses (normals-only duplicates), keeping order stable.
    used, inverse = np.unique(f.reshape(-1), return_inverse=True)
    return v[used], inverse.reshape(-1, 3)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("w", encoding="ascii") as out:
        out.write(f"# generated by simulator/shared/robot_models.py -- do not edit\n")
        np.savetxt(out, vertices, fmt="v %.7g %.7g %.7g")
        np.savetxt(out, faces + 1, fmt="f %d %d %d")
    os.replace(tmp, path)


# ------------------------------------------------------------------------ URDF


class Urdf:
    """A URDF's links and joints, as data."""

    def __init__(self, path: Path) -> None:
        self.path = path
        root = ET.parse(path).getroot()
        self.name = root.get("name", "")
        self.links = {l.get("name"): l for l in root.findall("link")}
        self.joints = [j for j in root.findall("joint")]
        self.children: dict[str, list] = {}
        for j in self.joints:
            self.children.setdefault(j.find("parent").get("link"), []).append(j)
        child_links = {j.find("child").get("link") for j in self.joints}
        roots = [n for n in self.links if n not in child_links]
        if len(roots) != 1:
            raise ValueError(f"{path}: expected one root link, found {roots}")
        self.root = roots[0]

    def mimics(self) -> list[tuple[str, str, float, float]]:
        """(joint, driver, multiplier, offset) for every `<mimic>`."""
        out = []
        for j in self.joints:
            m = j.find("mimic")
            if m is not None:
                out.append((j.get("name"), m.get("joint"), float(m.get("multiplier", "1")),
                            float(m.get("offset", "0"))))
        return out


def _origin(el) -> tuple[list[float], tuple[float, float, float, float]]:
    o = el.find("origin") if el is not None else None
    xyz = _floats(o.get("xyz") if o is not None else None)
    rpy = _floats(o.get("rpy") if o is not None else None)
    return xyz, rpy_to_quat(*rpy)


class MeshTable:
    """The `<asset>` meshes a model declares, one per (file, role)."""

    def __init__(self) -> None:
        self.meshes: dict[str, str] = {}  # name -> file, relative to meshdir

    def add(self, name: str, file: str) -> str:
        if self.meshes.get(name, file) != file:
            raise ValueError(f"mesh name {name} names two files")
        self.meshes[name] = file
        return name


def emit_link(urdf: Urdf, link: str, body: ET.Element, mesh_for, *, prefix: str = "",
              inertial_for=None, collide=True) -> None:
    """Fill `body` (the frame of `link`) with its inertial, visuals and collision hulls,
    then add a child body per joint below it, recursively."""
    el = urdf.links[link]
    inertial = inertial_for(link) if inertial_for else None
    if inertial is None:
        inertial = _urdf_inertial(el)
    if inertial is not None:
        ET.SubElement(body, "inertial", inertial)
    for kind in ("visual", "collision"):
        if kind == "collision" and not collide:
            continue
        for i, part in enumerate(el.findall(kind)):
            mesh = part.find("geometry/mesh")
            if mesh is None:
                continue
            xyz, quat = _origin(part)
            name = mesh_for(link, kind, mesh.get("filename"))
            attrs = {"class": f"{prefix}{kind}", "type": "mesh", "mesh": name}
            if any(abs(v) > 0 for v in xyz):
                attrs["pos"] = _fmt(xyz)
            if not np.allclose(quat, (1, 0, 0, 0), atol=0):
                attrs["quat"] = _fmt(quat)
            if kind == "visual":
                color = part.find("material/color")
                if color is not None:
                    attrs["rgba"] = _fmt(_floats(color.get("rgba"), "0.7 0.7 0.7 1"))
            ET.SubElement(body, "geom", attrs)
    for joint in urdf.children.get(link, []):
        child = joint.find("child").get("link")
        xyz, quat = _origin(joint)
        attrs = {"name": child, "pos": _fmt(xyz)}
        if not np.allclose(quat, (1, 0, 0, 0), atol=0):
            attrs["quat"] = _fmt(quat)
        sub = ET.SubElement(body, "body", attrs)
        kind = joint.get("type")
        if kind in ("revolute", "continuous", "prismatic"):
            j = {"name": joint.get("name"),
                 "type": "slide" if kind == "prismatic" else "hinge",
                 "axis": _fmt(_floats(joint.find("axis").get("xyz") if joint.find("axis") is
                                      not None else None, "1 0 0"))}
            limit = joint.find("limit")
            if kind != "continuous" and limit is not None:
                j["range"] = _fmt((float(limit.get("lower", "0")), float(limit.get("upper", "0"))))
            else:
                j["limited"] = "false"
            ET.SubElement(sub, "joint", j)
        elif kind != "fixed":
            raise ValueError(f"{urdf.path.name}: joint type {kind!r} is not supported")
        emit_link(urdf, child, sub, mesh_for, prefix=prefix, inertial_for=inertial_for,
                  collide=collide)


def _urdf_inertial(link_el) -> dict | None:
    inertial = link_el.find("inertial")
    if inertial is None:
        return None
    xyz, quat = _origin(inertial)
    i = inertial.find("inertia")
    full = [float(i.get(k, "0")) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    out = {"pos": _fmt(xyz), "mass": _fmt([float(inertial.find("mass").get("value"))]),
           "fullinertia": _fmt(full)}
    if not np.allclose(quat, (1, 0, 0, 0)):
        raise ValueError("a rotated URDF inertial frame is not supported")
    return out


# ------------------------------------------------------------------------ shared parts


PREAMBLE = """
  GENERATED by simulator/shared/robot_models.py from {sources} -- do not edit by hand.
  Regenerate with `python simulator/shared/robot_models.py {rid}`; `--check` verifies it.
"""


def _model(rid: str, sources: str) -> ET.Element:
    root = ET.Element("mujoco", {"model": rid})
    root.append(ET.Comment(PREAMBLE.format(sources=sources, rid=rid)))
    return root


def _defaults(root: ET.Element, cls: str, *, servo_kp: float, prefix: str = "",
              drive_kp: float | None = None, turn_kp: float | None = None) -> None:
    """The model's default classes. `prefix` keeps their names apart from the classes of a
    model this one includes (the composite carries the myAGV's own `visual`, `collision`,
    `drive` and `turn`); the planar-drive classes are written only when given gains."""
    default = ET.SubElement(ET.SubElement(root, "default"), "default", {"class": cls})
    # Servo-driven joints: a little armature and damping, as a geared servo has.
    ET.SubElement(default, "joint", {"armature": "0.005", "damping": "0.05"})
    # Visual geoms carry no mass: every body's inertia is its source's `inertial`.
    ET.SubElement(ET.SubElement(default, "default", {"class": f"{prefix}visual"}), "geom",
                  {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2",
                   "density": "0"})
    ET.SubElement(ET.SubElement(default, "default", {"class": f"{prefix}collision"}), "geom",
                  {"group": "3", "contype": "1", "conaffinity": "0", "condim": "3",
                   "density": "0", "friction": "0.6 0.005 0.0001"})
    if drive_kp is not None:
        ET.SubElement(ET.SubElement(default, "default", {"class": "drive"}), "position",
                      {"kp": _fmt([drive_kp]), "dampratio": "1"})
        ET.SubElement(ET.SubElement(default, "default", {"class": "turn"}), "position",
                      {"kp": _fmt([turn_kp]), "dampratio": "1"})
    ET.SubElement(ET.SubElement(default, "default", {"class": f"{prefix}servo"}), "position",
                  {"kp": _fmt([servo_kp]), "dampratio": "1"})


def _planar_joints(base: ET.Element) -> None:
    """The myAGV's virtual holonomic joints, world-aligned, named for the planar base."""
    ET.SubElement(base, "joint", {"name": "base_x", "type": "slide", "axis": "1 0 0",
                                  "limited": "false", "damping": "5"})
    ET.SubElement(base, "joint", {"name": "base_y", "type": "slide", "axis": "0 1 0",
                                  "limited": "false", "damping": "5"})
    ET.SubElement(base, "joint", {"name": "base_theta", "type": "hinge", "axis": "0 0 1",
                                  "limited": "false", "damping": "0.5"})


def _planar_actuators(actuator: ET.Element) -> None:
    ET.SubElement(actuator, "position", {"class": "drive", "name": "base_x_act",
                                         "joint": "base_x", "ctrlrange": "-25 25"})
    ET.SubElement(actuator, "position", {"class": "drive", "name": "base_y_act",
                                         "joint": "base_y", "ctrlrange": "-25 25"})
    ET.SubElement(actuator, "position", {"class": "turn", "name": "base_theta_act",
                                         "joint": "base_theta", "ctrlrange": "-100 100"})


def _servos(actuator: ET.Element, joints: list[tuple[str, tuple[float, float]]],
            prefix: str = "") -> None:
    for name, (lo, hi) in joints:
        ET.SubElement(actuator, "position", {"class": f"{prefix}servo", "name": name, "joint": name,
                                             "ctrlrange": _fmt((lo, hi))})


def _mimics(root: ET.Element, urdf: Urdf) -> None:
    mimics = urdf.mimics()
    if not mimics:
        return
    equality = ET.SubElement(root, "equality")
    for joint, driver, mult, offset in mimics:
        ET.SubElement(equality, "joint", {"name": f"mimic_{joint}", "joint1": joint,
                                          "joint2": driver,
                                          "polycoef": _fmt((offset, mult, 0, 0, 0))})


def _box(body: ET.Element, name: str, lo, hi) -> None:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ET.SubElement(body, "geom", {"name": name, "class": "collision", "type": "box",
                                 "pos": _fmt((lo + hi) / 2), "size": _fmt((hi - lo) / 2)})


def _set_class(body: ET.Element, cls: str) -> None:
    body.set("childclass", cls)


def _joint_ranges(body: ET.Element) -> list[tuple[str, tuple[float, float]]]:
    out = []
    for j in body.iter("joint"):
        if j.get("range"):
            lo, hi = _floats(j.get("range"))
            out.append((j.get("name"), (lo, hi)))
    return out


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


#: What a body the sources give no mass (a URDF link with no `<inertial>` and nothing
#: to derive one from, such as the myCobot's fixed base plate) carries: next to nothing,
#: and explicit, because older MuJoCo refuses a body whose inertia it would have to
#: derive from massless geoms.
PLACEHOLDER_INERTIAL = {"pos": "0 0 0", "mass": "1e-05", "diaginertia": "1e-09 1e-09 1e-09"}


def _placeholder_inertials(root: ET.Element) -> None:
    for body in root.iter("body"):
        if body.find("inertial") is not None:
            continue
        if any(float(g.get("density", "1")) > 0 for g in body.findall("geom")
               if g.get("density") is not None):
            continue  # its geoms give it a mass
        body.insert(0, ET.Element("inertial", PLACEHOLDER_INERTIAL))


def _serialise(root: ET.Element) -> str:
    _placeholder_inertials(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _relpath(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


# ------------------------------------------------------------------------ ROSMASTER X3 PLUS


#: The X3 PLUS's published figures that its URDF does not carry (ros_surfaces'
#: `rosmaster_x3_plus.PHYSICAL_FIGURES` quotes the pages): the assembled robot weighs
#: about 4.35 kg, and the URDF's links add up to 1.37 kg. A manufacturer's figure outranks
#: the URDF (spec §3), so the chassis link carries the difference, its inertia scaled
#: with its mass.
X3_MASS_KG = 4.35

#: Where the X3 PLUS's ROS interface puts its Astra camera frames: `camera_link`, the
#: URDF's, with the colour and depth sensors at its origin (astra_frames.launch's
#: identity `camera_link -> camera_rgb_frame / camera_depth_frame`). Vertical fields of
#: view at 640x480 are the published Astra Pro Plus figures.
X3_RGB_FOVY_DEG = 46.81
X3_DEPTH_FOVY_DEG = 45.8


#: MuJoCo reads at most this many triangles from one STL; the X3 PLUS's larger visual
#: meshes (its chassis has 567k) are written out as OBJ instead, which has no such cap.
STL_FACE_LIMIT = 200_000

#: The X3 PLUS's CAD exports are tessellated far past what any camera resolves (1.6 M
#: triangles in all, screw threads included), and every camera render draws every
#: triangle: its Astra's two 30 Hz views took 26-30 ms each. A visual mesh over
#: `VISUAL_FACE_BUDGET` triangles is welded on a `WELD_CELL_M` grid -- vertices within one
#: cell merge, triangles that collapse are dropped -- so no surface moves by more than
#: half a cell's diagonal (0.87 mm), under a pixel at any distance the Astra sees.
#: Collision geometry is not touched (it is convex hulls and boxes regardless).
VISUAL_FACE_BUDGET = 20_000
WELD_CELL_M = 0.001


def weld(vertices: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: every vertex moved to the mean of its grid cell's, degenerate and
    repeated triangles dropped. Returns `(vertices, faces)`."""
    keys = np.floor(vertices / cell).astype(np.int64)
    uniq, cluster = np.unique(keys, axis=0, return_inverse=True)
    cluster = cluster.reshape(-1)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, cluster, vertices)
    counts = np.bincount(cluster, minlength=len(uniq)).astype(float)
    welded = sums / counts[:, None]
    f = cluster[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return welded, f[np.sort(first)]


def _stl_faces(path: Path) -> int:
    with path.open("rb") as f:
        f.seek(80)
        return struct.unpack("<I", f.read(4))[0]


def x3_converted() -> list[Path]:
    """The X3 PLUS meshes MuJoCo cannot read as they are."""
    urdf = Urdf(robots_spec.urdf_path("rosmaster_x3_plus"))
    paths = sorted({robots_spec.resolve_mesh("rosmaster_x3_plus", m.get("filename"))
                    for link in urdf.links.values() for m in link.iter("mesh")})
    return [p for p in paths if p.exists() and (
        _stl_faces(p) > STL_FACE_LIMIT
        or (p.parent.name == "visual" and _stl_faces(p) > VISUAL_FACE_BUDGET))]


def _x3_obj(src: Path) -> Path:
    kind = src.parent.name  # visual / collision
    return robots_spec.model_dir("rosmaster_x3_plus") / "assets" / f"{kind}_{src.stem}.obj"


def convert_x3_meshes(force: bool = False) -> list[Path]:
    written = []
    for src in x3_converted():
        dst = _x3_obj(src)
        if dst.exists() and not force:
            continue
        corners = read_stl(src)
        v, inverse = np.unique(corners.astype(np.float32), axis=0, return_inverse=True)
        f = inverse.reshape(-1, 3)
        v = v.astype(np.float64)
        if src.parent.name == "visual" and len(f) > VISUAL_FACE_BUDGET:
            v, f = weld(v, f, WELD_CELL_M)
        write_obj(dst, v, f)
        written.append(dst)
        print(f"  {src.parent.name}/{src.name} -> {dst.name}: {len(v)} vertices, "
              f"{len(f)} triangles (of {len(corners) // 3})", file=sys.stderr)
    return written


def build_x3(write_assets: bool = True) -> str:
    rid = "rosmaster_x3_plus"
    spec = robots_spec.robot(rid)
    urdf = Urdf(spec.urdf)
    out_dir = robots_spec.model_dir(rid)
    mesh_root = spec.meshes[0]
    root = _model(rid, f"robots_specs/{rid}/{spec.urdf.name} and its meshes")
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true",
                                     "meshdir": "assets"})
    ET.SubElement(root, "option", {"integrator": "implicitfast", "timestep": "0.002"})
    _defaults(root, rid, drive_kp=2000, turn_kp=120, servo_kp=20)
    asset = ET.SubElement(root, "asset")
    table = MeshTable()
    big = set(x3_converted())

    def mesh_for(link: str, kind: str, filename: str) -> str:
        path = robots_spec.resolve_mesh(rid, filename)
        rel = _relpath(path, mesh_root)
        name = f"{kind}_{Path(rel).parts[0]}_{path.stem}"
        if path in big:
            return table.add(name, _x3_obj(path).name)
        return table.add(name, _relpath(path, out_dir / "assets"))

    masses = {n: float(l.find("inertial/mass").get("value"))
              for n, l in urdf.links.items() if l.find("inertial") is not None}
    others = sum(m for n, m in masses.items() if n != "base_link")

    def inertial_for(link: str):
        if link != "base_link":
            return None
        base = _urdf_inertial(urdf.links[link])
        mass = X3_MASS_KG - others
        k = mass / masses["base_link"]
        base["mass"] = _fmt([mass])
        base["fullinertia"] = _fmt([v * k for v in _floats(base["fullinertia"])])
        return base

    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(world, "body", {"name": "base", "childclass": rid})
    _planar_joints(base)
    ET.SubElement(base, "site", {"name": "base_site", "pos": "0 0 0", "size": "0.01",
                                 "group": "3"})
    # The chassis link carries only its visual mesh; its collision is the boxes below, cut
    # from its own collision mesh: the chassis between the wheels' tops and the floor
    # clearance, the upper deck, and the camera mast. The mesh itself reaches 3.6 mm under
    # the floor (its wheels), which a hull would drag along it.
    emit_link(urdf, urdf.root, base, mesh_for, inertial_for=inertial_for, collide=True)
    base_link = _find_body(base, "base_link")
    for geom in [g for g in base_link.findall("geom") if g.get("class") == "collision"]:
        base_link.remove(geom)
    z0 = float(_floats(_find_body(base, "base_link").get("pos"))[2])
    corners = read_stl(mesh_root / "X3plus" / "collision" / "base_link.STL")
    z = corners[:, 2] + z0

    def extent(sel):
        pts = corners[sel]
        return pts.min(axis=0), pts.max(axis=0)

    lo, hi = extent(z < 0.13)
    _box(base_link, "chassis", [lo[0], lo[1], GROUND_CLEARANCE - z0], [hi[0], hi[1], 0.13 - z0])
    lo, hi = extent((z >= 0.13) & (z < 0.2))
    _box(base_link, "deck", [lo[0], lo[1], 0.13 - z0], [hi[0], hi[1], 0.2 - z0])
    lo, hi = extent(z >= 0.2)
    _box(base_link, "mast", [lo[0], -0.035, 0.2 - z0], [-0.035, 0.035, hi[2]])
    # The touch screen on the mast's back, which the collision mesh leaves out: a box
    # around its panel, read off the visual mesh.
    visual = read_stl(mesh_root / "X3plus" / "visual" / "base_link.STL")
    panel = visual[(visual[:, 2] + z0 >= 0.28) & (visual[:, 0] < lo[0] - 0.005)]
    _box(base_link, "screen", panel.min(axis=0), panel.max(axis=0))

    camera = _find_body(base, "camera_link")
    # MuJoCo cameras look along -z with +y up: xyaxes maps -z_cam onto the link's +x.
    ET.SubElement(camera, "camera", {"name": "rgb_camera", "mode": "fixed", "pos": "0 0 0",
                                     "xyaxes": "0 -1 0 0 0 1",
                                     "fovy": _fmt([X3_RGB_FOVY_DEG])})
    ET.SubElement(camera, "camera", {"name": "depth_camera", "mode": "fixed", "pos": "0 0 0",
                                     "xyaxes": "0 -1 0 0 0 1",
                                     "fovy": _fmt([X3_DEPTH_FOVY_DEG])})
    for name, file in table.meshes.items():
        ET.SubElement(asset, "mesh", {"name": name, "file": file})
    actuator = ET.SubElement(root, "actuator")
    _planar_actuators(actuator)
    driven = {j for j, *_ in urdf.mimics()}
    _servos(actuator, [(n, r) for n, r in _joint_ranges(base) if n not in driven])
    _mimics(root, urdf)
    return _serialise(root)


# ------------------------------------------------------------------------ myAGV + myCobot 280


#: The myCobot 280's arm stands on the myAGV's top deck with its X axis along the AGV's
#: (composite manual 7.1, "arm X axis aligned with the AGV X axis"), centred on it. The
#: manual allows a front or a rear mount and gives no hole position; the centre is the
#: one the robot stays symmetric about.
ARM_MOUNT_XY = (0.0, 0.0)

#: The composite's own default classes are named apart from the myAGV's it includes.
ARM_CLASS = "arm_"

#: Material colours: the myCobot 280 Pi's white shell and the gripper's dark plastic.
ARM_RGBA = "0.92 0.92 0.92 1"
GRIPPER_RGBA = "0.2 0.2 0.22 1"

#: Density given to the adaptive gripper's hulls, kg/m^3, for its links' inertia: its
#: URDF declares no inertials and Elephant publishes no per-link figures. A printed-nylon
#: value; the gripper is ~0.1 kg of a ~5 kg robot either way.
GRIPPER_DENSITY = 1100.0


def composite_assets() -> Path:
    return robots_spec.model_dir("myagv_mycobot280") / "assets"


def _arm_meshes(urdf: Urdf, rid: str) -> dict[str, Path]:
    """URDF mesh filename -> the COLLADA file it names."""
    out = {}
    for link in urdf.links.values():
        for mesh in link.iter("mesh"):
            out[mesh.get("filename")] = robots_spec.resolve_mesh(rid, mesh.get("filename"))
    return out


def convert_composite_meshes(force: bool = False) -> list[Path]:
    """The arm and gripper COLLADA meshes as OBJ, under `composite_assets()`."""
    rid = "myagv_mycobot280"
    urdf = Urdf(robots_spec.urdf_path(rid))
    written = []
    for src in sorted(set(_arm_meshes(urdf, rid).values())):
        dst = composite_assets() / f"{src.stem}.obj"
        if dst.exists() and not force:
            continue
        if not src.exists():
            raise SystemExit(f"{src} is missing: run ./fetch_robot_assets.sh {rid}")
        v, f = read_dae(src)
        write_obj(dst, v, f)
        written.append(dst)
        print(f"  {src.parent.name}/{src.name} -> {dst.name}: {len(v)} vertices, "
              f"{len(f)} triangles", file=sys.stderr)
    return written


def _mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[float(x) for x in line.split()[1:4]] for line in path.open()
                  if line.startswith("v ")])
    return v.min(axis=0), v.max(axis=0), v


def build_composite(write_assets: bool = True) -> str:
    rid = "myagv_mycobot280"
    spec = robots_spec.robot(rid)
    urdf = Urdf(spec.urdf)
    out_dir = robots_spec.model_dir(rid)
    if write_assets:
        convert_composite_meshes()
    myagv_xml = robots_spec.model_xml(spec.base)
    agv = ET.parse(myagv_xml).getroot()
    agv_assets = myagv_xml.parent / agv.find("compiler").get("meshdir", "")

    root = _model(rid, f"robots_specs/{rid}/{spec.urdf.name}, its meshes, and the myAGV's "
                       f"generated model ({_relpath(myagv_xml, out_dir.parent)})")
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true",
                                     "meshdir": "assets"})
    ET.SubElement(root, "option", {"integrator": "implicitfast", "timestep": "0.002"})
    # The myAGV's own defaults (its planar drive gains and chassis classes), then the arm's.
    root.append(agv.find("default"))
    _defaults(root, rid, servo_kp=20, prefix=ARM_CLASS)
    asset = ET.SubElement(root, "asset")
    for mesh in agv.find("asset").findall("mesh"):
        ET.SubElement(asset, "mesh", {"name": mesh.get("name"),
                                      "file": _relpath(agv_assets / mesh.get("file"),
                                                       out_dir / "assets")})
    for material in agv.find("asset").findall("material"):
        asset.append(material)

    table = MeshTable()

    def mesh_for(link: str, kind: str, filename: str) -> str:
        stem = Path(filename).stem
        return table.add(stem, f"{stem}.obj")

    world = ET.SubElement(root, "worldbody")
    base = agv.find("worldbody/body")
    world.append(base)
    # The arm's base plate stands on the deck: its lowest point on the myAGV's top plate
    # where it stands (both measured off the meshes, at their own scale).
    g_base = urdf.links[urdf.root]
    visual = g_base.find("visual")
    g_xyz, g_quat = _origin(visual)
    lo, hi, pts = _mesh_bounds(composite_assets() / f"{Path(visual.find('geometry/mesh').get('filename')).stem}.obj")
    w, x, y, z = g_quat
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    with np.errstate(all="ignore"):  # numpy 2.2 on Accelerate warns spuriously here
        in_frame = pts @ rot.T + np.asarray(g_xyz)
    plate_bottom = float(in_frame[:, 2].min())
    reach = in_frame[:, :2].max(axis=0), in_frame[:, :2].min(axis=0)
    deck = [l for l in agv.iter("mesh")]  # noqa: F841  (asset list read above)
    _, _, up = _mesh_bounds(agv_assets / "myagv_up.obj")
    under = up[(up[:, 0] >= reach[1][0] + ARM_MOUNT_XY[0]) & (up[:, 0] <= reach[0][0] + ARM_MOUNT_XY[0])
               & (up[:, 1] >= reach[1][1] + ARM_MOUNT_XY[1]) & (up[:, 1] <= reach[0][1] + ARM_MOUNT_XY[1])]
    deck_z = float(under[:, 2].max())
    mount = ET.SubElement(base, "body", {"name": urdf.root, "childclass": rid,
                                         "pos": _fmt((ARM_MOUNT_XY[0], ARM_MOUNT_XY[1],
                                                      deck_z - plate_bottom))})

    link_inertials = _mjcf_inertials(spec.mjcf)

    def inertial_for(link: str):
        return link_inertials.get(link)

    emit_link(urdf, urdf.root, mount, mesh_for, prefix=ARM_CLASS, inertial_for=inertial_for,
              collide=True)
    for body in mount.iter("body"):
        gripper = body.get("name", "").startswith("gripper")
        for geom in body.findall("geom"):
            if geom.get("class") == f"{ARM_CLASS}visual":
                geom.set("rgba", GRIPPER_RGBA if gripper else ARM_RGBA)
            elif gripper:
                geom.set("density", _fmt([GRIPPER_DENSITY]))
    for name, file in table.meshes.items():
        ET.SubElement(asset, "mesh", {"name": name, "file": file})
    actuator = ET.SubElement(root, "actuator")
    for act in agv.find("actuator"):
        actuator.append(act)
    driven = {j for j, *_ in urdf.mimics()}
    _servos(actuator, [(n, r) for n, r in _joint_ranges(mount) if n not in driven],
            prefix=ARM_CLASS)
    _mimics(root, urdf)
    return _serialise(root)


def _mjcf_inertials(path: Path) -> dict[str, dict]:
    """Body name -> its `<inertial>` attributes, from an official MJCF."""
    out = {}
    for body in ET.parse(path).getroot().iter("body"):
        inertial = body.find("inertial")
        if inertial is not None:
            out[body.get("name")] = dict(inertial.attrib)
    return out


# ------------------------------------------------------------------------ entry points


BUILDERS = {"myagv_mycobot280": build_composite, "rosmaster_x3_plus": build_x3}


def generated_assets(rid: str) -> list[Path]:
    """The mesh files `rid`'s model.xml needs that this module writes (none for most)."""
    if rid == "myagv_mycobot280":
        urdf = Urdf(robots_spec.urdf_path(rid))
        return [composite_assets() / f"{p.stem}.obj"
                for p in sorted(set(_arm_meshes(urdf, rid).values()))]
    if rid == "rosmaster_x3_plus":
        return [_x3_obj(p) for p in x3_converted()]
    return []


def ensure(rid: str) -> None:
    """Make `rid`'s model loadable: write any generated mesh it is missing."""
    if any(not p.exists() for p in generated_assets(rid)):
        print(f"{rid}: converting the meshes MuJoCo cannot read (once)", file=sys.stderr)
        (convert_composite_meshes if rid == "myagv_mycobot280" else convert_x3_meshes)()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("robots", nargs="*", help=f"ids (default: {', '.join(BUILT)})")
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and fail if a committed model.xml differs")
    ap.add_argument("--force", action="store_true", help="re-convert every generated mesh")
    ap.add_argument("--assets", action="store_true",
                    help="only write the converted meshes a committed model.xml needs "
                         "(what each engine's run.sh setup runs)")
    args = ap.parse_args(argv)
    if args.assets:
        for rid in args.robots or BUILT:
            ensure(rid)
        return 0
    bad = 0
    for rid in args.robots or BUILT:
        if rid not in BUILDERS:
            raise SystemExit(f"{rid}: not built here (built: {', '.join(BUILT)})")
        if args.force:
            (convert_composite_meshes if rid == "myagv_mycobot280"
             else convert_x3_meshes)(force=True)
        ensure(rid)
        text = BUILDERS[rid]()
        path = robots_spec.model_xml(rid)
        if args.check:
            same = path.exists() and path.read_text() == text
            print(f"{rid}: {'up to date' if same else 'DIFFERS from what its sources generate'}")
            bad += not same
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
