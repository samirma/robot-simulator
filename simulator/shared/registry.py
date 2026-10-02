"""The robot registry: `robots_specs/high_level_spec.md`, read as it is now.

There is no separate YAML registry (workspace spec §2): every robot id the simulator
accepts, its name, kind, files, ROS dialect and, for a composite, its components and
mounting transform, are parsed from the robot specification's per-robot sections.

Stdlib only and Python 3.8-compatible: `spawn.sh` runs it under any python3, and the
wire containers import it through the read-only mount.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SIM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("RSIM_REPO_ROOT", str(SIM_ROOT.parent)))
SPEC = REPO_ROOT / "robots_specs" / "high_level_spec.md"

KINDS = ("mobile_base", "arm", "humanoid", "mobile_manipulator")


class RegistryError(ValueError):
    pass


class Robot:
    """One robot section of the robot specification."""

    def __init__(self, rid: str, name: str, section: str, fields: Dict[str, str]):
        self.id = rid
        self.name = name
        self.section = section
        self.fields = fields
        self.kind = _code(fields.get("kind", "")) or ""
        self.base = _code(fields.get("`base`", "")) or None
        self.arm = _code(fields.get("`arm`", "")) or None
        self.folder = _code(fields.get("folder", "")) or (f"robots_specs/{rid}/" if not self.base else None)
        self.urdf = _code(fields.get("official urdf", "")) or None
        self.mjcf = _code(fields.get("official mjcf", "")) or None
        ros = fields.get("ros interface", "")
        self.ros_file = _code(ros) or None
        self.mounting = fields.get("mounting transform", "")

    # ------------------------------------------------------------------ derived facts

    @property
    def composite(self) -> bool:
        return bool(self.base and self.arm)

    @property
    def mobile(self) -> bool:
        return self.kind != "arm"

    @property
    def dialect(self) -> Optional[str]:
        if not self.ros_file:
            return None
        return "ros2" if self.ros_file.endswith("ros2.yml") else "ros1"

    def path(self, rel: Optional[str]) -> Optional[Path]:
        return (REPO_ROOT / rel) if rel else None

    @property
    def folder_path(self) -> Optional[Path]:
        return self.path(self.folder)

    def model_path(self) -> Optional[Path]:
        """The MJCF the simulator compiles: the official MJCF where the robot folder's
        `model.xml` does not supersede it, else the derived `model.xml`."""
        folder = self.folder_path
        if folder is not None and (folder / "model.xml").is_file():
            return folder / "model.xml"
        if self.mjcf:
            return self.path(self.mjcf)
        return folder / "model.xml" if folder is not None else None

    def mounting_transform(self) -> Tuple[str, List[float], List[float]]:
        """(base link, xyz, rpy) of the arm's root link relative to a named base link, read
        from the robot specification's mounting-transform entry (its only copy):
        "`<arm>` link `<arm link>` relative to `<base>` link `<base link>`: xyz = (..) m,
        rpy = (..) rad"."""
        text = self.mounting
        m = re.search(r"`([^`]+)`\s+link\s+`([^`]+)`\s+relative\s+to\s+`([^`]+)`\s+link\s+"
                      r"`([^`]+)`", text)
        xyz = _triple(text, "xyz")
        rpy = _triple(text, "rpy")
        if m is None or xyz is None or rpy is None:
            raise RegistryError(
                f"{self.id}: the robot specification's mounting transform is not recorded "
                f"as a named link with xyz and rpy: {text.strip()[:200]!r}")
        self.mount_arm_link = m.group(2)
        return m.group(4), xyz, rpy


def _code(text: str) -> Optional[str]:
    m = re.search(r"`([^`]+)`", text or "")
    return m.group(1) if m else None


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _triple(text: str, key: str) -> Optional[List[float]]:
    m = re.search(key + r"[^0-9+\-.]{0,40}(" + _NUM + r")[\s,;/]+(" + _NUM + r")[\s,;/]+(" + _NUM + ")",
                  text, re.I)
    if not m:
        return None
    return [float(m.group(i)) for i in (1, 2, 3)]


def _parse(text: str) -> List[Robot]:
    robots = []
    sections = re.split(r"^## +", text, flags=re.M)[1:]
    for sec in sections:
        title, _, body = sec.partition("\n")
        name = re.sub(r"^\d+\.\s*", "", title).strip()
        fields: Dict[str, str] = {}
        # A bullet runs until the next bullet or a blank line.
        for m in re.finditer(r"^\* +\*\*(.+?):?\*\*:?\s*(.*?)(?=^\* |^\s*$)", body, re.M | re.S):
            key = m.group(1).strip().rstrip(":").strip().lower()
            if key.startswith("`"):
                key = m.group(1).strip().rstrip(":").strip()
            fields[key] = " ".join(m.group(2).split())
        rid = _code(fields.get("robot id", ""))
        if rid:
            robots.append(Robot(rid, name, title.strip(), fields))
    return robots


_cache: Dict[str, Tuple[float, List[Robot]]] = {}


def robots() -> List[Robot]:
    """Every robot in the robot specification, in document order (re-read on change)."""
    try:
        mtime = SPEC.stat().st_mtime
    except OSError:
        raise RegistryError(f"the robot specification {SPEC} is missing")
    hit = _cache.get(str(SPEC))
    if hit and hit[0] == mtime:
        return hit[1]
    found = _parse(SPEC.read_text(encoding="utf-8"))
    _cache[str(SPEC)] = (mtime, found)
    return found


def ids() -> List[str]:
    return [r.id for r in robots()]


def get(rid: str) -> Robot:
    for r in robots():
        if r.id == rid:
            return r
    raise RegistryError(f"unknown robot id {rid!r}; accepted ids:\n{listing('  ')}")


def listing(indent: str = "  ") -> str:
    rows = robots()
    width = max((len(r.id) for r in rows), default=0)
    return "\n".join(f"{indent}{r.id.ljust(width)}  {r.name}" for r in rows)


def components(robot: Robot) -> List[Robot]:
    """The robots whose files and interfaces make this one: itself, or base and arm."""
    if robot.composite:
        return [get(robot.base), get(robot.arm)]
    return [robot]


def wires(robot: Robot) -> List[Tuple[str, Robot]]:
    """(wire role, interface owner) per wire: ("main", robot) or ("base", ..), ("arm", ..)."""
    if robot.composite:
        return [("base", get(robot.base)), ("arm", get(robot.arm))]
    return [("main", robot)]


# ---------------------------------------------------------------- required files


_MESH_EXT = (".stl", ".obj", ".dae", ".msh", ".png")


def _model_files(model: Path) -> List[Path]:
    """Every file an MJCF references (meshes, textures, includes), resolved."""
    out: List[Path] = []
    try:
        text = model.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    meshdir = re.search(r"meshdir\s*=\s*\"([^\"]*)\"", text)
    texdir = re.search(r"texturedir\s*=\s*\"([^\"]*)\"", text)
    assetdir = re.search(r"assetdir\s*=\s*\"([^\"]*)\"", text)
    for tag, attr_dir in (("mesh", meshdir or assetdir), ("texture", texdir or assetdir),
                          ("hfield", assetdir), ("include", None), ("skin", assetdir)):
        for m in re.finditer(r"<" + tag + r"\b[^>]*?\bfile\s*=\s*\"([^\"]+)\"", text):
            base = model.parent / (attr_dir.group(1) if attr_dir else "")
            p = base / m.group(1)
            out.append(p)
            if tag == "include" and p.is_file():
                out.extend(_model_files(p))
    return out


def _urdf_files(urdf: Path) -> List[Path]:
    out: List[Path] = []
    try:
        text = urdf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for m in re.finditer(r"filename\s*=\s*\"([^\"]+)\"", text):
        ref = m.group(1)
        if ref.startswith("package://"):
            # package://<pkg>/<rest>: resolved inside the robot folder, keeping upstream
            # relative paths (robot specification §1).
            rest = ref[len("package://"):].split("/", 1)
            rel = rest[1] if len(rest) > 1 else rest[0]
            cands = [urdf.parent / rel, urdf.parent / ref[len("package://"):]]
            out.append(next((c for c in cands if c.exists()), cands[0]))
        elif ref.startswith("file://"):
            out.append(Path(ref[len("file://"):]))
        else:
            out.append(urdf.parent / ref)
    return out


def missing_files(robot: Robot) -> List[str]:
    """Required files of this robot (and its components) that are not on disk."""
    missing: List[str] = []
    for comp in components(robot):
        need: List[Path] = []
        for rel in (comp.urdf, comp.mjcf, comp.ros_file):
            if rel:
                need.append(REPO_ROOT / rel)
        model = comp.model_path()
        if model is not None:
            need.append(model)
            need.extend(_model_files(model))
        if comp.urdf:
            need.extend(_urdf_files(REPO_ROOT / comp.urdf))
        for p in need:
            if not p.exists():
                try:
                    missing.append(str(p.relative_to(REPO_ROOT)))
                except ValueError:
                    missing.append(str(p))
    return sorted(dict.fromkeys(missing))


if __name__ == "__main__":  # python registry.py [list|missing <id>]
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "missing":
        print("\n".join(missing_files(get(sys.argv[2]))))
    else:
        print(listing())
