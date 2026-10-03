"""The robot registry: the robot specification's robot files, `robots_specs/<id>.md`,
read as they are now.

There is no separate YAML registry (workspace spec §2): every robot id the simulator
accepts, its name, kind, files and ROS dialect are parsed from the robot files (robot
specification §2), one robot per file, named by its id.

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
SPECS = REPO_ROOT / "robots_specs"
SPEC = SPECS / "high_level_spec.md"


class RegistryError(ValueError):
    pass


class Robot:
    """One robot file of the robot specification."""

    def __init__(self, rid: str, name: str, fields: Dict[str, str]):
        self.id = rid
        self.name = name
        self.fields = fields
        self.kind = _code(fields.get("kind", "")) or ""
        self.folder = _code(fields.get("folder", "")) or f"robots_specs/{rid}/"
        self.urdf = _code(fields.get("official urdf", "")) or None
        self.mjcf = _code(fields.get("official mjcf", "")) or None
        ros = fields.get("ros interface", "")
        self.ros_file = _code(ros) or None

    # ------------------------------------------------------------------ derived facts

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


def _code(text: str) -> Optional[str]:
    m = re.search(r"`([^`]+)`", text or "")
    return m.group(1) if m else None


def _parse(text: str) -> Optional[Robot]:
    """The robot of one robot file: `# <name>` then `* **Key:** value` bullets."""
    title, _, body = text.lstrip().partition("\n")
    name = title.lstrip("#").strip()
    fields: Dict[str, str] = {}
    # A bullet runs until the next bullet or a blank line.
    for m in re.finditer(r"^\* +\*\*(.+?):?\*\*:?\s*(.*?)(?=^\* |^\s*$|\Z)", body, re.M | re.S):
        key = m.group(1).strip().rstrip(":").strip().lower()
        fields[key] = " ".join(m.group(2).split())
    rid = _code(fields.get("robot id", ""))
    return Robot(rid, name, fields) if rid else None


_cache: Dict[str, Tuple[tuple, List[Robot]]] = {}


def _robot_files() -> List[Path]:
    """Every `.md` of the robot specification but its main document and schema, by name."""
    if not SPEC.is_file():
        raise RegistryError(f"the robot specification {SPEC} is missing")
    return sorted(p for p in SPECS.glob("*.md") if p.name not in ("high_level_spec.md", "SCHEMA.md"))


def robots() -> List[Robot]:
    """Every robot of the robot specification, ordered by file name (re-read on change)."""
    files = _robot_files()
    stamp = tuple((p.name, p.stat().st_mtime) for p in files)
    hit = _cache.get(str(SPECS))
    if hit and hit[0] == stamp:
        return hit[1]
    found = []
    for p in files:
        r = _parse(p.read_text(encoding="utf-8"))
        if r is None:
            continue
        if r.id != p.stem:
            raise RegistryError(f"{p.relative_to(REPO_ROOT)} records robot id {r.id!r}; a robot "
                                f"file is named by its robot id")
        found.append(r)
    _cache[str(SPECS)] = (stamp, found)
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


# ---------------------------------------------------------------- required files


def _xml_text(path: Path) -> Optional[str]:
    """An XML file's text without its comments (a reference commented out is no reference),
    or None when it cannot be read."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _model_files(model: Path) -> List[Path]:
    """Every file an MJCF references (meshes, textures, includes), resolved."""
    out: List[Path] = []
    text = _xml_text(model)
    if text is None:
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
    """Every file a URDF references (meshes), resolved."""
    out: List[Path] = []
    text = _xml_text(urdf)
    if text is None:
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
    """Required files of this robot that are not on disk."""
    missing: List[str] = []
    need: List[Path] = []
    for rel in (robot.urdf, robot.mjcf, robot.ros_file):
        if rel:
            need.append(REPO_ROOT / rel)
    model = robot.model_path()
    if model is not None:
        need.append(model)
        need.extend(_model_files(model))
    if robot.urdf:
        need.extend(_urdf_files(REPO_ROOT / robot.urdf))
    for p in need:
        if not p.exists():
            try:
                missing.append(str(p.relative_to(REPO_ROOT)))
            except ValueError:
                missing.append(str(p))
    return sorted(dict.fromkeys(missing))
