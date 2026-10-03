"""The robot registry as recorded in the robot files `robots_specs/<id>.md`, read for the tests.

The specification is the registry (there is no YAML registry); this module only parses
the robot files (robot specification §2): `# <name>` then `* **Key:** value` bullets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SPECS = Path(__file__).resolve().parent.parent
ROOT = SPECS.parent
SPEC = SPECS / "high_level_spec.md"
NOT_ROBOT_FILES = ("high_level_spec.md", "SCHEMA.md")


@dataclass
class Robot:
    id: str
    name: str
    kind: str
    section: str
    file: Path
    urdf: str | None = None
    mjcf: str | None = None
    model: str | None = None
    ros: str | None = None
    bullets: dict[str, str] = field(default_factory=dict)
    revisions: list[str] = field(default_factory=list)
    sha256: dict[str, str] = field(default_factory=dict)


def _code(value: str) -> str | None:
    m = re.search(r"`([^`]+)`", value)
    return m.group(1) if m else None


def robot_files() -> list[Path]:
    return sorted(p for p in SPECS.glob("*.md") if p.name not in NOT_ROBOT_FILES)


def robots() -> list[Robot]:
    out = []
    for path in robot_files():
        text = path.read_text()
        m = re.match(r"# (.+?)\n(.*)", text, re.S)
        if not m:
            continue
        name, body = m.group(1).strip(), m.group(2)
        bullets = {}
        for b in re.finditer(r"^\* \*\*(.+?):\*\*\s*(.*(?:\n  .*)*)", body, re.M):
            bullets[b.group(1).strip("` ")] = " ".join(b.group(2).split())
        if "Robot id" not in bullets:
            continue
        r = Robot(id=_code(bullets["Robot id"]), name=name, kind=_code(bullets.get("Kind", "")) or "",
                  section=body, file=path)
        r.urdf = _code(bullets.get("Official URDF", ""))
        r.mjcf = _code(bullets.get("Official MJCF", ""))
        r.model = _code(bullets.get("MuJoCo model", ""))
        r.bullets = bullets
        r.ros = _code(bullets.get("ROS interface", ""))
        r.revisions = re.findall(r"revision\s+`([0-9a-f]{40})`", body)
        for key, val in bullets.items():
            if key.endswith("SHA-256"):
                r.sha256[key] = _code(val)
        out.append(r)
    return out


def by_id() -> dict[str, Robot]:
    return {r.id: r for r in robots()}
