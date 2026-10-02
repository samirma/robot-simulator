"""The robot registry as recorded in robots_specs/high_level_spec.md, read for the tests.

The specification is the registry (there is no YAML registry); this module only parses
its robot sections: `## N. <name>` headings with `* **Key:** value` bullets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SPECS = Path(__file__).resolve().parent.parent
ROOT = SPECS.parent
SPEC = SPECS / "high_level_spec.md"


@dataclass
class Robot:
    id: str
    name: str
    kind: str
    section: str
    urdf: str | None = None
    mjcf: str | None = None
    ros: str | None = None
    base: str | None = None
    arm: str | None = None
    revisions: list[str] = field(default_factory=list)
    sha256: dict[str, str] = field(default_factory=dict)

    @property
    def composite(self) -> bool:
        return self.base is not None


def _code(value: str) -> str | None:
    m = re.search(r"`([^`]+)`", value)
    return m.group(1) if m else None


def robots() -> list[Robot]:
    text = SPEC.read_text()
    out = []
    for m in re.finditer(r"^## \d+\. (.+?)\n(.*?)(?=^## |\Z)", text, re.S | re.M):
        name, body = m.group(1).strip(), m.group(2)
        bullets = {}
        for b in re.finditer(r"^\* \*\*(.+?):\*\*\s*(.*(?:\n  .*)*)", body, re.M):
            bullets[b.group(1).strip("` ")] = " ".join(b.group(2).split())
        if "Robot id" not in bullets:
            continue
        r = Robot(id=_code(bullets["Robot id"]), name=name, kind=_code(bullets.get("Kind", "")) or "",
                  section=body)
        r.urdf = _code(bullets.get("Official URDF", ""))
        r.mjcf = _code(bullets.get("Official MJCF", ""))
        r.ros = _code(bullets.get("ROS interface", ""))
        r.base = _code(bullets.get("base", "")) if "base" in bullets else None
        r.arm = _code(bullets.get("arm", "")) if "arm" in bullets else None
        r.revisions = re.findall(r"revision\s+`([0-9a-f]{40})`", body)
        for key, val in bullets.items():
            if key.endswith("SHA-256"):
                r.sha256[key] = _code(val)
        out.append(r)
    return out


def by_id() -> dict[str, Robot]:
    return {r.id: r for r in robots()}
