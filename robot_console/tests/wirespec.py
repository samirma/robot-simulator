"""Build a wire description (for the fake bridge or a real container node) from a profile.

The wire presents a profile's required rows (optionally also its optional rows), with the
documented types, and its cameras publishing small frames in a documented encoding. Tests
mutate the result to produce wrong types, missing endpoints, extra endpoints etc.
"""

from __future__ import annotations

import copy
from typing import Optional

from robot_console.profiles import Profile


def wire_spec(p: Profile, *, namespace: str = "", include_optional: bool = False,
              camera_rate: Optional[float] = None) -> dict:
    cams = {c.topic: c for c in p.cameras}
    spec = {"dialect": p.dialect, "topics": [], "services": [], "actions": []}
    for e in p.endpoints:
        if e.optional and not include_optional:
            continue
        row = {"name": p.resolve(e.name, namespace), "type": e.type}
        if e.kind == "topic":
            row["direction"] = e.direction or "out"
            if e.name in cams:
                c = cams[e.name]
                row.update(camera=True, direction="out",
                           encoding=(c.encodings or ("rgb8",))[0],
                           rate_hz=camera_rate or c.rate_hz or 10)
            elif e.rate_hz and row["direction"] == "out":
                row["rate_hz"] = min(float(e.rate_hz), 5.0)
            spec["topics"].append(row)
        elif e.kind == "service":
            spec["services"].append(row)
        else:
            spec["actions"].append(dict(row, exec_s=3.0))
    return spec


def retype(spec: dict, name: str, new_type: str) -> dict:
    s = copy.deepcopy(spec)
    for sec in ("topics", "services", "actions"):
        for r in s[sec]:
            if r["name"] == name:
                r["type"] = new_type
                return s
    raise KeyError(name)


def drop(spec: dict, name: str) -> dict:
    s = copy.deepcopy(spec)
    for sec in ("topics", "services", "actions"):
        s[sec] = [r for r in s[sec] if r["name"] != name]
    return s


def merge(*specs: dict) -> dict:
    out = {"dialect": specs[0]["dialect"], "topics": [], "services": [], "actions": []}
    for s in specs:
        for sec in ("topics", "services", "actions"):
            seen = {r["name"] for r in out[sec]}
            out[sec] += [copy.deepcopy(r) for r in s[sec] if r["name"] not in seen]
    return out
