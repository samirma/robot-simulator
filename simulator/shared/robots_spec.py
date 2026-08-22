"""Locate engine-neutral robot specification files.

The MJCF (`model.xml` + `assets/`), the URDF, and the upstream source models for each
robot live under `shared/robots/<name>/` so every simulator engine loads *the same*
hardware description. Only the engine-specific adapter code (how that model is wired into
MolmoSpaces or robosuite) differs per engine.

This is deliberately a plain module, not a package named `robots`: each engine already
puts its own `robots/` adapter package on `PYTHONPATH`, so a second importable `robots`
package here would shadow it. The specs are referenced by filesystem path, never imported.
"""

from __future__ import annotations

from pathlib import Path

SHARED_ROBOTS_DIR = Path(__file__).resolve().parent / "robots"


def spec_dir(name: str) -> Path:
    """Return the shared spec directory for a robot, e.g. `.../shared/robots/so101`."""
    d = SHARED_ROBOTS_DIR / name
    if not d.is_dir():
        raise FileNotFoundError(f"no shared robot spec for {name!r} at {d}")
    return d


def model_xml(name: str) -> Path:
    """Return the engine-neutral MJCF entry point for a robot."""
    return spec_dir(name) / "model.xml"
