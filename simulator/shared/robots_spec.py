"""The one reader of `robots_specs/robots.yml`, and where every robot file is found.

Robot embodiments are recorded once, in `robots_specs/` at the workspace root: each
robot's id, placement, whether the simulator hosts it, and the paths of its URDF, MJCF,
meshes and ROS interface file. Everything in the simulator that needs a robot id or a
robot file asks this module, so there is no second list of robots to drift from it.

`shared/robots/<id>/` holds only what the simulator builds *from* those files -- a
generated MJCF where no usable official one exists, actuators, collision
simplifications -- never a copy of them (`model_dir`, `model_xml`).

Deliberately a plain module, not a package named `robots`: each engine already puts its
own `robots/` adapter package on `PYTHONPATH`, and a second importable `robots` package
here would shadow it.

Shell scripts ask it too, through its command line:

    python robots_spec.py list      # the simulated ids: `id  name (placement)`, one per line
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

#: The workspace root: `simulator/shared/robots_spec.py` -> `.`.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPECS_DIR = WORKSPACE_ROOT / "robots_specs"
ROBOTS_YML = SPECS_DIR / "robots.yml"

#: Simulator-built additions derived from `robots_specs/<id>/`.
MODELS_DIR = Path(__file__).resolve().parent / "robots"

PACKAGE_PREFIX = "package://"

#: The one `kind` that stays where it is placed; every other kind moves (robots.yml).
FIXED_KIND = "arm"


@dataclass(frozen=True)
class Robot:
    """One `robots.yml` entry, with its file paths resolved against `robots_specs/`."""

    id: str
    name: str
    kind: str
    placement: str
    simulated: bool
    base: str | None
    urdf: Path | None
    mjcf: Path | None
    meshes: tuple[Path, ...]
    ros: Path | None

    @property
    def mobile(self) -> bool:
        """Whether it moves about the scene: every kind but a fixed `arm` (robots.yml)."""
        return self.kind != FIXED_KIND

    @property
    def folder(self) -> Path:
        """`robots_specs/<id>/`, which a `package://<pkg>/` mesh path resolves to."""
        return SPECS_DIR / self.id


def _path(value) -> Path | None:
    return SPECS_DIR / value if value else None


@lru_cache(maxsize=1)
def robots() -> tuple[Robot, ...]:
    """Every robot in `robots.yml`, in file order."""
    data = yaml.safe_load(ROBOTS_YML.read_text())
    out = []
    for entry in data["robots"]:
        meshes = entry.get("meshes") or ()
        if isinstance(meshes, str):
            meshes = (meshes,)
        out.append(Robot(
            id=str(entry["id"]),
            name=str(entry.get("name", entry["id"])),
            kind=str(entry.get("kind", "")),
            placement=str(entry["placement"]),
            simulated=bool(entry.get("simulated", False)),
            base=entry.get("base"),
            urdf=_path(entry.get("urdf")),
            mjcf=_path(entry.get("mjcf")),
            meshes=tuple(SPECS_DIR / m for m in meshes),
            ros=_path(entry.get("ros")),
        ))
    ids = [r.id for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{ROBOTS_YML}: duplicate robot ids in {ids}")
    return tuple(out)


def robot(robot_id: str) -> Robot:
    for r in robots():
        if r.id == robot_id:
            return r
    raise KeyError(f"no robot {robot_id!r} in {ROBOTS_YML}; it has: {', '.join(ids())}")


def ids() -> tuple[str, ...]:
    return tuple(r.id for r in robots())


def simulated_ids() -> tuple[str, ...]:
    """The ids every `--robot`/`--robots` flag in the simulator accepts."""
    return tuple(r.id for r in robots() if r.simulated)


def describe_simulated(indent: str = "") -> str:
    """Every simulated id with its name and placement, one per line, read from `robots.yml`
    each time: the list every `--robot`/`--robots` help and unknown-id refusal shows."""
    rows = [(r.id, r.name, r.placement) for r in robots() if r.simulated]
    width = max((len(i) for i, _, _ in rows), default=0)
    return "\n".join(f"{indent}{i:<{width}}  {n} ({p})" for i, n, p in rows)


def placement(robot_id: str) -> str:
    """`floor` or `worktop`."""
    return robot(robot_id).placement


def mobile(robot_id: str) -> bool:
    """Whether the robot moves about the scene, on a planar base or a gait (robots.yml)."""
    return robot(robot_id).mobile


def worktop_ids() -> tuple[str, ...]:
    """The simulated robots that stand on the worktop rather than the floor."""
    return tuple(i for i in simulated_ids() if placement(i) == "worktop")


def urdf_path(robot_id: str) -> Path:
    p = robot(robot_id).urdf
    if p is None:
        raise KeyError(f"{robot_id!r} records no urdf in {ROBOTS_YML}")
    return p


def mjcf_path(robot_id: str) -> Path:
    p = robot(robot_id).mjcf
    if p is None:
        raise KeyError(f"{robot_id!r} records no official mjcf in {ROBOTS_YML}")
    return p


def mesh_dirs(robot_id: str) -> tuple[Path, ...]:
    return robot(robot_id).meshes


def ros_path(robot_id: str) -> Path:
    p = robot(robot_id).ros
    if p is None:
        raise KeyError(f"{robot_id!r} records no ros file in {ROBOTS_YML}")
    return p


def resolve_mesh(robot_id: str, filename: str, relative_to: Path | None = None) -> Path:
    """Where a mesh path found in one of the robot's description files is on disk.

    `package://<pkg>/<rest>` resolves to `robots_specs/<id>/<rest>`, whatever `<pkg>` is:
    each robot's folder is laid out as its upstream package. A plain relative path is
    taken relative to `relative_to` (the referencing file's folder), by default the URDF's.
    """
    r = robot(robot_id)
    if filename.startswith(PACKAGE_PREFIX):
        rest = filename[len(PACKAGE_PREFIX):]
        _, _, rest = rest.partition("/")
        return r.folder / rest
    p = Path(filename)
    if p.is_absolute():
        return p
    base = relative_to if relative_to is not None else (r.urdf.parent if r.urdf else r.folder)
    return base / p


def model_dir(robot_id: str) -> Path:
    """`shared/robots/<id>/`: what the simulator builds from the robot's spec files."""
    return MODELS_DIR / robot_id


def model_xml(robot_id: str) -> Path:
    """The simulator-generated MJCF for a robot whose model is a generated file."""
    return model_dir(robot_id) / "model.xml"


def check_simulated(names) -> list[str]:
    """`names` (a list, or a comma-separated string) as a list of simulated ids.

    Raises `ValueError` naming what is empty, unknown or not simulated.
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]
    names = [n for n in names if n]
    if not names:
        raise ValueError("no robot named")
    sim = simulated_ids()
    bad = [n for n in names if n not in sim]
    if bad:
        known = set(ids())
        why = ", ".join(
            f"{n!r} ({'not simulated' if n in known else 'not in robots_specs/robots.yml'})"
            for n in bad)
        raise ValueError(f"unknown robot(s) {why}; the simulated robots are:\n"
                         f"{describe_simulated('  ')}")
    return names


def main(argv: list[str]) -> int:
    if argv == ["list"]:
        print(describe_simulated())
        return 0
    print("usage: robots_spec.py list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
