"""Every simulated robot stands at its placement, without interpenetration (spec §5).

Run from an engine's venv, with the engine named -- each engine's `tools/test_placement.py`
runs it for its own:

    molmospaces/.venv/bin/python shared/tests/placement_check.py --engine molmospaces
    robocasa/.venv/bin/python    shared/tests/placement_check.py --engine robocasa

For each robot alone, and for the full fleet, it builds the world `kitchen.sh serve`
builds -- the engine's default scene, every staging flag at its default -- through the
shared spawn tool, without serving it, and checks:

* each robot stands on what its `placement` in robots.yml says: a floor robot on the
  floor, a worktop robot on the worktop, with that surface under its whole footprint;
* no two robots' footprints overlap, and no robot is in contact 1 mm deep with anything
  at its spawn pose -- `spawn.build_world` collects both, and the AiNex's soles are on
  its surface;
* a fixed robot's base is not sunk into its surface (collisions between a fixed base
  and a static worktop are never generated, so only geometry can say so);
* every worktop robot can get at the staged objects: the SO-101 by a top grasp solved on
  the compiled model, the AiNex by a walk over the worktop.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from contextlib import redirect_stderr
from pathlib import Path

import mujoco
import numpy as np

SHARED = Path(__file__).resolve().parents[1]
SIM = SHARED.parent
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import placement  # noqa: E402
import reach  # noqa: E402
import robots_spec  # noqa: E402
import spawn  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def load_engine(name: str):
    root = SIM / name
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools import spawn_robot  # noqa: PLC0415

    return spawn_robot.ENGINE


def scene_flags(engine: str) -> list[str]:
    """The engine's default scene (spec §2.3)."""
    if engine == "molmospaces":
        xml = subprocess.run(
            [sys.executable, str(SIM / "molmospaces/tools/resolve_scene.py"), "ithor", "1"],
            check=True, capture_output=True, text=True).stdout.strip()
        return ["--scene", xml]
    return ["--layout", "1", "--style", "1"]


def build(engine, flags, robots: str):
    args = spawn.build_parser(engine).parse_args([robots, *flags, "--ros-port", "0"])
    log = io.StringIO()
    with redirect_stderr(log):
        world = spawn.build_world(engine, args)
    return world, log.getvalue()


def surface_under(scene, xy, z: float) -> bool:
    """A top face at `z` under `xy` in the bare scene (no robot, no task)."""
    geomid = np.zeros(1, dtype=np.int32)
    start = np.array([xy[0], xy[1], z + 0.05])
    d = mujoco.mj_ray(scene.model, scene.data, start, np.array([0.0, 0.0, -1.0]), None, 1, -1,
                      geomid)
    return geomid[0] >= 0 and abs(d - 0.05) <= 0.015


def lowest_collider(model, data, prefix: str) -> float:
    """The lowest point of a robot's collision geoms, from their world AABBs."""
    low = np.inf
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if not body.startswith(prefix) or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        centre = data.geom_xpos[g] + rot @ model.geom_aabb[g][:3]
        extent = np.abs(rot) @ model.geom_aabb[g][3:]
        low = min(low, float(centre[2] - extent[2]))
    return low


def examine(label: str, world, log: str) -> None:
    print(f"{label}:")
    model, data = world.model, world.data
    check(f"{label}: no placement or start-up problem", not world.problems,
          "; ".join(world.problems))
    for inst in world.instances:
        want = robots_spec.placement(inst.name)
        check(f"{label}: {inst.name} stands on the {want}", inst.on == want,
              f"on the {inst.on or 'nothing'}")
        if want == "floor":
            check(f"{label}: {inst.name} on the floor at z 0", abs(inst.surface_z) < 1e-9,
                  f"z {inst.surface_z:.3f}")
            ring = [inst.xy + placement.SUPPORT_RADIUS[inst.name] * np.array([np.cos(a), np.sin(a)])
                    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
            check(f"{label}: floor under {inst.name}'s footprint",
                  all(surface_under(world.scene, p, 0.0) for p in [inst.xy, *ring]))
        else:
            # What it stands on has the worktop under all of it; a robot whose body does
            # not collide with the scene (the AiNex) also needs the headroom checked here,
            # where an arm's links would show as contacts in `world.problems`.
            surface = placement.SurfaceMap(world.scene.model, world.scene.data, world.worktop)
            height = placement.ROBOT_HEIGHT[inst.name] if inst.name == "ainex" else 0.0
            check(f"{label}: the worktop under all of what {inst.name} stands on",
                  surface.footprint_fits(inst.xy, placement.SUPPORT_RADIUS[inst.name], height),
                  f"at ({inst.xy[0]:.2f}, {inst.xy[1]:.2f}) on {world.worktop.name}")
        if not inst.holonomic:
            low = lowest_collider(model, data, inst.mjcf)
            check(f"{label}: {inst.name}'s base is not sunk into the {inst.on}",
                  low >= inst.surface_z - 0.001,
                  f"lowest collider {1000 * (low - inst.surface_z):+.1f} mm")
    pairs = placement.overlaps(world.instances)
    check(f"{label}: no two robots' footprints overlap", not pairs, "; ".join(pairs))
    if world.task is not None:
        objects = world.task.object_positions(data)
        for inst in world.instances:
            if inst.on != "worktop":
                continue
            if inst.name == "so101":
                for name, xyz in objects.items():
                    g = reach.top_grasp(model, data, inst.mjcf, xyz)
                    check(f"{label}: so101 has a top grasp at the {name}", g.solved,
                          f"{g.position_error * 1000:.1f} mm, {g.tilt_deg:.1f} deg")
            else:
                for name in objects:
                    line = next((l for l in log.splitlines()
                                 if l.startswith(f"reach {inst.name} {name}:")), "")
                    check(f"{label}: {inst.name} can walk to the {name}", " in reach" in line,
                          line)


def run(engine_name: str) -> int:
    """Every simulated robot alone, then all of them together; 0 when every check passes."""
    engine = load_engine(engine_name)
    flags = scene_flags(engine_name)
    ids = list(robots_spec.simulated_ids())
    for robots in [*ids, ",".join(ids)]:
        world, log = build(engine, flags, robots)
        examine(f"{engine_name} {robots}", world, log)
    print("all checks passed" if not failures else f"FAILED: {len(failures)} check(s)")
    return 0 if not failures else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=("molmospaces", "robocasa"))
    return run(ap.parse_args().engine)


if __name__ == "__main__":
    raise SystemExit(main())
