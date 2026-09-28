"""`kitchen.sh serve`'s command line, as spec §2.3 states it: what it refuses, which
namespaces it accepts, and what it stages for which robots.

Runs from either engine's venv, and starts no engine: every refusal here must happen
before a scene is compiled, which is also what makes it quick.

    molmospaces/.venv/bin/python shared/tests/test_kitchen_cli.py
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1]
SIM = SHARED.parent
KITCHEN = SIM / "kitchen.sh"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import placement  # noqa: E402
import robots_spec  # noqa: E402
import serve_args  # noqa: E402
import spawn  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def serve(*argv: str) -> tuple[int, str, float]:
    start = time.monotonic()
    done = subprocess.run([str(KITCHEN), "serve", *argv], capture_output=True, text=True,
                          timeout=120)
    return done.returncode, done.stdout + done.stderr, time.monotonic() - start


def refused(label: str, argv, pattern: str) -> None:
    code, out, took = serve(*argv)
    check(f"refuses {label}", code != 0 and re.search(pattern, out) is not None
          and ">> " not in out, f"exit {code} in {took:.1f} s: {out.strip()[-200:]}")


def help_text() -> str:
    code, out, _ = serve("--help")
    check("serve --help exits 0", code == 0, out[-200:])
    return out


def help_flags(text: str) -> set[str]:
    """The flags --help documents: those heading a `  --flag ...  description` line."""
    flags: set[str] = set()
    for line in text.splitlines():
        if line.startswith("  --"):
            flags |= set(re.findall(r"--[a-z-]+", re.split(r"\s{2,}", line.strip())[0]))
    return flags


def engine_parser(name: str):
    """The engine's spawn-tool parser, loaded by path so both fit in one process."""
    spec = importlib.util.spec_from_file_location(f"_spawn_{name}",
                                                  SIM / name / "tools" / "spawn_robot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return spawn.build_parser(module.ENGINE)


def test_help() -> None:
    print("--help:")
    text = help_text()
    listed = help_flags(text)
    synopsis = {"--engine", "--robots", "--ros-namespace", "--mujoco", "--port", "--scene",
                "--layout", "--style"}
    staging = {"--reference-table", "--no-dressing", "--reference-lighting",
               "--extra-lights", "--swap-objects"}
    check("lists every synopsis flag", synopsis <= listed, str(synopsis - listed))
    check("only the specified options are advertised", listed == synopsis,
          str(listed - synopsis))
    for name in ("molmospaces", "robocasa"):
        ap = engine_parser(name)
        known = {o for a in ap._actions for o in a.option_strings}  # noqa: SLF001
        check(f"{name}: staging has no command-line options", not (staging & known))
        passed = {"--headless", "--ros-port", "--ros-namespace"}
        passed |= {"--scene"} if name == "molmospaces" else {"--layout", "--style"}
        check(f"{name}: the spawn tool takes every flag kitchen.sh passes it", passed <= known,
              str(passed - known))
        dead = {"--camera", "--camera-hz", "--camera-size", "--scan-beams", "--scan-hz",
                "--scan-range", "--scan-min-range", "--scan-offset", "--no-scan",
                "--depth-hz", "--depth-size", "--depth-range", "--no-depth",
                "--no-scene-cameras", "--wrist-camera", "--task", "--task-objects",
                "--side-camera-mirror", "--objects", "--target", "--gripper",
                "--spawn-objects", "--mount-centre", "--reach", "--pos", "--yaw",
                "--host", "--control-hz", "--jpeg-quality", "--action-dir", "--timeout",
                "--render", "--render-camera", "--render-framing", "--width", "--height",
                "--distance", "--azimuth", "--elevation"}
        check(f"{name}: no dead flag is left on the spawn tool", not (dead & known),
              str(sorted(dead & known)))


def test_refusals() -> None:
    print("refusals:")
    refused("the other engine's scene flag, by name (--scene on robocasa)",
            ["--engine", "robocasa", "--scene", "ithor:1"], r"--scene is a MolmoSpaces scene flag")
    refused("the other engine's scene flag, by name (--layout on molmospaces)",
            ["--layout", "2"], r"--layout is a RoboCasa scene flag")
    refused("the other engine's scene flag, by name (--style on molmospaces)",
            ["--style", "2"], r"--style is a RoboCasa scene flag")
    for flag in ("--bogus", "--no-scene-cameras", "--camera-hz", "--scan-hz", "--depth-hz",
                 "--task-objects", "--side-camera-mirror", "--no-swap-objects",
                 "--no-reference-table", "--wrist-camera", "--control-hz",
                 "--reference-table", "--no-dressing", "--reference-lighting",
                 "--extra-lights", "--swap-objects"):
        refused(f"a flag --help does not list ({flag})", [flag], rf"unknown flag '{flag}'")
    refused("an unknown engine", ["--engine", "gazebo"], r"--engine: expected")
    refused("a scene that is not ithor:<n> or procthor:<n>", ["--scene", "kitchen:1"],
            r"--scene: expected")
    refused("a layout out of 1-60", ["--engine", "robocasa", "--layout", "61"], r"--layout")
    refused("a port out of range", ["--port", "70000"], r"--port")
    refused("a flag missing its value", ["--robots"], r"needs a value")
    physical = [r for r in robots_spec.ids() if r not in robots_spec.simulated_ids()]
    if physical:  # every robot robots.yml has is simulated today
        refused("a robot that is not simulated", ["--robots", physical[0]], r"not simulated")
    refused("a robot robots.yml does not have", ["--robots", "forklift"],
            r"not in robots_specs/robots.yml")
    refused("a robot named twice", ["--robots", "so101,so101"], r"more than once")


def test_namespaces() -> None:
    print("--ros-namespace:")
    refused("a namespace with more than one robot", ["--robots", "so101,myagv",
                                                      "--ros-namespace", "arm"], r"lone robot")
    refused("'' with more than one robot", ["--robots", "myagv,ainex", "--ros-namespace", ""],
            r"lone robot")
    for ns in ("scene", "rosapi", "rosbridge_websocket", "simulator", "/scene/cam"):
        refused(f"one that collides with another provider ({ns})",
                ["--robots", "myagv", "--ros-namespace", ns], r"collides with")
    for ns in ("1abc", "a//b", "a/", "a b", "a__b", "a-b", "~x"):
        refused(f"an invalid ROS namespace ({ns!r})", ["--robots", "ainex", "--ros-namespace", ns],
                r"--ros-namespace")
    for ns, wire in (("", ""), ("arm", "arm"), ("/arm", "arm"), ("lab/arm_1", "lab/arm_1")):
        try:
            got = serve_args.fleet_namespaces(["so101"], ns)
        except ValueError as exc:
            got = [f"refused: {exc}"]
        check(f"accepts {ns!r} for a lone robot, as {wire!r} on the wire", got == [wire], str(got))
    check("the rig does not count: a lone robot with the task takes a namespace",
          serve_args.fleet_namespaces(["ainex"], "biped") == ["biped"]
          and serve_args.staging(["ainex"]).rig)
    check("without --ros-namespace each robot is under its own id",
          serve_args.fleet_namespaces(["so101", "myagv"], None) == ["so101", "myagv"])


def test_staging() -> None:
    print("staging:")
    cases = {
        "so101": ("so101", ("so101",), True),
        "ainex": ("ainex", ("ainex",), False),
        "myagv": (None, (), False),
        "myagv,ainex": ("ainex", ("ainex",), False),
        "ainex,so101": ("so101", ("so101", "ainex"), True),
        "myagv,ainex,so101": ("so101", ("so101", "ainex"), True),
    }
    for robots, (lead, worktop, reset) in cases.items():
        names = robots.split(",")
        plan = serve_args.staging(names)
        check(f"{robots}: task {'in front of ' + lead if lead else 'not staged'}, "
              f"rig {'served' if lead else 'not served'}, /reset "
              f"{'served' if reset else 'not served'}",
              plan.task_robot == lead and plan.worktop_robots == worktop
              and plan.rig == bool(lead) and plan.reset(names) == reset,
              f"{plan} reset={plan.reset(names)}")

    def no_worktop(_robot):
        raise SystemExit("no worktop")

    inst = placement.Instance("so101", "robot_0/", "so101")
    try:
        placement.stand_fleet([inst], task_robot="so101", find_worktop=no_worktop,
                              floor_spot=None, surface_map=None, task_objects=None)
        ok = False
    except SystemExit:
        ok = True
    check("a worktop robot in a scene with no worktop is a start-up error", ok)


def main() -> int:
    test_help()
    test_refusals()
    test_namespaces()
    test_staging()
    print("all checks passed" if not failures else f"FAILED: {len(failures)} check(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
