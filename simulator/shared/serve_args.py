#!/usr/bin/env python
"""What a served fleet is made of: its namespaces and what gets staged around it.

Spec §2.3 in code, for the parts of it that need no scene: which ROS namespace each robot
is under, which namespaces are refused, and -- from the robots' `placement` in
`robots_specs/robots.yml` -- whether the task is staged, whether the rig is served and
whether `/reset` is. Both engines' spawn tools and `kitchen.sh` ask here, so there is one
answer to each of those questions rather than one per caller.

Stdlib plus `robots_spec` only, so `kitchen.sh` can ask before an engine has spent a
minute compiling a kitchen:

    python serve_args.py check --robots so101,myagv [--ros-namespace NS]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import robots_spec  # noqa: E402

#: First namespace tokens another provider on the port already owns (spec §3): the rig's
#: `/scene`, the discovery services' `/rosapi` and their node, the bridge's own node for
#: clients' advertisements, and `/simulator`, the node `/reset` belongs to. A lone robot
#: put under one of these would share names with that provider.
RESERVED_NAMESPACES: dict[str, str] = {
    "scene": "the worktop's camera rig (/scene/*)",
    "rosapi": "the discovery services (/rosapi/*) and their node",
    "rosbridge_websocket": "the bridge's node, which owns clients' advertisements",
    "simulator": "the node that provides /reset",
}

#: One namespace token: what both ROS 1 graph names and ROS 2 names accept -- a letter,
#: then letters, digits and underscores. ROS 2 also refuses a doubled underscore.
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def validate_namespace(namespace: str) -> str:
    """The namespace as it composes onto the wire (no slashes at either end), or ValueError.

    `''` is the bare vendor interface and is valid. A leading `/` is accepted and dropped,
    since every namespace here is absolute.
    """
    if namespace == "" or namespace == "/":
        return ""
    body = namespace[1:] if namespace.startswith("/") else namespace
    if body.endswith("/"):
        raise ValueError(f"--ros-namespace {namespace!r}: a namespace does not end in '/'")
    tokens = body.split("/")
    for token in tokens:
        if not token:
            raise ValueError(f"--ros-namespace {namespace!r}: empty name token ('//')")
        if not _TOKEN.fullmatch(token):
            raise ValueError(
                f"--ros-namespace {namespace!r}: {token!r} is not a valid ROS name token "
                "(a letter, then letters, digits and underscores)")
        if "__" in token:
            raise ValueError(f"--ros-namespace {namespace!r}: ROS 2 names may not contain "
                             "'__'")
    if tokens[0] in RESERVED_NAMESPACES:
        raise ValueError(
            f"--ros-namespace {namespace!r} collides with {RESERVED_NAMESPACES[tokens[0]]}")
    return body


def fleet_namespaces(names: list[str], ros_namespace: str | None) -> list[str]:
    """Each robot's namespace: its own id, or the one `--ros-namespace` names for a lone robot.

    The rig is not a robot and does not count towards "more than one".
    """
    repeated = sorted({n for n in names if list(names).count(n) > 1})
    if repeated:
        raise ValueError(f"--robots names {', '.join(repeated)} more than once; each robot "
                         "is served once, under its own id")
    if ros_namespace is None:
        return list(names)
    if len(names) != 1:
        raise ValueError(
            f"--ros-namespace applies to a lone robot; --robots names {len(names)} "
            f"({','.join(names)}), each of which is under its own id")
    return [validate_namespace(ros_namespace)]


@dataclass(frozen=True)
class Staging:
    """What spec §2.3 stages around a set of robots."""

    #: The robot the task is staged in front of, whose base frame is the task frame:
    #: the SO-101 when it is served, otherwise the first worktop robot. None: no task.
    task_robot: str | None
    #: Every robot standing on the worktop, `task_robot` first.
    worktop_robots: tuple[str, ...]

    @property
    def task(self) -> bool:
        return self.task_robot is not None

    @property
    def rig(self) -> bool:
        """The rig watches the staged task, so it is served exactly when the task is."""
        return self.task

    def reset(self, names) -> bool:
        """`/reset` is served whenever the SO-101 is, and only then."""
        return "so101" in names


def staging(names) -> Staging:
    worktop = [n for n in names if robots_spec.placement(n) == "worktop"]
    if not worktop:
        return Staging(None, ())
    first = "so101" if "so101" in worktop else worktop[0]
    return Staging(first, (first, *[n for n in worktop if n != first]))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="serve_args.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="refuse what kitchen.sh serve must refuse")
    check.add_argument("--robots", required=True)
    check.add_argument("--ros-namespace", default=None, dest="ros_namespace")
    args = ap.parse_args(argv)
    try:
        names = robots_spec.check_simulated(args.robots)
        fleet_namespaces(names, args.ros_namespace)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
