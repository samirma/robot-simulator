"""The robot ids every `--robot` flag takes, each with its name.

`robots_specs/robots.yml` is the workspace's one record of the robots (workspace spec §2);
the console keeps this copy of the `simulated` entries' ids and names so it installs with
no workspace around it, and the workspace parity tests hold the copy equal to that file.
Every `--robot` flag's `--help` lists the ids it accepts, each with its name, and the same
list is in the message that refuses an unknown id -- both built from `ROBOT_NAMES` here,
never typed into a help string.

Standard library only: `run_task.sh --help` runs this before any venv exists.
"""

from __future__ import annotations

import sys
from typing import Iterable, Mapping, Sequence

#: `id -> name` of every `simulated` robot in `robots_specs/robots.yml`, in the file's
#: order.
ROBOT_NAMES: Mapping[str, str] = {
    "myagv": "myAGV",
    "so101": "SO-101",
    "ainex": "AiNex",
    "myagv_mycobot280": "myAGV + myCobot 280",
    "rosmaster_x3_plus": "ROSMASTER X3 PLUS",
}

#: The `kind` of each, as `robots.yml` records it. Teleop drives every kind but `arm`.
ROBOT_KINDS: Mapping[str, str] = {
    "myagv": "mobile_base",
    "so101": "arm",
    "ainex": "humanoid",
    "myagv_mycobot280": "mobile_manipulator",
    "rosmaster_x3_plus": "mobile_manipulator",
}

ROBOT_IDS = tuple(ROBOT_NAMES)

#: Ids teleop accepts: every simulated robot whose kind is not `arm`.
TELEOP_IDS = tuple(rid for rid in ROBOT_IDS if ROBOT_KINDS[rid] != "arm")


def listing(ids: Iterable[str] = ROBOT_IDS, indent: str = "  ") -> str:
    """One `id  name` line per id, for a `--help` text or an error message."""
    ids = tuple(ids)
    width = max((len(i) for i in ids), default=0)
    return "\n".join(f"{indent}{rid:<{width}}  {ROBOT_NAMES[rid]}" for rid in ids)


def accepted(ids: Iterable[str] = ROBOT_IDS) -> str:
    return "accepted ids:\n" + listing(ids)


def refusal(value: str, ids: Sequence[str] = ROBOT_IDS) -> str:
    """The message that refuses an unknown id, listing the accepted ones."""
    return f"unknown robot id {value!r}; {accepted(ids)}"


if __name__ == "__main__":
    # For shell callers: `python3 -m robot_console.robot_ids [teleop]` prints the listing;
    # `python3 -m robot_console.robot_ids check <id>` exits 1 with the refusal for an
    # unknown id.
    args = sys.argv[1:]
    if args[:1] == ["check"]:
        if args[1] not in ROBOT_IDS:
            print(f"error: {refusal(args[1])}", file=sys.stderr)
            sys.exit(1)
    else:
        print(listing(TELEOP_IDS if args[:1] == ["teleop"] else ROBOT_IDS))
