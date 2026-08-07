"""`slam.sh explore` -- drive the robot around by itself until the map is complete.

    python -m robot_console.explore --out runs/house

Frontier exploration: repeatedly pick the boundary between mapped and unmapped space,
plan to it, drive there, and fold in what the lidar sees on the way. It terminates
because a map with no frontiers left is finished by definition.

Space pauses; the drive keys take over whenever you press one.
"""

from __future__ import annotations

from typing import Optional, Sequence

from robot_console.slam.cli import main as _main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _main(argv, mode="explore")


if __name__ == "__main__":
    raise SystemExit(main())
