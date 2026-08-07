"""`slam.sh navigate` -- load a map, click where the robot should go, and it drives there.

    python -m robot_console.navigate --map runs/house

Left click sets a goal and plans an A* path to it; right click or Space cancels and stops.
The robot localizes itself into the loaded map by scan matching, so it does not need to
be put back exactly where the mapping run started.

The live map keeps updating while navigating -- furniture that has moved since the map
was built gets planned around -- but the saved map is left alone unless you press M. A
map is usually the product of a whole exploration run, and one navigation pass through a
crowded room should not be able to overwrite it.
"""

from __future__ import annotations

from typing import Optional, Sequence

from robot_console.slam.cli import main as _main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _main(argv, mode="navigate")


if __name__ == "__main__":
    raise SystemExit(main())
