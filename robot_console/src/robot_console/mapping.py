"""`slam.sh map` -- teleoperation with the map building live beside the camera.

    python -m robot_console.mapping --out runs/house

The same keymap as `bin/teleop.sh`, plus a map window that fills in as you drive. M saves
without quitting; the map is also saved on the way out. Useful where autonomous
exploration is not what you want -- checking a specific room, or steering around
something the planner would rather not deal with.
"""

from __future__ import annotations

from typing import Optional, Sequence

from robot_console.slam.cli import main as _main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _main(argv, mode="map")


if __name__ == "__main__":
    raise SystemExit(main())
