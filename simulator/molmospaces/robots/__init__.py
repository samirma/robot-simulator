"""Out-of-tree robot definitions for the MolmoSpaces simulator.

Each subpackage supplies an MJCF plus the move groups, RobotView, Robot and
BaseRobotConfig subclasses MolmoSpaces needs. Because `BaseRobotConfig.robot_dir`
accepts an external directory, none of this requires forking molmospaces.

See molmospaces/docs/tutorials/add_robot.md.
"""

# Engine-neutral robot specs and the wire bridge live in `simulator/shared/`. Put it on
# the path when this adapter package is imported, so `robots_spec` and `contracts` resolve
# however the code was launched (run.sh, a standalone test script, or pytest).
import sys as _sys
from pathlib import Path as _Path

_shared = _Path(__file__).resolve().parents[2] / "shared"
if _shared.is_dir() and str(_shared) not in _sys.path:
    _sys.path.insert(0, str(_shared))
