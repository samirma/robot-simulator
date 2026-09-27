#!/usr/bin/env python
"""Every simulated robot stands at its placement in a RoboCasa kitchen (spec §5).

Each robot alone and the full fleet, on the floor or the worktop, without
interpenetration; the checks are `shared/tests/placement_check.py`, the same for both
engines.

    python tools/test_placement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "tests"))

import placement_check  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(placement_check.run("robocasa"))
