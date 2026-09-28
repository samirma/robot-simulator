#!/usr/bin/env python
"""The myAGV + myCobot 280 attaches to a RoboCasa kitchen, compiles and moves (spec §5).

    python robots/myagv_mycobot280/test_attach.py

RoboCasa grafts every robot through `tools/spawn_robot.py`'s one adapter, so there is no
per-robot code here: the checks are `shared/tests/attach_check.py`'s, the same for every
robot on both engines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared" / "tests"))

import attach_check  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(attach_check.run("robocasa", "myagv_mycobot280"))
