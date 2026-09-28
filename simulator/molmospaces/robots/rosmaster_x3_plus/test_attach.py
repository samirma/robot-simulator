#!/usr/bin/env python
"""The ROSMASTER X3 PLUS attaches to a MolmoSpaces house, compiles and moves (spec §5).

    python robots/rosmaster_x3_plus/test_attach.py

The checks are `shared/tests/attach_check.py`'s, the same for every robot on both engines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared" / "tests"))

import attach_check  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(attach_check.run("molmospaces", "rosmaster_x3_plus"))
