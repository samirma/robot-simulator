"""The simulator's own documents describe what the specification now requires: one body and
one wire per robot (spec §2.3, amended 2026-10-02), no assembly and no `--arm-port`."""

import pytest

from conftest import SIM

DOCS = [SIM / "README.md", SIM / "tests" / "README.md", SIM / "spawn.sh"]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(SIM)))
def test_no_composite_robot_is_documented(doc):
    text = doc.read_text().lower()
    for word in ("--arm-port", "myagv_mycobot280", "composite"):
        assert word not in text, f"{doc.name} mentions {word!r}"
