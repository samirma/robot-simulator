"""One body and one wire per robot (spec §2.3, amended 2026-10-02): no assembly machinery is
left in the simulator's code, and a wire attaches to its robot by the spawn token alone."""

import re

import common
from conftest import SHARED

LEFTOVERS = ("RSIM_ROLE", "RSIM_TEST_FAIL_ROLE", "ARM_PREFIX", "mounting_transform",
             "registry.wires", "registry.components")


def code_files():
    return sorted(p for p in SHARED.rglob("*") if p.suffix in (".py", ".sh")
                  and "__pycache__" not in p.parts)


def test_no_assembly_machinery_is_left():
    found = []
    for path in code_files():
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            code = line.split("#", 1)[0]
            if any(word in code for word in LEFTOVERS) or re.search(r"composite", code, re.I):
                found.append(f"{path.relative_to(SHARED)}:{n}: {line.strip()}")
    assert found == [], "\n".join(found)


def test_a_wire_attaches_by_its_token_alone(monkeypatch):
    calls = []

    class Simulation:
        def __init__(self, *args, **kw):
            pass

        def call(self, op, **fields):
            calls.append((op, fields))
            return {"robot": "so101", "describe": {}}

    monkeypatch.setattr(common.protocol, "Client", Simulation)
    monkeypatch.setenv("RSIM_TOKEN", "t0k3n")
    link = common.SimLink()
    assert calls == [("wire", {"token": "t0k3n"})] and link.robot_id == "so101"
