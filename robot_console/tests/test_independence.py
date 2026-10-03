"""The console is independent of the simulator (root spec §1.4, console spec): it derives
its facts from the pinned vendor sources and never reads, imports or cites the simulator or
the workspace's robot records -- not in its modules, profiles, page or launchers."""

import ast
import os

import pytest

import robot_console

PKG = os.path.dirname(os.path.abspath(robot_console.__file__))
ROOT = os.path.dirname(os.path.dirname(PKG))          # robot_console/
WORDS = ("robots_specs/", "simulator/", "model.xml")


def console_files():
    out = []
    for base, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        out += [os.path.join(base, f) for f in files
                if f.endswith((".py", ".yaml", ".js", ".html", ".css"))]
    out += [os.path.join(ROOT, f) for f in os.listdir(ROOT) if f.endswith(".sh")]
    return sorted(out)


def test_the_console_has_files():
    names = {os.path.basename(f) for f in console_files()}
    assert {"teleop_core.py", "teleop.py", "fleet.py", "view.py", "teleop.sh"} <= names


@pytest.mark.parametrize("path", console_files(), ids=lambda p: os.path.relpath(p, ROOT))
def test_no_file_cites_the_simulator_or_the_robot_records(path):
    text = open(path, encoding="utf-8").read()
    for word in WORDS:
        assert word not in text, f"{os.path.relpath(path, ROOT)} mentions {word}"


@pytest.mark.parametrize("path", [f for f in console_files() if f.endswith(".py")],
                         ids=lambda p: os.path.relpath(p, ROOT))
def test_modules_import_nothing_of_the_simulator(path):
    """Only the standard library, third-party packages and the console's own modules."""
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    simulator_modules = {"protocol", "registry", "simulation", "spawn", "wirecheck", "motion",
                         "rosbridge_client", "common", "world", "placement", "sensors"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name.split(".")[0] not in simulator_modules, (path, name)
