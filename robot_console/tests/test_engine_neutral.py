"""No client branch selects behaviour from an engine identity (console spec §4).

The console drives whatever serves the contract -- either simulator engine or the real
robot -- and must behave the same on each. So nothing in its code may even be able to
tell them apart: no engine name as a value, no test of anything called an engine, and no
reading of an engine name out of the environment. Comments, docstrings and messages to
the user may talk about engines; no condition may look at one. `run_task.sh --label` names the engine for the
report's directory only, and is a plain string passed through, never compared.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import robot_console

SRC = Path(robot_console.__file__).resolve().parent
CONSOLE = SRC.parents[1]
ENGINE_NAMES = re.compile(r"molmo_?spaces|robocasa|robosuite|mujoco", re.I)


def _names_in(node: ast.AST) -> set[str]:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            names.add(sub.value)
    return names


def offences(source: str) -> list[str]:
    """Every place `source` could select behaviour from an engine identity."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _names_in(node.func) & {"getenv", "get", "environ"} \
                and any(re.search(r"engine", n, re.I) or ENGINE_NAMES.search(n)
                        for arg in node.args for n in _names_in(arg)):
            found.append(f"line {node.lineno}: reads an engine identity from the environment")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            if any(ENGINE_NAMES.search(m) for m in modules):
                found.append(f"line {node.lineno}: imports an engine")
        tests = []
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            tests.append(node.test)
        elif isinstance(node, ast.Compare):
            tests.append(node)
        elif isinstance(node, ast.Match):
            tests.append(node.subject)
            tests += [case.pattern for case in node.cases]
        for test in tests:
            hit = [n for n in _names_in(test) if re.search(r"engine", n, re.I)
                   or ENGINE_NAMES.search(n)]
            if hit:
                found.append(f"line {test.lineno}: a condition on {', '.join(sorted(hit))}")
    return found


def test_no_source_file_selects_behaviour_from_an_engine() -> None:
    bad = {}
    for path in sorted(SRC.rglob("*.py")):
        problems = offences(path.read_text(encoding="utf-8"))
        if problems:
            bad[str(path.relative_to(SRC))] = problems
    assert bad == {}


def test_the_page_and_the_launchers_name_no_engine_outside_comments() -> None:
    page = (CONSOLE / "live_cameras.html").read_text(encoding="utf-8")
    script = re.sub(r"//[^\n]*|/\*.*?\*/|<!--.*?-->", "", page, flags=re.S)
    assert not ENGINE_NAMES.search(script)
    for launcher in (*sorted((CONSOLE / "bin").glob("*.sh")), CONSOLE / "run_task.sh"):
        code = "\n".join(line.split("#", 1)[0] for line in
                         launcher.read_text(encoding="utf-8").splitlines())
        assert not ENGINE_NAMES.search(code), launcher.name


def test_the_detector_catches_what_it_is_for() -> None:
    """The rule itself, on code that breaks it and code that does not."""
    assert offences('if engine == "robocasa":\n    pass\n')
    assert offences('x = 1 if os.environ.get("ENGINE") else 2\n')
    assert offences('name = os.getenv("SIM_ENGINE")\n')
    assert offences('import mujoco\n')
    assert offences('match wire.engine:\n    case "a": pass\n')
    assert not offences('"""Works on both engines, MolmoSpaces and RoboCasa."""\n'
                        '# robocasa renders darker\nprint("start one: kitchen.sh serve")\n')
