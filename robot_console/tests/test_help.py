"""Help (console spec §4): the ids `teleop.sh --help` and `run_task.sh --help` list are the
ids in `robots_specs/robots.yml`, each with its name -- teleop's exactly the entries whose
kind is not `arm`, `run_task.sh`'s the `simulated` ones -- and an unknown id is refused
with the same list.

The console keeps its own copy of those ids and names (`robot_ids.py`) so it installs
with no workspace around it, and `robots.yml` is read here from the sibling checkout when
there is one; without it the comparison is skipped, and the workspace parity tests are
the other half of the check.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from robot_console import robot_ids
from robot_console.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
ROBOTS_YML = ROOT.parent / "robots_specs" / "robots.yml"


def robots_yml() -> list[dict[str, str]]:
    """Each entry's scalar `id`, `name`, `kind`, `placement` and `simulated`."""
    if not ROBOTS_YML.exists():
        pytest.skip("no sibling robots_specs/robots.yml")
    entries: list[dict[str, str]] = []
    for line in ROBOTS_YML.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"  - id: (\S+)", line):
            entries.append({"id": m.group(1)})
        elif entries and (m := re.match(r"    (name|kind|placement|simulated): (.+?)\s*(#.*)?$", line)):
            entries[-1][m.group(1)] = m.group(2)
    return entries


def listed(text: str) -> dict[str, str]:
    """`{id: name}` from the `  id  name` lines of a help text."""
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^  (\w+)  +(\S.*)$", text, re.M)
            if m.group(1) in robot_ids.ROBOT_NAMES}


def run_help(script: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          timeout=60, env=env)


@pytest.fixture
def stub_venv(tmp_path: Path) -> dict:
    """An environment whose `.venv` is this interpreter, so `bin/teleop.sh` starts without
    creating or installing anything."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    (venv / ".teleop-stamp").touch()
    env = dict(os.environ, ROBOT_CONSOLE_VENV=str(venv))
    return env


# ------------------------------------------------------------------ the copy is robots.yml


def test_the_console_copy_is_the_simulated_robots_ids_and_names() -> None:
    simulated = {e["id"]: e["name"] for e in robots_yml() if e.get("simulated") == "true"}
    assert dict(robot_ids.ROBOT_NAMES) == simulated
    assert list(robot_ids.ROBOT_NAMES) == list(simulated), "in the file's order"


def test_the_console_copy_of_the_kinds_is_robots_ymls() -> None:
    kinds = {e["id"]: e["kind"] for e in robots_yml() if e.get("simulated") == "true"}
    assert dict(robot_ids.ROBOT_KINDS) == kinds


# ------------------------------------------------------------------ teleop.sh --help


def test_teleop_help_lists_exactly_the_ids_that_are_not_arms(stub_venv) -> None:
    expected = {e["id"]: e["name"] for e in robots_yml()
                if e.get("simulated") == "true" and e["kind"] != "arm"}
    result = run_help(ROOT / "bin" / "teleop.sh", "--help", env=stub_venv)
    assert result.returncode == 0, result.stderr
    assert listed(result.stdout) == expected
    assert "so101" not in listed(result.stdout)


def test_teleop_parser_help_and_refusal_list_the_same_ids() -> None:
    expected = robot_ids.listing(robot_ids.TELEOP_IDS)
    assert expected in build_parser().format_help()
    with pytest.raises(SystemExit) as err:
        build_parser().parse_args(["--robot", "so101"])
    assert err.value.code == 2


def test_teleop_refuses_an_unknown_id_with_the_list(stub_venv, capsys) -> None:
    result = run_help(ROOT / "bin" / "teleop.sh", "--robot", "rover", env=stub_venv)
    assert result.returncode == 2
    assert "unknown robot id 'rover'" in result.stderr
    assert listed(result.stderr) == listed(run_help(
        ROOT / "bin" / "teleop.sh", "--help", env=stub_venv).stdout)


# ------------------------------------------------------------------ run_task.sh --help


def test_run_task_help_lists_the_simulated_ids(tmp_path) -> None:
    expected = {e["id"]: e["name"] for e in robots_yml() if e.get("simulated") == "true"}
    env = dict(os.environ, ROBOT_CONSOLE_VLA_VENV=str(tmp_path / "never-vla"))
    result = run_help(ROOT / "run_task.sh", "--help", env=env)
    assert result.returncode == 0, result.stderr
    assert listed(result.stdout) == expected
    assert not (tmp_path / "never-vla").exists()


def test_run_task_refuses_an_unknown_id_with_the_same_list(tmp_path) -> None:
    env = dict(os.environ, ROBOT_CONSOLE_VLA_VENV=str(tmp_path / "never-vla"))
    result = run_help(ROOT / "run_task.sh", "--robot", "rover", env=env)
    assert result.returncode == 1
    assert "unknown robot id 'rover'" in result.stderr
    assert listed(result.stderr) == dict(robot_ids.ROBOT_NAMES)
    assert not (tmp_path / "never-vla").exists()


# ------------------------------------------------------------------ the other id flags


def test_fleet_and_smoke_list_their_ids_too(capsys) -> None:
    from robot_console import fleet, smoke

    with pytest.raises(SystemExit):
        fleet.main(["--help"])
    assert listed(capsys.readouterr().out) == dict(robot_ids.ROBOT_NAMES)
    with pytest.raises(SystemExit):
        smoke.main(["--help"])
    from robot_console.robots import WHEELED_ROBOTS

    assert set(listed(capsys.readouterr().out)) == set(WHEELED_ROBOTS)
    with pytest.raises(SystemExit):
        fleet.main(["--expect", "rover"])
    assert listed(capsys.readouterr().err) == dict(robot_ids.ROBOT_NAMES)
