"""The shell launchers parse, and `run_task.sh --help` never installs anything."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = [ROOT / "run_task.sh", ROOT / "bin" / "teleop.sh", ROOT / "bin" / "slam.sh",
             ROOT / "bin" / "view.sh"]


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_parses(script: Path) -> None:
    assert script.exists(), script
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_shellchecks(script: Path) -> None:
    subprocess.run(["shellcheck", "-S", "warning", str(script)], check=True)


def test_run_task_help_needs_no_venv_and_installs_nothing(tmp_path: Path) -> None:
    """--help is parsed before any bootstrap: pointing both venv paths at directories that
    do not exist, the help text must come back and neither directory may appear."""
    env = dict(os.environ)
    env["ROBOT_CONSOLE_VENV"] = str(tmp_path / "never-light")
    env["ROBOT_CONSOLE_VLA_VENV"] = str(tmp_path / "never-vla")
    result = subprocess.run(
        ["bash", str(ROOT / "run_task.sh"), "--help"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--episodes", "--label", "--url", "--robot", "--namespace",
                 "--instruction", "--instruction-file", "kitchen.sh serve"):
        assert flag in result.stdout
    assert not (tmp_path / "never-light").exists()
    assert not (tmp_path / "never-vla").exists()


def test_run_task_offers_only_the_specified_flags(tmp_path: Path) -> None:
    """The synopsis is the spec's; the retired flags are unknown now, not ignored."""
    env = dict(os.environ)
    env["ROBOT_CONSOLE_VLA_VENV"] = str(tmp_path / "never-vla")
    for retired in ("--policy", "--steps", "--log-dir", "--wait", "--reinstall", "--robots"):
        result = subprocess.run(
            ["bash", str(ROOT / "run_task.sh"), retired, "x"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1 and "unknown flag" in result.stderr, retired
    assert not (tmp_path / "never-vla").exists()


def test_run_task_refuses_an_unknown_robot_id_listing_the_known_ones(tmp_path: Path) -> None:
    """--robot takes one robot id from the console's parity-held copy of robots.yml; an
    unknown one (or a list) is refused before anything touches the wire. The VLA venv is stood in by
    one whose python is this test's own, so nothing is installed."""
    import sys

    from robot_console.fleet import ROBOT_IDS

    venv = tmp_path / "venv-vla"
    (venv / "bin").mkdir(parents=True)
    for name, body in (("python", f'exec "{sys.executable}" "$@"'), ("inspect-robot", "exit 0")):
        script = venv / "bin" / name
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
    (venv / ".run-task-stamp").touch()
    env = dict(os.environ, ROBOT_CONSOLE_VLA_VENV=str(venv))
    for value in ("rover", "so101,myagv"):
        result = subprocess.run(
            ["bash", str(ROOT / "run_task.sh"), "--robot", value],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 1, result.stderr
        assert f"unknown robot id '{value}'" in result.stderr
        assert all(rid in result.stderr for rid in ROBOT_IDS)


def test_run_task_rejects_unknown_flags_before_installing(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["ROBOT_CONSOLE_VLA_VENV"] = str(tmp_path / "never-vla")
    result = subprocess.run(
        ["bash", str(ROOT / "run_task.sh"), "--bogus"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "unknown flag" in result.stderr
    assert not (tmp_path / "never-vla").exists()
