"""The shell launchers parse, and `run_task.sh --help` never installs anything."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = [ROOT / "run_task.sh", ROOT / "bin" / "teleop.sh", ROOT / "bin" / "slam.sh"]


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
    for flag in ("--instruction-file", "--episodes", "--label", "--policy", "kitchen.sh serve"):
        assert flag in result.stdout
    assert not (tmp_path / "never-light").exists()
    assert not (tmp_path / "never-vla").exists()


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
