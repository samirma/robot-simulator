"""view.sh refuses unknown robot ids before serving anything (console spec §1.1, §4; amended
2026-10-02: the former assembly id is an ordinary unknown id)."""

import os
import subprocess

import pytest

from robot_console import view

CONSOLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCEPTED = "Accepted ids: myagv, ainex, rosmaster_x3_plus, so101, mycobot280"


@pytest.mark.parametrize("rid", ["turtlebot", "myagv_mycobot280"])
def test_view_refuses_unknown_robot_ids(rid, capsys):
    assert view.main(["--robot", rid, "--no-open"]) == 2              # returns before binding a port
    err = capsys.readouterr().err
    assert f"unknown robot id '{rid}'" in err and ACCEPTED in err


@pytest.mark.parametrize("rid", ["myagv", "ainex", "rosmaster_x3_plus", "mycobot280"])
def test_view_refuses_a_namespace_where_the_hardware_documents_none(rid, capsys):
    """Console spec §1.1: only so101's hardware interface is namespace-configurable."""
    assert view.main(["--robot", rid, "--namespace", "robot1", "--no-open"]) == 2
    assert "no namespace override" in capsys.readouterr().err


@pytest.mark.parametrize("rid", ["turtlebot", "myagv_mycobot280"])
def test_view_launcher_refuses_unknown_robot_ids(rid):
    r = subprocess.run([os.path.join(CONSOLE, "view.sh"), "--robot", rid, "--no-open"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, r.stderr
    assert f"unknown robot id '{rid}'" in r.stderr and ACCEPTED in r.stderr
