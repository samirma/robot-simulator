"""Self-installing launchers and an installed console without any sibling source tree
(console spec §3, §4)."""

import os
import shutil
import subprocess
import sys

import pytest

from robot_console.profiles import load
from wirespec import wire_spec

CONSOLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(shutil.which("uv") is None and shutil.which("python3") is None,
                                reason="needs uv or python3 to build a venv")


def copy_console(dst):
    shutil.copytree(CONSOLE, dst, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache",
                                                                "tests"))
    return dst


def run(cmd, cwd="/", env=None, timeout=600):
    e = dict(os.environ, ROBOT_CONSOLE_EXTRAS="", SDL_VIDEODRIVER="dummy")
    e.pop("VIRTUAL_ENV", None)
    e.update(env or {})
    return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)


def test_launchers_self_install_and_reinstall_on_pyproject_change(tmp_path, fake):
    root = copy_console(str(tmp_path / "robot_console"))      # no simulator/ or robots_specs/ beside it
    r = run([f"{root}/python.sh", "-c", "import robot_console.profiles as p; print(p.__file__)"])
    assert r.returncode == 0, r.stderr
    assert "installing venv" in r.stderr and r.stdout.strip().startswith(root)
    r = run([f"{root}/teleop.sh", "--help"])
    assert r.returncode == 0 and "installing venv" not in r.stderr and "Teleoperable ids" in r.stdout
    r = run([f"{root}/view.sh", "--help"])
    assert r.returncode == 0 and "installing venv" not in r.stderr
    with open(f"{root}/pyproject.toml", "a") as f:
        f.write("\n# changed\n")
    r = run([f"{root}/python.sh", "-c", "print('ok')"])
    assert r.returncode == 0 and "installing venv" in r.stderr, r.stderr
    # The installed console checks a wire with its packaged profiles, from anywhere.
    srv = fake(wire_spec(load("myagv")))
    r = run([f"{root}/python.sh", "-m", "robot_console.fleet", "--url", srv.url, "--expect", "myagv"])
    assert r.returncode == 0 and r.stdout.rstrip().endswith("PASS"), r.stdout + r.stderr


def test_wheel_install_runs_without_the_source_tree(tmp_path, fake):
    if shutil.which("uv") is None:
        pytest.skip("needs uv")
    src = copy_console(str(tmp_path / "src" / "robot_console"))
    r = run(["uv", "build", "--wheel", "-q", "-o", str(tmp_path / "dist"), src])
    assert r.returncode == 0, r.stderr
    wheel = next((tmp_path / "dist").glob("*.whl"))
    venv = tmp_path / "venv"
    assert run(["uv", "venv", "-q", "--python", sys.executable, str(venv)]).returncode == 0
    r = run(["uv", "pip", "install", "-q", "--python", str(venv / "bin" / "python"), str(wheel)])
    assert r.returncode == 0, r.stderr
    shutil.rmtree(tmp_path / "src")                            # no source tree at all
    srv = fake(wire_spec(load("so101")))
    r = run([str(venv / "bin" / "python"), "-m", "robot_console.fleet", "--url", srv.url])
    assert r.returncode == 0 and "robot so101: typed validation PASS" in r.stdout, r.stdout + r.stderr
    r = run([str(venv / "bin" / "python"), "-c",
             "from robot_console.view import make_server; s = make_server('ws://127.0.0.1:1', None, None);"
             "import urllib.request, threading; threading.Thread(target=s.serve_forever, daemon=True).start();"
             "print(len(urllib.request.urlopen(f'http://127.0.0.1:{s.server_address[1]}/console.js').read()))"])
    assert r.returncode == 0 and int(r.stdout) > 1000, r.stderr
