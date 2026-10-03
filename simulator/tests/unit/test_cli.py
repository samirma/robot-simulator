"""Entry-point argument handling: help, unknown flags, unknown ids, refusals that need no
running simulation."""

import os
import shutil
import signal
import socket
import subprocess
import threading
import time

import pytest

from conftest import SIM, port_free

import registry


def run(*args, timeout=120, env=None):
    return subprocess.run([str(a) for a in args], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout, env=env)


def test_spawn_help_lists_every_id_with_its_name_and_no_other():
    out = run(SIM / "spawn.sh", "--help")
    assert out.returncode == 0
    for r in registry.robots():
        assert r.id in out.stdout and r.name in out.stdout
    listed = out.stdout.split("accepted robot ids", 1)[1].splitlines()[1:]
    assert {line.split()[0] for line in listed if line.strip()} == set(registry.ids())
    for flag in ("--placement", "--sim-port", "--port"):
        assert flag in out.stdout
    # one wire per robot: no assembly, no --arm-port (spec §2.3, amended 2026-10-02)
    for word in ("--arm-port", "composite", "myagv_mycobot280"):
        assert word not in out.stdout, word


def test_spawn_refuses_unknown_id_with_the_list():
    out = run(SIM / "spawn.sh", "robonaut")
    assert out.returncode != 0 and "unknown robot id 'robonaut'" in out.stdout
    for r in registry.robots():
        assert r.id in out.stdout and r.name in out.stdout


def test_spawn_refuses_unknown_and_abbreviated_flags():
    assert run(SIM / "spawn.sh", "myagv", "--namespace", "x").returncode != 0
    assert run(SIM / "spawn.sh", "myagv", "--place", "floor").returncode != 0


def test_spawn_refuses_arm_port_as_unknown_flag():
    out = run(SIM / "spawn.sh", "so101", "--arm-port", "9999")
    assert out.returncode != 0 and "unrecognized arguments: --arm-port" in out.stdout


def test_spawn_refuses_when_no_simulation_runs():
    port = 9179
    assert port_free(port)
    out = run(SIM / "spawn.sh", "myagv", "--sim-port", port, "--port", 9178)
    assert out.returncode != 0 and "no simulation is running" in out.stdout


def test_spawn_refuses_a_taken_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 9177))
    s.listen(1)
    try:
        out = run(SIM / "spawn.sh", "myagv", "--port", 9177, "--sim-port", 9176)
        assert out.returncode != 0 and "port 9177 is already in use" in out.stdout
    finally:
        s.close()


def fake_docker(tmp_path, delay: float):
    """A `docker` first on PATH that answers everything with success, `docker info` only
    after `delay` seconds (marking that it was asked), so no test here needs Docker."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    mark = tmp_path / "docker-info-asked"
    exe = bindir / "docker"
    exe.write_text(f'#!/bin/sh\n[ "$1" = info ] && {{ : > "{mark}"; sleep {delay}; }}\nexit 0\n')
    exe.chmod(0o755)
    return dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}"), mark


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_spawn_ended_by_a_signal_during_its_checks_exits_zero(tmp_path, sig):
    """A spawn ended by SIGINT or SIGTERM exits 0 (spec §2.3), also while it is still
    asking Docker, before it reaches the simulation."""
    env, mark = fake_docker(tmp_path, delay=5)
    sim_port, port = 9171, 9170
    assert port_free(sim_port) and port_free(port)
    p = subprocess.Popen([str(SIM / "spawn.sh"), "so101", "--sim-port", str(sim_port),
                          "--port", str(port)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env)
    t0 = time.monotonic()
    while not mark.exists() and p.poll() is None and time.monotonic() - t0 < 60:
        time.sleep(0.05)
    if not mark.exists():
        p.kill()
        raise AssertionError("spawn.sh never asked Docker: " + p.communicate()[0])
    p.send_signal(sig)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out
    assert "Traceback" not in out, out


def test_spawn_refuses_a_sim_port_that_does_not_answer(tmp_path):
    """Something that is not a simulation on --sim-port (it accepts and hangs up) is a
    refusal with a message, not a traceback."""
    env, _ = fake_docker(tmp_path, delay=0)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    srv.settimeout(0.2)
    stop = threading.Event()

    def hang_up():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            conn.close()

    th = threading.Thread(target=hang_up, daemon=True)
    th.start()
    port = 9169
    assert port_free(port)
    try:
        out = run(SIM / "spawn.sh", "so101", "--sim-port", srv.getsockname()[1], "--port", port,
                  env=env)
    finally:
        stop.set()
        th.join()
        srv.close()
    assert out.returncode != 0 and "spawn.sh: refused" in out.stdout, out.stdout
    assert "Traceback" not in out.stdout, out.stdout


def test_spawn_help_and_unknown_id_need_no_engine_venv(tmp_path):
    """`--help` lists every id, and an unknown id is refused with the list, on a checkout
    where no engine is set up yet."""
    shutil.copy2(SIM / "spawn.sh", tmp_path / "spawn.sh")
    (tmp_path / "shared").symlink_to(SIM / "shared")
    out = run(tmp_path / "spawn.sh", "--help")
    assert out.returncode == 0, out.stdout
    for r in registry.robots():
        assert r.id in out.stdout and r.name in out.stdout
    out = run(tmp_path / "spawn.sh", "robonaut")
    assert out.returncode != 0 and "unknown robot id 'robonaut'" in out.stdout, out.stdout
    for r in registry.robots():
        assert r.id in out.stdout and r.name in out.stdout


def test_kitchen_help_and_flag_refusal():
    out = run(SIM / "kitchen.sh", "start", "--help")
    assert out.returncode == 0
    for word in ("--engine", "--scene", "--mujoco", "--sim-port", "ithor:", "robocasa:", "test:1"):
        assert word in out.stdout
    out = run(SIM / "kitchen.sh", "start", "--robots", "so101")
    assert out.returncode != 0 and "unknown flag" in out.stdout
    out = run(SIM / "kitchen.sh", "serve")
    assert out.returncode != 0


def test_kitchen_refuses_other_engines_source_and_out_of_range_ids():
    out = run(SIM / "kitchen.sh", "start", "--scene", "robocasa:1-1", "--sim-port", 9175)
    assert out.returncode != 0 and "robocasa" in out.stdout and "--engine robocasa" in out.stdout
    out = run(SIM / "kitchen.sh", "start", "--engine", "robocasa", "--scene", "robocasa:99-1",
              "--sim-port", 9175)
    assert out.returncode != 0 and "out of range 1-60" in out.stdout
    out = run(SIM / "kitchen.sh", "start", "--scene", "ithor:77", "--sim-port", 9175)
    assert out.returncode != 0 and "out of range" in out.stdout and "1-12, 201-212" in out.stdout


def test_kitchen_refuses_a_taken_sim_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 9174))
    s.listen(1)
    try:
        out = run(SIM / "kitchen.sh", "start", "--scene", "test:1", "--sim-port", 9174)
        assert out.returncode != 0 and "--sim-port 9174 is already in use" in out.stdout
    finally:
        s.close()


def test_run_sh_commands_and_refusals():
    for engine in ("molmospaces", "robocasa"):
        out = run(SIM / engine / "run.sh", "--help")
        assert out.returncode == 0
        for cmd in ("setup", "assets", "start", "repair", "--mujoco", "--sim-port", "--scene"):
            assert cmd in out.stdout
        assert run(SIM / engine / "run.sh", "frobnicate").returncode != 0
        assert run(SIM / engine / "run.sh", "start", "--bogus").returncode != 0
        assert run(SIM / engine / "run.sh", "repair").returncode == 0
