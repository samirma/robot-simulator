"""Entry-point argument handling: help, unknown flags, unknown ids, refusals that need no
running simulation."""

import socket
import subprocess

from conftest import SIM, port_free

import registry


def run(*args, timeout=120):
    return subprocess.run([str(a) for a in args], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout)


def test_spawn_help_lists_every_id_with_its_name_and_no_other():
    out = run(SIM / "spawn.sh", "--help")
    assert out.returncode == 0
    for r in registry.robots():
        assert r.id in out.stdout and r.name in out.stdout
    for flag in ("--placement", "--sim-port", "--port", "--arm-port"):
        assert flag in out.stdout


def test_spawn_refuses_unknown_id_with_the_list():
    out = run(SIM / "spawn.sh", "robonaut")
    assert out.returncode != 0 and "unknown robot id 'robonaut'" in out.stdout
    for r in registry.robots():
        assert r.id in out.stdout and r.name in out.stdout


def test_spawn_refuses_unknown_and_abbreviated_flags():
    assert run(SIM / "spawn.sh", "myagv", "--namespace", "x").returncode != 0
    assert run(SIM / "spawn.sh", "myagv", "--place", "floor").returncode != 0


def test_spawn_refuses_arm_port_without_an_arm_interface():
    out = run(SIM / "spawn.sh", "so101", "--arm-port", "9999")
    assert out.returncode != 0 and "no separate arm interface" in out.stdout


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
