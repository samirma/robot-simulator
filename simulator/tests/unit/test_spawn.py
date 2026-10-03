"""A failed spawn attempt cleans up only what it created (spec §2.3 Admission: a failed
attempt releases only its own reservations), with Docker and the simulation stubbed."""

import argparse
import subprocess
import threading

import pytest

import protocol
import spawn


class RefusingSimulation:
    """A simulation that refuses the `spawn` (the id is already another spawn's)."""

    def __init__(self):
        self.closed = threading.Event()
        self.calls = []

    def call(self, op, **fields):
        self.calls.append(op)
        if op == "spawn":
            raise protocol.RemoteError("refused: robot 'so101' is already spawned in this "
                                       "simulation (running); end that spawn first")
        return {}

    def close(self):
        self.closed.set()


def test_a_refused_attempt_leaves_the_running_spawns_container_alone(monkeypatch):
    """Two spawns of one id both pass their checks; the second is refused by the
    simulation and must not remove the first one's container, which has the same name."""
    docker_calls = []

    def docker(*args, **kw):
        docker_calls.append(list(args))
        return subprocess.CompletedProcess(["docker", *args], 0, "", "")

    monkeypatch.setattr(spawn, "docker", docker)
    monkeypatch.setattr(spawn, "docker_running", lambda: True)
    monkeypatch.setattr(spawn, "port_taken", lambda port: False)
    monkeypatch.setattr(spawn, "ensure_image", lambda robot: "rsim-wire/test:0")
    sp = spawn.Spawn(argparse.Namespace(robot="so101", placement="worktop", sim_port=9179,
                                        port=9178))
    sp.check()
    sp.client = RefusingSimulation()
    with pytest.raises(spawn.Refusal, match="already spawned"):
        sp.start()
    sp.cleanup()
    assert "spawn" in sp.client.calls
    name = "rsim-9179-so101-main"
    assert not [c for c in docker_calls if c[:1] in (["rm"], ["run"]) and name in c], docker_calls
