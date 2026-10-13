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
            raise protocol.RemoteError("refused: robot 'rebot_b601_rs' is already spawned in this "
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
    sp = spawn.Spawn(argparse.Namespace(robot="rebot_b601_rs", placement="worktop", sim_port=9179,
                                        port=9178))
    sp.check()
    sp.client = RefusingSimulation()
    with pytest.raises(spawn.Refusal, match="already spawned"):
        sp.start()
    sp.cleanup()
    assert "spawn" in sp.client.calls
    name = "rsim-9179-rebot_b601_rs-main"
    assert not [c for c in docker_calls if c[:1] in (["rm"], ["run"]) and name in c], docker_calls


# ------------------------------------------------- the foreground watch of a ready wire


class FakeClock:
    """spawn's `time`: it advances only when the watch waits or sleeps."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s


class ClockedDone(threading.Event):
    """`Spawn.done` whose waits advance the clock; the spawn is ended from outside (as by
    Ctrl-C) once `until` seconds have passed."""

    def __init__(self, clock, until):
        super().__init__()
        self.clock, self.until = clock, clock.now + until

    def wait(self, timeout=None):
        self.clock.now += timeout or 0
        if self.clock.now >= self.until:
            self.set()
        return self.is_set()


class Rosapi:
    """A wire's rosbridge whose `/rosapi/nodes` raises `error` for its first `failures` calls,
    then lists the robot's node."""

    def __init__(self, failures, error=None):
        self.closed = threading.Event()
        self.failures, self.calls = failures, 0
        self.error = error or RuntimeError("/rosapi/nodes failed: Service /rosapi/nodes does "
                                           "not exist")

    def call(self, service, timeout=10.0):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return {"nodes": ["/reBotArmController"]}


def watch(monkeypatch, rb, until):
    clock = FakeClock()
    monkeypatch.setattr(spawn, "time", clock)
    monkeypatch.setattr(spawn, "docker",
                        lambda *args, **kw: subprocess.CompletedProcess(args, 0, "", ""))
    sp = spawn.Spawn(argparse.Namespace(robot="rebot_b601_rs", placement="worktop", sim_port=9179,
                                        port=9178))
    sp.done = ClockedDone(clock, until)
    sp.wire = spawn.Wire(None, 9178, "rsim-9179-rebot_b601_rs-main")
    sp.wire.required_nodes = ["/reBotArmController"]
    sp.wire.rb = rb
    sp._container_running = lambda name: True
    start = clock.now
    sp.watch()
    return sp, clock.now - start


def test_a_passing_rosapi_error_does_not_end_the_spawn(monkeypatch):
    """rosbridge answers "Service /rosapi/nodes does not exist" when the client it makes for a
    call misses the server for 1 s; a few such answers are not a wire that stopped serving."""
    rb = Rosapi(failures=3)
    sp, _ = watch(monkeypatch, rb, until=60)
    assert (sp.exit_code, sp.reason) == (0, "")
    assert rb.calls > 3


def test_a_lasting_rosapi_error_ends_the_spawn(monkeypatch):
    sp, elapsed = watch(monkeypatch, Rosapi(failures=10**6), until=120)
    assert sp.exit_code == spawn.EXIT_FAILED
    assert "stopped serving (rosapi failing" in sp.reason and "does not exist" in sp.reason
    assert spawn.WIRE_SILENCE <= elapsed < spawn.WIRE_SILENCE + 5


def test_a_closed_rosbridge_ends_the_spawn_at_once(monkeypatch):
    rb = Rosapi(failures=10**6, error=spawn.WebSocketClosed("rosbridge connection closed"))
    sp, elapsed = watch(monkeypatch, rb, until=120)
    assert sp.exit_code == spawn.EXIT_FAILED
    assert "stopped serving (rosbridge: rosbridge connection closed)" in sp.reason
    assert elapsed < 10
