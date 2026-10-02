"""Shared fixtures of the simulator's checks.

Run with an engine venv's python (it has MuJoCo, NumPy, PyYAML):

    simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/unit simulator/tests/integration
    simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/e2e     # Docker; slow

The end-to-end suite starts real simulations (`kitchen.sh start`) on their own control
ports and real spawns (`spawn.sh`) on their own wire ports, so it can run beside a
simulation someone else is using on the default ports.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SIM = TESTS.parent
SHARED = SIM / "shared"
REPO = SIM.parent
for p in (str(SHARED), str(SHARED / "wire"), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


def port_free(port: int) -> bool:
    """Nothing listens on the port (TIME_WAIT leftovers do not count)."""
    for host in ("0.0.0.0", "127.0.0.1"):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        finally:
            s.close()
    return True


def wait_port_free(port: int, timeout: float = 20.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if port_free(port):
            return True
        time.sleep(0.2)
    return False


class Sim:
    """A running `kitchen.sh start` simulation."""

    def __init__(self, engine: str, scene: str, port: int, log: Path):
        self.engine, self.scene, self.port, self.log = engine, scene, port, log
        self.proc = None

    def start(self, timeout: float = 240.0):
        self.logf = open(self.log, "w")
        self.proc = subprocess.Popen(
            [str(SIM / "kitchen.sh"), "start", "--engine", self.engine, "--scene", self.scene,
             "--sim-port", str(self.port)], stdout=self.logf, stderr=subprocess.STDOUT,
            start_new_session=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"simulation exited: {self.log.read_text()[-2000:]}")
            if "simulation ready" in self.log.read_text():
                return self
            time.sleep(0.5)
        raise RuntimeError("simulation did not become ready")

    def client(self):
        import protocol

        return protocol.Client("127.0.0.1", self.port)

    def stop(self, sig=signal.SIGINT) -> int:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, sig)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(10)
        wait_port_free(self.port)
        return self.proc.returncode if self.proc else 0


class Spawn:
    """A running `spawn.sh` process."""

    def __init__(self, robot, sim_port, port, placement=None, arm_port=None, log=None):
        self.robot, self.sim_port, self.port = robot, sim_port, port
        self.arm_port = arm_port
        args = [str(SIM / "spawn.sh"), robot, "--sim-port", str(sim_port), "--port", str(port)]
        if placement:
            args += ["--placement", placement]
        if arm_port:
            args += ["--arm-port", str(arm_port)]
        self.log = log
        self.logf = open(log, "w")
        self.proc = subprocess.Popen(args, stdout=self.logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def text(self) -> str:
        return self.log.read_text()

    def wait_ready(self, timeout: float = 900.0) -> str:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            txt = self.text()
            for line in txt.splitlines():
                if line.startswith("spawn ready:"):
                    return line
            if self.proc.poll() is not None:
                raise RuntimeError(f"spawn exited {self.proc.returncode}: {txt[-3000:]}")
            time.sleep(0.5)
        raise RuntimeError("spawn did not become ready: " + self.text()[-3000:])

    def wait_exit(self, timeout: float = 60.0) -> int:
        return self.proc.wait(timeout)

    def stop(self, sig=signal.SIGINT, timeout: float = 60.0) -> int:
        if self.proc.poll() is None:
            os.kill(self.proc.pid, sig)
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            return self.proc.wait(10)


def docker_containers(label_filter: str) -> list:
    out = subprocess.run(["docker", "ps", "-a", "--filter", label_filter, "--format",
                          "{{.Names}}"], stdout=subprocess.PIPE, text=True)
    return [x for x in out.stdout.split() if x]


def wait_no_containers(label_filter: str, timeout: float = 30.0) -> list:
    t0 = time.monotonic()
    left = docker_containers(label_filter)
    while left and time.monotonic() - t0 < timeout:
        time.sleep(0.5)
        left = docker_containers(label_filter)
    return left


@pytest.fixture(scope="session")
def logdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("rsim-logs")


@contextlib.contextmanager
def running_sim(engine, scene, port, logdir):
    assert port_free(port), f"sim port {port} is taken"
    sim = Sim(engine, scene, port, logdir / f"sim-{engine}-{port}.log").start()
    try:
        yield sim
    finally:
        sim.stop()


def engine_ready(engine: str) -> bool:
    return (SIM / engine / ".venv" / "bin" / "python").exists()


def docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=20).returncode == 0
    except Exception:
        return False
