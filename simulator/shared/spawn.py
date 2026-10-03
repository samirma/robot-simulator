"""spawn.sh: add one robot to a running simulation and serve its vendor interface.

The only entry point that serves a wire. It asks the simulation on `--sim-port` to add the
robot (admission and placement happen there), starts the robot's one wire, a Docker
container (ROS graph + rosbridge_suite, the simulator code mounted read-only), waits until
it serves the recorded interface, prints one readiness line and stays in the foreground
watching the simulation, the container and the wire. Ending it removes the robot and its
container; the simulation also removes the robot if this process dies (even by SIGKILL),
and the container then exits by itself when its link to the simulation closes.

Exit status: 0 when ended by SIGINT/SIGTERM; non-zero for every refusal, failed startup,
wire failure, or when the simulation ends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol  # noqa: E402
import registry  # noqa: E402
from rosbridge_client import Rosbridge, WebSocketClosed, wait_for  # noqa: E402

SIM_ROOT = HERE.parent
REPO_ROOT = SIM_ROOT.parent
DOCKER_DIR = HERE / "wire" / "docker"
#: from the container's start to RSIM-WIRE-READY: a first use's overlay build, the graph (the
#: supervisor waits up to 150 s) and the first samples of the periodic outputs
READY_TIMEOUT = 300.0
READY_MARK = "RSIM-WIRE-READY"
#: a wire whose rosapi stays silent this long has stopped serving
WIRE_SILENCE = 30.0

EXIT_REFUSED = 2
EXIT_FAILED = 1


class Refusal(Exception):
    pass


def log(msg: str) -> None:
    print(f"spawn: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- arguments


def build_parser() -> argparse.ArgumentParser:
    try:
        ids = registry.listing("    ")
    except registry.RegistryError as exc:
        ids = f"    ({exc})"
    ap = argparse.ArgumentParser(
        prog="spawn.sh", allow_abbrev=False, formatter_class=argparse.RawTextHelpFormatter,
        description="Spawn one robot by id into the running simulation and serve its vendor "
                    "interface on its own rosbridge websocket. Runs in the foreground; "
                    "Ctrl-C removes the robot.",
        epilog="accepted robot ids (robots_specs/<id>.md):\n" + ids)
    ap.add_argument("robot", metavar="<id>", help="robot id (listed below)")
    ap.add_argument("--placement", choices=("worktop", "floor"), default="worktop",
                    help="stand the robot on the worktop (default) or on the floor")
    ap.add_argument("--sim-port", type=int, default=protocol.DEFAULT_SIM_PORT,
                    help="the simulation's control port (default %(default)s)")
    ap.add_argument("--port", type=int, default=9090,
                    help="rosbridge websocket port of the robot (default %(default)s)")
    return ap


def port_taken(port: int) -> bool:
    """Something listens on the port (connections left in TIME_WAIT do not count)."""
    for host in ("0.0.0.0", "127.0.0.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return True
        finally:
            s.close()
    return False


# ---------------------------------------------------------------- docker


def docker(*args, check=True, capture=True, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None, text=True)


def docker_running() -> bool:
    try:
        return docker("info", "--format", "{{.ServerVersion}}", check=False,
                      timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _hash_dir(path: Path, extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _interface(robot: registry.Robot) -> dict:
    """The robot's recorded interface file (PyYAML comes with an engine's venv)."""
    try:
        import yaml
    except ImportError:
        raise Refusal("PyYAML is not available to this python; run "
                      "simulator/<engine>/run.sh setup")
    with open(robot.path(robot.ros_file), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def distro_of(robot: registry.Robot) -> str:
    return _interface(robot)["ros_distribution"]


def ensure_image(robot: registry.Robot) -> str:
    """The robot's wire image, built (from pinned sources) if missing:
    the distribution's base image, and the robot's own layer when it has one."""
    distro = distro_of(robot)
    base_dir = DOCKER_DIR / distro
    if not (base_dir / "Dockerfile").is_file():
        raise Refusal(f"no wire image definition for ROS {distro} ({base_dir})")
    base_tag = f"rsim-wire/{distro}:{_hash_dir(base_dir)}"
    _build(base_tag, base_dir, {})
    rdir = DOCKER_DIR / robot.id
    if not (rdir / "Dockerfile").is_file():
        return base_tag
    tag = f"rsim-wire/{robot.id}:{_hash_dir(rdir, base_tag)}"
    if docker("image", "inspect", tag, check=False).returncode == 0:
        return tag
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rsim-ctx-") as tmp:
        ctx = Path(tmp) / "ctx"
        shutil.copytree(rdir, ctx)
        spec = ctx / "context.json"
        if spec.is_file():
            _vendor_files(json.loads(spec.read_text()), ctx)
        _build(tag, ctx, {"BASE": base_tag})
    return tag


def _vendor_files(spec: dict, ctx: Path) -> None:
    """Extract pinned vendor files into a build context from their verified archive (in
    the robots_specs fetch cache; fetched there first when missing)."""
    import zipfile

    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    for item in spec.get("extract", []):
        archive = cache / item["archive"]
        if not archive.is_file() and item.get("fetch"):
            subprocess.run(item["fetch"], cwd=str(REPO_ROOT), check=False)
        if not archive.is_file():
            raise Refusal(f"the pinned archive {archive.name} is not available (run "
                          "simulator/<engine>/run.sh setup, which fetches it)")
        h = hashlib.sha256()
        with open(archive, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != item["sha256"]:
            raise Refusal(f"{archive} does not match its pinned sha256; refusing to use it")
        dest = ctx / item["dest"]
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.startswith(item["prefix"]) and not name.endswith("/"):
                    out = dest / name[len(item["prefix"]):]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))


def _build(tag: str, context: Path, args: dict) -> None:
    if docker("image", "inspect", tag, check=False).returncode == 0:
        return
    log(f"building wire image {tag} (first use; this can take several minutes)")
    cmd = ["build", "-t", tag]
    for k, v in args.items():
        cmd += ["--build-arg", f"{k}={v}"]
    res = docker(*cmd, str(context), check=False, capture=True)
    if res.returncode != 0:
        tail = "\n".join((res.stderr or res.stdout or "").splitlines()[-25:])
        raise Refusal(f"building the wire image {tag} failed:\n{tail}")


# ---------------------------------------------------------------- the spawn


class Wire:
    """The robot's one wire: the container serving its interface on `port`."""

    def __init__(self, robot: registry.Robot, port: int, name: str):
        self.robot, self.port, self.name = robot, port, name
        self.image = None
        self.rb: Rosbridge | None = None
        self.required_nodes: list = []
        self.distro = None
        self.logfile: Path | None = None

    def describe(self) -> str:
        label = f"ROS {'1' if self.robot.dialect == 'ros1' else '2'} {self.distro}"
        return f"{self.robot.id} ws://127.0.0.1:{self.port} [{label}]"


def required_nodes(robot: registry.Robot) -> list:
    iface = _interface(robot)
    return sorted(n["name"] for n in iface.get("nodes", []) if not n.get("optional"))


class Spawn:
    def __init__(self, args):
        self.args = args
        self.robot = None
        self.client: protocol.Client | None = None
        self.wire: Wire | None = None
        self.done = threading.Event()
        self.exit_code = 0
        self.reason = ""
        self.cleaned = False
        #: the wire container is this attempt's to remove (set once the simulation admitted
        #: it, just before the container is created)
        self.container_started = False
        self.sim_ended = False

    # -------------------------------------------------------------- checks

    def check(self):
        a = self.args
        try:
            self.robot = registry.get(a.robot)
        except registry.RegistryError as exc:
            raise Refusal(str(exc))
        missing = registry.missing_files(self.robot)
        if missing:
            raise Refusal(f"{a.robot}: required files are missing "
                          f"({', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}); "
                          "run simulator/<engine>/run.sh setup")
        if not 1 <= a.port <= 65535:
            raise Refusal(f"port {a.port} is out of range")
        w = self.wire = Wire(self.robot, a.port, f"rsim-{a.sim_port}-{a.robot}-main")
        if not docker_running():
            raise Refusal("Docker is not running; start Docker Desktop and try again")
        if port_taken(w.port):
            raise Refusal(f"port {w.port} is already in use; pick another with --port")
        w.required_nodes = required_nodes(self.robot)
        w.distro = distro_of(self.robot)

    def connect(self):
        p = self.args.sim_port
        try:
            self.client = protocol.Client("127.0.0.1", p, on_event=self._event,
                                          on_close=self._sim_closed)
        except OSError:
            raise Refusal(f"no simulation is running on --sim-port {p}; "
                          "start one with simulator/kitchen.sh start")
        try:
            self.hello = self.client.call("hello")
            self.client.call("precheck", robot=self.robot.id, placement=self.args.placement,
                             ports=[self.wire.port])
        except (TimeoutError, OSError) as exc:   # OSError: ConnectionError included
            raise Refusal(f"no simulation is answering on --sim-port {p} ({exc}); "
                          "start one with simulator/kitchen.sh start")

    def _event(self, header, payload):
        if header.get("event") == "shutdown":
            self.sim_ended = True
            self.finish(EXIT_FAILED, f"the simulation ended ({header.get('reason', '')})")

    def _sim_closed(self):
        if not self.done.is_set():
            self.sim_ended = True
            self.finish(EXIT_FAILED, "the simulation ended (its control port closed)")

    def finish(self, code: int, reason: str):
        if self.done.is_set():
            return
        self.exit_code, self.reason = code, reason
        self.done.set()

    # -------------------------------------------------------------- startup

    def start(self):
        w = self.wire
        w.image = ensure_image(self.robot)
        # Ended (a signal, the simulation gone) during a possibly long build: nothing was
        # admitted yet, so nothing is asked of the simulation.
        if self.done.is_set():
            raise Refusal(self.reason)
        # Re-check the port after a possibly long build.
        if port_taken(w.port):
            raise Refusal(f"port {w.port} is already in use")
        try:
            res = self.client.call("spawn", robot=self.robot.id, placement=self.args.placement,
                                   ports=[w.port], timeout=600)
        except protocol.RemoteError as exc:
            raise Refusal(str(exc))
        self.spawned = res
        log(f"{self.robot.id} placed on the {res['placement']} at "
            f"({res['xyz'][0]:.3f}, {res['xyz'][1]:.3f}, {res['xyz'][2]:.3f}), heading "
            f"{res['yaw'] * 57.29578:.1f} deg")
        if res.get("staged"):
            cleared = res.get("cleared") or []
            log(f"worktop objects staged: {', '.join(res['staged'])}; scene objects cleared "
                f"from the working area: {', '.join(cleared) if cleared else 'none'}")
        self._run_container(w, res["token"])
        self._wait_ready(w, time.monotonic() + READY_TIMEOUT)
        self.client.call("commit")

    def _run_container(self, w: Wire, token: str):
        # The simulation admitted this attempt, so a container under its name is a
        # leftover; from here on the container is this attempt's to remove (cleanup),
        # even one `docker run` creates but fails to start.
        docker("rm", "-f", w.name, check=False)
        self.container_started = True
        env = {"RSIM_SIM_HOST": "host.docker.internal", "RSIM_SIM_PORT": str(self.args.sim_port),
               "RSIM_TOKEN": token, "RSIM_WIRE_PORT": str(w.port), "RSIM_ROBOT": self.robot.id,
               "ROS_DOMAIN_ID": str(7 + (w.port % 90)),
               "ROS_LOCALHOST_ONLY": "1", "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
        if os.environ.get("RSIM_TEST_FAIL_WIRE"):   # test hook (e2e/test_lifecycle.py)
            env["RSIM_TEST_FAIL_WIRE"] = os.environ["RSIM_TEST_FAIL_WIRE"]
        cmd = ["run", "-d", "--rm", "--name", w.name,
               "--label", "rsim.wire=1", "--label", f"rsim.sim_port={self.args.sim_port}",
               "--label", f"rsim.robot={self.robot.id}",
               "--label", f"rsim.spawn_pid={os.getpid()}",
               # rosbridge listens inside on the port number it is published on: the ROS 1
               # server (autobahn) refuses a handshake whose Host header names another port
               "-p", f"{w.port}:{w.port}",
               # ROS 1's roslaunch/rosmaster crawl through every possible descriptor when
               # they fork; Docker's default of 1M open files makes that take minutes.
               "--ulimit", "nofile=1024:524288",
               "--add-host", "host.docker.internal:host-gateway",
               "-v", f"{SIM_ROOT}:/opt/rsim/simulator:ro",
               "-v", f"{REPO_ROOT / 'robots_specs'}:/opt/rsim/robots_specs:ro",
               # compiled output of the simulator's ROS packages, built once per revision
               # from the read-only mount (never a copy of the code in the image)
               "-v", "rsim-wire-build:/opt/rsim_build"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [w.image, "python3", "-u", "/opt/rsim/simulator/shared/wire/supervisor.py"]
        res = docker(*cmd, check=False)
        if res.returncode != 0:
            raise Refusal(f"could not start the wire container: {res.stderr.strip()}")
        # The container removes itself when it exits (--rm); keep its log for diagnostics.
        import tempfile

        w.logfile = Path(tempfile.gettempdir()) / f"{w.name}.log"
        w.logproc = subprocess.Popen(["docker", "logs", "-f", w.name],
                                     stdout=open(w.logfile, "w"), stderr=subprocess.STDOUT)

    def _container_running(self, name: str) -> bool:
        res = docker("inspect", "-f", "{{.State.Running}}", name, check=False)
        return res.returncode == 0 and res.stdout.strip() == "true"

    def _logs(self, w: Wire, n: int = 30) -> str:
        if w.logfile is not None and w.logfile.exists():
            time.sleep(0.3)
            lines = w.logfile.read_text(errors="replace").splitlines()
            return "\n".join(lines[-n:])
        res = docker("logs", "--tail", str(n), w.name, check=False)
        return (res.stdout or "") + (res.stderr or "")

    def _wait_ready(self, w: Wire, deadline: float):
        while time.monotonic() < deadline:
            if self.done.is_set():
                raise Refusal(self.reason)
            if not self._container_running(w.name):
                raise Refusal("the wire container exited during startup:\n" + self._logs(w))
            if READY_MARK in self._logs(w, 200):
                break
            time.sleep(0.5)
        else:
            raise Refusal(f"the wire did not become ready within {READY_TIMEOUT:.0f} s:\n"
                          + self._logs(w))
        # Confirm from outside, over rosbridge: every required node is on the graph.
        w.rb = wait_for("127.0.0.1", w.port, max(5.0, deadline - time.monotonic()))
        missing = self._missing_nodes(w)
        if missing:
            raise Refusal(f"the wire is up but lacks node(s) {', '.join(missing)}")

    def _missing_nodes(self, w: Wire) -> list:
        nodes = set(w.rb.call("/rosapi/nodes", timeout=10).get("nodes", []))
        return [n for n in w.required_nodes if n not in nodes]

    # -------------------------------------------------------------- foreground

    def watch(self):
        w = self.wire
        last_probe = 0.0
        unanswered_since = None
        while not self.done.wait(1.0):
            if not self._container_running(w.name):
                self.finish(EXIT_FAILED, self._wire_reason(w, "container exited"))
                return
            if time.monotonic() - last_probe < 2.0:
                continue
            last_probe = time.monotonic()
            try:
                if w.rb is None or w.rb.closed.is_set():
                    w.rb = Rosbridge("127.0.0.1", w.port, timeout=3.0)
                missing = self._missing_nodes(w)
                unanswered_since = None
            except TimeoutError:
                # rosapi answers one request at a time and a client may keep it busy;
                # only a silence of WIRE_SILENCE s is a failure
                now = time.monotonic()
                unanswered_since = unanswered_since or now
                if now - unanswered_since >= WIRE_SILENCE:
                    self.finish(EXIT_FAILED, "the wire stopped serving (rosapi silent for "
                                             f"{WIRE_SILENCE:.0f} s)")
                    return
                continue
            except (OSError, WebSocketClosed, RuntimeError) as exc:
                self.finish(EXIT_FAILED, self._wire_reason(
                    w, f"stopped serving (rosbridge: {exc})"))
                return
            if missing:
                self.finish(EXIT_FAILED, f"the wire lost node(s) {', '.join(missing)}")
                return

    def _wire_reason(self, w: Wire, default: str) -> str:
        """Why the wire ended: its own report (a lost node) when it gave one -- waiting
        briefly, since rosbridge goes down a moment before the report lands -- else
        `default`."""
        deadline = time.monotonic() + 3.0
        while True:
            why = [l for l in self._logs(w, 400).splitlines()
                   if l.startswith("[wire] lost node(s)")]
            if why:
                return f"the wire {why[-1][len('[wire] '):]}"
            if time.monotonic() > deadline:
                return f"the wire {default}"
            time.sleep(0.3)

    def cleanup(self):
        """Remove what this attempt created -- its robot and, once admitted, its wire
        container and port -- and nothing of another spawn's."""
        if self.cleaned:
            return
        self.cleaned = True
        w = self.wire
        if w is not None and w.rb is not None:
            try:
                w.rb.close()
            except Exception:
                pass
        if self.client is not None and not self.client.closed.is_set():
            try:
                self.client.call("remove", reason=self.reason or "its spawn ended", timeout=10)
            except Exception:
                pass
            self.client.close()
        if self.container_started:
            docker("rm", "-f", w.name, check=False)
            deadline = time.monotonic() + 15
            while port_taken(w.port) and time.monotonic() < deadline:
                time.sleep(0.2)


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    sp = Spawn(args)

    def on_signal(signum, frame):
        sp.finish(0, f"ended by {signal.Signals(signum).name}")

    # From the start: a spawn ended by SIGINT or SIGTERM exits with status zero, even
    # while it is still checking (a terminal's Ctrl-C also ends `docker info`, whose
    # failure then reads as Docker not running).
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    interrupted = lambda: sp.done.is_set() and sp.exit_code == 0
    try:
        sp.check()
        sp.connect()
    except (Refusal, protocol.RemoteError) as exc:
        if sp.client is not None:
            sp.client.close()
        if interrupted():
            log(f"{sp.reason} before startup")
            return 0
        print(f"spawn.sh: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if interrupted():
        sp.client.close()
        log(f"{sp.reason} before startup")
        return 0
    try:
        sp.start()
    except Exception as exc:   # every startup failure is reported and cleaned up
        if interrupted():
            log("interrupted during startup; cleaning up")
            sp.cleanup()
            return 0
        print(f"spawn.sh: {sp.robot.id} failed to start: {exc}", file=sys.stderr)
        sp.reason = f"startup failed: {exc}"
        sp.cleanup()
        return EXIT_FAILED
    if sp.done.is_set():
        sp.cleanup()
        print(f"spawn.sh: {sp.robot.id} ended: {sp.reason}", file=sys.stderr)
        return sp.exit_code
    print(f"spawn ready: {sp.robot.id} ({sp.robot.name}) in {sp.hello['engine']} "
          f"{sp.hello['scene']} on the {args.placement}; wire(s): "
          + sp.wire.describe(), flush=True)
    try:
        sp.watch()
    finally:
        if sp.exit_code != 0:
            print(f"spawn.sh: {sp.robot.id}: {sp.reason}; removing it", file=sys.stderr)
        sp.cleanup()
    log(f"{sp.robot.id} removed ({sp.reason})")
    return sp.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
