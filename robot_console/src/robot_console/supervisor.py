"""The safety supervisor: the one process that may move a robot (console spec §2.1, §3).

Neither the myAGV nor the AiNex has a command watchdog -- `myagv_odometry_node` re-sends
the last Twist forever, and a walking AiNex walks until told `stop` -- so a UI that
freezes, crashes or is killed must not be able to leave its last command running. The UI
therefore never touches rosbridge. It starts this process, which owns the rosbridge
connection and every motion publication, and speaks to it over its stdin/stdout pipes:

    UI -> supervisor (stdin, JSON lines)
        {"op": "hb"}                         heartbeat
        {"op": "cmd", "vx":, "vy":, "wz":}   desired body velocity (also a heartbeat)
        {"op": "head", "pan":, "tilt":}      AiNex head position (also a heartbeat)
        {"op": "enable"}                     the operator confirmed the independent e-stop
        {"op": "subscribe", "stream": s}     s in camera | odom | scan
        {"op": "quit"}                       normal exit
    supervisor -> UI (stdout, JSON lines)
        {"op": "ready", "robot":, "namespace":, "camera_topic":}
        {"op": "error", "message":}
        {"op": "msg", "stream":, "msg":}
        {"op": "stopped", "reason":}

The watchdog arms on the first heartbeat. From then on, if the heartbeat stops for the
safety timeout (`SAFETY_TIMEOUT`), or the pipe closes (the UI died or closed its end), or
the parent process changes, or a quit or SIGTERM arrives, the supervisor sends the robot's
stop command three times, 50 ms apart, and exits. Nothing else is published after the
first stop. SIGINT and SIGHUP are ignored: a terminal's Ctrl-C reaches the whole process
group, and it is the UI's job to turn that into a quit -- or, if the UI is wedged, the
heartbeat's.

Non-zero commands are refused until the UI says the operator confirmed an independent
physical emergency stop. That device, not this process, is the protection against host
failure or network loss: software on the failed path cannot stop anything.

    python -m robot_console.supervisor --url ws://127.0.0.1:9090 [--robot R] [--namespace NS]
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from robot_console.teleop import Command
from robot_console.wire import DEFAULT_URL, parse_url, url_arg

#: The safety timeout, seconds (console spec §2.1): a fixed constant, not a flag.
SAFETY_TIMEOUT = 0.25
STOP_REPEATS = 3
STOP_SPACING_S = 0.05

#: Re-send rate of the desired command. The myAGV holds the last Twist, so this is
#: redundancy against a dropped message rather than a keep-alive.
PUBLISH_HZ = 20.0
#: How often the watchdog looks. Small against the 100 ms slack the spec allows.
POLL_S = 0.005
#: Stream messages queued for a slow UI before the oldest are dropped. The writer thread
#: may block on a full pipe; the watchdog never does.
OUT_QUEUE = 256

STREAMS = ("camera", "odom", "scan")

EXIT_OK = 0
EXIT_ERROR = 2


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m robot_console.supervisor")
    parser.add_argument("--url", type=url_arg, default=DEFAULT_URL)
    parser.add_argument("--robot", default=None)
    # `--namespace=` (empty) is the bare contract; absent means "discover".
    parser.add_argument("--namespace", default=None)
    # Ask /rosapi even when both are given, so they only narrow what is on the wire (slam).
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    return parser.parse_args(argv)


class _Out:
    """Serialised writes to stdout, off the watchdog's thread."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._queue: "queue.Queue" = queue.Queue(maxsize=OUT_QUEUE)
        self._thread = threading.Thread(target=self._run, name="supervisor-out", daemon=True)
        self._thread.start()

    def send(self, message: dict, *, droppable: bool = False) -> None:
        if not droppable:
            try:
                self._queue.put(message, timeout=0.5)
            except queue.Full:
                pass
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                pass

    def drain(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.005)

    def _run(self) -> None:
        while True:
            message = self._queue.get()
            try:
                self._stream.write((json.dumps(message) + "\n").encode())
                self._stream.flush()
            except Exception:
                return


class Supervisor:
    """The watchdog loop. `link` is a connected RobotLink or AiNexLink."""

    def __init__(self, link, *, out: _Out, inp, parent_pid: int,
                 has_head: bool = False) -> None:
        self.link = link
        self.safety_timeout = SAFETY_TIMEOUT
        self.out = out
        self.inp = inp
        self.parent_pid = parent_pid
        self.has_head = has_head
        self._lock = threading.Lock()
        self._desired = Command()
        self._head: Optional[tuple] = None
        self._head_sent: Optional[tuple] = None
        self._last_hb: Optional[float] = None
        self._enabled = False
        self._eof = False
        self._quit = False
        self._terminated = False
        self._subscribed: set = set()

    # ------------------------------------------------------------------ input

    def on_sigterm(self, *_):
        self._terminated = True

    def read_loop(self) -> None:
        try:
            for raw in self.inp:
                try:
                    message = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(message, dict):
                    self.handle(message)
        except Exception:
            pass
        self._eof = True

    def handle(self, message: Dict[str, Any]) -> None:
        op = message.get("op")
        now = time.monotonic()
        with self._lock:
            if op in ("hb", "cmd", "head", "enable"):
                self._last_hb = now
            if op == "cmd":
                self._desired = Command(
                    vx=float(message.get("vx", 0.0)),
                    vy=float(message.get("vy", 0.0)),
                    wz=float(message.get("wz", 0.0)),
                )
            elif op == "head":
                self._head = (float(message.get("pan", 0.0)), float(message.get("tilt", 0.0)))
            elif op == "enable":
                self._enabled = True
            elif op == "quit":
                self._quit = True
        if op == "subscribe":
            self.subscribe(str(message.get("stream", "")))

    def subscribe(self, stream: str) -> None:
        if stream not in STREAMS or stream in self._subscribed:
            return
        self._subscribed.add(stream)

        def forward(msg, _stream=stream):
            self.out.send({"op": "msg", "stream": _stream, "msg": msg}, droppable=True)

        try:
            if stream == "camera":
                self.link.subscribe_camera(forward)
            elif stream == "scan":
                self.link.subscribe_scan(forward)
            elif stream == "odom":
                self.link.subscribe_odom_raw(forward)
        except Exception as exc:  # noqa: BLE001 - e.g. odom on a robot that has none
            self.out.send({"op": "warning", "message": f"cannot subscribe {stream}: {exc}"})

    # ------------------------------------------------------------------ the loop

    def _why_stop(self, now: float) -> Optional[str]:
        if self._terminated:
            return "SIGTERM"
        if self._quit:
            return "quit"
        if self._eof:
            return "IPC closed"
        if os.getppid() != self.parent_pid:
            return "parent process gone"
        with self._lock:
            last = self._last_hb
        if last is not None and now - last > self.safety_timeout:
            return f"no heartbeat for {self.safety_timeout:.3g} s"
        if not self.link.is_connected:
            return "rosbridge connection lost"
        return None

    def run(self) -> str:
        period = 1.0 / PUBLISH_HZ
        next_publish = time.monotonic()
        while True:
            now = time.monotonic()
            reason = self._why_stop(now)
            if reason is not None:
                break
            with self._lock:
                armed = self._last_hb is not None
                command = self._desired if self._enabled else Command()
                head = self._head if self._enabled else None
            if armed and now >= next_publish:
                self.link.publish_cmd_vel(command)
                next_publish = max(next_publish + period, now)
            if self.has_head and head is not None and head != self._head_sent:
                self.link.publish_head(*head)
                self._head_sent = head
            time.sleep(POLL_S)
        self.stop_robot()
        return reason

    def stop_robot(self) -> None:
        """The robot's stop command, three times, 50 ms apart. Never raises."""
        for i in range(STOP_REPEATS):
            if i:
                time.sleep(STOP_SPACING_S)
            try:
                self.link.stop()
            except Exception:
                pass


def _connect_and_resolve(args, out: _Out):
    """Connect, discover, and build the link. Returns (link, ready) or exits."""
    import roslibpy

    from robot_console import discovery
    from robot_console.bridge import quiet_roslibpy_logging
    from robot_console.robots import TELEOP_ROBOTS, profile

    quiet_roslibpy_logging()
    host, port = parse_url(args.url)
    ros = roslibpy.Ros(host=host, port=port)
    try:
        ros.run(timeout=args.connect_timeout)
    except Exception as exc:  # noqa: BLE001
        raise discovery.DiscoveryError(f"could not connect to {args.url}: {exc}") from None
    if not ros.is_connected:
        raise discovery.DiscoveryError(f"could not connect to {args.url}")

    if args.robot is not None and args.robot not in TELEOP_ROBOTS:
        raise discovery.DiscoveryError(f"unknown robot {args.robot!r}")
    if args.robot is not None and args.namespace is not None and not args.discover:
        robot, namespace = args.robot, args.namespace
        camera = None
    else:
        from robot_console.fleet import topics_from

        try:
            present = topics_from(ros, 5.0)
        except Exception as exc:  # noqa: BLE001 - no rosapi, a timeout, a refusal
            raise discovery.unreachable(args.url, args.robot, args.namespace, exc) from None
        found = discovery.discover_from(present, args.robot, args.namespace)
        robot, namespace, camera = found.robot, found.namespace, found.camera_topic

    prof = profile(robot)
    link = prof.make_link(host, port, namespace, camera)
    link.attach(ros)
    ready = {
        "op": "ready",
        "robot": robot,
        "namespace": namespace,
        "camera_topic": link._camera_name,
        "has_odom": prof.has_odom,
        "stop_command": prof.stop_command,
        "url": args.url,
    }
    if prof.has_odom:
        ready["odom_topic"] = link._odom_name
        ready["scan_topic"] = link._scan_name
        ready["cmd_topic"] = link._cmd_name
    return ros, link, prof, ready


def main(argv=None) -> int:
    args = _parse_args(argv)
    # Ctrl-C and a closing terminal reach the whole group; the UI owns those. A UI that
    # cannot act on them stops heartbeating, and that is handled below.
    for sig in (signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
    parent = os.getppid()
    out = _Out(sys.stdout.buffer)

    from robot_console.discovery import DiscoveryError

    try:
        ros, link, prof, ready = _connect_and_resolve(args, out)
    except DiscoveryError as exc:
        out.send({"op": "error", "message": str(exc)})
        out.drain(1.0)
        os._exit(EXIT_ERROR)
    except Exception as exc:  # noqa: BLE001
        out.send({"op": "error", "message": f"{type(exc).__name__}: {exc}"})
        out.drain(1.0)
        os._exit(EXIT_ERROR)

    supervisor = Supervisor(
        link, out=out, inp=sys.stdin.buffer,
        parent_pid=parent, has_head=prof.has_head,
    )
    signal.signal(signal.SIGTERM, supervisor.on_sigterm)
    out.send(ready)
    threading.Thread(target=supervisor.read_loop, name="supervisor-in", daemon=True).start()

    try:
        reason = supervisor.run()
    except BaseException as exc:  # noqa: BLE001 - whatever happened, stop the robot
        supervisor.stop_robot()
        reason = f"supervisor error: {exc}"
    print(f"safety supervisor: stopped the {ready['robot']} ({reason})", file=sys.stderr)
    out.send({"op": "stopped", "reason": reason})
    # Let the reactor put the stops on the socket and the UI read why, then leave without
    # waiting on Twisted's shutdown, which can hang.
    time.sleep(0.15)
    out.drain(0.3)
    try:
        ros.close()
    except Exception:
        pass
    time.sleep(0.05)
    os._exit(EXIT_OK)


# ====================================================================== the UI's side


class SupervisorError(RuntimeError):
    """The supervisor could not start, or refused (discovery, connection)."""


class SupervisedLink:
    """What the UI holds instead of a link: the supervisor process, behind a pipe.

    Link-shaped (`subscribe_camera`, `publish_cmd_vel`, `stop`, `close` ...) so the loops
    that used to drive a `RobotLink` read the same. Every call is a line on the pipe;
    nothing here opens a socket. `heartbeat()` must be called from the UI loop itself --
    never from a helper thread, which would keep beating for a UI that has frozen.
    Callbacks run on this object's reader thread, as roslibpy's ran on its reactor.
    """

    def __init__(
        self,
        url: str,
        *,
        robot: Optional[str] = None,
        namespace: Optional[str] = None,
        python: str = sys.executable,
        discover: bool = False,
    ) -> None:
        self.url = url
        self.robot = robot
        self.namespace = namespace
        self.discover = discover
        self.python = python
        self.proc: Optional[subprocess.Popen] = None
        self.ready: Optional[dict] = None
        self.stopped_reason: Optional[str] = None
        self._callbacks: Dict[str, Callable] = {}
        self._ready_event = threading.Event()
        self._error: Optional[str] = None
        self._write_lock = threading.Lock()
        self._broken = False
        self._closed = False
        self._reader: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle

    def command_line(self) -> list:
        cmd = [self.python, "-m", "robot_console.supervisor", "--url", self.url]
        if self.robot is not None:
            cmd += ["--robot", self.robot]
        if self.namespace is not None:
            cmd.append(f"--namespace={self.namespace}")
        if self.discover:
            cmd.append("--discover")
        return cmd

    def start(self, timeout: float = 20.0) -> dict:
        """Spawn the supervisor and wait for it to connect and discover. Returns `ready`."""
        self.proc = subprocess.Popen(
            self.command_line(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            close_fds=True,
        )
        self._reader = threading.Thread(target=self._read_loop, name="supervised-link", daemon=True)
        self._reader.start()
        if not self._ready_event.wait(timeout):
            self.kill()
            raise SupervisorError(f"the safety supervisor did not come up within {timeout:.0f} s")
        if self._error is not None or self.ready is None:
            self.wait(2.0)
            raise SupervisorError(self._error or "the safety supervisor exited during start-up")
        return self.ready

    def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                message = json.loads(raw)
            except (ValueError, TypeError):
                continue
            op = message.get("op")
            if op == "msg":
                callback = self._callbacks.get(message.get("stream"))
                if callback is not None:
                    try:
                        callback(message.get("msg") or {})
                    except Exception:
                        pass
            elif op == "ready":
                self.ready = message
                self._ready_event.set()
            elif op == "error":
                self._error = str(message.get("message"))
                self._ready_event.set()
            elif op == "stopped":
                self.stopped_reason = str(message.get("reason"))
            elif op == "warning":
                print(f"warning: {message.get('message')}", file=sys.stderr)
        if self.stopped_reason is None:
            self.stopped_reason = self._error or "the safety supervisor exited"
        self._ready_event.set()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not self._broken

    @property
    def is_connected(self) -> bool:
        return self.alive and self.ready is not None

    # ------------------------------------------------------------------ the pipe

    def _send(self, message: dict) -> bool:
        if self.proc is None or self.proc.stdin is None or self._broken:
            return False
        line = (json.dumps(message) + "\n").encode()
        with self._write_lock:
            try:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
                return True
            except (BrokenPipeError, OSError, ValueError):
                self._broken = True
                return False

    def heartbeat(self) -> bool:
        return self._send({"op": "hb"})

    def enable_motion(self) -> None:
        """Tell the supervisor the operator confirmed the independent emergency stop."""
        self._send({"op": "enable"})

    def publish_cmd_vel(self, command: Command) -> None:
        self._send({"op": "cmd", "vx": command.vx, "vy": command.vy, "wz": command.wz})

    def publish_head(self, pan: float, tilt: float) -> None:
        self._send({"op": "head", "pan": float(pan), "tilt": float(tilt)})

    def stop(self) -> None:
        self.publish_cmd_vel(Command())

    def _subscribe(self, stream: str, callback: Callable) -> None:
        self._callbacks[stream] = callback
        self._send({"op": "subscribe", "stream": stream})

    def subscribe_camera(self, callback: Callable[[dict], None]) -> None:
        self._subscribe("camera", callback)

    def subscribe_scan(self, callback: Callable[[dict], None]) -> None:
        self._subscribe("scan", callback)

    def subscribe_odom(self, callback) -> None:
        from robot_console.bridge import parse_odom

        self._subscribe("odom", lambda msg: callback(parse_odom(msg)))

    def close_ipc(self) -> None:
        """Close our end of the pipe. The supervisor stops the robot and exits."""
        if self.proc is not None and self.proc.stdin is not None:
            with self._write_lock:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            self._broken = True

    def wait(self, timeout: float) -> Optional[int]:
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()   # SIGTERM: the supervisor still stops the robot
            except Exception:
                pass
            if self.wait(1.0) is None:
                self.proc.kill()

    def close(self, timeout: float = 2.0) -> None:
        """Normal exit: ask the supervisor to stop the robot and go, then make sure it has."""
        if self._closed:
            return
        self._closed = True
        self._send({"op": "quit"})
        self.close_ipc()
        if self.wait(timeout) is None:
            self.kill()
        if self._reader is not None:
            self._reader.join(0.5)

    def describe(self) -> str:
        return self.url


if __name__ == "__main__":
    sys.exit(main())
