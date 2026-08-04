"""Is anything actually listening before we try to drive a robot?

Deliberately a plain TCP connect rather than a rosbridge one. Two reasons:

  * The bridge sends no handshake on connect and never echoes message ids, so a
    successful rosbridge connect proves nothing that a TCP connect does not. Real
    validation means subscribing and waiting for data, which is the application, not a
    preflight.
  * roslibpy drives a process-global Twisted reactor that cannot be reliably restarted
    once terminated. A preflight that connects and tears down risks leaving the real
    connection unable to start.
"""

from __future__ import annotations

import dataclasses
import socket
import sys
import time
from typing import TextIO


@dataclasses.dataclass(frozen=True)
class PreflightResult:
    ok: bool
    host: str
    port: int
    detail: str
    elapsed: float

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"


def probe_tcp(host: str, port: int, timeout: float = 1.5) -> PreflightResult:
    """Try to open a TCP connection. Never raises.

    The three failure modes are distinguished because they point at three different
    fixes: nothing started, wrong host or a firewall, or a typo in the name.
    """
    started = time.monotonic()
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            pass
    except socket.gaierror:
        detail = "hostname does not resolve"
        ok = False
    except (ConnectionRefusedError, ConnectionResetError):
        detail = "connection refused"
        ok = False
    except (socket.timeout, TimeoutError):
        detail = "timed out (host unreachable or filtered)"
        ok = False
    except OSError as exc:
        detail = f"{type(exc).__name__}: {exc}"
        ok = False
    else:
        detail = "connected"
        ok = True
    return PreflightResult(ok, host, int(port), detail, time.monotonic() - started)


def startup_instructions(host: str, port: int) -> str:
    """What to start, for whichever of the three robots the user meant."""
    return f"""Start the simulator in another terminal:

    cd ../simulator
    ./run.sh view --robot myagv --scene ithor:1 --ros-port {port}

Or, without MuJoCo, the standalone protocol server (odom only, no camera):

    cd ../simulator
    python bridge/rosbridge_server.py --port {port} --echo

Or, on a real myAGV over the network:

    # on the AGV
    roslaunch myagv_odometry myagv_active.launch
    roslaunch rosbridge_server rosbridge_websocket.launch
    # then, here
    ./bin/teleop.sh --host <agv-ip>

Bypass this check with --no-preflight."""


def preflight(
    host: str,
    port: int,
    *,
    timeout: float = 1.5,
    stream: TextIO = sys.stderr,
) -> bool:
    """Probe, and print the startup instructions if nothing is there."""
    result = probe_tcp(host, port, timeout)
    if result.ok:
        return True
    print(f"error: no rosbridge on {result.url} ({result.detail})\n", file=stream)
    print(startup_instructions(host, port), file=stream)
    return False
