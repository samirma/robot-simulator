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
from typing import Callable, Optional, TextIO


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


def startup_instructions_any(host: str, port: int) -> str:
    """What to start when the robot is not known yet -- before the wire has been asked.

    Which robot this console is about to drive is normally the wire's answer, and there is
    no wire. So this names the two the console can drive and stops there: telling someone
    to start a myAGV when they meant an AiNex is worse than naming both.
    """
    return f"""Start the simulator in another terminal:

    cd ../simulator/molmospaces
    ./run.sh view --robot myagv --scene ithor:1 --ros-port {port}   # or --robot ainex

Or a kitchen with a fleet in it:

    cd ../simulator
    ./kitchen.sh serve --robots myagv --port {port}

The robot and its namespace are then read off the wire; --robot and --namespace override
that. On real hardware, point --host at the robot instead.

Bypass this check with --no-preflight."""


def startup_instructions(host: str, port: int) -> str:
    """What to start for a myAGV, in whichever of its three forms the user meant."""
    return f"""Start the simulator in another terminal:

    cd ../simulator/molmospaces
    ./run.sh view --robot myagv --scene ithor:1 --ros-port {port}

Or, without MuJoCo, the standalone protocol server (odom only, no camera):

    cd ../simulator
    python shared/contracts/rosbridge_server.py --port {port} --echo

Or, on a real myAGV over the network:

    # on the AGV
    roslaunch myagv_odometry myagv_active.launch
    roslaunch rosbridge_server rosbridge_websocket.launch
    # then, here
    ./bin/teleop.sh --host <agv-ip>

Bypass this check with --no-preflight."""


def startup_instructions_ainex(host: str, port: int) -> str:
    """What to start when the robot being driven is the AiNex."""
    return f"""Start the simulator in another terminal:

    cd ../simulator/molmospaces
    ./run.sh view --robot ainex --scene ithor:1 --ros-port {port}

Or, on a real AiNex over the network:

    # on the robot (the vendor stack brings rosbridge up itself)
    roslaunch ainex_bringup bringup.launch
    # then, here
    --host <ainex-ip>

Bypass this check with --no-preflight."""


def preflight(
    host: str,
    port: int,
    *,
    timeout: float = 1.5,
    stream: TextIO = sys.stderr,
    instructions: Optional[Callable[[str, int], str]] = None,
) -> bool:
    """Probe, and print the startup instructions if nothing is there.

    `instructions` lets a caller substitute the text for the robot it is actually driving
    -- the thing to launch differs per robot, and telling someone to start a myAGV when
    they asked for something else is worse than saying nothing.
    """
    result = probe_tcp(host, port, timeout)
    if result.ok:
        return True
    print(f"error: no rosbridge on {result.url} ({result.detail})\n", file=stream)
    print((instructions or startup_instructions)(host, port), file=stream)
    return False
