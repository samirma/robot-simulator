"""Client for the simulator's generic msgpack-numpy robot-control protocol."""

from __future__ import annotations

import argparse
import importlib
import math
import time
import traceback
from typing import Any, Callable

import msgpack_numpy
import numpy as np
import websockets.sync.client as ws_client

Controller = Callable[[dict[str, Any]], dict[str, Any]]


class ArmClient:
    """Connect to a simulated robot and answer each observation with one action."""

    def __init__(
        self,
        controller: Controller,
        host: str = "127.0.0.1",
        port: int = 8000,
        connect_timeout: float = 180.0,
    ) -> None:
        self._controller = controller
        self._uri = f"ws://{host}:{port}"
        self._connect_timeout = connect_timeout
        self._connection = None
        self.metadata: dict[str, Any] | None = None

    def run(self) -> None:
        deadline = time.monotonic() + self._connect_timeout
        announced = False
        while True:
            try:
                self._connection = ws_client.connect(
                    self._uri, compression=None, max_size=None, open_timeout=30
                )
                break
            except (ConnectionRefusedError, OSError) as exc:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"no simulator control server at {self._uri} after "
                        f"{self._connect_timeout:.0f}s"
                    ) from exc
                if not announced:
                    print(f"waiting for simulator control server at {self._uri} ...")
                    announced = True
                time.sleep(2.0)

        try:
            self.metadata = msgpack_numpy.unpackb(self._connection.recv(timeout=30))
            print(f"connected to simulator: {self.metadata}")
            for raw in self._connection:
                observation = msgpack_numpy.unpackb(raw)
                try:
                    action = self._controller(observation)
                except Exception:
                    self._connection.send(traceback.format_exc())
                    raise
                self._connection.send(msgpack_numpy.packb(action))
        finally:
            self.close()

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()


def hold(obs: dict[str, Any]) -> dict[str, Any]:
    """Echo the last joint-position command so the robot stays still."""
    commanded = obs.get("actions/joint_pos")
    if not isinstance(commanded, dict):
        raise TypeError("observation has no actions/joint_pos mapping")
    return {name: np.asarray(target, dtype=np.float64) for name, target in commanded.items()}


def make_wave(amplitude: float = 0.35, period_s: float = 4.0) -> Controller:
    """Create a controller that sweeps the final arm joint around its initial target."""
    home = None
    started = time.monotonic()

    def wave(obs: dict[str, Any]) -> dict[str, Any]:
        nonlocal home
        action = hold(obs)
        arm = np.asarray(action["arm"], dtype=np.float64).copy()
        if home is None:
            home = arm.copy()
        phase = 2 * math.pi * (time.monotonic() - started) / period_s
        arm[-1] = home[-1] + amplitude * math.sin(phase)
        action["arm"] = arm
        return action

    return wave


def load_controller(spec: str) -> Controller:
    if spec == "hold":
        return hold
    if spec == "wave":
        return make_wave()
    if ":" not in spec:
        raise ValueError(f"unknown controller {spec!r}; use hold, wave, or module:callable")
    module_name, attr = spec.split(":", 1)
    controller = getattr(importlib.import_module(module_name), attr)
    if not callable(controller):
        raise TypeError(f"{spec!r} is not callable")
    return controller


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--connect-timeout", type=float, default=180.0)
    ap.add_argument("--controller", default="hold", help="hold, wave, or module:callable")
    args = ap.parse_args()
    ArmClient(
        load_controller(args.controller),
        host=args.host,
        port=args.port,
        connect_timeout=args.connect_timeout,
    ).run()


if __name__ == "__main__":
    main()
