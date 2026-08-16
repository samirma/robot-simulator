#!/usr/bin/env python
"""External control server for the MolmoSpaces simulator.

The simulator is the *client*: it connects out over a websocket
(``tools/spawn_robot.py::connect_control``, wired behind ``view --control``),
sends an observation dict each step, and applies the action dict it gets back.
This module is a dependency-light server for driving the robot from your own
code.

Wire protocol (msgpack-numpy framed, binary websocket messages):

    server -> client   metadata dict, once, on connect
    client -> server   observation dict   {"qpos", "qvel", "actions/joint_pos",
                                           <camera images>, ...}
    server -> client   action dict        {"arm": ndarray, "gripper": ndarray}

Action semantics follow the robot's ``command_mode`` (see
``molmo_spaces/configs/robot_configs.py``); for joint-position robots both arm
and gripper are absolute joint targets in radians / metres.

Usage:
    ./run.sh serve                       # hold position (default)
    ./run.sh serve --controller wave     # visibly move the arm
    ./run.sh serve --controller mymod:fn # your own callable
"""

from __future__ import annotations

import argparse
import asyncio
import http
import importlib
import logging
import math
import time
import traceback
from typing import Any, Callable

import msgpack_numpy
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames

log = logging.getLogger("bridge")

Controller = Callable[[dict], dict]
"""Takes an observation dict, returns an action dict with "arm" and "gripper"."""


# --------------------------------------------------------------------------
# Built-in controllers
# --------------------------------------------------------------------------


def _arm_gripper_from_obs(obs: dict, n_arm: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Read the current joint-position command out of an observation.

    Prefer ``actions/joint_pos``: it is the last commanded joint-position action,
    so it is already the exact width the controllers expect. This matters for the
    gripper, where the *command* is one value per actuator while ``qpos`` reports
    one value per finger joint (2 for the Franka) -- feeding qpos straight back
    raises a shape mismatch in the sim.

    Falls back to ``qpos`` (a dict keyed by move group), truncating the gripper to
    a single value, and finally to a flat vector split at ``n_arm``.
    """
    commanded = obs.get("actions/joint_pos")
    if isinstance(commanded, dict) and commanded.get("arm") is not None:
        arm = np.asarray(commanded["arm"], dtype=np.float64).reshape(-1)
        gripper = np.asarray(commanded.get("gripper", [0.0]), dtype=np.float64).reshape(-1)
        return arm, gripper

    qpos = obs.get("qpos")
    if isinstance(qpos, dict):
        arm = np.asarray(qpos.get("arm", []), dtype=np.float64).reshape(-1)
        gripper = np.asarray(qpos.get("gripper", []), dtype=np.float64).reshape(-1)[:1]
    else:
        flat = np.asarray(qpos if qpos is not None else np.zeros(9), dtype=np.float64).reshape(-1)
        arm = flat[:n_arm].copy()
        gripper = flat[n_arm : n_arm + 1].copy()

    if arm.size == 0:
        raise ValueError(
            "no arm joints in observation; expected 'actions/joint_pos' or 'qpos', got keys "
            f"{sorted(obs)}"
        )
    if gripper.size == 0:
        gripper = np.zeros(1)
    return arm, gripper


def hold(obs: dict) -> dict:
    """Command the arm to stay exactly where it is. Safe default."""
    arm, gripper = _arm_gripper_from_obs(obs)
    return {"arm": arm, "gripper": gripper}


def _make_wave(amplitude: float = 0.35, period_s: float = 4.0) -> Controller:
    """Sinusoidally sweep the last arm joint around its position at connect time.

    Used to prove end-to-end that commands from this server actually move the
    robot in the viewer.
    """
    state: dict[str, Any] = {"home": None, "t0": None}

    def wave(obs: dict) -> dict:
        arm, gripper = _arm_gripper_from_obs(obs)
        if state["home"] is None:
            state["home"] = arm.copy()
            state["t0"] = time.monotonic()
        target = state["home"].copy()
        phase = 2 * math.pi * (time.monotonic() - state["t0"]) / period_s
        target[-1] = state["home"][-1] + amplitude * math.sin(phase)
        return {"arm": target, "gripper": gripper}

    return wave


def _load_controller(spec: str) -> Controller:
    """Resolve ``hold``, ``wave``, or a dotted ``module:callable`` reference."""
    if spec == "hold":
        return hold
    if spec == "wave":
        return _make_wave()
    if ":" not in spec:
        raise SystemExit(f"unknown controller {spec!r}; use hold, wave, or module:callable")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise SystemExit(f"{spec} is not callable")
    return fn


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


class BridgeServer:
    def __init__(
        self,
        controller: Controller,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        verbose: bool = False,
    ) -> None:
        self._controller = controller
        self._host = host
        self._port = port
        self._metadata = {"model_name": "bridge", **(metadata or {})}
        self._verbose = verbose

    def serve_forever(self) -> None:
        log.info("listening on ws://%s:%d", self._host, self._port)
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        log.info("simulator connected from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        step = 0
        try:
            while True:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if step == 0:
                    log.info("observation keys: %s", sorted(obs.keys()))
                    if isinstance(obs.get("qpos"), dict):
                        log.info(
                            "move groups: %s",
                            {k: len(v) for k, v in obs["qpos"].items()},
                        )

                # The controller is user code and may block; keep it off the
                # event loop so slow controllers don't stall the websocket.
                action = await asyncio.get_event_loop().run_in_executor(
                    None, self._controller, obs
                )

                if self._verbose and step % 20 == 0:
                    log.info("step %d arm=%s", step, np.round(action["arm"], 3))

                await websocket.send(packer.pack(action))
                step += 1
        except websockets.ConnectionClosed:
            log.info("simulator disconnected after %d steps", step)
        except Exception:
            log.exception("controller error")
            # Sending a text frame is how this protocol reports server-side
            # failures; the client raises with the traceback included.
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="controller error; traceback in previous frame",
            )
            raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--controller",
        default="hold",
        help="hold (default), wave, or module:callable taking obs -> action",
    )
    ap.add_argument("--verbose", action="store_true", help="log actions periodically")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    BridgeServer(
        controller=_load_controller(args.controller),
        host=args.host,
        port=args.port,
        metadata={"controller": args.controller},
        verbose=args.verbose,
    ).serve_forever()


if __name__ == "__main__":
    main()
