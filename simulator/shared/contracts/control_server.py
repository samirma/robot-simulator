"""Generic observation/action server for externally controlled simulated robots.

The simulator owns this server. A client receives one metadata message, then exchanges
one action for each simulator observation. This module contains transport only: it never
chooses an action or implements robot behaviour.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import msgpack_numpy
import websockets.sync.server as ws_server
from websockets.exceptions import ConnectionClosed

log = logging.getLogger("control_server")


class ControlServer:
    """Serve one external controller over binary msgpack-numpy websocket frames."""

    def __init__(
        self,
        host: str,
        port: int,
        metadata: dict[str, Any],
    ) -> None:
        self._host = host
        self._port = port
        self._metadata = metadata
        self._server: ws_server.Server | None = None
        self._connection: Any | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._connected = threading.Event()
        self._shutdown = threading.Event()
        self._observation: dict[str, Any] | None = None
        self._observation_sequence = 0
        self._action: dict[str, Any] | None = None

    def start(self) -> None:
        self._server = ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        )
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        log.info("robot control listening on ws://%s:%d", self._host, self.port)

    @property
    def port(self) -> int:
        """The bound port, including the OS-assigned value when constructed with zero."""
        if self._server is None:
            return self._port
        return int(self._server.socket.getsockname()[1])

    def publish(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        """Publish the newest observation and return any action received since last tick."""
        with self._condition:
            self._observation = observation
            self._observation_sequence += 1
            action, self._action = self._action, None
            self._condition.notify_all()
        return action

    def stop(self) -> None:
        self._shutdown.set()
        with self._condition:
            connection = self._connection
            self._connection = None
            self._condition.notify_all()
        if connection is not None:
            connection.close()
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        self._connected.clear()

    def _handler(self, websocket) -> None:
        with self._lock:
            if self._connection is not None:
                websocket.close(code=1013, reason="a robot control client is already connected")
                return
            self._connection = websocket
        try:
            websocket.send(msgpack_numpy.packb(self._metadata))
            self._connected.set()
            log.info("robot control client connected from %s", websocket.remote_address)
            seen_sequence = 0
            while not self._shutdown.is_set():
                with self._condition:
                    ready = self._condition.wait_for(
                        lambda seen=seen_sequence: self._shutdown.is_set()
                        or self._observation_sequence > seen,
                        timeout=0.5,
                    )
                    if self._shutdown.is_set():
                        return
                    if not ready or self._observation is None:
                        continue
                    observation = self._observation
                    seen_sequence = self._observation_sequence
                websocket.send(msgpack_numpy.packb(observation))
                reply = websocket.recv(timeout=30)
                if isinstance(reply, str):
                    raise RuntimeError(f"robot control client error:\n{reply}")  # noqa: TRY004
                action = msgpack_numpy.unpackb(reply)
                if not isinstance(action, dict):
                    raise TypeError(
                        f"robot control action must be a dict, got {type(action).__name__}"
                    )
                with self._lock:
                    self._action = action
        except (ConnectionClosed, TimeoutError, OSError) as exc:
            log.info("robot control client disconnected: %s", exc)
        except Exception:
            log.exception("robot control client protocol error")
        finally:
            self._drop(websocket)

    def _drop(self, websocket) -> None:
        with self._lock:
            if self._connection is websocket:
                self._connection = None
                self._connected.clear()
