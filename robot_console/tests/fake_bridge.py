"""A minimal rosbridge server, for testing the console without any robot.

Deliberately reimplemented here rather than imported from `../simulator`: the console
has to install and test on a machine that has never seen the simulator checkout, and a
test double that imports the thing it is standing in for proves nothing anyway.

It mirrors the real bridge's quirks on purpose -- no status handshake on connect,
message ids ignored and never echoed, no loopback of published topics, and relative
topic names normalised to absolute.
"""

from __future__ import annotations

import json
import threading
from typing import Dict, List, Optional, Set, Tuple

from websockets.sync.server import serve


def normalise(topic: str) -> str:
    return topic if topic.startswith("/") else "/" + topic


class FakeBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._clients: Dict[object, Set[str]] = {}
        self._received: List[Tuple[str, dict]] = []
        self._event = threading.Event()

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> int:
        # Port 0 so tests never collide with a real bridge on 9090.
        self._server = serve(self._handle, self.host, self.port, compression=None, max_size=None)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    def __enter__(self) -> "FakeBridge":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---------------------------------------------------------------- protocol

    def _handle(self, connection) -> None:
        with self._lock:
            self._clients[connection] = set()
        try:
            for raw in connection:
                try:
                    message = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                op = message.get("op")
                topic = message.get("topic")
                topic = normalise(topic) if isinstance(topic, str) and topic else None
                if op == "subscribe" and topic:
                    with self._lock:
                        self._clients[connection].add(topic)
                    self._event.set()
                elif op == "unsubscribe" and topic:
                    with self._lock:
                        self._clients[connection].discard(topic)
                elif op == "publish" and topic:
                    with self._lock:
                        self._received.append((topic, message.get("msg") or {}))
                    self._event.set()
                # advertise/unadvertise are accepted and ignored, as upstream does.
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.pop(connection, None)

    def publish(self, topic: str, msg: dict) -> None:
        """Push a message to every client subscribed to `topic`."""
        name = normalise(topic)
        frame = json.dumps({"op": "publish", "topic": name, "msg": msg})
        with self._lock:
            targets = [c for c, topics in self._clients.items() if name in topics]
        for connection in targets:
            try:
                connection.send(frame)
            except Exception:
                pass

    # ---------------------------------------------------------------- assertions

    @property
    def received(self) -> List[Tuple[str, dict]]:
        with self._lock:
            return list(self._received)

    def received_on(self, topic: str) -> List[dict]:
        name = normalise(topic)
        return [msg for t, msg in self.received if t == name]

    @property
    def subscriptions(self) -> Set[str]:
        with self._lock:
            out: Set[str] = set()
            for topics in self._clients.values():
                out |= topics
            return out

    def _wait(self, predicate, timeout: float) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            self._event.wait(0.05)
            self._event.clear()
        return predicate()

    def wait_for_subscription(self, topic: str, timeout: float = 5.0) -> bool:
        name = normalise(topic)
        return self._wait(lambda: name in self.subscriptions, timeout)

    def wait_for_publish(self, topic: str, count: int = 1, timeout: float = 5.0) -> bool:
        return self._wait(lambda: len(self.received_on(topic)) >= count, timeout)
