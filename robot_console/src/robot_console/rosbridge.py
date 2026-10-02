"""A small rosbridge v2 protocol client (JSON over a websocket), shared by every entry point.

It speaks to ROS 1 and ROS 2 rosbridge servers alike; the dialect only shows in type
spellings, which callers take from their profile. One reader thread dispatches incoming
messages; sends are serialised. A lost connection is reported once through ``on_close``.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

DEFAULT_URL = "ws://127.0.0.1:9090"


class TransportError(RuntimeError):
    """The transport could not send, or the connection is gone."""


class ServiceError(RuntimeError):
    """A service call returned an error (``result: false``)."""


def check_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("ws", "wss") or not parts.hostname:
        raise ValueError(f"--url must look like ws://<host>:<port>, got {url!r}")
    return url


class Rosbridge:
    def __init__(self, url: str = DEFAULT_URL) -> None:
        self.url = check_url(url)
        self._ws = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: Dict[str, concurrent.futures.Future] = {}
        self._subs: Dict[str, List[Tuple[str, Callable[[dict], None]]]] = {}
        self._advertised: Dict[str, str] = {}
        self._close_cbs: List[Callable[[str], None]] = []
        self._closed = threading.Event()
        self._close_reason = ""
        self._user_closed = False
        self.status_messages: List[dict] = []
        self._reader: Optional[threading.Thread] = None

    # ------------------------------------------------------------ lifecycle
    def connect(self, timeout: float = 5.0) -> "Rosbridge":
        try:
            # Entered as a context manager (the supported form); closed in close().
            self._ws = ws_connect(self.url, open_timeout=timeout, close_timeout=1.0,
                                  max_size=None, compression=None).__enter__()
        except Exception as exc:  # noqa: BLE001 - any failure means unreachable
            raise TransportError(f"cannot connect to {self.url}: {exc}") from None
        self._reader = threading.Thread(target=self._read_loop, name="rosbridge-reader", daemon=True)
        self._reader.start()
        return self

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._closed.is_set()

    @property
    def close_reason(self) -> str:
        return self._close_reason

    def on_close(self, cb: Callable[[str], None]) -> None:
        self._close_cbs.append(cb)

    def wait_closed(self, timeout: Optional[float] = None) -> bool:
        return self._closed.wait(timeout)

    def close(self) -> None:
        """Unsubscribe, unadvertise and close. Idempotent."""
        if self._ws is None or self._closed.is_set():
            self._closed.set()
            return
        self._user_closed = True
        for topic, subs in list(self._subs.items()):
            for sid, _ in subs:
                self._send_quiet({"op": "unsubscribe", "id": sid, "topic": topic})
        for topic in list(self._advertised):
            self._send_quiet({"op": "unadvertise", "topic": topic})
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass
        self._mark_closed("closed by the console")

    def _mark_closed(self, reason: str) -> None:
        with self._state_lock:
            if self._closed.is_set():
                return
            self._close_reason = reason
            self._closed.set()
            pending = list(self._pending.values())
            self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(TransportError(f"connection lost: {reason}"))
        if not self._user_closed:
            for cb in list(self._close_cbs):
                try:
                    cb(reason)
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------ io
    def _send(self, msg: dict) -> None:
        if not self.connected:
            raise TransportError(f"not connected to {self.url}" +
                                 (f" ({self._close_reason})" if self._close_reason else ""))
        data = json.dumps(msg, separators=(",", ":"), allow_nan=False)
        try:
            with self._send_lock:
                self._ws.send(data)
        except (ConnectionClosed, OSError, RuntimeError) as exc:
            self._mark_closed(f"send failed: {exc}")
            raise TransportError(f"send failed: {exc}") from None

    def _send_quiet(self, msg: dict) -> None:
        try:
            self._send(msg)
        except TransportError:
            pass

    def _read_loop(self) -> None:
        reason = "connection closed by the server"
        try:
            while True:
                raw = self._ws.recv()
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                self._dispatch(msg)
        except ConnectionClosed as exc:
            reason = f"connection closed ({exc.rcvd.code if exc.rcvd else 'no close frame'})"
        except Exception as exc:  # noqa: BLE001
            reason = f"connection failed: {exc}"
        self._mark_closed(reason)

    def _dispatch(self, msg: dict) -> None:
        op = msg.get("op")
        if op == "publish":
            for _, cb in list(self._subs.get(msg.get("topic"), ())):
                try:
                    cb(msg.get("msg") or {})
                except Exception:  # noqa: BLE001 - a bad frame must not kill the reader
                    pass
        elif op == "service_response":
            fut = self._pending.pop(str(msg.get("id")), None)
            if fut is not None and not fut.done():
                if msg.get("result", True) is False:
                    fut.set_exception(ServiceError(str(msg.get("values"))))
                else:
                    fut.set_result(msg.get("values") or {})
        elif op == "status":
            self.status_messages.append(msg)
            fut = self._pending.get(str(msg.get("id")))
            if fut is not None and msg.get("level") == "error" and not fut.done():
                self._pending.pop(str(msg.get("id")), None)
                fut.set_exception(ServiceError(str(msg.get("msg"))))

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}:{next(self._ids)}"

    # ------------------------------------------------------------ services
    def call_service_async(self, service: str, args: Optional[dict] = None,
                           type: Optional[str] = None) -> concurrent.futures.Future:
        cid = self._next_id("call")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending[cid] = fut
        msg = {"op": "call_service", "id": cid, "service": service, "args": args or {}}
        if type:
            msg["type"] = type
        try:
            self._send(msg)
        except TransportError as exc:
            self._pending.pop(cid, None)
            fut.set_exception(exc)
        return fut

    def call_service(self, service: str, args: Optional[dict] = None, *, timeout: float = 5.0,
                     type: Optional[str] = None) -> dict:
        fut = self.call_service_async(service, args, type)
        try:
            return fut.result(timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"{service} did not answer within {timeout:g} s") from None

    # ------------------------------------------------------------ topics
    def subscribe(self, topic: str, type: Optional[str], callback: Callable[[dict], None], *,
                  throttle_rate: int = 0, queue_length: int = 1) -> str:
        sid = self._next_id("sub")
        msg = {"op": "subscribe", "id": sid, "topic": topic,
               "throttle_rate": int(throttle_rate), "queue_length": int(queue_length)}
        if type:
            msg["type"] = type
        self._subs.setdefault(topic, []).append((sid, callback))
        self._send(msg)
        return sid

    def unsubscribe(self, sid: str) -> None:
        for topic, subs in list(self._subs.items()):
            keep = [s for s in subs if s[0] != sid]
            if len(keep) != len(subs):
                self._subs[topic] = keep
                self._send_quiet({"op": "unsubscribe", "id": sid, "topic": topic})

    def advertise(self, topic: str, type: str) -> None:
        if self._advertised.get(topic) == type:
            return
        self._send({"op": "advertise", "id": self._next_id("adv"), "topic": topic, "type": type})
        self._advertised[topic] = type

    def publish(self, topic: str, msg: Any, type: Optional[str] = None) -> None:
        """Send one message. Raises TransportError when it cannot be sent; a publish is
        never acknowledged by rosbridge, so success only means it left this process."""
        if type is not None:
            self.advertise(topic, type)
        self._send({"op": "publish", "topic": topic, "msg": msg})


def wait_until(pred: Callable[[], bool], timeout: float, step: float = 0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()
