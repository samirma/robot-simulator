"""The simulation's private control-port protocol (`--sim-port`, default 9080).

One TCP connection carries framed messages both ways. A frame is

    u32 big-endian header length | UTF-8 JSON header | optional binary payload

where the payload is present when the header has `"nbytes": N` and is exactly N bytes.
Requests carry `"op"` and an `"id"`; the reply echoes the `id` with `"ok": true|false`
(and `"error"` when false). The server may also push `"event"` frames (samples of a
subscribed stream, `shutdown`, `removed`).

Nothing of this reaches a ROS wire. Stdlib only and Python 3.8-compatible: the spawn
client runs it under any python3 and the wire containers import it through the
read-only mount. See `simulator/README.md` for the operations.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any, Dict, Optional, Tuple

DEFAULT_SIM_PORT = 9080
MAX_HEADER = 16 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class RemoteError(RuntimeError):
    """The server answered `ok: false`."""


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def send(sock: socket.socket, header: Dict[str, Any], payload: Optional[bytes] = None,
         lock: Optional[threading.Lock] = None) -> None:
    if payload is not None:
        header = dict(header, nbytes=len(payload))
    raw = json.dumps(header, separators=(",", ":")).encode()
    parts = [struct.pack(">I", len(raw)), raw]
    if payload is not None:
        parts.append(payload)
    data = b"".join(parts) if payload is None or len(payload) < 65536 else None
    if lock is not None:
        lock.acquire()
    try:
        if data is not None:
            sock.sendall(data)
        else:
            sock.sendall(parts[0] + parts[1])
            sock.sendall(payload)
    finally:
        if lock is not None:
            lock.release()


def recv(sock: socket.socket) -> Tuple[Dict[str, Any], Optional[bytes]]:
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    if n > MAX_HEADER:
        raise ProtocolError(f"header of {n} bytes")
    header = json.loads(_recv_exact(sock, n).decode())
    payload = None
    if "nbytes" in header:
        payload = _recv_exact(sock, int(header["nbytes"]))
    return header, payload


class Client:
    """A blocking client with a reader thread: replies are matched by id; events go to
    `on_event(header, payload)` (called on the reader thread)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_SIM_PORT,
                 timeout: float = 5.0, on_event=None, on_close=None):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._wlock = threading.Lock()
        self._next = 0
        self._pending: Dict[int, list] = {}
        self._plock = threading.Lock()
        self.on_event = on_event
        self.on_close = on_close
        self.closed = threading.Event()
        self.close_reason = None
        self._reader = threading.Thread(target=self._read, daemon=True, name="simport-reader")
        self._reader.start()

    def _read(self) -> None:
        try:
            while True:
                header, payload = recv(self.sock)
                if "event" in header:
                    if self.on_event is not None:
                        self.on_event(header, payload)
                    continue
                with self._plock:
                    slot = self._pending.pop(header.get("id"), None)
                if slot is not None:
                    slot[1], slot[2] = header, payload
                    slot[0].set()
        except (ConnectionError, OSError, ProtocolError, ValueError) as exc:
            self.close_reason = str(exc)
        finally:
            self.closed.set()
            with self._plock:
                for slot in self._pending.values():
                    slot[0].set()
                self._pending.clear()
            if self.on_close is not None:
                try:
                    self.on_close()
                except Exception:
                    pass

    def request(self, op: str, payload: Optional[bytes] = None, timeout: Optional[float] = 60.0,
                **fields) -> Tuple[Dict[str, Any], Optional[bytes]]:
        with self._plock:
            self._next += 1
            rid = self._next
            slot = [threading.Event(), None, None]
            self._pending[rid] = slot
        send(self.sock, dict(fields, op=op, id=rid), payload, self._wlock)
        if not slot[0].wait(timeout):
            with self._plock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"no reply to {op} within {timeout} s")
        if slot[1] is None:
            raise ConnectionError(f"connection closed while waiting for {op}"
                                  + (f" ({self.close_reason})" if self.close_reason else ""))
        return slot[1], slot[2]

    def call(self, op: str, payload: Optional[bytes] = None, timeout: Optional[float] = 60.0,
             **fields) -> Dict[str, Any]:
        header, data = self.request(op, payload, timeout, **fields)
        if not header.get("ok", False):
            raise RemoteError(header.get("error", "request failed"))
        if data is not None:
            header["_payload"] = data
        return header

    def notify(self, op: str, payload: Optional[bytes] = None, **fields) -> None:
        """Send without waiting for (or getting) a reply."""
        send(self.sock, dict(fields, op=op, id=None), payload, self._wlock)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
