"""A minimal rosbridge client over a stdlib WebSocket (RFC 6455): enough to call rosapi
services, subscribe, publish and advertise. Used by `spawn.sh` to confirm and watch its
wires, and by the tests. Stdlib only, Python 3.8-compatible.
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import queue
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, Optional


class WebSocketClosed(ConnectionError):
    pass


class WebSocket:
    def __init__(self, host: str, port: int, path: str = "/", timeout: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WebSocketClosed("handshake failed: connection closed")
            resp += chunk
        head, _, self._buf = resp.partition(b"\r\n\r\n")
        status = head.split(b"\r\n")[0]
        if b" 101 " not in status:
            raise WebSocketClosed("handshake failed: %r" % status)
        self.sock.settimeout(None)
        self._wlock = threading.Lock()

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self._buf)))
            if not chunk:
                raise WebSocketClosed("connection closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def send_text(self, text: str) -> None:
        self._send(0x1, text.encode())

    def _send(self, opcode: int, payload: bytes) -> None:
        head = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload)) if n < 4096 else \
            _mask_fast(payload, mask)
        with self._wlock:
            self.sock.sendall(bytes(head) + body)

    def recv(self):
        """(opcode, payload) of the next complete message; pings answered."""
        chunks, first_op = [], None
        while True:
            b0, b1 = self._read(2)
            fin, op = b0 & 0x80, b0 & 0x0F
            n = b1 & 0x7F
            if n == 126:
                (n,) = struct.unpack(">H", self._read(2))
            elif n == 127:
                (n,) = struct.unpack(">Q", self._read(8))
            if b1 & 0x80:
                mask = self._read(4)
                data = _mask_fast(self._read(n), mask)
            else:
                data = self._read(n)
            if op == 0x8:
                raise WebSocketClosed("closed by peer")
            if op == 0x9:
                self._send(0xA, data)
                continue
            if op == 0xA:
                continue
            if first_op is None:
                first_op = op
            chunks.append(data)
            if fin:
                return first_op, b"".join(chunks)

    def close(self):
        try:
            self._send(0x8, b"")
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def _mask_fast(data: bytes, mask: bytes) -> bytes:
    if not data:
        return data
    n = len(data)
    m = (mask * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(m, "little")).to_bytes(n, "little")


def cbor_decode(buf: bytes, i: int):
    """(value, next index) of the CBOR item at buf[i] (RFC 8949: the subset rosbridge
    emits -- integers, byte/text strings, arrays, maps, tags, floats, simple values)."""
    ib = buf[i]
    major, info = ib >> 5, ib & 0x1F
    i += 1
    if info < 24:
        arg = info
    elif info == 24:
        arg, i = buf[i], i + 1
    elif info == 25:
        arg, i = struct.unpack_from(">H", buf, i)[0], i + 2
    elif info == 26:
        if major == 7:
            return struct.unpack_from(">f", buf, i)[0], i + 4
        arg, i = struct.unpack_from(">I", buf, i)[0], i + 4
    elif info == 27:
        if major == 7:
            return struct.unpack_from(">d", buf, i)[0], i + 8
        arg, i = struct.unpack_from(">Q", buf, i)[0], i + 8
    else:
        raise ValueError("indefinite-length CBOR is not supported")
    if major == 0:
        return arg, i
    if major == 1:
        return -1 - arg, i
    if major == 2:
        return bytes(buf[i:i + arg]), i + arg
    if major == 3:
        return buf[i:i + arg].decode("utf-8", "replace"), i + arg
    if major == 4:
        out = []
        for _ in range(arg):
            v, i = cbor_decode(buf, i)
            out.append(v)
        return out, i
    if major == 5:
        out = {}
        for _ in range(arg):
            k, i = cbor_decode(buf, i)
            v, i = cbor_decode(buf, i)
            out[k] = v
        return out, i
    if major == 6:   # tag (typed arrays): return the tagged item
        return cbor_decode(buf, i)
    if major == 7:
        if info == 25:
            return float(arg), i   # half floats are not produced by rosbridge
        return {20: False, 21: True, 22: None}.get(arg), i
    raise ValueError(f"bad CBOR major type {major}")


class Rosbridge:
    """rosbridge v2 protocol over one websocket, with a reader thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9090, timeout: float = 5.0):
        self.ws = WebSocket(host, port, timeout=timeout)
        self._ids = itertools.count(1)
        self._pending: Dict[str, "queue.Queue"] = {}
        self._subs: Dict[str, Callable[[dict], None]] = {}
        self._lock = threading.Lock()
        self.closed = threading.Event()
        self.reader = threading.Thread(target=self._loop, daemon=True, name="rosbridge")
        self.reader.start()

    def _loop(self):
        try:
            while True:
                op, data = self.ws.recv()
                if op == 0x2:   # CBOR (compression "cbor" / "cbor-raw")
                    msg, _ = cbor_decode(data, 0)
                else:
                    msg = json.loads(data.decode())
                kind = msg.get("op")
                if kind in ("service_response", "action_result"):
                    q = self._pending.pop(msg.get("id"), None)
                    if q is not None:
                        q.put(msg)
                elif kind == "publish":
                    cb = self._subs.get(msg.get("topic"))
                    if cb is not None:
                        cb(msg)
                elif kind == "status" and msg.get("id") in self._pending:
                    self._pending.pop(msg["id"]).put(msg)
        except (WebSocketClosed, OSError, ValueError):
            pass
        finally:
            self.closed.set()
            for q in list(self._pending.values()):
                q.put(None)

    def send(self, msg: dict) -> None:
        self.ws.send_text(json.dumps(msg))

    def call(self, service: str, args: Optional[dict] = None, timeout: float = 10.0) -> dict:
        rid = f"call:{next(self._ids)}"
        q: "queue.Queue" = queue.Queue(1)
        self._pending[rid] = q
        self.send({"op": "call_service", "service": service, "args": args or {}, "id": rid})
        try:
            res = q.get(timeout=timeout)
        except queue.Empty:
            self._pending.pop(rid, None)
            raise TimeoutError(f"{service} did not answer within {timeout} s")
        if res is None:
            raise WebSocketClosed("rosbridge connection closed")
        if not res.get("result", True):
            raise RuntimeError(f"{service} failed: {res.get('values')}")
        return res.get("values") or {}

    def action(self, action: str, action_type: str, goal: dict, timeout: float = 30.0) -> dict:
        """Send a ROS 2 action goal and wait for its result message (rosbridge 2.x)."""
        rid = f"action:{next(self._ids)}"
        q: "queue.Queue" = queue.Queue(1)
        self._pending[rid] = q
        self.send({"op": "send_action_goal", "action": action, "action_type": action_type,
                   "args": goal, "id": rid, "feedback": False})
        try:
            res = q.get(timeout=timeout)
        except queue.Empty:
            self._pending.pop(rid, None)
            raise TimeoutError(f"{action} gave no result within {timeout} s")
        if res is None:
            raise WebSocketClosed("rosbridge connection closed")
        return res

    def subscribe(self, topic: str, callback: Callable[[dict], None], msg_type: str = None,
                  throttle_rate: int = 0, queue_length: int = 0, compression: str = "none"):
        self._subs[topic] = callback
        msg = {"op": "subscribe", "topic": topic, "id": f"sub:{topic}",
               "throttle_rate": throttle_rate, "queue_length": queue_length,
               "compression": compression}
        if msg_type:
            msg["type"] = msg_type
        self.send(msg)

    def unsubscribe(self, topic: str):
        self._subs.pop(topic, None)
        self.send({"op": "unsubscribe", "topic": topic, "id": f"sub:{topic}"})

    def advertise(self, topic: str, msg_type: str):
        self.send({"op": "advertise", "topic": topic, "type": msg_type, "id": f"adv:{topic}"})

    def publish(self, topic: str, msg: dict):
        self.send({"op": "publish", "topic": topic, "msg": msg})

    def close(self):
        self.ws.close()


def wait_for(host: str, port: int, timeout: float) -> Rosbridge:
    """A connected client, retrying until rosbridge accepts within the timeout."""
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            return Rosbridge(host, port, timeout=3.0)
        except (OSError, WebSocketClosed) as exc:
            last = exc
            time.sleep(0.5)
    raise TimeoutError(f"no rosbridge on {host}:{port} within {timeout:.0f} s ({last})")
