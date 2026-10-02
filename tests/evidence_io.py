"""Transport helpers of the workspace evidence script (`tests/evidence.py`).

Written for the workspace, independently of both projects (workspace spec §1.4): a
rosbridge v2 client over `websocket-client`, a client of the simulation's private control
port (`--sim-port`; framing documented in `simulator/README.md`), and the decoding of
`sensor_msgs/Image` messages taken off a vendor wire into PNG pictures.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import math
import queue
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import websocket  # websocket-client
from PIL import Image


# --------------------------------------------------------------------------- rosbridge


class Rosbridge:
    """rosbridge v2 protocol (ROS 1 rosbridge 0.11 and ROS 2 rosbridge 2.x) over one
    websocket, with a reader thread."""

    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self.ws = websocket.create_connection(url, timeout=timeout,
                                              enable_multithread=True)
        self.ws.settimeout(None)
        self._ids = itertools.count(1)
        self._pending: Dict[str, "queue.Queue"] = {}
        self._subs: Dict[str, Callable[[dict], None]] = {}
        self.closed = threading.Event()
        self._reader = threading.Thread(target=self._loop, daemon=True)
        self._reader.start()

    def _loop(self):
        try:
            while True:
                data = self.ws.recv()
                if not data:
                    break
                msg = json.loads(data)
                op = msg.get("op")
                if op in ("service_response", "action_result"):
                    q = self._pending.pop(msg.get("id"), None)
                    if q is not None:
                        q.put(msg)
                elif op == "publish":
                    cb = self._subs.get(msg.get("topic"))
                    if cb is not None:
                        try:
                            cb(msg["msg"])
                        except Exception:  # a callback error must not end the reader
                            pass
                elif op == "status" and msg.get("id") in self._pending:
                    self._pending.pop(msg["id"]).put(msg)
        except Exception:
            pass
        finally:
            self.closed.set()
            for q in list(self._pending.values()):
                q.put(None)

    def send(self, msg: dict) -> None:
        self.ws.send(json.dumps(msg))

    def call(self, service: str, args: Optional[dict] = None, timeout: float = 15.0,
             check: bool = True) -> dict:
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
            raise ConnectionError("rosbridge connection closed")
        if check and not res.get("result", True):
            raise RuntimeError(f"{service} failed: {res.get('values')}")
        return res.get("values") or {}

    def action(self, action: str, action_type: str, goal: dict, timeout: float = 30.0) -> dict:
        """A ROS 2 action goal through rosbridge 2.x `send_action_goal`; the result message."""
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
            raise ConnectionError("rosbridge connection closed")
        return res

    def subscribe(self, topic: str, cb: Callable[[dict], None], msg_type: Optional[str] = None,
                  throttle_rate: int = 0, queue_length: int = 1):
        self._subs[topic] = cb
        m = {"op": "subscribe", "topic": topic, "id": f"sub:{topic}",
             "throttle_rate": throttle_rate, "queue_length": queue_length}
        if msg_type:
            m["type"] = msg_type
        self.send(m)

    def unsubscribe(self, topic: str):
        self._subs.pop(topic, None)
        self.send({"op": "unsubscribe", "topic": topic, "id": f"sub:{topic}"})

    def advertise(self, topic: str, msg_type: str):
        self.send({"op": "advertise", "topic": topic, "type": msg_type, "id": f"adv:{topic}"})

    def unadvertise(self, topic: str):
        self.send({"op": "unadvertise", "topic": topic, "id": f"adv:{topic}"})

    def publish(self, topic: str, msg: dict):
        self.send({"op": "publish", "topic": topic, "msg": msg})

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


class Latest:
    """The latest message of one topic, with a counter and its arrival time."""

    def __init__(self, rb: Rosbridge, topic: str, msg_type: Optional[str] = None,
                 throttle_rate: int = 0):
        self.msg = None
        self.count = 0
        self.t = 0.0
        self.history: list = []
        self.keep = False
        self._ev = threading.Event()
        rb.subscribe(topic, self._cb, msg_type=msg_type, throttle_rate=throttle_rate)

    def _cb(self, m):
        self.msg = m
        self.count += 1
        self.t = time.monotonic()
        if self.keep:
            self.history.append((self.t, m))
        self._ev.set()

    def wait(self, timeout: float = 15.0):
        if not self._ev.wait(timeout):
            raise TimeoutError("no message")
        return self.msg

    def next(self, timeout: float = 5.0):
        """The first message arriving after this call."""
        n = self.count
        t0 = time.monotonic()
        while self.count == n:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("no fresh message")
            time.sleep(0.01)
        return self.msg


# --------------------------------------------------------------------------- sim-port


class SimPort:
    """The simulation's private control port: `u32 BE header length | JSON header |
    payload (header nbytes)`; replies echo the request id (simulator/README.md)."""

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.settimeout(120)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise ConnectionError("sim-port closed")
            buf += chunk
        return bytes(buf)

    def call(self, op: str, **fields) -> dict:
        with self._lock:
            rid = next(self._ids)
            raw = json.dumps(dict(fields, op=op, id=rid)).encode()
            self.sock.sendall(struct.pack(">I", len(raw)) + raw)
            while True:
                (n,) = struct.unpack(">I", self._recv_exact(4))
                header = json.loads(self._recv_exact(n).decode())
                payload = self._recv_exact(int(header["nbytes"])) if "nbytes" in header else None
                if "event" in header or header.get("id") != rid:
                    continue
                if not header.get("ok", False):
                    raise RuntimeError(f"sim-port {op}: {header.get('error')}")
                if payload is not None:
                    header["_payload"] = payload
                return header

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- images


def _data_bytes(msg: dict) -> bytes:
    d = msg["data"]
    if isinstance(d, str):
        return base64.b64decode(d)
    return bytes(d)


def colourize_depth(depth: np.ndarray) -> np.ndarray:
    """A float depth map (m; 0/nan = no return) as a turbo-like RGB picture (near warm,
    far cool), black where there is no return."""
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros(depth.shape + (3,), np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(depth[valid], [1, 99])
    hi = max(hi, lo + 1e-3)
    t = np.clip((np.nan_to_num(depth) - lo) / (hi - lo), 0, 1)
    # a polynomial approximation of the "turbo" colour map, reversed so near is red and
    # far dark blue (kept clear of turbo's near-black end, which marks no return here)
    tt = 0.1 + 0.9 * (1.0 - t)
    r = 0.13572138 + tt * (4.61539260 + tt * (-42.66032258 + tt * (132.13108234 + tt * (-152.94239396 + tt * 59.28637943))))
    g = 0.09140261 + tt * (2.19418839 + tt * (4.84296658 + tt * (-14.18503333 + tt * (4.27729857 + tt * 2.82956604))))
    b = 0.10667330 + tt * (12.64194608 + tt * (-60.58204836 + tt * (110.36276771 + tt * (-89.90310912 + tt * 27.34824973))))
    rgb = np.clip(np.stack([r, g, b], -1), 0, 1)
    out[valid] = (rgb[valid] * 255).astype(np.uint8)
    return out


def decode_image(msg: dict) -> Dict[str, Any]:
    """A sensor_msgs/Image (as rosbridge JSON) as an RGB array, with its statistics."""
    h, w, enc = int(msg["height"]), int(msg["width"]), msg["encoding"]
    step = int(msg.get("step") or 0)
    big = bool(msg.get("is_bigendian"))
    raw = _data_bytes(msg)
    info: Dict[str, Any] = {"width": w, "height": h, "encoding": enc, "step": step,
                            "nbytes": len(raw),
                            "frame_id": (msg.get("header") or {}).get("frame_id")}

    def rows(bpp: int) -> np.ndarray:
        s = step or w * bpp
        a = np.frombuffer(raw, np.uint8)[: s * h].reshape(h, s)
        return a[:, : w * bpp]

    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        ch = 4 if enc.endswith("a8") else 3
        a = rows(ch).reshape(h, w, ch)[..., :3]
        rgb = a[..., ::-1] if enc.startswith("bgr") else a
    elif enc in ("mono8", "8UC1"):
        g = rows(1).reshape(h, w)
        rgb = np.repeat(g[..., None], 3, axis=2)
    elif enc in ("yuv422_yuy2", "yuyv", "yuv422"):
        a = rows(2).reshape(h, w // 2, 4).astype(np.float32)
        if enc == "yuv422":       # UYVY
            u, y0, v, y1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        else:                     # YUYV
            y0, u, y1, v = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        y = np.stack([y0, y1], -1).reshape(h, w)
        u = np.repeat(u, 2, axis=1) - 128.0
        v = np.repeat(v, 2, axis=1) - 128.0
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
        rgb = np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)
    elif enc in ("16UC1", "mono16"):
        dt = ">u2" if big else "<u2"
        a = np.frombuffer(rows(2).tobytes(), dt).reshape(h, w).astype(np.float32)
        info["min"], info["max"] = float(a.min()), float(a.max())
        if enc == "16UC1":        # depth in millimetres (REP 118)
            info["valid_fraction"] = float((a > 0).mean())
            rgb = colourize_depth(a / 1000.0)
        else:                     # 16-bit intensity: stretch to 8 bits
            lo, hi = np.percentile(a, [0.5, 99.5])
            g = np.clip((a - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
            rgb = np.repeat(g[..., None], 3, axis=2)
    elif enc == "32FC1":
        dt = ">f4" if big else "<f4"
        a = np.frombuffer(rows(4).tobytes(), dt).reshape(h, w)
        fin = np.isfinite(a)
        info["valid_fraction"] = float((fin & (a > 0)).mean())
        info["min"] = float(a[fin].min()) if fin.any() else None
        info["max"] = float(a[fin].max()) if fin.any() else None
        rgb = colourize_depth(a.astype(np.float32))
    else:
        raise ValueError(f"unsupported encoding {enc}")
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    info["mean_rgb"] = [round(float(x), 1) for x in rgb.reshape(-1, 3).mean(0)]
    info["std"] = round(float(rgb.std()), 2)
    return {"rgb": rgb, "info": info}


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def png_stats(data: bytes) -> dict:
    im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return {"width": int(im.shape[1]), "height": int(im.shape[0]),
            "std": round(float(im.std()), 2)}


# --------------------------------------------------------------------------- geometry


def yaw_of_quat_xyzw(q: dict) -> float:
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]),
                      1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


def yaw_of_wxyz(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
