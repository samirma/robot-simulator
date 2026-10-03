"""Live camera streams: raw ``sensor_msgs/Image`` decoding and live/stale/failed state.

rosbridge carries ``uint8[] data`` base64-encoded. Frames are decoded to an RGB numpy array
(H, W, 3). A stream is *live* only while its newest decodable frame is younger than its
stale threshold; a frozen frame is never presented as live.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import threading
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: Raw encodings the console can display.
SUPPORTED_ENCODINGS = ("rgb8", "bgr8", "rgba8", "bgra8", "mono8", "mono16", "8uc1", "16uc1",
                       "8uc3", "yuv422", "uyvy", "yuv422_yuy2", "yuyv")

#: Console policy for streams not tied to a profile (no documented rate).
UNTIED_STALE_AFTER_S = 2.0


class FrameError(ValueError):
    """A frame that cannot be shown: carries the visible reason."""


def _yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    u = u.astype(np.float32) - 128.0
    v = v.astype(np.float32) - 128.0
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def decode_image(msg: dict, allowed: Optional[Sequence[str]] = None) -> np.ndarray:
    """A ``sensor_msgs/Image`` dict -> RGB uint8 array (H, W, 3); raises FrameError."""
    try:
        w, h, step = int(msg["width"]), int(msg["height"]), int(msg["step"])
        enc = str(msg["encoding"]).lower()
        data = msg["data"]
    except (KeyError, TypeError, ValueError):
        raise FrameError("malformed image message") from None
    if allowed is not None and enc not in [a.lower() for a in allowed]:
        raise FrameError(f"unsupported encoding '{enc}' (profile documents {', '.join(allowed)})")
    if enc not in SUPPORTED_ENCODINGS:
        raise FrameError(f"unsupported encoding '{enc}'")
    if isinstance(data, str):
        try:
            raw = base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError):
            raise FrameError("undecodable image data") from None
    elif isinstance(data, list):
        raw = bytes(data)
    else:
        raise FrameError("undecodable image data")
    if w <= 0 or h <= 0:
        raise FrameError("empty image")
    bpp = {"rgb8": 3, "bgr8": 3, "8uc3": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1,
           "mono16": 2, "16uc1": 2}.get(enc, 2)
    if step < w * bpp:
        raise FrameError(f"row step {step} shorter than {w} pixels of {enc}")
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size < step * h:
        raise FrameError(f"truncated image ({buf.size} of {step * h} bytes)")
    rows = buf[: step * h].reshape(h, step)
    big = bool(msg.get("is_bigendian"))
    if enc in ("rgb8", "bgr8", "8uc3"):
        img = rows[:, : w * 3].reshape(h, w, 3)
        return img[:, :, ::-1].copy() if enc == "bgr8" else img.copy()
    if enc in ("rgba8", "bgra8"):
        img = rows[:, : w * 4].reshape(h, w, 4)[:, :, :3]
        return img[:, :, ::-1].copy() if enc == "bgra8" else img.copy()
    if enc in ("mono8", "8uc1"):
        g = rows[:, :w]
        return np.repeat(g[:, :, None], 3, axis=2).copy()
    if enc in ("mono16", "16uc1"):
        vals = rows[:, : w * 2].copy().view(">u2" if big else "<u2").reshape(h, w).astype(np.float32)
        top = float(vals.max()) or 1.0
        g = np.clip(vals * (255.0 / top), 0, 255).astype(np.uint8)
        return np.repeat(g[:, :, None], 3, axis=2)
    # 4:2:2 packed: yuv422/uyvy = U Y0 V Y1 ; yuv422_yuy2/yuyv = Y0 U Y1 V
    if w % 2:
        raise FrameError("odd width for a 4:2:2 image")
    px = rows[:, : w * 2].reshape(h, w // 2, 4)
    if enc in ("yuv422", "uyvy"):
        u, y0, v, y1 = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
    else:
        y0, u, y1, v = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
    y = np.stack([y0, y1], axis=-1).reshape(h, w)
    u = np.repeat(u, 2, axis=1)
    v = np.repeat(v, 2, axis=1)
    return _yuv_to_rgb(y, u, v)


# ------------------------------------------------------------------ stream state

WAITING, LIVE, STALE, FAILED, UNSUPPORTED, MISSING = (
    "waiting", "live", "stale", "failed", "unsupported", "missing")


@dataclasses.dataclass
class StreamSpec:
    topic: str
    type: str
    stale_after_s: float
    encodings: Optional[Tuple[str, ...]] = None   # None: any supported (untied stream)
    tied: bool = True
    optional: bool = False


class CameraStream:
    """Latest frame of one subscribed stream, with its visible state.

    ``offer`` runs on the transport reader thread and only stores the message; decoding
    happens in ``poll`` on the consumer's thread.
    """

    def __init__(self, spec: StreamSpec, clock: Callable[[], float] = time.monotonic) -> None:
        self.spec = spec
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: Optional[dict] = None
        self._pending_at = 0.0
        self.frame: Optional[np.ndarray] = None
        self.frame_at: Optional[float] = None     # arrival time of the shown frame
        self.frames = 0
        self.error: Optional[str] = None
        self.unsupported = False
        self.missing = False
        self.sid: Optional[str] = None
        self.started = clock()

    def offer(self, msg: dict) -> None:
        with self._lock:
            self._pending = msg
            self._pending_at = self._clock()

    def poll(self) -> None:
        with self._lock:
            msg, at = self._pending, self._pending_at
            self._pending = None
        if msg is None:
            return
        try:
            self.frame = decode_image(msg, self.spec.encodings)
            self.frame_at = at
            self.frames += 1
            self.error = None
            self.unsupported = False
        except FrameError as exc:
            self.error = str(exc)
            self.unsupported = "unsupported encoding" in str(exc)
            self.frame = None      # never keep showing an older frame as current

    def state(self, now: Optional[float] = None) -> str:
        now = self._clock() if now is None else now
        if self.missing:
            return MISSING
        if self.unsupported:
            return UNSUPPORTED
        if self.error:
            return FAILED
        if self.frame_at is None:
            return STALE if now - self.started > self.spec.stale_after_s else WAITING
        return LIVE if now - self.frame_at <= self.spec.stale_after_s else STALE

    def status_text(self, now: Optional[float] = None) -> str:
        now = self._clock() if now is None else now
        st = self.state(now)
        if st == LIVE:
            return "live"
        if st == STALE:
            if self.frame_at is None:
                return f"stale: no frame within {self.spec.stale_after_s:g} s"
            return f"stale: last frame {now - self.frame_at:.1f} s ago"
        if st == MISSING:
            return "unavailable: the topic is not on the wire"
        if st in (FAILED, UNSUPPORTED):
            return f"{st}: {self.error}"
        return "waiting for the first frame"


class CameraSet:
    """Subscribe to a set of streams on one connection; poll and close together."""

    def __init__(self, rb, specs: Sequence[StreamSpec], *, present: Optional[Dict[str, str]] = None) -> None:
        self.rb = rb
        self.streams: Dict[str, CameraStream] = {}
        for s in specs:
            cs = CameraStream(s)
            self.streams[s.topic] = cs
            if present is not None and s.topic not in present:
                cs.missing = True
                continue
            cs.sid = rb.subscribe(s.topic, s.type, cs.offer, queue_length=1)

    def poll(self) -> None:
        for s in self.streams.values():
            s.poll()

    def close(self) -> None:
        for s in self.streams.values():
            if s.sid:
                self.rb.unsubscribe(s.sid)
                s.sid = None


def target_specs(target) -> list:
    """Stream specs of a validated target's profile cameras (resolved names)."""
    return [StreamSpec(topic=target.wire(c.topic), type=c.type, stale_after_s=c.stale_after_s,
                       encodings=c.encodings or None, tied=True, optional=c.optional)
            for c in target.profile.cameras]


def untied_specs(graph) -> list:
    """Every supported raw image stream on the wire, not tied to any target."""
    return [StreamSpec(topic=n, type=t, stale_after_s=UNTIED_STALE_AFTER_S, encodings=None, tied=False)
            for n, t in sorted(graph.cameras().items())]
