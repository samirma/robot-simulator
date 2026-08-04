"""Decoding `sensor_msgs/CompressedImage`, and the hand-off to the render loop.

rosbridge transports a `uint8[]` field base64-encoded, so `data` arrives as an ASCII
string rather than a JSON array of integers. Decoding it as an array produces a message
that looks valid and renders as garbage.
"""

from __future__ import annotations

import base64
import binascii
import threading
import time
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np


def is_jpeg(fmt: object) -> bool:
    """Whether a CompressedImage `format` field describes JPEG.

    The simulator sends the bare string `"jpeg"`; a real `image_transport` republisher
    sends `"rgb8; jpeg compressed bgr8"`. Both are JPEG.
    """
    return isinstance(fmt, str) and "jpeg" in fmt.lower()


def decode_compressed_image(msg: Mapping) -> Optional[np.ndarray]:
    """Decode a CompressedImage message to a BGR ndarray, or None.

    Returns None rather than raising for every malformed input: one corrupt frame in a
    20 Hz stream must not take the teleop loop -- and therefore the watchdog feed --
    down with it.
    """
    if not isinstance(msg, Mapping):
        return None
    if not is_jpeg(msg.get("format")):
        return None
    data = msg.get("data")
    if not isinstance(data, str) or not data:
        return None
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    return frame


def header_seq(msg: Mapping) -> Optional[int]:
    """The `header.seq` counter, if present. Used to align video with the command log."""
    try:
        return int(msg["header"]["seq"])
    except (KeyError, TypeError, ValueError):
        return None


class LatestFrame:
    """A single-slot, newest-wins mailbox between the roslibpy thread and the loop.

    Deliberately not a Queue: if the render loop falls behind -- a window drag, a heavy
    scene -- a queue grows without bound and the operator ends up steering by video that
    is seconds old. Dropping stale frames keeps the feed honest, and `dropped` makes the
    drops visible instead of silent.

    `offer` is called on roslibpy's reactor thread and must stay O(1): no base64, no
    imdecode. Blocking that thread stalls `/odom` delivery and `/cmd_vel` egress too.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._msg: Optional[dict] = None
        self._at: float = 0.0
        self._dropped = 0
        self._received = 0
        self._last_arrival: Optional[float] = None
        self._interval: Optional[float] = None

    def offer(self, msg: dict) -> None:
        now = time.monotonic()
        with self._lock:
            if self._msg is not None:
                self._dropped += 1
            self._msg = msg
            self._at = now
            self._received += 1
            if self._last_arrival is not None:
                dt = now - self._last_arrival
                if dt > 0:
                    # EMA, so the reported rate settles quickly but survives one hiccup.
                    self._interval = dt if self._interval is None else 0.8 * self._interval + 0.2 * dt
            self._last_arrival = now

    def take(self) -> Optional[Tuple[dict, float]]:
        """Pop the pending message and its arrival time, or None if nothing is new."""
        with self._lock:
            if self._msg is None:
                return None
            msg, at = self._msg, self._at
            self._msg = None
            return msg, at

    @property
    def rate_hz(self) -> float:
        with self._lock:
            if not self._interval:
                return 0.0
            return 1.0 / self._interval

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def received(self) -> int:
        with self._lock:
            return self._received

    @property
    def last_arrival(self) -> Optional[float]:
        with self._lock:
            return self._last_arrival
