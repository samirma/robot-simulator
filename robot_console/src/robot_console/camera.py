"""Live camera streams: raw ``sensor_msgs/Image`` decoding and live/stale/failed state.

rosbridge carries ``uint8[] data`` base64-encoded. Frames are decoded to an RGB numpy array
(H, W, 3). A stream is *live* only while the image it shows is, in robot time, at most its
live bound old (``live_bound_s``); a frozen frame is never presented as live.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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


#: The depth false-colour map: (position 0 = near ... 1 = far, RGB). The page (``console.js``) carries
#: the same stops, so both entry points draw a depth image alike.
DEPTH_STOPS = ((0.0, (213, 32, 39)), (0.17, (244, 125, 32)), (0.34, (250, 224, 50)), (0.5, (80, 200, 100)),
               (0.67, (40, 180, 220)), (0.84, (50, 90, 210)), (1.0, (70, 40, 140)))
#: Colour of a pixel with no depth (value 0).
NO_DEPTH_RGB = (0, 0, 0)


def depth_lut(n: int = 256) -> np.ndarray:
    """The map as ``n`` RGB rows, near first (float stops interpolated, rounded to uint8)."""
    pos = [p for p, _ in DEPTH_STOPS]
    xs = np.linspace(0.0, 1.0, n)
    return np.stack([np.round(np.interp(xs, pos, [c[i] for _, c in DEPTH_STOPS])) for i in range(3)],
                    axis=-1).astype(np.uint8)


def depth_to_rgb(values: np.ndarray, unit_m: float, range_m: Tuple[float, float]) -> np.ndarray:
    """Depth values (H, W; each times ``unit_m`` is metres, 0 = no depth) -> RGB (H, W, 3): the
    false-colour map scaled to ``range_m`` (near, far), nearer or farther values at its ends."""
    lut = depth_lut()
    near, far = range_m
    t = np.clip((values.astype(np.float64) * unit_m - near) / (far - near), 0.0, 1.0)
    out = lut[np.floor(t * 255.0 + 0.5).astype(np.intp)]
    out[values == 0] = NO_DEPTH_RGB
    return out


def stretch(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Grey levels 0..255 of ``values`` stretched from ``low`` (black) to ``high`` (white), rounded; all
    black when the frame has a single value (as OpenCV's NORM_MINMAX normalisation)."""
    if high <= low:
        return np.zeros(values.shape, np.uint8)
    return np.clip(np.round((values.astype(np.float64) - low) * (255.0 / (high - low))), 0, 255).astype(np.uint8)


COMPRESSED_FORMATS = ("jpeg", "png")


def decode_compressed(msg: dict, allowed: Optional[Sequence[str]] = None) -> np.ndarray:
    """A ``sensor_msgs/CompressedImage`` dict -> RGB uint8 array (H, W, 3); raises FrameError. ``format`` names the
    codec (image_transport writes e.g. "rgb8; jpeg compressed bgr8"); ``allowed`` the codecs the camera documents.
    Decoded with pygame (a console dependency)."""
    fmt = str(msg.get("format", "")).lower()
    codec = next((k for k in COMPRESSED_FORMATS if k in fmt), None)
    if codec is None or (allowed and codec not in [a.lower() for a in allowed]):
        raise FrameError(f"unsupported format '{msg.get('format', '')}'"
                         + (f" (profile documents {', '.join(allowed)})" if allowed else ""))
    data = msg.get("data")
    try:
        raw = base64.b64decode(data, validate=False) if isinstance(data, str) else bytes(data or [])
    except (binascii.Error, ValueError, TypeError):
        raise FrameError("undecodable image data") from None
    import io
    import pygame
    try:
        surf = pygame.image.load(io.BytesIO(raw), f"frame.{'jpg' if codec == 'jpeg' else 'png'}")
    except Exception:  # noqa: BLE001
        raise FrameError(f"undecodable {codec} image") from None
    return np.ascontiguousarray(pygame.surfarray.array3d(surf).swapaxes(0, 1))


def decode_image(msg: dict, allowed: Optional[Sequence[str]] = None,
                 depth: Optional[Tuple[float, Tuple[float, float]]] = None, ir: bool = False) -> np.ndarray:
    """A ``sensor_msgs/Image`` dict -> RGB uint8 array (H, W, 3); raises FrameError.

    ``depth`` = (``unit_m``, ``range_m``) is a depth camera's scale: a 16-bit frame is then drawn as the
    false-colour map over that range instead of being stretched to its own maximum. ``ir``: an infrared
    camera's single-channel frame is drawn in grey stretched from its own minimum to its own maximum. A
    ``sensor_msgs/CompressedImage`` (a ``format`` and no ``encoding``) is decoded by ``decode_compressed``."""
    if isinstance(msg, dict) and "format" in msg and "encoding" not in msg:
        return decode_compressed(msg, allowed)
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
        if ir:
            g = stretch(g, float(g.min()), float(g.max()))
        return np.repeat(g[:, :, None], 3, axis=2).copy()
    if enc in ("mono16", "16uc1"):
        raw16 = rows[:, : w * 2].copy().view(">u2" if big else "<u2").reshape(h, w)
        if depth is not None:
            return depth_to_rgb(raw16, *depth)
        if ir:
            return np.repeat(stretch(raw16, float(raw16.min()), float(raw16.max()))[:, :, None], 3, axis=2)
        vals = raw16.astype(np.float32)
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

WAITING, LIVE, STALE, FAILED, UNSUPPORTED, MISSING, OFF, SILENT, LAGGING, UNKNOWN, PAUSED = (
    "waiting", "live", "stale", "failed", "unsupported", "missing", "off", "silent", "lagging", "unknown", "paused")

#: Console policy (console spec §2.3): a frame that arrived more than this after its stamp (robot time) is LAGGING.
LAG_LIMIT_S = 1.0
#: A shown image is live while its age (robot time, judged on every refresh) is at most LAG_LIMIT_S plus this many
#: frame periods at the stream's rate cap (``live_bound_s``): a steady lag below LAG_LIMIT_S never flickers between
#: frames. A frame is shown only once a time answer has read the robot's clock after the frame was taken (its question
#: sent after the frame arrived, or the time it read not before the frame's stamp: it is "confirmed"); its age is bounded with the lowest and highest offset of every answer from CLOCK_STALE_S before it
#: arrived until now (``ClockRef.span``), so a robot clock step before, while or after it was taken widens the bounds
#: and never makes an old image read live, as long as the clock's reading when it was taken is one an answer in
#: that window saw (see the console spec §2.3 for the one case left).
LIVE_PERIODS = 1.5
#: A frame stamped more than this after its own arrival (robot time) is impossible: the robot's clock stepped.
STEP_TOLERANCE_S = 0.25
#: The clock reference: /rosapi/get_time is asked every CLOCK_SYNC_S; a reference older than CLOCK_STALE_S is not
#: used; samples older than CLOCK_WINDOW_S are dropped (clock drift); an answer later than CLOCK_ANSWER_S is given up.
CLOCK_SYNC_S = 2.0
CLOCK_STALE_S = 10.0
CLOCK_WINDOW_S = 300.0
CLOCK_ANSWER_S = 3.0
#: Congestion (console spec §2.3; console.js congestion() applies the same three rules):
#:   R1 a stream LAGGING while the newest time answer took at least CLOCK_DOMAIN_RTT_S (answers share the link), once
#:      its lags span 0.8 CLOCK_DOMAIN_SPAN_S and show no steady offset (a steady offset is not a backlog);
#:   R2 a time answer took more than CONGESTED_RTT_S (answered within the last CONGESTED_WINDOW_S);
#:   R3 a time question unanswered for more than CLOCK_ANSWER_S, now or given up within the last CLOCK_ANSWER_S,
#:      while frames arrived within the last second.
#: A question sent inside a grace window (a large message the operator asked for, such as a point-cloud sample) is
#: not counted by R2 or R3, nor a frame that arrived inside it by R1.
CONGESTED_RTT_S = 1.0
CONGESTED_WINDOW_S = 5.0
#: A steady lag (within CLOCK_DOMAIN_SPREAD_S over CLOCK_DOMAIN_SPAN_S, at least 5 frames, beyond CLOCK_DOMAIN_MIN_S,
#: while the newest time answer took at most CONGESTED_RTT_S and less than half that lag: not a queue the answers
#: share) is either a real delay or stamps on another clock: it stays LAGGING, with that note, and R1 skips it.
CLOCK_DOMAIN_SPREAD_S = 0.25
CLOCK_DOMAIN_SPAN_S = 3.0
CLOCK_DOMAIN_MIN_S = 2.0
CLOCK_DOMAIN_RTT_S = 0.3
#: Console policy: ``throttle_rate`` of a profile camera without ``max_rate_hz`` (10 frames per second,
#: profiles.DEFAULT_CAMERA_MAX_HZ).
DEFAULT_THROTTLE_MS = 100
#: Console policy: a stream not tied to a target is a preview, at most 2 frames per second, one at a time.
UNTIED_THROTTLE_MS = 500


def live_bound_s(throttle_ms: int) -> float:
    """The age up to which a shown image of a stream capped at ``throttle_ms`` reads live."""
    return LAG_LIMIT_S + LIVE_PERIODS * throttle_ms / 1000.0




def stamp_s(msg: dict) -> Optional[float]:
    """The header stamp of a ROS 1 (secs/nsecs) or ROS 2 (sec/nanosec) message in seconds; None if absent or 0."""
    st = (msg.get("header") or {}).get("stamp") if isinstance(msg, dict) else None
    return time_s(st)


def time_s(st: Any) -> Optional[float]:
    """A ROS time {secs, nsecs} / {sec, nanosec} in seconds; None if absent or 0 (no time)."""
    if not isinstance(st, dict):
        return None
    sec, ns = st.get("secs", st.get("sec")), st.get("nsecs", st.get("nanosec", 0))
    if not isinstance(sec, (int, float)) or not isinstance(ns, (int, float)) or (sec == 0 and ns == 0):
        return None
    return float(sec) + float(ns) * 1e-9


class ClockRef:
    """The robot's clock against the console's monotonic clock, NTP-style, from /rosapi/get_time answers.

    A call sent at console time t0 and answered at t1 with robot time T bounds the offset (robot - console) to
    [T - t1, T - t0], whatever the delays on the way (only their sum, the round trip, matters). The estimate
    is the intersection of the recent samples' bounds. A sample whose bounds exclude the estimate's midpoint is
    taken as a clock step (also one hidden by a backlog's wide bounds): the samples before it are dropped. A
    delayed answer only widens its own bounds. With no usable answer (rosapi's time absent or not answering, a
    ROS 1 robot on simulated time, a reference older than CLOCK_STALE_S) the lag is unknown, never zero."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, dialect: Optional[str] = None) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self.dialect = dialect
        self.samples: list = []           # (t1, lo, hi)
        self.lo = self.hi = None
        self.last_ok: Optional[float] = None
        self.last_t0 = float("-inf")      # when the newest sample's question was sent
        self.last_rtt: Optional[float] = None
        self.last_rtt_at: Optional[float] = None   # when that answer came
        self.last_rtt_t0: Optional[float] = None   # when its question was sent
        self.last_timeout: Optional[float] = None  # when a question was last given up unanswered
        self.last_timeout_t0: Optional[float] = None
        self.grace: Tuple[float, float] = (float("-inf"), float("-inf"))   # see CONGESTED_* (a sample in flight)
        self.steps = 0
        self.resets = 0
        self.answers: list = []           # every answer of the last CLOCK_WINDOW_S, resets or not: (t1, lo, hi, T)
        # ROS 1: /use_sim_time, read through rosapi before any time is trusted (None: not known yet)
        self.sim: Optional[bool] = None if dialect == "ros1" else False
        self.reason = "measuring the robot's clock"
        self._inflight = False
        self._last_sent = float("-inf")

    def add(self, t0: float, t1: float, robot: Optional[float]) -> None:
        with self._lock:
            self.last_rtt, self.last_rtt_at, self.last_rtt_t0 = t1 - t0, t1, t0
            if robot is None:
                self.reason = "the robot's clock reads 0 (simulated time not started)"
                return
            lo, hi = robot - t1, robot - t0
            self.answers = [x for x in self.answers if t1 - x[0] <= CLOCK_WINDOW_S] + [(t1, lo, hi, robot)]
            keep = [x for x in self.samples if t1 - x[0] <= CLOCK_WINDOW_S]
            ilo, ihi = max([lo] + [x[1] for x in keep]), min([hi] + [x[2] for x in keep])
            mid = None if self.lo is None else (self.lo + self.hi) / 2
            if ilo > ihi or (mid is not None and not lo <= mid <= hi):
                # bounds that exclude every recent sample prove a clock step; ones that only exclude the estimate's
                # midpoint may hide one behind a wide answer: either way the older samples are dropped
                self.steps += ilo > ihi
                self.resets += 1
                keep, ilo, ihi = [], lo, hi
            self.samples = keep + [(t1, lo, hi)]
            self.lo, self.hi, self.last_ok, self.last_t0 = ilo, ihi, t1, t0

    def fail(self, reason: str) -> None:
        with self._lock:
            self.reason = reason

    def offset(self, now: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """(lo, hi) estimate of robot - console (the recent answers' intersection), or None when there is no usable
        reference. A frame's age is bounded with ``span`` instead, which holds whatever the clock did."""
        now = self._clock() if now is None else now
        with self._lock:
            if self.sim:
                self.reason = "the robot runs on simulated time (/use_sim_time)"
                return None
            if self.sim is None:
                self.reason = "checking whether the robot runs on simulated time (/use_sim_time)"
                return None
            if self.last_ok is None or now - self.last_ok > CLOCK_STALE_S:
                if self.last_ok is not None:
                    self.reason = f"the clock reference is older than {CLOCK_STALE_S:g} s"
                return None
            return self.lo, self.hi

    def confirms(self, arrival: float, stamp: float) -> bool:
        """Whether an answer read the robot's clock after a frame (arrived at ``arrival``, stamped ``stamp``) was taken:
        its question was sent after the frame arrived, or the time it read is not before the stamp (read earlier, it
        would be earlier, unless the clock stepped back since, which only raises the age's upper bound)."""
        with self._lock:
            return self.last_t0 >= arrival or any(
                x[3] >= stamp for x in self.answers if x[0] >= arrival - CLOCK_STALE_S)

    def span(self, arrival: float) -> Tuple[float, float]:
        """(lowest lo, highest hi) of robot - console over every answer from CLOCK_STALE_S before ``arrival`` until now:
        whatever the clock did, it holds the offset of a frame taken while any of those answers saw its reading."""
        with self._lock:
            got = [x for x in self.answers if x[0] >= arrival - CLOCK_STALE_S] or self.answers[-1:]
            return min(x[1] for x in got), max(x[2] for x in got)

    def unanswered_for(self, now: Optional[float] = None) -> float:
        """How long the question in flight has gone unanswered (0 when none is)."""
        now = self._clock() if now is None else now
        return now - self._last_sent if self._inflight else 0.0

    def maybe_sync(self, rb, force: bool = False) -> None:
        """Ask the robot's time without blocking (``rb.call_service_async``) every CLOCK_SYNC_S, or now when
        ``force`` (a frame's lag must be checked against a fresh sample); one question at a time. On ROS 1 it
        first reads /use_sim_time, again each period until that is known."""
        now = self._clock()
        if self._inflight and now - self._last_sent > CLOCK_ANSWER_S:
            # never answered in time: remembered (congestion R3), and asked again; a late answer still counts for R2
            self.last_timeout, self.last_timeout_t0 = now, self._last_sent
            self._inflight = False
            self._last_sent = now - CLOCK_SYNC_S
            self.fail(f"no clock reference: /rosapi/get_time did not answer within {CLOCK_ANSWER_S:g} s")
        if self._inflight or not (force or now - self._last_sent >= CLOCK_SYNC_S):
            return
        self._inflight, self._last_sent = True, now
        t0 = now
        sim_check = self.sim is None
        try:
            fut = (rb.call_service_async("/rosapi/get_param", {"name": "/use_sim_time", "default": "false"})
                   if sim_check else rb.call_service_async("/rosapi/get_time", {}))
        except Exception as exc:  # noqa: BLE001
            self._inflight = False
            self.fail(f"no clock reference: {exc}")
            return

        def done(f) -> None:
            t1 = self._clock()
            if t1 - t0 > CLOCK_ANSWER_S:     # given up on already: only its round trip is kept (congestion R2)
                if not sim_check:
                    try:
                        f.result()
                        with self._lock:
                            self.last_rtt, self.last_rtt_at, self.last_rtt_t0 = t1 - t0, t1, t0
                    except Exception:  # noqa: BLE001
                        pass
                return
            try:
                v = f.result()
                if sim_check:
                    self.sim = str((v or {}).get("value", "")).strip().lower() == "true"
                    self._last_sent = float("-inf")      # ask the time at once
                else:
                    self.add(t0, t1, time_s((v or {}).get("time")))
            except Exception as exc:  # noqa: BLE001
                what = "/rosapi/get_param /use_sim_time" if sim_check else "/rosapi/get_time"
                self.fail(f"no clock reference: {what} did not answer ({exc})")
            finally:
                self._inflight = False
        fut.add_done_callback(done)


def frame_lag(ref: Optional[ClockRef], arrival: float, stamp: Optional[float], now: float) -> Tuple[str, Any]:
    """("known", (arrival_lo, arrival_hi, age_lo, age_hi)): bounds of a shown frame's lag when it arrived and of its age
    now, both in robot time, from ``ClockRef.span``; or ("unknown", reason). The frame must have been confirmed
    (``ClockRef.confirms``)."""
    if stamp is None:
        return "unknown", "the frames carry no header stamp"
    if ref is None:
        return "unknown", "no clock reference"
    off = ref.offset(now)
    if off is None:
        return "unknown", ref.reason
    if not ref.confirms(arrival, stamp):
        return "unknown", "this frame arrived before the robot's clock could be checked"
    olo, ohi = ref.span(arrival)
    lo, hi = arrival + olo - stamp, arrival + ohi - stamp
    if hi < -STEP_TOLERANCE_S:
        return "unknown", "the frame is stamped after its arrival: the robot's clock is inconsistent"
    return "known", (lo, hi, now + olo - stamp, now + ohi - stamp)


@dataclasses.dataclass
class StreamSpec:
    topic: str
    type: str
    stale_after_s: float
    encodings: Optional[Tuple[str, ...]] = None   # None: any supported (untied stream)
    tied: bool = True
    optional: bool = False
    depth: Optional[Tuple[float, Tuple[float, float]]] = None   # (unit_m, range_m) of a depth camera
    ir: bool = False          # an infrared camera: each frame stretched from its own minimum to maximum
    missing_note: str = ""    # said when the camera is not on the wire (e.g. which launch starts it)
    throttle_ms: int = DEFAULT_THROTTLE_MS   # rosbridge throttle_rate: the most frames per second sent
    on_demand: bool = False   # subscribed only on request (the page's user; the wire check samples it)
    silent_note: str = ""     # an advertised stream that may never deliver: explained, not alarmed


class CameraStream:
    """Latest frame of one subscribed stream, with its visible state.

    ``offer`` runs on the transport reader thread and only stores the message; decoding happens in ``poll`` on
    the consumer's thread. With a usable clock reference a decoded frame is held until it is confirmed (a time answer
    whose question was sent after it arrived), then shown; its state is judged from its age in robot time at the
    moment it is asked, so an image never reads LIVE once it is older than its live bound."""

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
        self.paused = ""                          # why the set paused it (link congestion), else ""
        self.sid: Optional[str] = None
        self.started = clock()
        self.ref: Optional[ClockRef] = None     # the clock reference; None: delivery only (the wire check)
        self.last_stamp: Optional[float] = None  # the shown frame's header stamp (robot time)
        self.last_arrival: Optional[float] = None    # its arrival, monotonic
        self.needs_sync = False                  # a frame waits for a time answer sent after it arrived
        self._lags: list = []                    # (arrival, arrival-lag midpoint): a steady offset?
        self._held: Optional[tuple] = None       # (image, arrival, stamp): decoded, waiting to be confirmed
        self.arrived_at: Optional[float] = None  # when the newest message arrived, shown yet or not (congestion R3)
        self._prev = WAITING                     # the state last judged
        self._recheck: Optional[float] = None    # arrival of the first late-looking frame not yet re-checked

    def offer(self, msg: dict) -> None:
        with self._lock:
            self._pending = msg
            self._pending_at = self._clock()

    def poll(self) -> None:
        with self._lock:
            msg, at = self._pending, self._pending_at
            self._pending = None
        if msg is None:
            self._promote()
            return
        self.arrived_at = at
        try:
            img = decode_image(msg, self.spec.encodings, self.spec.depth, self.spec.ir)
        except FrameError as exc:
            self.error = str(exc)
            self.unsupported = "unsupported encoding" in str(exc) or "unsupported format" in str(exc)
            self.frame, self._held = None, None    # never keep showing an older frame as current
            return
        self.error, self.unsupported = None, False
        self._held = (img, at, stamp_s(msg))
        self._promote()

    def _promote(self) -> None:
        """Show the held frame once it is confirmed (or at once with no usable reference: its age is then unknown)."""
        if self._held is None:
            self.needs_sync = False
            return
        img, at, stamp = self._held
        usable = self.ref is not None and stamp is not None and self.ref.offset() is not None
        if usable and not self.ref.confirms(at, stamp):
            self.needs_sync = True                 # ask the robot's time now
            return
        self._held, self.needs_sync = None, False
        self.frame, self.frame_at, self.last_stamp, self.last_arrival = img, at, stamp, at
        self.frames += 1
        self._note_offset(at)

    def _note_offset(self, at: float) -> None:
        off = self.ref.offset(at) if self.ref is not None and self.last_stamp is not None else None
        if off is None:
            self._lags = []
            return
        self._lags = [x for x in self._lags if at - x[0] <= 2 * CLOCK_DOMAIN_SPAN_S]
        self._lags.append((at, at + (off[0] + off[1]) / 2 - self.last_stamp))

    def lag_span(self) -> float:
        """Over how long this stream's lags have been noted (enough frames to tell a steady offset from a backlog)."""
        return self._lags[-1][0] - self._lags[0][0] if len(self._lags) >= 5 else 0.0

    def steady_offset(self) -> Optional[float]:
        """The steady lag of this stream's frames (a real delay, or stamps on another clock), when its frames have sat
        at one offset beyond CLOCK_DOMAIN_MIN_S (or before their arrival) while the newest time answer took at most
        CONGESTED_RTT_S and less than half that lag (an unknown round trip does not count); else None."""
        rtt = self.ref.last_rtt if self.ref is not None else None
        if rtt is None or rtt > CONGESTED_RTT_S or len(self._lags) < 5:   # an unknown round trip is not quick
            return None
        recent = [x for x in self._lags if self._lags[-1][0] - x[0] <= CLOCK_DOMAIN_SPAN_S]
        if len(recent) < 5 or recent[-1][0] - recent[0][0] < CLOCK_DOMAIN_SPAN_S * 0.8:
            return None
        vals = [v for _, v in recent]
        mean = sum(vals) / len(vals)
        if max(vals) - min(vals) > CLOCK_DOMAIN_SPREAD_S or not (mean > CLOCK_DOMAIN_MIN_S or mean < -STEP_TOLERANCE_S):
            return None
        if rtt >= abs(mean) / 2:      # the answers wait about as long as the frames: a queue they share, not an offset
            return None
        return mean

    def state(self, now: Optional[float] = None) -> str:
        return self._judge(now)[0]

    def _judge(self, now: Optional[float] = None) -> Tuple[str, str]:
        st = self._judge_now(now)
        self._prev = st[0]
        return st

    def _judge_now(self, now: Optional[float] = None) -> Tuple[str, str]:
        now = self._clock() if now is None else now
        if self.missing:
            return MISSING, "unavailable: the topic is not on the wire" + (
                f". {self.spec.missing_note}" if self.spec.missing_note else "")
        if self.paused:
            return PAUSED, f"paused: {self.paused}"
        self._promote()                         # a held frame confirmed since the last look is shown now
        if self.unsupported:
            return UNSUPPORTED, f"unsupported: {self.error}"
        if self.error:
            return FAILED, f"failed: {self.error}"
        if self.sid is None and self.frame_at is None and self.spec.on_demand:
            return OFF, "off: started on request only (on the view page)"
        if self.frame_at is None:
            if now - self.started <= self.spec.stale_after_s:
                return WAITING, "waiting for the first frame"
            if self.spec.silent_note:
                return SILENT, f"no frames: {self.spec.silent_note}"
            return STALE, f"stale: no frame within {self.spec.stale_after_s:g} s"
        since = now - self.frame_at
        if since > self.spec.stale_after_s:
            return STALE, f"stale: last frame {since:.1f} s ago"
        if self.ref is None:
            return LIVE, "live"                 # delivery only: no clock reference was given (the wire check)
        bound = live_bound_s(self.spec.throttle_ms)
        steady = self.steady_offset()
        note = (f" (a steady offset of {steady:+.1f} s: either a real delay or stamps on another clock than the robot's"
                " ROS clock)" if steady is not None else "")
        kind, v = frame_lag(self.ref, self.frame_at, self.last_stamp, now)
        if kind != "known":
            if steady is not None and steady > 0:
                return LAGGING, f"lagging: frames arrive about {steady:.1f} s after their stamps{note}; not current"
            if since > bound:
                return STALE, f"stale: last frame {since:.1f} s ago (its age is unknown: {v})"
            return UNKNOWN, f"lag unknown ({v}){note}: frames arrive, but whether they are current cannot be checked"
        alo, ahi, glo, ghi = v
        if alo <= LAG_LIMIT_S:
            self._recheck = None
        elif self.ref.last_t0 < self.frame_at:
            # confirmed by a reading of the clock that may predate a backward step: the clock is asked again, and the
            # stream is first called late once an answer asked after its first late-looking frame arrived has come (a
            # stream already LAGGING stays so meanwhile, rather than flicker)
            self.needs_sync = True
            if self._recheck is None:
                self._recheck = self.frame_at
            if self._prev != LAGGING and self.ref.last_t0 < self._recheck:
                return UNKNOWN, (f"lag unknown (the image looks {alo:.1f}-{ahi:.1f} s old when it arrived; re-checking "
                                 "the robot's clock)")
        if alo > LAG_LIMIT_S:
            why = note or " (the link or the camera is behind)"
            return LAGGING, f"lagging: the image was {alo:.1f}-{ahi:.1f} s old when it arrived{why}; not current"
        if ghi <= bound:
            return LIVE, (f"live (image {max(0.0, (glo + ghi) / 2):.2f} ± {(ghi - glo) / 2:.2f} s old, at most {ghi:.2f} s;"
                          f" live while at most {bound:.2f} s)")
        if glo > bound:
            return STALE, f"stale: no newer frame; the image is {glo:.1f}-{ghi:.1f} s old (live only to {bound:.2f} s)"
        why = (f"; the robot's time answers since {CLOCK_STALE_S:g} s before it arrived disagree by {ghi - glo:.1f} s:"
               " a clock step or slow answers" if ghi - glo > 2 * STEP_TOLERANCE_S else "")
        return UNKNOWN, (f"lag unknown (the image is {glo:.2f}-{ghi:.2f} s old, live only to {bound:.2f} s{why}): "
                         "whether it is current cannot be decided")

    def status_text(self, now: Optional[float] = None) -> str:
        return self._judge(now)[1]


class CameraSet:
    """Subscribe to a set of streams on one connection; poll and close together. With a clock reference it watches
    the link and pauses every stream on congestion (rules R1-R3 above; console spec §2.3) until ``resume``."""

    def __init__(self, rb, specs: Sequence[StreamSpec], *, present: Optional[Dict[str, str]] = None,
                 on_demand: bool = False, clock: Optional[ClockRef] = None, start: bool = True) -> None:
        """Subscribe to every stream on the wire, each at its ``throttle_ms``; an ``on_demand`` stream
        only when ``on_demand`` (the wire check's sample), else it stays off; none when not ``start``."""
        self.rb = rb
        self.streams: Dict[str, CameraStream] = {}
        self.clock = clock
        self.paused = ""
        self._resume: list = []
        for s in specs:
            cs = CameraStream(s)
            cs.ref = clock
            self.streams[s.topic] = cs
            if present is not None and s.topic not in present:
                cs.missing = True
                continue
            if start and (on_demand or not s.on_demand):
                self.start(s.topic)

    def start(self, topic: str) -> None:
        s = self.streams[topic]
        if s.sid is None and not s.missing:
            # a fresh start: an earlier frame (paused, stopped) is not judged as this stream's current one
            s.started, s._lags, s.frame_at, s.last_stamp, s.last_arrival = s._clock(), [], None, None, None
            s._held, s.needs_sync, s.arrived_at, s._recheck = None, False, None, None
            s.sid = self.rb.subscribe(s.spec.topic, s.spec.type, s.offer, throttle_rate=s.spec.throttle_ms,
                                      queue_length=1)

    def stop(self, topic: str) -> None:
        """Unsubscribe one stream (its last frame stays, ageing to stale)."""
        s = self.streams[topic]
        if s.sid:
            self.rb.unsubscribe(s.sid)
            s.sid = None

    def congestion(self, now: Optional[float] = None) -> str:
        """Why the link is congested (rules R1-R3), or ""."""
        c = self.clock
        if c is None or self.paused:
            return ""
        now = c._clock() if now is None else now
        g0, g1 = c.grace
        graced = lambda t: t is not None and g0 <= t <= g1          # noqa: E731
        rtt = c.last_rtt if not graced(c.last_rtt_t0) else None
        for s in self.streams.values():                               # R1 (a steady offset is not a backlog)
            if (s.sid and rtt is not None and rtt >= CLOCK_DOMAIN_RTT_S and not graced(s.frame_at)
                    and s.lag_span() >= CLOCK_DOMAIN_SPAN_S * 0.8 and s.steady_offset() is None
                    and s.state(now) == LAGGING):
                return (f"frames on {s.spec.topic} arrive late while the robot's time answer took {rtt:.2f} s "
                        f"(at least {CLOCK_DOMAIN_RTT_S:g} s): {s.status_text(now)}")
        if rtt is not None and rtt > CONGESTED_RTT_S and c.last_rtt_at is not None \
                and now - c.last_rtt_at <= CONGESTED_WINDOW_S:            # R2
            return f"the robot's time answer took {rtt:.1f} s (the link is backing up)"
        arriving = any(s.sid and s.arrived_at is not None and now - s.arrived_at < 1.0 for s in self.streams.values())
        unanswered = c.unanswered_for(now) > CLOCK_ANSWER_S and not graced(c._last_sent)
        gave_up = (c.last_timeout is not None and now - c.last_timeout <= CLOCK_ANSWER_S
                   and not graced(c.last_timeout_t0))
        if arriving and (unanswered or gave_up):                      # R3
            return f"the robot's time answer is over {CLOCK_ANSWER_S:g} s late while frames keep arriving"
        return ""

    def pause(self, reason: str) -> None:
        self._resume = [t for t, s in self.streams.items() if s.sid]
        for t in self._resume:
            self.stop(t)
        self.paused = reason
        for s in self.streams.values():
            s.paused, s._held, s.needs_sync = reason, None, False

    def resume(self) -> None:
        self.paused = ""
        for s in self.streams.values():
            s.paused = ""
        if self.clock is not None:
            self.clock.last_rtt = self.clock.last_timeout = None
        for t in self._resume:
            self.start(t)
        self._resume = []

    def poll(self) -> None:
        for s in self.streams.values():
            s.poll()
        if self.clock is not None:
            for s in self.streams.values():
                s.state()                      # marks a stream whose lag needs a fresh clock sample
            why = self.congestion()            # judged before a new question resets the one in flight
            if why:
                self.pause(why)
            self.clock.maybe_sync(self.rb, force=any(s.needs_sync for s in self.streams.values()))
            for s in self.streams.values():
                s._promote()                   # an answer that came at once confirms its frames now

    def close(self) -> None:
        for s in self.streams.values():
            if s.sid:
                self.rb.unsubscribe(s.sid)
                s.sid = None


def target_specs(target) -> list:
    """Stream specs of a validated target's profile cameras (resolved names)."""
    return [StreamSpec(topic=target.wire(c.topic), type=c.type, stale_after_s=c.stale_after_s,
                       encodings=c.encodings or None, tied=True, optional=c.optional,
                       depth=(c.unit_m, c.range_m) if c.kind == "depth" else None, ir=c.kind == "ir",
                       missing_note=c.missing_note, throttle_ms=c.throttle_ms(), on_demand=c.on_demand,
                       silent_note=c.silent_note)
            for c in target.profile.cameras]


def untied_specs(graph) -> list:
    """Every supported raw image stream on the wire, not tied to any target: previews, started only on request
    (one at a time, on the view page), since no profile says what they cost the link."""
    return [StreamSpec(topic=n, type=t, stale_after_s=UNTIED_STALE_AFTER_S, encodings=None, tied=False,
                       throttle_ms=UNTIED_THROTTLE_MS, on_demand=True)
            for n, t in sorted(graph.cameras().items())]
