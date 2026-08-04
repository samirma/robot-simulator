"""`--record <dir>`: `feed.mp4` plus `commands.jsonl`.

The video writer cannot be opened until the first frame arrives, because neither the
frame size nor the true frame rate is known before then -- and stamping a guessed fps
produces a file that plays back at the wrong speed, which is a subtly wrong artefact
rather than an obviously broken one. So frames are buffered briefly, the rate is
measured, and the writer is opened with real numbers.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from robot_console.bridge import Odom
from robot_console.teleop import Command

FEED_NAME = "feed.mp4"
LOG_NAME = "commands.jsonl"
SCHEMA_VERSION = 1

# mp4v is the codec the opencv-python wheels ship everywhere. avc1/H264 is frequently
# absent from the bundled FFmpeg on macOS and fails by writing a 0-byte file with only
# an obscure warning.
FOURCC = "mp4v"

FPS_FALLBACK = 20.0
FPS_MIN, FPS_MAX = 1.0, 120.0
PROBE_FRAMES = 20  # one second at the simulator's 20 Hz


def estimate_fps(arrivals: Sequence[float], fallback: float = FPS_FALLBACK) -> float:
    """Frame rate from arrival timestamps, via the median interval.

    Median rather than mean: one scheduling hiccup or TCP stall in the sample would drag
    a mean down far enough to make the whole recording play back in slow motion.
    """
    if len(arrivals) < 2:
        return fallback
    deltas = [b - a for a, b in zip(arrivals, arrivals[1:]) if b > a]
    if not deltas:
        return fallback
    interval = statistics.median(deltas)
    if interval <= 0:
        return fallback
    return min(FPS_MAX, max(FPS_MIN, 1.0 / interval))


class Recorder:
    """Writes the video feed and a replayable command log into one directory."""

    def __init__(
        self,
        directory: Path,
        *,
        fps: Optional[float] = None,
        probe_frames: int = PROBE_FRAMES,
        t0: Optional[float] = None,
        stream=sys.stderr,
    ) -> None:
        self.directory = Path(directory)
        self.fixed_fps = fps
        self.probe_frames = max(1, int(probe_frames))
        self.t0 = time.monotonic() if t0 is None else t0
        self._stream = stream

        self._log = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[Tuple[int, int]] = None
        self._fps: Optional[float] = None
        self._pending: List[np.ndarray] = []
        self._arrivals: List[float] = []
        self._frames = 0
        self._commands = 0
        self._writer_failed = False
        self._closed = False

    # ---------------------------------------------------------------- lifecycle

    def start(self, meta: Mapping) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._log = (self.directory / LOG_NAME).open("w", encoding="utf-8")
        record = {"type": "meta", "schema": SCHEMA_VERSION}
        record.update(meta)
        self._write(record, stamp=False)

    def _write(self, record: dict, *, stamp: bool = True, t: Optional[float] = None) -> None:
        if self._log is None:
            return
        if stamp:
            now = time.monotonic() if t is None else t
            record = {**record, "t": round(now - self.t0, 4)}
            # Keep `type` and `t` first so the log reads well with head/tail.
            record = {"type": record.pop("type"), "t": record.pop("t"), **record}
        self._log.write(json.dumps(record) + "\n")

    # ---------------------------------------------------------------- video

    def add_frame(self, bgr: np.ndarray, *, t: Optional[float] = None, seq: Optional[int] = None) -> None:
        now = time.monotonic() if t is None else t
        if self._writer is None and not self._writer_failed:
            self._arrivals.append(now)
            self._pending.append(bgr)
            ready = self.fixed_fps is not None or len(self._pending) >= self.probe_frames
            if ready:
                self._open_writer()
        elif self._writer is not None:
            self._write_frame(bgr)
        index = self._frames - 1 if self._writer is not None else len(self._pending) - 1
        height, width = bgr.shape[:2]
        self._write(
            {"type": "frame", "index": index, "header_seq": seq, "width": int(width), "height": int(height)},
            t=now,
        )

    def _open_writer(self) -> None:
        first = self._pending[0]
        height, width = first.shape[:2]
        self._size = (int(width), int(height))
        self._fps = float(self.fixed_fps) if self.fixed_fps else estimate_fps(self._arrivals)
        path = self.directory / FEED_NAME
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*FOURCC), self._fps, self._size)
        if not writer.isOpened():
            # A codec problem must not abort the drive: the command log is still
            # worth having, and the operator is mid-teleoperation.
            print(f"warning: could not open {path} with codec {FOURCC}; video not recorded", file=self._stream)
            self._writer_failed = True
            self._pending.clear()
            return
        self._writer = writer
        for frame in self._pending:
            self._write_frame(frame)
        self._pending.clear()

    def _write_frame(self, bgr: np.ndarray) -> None:
        if self._writer is None or self._size is None:
            return
        height, width = bgr.shape[:2]
        if (width, height) != self._size:
            # Real image_transport can change resolution mid-stream; resizing keeps the
            # file valid rather than silently dropping the rest of the drive.
            bgr = cv2.resize(bgr, self._size)
        self._writer.write(bgr)
        self._frames += 1

    # ---------------------------------------------------------------- log

    def add_command(
        self,
        command: Command,
        *,
        speed: float,
        action: str,
        key: Optional[str] = None,
        t: Optional[float] = None,
    ) -> None:
        twist = command.to_twist()
        self._write(
            {
                "type": "cmd",
                "seq": self._commands,
                "key": key,
                "action": action,
                "speed": round(float(speed), 4),
                # The literal Twist sub-dicts, so a replayer can feed a line straight
                # back to /cmd_vel with no translation.
                "linear": twist["linear"],
                "angular": twist["angular"],
            },
            t=t,
        )
        self._commands += 1

    def add_odom(self, odom: Odom, *, t: Optional[float] = None) -> None:
        self._write(
            {
                "type": "odom",
                "x": round(odom.x, 4),
                "y": round(odom.y, 4),
                "yaw": round(odom.yaw, 4),
                "vx": round(odom.vx, 4),
                "vy": round(odom.vy, 4),
                "wz": round(odom.wz, 4),
            },
            t=t,
        )

    def add_event(self, kind: str, *, t: Optional[float] = None, **fields) -> None:
        self._write({"type": "event", "kind": kind, **fields}, t=t)

    # ---------------------------------------------------------------- teardown

    def close(self, *, t: Optional[float] = None) -> dict:
        if self._closed:
            return {}
        self._closed = True
        # Fewer frames arrived than the probe wanted: open the writer anyway so a short
        # drive still produces a video.
        if self._writer is None and self._pending and not self._writer_failed:
            self._open_writer()
        if self._writer is not None:
            self._writer.release()
        if not self._frames:
            # An empty or placeholder-filled mp4 is worse than an absent one.
            self._write({"type": "event", "kind": "no_camera"}, t=t)
        now = time.monotonic() if t is None else t
        summary = {
            "type": "summary",
            "duration": round(now - self.t0, 4),
            "commands": self._commands,
            "frames": self._frames,
            "fps": round(self._fps, 2) if self._fps else None,
            "feed": FEED_NAME if self._frames else None,
        }
        self._write(summary, t=now)
        if self._log is not None:
            self._log.close()
            self._log = None
        return summary

    @property
    def frames_written(self) -> int:
        return self._frames

    @property
    def commands_written(self) -> int:
        return self._commands

    @property
    def fps(self) -> Optional[float]:
        return self._fps
