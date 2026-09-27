"""The real teleop UI loop, driven by a script instead of a keyboard and a window.

Run as a subprocess by `test_motion_safety.py` (and usable by hand for a scripted live
session):

    python teleop_driver.py --url ws://127.0.0.1:PORT --robot myagv --namespace myagv \
        --event freeze --marker /tmp/marker [--latch] [--drive-s 1.0] [--key w]

It holds a motion key (auto-repeating every 90 ms like a held key, or pressed once with
`--latch`) for `--drive-s`, then writes `time.time()` to `--marker` and does `--event`:

    freeze     the UI loop stops dead (sleeps) -- the heartbeat stops with it
    close-ipc  the UI closes its end of the supervisor pipe and carries on
    quit       Esc
    exception  the loop raises
    wait       keeps driving; the test sends a signal (SIGINT, SIGTERM, SIGKILL)

No window is opened: the frontend is this script's.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from robot_console.app import Frontend, run
from robot_console.cli import Options
from robot_console.supervisor import SupervisedLink

KEY_ESC = 27


class ScriptedFrontend(Frontend):
    def __init__(self, *, key: str, latch: bool, drive_s: float, event: str, marker: Path,
                 link: SupervisedLink) -> None:
        self.key = ord(key)
        self.latch = latch
        self.drive_s = drive_s
        self.event = event
        self.marker = marker
        self.link = link
        self.opened_at = None
        self.last_key = -1.0
        self.fired = False

    def confirm_estop(self, prompt: str) -> bool:
        return True

    def open(self, title: str) -> None:
        self.opened_at = time.monotonic()

    def show(self, image) -> None:
        pass

    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def _mark(self) -> None:
        self.marker.write_text(repr(time.time()))

    def poll_key(self, timeout_ms: int) -> int:
        time.sleep(timeout_ms / 1000.0)
        now = time.monotonic()
        if now - self.opened_at >= self.drive_s and not self.fired:
            self.fired = True
            self._mark()
            if self.event == "freeze":
                time.sleep(60.0)
            elif self.event == "close-ipc":
                self.link.close_ipc()
            elif self.event == "quit":
                return KEY_ESC
            elif self.event == "exception":
                raise RuntimeError("scripted UI failure")
        # A held key: the OS repeats it every ~90 ms. Latched: one press is enough.
        if self.latch:
            if self.last_key < 0:
                self.last_key = now
                return self.key
            return -1
        if now - self.last_key >= 0.09:
            self.last_key = now
            return self.key
        return -1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--robot", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--event", default="wait")
    parser.add_argument("--marker", required=True)
    parser.add_argument("--latch", action="store_true")
    parser.add_argument("--drive-s", type=float, default=1.0)
    parser.add_argument("--key", default="w")
    parser.add_argument("--safety-timeout", type=float, default=0.25)
    args = parser.parse_args(argv)

    options = Options(url=args.url, robot=args.robot, namespace=args.namespace,
                      latch=args.latch, safety_timeout=args.safety_timeout).in_envelope()
    link = SupervisedLink(args.url, robot=args.robot, namespace=args.namespace,
                          safety_timeout=args.safety_timeout)
    frontend = ScriptedFrontend(key=args.key, latch=args.latch, drive_s=args.drive_s,
                                event=args.event, marker=Path(args.marker), link=link)
    return run(options, frontend, link)


if __name__ == "__main__":
    sys.exit(main())
