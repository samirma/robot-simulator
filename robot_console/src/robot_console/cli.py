"""Command line: `bin/teleop.sh` and `python -m robot_console` both land here."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from robot_console import __version__
from robot_console.teleop import HOLD_TIMEOUT, SPEED_DEFAULT, SPEED_MAX, SPEED_MIN
from robot_console.topics import TOPIC_CAMERA, TOPIC_CMD_VEL, TOPIC_ODOM

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9090


@dataclasses.dataclass(frozen=True)
class Options:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    record: Optional[Path] = None
    preflight: bool = True
    preflight_timeout: float = 1.5
    connect_timeout: float = 10.0
    publish_hz: float = 20.0
    loop_hz: float = 60.0
    speed: float = SPEED_DEFAULT
    max_speed: float = SPEED_MAX
    hold_timeout: Optional[float] = HOLD_TIMEOUT
    record_fps: Optional[float] = None
    cmd_topic: str = TOPIC_CMD_VEL
    odom_topic: str = TOPIC_ODOM
    camera_topic: str = TOPIC_CAMERA

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"


def split_host_port(value: str, default_port: int = DEFAULT_PORT) -> Tuple[str, int]:
    """Accept `host` or `host:port`, so `--host 192.168.1.42:9090` does what it looks like."""
    text = value.strip()
    if text.startswith("["):  # bracketed IPv6, optionally with :port
        close = text.find("]")
        if close != -1:
            host = text[1:close]
            rest = text[close + 1 :]
            if rest.startswith(":") and rest[1:].isdigit():
                return host, int(rest[1:])
            return host, default_port
    if text.count(":") == 1:
        host, _, port = text.partition(":")
        if port.isdigit():
            return (host or DEFAULT_HOST), int(port)
    return text, default_port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teleop",
        description="Keyboard teleoperation with a live camera feed for a myAGV over rosbridge.",
        epilog=(
            "keys: W/S forward-back, A/D strafe, Q/E rotate, Space stop, +/- speed, "
            "H hints, Esc quit. Hold a key to drive; the robot stops shortly after you "
            "let go. The camera window must have focus."
        ),
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="rosbridge host, or host:port (default %(default)s)")
    parser.add_argument("--port", type=int, default=None, help=f"rosbridge port (default {DEFAULT_PORT})")
    parser.add_argument("--record", metavar="DIR", default=None, help="write feed.mp4 and commands.jsonl to DIR")
    parser.add_argument("--no-preflight", dest="preflight", action="store_false", help="skip the reachability check")
    parser.add_argument("--speed", type=float, default=SPEED_DEFAULT, help="initial speed in m/s (default %(default)s)")
    parser.add_argument(
        "--max-speed",
        type=float,
        default=SPEED_MAX,
        help=f"speed cap in m/s (default %(default)s, the real myAGV limit)",
    )
    parser.add_argument(
        "--hold-timeout",
        type=float,
        default=HOLD_TIMEOUT,
        metavar="SECONDS",
        help="stop this long after the last key repeat (default %(default)s)",
    )
    parser.add_argument(
        "--latch",
        action="store_true",
        help="keep moving until another direction, Space or Esc, instead of stopping on release",
    )
    parser.add_argument("--publish-hz", type=float, default=20.0, help="cmd_vel rate (default %(default)s)")
    parser.add_argument("--record-fps", type=float, default=None, help="force the recorded video frame rate")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="rosbridge connect timeout in seconds")
    parser.add_argument("--cmd-topic", default=TOPIC_CMD_VEL)
    parser.add_argument("--odom-topic", default=TOPIC_ODOM)
    parser.add_argument("--camera-topic", default=TOPIC_CAMERA)
    parser.add_argument("--version", action="version", version=f"robot_console {__version__}")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    args = build_parser().parse_args(argv)
    host, port = split_host_port(args.host, DEFAULT_PORT)
    if args.port is not None:
        port = args.port

    max_speed = float(args.max_speed)
    if max_speed > SPEED_MAX:
        print(
            f"warning: --max-speed {max_speed} exceeds the real myAGV limit of {SPEED_MAX} m/s; "
            "simulated motion above it will not match hardware",
            file=sys.stderr,
        )
    max_speed = max(SPEED_MIN, max_speed)

    return Options(
        host=host,
        port=port,
        record=Path(args.record) if args.record else None,
        preflight=args.preflight,
        connect_timeout=float(args.connect_timeout),
        publish_hz=float(args.publish_hz),
        speed=min(max_speed, max(SPEED_MIN, float(args.speed))),
        max_speed=max_speed,
        hold_timeout=None if args.latch else max(0.05, float(args.hold_timeout)),
        record_fps=args.record_fps,
        cmd_topic=args.cmd_topic,
        odom_topic=args.odom_topic,
        camera_topic=args.camera_topic,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_args(argv)
    # Imported here so `--help` and `--version` work even where OpenCV cannot open a
    # display, and so the import cost is not paid to print usage.
    from robot_console.app import run

    return run(options)
