"""Command line: `bin/teleop.sh` and `python -m robot_console` both land here."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from robot_console import __version__
from robot_console.robots import DEFAULT_ROBOT, PROFILES
from robot_console.teleop import HOLD_TIMEOUT, SPEED_DEFAULT, SPEED_MAX
from robot_console.topics import TOPIC_CAMERA, TOPIC_CMD_VEL, TOPIC_ODOM

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9090


@dataclasses.dataclass(frozen=True)
class Options:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    robot: str = DEFAULT_ROBOT
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
        description="Keyboard teleoperation with a live camera feed for a mobile robot over rosbridge.",
        epilog=(
            "keys: W/S forward-back, A/D strafe, Q/E rotate, Space stop, +/- speed, "
            "H hints, Esc quit. Hold a key to drive; the robot stops shortly after you "
            "let go. The camera window must have focus. On the AiNex the same keys walk "
            "and sidestep."
        ),
    )
    # `sorted(PROFILES)` iterates the factory table without calling any of it, so listing
    # the robots never depends on being able to construct every one of them.
    parser.add_argument(
        "--robot",
        choices=sorted(PROFILES),
        default=DEFAULT_ROBOT,
        help="which robot to drive (default %(default)s)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="rosbridge host, or host:port (default %(default)s)")
    parser.add_argument("--port", type=int, default=None, help=f"rosbridge port (default {DEFAULT_PORT})")
    parser.add_argument("--record", metavar="DIR", default=None, help="write feed.mp4 and commands.jsonl to DIR")
    parser.add_argument("--no-preflight", dest="preflight", action="store_false", help="skip the reachability check")
    # The speed defaults belong to the robot, not to this parser: a printed default of
    # 0.15 m/s would be wrong for two of the three. None here, resolved from the profile
    # in parse_args.
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        metavar="M_PER_S",
        help=f"initial speed in m/s (per-robot default; myagv {SPEED_DEFAULT})",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=None,
        metavar="M_PER_S",
        help=f"speed cap in m/s (per-robot default; myagv {SPEED_MAX}, the real hardware limit)",
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
    # The AiNex has no equivalent of either, and its profile ignores both rather than
    # pretending to honour them; say so instead of failing quietly.
    parser.add_argument("--cmd-topic", default=TOPIC_CMD_VEL, help="Twist robots only (myagv)")
    parser.add_argument("--odom-topic", default=TOPIC_ODOM, help="Twist robots only (myagv)")
    parser.add_argument("--camera-topic", default=TOPIC_CAMERA)
    parser.add_argument("--version", action="version", version=f"robot_console {__version__}")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    args = build_parser().parse_args(argv)
    host, port = split_host_port(args.host, DEFAULT_PORT)
    if args.port is not None:
        port = args.port

    # Resolving the profile can raise for a robot whose link module is missing; that
    # message is the whole point of the lazy factories, so it goes to the user as an
    # error rather than as a traceback.
    profile = PROFILES[args.robot]

    max_speed = profile.speed_max if args.max_speed is None else float(args.max_speed)
    if max_speed > profile.speed_max:
        print(
            f"warning: --max-speed {max_speed} exceeds {profile.speed_limit_label} of "
            f"{profile.speed_max} m/s; simulated motion above it will not match hardware",
            file=sys.stderr,
        )
    max_speed = max(profile.speed_min, max_speed)
    speed = profile.speed_default if args.speed is None else float(args.speed)

    return Options(
        host=host,
        port=port,
        robot=args.robot,
        record=Path(args.record) if args.record else None,
        preflight=args.preflight,
        connect_timeout=float(args.connect_timeout),
        publish_hz=float(args.publish_hz),
        speed=min(max_speed, max(profile.speed_min, speed)),
        max_speed=max_speed,
        hold_timeout=None if args.latch else max(0.05, float(args.hold_timeout)),
        record_fps=args.record_fps,
        cmd_topic=args.cmd_topic,
        odom_topic=args.odom_topic,
        camera_topic=args.camera_topic,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        options = parse_args(argv)
    except RuntimeError as exc:
        # A robot listed in `--robot` whose link module is missing. The profile's own
        # message names the file and the workaround; a traceback would bury both.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Imported here so `--help` and `--version` work even where OpenCV cannot open a
    # display, and so the import cost is not paid to print usage.
    from robot_console.app import run

    return run(options)
