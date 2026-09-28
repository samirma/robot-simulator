"""Command line: `bin/teleop.sh` and `python -m robot_console` both land here.

    teleop.sh [--robot <id>] [--namespace <ns>] [--url ws://…] [--record <dir>]
              [--speed <m/s>] [--max-speed <m/s>] [--latch]

That synopsis is console spec §2.1, and the parser takes exactly those flags.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from robot_console import __version__
from robot_console.robots import MYAGV, PROFILES, TELEOP_ROBOTS
from robot_console.teleop import HOLD_TIMEOUT
from robot_console.topics import TOPIC_CAMERA, namespaced
from robot_console.wire import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_URL, add_url_argument, parse_url

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "DEFAULT_URL", "Options", "build_parser",
           "parse_args", "main", "positive_seconds"]


@dataclasses.dataclass(frozen=True)
class Options:
    """What the user asked for, and -- after `resolved()` -- what will be driven.

    `robot` and `namespace` are `None` for "not given", which means "ask the wire"
    (`--namespace ''` is the bare contract on purpose). The speeds are clamped into the
    envelope of whichever robot is being driven, and re-clamped once discovery names it.
    """

    url: str = DEFAULT_URL
    robot: Optional[str] = None
    namespace: Optional[str] = None
    record: Optional[Path] = None
    latch: bool = False
    loop_hz: float = 60.0
    speed: float = 0.0
    max_speed: float = 0.0
    speed_request: Optional[float] = None
    max_speed_request: Optional[float] = None
    camera_topic: Optional[str] = None

    @property
    def host(self) -> str:
        return parse_url(self.url)[0]

    @property
    def port(self) -> int:
        return parse_url(self.url)[1]

    @property
    def hold_timeout(self) -> Optional[float]:
        """0.6 s without a motion-key event stops the robot; `--latch` turns that off."""
        return None if self.latch else HOLD_TIMEOUT

    @property
    def needs_discovery(self) -> bool:
        return self.robot is None or self.namespace is None

    def in_envelope(self, robot: Optional[str] = None, *, stream: Optional[TextIO] = None) -> "Options":
        """This, with the speeds clamped into `robot`'s envelope (the myAGV's until known).

        Always derived from the requests, never from a previous pass, so re-applying it
        once discovery names the robot gives that robot's own limits.
        """
        profile = PROFILES[robot or self.robot or MYAGV]
        requested = self.max_speed_request
        max_speed = profile.speed_max if requested is None else float(requested)
        if max_speed > profile.speed_max:
            print(
                f"warning: --max-speed {max_speed} exceeds {profile.speed_limit_label} of "
                f"{profile.speed_max} m/s; simulated motion above it will not match hardware",
                file=stream or sys.stderr,
            )
        max_speed = max(profile.speed_min, max_speed)
        speed = profile.speed_default if self.speed_request is None else float(self.speed_request)
        return dataclasses.replace(
            self,
            speed=min(max_speed, max(profile.speed_min, speed)),
            max_speed=max_speed,
        )

    def resolved(
        self,
        robot: str,
        namespace: str,
        *,
        camera_topic: Optional[str] = None,
        stream: Optional[TextIO] = None,
    ) -> "Options":
        """This, with the robot, namespace, camera and speeds settled by the wire's answer."""
        robot = self.robot or robot
        namespace = self.namespace if self.namespace is not None else (namespace or "")
        return dataclasses.replace(
            self.in_envelope(robot, stream=stream),
            robot=robot,
            namespace=namespace,
            camera_topic=camera_topic or namespaced(TOPIC_CAMERA, namespace),
        )


def positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of seconds") from None
    if not seconds > 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teleop",
        description="Keyboard teleoperation of a mobile robot (a myAGV, a myAGV + myCobot "
                    "280, a ROSMASTER X3 PLUS or an AiNex) over rosbridge, with its "
                    "live camera. A separate safety supervisor process publishes every "
                    "motion command and stops the robot if this UI stops answering.",
        epilog=(
            "keys: W/S forward-back, A/D strafe, Q/E rotate, Space stop, +/- speed, "
            "H help, Esc quit; on the AiNex the arrows turn the head and 0 centres it. "
            "Hold a key to drive; the robot stops 0.6 s after the last key event unless "
            "--latch. The camera window must have focus."
        ),
    )
    parser.add_argument(
        "--robot", choices=TELEOP_ROBOTS, default=None,
        help="robot id from robots_specs/robots.yml (default: discovered from /rosapi)")
    parser.add_argument(
        "--namespace", default=None, metavar="NS",
        help="ROS namespace the robot is under, e.g. `myagv` for /myagv/cmd_vel "
             "(default: discovered from /rosapi; '' for the bare contract)")
    add_url_argument(parser)
    parser.add_argument("--record", metavar="DIR", default=None,
                        help="write feed.mp4 (every decoded camera frame) and commands.jsonl to DIR")
    parser.add_argument("--speed", type=float, default=None, metavar="M_PER_S",
                        help="initial linear speed (myAGV and myAGV + myCobot 280 0.15, "
                             "X3 PLUS 0.20, AiNex 0.10)")
    parser.add_argument("--max-speed", type=float, default=None, metavar="M_PER_S",
                        help="speed cap (myAGV and myAGV + myCobot 280 0.28, X3 PLUS 0.70, "
                             "AiNex 0.20)")
    parser.add_argument("--latch", action="store_true",
                        help="a motion key keeps the robot moving until another motion key, "
                             "Space or Esc")
    parser.add_argument("--version", action="version", version=f"robot_console {__version__}")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    args = build_parser().parse_args(argv)
    options = Options(
        url=args.url,
        robot=args.robot,
        namespace=args.namespace,
        record=Path(args.record) if args.record else None,
        latch=bool(args.latch),
        speed_request=None if args.speed is None else float(args.speed),
        max_speed_request=None if args.max_speed is None else float(args.max_speed),
    )
    return options.in_envelope()


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_args(argv)
    # Imported here so `--help` and `--version` work even where OpenCV cannot open a
    # display, and so the import cost is not paid to print usage.
    from robot_console.app import run

    return run(options)
