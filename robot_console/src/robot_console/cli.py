"""Command line: `bin/teleop.sh` and `python -m robot_console` both land here.

    teleop.sh [--robot <id>] [--namespace <ns>] [--url ws://…] [--record <dir>]
              [--speed <m/s>] [--max-speed <m/s>] [--latch]

That synopsis is console spec §2.1, and the parser takes exactly those flags.
"""

from __future__ import annotations

import argparse
import dataclasses
import textwrap
from pathlib import Path
from typing import Optional, Sequence

from robot_console import __version__
from robot_console.robot_ids import ROBOT_NAMES, TELEOP_IDS, listing, refusal
from robot_console.robots import MYAGV, PROFILES
from robot_console.teleop import HOLD_TIMEOUT
from robot_console.topics import TOPIC_CAMERA, namespaced
from robot_console.wire import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_URL, add_url_argument, parse_url

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "DEFAULT_URL", "Options", "SpeedLimitError",
           "build_parser", "parse_args", "main", "positive_seconds"]


class SpeedLimitError(ValueError):
    """A requested speed above the cap (console spec §2.1). The message names the limit."""


def _over_limit(flag: str, value: float, profile) -> SpeedLimitError:
    return SpeedLimitError(
        f"{flag} {value:g} m/s is above {profile.speed_limit_label} of "
        f"{profile.speed_max:g} m/s; teleop never commands more than that")


@dataclasses.dataclass(frozen=True)
class Options:
    """What the user asked for, and -- after `resolved()` -- what will be driven.

    `robot` and `namespace` are `None` for "not given", which means "ask the wire"
    (`--namespace ''` is the bare contract on purpose). The speeds are checked against the
    cap of whichever robot is being driven -- its hardware limit, which `--max-speed` may
    only lower -- and checked again once discovery names it. Above the cap is refused
    (`SpeedLimitError`), never clamped.
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

    def in_envelope(self, robot: Optional[str] = None) -> "Options":
        """This, with the speeds settled in `robot`'s envelope. Raises `SpeedLimitError`.

        The cap is the robot's hardware limit, or a lower `--max-speed`. A `--speed` or
        `--max-speed` above the hardware limit, or a `--speed` above `--max-speed`, is
        refused rather than clamped; a speed below the envelope's floor is raised to it.
        Before discovery has named the robot, a request is checked against every robot
        teleop drives -- it is refused only if it is above all of their limits -- and
        `resolved()` checks it again against the robot actually found. Always derived
        from the requests, never from a previous pass.
        """
        name = robot or self.robot
        if name is None:
            fastest = max((PROFILES[r] for r in TELEOP_IDS), key=lambda p: p.speed_max)
            for flag, value in (("--speed", self.speed_request),
                                ("--max-speed", self.max_speed_request)):
                if value is not None and float(value) > fastest.speed_max:
                    raise SpeedLimitError(
                        f"{flag} {float(value):g} m/s is above every teleop robot's hardware "
                        f"limit, the highest being {fastest.speed_limit_label} of "
                        f"{fastest.speed_max:g} m/s; teleop never commands more than that")
            name = MYAGV
            if any(v is not None and float(v) > PROFILES[MYAGV].speed_max
                   for v in (self.speed_request, self.max_speed_request)):
                # Not refused yet, and not settled either: only the wire can say.
                return self
        profile = PROFILES[name]
        max_speed = profile.speed_max
        if self.max_speed_request is not None:
            requested = float(self.max_speed_request)
            if requested > profile.speed_max:
                raise _over_limit("--max-speed", requested, profile)
            if requested < profile.speed_min:
                raise SpeedLimitError(
                    f"--max-speed {requested:g} m/s is below the slowest speed teleop "
                    f"drives the {ROBOT_NAMES[profile.name]} at, {profile.speed_min:g} m/s")
            max_speed = requested
        speed = profile.speed_default if self.speed_request is None else float(self.speed_request)
        if self.speed_request is not None:
            if speed > profile.speed_max:
                raise _over_limit("--speed", speed, profile)
            if speed > max_speed:
                raise SpeedLimitError(
                    f"--speed {speed:g} m/s is above --max-speed {max_speed:g} m/s")
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
    ) -> "Options":
        """This, with the robot, namespace, camera and speeds settled by the wire's answer.

        Raises `SpeedLimitError` if a requested speed is above that robot's cap.
        """
        robot = self.robot or robot
        namespace = self.namespace if self.namespace is not None else (namespace or "")
        return dataclasses.replace(
            self.in_envelope(robot),
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


def teleop_robot(value: str) -> str:
    """`--robot`'s type: a robot id teleop drives, or a refusal listing the accepted ids."""
    if value not in TELEOP_IDS:
        raise argparse.ArgumentTypeError(refusal(value, TELEOP_IDS))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teleop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.fill(
            "Keyboard teleoperation of a mobile robot over rosbridge, with its live camera. "
            "A separate safety supervisor process publishes every motion command and stops "
            "the robot if this UI stops answering.", 78),
        epilog=(
            "keys: W/S forward-back, A/D strafe, Q/E rotate, Space stop, +/- speed,\n"
            "H help, Esc quit; on the AiNex the arrows turn the head and 0 centres it.\n"
            "Hold a key to drive; the robot stops 0.6 s after the last key event unless\n"
            "--latch. The camera window must have focus.\n\n"
            "--robot accepts (every robot in robots_specs/robots.yml whose kind is not arm):\n"
            + listing(TELEOP_IDS)
        ),
    )
    parser.add_argument(
        "--robot", type=teleop_robot, default=None, metavar="ID",
        help="robot id, listed below (default: discovered from /rosapi)")
    parser.add_argument(
        "--namespace", default=None, metavar="NS",
        help="ROS namespace the robot is under, e.g. `myagv` for /myagv/cmd_vel "
             "(default: discovered from /rosapi; '' for the bare contract)")
    add_url_argument(parser)
    parser.add_argument("--record", metavar="DIR", default=None,
                        help="write feed.mp4 (every decoded camera frame) and commands.jsonl to DIR")
    parser.add_argument("--speed", type=float, default=None, metavar="M_PER_S",
                        help="initial linear speed (default: myAGV and myAGV + myCobot 280 "
                             "0.15, X3 PLUS 0.20, AiNex 0.10); above the cap is refused")
    parser.add_argument("--max-speed", type=float, default=None, metavar="M_PER_S",
                        help="lower the speed cap below the robot's hardware limit (myAGV "
                             "and myAGV + myCobot 280 0.28, X3 PLUS 0.70, AiNex 0.20); "
                             "above it is refused")
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
    try:
        return options.in_envelope()
    except SpeedLimitError as exc:
        build_parser().error(str(exc))


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_args(argv)
    # Imported here so `--help` and `--version` work even where OpenCV cannot open a
    # display, and so the import cost is not paid to print usage.
    from robot_console.app import run

    return run(options)
