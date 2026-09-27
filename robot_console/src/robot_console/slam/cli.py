"""Command line for the three SLAM modes (console spec §2.2).

    slam.sh explore  --out <map-dir> [--namespace <ns>] [--url ws://…]
                     [--max-duration <s>] [--max-goals <n>] [--safety-timeout <s>]
    slam.sh map      --out <map-dir> [--namespace <ns>] [--url ws://…] [--safety-timeout <s>]
    slam.sh navigate --map <map-dir> [--namespace <ns>] [--url ws://…] [--safety-timeout <s>]

The parser takes exactly those flags. Everything else in `SlamOptions` is a fixed
constant of the console (the 0.05 m grid among them), kept as a field so tests can build
a session directly.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Optional, Sequence

from robot_console import __version__
from robot_console.cli import positive_seconds
from robot_console.slam import mapio
from robot_console.slam.explorer import DEFAULT_MAX_GOALS
from robot_console.slam.frontier import DISTANCE_BIAS_M
from robot_console.slam.grid import DEFAULT_RESOLUTION
from robot_console.slam.planner import ROBOT_RADIUS_M
from robot_console.supervisor import DEFAULT_SAFETY_TIMEOUT
from robot_console.teleop import HOLD_TIMEOUT, SPEED_DEFAULT, SPEED_MAX
from robot_console.wire import DEFAULT_URL, add_url_argument, parse_url

MODES = ("explore", "map", "navigate")
DEFAULT_MAX_DURATION = 3600.0


@dataclasses.dataclass(frozen=True)
class SlamOptions:
    mode: str = "map"
    url: str = DEFAULT_URL
    # None -> discovered from /rosapi; '' -> the bare contract.
    namespace: Optional[str] = None
    out: Optional[Path] = None           # explore, map: where the map is saved/continued
    load: Optional[Path] = None          # navigate: the map to drive on
    safety_timeout: float = DEFAULT_SAFETY_TIMEOUT
    max_duration: float = DEFAULT_MAX_DURATION
    max_goals: int = DEFAULT_MAX_GOALS

    # Fixed by the console, not flags.
    preflight: bool = True
    preflight_timeout: float = 1.5
    resolution: float = DEFAULT_RESOLUTION
    max_range: float = 8.0
    robot_radius: float = ROBOT_RADIUS_M
    loop_hz: float = 60.0
    slam_hz: float = 5.0
    speed: float = SPEED_DEFAULT
    max_speed: float = SPEED_MAX
    hold_timeout: Optional[float] = HOLD_TIMEOUT
    zoom: int = 4
    camera_window: bool = True
    no_match: bool = False
    distance_bias: float = DISTANCE_BIAS_M
    autosave: float = 30.0

    @property
    def host(self) -> str:
        return parse_url(self.url)[0]

    @property
    def port(self) -> int:
        return parse_url(self.url)[1]

    @property
    def map_source(self) -> Optional[Path]:
        """The saved map this run starts from: `--map` for navigate, and for explore and
        map a map already in `--out`, which the run continues."""
        if self.mode == "navigate":
            return self.load
        if self.out is not None and mapio.exists(self.out):
            return self.out
        return None

    @property
    def save_to(self) -> Optional[Path]:
        return self.load if self.mode == "navigate" else self.out


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _add_mode_arguments(parser: argparse.ArgumentParser, mode: str) -> None:
    if mode == "navigate":
        parser.add_argument("--map", dest="load", metavar="MAP_DIR", required=True,
                            help="the saved map (map.yaml + map.pgm [+ map.npz]) to drive on")
    else:
        parser.add_argument("--out", metavar="MAP_DIR", required=True,
                            help="where map.pgm, map.yaml and map.npz are written; a map "
                                 "already there is continued, from the robot pose it saved")
    parser.add_argument(
        "--namespace", default=None, metavar="NS",
        help="ROS namespace of the myAGV (default: discovered from /rosapi; '' for the "
             "bare contract)")
    add_url_argument(parser)
    if mode == "explore":
        parser.add_argument("--max-duration", type=positive_seconds,
                            default=DEFAULT_MAX_DURATION, metavar="S",
                            help="hard run limit in seconds (default %(default)s)")
        parser.add_argument("--max-goals", type=_positive_int, default=DEFAULT_MAX_GOALS,
                            metavar="N", help="hard limit on goals attempted (default %(default)s)")
    parser.add_argument("--safety-timeout", type=positive_seconds,
                        default=DEFAULT_SAFETY_TIMEOUT, metavar="S",
                        help="the safety supervisor stops the robot after this long without "
                             "a UI heartbeat (default %(default)s)")
    parser.add_argument("--version", action="version", version=f"robot_console {__version__}")


DESCRIPTIONS = {
    "explore": "Explore a space autonomously by frontier exploration and build a map of it.",
    "map": "Drive with the keyboard while the map builds in real time.",
    "navigate": "Load a map, click a point, and drive the robot there.",
}


def build_parser(mode: Optional[str] = None) -> argparse.ArgumentParser:
    if mode is not None:
        parser = argparse.ArgumentParser(prog=f"slam {mode}", description=DESCRIPTIONS[mode])
        _add_mode_arguments(parser, mode)
        return parser
    parser = argparse.ArgumentParser(prog="slam", description="Occupancy-grid SLAM for the myAGV.")
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in MODES:
        _add_mode_arguments(sub.add_parser(name, description=DESCRIPTIONS[name]), name)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None, mode: Optional[str] = None) -> SlamOptions:
    args = build_parser(mode).parse_args(argv)
    mode = mode or args.mode
    return SlamOptions(
        mode=mode,
        url=args.url,
        namespace=args.namespace,
        out=Path(args.out) if getattr(args, "out", None) else None,
        load=Path(args.load) if getattr(args, "load", None) else None,
        safety_timeout=float(args.safety_timeout),
        max_duration=float(getattr(args, "max_duration", DEFAULT_MAX_DURATION)),
        max_goals=int(getattr(args, "max_goals", DEFAULT_MAX_GOALS)),
    )


def main(argv: Optional[Sequence[str]] = None, mode: Optional[str] = None) -> int:
    options = parse_args(argv, mode)
    # Imported late so --help works where OpenCV cannot open a display.
    from robot_console.slam.app import run

    return run(options)
