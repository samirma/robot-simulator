"""Command line for the three SLAM commands.

They share almost everything -- connection, map geometry, speed, save location -- so they
share a parser and an options object, with each mode adding only what is genuinely its
own. `bin/slam.sh explore|map|navigate` and `python -m robot_console.<mode>` both land here.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Optional, Sequence

from robot_console import __version__
from robot_console.cli import DEFAULT_HOST, DEFAULT_PORT, split_host_port
from robot_console.slam.grid import DEFAULT_RESOLUTION
from robot_console.slam.planner import ROBOT_RADIUS_M
from robot_console.teleop import HOLD_TIMEOUT, SPEED_DEFAULT, SPEED_MAX, SPEED_MIN
from robot_console.topics import TOPIC_CAMERA, TOPIC_CMD_VEL, TOPIC_ODOM, TOPIC_SCAN

DEFAULT_MAP_DIR = Path("runs/map")

MODES = ("explore", "map", "navigate")


@dataclasses.dataclass(frozen=True)
class SlamOptions:
    mode: str = "map"

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    preflight: bool = True
    preflight_timeout: float = 1.5
    connect_timeout: float = 10.0

    cmd_topic: str = TOPIC_CMD_VEL
    odom_topic: str = TOPIC_ODOM
    camera_topic: str = TOPIC_CAMERA
    scan_topic: str = TOPIC_SCAN

    out: Path = DEFAULT_MAP_DIR
    load: Optional[Path] = None

    resolution: float = DEFAULT_RESOLUTION
    max_range: float = 8.0
    robot_radius: float = ROBOT_RADIUS_M

    publish_hz: float = 20.0
    loop_hz: float = 60.0
    slam_hz: float = 5.0
    speed: float = SPEED_DEFAULT
    max_speed: float = SPEED_MAX
    hold_timeout: Optional[float] = HOLD_TIMEOUT

    zoom: int = 4
    camera_window: bool = True
    no_match: bool = False
    timeout: Optional[float] = None
    autosave: float = 30.0
    record: Optional[Path] = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    @property
    def map_source(self) -> Optional[Path]:
        """Where an existing map should be read from, if anywhere."""
        if self.load is not None:
            return self.load
        return self.out if self.mode == "navigate" else None


def build_parser(mode: Optional[str] = None) -> argparse.ArgumentParser:
    descriptions = {
        "explore": "Explore a space autonomously and build a map of it.",
        "map": "Drive with the keyboard while the map builds in real time.",
        "navigate": "Load a map, click a point, and drive the robot there.",
    }
    parser = argparse.ArgumentParser(
        prog=f"slam {mode}" if mode else "slam",
        description=descriptions.get(mode or "", "Occupancy-grid SLAM for the myAGV."),
    )
    if mode is None:
        parser.add_argument("mode", choices=MODES, help="what to do")

    parser.add_argument("--host", default=DEFAULT_HOST, help="rosbridge host, or host:port (default %(default)s)")
    parser.add_argument("--port", type=int, default=None, help=f"rosbridge port (default {DEFAULT_PORT})")
    parser.add_argument("--no-preflight", dest="preflight", action="store_false",
                        help="skip the reachability check")
    parser.add_argument("--connect-timeout", type=float, default=10.0)

    parser.add_argument("--out", metavar="DIR", default=str(DEFAULT_MAP_DIR),
                        help="where map.pgm/map.yaml are written (default %(default)s)")
    parser.add_argument("--map", dest="load", metavar="DIR", default=None,
                        help="load an existing map from here and keep building on it. "
                             "`navigate` defaults to --out")
    parser.add_argument("--record", metavar="DIR", default=None,
                        help="also record feed.mp4 and commands.jsonl")

    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, metavar="M",
                        help="map cell size in metres (default %(default)s, what myagv_navigation uses)")
    parser.add_argument("--max-range", type=float, default=8.0, metavar="M",
                        help="ignore laser returns beyond this. The X2 reaches 12 m, but far "
                             "beams are the least accurate and the most expensive to trace "
                             "(default %(default)s)")
    parser.add_argument("--robot-radius", type=float, default=ROBOT_RADIUS_M, metavar="M",
                        help="footprint radius used to inflate obstacles (default %(default)s)")

    parser.add_argument("--speed", type=float, default=SPEED_DEFAULT, help="drive speed in m/s (default %(default)s)")
    parser.add_argument("--max-speed", type=float, default=SPEED_MAX,
                        help="speed cap in m/s (default %(default)s, the real myAGV limit)")
    parser.add_argument("--publish-hz", type=float, default=20.0, help="cmd_vel rate (default %(default)s)")
    parser.add_argument("--slam-hz", type=float, default=5.0,
                        help="cap on scan-matching rate. This runs on the same thread as the "
                             "command stream, so it is a budget, not a target (default %(default)s)")
    parser.add_argument("--hold-timeout", type=float, default=HOLD_TIMEOUT, metavar="SECONDS")
    parser.add_argument("--latch", action="store_true",
                        help="manual driving keeps moving until another key, instead of on release")

    parser.add_argument("--zoom", type=int, default=4, help="map window pixels per cell (default %(default)s)")
    parser.add_argument("--no-camera-window", dest="camera_window", action="store_false",
                        help="map window only")
    parser.add_argument("--no-match", action="store_true",
                        help="trust /odom and skip scan matching. Reasonable against the "
                             "simulator, where odom is ground truth; not on hardware")
    parser.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                        help="stop after this long (explore defaults to unlimited)")
    parser.add_argument("--autosave", type=float, default=30.0, metavar="SECONDS",
                        help="save the map this often while running; 0 disables")

    parser.add_argument("--cmd-topic", default=TOPIC_CMD_VEL)
    parser.add_argument("--odom-topic", default=TOPIC_ODOM)
    parser.add_argument("--camera-topic", default=TOPIC_CAMERA)
    parser.add_argument("--scan-topic", default=TOPIC_SCAN)
    parser.add_argument("--version", action="version", version=f"robot_console {__version__}")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None, mode: Optional[str] = None) -> SlamOptions:
    args = build_parser(mode).parse_args(argv)
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

    return SlamOptions(
        mode=mode or args.mode,
        host=host,
        port=port,
        preflight=args.preflight,
        connect_timeout=float(args.connect_timeout),
        cmd_topic=args.cmd_topic,
        odom_topic=args.odom_topic,
        camera_topic=args.camera_topic,
        scan_topic=args.scan_topic,
        out=Path(args.out),
        load=Path(args.load) if args.load else None,
        record=Path(args.record) if args.record else None,
        resolution=max(0.01, float(args.resolution)),
        max_range=max(0.5, float(args.max_range)),
        robot_radius=max(0.01, float(args.robot_radius)),
        publish_hz=float(args.publish_hz),
        slam_hz=max(0.5, float(args.slam_hz)),
        speed=min(max_speed, max(SPEED_MIN, float(args.speed))),
        max_speed=max_speed,
        hold_timeout=None if args.latch else max(0.05, float(args.hold_timeout)),
        zoom=max(1, int(args.zoom)),
        camera_window=args.camera_window,
        no_match=args.no_match,
        timeout=float(args.timeout) if args.timeout else None,
        autosave=max(0.0, float(args.autosave)),
    )


def main(argv: Optional[Sequence[str]] = None, mode: Optional[str] = None) -> int:
    options = parse_args(argv, mode)
    # Imported late so --help works where OpenCV cannot open a display.
    from robot_console.slam.app import run

    return run(options)
