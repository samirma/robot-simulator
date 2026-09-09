"""Command line: `bin/teleop.sh` and `python -m robot_console` both land here."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO, Tuple

from robot_console import __version__
from robot_console.robots import DEFAULT_ROBOT, PROFILES
from robot_console.teleop import HOLD_TIMEOUT, SPEED_DEFAULT, SPEED_MAX
from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    namespaced,
    normalise,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9090


@dataclasses.dataclass(frozen=True)
class Options:
    """What the user asked for, and -- after `resolved()` -- what will be driven.

    `robot`, `namespace` and the three topic fields are `Optional` and mean "not given"
    when `None`; `resolved()` fills them in and everything downstream sees plain strings.
    That is the same idiom `resolve_topic` documents below, extended to the robot itself,
    and it is what lets the console ask the wire which robot is on it and under what name
    instead of assuming both.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    robot: Optional[str] = None
    namespace: Optional[str] = None
    record: Optional[Path] = None
    preflight: bool = True
    preflight_timeout: float = 1.5
    connect_timeout: float = 10.0
    publish_hz: float = 20.0
    loop_hz: float = 60.0
    speed: float = SPEED_DEFAULT
    max_speed: float = SPEED_MAX
    # What `--speed` and `--max-speed` asked for, kept beside what they resolved to. The
    # envelope they are clamped into is the robot's own -- 0.25 m/s is the myAGV's real
    # limit and means nothing to a walking AiNex -- and the robot can still be the wire's
    # answer, so the request has to survive being clamped once in order to be re-clamped
    # against whatever discovery finds.
    speed_request: Optional[float] = None
    max_speed_request: Optional[float] = None
    hold_timeout: Optional[float] = HOLD_TIMEOUT
    record_fps: Optional[float] = None
    cmd_topic: Optional[str] = None
    odom_topic: Optional[str] = None
    camera_topic: Optional[str] = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    @property
    def needs_discovery(self) -> bool:
        """True when the wire has to be asked: either the robot or its namespace is open."""
        return self.robot is None or self.namespace is None

    def in_envelope(self, robot: Optional[str] = None, *, stream: Optional[TextIO] = None) -> "Options":
        """This, with the speeds clamped into `robot`'s envelope.

        Always derived from `speed_request`/`max_speed_request`, never from the values a
        previous pass produced, so applying it again with the robot discovery actually
        found gives what that robot's own limits say and not what the fallback's did.

        Does **not** settle `robot`: which robot is being driven can still be the wire's
        answer while the numbers already have to be usable.
        """
        # Resolving the profile can raise for a robot whose link module is missing; that
        # message is the whole point of the lazy factories, so it reaches the user as an
        # error rather than as a traceback.
        profile = PROFILES[robot or self.robot or DEFAULT_ROBOT]
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

    def under(
        self, namespace: Optional[str] = None, *, camera_topic: Optional[str] = None
    ) -> "Options":
        """This, with every topic left unnamed resolved under `namespace`.

        Which topics a robot is on depends on the namespace alone, not on which robot it
        is, so this can settle as soon as `--namespace` is given even while the robot is
        still the wire's to name.

        `camera_topic` is what discovery *saw*, and is a weaker claim than a flag: it fills
        in only when the user named no camera topic. What the user said wins over what the
        wire said throughout -- the arguments here are what discovery found, and a flag is
        the more specific instruction, which is the rule `resolve_topic` already follows.
        """
        namespace = self.namespace if self.namespace is not None else (namespace or "")
        seen = self.camera_topic if self.camera_topic is not None else camera_topic
        return dataclasses.replace(
            self,
            namespace=namespace,
            cmd_topic=resolve_topic(self.cmd_topic, TOPIC_CMD_VEL, namespace),
            odom_topic=resolve_topic(self.odom_topic, TOPIC_ODOM, namespace),
            camera_topic=resolve_topic(seen, TOPIC_CAMERA, namespace),
        )

    def resolved(
        self,
        robot: Optional[str] = None,
        namespace: Optional[str] = None,
        *,
        camera_topic: Optional[str] = None,
        stream: Optional[TextIO] = None,
    ) -> "Options":
        """This, with the robot, the namespace, the topics and the speeds all settled.

        Idempotent, so calling it twice, or on options that were already explicit, changes
        nothing.
        """
        robot = self.robot or robot or DEFAULT_ROBOT
        return dataclasses.replace(
            self.in_envelope(robot, stream=stream).under(namespace, camera_topic=camera_topic),
            robot=robot,
        )


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



def resolve_topic(explicit: Optional[str], default: str, namespace: str) -> str:
    """The topic to use: what the caller named, else the contract default namespaced.

    An explicitly named topic is taken as given -- naming one is a more specific
    instruction than naming a namespace, and silently prefixing it would make
    `--odom-topic /elsewhere/odom` mean something the caller did not ask for. The flags
    therefore default to `None` rather than to the constant, which is the only way to
    tell "left alone" from "set to the default value on purpose".
    """
    return normalise(explicit) if explicit is not None else namespaced(default, namespace)


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
    # Not given means "ask the wire", not "assume the myAGV": a rosbridge serving one
    # AiNex answers this question better than a default can, and defaulting to the myAGV
    # against one produced a black window and no error. `DEFAULT_ROBOT` is still the
    # fallback for a wire that cannot be asked.
    parser.add_argument(
        "--robot",
        choices=sorted(PROFILES),
        default=None,
        help=f"which robot to drive (default: discovered from the wire, else {DEFAULT_ROBOT})",
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
    # A namespace, not four flags. Several robots on one rosbridge each get one -- the
    # simulator names it after the robot -- so `--namespace myagv` reaches
    # `/myagv/cmd_vel` and friends without spelling any of them out. Applied only to
    # topics left at their default, so an explicit `--odom-topic` still wins.
    #
    # Not given means "ask the wire"; `--namespace ''` is how the bare contract a real
    # vendor bringup presents is asked for on purpose. The two were the same thing when
    # the default was `''`, which is why a namespaced simulator drove nothing.
    parser.add_argument(
        "--namespace", default=None, metavar="NAME",
        help="ROS namespace the robot is under, e.g. `myagv` for /myagv/cmd_vel "
             "(default: discovered from the wire; pass '' for the bare contract)")
    parser.add_argument("--cmd-topic", default=None, help="Twist robots only (myagv)")
    parser.add_argument("--odom-topic", default=None, help="Twist robots only (myagv)")
    parser.add_argument("--camera-topic", default=None)
    parser.add_argument("--version", action="version", version=f"robot_console {__version__}")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    args = build_parser().parse_args(argv)
    host, port = split_host_port(args.host, DEFAULT_PORT)
    if args.port is not None:
        port = args.port

    # What the user said, unresolved: the robot and its namespace may still be the wire's
    # to answer, and the speed envelope is the robot's. `Options.resolved()` settles all
    # of it, once, when the answer is in -- and settles it immediately here when the user
    # left nothing open, so a run that names everything never touches the network for it.
    options = Options(
        host=host,
        port=port,
        robot=args.robot,
        namespace=args.namespace,
        record=Path(args.record) if args.record else None,
        preflight=args.preflight,
        connect_timeout=float(args.connect_timeout),
        publish_hz=float(args.publish_hz),
        speed_request=None if args.speed is None else float(args.speed),
        max_speed_request=None if args.max_speed is None else float(args.max_speed),
        hold_timeout=None if args.latch else max(0.05, float(args.hold_timeout)),
        record_fps=args.record_fps,
        cmd_topic=args.cmd_topic,
        odom_topic=args.odom_topic,
        camera_topic=args.camera_topic,
    )
    # Everything that can be settled now is: the speeds, clamped into the named robot's
    # envelope or the fallback's until the wire names another, and the topics as soon as a
    # namespace is given, since which names a robot is on depends on that alone. What is
    # left open is left open, and `app.resolve` asks the wire about it.
    options = options.in_envelope()
    return options if options.namespace is None else options.under()


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
