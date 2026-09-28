"""A scripted live check of a `/cmd_vel` base: `python -m robot_console.smoke [--url ws://…]`.

The base is the one on the wire -- a myAGV, a myAGV + myCobot 280 or a ROSMASTER X3 PLUS --
narrowed by `--robot` and `--namespace` as teleop narrows it.

Headless and non-interactive, but it **drives the robot**: 2 s forward, 2 s back and 2 s
sideways at teleop's default speed (0.15 m/s), then 2 s turning in place at 0.5 rad/s. It
passes when each straight leg covers at least 0.10 m, the sideways leg 0.06 m, the turn
0.30 rad, and a camera frame decodes.

Motion goes through the same safety supervisor teleop uses, so the robot's stop command
is sent on every exit path -- a pass, a failure, an exception, Ctrl-C, SIGTERM -- and a
wedged check stops heartbeating and is stopped like a wedged UI.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from typing import List, Optional

from robot_console.bridge import Odom, wrap_angle
from robot_console.camera import LatestFrame, decode_image
from robot_console.preflight import probe_tcp, startup_instructions
from robot_console.robots import WHEELED_ROBOTS
from robot_console.supervisor import DEFAULT_SAFETY_TIMEOUT, SupervisedLink, SupervisorError
from robot_console.teleop import SPEED_DEFAULT, Command
from robot_console.wire import DEFAULT_URL, add_url_argument, parse_url

DRIVE_SPEED = SPEED_DEFAULT      # teleop's default for the myAGV, 0.15 m/s
DRIVE_SECONDS = 2.0
TURN_RATE = 0.5
TURN_SECONDS = 2.0

MIN_STRAIGHT = 0.10
MIN_LATERAL = 0.06
MIN_YAW = 0.30

SETTLE_SECONDS = 0.7
TICK = 0.02
CAMERA_WAIT = 5.0
ODOM_WAIT = 5.0


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ok: Optional[bool] = None
        self.detail = ""

    def passed(self, detail: str = "") -> "Check":
        self.ok, self.detail = True, detail
        return self

    def failed(self, detail: str = "") -> "Check":
        self.ok, self.detail = False, detail
        return self

    def judge(self, ok: bool, detail: str) -> "Check":
        return self.passed(detail) if ok else self.failed(detail)

    @property
    def label(self) -> str:
        return "ok" if self.ok else "FAIL"


class Runner:
    def __init__(self, link: SupervisedLink, quiet: bool) -> None:
        self.link = link
        self.quiet = quiet
        self.checks: List[Check] = []
        self.odoms: List[Odom] = []
        self.latest = LatestFrame()

    def record(self, check: Check) -> Check:
        self.checks.append(check)
        if not self.quiet:
            print(f"  [{check.label:4s}] {check.name:<14s} {check.detail}")
        return check

    def pose(self) -> Odom:
        return self.odoms[-1] if self.odoms else Odom()

    def hold(self, command: Command, seconds: float) -> None:
        """Keep `command` wanted for `seconds`, heartbeating the supervisor throughout."""
        deadline = time.monotonic() + seconds
        self.link.publish_cmd_vel(command)
        while time.monotonic() < deadline:
            if not self.link.alive:
                raise RuntimeError(f"the safety supervisor stopped ({self.link.stopped_reason})")
            self.link.publish_cmd_vel(command)
            time.sleep(TICK)

    def leg(self, command: Command, seconds: float) -> None:
        self.hold(command, seconds)
        self.hold(Command(), SETTLE_SECONDS)

    def wait_for(self, predicate, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.link.heartbeat()
            if predicate():
                return True
            time.sleep(TICK)
        return predicate()


def execute(url: str, *, namespace: Optional[str] = None, quiet: bool = False,
            robot: Optional[str] = None):
    """Run every check. Returns `(checks, exit_code)`."""
    host, port = parse_url(url)
    if not quiet:
        print(f"robot_console smoke  {url}")
        print("  (this drives the robot about 0.3 m in each direction)\n")

    probe = probe_tcp(host, port)
    if not probe.ok:
        print(f"  [FAIL] preflight      {probe.detail}\n", file=sys.stderr)
        print(startup_instructions(host, port), file=sys.stderr)
        return [Check("preflight").failed(probe.detail)], 2

    started = time.monotonic()
    link = SupervisedLink(url, robot=robot, namespace=namespace,
                          safety_timeout=DEFAULT_SAFETY_TIMEOUT)
    try:
        ready = link.start()
    except SupervisorError as exc:
        print(f"  [FAIL] connect        {exc}", file=sys.stderr)
        return [Check("connect").failed(str(exc))], 1
    if ready.get("robot") not in WHEELED_ROBOTS:
        link.close()
        detail = f"{ready.get('robot')} is not a /cmd_vel base ({', '.join(WHEELED_ROBOTS)})"
        print(f"  [FAIL] connect        {detail}", file=sys.stderr)
        return [Check("connect").failed(detail)], 1

    runner = Runner(link, quiet)

    def _bail(signum, _frame):
        raise KeyboardInterrupt

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass

    try:
        link.enable_motion()
        runner.record(Check("connect").passed(
            f"{ready['robot']} on {ready.get('cmd_topic')} in "
            f"{(time.monotonic() - started) * 1000:.0f} ms"))
        link.subscribe_odom(runner.odoms.append)
        link.subscribe_camera(runner.latest.offer)

        # ------------------------------------------------------------ streams
        runner.wait_for(lambda: len(runner.odoms) > 0, ODOM_WAIT)
        runner.record(Check("odom").judge(bool(runner.odoms),
                                          f"{len(runner.odoms)} messages"))

        frame_box: dict = {}

        def decoded() -> bool:
            pending = runner.latest.take()
            if pending is not None:
                frame = decode_image(pending[0])
                if frame is not None:
                    frame_box["frame"] = frame
            return "frame" in frame_box

        runner.wait_for(decoded, CAMERA_WAIT)
        frame = frame_box.get("frame")
        runner.record(Check("camera").judge(
            frame is not None,
            f"{frame.shape[1]}x{frame.shape[0]} decoded" if frame is not None
            else f"{runner.latest.received} messages, none decodable"))

        # ------------------------------------------------------------ motion
        runner.hold(Command(), SETTLE_SECONDS)

        def along_and_across(start: Odom, end: Odom):
            dx, dy = end.x - start.x, end.y - start.y
            return (dx * math.cos(start.yaw) + dy * math.sin(start.yaw),
                    -dx * math.sin(start.yaw) + dy * math.cos(start.yaw))

        start = runner.pose()
        runner.leg(Command(vx=DRIVE_SPEED), DRIVE_SECONDS)
        along, _ = along_and_across(start, runner.pose())
        runner.record(Check("forward").judge(along >= MIN_STRAIGHT, f"{along:+.3f} m"))

        start = runner.pose()
        runner.leg(Command(vx=-DRIVE_SPEED), DRIVE_SECONDS)
        along, _ = along_and_across(start, runner.pose())
        runner.record(Check("back").judge(-along >= MIN_STRAIGHT, f"{along:+.3f} m"))

        start = runner.pose()
        runner.leg(Command(vy=DRIVE_SPEED), DRIVE_SECONDS)
        _, across = along_and_across(start, runner.pose())
        runner.record(Check("sideways").judge(across >= MIN_LATERAL, f"{across:+.3f} m"))

        start = runner.pose()
        runner.leg(Command(wz=TURN_RATE), TURN_SECONDS)
        turned = wrap_angle(runner.pose().yaw - start.yaw)
        runner.record(Check("turn").judge(turned >= MIN_YAW, f"{turned:+.3f} rad"))
    except KeyboardInterrupt:
        runner.record(Check("interrupted").failed("stopped by a signal"))
    except Exception as exc:  # noqa: BLE001 - reported as a failed check, robot stopped
        runner.record(Check("error").failed(f"{type(exc).__name__}: {exc}"))
    finally:
        link.close()   # the supervisor sends the stop command, three times
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    failed = [c for c in runner.checks if not c.ok]
    if not quiet:
        print(f"\n{len(runner.checks) - len(failed)} passed, {len(failed)} failed "
              f"in {time.monotonic() - started:.1f} s; robot stopped ({link.stopped_reason})")
    return runner.checks, (1 if failed else 0)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m robot_console.smoke",
        description="Scripted live check against a /cmd_vel base over rosbridge. DRIVES the "
                    "robot.",
    )
    add_url_argument(parser)
    parser.add_argument("--robot", choices=WHEELED_ROBOTS, default=None,
                        help="robot id (default: discovered from /rosapi)")
    parser.add_argument("--namespace", default=None, metavar="NS",
                        help="the robot's namespace (default: discovered from /rosapi)")
    parser.add_argument("--json", action="store_true", help="one JSON object instead of a table")
    args = parser.parse_args(argv)

    checks, code = execute(args.url, namespace=args.namespace, quiet=args.json,
                           robot=args.robot)
    if args.json:
        print(json.dumps({
            "ok": code == 0, "exit": code, "url": args.url,
            "checks": [{"name": c.name, "ok": bool(c.ok), "detail": c.detail} for c in checks],
        }))
    return code


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "execute", "DEFAULT_URL"]
