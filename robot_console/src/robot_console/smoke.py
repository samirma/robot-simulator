"""A scripted live integration check: `python -m robot_console.smoke`.

Headless and non-interactive, but it *drives the robot* -- roughly 0.3 m forward, back,
sideways, then a turn in place. It uses `RobotLink` and `decode_compressed_image`, the
same code paths teleop uses, so a pass is evidence about the console and not just about
the server.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import List, Optional

from robot_console.bridge import Odom, RobotLink, quiet_roslibpy_logging, wrap_angle
from robot_console.camera import LatestFrame, decode_compressed_image
from robot_console.cli import DEFAULT_HOST, DEFAULT_PORT, split_host_port
from robot_console.preflight import probe_tcp, startup_instructions
from robot_console.teleop import Command
from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TOPIC_SCAN,
    namespaced,
)

DRIVE_SPEED = 0.15
DRIVE_SECONDS = 2.0
TURN_RATE = 0.5
PUBLISH_HZ = 20.0

# Thresholds are about two thirds of the theoretical distance (0.15 x 2.0 = 0.30 m).
# The simulator carries a 0.12 m setpoint lead and real contact physics, so it
# under-travels; the point is to prove the contract carries motion, not to measure a
# servo.
MIN_FORWARD = 0.10
MIN_LATERAL = 0.06
MIN_YAW = 0.30


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ok: Optional[bool] = None
        self.detail = ""
        self.skipped = False

    def passed(self, detail: str = "") -> "Check":
        self.ok, self.detail = True, detail
        return self

    def failed(self, detail: str = "") -> "Check":
        self.ok, self.detail = False, detail
        return self

    def skip(self, detail: str = "") -> "Check":
        self.ok, self.skipped, self.detail = True, True, detail
        return self

    @property
    def label(self) -> str:
        return "skip" if self.skipped else ("ok" if self.ok else "FAIL")


class Runner:
    def __init__(self, link: RobotLink, odoms: List[Odom], latest: LatestFrame, quiet: bool) -> None:
        self.link = link
        self.odoms = odoms
        self.latest = latest
        self.quiet = quiet
        self.checks: List[Check] = []

    def record(self, check: Check) -> Check:
        self.checks.append(check)
        if not self.quiet:
            print(f"  [{check.label:4s}] {check.name:<20s} {check.detail}")
        return check

    def pose(self) -> Odom:
        return self.odoms[-1] if self.odoms else Odom()

    def drive(self, command: Command, seconds: float) -> None:
        """Publish at the teleop rate for `seconds`, then stop and let the base settle."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.link.publish_cmd_vel(command)
            time.sleep(1.0 / PUBLISH_HZ)
        self.settle()

    def settle(self, seconds: float = 0.7) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.link.publish_cmd_vel(Command())
            time.sleep(1.0 / PUBLISH_HZ)
        time.sleep(0.3)


def run(host: str, port: int, *, require_camera: bool, quiet: bool = False,
        namespace: str = "") -> int:
    checks, code = execute(host, port, require_camera=require_camera, quiet=quiet,
                           namespace=namespace)
    return code


def execute(host: str, port: int, *, require_camera: bool, quiet: bool = False,
            namespace: str = ""):
    """Run every check. Returns `(checks, exit_code)`.

    `namespace` addresses one robot on a shared graph: with several robots on one
    rosbridge each is under its own prefix, so this is what points the check at the base
    rather than at whatever else is on the port.
    """
    quiet_roslibpy_logging()
    url = f"ws://{host}:{port}"
    if not quiet:
        print(f"robot_console smoke  {url}")
        print("  (this drives the robot roughly 0.3 m in each direction)\n")

    probe = probe_tcp(host, port)
    if not probe.ok:
        print(f"  [FAIL] preflight            {probe.detail}\n", file=sys.stderr)
        print(startup_instructions(host, port), file=sys.stderr)
        return [Check("preflight").failed(probe.detail)], 2

    odoms: List[Odom] = []
    latest = LatestFrame()
    link = RobotLink(
        host, port,
        cmd_topic=namespaced(TOPIC_CMD_VEL, namespace),
        odom_topic=namespaced(TOPIC_ODOM, namespace),
        camera_topic=namespaced(TOPIC_CAMERA, namespace),
        scan_topic=namespaced(TOPIC_SCAN, namespace),
    )
    # Frames carry the namespace too, joined by `/` and with no leading slash -- that is
    # what ROS 1's `tf_prefix` and ROS 2's frame prefixing produce, and checking it here
    # is what keeps the frame half of the contract verified rather than merely emitted.
    # This is the only place in the console that reads a `frame_id` at all.
    odom_frame = f"{namespace}/odom" if namespace else "odom"
    base_frame = f"{namespace}/base_footprint" if namespace else "base_footprint"

    started = time.monotonic()
    try:
        link.connect(timeout=5.0)
    except Exception as exc:
        print(f"  [FAIL] connect              {exc}", file=sys.stderr)
        return [Check("connect").failed(str(exc))], 1

    runner = Runner(link, odoms, latest, quiet)
    runner.record(Check("preflight").passed(f"tcp connect in {probe.elapsed * 1000:.1f} ms"))
    runner.record(Check("connect").passed(f"roslibpy connected in {(time.monotonic() - started) * 1000:.0f} ms"))

    link.subscribe_odom(odoms.append)
    link.subscribe_camera(latest.offer)

    try:
        # ------------------------------------------------------------ streams
        time.sleep(1.5)
        rate = len(odoms) / 1.5
        odom = runner.pose()
        if len(odoms) >= 12 and odom.frame_id == odom_frame and odom.child_frame_id == base_frame:
            runner.record(
                Check("odom stream").passed(f"{rate:.1f} Hz, frames {odom.frame_id} -> {odom.child_frame_id}")
            )
        else:
            runner.record(
                Check("odom stream").failed(
                    f"{len(odoms)} msgs in 1.5 s ({rate:.1f} Hz), "
                    f"frames {odom.frame_id!r} -> {odom.child_frame_id!r}"
                )
            )

        camera_check = Check("camera stream")
        pending = latest.take()
        frame = decode_compressed_image(pending[0]) if pending else None
        if frame is None:
            detail = f"{latest.received} msgs in 1.5 s; no decodable frame"
            runner.record(camera_check.failed(detail) if require_camera else camera_check.skip(detail))
        else:
            height, width = frame.shape[:2]
            std = float(frame.std())
            detail = f"{latest.rate_hz:.1f} Hz, {width}x{height} jpeg, std {std:.1f}"
            # A flat buffer decodes fine and tells you nothing; std proves a real render.
            if latest.received >= 8 and frame.dtype.name == "uint8" and frame.ndim == 3 and std > 1.0:
                runner.record(camera_check.passed(detail))
            else:
                runner.record(camera_check.failed(detail))

        # ------------------------------------------------------------ motion
        runner.settle(0.5)
        start = runner.pose()
        runner.drive(Command(vx=DRIVE_SPEED), DRIVE_SECONDS)
        end = runner.pose()
        # Project the displacement onto the heading the robot started from, so a base
        # that is not axis-aligned in the house still measures correctly.
        dx, dy = end.x - start.x, end.y - start.y
        along = dx * math.cos(start.yaw) + dy * math.sin(start.yaw)
        lateral = -dx * math.sin(start.yaw) + dy * math.cos(start.yaw)
        detail = f"{along:+.3f} m along heading (lateral {lateral:+.3f} m)"
        runner.record(Check("forward").passed(detail) if along > MIN_FORWARD else Check("forward").failed(detail))

        home = start
        runner.drive(Command(vx=-DRIVE_SPEED), DRIVE_SECONDS)
        end = runner.pose()
        back_err = math.hypot(end.x - home.x, end.y - home.y)
        detail = f"returned to {back_err:.3f} m of start"
        if back_err < 0.15:
            runner.record(Check("back").passed(detail))
        else:
            # The usual cause is furniture, not a broken contract: the base is driven
            # blind here, so a wall on one leg of the round trip shows up as this.
            runner.record(Check("back").failed(detail + "; is the robot against something?"))

        start = runner.pose()
        runner.drive(Command(vy=DRIVE_SPEED), 1.5)
        end = runner.pose()
        dx, dy = end.x - start.x, end.y - start.y
        lateral = -dx * math.sin(start.yaw) + dy * math.cos(start.yaw)
        detail = f"{lateral:+.3f} m lateral"
        runner.record(Check("strafe").passed(detail) if lateral > MIN_LATERAL else Check("strafe").failed(detail))

        start = runner.pose()
        runner.drive(Command(wz=TURN_RATE), DRIVE_SECONDS)
        end = runner.pose()
        dyaw = wrap_angle(end.yaw - start.yaw)
        detail = f"{dyaw:+.3f} rad"
        runner.record(Check("rotate").passed(detail) if dyaw > MIN_YAW else Check("rotate").failed(detail))

        # ------------------------------------------------------------ watchdog
        # The simulator stops the base 0.5 s after commands stop. A real myAGV does
        # NOT: myagv_odometry_node keeps writing the last Twist to the motors forever,
        # so on hardware this check is expected to fail and the console's zero-on-exit
        # is the only thing that stops the robot.
        watchdog = Check("watchdog")
        link.publish_cmd_vel(Command(vx=DRIVE_SPEED))
        time.sleep(1.5)
        moving = runner.pose()
        detail = f"velocity {moving.vx:.3f} after 1.5 s of silence"
        if abs(moving.vx) < 1e-6:
            runner.record(watchdog.passed(detail))
        elif latest.received == 0:
            runner.record(watchdog.skip(detail + "; no camera, so likely the standalone --echo server"))
        else:
            runner.record(watchdog.failed(detail + "; expected the 0.5 s bridge watchdog to zero it"))
        runner.settle(0.5)

        zero = runner.pose()
        detail = f"velocity {zero.vx:.3f}, {zero.wz:.3f}"
        runner.record(
            Check("zero on quit").passed(detail)
            if abs(zero.vx) < 1e-6 and abs(zero.wz) < 1e-6
            else Check("zero on quit").failed(detail)
        )
    finally:
        link.close(hard_exit_after=None)

    failed = [c for c in runner.checks if not c.ok]
    skipped = [c for c in runner.checks if c.skipped]
    if not quiet:
        note = f", {len(skipped)} skipped" if skipped else ""
        print(
            f"\n{len(runner.checks) - len(failed) - len(skipped)} passed, {len(failed)} failed{note} "
            f"in {time.monotonic() - started:.1f} s"
        )
    return runner.checks, (1 if failed else 0)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m robot_console.smoke",
        description="Scripted live integration check against a rosbridge robot. Drives the robot.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="rosbridge host, or host:port")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--require-camera", action="store_true", help="fail, rather than skip, without a camera")
    parser.add_argument("--json", action="store_true", help="emit one JSON object instead of a table")
    parser.add_argument(
        "--namespace", default="", metavar="NAME",
        help="ROS namespace the base is under, e.g. `myagv` for /myagv/cmd_vel. Empty "
             "(the default) is the bare single-robot contract.",
    )
    args = parser.parse_args(argv)

    host, port = split_host_port(args.host, DEFAULT_PORT)
    if args.port is not None:
        port = args.port

    if not args.json:
        return run(host, port, require_camera=args.require_camera,
                   namespace=args.namespace)

    checks, code = execute(host, port, require_camera=args.require_camera, quiet=True,
                           namespace=args.namespace)
    print(
        json.dumps(
            {
                "ok": code == 0,
                "exit": code,
                "host": host,
                "port": port,
                "checks": [
                    {"name": c.name, "ok": bool(c.ok), "skipped": c.skipped, "detail": c.detail} for c in checks
                ],
            }
        )
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
