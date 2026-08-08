"""The teleop loop.

One loop, on the main thread, doing everything: read a key, fold it into the state,
publish `/cmd_vel`, draw the frame. `cv2.imshow`/`waitKey` must own the main thread on
macOS, and a second publisher thread would need a lock around the teleop state and --
worse -- would keep the robot driving while the UI was wedged. With a single loop, a UI
stall stops feeding the command stream and the robot halts, so a freeze degrades into a
stop rather than a runaway.

Stopping the robot on the way out is not best-effort here. The simulator has a 0.5 s
watchdog, but the **real myAGV has none**: `myagv_odometry_node` stores the last Twist
in a global and writes it to the motors at 100 Hz forever, so a console that exits
without sending zeros leaves the AGV driving. Hence the signal handlers and the
`finally`. The same holds harder for the AiNex, which is a state machine: nothing about
falling silent means "stop walking" -- only the `stop` service call does.

Which robot is being driven is a `robots.RobotProfile`, resolved once here. Everything
that differs -- the link, the speed envelope, the on-screen wording, whether there is any
odometry to subscribe to -- comes from it, so this loop stays one loop.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Optional

import cv2

from robot_console.bridge import Odom, quiet_roslibpy_logging
from robot_console.camera import LatestFrame, decode_compressed_image, header_seq
from robot_console.cli import Options
from robot_console.hud import draw_overlay, placeholder
from robot_console.preflight import preflight
from robot_console.recorder import Recorder
from robot_console.robots import PROFILES, RobotProfile
from robot_console.teleop import (
    Action,
    TeleopState,
    action_for_key,
    key_label,
)

WINDOW = "robot_console - teleop"

BANNER = """robot_console {version}  ->  {url}   [{robot}]

{keys}

Hold a key to keep moving; the robot stops shortly after you let go.
The camera window must have focus for keys to register.
"""


def _banner_keys(profile: RobotProfile) -> str:
    """The key block, in the robot's own words -- the AiNex walks where the AGV drives."""
    return "\n".join(f"  {key:<7s} {text}" for key, text in profile.hints)


def run(options: Options) -> int:
    quiet_roslibpy_logging()

    profile = PROFILES[options.robot]

    if options.preflight and not preflight(
        options.host,
        options.port,
        timeout=options.preflight_timeout,
        instructions=profile.startup_instructions,
    ):
        return 2

    link = profile.make_link(options)
    try:
        link.connect(timeout=options.connect_timeout)
    except Exception as exc:
        print(f"error: could not connect to {options.url}: {exc}", file=sys.stderr)
        if not options.preflight:
            print("(--no-preflight was given, so the reachability check was skipped)", file=sys.stderr)
        return 2

    latest = LatestFrame()
    odom_box: dict = {"value": None, "count": 0}

    def on_odom(odom: Odom) -> None:
        odom_box["value"] = odom
        odom_box["count"] += 1

    # The AiNex publishes no odometry at all, and its link raises rather than shrug when
    # asked: a status line claiming a stream that cannot exist is worse than saying none.
    if profile.has_odom:
        link.subscribe_odom(on_odom)
    link.subscribe_camera(latest.offer)

    state = TeleopState(
        speed=options.speed,
        speed_max=options.max_speed,
        hold_timeout=options.hold_timeout,
        speed_min=profile.speed_min,
        speed_step=profile.speed_step,
        turn_ratio=profile.turn_ratio,
        turn_max=profile.turn_max,
    )
    recorder: Optional[Recorder] = None
    t0 = time.monotonic()

    if options.record:
        recorder = Recorder(options.record, fps=options.record_fps, t0=t0)
        # Only the topics this run actually touches: a header naming /cmd_vel and /odom
        # on an AiNex recording would describe a session that never happened. `has_odom`
        # is the right test for both because the split is one contract, not two knobs --
        # the Twist robots have odometry and the gait robot has neither.
        topics = {"camera": options.camera_topic}
        if profile.has_odom:
            topics["cmd_vel"] = options.cmd_topic
            topics["odom"] = options.odom_topic
        recorder.start(
            {
                "robot": profile.name,
                "host": options.host,
                "port": options.port,
                "topics": topics,
                "speed": options.speed,
                "speed_max": options.max_speed,
                "publish_hz": options.publish_hz,
                "hold_timeout": options.hold_timeout,
                "robot_console_version": __import__("robot_console").__version__,
            }
        )

    print(
        BANNER.format(
            version=__import__("robot_console").__version__,
            url=options.url,
            robot=profile.name,
            keys=_banner_keys(profile),
        )
    )
    if recorder:
        print(f"recording to {options.record}")

    # A console killed with Ctrl-C or SIGTERM must still stop a real AGV, which has no
    # watchdog of its own to fall back on.
    def _bail(signum, _frame):
        state.running = False
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass

    # The title carries the robot, so two consoles side by side are tellable apart.
    window = f"{WINDOW} ({profile.name})"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    frame = placeholder(message=f"waiting for {options.camera_topic} ...")
    cv2.imshow(
        window,
        draw_overlay(frame, show_help=state.show_help, speed=state.speed, hints=profile.hints),
    )

    tick_ms = max(1, int(1000.0 / options.loop_hz))
    publish_period = 1.0 / options.publish_hz
    next_publish = time.monotonic()
    last_odom_logged = -1
    last_status = 0.0
    exit_reason = "esc"

    try:
        while state.running:
            key = cv2.waitKey(tick_ms)
            action = action_for_key(key)
            now = time.monotonic()
            if action is not Action.NONE:
                state.apply(action, now)
                if action is Action.QUIT:
                    exit_reason = "esc"
                if recorder and action in (Action.FASTER, Action.SLOWER):
                    recorder.add_event("speed", speed=round(state.speed, 4), t=now)

            # No key-up event exists, so a held key is recognised by its OS auto-repeat
            # and the motion is dropped once the repeats stop.
            if state.expire(now) and recorder:
                recorder.add_event("release", t=now)

            # The window's close button is a legitimate way to quit, and ignoring it
            # would leave a headless loop still driving the robot.
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = "window_closed"
                    break
            except cv2.error:
                exit_reason = "window_closed"
                break

            if now >= next_publish:
                command = state.command()
                link.publish_cmd_vel(command)
                if recorder:
                    recorder.add_command(
                        command,
                        speed=state.speed,
                        action=state.last_action.value,
                        key=key_label(key),
                        t=now,
                    )
                next_publish += publish_period
                if next_publish < now:
                    # After a stall, resync rather than firing a catch-up burst.
                    next_publish = now + publish_period

            pending = latest.take()
            if pending is not None:
                message, arrival = pending
                decoded = decode_compressed_image(message)
                if decoded is not None:
                    frame = decoded
                    if recorder:
                        recorder.add_frame(frame, t=arrival, seq=header_seq(message))

            odom = odom_box["value"]
            if recorder and odom is not None and odom_box["count"] != last_odom_logged:
                recorder.add_odom(odom, t=now)
                last_odom_logged = odom_box["count"]

            cv2.imshow(
                window,
                draw_overlay(
                    frame,
                    show_help=state.show_help,
                    speed=state.speed,
                    speed_max=state.speed_max,
                    moving=state.is_moving,
                    hints=profile.hints,
                ),
            )

            if now - last_status >= 1.0:
                last_status = now
                _print_status(state, odom, latest, has_odom=profile.has_odom)

    except KeyboardInterrupt:
        exit_reason = "interrupt"
    finally:
        # Order matters: stop the robot before spending time on file handles.
        try:
            link.stop()
        except Exception:
            pass
        if recorder:
            recorder.add_event("quit", reason=exit_reason)
            summary = recorder.close()
            print(
                f"\nrecorded {summary.get('frames', 0)} frames, "
                f"{summary.get('commands', 0)} commands to {options.record}"
            )
            if not summary.get("frames"):
                print("(no camera frames arrived, so no feed.mp4 was written)")
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass
        link.close()

    print("stopped.")
    return 0


def _print_status(
    state: TeleopState, odom: Optional[Odom], latest: LatestFrame, *, has_odom: bool = True
) -> None:
    command = state.command()
    speed_note = ""
    if state.at_max_speed:
        speed_note = " (max)"
    elif state.speed <= state.speed_min + 1e-9:
        speed_note = " (min)"
    camera = f"{latest.rate_hz:4.1f} Hz" if latest.received else "  none"
    if odom is not None:
        pose = f"x={odom.x:+.2f} y={odom.y:+.2f} yaw={odom.yaw:+.2f}"
    else:
        # "no odom" on a robot that has none reads as a dropped stream; it is not one.
        pose = "no odom" if has_odom else "odom n/a"
    sys.stdout.write(
        f"\rspeed {state.speed:.2f}{speed_note:6s} "
        f"cmd [{command.vx:+.2f} {command.vy:+.2f} {command.wz:+.2f}]  "
        f"camera {camera}  {pose}    "
    )
    sys.stdout.flush()
