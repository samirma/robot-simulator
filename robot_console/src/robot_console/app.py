"""The teleop UI loop.

One loop, on the main thread (console spec §3): read a key, fold it into the state, tell
the safety supervisor what motion is wanted, draw the frame. `cv2.imshow`/`waitKey` must
own the main thread on macOS, and this loop **never publishes motion itself**: every
command goes over a pipe to `robot_console.supervisor`, which owns the rosbridge
connection and is the only thing that talks to the robot.

The loop sends a heartbeat every tick, from the loop itself. If it freezes, the heartbeat
stops and the supervisor stops the robot within the safety timeout; if it dies, the pipe
closes and the same happens. Neither robot has a command watchdog, so that -- and the
explicit quit on every exit path here (Esc, window close, exception, SIGINT/SIGTERM) -- is
what stops it.

Display and keyboard come from a `Frontend`, so the same loop runs against OpenCV windows
for a person and against a script for tests and scripted sessions, with no window.
"""

from __future__ import annotations

import queue
import signal
import sys
import time
from typing import Optional

from robot_console.camera import decode_image, header_seq
from robot_console.cli import Options, SpeedLimitError
from robot_console.hud import draw_overlay, placeholder
from robot_console.recorder import Recorder
from robot_console.robots import PROFILES, RobotProfile
from robot_console.supervisor import SAFETY_TIMEOUT, SupervisedLink, SupervisorError
from robot_console.teleop import (
    HEAD_ACTIONS,
    Action,
    HeadPose,
    TeleopState,
    action_for_key,
    key_label,
)

WINDOW = "robot_console - teleop"

BANNER = """robot_console {version}  ->  {url}   [{robot}]

{keys}

Hold a key to keep moving; the robot stops {release}.
The camera window must have focus for keys to register.
"""

ESTOP_PROMPT = (
    "Before any motion: confirm that an independent physical emergency stop or motor-power\n"
    "dead-man is armed and within reach of the operator. It is the only protection against\n"
    "host failure or network loss -- software on the failed path cannot stop the robot.\n"
    "Type 'yes' to confirm: "
)


class Frontend:
    """Display and keyboard. The OpenCV one is below; tests supply a scripted one."""

    def confirm_estop(self, prompt: str) -> bool:
        raise NotImplementedError

    def open(self, title: str) -> None:
        raise NotImplementedError

    def poll_key(self, timeout_ms: int) -> int:
        raise NotImplementedError

    def show(self, image) -> None:
        raise NotImplementedError

    def is_open(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class CvFrontend(Frontend):
    """An OpenCV window, and the terminal for the emergency-stop confirmation."""

    def __init__(self) -> None:
        self.title = WINDOW

    def confirm_estop(self, prompt: str) -> bool:
        try:
            sys.stderr.write(prompt)
            sys.stderr.flush()
            answer = sys.stdin.readline()
        except (OSError, ValueError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("yes", "y")

    def open(self, title: str) -> None:
        import cv2

        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def poll_key(self, timeout_ms: int) -> int:
        import cv2

        # waitKeyEx, not waitKey: the arrows need the untruncated code, because their
        # low byte collides with a letter (see teleop.KEYMAP_EXTENDED).
        return cv2.waitKeyEx(timeout_ms)

    def show(self, image) -> None:
        import cv2

        cv2.imshow(self.title, image)

    def is_open(self) -> bool:
        import cv2

        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        import cv2

        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass


def _banner_keys(profile: RobotProfile) -> str:
    return "\n".join(f"  {key:<7s} {text}" for key, text in profile.hints)


def run(options: Options, frontend: Optional[Frontend] = None,
        link: Optional[SupervisedLink] = None) -> int:
    frontend = frontend or CvFrontend()

    # Asked before the supervisor exists, so a person thinking about it does not count
    # against the heartbeat.
    if not frontend.confirm_estop(ESTOP_PROMPT):
        print("error: motion needs a confirmed independent emergency stop; not starting.",
              file=sys.stderr)
        return 2

    link = link or SupervisedLink(options.url, robot=options.robot, namespace=options.namespace)
    try:
        ready = link.start()
    except SupervisorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        options = options.resolved(ready["robot"], ready["namespace"],
                                   camera_topic=ready.get("camera_topic"))
    except SpeedLimitError as exc:
        # Refused before motion is enabled: nothing has been commanded.
        print(f"error: {exc}", file=sys.stderr)
        link.close()
        return 2
    print(f"discovered {ready['robot']} on "
          f"{'/' + ready['namespace'] + '/*' if ready['namespace'] else 'the bare contract'}; "
          f"stop command: {ready.get('stop_command')}")
    link.enable_motion()

    try:
        return _loop(options, PROFILES[options.robot], frontend, link)
    finally:
        # Every exit path: the supervisor sends the stop command three times and goes.
        link.close()
        if link.stopped_reason:
            print(f"robot stopped ({link.stopped_reason}).")


def _loop(options: Options, profile: RobotProfile, frontend: Frontend, link: SupervisedLink) -> int:
    frames: "queue.Queue" = queue.Queue(maxsize=64)
    odom_box: dict = {"value": None, "count": 0}

    def on_frame(msg: dict) -> None:
        try:
            frames.put_nowait((msg, time.monotonic()))
        except queue.Full:
            pass

    def on_odom(odom) -> None:
        odom_box["value"] = odom
        odom_box["count"] += 1

    if profile.has_odom:
        link.subscribe_odom(on_odom)
    link.subscribe_camera(on_frame)

    state = TeleopState(
        speed=options.speed,
        speed_max=options.max_speed,
        hold_timeout=options.hold_timeout,
        speed_min=profile.speed_min,
        speed_step=profile.speed_step,
        turn_ratio=profile.turn_ratio,
        turn_max=profile.turn_max,
    )
    head: Optional[HeadPose] = None
    if profile.has_head:
        from robot_console import ainex_topics
        from robot_console.ainex_link import HEAD_RATE

        head = HeadPose(pan_limit=ainex_topics.HEAD_PAN_LIMIT,
                        tilt_limit=ainex_topics.HEAD_TILT_LIMIT, rate=HEAD_RATE)

    recorder: Optional[Recorder] = None
    t0 = time.monotonic()
    if options.record:
        recorder = Recorder(options.record, t0=t0)
        topics = {"camera": options.camera_topic}
        recorder.start({
            "robot": profile.name,
            "url": options.url,
            "namespace": options.namespace,
            "topics": topics,
            "speed": options.speed,
            "speed_max": options.max_speed,
            "hold_timeout": options.hold_timeout,
            "safety_timeout": SAFETY_TIMEOUT,
            "robot_console_version": __import__("robot_console").__version__,
        })

    release = ("when you press another motion key, Space or Esc" if options.latch
               else "0.6 s after you let go")
    print(BANNER.format(version=__import__("robot_console").__version__, url=options.url,
                        robot=profile.name, keys=_banner_keys(profile), release=release))
    if recorder:
        print(f"recording to {options.record}")

    def _bail(signum, _frame):
        state.running = False
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass

    window = f"{WINDOW} ({profile.name})"
    frontend.open(window)
    frame = placeholder(message=f"waiting for {options.camera_topic} ...")
    head_pose = None if head is None else (head.pan, head.tilt)
    frontend.show(draw_overlay(frame, show_help=state.show_help, speed=state.speed,
                               hints=profile.hints, head=head_pose))

    tick_ms = max(1, int(1000.0 / options.loop_hz))
    last_sent = None
    last_resend = 0.0
    last_odom_logged = -1
    last_status = 0.0
    last_head_key = time.monotonic()
    exit_reason = "esc"
    code = 0

    try:
        while state.running:
            key = frontend.poll_key(tick_ms)
            now = time.monotonic()
            if not link.alive:
                exit_reason = "supervisor"
                print(f"\nerror: the safety supervisor stopped the robot "
                      f"({link.stopped_reason or 'it exited'}); quitting.", file=sys.stderr)
                code = 1
                break
            link.heartbeat()

            action = action_for_key(key)
            if action is not Action.NONE:
                state.apply(action, now)
                if action is Action.QUIT:
                    exit_reason = "esc"
                    break
                if recorder and action in (Action.FASTER, Action.SLOWER):
                    recorder.add_event("speed", speed=round(state.speed, 4), t=now)
                if head is not None and action in HEAD_ACTIONS:
                    if head.apply(action, now - last_head_key):
                        link.publish_head(head.pan, head.tilt)
                        if recorder:
                            recorder.add_event("head", pan=round(head.pan, 4),
                                               tilt=round(head.tilt, 4), t=now)
                    last_head_key = now

            # No key-up event exists, so a held key is recognised by its OS auto-repeat
            # and the motion is dropped 0.6 s after the repeats stop (unless latched).
            if state.expire(now) and recorder:
                recorder.add_event("release", t=now)

            if not frontend.is_open():
                exit_reason = "window_closed"
                break

            # The desired command goes to the supervisor when it changes and at 5 Hz
            # anyway; the heartbeat above is what keeps it alive between.
            command = state.command()
            if command != last_sent or now - last_resend >= 0.2:
                link.publish_cmd_vel(command)
                if recorder and command != last_sent:
                    recorder.add_command(command, speed=state.speed,
                                         action=state.last_action.value,
                                         key=key_label(key), t=now)
                last_sent = command
                last_resend = now

            while True:
                try:
                    message, arrival = frames.get_nowait()
                except queue.Empty:
                    break
                decoded = decode_image(message)
                if decoded is not None:
                    frame = decoded
                    if recorder:
                        recorder.add_frame(frame, t=arrival, seq=header_seq(message))

            odom = odom_box["value"]
            if recorder and odom is not None and odom_box["count"] != last_odom_logged:
                recorder.add_odom(odom, t=now)
                last_odom_logged = odom_box["count"]

            frontend.show(draw_overlay(
                frame, show_help=state.show_help, speed=state.speed,
                speed_max=state.speed_max, moving=state.is_moving, hints=profile.hints,
                head=None if head is None else (head.pan, head.tilt),
            ))

            if now - last_status >= 1.0:
                last_status = now
                _print_status(state, odom, has_odom=profile.has_odom)
    except KeyboardInterrupt:
        exit_reason = "interrupt"
    finally:
        # Stop the robot before anything slow: the supervisor sends the stop command.
        link.close()
        if recorder:
            recorder.add_event("quit", reason=exit_reason)
            summary = recorder.close()
            print(f"\nrecorded {summary.get('frames', 0)} frames, "
                  f"{summary.get('commands', 0)} commands to {options.record}")
            if not summary.get("frames"):
                print("(no camera frames arrived, so no feed.mp4 was written)")
        frontend.close()

    print("stopped.")
    return code


def _print_status(state: TeleopState, odom, *, has_odom: bool = True) -> None:
    command = state.command()
    speed_note = ""
    if state.at_max_speed:
        speed_note = " (max)"
    elif state.speed <= state.speed_min + 1e-9:
        speed_note = " (min)"
    if odom is not None:
        pose = f"x={odom.x:+.2f} y={odom.y:+.2f} yaw={odom.yaw:+.2f}"
    else:
        pose = "no odom" if has_odom else "odom n/a"
    sys.stdout.write(
        f"\rspeed {state.speed:.2f}{speed_note:6s} "
        f"cmd [{command.vx:+.2f} {command.vy:+.2f} {command.wz:+.2f}]  {pose}    "
    )
    sys.stdout.flush()
