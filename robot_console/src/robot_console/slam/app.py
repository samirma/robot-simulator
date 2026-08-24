"""The loop all three SLAM commands are modes of.

Same shape as `robot_console.app`, and for the same reasons: **one loop, on the main
thread**. `cv2.imshow`/`waitKey` must own the main thread on macOS, and a second thread
publishing `/cmd_vel` would keep the robot driving while the UI was wedged. Here that
matters more than in teleop, because an autonomous mode has no human noticing that the
window has stopped repainting -- a UI stall must starve the command stream and stop the
robot, not free it to keep going.

Which is also why the expensive part of SLAM is keyframed rather than run every tick. The
budget is real and it is checked: `_Budget` watches the loop's own tick time and says so
when a tick overruns the publish period, because the failure mode otherwise is a robot
that drives fine and maps badly with nothing in the log to explain it.

Stopping the robot on the way out is not best-effort. The simulator has a 0.5 s watchdog;
the **real myAGV has none** -- `myagv_odometry_node` latches the last Twist and writes it
to the motors at 100 Hz forever. Hence the signal handlers and the `finally`.
"""

from __future__ import annotations

import math
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from robot_console.bridge import Odom, RobotLink, quiet_roslibpy_logging
from robot_console.camera import LatestFrame, decode_compressed_image, header_seq
from robot_console.hud import draw_overlay, placeholder as camera_placeholder
from robot_console.preflight import preflight
from robot_console.recorder import Recorder
from robot_console.slam import frontier as frontier_mod
from robot_console.slam import mapio
from robot_console.slam.cli import SlamOptions
from robot_console.slam.controller import PathFollower, scale_to_limits
from robot_console.slam.explorer import SWEEP_RATE, Explorer
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.mapview import MapView, placeholder as map_placeholder
from robot_console.slam.planner import CLEARANCE_M, CostMap, costmap_for, plan
from robot_console.slam.pose import PoseTracker
from robot_console.slam.scan import LaserScan, parse_scan, scan_points, transform_points
from robot_console.teleop import Action, Command, TeleopState, action_for_key

MAP_WINDOW = "robot_console - map"
CAMERA_WINDOW = "robot_console - camera"

KEY_SAVE = ord("m")
KEY_PLAN = ord("p")

# How far the robot may be from the last planned-against pose before the path is stale.
REPLAN_DISTANCE_M = 0.6
REPLAN_SECONDS = 3.0

# Map redraw rate. Well below the loop rate on purpose -- see the render call.
MAP_RENDER_HZ = 15.0

BANNER = """robot_console {version} -- {mode}  ->  {url}

{keys}
The map window must have focus for keys to register.
"""

KEYS_MANUAL = """  W / S   forward / back        Space  stop
  A / D   strafe left / right   M      save the map now
  Q / E   rotate left / right   Esc    quit (saves)"""

KEYS_EXPLORE = """  Space   pause / resume        M      save the map now
  W/A/S/D/Q/E             drive manually while paused
  Esc     quit (saves)"""

KEYS_NAVIGATE = """  left click   drive to that point   Space  stop / cancel
  right click  cancel the goal        M      save the map
  W/A/S/D/Q/E  drive manually         Esc    quit"""


class _Budget:
    """Watches tick time against the publish period.

    An overrun means `/cmd_vel` went out late, which on the simulator trips the 0.5 s
    watchdog and on hardware just means the robot kept its last command for longer than
    intended. Either way it is worth knowing, and it is invisible without measuring.
    """

    def __init__(self, period: float) -> None:
        self.period = period
        self.worst = 0.0
        self.overruns = 0
        self._warned = False

    def sample(self, elapsed: float, stream=sys.stderr) -> None:
        self.worst = max(self.worst, elapsed)
        if elapsed > self.period:
            self.overruns += 1
            if not self._warned and self.overruns > 5:
                self._warned = True
                print(
                    f"\nwarning: SLAM ticks are overrunning the {self.period * 1000:.0f} ms "
                    f"publish period (worst {self.worst * 1000:.0f} ms). Lower --slam-hz, "
                    f"raise --resolution, or lower --max-range.",
                    file=stream,
                )


def run(options: SlamOptions) -> int:
    quiet_roslibpy_logging()

    if options.preflight and not preflight(
        options.host, options.port, timeout=options.preflight_timeout
    ):
        return 2

    grid = _initial_grid(options)
    tracker = PoseTracker(match_enabled=not options.no_match, min_interval=1.0 / options.slam_hz)

    link = RobotLink(
        options.host,
        options.port,
        cmd_topic=options.cmd_topic,
        odom_topic=options.odom_topic,
        camera_topic=options.camera_topic,
        scan_topic=options.scan_topic,
    )
    try:
        link.connect(timeout=options.connect_timeout)
    except Exception as exc:
        print(f"error: could not connect to {options.url}: {exc}", file=sys.stderr)
        if not options.preflight:
            print("(--no-preflight was given, so the reachability check was skipped)", file=sys.stderr)
        return 2

    latest_scan = LatestFrame()
    latest_frame = LatestFrame()
    odom_box: dict = {"value": None, "count": 0}

    def on_odom(odom: Odom) -> None:
        odom_box["value"] = odom
        odom_box["count"] += 1

    link.subscribe_odom(on_odom)
    link.subscribe_scan(latest_scan.offer)
    if options.camera_window:
        link.subscribe_camera(latest_frame.offer)

    state = TeleopState(
        speed=options.speed, speed_max=options.max_speed, hold_timeout=options.hold_timeout
    )
    follower = PathFollower(speed=options.speed, speed_max=options.max_speed)
    view = MapView(zoom=options.zoom)

    keys = {"explore": KEYS_EXPLORE, "navigate": KEYS_NAVIGATE}.get(options.mode, KEYS_MANUAL)
    print(BANNER.format(
        version=__import__("robot_console").__version__,
        mode=options.mode, url=options.url, keys=keys,
    ))
    print(f"map -> {options.out}")

    recorder: Optional[Recorder] = None
    t0 = time.monotonic()
    if options.record:
        recorder = Recorder(options.record, t0=t0)
        recorder.start({
            "mode": options.mode,
            "host": options.host,
            "port": options.port,
            "topics": {
                "cmd_vel": options.cmd_topic, "odom": options.odom_topic,
                "camera": options.camera_topic, "scan": options.scan_topic,
            },
            "resolution": options.resolution,
            "map_out": str(options.out),
            "robot_console_version": __import__("robot_console").__version__,
        })

    def _bail(signum, _frame):
        state.running = False
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass

    cv2.namedWindow(MAP_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(MAP_WINDOW, view.on_mouse)
    cv2.imshow(MAP_WINDOW, map_placeholder(f"waiting for {options.scan_topic} ..."))
    if options.camera_window:
        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(CAMERA_WINDOW, camera_placeholder(message=f"waiting for {options.camera_topic} ..."))

    session = _Session(options, grid, tracker, follower, view, link, state, recorder)
    budget = _Budget(1.0 / options.publish_hz)
    tick_ms = max(1, int(1000.0 / options.loop_hz))
    publish_period = 1.0 / options.publish_hz
    next_publish = time.monotonic()
    last_status = 0.0
    last_autosave = time.monotonic()
    map_render_period = 1.0 / MAP_RENDER_HZ
    next_render = 0.0
    map_image = None
    exit_reason = "esc"
    frame = camera_placeholder(message=f"waiting for {options.camera_topic} ...")

    try:
        while state.running:
            key = cv2.waitKey(tick_ms)
            # Timed from *after* waitKey: its up-to-`tick_ms` wait is deliberate idle, and
            # counting it as work makes every tick look like an overrun on a loop that is
            # keeping up perfectly well.
            tick_started = time.monotonic()
            now = tick_started

            action = action_for_key(key)
            if action is not Action.NONE:
                state.apply(action, now)
                session.on_action(action, now)
            elif (key & 0xFF) == KEY_SAVE:
                session.save("manual")
            state.expire(now)

            if not _window_alive(MAP_WINDOW):
                exit_reason = "window_closed"
                break

            # Odom before the scan, not after: integrating a scan needs a pose to put it
            # at, and taking them the other way round silently drops the first scan --
            # and with it the first frontier the explorer would have aimed for.
            odom = odom_box["value"]
            if odom is not None:
                tracker.update_odom(odom)
                if recorder and odom_box["count"] != session.last_odom_logged:
                    recorder.add_odom(odom, t=now)
                    session.last_odom_logged = odom_box["count"]

            pending = latest_scan.take()
            if pending is not None:
                session.on_scan(parse_scan(pending[0]), now)

            command = session.decide(now)

            if now >= next_publish:
                link.publish_cmd_vel(command)
                if recorder:
                    recorder.add_command(
                        command, speed=state.speed, action=session.activity, t=now
                    )
                next_publish += publish_period
                if next_publish < now:
                    # After a stall, resync rather than firing a catch-up burst.
                    next_publish = now + publish_period

            if options.camera_window:
                pending_frame = latest_frame.take()
                if pending_frame is not None:
                    decoded = decode_compressed_image(pending_frame[0])
                    if decoded is not None:
                        frame = decoded
                        if recorder:
                            recorder.add_frame(frame, t=pending_frame[1], seq=header_seq(pending_frame[0]))
                if not _window_alive(CAMERA_WINDOW):
                    exit_reason = "window_closed"
                    break
                cv2.imshow(CAMERA_WINDOW, draw_overlay(
                    frame, show_help=False, speed=state.speed,
                    speed_max=state.speed_max, moving=not command.is_zero(),
                ))

            # The map is redrawn at MAP_RENDER_HZ, not at the loop rate. Rebuilding and
            # rescaling a house-sized grid 60 times a second is the single most expensive
            # thing in the loop and nobody can see the difference above ~15 Hz.
            if now >= next_render:
                next_render = now + map_render_period
                map_image = session.render(keys.splitlines())
            if map_image is not None:
                cv2.imshow(MAP_WINDOW, map_image)

            if options.autosave and now - last_autosave >= options.autosave:
                last_autosave = now
                session.save("autosave", quiet=True)

            if options.timeout and now - t0 >= options.timeout:
                exit_reason = "timeout"
                break
            if session.finished:
                exit_reason = session.finished
                break

            if now - last_status >= 1.0:
                last_status = now
                session.print_status()

            budget.sample(time.monotonic() - tick_started)

    except KeyboardInterrupt:
        exit_reason = "interrupt"
    finally:
        # Order matters: stop the robot before spending time on file handles.
        try:
            link.stop()
        except Exception:
            pass
        saved = session.save(exit_reason)
        if recorder:
            recorder.add_event("quit", reason=exit_reason)
            summary = recorder.close()
            print(f"\nrecorded {summary.get('frames', 0)} frames, "
                  f"{summary.get('commands', 0)} commands to {options.record}")
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass
        link.close()

    print(f"\nstopped ({exit_reason}).")
    if saved:
        print(f"map saved: {saved}")
        detail = mapio.describe(saved)
        if detail:
            print(f"  {detail}")
    if budget.overruns:
        print(f"note: {budget.overruns} tick(s) overran the publish period, "
              f"worst {budget.worst * 1000:.0f} ms")
    return 0


class _Session:
    """Per-mode behaviour, kept out of the loop above so the loop stays readable."""

    def __init__(self, options, grid, tracker, follower, view, link, state, recorder):
        self.options = options
        self.grid: OccupancyGrid = grid
        self.tracker: PoseTracker = tracker
        self.follower: PathFollower = follower
        self.view: MapView = view
        self.link = link
        self.state: TeleopState = state
        self.recorder = recorder

        self.scan: Optional[LaserScan] = None
        self.points = np.empty((0, 2))
        self.path: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None
        self.cost: Optional[CostMap] = None
        self.frontiers: List = []
        self.trail: List[np.ndarray] = []
        self.finished: Optional[str] = None
        # `--robot-radius` used to be parsed, clamped and then dropped on the floor: every
        # costmap was built from the module default regardless of what was asked for.
        self._plan_radius = float(options.robot_radius) + CLEARANCE_M
        self.explorer = Explorer(
            min_cells=options.frontier_min_cells,
            distance_bias=options.distance_bias,
            stall_seconds=options.stall_timeout,
        )
        self.last_odom_logged = -1
        self.scans = 0
        self.integrated = 0
        self.activity = "idle"
        self.note = ""
        self._planned_at: Optional[np.ndarray] = None
        self._planned_when = 0.0
        self._followed_goal: Optional[np.ndarray] = None
        self._localized = options.mode != "navigate" or options.map_source is None
        # Exploring starts driving by itself; the other two wait to be told.
        self.auto = options.mode == "explore"

    # ------------------------------------------------------------------ inputs

    def on_action(self, action: Action, now: float) -> None:
        if action is Action.QUIT:
            self.state.running = False
        elif action is Action.STOP:
            # Space means stop, and in an autonomous mode it also means "stop deciding".
            if self.options.mode == "explore":
                self.auto = not self.auto
                self.note = "paused" if not self.auto else "exploring"
            self.goal = None
            self.path = None
        elif action in (Action.FORWARD, Action.BACK, Action.STRAFE_LEFT,
                        Action.STRAFE_RIGHT, Action.ROT_LEFT, Action.ROT_RIGHT):
            # Any manual input takes precedence: a human reaching for the keys while the
            # robot is driving itself wants it to stop driving itself.
            if self.options.mode == "explore":
                self.auto = False
                self.note = "manual"
            self.goal = None
            self.path = None

    def on_scan(self, scan: LaserScan, now: float) -> None:
        self.scan = scan
        self.scans += 1
        self.points = scan_points(scan, max_range=self.options.max_range)
        if self.points.shape[0] == 0 or not self.tracker.has_odom:
            return

        if not self._localized:
            # A loaded map's frame has nothing to do with the odom frame the robot booted
            # in. Seed at the odom origin and let one unrestricted match find the offset.
            self.tracker.seed(self.tracker.pose)
            self.tracker.refine(self.grid, self.points, now=now)
            self._localized = True

        if self.tracker.keyframe_due(now):
            self.tracker.refine(self.grid, self.points, now=now)

        pose = self.tracker.pose
        self.grid.integrate(pose, transform_points(self.points, pose),
                            max_range=self.options.max_range)
        self.integrated += 1
        if not self.trail or float(np.hypot(*(pose[:2] - self.trail[-1]))) > 0.1:
            self.trail.append(pose[:2].copy())

    # ------------------------------------------------------------------ decision

    def decide(self, now: float) -> Command:
        if self.state.is_moving:
            self.activity = "manual"
            return self.state.command()

        if self.options.mode == "explore" and self.auto:
            return self._explore(now)
        if self.options.mode == "navigate" and self.goal is not None:
            return self._navigate(now)

        # `map` mode, and the others when idle: read the mouse anyway so a click always
        # does something predictable.
        click = self.view.take_click()
        if click is not None and self.options.mode == "navigate":
            self._set_goal(click, now)
        if self.view.take_cancel():
            self.goal = None
            self.path = None
        self.activity = "idle"
        return Command()

    def _explore(self, now: float) -> Command:
        if not self.tracker.has_odom or self.integrated == 0:
            # Nothing has been mapped yet, so "no frontiers" would mean "no data", not
            # "finished". Waiting a tick for the first scan is the difference between
            # exploring a house and exiting immediately.
            return Command()
        pose = self.tracker.pose

        if self.explorer.sweeping(now):
            self.activity = "explore"
            return Command(wz=SWEEP_RATE)

        if self.explorer.needs_replan(pose, now):
            self.cost = costmap_for(
                self.grid, self.cost, allow_unknown=True, radius=self._plan_radius
            )
            self._apply(self.explorer.replan(self.grid, self.cost, pose, now))
        if self.finished is not None:
            return Command()
        if self.path is None:
            return Command(wz=SWEEP_RATE) if self.explorer.sweeping(now) else Command()

        result = self.follower.step(pose, self.path, scan=self.scan, now=now)
        if result.arrived:
            self.note = "reached frontier"
            self.explorer.on_arrived(now)
            self.path = None
            return Command()
        if self.follower.is_stuck(now):
            # Real failure: time has passed and the robot got no closer.
            strikes = self.explorer.on_stuck(now)
            self.path = None
            self.goal = None
            self.note = f"stuck, trying elsewhere (strike {strikes})"
            self.follower.reset()
            return Command()
        if result.should_replan:
            # Only a local obstruction. It says nothing about whether the frontier is a
            # good goal, so the goal is kept and just the route is redrawn -- suppressing
            # it here burns through every frontier in the house in a couple of seconds
            # while the robot never moves.
            self.explorer.on_blocked(now)
            self.path = None
            self.note = "blocked, rerouting"
            return Command()
        self.activity = "explore"
        return scale_to_limits(result.command, self.options.max_speed)

    def _apply(self, decision) -> None:
        """Fold an `Explorer.Decision` into the session's own state."""
        self.path = decision.path
        self.goal = decision.goal
        self.frontiers = decision.frontiers
        if decision.note:
            self.note = decision.note
        if decision.finished:
            self.finished = decision.finished
        if decision.path is not None:
            self._reset_follower_for(decision.goal)

    def _navigate(self, now: float) -> Command:
        pose = self.tracker.pose
        if self.view.take_cancel():
            self.goal = None
            self.path = None
            self.note = "cancelled"
            return Command()
        click = self.view.take_click()
        if click is not None:
            self._set_goal(click, now)

        if self.path is None:
            return Command()
        result = self.follower.step(pose, self.path, scan=self.scan, now=now)
        if result.arrived:
            self.note = "arrived"
            self.goal = None
            self.path = None
            return Command()
        if result.should_replan or self.follower.is_stuck(now):
            self._replan_goal(pose, now)
            return Command()
        if self._needs_replan(pose, now):
            self._replan_goal(pose, now)
        self.activity = "navigate"
        return scale_to_limits(result.command, self.options.max_speed)

    # ------------------------------------------------------------------ planning

    def _set_goal(self, point: np.ndarray, now: float) -> None:
        self.goal = np.asarray(point, dtype=np.float64)[:2]
        self._replan_goal(self.tracker.pose, now, announce=True)

    def _replan_goal(self, pose, now: float, *, announce: bool = False) -> None:
        if self.goal is None:
            return
        self.cost = costmap_for(
            self.grid, self.cost, allow_unknown=False, radius=self._plan_radius
        )
        self.path = plan(self.cost, pose[:2], self.goal)
        self._planned_at = np.asarray(pose[:2]).copy()
        self._planned_when = now
        self._reset_follower_for(self.goal)
        if self.path is None:
            self.note = f"no path to ({self.goal[0]:.2f}, {self.goal[1]:.2f})"
            self.goal = None
        elif announce:
            self.note = f"driving to ({self.goal[0]:.2f}, {self.goal[1]:.2f})"

    def _reset_follower_for(self, goal) -> None:
        """Clear the stuck watchdog only when the goal genuinely changed.

        Resetting on every replan looks harmless and is not: a robot pinned against
        something reroutes to the *same* goal every tick, and each reset restarts the
        watchdog, so `is_stuck` can never fire and the robot spins in place forever
        instead of blacklisting the goal and going elsewhere.
        """
        goal = np.asarray(goal, dtype=np.float64)[:2]
        if self._followed_goal is None or float(np.hypot(*(goal - self._followed_goal))) > 1e-6:
            self._followed_goal = goal.copy()
            self.follower.reset()

    def _needs_replan(self, pose, now: float) -> bool:
        if self.path is None or self._planned_at is None:
            return True
        if now - self._planned_when >= REPLAN_SECONDS:
            return True
        return float(np.hypot(*(pose[:2] - self._planned_at))) >= REPLAN_DISTANCE_M

    # ------------------------------------------------------------------ output

    def save(self, reason: str, *, quiet: bool = False) -> Optional[Path]:
        if self.options.mode == "navigate" and reason not in ("manual",):
            # Navigating updates the map with whatever it sees, but the saved map is the
            # user's artefact; overwriting it on every exit would let one bad run corrupt
            # a map that took a full exploration to build.
            return None
        try:
            path = mapio.save_map(self.grid, self.options.out)
        except OSError as exc:
            print(f"\nwarning: could not save map: {exc}", file=sys.stderr)
            return None
        if not quiet and reason == "manual":
            print(f"\nsaved {path}")
        if self.recorder:
            self.recorder.add_event("map_saved", reason=reason, path=str(path))
        return path

    def render(self, hints) -> np.ndarray:
        pose = self.tracker.pose if self.tracker.has_odom else None
        world_points = (
            transform_points(self.points, pose) if pose is not None and len(self.points) else None
        )
        return self.view.render(
            self.grid,
            pose=pose,
            path=self.path,
            goal=self.goal,
            scan_points=world_points,
            frontiers=self.frontiers if self.options.mode == "explore" else None,
            trail=self.trail,
            status=self._status_lines(),
            hints=hints,
        )

    def _status_lines(self) -> List[str]:
        pose = self.tracker.pose
        stats = self.tracker.stats
        area = frontier_mod.explored_area(self.grid)
        match = "off" if self.options.no_match else (
            f"{stats.matches}/{stats.keyframes} score {stats.last_score:.2f}"
        )
        lines = [
            f"{self.options.mode}  pose {pose[0]:+.2f} {pose[1]:+.2f} {math.degrees(pose[2]):+.0f}deg"
            f"   mapped {area:.1f} m2   scans {self.scans}",
            f"match {match}   slam {stats.last_ms:.0f} ms (worst {stats.worst_ms:.0f})"
            f"   {self.activity}",
        ]
        if self.note:
            lines.append(self.note)
        return lines

    def print_status(self) -> None:
        pose = self.tracker.pose
        sys.stdout.write(
            f"\r{self.options.mode:9s} pose {pose[0]:+.2f} {pose[1]:+.2f} "
            f"{math.degrees(pose[2]):+6.1f}deg  mapped {frontier_mod.explored_area(self.grid):6.1f} m2  "
            f"scans {self.scans:5d}  {self.activity:9s} {self.note[:40]:40s}"
        )
        sys.stdout.flush()


def _initial_grid(options: SlamOptions) -> OccupancyGrid:
    source = options.map_source
    if source is not None:
        try:
            grid = mapio.load_map(source)
            print(f"loaded map from {source}: {mapio.describe(source)}")
            return grid
        except (OSError, ValueError) as exc:
            if options.mode == "navigate":
                raise SystemExit(
                    f"error: navigate needs a map and could not load one from {source}: {exc}\n"
                    f"Build one first with `slam.sh explore --out {source}`."
                )
            print(f"warning: could not load {source} ({exc}); starting a new map", file=sys.stderr)
    return OccupancyGrid(options.resolution)


def _window_alive(name: str) -> bool:
    """The window's close button is a legitimate quit, and ignoring it leaves a headless
    loop still driving the robot."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False
