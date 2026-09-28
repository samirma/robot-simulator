"""The loop all three SLAM modes are modes of.

Same shape as `robot_console.app`, and for the same reasons (console spec §3): **one
loop, on the main thread**, which never publishes motion itself. Every command goes over
a pipe to the safety supervisor (`robot_console.supervisor`), which owns the rosbridge
connection. The loop heartbeats every tick; a wedged loop -- and an autonomous mode has no
human watching the window repaint -- stops heartbeating, and the supervisor sends the
myAGV's stop command. That is also why the expensive part of SLAM is keyframed: a tick
that overruns the safety timeout stops the robot, and `_Budget` says when ticks get close.

Stopping on the way out is not best-effort: the myAGV has no command watchdog. Every exit
path here (Esc, window close, `explored`, `limit`, exception, SIGINT/SIGTERM) closes the
supervisor link, and the supervisor sends the stop command three times.
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

from robot_console.camera import LatestFrame, decode_image
from robot_console.hud import draw_overlay, placeholder as camera_placeholder
from robot_console.robots import SLAM_ROBOT
from robot_console.slam import frontier as frontier_mod
from robot_console.slam import mapio
from robot_console.slam.cli import SlamOptions
from robot_console.slam.controller import PathFollower, scale_to_limits
from robot_console.slam.explorer import LIMIT, SWEEP_RATE, Explorer
from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.mapview import MapView, placeholder as map_placeholder
from robot_console.slam.planner import CLEARANCE_M, CostMap, costmap_for, plan
from robot_console.slam.pose import PoseTracker
from robot_console.slam.scan import LaserScan, parse_scan, scan_points, transform_points
from robot_console.supervisor import SAFETY_TIMEOUT, SupervisedLink, SupervisorError
from robot_console.teleop import Action, Command, TeleopState, action_for_key

MAP_WINDOW = "robot_console - map"
CAMERA_WINDOW = "robot_console - camera"

KEY_SAVE = ord("m")

REPLAN_DISTANCE_M = 0.6
REPLAN_SECONDS = 3.0

# Map redraw rate. Well below the loop rate on purpose -- see the render call.
MAP_RENDER_HZ = 15.0

# The desired command is re-sent to the supervisor at least this often; it re-publishes
# to the robot at its own 20 Hz in between.
RESEND_S = 0.2

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
    """Watches tick time against half the safety timeout.

    A tick longer than the safety timeout is a missed heartbeat, and the supervisor stops
    the robot for it; this says so before it happens rather than after.
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
                    f"\nwarning: SLAM ticks are taking up to {self.worst * 1000:.0f} ms, "
                    f"close to the safety timeout; the supervisor stops the robot on a "
                    f"missed heartbeat.",
                    file=stream,
                )


def run(options: SlamOptions) -> int:
    from robot_console.app import ESTOP_PROMPT, CvFrontend

    try:
        grid, seed_pose = _initial_grid(options)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    if not CvFrontend().confirm_estop(ESTOP_PROMPT):
        print("error: motion needs a confirmed independent emergency stop; not starting.",
              file=sys.stderr)
        return 2

    # Always discovered: --namespace narrows the myAGVs /rosapi reports (console spec §2.2).
    link = SupervisedLink(options.url, robot=SLAM_ROBOT, namespace=options.namespace,
                          discover=True)
    try:
        ready = link.start()
    except SupervisorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"discovered {ready['robot']} on "
          f"{'/' + ready['namespace'] + '/*' if ready['namespace'] else 'the bare contract'}")
    link.enable_motion()
    try:
        return _run(options, grid, seed_pose, link, ready)
    finally:
        link.close()


def _run(options: SlamOptions, grid: OccupancyGrid, seed_pose, link: SupervisedLink,
         ready: dict) -> int:
    tracker = PoseTracker(match_enabled=not options.no_match, min_interval=1.0 / options.slam_hz)

    latest_scan = LatestFrame()
    latest_frame = LatestFrame()
    odom_box: dict = {"value": None, "count": 0}

    def on_odom(odom) -> None:
        odom_box["value"] = odom
        odom_box["count"] += 1

    link.subscribe_odom(on_odom)
    link.subscribe_scan(latest_scan.offer)
    if options.camera_window:
        link.subscribe_camera(latest_frame.offer)

    state = TeleopState(speed=options.speed, speed_max=options.max_speed,
                        hold_timeout=options.hold_timeout)
    follower = PathFollower(speed=options.speed, speed_max=options.max_speed)
    view = MapView(zoom=options.zoom)

    keys = {"explore": KEYS_EXPLORE, "navigate": KEYS_NAVIGATE}.get(options.mode, KEYS_MANUAL)
    print(BANNER.format(version=__import__("robot_console").__version__,
                        mode=options.mode, url=options.url, keys=keys))
    print(f"map -> {options.save_to}")

    def _bail(signum, _frame):
        state.running = False
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass

    session = _Session(options, grid, tracker, follower, view, link, state,
                       seed_pose=seed_pose)
    budget = _Budget(SAFETY_TIMEOUT / 2.0)
    tick_ms = max(1, int(1000.0 / options.loop_hz))
    last_status = 0.0
    last_autosave = time.monotonic()
    map_render_period = 1.0 / MAP_RENDER_HZ
    next_render = 0.0
    map_image = None
    exit_reason = "esc"
    code = 0
    last_sent: Optional[Command] = None
    last_resend = 0.0
    frame = camera_placeholder(message=f"waiting for {ready.get('camera_topic')} ...")

    cv2.namedWindow(MAP_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(MAP_WINDOW, view.on_mouse)
    cv2.imshow(MAP_WINDOW, map_placeholder(f"waiting for {ready.get('scan_topic')} ..."))
    if options.camera_window:
        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(CAMERA_WINDOW, frame)

    try:
        while state.running:
            key = cv2.waitKey(tick_ms)
            tick_started = time.monotonic()
            now = tick_started
            if not link.alive:
                exit_reason = "supervisor"
                print(f"\nerror: the safety supervisor stopped the robot "
                      f"({link.stopped_reason or 'it exited'}).", file=sys.stderr)
                code = 1
                break
            link.heartbeat()

            action = action_for_key(key)
            if action is Action.QUIT:
                exit_reason = "esc"
                break
            if action is not Action.NONE:
                state.apply(action, now)
                session.on_action(action, now)
            elif (key & 0xFF) == KEY_SAVE:
                session.save("manual")
            state.expire(now)

            if not _window_alive(MAP_WINDOW):
                exit_reason = "window_closed"
                break

            # Odom before the scan: integrating a scan needs a pose to put it at.
            odom = odom_box["value"]
            if odom is not None:
                session.on_odom(odom)

            pending = latest_scan.take()
            if pending is not None:
                session.on_scan(parse_scan(pending[0]), now)
                link.heartbeat()

            command = session.decide(now)
            link.heartbeat()
            if command != last_sent or now - last_resend >= RESEND_S:
                link.publish_cmd_vel(command)
                last_sent, last_resend = command, now

            if options.camera_window:
                pending_frame = latest_frame.take()
                if pending_frame is not None:
                    decoded = decode_image(pending_frame[0])
                    if decoded is not None:
                        frame = decoded
                if not _window_alive(CAMERA_WINDOW):
                    exit_reason = "window_closed"
                    break
                cv2.imshow(CAMERA_WINDOW, draw_overlay(
                    frame, show_help=False, speed=state.speed,
                    speed_max=state.speed_max, moving=not command.is_zero(),
                ))

            if now >= next_render:
                next_render = now + map_render_period
                map_image = session.render(keys.splitlines())
            if map_image is not None:
                cv2.imshow(MAP_WINDOW, map_image)

            if options.autosave and now - last_autosave >= options.autosave:
                last_autosave = now
                session.save("autosave", quiet=True)

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
        # Stop the robot before anything slow.
        link.close()
        saved = session.save(exit_reason)
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass

    print(f"\nstopped ({exit_reason}).")
    if options.mode == "explore":
        print(session.report(time.monotonic()))
    if saved:
        print(f"map saved: {saved}")
        detail = mapio.describe(saved)
        if detail:
            print(f"  {detail}")
    if budget.overruns:
        print(f"note: {budget.overruns} tick(s) took over half the safety timeout, "
              f"worst {budget.worst * 1000:.0f} ms")
    return code


class _Session:
    """Per-mode behaviour, kept out of the loop above so the loop stays readable and so a
    whole run can be driven offline (`tests/simworld.py`)."""

    def __init__(self, options, grid, tracker, follower, view, link, state,
                 *, seed_pose=None):
        self.options = options
        self.grid: OccupancyGrid = grid
        self.tracker: PoseTracker = tracker
        self.follower: PathFollower = follower
        self.view: MapView = view
        self.link = link
        self.state: TeleopState = state

        self.scan: Optional[LaserScan] = None
        self.points = np.empty((0, 2))
        self.path: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None
        self.cost: Optional[CostMap] = None
        self.frontiers: List = []
        self.trail: List[np.ndarray] = []
        self.finished: Optional[str] = None
        self._plan_radius = float(options.robot_radius) + CLEARANCE_M
        self.explorer = Explorer(distance_bias=options.distance_bias,
                                 max_goals=options.max_goals)
        self.last_odom_logged = -1
        self.scans = 0
        self.integrated = 0
        self.activity = "idle"
        self.note = ""
        self.started: Optional[float] = None
        self._area_cache = (-1, 0.0)
        self._planned_at: Optional[np.ndarray] = None
        self._planned_when = 0.0
        self._followed_goal: Optional[np.ndarray] = None
        # A continued or loaded map is in its own frame; the saved pose says where the
        # robot is in it. Navigate additionally refines that by one scan match.
        self._seed_pose = None if seed_pose is None else np.asarray(seed_pose, dtype=np.float64)
        self._localized = options.map_source is None
        # Exploring starts driving by itself; the other two wait to be told.
        self.auto = options.mode == "explore"

    # ------------------------------------------------------------------ inputs

    def on_odom(self, odom) -> None:
        self.tracker.update_odom(odom)

    def on_action(self, action: Action, now: float) -> None:
        if action is Action.QUIT:
            self.state.running = False
        elif action is Action.STOP:
            if self.options.mode == "explore":
                self.auto = not self.auto
                self.note = "paused" if not self.auto else "exploring"
            self.goal = None
            self.path = None
        elif action in (Action.FORWARD, Action.BACK, Action.STRAFE_LEFT,
                        Action.STRAFE_RIGHT, Action.ROT_LEFT, Action.ROT_RIGHT):
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
            self.tracker.seed(self._seed_pose if self._seed_pose is not None
                              else self.tracker.pose)
            if self.options.mode == "navigate":
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

    def area(self) -> float:
        revision = getattr(self.grid, "revision", None)
        if revision is None or revision != self._area_cache[0]:
            self._area_cache = (revision if revision is not None else -1,
                                frontier_mod.explored_area(self.grid))
        return self._area_cache[1]

    def decide(self, now: float) -> Command:
        if self.options.mode == "explore" and self.started is None:
            # From the first tick, so a run that never gets data still ends at the limit.
            self.started = now
        if self.options.mode == "explore" and self.finished is None:
            if now - self.started >= self.options.max_duration:
                self._apply(self.explorer.stop_for_limit(
                    f"--max-duration {self.options.max_duration:g} s reached"))
                return Command()

        if self.state.is_moving:
            self.activity = "manual"
            return self.state.command()

        if self.options.mode == "explore" and self.auto:
            return self._explore(now)
        if self.options.mode == "navigate" and self.goal is not None:
            return self._navigate(now)

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
            # Nothing mapped yet: "no frontiers" would mean "no data", not "finished".
            return Command()
        pose = self.tracker.pose
        self.activity = "explore"

        if self.explorer.sweeping(now):
            return Command(wz=SWEEP_RATE)

        if self.explorer.track(self.grid, pose, now, area=self.area()) == "timeout":
            self.path = None
            self.goal = None
            self.note = "goal blacklisted: no progress"
            self.follower.reset()

        if self.explorer.needs_replan(pose, now):
            self.cost = costmap_for(self.grid, self.cost, allow_unknown=True,
                                    radius=self._plan_radius)
            self._apply(self.explorer.replan(self.grid, self.cost, pose, now, area=self.area()))
        if self.finished is not None:
            return Command()
        if self.explorer.sweeping(now):
            return Command(wz=SWEEP_RATE)
        if self.path is None:
            return Command()

        result = self.follower.step(pose, self.path, scan=self.scan, now=now)
        if result.arrived:
            fruitless = self.explorer.on_arrived(self.grid, now, area=self.area())
            self.note = "reached a fruitless goal" if fruitless else "reached frontier"
            self.path = None
            return Command()
        if self.follower.is_stuck(now):
            # Not a blacklisting on its own: the goal's 90 s / 300 s clock does that.
            self.explorer.on_stuck(now)
            self.path = None
            self.note = "stuck, rerouting"
            self.follower.reset()
            return Command()
        if result.should_replan:
            self.explorer.on_blocked(now)
            self.path = None
            self.note = "blocked, rerouting"
            return Command()
        return scale_to_limits(result.command, self.options.max_speed)

    def _apply(self, decision) -> None:
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
        """Navigate plans with unknown space blocked (explore plans with it free)."""
        if self.goal is None:
            return
        self.cost = costmap_for(self.grid, self.cost, allow_unknown=False,
                                radius=self._plan_radius)
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
        """Clear the stuck watchdog only when the goal genuinely changed."""
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
        target = self.options.save_to
        if target is None:
            return None
        if self.options.mode == "navigate" and reason != "manual":
            # The navigated map is the user's artefact; only M overwrites it.
            return None
        if not self._localized:
            # Not yet placed in the continued map: the pose stored with it still holds.
            pose = self._seed_pose
        else:
            pose = self.tracker.pose if self.tracker.has_odom else None
        try:
            path = mapio.save_map(self.grid, target, pose=pose)
        except OSError as exc:
            print(f"\nwarning: could not save map: {exc}", file=sys.stderr)
            return None
        if not quiet and reason == "manual":
            print(f"\nsaved {path}")
        return path

    def report(self, now: float) -> str:
        elapsed = 0.0 if self.started is None else now - self.started
        return (f"explore ended: {self.finished or 'interrupted'} after {elapsed:.1f} s, "
                f"{self.explorer.attempted} goals attempted")

    def render(self, hints) -> np.ndarray:
        pose = self.tracker.pose if self.tracker.has_odom else None
        world_points = (
            transform_points(self.points, pose) if pose is not None and len(self.points) else None
        )
        return self.view.render(
            self.grid, pose=pose, path=self.path, goal=self.goal, scan_points=world_points,
            frontiers=self.frontiers if self.options.mode == "explore" else None,
            trail=self.trail, status=self._status_lines(), hints=hints,
        )

    def _status_lines(self) -> List[str]:
        pose = self.tracker.pose
        stats = self.tracker.stats
        match = "off" if self.options.no_match else (
            f"{stats.matches}/{stats.keyframes} score {stats.last_score:.2f}"
        )
        lines = [
            f"{self.options.mode}  pose {pose[0]:+.2f} {pose[1]:+.2f} {math.degrees(pose[2]):+.0f}deg"
            f"   mapped {self.area():.1f} m2   scans {self.scans}",
            f"match {match}   slam {stats.last_ms:.0f} ms (worst {stats.worst_ms:.0f})"
            f"   {self.activity}   goals {self.explorer.attempted}",
        ]
        if self.note:
            lines.append(self.note)
        return lines

    def print_status(self) -> None:
        pose = self.tracker.pose
        sys.stdout.write(
            f"\r{self.options.mode:9s} pose {pose[0]:+.2f} {pose[1]:+.2f} "
            f"{math.degrees(pose[2]):+6.1f}deg  mapped {self.area():6.1f} m2  "
            f"goals {self.explorer.attempted:4d}  {self.activity:9s} {self.note[:40]:40s}"
        )
        sys.stdout.flush()


def _initial_grid(options: SlamOptions):
    """The grid to start from and the robot pose saved with it (or None)."""
    source = options.map_source
    if source is not None:
        try:
            grid = mapio.load_map(source)
        except (OSError, ValueError, KeyError) as exc:
            if options.mode == "navigate":
                raise SystemExit(
                    f"error: navigate needs a map and could not load one from {source}: {exc}\n"
                    f"Build one first with `slam.sh explore --out {source}`."
                )
            print(f"warning: could not load {source} ({exc}); starting a new map", file=sys.stderr)
        else:
            if not math.isclose(grid.resolution, options.resolution):
                raise SystemExit(
                    f"error: {source} is a {grid.resolution:g} m/cell map; the console maps "
                    f"at {options.resolution:g} m/cell")
            pose = mapio.load_pose(source)
            print(f"loaded map from {source}: {mapio.describe(source)}"
                  + ("" if pose is None else
                     f"; the robot is assumed to start at the saved pose "
                     f"({pose[0]:+.2f}, {pose[1]:+.2f}, {math.degrees(pose[2]):+.0f} deg)"))
            return grid, pose
    elif options.mode == "navigate":
        raise SystemExit("error: navigate needs --map")
    return OccupancyGrid(options.resolution), None


def _window_alive(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


__all__ = ["run", "LIMIT"]
