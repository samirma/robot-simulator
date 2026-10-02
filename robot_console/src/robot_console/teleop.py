"""``teleop.sh``: keyboard teleoperation of a supported mobile robot, cameras shown live.

    teleop.sh [--robot <id>] [--namespace <name>] [--url ws://host:port]

A local pygame window owns keyboard focus. Held/released keys come from SDL key-down and
key-up events (key repeat is disabled and ignored), so a held key is known to be held.

Exit status: 0 ordinary exit (Esc or closing the window); 1 no validated target; 2 refused
(unknown id, arm, namespace override); 3 wire unreachable; 4 connection lost;
5 keyboard input lost; 130/143 after SIGINT/SIGTERM.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import Callable, List, Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from robot_console import camera as cam  # noqa: E402
from robot_console.discovery import (  # noqa: E402
    DiscoveryError, SelectionError, fetch_graph, select_target,
)
from robot_console.profiles import (  # noqa: E402
    ProfileError, check_namespace_allowed, refuse_arm, select_id, teleop_ids,
)
from robot_console.rosbridge import DEFAULT_URL, Rosbridge, TransportError, check_url  # noqa: E402
from robot_console.teleop_core import (  # noqa: E402
    REENABLE_LIMITATION, Sender, TeleopCore, no_delivery_statement,
)

EXIT_OK, EXIT_NO_TARGET, EXIT_REFUSED, EXIT_UNREACHABLE, EXIT_CONN_LOST, EXIT_INPUT_LOST = 0, 1, 2, 3, 4, 5
ADVERTISE_SETTLE_S = 0.6     # let a fresh ROS 1 publisher connect before the start-up stop

HELP = ("W/S forward/back  A/D strafe  Q/E rotate  Space stop  Enter re-enable  Esc quit")
HEAD_HELP = "Arrows: head (Left/Right pan, Up/Down tilt)"


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="teleop.sh",
        description="Keyboard teleoperation of a supported mobile robot over rosbridge, with its "
                    "cameras shown live. Motion is hold-to-move: releasing the last motion key, "
                    "Space, focus loss, input loss and exit request the robot's documented stop.",
        epilog=f"Teleoperable ids: {', '.join(teleop_ids())}. Keys: {HELP}. AiNex: {HEAD_HELP}. "
               "After validation teleop sends the documented stop once and, if it was delivered, "
               "enables commands at once. A failed stop (at start-up or later) disables commands "
               "until Enter and a fresh key press. If teleop is killed (e.g. SIGKILL) or the connection breaks, no "
               "stop can be sent and stopping is not guaranteed; none of the supported profiles "
               "documents a command watchdog. A lost connection ends teleop (status 4).")
    ap.add_argument("--robot", metavar="ID", help="robot profile (default: identify from the wire)")
    ap.add_argument("--namespace", metavar="NAME",
                    help="only for profiles whose hardware interface documents namespaces")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"rosbridge websocket (default {DEFAULT_URL})")
    return ap


def _key_name(pg, key: int) -> Optional[str]:
    table = {pg.K_w: "w", pg.K_s: "s", pg.K_a: "a", pg.K_d: "d", pg.K_q: "q", pg.K_e: "e",
             pg.K_SPACE: "space", pg.K_RETURN: "enter", pg.K_KP_ENTER: "enter",
             pg.K_ESCAPE: "esc", pg.K_LEFT: "left", pg.K_RIGHT: "right", pg.K_UP: "up",
             pg.K_DOWN: "down"}
    return table.get(key)


class TeleopApp:
    def __init__(self, url: str, profile, namespace: Optional[str],
                 events: Optional[Callable[[], list]] = None, max_seconds: Optional[float] = None,
                 err=sys.stderr) -> None:
        self.url = url
        self.profile = profile
        self.namespace = namespace
        self._events = events
        self.max_seconds = max_seconds
        self.err = err
        self.signal: Optional[int] = None
        self.lost: Optional[str] = None
        self.core: Optional[TeleopCore] = None
        self.target = None
        self.reason = ""
        self.cams: Optional[cam.CameraSet] = None
        self.lines: List[str] = []

    def say(self, text: str) -> None:
        self.lines.append(text)
        del self.lines[:-8]
        print(f"teleop: {text}", file=self.err, flush=True)

    # ------------------------------------------------------------ main
    def run(self) -> int:
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, self._on_signal)
        rb = Rosbridge(self.url)
        try:
            rb.connect(5.0)
        except TransportError as exc:
            self.say(f"wire unreachable: {exc}")
            return EXIT_UNREACHABLE
        rb.on_close(self._on_close)
        import pygame as pg
        self.pg = pg
        pg.display.init()
        pg.font.init()
        self.screen = pg.display.set_mode((1000, 640), pg.RESIZABLE)
        pg.display.set_caption(f"robot console teleop - {self.url}")
        pg.key.set_repeat()           # disabled: only real key-down/key-up events
        self.font = pg.font.Font(None, 20)
        self.big = pg.font.Font(None, 26)
        code = EXIT_OK
        try:
            code = self._session(rb)
        finally:
            if self.cams:
                self.cams.close()
            rb.close()
            pg.quit()
        return code

    def _on_signal(self, signum, _frame) -> None:
        self.signal = signum

    def _on_close(self, reason: str) -> None:
        self.lost = reason

    def _events_now(self) -> list:
        if self._events is not None:
            return self._events()
        return self.pg.event.get()

    def _session(self, rb: Rosbridge) -> int:
        pg = self.pg
        try:
            graph = fetch_graph(rb)
        except DiscoveryError as exc:
            self.say(f"cannot read the wire: {exc}")
            return EXIT_UNREACHABLE
        self.cams = cam.CameraSet(rb, cam.untied_specs(graph))
        try:
            self.target = select_target(graph, self.profile, self.namespace)
        except ProfileError as exc:
            self.say(f"refused: {exc}")
            return EXIT_REFUSED
        except SelectionError as exc:
            self.reason = exc.reason
            self.say(f"no validated target, commands refused: {exc.reason}. Select one explicitly "
                     f"with --robot ({', '.join(teleop_ids())}).")
        if self.target is not None and self.target.profile.is_arm:
            self.say("refused: " + refuse_arm(self.target.profile.id) + " (identified automatically)")
            return EXIT_REFUSED
        if self.target is not None:
            self.cams.close()
            self.cams = cam.CameraSet(rb, cam.target_specs(self.target), present=graph.topics)
            self.core = TeleopCore(self.target, Sender(rb), log=self.say)
            for topic, typ in self.core.command_topics():
                rb.advertise(topic, typ)
            self.say(f"target {self.target.label} validated; stop command: "
                     f"{self.target.profile.stop_description()}")
            self.say("if this process is killed (SIGKILL, power loss) or the connection breaks, no "
                     "stop can be sent: " + no_delivery_statement(self.target.profile))
            settle = time.monotonic() + ADVERTISE_SETTLE_S
            while time.monotonic() < settle and self.signal is None and self.lost is None:
                self._pump_display()
                time.sleep(0.02)
            if self.lost is None and self.signal is None:
                if self.core.startup_stop().ok:   # logs its outcome
                    self.core.enable()            # no Enter needed after a delivered stop
                else:
                    self.say("press Enter to enable commands")
        started = time.monotonic()
        clock = pg.time.Clock()
        code = EXIT_OK if self.target is not None else EXIT_NO_TARGET
        end_reason = "exit"
        while True:
            if self.signal is not None:
                code = 128 + int(self.signal)
                end_reason = f"signal {signal.Signals(self.signal).name}"
                break
            if self.lost is not None:
                if self.core:
                    self.core.connection_lost(self.lost)
                else:
                    self.say(f"connection lost ({self.lost})")
                self.say("teleop ends; relaunch to reconnect")
                self._draw()
                return EXIT_CONN_LOST
            if self.max_seconds is not None and time.monotonic() - started > self.max_seconds:
                end_reason = "time limit"
                break
            try:
                events = self._events_now()
            except Exception as exc:  # noqa: BLE001 - the key-event source failed
                if self.core:
                    self.core.input_lost(str(exc))
                else:
                    self.say(f"keyboard input lost: {exc}")
                return EXIT_INPUT_LOST
            if events is None:            # the key-event source ended
                if self.core:
                    self.core.input_lost("event source ended")
                return EXIT_INPUT_LOST
            quit_now = False
            for ev in events:
                if ev.type == pg.QUIT:
                    quit_now, end_reason = True, "window closed"
                elif ev.type == pg.KEYDOWN:
                    k = _key_name(pg, ev.key)
                    if k == "esc":
                        quit_now, end_reason = True, "Esc"
                    elif k and self.core:
                        self.core.key_down(k)
                    elif k:
                        self.say("commands refused: no validated target")
                elif ev.type == pg.KEYUP:
                    k = _key_name(pg, ev.key)
                    if k and self.core:
                        self.core.key_up(k)
                elif ev.type == getattr(pg, "WINDOWFOCUSLOST", -1) or (
                        ev.type == getattr(pg, "ACTIVEEVENT", -2) and getattr(ev, "gain", 1) == 0
                        and getattr(ev, "state", 0) & 2):
                    if self.core and (self.core.held or self.core.held_head or self.core.enabled):
                        self.core.focus_lost()
            if quit_now:
                break
            if self.core:
                self.core.tick()
            self._pump_display()
            clock.tick(30)
        if self.core:
            out = self.core.shutdown(f"teleop ending ({end_reason})")
            if not out.ok:
                self.say(REENABLE_LIMITATION)
        return code

    # ------------------------------------------------------------ drawing
    def _pump_display(self) -> None:
        if self.cams:
            self.cams.poll()
        self._draw()

    def _draw(self) -> None:
        pg = self.pg
        scr = self.screen
        W, H = scr.get_size()
        scr.fill((18, 20, 24))
        streams = list(self.cams.streams.values()) if self.cams else []
        hud_h = 190
        area_h = H - hud_h
        now = time.monotonic()
        if not streams:
            msg = "No cameras" + (" for this robot's profile" if self.target else " discovered on the wire")
            scr.blit(self.big.render(msg, True, (200, 200, 200)), (20, 20))
        else:
            n = len(streams)
            cols = 1 if n == 1 else 2
            rows = (n + cols - 1) // cols
            tw, th = W // cols, area_h // rows
            for i, s in enumerate(streams):
                x, y = (i % cols) * tw, (i // cols) * th
                st = s.state(now)
                if s.frame is not None and st == cam.LIVE:
                    surf = pg.surfarray.make_surface(s.frame.swapaxes(0, 1))
                    fh, fw = s.frame.shape[:2]
                    scale = min((tw - 8) / fw, (th - 30) / fh)
                    surf = pg.transform.smoothscale(surf, (max(1, int(fw * scale)), max(1, int(fh * scale))))
                    scr.blit(surf, (x + 4, y + 26))
                else:
                    pg.draw.rect(scr, (40, 40, 44), (x + 4, y + 26, tw - 8, th - 30))
                    txt = s.status_text(now).upper() if st != cam.WAITING else "WAITING FOR FRAMES"
                    scr.blit(self.big.render(txt, True, (240, 160, 60)), (x + 14, y + th // 2))
                colour = (120, 220, 120) if st == cam.LIVE else (240, 160, 60)
                tag = "" if s.spec.tied else "  [not tied to a target]"
                scr.blit(self.font.render(f"{s.spec.topic}: {s.status_text(now)}{tag}", True, colour),
                         (x + 6, y + 6))
        y = area_h + 6
        if self.target is not None:
            core = self.core
            state = "ENABLED" if core and core.enabled else "DISABLED (press Enter)"
            if core and not core.connected:
                state = "DISCONNECTED"
            head = f"{self.target.label} on {self.url}   commands: {state}"
        else:
            head = f"NO VALIDATED TARGET - commands refused: {self.reason}"
        scr.blit(self.big.render(head[:140], True, (255, 255, 255)), (10, y))
        y += 26
        help_line = HELP + ("   " + HEAD_HELP if self.target is not None and self.target.profile.head else "")
        scr.blit(self.font.render(help_line, True, (170, 170, 190)), (10, y))
        y += 22
        if self.core and self.core.held:
            a = self.core.axes()
            scr.blit(self.font.render(f"held {''.join(self.core.held).upper()}  x={a['x']:+.3f} "
                                      f"y={a['y']:+.3f} yaw={a['yaw']:+.3f}", True, (120, 220, 120)), (10, y))
        y += 20
        for line in self.lines[-6:]:
            colour = (255, 110, 110) if ("FAIL" in line or "LOST" in line or "NOT" in line) else (210, 210, 210)
            scr.blit(self.font.render(line[:170], True, colour), (10, y))
            y += 19
        pg.display.flip()


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        check_url(args.url)
        profile = select_id(args.robot, teleop=True)
        if profile is not None:
            check_namespace_allowed(profile, args.namespace)
    except (ProfileError, ValueError) as exc:
        print(f"teleop: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    return TeleopApp(args.url, profile, args.namespace).run()


if __name__ == "__main__":
    sys.exit(main())
