"""Teleoperation state machine, independent of any window or keyboard library.

Keys arrive as held/released events (``key_down``/``key_up``); focus loss, input loss,
connection loss and exit arrive as their own events. The core turns held motion keys into
the profile's command (a Twist for a wheeled base, gait amplitudes for the AiNex), applies
the documented stop when the last motion key is released, on Space, on focus or input loss
and on exit, and disables commands after an explicitly failed stop until the user presses
Enter and presses a motion key afresh. No motion or head command is sent while commands are
disabled (stop requests still are); the window enables them once the start-up stop was
delivered, so Enter is only for re-enabling.
A head axis starts from the head's measured position (the profile's documented read),
read when its first arrow key is pressed; while that position is unknown the key is ignored.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from robot_console.profiles import Op, Profile
from robot_console.rosbridge import ServiceError, TransportError

MOTION_KEYS = ("w", "s", "a", "d", "q", "e")
HEAD_KEYS = ("left", "right", "up", "down")
HEAD_AXIS = {"left": "pan", "right": "pan", "up": "tilt", "down": "tilt"}
STOP_CALL_TIMEOUT_S = 3.0
HEAD_PERIOD_S = 0.1          # console policy: head target re-sent at most at 10 Hz while held

#: Shown with every failed stop and with connection loss (console spec §2.1).
REENABLE_LIMITATION = ("Re-enabling does not check or confirm that the robot stopped; "
                       "check the robot before continuing.")


def no_delivery_statement(profile: Profile) -> str:
    """What happens when no stop can be delivered: the profile's documented watchdog finding,
    and the explicit statement that stopping is then not guaranteed."""
    wd = " ".join((profile.watchdog or "").split())
    return ("No stop was delivered and stopping is NOT guaranteed: use the robot's physical "
            "stop." + (f" Documented command watchdog: {wd}" if wd else
                       " No command watchdog is documented for this robot."))


class Sender:
    """What the core needs from the transport (a Rosbridge adapter in production)."""

    def __init__(self, rb, call_timeout: float = STOP_CALL_TIMEOUT_S) -> None:
        self.rb = rb
        self.call_timeout = call_timeout

    def publish(self, topic: str, type: str, msg: Any) -> None:
        self.rb.publish(topic, msg, type)

    def call(self, service: str, type: str, args: Any) -> dict:
        return self.rb.call_service(service, args, timeout=self.call_timeout)


@dataclasses.dataclass
class Outcome:
    ok: bool
    text: str


class TeleopCore:
    def __init__(self, target, sender: Sender, clock: Callable[[], float] = time.monotonic,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.target = target
        self.p: Profile = target.profile
        self.sender = sender
        self.clock = clock
        self._log_cb = log
        self.enabled = False
        self.connected = True
        self.held: List[str] = []            # motion keys, in press order
        self.held_head: List[str] = []
        self.walking = False                 # AiNex gait started by this session
        self.last_cmd: Optional[dict] = None
        self.last_sent_at = 0.0
        self.head_pos: Dict[str, float] = {}
        self.head_last_at: Optional[float] = None
        self.head_sent: Dict[str, float] = {}
        self.head_send_at: Dict[str, float] = {}
        self.messages: List[str] = []
        self.stop_outcomes: List[Outcome] = []

    # ------------------------------------------------------------ helpers
    def log(self, text: str) -> None:
        self.messages.append(text)
        del self.messages[:-12]
        if self._log_cb:
            self._log_cb(text)

    def w(self, name: str) -> str:
        return self.target.wire(name)

    def _run_ops(self, ops: Sequence[Op], param: Optional[dict] = None) -> Outcome:
        """Execute documented ops; a publish is 'sent' (never acknowledged), a call must answer."""
        texts = []
        for o in ops:
            if o.op == "publish":
                msg = param if o.msg == "$param" else o.msg
                self.sender.publish(self.w(o.name), o.type, msg)
                texts.append(f"published {o.name} (no acknowledgement)")
            elif o.op == "call":
                values = self.sender.call(self.w(o.name), o.type, o.msg)
                if isinstance(values, dict) and (values.get("result") is False or values.get("success") is False):
                    raise ServiceError(f"{o.name} answered {values}")
                texts.append(f"{o.name} answered")
        return Outcome(True, "; ".join(texts))

    # ------------------------------------------------------------ motion intent
    def axes(self) -> Dict[str, float]:
        """Per-axis command from the held keys: opposite keys cancel, each axis is clamped
        to its documented limit."""
        ax = self.p.teleop_base.axes if self.p.teleop_base else self.p.teleop_walk.axes
        h = set(self.held)

        def one(pos: str, neg: str, axis: str) -> float:
            a = ax.get(axis)
            if a is None:
                return 0.0
            v = a.speed * ((pos in h) - (neg in h))
            return max(-a.limit, min(a.limit, v))
        return {"x": one("w", "s", "x"), "y": one("a", "d", "y"), "yaw": one("q", "e", "yaw")}

    def command(self) -> dict:
        a = self.axes()
        if self.p.teleop_base:
            return {"linear": {"x": a["x"], "y": a["y"], "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": a["yaw"]}}
        wk = self.p.teleop_walk
        msg = dict(wk.param_template)
        for axis, spec in wk.axes.items():
            msg[spec.field] = a[axis]
        return msg

    def _send_motion(self) -> None:
        if not self.held:
            return
        cmd = self.command()
        try:
            if self.p.teleop_base:
                tb = self.p.teleop_base
                self.sender.publish(self.w(tb.topic), tb.type, cmd)
            else:
                wk = self.p.teleop_walk
                if not self.walking:
                    self._run_ops(wk.start, param=cmd)
                    self.walking = True
                else:
                    self.sender.publish(self.w(wk.param_topic), wk.param_type, cmd)
        except (TransportError, ServiceError, TimeoutError) as exc:
            self.log(f"motion command failed: {exc}")
            self.request_stop("motion command failed")
            return
        self.last_cmd = cmd
        self.last_sent_at = self.clock()

    # ------------------------------------------------------------ events
    def disabled_reason(self) -> str:
        """Why keys are ignored: before the start-up stop no key is needed (commands enable on
        their own once it is delivered); after a failed stop Enter re-enables them."""
        if not self.connected:
            return "commands are disabled: not connected"
        if not self.stop_outcomes:
            return "commands are not enabled yet: waiting for the start-up stop"
        if not self.stop_outcomes[-1].ok:
            return "commands are disabled after a failed stop: press Enter to re-enable, then press the keys again"
        return "commands are disabled: press Enter to enable"

    def key_down(self, key: str) -> None:
        if key == "space":
            self.clear_and_stop("Space")
        elif key == "enter":
            self.enable()
        elif key in MOTION_KEYS:
            if not self.enabled:
                self.log(self.disabled_reason())
                return
            if key in self.held:          # a repeat is not a fresh press
                return
            self.held.append(key)
            self._send_motion()
        elif key in HEAD_KEYS and self.p.head:
            if not self.enabled:
                self.log(self.disabled_reason())
                return
            if key in self.held_head:
                return
            axis = HEAD_AXIS[key]
            if not any(HEAD_AXIS[k] == axis for k in self.held_head) and not self._read_head(axis):
                return
            self.held_head.append(key)
            self.head_last_at = self.clock()

    def key_up(self, key: str) -> None:
        if key in self.held:
            self.held.remove(key)
            if self.held:
                self._send_motion()
            else:
                self.request_stop("last motion key released")
        elif key in self.held_head:
            self.held_head.remove(key)       # head target stops changing on release

    def clear_intent(self) -> None:
        self.held.clear()
        self.held_head.clear()

    def clear_and_stop(self, reason: str) -> Outcome:
        self.clear_intent()
        return self.request_stop(reason)

    def focus_lost(self) -> Outcome:
        return self.clear_and_stop("window lost keyboard focus")

    def input_lost(self, why: str = "") -> Outcome:
        return self.clear_and_stop("keyboard input lost" + (f" ({why})" if why else ""))

    def enable(self) -> None:
        if not self.connected:
            self.log("cannot enable: not connected")
            return
        if self.enabled:
            return
        self.clear_intent()               # only fresh presses after Enter count
        if self.p.teleop_walk and self.p.teleop_walk.enable:
            try:
                self._run_ops(self.p.teleop_walk.enable)
            except (TransportError, ServiceError, TimeoutError) as exc:
                self.log(f"enable failed: {exc}")
                return
        self.enabled = True
        self.log("commands ENABLED: hold motion keys to move")

    # ------------------------------------------------------------ stop
    def stop_ops(self) -> Sequence[Op]:
        return self.p.teleop_walk.stop if self.p.teleop_walk else self.p.stop

    def request_stop(self, reason: str) -> Outcome:
        """Attempt the documented stop. An explicit failure disables commands."""
        self.clear_intent()
        self.last_cmd = None
        try:
            out = self._run_ops(self.stop_ops())
            self.walking = False
            outcome = Outcome(True, f"stop requested ({reason}): {out.text}")
        except (TransportError, ServiceError, TimeoutError) as exc:
            self.enabled = False
            outcome = Outcome(False, f"STOP FAILED ({reason}): {exc}. Commands disabled; press "
                                     f"Enter and press keys again to continue. {REENABLE_LIMITATION}")
        self.stop_outcomes.append(outcome)
        self.log(outcome.text)
        return outcome

    def startup_stop(self) -> Outcome:
        """Once the target is validated and before commands are enabled."""
        out = self.request_stop("start-up, after validation")
        self.enabled = False
        return out

    def shutdown(self, reason: str) -> Outcome:
        self.enabled = False
        if not self.connected:
            self.clear_intent()
            o = Outcome(False, f"{reason}: not connected, no stop could be sent. "
                               + no_delivery_statement(self.p))
            self.log(o.text)
            return o
        return self.clear_and_stop(reason)

    def connection_lost(self, reason: str) -> None:
        self.connected = False
        self.enabled = False
        self.clear_intent()
        self.log(f"CONNECTION LOST ({reason}). " + no_delivery_statement(self.p))

    # ------------------------------------------------------------ periodic
    def tick(self) -> None:
        now = self.clock()
        if not (self.enabled and self.connected):
            return
        if self.held:
            rate = self.p.teleop_base.rate_hz if self.p.teleop_base else self.p.teleop_walk.rate_hz
            if now - self.last_sent_at >= 1.0 / rate:
                self._send_motion()
        if self.held_head and self.p.head:
            dt = now - (self.head_last_at or now)
            self.head_last_at = now
            # the vendor model's head_pan joint axis is -Z (ainex:src/ainex_simulations/
            # ainex_description/urdf/ainex.urdf.xacro#L786-L787; console spec §2.1 as amended
            # 2026-10-02): a positive pan turns the head to the robot's right, so Left is negative
            dirs = {"pan": (("right" in self.held_head) - ("left" in self.held_head)),
                    "tilt": (("up" in self.held_head) - ("down" in self.held_head))}
            for axis, h in self.p.head.items():
                d = dirs.get(axis, 0)
                if not d:
                    continue
                self.head_pos[axis] = max(h.min, min(h.max, self.head_pos[axis] + d * h.rate * dt))
                last = self.head_sent.get(axis)
                if last is None or (abs(last - self.head_pos[axis]) > 1e-6
                                    and now - self.head_send_at.get(axis, 0.0) >= HEAD_PERIOD_S):
                    self._send_head(axis)
        elif self.head_last_at is not None:
            self.head_last_at = None

    def _read_head(self, axis: str) -> bool:
        """Start ``axis`` from the head's measured position, so a key moves the head its way
        wherever it was left. False (reason shown, key ignored) when the position is unknown:
        the template's 0 is never assumed. The position is the profile's documented read."""
        h = self.p.head[axis]
        found = self.p.head_read(axis)
        try:
            if found is None:
                raise ValueError("the profile documents no position read")
            read, value = found
            pos = read.parse(self.sender.call(self.w(read.service), read.type, read.request))[
                (value.control, value.field)]
            if pos is None:
                raise ValueError(f"{read.service} reported no position for it")
        except (TransportError, ServiceError, TimeoutError, ValueError) as exc:
            self.log(f"head position unknown ({axis}): {exc}; key ignored")
            return False
        self.head_pos[axis] = max(h.min, min(h.max, pos))
        return True

    def _send_head(self, axis: str) -> None:
        h = self.p.head[axis]
        msg = dict(h.template)
        msg[h.field] = round(self.head_pos[axis], 4)
        try:
            self.sender.publish(self.w(h.topic), h.type, msg)
        except TransportError as exc:
            self.log(f"head command failed: {exc}")
            return
        self.head_sent[axis] = self.head_pos[axis]
        self.head_send_at[axis] = self.clock()

    def command_topics(self) -> List[tuple]:
        """(wire topic, type) this session publishes, to advertise before the first stop."""
        out = []
        if self.p.teleop_base:
            out.append((self.w(self.p.teleop_base.topic), self.p.teleop_base.type))
        if self.p.teleop_walk:
            out.append((self.w(self.p.teleop_walk.param_topic), self.p.teleop_walk.param_type))
        for o in self.stop_ops():
            if o.op == "publish":
                out.append((self.w(o.name), o.type))
        for h in (self.p.head or {}).values():
            out.append((self.w(h.topic), h.type))
        return list(dict.fromkeys(out))
