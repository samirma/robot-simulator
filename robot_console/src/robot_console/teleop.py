"""Keymap, speed model, and latched command state.

Pure: stdlib only, no OpenCV, no network. Everything here is directly unit-testable,
which is why the interesting behaviour of the console lives in this module rather than
in the render loop.

Hold-to-drive, without a key-up event: OpenCV's `waitKey` reports key-down only, and a
real key-up would mean the global keyboard hook the spec rules out. What it does give
is OS key auto-repeat -- holding `W` delivers `w` over and over. So motion is armed by a
key and expires `hold_timeout` seconds after the last one, which makes releasing the key
stop the robot.

The timeout has to clear the OS's *initial* repeat delay, or a held key would stutter:
move, expire, then resume once repeat kicks in. macOS defaults to 375 ms before the
first repeat and 90 ms between them, so 0.6 s leaves margin without making the robot
coast noticeably. The vendor's own teleop makes the same trade at 0.52 s.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Optional

# The real myAGV tops out around 0.28 m/s. The simulator applies no velocity limit of
# its own -- it only saturates through a 0.12 m setpoint-lead clamp -- so an uncapped
# console would let you command speeds that behave one way in sim and another on
# hardware. Capping at the hardware limit makes the sim an honest rehearsal.
SPEED_MIN = 0.05
SPEED_MAX = 0.28
SPEED_STEP = 0.05
SPEED_DEFAULT = 0.15

# One knob scales the whole motion envelope: two independent speeds on a six-key layout
# with no modifiers is a UI you would have to explain.
#
# The ratio and the cap come from the vendor's own teleop
# (myagv_ros/myagv_teleop/scripts/myagv_teleop.py), which defaults to speed 0.25 m/s
# with turn 0.5 rad/s -- a ratio of 2 -- and a turn_limit of 1.0 rad/s. Driving the
# same envelope means a drive rehearsed in the simulator behaves the same on hardware.
TURN_RATIO = 2.0
TURN_MAX = 1.00

KEY_ESC = 27
KEY_SPACE = 32

# Long enough to bridge the OS initial key-repeat delay (375 ms on a stock macOS), short
# enough that the robot does not coast far after a release: 0.6 s at the 0.15 m/s
# default is about 9 cm.
HOLD_TIMEOUT = 0.6


class Action(enum.Enum):
    NONE = "NONE"
    FORWARD = "FORWARD"
    BACK = "BACK"
    STRAFE_LEFT = "STRAFE_LEFT"
    STRAFE_RIGHT = "STRAFE_RIGHT"
    ROT_LEFT = "ROT_LEFT"
    ROT_RIGHT = "ROT_RIGHT"
    STOP = "STOP"
    FASTER = "FASTER"
    SLOWER = "SLOWER"
    HELP = "HELP"
    QUIT = "QUIT"


# Unit body-frame direction for each motion action: (forward, left, ccw).
# myAGV/ROS convention: +x forward, +y left, +z yaw counter-clockwise.
_AXES = {
    Action.FORWARD: (1.0, 0.0, 0.0),
    Action.BACK: (-1.0, 0.0, 0.0),
    Action.STRAFE_LEFT: (0.0, 1.0, 0.0),
    Action.STRAFE_RIGHT: (0.0, -1.0, 0.0),
    Action.ROT_LEFT: (0.0, 0.0, 1.0),
    Action.ROT_RIGHT: (0.0, 0.0, -1.0),
}

KEYMAP = {
    ord("w"): Action.FORWARD,
    ord("s"): Action.BACK,
    ord("a"): Action.STRAFE_LEFT,
    ord("d"): Action.STRAFE_RIGHT,
    ord("q"): Action.ROT_LEFT,
    ord("e"): Action.ROT_RIGHT,
    KEY_SPACE: Action.STOP,
    KEY_ESC: Action.QUIT,
    # '+' needs shift on most layouts, so accept the unshifted '=' too. Same for '_'.
    ord("+"): Action.FASTER,
    ord("="): Action.FASTER,
    ord("-"): Action.SLOWER,
    ord("_"): Action.SLOWER,
    ord("h"): Action.HELP,
    ord("?"): Action.HELP,
}

# Holding one of these is what keeps the robot moving; everything else is a one-shot.
MOTION_ACTIONS = frozenset(
    {
        Action.FORWARD,
        Action.BACK,
        Action.STRAFE_LEFT,
        Action.STRAFE_RIGHT,
        Action.ROT_LEFT,
        Action.ROT_RIGHT,
    }
)


def action_for_key(key: int) -> Action:
    """Map a `cv2.waitKey` return value to an `Action`.

    `waitKey` returns -1 on timeout and, on some platforms, sets high bits above the
    ASCII code, so the low byte is the only portable part.
    """
    if key is None or key < 0:
        return Action.NONE
    code = key & 0xFF
    if 65 <= code <= 90:  # normalise upper case; W and w mean the same thing
        code += 32
    return KEYMAP.get(code, Action.NONE)


def key_label(key: int) -> str | None:
    """A human-readable name for a key, for the command log. None if unmapped."""
    if key is None or key < 0:
        return None
    code = key & 0xFF
    if 65 <= code <= 90:
        code += 32
    if code == KEY_SPACE:
        return "space"
    if code == KEY_ESC:
        return "esc"
    if code == ord("?"):
        return "?"
    if code in KEYMAP:
        return chr(code)
    return None


def clamp_speed(value: float) -> float:
    """Clamp a linear speed into the supported range."""
    return min(SPEED_MAX, max(SPEED_MIN, float(value)))


@dataclasses.dataclass(frozen=True)
class Command:
    """A body-frame velocity command. vx forward, vy left, wz counter-clockwise."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def is_zero(self, eps: float = 1e-9) -> bool:
        return abs(self.vx) < eps and abs(self.vy) < eps and abs(self.wz) < eps

    def to_twist(self) -> dict:
        """A complete `geometry_msgs/Twist`.

        The bridge reads only linear.x/linear.y/angular.z, but a real ROS subscriber
        deserialises the whole message, so every field is present.
        """
        return {
            "linear": {"x": float(self.vx), "y": float(self.vy), "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": float(self.wz)},
        }


@dataclasses.dataclass
class TeleopState:
    """The direction currently being held, plus the speed setting.

    `hold_timeout` is how long a motion survives without another key event. Set it to
    None to latch instead -- motion then persists until another direction, `Space`, or
    `Esc`, which is useful over a link too laggy to deliver key repeat reliably.
    """

    speed: float = SPEED_DEFAULT
    speed_max: float = SPEED_MAX
    hold_timeout: Optional[float] = HOLD_TIMEOUT
    # The rest of the speed envelope, per instance because it belongs to the robot rather
    # than to this module. The defaults are the myAGV's, so a caller that says nothing
    # behaves exactly as it did when there was only one robot.
    speed_min: float = SPEED_MIN
    speed_step: float = SPEED_STEP
    turn_ratio: float = TURN_RATIO
    turn_max: float = TURN_MAX
    axis: tuple = (0.0, 0.0, 0.0)
    last_action: Action = Action.NONE
    running: bool = True
    show_help: bool = True
    held_since: Optional[float] = None
    _armed_at: Optional[float] = None

    def apply(self, action: Action, now: Optional[float] = None) -> Action:
        """Fold a key action into the state. Returns the action, for logging."""
        self.last_action = action
        if action is Action.NONE:
            return action
        if action is Action.QUIT:
            self._disarm()
            self.running = False
        elif action is Action.STOP:
            self._disarm()
        elif action is Action.FASTER:
            self.speed = self._clamp(self.speed + self.speed_step)
        elif action is Action.SLOWER:
            self.speed = self._clamp(self.speed - self.speed_step)
        elif action is Action.HELP:
            self.show_help = not self.show_help
        elif action in MOTION_ACTIONS:
            stamp = time.monotonic() if now is None else now
            if self.axis != _AXES[action]:
                # A new direction replaces the old rather than combining, so W then D
                # strafes instead of driving diagonally.
                self.held_since = stamp
            # Every repeat of the held key re-arms the motion; when the key comes up the
            # repeats stop and `expire` takes it away.
            self.axis = _AXES[action]
            self._armed_at = stamp
        return action

    def expire(self, now: Optional[float] = None) -> bool:
        """Zero the motion if the key that armed it has stopped repeating.

        Called every tick. Returns True on the tick where the motion was dropped, so the
        caller can log a release.
        """
        if self.hold_timeout is None or self._armed_at is None:
            return False
        stamp = time.monotonic() if now is None else now
        if stamp - self._armed_at <= self.hold_timeout:
            return False
        self._disarm()
        return True

    def _disarm(self) -> None:
        self.axis = (0.0, 0.0, 0.0)
        self._armed_at = None
        self.held_since = None

    @property
    def is_moving(self) -> bool:
        return self.axis != (0.0, 0.0, 0.0)

    def _clamp(self, value: float) -> float:
        return min(self.speed_max, max(self.speed_min, float(value)))

    def command(self) -> Command:
        fx, fy, fw = self.axis
        return Command(
            vx=fx * self.speed,
            vy=fy * self.speed,
            wz=fw * min(self.speed * self.turn_ratio, self.turn_max),
        )

    @property
    def at_max_speed(self) -> bool:
        return self.speed >= self.speed_max - 1e-9

    @property
    def at_min_speed(self) -> bool:
        return self.speed <= self.speed_min + 1e-9
