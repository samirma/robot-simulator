"""Which robot the console is driving, and everything that differs between them.

Two robots, two contracts with almost nothing in common: the myAGV is a velocity stream
(`/cmd_vel` at 20 Hz, `/odom` back), the AiNex a walking state machine
(`/walking/set_param` + a `start`/`stop` service, nothing back but the camera). The
console keeps one loop, one keymap and one `Command` intent type; a `RobotProfile`
carries the parts that genuinely differ -- the link that encodes `Command` for the wire,
the speed envelope, the HUD wording and the what-to-start text.

Deliberately one small file, not a plugin system: the myAGV entries point at the same
objects the console used before there was a second robot, so `--robot` unspecified
behaves byte-for-byte as it always has.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Tuple

from robot_console import ainex_link, hud, preflight, teleop
from robot_console.ainex_link import AiNexLink
from robot_console.bridge import RobotLink

# The AiNex walks; it does not roll. Same keys, honest words.
AINEX_HINTS: Sequence[Tuple[str, str]] = (
    ("W / S", "walk forward / back"),
    ("A / D", "sidestep left / right"),
    ("Q / E", "turn left / right"),
    ("Space", "stop"),
    ("+ / -", "speed"),
    ("H", "hide these hints"),
    ("Esc", "quit"),
)


@dataclasses.dataclass(frozen=True)
class RobotProfile:
    """Everything the console needs to know about one kind of robot."""

    name: str
    # (options) -> a connected-shape link: connect/subscribe_camera/publish_cmd_vel/
    # stop/close/describe. Takes the whole Options so a profile can pick the topic
    # overrides that apply to it and ignore the rest.
    make_link: Callable[[Any], Any]
    speed_min: float
    speed_max: float
    speed_step: float
    speed_default: float
    turn_ratio: float
    turn_max: float
    # False -> the app never subscribes /odom. The AiNex publishes no odometry at all;
    # subscribing would not error, just never deliver, and the status line would claim
    # a stream that cannot exist.
    has_odom: bool
    hints: Sequence[Tuple[str, str]]
    startup_instructions: Callable[[str, int], str]
    # For the --max-speed warning: what the cap is, in the robot's own terms.
    speed_limit_label: str


def _make_myagv_link(options: Any) -> RobotLink:
    return RobotLink(
        options.host,
        options.port,
        cmd_topic=options.cmd_topic,
        odom_topic=options.odom_topic,
        camera_topic=options.camera_topic,
    )


def _make_ainex_link(options: Any) -> AiNexLink:
    # --cmd-topic and --odom-topic are myAGV knobs; the AiNex has no equivalent of
    # either, so only the camera override applies.
    return AiNexLink(options.host, options.port, camera_topic=options.camera_topic)


PROFILES = {
    "myagv": RobotProfile(
        name="myagv",
        make_link=_make_myagv_link,
        speed_min=teleop.SPEED_MIN,
        speed_max=teleop.SPEED_MAX,
        speed_step=teleop.SPEED_STEP,
        speed_default=teleop.SPEED_DEFAULT,
        turn_ratio=teleop.TURN_RATIO,
        turn_max=teleop.TURN_MAX,
        has_odom=True,
        hints=hud.HINTS,
        startup_instructions=preflight.startup_instructions,
        speed_limit_label="the real myAGV limit",
    ),
    "ainex": RobotProfile(
        name="ainex",
        make_link=_make_ainex_link,
        speed_min=ainex_link.SPEED_MIN,
        speed_max=ainex_link.SPEED_MAX,
        speed_step=ainex_link.SPEED_STEP,
        speed_default=ainex_link.SPEED_DEFAULT,
        turn_ratio=ainex_link.TURN_RATIO,
        turn_max=ainex_link.TURN_MAX,
        has_odom=False,
        hints=AINEX_HINTS,
        startup_instructions=preflight.startup_instructions_ainex,
        speed_limit_label="the AiNex gait envelope",
    ),
}

DEFAULT_ROBOT = "myagv"
