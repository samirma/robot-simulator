"""Which robot the console is driving, and everything that differs between them.

Two robots, two contracts with almost nothing in common: the myAGV is a velocity stream
(`/cmd_vel`, `/odom` back), the AiNex a walking state machine (`/walking/set_param` + the
`/walking/command` service, nothing back but the camera). The console keeps one loop, one
keymap and one `Command` intent type; a `RobotProfile` carries the parts that genuinely
differ -- the link that encodes `Command` for the wire, the speed envelope, the HUD
wording and the what-to-start text.

The robot ids are the console's copy of the ids in `robots_specs/robots.yml` for the
robots teleop drives. `tests/test_robot_ids.py` holds the two equal by reading that file.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence, Tuple

from robot_console import hud, preflight, teleop

#: `robots_specs/robots.yml` ids of the robots teleop drives (console spec §2.1).
MYAGV = "myagv"
AINEX = "ainex"
TELEOP_ROBOTS: Tuple[str, ...] = (MYAGV, AINEX)

#: The only robot `slam.sh` maps with: it needs `/scan` and `/odom`.
SLAM_ROBOT = MYAGV

# The AiNex walks; it does not roll. Same keys, honest words -- plus the arrows, which
# only this robot has anything to point.
AINEX_HINTS: Sequence[Tuple[str, str]] = (
    ("W / S", "walk forward / back"),
    ("A / D", "sidestep left / right"),
    ("Q / E", "turn left / right"),
    ("Arrows", "look around (the camera is on the head)"),
    ("0", "centre the head"),
    ("Space", "stop"),
    ("+ / -", "speed"),
    ("H", "hide these hints"),
    ("Esc", "quit"),
)

#: Each robot's `stop_command`, as its ROS file states it. The links implement these
#: (`RobotLink.stop`, `AiNexLink.stop`); the supervisor sends them.
STOP_COMMANDS: Mapping[str, str] = {
    MYAGV: "publish a zero geometry_msgs/Twist on /cmd_vel",
    AINEX: "call /walking/command with 'enable_control', then with 'stop'",
}


@dataclasses.dataclass(frozen=True)
class RobotProfile:
    """Everything the console needs to know about one kind of robot."""

    name: str
    # (host, port, namespace, camera_topic) -> an unconnected link with the shape
    # connect/attach/subscribe_camera/publish_cmd_vel/stop/close. Only the safety
    # supervisor ever calls it: the UI never holds a link of its own.
    make_link: Callable[..., Any]
    speed_min: float
    speed_max: float
    speed_step: float
    speed_default: float
    turn_ratio: float
    turn_max: float
    # False -> nothing subscribes /odom. The AiNex publishes no odometry at all.
    has_odom: bool
    # False -> the arrow keys do nothing and no link needs `publish_head`.
    has_head: bool
    hints: Sequence[Tuple[str, str]]
    startup_instructions: Callable[[str, int], str]
    # For the --max-speed warning: what the cap is, in the robot's own terms.
    speed_limit_label: str
    stop_command: str


def _make_myagv_link(host: str, port: int, namespace: str = "", camera_topic=None):
    from robot_console.bridge import RobotLink
    from robot_console.topics import (
        TOPIC_CAMERA, TOPIC_CMD_VEL, TOPIC_ODOM, TOPIC_SCAN, namespaced,
    )

    return RobotLink(
        host, port,
        cmd_topic=namespaced(TOPIC_CMD_VEL, namespace),
        odom_topic=namespaced(TOPIC_ODOM, namespace),
        camera_topic=camera_topic or namespaced(TOPIC_CAMERA, namespace),
        scan_topic=namespaced(TOPIC_SCAN, namespace),
    )


def _make_ainex_link(host: str, port: int, namespace: str = "", camera_topic=None):
    from robot_console.ainex_link import AiNexLink
    from robot_console.ainex_topics import TOPIC_CAMERA
    from robot_console.topics import namespaced

    return AiNexLink(
        host, port,
        camera_topic=camera_topic or namespaced(TOPIC_CAMERA, namespace),
        namespace=namespace,
    )


def _myagv_profile() -> RobotProfile:
    return RobotProfile(
        name=MYAGV,
        make_link=_make_myagv_link,
        speed_min=teleop.SPEED_MIN,
        speed_max=teleop.SPEED_MAX,
        speed_step=teleop.SPEED_STEP,
        speed_default=teleop.SPEED_DEFAULT,
        turn_ratio=teleop.TURN_RATIO,
        turn_max=teleop.TURN_MAX,
        has_odom=True,
        has_head=False,
        hints=hud.HINTS,
        startup_instructions=preflight.startup_instructions,
        speed_limit_label="the real myAGV limit",
        stop_command=STOP_COMMANDS[MYAGV],
    )


def _ainex_profile() -> RobotProfile:
    from robot_console import ainex_link

    return RobotProfile(
        name=AINEX,
        make_link=_make_ainex_link,
        speed_min=ainex_link.SPEED_MIN,
        speed_max=ainex_link.SPEED_MAX,
        speed_step=ainex_link.SPEED_STEP,
        speed_default=ainex_link.SPEED_DEFAULT,
        turn_ratio=ainex_link.TURN_RATIO,
        turn_max=ainex_link.TURN_MAX,
        has_odom=False,
        has_head=True,
        hints=AINEX_HINTS,
        startup_instructions=preflight.startup_instructions_ainex,
        speed_limit_label="the AiNex gait envelope",
        stop_command=STOP_COMMANDS[AINEX],
    )


_FACTORIES = {MYAGV: _myagv_profile, AINEX: _ainex_profile}


def profile(name: str) -> RobotProfile:
    """The `RobotProfile` for `name`, built on demand."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise KeyError(f"unknown robot {name!r}; known: {', '.join(TELEOP_ROBOTS)}") from None
    return factory()


class _Profiles(Mapping):
    """`PROFILES[name]` -> profile, resolved lazily; iterates the teleop robot ids."""

    def __getitem__(self, name: str) -> RobotProfile:
        return profile(name)

    def __iter__(self):
        return iter(TELEOP_ROBOTS)

    def __len__(self) -> int:
        return len(TELEOP_ROBOTS)


PROFILES = _Profiles()
