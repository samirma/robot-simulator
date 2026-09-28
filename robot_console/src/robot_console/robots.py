"""Which robot the console is driving, and everything that differs between them.

Teleop drives every mobile robot in `robots_specs/robots.yml` -- every kind but a fixed
arm. Two contracts with almost nothing in common: the `/cmd_vel` bases (the myAGV, the
myAGV + myCobot 280 on the same base, the ROSMASTER X3 PLUS) are a velocity stream with
`/odom` back, the AiNex a walking state machine (`/walking/set_param` + the
`/walking/command` service, nothing back but the camera). The console keeps one loop, one
keymap and one `Command` intent type; a `RobotProfile` carries the parts that genuinely
differ -- the link that encodes `Command` for the wire, the speed envelope and the HUD
wording.

The robot ids are the console's copy of the ids in `robots_specs/robots.yml` for the
robots teleop drives. The workspace parity tests (`tests/test_contract_parity.py` at the
workspace root) hold the two equal by reading that file.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence, Tuple

from robot_console import hud, teleop

#: `robots_specs/robots.yml` ids of the robots teleop drives (console spec §2.1).
MYAGV = "myagv"
AINEX = "ainex"
MYAGV_MYCOBOT280 = "myagv_mycobot280"
ROSMASTER_X3_PLUS = "rosmaster_x3_plus"
TELEOP_ROBOTS: Tuple[str, ...] = (MYAGV, AINEX, MYAGV_MYCOBOT280, ROSMASTER_X3_PLUS)

#: The `/cmd_vel` bases, which report `/odom` and a `/scan`: what `smoke` drives.
WHEELED_ROBOTS: Tuple[str, ...] = (MYAGV, MYAGV_MYCOBOT280, ROSMASTER_X3_PLUS)

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
    MYAGV_MYCOBOT280: "publish an empty actionlib_msgs/GoalID on /move_base/cancel, then a "
                      "zero geometry_msgs/Twist on /cmd_vel",
    ROSMASTER_X3_PLUS: "publish a zero geometry_msgs/Twist on /cmd_vel",
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


def _make_composite_link(host: str, port: int, namespace: str = "", camera_topic=None):
    from robot_console.bridge import RobotLink
    from robot_console.composite_topics import CANCEL_ALL, TOPIC_CANCEL, TYPE_GOAL_ID
    from robot_console.topics import (
        TOPIC_CAMERA, TOPIC_CMD_VEL, TOPIC_ODOM, TOPIC_SCAN, namespaced,
    )

    return RobotLink(
        host, port,
        cmd_topic=namespaced(TOPIC_CMD_VEL, namespace),
        odom_topic=namespaced(TOPIC_ODOM, namespace),
        camera_topic=camera_topic or namespaced(TOPIC_CAMERA, namespace),
        scan_topic=namespaced(TOPIC_SCAN, namespace),
        cancel_topic=namespaced(TOPIC_CANCEL, namespace),
        cancel_type=TYPE_GOAL_ID,
        cancel_msg=CANCEL_ALL,
    )


def _make_x3_link(host: str, port: int, namespace: str = "", camera_topic=None):
    from robot_console import x3plus_topics as x3
    from robot_console.bridge import RobotLink
    from robot_console.topics import namespaced

    return RobotLink(
        host, port,
        cmd_topic=namespaced(x3.TOPIC_CMD_VEL, namespace),
        odom_topic=namespaced(x3.TOPIC_ODOM, namespace),
        # Its only colour stream is raw: discovery's fallback to "the namespace's one
        # CompressedImage" never applies, so the contract name is used as it is.
        camera_topic=namespaced(x3.TOPIC_CAMERA, namespace),
        scan_topic=namespaced(x3.TOPIC_SCAN, namespace),
        camera_type=x3.TYPE_IMAGE,
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
        speed_limit_label="the AiNex gait envelope",
        stop_command=STOP_COMMANDS[AINEX],
    )


def _composite_profile() -> RobotProfile:
    """The myAGV + myCobot 280 drives as the myAGV it stands on, with its own stop."""
    return dataclasses.replace(
        _myagv_profile(), name=MYAGV_MYCOBOT280, make_link=_make_composite_link,
        speed_limit_label="the real myAGV limit", stop_command=STOP_COMMANDS[MYAGV_MYCOBOT280])


def _x3_profile() -> RobotProfile:
    from robot_console import x3plus_topics as x3

    return RobotProfile(
        name=ROSMASTER_X3_PLUS,
        make_link=_make_x3_link,
        speed_min=x3.SPEED_MIN,
        speed_max=x3.SPEED_MAX,
        speed_step=x3.SPEED_STEP,
        speed_default=x3.SPEED_DEFAULT,
        turn_ratio=x3.TURN_RATIO,
        turn_max=x3.TURN_MAX,
        has_odom=True,
        has_head=False,
        hints=hud.HINTS,
        speed_limit_label="the X3 PLUS board's 0.7 m/s input range",
        stop_command=STOP_COMMANDS[ROSMASTER_X3_PLUS],
    )


_FACTORIES = {MYAGV: _myagv_profile, AINEX: _ainex_profile,
              MYAGV_MYCOBOT280: _composite_profile, ROSMASTER_X3_PLUS: _x3_profile}


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
