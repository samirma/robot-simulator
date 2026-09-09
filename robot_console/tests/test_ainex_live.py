"""The console actually drives the AiNex: keys in, robot moving, against a live engine.

    cd simulator && ./kitchen.sh serve --robots ainex          # or --engine robocasa
    cd robot_console && .venv/bin/python -m pytest -m live tests/test_ainex_live.py

Nothing else in either project joins these two halves. `test_cli.py` checks the console's
gait arithmetic with no wire; the simulator's `robots/ainex/test_ros.py` drives the
surface over a raw websocket, and through `/app/set_walking_param` -- the *other* command
topic, the tiered preset one -- so the topic teleop actually publishes to had no
end-to-end check at all. That gap hid a real bug: `AiNexLink` composed the camera under
the discovered namespace and left `/walking/set_param` and `/walking/command` bare, so
against any engine (they namespace every robot after itself) the view streamed and every
key did nothing. rosbridge acks nothing, so there was no error to see.

Yaw is read back from `/imu`, which is the only pose this robot's vendor contract carries
-- it has no `/odom` and no `/tf`, by the vendor's design. That makes the turn the
measurable half here; that the gait is running at all is read from `/walking/is_walking`
and from the legs moving in `/joint_states`. How far it walks is the simulator's own
measurement to make, against a model this side cannot see.
"""

from __future__ import annotations

import math
import time

import pytest

from robot_console.ainex_link import AiNexLink
from robot_console.ainex_topics import (
    TOPIC_IMU,
    TOPIC_IS_WALKING,
    TOPIC_JOINT_STATES,
    TYPE_BOOL,
    TYPE_IMU,
    TYPE_JOINT_STATE,
)
from robot_console.bridge import quiet_roslibpy_logging
from robot_console.cli import DEFAULT_HOST, DEFAULT_PORT
from robot_console.ainex_link import HEAD_RATE
from robot_console.ainex_topics import HEAD_PAN_LIMIT, HEAD_TILT_LIMIT
from robot_console.teleop import Action, Command, HeadPose
from robot_console.topics import namespaced

pytestmark = pytest.mark.live

#: Long enough to be unambiguous against the gait's own 0.4 s cycle, short enough that
#: three of them plus a settle is a few seconds.
DRIVE_S = 3.0
#: What a Q or an E produces at the AiNex's default speed setting: 0.10 m/s * TURN_RATIO
#: 4.0, capped at TURN_MAX. See `robots.py` and `teleop.TeleopState.command`.
TURN_WZ = 0.40


@pytest.fixture(scope="module")
def robot():
    """A connected link to whatever AiNex is on the port, or a skip.

    Discovered, not assumed: the namespace is the engine's to choose, and asking is what
    teleop itself does. One module-scoped link, because roslibpy's reactor is
    process-global and single-shot -- see `test_link_roundtrip.py`.
    """
    quiet_roslibpy_logging()
    from robot_console.discovery import DiscoveryError, discover

    try:
        found = discover(f"ws://{DEFAULT_HOST}:{DEFAULT_PORT}", "ainex")
    except DiscoveryError as exc:
        pytest.skip(f"no AiNex on ws://{DEFAULT_HOST}:{DEFAULT_PORT}: {exc}")
    except Exception as exc:  # noqa: BLE001 - nothing listening is the usual case
        pytest.skip(f"no rosbridge on ws://{DEFAULT_HOST}:{DEFAULT_PORT}: {exc}")

    link = AiNexLink(
        DEFAULT_HOST, DEFAULT_PORT,
        camera_topic=found.camera_topic, namespace=found.namespace,
    )
    link.connect(timeout=10.0)
    yield link, found.namespace
    link.stop()
    # No terminate(): the reactor must survive for whatever runs after this module.
    link.close(hard_exit_after=None)


def _latest(link, topic, message_type, namespace, timeout=5.0):
    """Subscribe and return a getter for the most recent message, or skip on silence."""
    import roslibpy

    box = {"msg": None}
    handle = roslibpy.Topic(link._ros, namespaced(topic, namespace), message_type)
    handle.subscribe(lambda msg: box.update(msg=msg))
    deadline = time.monotonic() + timeout
    while box["msg"] is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if box["msg"] is None:
        pytest.skip(f"nothing published on {namespaced(topic, namespace)}")
    return lambda: box["msg"]


def _yaw(msg) -> float:
    q = msg["orientation"]
    return math.atan2(
        2.0 * (q["w"] * q["z"] + q["x"] * q["y"]),
        1.0 - 2.0 * (q["y"] ** 2 + q["z"] ** 2),
    )


def _drive(link, command: Command, seconds: float = DRIVE_S) -> None:
    """Hold a key for `seconds`, at the rate the teleop loop publishes at, then release.

    Through `publish_cmd_vel` rather than a raw publish, because the state machine in it
    -- parameter block on change, `start` on the transition, `stop` on release -- is
    exactly what is under test.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        link.publish_cmd_vel(command)
        time.sleep(0.05)
    link.publish_cmd_vel(Command())
    time.sleep(0.5)


def _turned(imu, link, command) -> float:
    """Degrees of yaw a held key produces, wrapped to (-180, 180]."""
    before = _yaw(imu())
    _drive(link, command)
    delta = _yaw(imu()) - before
    return math.degrees(math.atan2(math.sin(delta), math.cos(delta)))


def test_q_turns_left_and_e_turns_right(robot) -> None:
    """The keys the user presses, the direction the robot goes.

    Both directions, not one: a sign error is invisible in a single turn, and the console
    and the simulator each apply their own sign convention to `angle_move_amplitude`.
    """
    link, namespace = robot
    imu = _latest(link, TOPIC_IMU, TYPE_IMU, namespace)

    left = _turned(imu, link, Command(wz=+TURN_WZ))
    right = _turned(imu, link, Command(wz=-TURN_WZ))

    # The gait model round-trips exactly -- the console solves A = wz*T/4 and the surface
    # reads wz = 4A/T back out -- so this is `TURN_WZ * DRIVE_S`, about 69 degrees, and
    # the tolerance is for the release ramp rather than for any modelling slack.
    expected = math.degrees(TURN_WZ * DRIVE_S)
    assert left > 0.5 * expected, f"Q turned {left:+.1f} deg, expected about {expected:+.1f}"
    assert right < -0.5 * expected, f"E turned {right:+.1f} deg, expected about {-expected:+.1f}"
    assert abs(left + right) < 0.2 * expected, (
        f"the two turns are not mirror images: {left:+.1f} and {right:+.1f} deg"
    )


def test_w_walks_without_turning(robot) -> None:
    """Forward is forward: the gait runs and the heading does not drift.

    How *far* it walks is the simulator's own measurement (`robots/ainex/test_ros.py`
    against the model); from here the honest statements are that the state machine is
    running and that walking straight does not yaw.
    """
    link, namespace = robot
    imu = _latest(link, TOPIC_IMU, TYPE_IMU, namespace)
    walking = _latest(link, TOPIC_IS_WALKING, TYPE_BOOL, namespace)
    joints = _latest(link, TOPIC_JOINT_STATES, TYPE_JOINT_STATE, namespace)

    assert walking()["data"] is False, "the robot was already walking before any key"
    knee = joints()["name"].index("l_knee")
    still = joints()["position"][knee]

    before = _yaw(imu())
    moved = {"walking": False, "knee": False}
    deadline = time.monotonic() + DRIVE_S
    while time.monotonic() < deadline:
        link.publish_cmd_vel(Command(vx=0.10))
        moved["walking"] = moved["walking"] or bool(walking()["data"])
        moved["knee"] = moved["knee"] or abs(joints()["position"][knee] - still) > 0.02
        time.sleep(0.05)
    link.publish_cmd_vel(Command())
    time.sleep(0.5)

    assert moved["walking"], "/walking/is_walking never went true -- `start` never landed"
    assert moved["knee"], "the legs never moved -- the parameter block never landed"
    drift = math.degrees(abs(math.atan2(math.sin(_yaw(imu()) - before),
                                        math.cos(_yaw(imu()) - before))))
    assert drift < 5.0, f"walking forward yawed {drift:.1f} deg"
    assert walking()["data"] is False, "releasing the key did not stop the gait"


def test_the_arrow_keys_point_the_head(robot) -> None:
    """Keys in, head joints moved -- the other half of what `test_teleop.py` cannot see.

    `HeadPose` is pure and tested offline, and `publish_head` is one publish per
    controller, so the only thing that can go wrong between them is a name: the head
    topics take the namespace like every other, and a head published bare against an
    engine that namespaces every robot after itself moves nothing and errors on nothing.
    That is the exact bug this file was written for, one contract further along.
    """
    link, namespace = robot
    states = _latest(link, TOPIC_JOINT_STATES, TYPE_JOINT_STATE, namespace)

    def head_angles() -> tuple[float, float]:
        msg = states()
        by_name = dict(zip(msg["name"], msg["position"]))
        return by_name["head_pan"], by_name["head_tilt"]

    head = HeadPose(
        pan_limit=HEAD_PAN_LIMIT, tilt_limit=HEAD_TILT_LIMIT, rate=HEAD_RATE,
    )
    # Hold the left arrow, then the up arrow, at the OS's repeat interval.
    for action, _ in ((Action.HEAD_LEFT, None),) * 12:
        head.apply(action, 0.09)
    link.publish_head(head.pan, head.tilt)
    time.sleep(1.0)
    pan, tilt = head_angles()
    assert pan == pytest.approx(head.pan, abs=0.05), f"pan {pan:.3f} vs {head.pan:.3f}"
    assert abs(tilt) < 0.05, f"tilt moved to {tilt:.3f} on a pan-only command"

    for _ in range(12):
        head.apply(Action.HEAD_UP, 0.09)
    link.publish_head(head.pan, head.tilt)
    time.sleep(1.0)
    pan_after, tilt_after = head_angles()
    assert tilt_after == pytest.approx(head.tilt, abs=0.05)
    assert pan_after == pytest.approx(pan, abs=0.05), "panning changed when tilting"

    # ...and `0` brings it back, which is the only way out of a limit from the keyboard.
    head.apply(Action.HEAD_CENTRE, 0.0)
    link.publish_head(head.pan, head.tilt)
    time.sleep(1.5)
    pan_home, tilt_home = head_angles()
    assert abs(pan_home) < 0.05 and abs(tilt_home) < 0.05, (pan_home, tilt_home)
