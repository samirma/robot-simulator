"""Which robot is on this rosbridge, and under what name?

The console's constants are the *bare* vendor contract -- `/cmd_vel`, `/odom`,
`/camera/image_raw/compressed` -- because that is what one real robot's stack presents.
The simulator, meanwhile, gives every robot a namespace and defaults it to the robot's own
name, so a lone myAGV is on `/myagv/*`. Both defaults are right and they do not meet: with
neither `--namespace` nor `--robot` given the console published into a void and subscribed
to topics nobody fed, and **nothing errored** -- roslibpy subscribes happily to a name that
does not exist and the bridge never acks. The symptom was a black camera window and a
robot that ignored every key.

So the console asks. This module is the pure half of that question: it takes
`{topic: type}` as `/rosapi/topics` reports it and answers which robots are there. The
transport is `fleet.list_topics`, so there is one rosapi implementation in the console and
not two.

A robot is identified by a signature **command** topic, not by its camera or its odometry:
a robot whose first frame has not been encoded yet is still identifiable, and rosapi keeps
declared subscriptions in its answer precisely so a client can discover how to *drive*
something. The console's camera and control page (`live_cameras.html`, served by
`bin/view.sh`) identifies members the same way, from a copy of `MEMBER_SIGNATURES` in its
`CONTRACT` block. The two are duplicated rather than shared because the page is one
static file with no Python behind it; `tests/test_view_page.py` holds them equal.

`find_members` is the wider question the fleet check and the page ask: every member on
the wire, typed, including the ones teleop never drives (an SO-101, the worktop rig).
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Sequence

from robot_console import ainex_topics
from robot_console.topics import TOPIC_CAMERA, TOPIC_CMD_VEL, namespaced

#: `(robot, signature topic)`, most specific first. Only robots this console can *drive*
#: are here: an SO-101 has a signature of its own on the wire and no place in teleop, and
#: the worktop's camera rig under `scene` is not a robot at all.
SIGNATURES: tuple[tuple[str, str], ...] = (
    ("ainex", ainex_topics.TOPIC_SET_WALKING_PARAM),
    ("myagv", TOPIC_CMD_VEL),
)

#: Both dialects of the same message type. Two robots on one graph can speak two: the
#: myAGV and the AiNex are ROS 1 stacks and the SO-101 is a ROS 2 bringup, and rosapi
#: reports each one's strings verbatim. Matching one of these found half the cameras on
#: the wire, which reads as a simulator that failed to render.
CAMERA_TYPES: frozenset[str] = frozenset(
    {"sensor_msgs/CompressedImage", "sensor_msgs/msg/CompressedImage"}
)


#: `(member kind, signature topic, its type)`, most specific first: every kind of fleet
#: member this console knows a contract for. Typed, because a name alone is a claim
#: anybody can make -- a `/cmd_vel` that is a `std_msgs/String` is not a myAGV, and a
#: fleet check that counted it as one would then report every real topic of the robot as
#: missing instead of saying the one thing that is wrong. Each type is its robot's own
#: dialect, exactly as its ROS file spells it; the two dialects are never folded together.
MEMBER_SIGNATURES: tuple[tuple[str, str, str], ...] = (
    ("so101", "/joint_trajectory_controller/joint_trajectory",
     "trajectory_msgs/msg/JointTrajectory"),
    ("ainex", ainex_topics.TOPIC_SET_WALKING_PARAM, ainex_topics.TYPE_WALKING_PARAM),
    ("myagv", TOPIC_CMD_VEL, "geometry_msgs/Twist"),
)

#: The worktop's fixed camera rig (simulator spec §3): a workspace-owned member under a
#: namespace of its own, identified by its overhead view. Not a robot and not drivable.
RIG_KIND = "scene"
RIG_NAMESPACE = "scene"
RIG_SIGNATURE = ("/scene/overhead/color/compressed", "sensor_msgs/msg/CompressedImage")


class DiscoveryError(RuntimeError):
    """No single robot could be picked. The message is what the user is shown."""


@dataclasses.dataclass(frozen=True)
class Discovered:
    """One drivable robot on the wire."""

    robot: str
    namespace: str
    camera_topic: str

    def describe(self) -> str:
        where = f"/{self.namespace}/*" if self.namespace else "the bare contract (no namespace)"
        return f"{self.robot} on {where}, camera {self.camera_topic}"


@dataclasses.dataclass(frozen=True)
class Member:
    """One fleet member on the wire: its kind (a `robots.yml` id, or `scene`) and namespace."""

    kind: str
    namespace: str

    def describe(self) -> str:
        where = f"/{self.namespace}/*" if self.namespace else "the bare contract"
        return f"{self.kind} on {where}"


def namespace_of(topic: str, signature: str) -> Optional[str]:
    """The namespace that makes `topic` be `signature`, or None if it is not.

    Derived from the signature rather than by taking the first path segment, because a
    signature can have several segments of its own: a bare AiNex's `/walking/set_param` is
    the whole contract name at the root, and reading `walking` off the front of it would
    invent a namespace and then look for the robot's camera inside it.

    An empty namespace is not a failure to find one: it is the bare single-robot contract,
    what a real vendor bringup presents and what `--ros-namespace ''` reproduces exactly.
    """
    signature = "/" + signature.strip("/")
    if topic == signature:
        return ""
    if topic.endswith(signature):
        namespace = topic[: -len(signature)].strip("/")
        # Round-trip through the composer, so this agrees with the rule that put the
        # prefix on rather than merely with the shape of the string.
        if namespace and namespaced(signature, namespace) == topic:
            return namespace
    return None


def in_namespace(topic: str, namespace: str) -> bool:
    """Is `topic` one of the names the robot under `namespace` presents?

    Everything is in the empty namespace: a bare wire is a single robot's, which is what
    makes it bare.
    """
    return not namespace or topic.startswith(f"/{namespace.strip('/')}/")


def _camera_for(present: Mapping[str, str], namespace: str) -> str:
    """The camera topic to subscribe to for the robot under `namespace`.

    The contract name first, so a robot presenting what it should is never second-guessed.
    Otherwise the one CompressedImage topic in that namespace, which is how a stream named
    by its driver rather than by the contract is still found -- `/color/compressed` behind
    a RealSense-style node against `/image_raw/compressed` behind `usb_cam`. Ambiguity
    falls back to the contract name rather than guessing between two streams.
    """
    contract = namespaced(TOPIC_CAMERA, namespace)
    if contract in present:
        return contract
    cameras = [
        topic
        for topic, kind in present.items()
        if kind in CAMERA_TYPES and in_namespace(topic, namespace)
    ]
    return cameras[0] if len(cameras) == 1 else contract


def find_robots(present: Mapping[str, str]) -> list[Discovered]:
    """Every drivable robot the wire is offering, in the order their namespaces sort.

    `present` is `{topic: type}` from `/rosapi/topics`. A namespace matches at most one
    signature -- the first in `SIGNATURES` -- so a robot presenting both an AiNex's walking
    topics and a `/cmd_vel` is read as the AiNex it is.
    """
    hits: dict[str, set[str]] = {}
    for topic in present:
        for robot, signature in SIGNATURES:
            namespace = namespace_of(topic, signature)
            if namespace is not None:
                hits.setdefault(namespace, set()).add(robot)
                break
    found = []
    for namespace in sorted(hits):
        # `SIGNATURES` order decides, not the order rosapi happened to list the topics in.
        robot = next(r for r, _ in SIGNATURES if r in hits[namespace])
        found.append(Discovered(robot, namespace, _camera_for(present, namespace)))
    return found


def find_members(present: Mapping[str, str]) -> tuple[list[Member], list[str]]:
    """Every fleet member on the wire, and what looked like one but had the wrong type.

    `present` is `{topic: type}` from `/rosapi/topics`. A member is a namespace composing
    a `MEMBER_SIGNATURES` topic **with that signature's type**; a name match with any
    other type is returned in the second list as `"<topic> is <type>, not <expected>"`,
    so a caller can fail on it rather than silently ignore it. A namespace holds at most
    one member, the first kind in `MEMBER_SIGNATURES` that it composes. The rig is a
    member when its signature is on the wire with its type. Sorted by namespace.
    """
    order = [kind for kind, _, _ in MEMBER_SIGNATURES]
    kinds: dict[str, str] = {}
    wrong: list[str] = []
    for topic, topic_type in present.items():
        for kind, signature, expected in MEMBER_SIGNATURES:
            namespace = namespace_of(topic, signature)
            if namespace is None:
                continue
            if topic_type != expected:
                wrong.append(f"{topic} is {topic_type or 'untyped'}, not {expected}")
                continue
            previous = kinds.get(namespace)
            if previous is None or order.index(kind) < order.index(previous):
                kinds[namespace] = kind
            break
    members = [Member(kind, namespace) for namespace, kind in kinds.items()]
    rig_topic, rig_type = RIG_SIGNATURE
    if rig_topic in present:
        if present[rig_topic] == rig_type:
            members.append(Member(RIG_KIND, RIG_NAMESPACE))
        else:
            wrong.append(f"{rig_topic} is {present[rig_topic] or 'untyped'}, not {rig_type}")
    members.sort(key=lambda m: (m.namespace, m.kind))
    return members, sorted(wrong)


def choose(
    found: Sequence[Discovered],
    want: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Discovered:
    """The one robot to drive, or a `DiscoveryError` saying why there isn't one.

    `want` and `namespace` are `--robot` and `--namespace` when the user gave them. They
    narrow the search rather than confirm a guess: with an arm and a base on one port,
    `--robot myagv` is how the base is asked for, and two of the same kind is a question
    only the user can answer. Either one given still leaves the other worth asking about,
    which is why discovery runs at all when only one of the two flags is set.
    """
    candidates = [
        d
        for d in found
        if (want is None or d.robot == want) and (namespace is None or d.namespace == namespace)
    ]
    if not candidates:
        offered = ", ".join(sorted(d.describe() for d in found)) or "none"
        wanted = f"no {want} " if want else "no robot this console can drive "
        where = f" under namespace {namespace!r}" if namespace is not None else ""
        raise DiscoveryError(
            f"{wanted}is on the wire{where} (drivable robots there: {offered}). "
            "Name the robot and its namespace explicitly with --robot and --namespace, "
            "or check what the simulator was started with."
        )
    if len(candidates) > 1:
        names = ", ".join(f"--namespace {d.namespace or ''!r}" for d in candidates)
        raise DiscoveryError(
            f"{len(candidates)} {candidates[0].robot} robots are on the wire; "
            f"say which with one of: {names}"
        )
    return candidates[0]


def discover(
    url: str,
    want: Optional[str] = None,
    namespace: Optional[str] = None,
    timeout: float = 2.0,
) -> Discovered:
    """Ask the rosbridge at `url` what it has, and pick the robot to drive.

    Raises `DiscoveryError` for a wire that answers and holds nothing usable, and lets a
    transport failure through as itself -- a rosbridge that cannot be reached or has no
    rosapi node is a different problem with a different answer, and the caller says so.

    **Opens its own connection and closes it without terminating.** roslibpy drives a
    process-global Twisted reactor that cannot be restarted once terminated, and the link
    this console goes on to drive the robot with connects afterwards, in this same process.
    `fleet.list_topics` closes and does not terminate for exactly that reason; keep it so.
    """
    from robot_console.fleet import list_topics

    return choose(find_robots(list_topics(url, timeout)), want, namespace)
