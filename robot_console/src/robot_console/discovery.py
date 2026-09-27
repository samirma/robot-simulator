"""Which robot is on this rosbridge, and under what name?

The console's constants are the *bare* vendor contract -- `/cmd_vel`, `/odom`,
`/camera/image_raw/compressed` -- because that is what one real robot's stack presents.
The simulator, meanwhile, gives every robot a namespace and defaults it to the robot's own
name, so a lone myAGV is on `/myagv/*`. Both defaults are right and they do not meet: with
neither `--namespace` nor `--robot` given the console published into a void and subscribed
to topics nobody fed, and **nothing errored** -- roslibpy subscribes happily to a name that
does not exist and the bridge never acks.

So the console asks `/rosapi/topics` (console spec §2.1: robot and namespace default to
discovered). This module is the pure half of that question: it takes `{topic: type}` as
rosapi reports it and answers which robots are there. After `--robot` and `--namespace`
narrow the candidates exactly one must remain; none, or more than one, is an error that
names what was found -- including candidates rejected for a missing or mistyped topic, so
"nothing drivable" never hides "something almost drivable".

A robot is identified by a signature **command** topic, confirmed by its distinguishing
companions, all with the contract's types. Command topics because a robot whose first
frame has not been encoded yet is still identifiable, and rosapi keeps declared
subscriptions in its answer precisely so a client can discover how to *drive* something.

There is one table of signatures, `MEMBER_SIGNATURES`, typed and covering every fleet
member the console knows a contract for. Two questions are asked of it:

* `survey` / `discover`: which robot does teleop drive? Only the kinds with
  `COMPANIONS` (a myAGV, an AiNex) are candidates, and each must also carry its
  distinguishing companions.
* `find_members`: what is on the wire at all? The fleet check and the camera page ask
  this; it counts every kind, including the ones teleop never drives (an SO-101, the
  worktop rig), and reports a signature of the wrong type rather than counting it.

The two differ in one respect, on purpose: `find_members` treats a signature rosapi gives
no type for as wrong (the fleet check exists to prove the typed interface is there),
while teleop takes an empty type as rosapi not saying and only rejects a stated type that
contradicts the contract.

The console's camera and control page (`live_cameras.html`, served by `bin/view.sh`)
identifies members the same way, from a copy of `MEMBER_SIGNATURES` in its `CONTRACT`
block. The two are duplicated rather than shared because the page is one static file with
no Python behind it; `tests/test_view_page.py` holds them equal.
"""

from __future__ import annotations

import dataclasses
from typing import List, Mapping, Optional, Sequence, Tuple

from robot_console import ainex_topics
from robot_console.robots import AINEX, MYAGV

SO101 = "so101"
from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TYPE_ODOM,
    TYPE_TWIST,
    namespaced,
)

#: `(member kind, signature topic, its type)`, most specific first: every kind of fleet
#: member this console knows a contract for. A robot is identified by its signature
#: **command** topic. Typed, because a name alone is a claim anybody can make -- a
#: `/cmd_vel` that is a `std_msgs/String` is not a myAGV. Each type is its robot's own
#: dialect, exactly as its ROS file spells it; the two dialects are never folded together.
MEMBER_SIGNATURES: Tuple[Tuple[str, str, str], ...] = (
    (SO101, "/joint_trajectory_controller/joint_trajectory",
     "trajectory_msgs/msg/JointTrajectory"),
    (AINEX, ainex_topics.TOPIC_SET_WALKING_PARAM, ainex_topics.TYPE_WALKING_PARAM),
    (MYAGV, TOPIC_CMD_VEL, TYPE_TWIST),
)

#: The worktop's fixed camera rig (simulator spec §3): a workspace-owned member under a
#: namespace of its own, identified by its overhead view. Not a robot and not drivable.
RIG_KIND = "scene"
RIG_NAMESPACE = "scene"
RIG_SIGNATURE = ("/scene/overhead/color/compressed", "sensor_msgs/msg/CompressedImage")

#: The kinds teleop drives, and the topics besides the signature that must be on the wire,
#: with which type, for a signature hit to be that robot: what distinguishes it from
#: anything else that happens to take the same command (a Twist base without odometry is
#: not a myAGV). An SO-101 has no place in teleop, so it has no entry.
COMPANIONS: Mapping[str, Mapping[str, str]] = {
    MYAGV: {TOPIC_ODOM: TYPE_ODOM},
    AINEX: {ainex_topics.TOPIC_IS_WALKING: ainex_topics.TYPE_BOOL},
}

#: `(robot, signature topic)` of the drivable kinds, in `MEMBER_SIGNATURES` order.
SIGNATURES: Tuple[Tuple[str, str], ...] = tuple(
    (kind, topic) for kind, topic, _ in MEMBER_SIGNATURES if kind in COMPANIONS
)

#: Everything that must be on the wire for a drivable kind: its signature and companions.
REQUIRED: Mapping[str, Mapping[str, str]] = {
    kind: {topic: topic_type, **COMPANIONS[kind]}
    for kind, topic, topic_type in MEMBER_SIGNATURES
    if kind in COMPANIONS
}

#: Both dialects of the same message type. Two robots on one graph can speak two: the
#: myAGV and the AiNex are ROS 1 stacks and the SO-101 is a ROS 2 bringup, and rosapi
#: reports each one's strings verbatim.
CAMERA_TYPES: frozenset = frozenset(
    {"sensor_msgs/CompressedImage", "sensor_msgs/msg/CompressedImage"}
)


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


@dataclasses.dataclass(frozen=True)
class Rejected:
    """A signature hit that is not a usable robot, and why."""

    robot: str
    namespace: str
    reason: str

    def describe(self) -> str:
        where = f"/{self.namespace}/*" if self.namespace else "the bare contract"
        return f"{self.robot}? on {where}: {self.reason}"


def namespace_of(topic: str, signature: str) -> Optional[str]:
    """The namespace that makes `topic` be `signature`, or None if it is not.

    Derived from the signature rather than by taking the first path segment, because a
    signature can have several segments of its own: a bare AiNex's `/walking/set_param`
    must not be read as a robot called `walking`.
    """
    signature = "/" + signature.strip("/")
    if topic == signature:
        return ""
    if topic.endswith(signature):
        namespace = topic[: -len(signature)].strip("/")
        if namespace and namespaced(signature, namespace) == topic:
            return namespace
    return None


def in_namespace(topic: str, namespace: str) -> bool:
    """Is `topic` one of the names the robot under `namespace` presents?"""
    return not namespace or topic.startswith(f"/{namespace.strip('/')}/")


def _camera_for(present: Mapping[str, str], namespace: str) -> str:
    """The contract camera name if present, else the namespace's one CompressedImage
    topic, else (ambiguous or none) the contract name."""
    contract = namespaced(TOPIC_CAMERA, namespace)
    if contract in present:
        return contract
    cameras = [
        topic
        for topic, kind in present.items()
        if kind in CAMERA_TYPES and in_namespace(topic, namespace)
    ]
    return cameras[0] if len(cameras) == 1 else contract


def _problems(present: Mapping[str, str], robot: str, namespace: str) -> List[str]:
    problems = []
    for topic, kind in REQUIRED[robot].items():
        name = namespaced(topic, namespace)
        if name not in present:
            problems.append(f"missing {name}")
        # An empty type is rosapi not saying, not a contradiction; only a stated type
        # that differs from the contract rules a candidate out.
        elif present[name] and present[name] != kind:
            problems.append(f"{name} is {present[name]}, not {kind}")
    return problems


def _signature_hits(present: Mapping[str, str]) -> dict:
    """`{namespace: {kind: (topic, stated type)}}` for every name match of a signature.

    The one scan both questions share; each then decides what a type mismatch means.
    """
    hits: dict = {}
    for topic, topic_type in present.items():
        for kind, signature, _ in MEMBER_SIGNATURES:
            namespace = namespace_of(topic, signature)
            if namespace is not None:
                hits.setdefault(namespace, {})[kind] = (topic, topic_type)
                break
    return hits


def survey(present: Mapping[str, str]) -> Tuple[List[Discovered], List[Rejected]]:
    """Every drivable robot the wire offers, and every near miss with its reason."""
    hits = _signature_hits(present)
    found: List[Discovered] = []
    rejected: List[Rejected] = []
    for namespace in sorted(hits):
        # `SIGNATURES` order decides, not the order rosapi listed the topics in.
        robot = next((r for r, _ in SIGNATURES if r in hits[namespace]), None)
        if robot is None:
            continue  # only members teleop does not drive (an SO-101) live here
        problems = _problems(present, robot, namespace)
        if problems:
            rejected.append(Rejected(robot, namespace, "; ".join(problems)))
        else:
            found.append(Discovered(robot, namespace, _camera_for(present, namespace)))
    return found, rejected


def find_robots(present: Mapping[str, str]) -> List[Discovered]:
    """The drivable robots only; see `survey` for the near misses."""
    return survey(present)[0]


def find_members(present: Mapping[str, str]) -> Tuple[List[Member], List[str]]:
    """Every fleet member on the wire, and what looked like one but had the wrong type.

    `present` is `{topic: type}` from `/rosapi/topics`. A member is a namespace composing
    a `MEMBER_SIGNATURES` topic **with that signature's type**; a name match with any
    other type (an empty one included) is returned in the second list as
    `"<topic> is <type>, not <expected>"`, so a caller can fail on it rather than silently
    ignore it. A namespace holds at most one member, the first kind in `MEMBER_SIGNATURES`
    that it composes. The rig is a member when its signature is on the wire with its type.
    Sorted by namespace.
    """
    expected = {kind: topic_type for kind, _, topic_type in MEMBER_SIGNATURES}
    members: List[Member] = []
    wrong: List[str] = []
    for namespace, kinds in _signature_hits(present).items():
        typed = []
        for kind, (topic, topic_type) in kinds.items():
            if topic_type == expected[kind]:
                typed.append(kind)
            else:
                wrong.append(f"{topic} is {topic_type or 'untyped'}, not {expected[kind]}")
        if typed:
            members.append(Member(next(k for k, _, _ in MEMBER_SIGNATURES if k in typed),
                                  namespace))
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
    rejected: Sequence[Rejected] = (),
) -> Discovered:
    """The one robot to drive, or a `DiscoveryError` naming every candidate found."""
    candidates = [
        d
        for d in found
        if (want is None or d.robot == want) and (namespace is None or d.namespace == namespace)
    ]
    if len(candidates) == 1:
        return candidates[0]

    offered = "; ".join(d.describe() for d in found) or "none"
    near = "; ".join(r.describe() for r in rejected)
    seen = f"drivable robots there: {offered}" + (f"; rejected: {near}" if near else "")
    if not candidates:
        wanted = f"no {want} " if want else "no robot this console can drive "
        where = f" under namespace {namespace!r}" if namespace is not None else ""
        raise DiscoveryError(
            f"{wanted}is on the wire{where} ({seen}). "
            "Name the robot and its namespace with --robot and --namespace, "
            "or check what the simulator was started with."
        )
    names = ", ".join(f"--namespace {d.namespace!r} ({d.robot})" for d in candidates)
    raise DiscoveryError(
        f"{len(candidates)} robots match on the wire ({seen}); say which with one of: {names}"
    )


def discover_from(
    present: Mapping[str, str], want: Optional[str] = None, namespace: Optional[str] = None
) -> Discovered:
    found, rejected = survey(present)
    return choose(found, want, namespace, rejected)


def unreachable(url: str, want: Optional[str], namespace: Optional[str], exc: BaseException) -> DiscoveryError:
    """The error for a wire whose `/rosapi` cannot be asked: no candidates were found."""
    asked = []
    if want:
        asked.append(f"--robot {want}")
    if namespace is not None:
        asked.append(f"--namespace {namespace!r}")
    return DiscoveryError(
        f"could not ask {url} which robots are on it: /rosapi/topics failed ({exc}); "
        f"candidates found: none{' for ' + ' '.join(asked) if asked else ''}. "
        "Start rosapi beside rosbridge, or name both --robot and --namespace."
    )


def discover(
    url: str,
    want: Optional[str] = None,
    namespace: Optional[str] = None,
    timeout: float = 2.0,
) -> Discovered:
    """Ask the rosbridge at `url` what it has, and pick the robot to drive.

    Raises `DiscoveryError` for a wire that answers and holds nothing usable, and lets a
    transport failure through as itself. Opens its own connection and closes it without
    terminating (roslibpy's reactor is process-global and single-shot).
    """
    from robot_console.fleet import list_topics

    return discover_from(list_topics(url, timeout), want, namespace)
