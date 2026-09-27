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
"""

from __future__ import annotations

import dataclasses
from typing import List, Mapping, Optional, Sequence, Tuple

from robot_console import ainex_topics
from robot_console.robots import AINEX, MYAGV
from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TYPE_ODOM,
    TYPE_TWIST,
    namespaced,
)

#: `(robot, signature topic)`, most specific first. Only robots this console can *drive*
#: are here: an SO-101 has a signature of its own on the wire and no place in teleop, and
#: the worktop's camera rig under `scene` is not a robot at all.
SIGNATURES: Tuple[Tuple[str, str], ...] = (
    (AINEX, ainex_topics.TOPIC_SET_WALKING_PARAM),
    (MYAGV, TOPIC_CMD_VEL),
)

#: What must be on the wire, with which type, for a signature hit to be that robot: the
#: command topic itself and the topic that distinguishes the robot from anything else
#: that happens to take the same command (a Twist base without odometry is not a myAGV).
REQUIRED: Mapping[str, Mapping[str, str]] = {
    MYAGV: {TOPIC_CMD_VEL: TYPE_TWIST, TOPIC_ODOM: TYPE_ODOM},
    AINEX: {
        ainex_topics.TOPIC_SET_WALKING_PARAM: ainex_topics.TYPE_WALKING_PARAM,
        ainex_topics.TOPIC_IS_WALKING: ainex_topics.TYPE_BOOL,
    },
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


def survey(present: Mapping[str, str]) -> Tuple[List[Discovered], List[Rejected]]:
    """Every drivable robot the wire offers, and every near miss with its reason."""
    hits: dict = {}
    for topic in present:
        for robot, signature in SIGNATURES:
            namespace = namespace_of(topic, signature)
            if namespace is not None:
                hits.setdefault(namespace, set()).add(robot)
                break
    found: List[Discovered] = []
    rejected: List[Rejected] = []
    for namespace in sorted(hits):
        # `SIGNATURES` order decides, not the order rosapi listed the topics in.
        robot = next(r for r, _ in SIGNATURES if r in hits[namespace])
        problems = _problems(present, robot, namespace)
        if problems:
            rejected.append(Rejected(robot, namespace, "; ".join(problems)))
        else:
            found.append(Discovered(robot, namespace, _camera_for(present, namespace)))
    return found, rejected


def find_robots(present: Mapping[str, str]) -> List[Discovered]:
    """The drivable robots only; see `survey` for the near misses."""
    return survey(present)[0]


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
