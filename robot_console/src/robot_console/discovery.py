"""Typed discovery of robots on a rosbridge wire, validation against profiles, and target
selection. Shared by teleop, fleet and (as the same algorithm in ``web/console.js``) the page.

Discovery reads the graph through ``rosapi`` only (never publishes). A profile is a
*candidate* at a namespace when every one of its required endpoint names is present there;
it then passes validation when every present endpoint has its documented type and nothing
else is under the profile's names (its owned prefixes, or the whole namespace) but its
optional rows and ROS infrastructure.

Stock ROS 2 rosapi cannot report action types (its ``action_type`` service crashes the
rosapi node on Jazzy and hangs on Humble, and hidden ``_action`` endpoints are not listed),
so a ROS 2 action is identified by name through ``/rosapi/action_servers`` and its type is
reported as not verifiable over rosbridge. The console never calls ``action_type``.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from robot_console import dialect as d
from robot_console.profiles import Profile, ProfileError, check_namespace_allowed, expand_endpoint, load_all
from robot_console.rosbridge import Rosbridge, ServiceError, TransportError


class DiscoveryError(RuntimeError):
    """The wire could not be read (unreachable, rosapi missing or failing)."""


class SelectionError(RuntimeError):
    """No unique validated target: carries the reason and the candidates found."""

    def __init__(self, reason: str, candidates: Sequence["Target"] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.candidates = list(candidates)


# ------------------------------------------------------------------ graph

@dataclasses.dataclass
class Graph:
    dialect: Optional[str]
    topics: Dict[str, str]
    services: Dict[str, str]            # "" when rosapi could not tell the type
    actions: Dict[str, Optional[str]]   # ROS 2: None (type not reportable)
    action_parts: Set[str]              # wire names that belong to a discovered action
    distro: str = ""

    def names(self) -> Set[str]:
        return set(self.topics) | set(self.services) | set(self.actions)

    def has(self, kind: str, name: str) -> bool:
        return name in {"topic": self.topics, "service": self.services, "action": self.actions}[kind]

    def type_of(self, kind: str, name: str) -> Optional[str]:
        return {"topic": self.topics, "service": self.services, "action": self.actions}[kind].get(name)

    def cameras(self) -> Dict[str, str]:
        """Every topic of a supported raw image type (sensor_msgs/Image)."""
        return {n: t for n, t in self.topics.items() if d.normalize_type(t) == "sensor_msgs/Image"}


def _call(rb: Rosbridge, service: str, args: Optional[dict] = None, timeout: float = 5.0) -> dict:
    try:
        return rb.call_service(service, args, timeout=timeout)
    except (ServiceError, TimeoutError, TransportError) as exc:
        raise DiscoveryError(f"rosapi {service} failed: {exc}") from None


def fetch_graph(rb: Rosbridge, timeout: float = 5.0) -> Graph:
    """Read topics, services (typed), actions and the dialect through rosapi."""
    tr = _call(rb, "/rosapi/topics", timeout=timeout)
    topics = dict(zip(tr.get("topics") or [], tr.get("types") or []))
    services_list = list((_call(rb, "/rosapi/services", timeout=timeout).get("services")) or [])
    dia: Optional[str] = None
    for t in topics.values():
        dia = d.type_dialect(t)
        if dia == d.ROS2:
            break
    # ROS 2 stock nodes always publish /parameter_events and /rosout with ROS 2 spellings.
    distro = ""
    if "/rosapi/get_ros_version" in services_list:
        try:
            v = rb.call_service("/rosapi/get_ros_version", timeout=timeout)
            dia = d.ROS2 if int(v.get("version", 1)) == 2 else d.ROS1
            distro = str(v.get("distro", ""))
        except (ServiceError, TimeoutError, TransportError, ValueError):
            pass
    # Service types, pipelined; infrastructure services are skipped (never part of a robot).
    services: Dict[str, str] = {}
    futs: Dict[str, concurrent.futures.Future] = {}
    for s in services_list:
        if d.is_infrastructure(s, "service"):
            continue
        futs[s] = rb.call_service_async("/rosapi/service_type", {"service": s})
    for s, f in futs.items():
        try:
            services[s] = str(f.result(timeout).get("type") or "")
        except (ServiceError, TransportError, concurrent.futures.TimeoutError):
            services[s] = ""
        except Exception:  # noqa: BLE001
            services[s] = ""
    if dia is None:
        for t in services.values():
            dia = d.type_dialect(t) or dia
    actions: Dict[str, Optional[str]] = {}
    parts: Set[str] = set()
    if dia == d.ROS1:
        for n, t in topics.items():
            hit = d.ros1_action_from_goal_topic(n, t)
            if hit and all(f"{hit[0]}/{p}" in topics for p in ("cancel", "status", "feedback", "result")):
                actions[hit[0]] = hit[1]
                parts |= set(d.ros1_action_topics(hit[0], hit[1]))
    elif dia == d.ROS2:
        if "/rosapi/action_servers" in services_list:
            ar = _call(rb, "/rosapi/action_servers", timeout=timeout)
            for a in ar.get("action_servers") or []:
                actions[a] = None
        for s, t in list(services.items()):   # a rosapi that does list hidden services
            hit = d.ros2_action_from_send_goal(s, t)
            if hit:
                actions[hit[0]] = hit[1]
        for a, t in actions.items():
            tp, sv = d.ros2_action_endpoints(a, t or "x/action/X")
            parts |= set(tp) | set(sv)
    return Graph(dialect=dia, topics=topics, services=services, actions=actions,
                 action_parts=parts, distro=distro)


# ------------------------------------------------------------------ validation

@dataclasses.dataclass
class Validation:
    missing: List[Tuple[str, str, str]] = dataclasses.field(default_factory=list)      # kind, name, type
    wrong_type: List[Tuple[str, str, str, str]] = dataclasses.field(default_factory=list)  # kind, name, want, got
    unexpected: List[Tuple[str, str]] = dataclasses.field(default_factory=list)       # kind, name
    unverified: List[Tuple[str, str, str]] = dataclasses.field(default_factory=list)   # kind, name, why
    dialect_mismatch: Optional[str] = None
    optional_present: List[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.wrong_type or self.unexpected or self.dialect_mismatch)

    def problems(self) -> List[str]:
        out = []
        if self.dialect_mismatch:
            out.append(self.dialect_mismatch)
        out += [f"missing {k} {n} ({t})" for k, n, t in self.missing]
        out += [f"wrong type on {k} {n}: expected {w}, found {g or 'unknown'}" for k, n, w, g in self.wrong_type]
        out += [f"unexpected {k} {n} under the profile's names" for k, n in self.unexpected]
        return out


@dataclasses.dataclass
class Target:
    profile: Profile
    namespace: str
    validation: Validation

    @property
    def label(self) -> str:
        return self.profile.id + (f" (namespace /{self.namespace})" if self.namespace else "")

    def wire(self, name: str) -> str:
        return self.profile.resolve(name, self.namespace)

    def required_names(self) -> Set[str]:
        return {self.wire(e.name) for e in self.profile.required()}


def validate(p: Profile, ns: str, g: Graph) -> Validation:
    v = Validation()
    if g.dialect and g.dialect != p.dialect:
        v.dialect_mismatch = (f"the wire is {g.dialect.upper()} but profile '{p.id}' is "
                              f"{p.dialect.upper()}")
    claimed: Set[str] = set()
    for e in p.endpoints:
        wire = p.resolve(e.name, ns)
        for _, part, _ in expand_endpoint(p, e):
            claimed.add(p.resolve(part, ns))
        claimed.add(wire)
        if not g.has(e.kind, wire):
            if not e.optional:
                v.missing.append((e.kind, wire, e.type))
            continue
        if e.optional:
            v.optional_present.append(wire)
        got = g.type_of(e.kind, wire)
        if e.kind == "action" and got is None:
            v.unverified.append((e.kind, wire, "stock ROS 2 rosapi cannot report action types"))
            continue
        if not got or not d.types_equal(got, e.type):
            v.wrong_type.append((e.kind, wire, e.type, got or ""))
    # Nothing else under the profile's names.
    owned = [p.resolve(pref, ns) for pref in p.owned_prefixes]
    nsp = f"/{ns.strip('/')}/" if ns else None
    for kind, table in (("topic", g.topics), ("service", g.services), ("action", g.actions)):
        for name in table:
            if name in claimed or d.is_infrastructure(name, kind):
                continue
            if name in g.action_parts and any(name.startswith(a + "/") for a in claimed):
                continue
            if any(name.startswith(o) for o in owned) or (nsp and name.startswith(nsp)):
                v.unexpected.append((kind, name))
    return v


def _namespaces_for(p: Profile, g: Graph) -> List[str]:
    """Candidate namespaces where ``p``'s required names could be (default first)."""
    if p.namespace is None:
        return [""]
    out = [p.namespace.default.strip("/")]
    wire = g.names()
    for e in p.required():
        if e.name in p.namespace.global_names:
            continue
        for w in wire:
            if w != e.name and w.endswith(e.name):
                prefix = w[: -len(e.name)].strip("/")
                if prefix and prefix not in out:
                    out.append(prefix)
    return out


def candidates(g: Graph, profiles: Iterable[Profile], namespace: Optional[str] = None
               ) -> Tuple[List[Target], List[Tuple[Target, List[str]]]]:
    """(full candidates, near misses with their missing names)."""
    full: List[Target] = []
    near: List[Tuple[Target, List[str]]] = []
    for p in profiles:
        if g.dialect and g.dialect != p.dialect:
            continue
        nss = [namespace.strip("/")] if (namespace is not None and p.namespace) else _namespaces_for(p, g)
        for ns in nss:
            req = p.required()
            present = [e for e in req if g.has(e.kind, p.resolve(e.name, ns))]
            t = Target(p, ns, validate(p, ns, g))
            if req and len(present) == len(req):
                full.append(t)
            elif req and len(present) * 2 >= len(req):
                near.append((t, [p.resolve(e.name, ns) for e in req if e not in present]))
    return full, near


def _prune_dominated(full: List[Target]) -> Tuple[List[Target], List[Tuple[Target, Target]]]:
    """Drop a candidate whose required names are a strict subset of another candidate's
    (the other presents strictly more typed evidence). Returns (kept, [(dropped, by)])."""
    kept, dropped = [], []
    for a in full:
        by = next((b for b in full if b is not a and a.required_names() < b.required_names()), None)
        if by is None:
            kept.append(a)
        else:
            dropped.append((a, by))
    return kept, dropped


@dataclasses.dataclass
class Discovery:
    graph: Graph
    targets: List[Target]                       # distinct, non-overlapping identified targets
    ambiguous: List[List[Target]]               # overlapping candidate groups
    dominated: List[Tuple[Target, Target]]
    near: List[Tuple[Target, List[str]]]


def discover(g: Graph, profiles: Optional[Iterable[Profile]] = None,
             namespace: Optional[str] = None) -> Discovery:
    profiles = list(profiles) if profiles is not None else load_all()
    full, near = candidates(g, profiles, namespace)
    kept, dominated = _prune_dominated(full)
    # Group overlapping candidates (shared wire names) -> ambiguity.
    groups: List[List[Target]] = []
    for t in kept:
        for grp in groups:
            if any(t.required_names() & o.required_names() for o in grp):
                grp.append(t)
                break
        else:
            groups.append([t])
    targets = [grp[0] for grp in groups if len(grp) == 1]
    ambiguous = [grp for grp in groups if len(grp) > 1]
    return Discovery(g, targets, ambiguous, dominated, near)


def describe_candidates(ts: Iterable[Target]) -> str:
    return ", ".join(t.label for t in ts)


def select_target(g: Graph, robot: Optional[Profile], namespace: Optional[str]) -> Target:
    """The unique validated target for an entry point, or SelectionError."""
    if robot is not None:
        check_namespace_allowed(robot, namespace)
    disc = discover(g, None, namespace if (robot is None or robot.namespace) else None)
    pool = [t for t in disc.targets + [x for grp in disc.ambiguous for x in grp]]
    if robot is not None:
        mine = [t for t in pool if t.profile.id == robot.id]
        mine_dom = [(a, b) for a, b in disc.dominated if a.profile.id == robot.id]
        if not mine and mine_dom:
            a, b = mine_dom[0]
            raise SelectionError(f"'{robot.id}' is not uniquely identified: the wire presents "
                                 f"{b.label}, whose interface contains all of {robot.id}'s required "
                                 f"names", [a, b])
        if not mine:
            near = [(t, m) for t, m in disc.near if t.profile.id == robot.id]
            if near:
                t, missing = near[0]
                raise SelectionError(f"'{robot.id}' not found on the wire: missing "
                                     + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else ""), [t])
            raise SelectionError(f"'{robot.id}' not found on the wire (none of its required "
                                 f"interface is present)")
        if len(mine) > 1:
            default = robot.namespace.default.strip("/") if robot.namespace else ""
            pick = [t for t in mine if t.namespace == default] if namespace is None else []
            if len(pick) == 1:
                mine = pick
            else:
                raise SelectionError(f"several '{robot.id}' targets found; select one by its namespace: "
                                     f"{describe_candidates(mine)}", mine)
        t = mine[0]
        others = [grp for grp in disc.ambiguous if t in grp]
        if others and robot.namespace is None:
            rivals = [x for x in others[0] if x.profile.id != robot.id]
            if rivals:
                raise SelectionError(f"ambiguous wire: {describe_candidates(others[0])} all match",
                                     others[0])
    else:
        if disc.ambiguous:
            grp = disc.ambiguous[0]
            same = {x.profile.id for x in grp}
            if len(same) == 1:
                p = grp[0].profile
                default = p.namespace.default.strip("/") if p.namespace else ""
                pick = [x for x in grp if x.namespace == default]
                if len(pick) == 1 and namespace is None:
                    grp = pick
            if len(grp) > 1:
                raise SelectionError("ambiguous wire, select a robot explicitly (--robot/--namespace); "
                                     f"candidates: {describe_candidates(grp)}", grp)
            disc.targets.append(grp[0])
        if len(disc.targets) != 1:
            if not disc.targets:
                near = "; ".join(f"{t.label} (missing {', '.join(m[:4])})" for t, m in disc.near)
                raise SelectionError("no supported robot identified on the wire"
                                     + (f"; incomplete candidates: {near}" if near else ""),
                                     [t for t, _ in disc.near])
            raise SelectionError("several robots on the wire, select one with --robot: "
                                 f"{describe_candidates(disc.targets)}", disc.targets)
        t = disc.targets[0]
    if namespace is not None and t.profile.namespace is None:
        raise ProfileError(f"'{t.profile.id}' offers no namespace override")
    if not t.validation.ok:
        raise SelectionError(f"{t.label} failed typed validation: " + "; ".join(t.validation.problems()),
                             [t])
    return t
