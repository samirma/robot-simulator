"""The console's packaged robot profiles (``profiles/<id>.yaml``) and robot-id rules.

Each profile is the console's normative typed interface and control catalog for one robot,
derived from that robot's pinned authoritative sources (every entry carries a ``source``
trace or an explicit console ``policy``). Profiles ship inside the package and are read
with ``importlib.resources``, so an installed console needs no sibling source tree.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import re
from importlib import resources
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from robot_console import dialect as d

#: Every accepted robot id, in display order (console spec §1.1).
SUPPORTED_IDS: Tuple[str, ...] = ("myagv", "ainex", "rosmaster_x3_plus", "so101", "mycobot280")


class ProfileError(ValueError):
    """A robot id or namespace request the console refuses."""


# ------------------------------------------------------------------ data model

@dataclasses.dataclass(frozen=True)
class Endpoint:
    kind: str                 # topic | service | action
    name: str                 # bare (documented hardware) name
    type: str                 # native dialect spelling
    optional: bool = False
    direction: Optional[str] = None   # topics: in (robot subscribes) / out (robot publishes)
    rate_hz: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class Camera:
    topic: str
    type: str
    encodings: Tuple[str, ...]
    stale_after_s: float
    rate_hz: Optional[float] = None
    optional: bool = False


@dataclasses.dataclass(frozen=True)
class Op:
    """One step of a documented stop/start/enable operation."""

    op: str                   # publish | call | cancel
    name: str = ""
    type: str = ""
    msg: Any = None           # publish payload / call request


@dataclasses.dataclass(frozen=True)
class Field:
    name: str
    label: str
    min: Optional[float] = None
    max: Optional[float] = None
    default: Any = None
    unit: str = ""
    integer: bool = False
    choices: Optional[Tuple[str, ...]] = None


@dataclasses.dataclass(frozen=True)
class Control:
    """A bounded page control. Only the page sends controls: it fills ``template`` (``"$field"``
    placeholders, ``{joint_prefix}`` in strings) with the bounds-checked field values."""

    id: str
    label: str
    kind: str                 # publish | call | action
    name: str
    type: str
    template: Any
    fields: Tuple[Field, ...]
    stop: Optional[Tuple[Op, ...]]    # the documented stop/cancel (the page never sends it)
    prerequisites: Tuple[str, ...]
    ok_field: str = ""        # call: the documented response field that is false on failure
    # publish: incompatible with any other (non-infrastructure) publisher of its topic, whose
    # messages would re-command the robot (checked through rosapi when the target validates)
    sole_publisher: bool = False


@dataclasses.dataclass(frozen=True)
class ReadValue:
    """One answered value of a read, as a control field: field = (raw - offset) / scale."""

    control: str
    field: str
    path: str                 # "a.b", "list.0" or "list[key=value].b" into the service answer
    offset: float = 0.0
    scale: float = 1.0


@dataclasses.dataclass(frozen=True)
class Read:
    """A documented read service returning measured joint positions. The page calls it once per
    click of its read button; teleop reads the AiNex head position through it."""

    id: str
    service: str
    type: str
    request: Any
    values: Tuple[ReadValue, ...]
    ok_field: str = ""                # answer field that is false when the read failed
    invalid: Tuple[float, ...] = ()   # raw values that mean "not read"
    invalid_if_all_zero: bool = False

    def parse(self, answer: Any) -> Dict[Tuple[str, str], Optional[float]]:
        """{(control, field): value in the field's unit, or None when not read}; ValueError when
        the answer reports a failed read."""
        if not isinstance(answer, Mapping) or (self.ok_field and answer.get(self.ok_field) is False):
            raise ValueError(f"{self.service} answered {answer}")
        raws = [read_path(answer, v.path) for v in self.values]
        nums = [float(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else None for r in raws]
        none = self.invalid_if_all_zero and all(n == 0 for n in nums)
        return {(v.control, v.field): None if none or n is None or n in self.invalid
                else (n - v.offset) / v.scale for v, n in zip(self.values, nums)}


_PATH_KEY = re.compile(r"^(\w+)\[(\w+)=(-?[\d.]+)\]$")


def read_path(value: Any, path: str) -> Any:
    """``path`` into a service answer (the same grammar as the page's ``readPath``)."""
    for seg in str(path).split("."):
        if value is None:
            return None
        m = _PATH_KEY.match(seg)
        if m:
            items = value.get(m[1]) if isinstance(value, Mapping) else None
            value = next((x for x in items or [] if isinstance(x, Mapping)
                          and float(x.get(m[2], "nan")) == float(m[3])), None)
        elif seg.isdigit():
            value = value[int(seg)] if isinstance(value, list) and int(seg) < len(value) else None
        else:
            value = value.get(seg) if isinstance(value, Mapping) else None
    return value


@dataclasses.dataclass(frozen=True)
class Axis:
    speed: float
    limit: float
    field: Optional[str] = None      # teleop_walk: gait field this axis sets


@dataclasses.dataclass(frozen=True)
class BaseTeleop:
    topic: str
    type: str
    rate_hz: float
    axes: Mapping[str, Axis]


@dataclasses.dataclass(frozen=True)
class WalkTeleop:
    param_topic: str
    param_type: str
    param_template: Mapping[str, Any]
    axes: Mapping[str, Axis]
    start: Tuple[Op, ...]
    stop: Tuple[Op, ...]
    enable: Tuple[Op, ...]
    rate_hz: float


@dataclasses.dataclass(frozen=True)
class HeadAxis:
    topic: str
    type: str
    template: Mapping[str, Any]
    field: str
    min: float
    max: float
    rate: float                       # slew while an arrow key is held, unit/s


@dataclasses.dataclass(frozen=True)
class NamespaceRule:
    default: str
    compose: str
    global_names: Tuple[str, ...]
    joint_prefix: str = ""            # e.g. "{ns}/" when joints are prefixed too


@dataclasses.dataclass(frozen=True)
class Profile:
    id: str
    name: str
    dialect: str
    teleop: str                       # base | walk | none
    sources: Mapping[str, Mapping[str, Any]]
    boot: Any
    namespace: Optional[NamespaceRule]
    owned_prefixes: Tuple[str, ...]
    endpoints: Tuple[Endpoint, ...]
    cameras: Tuple[Camera, ...]
    stop: Tuple[Op, ...]
    watchdog: str
    teleop_base: Optional[BaseTeleop]
    teleop_walk: Optional[WalkTeleop]
    head: Optional[Mapping[str, HeadAxis]]
    controls: Tuple[Control, ...]
    reads: Tuple[Read, ...]
    raw: Mapping[str, Any] = dataclasses.field(repr=False, compare=False, default_factory=dict)

    # ---- names
    @property
    def is_arm(self) -> bool:
        return self.teleop == "none"

    def resolve(self, name: str, namespace: str = "") -> str:
        """The wire name of documented name ``name`` for the selected namespace."""
        ns = (namespace or "").strip("/")
        if not ns or self.namespace is None or name in self.namespace.global_names:
            return name
        return self.namespace.compose.format(ns=ns, name=name)

    def required(self) -> List[Endpoint]:
        return [e for e in self.endpoints if not e.optional]

    def endpoint(self, name: str) -> Optional[Endpoint]:
        for e in self.endpoints:
            if e.name == name:
                return e
        return None

    def stop_description(self) -> str:
        return describe_ops(self.stop)

    def head_read(self, axis: str) -> Optional[Tuple[Read, ReadValue]]:
        """The documented read of head ``axis``'s measured position: the read value of the page
        control that publishes the same topic field."""
        h = (self.head or {}).get(axis)
        if h is None:
            return None
        ids = {c.id for c in self.controls if c.kind == "publish" and c.name == h.topic}
        return next(((r, v) for r in self.reads for v in r.values if v.control in ids and v.field == h.field), None)


def describe_ops(ops: Iterable[Op]) -> str:
    parts = []
    for o in ops:
        if o.op == "publish":
            parts.append(f"publish {o.type} {json.dumps(o.msg, separators=(',', ':'))} on {o.name}")
        elif o.op == "call":
            parts.append(f"call {o.name} ({o.type}) with {json.dumps(o.msg, separators=(',', ':'))}")
    return "; then ".join(parts) or "none documented"


# ------------------------------------------------------------------ loading

def _ops(rows: Any) -> Tuple[Op, ...]:
    out = []
    for r in rows or []:
        out.append(Op(op=r["op"], name=r.get("name", ""), type=r.get("type", ""),
                      msg=r.get("msg", r.get("request", {}))))
    return tuple(out)


def _field(r: Mapping[str, Any]) -> Field:
    ch = r.get("choices")
    return Field(name=r["name"], label=str(r.get("label", r["name"])),
                 min=None if r.get("min") is None else float(r["min"]),
                 max=None if r.get("max") is None else float(r["max"]),
                 default=r.get("default"), unit=str(r.get("unit", "")),
                 integer=bool(r.get("integer", False)),
                 choices=tuple(ch) if ch is not None else None)


def _axes(rows: Mapping[str, Any]) -> Dict[str, Axis]:
    return {k: Axis(speed=float(v["speed"]), limit=float(v["limit"]), field=v.get("field"))
            for k, v in (rows or {}).items()}


def _read(r: Mapping[str, Any]) -> Read:
    return Read(id=r["id"], service=r["service"], type=r["type"],
                request=r.get("request") or {}, ok_field=str(r.get("ok_field", "")),
                invalid=tuple(float(x) for x in r.get("invalid") or ()),
                invalid_if_all_zero=bool(r.get("invalid_if_all_zero", False)),
                values=tuple(ReadValue(control=v["control"], field=v["field"], path=str(v["path"]),
                                       offset=float(v.get("offset", 0)), scale=float(v.get("scale", 1)))
                             for v in r["values"]))


def parse_profile(raw: Mapping[str, Any]) -> Profile:
    dia = raw["dialect"]
    if dia not in d.DIALECTS:
        raise ValueError(f"{raw.get('id')}: unknown dialect {dia!r}")
    eps: List[Endpoint] = []
    for kind, key in (("topic", "topics"), ("service", "services"), ("action", "actions")):
        for r in raw.get(key) or []:
            eps.append(Endpoint(kind=kind, name=r["name"], type=r["type"],
                                optional=bool(r.get("optional", False)),
                                direction=r.get("direction"), rate_hz=r.get("rate_hz")))
    cams = tuple(Camera(topic=c["topic"], type=c["type"], encodings=tuple(c.get("encodings") or ()),
                        stale_after_s=float(c["stale_after_s"]), rate_hz=c.get("rate_hz"),
                        optional=bool(c.get("optional", False)))
                 for c in raw.get("cameras") or [])
    ns = raw.get("namespace")
    nsr = None
    if ns:
        nsr = NamespaceRule(default=str(ns.get("default", "")), compose=ns.get("compose", "/{ns}{name}"),
                            global_names=tuple(ns.get("global_names") or ()),
                            joint_prefix=str(ns.get("joint_prefix", "")))
    tb = raw.get("teleop_base")
    base = None
    if tb:
        base = BaseTeleop(topic=tb["topic"], type=tb["type"], rate_hz=float(tb.get("rate_hz", 10)),
                          axes=_axes(tb["axes"]))
    tw = raw.get("teleop_walk")
    walk = None
    if tw:
        walk = WalkTeleop(param_topic=tw["param_topic"], param_type=tw["param_type"],
                          param_template=dict(tw["param_template"]), axes=_axes(tw["axes"]),
                          start=_ops(tw.get("start")), stop=_ops(tw.get("stop")),
                          enable=_ops(tw.get("enable")), rate_hz=float(tw.get("rate_hz", 5)))
    hd = raw.get("head")
    head = None
    if hd:
        head = {k: HeadAxis(topic=v["topic"], type=v["type"], template=dict(v["template"]),
                            field=v["field"], min=float(v["min"]), max=float(v["max"]),
                            rate=float(v["rate"]))
                for k, v in hd.items()}
    controls = []
    for c in raw.get("controls") or []:
        controls.append(Control(
            id=c["id"], label=str(c.get("label", c["id"])), kind=c["kind"], name=c["name"],
            type=c["type"], template=c.get("template", {}),
            fields=tuple(_field(f) for f in c.get("fields") or []),
            stop=_ops(c["stop"]) if c.get("stop") else None,
            prerequisites=tuple(c.get("prerequisites") or ()), ok_field=str(c.get("ok_field", "")),
            sole_publisher=bool(c.get("sole_publisher", False))))
    return Profile(
        id=raw["id"], name=raw.get("name", raw["id"]), dialect=dia,
        teleop=raw.get("teleop", "none"), sources=raw.get("sources") or {}, boot=raw.get("boot"),
        namespace=nsr, owned_prefixes=tuple(raw.get("owned_prefixes") or ()),
        endpoints=tuple(eps), cameras=cams, stop=_ops(raw.get("stop")),
        watchdog=str(raw.get("watchdog", "")), teleop_base=base, teleop_walk=walk, head=head,
        controls=tuple(controls), reads=tuple(_read(r) for r in raw.get("reads") or []), raw=raw)


@functools.lru_cache(maxsize=None)
def load(robot_id: str) -> Profile:
    """The packaged profile for ``robot_id`` (must be a supported id)."""
    if robot_id not in SUPPORTED_IDS:
        raise ProfileError(refuse_unknown(robot_id))
    text = resources.files("robot_console").joinpath("profiles", f"{robot_id}.yaml").read_text("utf-8")
    return parse_profile(yaml.safe_load(text))


def load_all() -> List[Profile]:
    return [load(i) for i in SUPPORTED_IDS]


def teleop_ids() -> Tuple[str, ...]:
    return tuple(p.id for p in load_all() if not p.is_arm)


# ------------------------------------------------------------------ id selection

def refuse_unknown(robot_id: str, allowed: Sequence[str] = SUPPORTED_IDS) -> str:
    return f"unknown robot id '{robot_id}'. Accepted ids: {', '.join(allowed)}"


def refuse_arm(robot_id: str) -> str:
    return (f"'{robot_id}' is an arm and cannot be teleoperated. "
            f"Teleoperable ids: {', '.join(teleop_ids())}")


def select_id(robot_id: Optional[str], *, teleop: bool = False) -> Optional[Profile]:
    """Resolve an explicit ``--robot`` (None stays None); raise ProfileError to refuse."""
    if robot_id is None:
        return None
    allowed = teleop_ids() if teleop else SUPPORTED_IDS
    if robot_id not in SUPPORTED_IDS:
        raise ProfileError(refuse_unknown(robot_id, allowed))
    p = load(robot_id)
    if teleop and p.is_arm:
        raise ProfileError(refuse_arm(robot_id))
    return p


def check_namespace_allowed(profile: Profile, namespace: Optional[str]) -> None:
    """Refuse ``--namespace`` for a profile whose hardware interface documents none."""
    if namespace is not None and profile.namespace is None:
        raise ProfileError(
            f"'{profile.id}' offers no namespace override: its pinned hardware interface does "
            f"not document namespace configuration, so its documented names are used as they are")


# ------------------------------------------------------------------ page JSON

def expand_endpoint(p: Profile, e: Endpoint) -> List[Tuple[str, str, str]]:
    """(kind, bare name, type) of every wire endpoint an endpoint occupies."""
    if e.kind != "action":
        return [(e.kind, e.name, e.type)]
    if p.dialect == d.ROS1:
        return [("topic", n, t) for n, t in d.ros1_action_topics(e.name, e.type).items()]
    topics, services = d.ros2_action_endpoints(e.name, e.type)
    return ([("topic", n, t) for n, t in topics.items()]
            + [("service", n, t) for n, t in services.items()])


def profile_json(p: Profile) -> dict:
    """What the page needs of a profile: the typed interface it validates, its cameras, its
    bounded controls (never their stop: the page stops nothing), its measured reads and the
    ``model`` section for its 3D model (the embodiment's joint tree from the pinned vendor URDF,
    the control fields that command each joint and the documented joint-position feedback)."""
    return {
        "id": p.id, "name": p.name, "dialect": p.dialect,
        "namespace": dataclasses.asdict(p.namespace) if p.namespace else None,
        "owned_prefixes": list(p.owned_prefixes),
        "endpoints": [dict(kind=e.kind, name=e.name, type=e.type, optional=e.optional, direction=e.direction,
                           parts=[{"kind": k, "name": n, "type": t} for k, n, t in expand_endpoint(p, e)])
                      for e in p.endpoints],
        "cameras": [dict(topic=c.topic, type=c.type, encodings=list(c.encodings), stale_after_s=c.stale_after_s)
                    for c in p.cameras],
        "controls": [{k: v for k, v in dataclasses.asdict(c).items() if k != "stop"} for c in p.controls],
        "reads": [dataclasses.asdict(r) for r in p.reads],
        "model": p.raw.get("model"),
    }


def all_profiles_json() -> dict:
    """``/profiles.json``: every packaged profile and the ROS infrastructure rules."""
    return {
        "infrastructure": {
            "topics": sorted(d.INFRA_TOPICS), "prefixes": list(d.INFRA_PREFIXES),
            "node_service_suffixes": list(d._NODE_SERVICE_SUFFIXES),
            "nodes": list(d.INFRA_NODES),
        },
        "profiles": [profile_json(p) for p in load_all()],
    }
