"""The console's packaged robot profiles (``profiles/<id>.yaml``) and robot-id rules.

Each profile is the console's normative typed interface and control catalog for one robot,
derived from that robot's pinned authoritative sources (every entry carries a ``source``
trace or an explicit console ``policy``). Profiles ship inside the package and are read
with ``importlib.resources``, so an installed console needs no sibling source tree.
"""

from __future__ import annotations

import copy
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
    source: str = ""


@dataclasses.dataclass(frozen=True)
class Camera:
    topic: str
    type: str
    encodings: Tuple[str, ...]
    stale_after_s: float
    rate_hz: Optional[float] = None
    optional: bool = False
    source: str = ""


@dataclasses.dataclass(frozen=True)
class Op:
    """One step of a documented stop/start/enable operation."""

    op: str                   # publish | call | cancel
    name: str = ""
    type: str = ""
    msg: Any = None           # publish payload / call request
    source: str = ""


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

    def coerce(self, value: Any) -> Any:
        """Validate one user value against the documented bounds; raise ValueError."""
        if self.choices is not None:
            if value not in self.choices:
                raise ValueError(f"{self.name}: {value!r} is not one of {list(self.choices)}")
            return value
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name}: {value!r} is not a number") from None
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"{self.name}: not finite")
        if self.min is not None and v < self.min:
            raise ValueError(f"{self.name}: {v} below documented minimum {self.min}")
        if self.max is not None and v > self.max:
            raise ValueError(f"{self.name}: {v} above documented maximum {self.max}")
        if self.integer:
            if v != int(v):
                raise ValueError(f"{self.name}: must be an integer")
            return int(v)
        return v


@dataclasses.dataclass(frozen=True)
class Control:
    id: str
    label: str
    kind: str                 # publish | call | action
    name: str
    type: str
    template: Any
    fields: Tuple[Field, ...]
    stop: Optional[Tuple[Op, ...]]
    prerequisites: Tuple[str, ...]
    source: str = ""
    note: str = ""

    def build(self, values: Mapping[str, Any], joint_prefix: str = "") -> Any:
        """The command payload for ``values`` (bounds-checked), from the template."""
        checked = {}
        for f in self.fields:
            checked[f.name] = f.coerce(values.get(f.name, f.default))
        return fill_template(self.template, checked, joint_prefix)


@dataclasses.dataclass(frozen=True)
class Axis:
    speed: float
    limit: float
    unit: str
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
    unit: str


@dataclasses.dataclass(frozen=True)
class NamespaceRule:
    default: str
    compose: str
    global_names: Tuple[str, ...]
    joint_prefix: str = ""            # e.g. "{ns}/" when joints are prefixed too
    source: str = ""


@dataclasses.dataclass(frozen=True)
class Profile:
    id: str
    name: str
    dialect: str
    kind: str
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
    raw: Mapping[str, Any] = dataclasses.field(repr=False, compare=False, default_factory=dict)

    # ---- names
    @property
    def is_arm(self) -> bool:
        return self.teleop == "none"

    @property
    def namespaced(self) -> bool:
        return self.namespace is not None

    def resolve(self, name: str, namespace: str = "") -> str:
        """The wire name of documented name ``name`` for the selected namespace."""
        ns = (namespace or "").strip("/")
        if not ns or self.namespace is None or name in self.namespace.global_names:
            return name
        return self.namespace.compose.format(ns=ns, name=name)

    def joint_prefix(self, namespace: str = "") -> str:
        ns = (namespace or "").strip("/")
        if not ns or self.namespace is None:
            return ""
        return self.namespace.joint_prefix.format(ns=ns)

    def required(self) -> List[Endpoint]:
        return [e for e in self.endpoints if not e.optional]

    def endpoint(self, name: str) -> Optional[Endpoint]:
        for e in self.endpoints:
            if e.name == name:
                return e
        return None

    def stop_description(self) -> str:
        return describe_ops(self.stop)

    def to_json(self) -> dict:
        """The profile as the page consumes it (``/profiles.json``)."""
        return profile_json(self)


# ------------------------------------------------------------------ templates

_PLACEHOLDER = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")


def fill_template(template: Any, values: Mapping[str, Any], joint_prefix: str = "") -> Any:
    """Replace ``"$field"`` strings by values and ``{joint_prefix}`` inside strings."""
    if isinstance(template, Mapping):
        return {k: fill_template(v, values, joint_prefix) for k, v in template.items()}
    if isinstance(template, list):
        return [fill_template(v, values, joint_prefix) for v in template]
    if isinstance(template, str):
        m = _PLACEHOLDER.match(template)
        if m:
            if m.group(1) not in values:
                raise KeyError(f"template field ${m.group(1)} has no value")
            return values[m.group(1)]
        return template.replace("{joint_prefix}", joint_prefix)
    return copy.deepcopy(template)


def describe_ops(ops: Iterable[Op]) -> str:
    parts = []
    for o in ops:
        if o.op == "publish":
            parts.append(f"publish {o.type} {json.dumps(o.msg, separators=(',', ':'))} on {o.name}")
        elif o.op == "call":
            parts.append(f"call {o.name} ({o.type}) with {json.dumps(o.msg, separators=(',', ':'))}")
        elif o.op == "cancel":
            parts.append("cancel the goal")
    return "; then ".join(parts) or "none documented"


# ------------------------------------------------------------------ loading

def _ops(rows: Any) -> Tuple[Op, ...]:
    out = []
    for r in rows or []:
        out.append(Op(op=r["op"], name=r.get("name", ""), type=r.get("type", ""),
                      msg=r.get("msg", r.get("request", {})), source=str(r.get("source", ""))))
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
    return {k: Axis(speed=float(v["speed"]), limit=float(v["limit"]), unit=str(v.get("unit", "")),
                    field=v.get("field")) for k, v in (rows or {}).items()}


def parse_profile(raw: Mapping[str, Any]) -> Profile:
    dia = raw["dialect"]
    if dia not in d.DIALECTS:
        raise ValueError(f"{raw.get('id')}: unknown dialect {dia!r}")
    eps: List[Endpoint] = []
    for kind, key in (("topic", "topics"), ("service", "services"), ("action", "actions")):
        for r in raw.get(key) or []:
            eps.append(Endpoint(kind=kind, name=r["name"], type=r["type"],
                                optional=bool(r.get("optional", False)),
                                direction=r.get("direction"), rate_hz=r.get("rate_hz"),
                                source=str(r.get("source", ""))))
    cams = tuple(Camera(topic=c["topic"], type=c["type"], encodings=tuple(c.get("encodings") or ()),
                        stale_after_s=float(c["stale_after_s"]), rate_hz=c.get("rate_hz"),
                        optional=bool(c.get("optional", False)), source=str(c.get("source", "")))
                 for c in raw.get("cameras") or [])
    ns = raw.get("namespace")
    nsr = None
    if ns:
        nsr = NamespaceRule(default=str(ns.get("default", "")), compose=ns.get("compose", "/{ns}{name}"),
                            global_names=tuple(ns.get("global_names") or ()),
                            joint_prefix=str(ns.get("joint_prefix", "")), source=str(ns.get("source", "")))
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
                            rate=float(v["rate"]), unit=str(v.get("unit", "")))
                for k, v in hd.items()}
    controls = []
    for c in raw.get("controls") or []:
        controls.append(Control(
            id=c["id"], label=str(c.get("label", c["id"])), kind=c["kind"], name=c["name"],
            type=c["type"], template=c.get("template", {}),
            fields=tuple(_field(f) for f in c.get("fields") or []),
            stop=_ops(c["stop"]) if c.get("stop") else None,
            prerequisites=tuple(c.get("prerequisites") or ()), source=str(c.get("source", "")),
            note=str(c.get("note", ""))))
    boot = raw.get("boot")
    return Profile(
        id=raw["id"], name=raw.get("name", raw["id"]), dialect=dia, kind=raw.get("kind", ""),
        teleop=raw.get("teleop", "none"), sources=raw.get("sources") or {}, boot=boot,
        namespace=nsr, owned_prefixes=tuple(raw.get("owned_prefixes") or ()),
        endpoints=tuple(eps), cameras=cams, stop=_ops(raw.get("stop")),
        watchdog=str(raw.get("watchdog", "")), teleop_base=base, teleop_walk=walk, head=head,
        controls=tuple(controls), raw=raw)


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
    def ops(o: Optional[Sequence[Op]]):
        return None if o is None else [dataclasses.asdict(x) for x in o]

    return {
        "id": p.id, "name": p.name, "dialect": p.dialect, "kind": p.kind, "teleop": p.teleop,
        "sources": p.sources,
        "namespace": dataclasses.asdict(p.namespace) if p.namespace else None,
        "owned_prefixes": list(p.owned_prefixes),
        "endpoints": [dict(dataclasses.asdict(e),
                           parts=[{"kind": k, "name": n, "type": t} for k, n, t in expand_endpoint(p, e)])
                      for e in p.endpoints],
        "cameras": [dataclasses.asdict(c) for c in p.cameras],
        "stop": ops(p.stop),
        "watchdog": p.watchdog,
        "controls": [dict(dataclasses.asdict(c), stop=ops(c.stop)) for c in p.controls],
    }


def all_profiles_json() -> dict:
    return {
        "supported_ids": list(SUPPORTED_IDS),
        "infrastructure": {
            "topics": sorted(d.INFRA_TOPICS), "prefixes": list(d.INFRA_PREFIXES),
            "node_service_suffixes": list(d._NODE_SERVICE_SUFFIXES),
        },
        "profiles": [profile_json(p) for p in load_all()],
    }
