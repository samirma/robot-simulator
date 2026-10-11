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
SUPPORTED_IDS: Tuple[str, ...] = ("myagv", "ainex", "rosmaster_x3_plus", "rebot_b601_rs")


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
    """A raw image stream the page shows. ``kind`` ``color`` is shown as the image it carries; ``depth``
    is a single-channel 16-bit range image: each value times ``unit_m`` is a distance in metres (0 = no
    depth) and is shown as a false-colour map scaled to ``range_m`` (near, far), the camera's range;
    ``ir`` is a single-channel intensity image (an infrared stream) shown in greyscale, each frame
    stretched from its own darkest to its own brightest value. ``camera_info`` is the stream's
    documented ``sensor_msgs/CameraInfo`` topic, where there is one.

    Bandwidth (console spec §2.3): ``max_rate_hz`` caps the frames rosbridge sends for the stream
    (``throttle_rate``; None: the console default, ``DEFAULT_CAMERA_MAX_HZ``). Every camera is subscribed on
    validation (the page's user may stop and start it). ``silent_note`` explains a stream that is advertised but may never deliver a frame: shown,
    instead of an alarm, while no frame has arrived."""
    topic: str
    type: str
    encodings: Tuple[str, ...]
    stale_after_s: float
    rate_hz: Optional[float] = None
    optional: bool = False
    kind: str = "color"
    range_m: Optional[Tuple[float, float]] = None
    unit_m: Optional[float] = None
    camera_info: Optional[str] = None
    missing_note: str = ""    # shown when an optional camera is not on the wire (e.g. which launch starts it)
    max_rate_hz: Optional[float] = None
    silent_note: str = ""
    caption: str = ""         # a short note shown on the card (e.g. how the stream is registered)

    @property
    def compressed(self) -> bool:
        """A sensor_msgs/CompressedImage stream (its ``encodings`` are the codecs it documents: jpeg, png)."""
        return d.normalize_type(self.type) == "sensor_msgs/CompressedImage"

    def nominal_mbps(self) -> float:
        """The link load the console assumes before a frame is measured (console spec §2.3): a 640x480 frame of the
        first encoding (a compressed one taken as a tenth of rgb8) as base64 JSON, at the rate cap."""
        enc = (self.encodings or ("rgb8",))[0].lower()
        bpp = {"mono8": 1, "8uc1": 1, "mono16": 2, "16uc1": 2, "rgba8": 4, "bgra8": 4}.get(enc, 3)
        raw = 640 * 480 * (0.3 if self.compressed else bpp)
        return (raw * 4 / 3 + 300) * 8 * (1000.0 / self.throttle_ms()) / 1e6

    def live_bound_s(self) -> float:
        """The age (robot time) up to which a shown image of this stream reads live (console spec §2.3)."""
        return LAG_LIMIT_S + LIVE_PERIODS * self.throttle_ms() / 1000.0

    def throttle_ms(self) -> int:
        """rosbridge ``throttle_rate`` of a subscription to this stream."""
        return int(round(1000.0 / (self.max_rate_hz or DEFAULT_CAMERA_MAX_HZ)))


#: Console policy: frames per second rosbridge sends for a camera whose profile records no ``max_rate_hz``. The
#: real protection against a backlog is queue_length 1 (rosbridge keeps only the newest frame between sends); a
#: profile that needs a lower cap (a base whose single stop shares the link, say) records its own, with its reason.
DEFAULT_CAMERA_MAX_HZ = 10.0
#: Image age (console spec §2.3; camera.py): a frame arriving more than LAG_LIMIT_S after its stamp is LAGGING, and a
#: shown image is live while at most LAG_LIMIT_S + LIVE_PERIODS frame periods old, so a steady transport lag below
#: LAG_LIMIT_S never flickers between frames. MAX_PERIOD_S bounds the frame period, so that bound stays short.
LAG_LIMIT_S = 1.0
LIVE_PERIODS = 1.5
MAX_PERIOD_S = 1.0
#: Console policy: frames per second for a stream not tied to a target (a preview before validation).
UNTIED_CAMERA_MAX_HZ = 2.0

CAMERA_KINDS = ("color", "depth", "ir")
#: The codecs a sensor_msgs/CompressedImage camera may document (the browser decodes them).
COMPRESSED_FORMATS = ("jpeg", "png")
#: The single-channel encodings an ``ir`` camera may document.
IR_ENCODINGS = ("mono8", "8uc1", "mono16", "16uc1")


#: How a sensor field is shown (console spec §2.5): ``number`` a finite number (``unit``, ``decimals``),
#: ``text`` a string, ``bool`` true/false, ``list`` the elements of an array joined by ", ", ``bits`` an
#: unsigned integer whose documented bits (``bits``) are each shown set or clear, ``vector3`` the x, y
#: and z of a vector (``unit``, ``decimals``), ``rpy`` a quaternion (x, y, z, w) shown as roll, pitch and
#: yaw in degrees, ``count`` the number of elements of an array.
SENSOR_KINDS = ("number", "text", "bool", "list", "bits", "vector3", "rpy", "count")
#: How a sensor's whole message may be drawn besides its fields: ``scan`` a ``sensor_msgs/LaserScan`` as a
#: top-down plot of its returns.
SENSOR_PLOTS = ("scan",)


@dataclasses.dataclass(frozen=True)
class SensorBit:
    bit: int
    label: str


@dataclasses.dataclass(frozen=True)
class SensorField:
    """One value the sensor panel shows from a message of a sensor topic: ``path`` into the message
    (``read_path``'s grammar, e.g. ``position.0``)."""

    name: str
    label: str
    path: str
    kind: str = "number"
    unit: str = ""
    decimals: Optional[int] = None
    bits: Tuple[SensorBit, ...] = ()
    group: str = ""           # fields of one group are shown together under the group's label
    headline: bool = False    # a number also shown large at the top of the panel (e.g. battery voltage)
    warn_below: Optional[float] = None   # a number below this documented level is marked low
    no_reading: Tuple[float, ...] = ()   # documented values that mean "nothing read yet / no answer"
    no_reading_label: str = ""           # what such a value means, shown instead of it
    # an rpy field: the rpy (URDF convention) of the message's frame in the frame to show it in; the orientation is
    # shown re-expressed in that frame (q * conj(q_mount)), e.g. an IMU's orientation as the robot base's
    mount_rpy: Optional[Tuple[float, float, float]] = None


@dataclasses.dataclass(frozen=True)
class Sensor:
    """A manufacturer sensor topic the view page subscribes to and shows read-only (console spec §2.5):
    an ``out`` topic of the profile. ``stale_after_s`` None: a latched or non-periodic topic, shown
    with its last message's age but never as stale. ``throttle_s``: the page asks rosbridge for at
    most one message per this period (None: the page's default, 0.1 s); rosbridge still receives every
    message from ROS, so this bounds only the websocket; a sensor with a ``throttle_s`` can be stopped and
    started on the page, and pauses with the cameras on congestion. Every sensor is subscribed on
    validation. ``plot``: how the whole message is
    also drawn (``SENSOR_PLOTS``). ``caption``: a short note shown on the card (e.g. a unit caveat)."""

    id: str
    label: str
    topic: str                # bare (documented) name, resolved like every profile name
    type: str
    stale_after_s: Optional[float]
    fields: Tuple[SensorField, ...]
    throttle_s: Optional[float] = None
    plot: str = ""
    caption: str = ""
    event_only: bool = False   # published only on an event (no latching, no period): waiting is normal


@dataclasses.dataclass(frozen=True)
class Op:
    """One step of a documented stop/start/enable operation."""

    op: str                   # publish | call | cancel
    name: str = ""
    type: str = ""
    msg: Any = None           # publish payload / call request


@dataclasses.dataclass(frozen=True)
class WireMap:
    """A field's documented unit mapping onto the wire, the inverse of a read's: the template gets
    offset + scale * value, rounded to an integer when ``integer`` (the vendor rounds the scaled
    value, then adds its integer offset), and that wire value must lie within ``min``..``max``."""

    offset: float = 0.0
    scale: float = 1.0
    integer: bool = False
    min: Optional[float] = None
    max: Optional[float] = None

    def apply(self, value: float) -> float:
        """The wire value of field value ``value``; ValueError when it is outside the wire range."""
        x = self.offset + (round(self.scale * value) if self.integer else self.scale * value)
        if self.integer:
            x = int(round(x))
        if (self.min is not None and x < self.min) or (self.max is not None and x > self.max):
            raise ValueError(f"{value} maps to {x}, outside {self.min}..{self.max} on the wire")
        return x


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
    to_wire: Optional[WireMap] = None  # documented field -> wire mapping (None: sent as entered)

    def wire(self, value: float) -> float:
        """The value the template gets for field value ``value``."""
        return self.to_wire.apply(value) if self.to_wire else value


@dataclasses.dataclass(frozen=True)
class Group:
    """Controls the page lists together (e.g. the servos of one limb), with a note shown above them."""

    id: str
    label: str
    note: str = ""


#: What an action server documentedly does with a goal that arrives while one of its goals runs
#: (a control's ``busy.new_goal``): ``preempt`` ends the running goal and runs the new one;
#: ``reject`` refuses the new goal; ``parallel`` accepts it and runs it beside the running one.
#: The page streams goals only to a ``preempt`` server; to any other it keeps one goal of the
#: control outstanding and sends the latest value when that goal's result arrives (console spec §2.2).
BUSY = ("preempt", "reject", "parallel")


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
    group: str = ""           # the profile group the page lists it in ("" = a card of its own)
    busy: str = ""            # action controls: the server's handling of a goal while one runs (BUSY)


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
    click of its read button, or, for the read with ``poll_s`` (the profile's documented read-only
    joint-position query, console spec §2.2), about every ``poll_s`` seconds
    once the target is validated; teleop reads the AiNex head position through it."""

    id: str
    service: str
    type: str
    request: Any
    values: Tuple[ReadValue, ...]
    ok_field: str = ""                # answer field that is false when the read failed
    invalid: Tuple[float, ...] = ()   # raw values that mean "not read"
    poll_s: Optional[float] = None    # the page polls it (model feedback) at about this period

    def parse(self, answer: Any) -> Dict[Tuple[str, str], Optional[float]]:
        """{(control, field): value in the field's unit, or None when not read}; ValueError when
        the answer reports a failed read."""
        if not isinstance(answer, Mapping) or (self.ok_field and answer.get(self.ok_field) is False):
            raise ValueError(f"{self.service} answered {answer}")
        raws = [read_path(answer, v.path) for v in self.values]
        nums = [float(r) if isinstance(r, (int, float)) and not isinstance(r, bool) else None for r in raws]
        return {(v.control, v.field): None if n is None or n in self.invalid
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
    groups: Tuple[Group, ...] = ()
    sensors: Tuple[Sensor, ...] = ()
    # the most the page's camera streams may take of the link together (Mbit/s; console policy); None: no budget
    link_budget_mbps: Optional[float] = None
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


def _opt_float(v: Any) -> Optional[float]:
    return None if v is None else float(v)


def _wire_map(r: Optional[Mapping[str, Any]]) -> Optional[WireMap]:
    if r is None:
        return None
    return WireMap(offset=float(r.get("offset", 0)), scale=float(r.get("scale", 1)),
                   integer=bool(r.get("integer", False)), min=_opt_float(r.get("min")), max=_opt_float(r.get("max")))


def _field(r: Mapping[str, Any]) -> Field:
    ch = r.get("choices")
    return Field(name=r["name"], label=str(r.get("label", r["name"])),
                 min=_opt_float(r.get("min")), max=_opt_float(r.get("max")),
                 default=r.get("default"), unit=str(r.get("unit", "")),
                 integer=bool(r.get("integer", False)),
                 choices=tuple(ch) if ch is not None else None, to_wire=_wire_map(r.get("to_wire")))


def _check_field(where: str, f: Field) -> None:
    """A mapped field's documented limits map inside the wire range; it is numeric and bounded."""
    w = f.to_wire
    if w is None:
        return
    if f.choices is not None or f.min is None or f.max is None or w.scale == 0:
        raise ValueError(f"{where}: to_wire needs a bounded numeric field and a non-zero scale")
    for v in (f.min, f.max):
        try:
            w.apply(v)
        except ValueError as exc:
            raise ValueError(f"{where}: documented limit {exc}") from None


def _busy(where: str, c: Mapping[str, Any]) -> str:
    """An action control's documented ``busy.new_goal`` (required); no other control has one."""
    b = c.get("busy")
    if c["kind"] != "action":
        if b is not None:
            raise ValueError(f"{where}: busy is recorded for action controls only")
        return ""
    v = b.get("new_goal") if isinstance(b, Mapping) else None
    if v not in BUSY:
        raise ValueError(f"{where}: an action control records busy.new_goal, one of {', '.join(BUSY)} (got {v!r})")
    return v


def _axes(rows: Mapping[str, Any]) -> Dict[str, Axis]:
    return {k: Axis(speed=float(v["speed"]), limit=float(v["limit"]), field=v.get("field"))
            for k, v in (rows or {}).items()}


def _read(r: Mapping[str, Any]) -> Read:
    return Read(id=r["id"], service=r["service"], type=r["type"],
                request=r.get("request") or {}, ok_field=str(r.get("ok_field", "")),
                invalid=tuple(float(x) for x in r.get("invalid") or ()),
                poll_s=_opt_float(r.get("poll_s")),
                values=tuple(ReadValue(control=v["control"], field=v["field"], path=str(v["path"]),
                                       offset=float(v.get("offset", 0)), scale=float(v.get("scale", 1)))
                             for v in r["values"]))


def _sensor(pid: str, r: Mapping[str, Any]) -> Sensor:
    where = f"{pid}.sensors.{r['id']}"
    fields = []
    for f in r.get("fields") or []:
        kind = str(f.get("kind", "number"))
        if kind not in SENSOR_KINDS:
            raise ValueError(f"{where}.{f['name']}: kind {kind!r} is none of {', '.join(SENSOR_KINDS)}")
        bits = tuple(SensorBit(bit=int(b["bit"]), label=str(b["label"])) for b in f.get("bits") or ())
        if bool(bits) != (kind == "bits"):
            raise ValueError(f"{where}.{f['name']}: bits are recorded for (and only for) kind bits")
        if len({b.bit for b in bits}) != len(bits) or any(not 0 <= b.bit <= 31 for b in bits):
            raise ValueError(f"{where}.{f['name']}: bits are distinct and within 0..31")
        dec = f.get("decimals")
        if dec is not None and (isinstance(dec, bool) or not isinstance(dec, int) or dec < 0):
            raise ValueError(f"{where}.{f['name']}: decimals is a non-negative integer")
        if not str(f.get("path", "")):
            raise ValueError(f"{where}.{f['name']}: no path")
        headline, warn = bool(f.get("headline", False)), _opt_float(f.get("warn_below"))
        nr = tuple(float(x) for x in f.get("no_reading") or ())
        if (headline or warn is not None or nr) and kind != "number":
            raise ValueError(f"{where}.{f['name']}: headline, warn_below and no_reading belong to a number")
        mount = f.get("mount_rpy")
        if mount is not None:
            if kind != "rpy" or not (isinstance(mount, (list, tuple)) and len(mount) == 3
                                     and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in mount)):
                raise ValueError(f"{where}.{f['name']}: mount_rpy is [roll, pitch, yaw] of an rpy field")
            mount = tuple(float(x) for x in mount)
        if bool(nr) != bool(str(f.get("no_reading_label") or "")):
            raise ValueError(f"{where}.{f['name']}: no_reading values come with their no_reading_label")
        fields.append(SensorField(name=f["name"], label=str(f.get("label", f["name"])), path=str(f["path"]),
                                  kind=kind, unit=str(f.get("unit", "")), decimals=dec, bits=bits,
                                  group=str(f.get("group") or ""), headline=headline, warn_below=warn,
                                  no_reading=nr, no_reading_label=str(f.get("no_reading_label") or ""),
                                  mount_rpy=mount))
    if not fields or len({f.name for f in fields}) != len(fields):
        raise ValueError(f"{where}: one or more fields with distinct names")
    stale = _opt_float(r.get("stale_after_s"))
    if stale is not None and stale <= 0:
        raise ValueError(f"{where}: stale_after_s is positive (or null for a latched topic)")
    throttle = _opt_float(r.get("throttle_s"))
    if throttle is not None and (throttle <= 0 or (stale is not None and stale <= throttle)):
        raise ValueError(f"{where}: throttle_s is positive and shorter than stale_after_s")
    plot = str(r.get("plot") or "")
    if plot and plot not in SENSOR_PLOTS:
        raise ValueError(f"{where}: plot {plot!r} is none of {', '.join(SENSOR_PLOTS)}")
    if plot == "scan" and d.normalize_type(r["type"]) != "sensor_msgs/LaserScan":
        raise ValueError(f"{where}: a scan plot draws a sensor_msgs/LaserScan")
    if "on_demand" in r:
        raise ValueError(f"{where}: no on_demand: every sensor is subscribed on validation (console spec §2.5)")
    event_only = bool(r.get("event_only", False))
    if event_only and stale is not None:
        raise ValueError(f"{where}: an event_only sensor has no stale_after_s")
    return Sensor(id=r["id"], label=str(r.get("label", r["id"])), topic=r["topic"], type=r["type"],
                  stale_after_s=stale, fields=tuple(fields), throttle_s=throttle, plot=plot,
                  caption=" ".join(str(r.get("caption") or "").split()), event_only=event_only)


def _check_sensors(raw: Mapping[str, Any], sensors: Sequence[Sensor], eps: Sequence[Endpoint]) -> None:
    """Every sensor reads an ``out`` topic of the profile, of the profile's type for it; none is a camera."""
    if len({s.id for s in sensors}) != len(sensors):
        raise ValueError(f"{raw['id']}: sensor ids are distinct")
    cams = {c["topic"] for c in raw.get("cameras") or []}
    for s in sensors:
        e = next((x for x in eps if x.kind == "topic" and x.name == s.topic), None)
        if e is None or e.direction != "out" or e.type != s.type:
            raise ValueError(f"{raw['id']}.sensors.{s.id}: {s.topic} ({s.type}) is not an out topic "
                             f"of that type in the profile")
        if s.topic in cams:
            raise ValueError(f"{raw['id']}.sensors.{s.id}: {s.topic} is a camera, not a sensor")


def _camera(raw: Mapping[str, Any], c: Mapping[str, Any]) -> Camera:
    where = f"{raw['id']}.cameras.{c['topic']}"
    kind = c.get("kind", "color")
    if kind not in CAMERA_KINDS:
        raise ValueError(f"{where}: kind {kind!r} is none of {', '.join(CAMERA_KINDS)}")
    rng = unit = None
    if kind == "depth":
        enc = [str(e).lower() for e in c.get("encodings") or ()]
        if enc != ["16uc1"]:
            raise ValueError(f"{where}: a depth camera's only encoding is 16UC1")
        r, u = c.get("range_m"), c.get("unit_m")
        num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(num(x) for x in r) and 0 <= r[0] < r[1]):
            raise ValueError(f"{where}: range_m is [near, far] metres with 0 <= near < far")
        if not num(u) or u <= 0:
            raise ValueError(f"{where}: unit_m is the positive metres per depth value")
        rng, unit = (float(r[0]), float(r[1])), float(u)
    elif "range_m" in c or "unit_m" in c:
        raise ValueError(f"{where}: range_m and unit_m belong to a depth camera")
    if d.normalize_type(c["type"]) == "sensor_msgs/CompressedImage":
        enc = [str(e).lower() for e in c.get("encodings") or ()]
        if kind != "color" or not enc or any(e not in COMPRESSED_FORMATS for e in enc):
            raise ValueError(f"{where}: a compressed camera is a colour camera whose encodings are its codecs "
                             f"({', '.join(COMPRESSED_FORMATS)})")
    if kind == "ir":
        enc = [str(e).lower() for e in c.get("encodings") or ()]
        if not enc or any(e not in IR_ENCODINGS for e in enc):
            raise ValueError(f"{where}: an ir camera's encodings are single-channel ({', '.join(IR_ENCODINGS)})")
    mx = _opt_float(c.get("max_rate_hz"))
    stale = float(c["stale_after_s"])
    if mx is not None and mx <= 0:
        raise ValueError(f"{where}: max_rate_hz is positive")
    period = 1.0 / (mx or DEFAULT_CAMERA_MAX_HZ)          # every camera, at its cap or the default
    if period > min(stale / 2, MAX_PERIOD_S):
        raise ValueError(f"{where}: the frame period at max_rate_hz (or the default {DEFAULT_CAMERA_MAX_HZ:g} Hz) "
                         f"must be at most half of stale_after_s (one missing frame is not 'no frame') and at most "
                         f"{MAX_PERIOD_S:g} s (the live bound {LAG_LIMIT_S:g} s + {LIVE_PERIODS:g} periods stays short)")
    if "on_demand" in c:
        raise ValueError(f"{where}: no on_demand: every camera is subscribed on validation (console spec §2.3)")
    text = lambda k: " ".join(str(c.get(k) or "").split())      # noqa: E731
    return Camera(topic=c["topic"], type=c["type"], encodings=tuple(c.get("encodings") or ()),
                  stale_after_s=stale, rate_hz=c.get("rate_hz"),
                  optional=bool(c.get("optional", False)), kind=kind, range_m=rng, unit_m=unit,
                  camera_info=c.get("camera_info") or None, missing_note=text("missing_note"),
                  max_rate_hz=mx, silent_note=text("silent_note"),
                  caption=text("caption"))


def _camera_endpoints(raw: Mapping[str, Any], cams: Sequence[Camera], eps: List[Endpoint]) -> None:
    """A camera row whose topic, or ``camera_info`` topic, has no ``topics`` row declares that topic
    itself (an ``out`` topic of the camera's type, or of the dialect's CameraInfo type, optional as the
    camera is), so one more camera is one more ``cameras`` row. A ``topics`` row a camera names keeps
    its own type, which must fit."""
    info_type = "sensor_msgs/msg/CameraInfo" if raw["dialect"] == d.ROS2 else "sensor_msgs/CameraInfo"
    for c in cams:
        e = next((x for x in eps if x.kind == "topic" and x.name == c.topic), None)
        if e is None:
            eps.append(Endpoint(kind="topic", name=c.topic, type=c.type, optional=c.optional,
                                direction="out", rate_hz=c.rate_hz))
        if c.camera_info is not None:
            i = next((x for x in eps if x.kind == "topic" and x.name == c.camera_info), None)
            if i is None:
                eps.append(Endpoint(kind="topic", name=c.camera_info, type=info_type, optional=c.optional,
                                    direction="out", rate_hz=c.rate_hz))
            elif i.direction != "out" or d.normalize_type(i.type) != "sensor_msgs/CameraInfo":
                raise ValueError(f"{raw['id']}.cameras.{c.topic}: camera_info {c.camera_info} is not a "
                                 f"sensor_msgs/CameraInfo out topic of the profile")


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
    cams = tuple(_camera(raw, c) for c in raw.get("cameras") or [])
    _camera_endpoints(raw, cams, eps)
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
    groups = tuple(Group(id=g["id"], label=str(g.get("label", g["id"])), note=str(g.get("note") or ""))
                   for g in raw.get("groups") or [])
    controls = []
    for c in raw.get("controls") or []:
        controls.append(Control(
            id=c["id"], label=str(c.get("label", c["id"])), kind=c["kind"], name=c["name"],
            type=c["type"], template=c.get("template", {}),
            fields=tuple(_field(f) for f in c.get("fields") or []),
            stop=_ops(c["stop"]) if c.get("stop") else None,
            prerequisites=tuple(c.get("prerequisites") or ()), group=str(c.get("group") or ""),
            busy=_busy(f"{raw['id']}.controls.{c['id']}", c)))
        for f in controls[-1].fields:
            _check_field(f"{raw['id']}.controls.{c['id']}.{f.name}", f)
        if controls[-1].group and controls[-1].group not in {g.id for g in groups}:
            raise ValueError(f"{raw['id']}.controls.{c['id']}: unknown group {controls[-1].group!r}")
    reads = tuple(_read(r) for r in raw.get("reads") or [])
    sensors = tuple(_sensor(raw["id"], r) for r in raw.get("sensors") or [])
    _check_sensors(raw, sensors, eps)
    polled = [r for r in reads if r.poll_s is not None]
    if len(polled) > 1 or any(r.poll_s <= 0 for r in polled):
        raise ValueError(f"{raw['id']}: at most one read is polled, with a positive poll_s")
    return Profile(
        id=raw["id"], name=raw.get("name", raw["id"]), dialect=dia,
        teleop=raw.get("teleop", "none"), sources=raw.get("sources") or {}, boot=raw.get("boot"),
        namespace=nsr, owned_prefixes=tuple(raw.get("owned_prefixes") or ()),
        endpoints=tuple(eps), cameras=cams, stop=_ops(raw.get("stop")),
        watchdog=str(raw.get("watchdog", "")), teleop_base=base, teleop_walk=walk, head=head,
        controls=tuple(controls), reads=reads, groups=groups, sensors=sensors, raw=raw,
        link_budget_mbps=_budget(raw, cams))


def _budget(raw: Mapping[str, Any], cams: Sequence[Camera] = ()) -> Optional[float]:
    """The profile's camera link budget; the cameras, which all start on validation, must fit it together."""
    b = raw.get("link_budget_mbps")
    if b is None:
        return None
    if isinstance(b, bool) or not isinstance(b, (int, float)) or b <= 0:
        raise ValueError(f"{raw['id']}: link_budget_mbps is a positive Mbit/s")
    auto = sum(c.nominal_mbps() for c in cams)
    if auto > b:
        raise ValueError(f"{raw['id']}: the cameras, all started on validation, take about {auto:.0f} Mbit/s, over "
                         f"link_budget_mbps {b:g}")
    return float(b)


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
    read-only ``sensors`` (topics the page subscribes to and shows), its bounded controls (never their stop: the page stops or cancels no bounded control) with their
    groups, wire maps and, for an action control, ``busy`` (how its server handles a goal that
    arrives while one runs, which decides whether the page streams goals), its measured reads (one of them possibly polled), the ``model`` section
    for its 3D model (the embodiment's joint tree from the pinned vendor URDF, the control fields
    that command each joint and the documented joint-position feedback) and, for a teleoperable
    profile, ``teleop``: the base motion contract teleop runs (``teleop_core.page_contract``)."""
    from robot_console import teleop_core     # teleop_core imports this module
    return {
        "id": p.id, "name": p.name, "dialect": p.dialect,
        "namespace": dataclasses.asdict(p.namespace) if p.namespace else None,
        "owned_prefixes": list(p.owned_prefixes),
        "endpoints": [dict(kind=e.kind, name=e.name, type=e.type, optional=e.optional, direction=e.direction,
                           parts=[{"kind": k, "name": n, "type": t} for k, n, t in expand_endpoint(p, e)])
                      for e in p.endpoints],
        "cameras": [dict(topic=c.topic, type=c.type, encodings=list(c.encodings), stale_after_s=c.stale_after_s,
                         optional=c.optional, kind=c.kind, range_m=list(c.range_m) if c.range_m else None,
                         unit_m=c.unit_m, camera_info=c.camera_info, missing_note=c.missing_note,
                         live_bound_s=c.live_bound_s(),
                         throttle_ms=c.throttle_ms(), silent_note=c.silent_note,
                         caption=c.caption)
                    for c in p.cameras],
        "sensors": [dataclasses.asdict(x) for x in p.sensors],
        "link_budget_mbps": p.link_budget_mbps,
        "controls": [{k: v for k, v in dataclasses.asdict(c).items() if k != "stop"} for c in p.controls],
        "groups": [dataclasses.asdict(g) for g in p.groups],
        "reads": [dataclasses.asdict(r) for r in p.reads],
        "model": p.raw.get("model"),
        "teleop": teleop_core.page_contract(p),
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
