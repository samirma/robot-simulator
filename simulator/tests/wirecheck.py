"""Checks of a live wire against the robot's interface file, over rosbridge (the same way a
client sees it). Used by the contract, frame, transform, rate and engine-consistency
checks. The judging functions take what was observed as plain data, so they can be
exercised without a wire."""

from __future__ import annotations

import json
import math
import os
import time

import registry
from rosbridge_client import Rosbridge

ROS2_PARAM_SERVICES = ("describe_parameters", "get_parameter_types", "get_parameters",
                       "list_parameters", "set_parameters", "set_parameters_atomically",
                       "get_type_description")
LOGGER_SERVICES = ("get_logger_levels", "set_logger_levels")
INFRA_NODES = {"/rosbridge_websocket", "/rosapi", "/rosapi_params", "/rosout"}
INFRA_TOPICS = {"/rosout", "/rosout_agg", "/parameter_events", "/client_count",
                "/connected_clients"}


def interface(rid: str) -> dict:
    import yaml

    r = registry.get(rid)
    with open(r.path(r.ros_file)) as fh:
        return yaml.safe_load(fh)


def served(rows):
    return [r for r in rows or [] if not r.get("optional")]


def infra_service(name: str, dialect: str) -> bool:
    """ROS infrastructure as the simulator spec's Terms define it: rosbridge's and rosapi's
    own services and each node's client-library logger, parameter and type-description
    services; and ROS 2 action internals, which no interface file lists (SCHEMA.md)."""
    if name.startswith(("/rosapi/", "/rosbridge_websocket/", "/rosapi_params/")):
        return True
    if "/_action/" in name:
        return True
    leaf = name.rsplit("/", 1)[-1]
    if dialect == "ros1":
        return leaf in ("get_loggers", "set_logger_level")
    return leaf in ROS2_PARAM_SERVICES + LOGGER_SERVICES


def infra_param(name: str, dialect: str) -> bool:
    """ROS infrastructure parameters as the simulator spec's Terms define them: the ROS
    master's own (ROS 1), rosbridge's and rosapi's, and the parameters the client library
    declares on every ROS 2 node -- `use_sim_time`, `start_type_description_service` and
    the QoS overrides of the node's own /rosout and /parameter_events publishers."""
    if dialect == "ros1":
        return name in ("/rosdistro", "/rosversion", "/run_id") or \
            name.startswith(("/roslaunch/", "/rosbridge_websocket/", "/rosapi/"))
    node, _, p = name.partition(":")
    return node in INFRA_NODES or p in ("use_sim_time", "start_type_description_service") \
        or p.startswith("qos_overrides./rosout") or p.startswith("qos_overrides./parameter_events")


def _value(v):
    try:
        return json.loads(v) if isinstance(v, str) else v
    except ValueError:
        return v


def snapshot(port: int, rid: str) -> dict:
    """What the wire presents: nodes, topics with types, services with types, the
    recorded parameters' values (asked for by name), every other parameter name, and the
    action servers (ROS 2)."""
    iface = interface(rid)
    dialect = iface["dialect"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        nodes = set(rb.call("/rosapi/nodes", timeout=30)["nodes"])
        t = rb.call("/rosapi/topics", timeout=30)
        topics = dict(zip(t["topics"], t["types"]))
        services = {}
        for s in rb.call("/rosapi/services", timeout=30)["services"]:
            if infra_service(s, dialect):
                services[s] = None
                continue
            services[s] = rb.call("/rosapi/service_type", {"service": s}, timeout=30).get("type")
        params = {}
        for row in served(iface.get("parameters")):
            key = param_key(row, dialect)
            if not rb.call("/rosapi/has_param", {"name": key}, timeout=30).get("exists"):
                continue
            params[key] = _value(rb.call("/rosapi/get_param", {"name": key}, timeout=30).get("value"))
        actions = []
        if dialect == "ros2":
            actions = rb.call("/rosapi/action_servers", {}, timeout=30).get("action_servers", [])
        names = param_names(rb, params)
        return {"nodes": nodes, "topics": topics, "services": services, "params": params,
                "param_names": [n for n in names if not infra_param(n, dialect)],
                "actions": actions}
    finally:
        rb.close()


#: attempts at listing every parameter name before the snapshot fails
PARAM_NAMES_ATTEMPTS = 3


def param_names(rb, found: dict, attempts: int = PARAM_NAMES_ATTEMPTS) -> list:
    """Every parameter name on the wire (/rosapi/get_param_names). rosapi can give up on a
    slow node and answer an error or a partial list; a list that lacks a parameter the wire
    answered for by name (`found`) is partial. Asked again on either, and after `attempts`
    an error, never an empty list: the contract's and engine consistency's parameter checks
    must not pass on a listing that was never made."""
    last = None
    for k in range(attempts):
        if k:
            time.sleep(2.0)
        try:
            names = list(rb.call("/rosapi/get_param_names", timeout=40).get("names") or [])
        except Exception as exc:
            last = f"failed: {exc}"
            continue
        # a namespace parameter (a dict) is listed as its leaves
        lacking = sorted(k for k in found if k not in names and not any(
            n.startswith((k + "/", k + ".")) for n in names))
        if not names or lacking:
            last = f"lacks {lacking[:5] or 'every name'}"
            continue
        return names
    raise RuntimeError(f"/rosapi/get_param_names {last} ({attempts} attempts)")


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def param_key(row: dict, dialect: str) -> str:
    return row["name"] if dialect == "ros1" else f"{row['node']}:{row['name']}"


def contract_problems(rid: str, snap: dict) -> list:
    """Every difference between the live wire and the recorded interface."""
    iface = interface(rid)
    dialect = iface["dialect"]
    robot = registry.get(rid)
    out = []
    want_nodes = {n["name"] for n in served(iface["nodes"])}
    optional_nodes = {n["name"] for n in iface["nodes"] if n.get("optional")}
    for n in sorted(want_nodes - snap["nodes"]):
        out.append(f"missing node {n}")
    for n in sorted(snap["nodes"] - want_nodes - INFRA_NODES):
        out.append(f"extra node {n}" + (" (optional row served)" if n in optional_nodes else ""))
    want_topics = {t["name"]: t["type"] for t in served(iface["topics"])}
    for name, typ in want_topics.items():
        if name not in snap["topics"]:
            out.append(f"missing topic {name}")
        elif snap["topics"][name] != typ:
            out.append(f"topic {name} is {snap['topics'][name]}, recorded {typ}")
    for name in sorted(set(snap["topics"]) - set(want_topics) - INFRA_TOPICS):
        if "/_action/" in name:
            continue   # ROS 2 action internals
        out.append(f"extra topic {name}")
    want_srv = {s["name"]: s["type"] for s in served(iface.get("services"))}
    for name, typ in want_srv.items():
        if name not in snap["services"]:
            out.append(f"missing service {name}")
        elif snap["services"][name] not in (typ, None):
            out.append(f"service {name} is {snap['services'][name]}, recorded {typ}")
    for name in sorted(set(snap["services"]) - set(want_srv)):
        if not infra_service(name, dialect):
            out.append(f"extra service {name}")
    if dialect == "ros2":
        want_actions = {a["name"] for a in served(iface.get("actions"))}
        for a in sorted(want_actions - set(snap["actions"])):
            out.append(f"missing action {a}")
        for a in sorted(set(snap["actions"]) - want_actions):
            out.append(f"extra action {a}")
    recorded = [param_key(r, dialect) for r in served(iface.get("parameters"))]
    for n in snap.get("param_names", []):
        if not any(n == k or n.startswith(k + "/") or n.startswith(k + ".") for k in recorded):
            out.append(f"extra parameter {n}")
    for row in served(iface.get("parameters")):
        key = param_key(row, dialect)
        if key not in snap["params"]:
            out.append(f"missing parameter {key}")
            continue
        v = row.get("value")
        if isinstance(v, dict) and "generated" in v:
            continue
        if isinstance(v, dict) and "file" in v and len(v) <= 2:
            v = (robot.folder_path / v["file"]).read_text()
        got = snap["params"][key]
        if not _same(got, v):
            out.append(f"parameter {key} = {str(got)[:80]!r}, recorded {str(v)[:80]!r}")
    return out


# ---------------------------------------------------------------- message fields

_BOOL = {"bool", "boolean"}
_INT = {"byte", "char", "octet", "int8", "uint8", "int16", "uint16", "int32", "uint32",
        "int64", "uint64"}
_FLOAT = {"float32", "float64", "float", "double"}
#: arrays of these rosbridge sends as one base64 string
_BINARY = {"uint8", "char", "octet", "byte"}


def norm_type(t: str) -> str:
    """`pkg/msg/Type`, `pkg/srv/Type_Response`... as rosapi's typedefs name them:
    `pkg/Type` (ROS 2 rosapi drops the interface kind)."""
    parts = t.split("/")
    return f"{parts[0]}/{parts[-1]}" if len(parts) == 3 else t


def typedef_problems(value, root: str, typedefs: list, where: str = "") -> list:
    """Every way a message (as rosbridge delivers it) differs from its declared type, field
    for field and through every nested message: a missing or extra field, a value of the
    wrong kind (bool, integer, number, string, nested message, ROS 1 time), an array that
    is not one or a fixed-size array of another length. `typedefs`: rosapi's
    message_details (or service_response_details, action_result_details) answer, `root`
    the type of `value` among them."""
    defs = {norm_type(d["type"]): d for d in typedefs}
    out = []

    def scalar(v, t, at):
        base = t.split("<=")[0]
        if base in _BOOL:
            ok = isinstance(v, bool)
        elif base in _INT:
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif base in _FLOAT:
            ok = v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
        elif base in ("string", "wstring"):
            ok = isinstance(v, str)
        elif base in ("time", "duration"):           # ROS 1 builtins
            ok = isinstance(v, dict) and set(v) == {"secs", "nsecs"} and \
                all(isinstance(x, int) for x in v.values())
        elif norm_type(base) in defs:
            check(v, norm_type(base), at)
            return
        else:
            out.append(f"{at}: type {t} has no definition in rosapi's answer")
            return
        if not ok:
            out.append(f"{at}: {str(v)[:40]!r} is no {t}")

    def check(v, typ, at):
        d = defs.get(norm_type(typ))
        if d is None:
            out.append(f"{at or typ}: no definition of {typ}")
            return
        if not isinstance(v, dict):
            out.append(f"{at or typ}: {str(v)[:40]!r} is no {typ} message")
            return
        names = list(d["fieldnames"])
        if set(v) != set(names):
            out.append(f"{at or typ}: fields {sorted(v)}, declared {sorted(names)}")
        for name, t, n in zip(names, d["fieldtypes"], d["fieldarraylen"]):
            if name not in v:
                continue
            x, here = v[name], f"{at}.{name}" if at else name
            if n < 0:
                scalar(x, t, here)
                continue
            if t.split("<=")[0] in _BINARY and isinstance(x, str):
                import base64

                try:
                    size = len(base64.b64decode(x, validate=True))
                except ValueError:
                    out.append(f"{here}: not base64 {t}[]")
                    continue
            elif isinstance(x, list):
                size = len(x)
                for i, e in enumerate(x):
                    scalar(e, t, f"{here}[{i}]")
            else:
                out.append(f"{here}: {str(x)[:40]!r} is no {t}[] array")
                continue
            if n > 0 and size != n:
                out.append(f"{here}: {size} elements, declared {t}[{n}]")

    check(value, root, where)
    return out


def type_details(rb, kind: str, typ: str) -> list:
    """rosapi's typedefs of a message type (`kind` "message"), a service's response
    ("service_response") or an action's result ("action_result"), asked once per wire
    connection and type."""
    _cache = rb.__dict__.setdefault("_rsim_typedefs", {})
    key = (kind, typ)
    if key not in _cache:
        service = {"message": "/rosapi/message_details",
                   "service_response": "/rosapi/service_response_details",
                   "action_result": "/rosapi/action_result_details"}[kind]
        _cache[key] = rb.call(service, {"type": typ}, timeout=20)["typedefs"]
    return _cache[key]


def typed_problems(rb, kind: str, typ: str, value, where: str) -> list:
    """`typedef_problems` of a message, service response or action result received over
    `rb`, against its declared type as the wire's own rosapi defines it."""
    defs = type_details(rb, kind, typ)
    if not defs:
        return [f"{where}: rosapi has no definition of {typ}"]
    return typedef_problems(value, defs[0]["type"], defs, where)


# ---------------------------------------------------------------- frames and transforms


def sample(port: int, topics, timeout: float = 15.0) -> dict:
    """One message of each topic, over rosbridge (throttled, so big images get through)."""
    rb = Rosbridge("127.0.0.1", port)
    got = {}
    try:
        for t in topics:
            rb.subscribe(t, lambda m, t=t: got.setdefault(t, m["msg"]), throttle_rate=500,
                         queue_length=1)
        deadline = time.monotonic() + timeout
        while len(got) < len(topics) and time.monotonic() < deadline:
            time.sleep(0.2)
    finally:
        rb.close()
    return dict(got)


def header_frames(port: int, rid: str) -> dict:
    """{topic: [header frame_id, child_frame_id]} of one message of each served periodic
    output topic (None where the type has neither)."""
    got = sample(port, list(periodic_topics(rid)))
    return {t: [(m.get("header") or {}).get("frame_id"), m.get("child_frame_id")]
            for t, m in got.items()}


def rpy_quat(roll: float, pitch: float, yaw: float) -> list:
    """[x, y, z, w] of URDF fixed-axis roll, pitch, yaw (tf2's setRPY)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]


def _pose_diff(xyz_a, q_a, xyz_b, q_b) -> float:
    """Largest component difference of two transforms (q and -q are one rotation)."""
    dq = min(max(abs(a - b) for a, b in zip(q_a, q_b)),
             max(abs(a + b) for a, b in zip(q_a, q_b)))
    return max(max(abs(a - b) for a, b in zip(xyz_a, xyz_b)), dq)


def tf_window(iface: dict) -> float:
    """How long to watch the transform tree: five periods of its slowest periodic edge."""
    rates = [float(r["rate"]) for r in served(iface.get("tf"))
             if isinstance(r.get("rate"), (int, float)) and not isinstance(r["rate"], bool)
             and r["rate"] > 0]
    return max(3.0, 5.0 / min(rates)) if rates else 3.0


def tf_edges(port: int, seconds: float) -> dict:
    """The transforms the wire sends on /tf and /tf_static (latched or transient-local, so
    sent before the subscription too) over `seconds`: {(parent, child): {"topic", "xyz",
    "quat" [x, y, z, w], "changed"}}, `topic` "both" when an edge came on both and
    `changed` when its transform did not stay the same."""
    rb = Rosbridge("127.0.0.1", port)
    edges = {}
    try:
        for topic in ("/tf", "/tf_static"):
            def cb(m, topic=topic):
                for t in (m.get("msg") or {}).get("transforms") or []:
                    tr = t["transform"]
                    xyz = [tr["translation"][k] for k in "xyz"]
                    quat = [tr["rotation"][k] for k in "xyzw"]
                    key = (t["header"]["frame_id"], t["child_frame_id"])
                    old = edges.get(key)
                    edges[key] = {
                        "topic": topic if old is None or old["topic"] == topic else "both",
                        "xyz": xyz, "quat": quat,
                        "changed": old is not None and (
                            old["changed"] or _pose_diff(old["xyz"], old["quat"], xyz, quat) > 1e-9)}
            rb.subscribe(topic, cb, queue_length=1000)
        time.sleep(seconds)
    finally:
        rb.close()
    return edges


def tf_problems(iface: dict, edges: dict) -> list:
    """Every difference between the observed transform tree (`tf_edges`) and the recorded
    `tf` rows: a missing or an extra edge; an edge on the wrong topic (a fixed transform
    recorded as non-periodic is latched on /tf_static, every other edge is sent on /tf);
    a fixed transform that changes or does not carry its recorded xyz/rpy. A moving edge
    recorded as non-periodic (sent only when its input arrives) may be absent."""
    out = []
    rows = {(r["parent"], r["child"]): r for r in served(iface.get("tf"))}
    for key, r in rows.items():
        edge = f"{key[0]} -> {key[1]}"
        want = "/tf_static" if r.get("static") and r.get("rate") == "non_periodic" else "/tf"
        got = edges.get(key)
        if got is None:
            if r.get("static") or r.get("rate") != "non_periodic":
                out.append(f"missing transform {edge}")
            continue
        if got["topic"] != want:
            out.append(f"transform {edge} on {got['topic']}, recorded on {want}")
        if not r.get("static"):
            continue
        if got["changed"]:
            out.append(f"fixed transform {edge} changed while observed")
        if "xyz" in r or "rpy" in r:
            xyz = [float(v) for v in r.get("xyz", [0.0, 0.0, 0.0])]
            q = rpy_quat(*[float(v) for v in r.get("rpy", [0.0, 0.0, 0.0])])
            if _pose_diff(xyz, q, got["xyz"], got["quat"]) > 1e-6:
                out.append(f"fixed transform {edge} is xyz {got['xyz']} quat {got['quat']}, "
                           f"recorded xyz {xyz} rpy {r.get('rpy', [0.0, 0.0, 0.0])}")
    for key in sorted(set(edges) - set(rows)):
        out.append(f"extra transform {key[0]} -> {key[1]}")
    return out


# ---------------------------------------------------------------- rates and stamps


def _rate(row: dict):
    r = row.get("rate")
    if isinstance(r, bool) or not isinstance(r, (int, float)) or not r > 0:
        return None
    return float(r)


def periodic_topics(rid: str) -> dict:
    """{topic: recorded rate} of the served periodic output topics."""
    out = {}
    for t in served(interface(rid)["topics"]):
        if t.get("direction") == "out" and _rate(t) is not None:
            out[t["name"]] = _rate(t)
    return out


def rate_record_problems(iface: dict) -> list:
    """Served output topics whose recorded rate is neither a positive number nor
    `non_periodic`: such a topic fails the rate check, it is never skipped (spec §5)."""
    return [f"{t['name']}: no recorded rate ({t.get('rate')!r})"
            for t in served(iface.get("topics"))
            if t.get("direction") == "out" and t.get("rate") != "non_periodic"
            and _rate(t) is None]


def multi_publisher(iface: dict) -> set:
    """Output topics several recorded nodes publish (their stamps interleave)."""
    return {t["name"] for t in served(iface.get("topics"))
            if t.get("direction") == "out" and len(t.get("nodes") or []) > 1}


def measure(port: int, topics, seconds: float) -> dict:
    """Arrival times (rosbridge's own receive stamps, from cbor-raw) per topic."""
    rb = Rosbridge("127.0.0.1", port)
    times = {t: [] for t in topics}
    try:
        for t in topics:
            def cb(m, t=t):
                msg = m.get("msg") or {}
                if "secs" in msg:
                    times[t].append(msg["secs"] + msg["nsecs"] * 1e-9)
                else:
                    times[t].append(time.time())
            rb.subscribe(t, cb, compression="cbor-raw", queue_length=100)
        time.sleep(1.0)            # subscriptions settle
        for t in topics:
            times[t].clear()
        t0 = time.time()
        time.sleep(seconds)
        window = time.time() - t0
        snap = {t: list(v) for t, v in times.items()}
    finally:
        rb.close()
    return {"window": window, "times": snap}


# new DDS participants joining the graph (the probes themselves) briefly stall the
# publishers' writers; measure only once discovery has settled
PROBE_WARMUP_S = float(os.environ.get("RSIM_PROBE_WARMUP", "5"))


def measure_on_graph(container: str, rid: str, topics, seconds: float) -> dict:
    """Arrival times of the topics on the wire's own ROS graph, from inside its container
    (rosbridge, a Python server, cannot carry every camera stream and point cloud at once;
    this measures what the robot publishes, as a native subscriber sees it), and the
    header stamps of the messages that have one, in arrival order."""
    import subprocess

    iface = interface(rid)
    distro = iface["ros_distribution"]
    if iface["dialect"] == "ros1":
        probe = "/opt/rsim/simulator/tests/rate_probe_ros1.py"
        args = list(topics)
    else:
        types = {t["name"]: t["type"] for t in iface["topics"]}
        probe = "/opt/rsim/simulator/tests/rate_probe_ros2.py"
        args = [f"{t}={types[t]}" for t in topics]
    # one probe process per topic, all at once: a single subscriber process serving every
    # stream of a robot would add its own scheduling jitter to the gaps it measures
    procs = []
    for a in args:
        cmd = (f"source /opt/ros/{distro}/setup.bash; for f in /opt/rsim_ws/install/setup.bash "
               f"/opt/rsim_ws/devel/setup.bash /home/ubuntu/ros_ws/devel/setup.bash; do "
               f"[ -f $f ] && source $f; done; "
               f"RSIM_PROBE_WARMUP={PROBE_WARMUP_S} python3 {probe} {seconds} {a}")
        procs.append(subprocess.Popen(["docker", "exec", container, "bash", "-c", cmd],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    merged = {"window": seconds, "times": {}, "stamps": {}}
    for p in procs:
        out, err = p.communicate(timeout=seconds + 90)
        lines = [l for l in out.splitlines() if l.startswith("{")]
        if not lines:
            raise RuntimeError(f"rate probe failed: {err[-2000:]}")
        got = json.loads(lines[-1])
        merged["window"] = min(merged["window"], got["window"]) if merged["times"] else got["window"]
        merged["times"].update(got["times"])
        merged["stamps"].update(got.get("stamps", {}))
    return merged


def tf_rates(iface: dict) -> dict:
    """{"<parent> -> <child>": recorded rate} of the served transforms the record gives a
    rate of their own (sent on /tf, each by its publisher at that rate)."""
    return {f"{r['parent']} -> {r['child']}": _rate(r) for r in served(iface.get("tf"))
            if _rate(r) is not None}


def measure_tf_on_graph(container: str, rid: str, seconds: float) -> dict:
    """Arrival times and header stamps of each transform on /tf, on the wire's own graph
    (`tf_rate_probe.py` inside its container): {"window", "times", "stamps"} keyed
    "<parent> -> <child>"."""
    import subprocess

    iface = interface(rid)
    distro = iface["ros_distribution"]
    cmd = (f"source /opt/ros/{distro}/setup.bash; for f in /opt/rsim_ws/install/setup.bash "
           f"/opt/rsim_ws/devel/setup.bash /home/ubuntu/ros_ws/devel/setup.bash; do "
           f"[ -f $f ] && source $f; done; RSIM_PROBE_WARMUP={PROBE_WARMUP_S} python3 "
           f"/opt/rsim/simulator/tests/tf_rate_probe.py {iface['dialect']} {seconds}")
    res = subprocess.run(["docker", "exec", container, "bash", "-c", cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         timeout=seconds + 90)
    lines = [l for l in res.stdout.splitlines() if l.startswith("{")]
    if not lines:
        raise RuntimeError(f"tf rate probe failed: {res.stderr[-2000:]}")
    return json.loads(lines[-1])


def tf_rate_problems(measured: dict, iface: dict) -> list:
    """Spec §5 acceptance bounds for each transform with a recorded rate of its own: on
    /tf at its rate within +-10%, no gap over three of its periods, its header stamps the
    times it stands for (`rate_problems`), over a window of at least five of its periods;
    a transform with a recorded rate that never comes fails."""
    rates = tf_rates(iface)
    return rate_problems(measured, rates)


def rate_problems(measured: dict, rates: dict, multi=()) -> list:
    """The acceptance bounds of spec §5 on what `measure_on_graph` saw: over a window of at
    least five periods, the arrival rate within ±10% of the recorded rate and no gap
    between consecutive arrivals over three recorded periods. Where the type has a header
    its stamps are the samples' acquisition times in the wall clock (spec §3 Timing):
    each within three recorded periods of its arrival, and one publisher's never going
    back (`multi`: topics several nodes publish, whose stamps interleave)."""
    out = []
    window = measured["window"]
    for topic, rate in rates.items():
        ts = measured["times"].get(topic, [])
        period = 1.0 / rate
        if window < 5 * period:
            out.append(f"{topic}: window {window:.1f} s is shorter than five periods")
            continue
        if len(ts) < 2:
            out.append(f"{topic}: {len(ts)} messages (recorded {rate} Hz)")
            continue
        arrivals = sorted(ts)
        got = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
        if not 0.9 * rate <= got <= 1.1 * rate:
            out.append(f"{topic}: {got:.2f} Hz, recorded {rate} Hz")
        gap = max(b - a for a, b in zip(arrivals, arrivals[1:]))
        if gap > 3 * period:
            out.append(f"{topic}: gap {gap:.3f} s > 3 periods ({3 * period:.3f} s)")
        stamps = measured.get("stamps", {}).get(topic) or []
        if not stamps:
            continue
        lag = max((a - s for a, s in zip(ts, stamps)), key=abs)
        if abs(lag) > 3 * period:
            out.append(f"{topic}: a header stamp {lag:+.3f} s from its arrival, over 3 periods "
                       f"({3 * period:.3f} s): not the acquisition time in the wall clock")
        back = min((b - a for a, b in zip(stamps, stamps[1:])), default=0.0)
        if topic not in multi and back < 0:
            out.append(f"{topic}: header stamps go back {-back:.3f} s")
    return out
