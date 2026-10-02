"""Checks of a live wire against the robot's interface file, over rosbridge (the same way a
client sees it). Used by the contract, rate and engine-consistency checks."""

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
INFRA_NODES = {"/rosbridge_websocket", "/rosapi", "/rosapi_params", "/rosout"}
INFRA_TOPICS = {"/rosout", "/rosout_agg", "/parameter_events", "/client_count",
                "/connected_clients"}


def interface(rid: str) -> dict:
    import yaml

    r = registry.get(rid)
    with open(r.path(r.ros_file)) as fh:
        return yaml.safe_load(fh)


#: Differences between the live wire and the interface file that come from the unchanged
#: stock code the recorded boot runs, i.e. from the interface file, not the simulator.
#: Reported (tests print them) and raised with robots_specs; not counted as failures.
INTERFACE_FILE_GAPS = {
    "ainex": {
        "extra topic /tf": "the stock imu_complementary_filter (imu_tools 1.2.7) creates a tf "
                           "broadcaster, which advertises /tf, even with publish_tf false (it "
                           "never sends); ros.yml records no /tf",
    },
    "rosmaster_x3_plus": {
        f"extra parameter /imu_filter_madgwick/{p}":
            "the stock imu_filter_madgwick's dynamic_reconfigure server writes its whole "
            "config (gain, zeta, magnetometer bias) to the parameter server at start-up; "
            "ros.yml records only the launch file's parameters"
        for p in ("gain", "zeta", "mag_bias_x", "mag_bias_y", "mag_bias_z")
    },
    "so101": {
        "missing parameter /robot_state_publisher:qos_overrides./joint_states.subscription."
        "durability": "robot_state_publisher 3.3.4 (the pinned release) declares only "
                      "history, depth and reliability overrides for its /joint_states "
                      "subscription (QosOverridingOptions::with_default_policies)",
    },
}


def served(rows):
    return [r for r in rows or [] if not r.get("optional")]


def infra_service(name: str, dialect: str) -> bool:
    if name.startswith("/rosapi/") or name.startswith("/rosbridge_websocket/") or \
            name.startswith("/rosapi_params/"):
        return True
    if "/_action/" in name:
        return True
    leaf = name.rsplit("/", 1)[-1]
    if dialect == "ros1":
        return leaf in ("get_loggers", "set_logger_level")
    # rclcpp's parameter, logger and type-description services, and rclcpp_lifecycle's own
    # state services, which every lifecycle node (the ros2_control controllers) carries
    return leaf in ROS2_PARAM_SERVICES + LOGGER_SERVICES + LIFECYCLE_SERVICES


LOGGER_SERVICES = ("get_logger_levels", "set_logger_levels")
LIFECYCLE_SERVICES = ("change_state", "get_state", "get_available_states",
                      "get_available_transitions", "get_transition_graph")


def infra_param(name: str, dialect: str) -> bool:
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
        try:   # every parameter name; Humble's rosapi can give up on a slow node
            names = rb.call("/rosapi/get_param_names", timeout=40).get("names", [])
        except Exception:
            names = []
        return {"nodes": nodes, "topics": topics, "services": services, "params": params,
                "param_names": [n for n in names if not infra_param(n, dialect)],
                "actions": actions}
    finally:
        rb.close()


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
        if "/_action/" in name or (dialect == "ros2" and name.endswith("/transition_event")):
            continue   # action internals; rclcpp_lifecycle's own state-change topic
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


def periodic_topics(rid: str) -> dict:
    """{topic: recorded rate} of the served periodic output topics."""
    out = {}
    for t in served(interface(rid)["topics"]):
        if t.get("direction") == "out" and isinstance(t.get("rate"), (int, float)):
            out[t["name"]] = float(t["rate"])
    return out


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
    this measures what the robot publishes, as a native subscriber sees it)."""
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
        cmd = (f"source /opt/ros/{distro}/setup.bash; for f in /opt/rsim_ws/devel/setup.bash "
               f"/home/ubuntu/ros_ws/devel/setup.bash; do [ -f $f ] && source $f; done; "
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


def rate_problems(measured: dict, rates: dict) -> list:
    out = []
    window = measured["window"]
    for topic, rate in rates.items():
        ts = sorted(measured["times"].get(topic, []))
        period = 1.0 / rate
        if window < 5 * period:
            out.append(f"{topic}: window {window:.1f} s is shorter than five periods")
            continue
        if len(ts) < 2:
            out.append(f"{topic}: {len(ts)} messages (recorded {rate} Hz)")
            continue
        got = (len(ts) - 1) / (ts[-1] - ts[0])
        if not 0.9 * rate <= got <= 1.1 * rate:
            out.append(f"{topic}: {got:.2f} Hz, recorded {rate} Hz")
        # Gaps between samples: by their header stamps (acquisition times) where the type
        # has a header, else by arrival.
        st = measured.get("stamps", {}).get(topic) or []
        seq = st if len(st) >= 2 else ts
        gap = max(b - a for a, b in zip(seq, seq[1:]))
        if gap > 3 * period:
            out.append(f"{topic}: gap {gap:.3f} s > 3 periods ({3 * period:.3f} s)")
    return out
