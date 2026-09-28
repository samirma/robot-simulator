#!/usr/bin/env python
"""The two mobile manipulators' contract modules against their ROS files, then on the wire.

    molmospaces/.venv/bin/python shared/contracts/test_mobile_contracts.py

Run by `test_fleet.py` as well (spec §5, "Contracts"). For the ROSMASTER X3 PLUS
(`ros_surfaces/rosmaster_x3_plus.py`) and the myAGV + myCobot 280
(`ros_surfaces/myagv_mycobot280.py`, which `extends` the myAGV's): every topic row
(name, type, direction, node, rate), every service, every parameter, compared with the
robot's ROS file; then the surface attached to a server with no MuJoCo model (a planar
base that goes where it is told), so that what `rosapi` reports -- names, types,
publishing and subscribing nodes, services with their nodes and types, parameters -- is
compared with the file, every message it publishes is held to its declared type field for
field, and the documented behaviour is exercised: `/cmd_vel` clamped and held with no
timeout, the stop command stopping the base, the X3 PLUS's servo commands echoed and
reported, the composite's navigation goal driving the base and a cancel ending it.
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SHARED = Path(__file__).resolve().parents[1]
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

PORT = 9407


class FakeBase:
    """A planar base that goes exactly where it is told."""

    def __init__(self) -> None:
        self.xyz = np.zeros(3)

    @property
    def pose(self):
        x, y, yaw = self.xyz
        out = np.eye(4)
        out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        out[0, 3], out[1, 3] = x, y
        return out

    @property
    def ctrl(self):
        return self.xyz

    @ctrl.setter
    def ctrl(self, value) -> None:
        self.xyz = np.asarray(value, dtype=float).copy()


class FakeData:
    time = 0.0


def _node(name: str) -> str:
    return str(name).lstrip("/")


def _rate(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else str(value)


def _rows(spec: dict) -> list[tuple]:
    return sorted((t["name"], t["type"], t["direction"], _node(t["node"]), _rate(t["rate_hz"]))
                  for t in spec["topics"])


def _module_rows(module) -> list[tuple]:
    return sorted((n, ty, d, node, _rate(r)) for n, ty, d, node, r in module.TOPICS)


def expand(pattern: str) -> list[str]:
    """A ROS file parameter row as name patterns: `{a,b}` alternatives, `*` wildcards."""
    parts = []
    for chunk in pattern.replace("}", "{").split("{"):
        parts.append(chunk)
    out = [""]
    for i, part in enumerate(parts):
        options = part.split(",") if i % 2 else [part]
        out = [o + p for o in out for p in options]
    return out


def covers(row: str, name: str) -> bool:
    return any(name == p or name.startswith(p.rstrip("/") + "/") or fnmatch.fnmatchcase(name, p)
               for p in expand(row))


def _fields_match(schemas, mtype: str, msg: dict) -> bool:
    declared = [f[0] for f in schemas.fields_of(mtype) or []]
    return sorted(declared) == sorted(msg)


def _wire(check, label: str, spec_topics: list[dict], spec_services: list[dict],
          param_names: set[str], ns: str, attach, extra) -> None:
    """Attach, then compare rosapi's answers and the published messages with the file."""
    from contracts import message_schemas as schemas
    from contracts.namespace import ns_topic
    from contracts.test_transport import Client
    from ros_surfaces import RobotFleet

    fleet = RobotFleet(port=PORT, default_hz=10.0)
    base = FakeBase()
    fleet.attach(ns, attach, base=base, model=None, camera=None)
    fleet.start()
    data = FakeData()
    hz = fleet.rate_hz

    def run_for(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            data.time += 1.0 / hz
            fleet(data)
            time.sleep(1.0 / hz)

    wire_type = {}
    for t in spec_topics:
        if t["direction"] == "out":
            wire_type.setdefault(t["name"], t["type"])
    for t in spec_topics:
        wire_type.setdefault(t["name"], t["type"])
    composed = {(ns_topic(ns, n), ty) for n, ty in wire_type.items()}
    try:
        client = Client(PORT)
        ok, v = client.call("/rosapi/topics")
        wire = {(t, ty) for t, ty in zip(v.get("topics", []), v.get("types", []))}
        check(f"{label}: rosapi lists exactly the file's topics, typed", wire == composed,
              str(sorted(wire ^ composed))[:400])
        bad = []
        for topic in sorted(wire_type):
            for role, direction in (("publishers", "out"), ("subscribers", "in")):
                want = sorted({f"/{ns}/{_node(t['node'])}" for t in spec_topics
                               if t["name"] == topic and t["direction"] == direction})
                ok, v = client.call(f"/rosapi/{role}", {"topic": ns_topic(ns, topic)})
                if v.get(role, []) != want:
                    bad.append(f"{topic} {role} {v.get(role)} vs {want}")
        check(f"{label}: each topic's publishers and subscribers are its file's nodes",
              not bad, "; ".join(bad[:5]))
        ok, v = client.call("/rosapi/services")
        mine = {s for s in v.get("services", []) if s.startswith(f"/{ns}/")}
        check(f"{label}: the file's services, and no others",
              mine == {ns_topic(ns, s["name"]) for s in spec_services},
              str(sorted(mine ^ {ns_topic(ns, s["name"]) for s in spec_services}))[:300])
        bad = []
        for s in spec_services:
            ok, n = client.call("/rosapi/service_node", {"service": ns_topic(ns, s["name"])})
            ok2, t = client.call("/rosapi/service_type", {"service": ns_topic(ns, s["name"])})
            if n.get("node") != f"/{ns}/{_node(s['node'])}" or t.get("type") != s["type"]:
                bad.append(f"{s['name']}: {n} {t}")
        check(f"{label}: each service's node and type", not bad, "; ".join(bad[:5]))
        ok, v = client.call("/rosapi/get_param_names")
        got = set(v.get("names", []))
        want = {ns_topic(ns, p) for p in param_names}
        check(f"{label}: the file's parameters, namespaced", got == want,
              str(sorted(got ^ want))[:300])

        for topic in wire_type:
            if any(t["name"] == topic and t["direction"] == "out" for t in spec_topics):
                client.send({"op": "subscribe", "topic": ns_topic(ns, topic)})
        client.sync()
        extra(client, run_for, base)
        seen: dict[str, dict] = {}
        for msg in client.inbox:
            if msg.get("op") == "publish":
                seen.setdefault(msg["topic"], msg["msg"])
        bad = [t for t in wire_type if ns_topic(ns, t) in seen
               and not _fields_match(schemas, wire_type[t], seen[ns_topic(ns, t)])]
        check(f"{label}: every published message matches its type field for field", not bad,
              str(bad))
        client.close()
        return seen
    finally:
        fleet(None)


def run_x3(check) -> None:
    import robots_spec
    from contracts import message_schemas as schemas
    from ros_surfaces import rosmaster_x3_plus as m

    print("ROSMASTER X3 PLUS contract vs robots_specs/rosmaster_x3_plus/ros.yml")
    spec = yaml.safe_load(robots_spec.ros_path("rosmaster_x3_plus").read_text())
    check("x3: every topic row: name, type, direction, node, rate",
          _rows(spec) == _module_rows(m), str(set(_rows(spec)) ^ set(_module_rows(m)))[:400])
    want_srv = sorted((s["name"], s["type"], _node(s["node"])) for s in spec["services"])
    check("x3: every service: name, type, node", want_srv == sorted(m.SERVICES),
          str(set(want_srv) ^ set(m.SERVICES))[:300])
    check("x3: no action server", list(spec.get("actions") or []) == list(m.ACTIONS))
    names = {p["name"] for p in spec["parameters"]}
    check("x3: every parameter", names == set(m.PARAMETERS) | {m.PARAM_ROBOT_DESCRIPTION},
          str(sorted(names ^ (set(m.PARAMETERS) | {m.PARAM_ROBOT_DESCRIPTION}))))
    wrong = []
    for p in spec["parameters"]:
        value = p.get("default")
        if p["name"] == m.PARAM_ROBOT_DESCRIPTION or p["name"] not in m.PARAMETERS:
            continue
        mine = m.PARAMETERS[p["name"]]
        if isinstance(mine, list):
            continue  # the EKF's 15-entry configs, which the file summarises by name
        if isinstance(value, str) and " (" in value:
            value = value.split(" (")[0]
        if isinstance(mine, str) and isinstance(value, int) and not isinstance(value, bool):
            # YAML reads the file's 0x2bc5 as a number; roslaunch keeps it a string.
            value = hex(value) if mine.startswith("0x") else str(value)
        if mine != value and not (isinstance(mine, float) and isinstance(value, (int, float))
                                  and float(value) == mine):
            wrong.append(f"{p['name']}: {mine!r} vs {value!r}")
    check("x3: every parameter's value is the file's", not wrong, "; ".join(wrong))
    check("x3: the joints are the file's, in its order", list(spec["joints"]) == list(m.JOINTS))
    types = {t["type"] for t in spec["topics"]} | {s["type"] for s in spec["services"]}
    unknown = sorted(t for t in types if schemas.canonical(t) not in schemas.known_types())
    check("x3: the schema table answers for every type it serves", not unknown, str(unknown))
    check("x3: the stop command is a zero Twist on /cmd_vel",
          "/cmd_vel" in spec["stop_command"] and all(
              v == 0.0 for part in m.STOP_COMMAND.values() for v in part.values()))

    ns = "rosmaster_x3_plus"

    def extra(client, run_for, base):
        client.send({"op": "publish", "topic": f"/{ns}/cmd_vel",
                     "msg": {"linear": {"x": 2.0, "y": -2.0, "z": 0.0},
                             "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}})
        client.send({"op": "publish", "topic": f"/{ns}/TargetAngle",
                     "msg": {"id": 0, "run_time": 500, "angle": 0.0,
                             "joints": [100, 120, 30, 60, 90, 90]}})
        client.sync()
        run_for(1.5)
        client.quiet(lambda msg: False, window=0.3)
        vel = [x["msg"] for x in client.inbox if x.get("topic") == f"/{ns}/vel_raw"]
        lx, ly, lz = m.CMD_VEL_LIMITS
        sx, sy = m.LINEAR_SCALE
        last = vel[-1] if vel else {}
        check("x3: /cmd_vel is carried out within the board's range, held with no timeout",
              last and abs(last["linear"]["x"] * sx - lx) < 1e-6
              and abs(last["linear"]["y"] * sy + ly) < 1e-6
              and abs(last["angular"]["z"]) < 1e-6, str(last))
        client.send({"op": "publish", "topic": f"/{ns}/cmd_vel",
                     "msg": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                             "angular": {"x": 0.0, "y": 0.0, "z": 9.0}}})
        client.sync()
        run_for(0.6)
        client.quiet(lambda msg: False, window=0.2)
        vel = [x["msg"] for x in client.inbox if x.get("topic") == f"/{ns}/vel_raw"]
        check("x3: ...and its yaw rate too", vel and abs(vel[-1]["angular"]["z"] - lz) < 1e-6,
              str(vel[-1] if vel else None))
        echoes = [x["msg"] for x in client.inbox if x.get("topic") == f"/{ns}/ArmAngleUpdate"]
        check("x3: a servo command is echoed twice, with the joints it replaced",
              len(echoes) == 2 and all(list(e["joints"]) == list(m.STARTUP_JOINTS_DEG)
                                       for e in echoes), str(echoes))
        js = [x["msg"] for x in client.inbox if x.get("topic") == f"/{ns}/joint_states"]
        want = m.joint_positions([100, 120, 30, 60, 90, 90])
        check("x3: /joint_states reports the commanded angles, as the driver maps them",
              js and np.allclose(js[-1]["position"], want) and js[-1]["name"] == list(m.JOINTS),
              str(js[-1] if js else None))
        ok, v = client.call(f"/{ns}/CurrentAngle", {"apply": ""})
        check("x3: /CurrentAngle reads the servos back",
              ok and np.allclose(v.get("angles"), [100, 120, 30, 60, 90, 90], atol=1.0), str(v))
        ok, v = client.call(f"/{ns}/driver_node/set_parameters",
                            {"config": {"doubles": [{"name": "Kp", "value": 2.0}]}})
        check("x3: dynamic_reconfigure answers with the updated config",
              ok and {"name": "Kp", "value": 2.0} in v.get("config", {}).get("doubles", []),
              str(v)[:200])
        client.send({"op": "publish", "topic": f"/{ns}/cmd_vel", "msg": m.STOP_COMMAND})
        client.sync()
        run_for(0.2)
        before = base.xyz.copy()
        run_for(0.4)
        check("x3: a zero Twist stops it", np.allclose(before, base.xyz), f"{before} {base.xyz}")

    names_wire = set(m.PARAMETERS) | {m.PARAM_ROBOT_DESCRIPTION}
    _wire(check, "x3", spec["topics"], spec["services"], names_wire, ns, m.attach_ros, extra)


def run_composite(check) -> None:
    import robots_spec
    from contracts import message_schemas as schemas
    from ros_surfaces import myagv as b
    from ros_surfaces import myagv_mycobot280 as m

    print("myAGV + myCobot 280 contract vs robots_specs/myagv_mycobot280/ros.yml "
          "(extends myagv/ros.yml)")
    spec = yaml.safe_load(robots_spec.ros_path("myagv_mycobot280").read_text())
    base_spec = yaml.safe_load(robots_spec.ros_path("myagv").read_text())
    check("composite: it extends the myAGV's file", spec.get("extends") == "myagv/ros.yml")
    check("composite: every added topic row: name, type, direction, node, rate",
          _rows(spec) == _module_rows(m), str(set(_rows(spec)) ^ set(_module_rows(m)))[:400])
    conditional = {t["name"] for t in spec["topics"] if t.get("active_while")}
    check("composite: the conditional rows are the file's `active_while` rows",
          conditional == set(m.CONDITIONAL), str(conditional ^ set(m.CONDITIONAL)))
    want_srv = sorted((s["name"], s["type"], _node(s["node"])) for s in spec["services"])
    check("composite: every added service: name, type, node", want_srv == sorted(m.SERVICES),
          str(set(want_srv) ^ set(m.SERVICES))[:300])
    params = m.parameters(robots_spec.robot("myagv_mycobot280").folder / m.NAVIGATION_DIR)
    rows = [p["name"] for p in spec["parameters"]]
    uncovered = sorted(n for n in params if not any(covers(r, n) for r in rows))
    unmatched = sorted(r for r in rows if not any(covers(r, n) for n in params))
    check("composite: every parameter the launch sets is one of the file's rows",
          not uncovered, str(uncovered))
    check("composite: every parameter row names parameters the launch sets",
          not unmatched, str(unmatched))
    explicit = {r["name"]: r for r in spec["parameters"] if "*" not in r["name"]
                and "{" not in r["name"]}
    wrong = [n for n in ("/amcl/initial_pose_x", "/amcl/initial_pose_y", "/amcl/initial_pose_a",
                         "/move_base/planner_frequency", "/move_base/controller_frequency")
             if n in explicit and not str(explicit[n]["description"]).startswith(
                 f"{params[n]:g}")]
    check("composite: the launch's own values", not wrong, str(wrong))
    types = {t["type"] for t in spec["topics"]} | {s["type"] for s in spec["services"]}
    unknown = sorted(t for t in types if schemas.canonical(t) not in schemas.known_types())
    check("composite: the schema table answers for every type it adds", not unknown,
          str(unknown))
    check("composite: the stop command cancels navigation, then sends a zero Twist",
          "/move_base/cancel" in spec["stop_command"] and "/cmd_vel" in spec["stop_command"])

    ns = "myagv_mycobot280"

    def extra(client, run_for, base):
        run_for(0.5)
        client.quiet(lambda msg: False, window=0.2)
        client.inbox.clear()
        amcl = client.expect(lambda x: x.get("topic") == f"/{ns}/amcl_pose", timeout=0.5)
        # A goal 0.5 m ahead of where amcl starts, facing the same way.
        x0, y0, a0 = (m.LAUNCH_PARAMS[k] for k in ("/amcl/initial_pose_x", "/amcl/initial_pose_y",
                                                  "/amcl/initial_pose_a"))
        goal = {"header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": "map"},
                "pose": {"position": {"x": x0 + 0.5 * math.cos(a0), "y": y0 + 0.5 * math.sin(a0),
                                      "z": 0.0},
                         "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(a0 / 2),
                                         "w": math.cos(a0 / 2)}}}
        client.send({"op": "publish", "topic": f"/{ns}/move_base_simple/goal", "msg": goal})
        client.sync()
        start = base.xyz.copy()
        run_for(1.5)
        client.quiet(lambda msg: False, window=0.3)
        drives = [x["msg"] for x in client.inbox if x.get("topic") == f"/{ns}/cmd_vel"]
        check("composite: a navigation goal drives the base through /cmd_vel",
              drives and any(d["linear"]["x"] > 0 for d in drives)
              and np.linalg.norm(base.xyz[:2] - start[:2]) > 0.1,
              f"{len(drives)} commands, moved {np.linalg.norm(base.xyz[:2] - start[:2]):.3f} m")
        status = [x["msg"] for x in client.inbox if x.get("topic") == f"/{ns}/move_base/status"]
        check("composite: move_base reports the goal ACTIVE",
              status and any(s["status"] == 1 for s in status[-1]["status_list"]),
              str(status[-1] if status else None)[:200])
        client.inbox.clear()
        client.send({"op": "publish", "topic": f"/{ns}/move_base/cancel", "msg": m.STOP_CANCEL})
        client.send({"op": "publish", "topic": f"/{ns}/cmd_vel", "msg": m.STOP_COMMAND})
        client.sync()
        run_for(0.3)
        result = client.expect(lambda x: x.get("topic") == f"/{ns}/move_base/result", timeout=1.0)
        check("composite: the stop command's cancel ends the goal PREEMPTED",
              result is not None and result["msg"]["status"]["status"] == 2, str(result)[:200])
        before = base.xyz.copy()
        run_for(0.4)
        check("composite: ...and the base stays stopped", np.allclose(before, base.xyz),
              f"{before} {base.xyz}")
        check("composite: amcl publishes its pose on the map frame",
              amcl is None or amcl["msg"]["header"]["frame_id"] == f"{ns}/map")

    topics = base_spec["topics"] + spec["topics"]
    services = base_spec["services"] + spec["services"]
    names = {p["name"] for p in base_spec["parameters"]} | set(params)
    _wire(check, "composite", topics, services, names, ns, m.attach_ros, extra)
    del b


def run(check) -> None:
    run_x3(check)
    run_composite(check)


if __name__ == "__main__":
    failures: list[str] = []

    def _check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail and not ok else ''}")
        if not ok:
            failures.append(name)

    run(_check)
    print(f"{len(failures)} failure(s)" if failures else "all checks passed")
    raise SystemExit(1 if failures else 0)
