#!/usr/bin/env python
"""Two robots on one server: are they actually separate?

A standalone script rather than pytest, following the convention the robot self-tests
use -- a failure here points at the transport, not at a scene. It needs `websockets` and
nothing else, so either engine's venv runs it:

    molmospaces/.venv/bin/python shared/contracts/test_fleet.py

What it pins is the part of multi-robot that is silent when wrong. Before namespacing,
`RosBridgeServer.on` stored one callback per topic and overwrote without complaint, so a
second robot took the first one's `/cmd_vel` away and both published `/odom` onto one
topic. Nothing raised; the first robot simply stopped responding, which reads as a
physics or a wiring fault and is neither.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.namespace import ns_frame, ns_topic  # noqa: E402
from contracts.rosbridge_server import (  # noqa: E402
    TYPE_ODOM,
    TYPE_TWIST,
    NamespacedBus,
    RosBridgeServer,
)

PORT = 9399
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_naming() -> None:
    print("namespace composition")
    check("a topic is an absolute path", ns_topic("myagv", "/cmd_vel") == "/myagv/cmd_vel")
    check("a frame is a tf_prefix join, with no leading slash",
          ns_frame("myagv", "base_footprint") == "myagv/base_footprint")
    check("an empty namespace is the identity",
          ns_topic("", "/cmd_vel") == "/cmd_vel" and ns_frame("", "odom") == "odom")
    check("composition is idempotent",
          ns_topic("myagv", "/myagv/cmd_vel") == "/myagv/cmd_vel")
    check("an empty frame stays empty (a real JointState carries no frame)",
          ns_frame("so101", "") == "")


def test_collisions() -> None:
    print("collisions are refused, not silently won")
    server = RosBridgeServer(port=0)
    a, b = NamespacedBus(server, "a"), NamespacedBus(server, "b")
    a.on("/cmd_vel", lambda m: None, TYPE_TWIST)
    b.on("/cmd_vel", lambda m: None, TYPE_TWIST)
    check("two namespaced robots may both take /cmd_vel",
          a.subscribed == ["/a/cmd_vel"] and b.subscribed == ["/b/cmd_vel"])

    bare1, bare2 = NamespacedBus(server, ""), NamespacedBus(server, "")
    bare1.on("/scan", lambda m: None)
    try:
        bare2.on("/scan", lambda m: None)
        check("two UNnamespaced robots on one topic raise", False, "it was accepted")
    except ValueError:
        check("two UNnamespaced robots on one topic raise", True)

    check("per-bus sequence counters are independent",
          (a.next_seq(), a.next_seq(), b.next_seq()) == (1, 2, 1))


def test_routing_and_discovery() -> None:
    """Over a real socket: a command reaches one robot, and rosapi lists both."""
    print("routing and discovery, over the wire")
    import websockets.sync.client as ws_client

    server = RosBridgeServer(port=PORT)
    got: dict[str, list] = {"a": [], "b": []}
    for name in ("a", "b"):
        bus = NamespacedBus(server, name)
        bus.on("/cmd_vel", (lambda n: lambda msg: got[n].append(msg))(name), TYPE_TWIST)
        bus.publish("/odom", {"seeded": True}, TYPE_ODOM)
    server.serve_rosapi()
    server.start()
    try:
        with ws_client.connect(f"ws://127.0.0.1:{PORT}") as conn:
            conn.send(json.dumps({
                "op": "publish", "topic": "/a/cmd_vel",
                "msg": {"linear": {"x": 1.0}, "angular": {"z": 0.0}},
            }))
            deadline = time.monotonic() + 3.0
            while not got["a"] and time.monotonic() < deadline:
                time.sleep(0.01)
            check("a command reaches the robot it is addressed to", len(got["a"]) == 1)
            check("...and only that one", len(got["b"]) == 0)

            conn.send(json.dumps({
                "op": "call_service", "service": "/rosapi/topics", "id": "q", "args": {},
            }))
            reply = json.loads(conn.recv(timeout=5))
            topics = set(reply["values"]["topics"])
            check("rosapi lists both robots' command topics",
                  {"/a/cmd_vel", "/b/cmd_vel"} <= topics, str(sorted(topics)))
            check("rosapi lists both robots' published topics",
                  {"/a/odom", "/b/odom"} <= topics)
            check("nothing is left on a bare, unnamespaced name",
                  not any(t in topics for t in ("/cmd_vel", "/odom")))
    finally:
        server.stop()


def test_rosapi_surface() -> None:
    """A rosapi client can ask this bridge what a real one answers.

    It used to answer two queries and refuse the rest with `no service` -- a client asking
    what type a topic was got nothing, which is the largest way it could tell this bridge
    from the robot it stands in for. Every query a real `rosapi_node` ships is called here
    over the wire, and the ones whose answers carry structure are checked for content, not
    just for answering.
    """
    print("rosapi introspection, over the wire")
    import websockets.sync.client as ws_client

    from contracts import message_schemas as schemas
    from contracts.rosbridge_server import (
        SRV_TYPE_TRIGGER, TYPE_JOINT_STATE, joint_state, odometry,
    )
    from contracts.tf import TYPE_TF_MESSAGE, tf_message

    server = RosBridgeServer(port=PORT + 1)
    a, b = NamespacedBus(server, "a"), NamespacedBus(server, "b")
    a.set_param("/robot_description", "<robot name='a'/>")
    b.set_param("/robot_description", "<robot name='b'/>")
    a.on("/cmd_vel", lambda m: None, TYPE_TWIST)
    b.on("/cmd_vel", lambda m: None, TYPE_TWIST)
    a.publish("/odom", odometry(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), TYPE_ODOM)
    b.publish("/joint_states", joint_state(["j"], [0.0], [0.0], 0.0), TYPE_JOINT_STATE)
    a.service("/reset", lambda args: {"success": True, "message": ""}, SRV_TYPE_TRIGGER)
    server.serve_rosapi()
    server.start()
    try:
        with ws_client.connect(f"ws://127.0.0.1:{PORT + 1}") as conn:
            counter = {"n": 0}

            def call(service: str, args: dict | None = None) -> tuple[bool, dict]:
                counter["n"] += 1
                conn.send(json.dumps({"op": "call_service", "service": service,
                                      "id": f"q{counter['n']}", "args": args or {}}))
                reply = json.loads(conn.recv(timeout=5))
                return bool(reply.get("result")), reply.get("values") or {}

            from contracts.test_transport import ROSAPI_31
            refused = [s for s in ROSAPI_31 if not call(s, {"type": "", "topic": "",
                                                          "service": "", "node": ""})[0]]
            check("every one of the spec's 31 rosapi services answers", not refused,
                  str(refused))

            ok, v = call("/rosapi/topic_type", {"topic": "/a/cmd_vel"})
            check("topic_type answers for a subscribe-only command topic",
                  ok and v.get("type") == TYPE_TWIST, str(v))

            ok, v = call("/rosapi/subscribers", {"topic": "/a/cmd_vel"})
            check("subscribers names the robot that owns the topic, and only it",
                  v.get("subscribers") == ["/a"], str(v))
            ok, v = call("/rosapi/publishers", {"topic": "/b/joint_states"})
            check("publishers likewise", v.get("publishers") == ["/b"], str(v))

            ok, v = call("/rosapi/nodes")
            check("nodes are the namespaces, beside the bridge's own runtime nodes",
                  set(v.get("nodes", [])) == {"/a", "/b", "/rosapi", "/rosbridge_websocket"},
                  str(v))
            ok, v = call("/rosapi/node_details", {"node": "/a"})
            check("node_details keeps one robot's names apart from the other's",
                  v.get("subscribing") == ["/a/cmd_vel"] and v.get("publishing") == ["/a/odom"]
                  and v.get("services") == ["/a/reset"], str(v))

            ok, v = call("/rosapi/services")
            check("services lists the robot's own beside rosapi's",
                  "/a/reset" in v.get("services", []) and "/rosapi/topics" in v["services"])
            ok, v = call("/rosapi/service_type", {"service": "/a/reset"})
            check("service_type answers", v.get("type") == SRV_TYPE_TRIGGER, str(v))

            ok, v = call("/rosapi/message_details", {"type": "sensor_msgs/msg/Imu"})
            names = [t["type"] for t in v.get("typedefs", [])]
            check("message_details returns the nested typedefs, not just the top level",
                  names[:1] == ["sensor_msgs/Imu"]
                  and {"std_msgs/Header", "geometry_msgs/Quaternion", "geometry_msgs/Vector3"}
                  <= set(names), str(names))
            ok, v = call("/rosapi/message_details", {"type": "ainex_interfaces/HeadState"})
            td = (v.get("typedefs") or [{}])[0]
            check("the vendor's HeadState comes back as the vendor defines it",
                  td.get("fieldnames") == ["position", "duration"]
                  and td.get("fieldtypes") == ["float64", "float64"], str(td))
            ok, v = call("/rosapi/service_response_details",
                         {"type": "ros_robot_controller/GetBusServosPosition"})
            td = (v.get("typedefs") or [{}])[0]
            check("a service response schema resolves, including its nested type",
                  td.get("fieldnames") == ["success", "position"]
                  and len(v.get("typedefs", [])) == 2, str(td.get("fieldnames")))

            # The description, and the parameter that carries it. `get_param` answered
            # the empty string for every name until the transform tree went in, so a
            # client could read a robot's joint angles and never learn what its body is.
            ok, v = call("/rosapi/get_param_names")
            check("get_param_names lists each robot's description, namespaced",
                  set(v.get("names", [])) == {"/a/robot_description", "/b/robot_description"},
                  str(v))
            ok, v = call("/rosapi/get_param", {"name": "/a/robot_description"})
            check("get_param returns the description JSON-encoded, as rosapi does",
                  json.loads(v.get("value", '""')) == "<robot name='a'/>", str(v)[:80])
            ok, v = call("/rosapi/get_param", {"name": "/nope", "default": "fallback"})
            check("an unset parameter comes back as the caller's own default",
                  v.get("value") == "fallback", str(v))

            # Drift: the table must describe what this bridge actually sends. Every
            # message a builder produced here is compared, key for key, with its schema.
            tf_msg = tf_message([("odom", "base_footprint", (0, 0, 0), (1, 0, 0, 0))],
                                stamp_s=0.0)
            for label, msg, mtype in (("Odometry", odometry(1, 0, 0, 0, 0, 0, 0), TYPE_ODOM),
                                      ("JointState", joint_state(["j"], [0.0], [0.0], 0.0),
                                       TYPE_JOINT_STATE),
                                      ("TFMessage", tf_msg, TYPE_TF_MESSAGE),
                                      ("TransformStamped", tf_msg["transforms"][0],
                                       "geometry_msgs/TransformStamped")):
                declared = [f[0] for f in schemas.fields_of(mtype) or []]
                check(f"{label} as built matches its declared schema, field for field",
                      sorted(declared) == sorted(msg), f"{sorted(declared)} vs {sorted(msg)}")
    finally:
        server.stop()


def test_myagv_contract() -> None:
    """`ros_surfaces/myagv.py` against `robots_specs/myagv/ros.yml`, then on the wire.

    The module is compared with the file row for row -- name, type, direction, node and
    rate of every topic, every service, every parameter -- and then attached to a server
    (no MuJoCo model: the camera and lidar render nothing, everything else runs) so that
    what `rosapi` reports is compared with the file too, and every message it publishes
    with its declared type field for field.
    """
    print("myAGV contract vs robots_specs/myagv/ros.yml")
    import math

    import numpy as np
    import yaml

    import robots_spec
    from contracts import message_schemas as schemas
    from contracts.test_transport import Client
    from ros_surfaces import RobotFleet
    from ros_surfaces import myagv as m

    spec = yaml.safe_load(robots_spec.ros_path("myagv").read_text())

    def rate(row):
        r = row.get("rate_hz")
        return float(r) if isinstance(r, (int, float)) else str(r)

    want_topics = sorted((t["name"], t["type"], t["direction"], t["node"], rate(t))
                         for t in spec["topics"])
    have_topics = sorted((n, ty, d, node, float(r) if isinstance(r, float) else r)
                         for n, ty, d, node, r in m.TOPICS)
    check("every topic row: name, type, direction, node, rate", want_topics == have_topics,
          str(sorted(set(want_topics) ^ set(have_topics))))
    want_srv = sorted((s["name"], s["type"], s["node"]) for s in spec["services"])
    check("every service: name, type, node", want_srv == sorted(m.SERVICES),
          str(sorted(set(want_srv) ^ set(m.SERVICES))))
    want_params = {p["name"] for p in spec["parameters"]}
    have_params = set(m.PARAMETERS) | {m.PARAM_ROBOT_DESCRIPTION}
    check("every parameter", want_params == have_params, str(sorted(want_params ^ have_params)))
    check("the loop runs at the fastest declared rate",
          m.LOOP_HZ == max(r for *_, r in have_topics if isinstance(r, float)))

    class FakeBase:
        """A planar base that goes exactly where it is told."""

        def __init__(self) -> None:
            self.xyz = np.zeros(3)
            self.target = np.zeros(3)

        @property
        def pose(self):
            x, y, yaw = self.xyz
            out = np.eye(4)
            out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            out[0, 3], out[1, 3] = x, y
            return out

        @property
        def ctrl(self):
            return self.target

        @ctrl.setter
        def ctrl(self, value) -> None:
            self.target = np.asarray(value, dtype=float)
            self.xyz = self.target.copy()

    class FakeData:
        time = 0.0

    port, ns = PORT + 2, "myagv"
    fleet = RobotFleet(port=port, default_hz=10.0)
    base = FakeBase()
    fleet.attach(ns, m.attach_ros, base=base, model=None, camera=None)
    check("the fleet steps at the myAGV's own rate", fleet.rate_hz == m.LOOP_HZ,
          str(fleet.rate_hz))
    fleet.start()
    data = FakeData()

    def run_for(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            data.time += 1.0 / m.LOOP_HZ
            fleet(data)
            time.sleep(1.0 / m.LOOP_HZ)

    composed = {(ns_topic(ns, t["name"]), t["type"]) for t in spec["topics"]}
    try:
        client = Client(port)
        ok, v = client.call("/rosapi/topics")
        wire = {(t, ty) for t, ty in zip(v.get("topics", []), v.get("types", []))}
        check("rosapi lists exactly the file's topics, typed", wire == composed,
              str(sorted(wire ^ composed)))
        for topic in sorted({t["name"] for t in spec["topics"]}):
            want = sorted({f"/{ns}/{t['node']}" for t in spec["topics"]
                           if t["name"] == topic})
            role = "subscribers" if topic == m.TOPIC_CMD_VEL else "publishers"
            ok, v = client.call(f"/rosapi/{role}", {"topic": ns_topic(ns, topic)})
            if v.get(role) != want:
                check(f"{topic}'s {role} are its nodes", False, f"{v.get(role)} vs {want}")
        ok, v = client.call("/rosapi/services")
        mine = {s for s in v.get("services", []) if s.startswith(f"/{ns}/")}
        check("the file's services, and no others",
              mine == {ns_topic(ns, s["name"]) for s in spec["services"]}, str(sorted(mine)))
        for s in spec["services"]:
            ok, v = client.call("/rosapi/service_node", {"service": ns_topic(ns, s["name"])})
            ok2, t = client.call("/rosapi/service_type", {"service": ns_topic(ns, s["name"])})
            if v.get("node") != f"/{ns}/{s['node']}" or t.get("type") != s["type"]:
                check(f"{s['name']} node and type", False, f"{v} {t}")
        ok, v = client.call("/rosapi/get_param_names")
        check("the file's parameters, namespaced",
              set(v.get("names", [])) == {ns_topic(ns, p) for p in want_params},
              str(sorted(set(v.get("names", [])) ^ {ns_topic(ns, p) for p in want_params})))
        ok, v = client.call("/rosapi/get_param", {"name": f"/{ns}/robot_description"})
        check("robot_description is the vendor URDF",
              json.loads(v.get("value", '""')) == robots_spec.urdf_path("myagv").read_text())

        # /tf_static is latched: a late subscriber still gets it, once.
        client.send({"op": "subscribe", "topic": f"/{ns}/tf_static"})
        got = client.expect(lambda msg: msg.get("topic") == f"/{ns}/tf_static", timeout=2.0)
        check("/tf_static is latched, and empty (the URDF has no fixed joint)",
              got is not None and got["msg"] == {"transforms": []}, str(got)[:120])

        for t in spec["topics"]:
            if t["direction"] == "out":
                client.send({"op": "subscribe", "topic": ns_topic(ns, t["name"])})
        client.sync()
        # 2.0 on every axis is clamped to 1.0, and held with no further command.
        client.send({"op": "publish", "topic": f"/{ns}/cmd_vel",
                     "msg": {"linear": {"x": 2.0, "y": -2.0, "z": 0.0},
                             "angular": {"x": 0.0, "y": 0.0, "z": 0.5}}})
        client.sync()
        run_for(1.5)
        client.quiet(lambda msg: False, window=0.3)
        odoms = [msg["msg"] for msg in client.inbox if msg.get("topic") == f"/{ns}/odom"]
        last = odoms[-1]["twist"]["twist"] if odoms else {}
        check("cmd_vel is clamped to +/-1 and held with no timeout",
              last.get("linear", {}).get("x") == 1.0 and last.get("linear", {}).get("y") == -1.0
              and last.get("angular", {}).get("z") == 0.5, str(last))
        check("...and the base is still being driven 1.5 s later",
              base.xyz[0] > 1.0, f"x {base.xyz[0]:.3f} m")

        seen: dict[str, dict] = {}
        tf_frames = set()
        for msg in client.inbox:
            if msg.get("op") == "publish":
                seen.setdefault(msg["topic"], msg["msg"])
                if msg["topic"] == f"/{ns}/tf":
                    tf_frames |= {(tr["header"]["frame_id"], tr["child_frame_id"])
                                  for tr in msg["msg"]["transforms"]}
        for t in spec["topics"]:
            name = ns_topic(ns, t["name"])
            if t["direction"] != "out" or name not in seen:
                continue
            declared = [f[0] for f in schemas.fields_of(t["type"]) or []]
            if sorted(declared) != sorted(seen[name]):
                check(f"{t['name']} matches {t['type']} field for field", False,
                      f"{sorted(declared)} vs {sorted(seen[name])}")
        missing = sorted({t["name"] for t in spec["topics"] if t["direction"] == "out"
                          and ns_topic(ns, t["name"]) not in seen
                          and t["name"] != m.TOPIC_TF_STATIC  # checked above
                          and t["node"] not in (m.NODE_LIDAR, m.NODE_CAMERA)})
        check("every non-sensor topic was published", not missing, str(missing))
        want_tf = {(ns_frame(ns, "odom"), ns_frame(ns, "base_footprint"))} | {
            (ns_frame(ns, p), ns_frame(ns, c))
            for p, c, *_ in m.static_transforms().values()} | {
            (ns_frame(ns, "base_footprint"), ns_frame(ns, "base_up"))}
        check("/tf carries the launch's tree, namespaced", tf_frames == want_tf,
              str(sorted(tf_frames ^ want_tf)))
        ok, v = client.call(f"/{ns}/robot_pose_ekf/get_status")
        check("get_status answers", ok and "Robot pose ekf filter" in v.get("status", ""))
        ok, v = client.call(f"/{ns}/camera/set_camera_info",
                            {"camera_info": {"width": 640, "height": 480}})
        check("set_camera_info stores a calibration", ok and v.get("success") is True, str(v))
        # The stop command is a zero Twist, and nothing else stops the base.
        client.send({"op": "publish", "topic": f"/{ns}/cmd_vel", "msg": m.STOP_COMMAND})
        client.sync()
        run_for(0.2)
        before = base.xyz.copy()
        run_for(0.3)
        check("a zero Twist stops it", np.allclose(before, base.xyz), f"{before} -> {base.xyz}")
        client.close()
    finally:
        fleet(None)


def main() -> int:
    print(f"fleet transport check ({threading.active_count()} threads at start)\n")
    test_naming()
    test_collisions()
    test_routing_and_discovery()
    test_rosapi_surface()
    test_myagv_contract()
    # Every transport operation, latching, routing, the action lifecycle and all 31 rosapi
    # services, over the wire (spec §5, "Contracts"). A sibling module for length only;
    # it reports into this file's FAILURES.
    from contracts import test_transport
    test_transport.run(check)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
