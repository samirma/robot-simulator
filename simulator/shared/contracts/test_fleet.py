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

            real_rosapi = [
                "/rosapi/topics", "/rosapi/topics_for_type", "/rosapi/topic_type",
                "/rosapi/services", "/rosapi/service_type", "/rosapi/publishers",
                "/rosapi/subscribers", "/rosapi/nodes", "/rosapi/node_details",
                "/rosapi/message_details", "/rosapi/service_request_details",
                "/rosapi/service_response_details", "/rosapi/get_param_names",
                "/rosapi/get_param", "/rosapi/action_servers", "/rosapi/get_ros_version",
            ]
            refused = [s for s in real_rosapi if not call(s, {"type": "", "topic": "",
                                                           "service": "", "node": ""})[0]]
            check("every service a real rosapi ships answers", not refused, str(refused))

            ok, v = call("/rosapi/topic_type", {"topic": "/a/cmd_vel"})
            check("topic_type answers for a subscribe-only command topic",
                  ok and v.get("type") == TYPE_TWIST, str(v))

            ok, v = call("/rosapi/subscribers", {"topic": "/a/cmd_vel"})
            check("subscribers names the robot that owns the topic, and only it",
                  v.get("subscribers") == ["/a"], str(v))
            ok, v = call("/rosapi/publishers", {"topic": "/b/joint_states"})
            check("publishers likewise", v.get("publishers") == ["/b"], str(v))

            ok, v = call("/rosapi/nodes")
            check("nodes are the namespaces", set(v.get("nodes", [])) == {"/a", "/b"}, str(v))
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


def main() -> int:
    print(f"fleet transport check ({threading.active_count()} threads at start)\n")
    test_naming()
    test_collisions()
    test_routing_and_discovery()
    test_rosapi_surface()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
