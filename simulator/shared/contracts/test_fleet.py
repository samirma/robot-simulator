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


def test_so101_contract() -> None:
    """`ros_surfaces/so101.py` against `robots_specs/so101/ros2.yml`, name for name.

    Every topic (type, direction, periodic rate, node), service, action and ROS parameter
    the ROS file lists is in the contract module, and nothing else is -- except the
    `unverified` image_transport plugin topics, which are left out on purpose and listed
    in `OMITTED_TOPICS`. Every type the module names must also be one the schema table
    can answer `message_details` for.
    """
    print("SO-101 contract vs robots_specs/so101/ros2.yml")
    import yaml

    from contracts import message_schemas as schemas
    from ros_surfaces import so101

    spec = yaml.safe_load((Path(__file__).resolve().parents[3] / "robots_specs" / "so101"
                           / "ros2.yml").read_text())
    check("joints are the ROS file's, in its order",
          list(spec["joints"]) == list(so101.JOINT_ORDER))

    listed = {t["name"]: t for t in spec["topics"]}
    omitted = {n for n, t in listed.items() if t.get("unverified") and "/image_raw/" in n}
    check("the omitted topics are exactly the unverified image_transport plugins",
          omitted == set(so101.OMITTED_TOPICS), str(sorted(omitted)))
    served = {n: t for n, t in listed.items() if n not in omitted}
    check("every served topic is in the ROS file and every listed one is served",
          set(served) == set(so101.TOPICS),
          f"missing {sorted(set(served) - set(so101.TOPICS))}, "
          f"extra {sorted(set(so101.TOPICS) - set(served))}")
    wrong = []
    for name, t in served.items():
        mine = so101.TOPICS.get(name)
        if mine is None:
            continue
        rate = t["rate_hz"]
        if (mine[0], mine[1], mine[3]) != (t["type"], t["direction"], t["node"]) or \
                (float(mine[2]) != float(rate) if isinstance(rate, (int, float))
                 else mine[2] != rate):
            wrong.append((name, mine, (t["type"], t["direction"], rate, t["node"])))
    check("each topic's type, direction, rate and node are the ROS file's", not wrong,
          str(wrong))

    services = {s["name"]: (s["type"], s["node"]) for s in spec["services"]}
    check("services are the ROS file's, with its types and nodes",
          services == so101.SERVICES,
          str(set(services.items()) ^ set(so101.SERVICES.items())))
    actions = {a["name"]: (a["type"], a["node"]) for a in spec["actions"]}
    check("actions are the ROS file's, with its types and nodes", actions == so101.ACTIONS,
          str(set(actions.items()) ^ set(so101.ACTIONS.items())))

    params = {}
    for p in spec["parameters"]:
        names = [n.strip() for n in str(p["name"]).split(",")]
        if p["type"] == "mixed":
            values = [v.strip() for v in str(p["value"]).split("#")[0].split(",")]
            for n, v in zip(names, values):
                params[(p["node"], n)] = yaml.safe_load(v)
        elif "/" not in names[0]:   # SO_ARM101/usb_port is inside the URDF, not a parameter
            params[(p["node"], names[0])] = p["value"]
    mismatched = [k for k in set(params) | set(so101.PARAMETERS)
                  if k not in params or k not in so101.PARAMETERS
                  or (k[1] != "robot_description" and params[k] != so101.PARAMETERS[k])]
    check("parameters are the ROS file's, name and value", not mismatched, str(mismatched))

    types = [t for t, *_ in so101.TOPICS.values()] + \
        [t for t, _ in so101.SERVICES.values()] + [t for t, _ in so101.ACTIONS.values()]
    unknown = [t for t in types if schemas.canonical(t) not in schemas.known_types()]
    check("the schema table answers for every type the SO-101 serves", not unknown,
          str(unknown))

    description = so101.bringup_description(
        (Path(__file__).resolve().parents[3] / "robots_specs" / "so101"
         / "so101_new_calib.urdf").read_text())
    from contracts.tf import UrdfTree

    tree = UrdfTree(description)
    check("the description's fixed joints are the ROS file's /tf_static "
          "(world->base_link, gripper_frame_joint)",
          sorted((p, c) for p, c, *_ in tree.fixed())
          == [("gripper_link", "gripper_frame_link"), ("world", "base_link")])
    check("the description's moving joints are the ROS file's joints",
          sorted(j["name"] for j in tree.joints if j["type"] != "fixed")
          == sorted(spec["joints"]))


def main() -> int:
    print(f"fleet transport check ({threading.active_count()} threads at start)\n")
    test_naming()
    test_collisions()
    test_routing_and_discovery()
    test_rosapi_surface()
    test_so101_contract()
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
