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


def main() -> int:
    print(f"fleet transport check ({threading.active_count()} threads at start)\n")
    test_naming()
    test_collisions()
    test_routing_and_discovery()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
