#!/usr/bin/env python
"""Every transport operation, over a real websocket.

Run by `test_fleet.py` (which reports into its own tally), or on its own the same way:

    molmospaces/.venv/bin/python shared/contracts/test_transport.py

What it pins is spec §2.2 and §3: the accepted operations and nothing else, a `status`
error for anything unsupported or malformed, latched delivery (each publisher's last
message once to each new subscriber, never republished), client advertisements routed to
other clients and to the surfaces, the ROS 2 action lifecycle (goal, feedback, status,
result, cancel, abort by `/reset`, a disconnect that leaves goals running), and each of
the 31 `rosapi` services.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets.sync.client as ws_client  # noqa: E402

from contracts import message_schemas as schemas  # noqa: E402
from contracts.rosbridge_server import (  # noqa: E402
    SIMULATOR_NODE,
    SRV_TYPE_TRIGGER,
    TYPE_TWIST,
    GoalStatus,
    NamespacedBus,
    RosBridgeServer,
)

PORT = 9411

ROSAPI_31 = [f"/rosapi/{name}" for name in (
    # the ROS 1 rosapi_node's 25
    "topics", "topics_for_type", "topics_and_raw_types", "topic_type", "services",
    "services_for_type", "service_type", "service_providers", "service_node",
    "service_host", "nodes", "node_details", "publishers", "subscribers", "action_servers",
    "message_details", "service_request_details", "service_response_details",
    "get_param_names", "get_param", "set_param", "has_param", "search_param",
    "delete_param", "get_time",
    # the ROS 2 rosapi_node's 6 more
    "interfaces", "action_type", "action_goal_details", "action_result_details",
    "action_feedback_details", "get_ros_version",
)]

TYPE_STRING2 = "std_msgs/msg/String"
ACTION_TYPE = "example_interfaces/action/Fibonacci"


class Client:
    """A rosbridge client that keeps what it has not yet been asked about."""

    def __init__(self, port: int) -> None:
        self.conn = ws_client.connect(f"ws://127.0.0.1:{port}")
        self.inbox: list[dict] = []
        self._n = 0

    def send(self, frame) -> None:
        self.conn.send(frame if isinstance(frame, str) else json.dumps(frame))

    def _pump(self, timeout: float) -> bool:
        try:
            raw = self.conn.recv(timeout=timeout)
        except TimeoutError:
            return False
        self.inbox.append(json.loads(raw))
        return True

    def expect(self, pred: Callable[[dict], bool], timeout: float = 3.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while True:
            for i, msg in enumerate(self.inbox):
                if pred(msg):
                    return self.inbox.pop(i)
            left = deadline - time.monotonic()
            if left <= 0 or not self._pump(left):
                if left <= 0:
                    return None

    def quiet(self, pred: Callable[[dict], bool], window: float = 0.4) -> list[dict]:
        """Everything matching `pred` that arrives within `window` (ideally nothing)."""
        deadline = time.monotonic() + window
        while time.monotonic() < deadline:
            self._pump(max(0.0, deadline - time.monotonic()))
        hits = [m for m in self.inbox if pred(m)]
        self.inbox = [m for m in self.inbox if not pred(m)]
        return hits

    def call(self, service: str, args: dict | None = None) -> tuple[bool, dict]:
        self._n += 1
        cid = f"call{self._n}"
        self.send({"op": "call_service", "service": service, "id": cid, "args": args or {}})
        reply = self.expect(lambda m: m.get("op") == "service_response" and m.get("id") == cid)
        if reply is None:
            return False, {"error": "no reply"}
        return bool(reply.get("result")), reply.get("values") or {}

    def sync(self) -> None:
        """A round trip, so every op this client sent before it has been handled."""
        self.call("/rosapi/get_ros_version")

    def close(self) -> None:
        self.conn.close()


def _status(mid=None):
    return lambda m: m.get("op") == "status" and m.get("level") == "error" and (
        mid is None or m.get("id") == mid)


def _pub(topic):
    return lambda m: m.get("op") == "publish" and m.get("topic") == topic


# ------------------------------------------------------------------------------ checks

def check_ops(check, port: int) -> None:
    print("transport ops: unsupported and malformed get a status error")
    c = Client(port)
    try:
        c.send("{not json")
        check("invalid JSON -> status error", c.expect(_status()) is not None)
        c.send("[1, 2]")
        check("a non-object -> status error", c.expect(_status()) is not None)
        c.send({"id": "x1"})
        check("no op -> status error, echoing id", c.expect(_status("x1")) is not None)
        for op in ("fragment", "png", "auth", "call_services", "service_response_x"):
            c.send({"op": op, "id": f"u-{op}"})
            check(f"unsupported op {op!r} -> status error",
                  c.expect(_status(f"u-{op}")) is not None)
        for op, missing in (("subscribe", "topic"), ("unsubscribe", "topic"),
                            ("advertise", "type"), ("unadvertise", "topic"),
                            ("publish", "topic"), ("call_service", "service"),
                            ("advertise_service", "type"), ("unadvertise_service", "service"),
                            ("advertise_action", "type"), ("unadvertise_action", "action"),
                            ("send_action_goal", "action_type"),
                            ("cancel_action_goal", "action")):
            frame = {"op": op, "id": f"m-{op}", "topic": "/t", "service": "/s",
                     "action": "/a", "type": "std_msgs/msg/String",
                     "action_type": ACTION_TYPE}
            del frame[missing]
            c.send(frame)
            check(f"{op} without '{missing}' -> status error",
                  c.expect(_status(f"m-{op}")) is not None)
        c.send({"op": "publish", "id": "badmsg", "topic": "/x", "msg": [1]})
        check("publish with a non-object msg -> status error", c.expect(_status("badmsg")) is not None)
        c.send({"op": "service_response", "id": "nobody", "service": "/s", "values": {},
                "result": True})
        check("service_response for no pending call -> status error",
              c.expect(_status("nobody")) is not None)
        c.send({"op": "action_result", "id": "nobody2", "action": "/a", "values": {},
                "status": 4, "result": True})
        check("action_result for no relayed goal -> status error",
              c.expect(_status("nobody2")) is not None)
        c.send({"op": "action_feedback", "id": "nobody3", "action": "/a", "values": {}})
        check("action_feedback for no relayed goal -> status error",
              c.expect(_status("nobody3")) is not None)

        c.send({"op": "status", "level": "info", "msg": "hello"})
        check("a client's own status op is accepted, silently",
              not c.quiet(lambda m: m.get("op") == "status"))
        c.send({"op": "set_level", "id": "lv", "level": "loud"})
        check("set_level with an unknown level -> status error", c.expect(_status("lv")) is not None)
        c.send({"op": "set_level", "level": "none"})
        c.send({"op": "bogus"})
        check("set_level none silences status messages",
              not c.quiet(lambda m: m.get("op") == "status"))
        c.send({"op": "set_level", "level": "error"})
        c.send({"op": "bogus", "id": "back"})
        check("set_level error brings errors back", c.expect(_status("back")) is not None)
    finally:
        c.close()


def check_topics(check, server: RosBridgeServer, port: int) -> None:
    print("topics: subscribe, advertise, publish, routing, unadvertise")
    got: list[dict] = []
    bus = NamespacedBus(server, "r")
    bus.on("/chatter", got.append, TYPE_STRING2, node="listener")
    bus.publish("/odom_like", {"data": "early"}, TYPE_STRING2, node="talker")
    a, b = Client(port), Client(port)
    try:
        b.send({"op": "subscribe", "topic": "/r/odom_like"})
        check("a non-latched message sent before subscribing is not replayed",
              not b.quiet(_pub("/r/odom_like")))
        b.send({"op": "subscribe", "id": "wrongtype", "topic": "/r/odom_like",
                "type": "sensor_msgs/msg/Image"})
        check("subscribe with a type that disagrees -> status error",
              b.expect(_status("wrongtype")) is not None)

        b.send({"op": "subscribe", "id": "s1", "topic": "/r/chatter"})
        b.sync()
        a.send({"op": "subscribe", "topic": "/r/chatter"})
        a.send({"op": "advertise", "id": "adv1", "topic": "/r/chatter", "type": TYPE_STRING2})
        a.send({"op": "publish", "topic": "/r/chatter", "msg": {"data": "hi"}})
        m = b.expect(_pub("/r/chatter"))
        check("a client's publish reaches another client's subscription",
              m is not None and m["msg"] == {"data": "hi"}, str(m))
        check("...and the publisher's own subscription", a.expect(_pub("/r/chatter")) is not None)
        deadline = time.monotonic() + 2
        while not got and time.monotonic() < deadline:
            time.sleep(0.01)
        check("...and the surface subscribed to it", got == [{"data": "hi"}], str(got))

        ok, v = a.call("/rosapi/publishers", {"topic": "/r/chatter"})
        check("a client's advertisement belongs to /rosbridge_websocket",
              "/rosbridge_websocket" in v.get("publishers", []), str(v))
        ok, v = a.call("/rosapi/subscribers", {"topic": "/r/chatter"})
        check("subscribers name the surface's composed node and the bridge",
              v.get("subscribers") == ["/r/listener", "/rosbridge_websocket"], str(v))
        ok, v = a.call("/rosapi/publishers", {"topic": "/r/odom_like"})
        check("publishers name the surface's composed node",
              v.get("publishers") == ["/r/talker"], str(v))

        a.send({"op": "advertise", "id": "conflict", "topic": "/r/chatter",
                "type": "sensor_msgs/msg/Image"})
        check("advertise with a conflicting type -> status error",
              a.expect(_status("conflict")) is not None)

        a.send({"op": "advertise", "topic": "/client_only", "type": TYPE_STRING2})
        ok, v = a.call("/rosapi/topics")
        check("a client's own advertised topic is listed",
              "/client_only" in v.get("topics", []), str(v.get("topics")))
        a.send({"op": "unadvertise", "topic": "/client_only"})
        ok, v = a.call("/rosapi/topics")
        check("...and unlisted once unadvertised", "/client_only" not in v.get("topics", []))
        a.send({"op": "unadvertise", "id": "again", "topic": "/client_only"})
        check("unadvertise of what is not advertised -> status error",
              a.expect(_status("again")) is not None)

        a.send({"op": "publish", "id": "untyped", "topic": "/nowhere", "msg": {}})
        check("publish on an unknown topic with no type -> status error",
              a.expect(_status("untyped")) is not None)
        b.send({"op": "subscribe", "topic": "/auto"})
        b.sync()
        a.send({"op": "publish", "topic": "/auto", "type": TYPE_STRING2, "msg": {"data": 1}})
        check("publish with a type advertises on the fly and is routed",
              b.expect(_pub("/auto")) is not None)

        b.send({"op": "unsubscribe", "id": "s1", "topic": "/r/chatter"})
        b.sync()
        a.send({"op": "publish", "topic": "/r/chatter", "msg": {"data": "after"}})
        check("unsubscribe stops delivery", not b.quiet(_pub("/r/chatter")))
        b.send({"op": "unsubscribe", "id": "s9", "topic": "/r/chatter"})
        check("unsubscribe of what is not subscribed -> status error",
              b.expect(_status("s9")) is not None)
        a.send({"op": "unadvertise", "id": "adv1", "topic": "/r/chatter"})
        check("unadvertise by id is accepted",
              not a.quiet(lambda m: m.get("op") == "status"))
    finally:
        a.close()
        b.close()


def check_latching(check, server: RosBridgeServer, port: int) -> None:
    print("latching: each publisher's last message, once to each new subscriber")
    bus = NamespacedBus(server, "lat")
    for node, n in (("pub_one", 1), ("pub_one", 2), ("pub_two", 7)):
        bus.publish("/tf_static", {"data": f"{node}:{n}"}, TYPE_STRING2, latched=True,
                    node=node)
    first, second = Client(port), Client(port)
    try:
        first.send({"op": "subscribe", "topic": "/lat/tf_static"})
        got = sorted(m["msg"]["data"] for m in first.quiet(_pub("/lat/tf_static"), 0.6))
        check("a new subscriber gets each publisher's last message, once",
              got == ["pub_one:2", "pub_two:7"], str(got))
        first.send({"op": "subscribe", "id": "again", "topic": "/lat/tf_static"})
        check("a second subscription on the same connection is not re-sent them",
              not first.quiet(_pub("/lat/tf_static")))
        second.send({"op": "subscribe", "topic": "/lat/tf_static"})
        got2 = sorted(m["msg"]["data"] for m in second.quiet(_pub("/lat/tf_static"), 0.6))
        check("another new subscriber gets them too", got2 == ["pub_one:2", "pub_two:7"],
              str(got2))
        check("...and they are never republished to the first",
              not first.quiet(_pub("/lat/tf_static")))

        first.send({"op": "advertise", "topic": "/lat/client_latched", "type": TYPE_STRING2,
                    "latch": True})
        first.send({"op": "publish", "topic": "/lat/client_latched", "msg": {"data": "last"}})
        first.send({"op": "advertise", "topic": "/lat/client_qos", "type": TYPE_STRING2,
                    "qos": {"durability": "transient_local"}})
        first.send({"op": "publish", "topic": "/lat/client_qos", "msg": {"data": "q"}})
        first.send({"op": "advertise", "topic": "/lat/client_plain", "type": TYPE_STRING2})
        first.send({"op": "publish", "topic": "/lat/client_plain", "msg": {"data": "p"}})
        first.sync()
        for topic in ("/lat/client_latched", "/lat/client_qos", "/lat/client_plain"):
            second.send({"op": "subscribe", "topic": topic})
        check("a client's latched advertisement latches",
              len(second.quiet(_pub("/lat/client_latched"))) == 1)
        check("...as does one with transient_local durability",
              len(second.quiet(_pub("/lat/client_qos"))) == 1)
        check("...and a plain one does not", not second.quiet(_pub("/lat/client_plain")))
        first.send({"op": "unadvertise", "topic": "/lat/client_latched"})
        first.sync()
        third = Client(port)
        third.send({"op": "subscribe", "topic": "/lat/client_latched"})
        check("an unadvertised publisher's latched message is gone with it",
              not third.quiet(_pub("/lat/client_latched")))
        third.close()
    finally:
        first.close()
        second.close()


def check_actions(check, server: RosBridgeServer, port: int) -> None:
    print("actions: goal, feedback, status, result, cancel, abort, disconnect")
    release = threading.Event()
    goals: dict[str, object] = {}

    def execute(goal):
        goals[goal.args.get("tag", "")] = goal
        order = int(goal.args.get("order", 0))
        seq = [0, 1]
        for _ in range(order):
            goal.publish_feedback({"sequence": list(seq)})
            seq.append(seq[-1] + seq[-2])
        if goal.args.get("hold"):
            while not goal.cancel_requested and not release.is_set():
                time.sleep(0.01)
        return {"sequence": seq}

    arm = NamespacedBus(server, "arm")
    base = NamespacedBus(server, "base")
    arm.action("/fib", ACTION_TYPE, execute, node="fib_server")
    base.action("/fib", ACTION_TYPE, execute)
    refuse = {"flag": False}
    arm.action("/stubborn", ACTION_TYPE, execute, cancel=lambda g: not refuse["flag"])
    try:
        arm.action("/fib", ACTION_TYPE, execute)
        check("a second server on one action name is refused", False)
    except ValueError:
        check("a second server on one action name is refused", True)

    c = Client(port)
    try:
        def result(cid):
            return c.expect(lambda m: m.get("op") == "action_result" and m.get("id") == cid)

        c.send({"op": "subscribe", "topic": "/arm/fib/_action/status"})
        c.send({"op": "send_action_goal", "id": "g1", "action": "/arm/fib",
                "action_type": ACTION_TYPE, "args": {"order": 3}, "feedback": True})
        r = result("g1")
        fb = c.quiet(lambda m: m.get("op") == "action_feedback" and m.get("id") == "g1", 0.2)
        check("a goal's feedback arrives, with id and action",
              len(fb) == 3 and fb[0]["action"] == "/arm/fib"
              and fb[-1]["values"] == {"sequence": [0, 1, 1, 2]}, str(fb[:1]))
        check("its result arrives SUCCEEDED, result true",
              r is not None and r["status"] == GoalStatus.SUCCEEDED and r["result"] is True
              and r["values"] == {"sequence": [0, 1, 1, 2, 3]} and r["action"] == "/arm/fib",
              str(r))
        statuses = [s["status"] for m in c.quiet(_pub("/arm/fib/_action/status"), 0.3)
                    for s in m["msg"]["status_list"]]
        check("status goes out on the hidden _action/status topic, through to SUCCEEDED",
              GoalStatus.EXECUTING in statuses and GoalStatus.SUCCEEDED in statuses,
              str(statuses))

        c.send({"op": "send_action_goal", "id": "g2", "action": "/arm/fib",
                "action_type": ACTION_TYPE, "args": {"order": 2}})
        r = result("g2")
        check("feedback is off unless asked for",
              r is not None and not c.quiet(lambda m: m.get("op") == "action_feedback"))

        c.send({"op": "send_action_goal", "id": "g3", "action": "/arm/fib",
                "action_type": ACTION_TYPE, "args": [1]})
        r = result("g3")
        check("list args are refused for an action type with no recorded definition",
              r is not None and r["result"] is False and r["status"] == GoalStatus.UNKNOWN)

        c.send({"op": "send_action_goal", "id": "hold1", "action": "/arm/fib",
                "action_type": ACTION_TYPE, "args": {"hold": True, "tag": "hold1"}})
        time.sleep(0.2)
        check("a held goal is still running", not goals["hold1"].done)
        c.send({"op": "cancel_action_goal", "id": "hold1", "action": "/arm/fib"})
        r = result("hold1")
        check("cancel_action_goal ends it CANCELED", r is not None
              and r["status"] == GoalStatus.CANCELED, str(r))
        c.send({"op": "cancel_action_goal", "id": "hold1", "action": "/arm/fib"})
        check("cancelling a finished goal -> status error", c.expect(_status("hold1")) is not None)
        c.send({"op": "cancel_action_goal", "id": "never", "action": "/arm/fib"})
        check("cancelling an unknown goal -> status error", c.expect(_status("never")) is not None)

        refuse["flag"] = True
        c.send({"op": "set_level", "level": "info"})
        c.send({"op": "send_action_goal", "id": "st", "action": "/arm/stubborn",
                "action_type": ACTION_TYPE, "args": {"hold": True, "tag": "st"}})
        time.sleep(0.1)
        c.send({"op": "cancel_action_goal", "id": "st", "action": "/arm/stubborn"})
        info = c.expect(lambda m: m.get("op") == "status" and m.get("id") == "st")
        check("a cancel the server rejects is reported (info) and the goal runs on",
              info is not None and info["level"] == "info" and not goals["st"].done)
        c.send({"op": "set_level", "level": "error"})

        c.send({"op": "send_action_goal", "id": "base1", "action": "/base/fib",
                "action_type": ACTION_TYPE, "args": {"hold": True, "tag": "base1"}})
        time.sleep(0.1)
        aborted = arm.abort_goals()
        r = result("st")
        check("the /reset API aborts the member's outstanding goals ABORTED",
              aborted == 1 and r is not None and r["status"] == GoalStatus.ABORTED
              and r["result"] is True, f"{aborted} {r}")
        check("...and not another member's", not goals["base1"].done)
        check("...and the aborted executor is told to stop", goals["st"].cancel_requested)
        check("abort_goals(None) aborts everything left", server.abort_goals() == 1
              and result("base1")["status"] == GoalStatus.ABORTED)

        c.send({"op": "send_action_goal", "id": "g4", "action": "/nope",
                "action_type": ACTION_TYPE})
        r = result("g4")
        check("a goal for no server -> action_result false, status UNKNOWN",
              r is not None and r["result"] is False and r["status"] == GoalStatus.UNKNOWN
              and isinstance(r["values"], str), str(r))
        c.send({"op": "send_action_goal", "id": "g5", "action": "/arm/fib",
                "action_type": "other_pkg/action/Other"})
        r = result("g5")
        check("a goal of the wrong action type -> action_result false",
              r is not None and r["result"] is False)
    finally:
        c.close()

    d = Client(port)
    d.send({"op": "send_action_goal", "id": "orphan", "action": "/arm/fib",
            "action_type": ACTION_TYPE, "args": {"hold": True, "tag": "orphan"},
            "feedback": True})
    deadline = time.monotonic() + 2
    while "orphan" not in goals and time.monotonic() < deadline:
        time.sleep(0.01)
    d.close()
    time.sleep(0.3)
    orphan = goals.get("orphan")
    check("a client that disconnects leaves its goal running",
          orphan is not None and not orphan.done and not orphan.cancel_requested)
    release.set()
    check("...which then finishes SUCCEEDED on its own",
          orphan is not None and orphan.wait(3) and orphan.status == GoalStatus.SUCCEEDED)
    release.clear()


def check_client_providers(check, port: int) -> None:
    print("client-provided services and actions")
    provider, caller = Client(port), Client(port)
    try:
        provider.send({"op": "advertise_service", "service": "/p/echo",
                       "type": "std_srvs/srv/Trigger"})
        provider.send({"op": "advertise_action", "action": "/p/fib", "type": ACTION_TYPE})
        provider.sync()
        caller.send({"op": "call_service", "service": "/p/echo", "id": "c1", "args": {}})
        req = provider.expect(lambda m: m.get("op") == "call_service")
        check("a call reaches the client that advertised the service",
              req is not None and req["service"] == "/p/echo" and req.get("id"), str(req))
        provider.send({"op": "service_response", "id": req["id"], "service": "/p/echo",
                       "values": {"success": True, "message": "pong"}, "result": True})
        r = caller.expect(lambda m: m.get("op") == "service_response" and m.get("id") == "c1")
        check("...and its response reaches the caller under the caller's id",
              r is not None and r["values"]["message"] == "pong" and r["result"] is True)
        ok, v = caller.call("/rosapi/service_node", {"service": "/p/echo"})
        check("a client's service belongs to /rosbridge_websocket",
              v.get("node") == "/rosbridge_websocket", str(v))
        ok, v = caller.call("/rosapi/action_servers")
        check("a client's action is listed", "/p/fib" in v.get("action_servers", []))

        caller.send({"op": "send_action_goal", "id": "cg", "action": "/p/fib",
                     "action_type": ACTION_TYPE, "args": {"order": 1}, "feedback": True})
        goal = provider.expect(lambda m: m.get("op") == "send_action_goal")
        check("a goal is relayed to the client-provided action server",
              goal is not None and goal["args"] == {"order": 1} and goal["feedback"] is True)
        provider.send({"op": "action_feedback", "id": goal["id"], "action": "/p/fib",
                       "values": {"sequence": [0]}})
        fb = caller.expect(lambda m: m.get("op") == "action_feedback" and m.get("id") == "cg")
        check("its feedback is relayed back", fb is not None and fb["values"] == {"sequence": [0]})
        caller.send({"op": "cancel_action_goal", "id": "cg", "action": "/p/fib"})
        cancel = provider.expect(lambda m: m.get("op") == "cancel_action_goal")
        check("a cancel is relayed to the provider under its goal id",
              cancel is not None and cancel["id"] == goal["id"])
        provider.send({"op": "action_result", "id": goal["id"], "action": "/p/fib",
                       "values": {"sequence": [0]}, "status": GoalStatus.CANCELED,
                       "result": True})
        r = caller.expect(lambda m: m.get("op") == "action_result" and m.get("id") == "cg")
        check("the provider's result is relayed with its status",
              r is not None and r["status"] == GoalStatus.CANCELED)

        provider.send({"op": "unadvertise_service", "service": "/p/echo"})
        provider.send({"op": "unadvertise_action", "action": "/p/fib"})
        provider.sync()
        ok, _ = caller.call("/p/echo")
        check("an unadvertised service is gone", not ok)
        provider.send({"op": "unadvertise_service", "id": "u2", "service": "/p/echo"})
        check("unadvertise_service twice -> status error", provider.expect(_status("u2")) is not None)
        provider.send({"op": "unadvertise_action", "id": "u3", "action": "/p/fib"})
        check("unadvertise_action twice -> status error", provider.expect(_status("u3")) is not None)
        provider.send({"op": "advertise_service", "id": "u4", "service": "/rosapi/topics",
                       "type": "rosapi_msgs/srv/Topics"})
        check("advertising a name a member provides -> status error",
              provider.expect(_status("u4")) is not None)
    finally:
        provider.close()
        caller.close()


def check_rosapi(check, server: RosBridgeServer, port: int) -> None:
    print("rosapi: all 31 services, over the wire")
    schemas.ACTIONS[schemas.canonical(ACTION_TYPE)] = (
        [("order", "int32", schemas.SCALAR)],
        [("sequence", "int32", schemas.VARIABLE)],
        [("sequence", "int32", schemas.VARIABLE)],
    )
    so101 = NamespacedBus(server, "so101")
    so101.service("/reset", lambda a: {"success": True, "message": ""}, SRV_TYPE_TRIGGER,
                  node=SIMULATOR_NODE)
    so101.on("/cmd", lambda m: None, TYPE_TWIST, node="controller")
    so101.set_param("/robot_description", "<robot/>")
    server.set_time(12.5)
    c = Client(port)
    try:
        answered = {}
        for name in ROSAPI_31:
            answered[name] = c.call(name, {"topic": "", "type": "", "service": "",
                                           "node": "", "name": "", "action": ""})[0]
        check("all 31 rosapi services answer",
              all(answered.values()), str([n for n, ok in answered.items() if not ok]))
        ok, v = c.call("/rosapi/services")
        check("...and each is listed", set(ROSAPI_31) <= set(v.get("services", [])))
        ok, v = c.call("/rosapi/service_node", {"service": "/rosapi/topics"})
        check("rosapi's services belong to /rosapi", v.get("node") == "/rosapi", str(v))

        ok, v = c.call("/rosapi/topics_for_type", {"type": TYPE_TWIST})
        check("topics_for_type", "/so101/cmd" in v.get("topics", []), str(v))
        ok, v = c.call("/rosapi/topics_and_raw_types")
        idx = v.get("topics", []).index("/so101/cmd") if "/so101/cmd" in v.get("topics", []) else -1
        check("topics_and_raw_types carries the full definition text",
              idx >= 0 and "geometry_msgs/Vector3 linear" in v["typedefs_full_text"][idx]
              and "MSG: geometry_msgs/Vector3" in v["typedefs_full_text"][idx])
        ok, v = c.call("/rosapi/services_for_type", {"type": SRV_TYPE_TRIGGER})
        check("services_for_type", "/so101/reset" in v.get("services", []), str(v))
        ok, v = c.call("/rosapi/service_providers", {"service": "/so101/reset"})
        check("/reset is provided by /simulator, not composed",
              v.get("providers") == ["/simulator"], str(v))
        ok, v = c.call("/rosapi/service_host", {"service": "/so101/reset"})
        check("service_host names a host", bool(v.get("host")), str(v))
        ok, v = c.call("/rosapi/node_details", {"node": "/simulator"})
        check("node_details for /simulator", v.get("services") == ["/so101/reset"], str(v))
        ok, v = c.call("/rosapi/node_details", {"node": "/so101/controller"})
        check("node_details for a composed vendor node",
              v.get("subscribing") == ["/so101/cmd"], str(v))
        ok, v = c.call("/rosapi/nodes")
        check("nodes include composed vendor nodes, /simulator and the runtime nodes",
              {"/so101/controller", "/simulator", "/rosapi", "/rosbridge_websocket",
               "/arm/fib_server"} <= set(v.get("nodes", [])), str(v))

        ok, v = c.call("/rosapi/action_servers")
        check("action_servers lists the registered actions",
              {"/arm/fib", "/base/fib", "/arm/stubborn"} <= set(v.get("action_servers", [])),
              str(v))
        ok, v = c.call("/rosapi/action_type", {"action": "/arm/fib"})
        check("action_type", v.get("type") == ACTION_TYPE, str(v))
        for part, field in (("goal", "order"), ("result", "sequence"), ("feedback", "sequence")):
            ok, v = c.call(f"/rosapi/action_{part}_details", {"type": ACTION_TYPE})
            td = (v.get("typedefs") or [{}])[0]
            check(f"action_{part}_details resolves from the recorded definition",
                  td.get("fieldnames") == [field]
                  and td.get("type") == f"example_interfaces/Fibonacci_{part.capitalize()}",
                  str(td))
        ok, v = c.call("/rosapi/interfaces")
        check("interfaces lists messages, services and actions in ROS 2 spelling",
              {"geometry_msgs/msg/Twist", "std_srvs/srv/Trigger", ACTION_TYPE}
              <= set(v.get("interfaces", [])))

        ok, v = c.call("/rosapi/get_time")
        check("get_time answers simulated time", v.get("time") == {"sec": 12,
                                                                   "nanosec": 500000000},
              str(v))
        ok, v = c.call("/rosapi/get_ros_version")
        check("get_ros_version answers ROS 2 Jazzy",
              v == {"version": 2, "distro": "jazzy"}, str(v))

        ok, v = c.call("/rosapi/set_param", {"name": "/cfg/gain", "value": json.dumps(1.5)})
        check("set_param takes a JSON-encoded value", v.get("successful") is True, str(v))
        ok, v = c.call("/rosapi/set_param", {"name": "/cfg/bad", "value": "{not json"})
        check("set_param refuses a value that is not JSON", v.get("successful") is False)
        ok, v = c.call("/rosapi/get_param", {"name": "/cfg/gain"})
        check("get_param returns it JSON-encoded", json.loads(v.get("value")) == 1.5, str(v))
        ok, v = c.call("/rosapi/get_param", {"name": "/cfg"})
        check("get_param on a namespace returns the subtree",
              json.loads(v.get("value")) == {"gain": 1.5}, str(v))
        ok, v = c.call("/rosapi/get_param", {"name": "/unset", "default_value": "7"})
        check("get_param on an unset name returns the ROS 2 default_value",
              v.get("value") == "7" and v.get("successful") is False, str(v))
        ok, v = c.call("/rosapi/has_param", {"name": "/cfg/gain"})
        check("has_param", v.get("exists") is True, str(v))
        ok, v = c.call("/rosapi/search_param", {"name": "cfg/gain"})
        check("search_param resolves from the root", v.get("global_name") == "/cfg/gain", str(v))
        ok, v = c.call("/rosapi/search_param", {"name": "nothing"})
        check("search_param of an unset key is empty", v.get("global_name") == "", str(v))
        ok, v = c.call("/rosapi/get_param_names")
        check("get_param_names", {"/cfg/gain", "/so101/robot_description"}
              <= set(v.get("names", [])), str(v))
        ok, v = c.call("/rosapi/delete_param", {"name": "/cfg/gain"})
        ok2, v2 = c.call("/rosapi/has_param", {"name": "/cfg/gain"})
        check("delete_param", v.get("successful") is True and v2.get("exists") is False)
        ok, v = c.call("/rosapi/delete_param", {"name": "/cfg/gain"})
        check("delete_param of an unset name is unsuccessful", v.get("successful") is False)
        ok, v = c.call("/rosapi/message_details", {"type": "geometry_msgs/msg/Twist"})
        check("message_details", [t["type"] for t in v.get("typedefs", [])][:1]
              == ["geometry_msgs/Twist"], str(v)[:120])
        ok, v = c.call("/rosapi/service_request_details", {"type": SRV_TYPE_TRIGGER})
        check("service_request_details", (v.get("typedefs") or [{}])[0].get("fieldnames") == [])
        ok, v = c.call("/rosapi/service_response_details", {"type": SRV_TYPE_TRIGGER})
        check("service_response_details",
              (v.get("typedefs") or [{}])[0].get("fieldnames") == ["success", "message"])
    finally:
        c.close()
        del schemas.ACTIONS[schemas.canonical(ACTION_TYPE)]


def run(check: Callable[[str, bool, str], None] | Callable[..., None],
        port: int = PORT) -> None:
    # The server logs every status error it sends; this file provokes dozens on purpose.
    logging.getLogger("rosbridge").setLevel(logging.CRITICAL)
    server = RosBridgeServer(host="127.0.0.1", port=port)
    server.serve_rosapi()
    server.start()
    try:
        check_ops(check, port)
        check_topics(check, server, port)
        check_latching(check, server, port)
        check_actions(check, server, port)
        check_client_providers(check, port)
        check_rosapi(check, server, port)
    finally:
        server.stop()


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail and not ok else ''}")
        if not ok:
            failures.append(name)

    run(check)
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
