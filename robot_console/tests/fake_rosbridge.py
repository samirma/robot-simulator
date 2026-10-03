"""An in-process rosbridge v2 server for tests, speaking the ROS 1 or ROS 2 dialect.

It serves ``rosapi`` from a wire description (see ``wirespec.py``), streams camera frames on
subscribed camera topics, records every client op (publishes, advertises, calls, action
goals and cancels), and lets tests inject failures: a failing or hanging service, a paused
camera, a dropped connection, a silent action server. A ROS 2 goal preempts the earlier
active goals of its action (they end CANCELED, as a ros2_control controller's preempted goal
does) and, when the client asks for feedback, is acknowledged with one feedback message. Its behaviour follows the stock servers the console was
checked against (rosbridge_suite on Noetic, Humble and Jazzy): ROS 2 hides ``_action``
endpoints from rosapi and lists actions through ``/rosapi/action_servers``.
"""

from __future__ import annotations

import base64
import copy
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from websockets.sync.server import serve

INFRA = {
    "ros1": {"topics": {"/rosout": "rosgraph_msgs/Log", "/rosout_agg": "rosgraph_msgs/Log",
                        "/client_count": "std_msgs/Int32",
                        "/connected_clients": "rosbridge_msgs/ConnectedClients"},
             "services": {"/rosbridge_websocket/get_loggers": "roscpp/GetLoggers",
                          "/rosbridge_websocket/set_logger_level": "roscpp/SetLoggerLevel",
                          "/rosapi/get_loggers": "roscpp/GetLoggers"}},
    "ros2": {"topics": {"/rosout": "rcl_interfaces/msg/Log",
                        "/parameter_events": "rcl_interfaces/msg/ParameterEvent",
                        "/client_count": "std_msgs/msg/Int32",
                        "/connected_clients": "rosbridge_msgs/msg/ConnectedClients"},
             "services": {"/rosbridge_websocket/get_parameters": "rcl_interfaces/srv/GetParameters",
                          "/rosapi_params/list_parameters": "rcl_interfaces/srv/ListParameters",
                          "/rosapi/describe_parameters": "rcl_interfaces/srv/DescribeParameters"}},
}


def image_msg(dialect: str, encoding: str = "rgb8", w: int = 8, h: int = 6, seq: int = 0) -> dict:
    ch = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4, "mono16": 2, "yuv422": 2,
          "yuv422_yuy2": 2}.get(encoding, 3)
    step = w * ch
    data = bytes(((i + seq * 13) * 7) % 256 for i in range(step * h))
    stamp = {"sec": seq, "nanosec": 0} if dialect == "ros2" else {"secs": seq, "nsecs": 0}
    return {"header": {"stamp": stamp, "frame_id": "camera"}, "height": h, "width": w,
            "encoding": encoding, "is_bigendian": 0, "step": step,
            "data": base64.b64encode(data).decode()}


class Client:
    def __init__(self, conn, server: "FakeRosbridge") -> None:
        self.conn = conn
        self.server = server
        self.subs: Dict[str, str] = {}          # sid -> topic
        self.lock = threading.Lock()
        self.closed = False

    def send(self, msg: dict) -> None:
        if self.closed:
            return
        try:
            with self.lock:
                self.conn.send(json.dumps(msg))
        except Exception:  # noqa: BLE001
            self.closed = True


class FakeRosbridge:
    def __init__(self, spec: dict, port: int = 0) -> None:
        self.spec = copy.deepcopy(spec)
        self.dialect = spec["dialect"]
        self.ops: List[dict] = []                 # every op received, in order
        self.ops_lock = threading.Lock()
        self.clients: List[Client] = []
        self.service_behaviour: Dict[str, Any] = {}   # name -> "fail" | "hang" | callable
        # ros2 action -> "silent" (no feedback, no preemption result) | "abort" (goals end ABORTED)
        self.action_behaviour: Dict[str, str] = {}
        self.paused: set = set()
        self.frame_encoding: Dict[str, str] = {}
        self.goals: Dict[str, dict] = {}          # ros2 goal id -> {client, action, t_end, done}
        self._stop = threading.Event()
        self._server = serve(self._handle, "127.0.0.1", port, compression=None, max_size=None)
        self.port = self._server.socket.getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._pump = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump.start()

    # ------------------------------------------------------------ control
    def close(self) -> None:
        self._stop.set()
        self.drop_all()
        self._server.shutdown()

    def drop_all(self) -> None:
        """Close every client connection (connection loss seen by the console)."""
        for c in list(self.clients):
            c.closed = True
            try:
                c.conn.close()
            except Exception:  # noqa: BLE001
                pass

    def set_service(self, name: str, behaviour: Any) -> None:
        self.service_behaviour[name] = behaviour

    def published(self, topic: Optional[str] = None) -> List[dict]:
        with self.ops_lock:
            return [o for o in self.ops if o.get("op") == "publish" and (topic is None or o["topic"] == topic)]

    def calls(self, service: Optional[str] = None) -> List[dict]:
        with self.ops_lock:
            return [o for o in self.ops if o.get("op") == "call_service"
                    and (service is None or o["service"] == service)
                    and not o["service"].startswith("/rosapi/")]

    def open_clients(self) -> int:
        """Websocket connections currently open."""
        return sum(1 for c in self.clients if not c.closed)

    def ops_of(self, op: str) -> List[dict]:
        with self.ops_lock:
            return [o for o in self.ops if o.get("op") == op]

    def command_ops(self) -> List[dict]:
        """Every op that could command a robot (anything but rosapi calls and subscriptions)."""
        with self.ops_lock:
            return [o for o in self.ops if o.get("op") in
                    ("publish", "advertise", "send_action_goal", "cancel_action_goal",
                     "advertise_service", "advertise_action")
                    or (o.get("op") == "call_service" and not o["service"].startswith("/rosapi/"))]

    def wait_for(self, pred: Callable[[], bool], timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            time.sleep(0.02)
        return pred()

    # ------------------------------------------------------------ rosapi
    def _topics(self) -> Dict[str, str]:
        t = dict(INFRA[self.dialect]["topics"])
        for r in self.spec["topics"]:
            t[r["name"]] = r["type"]
        if self.dialect == "ros1":
            for a in self.spec["actions"]:
                pkg, base = a["type"].split("/")
                base = base[:-6] if base.endswith("Action") else base
                for sfx, ty in (("goal", f"{pkg}/{base}ActionGoal"), ("cancel", "actionlib_msgs/GoalID"),
                                ("status", "actionlib_msgs/GoalStatusArray"),
                                ("feedback", f"{pkg}/{base}ActionFeedback"),
                                ("result", f"{pkg}/{base}ActionResult")):
                    t[f"{a['name']}/{sfx}"] = ty
        return t

    def _services(self) -> Dict[str, str]:
        s = dict(INFRA[self.dialect]["services"])
        for r in self.spec["services"]:
            s[r["name"]] = r["type"]
        rosapi = ["topics", "services", "service_type", "nodes", "get_param_names", "topic_type", "publishers"]
        if self.dialect == "ros2":
            rosapi += ["action_servers", "get_ros_version"]
        for n in rosapi:
            s[f"/rosapi/{n}"] = f"rosapi{'_msgs/srv' if self.dialect == 'ros2' else ''}/{n}"
        return s

    def _rosapi(self, name: str, args: dict) -> Optional[dict]:
        if name == "/rosapi/topics":
            t = self._topics()
            return {"topics": list(t), "types": list(t.values())}
        if name == "/rosapi/services":
            return {"services": list(self._services())}
        if name == "/rosapi/service_type":
            return {"type": self._services().get(args.get("service"), "")}
        if name == "/rosapi/topic_type":
            return {"type": self._topics().get(args.get("topic"), "")}
        if name == "/rosapi/nodes":
            return {"nodes": ["/rosbridge_websocket", "/rosapi", "/robot"]}
        if name == "/rosapi/publishers":
            # A row may list its publishing nodes; else the robot publishes its `out` topics.
            row = next((r for r in self.spec["topics"] if r["name"] == args.get("topic")), None)
            if row is None:
                return {"publishers": []}
            pubs = row.get("publishers", ["/robot"] if row.get("direction", "out") == "out" else [])
            return {"publishers": list(pubs)}
        if name == "/rosapi/action_servers" and self.dialect == "ros2":
            return {"action_servers": [a["name"] for a in self.spec["actions"]]}
        if name == "/rosapi/get_ros_version" and self.dialect == "ros2":
            return {"version": 2, "distro": "fake"}
        return None

    # ------------------------------------------------------------ protocol
    def _record(self, msg: dict) -> None:
        with self.ops_lock:
            self.ops.append(dict(msg, _t=time.monotonic()))

    def _handle(self, conn) -> None:
        c = Client(conn, self)
        self.clients.append(c)
        try:
            for raw in conn:
                msg = json.loads(raw)
                self._record(msg)
                self._dispatch(c, msg)
        except Exception:  # noqa: BLE001
            pass
        finally:
            c.closed = True

    def _dispatch(self, c: Client, msg: dict) -> None:
        op = msg.get("op")
        if op == "subscribe":
            c.subs[msg.get("id") or msg["topic"]] = msg["topic"]
        elif op == "unsubscribe":
            c.subs.pop(msg.get("id") or msg["topic"], None)
        elif op == "call_service":
            threading.Thread(target=self._call, args=(c, msg), daemon=True).start()
        elif op == "send_action_goal" and self.dialect == "ros2":
            spec = next((a for a in self.spec["actions"] if a["name"] == msg["action"]), None)
            if spec is None:
                c.send({"op": "action_result", "id": msg.get("id"), "action": msg["action"],
                        "values": "Action does not exist", "status": 6, "result": False})
                return
            silent = self.action_behaviour.get(msg["action"]) == "silent"
            if not silent:
                for gid, g in list(self.goals.items()):
                    if g["action"] == msg["action"] and not g["done"]:
                        self._finish_goal(gid, 5)           # preempted
            self.goals[msg["id"]] = {"client": c, "action": msg["action"],
                                     "t_end": time.monotonic() + float(spec.get("exec_s", 3.0)),
                                     "done": False}
            if msg.get("feedback") and not silent:
                c.send({"op": "action_feedback", "id": msg["id"], "action": msg["action"], "values": {}})
        elif op == "cancel_action_goal" and self.dialect == "ros2":
            g = self.goals.get(msg.get("id"))
            if g and not g["done"]:
                self._finish_goal(msg["id"], 5)

    def _finish_goal(self, gid: str, status: int) -> None:
        g = self.goals[gid]
        if g["done"]:
            return
        g["done"] = True
        g["client"].send({"op": "action_result", "id": gid, "action": g["action"], "values": {},
                          "status": status, "result": True})

    def _broadcast(self, topic: str, msg: dict) -> None:
        for c in list(self.clients):
            if topic in c.subs.values():
                c.send({"op": "publish", "topic": topic, "msg": msg})

    def _call(self, c: Client, msg: dict) -> None:
        name = msg["service"]
        args = msg.get("args") or {}
        resp = {"op": "service_response", "id": msg.get("id"), "service": name}
        if name.startswith("/rosapi/"):
            vals = self._rosapi(name, args)
            if vals is None:
                c.send(dict(resp, values=f"Service {name} does not exist", result=False))
            else:
                c.send(dict(resp, values=vals, result=True))
            return
        if self.dialect == "ros2" and name.endswith("/_action/cancel_goal"):
            action = name[: -len("/_action/cancel_goal")]
            if action in [a["name"] for a in self.spec["actions"]]:
                beh = self.service_behaviour.get(name)
                if beh == "fail":
                    c.send(dict(resp, values="cancel failed", result=False))
                    return
                if beh == "hang":
                    return
                canceling = []
                for gid, g in list(self.goals.items()):
                    if g["action"] == action and not g["done"]:
                        self._finish_goal(gid, 5)
                        canceling.append({"goal_id": {"uuid": "AAAAAAAAAAAAAAAAAAAAAA=="}})
                c.send(dict(resp, values={"return_code": 0, "goals_canceling": canceling}, result=True))
                return
        known = {r["name"] for r in self.spec["services"]}
        if name not in known:
            c.send(dict(resp, values=f"Service {name} does not exist", result=False))
            return
        beh = self.service_behaviour.get(name)
        if beh == "hang":
            return
        if beh == "fail":
            c.send(dict(resp, values="service call failed", result=False))
            return
        if callable(beh):
            try:
                c.send(dict(resp, values=beh(args), result=True))
            except Exception as exc:  # noqa: BLE001
                c.send(dict(resp, values=str(exc), result=False))
            return
        c.send(dict(resp, values={}, result=True))

    # ------------------------------------------------------------ periodic output
    def _pump_loop(self) -> None:
        seq = 0
        cams = [r for r in self.spec["topics"] if r.get("camera")]
        next_at: Dict[str, float] = {}
        while not self._stop.is_set():
            now = time.monotonic()
            for r in cams:
                if r["name"] in self.paused:
                    continue
                if now >= next_at.get(r["name"], 0):
                    next_at[r["name"]] = now + 1.0 / float(r.get("rate_hz") or 10)
                    seq += 1
                    enc = self.frame_encoding.get(r["name"], r.get("encoding", "rgb8"))
                    self._broadcast(r["name"], image_msg(self.dialect, enc, seq=seq))
            for gid, g in list(self.goals.items()):
                if not g["done"] and now >= g["t_end"]:
                    self._finish_goal(gid, 6 if self.action_behaviour.get(g["action"]) == "abort" else 4)
            time.sleep(0.01)
