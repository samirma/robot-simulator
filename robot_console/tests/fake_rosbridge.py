"""An in-process rosbridge v2 server for tests, speaking the ROS 1 or ROS 2 dialect.

It serves ``rosapi`` from a wire description (see ``wirespec.py``), streams camera frames on
subscribed camera topics, records every client op (publishes, advertises, calls, action
goals and cancels), and lets tests inject failures: a failing or hanging service, a paused
camera, a dropped connection, a silent action server, another client's running goal. A ROS 2
action server handles a goal that arrives while another of its goals runs as its spec row's
``busy`` says (``wirespec.wire_spec`` takes it from the profile): ``preempt`` (the default) ends
the earlier active goals (CANCELED, as a ros2_control controller's preempted goal does);
``reject`` refuses the new goal, answered as rosbridge_suite 2.5 and later answer a rejected
goal (result false, "Action goal was rejected", status 0), except that a "silent" server's
rejection is never answered, as before rosbridge 2.5; ``parallel`` runs it beside the others.
An accepted goal is acknowledged with one feedback message when the client asks for feedback;
a slow server's (``reply_delay``) acknowledgement and result reach the client late.
Its behaviour follows the stock servers the console was checked against (rosbridge_suite on
Noetic, Humble and Jazzy): ROS 2 hides ``_action`` endpoints from rosapi and lists actions
through ``/rosapi/action_servers``.
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


def depth_mm(i: int, n: int) -> int:
    """The depth sample of pixel ``i`` of ``n`` in a 16UC1 frame (millimetres): pixel 0 has no depth (0),
    the others rise from 70 mm (pixel 1) to 500 mm (the last), the D405's recorded range."""
    return 0 if i == 0 else 70 + round((i - 1) * 430 / (n - 2))


def image_msg(dialect: str, encoding: str = "rgb8", w: int = 8, h: int = 6, seq: int = 0) -> dict:
    ch = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4, "mono16": 2, "16UC1": 2, "yuv422": 2,
          "yuv422_yuy2": 2}.get(encoding, 3)
    step = w * ch
    if encoding == "16UC1":     # little-endian millimetres, as realsense2_camera publishes the aligned depth
        data = b"".join(depth_mm(i, w * h).to_bytes(2, "little") for i in range(w * h))
    else:
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
        # ros2 action -> "silent" (no feedback, no preemption result, no answer to a rejected goal)
        # | "abort" (goals end ABORTED)
        self.action_behaviour: Dict[str, str] = {}
        # ros2 action -> seconds by which what the server sends about a goal arriving now (its
        # acknowledging feedback, its result) reaches the client late: a slow server
        self.reply_delay: Dict[str, float] = {}
        self.paused: set = set()
        # topic -> message a sensor topic publishes (a spec row's ``sample``): periodically at the
        # row's ``rate_hz``, or, for a row marked ``latched``, once to each new subscriber and again
        # whenever a test changes it with ``set_sample`` (a transient-local publisher)
        self.samples: Dict[str, dict] = {r["name"]: copy.deepcopy(r["sample"])
                                         for r in spec["topics"] if r.get("sample") is not None}
        self.frame_encoding: Dict[str, str] = {}
        # ros2 goal id -> {client, action, t_sent, t_end, done, t_done} (client None: another client's)
        self.goals: Dict[str, dict] = {}
        self.rejected: List[dict] = []            # goals a rejecting ros2 server refused: {id, action, t}
        # services a test declares read-only queries (e.g. a profile's polled joint-position read):
        # calls of them are not counted by command_ops()
        self.read_only: set = set()
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

    def set_sample(self, topic: str, msg: dict) -> None:
        """Change the message a sensor topic publishes (a latched topic republishes it at once)."""
        self.samples[topic] = copy.deepcopy(msg)
        if self._row(topic).get("latched"):
            self._broadcast(topic, self.samples[topic])

    def _row(self, topic: str) -> dict:
        return next(r for r in self.spec["topics"] if r["name"] == topic)

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

    def occupy(self, action: str, seconds: float) -> str:
        """Another client's goal of ROS 2 ``action``, running for ``seconds`` (the server is busy)."""
        gid = f"other-{len(self.goals)}"
        now = time.monotonic()
        self.goals[gid] = {"client": None, "action": action, "t_sent": now, "t_end": now + seconds,
                           "done": False, "t_done": None}
        return gid

    def ops_of(self, op: str) -> List[dict]:
        with self.ops_lock:
            return [o for o in self.ops if o.get("op") == op]

    def command_ops(self) -> List[dict]:
        """Every op that could command a robot (anything but rosapi calls, subscriptions and calls
        of the services declared ``read_only``)."""
        with self.ops_lock:
            return [o for o in self.ops if o.get("op") in
                    ("publish", "advertise", "send_action_goal", "cancel_action_goal",
                     "advertise_service", "advertise_action")
                    or (o.get("op") == "call_service" and not o["service"].startswith("/rosapi/")
                        and o["service"] not in self.read_only)]

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
        rosapi = ["topics", "services", "service_type", "nodes", "get_param_names", "topic_type"]
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
            if msg["topic"] in self.samples and self._row(msg["topic"]).get("latched") and msg["topic"] not in self.paused:
                c.send({"op": "publish", "topic": msg["topic"], "msg": self.samples[msg["topic"]]})
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
            busy = spec.get("busy", "preempt")
            running = [gid for gid, g in list(self.goals.items()) if g["action"] == msg["action"] and not g["done"]]
            if busy == "reject" and running:
                self.rejected.append({"id": msg.get("id"), "action": msg["action"], "t": time.monotonic()})
                if not silent:
                    c.send({"op": "action_result", "id": msg.get("id"), "action": msg["action"],
                            "values": "Action goal was rejected", "status": 0, "result": False})
                return
            if busy == "preempt" and not silent:
                for gid in running:
                    self._finish_goal(gid, 5)               # preempted
            now = time.monotonic()
            delay = float(self.reply_delay.get(msg["action"], 0.0))
            self.goals[msg["id"]] = {"client": c, "action": msg["action"], "t_sent": now,
                                     "t_end": now + float(spec.get("exec_s", 3.0)),
                                     "done": False, "t_done": None, "delay": delay}
            if msg.get("feedback") and not silent:
                self._reply(c, {"op": "action_feedback", "id": msg["id"], "action": msg["action"], "values": {}}, delay)
        elif op == "cancel_action_goal" and self.dialect == "ros2":
            g = self.goals.get(msg.get("id"))
            if g and not g["done"]:
                self._finish_goal(msg["id"], 5)

    def _finish_goal(self, gid: str, status: int) -> None:
        g = self.goals[gid]
        if g["done"]:
            return
        g["done"] = True
        g["t_done"] = time.monotonic()
        if g["client"] is not None:
            self._reply(g["client"], {"op": "action_result", "id": gid, "action": g["action"], "values": {},
                                      "status": status, "result": True}, g.get("delay", 0.0))

    @staticmethod
    def _reply(c: Client, msg: dict, delay: float) -> None:
        if delay > 0:
            t = threading.Timer(delay, c.send, args=(msg,))
            t.daemon = True
            t.start()
        else:
            c.send(msg)

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
        sensors = [r for r in self.spec["topics"] if r.get("sample") is not None]
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
            for r in sensors:
                n = r["name"]
                if n in self.paused or r.get("latched") or now < next_at.get(n, 0):
                    continue
                next_at[n] = now + 1.0 / float(r.get("rate_hz") or 5)
                self._broadcast(n, self.samples[n])
            for gid, g in list(self.goals.items()):
                if not g["done"] and now >= g["t_end"]:
                    self._finish_goal(gid, 6 if self.action_behaviour.get(g["action"]) == "abort" else 4)
            time.sleep(0.01)
