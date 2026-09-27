"""An in-process rosbridge serving an SO-101 and the rig, for the embodiment's tests.

Separate from ``tests/fake_bridge.py`` because it speaks ROS 2 actions -- ``send_action_goal``
answered by ``action_result``, ``cancel_action_goal`` -- and publishes the arm's and the
rig's streams on stamped clocks. It imports nothing from the simulator.
"""

from __future__ import annotations

import base64
import json
import threading
import time

from websockets.sync.server import serve

from .rig_fixtures import AWAY, CAMERA_INFO, RESTING, render


class FakeRig:
    def __init__(self, namespace: str = "so101", *, reset_success: bool = True,
                 finish_goals: bool = False) -> None:
        self.ns = f"/{namespace}" if namespace else ""
        self.reset_success = reset_success
        self.finish_goals = finish_goals
        self.published: list[tuple[str, dict]] = []
        self.goals: list[dict] = []
        self.cancels: list[dict] = []
        self.service_calls: list[str] = []
        self._subs: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frames = {name: base64.b64encode(render(name, RESTING)).decode()
                        for name in ("overhead", "side")}
        self._wrist = self._frames["side"]

    def __enter__(self) -> "FakeRig":
        self._server = serve(self._handle, "127.0.0.1", 0, max_size=None)
        self.port = self._server.socket.getsockname()[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        threading.Thread(target=self._pump, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._server.shutdown()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    # -- server ------------------------------------------------------------------------
    def _send(self, conn, frame: dict) -> None:
        try:
            conn.send(json.dumps(frame))
        except Exception:
            pass

    def _handle(self, conn) -> None:
        try:
            for raw in conn:
                msg = json.loads(raw)
                op = msg.get("op")
                if op == "subscribe":
                    with self._lock:
                        self._subs.setdefault(msg["topic"], set()).add(conn)
                elif op == "publish":
                    with self._lock:
                        self.published.append((msg["topic"], msg["msg"]))
                elif op == "call_service":
                    self.service_calls.append(msg["service"])
                    values = {"success": self.reset_success,
                              "message": "reset" if self.reset_success else "engine busy"}
                    self._send(conn, {"op": "service_response", "id": msg.get("id"),
                                      "service": msg["service"], "values": values,
                                      "result": True})
                elif op == "send_action_goal":
                    with self._lock:
                        self.goals.append(msg)
                    if self.finish_goals:
                        self._send(conn, {"op": "action_result", "id": msg["id"],
                                          "action": msg["action"], "values": {},
                                          "status": 4, "result": True})
                elif op == "cancel_action_goal":
                    with self._lock:
                        self.cancels.append(msg)
                    self._send(conn, {"op": "action_result", "id": msg["id"],
                                      "action": msg["action"], "values": {}, "status": 5,
                                      "result": True})
        except Exception:
            pass
        finally:
            with self._lock:
                for conns in self._subs.values():
                    conns.discard(conn)

    def _publish(self, topic: str, msg: dict) -> None:
        with self._lock:
            conns = list(self._subs.get(topic, ()))
        for conn in conns:
            self._send(conn, {"op": "publish", "topic": topic, "msg": msg})

    def _pump(self) -> None:
        from robot_console.arm.kinematics import ARM_JOINTS, GRIPPER_JOINT

        names = sorted((*ARM_JOINTS, GRIPPER_JOINT))
        by_name = dict(zip((*ARM_JOINTS, GRIPPER_JOINT), AWAY))
        sim, tick = 50.0, 0
        while not self._stop.is_set():
            sim += 0.02
            tick += 1
            header = {"stamp": {"sec": int(sim), "nanosec": int(round((sim % 1) * 1e9))},
                      "frame_id": ""}
            self._publish(f"{self.ns}/joint_states", {
                "header": header, "name": names, "position": [by_name[n] for n in names],
                "velocity": [0.0] * 6, "effort": []})
            if tick % 5 == 0:
                for name in ("overhead", "side"):
                    self._publish(f"/scene/{name}/color/compressed", {
                        "header": {**header, "frame_id": f"scene/{name}"},
                        "format": "jpeg", "data": self._frames[name]})
                    self._publish(f"/scene/{name}/color/camera_info",
                                  {"header": header, **CAMERA_INFO[name]})
                self._publish(f"{self.ns}/wrist/image_raw/compressed", {
                    "header": header, "format": "jpeg", "data": self._wrist})
            time.sleep(0.01)
