#!/usr/bin/env python
"""Standalone check of the AiNex's ROS surface against `robots_specs/ainex/ros.yml`.

    python robots/ainex/test_ros.py [--port 9391]

Over a real websocket, on the bare contract (no namespace), with the simulation stepping
in real time on the main thread as an engine steps it. Two halves:

* **the interface** -- every topic, service and parameter the ROS file lists is on the
  wire with its type and node, nothing else is (no `/joint_states`, `/tf`, `/scan`,
  `robot_description` or per-joint controllers beyond the head), the frames are the
  drivers' and each periodic topic runs at its declared rate;
* **the behaviour** -- `/walking/command`'s control gate and blocking `stop`,
  `period_times`, `/app/set_action`, `/walking/init_pose`, `/app/enter`,
  `/app/set_running`, `/joy`, the servo bus, and no watchdog.

Everything here misbehaves *quietly* when wrong: a gated command still answers true, a
stop that does not block returns while the robot is still stepping, a missing topic is
just a topic nobody receives.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import websockets.sync.client as wsc
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FAIL: list[str] = []
REPO = Path(__file__).resolve().parents[4]


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(label)


class Client:
    """Just enough rosbridge client: a reader thread files publishes by topic and
    service responses by id, so a check can wait on either without draining the other."""

    def __init__(self, url: str) -> None:
        self._ws = wsc.connect(url, open_timeout=5, max_size=None)
        self._id = 0
        self._lock = threading.Lock()
        self._responses: dict[str, queue.Queue] = {}
        self.messages: dict[str, list[tuple[float, dict]]] = {}
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        try:
            for raw in self._ws:
                frame = json.loads(raw)
                if frame.get("op") == "publish":
                    with self._lock:
                        self.messages.setdefault(frame["topic"], []).append(
                            (time.monotonic(), frame["msg"]))
                elif frame.get("op") == "service_response":
                    self._responses.setdefault(frame.get("id"), queue.Queue()).put(frame)
        except Exception:
            pass

    def close(self) -> None:
        self._ws.close()

    def send(self, frame: dict) -> None:
        self._ws.send(json.dumps(frame))

    def publish(self, topic: str, msg: dict) -> None:
        self.send({"op": "publish", "topic": topic, "msg": msg})

    def subscribe(self, topic: str) -> None:
        self.send({"op": "subscribe", "topic": topic})

    def unsubscribe(self, topic: str) -> None:
        self.send({"op": "unsubscribe", "topic": topic})

    def call(self, service: str, args: dict | None = None, timeout: float = 10.0) -> dict:
        with self._lock:
            self._id += 1
            cid = f"call-{self._id}"
            box = self._responses.setdefault(cid, queue.Queue())
        self.send({"op": "call_service", "service": service, "args": args or {}, "id": cid})
        return box.get(timeout=timeout)

    def values(self, service: str, args: dict | None = None) -> dict:
        return self.call(service, args).get("values") or {}

    def clear(self, topic: str) -> None:
        with self._lock:
            self.messages[topic] = []

    def received(self, topic: str) -> list[tuple[float, dict]]:
        with self._lock:
            return list(self.messages.get(topic, []))


def interface_checks(client: Client, ros: dict, t) -> None:
    print("interface:")
    names = client.values("/rosapi/topics")
    served = dict(zip(names.get("topics", []), names.get("types", [])))
    want = {e["name"]: e["type"] for e in ros["topics"]}
    check("rosapi lists exactly the ROS file's topics", set(served) == set(want),
          f"missing {sorted(set(want) - set(served))}; extra {sorted(set(served) - set(want))}")
    wrong = {n: (served[n], want[n]) for n in want if n in served and served[n] != want[n]}
    check("...each with the file's type", not wrong, str(wrong))

    services = set(client.values("/rosapi/services").get("services", []))
    own = {s for s in services if not s.startswith("/rosapi/")}
    want_srv = {e["name"] for e in ros["services"]}
    check("rosapi lists exactly the ROS file's services, beside its own",
          own == want_srv, f"missing {sorted(want_srv - own)}; extra {sorted(own - want_srv)}")
    bad_types = [e["name"] for e in ros["services"]
                 if client.values("/rosapi/service_type", {"service": e["name"]}).get("type")
                 != e["type"]]
    check("...each with the file's type", not bad_types, str(bad_types))

    bad_nodes = []
    for e in ros["topics"]:
        role = "publishers" if e["direction"] == "out" else "subscribers"
        nodes = client.values(f"/rosapi/{role}", {"topic": e["name"]}).get(role, [])
        if f"/{e['node']}" not in nodes:
            bad_nodes.append((e["name"], role, nodes))
    for e in ros["services"]:
        node = client.values("/rosapi/service_node", {"service": e["name"]}).get("node")
        if node != f"/{e['node']}":
            bad_nodes.append((e["name"], "service", node))
    check("every name is provided by the node the file names", not bad_nodes, str(bad_nodes[:4]))

    params = set(client.values("/rosapi/get_param_names").get("names", []))
    want_params = {e["name"] for e in ros["parameters"]}
    check("the parameter server holds exactly the file's parameters", params == want_params,
          f"missing {sorted(want_params - params)}; extra {sorted(params - want_params)}")
    freq = client.values("/rosapi/get_param", {"name": "/ros_robot_controller/freq"})
    check("a parameter's value comes back JSON-encoded", json.loads(freq.get("value", "0")) == 100)

    absent = ["/joint_states", "/tf", "/tf_static", "/scan", "/l_knee_controller/command",
              "/camera/depth/image_raw", "/camera/rgb/camera_info"]
    check("nothing the boot chain does not present", not set(absent) & set(served),
          str(sorted(set(absent) & set(served))))
    check("no robot_description", "/robot_description" not in params)


def rate_checks(client: Client, t) -> None:
    print("\nrates and frames:")
    periodic = dict(t.RATES_HZ)
    for topic in periodic:
        client.subscribe(topic)
    time.sleep(1.0)
    for topic in periodic:
        client.clear(topic)
    window = 3.0
    time.sleep(window)
    for topic, hz in periodic.items():
        stamps = [s for s, _ in client.received(topic)]
        measured = len(stamps) / window
        gap = max(np.diff(stamps)) if len(stamps) > 1 else math.inf
        check(f"{topic} at {hz:g} Hz", abs(measured - hz) <= 0.1 * hz and gap < 3.0 / hz,
              f"{measured:.1f} Hz, worst gap {gap * 1000:.0f} ms")
    imu = client.received(t.TOPIC_IMU)[-1][1]
    check("/imu is framed imu_link", imu["header"]["frame_id"] == "imu_link")
    raw = client.received(t.TOPIC_IMU_RAW)[-1][1]
    check("imu_raw carries no orientation, as the board sends it",
          raw["orientation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0})
    corrected = client.received(t.TOPIC_IMU_CORRECTED)[-1][1]["linear_acceleration"]
    check("the calibrated accelerometer reads gravity standing still",
          abs(math.hypot(corrected["x"], corrected["y"], corrected["z"]) - 9.80665) < 0.01,
          str(corrected))
    info = client.received(t.TOPIC_CAMERA_INFO)[-1][1]
    check("the camera is framed camera, at 640x480",
          info["header"]["frame_id"] == "camera" and (info["width"], info["height"]) == (640, 480))
    image = client.received(t.TOPIC_CAMERA_RAW)[-1][1]
    check("image_raw is rgb8, as usb_cam converts yuyv",
          image["encoding"] == "rgb8" and image["step"] == 1920)
    jpeg = client.received(t.TOPIC_CAMERA)[-1][1]
    check("the compressed companion is image_transport's", "jpeg" in jpeg["format"])
    for topic in periodic:
        client.unsubscribe(topic)


def behaviour_checks(client: Client, t, sim) -> None:  # noqa: PLR0915
    from ros_surfaces.ainex import servos

    def walking() -> bool:
        return bool(client.values(t.SRV_IS_WALKING).get("state"))

    def cmd(command: str) -> dict:
        return client.values(t.SRV_WALKING_COMMAND, {"command": command})

    def x() -> float:
        return sim.base_x()

    forward = {"period_time": 400.0, "dsp_ratio": 0.2, "x_move_amplitude": 0.02,
               "init_z_offset": 0.025, "z_move_amplitude": 0.02, "y_swap_amplitude": 0.02,
               "z_swap_amplitude": 0.006, "arm_swing_gain": 0.5, "hip_pitch_offset": 15.0}

    print("\n/walking/command:")
    check("an unknown command still answers result true", cmd("nonsense").get("result") is True)
    client.subscribe(t.TOPIC_IS_WALKING)
    client.publish(t.TOPIC_SET_WALKING_PARAM, forward)
    time.sleep(0.2)
    check("period_time is milliseconds on the wire",
          client.values(t.SRV_GET_WALKING_PARAM)["parameters"]["period_time"] == 400.0)

    cmd("disable_control")
    reply = cmd("start")
    time.sleep(0.6)
    check("start while control is disabled answers true and does nothing",
          reply.get("result") is True and not walking())
    cmd("enable_control")
    before = x()
    cmd("start")
    time.sleep(2.0)
    moved = x() - before
    check("with control enabled, start walks forward at 4A/T",
          abs(moved - 0.2 * 2.0) < 0.25 * 0.4, f"moved {moved:.3f} m in 2 s")
    check("is_walking is true while walking", walking())

    cmd("disable_control")
    frozen = x()
    reply = cmd("stop")
    time.sleep(0.5)
    check("stop is ignored while control is disabled -- the gait is frozen, not stopped",
          reply.get("result") is True and walking() and abs(x() - frozen) < 0.005)
    cmd("enable_control")
    started = time.monotonic()
    cmd("stop")
    blocked = time.monotonic() - started
    at_stop = x()
    check("stop blocks until the gait has halted", not walking(), f"returned after {blocked:.2f} s")
    time.sleep(0.5)
    check("...and the robot stays put", abs(x() - at_stop) < 0.005, f"{x() - at_stop:+.4f} m")
    events = [m["data"] for _, m in client.received(t.TOPIC_IS_WALKING)]
    check("/walking/is_walking is published on transitions only", events[-2:] == [True, False],
          str(events))

    cmd("disable")
    before = x()
    cmd("start")
    time.sleep(1.0)
    check("start after disable walks (start implies enable)", x() - before > 0.05)
    cmd("stop")

    print("\nperiod_times:")
    client.clear(t.TOPIC_IS_WALKING)
    client.publish(t.TOPIC_SET_WALKING_PARAM, {**forward, "period_times": 2})
    time.sleep(0.2)
    started = time.monotonic()
    cmd("start")
    # As GaitManager does: wait for the gait to report it is moving, then for it to stop.
    deadline = time.monotonic() + 1.0
    while not walking() and time.monotonic() < deadline:
        time.sleep(0.01)
    deadline = time.monotonic() + 3.0
    while walking() and time.monotonic() < deadline:
        time.sleep(0.05)
    took = time.monotonic() - started
    check("period_times 2 walks two cycles and stops by itself",
          not walking() and 0.6 < took < 1.4, f"{took:.2f} s at 400 ms a cycle")
    check("get_param reports period_times as 0",
          client.values(t.SRV_GET_WALKING_PARAM)["parameters"]["period_times"] == 0)
    client.publish(t.TOPIC_SET_WALKING_PARAM, forward)

    print("\nno watchdog:")
    cmd("start")
    time.sleep(0.3)
    client.close()
    time.sleep(1.0)
    fresh = Client(sim.url)
    check("a client that disconnects mid-walk leaves the robot walking",
          bool(fresh.values(t.SRV_IS_WALKING).get("state")))
    fresh.values(t.SRV_WALKING_COMMAND, {"command": "stop"})
    return fresh


def behaviour_checks_2(client: Client, t, sim) -> None:  # noqa: PLR0915
    from ros_surfaces.ainex import servos

    def walking() -> bool:
        return bool(client.values(t.SRV_IS_WALKING).get("state"))

    def cmd(command: str) -> dict:
        return client.values(t.SRV_WALKING_COMMAND, {"command": command})

    def counts(*ids: int) -> dict[int, int]:
        reply = client.values(t.SRV_BUS_SERVO_GET, {"id": list(ids)})
        return {p["id"]: p["position"] for p in reply.get("position", [])}

    print("\nservo bus and head:")
    for sid, joint, count in ((23, "head_pan", 700), (13, "l_sho_pitch", 700)):
        client.publish(t.TOPIC_BUS_SERVO_SET,
                       {"duration": 0.2, "position": [{"id": sid, "position": count}]})
        time.sleep(1.0)
        got = counts(sid).get(sid)
        check(f"servo {sid} ({joint}) reaches count {count}, read back over get_position",
              got is not None and abs(got - count) <= 12, f"read {got}")
    check("get_position answers only the ids asked for", set(counts(5, 6)) == {5, 6})
    client.publish(t.TOPIC_HEAD_PAN, {"position": 0.5, "duration": 0.2})
    client.publish(t.TOPIC_HEAD_TILT, {"position": -0.3, "duration": 0.2})
    time.sleep(1.0)
    pan = servos.count_to_angle("head_pan", counts(23)[23])
    tilt = servos.count_to_angle("head_tilt", counts(24)[24])
    check("the head controllers move the head", abs(pan - 0.5) < 0.03 and abs(tilt + 0.3) < 0.03,
          f"pan {pan:+.3f} tilt {tilt:+.3f}")

    print("\n/walking/init_pose:")
    cmd("start")
    time.sleep(0.5)
    client.values(t.SRV_INIT_POSE)
    check("init_pose stops the walk before it returns", not walking())
    time.sleep(1.0)
    knee = servos.count_to_angle("l_knee", counts(5)[5])
    pan = servos.count_to_angle("head_pan", counts(23)[23])
    check("...puts the body in the init pose and leaves the head",
          abs(knee - servos.INIT_POSE["l_knee"]) < 0.05 and abs(pan - 0.5) < 0.03,
          f"l_knee {knee:+.3f}, head_pan {pan:+.3f}")

    print("\n/app/set_action:")
    # `wave` lifts the left arm (l_sho_pitch to 1.4) for about two seconds.
    client.publish(t.TOPIC_APP_ACTION, {"data": "wave"})
    time.sleep(0.3)
    cmd("start")
    time.sleep(0.4)
    check("walking commands are ignored while an action group plays", not walking())
    shoulder = servos.count_to_angle("l_sho_pitch", counts(13)[13])
    check("the group plays", shoulder > 0.5, f"l_sho_pitch {shoulder:+.3f}")
    time.sleep(2.5)
    shoulder = servos.count_to_angle("l_sho_pitch", counts(13)[13])
    check("...and ends in the init pose",
          abs(shoulder - servos.INIT_POSE["l_sho_pitch"]) < 0.05, f"l_sho_pitch {shoulder:+.3f}")
    cmd("start")
    time.sleep(0.4)
    check("control is enabled again afterwards", walking())
    cmd("stop")

    print("\n/app/enter and /app/set_running:")
    reply = client.values(t.SRV_APP_ENTER, {"data": 1})
    time.sleep(1.0)
    pan = servos.count_to_angle("head_pan", counts(23)[23])
    check("enter 1 (control) runs init_pose and centres the head",
          reply.get("success") is True and abs(pan) < 0.03, f"head_pan {pan:+.3f}")
    reply = client.values(t.SRV_APP_SET_RUNNING, {"data": True})
    check("set_running true in a mode answers true", reply.get("success") is True)
    cmd("start")
    time.sleep(0.5)
    reply = client.values(t.SRV_APP_SET_RUNNING, {"data": False})
    check("set_running false stops the gait, and answers false as the vendor does",
          not walking() and reply.get("success") is False)
    client.values(t.SRV_APP_ENTER, {"data": 7})
    cmd("start")
    time.sleep(0.5)
    check("enter 7 (fall_rise) leaves control disabled: start does nothing", not walking())
    client.values(t.SRV_APP_ENTER, {"data": 0})
    check("enter 0 (idle) enables control again, and set_running is refused there",
          client.values(t.SRV_APP_SET_RUNNING, {"data": True}).get("success") is False)
    cmd("start")
    time.sleep(0.3)
    check("...so start walks", walking())
    cmd("stop")
    check("heartbeat answers", client.values(t.SRV_APP_HEARTBEAT, {"data": True}).get("success"))

    print("\n/joy (joystick_control):")
    before = sim.base_x()
    axes = [0.0] * 8
    client.publish(t.TOPIC_JOY, {"axes": [0.0, 1.0] + axes[2:], "buttons": [0] * 21})
    time.sleep(1.5)
    check("the left stick walks forward", walking() and sim.base_x() - before > 0.05,
          f"moved {sim.base_x() - before:+.3f} m")
    client.publish(t.TOPIC_JOY, {"axes": axes, "buttons": [0] * 21})
    time.sleep(0.2)
    check("releasing the stick stops the gait", not walking())

    print("\nsensor node:")
    client.subscribe(t.TOPIC_BUTTON_STATE)
    time.sleep(0.5)
    client.values(t.SRV_BUTTON_ENABLE, {"data": False})
    time.sleep(0.2)
    client.clear(t.TOPIC_BUTTON_STATE)
    time.sleep(0.5)
    check("button/enable false stops the button stream",
          not client.received(t.TOPIC_BUTTON_STATE))
    client.values(t.SRV_BUTTON_ENABLE, {"data": True})


class Sim:
    """The simulation, stepped in real time on the main thread, as an engine steps it."""

    def __init__(self, port: int, control_hz: float) -> None:
        import ainex_model
        from robots.ainex import AiNexRobot, AiNexRobotConfig, AiNexRobotView
        from robots.ainex.ros_surface import serve_ros

        config = AiNexRobotConfig()
        ns = config.robot_namespace
        spec = mujoco.MjSpec()
        spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1],
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1],
                                rgba=[0.55, 0.56, 0.58, 1])
        AiNexRobot.add_robot_to_scene(config, spec, prefix=ns, pos=[0.0, 0.0, 0.0],
                                      quat=[1.0, 0.0, 0.0, 0.0])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        ainex_model.stand(self.model, self.data, ns)
        mujoco.mj_forward(self.model, self.data)
        self.view = AiNexRobotView(self.data, ns)
        self.control_hz = control_hz
        self.url = f"ws://127.0.0.1:{port}"
        self.step = serve_ros(port, self.view, self.model, f"{ns}{ainex_model.CAMERA_NAME}",
                              80, control_hz, None, host="127.0.0.1")
        self._base = self.view.get_move_group("base")

    def base_x(self) -> float:
        return float(np.asarray(self._base.joint_pos)[0])

    def run_until(self, done: threading.Event) -> None:
        """Physics pinned to the wall clock, the simulated clock moved on every step and
        the fleet called at its rate -- what `mujoco_bridge.run_sim_loop` does."""
        period = 1.0 / (getattr(self.step, "rate_hz", None) or self.control_hz)
        wall0, sim0 = time.monotonic(), float(self.data.time)
        next_tick = wall0
        while not done.is_set():
            target = sim0 + (time.monotonic() - wall0)
            while self.data.time < target:
                mujoco.mj_step(self.model, self.data)
                self.step.advance_clock(float(self.data.time))
            if time.monotonic() >= next_tick:
                self.step(self.data)
                next_tick += period
            time.sleep(0.001)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9391)
    ap.add_argument("--control-hz", type=float, default=50.0)
    args = ap.parse_args()

    from ros_surfaces.ainex import topics as t

    ros = yaml.safe_load((REPO / "robots_specs" / "ainex" / "ros.yml").read_text())
    sim = Sim(args.port, args.control_hz)
    done = threading.Event()

    def checks() -> None:
        client = None
        try:
            time.sleep(0.5)
            client = Client(sim.url)
            interface_checks(client, ros, t)
            rate_checks(client, t)
            client = behaviour_checks(client, t, sim)
            behaviour_checks_2(client, t, sim)
        except Exception as exc:  # noqa: BLE001 -- report, and let the loop end
            import traceback

            traceback.print_exc()
            FAIL.append(f"crashed: {exc!r}")
        finally:
            if client is not None:
                client.close()
            done.set()

    worker = threading.Thread(target=checks, daemon=True)
    worker.start()
    try:
        sim.run_until(done)
    finally:
        sim.step(None)
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
