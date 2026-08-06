#!/usr/bin/env python
"""Standalone check of the AiNex's ROS surface, against a real websocket.

    python robots/ainex/test_ros.py [--port 9391]

Separate from `test_attach.py` for the reason `robots/myagv/test_scan.py` is separate:
everything here misbehaves *quietly*. A service that drops the caller's `id` hangs the
client instead of failing it; a walk that ignores `start` looks like a stationary robot;
a scan that ranges the robot's own thigh produces a perfectly plausible map of a room
that is not there.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import websockets.sync.client as wsc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FAIL = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(label)


class Client:
    """Just enough rosbridge client to drive the surface under test."""

    def __init__(self, url: str) -> None:
        self._ws = wsc.connect(url, open_timeout=5)
        self._id = 0

    def close(self) -> None:
        self._ws.close()

    def publish(self, topic: str, msg: dict) -> None:
        self._ws.send(json.dumps({"op": "publish", "topic": topic, "msg": msg}))

    def subscribe(self, topic: str) -> None:
        self._ws.send(json.dumps({"op": "subscribe", "topic": topic}))

    def unsubscribe(self, topic: str) -> None:
        self._ws.send(json.dumps({"op": "unsubscribe", "topic": topic}))

    def drain(self) -> None:
        """Throw away anything already queued.

        The surface publishes /joint_states, /imu and /walking/is_walking every control
        tick, so by the time a later check runs there are thousands of frames buffered and
        a naive read for one scan spends its whole timeout draining them.
        """
        try:
            while True:
                self._ws.recv(timeout=0.05)
        except Exception:
            pass

    def call(self, service: str, args: dict | None = None, call_id: str | None = None) -> dict:
        self._id += 1
        cid = call_id or f"call-{self._id}"
        self._ws.send(
            json.dumps({"op": "call_service", "service": service, "args": args or {}, "id": cid})
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame = json.loads(self._ws.recv(timeout=5))
            if frame.get("op") == "service_response" and frame.get("service") == service:
                frame["_sent_id"] = cid
                return frame
        raise TimeoutError(f"no response to {service}")

    def next_message(self, topic: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = json.loads(self._ws.recv(timeout=timeout))
            if frame.get("op") == "publish" and frame.get("topic") == topic:
                return frame["msg"]
        raise TimeoutError(f"no message on {topic}")


def main() -> int:  # noqa: PLR0915 -- a checklist reads better in one piece
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9391)
    args = ap.parse_args()

    from robots.ainex import AiNexRobot, AiNexRobotConfig, AiNexRobotView
    from robots.ainex import servos, topics
    from robots.ainex.ros_surface import serve_ros
    from tools.spawn_robot import SCAN_DEFAULTS

    config = AiNexRobotConfig()
    ns = config.robot_namespace

    spec = mujoco.MjSpec()
    spec.worldbody.add_light(
        pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.55, 0.56, 0.58, 1]
    )
    # A wall to range, 2 m ahead.
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 3.0, 1.0], pos=[2.0, 0.0, 1.0]
    )
    AiNexRobot.add_robot_to_scene(
        config, spec, prefix=ns, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    view = AiNexRobotView(data, ns)
    for name, value in servos.INIT_POSE.items():
        data.ctrl[model.actuator(f"{ns}{name}").id] = value

    defaults = SCAN_DEFAULTS["ainex"]
    scan_cfg = {
        "beams": 360,
        "max_range": defaults["max_range"],
        "min_range": defaults["min_range"],
        "offset_x": defaults["offset"][0],
        "offset_z": defaults["offset"][1],
        "period": 0.1,
        "body": f"{ns}{AiNexRobot.robot_model_root_name()}",
        "exclude_bodies": frozenset(
            i for i in range(model.nbody)
            if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)) and n.startswith(ns)
        ),
    }

    control_hz = 50.0
    step = serve_ros(
        args.port, view, model, None, (640, 480), 80, control_hz, 0.5,
        scan=scan_cfg, depth=None, extra=None,
    )

    def run(seconds: float) -> None:
        for _ in range(int(seconds * control_hz)):
            for _ in range(int((1.0 / control_hz) / model.opt.timestep)):
                mujoco.mj_step(model, data)
            step(data)

    time.sleep(0.4)
    client = Client(f"ws://127.0.0.1:{args.port}")
    try:
        print("services:")
        reply = client.call(topics.SRV_WALKING_COMMAND, {"command": "enable"}, call_id="xyz")
        # A dropped id hangs a blocked caller instead of failing it, and nothing else here
        # would notice; roslibpy would simply never return.
        check("call_service echoes the caller's id", reply.get("id") == "xyz", str(reply.get("id")))
        check("known command returns result true", reply.get("result") is True, str(reply))

        reply = client.call("/walking/nonexistent")
        check("unknown service fails promptly rather than hanging",
              reply.get("result") is False, str(reply.get("values")))

        reply = client.call(topics.SRV_WALKING_COMMAND, {"command": "nonsense"})
        check("unknown walking command is rejected", reply["values"].get("result") is False)

        print("\nwalking parameters:")
        client.publish(topics.TOPIC_APP_WALKING_PARAM,
                       {"speed": 4, "height": 0.03, "x": 1.0, "y": 0.0, "angle": 0.0})
        run(0.2)
        reply = client.call(topics.SRV_GET_WALKING_PARAM, {"get_param": True})
        param = reply["values"]["parameters"]
        check("set_walking_param round-trips through get_param",
              abs(param["period_time"] - 300.0) < 1e-6 and param["x_move_amplitude"] > 0,
              f"period_time={param['period_time']} x={param['x_move_amplitude']}")
        check("period_time is milliseconds on the wire", param["period_time"] > 100,
              f"{param['period_time']}")

        print("\nwalking:")
        base = view.get_move_group("base")
        before = np.asarray(base.joint_pos).copy()
        reply = client.call(topics.SRV_IS_WALKING)
        check("is_walking false before start", reply["values"]["state"] is False)

        client.call(topics.SRV_WALKING_COMMAND, {"command": "start"})
        run(2.0)
        moved = np.asarray(base.joint_pos).copy() - before
        expected = 4.0 * param["x_move_amplitude"] / (param["period_time"] / 1000.0) * 2.0
        check("start walks forward at the commanded speed",
              abs(moved[0] - expected) < 0.25 * expected,
              f"moved {moved[0]:.3f} m in 2 s, expected ~{expected:.3f}")
        reply = client.call(topics.SRV_IS_WALKING)
        check("is_walking true while walking", reply["values"]["state"] is True)

        client.call(topics.SRV_WALKING_COMMAND, {"command": "stop"})
        run(0.2)
        halted = np.asarray(base.joint_pos).copy()
        run(0.5)
        check("stop halts within a control period or two",
              abs(np.asarray(base.joint_pos)[0] - halted[0]) < 0.005,
              f"drifted {np.asarray(base.joint_pos)[0] - halted[0]:+.4f} m after stopping")

        # `start` must imply `enable`, as the vendor controller's start branch sets
        # walking_enable itself. Gating on a prior enable made disable -> start walk on
        # hardware but not here, which is exactly the class of divergence a client
        # rehearsing against the sim cannot afford.
        client.call(topics.SRV_WALKING_COMMAND, {"command": "disable"})
        before_restart = np.asarray(base.joint_pos).copy()
        client.call(topics.SRV_WALKING_COMMAND, {"command": "start"})
        run(1.0)
        client.call(topics.SRV_WALKING_COMMAND, {"command": "stop"})
        restart_moved = np.asarray(base.joint_pos)[0] - before_restart[0]
        check("start after disable walks (start implies enable)",
              abs(restart_moved) > 0.05, f"moved {restart_moved:+.3f} m")
        run(0.2)

        print("\nturning and strafing:")
        for axis, key, index in (("yaw", "angle", 2), ("lateral", "y", 1)):
            start = np.asarray(base.joint_pos).copy()
            client.publish(topics.TOPIC_APP_WALKING_PARAM,
                           {"speed": 4, "height": 0.03, "x": 0.0, "y": 0.0, "angle": 0.0,
                            key: 1.0})
            client.call(topics.SRV_WALKING_COMMAND, {"command": "start"})
            run(1.5)
            client.call(topics.SRV_WALKING_COMMAND, {"command": "stop"})
            delta = np.asarray(base.joint_pos).copy() - start
            check(f"{axis} command moves the {axis} axis", abs(delta[index]) > 0.05,
                  f"delta={delta[index]:+.3f}")
            run(0.2)

        print("\njoint states:")
        client.subscribe(topics.TOPIC_JOINT_STATES)
        run(0.2)
        msg = client.next_message(topics.TOPIC_JOINT_STATES)
        check("24 joints, named, in servo-id order",
              msg["name"] == list(servos.BY_ID) and len(msg["position"]) == 24,
              f"{len(msg['name'])} names, first={msg['name'][0]}")

        print("\nbus servos:")
        # id 23 is head_pan; id 13 is l_sho_pitch, one of the two the vendor yaml flips.
        for servo_id, joint, count in ((23, "head_pan", 700), (13, "l_sho_pitch", 700)):
            client.publish(topics.TOPIC_BUS_SERVO_SET,
                           {"duration": 0.2, "position": [{"id": servo_id, "position": count}]})
            run(1.0)
            reached = float(data.qpos[model.jnt_qposadr[model.joint(f"{ns}{joint}").id]])
            expected_rad = servos.count_to_angle(joint, count)
            check(f"servo {servo_id} ({joint}) reaches the commanded count",
                  abs(reached - expected_rad) < 0.05,
                  f"{reached:+.3f} rad, expected {expected_rad:+.3f}")
        # servo_controller.yaml writes `min: 1000, max: 0` for the two sho_pitch servos,
        # which is how the vendor encodes a servo mounted the other way round. So counting
        # *up* from a joint's init must move a flipped joint negative and an unflipped one
        # positive. Comparing against a fixed count would only re-test the arithmetic.
        def direction(joint: str) -> float:
            init = servos.SERVOS[joint][1]
            return servos.count_to_angle(joint, init + 100)

        check("the flipped servos move opposite the unflipped ones",
              direction("l_sho_pitch") < 0 < direction("head_pan")
              and direction("r_sho_pitch") < 0,
              f"l_sho_pitch {direction('l_sho_pitch'):+.3f} vs head_pan {direction('head_pan'):+.3f}")

        print("\naction groups:")
        client.publish(topics.TOPIC_APP_ACTION, {"data": "hand_open"})
        run(1.5)
        gripper = float(data.qpos[model.jnt_qposadr[model.joint(f"{ns}l_gripper").id]])
        check("set_action replays a group", abs(gripper - 0.690) < 0.1, f"l_gripper={gripper:+.3f}")
        client.publish(topics.TOPIC_APP_ACTION, {"data": "no_such_action"})
        run(0.3)
        check("unknown action group is ignored rather than fatal", True)

        print("\nscan:")
        client.unsubscribe(topics.TOPIC_JOINT_STATES)
        client.subscribe(topics.TOPIC_SCAN)
        client.drain()
        run(0.5)
        msg = client.next_message(topics.TOPIC_SCAN)
        ranges = np.array(msg["ranges"])
        check("360 beams", ranges.shape == (360,), str(ranges.shape))
        check("every value finite", bool(np.isfinite(ranges).all()))
        # The robot's own limbs are within ~0.15 m of the torso-mounted scanner. Without
        # exclude_bodies most beams would come back at that distance, and the resulting
        # scan looks entirely reasonable until you try to map with it.
        close = int((ranges < 0.3).sum())
        check("the scan does not range the robot's own limbs", close == 0,
              f"{close} beams under 0.3 m")

        print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    finally:
        client.close()
        step(None)

    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
