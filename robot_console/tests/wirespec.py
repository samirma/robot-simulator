"""Build a wire description (for the fake bridge or a real container node) from a profile.

The wire presents a profile's required rows (optionally also its optional rows), with the
documented types, and its cameras publishing small frames in a documented encoding, under the
profile's documented default namespace unless another is given (as the hardware boots by
default). An action server handles a goal that arrives while one runs as the profile's control
of that action records (``busy``: preempt, reject or parallel; preempt where no control names
it). Tests mutate the result to produce wrong types, missing endpoints, extra endpoints etc.
"""

from __future__ import annotations

import copy
from typing import Optional

from robot_console.dialect import ros2_action_endpoints
from robot_console.profiles import Profile


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_joint1", "gripper_joint2"]
#: The status_code every sample motor message carries: bits 0 (undervoltage) and 2 (overtemperature).
SAMPLE_STATUS_CODE = 0b000101


def sensor_sample(topic: str, type_: str) -> dict:
    """A message (rosbridge JSON) a sensor topic of ``type_`` publishes, with values distinct per
    field so a test can tell which one the page shows. ``topic`` is the sensor's bare name."""
    if type_.endswith("JointState"):
        n = len(JOINT_NAMES)
        return {"header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": ""}, "name": list(JOINT_NAMES),
                "position": [round(0.1 * (i + 1), 3) for i in range(6)] + [0.02, 0.0286],     # rad, then the fingers' m
                "velocity": [round(-0.01 * (i + 1), 3) for i in range(n)],
                "effort": [round(0.5 * (i + 1), 3) for i in range(n)]}
    if type_.endswith("JointMotorState"):
        joint = "gripper" if topic == "/gripper/state" else topic.split("/")[2]
        k = 1 if joint == "gripper" else int(joint[5:])
        return {"header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": ""}, "joint_name": joint,
                "position": round(0.25 * k, 3), "velocity": round(-0.125 * k, 3), "torque": round(1.5 * k, 3),
                "status_code": SAMPLE_STATUS_CODE}
    if type_.endswith("ArmStatus"):
        return {"header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": ""}, "mode": "mit", "enabled": True,
                "control_loop_active": True, "state_machine": "IDLE", "joint_names": JOINT_NAMES[:6],
                "per_joint_status_code": [0, 0, 0, 0, 0, 0], "error_codes": []}
    raise KeyError(type_)


def wire_spec(p: Profile, *, namespace: Optional[str] = None, include_optional: bool = False,
              camera_rate: Optional[float] = None) -> dict:
    """The profile's wire. Its sensors publish `sample` messages (`rate_hz`, or once to each new
    subscriber for a sensor without `stale_after_s`, which is `latched`), in sensor-data QoS where
    periodic; its optional rows (its D405 camera, say) only with ``include_optional``."""
    if namespace is None:
        namespace = p.namespace.default if p.namespace else ""
    cams = {c.topic: c for c in p.cameras}
    sensors = {s.topic: s for s in p.sensors}
    busy = {c.name: c.busy for c in p.controls if c.kind == "action"}
    spec = {"dialect": p.dialect, "topics": [], "services": [], "actions": []}
    for e in p.endpoints:
        if e.optional and not include_optional:
            continue
        row = {"name": p.resolve(e.name, namespace), "type": e.type}
        if e.kind == "topic":
            row["direction"] = e.direction or "out"
            if e.name in cams:
                c = cams[e.name]
                row.update(camera=True, direction="out",
                           encoding=(c.encodings or ("rgb8",))[0],
                           rate_hz=camera_rate or c.rate_hz or 10)
            elif e.rate_hz and row["direction"] == "out":
                row["rate_hz"] = min(float(e.rate_hz), 5.0)
            if e.name in sensors:
                s = sensors[e.name]
                row.update(sample=sensor_sample(e.name, e.type), rate_hz=row.get("rate_hz") or 5.0)
                if s.stale_after_s is None:
                    row["latched"] = True
                else:
                    row["qos"] = "best_effort"
            spec["topics"].append(row)
        elif e.kind == "service":
            spec["services"].append(row)
        else:
            spec["actions"].append(dict(row, exec_s=3.0, busy=busy.get(e.name) or "preempt"))
    return spec


def retype(spec: dict, name: str, new_type: str) -> dict:
    s = copy.deepcopy(spec)
    for sec in ("topics", "services", "actions"):
        for r in s[sec]:
            if r["name"] == name:
                r["type"] = new_type
                return s
    raise KeyError(name)


def drop(spec: dict, name: str) -> dict:
    s = copy.deepcopy(spec)
    for sec in ("topics", "services", "actions"):
        s[sec] = [r for r in s[sec] if r["name"] != name]
    return s


def merge(*specs: dict) -> dict:
    out = {"dialect": specs[0]["dialect"], "topics": [], "services": [], "actions": []}
    for s in specs:
        for sec in ("topics", "services", "actions"):
            seen = {r["name"] for r in out[sec]}
            out[sec] += [copy.deepcopy(r) for r in s[sec] if r["name"] not in seen]
    return out


def with_hidden_action_endpoints(spec: dict, action: str, action_type: str) -> dict:
    """``spec`` as a ROS 2 rosapi that lists hidden endpoints shows it: ``action``'s ``_action``
    topics and services, typed as those of an action of ``action_type`` (its send_goal service
    reveals the action's type; stock rosapi hides them)."""
    s = copy.deepcopy(spec)
    topics, services = ros2_action_endpoints(action, action_type)
    s["topics"] += [{"name": n, "type": t, "direction": "out"} for n, t in topics.items()]
    s["services"] += [{"name": n, "type": t} for n, t in services.items()]
    return s
