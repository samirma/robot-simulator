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
import math
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
    hdr = {"stamp": {"secs": 1, "nsecs": 0}, "frame_id": "imu_link"}
    if type_.endswith("Float32"):
        return {"data": SAMPLE_FLOAT32.get(topic, 1.5)}
    if type_.endswith("Bool"):
        return {"data": True}
    if type_.endswith("/Imu"):
        return {"header": hdr, "orientation": dict(SAMPLE_QUAT_YAW90 if topic == "/imu/data" else ZERO_QUAT),
                "angular_velocity": {"x": 0.01, "y": -0.02, "z": 0.03},
                "linear_acceleration": dict(SAMPLE_ACCEL)}
    if type_.endswith("MagneticField"):
        return {"header": hdr, "magnetic_field": {"x": 12.5, "y": -3.25, "z": 40.0}}
    if type_.endswith("Twist"):
        return {"linear": {"x": 0.2, "y": -0.1, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.5}}
    if type_.endswith("Odometry"):
        return {"header": dict(hdr, frame_id="odom"), "child_frame_id": "base_footprint",
                "pose": {"pose": {"position": {"x": 1.5, "y": -0.25, "z": 0.0}, "orientation": dict(SAMPLE_QUAT_YAW30)}},
                "twist": {"twist": {"linear": {"x": 0.2, "y": -0.1, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.5}}}}
    if type_.endswith("ArmJoint"):
        return {"id": 6, "run_time": 0, "angle": 30.0, "joints": []}
    if type_.endswith("LaserScan"):
        return dict(SAMPLE_SCAN, header=dict(hdr, frame_id="laser"))
    if type_.endswith("PointCloud2"):          # as orbbec_camera publishes it: unorganised, valid points only
        reg = topic != "/camera/depth/points"
        step, n = (20, 251_000) if reg else (16, 263_412)
        f = lambda name, off: {"name": name, "offset": off, "datatype": 7, "count": 1}      # noqa: E731
        return {"header": dict(hdr, frame_id="camera_color_optical_frame" if reg else "camera_depth_optical_frame"),
                "height": 1, "width": n, "fields": [f("x", 0), f("y", 4), f("z", 8)] + ([f("rgb", 16)] if reg else []),
                "is_bigendian": False, "point_step": step, "row_step": n * step, "data": "", "is_dense": True}
    if type_.endswith("/PointCloud"):          # sensor_msgs/PointCloud (v1): three lidar points
        return {"header": dict(hdr, frame_id="laser"), "channels": [],
                "points": [{"x": 0.5, "y": 0.0, "z": 0.0}, {"x": 2.0, "y": 1.0, "z": 0.0}, {"x": 1.0, "y": -1.0, "z": 0.0}]}
    raise KeyError(type_)


#: Float32 samples by topic (a battery voltage, a firmware version).
SAMPLE_FLOAT32 = {"/voltage": 12.3, "/edition": 1.100000023841858}
ZERO_QUAT = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}
SAMPLE_QUAT_YAW90 = {"x": 0.0, "y": 0.0, "z": math.sin(math.pi / 4), "w": math.cos(math.pi / 4)}
SAMPLE_QUAT_YAW30 = {"x": 0.0, "y": 0.0, "z": math.sin(math.pi / 12), "w": math.cos(math.pi / 12)}
#: An acceleration that assumes no unit (the ICM20948 board's unit is not established).
SAMPLE_ACCEL = {"x": 0.125, "y": -0.25, "z": 0.5}
#: A scan as TG.launch configures the 4ROS: -85..85 deg, range 0.01-50 m, invalid ranges +inf (sent by Python's
#: json as bare Infinity), 945 slots: 2 m everywhere, 0.5 m straight ahead (index 472), +inf at indices 0..99 and
#: one return beyond range_max (60 m) at index 200.
SCAN_N = 945
SAMPLE_SCAN = {"angle_min": -math.radians(85), "angle_max": math.radians(85),
               "angle_increment": math.radians(170) / (SCAN_N - 1), "time_increment": 0.0, "scan_time": 0.1,
               "range_min": 0.01, "range_max": 50.0,
               "ranges": [math.inf] * 100 + [2.0] * 100 + [60.0] + [2.0] * 271 + [0.5] + [2.0] * 472,
               "intensities": [0.0] * SCAN_N}
assert len(SAMPLE_SCAN["ranges"]) == SCAN_N


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
                if c.kind == "depth":       # depth values across the camera's whole range
                    row["depth_mm"] = (round(c.range_m[0] / c.unit_m), round(c.range_m[1] / c.unit_m))
            elif e.rate_hz and row["direction"] == "out":
                row["rate_hz"] = min(float(e.rate_hz), 5.0)
            if e.name in sensors:
                s = sensors[e.name]
                row.update(sample=sensor_sample(e.name, e.type), rate_hz=row.get("rate_hz") or 5.0)
                if s.event_only:              # an ordinary publisher of events: nothing until the test emits one
                    row["event"] = True
                elif s.stale_after_s is None:
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
