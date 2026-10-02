"""Validator for robots_specs/<id>/ros.yml and ros2.yml (schema version 1, SCHEMA.md).

    python robots_specs/tests/schema.py robots_specs/*/ros*.yml

`validate(path)` returns a list of error strings; empty means valid. PyYAML only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

BASES = {"source", "manufacturer", "measured", "estimate", "uncalibrated"}
MOTIONS = {"drive", "walk", "arm", "gripper", "head", "action_group"}
VELOCITY_MOTIONS = {"drive", "walk"}
ROS1_TYPE = re.compile(r"^[A-Za-z][\w]*/[A-Za-z][\w]*$")
ROS2_TYPE = {
    "msg": re.compile(r"^[A-Za-z][\w]*/msg/[A-Za-z][\w]*$"),
    "srv": re.compile(r"^[A-Za-z][\w]*/srv/[A-Za-z][\w]*$"),
    "action": re.compile(r"^[A-Za-z][\w]*/action/[A-Za-z][\w]*$"),
}
NAME = re.compile(r"^/[A-Za-z0-9_/~]*$")


class _Errors(list):
    def need(self, cond, msg):
        if not cond:
            self.append(msg)
        return cond


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _type_ok(dialect: str, kind: str, t) -> bool:
    if not isinstance(t, str):
        return False
    if dialect == "ros1":
        return bool(ROS1_TYPE.match(t))
    return bool(ROS2_TYPE[kind].match(t))


def _pose_ok(row) -> bool:
    return (isinstance(row.get("xyz"), list) and len(row["xyz"]) == 3
            and all(_is_num(v) for v in row["xyz"])
            and isinstance(row.get("rpy"), list) and len(row["rpy"]) == 3
            and all(_is_num(v) for v in row["rpy"]))


def _rate_ok(r) -> bool:
    return r == "non_periodic" or (_is_num(r) and r > 0)


def validate(path: str | Path) -> list[str]:
    path = Path(path)
    e = _Errors()
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return [f"{path}: not YAML: {exc}"]
    if not isinstance(doc, dict):
        return [f"{path}: top level is not a map"]
    p = f"{path.parent.name}/{path.name}"

    e.need(doc.get("schema_version") == 1, f"{p}: schema_version must be 1")
    e.need(doc.get("robot_id") == path.parent.name, f"{p}: robot_id must equal the folder name")
    e.need(isinstance(doc.get("name"), str) and doc["name"], f"{p}: name missing")
    dialect = doc.get("dialect")
    want = {"ros.yml": "ros1", "ros2.yml": "ros2"}.get(path.name)
    e.need(dialect == want, f"{p}: dialect {dialect!r} does not match the filename ({want})")
    if dialect not in ("ros1", "ros2"):
        return list(e)
    e.need(isinstance(doc.get("ros_distribution"), str) and doc["ros_distribution"],
           f"{p}: ros_distribution missing")
    e.need(doc.get("interface_authority") in ("manufacturer", "approved_community"),
           f"{p}: interface_authority must be manufacturer or approved_community")

    # sources
    src_ids = set()
    for i, s in enumerate(doc.get("sources") or []):
        w = f"{p}: sources[{i}]"
        if not e.need(isinstance(s, dict), f"{w} not a map"):
            continue
        for k in ("id", "url", "revision", "role"):
            e.need(isinstance(s.get(k), str) and s[k], f"{w}.{k} missing")
        src_ids.add(s.get("id"))
    e.need(src_ids, f"{p}: sources empty")

    # boot
    boot = doc.get("boot") or {}
    e.need(isinstance(boot.get("summary"), str) and boot["summary"], f"{p}: boot.summary missing")
    e.need(isinstance(boot.get("commands"), list) and boot["commands"], f"{p}: boot.commands missing")
    e.need(isinstance(boot.get("launch_files"), list) and boot["launch_files"],
           f"{p}: boot.launch_files missing")
    for k in ("launch_files", "data_files"):
        for i, row in enumerate(boot.get(k) or []):
            e.need(isinstance(row, dict) and row.get("source") in src_ids and row.get("path"),
                   f"{p}: boot.{k}[{i}] needs a known source id and a path")
    e.need(isinstance(boot.get("hardware"), str) and boot["hardware"], f"{p}: boot.hardware missing")

    # model
    model = doc.get("model") or {}
    for k in ("urdf", "urdf_sha256", "mjcf"):
        e.need(isinstance(model.get(k), str) and model[k], f"{p}: model.{k} missing")
    e.need(isinstance(model.get("mjcf_official"), bool), f"{p}: model.mjcf_official must be a bool")
    e.need(isinstance(model.get("robot_description"), dict), f"{p}: model.robot_description missing")

    # nodes
    node_names = set()
    for i, n in enumerate(doc.get("nodes") or []):
        w = f"{p}: nodes[{i}]"
        if not e.need(isinstance(n, dict), f"{w} not a map"):
            continue
        e.need(isinstance(n.get("name"), str) and NAME.match(n["name"]), f"{w}.name not a ROS name")
        e.need(n.get("source"), f"{w} ({n.get('name')}) has no source")
        node_names.add(n.get("name"))
    e.need(node_names, f"{p}: nodes empty")

    def check_common(w, row):
        e.need(isinstance(row.get("name"), str) and NAME.match(row["name"]),
               f"{w}.name {row.get('name')!r} not a ROS name")
        e.need(row.get("source"), f"{w} ({row.get('name')}) has no source")
        if "optional" in row:
            e.need(isinstance(row["optional"], bool), f"{w}.optional must be a bool")

    # topics
    seen = set()
    for i, t in enumerate(doc.get("topics") or []):
        w = f"{p}: topics[{i}]"
        if not e.need(isinstance(t, dict), f"{w} not a map"):
            continue
        check_common(w, t)
        e.need(t.get("name") not in seen, f"{w}: duplicate topic {t.get('name')}")
        seen.add(t.get("name"))
        e.need(_type_ok(dialect, "msg", t.get("type")), f"{w} ({t.get('name')}) bad type {t.get('type')!r}")
        e.need(t.get("direction") in ("in", "out"), f"{w} ({t.get('name')}) direction must be in|out")
        e.need(_rate_ok(t.get("rate")), f"{w} ({t.get('name')}) rate must be Hz > 0 or non_periodic")
        if _is_num(t.get("rate")):
            e.need(t.get("rate_basis") in BASES, f"{w} ({t.get('name')}) rate_basis missing/invalid")
        nodes = t.get("nodes")
        e.need(isinstance(nodes, list) and nodes, f"{w} ({t.get('name')}) nodes missing")
        for n in nodes or []:
            e.need(n in node_names, f"{w} ({t.get('name')}) node {n} not in nodes")
        if "camera" in t:
            e.need(t["camera"] is True, f"{w}.camera must be true when present")
            e.need(t.get("type") in ("sensor_msgs/Image", "sensor_msgs/msg/Image"),
                   f"{w}: only a raw sensor_msgs Image stream is marked camera")

    for kind, key in (("srv", "services"), ("action", "actions")):
        seen = set()
        for i, s in enumerate(doc.get(key) or []):
            w = f"{p}: {key}[{i}]"
            if not e.need(isinstance(s, dict), f"{w} not a map"):
                continue
            check_common(w, s)
            e.need(s.get("name") not in seen, f"{w}: duplicate {s.get('name')}")
            seen.add(s.get("name"))
            e.need(_type_ok(dialect, kind, s.get("type")), f"{w} ({s.get('name')}) bad type {s.get('type')!r}")
            e.need(s.get("node") in node_names, f"{w} ({s.get('name')}) node {s.get('node')} not in nodes")

    seen = set()
    for i, prm in enumerate(doc.get("parameters") or []):
        w = f"{p}: parameters[{i}]"
        if not e.need(isinstance(prm, dict), f"{w} not a map"):
            continue
        e.need(isinstance(prm.get("name"), str) and prm["name"], f"{w}.name missing")
        e.need("value" in prm, f"{w} ({prm.get('name')}) value missing")
        e.need(prm.get("source"), f"{w} ({prm.get('name')}) has no source")
        if dialect == "ros2":
            e.need(prm.get("node") in node_names, f"{w} ({prm.get('name')}) node {prm.get('node')} not in nodes")
        else:
            e.need(prm["name"].startswith("/") if isinstance(prm.get("name"), str) else False,
                   f"{w} ({prm.get('name')}) ROS 1 parameter must be a global name")
        k = (prm.get("node"), prm.get("name"))
        e.need(k not in seen, f"{w}: duplicate parameter {k}")
        seen.add(k)

    for i, tf in enumerate(doc.get("tf") or []):
        w = f"{p}: tf[{i}]"
        if not e.need(isinstance(tf, dict), f"{w} not a map"):
            continue
        for k in ("parent", "child", "publisher"):
            e.need(isinstance(tf.get(k), str) and tf[k], f"{w}.{k} missing")
        e.need(tf.get("publisher") in node_names, f"{w} publisher {tf.get('publisher')} not in nodes")
        e.need(isinstance(tf.get("static"), bool), f"{w}.static must be a bool")
        e.need(_rate_ok(tf.get("rate")), f"{w}.rate must be Hz or non_periodic")
        if tf.get("static"):
            e.need(_pose_ok(tf), f"{w} static transform needs xyz and rpy")
        e.need(tf.get("source"), f"{w} has no source")

    topic_names = {t.get("name") for t in doc.get("topics") or [] if isinstance(t, dict)}
    all_eps = topic_names | {s.get("name") for k in ("services", "actions")
                             for s in doc.get(k) or [] if isinstance(s, dict)}

    motion_ids = set()
    for i, m in enumerate(doc.get("motions") or []):
        w = f"{p}: motions[{i}]"
        if not e.need(isinstance(m, dict), f"{w} not a map"):
            continue
        mid = m.get("id")
        e.need(mid in MOTIONS, f"{w}.id {mid!r} not one of {sorted(MOTIONS)}")
        motion_ids.add(mid)
        vel = m.get("velocity_driven")
        e.need(vel is (mid in VELOCITY_MOTIONS), f"{w} ({mid}) velocity_driven must be {mid in VELOCITY_MOTIONS}")
        cmd = m.get("command") or {}
        e.need(cmd.get("interface") in ("topic", "service", "action"), f"{w} ({mid}) command.interface")
        e.need(cmd.get("name") in all_eps, f"{w} ({mid}) command {cmd.get('name')} is not a recorded endpoint")
        fields = cmd.get("fields")
        e.need(isinstance(fields, list) and fields, f"{w} ({mid}) command.fields missing")
        for j, f in enumerate(fields or []):
            e.need(isinstance(f, dict) and f.get("field") and "unit" in f,
                   f"{w} ({mid}) command.fields[{j}] needs field and unit")
            if isinstance(f, dict) and ("min" in f or "max" in f):
                e.need(_is_num(f.get("min")) and _is_num(f.get("max")) and f["min"] <= f["max"],
                       f"{w} ({mid}) command.fields[{j}] min/max must be numbers, min <= max")
        e.need("example" in cmd, f"{w} ({mid}) command.example missing")
        if vel:
            stop = m.get("stop") or {}
            e.need(stop.get("name") in all_eps and "message" in stop and stop.get("type"),
                   f"{w} ({mid}) velocity-driven motion needs a stop command (name, type, message)")
        else:
            e.need(isinstance(m.get("end_state"), str) and m["end_state"],
                   f"{w} ({mid}) end_state missing")
        wd = m.get("watchdog") or {}
        if e.need(isinstance(wd.get("present"), bool), f"{w} ({mid}) watchdog.present must be a bool"):
            if wd["present"]:
                e.need(_is_num(wd.get("interval_s")) and wd["interval_s"] > 0,
                       f"{w} ({mid}) watchdog.interval_s must be a number when present")
        e.need(wd.get("basis") in ("source", "measured"), f"{w} ({mid}) watchdog.basis must be source|measured")
        e.need(isinstance(wd.get("method"), str) and wd["method"], f"{w} ({mid}) watchdog.method missing")
        e.need(wd.get("date") is not None, f"{w} ({mid}) watchdog.date missing")
        e.need("feedback" in m, f"{w} ({mid}) feedback missing (use [] and a note when none)")
    e.need(motion_ids, f"{p}: motions empty")

    sensors = doc.get("sensors")
    if e.need(isinstance(sensors, dict), f"{p}: sensors missing"):
        cam_topics = {t.get("name") for t in doc.get("topics") or [] if isinstance(t, dict) and t.get("camera")}
        cams = sensors.get("cameras") or []
        for i, c in enumerate(cams):
            w = f"{p}: sensors.cameras[{i}]"
            e.need(c.get("image_topic") in cam_topics,
                   f"{w} image_topic {c.get('image_topic')} is not a topic marked camera")
            for k in ("width", "height", "rate"):
                e.need(_is_num(c.get(k)) and c[k] > 0, f"{w}.{k} missing")
            e.need(isinstance(c.get("frame_id"), str), f"{w}.frame_id missing")
            intr = c.get("intrinsics") or {}
            e.need(intr.get("basis") in BASES, f"{w}.intrinsics.basis missing")
            mount = c.get("mount") or {}
            e.need(_pose_ok(mount) and mount.get("parent") and mount.get("basis") in BASES,
                   f"{w}.mount needs parent, xyz, rpy, basis")
        e.need({c.get("image_topic") for c in cams} == cam_topics,
               f"{p}: every camera topic needs a sensors.cameras row and vice versa")
        for i, l in enumerate(sensors.get("lidars") or []):
            w = f"{p}: sensors.lidars[{i}]"
            e.need(l.get("topic") in topic_names, f"{w} topic not recorded")
            for k in ("angle_min", "angle_max", "range_min", "range_max", "scan_rate"):
                e.need(_is_num(l.get(k)), f"{w}.{k} missing")
            e.need(l.get("basis") in BASES, f"{w}.basis missing")

    for i, t in enumerate(doc.get("tolerances") or []):
        w = f"{p}: tolerances[{i}]"
        e.need(isinstance(t, dict) and t.get("figure"), f"{w}.figure missing")
        tol = (t or {}).get("tolerance") or {}
        e.need(_is_num(tol.get("absolute")) or _is_num(tol.get("relative")),
               f"{w} needs tolerance.absolute or tolerance.relative")
        e.need((t or {}).get("basis") in ("manufacturer", "estimate", "measured", "source"),
               f"{w}.basis missing")
    e.need(isinstance(doc.get("tolerances"), list), f"{p}: tolerances must be a list")
    return list(e)


if __name__ == "__main__":
    bad = 0
    for arg in sys.argv[1:]:
        errs = validate(arg)
        for msg in errs:
            print(msg)
        bad += bool(errs)
        print(f"{arg}: {'OK' if not errs else f'{len(errs)} error(s)'}")
    sys.exit(1 if bad else 0)
