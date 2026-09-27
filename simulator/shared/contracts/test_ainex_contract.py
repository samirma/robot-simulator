#!/usr/bin/env python
"""The AiNex's contract module against its ROS file, entry for entry.

    molmospaces/.venv/bin/python shared/contracts/test_ainex_contract.py

Run by `test_fleet.py` as well (spec §5, "Contracts"). `ros_surfaces/ainex/topics.py`
transcribes `robots_specs/ainex/ros.yml`; this holds the transcription to the file:
every topic with its type, direction, node and rate, every service with its type and
node, every parameter with its type and -- where the file gives one -- its value, the
joint list and the two frames. Every type it names must also resolve to a schema, or
`/rosapi/message_details` would answer a client with nothing.

`topics.py` is loaded by path, as the console's parity test loads it: it is stdlib-only,
and this check should not need the rest of the surface (or MuJoCo) to run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

SHARED = Path(__file__).resolve().parents[1]
REPO = SHARED.parents[1]
ROS_FILE = REPO / "robots_specs" / "ainex" / "ros.yml"
TOPICS_PY = SHARED / "ros_surfaces" / "ainex" / "topics.py"

if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))


def _load_topics():
    spec = importlib.util.spec_from_file_location("_ainex_contract_topics", TOPICS_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(check) -> None:
    from contracts import message_schemas as schemas

    print("ainex contract vs robots_specs/ainex/ros.yml")
    t = _load_topics()
    ros = yaml.safe_load(ROS_FILE.read_text())

    def rate(value):
        return value if value == "event" else float(value)

    want_topics = {(e["name"], e["type"], e["direction"], e["node"], rate(e["rate_hz"]))
                   for e in ros["topics"]}
    have_topics = {(x.name, x.type, x.direction, x.node, rate(x.rate_hz)) for x in t.TOPICS}
    check("every topic, with type, direction, node and rate", want_topics == have_topics,
          f"missing {sorted(want_topics - have_topics)}; extra {sorted(have_topics - want_topics)}")
    check("no topic listed twice", len(t.TOPICS) == len(have_topics))

    want_srv = {(e["name"], e["type"], e["node"]) for e in ros["services"]}
    have_srv = {(x.name, x.type, x.node) for x in t.SERVICES}
    check("every service, with type and node", want_srv == have_srv,
          f"missing {sorted(want_srv - have_srv)}; extra {sorted(have_srv - want_srv)}")

    want_params = {e["name"]: e for e in ros["parameters"]}
    have_params = {p.name: p for p in t.PARAMETERS}
    check("every parameter", set(want_params) == set(have_params),
          f"missing {sorted(set(want_params) - set(have_params))}; "
          f"extra {sorted(set(have_params) - set(want_params))}")
    wrong = []
    for name, entry in want_params.items():
        have = have_params.get(name)
        if have is None:
            continue
        if have.type != entry["type"]:
            wrong.append(f"{name}: type {have.type} != {entry['type']}")
        value = entry["value"]
        if entry["type"] in ("bool", "int", "double", "string") and have.value != value:
            wrong.append(f"{name}: {have.value!r} != {value!r}")
        elif entry["type"] == "dict" and isinstance(value, str) and value.startswith("{p"):
            if have.value != yaml.safe_load(value):
                wrong.append(f"{name}: {have.value!r} != {value!r}")
    check("every parameter's type, and its value where the file gives one", not wrong,
          "; ".join(wrong))
    check("/app/color_track is the same as /color_track",
          have_params["/app/color_track"].value == have_params["/color_track"].value)
    check("init_pose holds all 24 joints, as init_pose.yaml does",
          list(have_params["/ainex_controller/init_pose"].value) == list(ros["joints"])
          and have_params["/ainex_controller/init_pose"].value["l_knee"] == 1.192)
    check("the magnetometer calibration is 16 numbers", len(t.MAG_CALIBRATION) == 16)

    check("the joints, in servo-id order", tuple(ros["joints"]) == t.JOINT_NAMES)
    check("the camera frame is usb_cam's camera_frame_id",
          t.FRAME_CAMERA == want_params["/camera/camera_frame_id"]["value"])
    check("the IMU frame is the board driver's imu_frame",
          t.FRAME_IMU == want_params["/ros_robot_controller/imu_frame"]["value"])
    check("the camera size is usb_cam's",
          t.CAMERA_SIZE == (want_params["/camera/image_width"]["value"],
                            want_params["/camera/image_height"]["value"]))

    periodic = {e["name"]: float(e["rate_hz"]) for e in ros["topics"]
                if e["direction"] == "out" and e["rate_hz"] != "event"}
    check("a rate for every periodic topic, and only those", t.RATES_HZ == periodic,
          f"{t.RATES_HZ} vs {periodic}")

    unknown = sorted({x.type for x in t.TOPICS if schemas.fields_of(x.type) is None}
                     | {x.type for x in t.SERVICES
                        if schemas.canonical(x.type) not in schemas.SERVICES})
    check("every type resolves to a schema", not unknown, str(unknown))

    for entry in ros["topics"] + ros["services"] + ros["parameters"]:
        if entry.get("unverified"):
            print(f"  (unverified in the ROS file, served: {entry['name']})")


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail and not ok else ''}")
        if not ok:
            failures.append(name)

    run(check)
    print(f"\n{len(failures)} check(s) failed" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
