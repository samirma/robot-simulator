"""The packaged profiles: complete, source-traced, typed in their dialect, bounded."""

import math
import re

import pytest

from robot_console import dialect as d
from robot_console.profiles import (
    SUPPORTED_IDS, ProfileError, check_namespace_allowed, load,
    load_all, select_id, teleop_ids,
)

EXPECTED = {  # console spec §1.1
    "myagv": ("ros1", "base"), "ainex": ("ros1", "walk"), "rosmaster_x3_plus": ("ros1", "base"),
    "so101": ("ros2", "none"), "mycobot280": ("ros2", "none"),
}
# Pinned revisions identified by the robot specification (robots_specs/<id>.md).
PINNED = {
    "myagv": {"c71f3cc574e5ed1973a925238eabe88662cfa701", "addab4a65fdf65c460fec2cc3eee8fab94699370"},
    "ainex": {"e8fe2a816797cf83054135160df5a82ec3596a69"},
    "rosmaster_x3_plus": {"9732c62247dfb57a899aea01e4fe72ed434bac6d"},
    "so101": {"e166df9d51f43b24da9b99047c6c51c306bda74f", "cb2ae6bc0a312de629f84fef933fb3e07dac1ef8"},
    "mycobot280": {"d42ff61a78122c79246623391540d75738b03b23"},
}
TRACE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+):(?P<path>[^#\s]+)#L\d+")


def test_supported_ids_and_dialects():
    assert SUPPORTED_IDS == tuple(EXPECTED)
    for p in load_all():
        assert (p.dialect, p.teleop) == EXPECTED[p.id]
    assert teleop_ids() == ("myagv", "ainex", "rosmaster_x3_plus")


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_pinned_sources(pid):
    p = load(pid)
    revs = {str(s.get("revision", "")) for s in p.sources.values()}
    assert PINNED[pid] <= revs, f"{pid} must record its pinned revisions"
    assert p.boot, "authoritative boot recorded"


def _traced(row, where):
    src, pol = str(row.get("source") or ""), str(row.get("policy") or "")
    assert src or pol, f"{where}: no source trace or console policy"
    return src


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_every_entry_is_traced(pid):
    raw = load(pid).raw
    keys = set(raw["sources"])
    for sec in ("topics", "services", "actions", "cameras", "stop", "controls"):
        for row in raw.get(sec) or []:
            src = _traced(row, f"{pid}.{sec}.{row.get('name', row.get('topic', row.get('id')))}")
            m = TRACE.match(src)
            if m:
                assert m["key"] in keys, f"{pid}: trace key {m['key']} is not a pinned source"
    for c in raw.get("controls") or []:
        for f in c.get("fields") or []:
            _traced(f, f"{pid}.controls.{c['id']}.{f['name']}")
    for sec in ("teleop_base", "teleop_walk"):
        for axis, a in ((raw.get(sec) or {}).get("axes") or {}).items():
            _traced(a, f"{pid}.{sec}.{axis}")


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_types_use_the_profile_dialect(pid):
    p = load(pid)
    for e in p.endpoints:
        assert d.type_dialect(e.type) == p.dialect, (e.name, e.type)
    for c in p.cameras:
        assert d.normalize_type(c.type) == "sensor_msgs/Image"


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_cameras_are_topics_with_finite_stale_thresholds(pid):
    p = load(pid)
    for c in p.cameras:
        e = p.endpoint(c.topic)
        assert e is not None and e.kind == "topic" and e.type == c.type
        assert 0 < c.stale_after_s < math.inf
        assert c.encodings


@pytest.mark.parametrize("pid", ["myagv", "ainex", "rosmaster_x3_plus"])
def test_mobile_speeds_within_documented_limits_and_stop(pid):
    p = load(pid)
    axes = (p.teleop_base or p.teleop_walk).axes
    assert set(axes) == {"x", "y", "yaw"}
    for a in axes.values():
        assert 0 < a.speed <= a.limit
    stop = p.teleop_walk.stop if p.teleop_walk else p.stop
    assert stop, "a documented stop"
    if p.teleop_base:
        assert p.stop[0].op == "publish" and p.stop[0].name == p.teleop_base.topic
        assert all(v == 0 for part in p.stop[0].msg.values() for v in part.values())


def test_camera_sets():
    """Every mobile profile has a required raw camera stream; arms as their boots define."""
    cams = {p.id: [(c.topic, c.optional) for c in p.cameras] for p in load_all()}
    assert cams == {
        "myagv": [("/usb_cam/image_raw", False)],
        "ainex": [("/camera/image_raw", False)],
        "rosmaster_x3_plus": [("/camera/rgb/image_raw", False)],
        "so101": [("/image_raw", False)],
        "mycobot280": [],
    }


def test_ainex_head_limits():
    p = load("ainex")
    assert set(p.head) == {"pan", "tilt"}
    for h in p.head.values():
        assert h.min < 0 < h.max and h.rate > 0


def test_ainex_head_position_read():
    """Teleop starts each head axis from its measured position (console spec §2.1), the page reads
    it on a click: one documented read, servo 23/24 through the board driver's service, mapped
    back with the controller's pulse<->rad relation (init 500, 180/pi/240*1000 pulses per rad,
    ainex_controller.py#L24-L25). A servo the board could not read is left out of the answer."""
    p = load("ainex")
    answer = {"success": True, "position": [{"id": 23, "position": 739}, {"id": 24, "position": 500}]}
    for axis, servo, want in (("pan", 23, 239 * math.radians(240) / 1000), ("tilt", 24, 0.0)):
        read, value = p.head_read(axis)
        assert (read.service, read.type) == (
            "/ros_robot_controller/bus_servo/get_position", "ros_robot_controller/GetBusServosPosition")
        assert servo in read.request["id"]
        e = p.endpoint(read.service)
        assert e is not None and e.kind == "service" and e.type == read.type and not e.optional
        assert read.parse(answer)[(value.control, value.field)] == pytest.approx(want, rel=1e-6, abs=1e-12)
        assert read.parse({"success": True, "position": []})[(value.control, value.field)] is None
        with pytest.raises(ValueError):
            read.parse({"success": False, "position": []})


def test_reads_parse_documented_invalid_answers():
    x3 = load("rosmaster_x3_plus").reads[0]
    got = x3.parse({"angles": [80.0, 120.0, 10.0, 45.0, -1, 60.0]})
    assert got[("arm_pose", "j2")] == 120.0 and got[("arm_pose", "j5")] is None and got[("gripper", "angle")] == 60.0
    cobot = load("mycobot280").reads[0]
    assert set(cobot.parse({f"joint_{i}": 0.0 for i in range(1, 7)}).values()) == {None}
    assert cobot.parse({f"joint_{i}": 10.0 * i for i in range(1, 7)})[("arm_gripper_target", "j1")] == \
        pytest.approx(math.radians(10))


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_controls_are_bounded_and_never_drive_the_base(pid):
    p = load(pid)
    sustained = set()
    if p.teleop_base:
        sustained.add(p.teleop_base.topic)
    if p.teleop_walk:
        sustained |= {p.teleop_walk.param_topic} | {o.name for o in p.teleop_walk.start}
    for c in p.controls:
        assert c.name not in sustained, f"{pid}: control {c.id} would drive the base/walk"
        assert c.kind in ("publish", "call", "action")
        if c.kind == "action":
            assert c.stop and c.stop[0].op == "cancel"
            assert p.dialect == "ros2", "the page sends ROS 2 goals only (one connection each, spec §2.2)"
        for f in c.fields:
            if f.choices is None:
                assert f.min is not None and f.max is not None and f.min <= f.max
                assert f.min <= float(f.default) <= f.max
            else:
                assert f.default in f.choices
        # every "$field" of the template is a field of the control, and every field is sent
        assert placeholders(c.template) == {f.name for f in c.fields}, c.id
        for pre in c.prerequisites:
            assert p.endpoint(pre) is not None


def placeholders(template):
    if isinstance(template, dict):
        return set().union(*map(placeholders, template.values())) if template else set()
    if isinstance(template, list):
        return set().union(*map(placeholders, template)) if template else set()
    return {template[1:]} if isinstance(template, str) and template.startswith("$") else set()


def test_namespace_only_where_documented():
    for p in load_all():
        assert (p.namespace is not None) == (p.id == "so101"), p.id
    p = load("so101")
    assert p.resolve("/joint_trajectory_controller/follow_joint_trajectory", "arm1") == \
        "/arm1/joint_trajectory_controller/follow_joint_trajectory"
    assert p.resolve("/tf", "arm1") == "/tf"
    assert p.resolve("/image_raw", "arm1") == "/image_raw"       # the camera boot runs un-namespaced
    assert p.resolve("/set_capture", "arm1") == "/set_capture"
    # camera_info_manager's private ~/set_camera_info lives under the camera node, not the root
    assert p.endpoint("/usb_cam/set_camera_info") is not None and p.endpoint("/set_camera_info") is None
    assert p.resolve("/usb_cam/set_camera_info", "arm1") == "/usb_cam/set_camera_info"
    assert p.resolve("/joint_states", "") == "/joint_states"
    for other in load_all():
        if other.id != "so101":
            with pytest.raises(ProfileError, match="no namespace override"):
                check_namespace_allowed(other, "robot1")
            check_namespace_allowed(other, None)
    check_namespace_allowed(p, "arm1")


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_reads_are_documented_measurements_of_control_fields(pid):
    """A profile's `reads` (documented read services returning measured joint positions, which the
    page calls only on a click) are traced, name a service row of the profile with its type, and
    map each answered value onto a bounded numeric field of one of its controls."""
    p = load(pid)
    raw = p.raw
    for r in raw.get("reads") or []:
        t = TRACE.match(str(r.get("source", "")))
        assert t and t["key"] in p.sources, r["id"]
        e = p.endpoint(r["service"])
        assert e is not None and e.kind == "service" and e.type == r["type"], r["id"]
        assert r["values"], r["id"]
        for v in r["values"]:
            assert TRACE.match(str(v.get("source", ""))), (r["id"], v)
            c = next(c for c in p.controls if c.id == v["control"])
            f = next(f for f in c.fields if f.name == v["field"])
            assert f.choices is None and f.min is not None and f.max is not None
            assert float(v.get("scale", 1)) != 0
    ids = {pid: [r["id"] for r in load(pid).raw.get("reads") or []] for pid in SUPPORTED_IDS}
    assert ids == {"myagv": [], "ainex": ["head_position"], "rosmaster_x3_plus": ["current_angle"],
                   "so101": [], "mycobot280": ["get_angles"]}


def test_sole_publisher_controls_are_traced_topic_publishes():
    """A control incompatible with other publishers of its topic says why, from the sources."""
    marked = {(p.id, c.id) for p in load_all() for c in p.controls if c.sole_publisher}
    assert marked == {("mycobot280", "arm_gripper_target")}
    for pid, cid in marked:
        p = load(pid)
        c = next(c for c in p.controls if c.id == cid)
        raw = next(r for r in p.raw["controls"] if r["id"] == cid)
        assert c.kind == "publish"
        m = TRACE.match(raw["sole_publisher_source"])
        assert m and m["key"] in p.sources and raw["sole_publisher_note"]


def test_call_controls_name_their_documented_failure_field():
    """mycobot_interfaces SetAngles/GripperStatus answer `bool flag` (false on failure)."""
    by = {c.id: c for c in load("mycobot280").controls}
    assert {i: by[i].ok_field for i in ("set_angles", "gripper_open", "gripper_close")} == dict.fromkeys(
        ("set_angles", "gripper_open", "gripper_close"), "flag")


@pytest.mark.parametrize("unknown", ["turtlebot", "myagv_mycobot280"])
def test_id_refusals(unknown):
    # myagv_mycobot280 is the former assembly: no longer a registry id (amended 2026-10-02)
    with pytest.raises(ProfileError, match=f"unknown robot id '{unknown}'. Accepted ids"):
        select_id(unknown)
    for arm in ("so101", "mycobot280"):
        with pytest.raises(ProfileError, match="myagv, ainex, rosmaster_x3_plus"):
            select_id(arm, teleop=True)

