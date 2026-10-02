"""The packaged profiles: complete, source-traced, typed in their dialect, bounded."""

import math
import re

import pytest

from robot_console import dialect as d
from robot_console.profiles import (
    ASSEMBLY_IDS, SUPPORTED_IDS, ProfileError, check_namespace_allowed, fill_template, load,
    load_all, select_id, teleop_ids,
)

EXPECTED = {  # console spec §1.1
    "myagv": ("ros1", "base"), "ainex": ("ros1", "walk"), "rosmaster_x3_plus": ("ros1", "base"),
    "so101": ("ros2", "none"), "mycobot280": ("ros2", "none"),
}
# Pinned revisions identified by the robot specification (§2-§7).
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
        for f in c.fields:
            if f.choices is None:
                assert f.min is not None and f.max is not None and f.min <= f.max
                assert f.min <= float(f.default) <= f.max
            else:
                assert f.default in f.choices
        payload = c.build({})              # defaults are valid and fill the template
        assert "$" not in repr(payload)
        for pre in c.prerequisites:
            assert p.endpoint(pre) is not None


def test_field_bounds_are_enforced():
    c = next(c for c in load("so101").controls if c.id == "arm_trajectory")
    with pytest.raises(ValueError):
        c.build({"shoulder_pan": 5.0})
    with pytest.raises(ValueError):
        c.build({"duration": 2.5})       # integer seconds
    ok = c.build({"shoulder_pan": 0.5})
    assert ok["trajectory"]["points"][0]["positions"][0] == 0.5


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
    with pytest.raises(ProfileError):
        check_namespace_allowed(load("myagv"), "robot1")
    check_namespace_allowed(p, "arm1")


def test_id_refusals():
    with pytest.raises(ProfileError, match="Accepted ids"):
        select_id("turtlebot")
    with pytest.raises(ProfileError, match="assembly") as e:
        select_id("myagv_mycobot280")
    assert "'myagv'" in str(e.value) and "'mycobot280'" in str(e.value)
    for arm in ("so101", "mycobot280"):
        with pytest.raises(ProfileError, match="myagv, ainex, rosmaster_x3_plus"):
            select_id(arm, teleop=True)
    assert "myagv_mycobot280" in ASSEMBLY_IDS and "myagv_mycobot280" not in SUPPORTED_IDS


def test_template_fill():
    t = {"a": "$x", "b": ["$y", "{joint_prefix}j1"], "c": 3}
    assert fill_template(t, {"x": 1, "y": "s"}, "ns/") == {"a": 1, "b": ["s", "ns/j1"], "c": 3}
