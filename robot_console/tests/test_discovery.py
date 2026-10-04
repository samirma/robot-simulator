"""Typed discovery, validation and target selection (console spec §1.1, §2.4, §4)."""

import copy

import pytest

from robot_console import camera as cam
from robot_console import discovery
from robot_console.discovery import SelectionError, discover, fetch_graph, select_target
from robot_console.profiles import SUPPORTED_IDS, ProfileError, load, load_all, parse_profile
from robot_console.rosbridge import Rosbridge
from wirespec import drop, merge, retype, wire_spec


def graph_of(fake, spec):
    srv = fake(spec)
    rb = Rosbridge(srv.url).connect()
    try:
        return fetch_graph(rb)
    finally:
        rb.close()


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_matching_wire_selects_the_profile(fake, pid):
    p = load(pid)
    g = graph_of(fake, wire_spec(p))
    assert g.dialect == p.dialect
    t = select_target(g, None, None)
    assert t.profile.id == pid and t.namespace == "" and t.validation.ok
    assert select_target(g, p, None).profile.id == pid       # explicit selection agrees


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_optional_rows_and_infrastructure_are_accepted(fake, pid):
    p = load(pid)
    g = graph_of(fake, wire_spec(p, include_optional=True))
    t = select_target(g, p, None)
    assert t.validation.ok, t.validation.problems()


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_explicit_selection_cannot_bypass_type_checks(fake, pid):
    p = load(pid)
    row = next(e for e in p.required() if e.kind != "action")
    bad = "std_msgs/Bool" if p.dialect == "ros1" else "std_msgs/msg/Bool"
    g = graph_of(fake, retype(wire_spec(p), row.name, bad))
    with pytest.raises(SelectionError, match="wrong type on"):
        select_target(g, p, None)
    with pytest.raises(SelectionError):
        select_target(g, None, None)


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_missing_required_endpoint_is_refused(fake, pid):
    p = load(pid)
    row = p.required()[-1]
    g = graph_of(fake, drop(wire_spec(p), row.name))
    with pytest.raises(SelectionError) as e:
        select_target(g, p, None)
    assert row.name in e.value.reason


def test_ambiguous_overlapping_candidates_name_them(fake):
    spec = merge(wire_spec(load("myagv")), wire_spec(load("rosmaster_x3_plus")))
    g = graph_of(fake, spec)
    with pytest.raises(SelectionError) as e:
        select_target(g, None, None)
    assert "myagv" in e.value.reason and "rosmaster_x3_plus" in e.value.reason
    with pytest.raises(SelectionError, match="ambiguous"):
        select_target(g, load("myagv"), None)
    d = discover(g)
    assert [sorted(t.profile.id for t in grp) for grp in d.ambiguous] == [["myagv", "rosmaster_x3_plus"]]


def without_camera(p, new_id=None):
    """Profile ``p`` booted without its camera: its camera rows dropped (and its id changed)."""
    raw = copy.deepcopy(dict(p.raw))
    cams = {c["topic"] for c in raw.get("cameras") or []}
    raw["topics"] = [r for r in raw["topics"] if r["name"] not in cams]
    raw["cameras"] = []
    raw["id"] = new_id or raw["id"]
    return parse_profile(raw)


def test_subsumed_profile_is_not_selected(fake, monkeypatch):
    """A wire presenting every name one profile requires, and more, is the larger profile's; the
    smaller one is not selected even explicitly. No two packaged profiles nest, so the catalog here
    also holds a profile of the SO-101 booted without its camera, whose required names the SO-101's
    contain."""
    so101 = load("so101")
    nocam = without_camera(so101, "so101_nocam")
    assert {e.name for e in nocam.required()} < {e.name for e in so101.required()}
    monkeypatch.setattr(discovery, "load_all", lambda: [*load_all(), nocam])
    g = graph_of(fake, wire_spec(so101))
    assert select_target(g, None, None).profile.id == "so101"
    with pytest.raises(SelectionError, match="presents so101, whose interface contains all of so101_nocam"):
        select_target(g, nocam, None)
    # Without the camera on the wire, the smaller profile is the wire's.
    g = graph_of(fake, drop(wire_spec(so101), "/image_raw"))
    assert select_target(g, None, None).profile.id == "so101_nocam"


def test_no_supported_robot(fake):
    g = graph_of(fake, {"dialect": "ros1", "topics": [{"name": "/chatter", "type": "std_msgs/String"}],
                        "services": [], "actions": []})
    with pytest.raises(SelectionError, match="no supported robot"):
        select_target(g, None, None)


def test_dialect_is_checked(fake):
    g = graph_of(fake, wire_spec(load("so101")))
    with pytest.raises(SelectionError, match="not found"):
        select_target(g, load("myagv"), None)


def test_unexpected_endpoint_under_owned_names_fails(fake):
    p = load("ainex")
    spec = wire_spec(p)
    spec["topics"].append({"name": "/walking/secret_mode", "type": "std_msgs/Bool", "direction": "in"})
    spec["topics"].append({"name": "/unrelated", "type": "std_msgs/Bool", "direction": "in"})
    g = graph_of(fake, spec)
    with pytest.raises(SelectionError, match="unexpected topic /walking/secret_mode"):
        select_target(g, p, None)


def test_ros2_actions_identified_by_name(fake):
    g = graph_of(fake, wire_spec(load("so101")))
    t = select_target(g, None, None)
    assert {n for _, n, _ in t.validation.unverified} == {
        "/joint_trajectory_controller/follow_joint_trajectory", "/gripper_controller/gripper_cmd"}


# ------------------------------------------------------------------ namespaces (so101 only)

def test_namespace_default_preferred(fake):
    p = load("so101")
    g = graph_of(fake, merge(wire_spec(p), wire_spec(p, namespace="arm2")))
    assert select_target(g, None, None).namespace == ""
    assert select_target(g, p, None).namespace == ""
    assert select_target(g, p, "arm2").namespace == "arm2"


def test_namespace_single_discovered_target(fake):
    p = load("so101")
    g = graph_of(fake, wire_spec(p, namespace="arm1"))
    t = select_target(g, None, None)
    assert (t.profile.id, t.namespace) == ("so101", "arm1")
    assert t.wire("/joint_trajectory_controller/follow_joint_trajectory") == \
        "/arm1/joint_trajectory_controller/follow_joint_trajectory"
    assert t.wire("/tf") == "/tf"


def test_namespace_several_targets_need_explicit_selection(fake):
    p = load("so101")
    g = graph_of(fake, merge(wire_spec(p, namespace="arm1"), wire_spec(p, namespace="arm2")))
    with pytest.raises(SelectionError) as e:
        select_target(g, None, None)
    assert "arm1" in e.value.reason and "arm2" in e.value.reason
    with pytest.raises(SelectionError):
        select_target(g, p, None)
    assert select_target(g, p, "arm1").namespace == "arm1"


def test_namespace_override_refused_for_other_profiles(fake):
    g = graph_of(fake, wire_spec(load("myagv")))
    with pytest.raises(ProfileError, match="no namespace override"):
        select_target(g, load("myagv"), "robot1")
    with pytest.raises(ProfileError, match="no namespace override"):
        select_target(g, None, "robot1")


# ------------------------------------------------------------------ cameras

def test_untied_cameras_list_every_supported_stream(fake):
    spec = merge(wire_spec(load("ainex")), wire_spec(load("rosmaster_x3_plus")))
    g = graph_of(fake, spec)
    topics = [s.topic for s in cam.untied_specs(g)]
    images = sorted(r["name"] for r in spec["topics"] if r["type"] == "sensor_msgs/Image")
    assert topics == images and {"/camera/image_raw", "/camera/rgb/image_raw"} <= set(topics)
    assert all(not s.tied for s in cam.untied_specs(g))
