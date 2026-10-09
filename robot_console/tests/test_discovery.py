"""Typed discovery, validation and target selection (console spec §1.1, §2.4, §4)."""

import copy

import pytest

from robot_console import camera as cam
from robot_console import discovery
from robot_console.discovery import SelectionError, discover, fetch_graph, select_target
from robot_console.profiles import SUPPORTED_IDS, ProfileError, load, load_all, parse_profile
from robot_console.rosbridge import Rosbridge
from wirespec import drop, merge, retype, wire_spec, with_hidden_action_endpoints


def wire_name(p, name):
    """Documented name ``name`` on a wire booted with the profile's default namespace (wire_spec's)."""
    return p.resolve(name, p.namespace.default if p.namespace else "")


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
    default = p.namespace.default if p.namespace else ""      # the documented hardware default
    assert t.profile.id == pid and t.namespace == default and t.validation.ok
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
    g = graph_of(fake, retype(wire_spec(p), wire_name(p, row.name), bad))
    with pytest.raises(SelectionError, match="wrong type on"):
        select_target(g, p, None)
    with pytest.raises(SelectionError):
        select_target(g, None, None)


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_missing_required_endpoint_is_refused(fake, pid):
    p = load(pid)
    row = p.required()[-1]
    g = graph_of(fake, drop(wire_spec(p), wire_name(p, row.name)))
    with pytest.raises(SelectionError) as e:
        select_target(g, p, None)
    assert wire_name(p, row.name) in e.value.reason


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


#: The reBot's required row the subset variant below drops: its Cartesian goal, which no page
#: control uses (any required row would do; the reBot has no camera to drop).
SUBSET_DROPPED = "/move_to_pose"


def without_row(p, name, new_id=None):
    """Profile ``p`` booted without its endpoint ``name``: that row dropped (and its id changed)."""
    raw = copy.deepcopy(dict(p.raw))
    for sec in ("topics", "services", "actions"):
        raw[sec] = [r for r in raw.get(sec) or [] if r["name"] != name]
    raw["cameras"] = [c for c in raw.get("cameras") or [] if c["topic"] != name]
    raw["id"] = new_id or raw["id"]
    return parse_profile(raw)


def test_subsumed_profile_is_not_selected(fake, monkeypatch):
    """A wire presenting every name one profile requires, and more, is the larger profile's; the
    smaller one is not selected even explicitly. No two packaged profiles nest, so the catalog here
    also holds a profile of the reBot booted without its move_to_pose action, whose required names
    the reBot's contain."""
    rebot = load("rebot_b601_rs")
    subset = without_row(rebot, SUBSET_DROPPED, "rebot_subset")
    assert {e.name for e in subset.required()} < {e.name for e in rebot.required()}
    monkeypatch.setattr(discovery, "load_all", lambda: [*load_all(), subset])
    g = graph_of(fake, wire_spec(rebot))
    assert select_target(g, None, None).profile.id == "rebot_b601_rs"
    with pytest.raises(SelectionError, match=r"presents rebot_b601_rs \(namespace /rebotarm\), whose interface "
                                             "contains all of rebot_subset"):
        select_target(g, subset, None)
    # Without that row on the wire, the smaller profile is the wire's.
    g = graph_of(fake, drop(wire_spec(rebot), rebot.resolve(SUBSET_DROPPED, "rebotarm")))
    assert select_target(g, None, None).profile.id == "rebot_subset"


def test_no_supported_robot(fake):
    g = graph_of(fake, {"dialect": "ros1", "topics": [{"name": "/chatter", "type": "std_msgs/String"}],
                        "services": [], "actions": []})
    with pytest.raises(SelectionError, match="no supported robot"):
        select_target(g, None, None)


def test_dialect_is_checked(fake):
    g = graph_of(fake, wire_spec(load("rebot_b601_rs")))
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
    g = graph_of(fake, wire_spec(load("rebot_b601_rs")))
    t = select_target(g, None, None)
    assert {n for _, n, _ in t.validation.unverified} == {
        "/rebotarm/move_to_pose", "/rebotarm/follow_joint_trajectory", "/rebotarm/gripper/command"}


def test_ros2_action_type_from_listed_hidden_send_goal_service(fake):
    """A rosapi that lists hidden `_action` services reveals the action's type through its
    send_goal service: the documented type is verified, another one is refused, also when the
    profile is selected explicitly (the page does the same, test_view_page.py)."""
    p = load("rebot_b601_rs")
    arm = "/rebotarm/follow_joint_trajectory"
    g = graph_of(fake, with_hidden_action_endpoints(wire_spec(p), arm, "control_msgs/action/FollowJointTrajectory"))
    t = select_target(g, p, None)
    assert t.validation.ok and arm not in {n for _, n, _ in t.validation.unverified}
    g = graph_of(fake, with_hidden_action_endpoints(wire_spec(p), arm, "control_msgs/action/GripperCommand"))
    for robot in (p, None):
        with pytest.raises(SelectionError, match=f"wrong type on action {arm}: expected "
                                                 "control_msgs/action/FollowJointTrajectory, found "
                                                 "control_msgs/action/GripperCommand"):
            select_target(g, robot, None)


# ------------------------------------------------------------------ namespaces (rebot_b601_rs only)

def test_namespace_default_preferred(fake):
    """The bringup's arm_namespace default is "rebotarm": an omitted namespace selects it when present."""
    p = load("rebot_b601_rs")
    g = graph_of(fake, merge(wire_spec(p), wire_spec(p, namespace="arm2")))
    assert select_target(g, None, None).namespace == "rebotarm"
    assert select_target(g, p, None).namespace == "rebotarm"
    assert select_target(g, p, "arm2").namespace == "arm2"


def test_namespace_single_discovered_target(fake):
    p = load("rebot_b601_rs")
    g = graph_of(fake, wire_spec(p, namespace="arm1"))
    t = select_target(g, None, None)
    assert (t.profile.id, t.namespace) == ("rebot_b601_rs", "arm1")
    assert t.wire("/follow_joint_trajectory") == "/arm1/follow_joint_trajectory"
    assert t.wire("/tf") == "/tf" and t.wire("/robot_description") == "/robot_description"


def test_namespace_several_targets_need_explicit_selection(fake):
    p = load("rebot_b601_rs")
    g = graph_of(fake, merge(wire_spec(p, namespace="arm1"), wire_spec(p, namespace="arm2")))
    with pytest.raises(SelectionError) as e:
        select_target(g, None, None)
    assert "arm1" in e.value.reason and "arm2" in e.value.reason
    with pytest.raises(SelectionError):
        select_target(g, p, None)
    assert select_target(g, p, "arm1").namespace == "arm1"


def ns_camera_arm(model_feedback=True):
    """A synthetic ROS 2 arm (tests only) carrying what the console supports but no packaged profile
    uses: joint names composed with the namespace (``namespace.joint_prefix`` "{ns}/", in the
    trajectory's joint names and the model's reported joints) and a ROS 2 ``sensor_msgs/msg/Image``
    camera under the namespace. Built, through the profile loader, from the reBot's profile;
    without ``model_feedback`` its model documents no feedback, so the page pairs the trajectory's
    joint names with its fields to show the measured joints."""
    raw = copy.deepcopy(load("rebot_b601_rs").raw)
    if not model_feedback:
        del raw["model"]["feedback"]
    raw.update(id="ns_camera_arm", name="Synthetic namespaced arm with a ROS 2 camera (tests only)")
    raw["namespace"]["joint_prefix"] = "{ns}/"
    raw["topics"].append({"name": "/camera/image_raw", "type": "sensor_msgs/msg/Image", "direction": "out",
                          "rate_hz": 10, "optional": False, "source": "test"})
    raw["cameras"] = [{"topic": "/camera/image_raw", "type": "sensor_msgs/msg/Image", "encodings": ["rgb8"],
                       "stale_after_s": 1.0, "rate_hz": 10, "source": "test"}]
    arm = next(c for c in raw["controls"] if c["id"] == "arm_trajectory")
    arm["template"]["trajectory"]["joint_names"] = [f"{{joint_prefix}}joint{i}" for i in range(1, 7)]
    for j in raw["model"]["joints"]:
        if j.get("reported"):
            j["reported"] = "{joint_prefix}" + j["reported"]
    return parse_profile(raw)


def test_joint_prefix_namespace_and_ros2_camera_of_a_synthetic_arm(fake, monkeypatch):
    """The namespace's joint prefix is parsed into the profile and handed to the page; a ROS 2 raw
    camera is resolved under the selected namespace, discovered as a supported stream and reported
    live from sensor_msgs/msg/Image frames (console spec §1.1, §2.3, §2.4)."""
    import io

    from robot_console.fleet import check_cameras
    from robot_console.profiles import profile_json
    p = ns_camera_arm()
    assert p.namespace.joint_prefix == "{ns}/" and profile_json(p)["namespace"]["joint_prefix"] == "{ns}/"
    assert p.cameras[0].type == "sensor_msgs/msg/Image"
    monkeypatch.setattr(discovery, "load_all", lambda: [*load_all(), p])
    srv = fake(merge(wire_spec(p, namespace="arm1"), wire_spec(p, namespace="arm2")))
    rb = Rosbridge(srv.url).connect()
    try:
        g = fetch_graph(rb)
        assert {s.topic for s in cam.untied_specs(g)} == {"/arm1/camera/image_raw", "/arm2/camera/image_raw"}
        t = select_target(g, p, "arm2")
        assert t.validation.ok and t.wire("/camera/image_raw") == "/arm2/camera/image_raw"
        (spec,) = cam.target_specs(t)
        assert (spec.topic, spec.type) == ("/arm2/camera/image_raw", "sensor_msgs/msg/Image")
        out = io.StringIO()
        assert check_cameras(rb, [t], g, out), out.getvalue()
        assert "camera /arm2/camera/image_raw: LIVE" in out.getvalue()
    finally:
        rb.close()
    assert srv.command_ops() == []


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
