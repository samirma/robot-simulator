"""The judging half of the wire checks (`wirecheck.py`), on synthetic observations: what
counts as ROS infrastructure (spec Terms), the interface contract, the transform tree, and
the rate, gap and timestamp bounds of spec §5 Acceptance bounds."""

import copy

import pytest

import registry
import wirecheck


def record_snapshot(rid: str) -> dict:
    """What a wire presenting exactly the recorded interface answers."""
    iface = wirecheck.interface(rid)
    dialect = iface["dialect"]
    params = {}
    for row in wirecheck.served(iface.get("parameters")):
        v = row.get("value")
        if isinstance(v, dict) and "file" in v and len(v) <= 2:
            v = (registry.get(rid).folder_path / v["file"]).read_text()
        params[wirecheck.param_key(row, dialect)] = v
    return {"nodes": {n["name"] for n in wirecheck.served(iface["nodes"])},
            "topics": {t["name"]: t["type"] for t in wirecheck.served(iface["topics"])},
            "services": {s["name"]: s["type"] for s in wirecheck.served(iface.get("services"))},
            "actions": [a["name"] for a in wirecheck.served(iface.get("actions"))],
            "params": params, "param_names": []}


def test_infrastructure_is_what_the_terms_list():
    for name in ("/rosapi/topics", "/reBotArmController/get_parameters", "/x/set_logger_levels",
                 "/x/get_type_description", "/arm/follow/_action/send_goal"):
        assert wirecheck.infra_service(name, "ros2"), name
    assert wirecheck.infra_service("/x/get_loggers", "ros1")
    # rclcpp_lifecycle's state services are no client-library service of the Terms
    for leaf in ("change_state", "get_state", "get_available_states",
                 "get_available_transitions", "get_transition_graph"):
        assert not wirecheck.infra_service(f"/joint_state_broadcaster/{leaf}", "ros2"), leaf


@pytest.mark.parametrize("rid", registry.ids())
def test_the_recorded_interface_passes_the_contract(rid):
    assert wirecheck.contract_problems(rid, record_snapshot(rid)) == []


def test_unrecorded_lifecycle_names_fail_the_contract():
    snap = record_snapshot("rebot_b601_rs")
    snap["topics"]["/x/transition_event"] = "lifecycle_msgs/msg/TransitionEvent"
    snap["services"]["/joint_state_broadcaster/get_state"] = "lifecycle_msgs/srv/GetState"
    problems = wirecheck.contract_problems("rebot_b601_rs", snap)
    assert "extra topic /x/transition_event" in problems
    assert "extra service /joint_state_broadcaster/get_state" in problems


def test_contract_reports_missing_extra_and_wrong_names():
    snap = record_snapshot("myagv")
    topic, typ = next(iter(snap["topics"].items()))
    snap["topics"][topic] = typ + "Stamped"
    node = sorted(snap["nodes"])[0]
    snap["nodes"].discard(node)
    snap["nodes"].add("/imposter")
    problems = wirecheck.contract_problems("myagv", snap)
    assert f"missing node {node}" in problems and "extra node /imposter" in problems
    assert f"topic {topic} is {typ}Stamped, recorded {typ}" in problems


def recorded_edges(iface) -> dict:
    """The transform tree a wire sending exactly the recorded `tf` rows shows."""
    edges = {}
    for r in wirecheck.served(iface.get("tf")):
        static = r.get("static") and r.get("rate") == "non_periodic"
        edges[(r["parent"], r["child"])] = {
            "topic": "/tf_static" if static else "/tf",
            "xyz": [float(v) for v in r.get("xyz", [0.0, 0.0, 0.0])],
            "quat": wirecheck.rpy_quat(*[float(v) for v in r.get("rpy", [0.0, 0.0, 0.0])]),
            "changed": False}
    return edges


@pytest.mark.parametrize("rid", registry.ids())
def test_the_recorded_transform_tree_passes(rid):
    iface = wirecheck.interface(rid)
    assert wirecheck.tf_problems(iface, recorded_edges(iface)) == []


def test_transform_tree_differences_are_reported():
    iface = wirecheck.interface("myagv")
    rows = [r for r in wirecheck.served(iface["tf"]) if r.get("static") and "xyz" in r]
    assert len(rows) >= 2
    edges = recorded_edges(iface)
    gone, moved = (rows[0]["parent"], rows[0]["child"]), (rows[1]["parent"], rows[1]["child"])
    del edges[gone]
    edges[moved] = copy.deepcopy(edges[moved])
    edges[moved]["xyz"][0] += 0.01
    edges[("map", "nowhere")] = {"topic": "/tf", "xyz": [0, 0, 0], "quat": [0, 0, 0, 1],
                                 "changed": False}
    problems = wirecheck.tf_problems(iface, edges)
    assert f"missing transform {gone[0]} -> {gone[1]}" in problems
    assert any(p.startswith(f"fixed transform {moved[0]} -> {moved[1]} is xyz") for p in problems)
    assert "extra transform map -> nowhere" in problems


def test_a_served_topic_without_a_recorded_rate_fails():
    iface = {"topics": [{"name": "/a", "direction": "out", "rate": None},
                        {"name": "/b", "direction": "out", "rate": "non_periodic"},
                        {"name": "/c", "direction": "out", "rate": 10},
                        {"name": "/d", "direction": "in"},
                        {"name": "/e", "direction": "out", "rate": None, "optional": True}]}
    problems = wirecheck.rate_record_problems(iface)
    assert len(problems) == 1 and problems[0].startswith("/a")
    for rid in registry.ids():
        assert wirecheck.rate_record_problems(wirecheck.interface(rid)) == [], rid


def measured(arrivals, stamps=None, window=10.0):
    return {"window": window, "times": {"/t": list(arrivals)},
            "stamps": {"/t": list(stamps)} if stamps is not None else {}}


def test_rates_gaps_and_stamps():
    period = 0.1
    t0 = 1.7e9
    even = [t0 + k * period for k in range(100)]
    assert wirecheck.rate_problems(measured(even, even), {"/t": 10.0}) == []
    # delivery stalls for four periods, then bursts: the stamps stay evenly spaced
    burst = even[:50] + [t0 + 49 * period + 4 * period + k * 0.001 for k in range(4)] + \
        [t0 + k * period for k in range(54, 101)]
    problems = wirecheck.rate_problems(measured(burst, even + [even[-1] + period]), {"/t": 10.0})
    assert any("gap" in p for p in problems), problems
    # stamps in another clock (simulation time), and stamps going back
    problems = wirecheck.rate_problems(measured(even, [s - 1000.0 for s in even]), {"/t": 10.0})
    assert any("stamp" in p for p in problems), problems
    back = list(even)
    back[40], back[41] = back[41], back[40]
    problems = wirecheck.rate_problems(measured(even, back), {"/t": 10.0})
    assert any("go back" in p for p in problems), problems
    # too short a window, or a wrong rate
    assert wirecheck.rate_problems(measured(even, window=0.4), {"/t": 10.0})
    assert wirecheck.rate_problems(measured(even[::2]), {"/t": 10.0})


def test_client_library_parameters_are_infrastructure():
    """Spec Terms: the parameters the client library declares on every ROS 2 node."""
    for p in ("use_sim_time", "start_type_description_service",
              "qos_overrides./parameter_events.publisher.depth",
              "qos_overrides./rosout.publisher.durability"):
        assert wirecheck.infra_param(f"/robot_state_publisher:{p}", "ros2"), p
    assert not wirecheck.infra_param("/robot_state_publisher:qos_overrides./tf.publisher.depth",
                                     "ros2")
    assert not wirecheck.infra_param("/robot_state_publisher:publish_frequency", "ros2")
    assert wirecheck.infra_param("/run_id", "ros1") and not wirecheck.infra_param("/use_sim_time", "ros1")


class Answers:
    """A rosbridge stand-in answering /rosapi/get_param_names from a script."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), 0

    def call(self, service, args=None, timeout=None):
        assert service == "/rosapi/get_param_names"
        self.asked += 1
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return {"names": a}


def test_the_parameter_listing_is_asked_again_then_fails(monkeypatch):
    """A listing rosapi gave up on (an error, an empty or a partial list) is asked again;
    one that never comes fails the snapshot instead of passing the parameter checks."""
    monkeypatch.setattr(wirecheck.time, "sleep", lambda s: None)
    found = {"/a": 1, "/ns": {"x": 1}}
    full = ["/a", "/ns/x", "/rosdistro"]
    rb = Answers(RuntimeError("timed out"), [], ["/a"], full)
    with pytest.raises(RuntimeError, match="get_param_names"):
        wirecheck.param_names(rb, found)
    assert rb.asked == wirecheck.PARAM_NAMES_ATTEMPTS == 3
    rb = Answers(RuntimeError("timed out"), full)
    assert wirecheck.param_names(rb, found) == full and rb.asked == 2


def test_each_transform_with_a_recorded_rate_is_judged_on_its_own():
    """Spec §5: where the record states a transform's own rate, that transform is held to
    the acceptance bounds, not only /tf as a whole."""
    iface = {"tf": [{"parent": "odom", "child": "base", "rate": 20},
                    {"parent": "/base", "child": "/laser", "rate": 100.0, "static": True},
                    {"parent": "base", "child": "cam", "rate": "non_periodic", "static": True},
                    {"parent": "map", "child": "odom", "rate": 5, "optional": True}]}
    assert wirecheck.tf_rates(iface) == {"odom -> base": 20.0, "/base -> /laser": 100.0}
    t0 = 1.7e9

    def every(period, n, skip=()):
        return [t0 + k * period for k in range(n) if k not in skip]

    good = {"window": 5.0, "times": {"odom -> base": every(0.05, 100),
                                     "/base -> /laser": every(0.01, 500)},
            "stamps": {"odom -> base": every(0.05, 100),
                       "/base -> /laser": [t + 0.01 for t in every(0.01, 500)]}}
    assert wirecheck.tf_rate_problems(good, iface) == []
    slow = dict(good, times=dict(good["times"], **{"/base -> /laser": every(0.0125, 400)}),
                stamps={})
    assert any(p.startswith("/base -> /laser") for p in wirecheck.tf_rate_problems(slow, iface))
    gap = dict(good, times=dict(good["times"], **{"odom -> base": every(0.05, 100, range(40, 44))}),
               stamps={})
    assert any("gap" in p for p in wirecheck.tf_rate_problems(gap, iface))
    gone = dict(good, times={"odom -> base": every(0.05, 100)}, stamps={})
    assert any(p.startswith("/base -> /laser: 0 messages")
               for p in wirecheck.tf_rate_problems(gone, iface))
    # every record's own transform rates are numbers the check can hold them to
    for rid in registry.ids():
        assert all(r > 0 for r in wirecheck.tf_rates(wirecheck.interface(rid)).values()), rid


#: rosapi's message_details answer for sensor_msgs/CameraInfo (ROS 1), abridged to the
#: fields the check walks
CAMERA_INFO = [
    {"type": "sensor_msgs/CameraInfo",
     "fieldnames": ["header", "height", "width", "distortion_model", "D", "K", "R", "P",
                    "binning_x", "binning_y", "roi"],
     "fieldtypes": ["std_msgs/Header", "uint32", "uint32", "string", "float64", "float64",
                    "float64", "float64", "uint32", "uint32", "sensor_msgs/RegionOfInterest"],
     "fieldarraylen": [-1, -1, -1, -1, 0, 9, 9, 12, -1, -1, -1]},
    {"type": "std_msgs/Header", "fieldnames": ["seq", "stamp", "frame_id"],
     "fieldtypes": ["uint32", "time", "string"], "fieldarraylen": [-1, -1, -1]},
    {"type": "sensor_msgs/RegionOfInterest",
     "fieldnames": ["x_offset", "y_offset", "height", "width", "do_rectify"],
     "fieldtypes": ["uint32", "uint32", "uint32", "uint32", "bool"],
     "fieldarraylen": [-1, -1, -1, -1, -1]}]


def camera_info():
    return {"header": {"seq": 3, "stamp": {"secs": 1, "nsecs": 2}, "frame_id": "cam"},
            "height": 480, "width": 640, "distortion_model": "plumb_bob", "D": [0.0] * 5,
            "K": [1.0] * 9, "R": [0.0] * 9, "P": [0] * 12, "binning_x": 0, "binning_y": 0,
            "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0,
                    "do_rectify": False}}


def test_a_message_is_judged_field_for_field_through_every_nested_type():
    """Spec §5 Contracts: every published message matches its declared type field for
    field -- the nested ones too."""
    ok = camera_info()
    assert wirecheck.typedef_problems(ok, "sensor_msgs/CameraInfo", CAMERA_INFO) == []
    bad = camera_info()
    bad["roi"]["do_rectify"] = 0                       # an integer, not a bool
    bad["header"]["stamp"] = 1.5                       # a float, not a ROS 1 time
    bad["K"] = [1.0] * 8                               # a fixed-size array of the wrong size
    bad["width"] = "640"                               # a string, not an integer
    del bad["binning_y"]                               # a missing field
    bad["extra"] = 1                                   # an extra one
    problems = wirecheck.typedef_problems(bad, "sensor_msgs/CameraInfo", CAMERA_INFO)
    for fragment in ("roi.do_rectify", "header.stamp", "K: 8 elements", "width:",
                     "fields ["):
        assert any(fragment in p for p in problems), (fragment, problems)
    # a uint8[] comes as base64 (rosbridge), and a ROS 2 type name as rosapi gives it
    image = [{"type": "sensor_msgs/Image", "fieldnames": ["data", "step"],
              "fieldtypes": ["uint8", "uint32"], "fieldarraylen": [0, -1]}]
    assert wirecheck.typedef_problems({"data": "AAEC", "step": 3}, "sensor_msgs/msg/Image",
                                      image) == []
    assert wirecheck.typedef_problems({"data": "%%", "step": 3}, "sensor_msgs/Image", image)
    assert wirecheck.norm_type("control_msgs/action/FollowJointTrajectory") == \
        "control_msgs/FollowJointTrajectory"
