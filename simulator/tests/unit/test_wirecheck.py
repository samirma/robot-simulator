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
    for name in ("/rosapi/topics", "/so101/get_parameters", "/x/set_logger_levels",
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
    snap = record_snapshot("so101")
    snap["topics"]["/x/transition_event"] = "lifecycle_msgs/msg/TransitionEvent"
    snap["services"]["/joint_state_broadcaster/get_state"] = "lifecycle_msgs/srv/GetState"
    problems = wirecheck.contract_problems("so101", snap)
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
