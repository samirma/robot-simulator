"""What the fleet check expects of the member `run_task.sh --robot` names, pinned.

`fleet.py --expect` is the answer to "is the robot I asked for actually on this
rosbridge": the id must be a typed member on the wire presenting its whole contract, from
the console's *own* contract constants rather than a list typed into a shell script.

Offline by construction -- `check_expected` is pure, and the topic list it is given here
is a dict a live `/rosapi/topics` would have returned.
"""

from __future__ import annotations

import pytest

from robot_console import fleet
from robot_console.discovery import MEMBER_SIGNATURES
from robot_console.fleet import ROBOT_IDS, check_expected, main
from robot_console.topics import namespaced


def _full_wire(*members: tuple[str, str]) -> dict[str, str]:
    wire: dict[str, str] = {}
    for kind, namespace in members:
        for bare, kind_type in fleet.contract_of(kind).items():
            wire[namespaced(bare, namespace)] = kind_type
    return wire


def test_the_ids_are_the_members_the_console_has_a_contract_for() -> None:
    assert set(ROBOT_IDS) == {kind for kind, _, _ in MEMBER_SIGNATURES}
    assert "scene" not in ROBOT_IDS


@pytest.mark.parametrize("kind", [k for k in ROBOT_IDS if k != "so101"])
def test_every_mobile_member_passes_on_its_own_contract(kind: str) -> None:
    """The X3 PLUS and the composite are checked as themselves, not as a myAGV."""
    members, problems = check_expected(_full_wire((kind, kind)), kind)
    assert problems == []
    assert [(m.kind, m.namespace) for m in members] == [(kind, kind)]


def test_an_absent_member_is_named_with_what_was_found() -> None:
    _, problems = check_expected(_full_wire(("myagv", "myagv")), "ainex")
    assert problems == ["no ainex is on the wire (found: myagv on /myagv/*)"]


def test_a_missing_or_mistyped_topic_is_reported_by_its_wire_name() -> None:
    wire = _full_wire(("rosmaster_x3_plus", "rosmaster_x3_plus"))
    del wire["/rosmaster_x3_plus/scan"]
    wire["/rosmaster_x3_plus/odom"] = "std_msgs/String"
    _, problems = check_expected(wire, "rosmaster_x3_plus")
    assert any("/rosmaster_x3_plus/scan is missing" in p for p in problems)
    assert any("/rosmaster_x3_plus/odom is std_msgs/String" in p for p in problems)


def test_a_mistyped_signature_is_not_the_member() -> None:
    wire = _full_wire(("myagv", "myagv"))
    wire["/myagv/cmd_vel"] = "std_msgs/String"
    members, problems = check_expected(wire, "myagv")
    assert members == []
    assert any("/myagv/cmd_vel is std_msgs/String, not geometry_msgs/Twist" in p
               for p in problems)


def test_members_not_expected_are_not_checked() -> None:
    wire = _full_wire(("myagv", "myagv"), ("ainex", "ainex"))
    del wire["/ainex/imu"]
    assert check_expected(wire, "myagv")[1] == []


@pytest.mark.parametrize("value", ["scene", "myagv,ainex"])
def test_the_expect_flag_takes_one_known_id_and_lists_them_otherwise(value, capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--expect", value, "--url", "ws://127.0.0.1:1"])
    err = capsys.readouterr().err
    assert all(rid in err for rid in ROBOT_IDS)


# ------------------------------------------------------------------ --dump


def test_dump_lists_every_kind_of_interface_name_sorted(bridge) -> None:
    """Console spec §2.5: nodes, topics, services, actions, types, frames and parameters,
    one line each, sorted, from the wire's own answers."""
    import threading
    import time

    from robot_console.fleet import dump_wire

    bridge.topics = {"/b/odom": "nav_msgs/Odometry", "/a/tf_static": "tf2_msgs/TFMessage",
                     "/a/cmd_vel": "geometry_msgs/Twist"}
    bridge.answers = {
        "/rosapi/nodes": {"nodes": ["/z", "/a"]},
        "/rosapi/services": {"services": ["/a/reset"]},
        "/rosapi/service_type": {"type": "std_srvs/Trigger"},
        "/rosapi/action_servers": {"action_servers": ["/a/go"]},
        "/rosapi/action_type": {"type": "pkg/action/Go"},
        "/rosapi/get_param_names": {"names": ["/a/p"]},
        "/rosapi/get_param": {"value": "1.5"},
    }

    def publish_tf() -> None:
        for _ in range(20):
            bridge.publish("/a/tf_static", {"transforms": [
                {"header": {"frame_id": "/a/base"}, "child_frame_id": "a/laser"}]})
            time.sleep(0.05)

    thread = threading.Thread(target=publish_tf, daemon=True)
    thread.start()
    lines = dump_wire(f"ws://127.0.0.1:{bridge.port}", timeout_s=5, frame_window_s=1.0)
    thread.join()
    assert lines == sorted(lines)
    assert lines == [
        "action\t/a/go\tpkg/action/Go",
        "frame\ta/base\ta/laser",
        "node\t/a",
        "node\t/z",
        "param\t/a/p\t1.5",
        "service\t/a/reset\tstd_srvs/Trigger",
        "topic\t/a/cmd_vel\tgeometry_msgs/Twist",
        "topic\t/a/tf_static\ttf2_msgs/TFMessage",
        "topic\t/b/odom\tnav_msgs/Odometry",
    ]
    assert not bridge.subscriptions, "the dump lets go of the transform topics"


def test_dump_of_an_unreachable_wire_is_a_transport_error(capsys) -> None:
    import socket

    from robot_console import fleet

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert fleet.main(["--url", f"ws://127.0.0.1:{port}", "--dump"]) == fleet.EXIT_TRANSPORT
    assert "cannot reach" in capsys.readouterr().out


def test_the_flags_are_exactly_the_spec_synopsis(monkeypatch) -> None:
    """Console spec §2.5: [--url] [--expect <id>] [--dump] [--rates [--gate]]."""
    import argparse

    seen: dict = {}
    real = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        seen["flags"] = {s for a in self._actions for s in a.option_strings if s.startswith("--")}
        return real(self, args, namespace)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    with pytest.raises(SystemExit):
        main(["--help"])
    assert seen["flags"] - {"--help"} == {"--url", "--expect", "--dump", "--rates", "--gate"}
