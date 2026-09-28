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
