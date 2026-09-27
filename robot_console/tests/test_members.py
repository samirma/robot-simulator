"""`discovery.find_members`: every fleet member on a wire, by typed signature.

The wider sibling of `find_robots`, which only knows what teleop can drive. The fleet
check and the camera page both ask this question; the page's copy of the table is held
equal to `MEMBER_SIGNATURES` in `test_view_page.py`.
"""

from __future__ import annotations

from robot_console.discovery import MEMBER_SIGNATURES, RIG_SIGNATURE, Member, find_members
from robot_console.topics import namespaced

SIG = {kind: (topic, kind_type) for kind, topic, kind_type in MEMBER_SIGNATURES}


def _member(kind: str, namespace: str) -> dict[str, str]:
    topic, kind_type = SIG[kind]
    return {namespaced(topic, namespace): kind_type}


def test_a_mixed_fleet_and_its_rig() -> None:
    wire = {**_member("so101", "so101"), **_member("myagv", "myagv"),
            **_member("ainex", "ainex"), RIG_SIGNATURE[0]: RIG_SIGNATURE[1]}
    members, wrong = find_members(wire)
    assert wrong == []
    assert members == [Member("ainex", "ainex"), Member("myagv", "myagv"),
                       Member("scene", "scene"), Member("so101", "so101")]


def test_a_lone_bare_robot_is_the_empty_namespace() -> None:
    members, _ = find_members(_member("so101", ""))
    assert members == [Member("so101", "")]
    assert members[0].describe() == "so101 on the bare contract"


def test_two_of_a_kind_are_two_members() -> None:
    members, _ = find_members({**_member("myagv", "a"), **_member("myagv", "b")})
    assert members == [Member("myagv", "a"), Member("myagv", "b")]


def test_a_signature_with_the_wrong_type_is_reported_not_counted() -> None:
    members, wrong = find_members({
        "/x/cmd_vel": "std_msgs/String",
        # The arm's type in the ROS 1 spelling is not the arm: dialects are not folded.
        "/y/joint_trajectory_controller/joint_trajectory": "trajectory_msgs/JointTrajectory",
        RIG_SIGNATURE[0]: "sensor_msgs/CompressedImage",
    })
    assert members == []
    assert wrong == sorted([
        "/x/cmd_vel is std_msgs/String, not geometry_msgs/Twist",
        "/y/joint_trajectory_controller/joint_trajectory is trajectory_msgs/JointTrajectory, "
        "not trajectory_msgs/msg/JointTrajectory",
        f"{RIG_SIGNATURE[0]} is sensor_msgs/CompressedImage, not {RIG_SIGNATURE[1]}",
    ])


def test_one_namespace_is_one_member_the_most_specific() -> None:
    members, _ = find_members({**_member("myagv", "r"), **_member("ainex", "r")})
    assert members == [Member("ainex", "r")]
