"""Reading a robot off `/rosapi/topics`.

Pure, over the `{topic: type}` dict rosapi answers with, so the whole of what the console
concludes about a wire is tested with no wire. The one thing these cannot cover -- that a
discovery pass leaves roslibpy's process-global reactor alive for the connection that
follows it -- is in `test_link_roundtrip.py`, behind the live marker.
"""

from __future__ import annotations

import pytest

from robot_console import ainex_topics
from robot_console.discovery import (
    CAMERA_TYPES,
    DiscoveryError,
    choose,
    discover,
    find_robots,
    namespace_of,
)
from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TOPIC_SCAN,
    TYPE_COMPRESSED_IMAGE,
    namespaced,
)

ROS2_IMAGE = "sensor_msgs/msg/CompressedImage"


def _myagv(namespace: str = "") -> dict[str, str]:
    """What a myAGV puts on the wire, under `namespace`."""
    return {
        namespaced(TOPIC_CMD_VEL, namespace): "geometry_msgs/Twist",
        namespaced(TOPIC_ODOM, namespace): "nav_msgs/Odometry",
        namespaced(TOPIC_SCAN, namespace): "sensor_msgs/LaserScan",
        namespaced(TOPIC_CAMERA, namespace): TYPE_COMPRESSED_IMAGE,
    }


def _ainex(namespace: str = "") -> dict[str, str]:
    return {
        namespaced(ainex_topics.TOPIC_SET_WALKING_PARAM, namespace): "ainex_interfaces/WalkingParam",
        namespaced(ainex_topics.TOPIC_IS_WALKING, namespace): "std_msgs/Bool",
        namespaced(TOPIC_CAMERA, namespace): TYPE_COMPRESSED_IMAGE,
    }


def _so101(namespace: str = "so101") -> dict[str, str]:
    return {
        namespaced("/joint_trajectory_controller/joint_trajectory", namespace):
            "trajectory_msgs/msg/JointTrajectory",
        namespaced("/joint_states", namespace): "sensor_msgs/msg/JointState",
        namespaced("/wrist/color/compressed", namespace): ROS2_IMAGE,
    }


def _scene() -> dict[str, str]:
    return {
        "/scene/overhead/color/compressed": ROS2_IMAGE,
        "/scene/side/color/compressed": ROS2_IMAGE,
    }


def test_the_simulators_default_namespace_is_found() -> None:
    """`./run.sh view --robot myagv` puts the base on /myagv/*, and nothing says so."""
    found = choose(find_robots(_myagv("myagv")))
    assert (found.robot, found.namespace) == ("myagv", "myagv")
    assert found.camera_topic == "/myagv/camera/image_raw/compressed"


def test_the_bare_contract_is_a_namespace_too() -> None:
    """A real vendor bringup, and `--ros-namespace ''`. An empty answer is an answer."""
    found = choose(find_robots(_myagv("")))
    assert (found.robot, found.namespace) == ("myagv", "")
    assert found.camera_topic == TOPIC_CAMERA


def test_a_bare_ainex_is_not_read_as_a_robot_called_walking() -> None:
    """Its signature has two segments of its own, so the namespace cannot be the first.

    Taking the leading segment off `/walking/set_param` invents a namespace `walking` and
    then hunts for the robot's camera inside it, which finds nothing.
    """
    found = choose(find_robots(_ainex("")))
    assert (found.robot, found.namespace) == ("ainex", "")


def test_an_ainex_is_found_by_its_walking_topic() -> None:
    found = choose(find_robots(_ainex("ainex")))
    assert (found.robot, found.namespace) == ("ainex", "ainex")
    assert found.camera_topic == "/ainex/camera/image_raw/compressed"


def test_an_arm_and_a_rig_are_not_robots_this_console_drives() -> None:
    """`--robots so101,myagv` plus the worktop rig: one drivable robot, and it is the base.

    The rig under `scene` has cameras and no command topic, which is exactly why the
    signature is a command topic -- it is the difference between a robot and a camera.
    """
    present = {**_so101(), **_myagv("myagv"), **_scene()}
    found = find_robots(present)
    assert [(d.robot, d.namespace) for d in found] == [("myagv", "myagv")]


def test_naming_the_robot_narrows_a_mixed_fleet() -> None:
    present = {**_myagv("myagv"), **_ainex("ainex")}
    assert choose(find_robots(present), "ainex").namespace == "ainex"
    assert choose(find_robots(present), "myagv").namespace == "myagv"


def test_two_of_a_kind_is_the_users_question_to_answer() -> None:
    present = {**_myagv("robot_1"), **_myagv("robot_2")}
    with pytest.raises(DiscoveryError) as exc:
        choose(find_robots(present), "myagv")
    assert "robot_1" in str(exc.value) and "robot_2" in str(exc.value)


def test_a_wire_with_nothing_drivable_on_it_says_what_is_there() -> None:
    with pytest.raises(DiscoveryError) as exc:
        choose(find_robots({**_so101(), **_scene()}))
    assert "--robot" in str(exc.value)


def test_asking_for_a_robot_that_is_not_there_names_the_one_that_is() -> None:
    with pytest.raises(DiscoveryError) as exc:
        choose(find_robots(_myagv("myagv")), "ainex")
    assert "myagv" in str(exc.value)


def test_a_camera_named_by_its_driver_is_still_found() -> None:
    """`/color/compressed` behind a RealSense-style node, not the contract's name."""
    present = dict(_myagv("myagv"))
    del present["/myagv/camera/image_raw/compressed"]
    present["/myagv/color/compressed"] = TYPE_COMPRESSED_IMAGE
    assert choose(find_robots(present)).camera_topic == "/myagv/color/compressed"


def test_both_dialects_of_the_image_type_are_matched() -> None:
    """Two robots on one graph can speak two dialects; rosapi reports each verbatim.

    Matching one string found half the cameras on the wire, which reads as a simulator
    that failed to render rather than as a client asking the wrong question.
    """
    assert {TYPE_COMPRESSED_IMAGE, ROS2_IMAGE} <= CAMERA_TYPES
    present = dict(_myagv("myagv"))
    del present["/myagv/camera/image_raw/compressed"]
    present["/myagv/color/compressed"] = ROS2_IMAGE
    assert choose(find_robots(present)).camera_topic == "/myagv/color/compressed"


def test_two_cameras_in_one_namespace_fall_back_to_the_contract_name() -> None:
    """A guess between two streams is worse than the name the contract already gives."""
    present = dict(_myagv("myagv"))
    del present["/myagv/camera/image_raw/compressed"]
    present["/myagv/color/compressed"] = TYPE_COMPRESSED_IMAGE
    present["/myagv/front/compressed"] = TYPE_COMPRESSED_IMAGE
    assert choose(find_robots(present)).camera_topic == "/myagv/camera/image_raw/compressed"


@pytest.mark.parametrize(
    "topic,signature,expected",
    [
        ("/cmd_vel", "/cmd_vel", ""),
        ("/myagv/cmd_vel", "/cmd_vel", "myagv"),
        ("/robot_1/walking/set_param", "/walking/set_param", "robot_1"),
        ("/walking/set_param", "/walking/set_param", ""),
        ("/myagv2/cmd_vel", "/odom", None),
        ("/cmd_vel_stamped", "/cmd_vel", None),
    ],
)
def test_the_namespace_is_whatever_composes_the_signature(topic, signature, expected) -> None:
    assert namespace_of(topic, signature) == expected

def test_a_transport_failure_is_not_a_discovery_error(monkeypatch) -> None:
    """The two arrive as different exceptions because the caller treats them differently:
    a wire that cannot be asked falls back to the bare contract, a wire that answers and
    holds nothing drivable stops and asks the user.

    Patched rather than dialled, because starting roslibpy's process-global reactor in the
    offline suite is exactly what `test_link_roundtrip.py` exists to keep to one place.
    """
    import robot_console.fleet as fleet

    def _boom(url, timeout_s=10.0):
        raise ConnectionError("no rosbridge there")

    monkeypatch.setattr(fleet, "list_topics", _boom)
    with pytest.raises(ConnectionError):
        discover("ws://127.0.0.1:9090")
