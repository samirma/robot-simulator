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
    discover_from,
    find_robots,
    survey,
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


def _composite(namespace: str = "") -> dict[str, str]:
    """The myAGV's topics, and move_base's beside them."""
    return {
        **_myagv(namespace),
        namespaced("/move_base/goal", namespace): "move_base_msgs/MoveBaseActionGoal",
        namespaced("/move_base/cancel", namespace): "actionlib_msgs/GoalID",
    }


def _x3(namespace: str = "") -> dict[str, str]:
    """A ROSMASTER X3 PLUS: the same Twist and odometry, its driver's arm topic, and a raw
    camera only."""
    return {
        namespaced(TOPIC_CMD_VEL, namespace): "geometry_msgs/Twist",
        namespaced(TOPIC_ODOM, namespace): "nav_msgs/Odometry",
        namespaced(TOPIC_SCAN, namespace): "sensor_msgs/LaserScan",
        namespaced("/TargetAngle", namespace): "yahboomcar_msgs/ArmJoint",
        namespaced("/camera/rgb/image_raw", namespace): "sensor_msgs/Image",
    }


def _scene() -> dict[str, str]:
    return {
        "/scene/overhead/color/compressed": ROS2_IMAGE,
        "/scene/side/color/compressed": ROS2_IMAGE,
    }


def test_the_simulators_default_namespace_is_found() -> None:
    """`./kitchen.sh serve --robots myagv` puts the base on /myagv/*, and nothing says so."""
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

def test_three_cmd_vel_bases_on_one_wire_are_told_apart() -> None:
    """The myAGV, the composite and the X3 PLUS all take /cmd_vel and report /odom; each
    is still itself, by the command topic only it has."""
    wire = {**_myagv("myagv"), **_composite("myagv_mycobot280"), **_x3("rosmaster_x3_plus"),
            **_so101(), **_scene()}
    found, rejected = survey(wire)
    assert {(d.robot, d.namespace) for d in found} == {
        ("myagv", "myagv"), ("myagv_mycobot280", "myagv_mycobot280"),
        ("rosmaster_x3_plus", "rosmaster_x3_plus")}
    assert rejected == []
    for robot in ("myagv", "myagv_mycobot280", "rosmaster_x3_plus"):
        assert choose(found, robot).namespace == robot
    with pytest.raises(DiscoveryError, match="3 robots match") as err:
        choose(found)
    for candidate in ("myagv on /myagv/*", "myagv_mycobot280 on /myagv_mycobot280/*",
                      "rosmaster_x3_plus on /rosmaster_x3_plus/*"):
        assert candidate in str(err.value)
    assert choose(found, "rosmaster_x3_plus").camera_topic == \
        "/rosmaster_x3_plus/camera/rgb/image_raw"


@pytest.mark.parametrize("make,robot", [(_composite, "myagv_mycobot280"),
                                        (_x3, "rosmaster_x3_plus")])
def test_a_lone_bare_mobile_manipulator_is_not_taken_for_a_myagv(make, robot) -> None:
    found = choose(find_robots(make("")))
    assert (found.robot, found.namespace) == (robot, "")
    with pytest.raises(DiscoveryError, match="no myagv"):
        discover_from(make(""), "myagv")


def test_a_composite_missing_its_cancel_topic_is_rejected_by_name() -> None:
    wire = _composite("c")
    del wire["/c/move_base/cancel"]
    found, rejected = survey(wire)
    assert found == []
    assert [(r.robot, r.namespace) for r in rejected] == [("myagv_mycobot280", "c")]
    assert "missing /c/move_base/cancel" in rejected[0].reason


def test_a_transport_failure_is_not_a_discovery_error(monkeypatch) -> None:
    """`discover` lets a transport failure through as itself; the supervisor turns it
    into the "could not ask /rosapi" error (see the end-to-end cases below).

    Patched rather than dialled, because starting roslibpy's process-global reactor in the
    offline suite is exactly what `test_link_roundtrip.py` exists to keep to one place.
    """
    import robot_console.fleet as fleet

    def _boom(url, timeout_s=10.0):
        raise ConnectionError("no rosbridge there")

    monkeypatch.setattr(fleet, "list_topics", _boom)
    with pytest.raises(ConnectionError):
        discover("ws://127.0.0.1:9090")


# ------------------------------------------------------------------ console spec §4 cases


def test_a_missing_distinguishing_topic_rules_a_candidate_out_and_says_so() -> None:
    """A `/cmd_vel` with no `/odom` beside it is some Twist robot, not a myAGV."""
    present = dict(_myagv("myagv"))
    del present["/myagv/odom"]
    found, rejected = survey(present)
    assert found == []
    assert [(r.robot, r.namespace) for r in rejected] == [("myagv", "myagv")]
    with pytest.raises(DiscoveryError) as exc:
        discover_from(present)
    assert "missing /myagv/odom" in str(exc.value)


def test_a_wrong_type_rules_a_candidate_out_and_says_so() -> None:
    present = dict(_ainex("ainex"))
    present["/ainex/walking/set_param"] = "std_msgs/String"
    with pytest.raises(DiscoveryError) as exc:
        discover_from(present)
    assert "std_msgs/String" in str(exc.value)
    assert "ainex_interfaces/WalkingParam" in str(exc.value)


@pytest.mark.parametrize("topic", ["/myagv/cmd_vel", "/myagv/odom"])
def test_an_untyped_signature_or_companion_is_a_wrong_type_and_says_so(topic) -> None:
    """One typing rule with `find_members`: an empty type is not the contract's type."""
    present = dict(_myagv("myagv"))
    present[topic] = ""
    found, rejected = survey(present)
    assert found == []
    assert [(r.robot, r.namespace) for r in rejected] == [("myagv", "myagv")]
    with pytest.raises(DiscoveryError) as exc:
        discover_from(present)
    assert f"{topic} is untyped, not " in str(exc.value)


def test_teleop_and_the_fleet_check_agree_on_every_typing_case() -> None:
    """For each signature typed right, wrong, and not at all, the two questions agree:
    a candidate teleop drives is a member, and a rejected one is a reported wrong type."""
    from robot_console.discovery import find_members

    for stated in ("geometry_msgs/Twist", "geometry_msgs/TwistStamped", ""):
        present = dict(_myagv("myagv"))
        present["/myagv/cmd_vel"] = stated
        found, rejected = survey(present)
        members, wrong = find_members(present)
        assert bool(found) == bool(members) == (stated == "geometry_msgs/Twist"), stated
        assert bool(rejected) == bool(wrong) == (stated != "geometry_msgs/Twist"), stated
        if wrong:
            detail = wrong[0].split(" is ", 1)[1]
            assert detail in rejected[0].reason


def test_a_wrong_type_does_not_hide_the_good_robot_beside_it() -> None:
    present = {**_myagv("good"), **_myagv("bad")}
    present["/bad/cmd_vel"] = "geometry_msgs/TwistStamped"
    assert discover_from(present).namespace == "good"


def test_a_mixed_fleet_is_ambiguous_until_narrowed_and_lists_every_candidate() -> None:
    present = {**_myagv("myagv"), **_ainex("ainex"), **_so101(), **_scene()}
    with pytest.raises(DiscoveryError) as exc:
        discover_from(present)
    message = str(exc.value)
    assert "myagv on /myagv/*" in message and "ainex on /ainex/*" in message
    assert discover_from(present, namespace="ainex").robot == "ainex"


def test_naming_a_namespace_that_is_not_there_lists_what_is() -> None:
    with pytest.raises(DiscoveryError) as exc:
        discover_from(_myagv("myagv"), namespace="")
    assert "myagv on /myagv/*" in str(exc.value)


# -------------------------------------------------------- end to end, through the supervisor


def _supervise(bridge, **kwargs):
    """Start the real supervisor process against the fake bridge; return (`ready` or the
    error message it answered with, the link)."""
    import os
    import sys
    from pathlib import Path

    import robot_console
    from robot_console.supervisor import SupervisedLink, SupervisorError

    src = str(Path(robot_console.__file__).resolve().parents[1])
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = src
    link = SupervisedLink(f"ws://127.0.0.1:{bridge.port}", python=sys.executable, **kwargs)
    try:
        return link.start(timeout=20), link
    except SupervisorError as exc:
        return str(exc), link
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH")
        else:
            os.environ["PYTHONPATH"] = old


@pytest.mark.parametrize(
    "topics,kwargs,expected",
    [
        (lambda: _myagv(""), {}, ("myagv", "")),
        (lambda: _ainex("ainex"), {}, ("ainex", "ainex")),
        (lambda: {**_myagv("myagv"), **_ainex("ainex"), **_scene()}, {"robot": "ainex"},
         ("ainex", "ainex")),
        (lambda: {**_myagv("myagv"), **_composite("myagv_mycobot280"),
                  **_x3("rosmaster_x3_plus")}, {"robot": "rosmaster_x3_plus"},
         ("rosmaster_x3_plus", "rosmaster_x3_plus")),
        (lambda: {**_myagv("myagv"), **_composite("myagv_mycobot280"),
                  **_x3("rosmaster_x3_plus")}, {"namespace": "myagv_mycobot280"},
         ("myagv_mycobot280", "myagv_mycobot280")),
    ],
    ids=["lone-bare", "namespaced", "mixed-fleet-narrowed", "three-bases-by-robot",
         "three-bases-by-namespace"],
)
def test_the_supervisor_discovers_the_robot(bridge, topics, kwargs, expected) -> None:
    bridge.topics = topics()
    ready, link = _supervise(bridge, **kwargs)
    try:
        assert isinstance(ready, dict), ready
        assert (ready["robot"], ready["namespace"]) == expected
    finally:
        link.close()


@pytest.mark.parametrize(
    "topics,fragments",
    [
        (lambda: {**_myagv("robot_1"), **_myagv("robot_2")},
         ["robot_1", "robot_2", "--namespace"]),
        (lambda: {**_myagv("myagv"), **_ainex("ainex")},
         ["myagv on /myagv/*", "ainex on /ainex/*"]),
        (lambda: {**_so101(), **_scene()}, ["no robot this console can drive", "none"]),
        (lambda: {"/myagv/cmd_vel": "std_msgs/String", "/myagv/odom": "nav_msgs/Odometry"},
         ["std_msgs/String"]),
        (lambda: {"/myagv/cmd_vel": "geometry_msgs/Twist"}, ["missing /myagv/odom"]),
        (lambda: {**_myagv("myagv"), **_composite("myagv_mycobot280"),
                  **_x3("rosmaster_x3_plus")},
         ["myagv on /myagv/*", "myagv_mycobot280 on /myagv_mycobot280/*",
          "rosmaster_x3_plus on /rosmaster_x3_plus/*"]),
    ],
    ids=["duplicates", "mixed-fleet", "nothing-drivable", "wrong-type", "missing-topic",
         "every-cmd_vel-base"],
)
def test_the_supervisor_refuses_an_ambiguous_wire_naming_the_candidates(
    bridge, topics, fragments
) -> None:
    bridge.topics = topics()
    error, link = _supervise(bridge)
    assert isinstance(error, str), error
    for fragment in fragments:
        assert fragment in error
    assert bridge.received == [], "a refused wire must see no publication at all"
    assert link.wait(5) is not None


def test_an_unreachable_rosapi_is_an_error_not_a_guess(bridge) -> None:
    """No fallback to "a myAGV on the bare contract": with /rosapi unanswerable there are
    no candidates, and the error says so."""
    bridge.topics = _myagv("")
    bridge.rosapi = False
    error, link = _supervise(bridge)
    assert isinstance(error, str), error
    assert "/rosapi/topics failed" in error and "candidates found: none" in error
    assert bridge.received == []


def test_naming_both_robot_and_namespace_needs_no_rosapi(bridge) -> None:
    bridge.rosapi = False
    ready, link = _supervise(bridge, robot="myagv", namespace="")
    try:
        assert isinstance(ready, dict), ready
        assert ready["cmd_topic"] == "/cmd_vel"
    finally:
        link.close()
