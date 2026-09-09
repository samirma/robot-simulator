"""What the fleet check expects of a wire, pinned.

`fleet.py` is the answer to "are the robots I asked for actually on this rosbridge", and
it answers it from the console's *own* contract constants rather than a list typed into a
shell script. That is the right design and it had no test at all, so nothing held the
expectation to the wire: a topic quietly leaving `arm_topics()` would show up only as a
live run that passed when it should have failed.

Offline by construction -- `missing_for` is pure, and the topic list it is given here is
a dict a live `/rosapi/topics` would have returned.
"""

from __future__ import annotations

from robot_console.fleet import BASE_TOPICS, arm_topics, missing_for, scene_topics
from robot_console.topics import TOPIC_TF, TOPIC_TF_STATIC, namespaced


def _wire(*namespaced_topics: str) -> dict[str, str]:
    """A topic list shaped like `list_topics`', types elided -- nothing reads them."""
    return {topic: "" for topic in namespaced_topics}


def _arm_wire(namespace: str = "so101") -> dict[str, str]:
    return _wire(
        *(namespaced(t, namespace) for t in arm_topics()),
        *(namespaced(t, "scene") for t in scene_topics()),
    )


def test_a_complete_arm_is_missing_nothing() -> None:
    present = _arm_wire()
    assert missing_for(present, "so101", arm_topics()) == []
    assert missing_for(present, "scene", scene_topics()) == []


def test_the_scene_rig_is_not_checked_under_the_arms_namespace() -> None:
    """The rig watches the worktop, so it is not in the robot's namespace and must not be
    looked for there. Checked from both sides: the rig's real names satisfy the scene
    expectation, and the arm expectation does not contain them at all."""
    assert not any("overhead" in t or "side" in t for t in arm_topics())
    present = _arm_wire()
    assert "/scene/overhead/color/compressed" in present
    assert "/so101/overhead/color/compressed" not in present


def test_a_missing_topic_is_reported_by_its_wire_name() -> None:
    """The report has to name the topic as it would appear on the wire, because that is
    what the reader will grep the simulator's output for."""
    present = _arm_wire()
    del present["/so101/joint_states"]
    assert missing_for(present, "so101", arm_topics()) == ["/so101/joint_states"]


def test_an_arm_under_another_namespace_is_a_different_arm() -> None:
    present = _arm_wire("so101")
    missing = missing_for(present, "robot_2", arm_topics())
    assert missing == [namespaced(t, "robot_2") for t in arm_topics()]


def test_the_bare_contract_is_expressible() -> None:
    """`--arm ''` is the single-robot vendor wire, which the simulator can still serve
    with `--ros-namespace ''`. An empty namespace has to change nothing."""
    present = _wire(
        *arm_topics(), *(namespaced(t, "scene") for t in scene_topics())
    )
    assert missing_for(present, "", arm_topics()) == []


def test_a_base_is_checked_against_the_myagv_contract() -> None:
    present = _wire(*(namespaced(t, "myagv") for t in BASE_TOPICS))
    assert missing_for(present, "myagv", BASE_TOPICS) == []
    # A base is not an arm: every topic that makes an arm an arm is missing from it. `/tf`
    # is the one exception, because both robots really do publish a tree -- but not
    # `/tf_static`, which the arm requires and the base does not have: the myAGV's static
    # transforms come from tf1 publishers that write to `/tf`, and its URDF has no fixed
    # joint. So a base's wire satisfies exactly one of the arm's two tf topics.
    assert TOPIC_TF in BASE_TOPICS and TOPIC_TF_STATIC not in BASE_TOPICS
    assert missing_for(present, "myagv", arm_topics()) == [
        namespaced(t, "myagv") for t in arm_topics() if t != TOPIC_TF
    ]
