"""How a topic is named when several robots share one ROS graph.

The console and the simulator each own a copy of this rule -- the console must install
with no simulator checkout at all, so it cannot import one. That makes the rule a
contract like any other, and `tests/arm/test_ros_contract.py` is where the two copies are
held to the same answers. This file pins the rule itself.
"""

from __future__ import annotations

import pytest

from robot_console.cli import resolve_topic
from robot_console.topics import TOPIC_CMD_VEL, TOPIC_ODOM, namespaced


def test_a_namespace_prefixes_an_absolute_topic() -> None:
    assert namespaced("/cmd_vel", "myagv") == "/myagv/cmd_vel"
    assert namespaced("/camera/image_raw/compressed", "myagv") == (
        "/myagv/camera/image_raw/compressed"
    )


def test_a_relative_topic_still_comes_back_absolute() -> None:
    """`normalise`'s job survives namespacing: what goes out is always absolute.

    Stock rosbridge resolves a relative name against the node namespace and silently
    misses; the simulator's own bridge normalises, but the console must keep working
    against both.
    """
    assert namespaced("cmd_vel", "myagv") == "/myagv/cmd_vel"
    assert namespaced("cmd_vel", "") == "/cmd_vel"


def test_no_namespace_is_the_identity() -> None:
    """The bare single-robot contract has to stay expressible, or it stops being testable.

    This is what a real myAGV bringup presents, and the claim that this simulator is
    indistinguishable from one rests on being able to ask for exactly it.
    """
    for topic in (TOPIC_CMD_VEL, TOPIC_ODOM, "/scan"):
        assert namespaced(topic, "") == topic


def test_composition_is_idempotent() -> None:
    """A topic already carrying the namespace is not prefixed twice.

    This is what lets `--namespace myagv` and an explicit `--scan-topic /myagv/scan` be
    given together without producing `/myagv/myagv/scan`.
    """
    assert namespaced("/myagv/cmd_vel", "myagv") == "/myagv/cmd_vel"
    assert namespaced(namespaced("/cmd_vel", "myagv"), "myagv") == "/myagv/cmd_vel"


def test_a_similar_prefix_is_not_mistaken_for_the_namespace() -> None:
    """`/myagv2/...` is not inside `/myagv`, and prefix matching must not think it is."""
    assert namespaced("/myagv2/cmd_vel", "myagv") == "/myagv/myagv2/cmd_vel"


@pytest.mark.parametrize("namespace", ["myagv", "/myagv", "myagv/", "/myagv/"])
def test_slashes_around_the_namespace_do_not_matter(namespace: str) -> None:
    assert namespaced("/cmd_vel", namespace) == "/myagv/cmd_vel"


def test_an_explicitly_named_topic_beats_the_namespace() -> None:
    """Naming a topic is more specific than naming a namespace, so it wins.

    `resolve_topic` is the rule the CLIs use, and the reason their topic flags default to
    `None` rather than to the constant: that is the only way to tell "left alone" from
    "set to the default value on purpose".
    """
    assert resolve_topic(None, TOPIC_CMD_VEL, "myagv") == "/myagv/cmd_vel"
    assert resolve_topic("/elsewhere/cmd_vel", TOPIC_CMD_VEL, "myagv") == "/elsewhere/cmd_vel"
    # Relative names given explicitly are still made absolute.
    assert resolve_topic("elsewhere/cmd_vel", TOPIC_CMD_VEL, "myagv") == "/elsewhere/cmd_vel"
    assert resolve_topic(None, TOPIC_CMD_VEL, "") == TOPIC_CMD_VEL
