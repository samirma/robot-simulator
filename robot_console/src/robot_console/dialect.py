"""ROS 1 / ROS 2 conventions: type spellings, action wire layouts and ROS infrastructure.

Everything dialect-specific that is *not* robot-specific lives here, so profiles, discovery,
fleet, teleop and the page (via ``profiles.json``) all apply the same rules.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

ROS1 = "ros1"
ROS2 = "ros2"
DIALECTS = (ROS1, ROS2)


def normalize_type(t: str) -> str:
    """``pkg/msg/Type`` / ``pkg/srv/Type`` / ``pkg/action/Type`` / ``pkg/Type`` -> ``pkg/Type``."""
    t = (t or "").strip()
    parts = t.split("/")
    if len(parts) == 3 and parts[1] in ("msg", "srv", "action"):
        return f"{parts[0]}/{parts[2]}"
    return t


def type_dialect(t: str) -> str | None:
    """The dialect a type spelling implies, or None when it cannot tell."""
    parts = (t or "").split("/")
    if len(parts) == 3 and parts[1] in ("msg", "srv", "action"):
        return ROS2
    if len(parts) == 2 and all(parts):
        return ROS1
    return None


def types_equal(a: str, b: str) -> bool:
    return normalize_type(a) == normalize_type(b)


# ------------------------------------------------------------------ action layouts

def ros1_action_topics(name: str, action_type: str) -> Dict[str, str]:
    """actionlib's five topics for a ROS 1 action server ``name`` of ``pkg/XAction``."""
    pkg, _, base = normalize_type(action_type).partition("/")
    if base.endswith("Action"):
        base = base[: -len("Action")]
    return {
        f"{name}/goal": f"{pkg}/{base}ActionGoal",
        f"{name}/cancel": "actionlib_msgs/GoalID",
        f"{name}/status": "actionlib_msgs/GoalStatusArray",
        f"{name}/feedback": f"{pkg}/{base}ActionFeedback",
        f"{name}/result": f"{pkg}/{base}ActionResult",
    }


def ros2_action_endpoints(name: str, action_type: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(topics, services) behind a ROS 2 action ``name`` of ``pkg/action/X``."""
    pkg, _, base = normalize_type(action_type).partition("/")
    topics = {
        f"{name}/_action/feedback": f"{pkg}/action/{base}_FeedbackMessage",
        f"{name}/_action/status": "action_msgs/msg/GoalStatusArray",
    }
    services = {
        f"{name}/_action/send_goal": f"{pkg}/action/{base}_SendGoal",
        f"{name}/_action/cancel_goal": "action_msgs/srv/CancelGoal",
        f"{name}/_action/get_result": f"{pkg}/action/{base}_GetResult",
    }
    return topics, services


_ROS1_GOAL = re.compile(r"^(?P<pkg>[^/]+)/(?P<base>.+)ActionGoal$")
_ROS2_SEND = re.compile(r"^(?P<pkg>[^/]+)/action/(?P<base>.+)_SendGoal$")


def ros1_action_from_goal_topic(topic: str, t: str) -> Tuple[str, str] | None:
    """``/x/goal`` of ``pkg/XActionGoal`` -> (``/x``, ``pkg/XAction``)."""
    m = _ROS1_GOAL.match(normalize_type(t))
    if topic.endswith("/goal") and m:
        return topic[: -len("/goal")], f"{m['pkg']}/{m['base']}Action"
    return None


def ros2_action_from_send_goal(service: str, t: str) -> Tuple[str, str] | None:
    """``/x/_action/send_goal`` of ``pkg/action/X_SendGoal`` -> (``/x``, ``pkg/action/X``)."""
    suffix = "/_action/send_goal"
    m = _ROS2_SEND.match((t or "").strip())
    if service.endswith(suffix) and m:
        return service[: -len(suffix)], f"{m['pkg']}/action/{m['base']}"
    return None


# ------------------------------------------------------------------ ROS infrastructure
# Workspace spec §1.3: rosbridge_websocket and rosapi and their stock endpoints, /rosout,
# /rosout_agg, /parameter_events and each node's client-library services (logger, parameter,
# type-description). Never part of a robot's vendor interface; may appear on any wire. (The
# console checks no parameters, so the parameter rules of §1.3 have no counterpart here.)

INFRA_TOPICS = {
    "/rosout", "/rosout_agg", "/parameter_events",
    "/client_count", "/connected_clients",
}
INFRA_PREFIXES = ("/rosapi/", "/rosbridge_websocket/", "/rosapi_params/", "/rosapi_node/")
#: The rosbridge and rosapi nodes themselves (a rosbridge client's publications appear as theirs).
INFRA_NODES = ("/rosbridge_websocket", "/rosapi", "/rosapi_params")

# Per-node client-library services.
_NODE_SERVICE_SUFFIXES = (
    # ROS 1 roscpp/rospy
    "/get_loggers", "/set_logger_level",
    # ROS 2 rcl parameter services
    "/describe_parameters", "/get_parameter_types", "/get_parameters",
    "/list_parameters", "/set_parameters", "/set_parameters_atomically",
    # ROS 2 type description (Iron+) and logger services (Jazzy+, opt-in)
    "/get_type_description", "/get_logger_levels", "/set_logger_levels",
)


def is_infrastructure(name: str, kind: str = "topic") -> bool:
    """Whether ``name`` (a topic, service or action) is ROS infrastructure."""
    if name in INFRA_TOPICS or name.startswith(INFRA_PREFIXES):
        return True
    if name in INFRA_NODES:
        return True
    if kind == "service" and name.endswith(_NODE_SERVICE_SUFFIXES):
        return True
    return False

