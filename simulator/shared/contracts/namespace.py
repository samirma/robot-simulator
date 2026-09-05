"""How a topic, a service and a frame are named when several robots share one graph.

Stdlib only, and deliberately so: `robot_console/tests/arm/test_ros_contract.py` loads
files out of this tree by path to hold the two projects' halves of the contract together,
and it can only import what the console itself can. Keeping the composition rule here --
rather than inside `rosbridge_server`, which needs `websockets` -- is what lets that test
pin *the rule*, not just the bare constant names.

The rule is ROS's own. Several robots on one graph is not several servers; it is one
graph, one bridge and a namespace per robot -- `ROS_NAMESPACE=robot1` / `<group
ns="robot1">` in ROS 1, `-r __ns:=/robot1` in ROS 2 -- with the matching prefix on every
`frame_id`. rosbridge sits above that: one websocket exposes the whole graph, so a fleet
is one port whose topic list reads `/so101/joint_states` beside `/myagv/cmd_vel` and a
client picks its robot by prefix. Two bridges on two ports would be two graphs: the
robots could not see each other and nothing could discover the fleet.

Three properties are load bearing, and each is a test:

* **Topics get a leading slash, frames do not.** A topic name is an absolute graph path;
  a `frame_id` is a tf node joined by `tf_prefix`, and no real stack emits `/so101/odom`
  as a frame. Getting this backwards produces a tf tree nothing will connect to.
* **An empty namespace is the identity.** That is what keeps the bare single-robot
  contract -- the thing this simulator claims to be indistinguishable from real hardware
  -- expressible, and therefore testable, rather than reduced to a comment.
* **Composition is idempotent.** `topic("/so101/joint_states")` under `so101` returns it
  unchanged, so a client that names a topic explicitly is never double-prefixed by a
  namespace it also passed.
"""

from __future__ import annotations

from dataclasses import dataclass


def normalise(topic: str) -> str:
    """ROS resolves a relative name against the node namespace; we only have the root.

    myagv_ros advertises `cmd_vel` while clients usually ask for `/cmd_vel`; treating
    them as the same name avoids a silent no-op subscription.
    """
    return topic if topic.startswith("/") else "/" + topic


def ns_topic(namespace: str, topic: str) -> str:
    """`("myagv", "/cmd_vel")` -> `/myagv/cmd_vel`. Empty namespace changes nothing."""
    topic = normalise(topic)
    namespace = namespace.strip("/")
    if not namespace:
        return topic
    if topic == f"/{namespace}" or topic.startswith(f"/{namespace}/"):
        return topic  # already namespaced; see the idempotency note above
    return f"/{namespace}{topic}"


def ns_frame(namespace: str, frame_id: str) -> str:
    """`("myagv", "base_footprint")` -> `myagv/base_footprint`, with no leading slash.

    An empty `frame_id` stays empty. `/joint_states` sends one -- a real
    `joint_state_broadcaster` publishes no frame there -- and inventing `so101/` would be
    a difference from hardware that this simulator exists not to have.
    """
    namespace = namespace.strip("/")
    if not namespace or not frame_id:
        return frame_id
    frame_id = frame_id.lstrip("/")
    if frame_id == namespace or frame_id.startswith(f"{namespace}/"):
        return frame_id
    return f"{namespace}/{frame_id}"


@dataclass(frozen=True)
class RobotNamespace:
    """One robot's slice of a shared ROS graph.

    Surfaces keep their topic constants bare and wrap them through here. Those constants
    are the record of what each vendor's stack actually presents; the namespace is applied
    at the one point where a name reaches the wire.
    """

    name: str = ""

    def topic(self, topic: str) -> str:
        return ns_topic(self.name, topic)

    def service(self, name: str) -> str:
        return ns_topic(self.name, name)

    def frame(self, frame_id: str) -> str:
        return ns_frame(self.name, frame_id)

    def __bool__(self) -> bool:
        return bool(self.name)

    def __str__(self) -> str:
        return self.name or "<bare>"
