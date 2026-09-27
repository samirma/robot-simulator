"""Publishing one robot's transform tree, in one place for every robot and engine.

Two publishers, one per way a real bringup produces its tree:

* `RobotStatePublisher` -- ROS 2 `robot_state_publisher` semantics, for a member whose
  interface has it (the SO-101): the published description (latched `/robot_description`
  and the node's `robot_description` parameter), its fixed joints once on a latched
  `/tf_static`, and its moving joints on `/tf` computed from the description and the
  joint positions the member publishes, throttled to `publish_frequency`. No MuJoCo: the
  tree is the description's, exactly as the real node computes it.
* `TfStream` -- a `mujoco_bridge.TransformTree` read off the compiled model, for the ROS 1
  members (myAGV, AiNex). Their static transforms go out on `/tf` every
  `STATIC_PERIOD_S`, as tf1's `static_transform_publisher` re-sends them; a ROS 2 tree
  given to it sends `/tf_static` once, latched.

Frames are namespaced once, here, with `bus.frame()`; the trees hand them over bare.
"""

from __future__ import annotations

import time

from contracts.tf import (
    PARAM_ROBOT_DESCRIPTION,
    STATIC_PERIOD_S,
    TOPIC_TF,
    TOPIC_TF_STATIC,
    TYPE_TF_MESSAGE,
    TYPE_TF_MESSAGE_ROS2,
    UrdfTree,
    tf_message,
)


class TfStream:
    """Publishes `/tf` every tick and the static half as the robot's dialect does.

    `extra` on each `publish` is for transforms a surface owns rather than the model:
    the myAGV's `odom -> base_footprint`, which is a measurement rather than a reading of
    the robot's own geometry.
    """

    def __init__(self, bus, tree, *, ros2: bool = False,
                 static_period: float = STATIC_PERIOD_S, node: str | None = None) -> None:
        self._bus = bus
        self._tree = tree
        self._type = TYPE_TF_MESSAGE_ROS2 if ros2 else TYPE_TF_MESSAGE
        self._ros2 = ros2
        self._node = node
        # A ROS 1 robot has no `/tf_static`: tf1's `static_transform_publisher` re-sends
        # onto `/tf` on a period. A ROS 2 one sends `/tf_static` once, latched.
        self._static_topic = TOPIC_TF_STATIC if ros2 else TOPIC_TF
        self._static_period = static_period
        self._next_static = 0.0
        self._static_sent = False
        self._static = [
            (bus.frame(parent), bus.frame(child), pos, quat)
            for parent, child, pos, quat in tree.static()
        ]

    @property
    def frames(self) -> tuple[str, ...]:
        """Every frame this stream publishes, namespaced. For the startup report."""
        return tuple(sorted(self._bus.frame(f) for f in self._tree.frames))

    def publish(self, data, seq: int, stamp_s: float, extra=()) -> None:
        entries = [
            (self._bus.frame(parent), self._bus.frame(child), pos, quat)
            for parent, child, pos, quat in self._tree.dynamic(data)
        ]
        entries.extend(extra)
        if entries:
            self._bus.publish(
                TOPIC_TF,
                tf_message(entries, stamp_s=stamp_s, seq=seq, ros2=self._ros2),
                self._type, node=self._node,
            )
        if not self._static:
            return
        if self._ros2:
            if not self._static_sent:
                self._static_sent = True
                self._bus.publish(
                    TOPIC_TF_STATIC,
                    tf_message(self._static, stamp_s=stamp_s, seq=seq, ros2=True),
                    self._type, latched=True, node=self._node,
                )
            return
        # A wall clock for the repeat: a paused simulation must still re-send it.
        now = time.monotonic()
        if now >= self._next_static:
            self._next_static = now + self._static_period
            self._bus.publish(
                self._static_topic,
                tf_message(self._static, stamp_s=stamp_s, seq=seq, ros2=False),
                self._type, node=self._node,
            )


def attach_tf(bus, tree, urdf_text: str, *, ros2: bool = False) -> TfStream:
    """Wire a robot's model-read tree and its description onto the bus (ROS 1 members)."""
    bus.set_param(PARAM_ROBOT_DESCRIPTION, urdf_text)
    return TfStream(bus, tree, ros2=ros2)


class RobotStatePublisher:
    """ROS 2 `robot_state_publisher` over a description, as its node behaves.

    `topic_description` is the latched `std_msgs/msg/String` description topic; the same
    text is the node's `robot_description` parameter. `/tf_static` carries the fixed
    joints once, latched, stamped with the first joint state's time. `/tf` carries the
    moving joints of each joint state, but no more often than `publish_frequency`.
    """

    def __init__(self, bus, urdf_text: str, *, node: str, topic_description: str,
                 type_description: str, param_description: str | None,
                 publish_frequency: float) -> None:
        self._bus = bus
        self._node = node
        self.tree = UrdfTree(urdf_text)
        self._period = 1.0 / publish_frequency if publish_frequency > 0 else 0.0
        self._last_tf: float | None = None
        self._static_sent = False
        bus.advertise(TOPIC_TF, TYPE_TF_MESSAGE_ROS2, node=node)
        bus.advertise(TOPIC_TF_STATIC, TYPE_TF_MESSAGE_ROS2, node=node)
        bus.publish(topic_description, {"data": urdf_text}, type_description,
                    latched=True, node=node)
        if param_description is not None:
            bus.set_param(param_description, urdf_text)

    def _frames(self, entries):
        return [(self._bus.frame(p), self._bus.frame(c), pos, quat)
                for p, c, pos, quat in entries]

    def publish(self, positions: dict, stamp_s: float, *, force: bool = False) -> None:
        """One joint state's worth: `/tf_static` once, `/tf` when the throttle allows."""
        if not self._static_sent:
            self._static_sent = True
            fixed = self._frames(self.tree.fixed())
            if fixed:
                self._bus.publish(TOPIC_TF_STATIC,
                                  tf_message(fixed, stamp_s=stamp_s, ros2=True),
                                  TYPE_TF_MESSAGE_ROS2, latched=True, node=self._node)
        # Throttled on the joint states' own (simulated) clock, with a quarter-period of
        # slack so a 50 Hz state stream divides into an even 20 Hz rather than beating.
        if (not force and self._last_tf is not None
                and stamp_s - self._last_tf < self._period * 0.75):
            return
        if self._last_tf is not None and not force:
            self._last_tf = max(self._last_tf + self._period, stamp_s - self._period)
        else:
            self._last_tf = stamp_s
        moving = self._frames(self.tree.moving(positions))
        if moving:
            self._bus.publish(TOPIC_TF, tf_message(moving, stamp_s=stamp_s, ros2=True),
                              TYPE_TF_MESSAGE_ROS2, node=self._node)


def read_description(path) -> str:
    """The URDF a robot is described by, as text.

    **The meshes it references are not served over this bridge, and cannot be.** rosbridge
    is a JSON websocket; a real client resolves `package://` against its own filesystem,
    and that is true of real rosbridge too.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()
