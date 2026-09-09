"""Publishing one robot's transform tree, in one place for every robot and engine.

A surface builds a `mujoco_bridge.TransformTree` for its robot and hands it here; this
does the wire half -- namespacing every frame, splitting static from dynamic, repeating
the static half often enough that a late client still gets it, and setting the
`robot_description` a client reads the tree against.

It is one module for the same reason `ros_surfaces/` is one directory: three robots and
two engines publishing tf from three copies of this loop would be three chances to
disagree about what a frame is called, and the whole claim of this simulator is that a
client cannot tell which robot -- or which engine -- it is talking to except by asking.

The two halves belong together and are attached together (`attach_tf`) because either
alone is a trap. A tree with no description names frames whose shape nothing knows; a
description with no tree describes a robot whose links are never placed, and RViz shows
it collapsed at the origin with every link on top of every other.
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
    tf_message,
)


class TfStream:
    """Publishes `/tf` every tick and `/tf_static` on a slow clock of its own.

    `extra` on each `publish` is for transforms a surface owns rather than the model:
    the myAGV's `odom -> base_footprint`, which is the odometry node's on real hardware
    and is the one transform in this contract that is a *measurement* rather than a
    reading of the robot's own geometry.
    """

    def __init__(self, bus, tree, *, ros2: bool = False,
                 static_period: float = STATIC_PERIOD_S) -> None:
        self._bus = bus
        self._tree = tree
        self._type = TYPE_TF_MESSAGE_ROS2 if ros2 else TYPE_TF_MESSAGE
        self._ros2 = ros2
        # **A ROS 1 robot has no `/tf_static`.** `tf`'s `static_transform_publisher` --
        # which is what `myagv_active.launch` runs, three times, `pkg="tf"` and not
        # `tf2_ros` -- re-publishes its transform onto `/tf` on a period. `/tf_static` is
        # a tf2 topic, and the myAGV's stack is tf1: nothing on that robot publishes to
        # it, and its URDF has no fixed joint for `robot_state_publisher` to put there
        # either (its one joint, `base_up`, is `continuous`). So the static half goes out
        # on `/tf` here too, on its own slow clock, and `/tf_static` never appears on a
        # ROS 1 robot's topic list. The SO-101 is a real ROS 2 bringup and does use it.
        self._static_topic = TOPIC_TF_STATIC if ros2 else TOPIC_TF
        self._static_period = static_period
        self._next_static = 0.0
        # Namespaced once, here, rather than at each publish: the frames come off the
        # tree bare (see `TransformTree`) and this is the one point they reach the wire.
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
                self._type,
            )

        # A wall clock for the repeat and the simulation's for the stamp. The repeat
        # exists because a client that connected a moment ago has no tree yet (rosbridge
        # does not latch), which is a fact about when the client arrived, not about
        # simulated time -- and a paused simulation would otherwise never repeat it.
        now = time.monotonic()
        if now >= self._next_static and self._static:
            self._next_static = now + self._static_period
            self._bus.publish(
                self._static_topic,
                tf_message(self._static, stamp_s=stamp_s, seq=seq, ros2=self._ros2),
                self._type,
            )


def attach_tf(bus, tree, urdf_text: str, *, ros2: bool = False) -> TfStream:
    """Wire a robot's tree and its description onto the bus, and return the stream."""
    bus.set_param(PARAM_ROBOT_DESCRIPTION, urdf_text)
    return TfStream(bus, tree, ros2=ros2)


def read_description(path) -> str:
    """The URDF a robot is described by, as text.

    A thin helper so every surface names its description the same way, and so the one
    thing worth saying about it is said once: **the meshes it references are not served
    over this bridge, and cannot be.** rosbridge is a JSON websocket; a real client
    resolves `package://` against its own filesystem or an out-of-band web server, and
    that is true of real rosbridge too, so it is not a divergence from hardware. A client
    with no copy of the meshes still gets every frame, every joint limit and the link
    tree -- everything except what the links look like.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()
