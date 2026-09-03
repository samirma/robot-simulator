"""A rosbridge client that stamps arm trajectories, which this simulator requires.

``inspect_robots_ros._msgs.build_joint_trajectory`` builds a ``JointTrajectory``
with ``joint_names`` and ``points`` only -- no ``header``. Against this
simulator's ``JointTrajectoryController`` that message is accepted by rosbridge
and then does nothing at all: the controller holds its pose, reports zero error,
and never moves. Adding an empty ``header`` (``stamp`` 0, which is ROS's "start
now") makes the identical goal execute exactly.

Measured on the running container, one publish per row, each asking for
+0.25 rad on ``shoulder_pan_joint`` from wherever the arm happened to be:

    variant                          pan before   pan after    travel  verdict
    tfs=0.1s  +header  no vel          +0.61531    +0.86531  +0.25000  TRACKS
    tfs=0.1s  no header +vel           +0.86531    +0.86531  +0.00000  *** NO ***
    tfs=0.1s  +header  no vel          +0.86531    +1.11531  +0.25000  TRACKS
    tfs=0.1s  no header +vel           +1.11531    +1.11531  +0.00000  *** NO ***

``time_from_start`` (0.1 s to 3.0 s) and ``velocities`` make no difference; the
``header`` decides it, and the result reproduces both ways.

This failure is silent in the worst way. The gripper is driven by a
``ForwardCommandController``, whose message needs no header, so the gripper keeps
working and the episode looks alive -- waypoints advance, the jaw opens and
closes -- while the arm sags under gravity and never leaves its start pose. It
also cannot be seen offline, because the MuJoCo path never builds a
``JointTrajectory`` at all.

The shim lives here rather than in the vendored ``inspect-robots`` tree so that
upstream stays pristine. It is a strict addition: a message that already carries
a ``header`` is published unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inspect_robots_ros._client import RosbridgeClient

#: ``stamp`` 0 means "execute starting now" to a ``JointTrajectoryController``.
ZERO_HEADER: dict[str, Any] = {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}


class HeaderStampingClient(RosbridgeClient):
    """Add the ``header`` the arm controller needs, for one configured topic.

    Scoped to a single topic on purpose: it is the arm trajectory that needs
    this, and blanket-stamping every outgoing message would quietly change the
    gripper and any future publisher too.
    """

    def __init__(self, url: str, *, stamped_topics: tuple[str, ...] = (), **kwargs: Any) -> None:
        super().__init__(url, **kwargs)
        self.stamped_topics = tuple(stamped_topics)
        #: How many messages this client actually had to fix up.
        self.headers_added = 0

    def publish(self, topic: str, msg: Mapping[str, Any]) -> None:
        """Publish, inserting an empty header first if this topic needs one."""
        if topic in self.stamped_topics and "header" not in msg:
            self.headers_added += 1
            msg = {"header": dict(ZERO_HEADER), **dict(msg)}
        super().publish(topic, msg)
