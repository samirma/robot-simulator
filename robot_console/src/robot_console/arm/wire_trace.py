"""Record what this client actually puts on, and takes off, the rosbridge socket.

Offline MuJoCo runs never build a ``JointTrajectory``, so a fault between the
policy's action vector and the published arm command is invisible to
``scripts/offline_eval.py`` and shows up only as "the live arm does not move".
This module makes that segment observable: it captures every advertise,
publish and service call verbatim, so the ``joint_names`` and
``points[0].positions`` that reach the controller can be read back and compared
against the joint order the controller is configured with.

Nothing here changes behaviour. ``TracingRosbridgeClient`` only records and
delegates, and it is opt-in — the shipped embodiment uses the plain client.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_console.arm.ros_client import ZERO_HEADER, HeaderStampingClient


@dataclass
class WireRecord:
    """One outgoing operation, tagged with the control step it belongs to."""

    step: int
    op: str
    topic: str
    payload: dict[str, Any]


@dataclass
class WireLog:
    """Everything the client sent, in order, grouped by control step."""

    records: list[WireRecord] = field(default_factory=list)
    step: int = -1

    def mark_step(self, step: int) -> None:
        """Tag subsequent operations with this control-step index."""
        self.step = step

    def add(self, op: str, topic: str, payload: Mapping[str, Any]) -> None:
        self.records.append(WireRecord(self.step, op, topic, dict(payload)))

    def publishes(self, topic: str) -> list[WireRecord]:
        """Every publish to one topic, oldest first."""
        return [r for r in self.records if r.op == "publish" and r.topic == topic]

    def write(self, path: str | Path) -> Path:
        """Dump the whole log as JSON for offline inspection."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                [
                    {"step": r.step, "op": r.op, "topic": r.topic, "payload": r.payload}
                    for r in self.records
                ],
                indent=2,
            )
        )
        return destination


class TracingRosbridgeClient(HeaderStampingClient):
    """A rosbridge client that records every operation it sends.

    Subclasses the header-stamping client the embodiment actually uses, so a
    trace shows the bytes that really go out -- including the ``header`` the arm
    controller needs -- rather than a differently-behaved debug path.
    """

    def __init__(self, url: str, *, log: WireLog | None = None, **kwargs: Any) -> None:
        super().__init__(url, **kwargs)
        self.log = log if log is not None else WireLog()

    def advertise(self, topic: str, *, message_type: str) -> None:
        self.log.add("advertise", topic, {"type": message_type})
        super().advertise(topic, message_type=message_type)

    def publish(self, topic: str, msg: Mapping[str, Any]) -> None:
        # Record after the stamping shim has run, so the log is the wire.
        if topic in self.stamped_topics and "header" not in msg:
            msg = {"header": dict(ZERO_HEADER), **dict(msg)}
        self.log.add("publish", topic, msg)
        super().publish(topic, msg)

    def call_service(self, service: str, args: Mapping[str, Any] | None = None) -> Any:
        self.log.add("call_service", service, dict(args or {}))
        return super().call_service(service, args)


def install_tracer(embodiment: Any, log: WireLog | None = None) -> WireLog:
    """Swap a not-yet-connected embodiment's client for a tracing one.

    Construction of the ROS embodiment is network-free, so the transport can be
    replaced any time before the first ``reset``. Raises if the client has
    already connected, because swapping it then would silently drop the
    subscriptions the old one holds.
    """
    client = embodiment._client
    if client.connected:
        raise RuntimeError("install_tracer must be called before the embodiment connects")
    traced = TracingRosbridgeClient(
        embodiment.url,
        log=log,
        stamped_topics=getattr(client, "stamped_topics", ()),
        clock=embodiment._clock,
        sleep=embodiment._sleep,
    )
    embodiment._client = traced
    return traced.log
