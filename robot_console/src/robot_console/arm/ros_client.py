"""The rosbridge client the ``so101_ros`` embodiment uses: actions, and short histories.

``inspect_robots_ros``'s client speaks topics and services and nothing else, and keeps
only the latest message per topic. The SO-101 needs two things beyond that:

* **A real ROS 2 action client** for ``/gripper_controller/gripper_cmd``
  (``control_msgs/action/ParallelGripperCommand``). rosbridge 2.0 carries actions as
  ``send_action_goal`` / ``cancel_action_goal``, answered by ``action_feedback`` and
  ``action_result``; this client sends the first two and tracks every goal to its
  result, so it can cancel exactly the goals still running when it closes. (roslibpy has
  an ``ActionClient`` too, but its Twisted reactor is process-global and single-shot; the
  embodiment's transport is this websocket client, and one robot gets one transport.)
* **A short history** of chosen topics, because the camera-verdict scorer pairs each
  overhead frame with the side frame and the joint state *of the same instant*, by stamp,
  and the latest-value slot keeps only whichever arrived last.

A ``status`` error answering one of this client's own cancels (the goal finished while
the cancel was in flight) is expected and is not latched as a transport failure; every
other status error still is.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from inspect_robots_ros._client import RosbridgeClient, TopicSample
from inspect_robots_ros._protocol import (
    PublishedMessage,
    RosbridgeError,
    ServiceResponse,
    StatusMessage,
    decode_message,
    parse_incoming,
)

#: ``action_msgs/msg/GoalStatus`` terminal codes.
STATUS_SUCCEEDED, STATUS_CANCELED, STATUS_ABORTED = 4, 5, 6


@dataclass
class Goal:
    """One action goal this client sent, and what became of it."""

    id: str
    action: str
    args: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    status: int | None = None
    result: Any = None
    cancel_sent: bool = False


class _HistoryTopics(dict):
    """The client's latest-value table, also appending chosen topics to a history."""

    def __init__(self, histories: Mapping[str, deque]) -> None:
        super().__init__()
        self.histories = histories

    def __setitem__(self, topic: str, sample: TopicSample) -> None:
        super().__setitem__(topic, sample)
        history = self.histories.get(topic)
        if history is not None:
            history.append(sample)


class ActionRosbridgeClient(RosbridgeClient):
    """``RosbridgeClient`` plus ROS 2 action goals and per-topic histories."""

    def __init__(self, url: str, *, history_topics: Iterable[str] = (),
                 history_length: int = 64, **kwargs: Any) -> None:
        super().__init__(url, **kwargs)
        self._histories: dict[str, deque] = {
            topic: deque(maxlen=history_length) for topic in history_topics
        }
        self._topics = _HistoryTopics(self._histories)
        self._goals: dict[str, Goal] = {}
        self._goal_ids = itertools.count(1)

    # -- histories ------------------------------------------------------------------
    def history(self, topic: str) -> list[TopicSample]:
        with self._lock:
            return list(self._histories.get(topic, ()))

    def clear_histories(self) -> None:
        with self._lock:
            for history in self._histories.values():
                history.clear()

    # -- actions --------------------------------------------------------------------
    def send_goal(self, action: str, action_type: str, args: Mapping[str, Any]) -> Goal:
        goal = Goal(id=f"send_action_goal:{action}:{next(self._goal_ids)}", action=action,
                    args=dict(args))
        with self._lock:
            self._goals[goal.id] = goal
        self._send({"op": "send_action_goal", "id": goal.id, "action": action,
                    "action_type": action_type, "args": dict(args), "feedback": False})
        return goal

    def cancel_goal(self, goal: Goal) -> bool:
        """Ask the server to cancel ``goal`` if it is still running; True if asked."""
        with self._lock:
            if goal.done.is_set() or goal.cancel_sent:
                return False
            goal.cancel_sent = True
        self._send({"op": "cancel_action_goal", "id": goal.id, "action": goal.action})
        return True

    def pending_goals(self) -> list[Goal]:
        with self._lock:
            return [g for g in self._goals.values() if not g.done.is_set()]

    def cancel_all(self, *, wait_s: float = 1.0) -> list[Goal]:
        """Cancel every unfinished goal and wait briefly for their results."""
        pending = self.pending_goals()
        for goal in pending:
            try:
                self.cancel_goal(goal)
            except Exception:  # noqa: BLE001 - a dead socket cancels nothing more
                break
        deadline = self._clock() + wait_s
        for goal in pending:
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            goal.done.wait(remaining)
        return pending

    def close(self) -> None:
        """Cancel unfinished goals, then close: a goal left running would run on."""
        if self.connected and self.latched_error is None:
            self.cancel_all()
        super().close()

    # -- receive --------------------------------------------------------------------
    def _receive_loop(self) -> None:  # noqa: C901 - one dispatch, kept flat
        with self._lock:
            ws = self._ws
        assert ws is not None
        try:
            while True:
                raw = ws.recv()
                if not isinstance(raw, (str, bytes)):
                    raise RosbridgeError("invalid_frame",
                                         f"rosbridge sent unsupported frame {type(raw).__name__}")
                message = decode_message(raw)
                op = message.get("op")
                if op in ("action_result", "action_feedback"):
                    self._on_action(message)
                    continue
                incoming = parse_incoming(message)
                if isinstance(incoming, PublishedMessage):
                    with self._lock:
                        previous = self._topics.get(incoming.topic)
                        seq = previous.seq + 1 if previous is not None else 1
                        self._topics[incoming.topic] = TopicSample(
                            msg=incoming.msg, stamp=self._clock(), seq=seq)
                elif isinstance(incoming, ServiceResponse):
                    with self._lock:
                        pending = self._pending_services.get(incoming.request_id)
                        if pending is not None:
                            pending.response = incoming
                elif isinstance(incoming, StatusMessage):
                    if self._is_own_cancel(incoming):
                        continue
                    error = incoming.as_error()
                    if error is not None:
                        self._latch(error)
                        return
        except Exception as exc:  # noqa: BLE001 - mirrors the base loop
            with self._lock:
                closing = self._closing
            if not closing:
                self._latch(exc if isinstance(exc, RosbridgeError) else ConnectionError(
                    f"connection to rosbridge at {self.url} lost while receiving: {exc}"))

    def _on_action(self, message: Mapping[str, Any]) -> None:
        if message.get("op") != "action_result":
            return
        with self._lock:
            goal = self._goals.get(str(message.get("id")))
        if goal is None:
            return
        try:
            goal.status = int(message.get("status"))
        except (TypeError, ValueError):
            goal.status = None
        goal.result = message.get("values")
        goal.done.set()

    def _is_own_cancel(self, status: StatusMessage) -> bool:
        with self._lock:
            goal = self._goals.get(status.request_id or "")
        return goal is not None and goal.cancel_sent
