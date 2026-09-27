"""Per-robot ROS surfaces, shared by every MuJoCo engine.

A *surface* is the set of topics one robot's vendor stack presents, plus the loop that
feeds them from a running MuJoCo model. It belongs to the robot rather than to any
engine: the myAGV speaks `cmd_vel`/`odom`, the AiNex speaks `/walking/*` and shares not
one topic with it, and both must look identical to `robot_console` no matter which
engine is hosting them. Keeping the implementation here is what makes that true by
construction instead of by two engines agreeing to stay in step.

The transport (`contracts/`) and the MuJoCo→wire helpers (`mujoco_bridge`) are the
layers below; a surface is the robot-specific wiring between them.

**Several robots share one server, one port and one graph** -- see `RobotFleet` below and
`contracts/namespace.py` for why that, and not a port per robot, is the shape. Each
surface exposes `attach_ros(bus, ...)`, which registers against a `NamespacedBus` and
returns a per-step closure; the thin `serve_ros(port, ...)` wrapper each surface keeps is
the single-robot path, and is what `run.sh view --robot myagv --ros-port 9090` and the
engine-side adapters still call.
"""

from __future__ import annotations

import sys
from typing import Any, Callable


class WorldReset:
    """Callbacks to run when something restores the whole simulated world.

    `tasks/apple_on_plate.py`'s `/reset` snapshots and restores the entire `qpos`, `qvel`
    and `ctrl` -- which, once a second robot is in the scene, includes that robot. So the
    arm's `/so101/reset` teleports the AGV back to its spawn pose, while the AGV's own
    `PlanarSetpoint` still holds the target it was driving toward: the base then lunges
    for a pose it no longer occupies, which reads as a physics glitch and is really a
    latched controller. Every surface that latches state registers here and drops it.

    Shared by every robot on one fleet on purpose: "the world was reset" is a fact about
    the world, not about the robot whose service happened to be called.
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], None]] = []

    def on_reset(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def fire(self) -> None:
        for callback in self._callbacks:
            callback()


class RobotFleet:
    """One `RosBridgeServer`, several attached surfaces, one fan-out controller.

    This is the whole of "run a ROS server with several robots at once": the robots share
    a graph and a port and are told apart by namespace, exactly as a real multi-robot
    bringup does it. Two servers on two ports would be two graphs -- nothing could then
    discover the fleet, and no single client could drive it.

    A fleet is callable with the signature `mujoco_bridge.run_sim_loop` already wants, so
    it drops in where a single robot's `step` used to go and that loop needs no change.
    Its one-thread rule is unaffected: every attached `step` runs on the simulation
    thread, in attach order.
    """

    def __init__(self, port: int, host: str = "0.0.0.0",
                 control_hz: float | None = None) -> None:
        from contracts.rosbridge_server import RosBridgeServer

        self.port = port
        self.host = host
        self.server = RosBridgeServer(host=host, port=port)
        self.world_reset = WorldReset()
        # A world reset aborts every outstanding action goal (spec §3, `/reset`): the
        # world a goal was pursuing no longer exists.
        self.world_reset.on_reset(self.server.abort_goals)
        # ...and every member runs on the next tick, so the observations a `/reset`
        # caller waits for are ones taken after the reset.
        self.world_reset.on_reset(self._rearm)
        # The rate a member without a rate of its own is stepped at (`--control-hz`).
        # `None` steps it on every call, which is the single-robot path's behaviour.
        self.control_hz = control_hz
        self._members: list[tuple[str, Any, Callable[[Any], None]]] = []
        # Per member: the simulated time its next step is due, or None for "now".
        self._due: list[float | None] = []

    @property
    def rate_hz(self) -> float:
        """How often the loop must call the fleet: its fastest member's rate.

        A member's step may carry a `rate_hz` of its own -- the SO-101's controller
        manager runs at its `update_rate`, the rig at its cameras' rate -- and the fleet
        then steps it on that clock whatever the loop's rate is.
        """
        rates = [float(r) for r in (getattr(step, "rate_hz", None)
                                    for _, _, step in self._members) if r]
        if self.control_hz:
            rates.append(float(self.control_hz))
        return max(rates) if rates else 0.0

    def _rearm(self) -> None:
        self._due = [None] * len(self._members)

    def bus(self, namespace: str):
        from contracts.rosbridge_server import NamespacedBus

        return NamespacedBus(self.server, namespace)

    def attach(self, namespace: str, attach_fn: Callable[..., Callable[[Any], None]],
               /, **kwargs: Any) -> None:
        """Register one robot's surface under `namespace`.

        `attach_fn(bus, **kwargs)` wires handlers and returns the per-step callback -- the
        same closure shape a standalone `serve_ros` returns. It must not start or stop the
        server, and `step(None)` must close only its own streams: the fleet owns the
        socket.
        """
        bus = self.bus(namespace)
        step = attach_fn(bus, world_reset=self.world_reset, **kwargs)
        self._members.append((str(bus.ns), bus, step))
        self._due.append(None)

    def start(self) -> None:
        # rosapi is per-server, not per-robot: one topic list answers for the whole fleet,
        # and that list is how a client finds out there is more than one robot here at
        # all. It used to be called from inside the SO-101's surface, which meant a
        # myAGV-only run had no topic discovery whatsoever.
        self.server.serve_rosapi()
        self.server.start()
        # "surface(s)", not "robot(s)": the worktop's camera rig is a member too, under
        # its own namespace, and it is not a robot.
        print(f"ROS fleet on ws://{self.host}:{self.port} — "
              f"{len(self._members)} surface(s)", file=sys.stderr)
        for name, bus, _ in self._members:
            print(f"  {name}: sub {', '.join(bus.subscribed) or '-'}", file=sys.stderr)

    def __call__(self, data) -> None:
        if data is None:
            # Close every surface's streams first, then stop the shared server exactly
            # once. A standalone `serve_ros` closure stops the server itself; letting each
            # member do that would shut the whole fleet down on the first robot.
            for _, _, step in self._members:
                step(None)
            self.server.stop()
            return
        # The simulated clock `/rosapi/get_time` answers from.
        now = float(getattr(data, "time", 0.0))
        self.server.set_time(now)
        for index, (_, _, step) in enumerate(self._members):
            rate = getattr(step, "rate_hz", None) or self.control_hz
            if not rate:
                step(data)
                continue
            period = 1.0 / float(rate)
            due = self._due[index]
            # A quarter period of slack: the loop's calls jitter around the period, and
            # a strict comparison would skip every other one of them.
            if due is not None and now < due - 0.25 * period:
                continue
            step(data)
            if index < len(self._due):
                nxt = (due + period) if due is not None else now + period
                self._due[index] = nxt if nxt > now - period else now + period
