"""Reset the simulator's world and verify, from the wire, that it took.

    python -m robot_console.arm.preflight [--url ws://127.0.0.1:9090]

Ported from the reference rig's ``scripts/scene_reset.py`` as-is: the guarantee it
provides is about the *wire*, not about any one simulator, so it did not need to change
when the simulator did. What follows is its original rationale, still true here.

WHY THIS EXISTS. World state persists between runs: a finished episode leaves
the apple wherever it came to rest, and the next run inherits it. That is not a
cosmetic problem. ``waypoints.PickPlaceConfig.apple_xyz`` is a hardcoded
constant, so the scripted policy against a drifted apple closes its jaws on
empty air at the spawn point -- the run completes, scores 0, and looks like a
policy failure rather than a world that was never reset. A recorded MolmoAct2
failure once left the apple at 0.52 m from the base, beyond the arm's 0.48 m
maximum, where no policy could recover it.

An episode does call ``/reset`` at its start, so why check? Because that
acknowledgement means less than it appears to. ``task_manager.reset_callback``
forwards to ``reset_world`` with ``call_async`` and returns ``success=True``
**without awaiting the result**, so the Trigger response says "requested", not
"done". Nothing downstream of it is ordered against step 0. This waits for the
observed consequence instead: it polls the apple's measured pose until the world
is actually back at spawn and still.

It is contract-legal. ``/reset`` restores the state captured at startup through
``mujoco_ros2_control``'s own ``reset_world``; no ``set_free_joint_state`` call
is involved, which CONTRACT.md section 6 forbids during an episode anyway. This
runs *before* an episode, and authors no pose.

Exit codes are meaningful, because ``run_task.sh`` branches on them:
    0  world verified at spawn and the plan solves
    2  transport: could not connect, or the sim publishes nothing
    3  the reset did not take within the deadline
    4  the apple is somewhere the arm cannot execute the plan from
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
from inspect_robots_ros._client import RosbridgeClient

from robot_console.arm.embodiment import apple_state_from
from robot_console.arm.policy import replace_config
from robot_console.arm.ros_settings import (
    APPLE_BODY,
    CAMERA_SPECS,
    DEFAULT_URL,
    FREE_JOINT_STATES_TOPIC,
    FREE_JOINT_STATES_TYPE,
    OVERHEAD_CAMERA_TYPE,
    TASK_MANAGER_RESET_SERVICE,
)
from robot_console.arm.task import APPLE_XYZ
from robot_console.arm.waypoints import PickPlaceConfig, build_plan

_SUBSCRIPTION_ID = "scene-reset-free-joint-states"

#: Position tolerance for "back at spawn", metres. Generous on purpose: the
#: apple settles to z = 0.0196 against the MJCF's 0.020, and the tolerance has
#: to absorb that 0.4 mm rather than chase it.
_SPAWN_TOLERANCE = 0.005

#: Speed under which the apple counts as still, m/s. A reset apple reads ~1e-12;
#: one still rolling reads orders of magnitude more.
_STILL_SPEED = 1e-3

#: Consecutive in-tolerance samples required. One could be a message in flight
#: from before the reset landed.
_STABLE_SAMPLES = 3

EXIT_OK, EXIT_TRANSPORT, EXIT_RESET_FAILED, EXIT_OUT_OF_REACH = 0, 2, 3, 4


def _sample(client: RosbridgeClient, *, after_seq: int, timeout_s: float):
    """Wait for a newer free-joint message and parse the apple out of it."""
    sample = client.wait_for_sample(
        FREE_JOINT_STATES_TOPIC, after_seq=after_seq, timeout_s=timeout_s
    )
    position, speed, _ = apple_state_from(sample.msg, APPLE_BODY)
    return sample.seq, position, speed


def _view_publishes(client: RosbridgeClient, view: str, *, timeout_s: float = 8.0) -> bool:
    """Whether a named camera view is actually delivering frames.

    Subscribing is not evidence -- rosbridge accepts a subscription to a topic
    nobody publishes and simply never sends anything. So this waits for a frame.
    The timeout is generous because cameras run at a few Hz, and slower still
    when several are enabled.
    """
    try:
        topic = CAMERA_SPECS[view][0]
    except KeyError:
        print(f"unknown camera view {view!r}; known: {sorted(CAMERA_SPECS)}")
        return False
    subscription = f"scene-reset-view-{view}"
    client.subscribe(
        topic,
        subscription_id=subscription,
        message_type=OVERHEAD_CAMERA_TYPE,
        throttle_rate=0,
        queue_length=1,
    )
    try:
        client.wait_for_sample(topic, after_seq=0, timeout_s=timeout_s)
    except TimeoutError:
        return False
    finally:
        with contextlib.suppress(Exception):
            client.unsubscribe(topic, subscription_id=subscription)
    return True


def _describe(position: np.ndarray) -> str:
    radius = float(np.linalg.norm(position[:2]))
    return (
        f"({position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f})  "
        f"r={radius:.4f} m from the base"
    )


def _plan_solves(apple: np.ndarray) -> tuple[bool, str]:
    """Ask whether the pick-and-place plan is executable from this apple pose.

    "In reach" is deliberately not a radius compared against a number typed
    here. It is ``build_plan`` succeeding -- the same call, the same
    ``max_position_error`` gate and the same jaw-centre refinement the scripted
    policy runs -- with the apple where it actually is. That cannot drift out of
    step with the policy, and it needs no second definition of reach.

    Note what this does and does not say: it is a kinematic verdict, that every
    waypoint has an IK solution. It says nothing about whether the arm can get
    there without colliding with something.
    """
    config = replace_config(
        PickPlaceConfig(), apple_xyz=(float(apple[0]), float(apple[1]), float(apple[2]))
    )
    try:
        plan = build_plan(config)
    except ValueError as exc:
        return False, str(exc)
    worst = max(solve.position_error for solve in plan.solves)
    return True, f"every waypoint solves, worst IK residual {worst * 1000:.3f} mm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="report where the apple actually is, and do not reset it",
    )
    parser.add_argument(
        "--allow-out-of-reach",
        action="store_true",
        help="warn instead of refusing when the plan does not solve; a diagnostic "
        "run against a displaced apple is a legitimate thing to want",
    )
    parser.add_argument(
        "--require-view",
        action="append",
        default=[],
        metavar="NAME",
        help="fail unless this camera view is publishing; repeatable. Used by run_task.sh "
        "to catch an eye-in-hand policy pointed at a simulator started without "
        "./start_sim.sh --wrist, which would otherwise surface minutes later as a "
        "missing-topic timeout naming a topic rather than the flag that creates it.",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="seconds to wait for the reset to land"
    )
    parser.add_argument("--json", default=None, help="write the verdict here as provenance")
    args = parser.parse_args()

    client = RosbridgeClient(args.url)
    try:
        client.connect()
    # Deliberately broad: a refused socket, a DNS failure, a TLS error and a
    # websocket handshake rejection all mean the same thing to the caller --
    # there is no simulator here -- and narrowing this would let some of them
    # escape as a traceback instead of the exit code run_task.sh branches on.
    except Exception as exc:  # noqa: BLE001
        print(f"cannot reach rosbridge at {args.url}: {exc}")
        return EXIT_TRANSPORT

    record: dict[str, object] = {"url": args.url, "reset": not args.no_reset}
    try:
        for view in args.require_view:
            if not _view_publishes(client, view):
                print(f"camera view {view!r} is not publishing on {args.url}")
                return EXIT_TRANSPORT
            record.setdefault("views_present", []).append(view)  # type: ignore[union-attr]
        client.subscribe(
            FREE_JOINT_STATES_TOPIC,
            subscription_id=_SUBSCRIPTION_ID,
            message_type=FREE_JOINT_STATES_TYPE,
            throttle_rate=0,
            queue_length=1,
        )
        # An open port is not a running simulator: Docker Desktop's proxy accepts
        # connections on 9090 with nothing listening behind it. Waiting for an
        # actual message is the only honest readiness test.
        try:
            seq, before, _ = _sample(client, after_seq=0, timeout_s=10.0)
        except TimeoutError:
            print(f"connected to {args.url}, but nothing publishes {FREE_JOINT_STATES_TOPIC}")
            print("the port can be open with no simulator behind it; try ./start_sim.sh --status")
            return EXIT_TRANSPORT
        if before is None:
            print(f"{FREE_JOINT_STATES_TOPIC} carries no {APPLE_BODY!r} entry")
            return EXIT_TRANSPORT

        print(f"apple before: {_describe(before)}")
        record["before"] = before.tolist()
        drift = float(np.linalg.norm(before - np.asarray(APPLE_XYZ)))

        if args.no_reset:
            print(f"--no-reset: leaving the world as it is ({drift:.4f} m from spawn)")
            after = before
        else:
            if drift > _SPAWN_TOLERANCE:
                print(f"  {drift:.4f} m from spawn — a previous episode left it there")
            client.call_service(TASK_MANAGER_RESET_SERVICE)
            # The Trigger reply is fire-and-forget downstream, so it proves
            # nothing. Wait for the apple to be measured back at spawn instead.
            deadline = time.monotonic() + args.timeout
            stable = 0
            after = before
            while stable < _STABLE_SAMPLES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(f"apple still at {_describe(after)} {args.timeout:g}s after /reset")
                    print("the reset was acknowledged but never landed; is task_manager alive?")
                    return EXIT_RESET_FAILED
                try:
                    seq, position, speed = _sample(
                        client, after_seq=seq, timeout_s=min(2.0, remaining)
                    )
                except TimeoutError:
                    continue
                if position is None:
                    continue
                after = position
                at_spawn = float(np.linalg.norm(position - np.asarray(APPLE_XYZ)))
                still = speed is None or speed < _STILL_SPEED
                stable = stable + 1 if (at_spawn <= _SPAWN_TOLERANCE and still) else 0
            print(f"apple after:  {_describe(after)}   (verified over {_STABLE_SAMPLES} samples)")

        record["after"] = after.tolist()
        record["radius_m"] = float(np.linalg.norm(after[:2]))

        reachable, detail = _plan_solves(after)
        record["reachable"] = reachable
        record["reach_detail"] = detail
        if reachable:
            print(f"in reach:     {detail}")
        else:
            print(f"OUT OF REACH: {detail}")
            if not args.allow_out_of_reach:
                print(
                    "refusing to start a run the arm cannot complete; --allow-out-of-reach overrides"
                )
                return EXIT_OUT_OF_REACH
            print("--allow-out-of-reach: continuing anyway")
        return EXIT_OK
    finally:
        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
