"""Reset the simulator's world and verify, from the wire, that it took.

    python -m robot_console.arm.preflight [--url ws://127.0.0.1:9090]

Ported from the reference rig's ``scripts/scene_reset.py`` as-is: the guarantee it
provides is about the *wire*, not about any one simulator, so it did not need to change
when the simulator did. What follows is its original rationale, still true here.

WHY THIS EXISTS. World state persists between runs: a finished episode leaves
the apple wherever it came to rest, and the next run inherits it. That is not a
cosmetic problem: a policy against a drifted apple is graded on clause 5's
displacement from a spawn the apple was never at -- the run completes, scores 0,
and looks like a policy failure rather than a world that was never reset. A
recorded MolmoAct2 failure once left the apple at 0.52 m from the base, beyond
the arm's 0.48 m maximum, where no policy could recover it.

Two layouts are legal (``task.LAYOUTS``): the apple may spawn at the contract's
point or at the plate's, with the plate at the other. Which one is on the wire is
*read*, from where the reset puts the apple, and written to the JSON as
``layout`` -- ``run_task.sh`` hands that to the task so the pose-derived
reference column grades the world that exists rather than the one a flag named.

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
    0  world verified at a known layout's spawn, and both objects are within reach
    2  transport: could not connect, or the sim publishes nothing
    3  the reset did not take within the deadline
    4  the apple is somewhere the arm cannot reach, or at no layout's spawn
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
from robot_console.arm.kinematics import ik_position
from robot_console.arm.ros_settings import (
    APPLE_BODY,
    CAMERA_SPECS,
    DEFAULT_URL,
    FREE_JOINT_STATES_TOPIC,
    FREE_JOINT_STATES_TYPE,
    OVERHEAD_CAMERA_TYPE,
    RosSettings,
    TASK_MANAGER_RESET_SERVICE,
)
from robot_console.arm.task import LAYOUTS, layout_of

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


def _sample(client: RosbridgeClient, *, after_seq: int, timeout_s: float,
            topic: str = FREE_JOINT_STATES_TOPIC):
    """Wait for a newer free-joint message and parse the apple out of it."""
    sample = client.wait_for_sample(topic, after_seq=after_seq, timeout_s=timeout_s)
    position, speed, _ = apple_state_from(sample.msg, APPLE_BODY)
    return sample.seq, position, speed


def _view_publishes(client: RosbridgeClient, view: str, settings: RosSettings,
                    *, timeout_s: float = 8.0) -> bool:
    """Whether a named camera view is actually delivering frames.

    Subscribing is not evidence -- rosbridge accepts a subscription to a topic
    nobody publishes and simply never sends anything. So this waits for a frame.
    The timeout is generous because cameras run at a few Hz, and slower still
    when several are enabled.
    """
    try:
        topic = settings.topic(CAMERA_SPECS[view][0])
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


#: The two poses a pick-and-place has to be able to reach, as (height above the
#: object's resting plane, tool pitch in radians). The grasp is top-down at the apple's
#: centre height; the release is the reference rig's 45 mm over the plate at its
#: shallower pitch. These are what the scripted plan used to solve on every preflight,
#: kept as the two waypoints that decide reachability now that the plan itself is gone.
_GRASP = (0.0, -1.3)
_RELEASE = (0.045, -0.5)
#: A residual over this and the pose is out of reach. 5 mm is the plan's own gate.
_MAX_IK_RESIDUAL = 5e-3


def _within_reach(apple: np.ndarray, plate: np.ndarray) -> tuple[bool, str]:
    """Ask whether a top-down grasp at the apple and a release over the plate both solve.

    "In reach" is deliberately not a radius compared against a number typed here. It is
    the console's own IK converging at the two poses that decide a pick-and-place, with
    the apple where it actually is and the plate where the layout says it is. That is a
    kinematic verdict -- it says nothing about whether the arm can get there without
    colliding with something -- but it is the same solver every waypoint used to go
    through, so it cannot drift from what the arm can do.
    """
    worst = 0.0
    for name, xyz, (lift, pitch) in (("grasp", apple, _GRASP), ("release", plate, _RELEASE)):
        target = (float(xyz[0]), float(xyz[1]), float(xyz[2]) + lift)
        solve = ik_position(target, pitch=pitch, pitch_weight=1.0, max_iterations=600)
        if solve.position_error > _MAX_IK_RESIDUAL:
            return False, (
                f"{name} pose {target} misses by {solve.position_error * 1000:.1f} mm "
                f"(limit {_MAX_IK_RESIDUAL * 1000:.0f} mm)"
            )
        worst = max(worst, solve.position_error)
    return True, f"grasp and release both solve, worst IK residual {worst * 1000:.3f} mm"


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
        "./kitchen.sh serve --wrist, which would otherwise surface minutes later as a "
        "missing-topic timeout naming a topic rather than the flag that creates it.",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="seconds to wait for the reset to land"
    )
    parser.add_argument("--json", default=None, help="write the verdict here as provenance")
    parser.add_argument(
        "--namespace", default=None,
        help="ROS namespace the arm is under (default: %(default)r, meaning the "
             "RosSettings default). Pass an empty string for the bare, unnamespaced "
             "contract a single-robot simulator without namespacing presents.",
    )
    args = parser.parse_args()

    # One settings object, so every topic and the reset service are composed by the same
    # rule the episode itself will use. Reading module constants directly is what let the
    # preflight and the run disagree about which topics they were talking about.
    settings = (RosSettings() if args.namespace is None
                else RosSettings(namespace=args.namespace))
    free_joint_topic = settings.topic(FREE_JOINT_STATES_TOPIC)
    reset_service = settings.topic(TASK_MANAGER_RESET_SERVICE)

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
            if not _view_publishes(client, view, settings):
                print(f"camera view {view!r} is not publishing on {args.url}")
                return EXIT_TRANSPORT
            record.setdefault("views_present", []).append(view)  # type: ignore[union-attr]
        client.subscribe(
            free_joint_topic,
            subscription_id=_SUBSCRIPTION_ID,
            message_type=FREE_JOINT_STATES_TYPE,
            throttle_rate=0,
            queue_length=1,
        )
        # An open port is not a running simulator: Docker Desktop's proxy accepts
        # connections on 9090 with nothing listening behind it. Waiting for an
        # actual message is the only honest readiness test.
        try:
            seq, before, _ = _sample(client, after_seq=0, timeout_s=10.0,
                                     topic=free_joint_topic)
        except TimeoutError:
            print(f"connected to {args.url}, but nothing publishes {free_joint_topic}")
            print("an open port is not a simulator: Docker Desktop's proxy answers with nothing "
                  "behind it (check `docker ps`), and an engine still compiling has bound the "
                  "socket long before it publishes. Start one with ./kitchen.sh serve")
            return EXIT_TRANSPORT
        if before is None:
            print(f"{free_joint_topic} carries no {APPLE_BODY!r} entry")
            return EXIT_TRANSPORT

        print(f"apple before: {_describe(before)}")
        record["before"] = before.tolist()
        # Which layout, if any, the apple is sitting at. Compared in xy: a native
        # engine apple rests a few millimetres higher or lower than the contract's
        # 20 mm sphere, and that is not drift.
        drift = min(
            float(np.hypot(before[0] - spawn[0], before[1] - spawn[1]))
            for spawn, _plate in LAYOUTS.values()
        )

        if args.no_reset:
            print(f"--no-reset: leaving the world as it is ({drift:.4f} m from a spawn)")
            after = before
        else:
            if drift > _SPAWN_TOLERANCE:
                print(f"  {drift:.4f} m from any spawn — a previous episode left it there")
            client.call_service(reset_service)
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
                        client, after_seq=seq, timeout_s=min(2.0, remaining),
                        topic=free_joint_topic,
                    )
                except TimeoutError:
                    continue
                if position is None:
                    continue
                after = position
                at_spawn = layout_of(position, tolerance=_SPAWN_TOLERANCE) is not None
                still = speed is None or speed < _STILL_SPEED
                stable = stable + 1 if (at_spawn and still) else 0
            print(f"apple after:  {_describe(after)}   (verified over {_STABLE_SAMPLES} samples)")

        record["after"] = after.tolist()
        record["radius_m"] = float(np.linalg.norm(after[:2]))

        # The layout is read off the world, not asserted: an apple resting at the
        # plate's contract position means the plate is at the apple's, and the
        # reference predicate downstream has to be told so or it grades against
        # the wrong centre on every step.
        layout = layout_of(after)
        record["layout"] = layout
        if layout is None:
            detail = f"apple is at no known layout's spawn ({_describe(after)})"
            record["reachable"] = False
            record["reach_detail"] = detail
            print(f"OUT OF REACH: {detail}")
            if not args.allow_out_of_reach:
                print("refusing to start a run against a world no layout describes; "
                      "--allow-out-of-reach overrides")
                return EXIT_OUT_OF_REACH
            print("--allow-out-of-reach: continuing anyway")
            return EXIT_OK
        print(f"layout:       {layout}")
        plate = np.asarray(LAYOUTS[layout][1], dtype=np.float64)

        reachable, detail = _within_reach(after, plate)
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
