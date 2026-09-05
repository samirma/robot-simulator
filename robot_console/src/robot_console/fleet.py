"""Ask a rosbridge what robots are on it, and check they are the ones expected.

A listening socket says nothing. Neither does a driving arm: an episode can run perfectly
while a second robot on the same port published nothing at all, because nothing the arm
does touches the base's topics. So the multi-robot claim needs its own check, and this is
it -- one `/rosapi/topics` call, compared against **the console's own contract
constants**, namespaced.

Checking against our own constants rather than a list typed into a shell script is the
point. `topics.py` and `arm/ros_settings.py` are what the console actually subscribes to;
if the wire and those disagree, the run was going to fail later and less legibly. A
hand-written expectation list would drift from both.

Built on roslibpy, which is a base runtime dependency, rather than on the arm extra's
client: a fleet check has to work on a console installed with nothing but numpy, OpenCV
and roslibpy. One shot per process, so roslibpy's process-global single-shot Twisted
reactor is not a constraint here -- see `bridge.py` for where it is.

    python -m robot_console.fleet --url ws://127.0.0.1:9090 --arm so101 --base myagv
    python -m robot_console.fleet --dump                 # just print what is out there
"""

from __future__ import annotations

import argparse
import sys

from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TOPIC_SCAN,
    namespaced,
)

#: Exit codes, so a shell can tell "nothing there" from "the wrong thing is there".
EXIT_OK = 0
EXIT_MISSING = 1
EXIT_TRANSPORT = 2

#: What a mobile base must present, from `topics.py` -- the myAGV contract.
BASE_TOPICS: tuple[str, ...] = (TOPIC_CMD_VEL, TOPIC_ODOM, TOPIC_CAMERA, TOPIC_SCAN)


def arm_topics() -> tuple[str, ...]:
    """What an arm must present, from `arm/ros_settings.py`.

    Imported lazily and tolerantly: `ros_settings` pulls in the arm's kinematics, which is
    part of the optional extra. A console without it can still check a base.
    """
    from robot_console.arm import ros_settings as rs

    return (
        rs.ARM_COMMAND_TOPIC,
        rs.GRIPPER_COMMAND_TOPIC,
        rs.JOINT_STATES_TOPIC,
        rs.FREE_JOINT_STATES_TOPIC,
        rs.OVERHEAD_CAMERA_TOPIC,
        rs.SIDE_CAMERA_TOPIC,
    )


def list_topics(url: str, timeout_s: float = 10.0) -> dict[str, str]:
    """`{topic: type}` as `/rosapi/topics` reports it. Raises on transport failure."""
    import roslibpy

    host, _, port = url.removeprefix("ws://").removeprefix("wss://").partition(":")
    client = roslibpy.Ros(host=host or "127.0.0.1", port=int(port or 9090))
    client.run(timeout=timeout_s)
    try:
        service = roslibpy.Service(client, "/rosapi/topics", "rosapi/Topics")
        result = service.call(roslibpy.ServiceRequest(), timeout=timeout_s)
        names = list(result.get("topics") or [])
        types = list(result.get("types") or [])
        # `types` is positional against `names` and a real rosapi can return fewer of
        # them; pad rather than zip short, so a missing type never hides a topic.
        types += [""] * (len(names) - len(types))
        return dict(zip(names, types))
    finally:
        client.close()


def missing_for(present: dict[str, str], namespace: str,
                expected: tuple[str, ...]) -> list[str]:
    """Which of `expected`, under `namespace`, the wire is not offering."""
    return [t for t in (namespaced(e, namespace) for e in expected) if t not in present]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:9090")
    parser.add_argument("--arm", action="append", default=[], metavar="NS",
                        help="an SO-101 is expected under this namespace; repeatable. "
                             "Pass an empty string for the bare, unnamespaced contract.")
    parser.add_argument("--base", action="append", default=[], metavar="NS",
                        help="a mobile base is expected under this namespace; repeatable")
    parser.add_argument("--dump", action="store_true",
                        help="print every topic on the wire, sorted, and exit 0. This is "
                             "how the two engines are compared: their lists must match, "
                             "which is the 'a client cannot tell them apart' invariant "
                             "checked rather than eyeballed across two terminals.")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    try:
        present = list_topics(args.url, args.timeout)
    except Exception as exc:  # noqa: BLE001 - every transport failure means the same thing
        print(f"cannot reach rosbridge at {args.url}: {exc}")
        return EXIT_TRANSPORT

    if args.dump:
        for topic in sorted(present):
            print(f"{topic}\t{present[topic]}")
        return EXIT_OK

    missing: list[str] = []
    for namespace in args.base:
        missing += missing_for(present, namespace, BASE_TOPICS)
    for namespace in args.arm:
        missing += missing_for(present, namespace, arm_topics())

    if missing:
        print(f"{args.url} is missing {len(missing)} expected topic(s):")
        for topic in missing:
            print(f"  {topic}")
        print(f"it offers {len(present)}: {', '.join(sorted(present))}")
        return EXIT_MISSING

    robots = [f"arm {ns or '<bare>'}" for ns in args.arm]
    robots += [f"base {ns or '<bare>'}" for ns in args.base]
    print(f"{args.url}: {len(present)} topics, all expected ones present "
          f"({'; '.join(robots) or 'nothing requested'})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
