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

from robot_console.ainex_topics import CONTRACT_TOPICS as AINEX_CONTRACT_TOPICS
from robot_console.topics import (
    TOPIC_CAMERA,
    TOPIC_CMD_VEL,
    TOPIC_ODOM,
    TOPIC_SCAN,
    TOPIC_TF,
    TOPIC_TF_STATIC,
    namespaced,
)

#: Exit codes, so a shell can tell "nothing there" from "the wrong thing is there".
EXIT_OK = 0
EXIT_MISSING = 1
EXIT_TRANSPORT = 2

#: What a mobile base must present, from `topics.py` -- the myAGV contract. `/tf` is the
#: hardware's: `myagv_active.launch` starts `robot_state_publisher`, `robot_pose_ekf` and
#: three static publishers, so a base with no tree is a base a mapping stack cannot use.
#:
#: **`/tf_static` is deliberately not here, and the reason is the vendor's tf version.**
#: Those three static publishers are `pkg="tf"`, not `tf2_ros`, and tf1's node
#: re-publishes onto `/tf` on a period. `robot_state_publisher` has nothing for
#: `/tf_static` either: the vendor URDF's only joint, `base_up`, is `continuous`, so the
#: description carries no fixed joint at all. A real myAGV therefore publishes nothing to
#: `/tf_static`, and requiring it would fail a real robot. The SO-101 is a genuine ROS 2
#: bringup and does have one -- see `arm_topics`.
BASE_TOPICS: tuple[str, ...] = (TOPIC_CMD_VEL, TOPIC_ODOM, TOPIC_CAMERA, TOPIC_SCAN,
                                TOPIC_TF)

#: What a humanoid must present, from `ainex_topics.py` -- the Hiwonder AiNex contract.
#: A third kind rather than a second flavour of base: the AiNex is commanded as a walking
#: state machine and has no wheels, so it shares not one topic with the myAGV beyond
#: `/joint_states` and its camera. Checking it as a base demanded `/cmd_vel` and `/odom`
#: of a biped, which is how `--robots so101,ainex` failed its fleet check even when every
#: topic it does present was on the wire.
HUMANOID_TOPICS: tuple[str, ...] = AINEX_CONTRACT_TOPICS


def arm_topics() -> tuple[str, ...]:
    """What an arm must present, from `arm/ros_settings.py`.

    Imported lazily and tolerantly: `ros_settings` pulls in the arm's kinematics, which is
    part of the optional extra. A console without it can still check a base.

    The two scene cameras used to be in here, and were checked under the arm's namespace
    with everything else. They are the worktop rig's, not the arm's -- see `scene_topics`.
    """
    from robot_console.arm import ros_settings as rs

    return (
        rs.ARM_COMMAND_TOPIC,
        rs.GRIPPER_COMMAND_TOPIC,
        rs.JOINT_STATES_TOPIC,
        rs.FREE_JOINT_STATES_TOPIC,
        rs.TF_TOPIC,
        rs.TF_STATIC_TOPIC,
    )


def scene_topics() -> tuple[str, ...]:
    """What the worktop's fixed camera rig presents, under its own namespace.

    Not a robot, and so not `--arm`'s or `--base`'s business: the rig watches the work
    surface and would still be there with every robot unbolted. An arm task needs it on
    the wire all the same -- the verdict is read off the overhead frame -- so it is
    checked whenever an arm is.
    """
    from robot_console.arm import ros_settings as rs

    return (rs.OVERHEAD_CAMERA_TOPIC, rs.SIDE_CAMERA_TOPIC)


def topics_from(client, timeout_s: float = 10.0) -> dict[str, str]:
    """`{topic: type}` from an already-connected roslibpy client."""
    import roslibpy

    service = roslibpy.Service(client, "/rosapi/topics", "rosapi/Topics")
    result = service.call(roslibpy.ServiceRequest(), timeout=timeout_s)
    names = list(result.get("topics") or [])
    types = list(result.get("types") or [])
    # `types` is positional against `names` and a real rosapi can return fewer of
    # them; pad rather than zip short, so a missing type never hides a topic.
    types += [""] * (len(names) - len(types))
    return dict(zip(names, types))


def list_topics(url: str, timeout_s: float = 10.0) -> dict[str, str]:
    """`{topic: type}` as `/rosapi/topics` reports it. Raises on transport failure.

    Closes the connection and deliberately does **not** terminate it. `close()` sends a
    websocket close; `terminate()` stops roslibpy's process-global Twisted reactor, which
    cannot be restarted -- and `discovery.discover` calls this and then hands the process
    to a `RobotLink` that has to connect afterwards. Adding a `terminate()` here would
    leave teleop unable to open its own connection, with nothing to say why.
    """
    import roslibpy

    host, _, port = url.removeprefix("ws://").removeprefix("wss://").partition(":")
    client = roslibpy.Ros(host=host or "127.0.0.1", port=int(port or 9090))
    client.run(timeout=timeout_s)
    try:
        return topics_from(client, timeout_s)
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
                             "Pass an empty string for the bare, unnamespaced contract. "
                             "The worktop's camera rig is checked too, under its own "
                             "namespace rather than the arm's -- it is not the arm's.")
    parser.add_argument("--base", action="append", default=[], metavar="NS",
                        help="a mobile base is expected under this namespace; repeatable")
    parser.add_argument("--humanoid", action="append", default=[], metavar="NS",
                        help="an AiNex is expected under this namespace; repeatable. Not "
                             "a --base: it walks, so it has no cmd_vel and no odometry.")
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
    for namespace in args.humanoid:
        missing += missing_for(present, namespace, HUMANOID_TOPICS)
    for namespace in args.arm:
        missing += missing_for(present, namespace, arm_topics())
    # The rig is the worktop's, and the simulator stages it for every robot that stands
    # at one -- the arm bolted to it and the humanoid standing on it. Checking it only
    # behind --arm let an AiNex-only kitchen that published no rig at all pass, which
    # agreed with the omission instead of catching it.
    if args.arm or args.humanoid:
        from robot_console.arm import ros_settings as rs

        missing += missing_for(present, rs.SCENE_NAMESPACE, scene_topics())

    if missing:
        print(f"{args.url} is missing {len(missing)} expected topic(s):")
        for topic in missing:
            print(f"  {topic}")
        print(f"it offers {len(present)}: {', '.join(sorted(present))}")
        return EXIT_MISSING

    robots = [f"arm {ns or '<bare>'}" for ns in args.arm]
    robots += [f"base {ns or '<bare>'}" for ns in args.base]
    robots += [f"humanoid {ns or '<bare>'}" for ns in args.humanoid]
    print(f"{args.url}: {len(present)} topics, all expected ones present "
          f"({'; '.join(robots) or 'nothing requested'})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
