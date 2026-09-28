"""The arm task's wire checks and reset, as ``run_task.sh`` runs them.

    python -m robot_console.arm.preflight discover --url URL
    python -m robot_console.arm.preflight check    --url URL [--namespace NS]
    python -m robot_console.arm.preflight wait     --url URL [--namespace NS] [--timeout S]
    python -m robot_console.arm.preflight reset    --url URL [--namespace NS]

* ``discover`` prints the SO-101's namespace, found by its signature command topic
  (``joint_trajectory_controller/joint_trajectory`` of type ``JointTrajectory``) in
  ``/rosapi/topics``. Two candidates, or none, is an error naming what was found.
* ``check`` compares the SO-101's typed interface and the rig's with what ``rosapi``
  reports, type for type, and **refuses a wire lacking the composed ``/reset`` or the
  rig**: those are simulation-only, so the arm task never runs on hardware.
* ``wait`` waits for the *topics* -- a message on each of ``/joint_states``, both rig
  views and both ``camera_info``s -- not for the port: a listening socket says nothing
  about whether a scene compiled.
* ``reset`` calls the composed ``/reset`` (``std_srvs/srv/Trigger``) and exits 3 with the
  server's message on ``success: false``.

Exit codes: 0 ok; 2 transport (unreachable, or nothing published in time); 3 reset
refused; 5 not a simulator (no ``/reset`` or no rig); 6 interface mismatch, or no single
SO-101 to be found.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Mapping

from inspect_robots_ros._client import RosbridgeClient

from robot_console.arm import ros_settings as rs
from robot_console.discovery import namespace_of

EXIT_OK, EXIT_TRANSPORT, EXIT_RESET_REFUSED, EXIT_NOT_SIMULATION, EXIT_INTERFACE = 0, 2, 3, 5, 6


class PreflightError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _connect(url: str) -> RosbridgeClient:
    client = RosbridgeClient(url, connect_timeout_s=5.0, request_timeout_s=15.0)
    try:
        client.connect()
    except Exception as exc:  # noqa: BLE001 - every failure means "nothing here"
        raise PreflightError(EXIT_TRANSPORT, f"cannot reach rosbridge at {url}: {exc}") from exc
    return client


def _call(client: RosbridgeClient, service: str, args: Mapping | None = None) -> dict:
    try:
        return dict(client.call_service(service, args).values)
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(EXIT_TRANSPORT, f"{service} did not answer: {exc}") from exc


def topics(client: RosbridgeClient) -> dict[str, str]:
    values = _call(client, "/rosapi/topics")
    names = list(values.get("topics") or [])
    types = list(values.get("types") or [])
    types += [""] * (len(names) - len(types))
    return dict(zip(names, types))


def discover_namespace(present: Mapping[str, str]) -> str:
    """The one SO-101 namespace on the wire, or `PreflightError` naming the candidates."""
    found = sorted({ns for topic, kind in present.items()
                    if kind == rs.ARM_COMMAND_TYPE
                    and (ns := namespace_of(topic, rs.ARM_COMMAND_TOPIC)) is not None})
    if not found:
        raise PreflightError(EXIT_INTERFACE, "no SO-101 on the wire: nothing publishes or "
                             f"subscribes {rs.ARM_COMMAND_TOPIC} as {rs.ARM_COMMAND_TYPE}")
    if len(found) > 1:
        shown = ", ".join(f"/{ns}" if ns else "<bare>" for ns in found)
        raise PreflightError(EXIT_INTERFACE, f"several SO-101s on the wire ({shown}); "
                             "choose one with --namespace")
    return found[0]


def interface_problems(present: Mapping[str, str], services: Mapping[str, str],
                       actions: Mapping[str, str], namespace: str) -> tuple[list[str], list[str]]:
    """``(not_simulation, mismatched)``: what refuses the wire, and what is wrong on it."""
    arm = rs.arm_interface(namespace)
    reset = rs.namespaced_reset(namespace)
    refuse: list[str] = []
    wrong: list[str] = []
    if reset not in services:
        refuse.append(f"no {reset} service: the arm task needs the simulation-only reset, "
                      "so it does not run on hardware")
    elif services[reset] != rs.RESET_SERVICE_TYPE:
        wrong.append(f"{reset} is {services[reset] or 'untyped'}, not {rs.RESET_SERVICE_TYPE}")
    rig_missing = [t for t in rs.rig_interface() if t not in present]
    if rig_missing:
        refuse.append(f"no worktop rig: missing {', '.join(rig_missing)}")
    for topic, kind in {**arm["topics"], **rs.rig_interface()}.items():
        if topic in present and present[topic] != kind:
            wrong.append(f"{topic} is {present[topic] or 'untyped'}, not {kind}")
        elif topic not in present and topic not in rig_missing:
            wrong.append(f"missing topic {topic} ({kind})")
    for action, kind in arm["actions"].items():
        if action not in actions:
            wrong.append(f"missing action {action} ({kind})")
        elif actions[action] != kind:
            wrong.append(f"{action} is {actions[action] or 'untyped'}, not {kind}")
    return refuse, wrong


def check(client: RosbridgeClient, namespace: str) -> None:
    present = topics(client)
    names = list(_call(client, "/rosapi/services").get("services") or [])
    services = {n: str(_call(client, "/rosapi/service_type", {"service": n}).get("type") or "")
                for n in names if n == rs.namespaced_reset(namespace)}
    services.update({n: "" for n in names if n not in services})
    wanted = rs.arm_interface(namespace)["actions"]
    listed = set(_call(client, "/rosapi/action_servers").get("action_servers") or [])
    actions = {a: str(_call(client, "/rosapi/action_type", {"action": a}).get("type") or "")
               for a in wanted if a in listed}
    refuse, wrong = interface_problems(present, services, actions, namespace)
    if refuse:
        raise PreflightError(EXIT_NOT_SIMULATION, "refusing this wire:\n  " + "\n  ".join(refuse))
    if wrong:
        raise PreflightError(EXIT_INTERFACE, "the SO-101's interface is not the official "
                             "one:\n  " + "\n  ".join(wrong))


def wait_topics(client: RosbridgeClient, namespace: str, timeout_s: float) -> None:
    settings = rs.RosSettings(namespace=namespace)
    wanted = {
        settings.topic(rs.JOINT_STATES_TOPIC): rs.JOINT_STATES_TYPE,
        settings.topic(rs.WRIST_CAMERA_TOPIC): rs.WRIST_CAMERA_TYPE,
        **{rs.rig_topic(rs.CAMERA_SPECS[v][0]): rs.OVERHEAD_CAMERA_TYPE
           for v in (rs.OVERHEAD_CAMERA_NAME, rs.SIDE_CAMERA_NAME)},
        **{t: rs.CAMERA_INFO_TYPE for t in settings.camera_info_topics().values()},
    }
    for index, (topic, kind) in enumerate(wanted.items()):
        client.subscribe(topic, subscription_id=f"preflight-{index}", message_type=kind,
                         throttle_rate=0, queue_length=1)
    deadline = time.monotonic() + timeout_s
    try:
        for topic in wanted:
            remaining = deadline - time.monotonic()
            try:
                client.wait_for_sample(topic, after_seq=0, timeout_s=max(remaining, 0.01))
            except TimeoutError as exc:
                raise PreflightError(EXIT_TRANSPORT, f"nothing on {topic} within "
                                     f"{timeout_s:g}s") from exc
    finally:
        for index, topic in enumerate(wanted):
            with contextlib.suppress(Exception):
                client.unsubscribe(topic, subscription_id=f"preflight-{index}")


def reset(client: RosbridgeClient, namespace: str) -> str:
    service = rs.namespaced_reset(namespace)
    try:
        response = client.call_service(service)
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(EXIT_TRANSPORT, f"{service} did not answer: {exc}") from exc
    values = response.values if isinstance(response.values, Mapping) else {}
    message = str(values.get("message") or "")
    if values.get("success") is not True:
        raise PreflightError(EXIT_RESET_REFUSED, f"{service} answered success: false"
                             + (f": {message}" if message else ""))
    return message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("discover", "check", "wait", "reset"))
    parser.add_argument("--url", default=rs.DEFAULT_URL)
    parser.add_argument("--namespace", default=None,
                        help="the SO-101's namespace; default: discovered ('' is the bare "
                             "contract)")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    client: RosbridgeClient | None = None
    try:
        client = _connect(args.url)
        namespace = args.namespace
        if namespace is None or args.command == "discover":
            namespace = discover_namespace(topics(client))
        if args.command == "discover":
            print(namespace)
        elif args.command == "check":
            check(client, namespace)
            print(f"SO-101 on {'/' + namespace if namespace else 'the bare contract'}, "
                  f"/reset and the rig present, every type as specified")
        elif args.command == "wait":
            wait_topics(client, namespace, args.timeout)
        else:
            message = reset(client, namespace)
            print(f"reset: {message}" if message else "reset")
        return EXIT_OK
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
