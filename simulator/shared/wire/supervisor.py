#!/usr/bin/env python3
"""A wire container's entry point: one robot interface (or one component's) on one ROS
graph with rosbridge_suite.

It attaches to the simulation with the spawn's token, starts the ROS graph (ROS 1: a
master; ROS 2: nothing to start), sets the recorded parameters, starts rosbridge_websocket
and rosapi on the wire port, then every node of the robot's wire plan (`robots/<id>.py`): the
stock ROS packages the recorded boot runs where they need no hardware, and the simulated
drivers for the rest. When every recorded node, topic and service is on the graph it
prints RSIM-WIRE-READY. It exits -- and the container with it -- when the simulation
removes the robot or ends.
"""

from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
import threading
import time
import xmlrpc.client
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
from rosbridge_client import wait_for  # noqa: E402

READY_MARK = "RSIM-WIRE-READY"
WIRE_PORT = int(os.environ.get("RSIM_WIRE_PORT", "9090"))
procs: list = []
stopping = threading.Event()


def log(msg):
    print(f"[wire] {msg}", flush=True)


def ros_env() -> dict:
    e = dict(os.environ)
    e.setdefault("ROS_MASTER_URI", "http://127.0.0.1:11311")
    e.setdefault("ROS_HOSTNAME", "127.0.0.1")
    e["PYTHONPATH"] = f"{HERE}:{HERE.parent}:" + e.get("PYTHONPATH", "")
    return e


OVERLAYS: list = []


def source_cmd(cmd) -> list:
    """Run a command inside the ROS environment (and the image's and plan's overlays)."""
    distro = os.environ.get("ROS_DISTRO", "noetic")
    overlays = ["/opt/rsim_ws/install/setup.bash", "/opt/rsim_ws/devel/setup.bash"] + OVERLAYS
    line = f"source /opt/ros/{distro}/setup.bash; "
    for ov in overlays:
        line += f"if [ -f {ov} ]; then source {ov}; fi; "
    import shlex

    return ["bash", "-c", line + "exec " + " ".join(shlex.quote(str(c)) for c in cmd)]


def start(name: str, cmd, env=None):
    log(f"starting {name}: {' '.join(map(str, cmd))[:200]}")
    p = subprocess.Popen(source_cmd(cmd), env=env or ros_env(), start_new_session=True)
    procs.append((name, p))
    return p


def stop_all():
    stopping.set()
    for name, p in reversed(procs):
        try:
            os.killpg(p.pid, signal.SIGINT)
        except OSError:
            pass
    deadline = time.time() + 4
    for name, p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass


def wait_master(timeout=30):
    m = xmlrpc.client.ServerProxy("http://127.0.0.1:11311")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            m.getSystemState("/rsim")
            return m
        except OSError:
            time.sleep(0.2)
    raise SystemExit("the ROS master did not start")


def main() -> int:
    robot = common.owner()
    iface = common.interface(robot)
    dialect = iface["dialect"]
    link = common.SimLink(on_lost=lambda: (log("the simulation removed the robot; exiting"),
                                           stop_all(), os._exit(0)))
    log(f"attached to the simulation as {link.role} of {link.robot_id} "
        f"({robot.id}, {dialect} {iface.get('ros_distribution')})")
    module = importlib.import_module(f"robots.{robot.id}")
    plan = module.plan(robot, iface, link.describe)

    def on_signal(signum, frame):
        stop_all()
        os._exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    for step in getattr(plan, "prepare", []):
        log("preparing: building the simulator's ROS packages (first use of this revision)")
        res = subprocess.run(["bash", "-c", step])
        if res.returncode != 0:
            log("preparing the wire failed")
            return 1
    OVERLAYS.extend(getattr(plan, "overlays", []))
    for name, cmd in getattr(plan, "helpers", []):
        start(name, cmd)

    if dialect == "ros1":
        start("roscore", ["roscore", "-p", "11311"])
        master = wait_master()
        # roscore sets /run_id once it has started /rosout; launches wait for it.
        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                master.getParam("/rsim", "/run_id")
                if master.lookupNode("/rsim", "/rosout")[0] == 1:
                    break
            except OSError:
                pass
            time.sleep(0.2)
        for row in common.served(iface.get("parameters")):
            value = plan.params.get(row["name"], common.param_value(robot, row)) \
                if hasattr(plan, "params") else common.param_value(robot, row)
            if isinstance(value, dict) and "generated" in value:
                raise SystemExit(f"parameter {row['name']} is generated but the wire plan "
                                 "gives no value for it")
            master.setParam("/rsim", row["name"], value)
        start("rosbridge", ["roslaunch", "--wait", "rosbridge_server",
                            "rosbridge_websocket.launch", f"port:={WIRE_PORT}", "address:=0.0.0.0"])
    else:
        start("rosbridge", ["ros2", "launch", "rosbridge_server",
                            "rosbridge_websocket_launch.xml", f"port:={WIRE_PORT}", "address:=0.0.0.0"])
    for name, cmd in plan.procs:
        start(name, cmd)
    if os.environ.get("RSIM_TEST_FAIL_ROLE") == link.role:
        # test hook (tests/e2e/test_lifecycle.py): this wire fails during startup
        log("RSIM_TEST_FAIL_ROLE: failing this wire's startup on purpose")
        stop_all()
        return 3

    # Ready once rosbridge answers and the graph holds every recorded node, topic and
    # service (all but the optional rows).
    need_nodes = {n["name"] for n in common.served(iface.get("nodes"))}
    need_topics = {t["name"] for t in common.served(iface.get("topics"))}
    need_services = {s["name"] for s in common.served(iface.get("services"))}
    rb = wait_for("127.0.0.1", WIRE_PORT, 120)
    t0 = time.time()
    missing = None
    while time.time() - t0 < 150:
        try:
            nodes = set(rb.call("/rosapi/nodes", timeout=10).get("nodes", []))
            topics = set(rb.call("/rosapi/topics", timeout=10).get("topics", []))
            services = set(rb.call("/rosapi/services", timeout=10).get("services", []))
        except Exception as exc:
            missing = f"rosapi: {exc}"
            time.sleep(1)
            continue
        missing = sorted(need_nodes - nodes) + sorted(need_topics - topics) + \
            sorted(need_services - services)
        if not missing:
            break
        time.sleep(0.5)
    rb.close()
    if missing:
        log(f"not ready after 150 s; missing: {missing}")
        stop_all()
        return 1
    log(f"{READY_MARK} {robot.id} role {link.role}: {len(need_nodes)} nodes, "
        f"{len(need_topics)} topics, {len(need_services)} services")
    # Stay until the simulation removes the robot (the link's on_lost exits the process).
    # A ROS 1 node that dies without unregistering (SIGKILL, a crash) stays on the master
    # and so on rosapi's node list; the wire checks each recorded node answers, and a lost
    # one ends the wire (the spawn reports it and cleans up).
    last_check = time.time()
    while not stopping.is_set():
        for name, p in procs:
            if p.poll() is not None and not getattr(p, "_reported", False):
                p._reported = True
                log(f"process {name} exited with status {p.returncode}")
        if dialect == "ros1" and time.time() - last_check >= 2.0:
            last_check = time.time()
            lost = ros1_lost_nodes(sorted(need_nodes))
            if lost:
                log(f"lost node(s) {', '.join(lost)}")
                stop_all()
                return 4
        time.sleep(1)
    return 0


class _TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = 3.0
        return conn


def ros1_lost_nodes(names) -> list:
    """Recorded nodes that are not registered or whose XML-RPC server no longer answers
    (twice in a row, so one slow reply is not a loss)."""
    master = xmlrpc.client.ServerProxy("http://127.0.0.1:11311", transport=_TimeoutTransport())
    lost = []
    for n in names:
        ok = False
        for _ in range(2):
            try:
                code, _, uri = master.lookupNode("/rsim", n)
                if code != 1:
                    break
                xmlrpc.client.ServerProxy(uri, transport=_TimeoutTransport()).getPid("/rsim")
                ok = True
                break
            except (OSError, xmlrpc.client.Error):
                time.sleep(0.5)
        if not ok:
            lost.append(n)
    return lost


if __name__ == "__main__":
    sys.exit(main())
