"""Engine consistency (spec §3, §5): the same robot, spawned with the same flags on the
`test` scene of either engine, presents an identical interface and behaves the same within
the acceptance bounds. (The household scenes differ by design and are not compared.)"""

import math
import time

import pytest

from conftest import Spawn, docker_ok, engine_ready, running_sim

import motion
import registry
import wirecheck
from rosbridge_client import Rosbridge

pytestmark = pytest.mark.skipif(not (docker_ok() and engine_ready("molmospaces")
                                     and engine_ready("robocasa")),
                                reason="needs Docker and both engines")

ENGINES = {"molmospaces": (9581, 9590), "robocasa": (9582, 9690)}
ROBOTS = ["myagv", "so101", "ainex", "mycobot280", "myagv_mycobot280", "rosmaster_x3_plus"]


@pytest.fixture(scope="module")
def sims(logdir):
    with running_sim("molmospaces", "test:1", 9581, logdir) as a, \
            running_sim("robocasa", "test:1", 9582, logdir) as b:
        yield {"molmospaces": a, "robocasa": b}


def comparable(snap):
    """The interface as the spec compares it: names and types of nodes, topics, services,
    actions; parameter names and values (ROS infrastructure left out)."""
    return {"nodes": sorted(snap["nodes"]), "topics": sorted(snap["topics"].items()),
            "services": sorted((k, v) for k, v in snap["services"].items() if v is not None),
            "actions": sorted(snap["actions"]),
            "params": sorted((k, repr(v)) for k, v in snap["params"].items())}


def forward_walk(port, sim_port, rid):
    rb = Rosbridge("127.0.0.1", port)
    try:
        r0 = motion.readings(sim_port, rid)["base"]
        prm = rb.call("/walking/get_param", {})["parameters"]
        prm["x_move_amplitude"] = 0.013
        rb.advertise("/walking/set_param", "ainex_interfaces/WalkingParam")
        time.sleep(0.5)
        rb.publish("/walking/set_param", prm)
        time.sleep(0.3)
        rb.call("/walking/command", {"command": "start"})
        time.sleep(4.0)
        rb.call("/walking/command", {"command": "stop"}, timeout=20)
        time.sleep(1.0)
        r1 = motion.readings(sim_port, rid)["base"]
        return math.hypot(r1["pos"][0] - r0["pos"][0], r1["pos"][1] - r0["pos"][1])
    finally:
        rb.close()


def drive_forward(port, rid, odom="/odom"):
    rb = Rosbridge("127.0.0.1", port)
    try:
        o = motion.Latest(rb, odom)
        o.wait()
        rb.advertise("/cmd_vel", "geometry_msgs/Twist")
        time.sleep(0.8)
        p0 = o.msg["pose"]["pose"]["position"]
        rb.publish("/cmd_vel", motion.twist(0.2))
        time.sleep(1.25)
        rb.publish("/cmd_vel", motion.twist())
        time.sleep(1.2)
        p1 = o.msg["pose"]["pose"]["position"]
        return math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
    finally:
        rb.close()


@pytest.mark.parametrize("rid", ROBOTS)
def test_same_interface_and_behaviour_on_both_engines(rid, sims, logdir):
    r = registry.get(rid)
    placement = "worktop" if r.kind == "arm" else "floor"
    spawns = {e: Spawn(rid, ENGINES[e][0], ENGINES[e][1], placement=placement,
                       log=logdir / f"ec-{e}-{rid}.log") for e in ENGINES}
    try:
        for sp in spawns.values():
            sp.wait_ready()
        # identical interface, wire by wire
        for role, owner in registry.wires(r):
            snaps = {}
            for e, sp in spawns.items():
                port = sp.port if role in ("main", "base") else sp.port + 1
                snaps[e] = comparable(wirecheck.snapshot(port, owner.id))
            assert snaps["molmospaces"] == snaps["robocasa"], owner.id
        # the same physical behaviour under the same commands
        if rid in ("myagv", "myagv_mycobot280", "rosmaster_x3_plus"):
            d = {e: drive_forward(sp.port, rid) for e, sp in spawns.items()}
            t = motion.tol("myagv" if rid != "rosmaster_x3_plus" else rid,
                           "displacement")
            assert motion.within(d["robocasa"], d["molmospaces"], t), d
        if rid == "ainex":
            d = {e: forward_walk(sp.port, sp.sim_port, rid) for e, sp in spawns.items()}
            assert d["molmospaces"] > 0.05 and \
                motion.within(d["robocasa"], d["molmospaces"], motion.tol(rid, "walk displacement")), d
        if rid in ("so101",):
            from test_robots import so101_arm

            for e, sp in spawns.items():
                assert so101_arm(sp.port, sp.sim_port) == [], e
        if rid in ("mycobot280",):
            from test_robots import mycobot_arm

            for e, sp in spawns.items():
                assert mycobot_arm(sp.port, sp.sim_port, rid, "") == [], e
    finally:
        for sp in spawns.values():
            sp.stop()
