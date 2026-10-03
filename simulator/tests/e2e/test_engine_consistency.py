"""Engine consistency (spec §3, §5): the same robot, spawned with the same flags on the
`test` scene of either engine, presents an identical interface (names and types, frame ids
and transform tree, parameter names and values) and behaves the same within the acceptance
bounds. (The household scenes differ by design and are not compared.)"""

import math
import time

import pytest

from conftest import Spawn, docker_ok, engine_ready, real_time_problems, running_sim

import motion
import registry
import wirecheck

pytestmark = pytest.mark.skipif(not (docker_ok() and engine_ready("molmospaces")
                                     and engine_ready("robocasa")),
                                reason="needs Docker and both engines")

ENGINES = {"molmospaces": (9581, 9590), "robocasa": (9582, 9690)}
ROBOTS = registry.ids()


@pytest.fixture(scope="module")
def sims(logdir):
    with running_sim("molmospaces", "test:1", 9581, logdir) as a, \
            running_sim("robocasa", "test:1", 9582, logdir) as b:
        yield {"molmospaces": a, "robocasa": b}


def comparable(snap, frames, edges):
    """The interface as the spec compares it: names and types of nodes, topics, services,
    actions; parameter names and values (ROS infrastructure left out); each periodic
    topic's frame ids; the transform tree's edges and the topic each comes on."""
    return {"nodes": sorted(snap["nodes"]), "topics": sorted(snap["topics"].items()),
            "services": sorted((k, v) for k, v in snap["services"].items() if v is not None),
            "actions": sorted(snap["actions"]),
            "params": sorted((k, repr(v)) for k, v in snap["params"].items()),
            "param_names": sorted(snap["param_names"]),
            "frames": sorted(frames.items()),
            "tf": sorted((p, c, e["topic"]) for (p, c), e in edges.items())}


def drive_differences(rid, a: dict, b: dict) -> list:
    """Each leg of the drive on engine b within the recorded tolerance of the same leg on
    engine a: the displacement vector, the turn's yaw, and what is left after the stop."""
    lin_t, yaw_t, res = motion.drive_tolerances(rid)
    out = []
    for name, la in a.items():
        lb = b[name]
        if name == "turn":
            if not motion.within(lb["yaw"], la["yaw"], yaw_t):
                out.append(f"turn: yaw {lb['yaw']:.3f} vs {la['yaw']:.3f} rad")
        else:
            err = math.hypot(lb["fwd"] - la["fwd"], lb["lat"] - la["lat"])
            lim = max(math.hypot(la["fwd"], la["lat"]) * lin_t.get("relative", 0.0),
                      lin_t.get("absolute", 0.0))
            if err > lim:
                out.append(f"{name}: ({lb['fwd']:.3f}, {lb['lat']:.3f}) vs "
                           f"({la['fwd']:.3f}, {la['lat']:.3f}) m")
        if abs(lb["residual"] - la["residual"]) > res["tolerance"]["absolute"]:
            out.append(f"{name}: {lb['residual']:.3f} vs {la['residual']:.3f} left after the stop")
    return out


@pytest.mark.parametrize("rid", ROBOTS)
def test_same_interface_and_behaviour_on_both_engines(rid, sims, logdir):
    from test_robots import MOTION_CHECKS, ainex_walk

    r = registry.get(rid)
    iface = wirecheck.interface(rid)
    placement = "worktop" if r.kind == "arm" else "floor"
    spawns = {e: Spawn(rid, ENGINES[e][0], ENGINES[e][1], placement=placement,
                       log=logdir / f"ec-{e}-{rid}.log") for e in ENGINES}
    try:
        for sp in spawns.values():
            sp.wait_ready()
        # identical interface
        seen = {e: comparable(wirecheck.snapshot(sp.port, rid),
                              wirecheck.header_frames(sp.port, rid),
                              wirecheck.tf_edges(sp.port, wirecheck.tf_window(iface)))
                for e, sp in spawns.items()}
        assert seen["molmospaces"] == seen["robocasa"], rid
        # the same physical behaviour under the same commands (compared only while both
        # simulations ran in real time: spec §3 Timing)
        t0 = time.time()
        motions = {m["id"] for m in iface.get("motions") or []}
        if "drive" in motions:
            legs = {e: motion.drive_legs(sp.port, rid) for e, sp in spawns.items()}
            assert drive_differences(rid, legs["molmospaces"], legs["robocasa"]) == []
        if "walk" in motions:
            # each engine against the record (the nominal commanded displacement within the
            # recorded tolerance), neither engine the other's reference
            walks = {e: ainex_walk(sp.port, sp.sim_port) for e, sp in spawns.items()}
            assert {e: w[0] for e, w in walks.items()} == {e: [] for e in walks}, walks
        # every other motion reaches its goal on both
        for ids, check in MOTION_CHECKS.get(rid, {}).items():
            if {"drive", "walk"} & set(ids):
                continue
            for e, sp in spawns.items():
                assert check(sp.port, sp.sim_port) == [], (e, ids)
        for e, sp in spawns.items():
            assert real_time_problems(sp.sim_port, t0) == [], e
    finally:
        for sp in spawns.values():
            sp.stop()
