"""Every robot on every engine's default scene (the scenes the reference project loads):
it spawns with its recorded embodiment; its wire presents exactly the recorded interface
at the recorded rates; its recorded commands move it (mobile robots on the floor, arms on
the worktop); its sensors report the simulated world.

    simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/e2e/test_robots.py
"""

import base64
import math
import time

import numpy as np
import pytest

from conftest import Spawn, docker_ok, engine_ready, running_sim, wait_no_containers

import motion
import protocol
import registry
import wirecheck
from rosbridge_client import Rosbridge

ENGINES = [("molmospaces", "ithor:1", 9481), ("robocasa", "robocasa:1-1", 9482)]
ROBOTS = ["myagv", "so101", "ainex", "mycobot280", "myagv_mycobot280", "rosmaster_x3_plus"]
WIRE_PORT = 9490

pytestmark = pytest.mark.skipif(not docker_ok(), reason="Docker is not running")


@pytest.fixture(scope="module", params=ENGINES, ids=[e[0] for e in ENGINES])
def sim(request, logdir):
    engine, scene, port = request.param
    if not engine_ready(engine):
        pytest.skip(f"{engine} is not set up")
    with running_sim(engine, scene, port, logdir) as s:
        yield s


@pytest.fixture(scope="module", params=ROBOTS)
def robot(request, sim, logdir):
    rid = request.param
    r = registry.get(rid)
    placement = "worktop" if r.kind == "arm" else "floor"
    sp = Spawn(rid, sim.port, WIRE_PORT, placement=placement,
               log=logdir / f"spawn-{sim.engine}-{rid}.log")
    line = sp.wait_ready()
    sp.ready_line = line
    sp.sim = sim
    sp.placement = placement
    yield sp
    assert sp.stop() == 0
    assert wait_no_containers(f"label=rsim.sim_port={sim.port}") == []


def wires(sp):
    r = registry.get(sp.robot)
    out = []
    for role, owner in registry.wires(r):
        port = sp.port if role in ("main", "base") else sp.port + 1
        out.append((role, owner.id, port, f"rsim-{sp.sim_port}-{sp.robot}-{role}"))
    return out


def test_loads_with_its_embodiment(robot):
    c = protocol.Client("127.0.0.1", robot.sim_port)
    try:
        rows = {r["id"]: r for r in c.call("robots")["robots"]}
        me = rows[robot.robot]
        assert me["state"] == "running" and me["placement"] == robot.placement
        comps = [x["robot"] for x in me["components"]]
        assert comps == [o.id for _, o in registry.wires(registry.get(robot.robot))]
        # an arm on the worktop has its six worktop objects around it (spec §2.3); the
        # motions below run with them there
        if robot.placement == "worktop":
            assert set(me["staged"]) == {"apple", "plate", "bowl", "mug", "banana", "lemon"}
            bodies = c.call("bodies", names=[o["body"] for o in me["staged"].values()])["bodies"]
            assert len(bodies) == 6
        else:
            assert me["staged"] == {} and me["cleared"] == []
        # every recorded (non-optional) camera is on the model, and every actuated joint
        for role, owner, port, _ in wires(robot):
            iface = wirecheck.interface(owner)
            d = c.call("describe", robot=robot.robot, component=role)
            cams = {x["name"]: x for x in d["cameras"]}
            for cam in (iface.get("sensors") or {}).get("cameras") or []:
                if cam.get("optional"):
                    continue
                assert cam["frame_id"] in cams, cam["frame_id"]
                assert cams[cam["frame_id"]]["resolution"] == [cam["width"], cam["height"]]
            for lid in (iface.get("sensors") or {}).get("lidars") or []:
                assert lid["frame_id"].lstrip("/") in d["sites"]
        # it stands: on the floor or the worktop, upright and at rest
        rd = c.call("readings", robot=robot.robot)
        h = c.call("hello")
        surface = h["floor_z"] if robot.placement == "floor" else h["worktop"]["z"]
        low = min(b["pos"][2] for b in rd["bodies"].values())
        assert low > surface - 0.05
        v = np.linalg.norm(rd["base"]["linvel_world"])
        assert v < 0.05
    finally:
        c.close()
    assert "spawn ready:" in robot.ready_line
    for role, owner, port, _ in wires(robot):
        assert f"ws://127.0.0.1:{port}" in robot.ready_line


def test_contract(robot):
    problems = []
    for role, owner, port, _ in wires(robot):
        snap = wirecheck.snapshot(port, owner)
        known = wirecheck.INTERFACE_FILE_GAPS.get(owner, {})
        for p in wirecheck.contract_problems(owner, snap):
            if p in known:
                print(f"[interface-file gap] {owner}: {p}: {known[p]}")
            else:
                problems.append(f"{owner}: {p}")
    assert problems == []


def test_frames_and_message_fields(robot):
    """Each recorded output topic with a frame carries it, and a message of every
    periodic topic has exactly its declared type's fields."""
    problems = []
    for role, owner, port, _ in wires(robot):
        iface = wirecheck.interface(owner)
        rb = Rosbridge("127.0.0.1", port)
        try:
            rows = [t for t in wirecheck.served(iface["topics"]) if t.get("direction") == "out"
                    and isinstance(t.get("rate"), (int, float))]
            got = {}
            for t in rows:
                def cb(m, name=t["name"]):
                    got.setdefault(name, m["msg"])
                rb.subscribe(t["name"], cb, throttle_rate=500, queue_length=1)
            deadline = time.monotonic() + 15
            while len(got) < len(rows) and time.monotonic() < deadline:
                time.sleep(0.2)
            for t in rows:
                msg = got.get(t["name"])
                if msg is None:
                    problems.append(f"{t['name']}: no message")
                    continue
                typ = t["type"]
                det = rb.call("/rosapi/message_details", {"type": typ}, timeout=20)
                top = det["typedefs"][0]
                if set(top["fieldnames"]) != set(msg):
                    problems.append(f"{t['name']}: fields {sorted(msg)} != {sorted(top['fieldnames'])}")
                fid = t.get("frame_id")
                if fid is not None and "header" in msg and msg["header"].get("frame_id") != fid:
                    problems.append(f"{t['name']}: frame {msg['header'].get('frame_id')!r}, "
                                    f"recorded {fid!r}")
        finally:
            rb.close()
    assert problems == []


def test_rates(robot):
    problems = []
    for role, owner, port, container in wires(robot):
        rates = wirecheck.periodic_topics(owner)
        if not rates:
            continue
        window = max(6.0, 5.5 / min(rates.values()))
        m = wirecheck.measure_on_graph(container, owner, list(rates), window)
        problems += [f"{owner}: {p}" for p in wirecheck.rate_problems(m, rates)]
    assert problems == []


def test_motion(robot):
    rid, sim_port, port = robot.robot, robot.sim_port, robot.port
    problems = []
    if rid in ("myagv", "myagv_mycobot280"):
        problems += motion.drive(port, "myagv")
    if rid == "rosmaster_x3_plus":
        problems += motion.drive(port, rid)
        problems += rosmaster_arm(port, sim_port)
    if rid == "so101":
        problems += so101_arm(port, sim_port)
    if rid == "mycobot280":
        problems += mycobot_arm(port, sim_port, rid, "")
    if rid == "myagv_mycobot280":
        problems += mycobot_arm(port + 1, sim_port, rid, "arm/")
    if rid == "ainex":
        problems += ainex_motions(port, sim_port)
    assert problems == []


# ---------------------------------------------------------------- arms and the humanoid


def so101_arm(port, sim_port):
    out = []
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_flex_joint", "wrist_flex_joint",
             "wrist_roll_joint"]
    target = [0.3, -0.3, 0.3, 0.3, 0.5]
    t_arm = motion.tol("so101", "arm joint")["absolute"]
    t_grip = motion.tol("so101", "gripper")["absolute"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        rb.advertise("/joint_trajectory_controller/joint_trajectory",
                     "trajectory_msgs/msg/JointTrajectory")
        time.sleep(0.8)
        rb.publish("/joint_trajectory_controller/joint_trajectory", {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}, "joint_names": names,
            "points": [{"positions": target, "velocities": [], "accelerations": [], "effort": [],
                        "time_from_start": {"sec": 2, "nanosec": 0}}]})
        err = motion.wait_joints(sim_port, "so101", dict(zip(names, target)), t_arm, 8)
        if max(err.values()) > t_arm:
            out.append(f"so101 arm error {err}")
        # the gripper action: goal reached, result reported
        res = rb.action("/gripper_controller/gripper_cmd",
                        "control_msgs/action/ParallelGripperCommand",
                        {"command": {"name": ["gripper_joint"], "position": [1.0],
                                     "velocity": [], "effort": []}}, timeout=20)
        if not res.get("result", False):
            out.append(f"so101 gripper action result: {res}")
        # no effort state interface is configured: effort is not measured and reads 0.0,
        # never the controller's uninitialised member
        effort = ((res.get("values") or {}).get("state") or {}).get("effort")
        if effort != [0.0]:
            out.append(f"so101 gripper result state.effort {effort}, expected [0.0]")
        err = motion.wait_joints(sim_port, "so101", {"gripper_joint": 1.0}, t_grip, 8)
        if err["gripper_joint"] > t_grip:
            out.append(f"so101 gripper error {err}")
        if not motion.at_rest(sim_port, "so101", names + ["gripper_joint"]):
            out.append("so101 not at rest after its goal")
    finally:
        rb.close()
    return out


def mycobot_arm(port, sim_port, rid, pre):
    out = []
    names = ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3", "joint5_to_joint4",
             "joint6_to_joint5", "joint6output_to_joint6", "gripper_controller"]
    target = [0.3, -0.3, 0.5, -0.4, 0.2, 0.5, 0.15]
    t_arm = motion.tol("mycobot280", "arm joint")["absolute"]
    t_grip = motion.tol("mycobot280", "gripper")["absolute"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        rb.advertise("/joint_states", "sensor_msgs/msg/JointState")
        time.sleep(0.8)
        rb.publish("/joint_states", {"header": {"stamp": {"sec": 0, "nanosec": 0},
                                                "frame_id": ""},
                                     "name": names, "position": target, "velocity": [],
                                     "effort": []})
        want = {pre + n: v for n, v in zip(names[:6], target[:6])}
        err = motion.wait_joints(sim_port, rid, want, t_arm, 8)
        if max(err.values()) > t_arm:
            out.append(f"{rid} arm error {err}")
        err = motion.wait_joints(sim_port, rid, {pre + "gripper_controller": 0.15}, t_grip, 5)
        if max(err.values()) > t_grip:
            out.append(f"{rid} gripper error {err}")
        if not motion.at_rest(sim_port, rid, list(want)):
            out.append(f"{rid} not at rest")
    finally:
        rb.close()
    return out


def rosmaster_arm(port, sim_port):
    out = []
    servo = [90, 120, 30, 45, 90, 146]
    names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint"]
    want = {n: math.radians(d - 90) for n, d in zip(names[:5], servo[:5])}
    want["grip_joint"] = math.radians((146 - 30) * 90 / 150 - 90)
    rb = Rosbridge("127.0.0.1", port)
    try:
        rb.advertise("/TargetAngle", "yahboomcar_msgs/ArmJoint")
        time.sleep(0.8)
        rb.publish("/TargetAngle", {"id": 0, "angle": 0.0, "joints": servo, "run_time": 1000})
        t_arm = motion.tol("rosmaster_x3_plus", "arm joint")["absolute"]
        t_grip = motion.tol("rosmaster_x3_plus", "gripper")["absolute"]
        err = motion.wait_joints(sim_port, "rosmaster_x3_plus", want, max(t_arm, t_grip), 6)
        bad = {k: e for k, e in err.items() if e > (t_grip if k == "grip_joint" else t_arm)}
        if bad:
            out.append(f"rosmaster arm/gripper error {bad}")
    finally:
        rb.close()
    return out


def ainex_motions(port, sim_port):
    out = []
    rb = Rosbridge("127.0.0.1", port)
    try:
        # head
        rb.advertise("/head_pan_controller/command", "ainex_interfaces/HeadState")
        time.sleep(0.8)
        rb.publish("/head_pan_controller/command", {"position": 0.5, "duration": 0.5})
        t_head = motion.tol("ainex", "head")["absolute"]
        err = motion.wait_joints(sim_port, "ainex", {"head_pan": 0.5}, t_head, 4)
        if err["head_pan"] > t_head:
            out.append(f"head error {err}")
        rb.publish("/head_pan_controller/command", {"position": 0.0, "duration": 0.5})
        time.sleep(1.0)
        # walk, then the recorded stop
        r0 = motion.readings(sim_port, "ainex")["base"]
        prm = rb.call("/walking/get_param", {})["parameters"]
        prm["x_move_amplitude"] = 0.013
        rb.advertise("/walking/set_param", "ainex_interfaces/WalkingParam")
        time.sleep(0.5)
        rb.publish("/walking/set_param", prm)
        time.sleep(0.3)
        rb.call("/walking/command", {"command": "start"})
        time.sleep(4.0)
        rb.call("/walking/command", {"command": "stop"}, timeout=20)
        time.sleep(1.5)
        r1 = motion.readings(sim_port, "ainex")["base"]
        q0 = r0["quat"]
        yaw0 = math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
        q1 = r1["quat"]
        yaw1 = math.atan2(2 * (q1[0] * q1[3] + q1[1] * q1[2]), 1 - 2 * (q1[2] ** 2 + q1[3] ** 2))
        dx, dy = r1["pos"][0] - r0["pos"][0], r1["pos"][1] - r0["pos"][1]
        fwd = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        drift = motion.tol("ainex", "heading drift")["absolute"]
        if not fwd > 0.05:
            out.append(f"walk: forward displacement {fwd:.3f} m")
        if abs(motion.wrap(yaw1 - yaw0)) > drift:
            out.append(f"walk: heading drift {yaw1 - yaw0:.3f} rad")
        if abs(r1["pos"][2] - r0["pos"][2]) > 0.03:
            out.append(f"walk: body height changed {r1['pos'][2] - r0['pos'][2]:.3f} m (fell?)")
        if np.linalg.norm(r1["linvel_world"][:2]) > 0.03:
            out.append("walk: not at rest after the stop")
        # an action group, back to the init pose
        j0 = motion.joints(sim_port, "ainex")
        rb.advertise("/app/set_action", "std_msgs/String")
        time.sleep(0.5)
        rb.publish("/app/set_action", {"data": "wave"})
        moved = 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6:
            j = motion.joints(sim_port, "ainex")
            moved = max(moved, abs(j["r_sho_roll"] - j0["r_sho_roll"]))
            time.sleep(0.1)
        time.sleep(1.0)
        j1 = motion.joints(sim_port, "ainex")
        t_ag = motion.tol("ainex", "action group")["absolute"]
        if moved < 0.5:
            out.append(f"action group: r_sho_roll moved only {moved:.2f} rad")
        back = {k: abs(j1[k] - j0[k]) for k in ("r_sho_roll", "r_sho_pitch", "r_el_pitch")}
        if max(back.values()) > t_ag:
            out.append(f"action group: not back at the init pose {back}")
    finally:
        rb.close()
    return out


# ---------------------------------------------------------------- sensors


def test_sensors(robot):
    """Cameras show what the simulation renders from the same camera; lidars and IMUs
    report the simulated world."""
    problems = []
    c = protocol.Client("127.0.0.1", robot.sim_port)
    try:
        for role, owner, port, _ in wires(robot):
            iface = wirecheck.interface(owner)
            rb = Rosbridge("127.0.0.1", port)
            try:
                for cam in (iface.get("sensors") or {}).get("cameras") or []:
                    if cam.get("optional") or cam["encoding"] not in ("rgb8", "yuv422_yuy2"):
                        continue
                    box = []
                    rb.subscribe(cam["image_topic"], lambda m: box.append(m["msg"]),
                                 throttle_rate=1000, queue_length=1)
                    t0 = time.monotonic()
                    while not box and time.monotonic() - t0 < 10:
                        time.sleep(0.1)
                    rb.unsubscribe(cam["image_topic"])
                    if not box:
                        problems.append(f"{cam['image_topic']}: no image")
                        continue
                    img = box[0]
                    raw = np.frombuffer(base64.b64decode(img["data"]), np.uint8)
                    h, w = img["height"], img["width"]
                    if cam["encoding"] == "rgb8":
                        luma = raw.reshape(h, w, 3).astype(float).mean(axis=2)
                    else:
                        luma = raw.reshape(h, w * 2)[:, 0::2].astype(float)
                    rr = c.call("render", view={"camera": f"{robot.robot}/"
                                                          f"{'arm/' if role == 'arm' else ''}"
                                                          f"{cam['frame_id']}"},
                                width=w, height=h, format="rgb")
                    ref = np.frombuffer(rr["_payload"], np.uint8).reshape(h, w, 3).astype(float)
                    diff = np.abs(luma - ref.mean(axis=2)).mean()
                    if not diff < 12.0:
                        problems.append(f"{cam['image_topic']}: differs from the render by "
                                        f"{diff:.1f} grey levels")
                for lid in (iface.get("sensors") or {}).get("lidars") or []:
                    box = []
                    rb.subscribe(lid["topic"], lambda m: box.append(m["msg"]), queue_length=1)
                    t0 = time.monotonic()
                    while not box and time.monotonic() - t0 < 10:
                        time.sleep(0.1)
                    rb.unsubscribe(lid["topic"])
                    if not box:
                        problems.append(f"{lid['topic']}: no scan")
                        continue
                    s = box[0]
                    if len(s["ranges"]) != lid["samples"]:
                        problems.append(f"{lid['topic']}: {len(s['ranges'])} samples")
                    if abs(s["angle_min"] - lid["angle_min"]) > 1e-4:
                        problems.append(f"{lid['topic']}: angle_min {s['angle_min']}")
                    valid = [r for r in s["ranges"] if r and lid["range_min"] <= r <= lid["range_max"]]
                    if len(valid) < 0.3 * len(s["ranges"]):
                        problems.append(f"{lid['topic']}: only {len(valid)} valid ranges")
            finally:
                rb.close()
    finally:
        c.close()
    assert problems == []
