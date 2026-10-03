"""Every robot of the robot specification on every engine's default scene (the scenes the
reference project loads): it spawns with its recorded embodiment and is ready to take its
recorded commands once the readiness line is printed; its wire presents exactly the
recorded interface, frames, transform tree and rates; its recorded commands move it
(mobile robots on the floor, arms on the worktop) within their limits and answer as
recorded; its sensors report the simulated world; every recorded figure is reproduced
within its recorded tolerance.

    simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/e2e/test_robots.py
"""

import base64
import math
import re
import subprocess
import time

import numpy as np
import pytest

from conftest import Spawn, docker_containers, docker_ok, engine_ready, real_time_problems, \
    running_sim, wait_no_containers

import motion
import protocol
import registry
import wirecheck
from rosbridge_client import Rosbridge

ENGINES = [("molmospaces", "ithor:1", 9481), ("robocasa", "robocasa:1-1", 9482)]
ROBOTS = registry.ids()
WIRE_PORT = 9490
G = 9.81   # MuJoCo's gravity, m/s^2

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
    sp.ready_line = sp.wait_ready()
    sp.at_ready = readiness(WIRE_PORT, rid) + outputs_at_ready(sp)
    sp.sim = sim
    sp.placement = placement
    sp.ended_by_test = False      # test_limits ends a robot whose node ends, as recorded
    yield sp
    code = sp.stop()
    assert code == 0 or sp.ended_by_test
    assert wait_no_containers(f"label=rsim.sim_port={sim.port}") == []


def container(sp) -> str:
    """The robot's wire container."""
    names = set(docker_containers(f"label=rsim.robot={sp.robot}")) & \
        set(docker_containers(f"label=rsim.sim_port={sp.sim_port}"))
    assert len(names) == 1, names
    return names.pop()


def readiness(port, rid) -> list:
    """What a client finds the moment the readiness line is out, asked once with no retry
    (spec §2.3: a spawn succeeds only when its wire is ready to accept its recorded
    commands): every recorded action server is up and, where the robot runs ros2_control,
    every loaded controller is active, those serving the recorded actions among them."""
    iface = wirecheck.interface(rid)
    if iface["dialect"] != "ros2":
        return []
    actions = wirecheck.served(iface.get("actions"))
    services = {s["name"] for s in wirecheck.served(iface.get("services"))}
    out = []
    try:
        rb = Rosbridge("127.0.0.1", port)
    except OSError as exc:
        return [f"no rosbridge at readiness: {exc}"]
    try:
        up = set(rb.call("/rosapi/action_servers", {}, timeout=30).get("action_servers", []))
        out += [f"action {a['name']} not served at readiness" for a in actions
                if a["name"] not in up]
        if "/controller_manager/list_controllers" in services:
            listed = rb.call("/controller_manager/list_controllers", {}, timeout=30)
            state = {c["name"]: c.get("state") for c in listed.get("controller") or []}
            out += [f"controller {n} is {s} at readiness" for n, s in state.items()
                    if s != "active"]
            out += [f"controller {a['node'].lstrip('/')} of {a['name']} not loaded at readiness"
                    for a in actions if a["node"].lstrip("/") not in state]
    except Exception as exc:
        out.append(f"asking at readiness: {exc}")
    finally:
        rb.close()
    return out


def outputs_at_ready(sp) -> list:
    """Spec §2.3: a ready wire provides its required outputs -- every recorded periodic
    output topic is publishing once the readiness line is out (seen by the native probes
    on the wire's own graph, over three periods of the slowest)."""
    rates = wirecheck.periodic_topics(sp.robot)
    if not rates:
        return []
    try:
        m = wirecheck.measure_on_graph(container(sp), sp.robot, list(rates),
                                       max(1.5, 3.0 / min(rates.values())))
    except Exception as exc:
        return [f"probing the outputs at readiness: {exc}"]
    return [f"{t}: no message right after readiness" for t in rates if not m["times"].get(t)]


def motion_row(iface, mid) -> dict:
    return next(m for m in iface.get("motions") or [] if m["id"] == mid)


def limited(row) -> list:
    """The command fields of a motion row that record a maximum."""
    return [f for f in row["command"].get("fields") or [] if f.get("max") is not None]


def field_joint(f) -> str:
    """The joint a command field names in parentheses: `positions[0] (shoulder_pan_joint)`."""
    return re.search(r"\((\w+)", f["field"]).group(1)


def test_loads_with_its_embodiment(robot):
    c = protocol.Client("127.0.0.1", robot.sim_port)
    try:
        rows = {r["id"]: r for r in c.call("robots")["robots"]}
        me = rows[robot.robot]
        assert me["state"] == "running" and me["placement"] == robot.placement
        # an arm on the worktop has its six worktop objects around it (spec §2.3); the
        # motions below run with them there
        if robot.placement == "worktop":
            assert set(me["staged"]) == {"apple", "plate", "bowl", "mug", "banana", "lemon"}
            bodies = c.call("bodies", names=[o["body"] for o in me["staged"].values()])["bodies"]
            assert len(bodies) == 6
        else:
            assert me["staged"] == {} and me["cleared"] == []
        # every recorded (non-optional) camera is on the model, and every lidar and IMU site
        iface = wirecheck.interface(robot.robot)
        sensors = iface.get("sensors") or {}
        d = c.call("describe", robot=robot.robot)
        cams = {x["name"]: x for x in d["cameras"]}
        for cam in wirecheck.served(sensors.get("cameras")):
            assert cam["frame_id"] in cams, cam["frame_id"]
            assert cams[cam["frame_id"]]["resolution"] == [cam["width"], cam["height"]]
        for s in wirecheck.served(sensors.get("lidars")) + wirecheck.served(sensors.get("imus")):
            assert s["frame_id"].lstrip("/") in d["sites"], s["frame_id"]
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
    assert f"ws://127.0.0.1:{robot.port}" in robot.ready_line


def test_ready_means_ready(robot):
    assert robot.at_ready == []


# ---------------------------------------------------------------- recorded figures

#: Where each recorded tolerance (`tolerances[]`, by a fragment of its figure) is judged;
#: "here" is test_recorded_figures. A recorded figure no check judges fails it.
JUDGED_BY = {
    "myagv": {"drive displacement": "test_motion", "drive rotation": "test_motion",
              "residual chassis speed": "test_motion",
              "lidar range, 0.12-1 m": "test_sensors", "lidar range, 1-6 m": "test_sensors",
              "lidar scan rate": "test_rates (within ±10%)"},
    "ainex": {"walk displacement": "test_motion", "walk heading drift": "test_motion",
              "head joint": "test_motion", "action group": "test_motion",
              "standing height": "here"},
    "rosmaster_x3_plus": {"translation displacement": "test_motion", "yaw change": "test_motion",
                          "residual motion": "test_motion", "arm joint": "test_motion",
                          "gripper": "test_motion", "camera and lidar mount poses": "here",
                          "camera horizontal field of view": "here"},
    "so101": {"arm joint": "test_motion", "gripper": "test_motion"},
    "mycobot280": {"end-effector position repeatability": "here", "arm joint": "test_motion",
                   "gripper": "test_motion"},
}

FLANGE = {"mycobot280": "joint6_flange"}   # the URDF link of the flange


def mount_poses(robot, iface, row) -> list:
    """Each camera and lidar of the compiled model (at its home pose) sits at its recorded
    mount, relative to the mount's parent link."""
    import mujoco

    import robot_model

    m = mujoco.MjModel.from_xml_path(str(robot_model.model_file(registry.get(robot.robot))))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    lim = row["tolerance"]["absolute"]
    sensors = iface.get("sensors") or {}
    out = []
    for kind, obj, rows in (("camera", mujoco.mjtObj.mjOBJ_CAMERA, sensors.get("cameras")),
                            ("lidar", mujoco.mjtObj.mjOBJ_SITE, sensors.get("lidars"))):
        for s in wirecheck.served(rows):
            mount = s["mount"]
            i = mujoco.mj_name2id(m, obj, s["frame_id"].lstrip("/"))
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mount["parent"].lstrip("/"))
            if i < 0 or b < 0:
                out.append(f"{kind} {s['frame_id']} or its mount parent {mount['parent']} is "
                           "not on the model")
                continue
            pos = d.cam_xpos[i] if kind == "camera" else d.site_xpos[i]
            local = d.xmat[b].reshape(3, 3).T @ (pos - d.xpos[b])
            err = float(np.linalg.norm(local - np.asarray(mount["xyz"], float)))
            if err > lim:
                out.append(f"{kind} {s['frame_id']} is {err * 1000:.1f} mm from its recorded mount")
    return out


def fields_of_view(robot, iface, row) -> list:
    """Each camera's horizontal field of view, from the rendered camera's fovy and from the
    wire's CameraInfo (where it carries a calibration), within the recorded figure."""
    lim = row["tolerance"]["absolute"]
    cams = [c for c in wirecheck.served((iface.get("sensors") or {}).get("cameras"))
            if c.get("hfov_deg") is not None]
    c = protocol.Client("127.0.0.1", robot.sim_port)
    try:
        fovy = {x["name"]: x["fovy"] for x in c.call("describe", robot=robot.robot)["cameras"]}
    finally:
        c.close()
    infos = wirecheck.sample(robot.port, [x["info_topic"] for x in cams if x.get("info_topic")])
    out = []
    for cam in cams:
        w, h = cam["width"], cam["height"]
        got = {"rendered": math.degrees(2 * math.atan(
            math.tan(math.radians(fovy[cam["frame_id"]]) / 2) * w / h))}
        info = infos.get(cam.get("info_topic"))
        if cam.get("info_topic") and info is None:
            out.append(f"{cam['info_topic']}: no message")
        k = (info or {}).get("K") or (info or {}).get("k") or [0.0]
        if k[0] > 0:
            got["CameraInfo"] = math.degrees(2 * math.atan(w / 2 / k[0]))
        for src, hfov in got.items():
            if abs(hfov - cam["hfov_deg"]) > lim:
                out.append(f"{cam['image_topic']}: {src} hfov {hfov:.2f} deg, recorded "
                           f"{cam['hfov_deg']}")
    return out


def standing_height(robot, iface, row) -> list:
    """The body's height above the floor at the init pose, against the recorded model
    value (the figure's notes: "Model value <h> m")."""
    body = re.search(r"of (\w+) origin", row["figure"]).group(1)
    want = float(re.search(r"([\d.]+) m\b", row.get("notes") or "").group(1))
    c = protocol.Client("127.0.0.1", robot.sim_port)
    try:
        z = c.call("readings", robot=robot.robot)["bodies"][body]["pos"][2] - \
            c.call("hello")["floor_z"]
    finally:
        c.close()
    if abs(z - want) > row["tolerance"]["absolute"]:
        return [f"{body} stands {z:.4f} m above the floor, recorded {want} m"]
    return []


def repeatability(robot, iface, row) -> list:
    """The flange's position after the same joint command, reached from two different
    poses, within the recorded repeatability."""
    rid, sim_port = robot.robot, robot.sim_port
    ex = motion_row(iface, "arm")["command"]["example"]
    names, target = ex["name"], ex["position"][:6]
    t_arm = motion.tol(rid, "arm joint")["absolute"]
    g = limited(motion_row(iface, "gripper"))[0]
    rb = Rosbridge("127.0.0.1", robot.port)
    try:
        rb.advertise("/joint_states", "sensor_msgs/msg/JointState")
        time.sleep(0.8)

        def go(arm):
            # the gripper stays as it is (inside its recorded range: outside, the node ends)
            grip = min(max(motion.joints(sim_port, rid)[names[6]], float(g["min"])),
                       float(g["max"]))
            rb.publish("/joint_states", {"header": {"stamp": {"sec": 0, "nanosec": 0},
                                                    "frame_id": ""},
                                         "name": names, "position": list(arm) + [grip],
                                         "velocity": [], "effort": []})
            motion.wait_joints(sim_port, rid, dict(zip(names, arm)), t_arm, 10)
            motion.at_rest(sim_port, rid, names[:6])
            return np.asarray(motion.readings(sim_port, rid)["bodies"][FLANGE[rid]]["pos"])

        first = go(target)
        go([0.0] * 6)
        again = go(target)
    finally:
        rb.close()
    err = float(np.linalg.norm(again - first))
    if err > row["tolerance"]["absolute"]:
        return [f"{FLANGE[rid]} returns {err * 1000:.2f} mm from where the same command put it"]
    return []


FIGURE_CHECKS = {"camera and lidar mount poses": mount_poses,
                 "camera horizontal field of view": fields_of_view,
                 "standing height": standing_height,
                 "end-effector position repeatability": repeatability}


def test_recorded_figures(robot):
    """Every recorded tolerance is judged by some check (`JUDGED_BY`); the figures no other
    check judges are judged here, before any motion (the robot in its init pose)."""
    iface = wirecheck.interface(robot.robot)
    judged = JUDGED_BY.get(robot.robot, {})
    problems = []
    for row in iface.get("tolerances") or []:
        hits = [f for f in judged if f in row["figure"]]
        if len(hits) != 1:
            problems.append(f"no check judges the recorded figure {row['figure']!r}")
        elif judged[hits[0]] == "here":
            problems += FIGURE_CHECKS[hits[0]](robot, iface, row)
    assert problems == []


# ---------------------------------------------------------------- the interface


def test_contract(robot):
    snap = wirecheck.snapshot(robot.port, robot.robot)
    assert wirecheck.contract_problems(robot.robot, snap) == []


def test_frames_and_message_fields(robot):
    """Each recorded output topic with a frame (and a child frame) carries it, and a
    message of every periodic topic has exactly its declared type's fields."""
    iface = wirecheck.interface(robot.robot)
    rows = [t for t in wirecheck.served(iface["topics"])
            if t["name"] in wirecheck.periodic_topics(robot.robot)]
    got = wirecheck.sample(robot.port, [t["name"] for t in rows])
    problems = []
    rb = Rosbridge("127.0.0.1", robot.port)
    try:
        for t in rows:
            msg = got.get(t["name"])
            if msg is None:
                problems.append(f"{t['name']}: no message")
                continue
            det = rb.call("/rosapi/message_details", {"type": t["type"]}, timeout=20)
            top = det["typedefs"][0]
            if set(top["fieldnames"]) != set(msg):
                problems.append(f"{t['name']}: fields {sorted(msg)} != {sorted(top['fieldnames'])}")
            fid = t.get("frame_id")
            if fid is not None and "header" in msg and msg["header"].get("frame_id") != fid:
                problems.append(f"{t['name']}: frame {msg['header'].get('frame_id')!r}, "
                                f"recorded {fid!r}")
            cid = t.get("child_frame_id")
            if cid is not None and "child_frame_id" in msg and msg["child_frame_id"] != cid:
                problems.append(f"{t['name']}: child frame {msg['child_frame_id']!r}, "
                                f"recorded {cid!r}")
    finally:
        rb.close()
    assert problems == []


def test_tf_tree(robot):
    """The transform tree is the recorded one: every edge, on its recorded topic, fixed
    transforms at their recorded poses, and no other edge."""
    iface = wirecheck.interface(robot.robot)
    edges = wirecheck.tf_edges(robot.port, wirecheck.tf_window(iface))
    assert wirecheck.tf_problems(iface, edges) == []


def test_rates(robot):
    """Every served periodic topic at its recorded rate with no gap over three periods and
    stamps that are acquisition times; a periodic topic with no recorded rate fails. Rates
    measured while the simulation ran below real time are not claimed (spec §3 Timing)."""
    iface = wirecheck.interface(robot.robot)
    problems = wirecheck.rate_record_problems(iface)
    rates = wirecheck.periodic_topics(robot.robot)
    if rates:
        window = max(6.0, 5.5 / min(rates.values()))
        t0 = time.time()
        m = wirecheck.measure_on_graph(container(robot), robot.robot, list(rates), window)
        problems += wirecheck.rate_problems(m, rates, wirecheck.multi_publisher(iface))
        problems += real_time_problems(robot.sim_port, t0)
    assert problems == []


# ---------------------------------------------------------------- motions


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
        rb.publish("/joint_trajectory_controller/joint_trajectory", trajectory(names, target))
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


def trajectory(names, positions, seconds=2):
    return {"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}, "joint_names": names,
            "points": [{"positions": list(positions), "velocities": [], "accelerations": [],
                        "effort": [], "time_from_start": {"sec": seconds, "nanosec": 0}}]}


def mycobot_arm(port, sim_port):
    out = []
    rid = "mycobot280"
    names = ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3", "joint5_to_joint4",
             "joint6_to_joint5", "joint6output_to_joint6", "gripper_controller"]
    target = [0.3, -0.3, 0.5, -0.4, 0.2, 0.5, 0.15]
    t_arm = motion.tol(rid, "arm joint")["absolute"]
    t_grip = motion.tol(rid, "gripper")["absolute"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        rb.advertise("/joint_states", "sensor_msgs/msg/JointState")
        time.sleep(0.8)
        rb.publish("/joint_states", {"header": {"stamp": {"sec": 0, "nanosec": 0},
                                                "frame_id": ""},
                                     "name": names, "position": target, "velocity": [],
                                     "effort": []})
        want = dict(zip(names[:6], target[:6]))
        err = motion.wait_joints(sim_port, rid, want, t_arm, 8)
        if max(err.values()) > t_arm:
            out.append(f"{rid} arm error {err}")
        err = motion.wait_joints(sim_port, rid, {"gripper_controller": 0.15}, t_grip, 5)
        if max(err.values()) > t_grip:
            out.append(f"{rid} gripper error {err}")
        if not motion.at_rest(sim_port, rid, list(want)):
            out.append(f"{rid} not at rest")
    finally:
        rb.close()
    return out


ROSMASTER_SERVOS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5",
                    "grip_joint"]


def rosmaster_arm(port, sim_port):
    out = []
    servo = [90, 120, 30, 45, 90, 146]
    names = ROSMASTER_SERVOS
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
        # the recorded measured feedback: /CurrentAngle answers the six servo angles (deg,
        # servo order 1..6; -1 only for a servo that did not answer) -- here every servo
        # answers, at the commanded angle within the recorded tolerance
        angles = rb.call("/CurrentAngle", {"apply": ""}, timeout=20).get("angles") or []
        lim = [math.degrees(t_arm)] * 5 + [math.degrees(t_grip) * 150 / 90]
        if len(angles) != 6 or any(a == -1 or abs(a - s) > l + 1
                                   for a, s, l in zip(angles, servo, lim)):
            out.append(f"/CurrentAngle answered {angles}, commanded {servo}")
    finally:
        rb.close()
    return out


def ainex_head(port, sim_port):
    """The head pan to the recorded example, judged by the joint readings and by the
    recorded feedback: /ros_robot_controller/bus_servo/get_position answers success and
    each head servo's (ids 23, 24) position in pulses, pulse = 500 + rad * 1000/240 deg."""
    out = []
    rb = Rosbridge("127.0.0.1", port)
    try:
        rb.advertise("/head_pan_controller/command", "ainex_interfaces/HeadState")
        time.sleep(0.8)
        rb.publish("/head_pan_controller/command", {"position": 0.5, "duration": 0.5})
        t_head = motion.tol("ainex", "head joint")["absolute"]
        err = motion.wait_joints(sim_port, "ainex", {"head_pan": 0.5}, t_head, 4)
        if err["head_pan"] > t_head:
            out.append(f"head error {err}")
        per_rad = 1000.0 / math.radians(240.0)
        reply = rb.call("/ros_robot_controller/bus_servo/get_position", {"id": [23, 24]},
                        timeout=20)
        got = {int(p["id"]): int(p["position"]) for p in reply.get("position") or []}
        want = {23: 500 + 0.5 * per_rad, 24: 500 + motion.joints(sim_port, "ainex")["head_tilt"]
                * per_rad}
        if reply.get("success") is not True or set(got) != {23, 24}:
            out.append(f"bus_servo/get_position answered {reply}, recorded success and "
                       "positions of servos 23 and 24")
        elif any(abs(got[i] - want[i]) > t_head * per_rad + 1 for i in want):
            out.append(f"bus_servo/get_position pulses {got}, recorded mapping gives "
                       f"{ {i: round(v) for i, v in want.items()} }")
        rb.publish("/head_pan_controller/command", {"position": 0.0, "duration": 0.5})
        time.sleep(1.0)
    finally:
        rb.close()
    return out


def _yaw(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def ainex_walk(port, sim_port):
    """A timed straight walk with the recorded example's parameters, then the recorded stop:
    (problems, forward displacement). The displacement is judged against the nominal
    commanded one -- x_move_amplitude per completed gait period, the record's
    `motions[walk].command.step_definition` -- within the recorded "walk displacement"
    tolerance; the stop blocks until the step cycle ends, so start to stop's return is a
    whole number of gait periods."""
    out = []
    ex = motion_row(wirecheck.interface("ainex"), "walk")["command"]["example"]
    prm = ex["set_param"]
    period = float(prm["period_time"]) / 1000.0
    rb = Rosbridge("127.0.0.1", port)
    try:
        r0 = motion.readings(sim_port, "ainex")["base"]
        rb.advertise("/walking/set_param", "ainex_interfaces/WalkingParam")
        time.sleep(0.5)
        rb.publish("/walking/set_param", prm)
        time.sleep(0.3)
        t0 = time.monotonic()
        rb.call(ex["start"]["service"], ex["start"]["request"])
        time.sleep(4.0)
        rb.call("/walking/command", {"command": "stop"}, timeout=20)
        periods = round((time.monotonic() - t0) / period)
        time.sleep(1.5)
        r1 = motion.readings(sim_port, "ainex")["base"]
        yaw0, yaw1 = _yaw(r0["quat"]), _yaw(r1["quat"])
        dx, dy = r1["pos"][0] - r0["pos"][0], r1["pos"][1] - r0["pos"][1]
        fwd = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        drift = motion.tol("ainex", "heading drift")["absolute"]
        nominal = float(prm["x_move_amplitude"]) * periods
        if not (fwd > 0 and motion.within(fwd, nominal, motion.tol("ainex", "walk displacement"))):
            out.append(f"walk: forward displacement {fwd:.3f} m over {periods} gait periods, "
                       f"nominal {nominal:.3f} m (x_move_amplitude per period)")
        if abs(motion.wrap(yaw1 - yaw0)) > drift:
            out.append(f"walk: heading drift {yaw1 - yaw0:.3f} rad")
        if abs(r1["pos"][2] - r0["pos"][2]) > 0.03:
            out.append(f"walk: body height changed {r1['pos'][2] - r0['pos'][2]:.3f} m (fell?)")
        if np.linalg.norm(r1["linvel_world"][:2]) > 0.03:
            out.append("walk: not at rest after the stop")
    finally:
        rb.close()
    return out, fwd


def ainex_action_group(port, sim_port):
    """The recorded example (no pinned source holds its .d6a, so, as the record says, the
    controller only returns to the init pose), then the record's smoke example, one of its
    estimated groups (robots_specs/ainex/action_groups.yml) -- it moves the right arm and
    ends back at the init pose."""
    out = []
    command = motion_row(wirecheck.interface("ainex"), "action_group")["command"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        j0 = motion.joints(sim_port, "ainex")
        rb.advertise("/app/set_action", "std_msgs/String")
        time.sleep(0.5)
        t_ag = motion.tol("ainex", "action group")["absolute"]
        example = command["example"]
        rb.publish("/app/set_action", example)
        time.sleep(2.0)
        j1 = motion.joints(sim_port, "ainex")
        off = {k: abs(j1[k] - j0[k]) for k in ("r_sho_roll", "r_sho_pitch", "r_el_pitch")}
        if max(off.values()) > t_ag:
            out.append(f"action group {example['data']!r}: not at the init pose {off}")
        rb.publish("/app/set_action", command["smoke_example"])
        moved = 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6:
            j = motion.joints(sim_port, "ainex")
            moved = max(moved, abs(j["r_sho_roll"] - j0["r_sho_roll"]))
            time.sleep(0.1)
        time.sleep(1.0)
        j1 = motion.joints(sim_port, "ainex")
        if moved < 0.5:
            out.append(f"action group: r_sho_roll moved only {moved:.2f} rad")
        back = {k: abs(j1[k] - j0[k]) for k in ("r_sho_roll", "r_sho_pitch", "r_el_pitch")}
        if max(back.values()) > t_ag:
            out.append(f"action group: not back at the init pose {back}")
    finally:
        rb.close()
    return out


#: The checks of each robot's recorded motions, by the `motions[].id`s each one covers, in
#: the order they run; a recorded motion no check covers fails test_motion.
MOTION_CHECKS = {
    "myagv": {("drive",): lambda port, sim_port: motion.drive(port, "myagv")},
    "rosmaster_x3_plus": {
        ("drive",): lambda port, sim_port: motion.drive(port, "rosmaster_x3_plus"),
        ("arm", "gripper"): rosmaster_arm},
    "so101": {("arm", "gripper"): so101_arm},
    "mycobot280": {("arm", "gripper"): mycobot_arm},
    "ainex": {("head",): ainex_head,
              ("walk",): lambda port, sim_port: ainex_walk(port, sim_port)[0],
              ("action_group",): ainex_action_group},
}


def test_motion(robot):
    rid = robot.robot
    checks = MOTION_CHECKS.get(rid, {})
    covered = {m for ids in checks for m in ids}
    recorded = [m["id"] for m in wirecheck.interface(rid).get("motions") or []]
    problems = [f"no check of the recorded motion {m!r}" for m in recorded if m not in covered]
    t0 = time.time()
    for check in checks.values():
        problems += check(robot.port, robot.sim_port)
    problems += real_time_problems(robot.sim_port, t0)
    assert problems == []


# ---------------------------------------------------------------- recorded behaviours


def internal_endpoints(port, iface) -> list:
    """Discovery answers as on the real robot: every node a topic's row lists as its
    internal publisher or subscriber is one, as /rosapi reports."""
    out = []
    rb = Rosbridge("127.0.0.1", port)
    try:
        for t in wirecheck.served(iface.get("topics")):
            for key, op in (("internal_publishers", "publishers"),
                            ("internal_subscribers", "subscribers")):
                if not t.get(key):
                    continue
                got = set(rb.call(f"/rosapi/{op}", {"topic": t["name"]}, timeout=20).get(op) or [])
                out += [f"{n} is not among the {op} of {t['name']}" for n in t[key] if n not in got]
    finally:
        rb.close()
    return out


def collect(rb, topic, seconds, **kw) -> list:
    """The messages of a topic that arrive over `seconds`."""
    got = []
    rb.subscribe(topic, lambda m: got.append(m["msg"]), **kw)
    time.sleep(seconds)
    rb.unsubscribe(topic)
    return got


def count(rb, topic, seconds, **kw) -> int:
    """How many messages of a topic arrive over `seconds`."""
    return len(collect(rb, topic, seconds, **kw))


def ainex_app_nodes(port, sim_port) -> list:
    """The recorded behaviour of the AiNex's app and sensor nodes: /app/set_running false
    stops the gait and answers success false (app_node.py answers the request's data);
    /sensor/button/enable gates /sensor/button/get_button_state (a released button reads
    true); /color_detection/enter and /face_detect/enter start their image_result (rgb8,
    labelled with the node's name)."""
    out = []
    iface = wirecheck.interface("ainex")
    ex = motion_row(iface, "walk")["command"]["example"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        rb.advertise("/walking/set_param", "ainex_interfaces/WalkingParam")
        time.sleep(0.5)
        rb.publish("/walking/set_param", ex["set_param"])
        time.sleep(0.3)
        rb.call(ex["start"]["service"], ex["start"]["request"])
        try:
            time.sleep(1.5)
            reply = rb.call("/app/set_running", {"data": False}, timeout=20)
            if reply.get("success") is not False:
                out.append(f"/app/set_running false answered {reply}, app_node answers "
                           f"success false")
            t0 = time.monotonic()
            while rb.call("/walking/is_walking", {}).get("state") and time.monotonic() - t0 < 5:
                time.sleep(0.2)
        finally:
            if rb.call("/walking/is_walking", {}).get("state"):
                out.append("/app/set_running false did not stop the gait")
                rb.call("/walking/command", {"command": "stop"}, timeout=20)
        rb.call("/sensor/button/enable", {"data": False})
        time.sleep(0.3)
        if count(rb, "/sensor/button/get_button_state", 1.0):
            out.append("/sensor/button/get_button_state still published while disabled")
        rb.call("/sensor/button/enable", {"data": True})
        states = collect(rb, "/sensor/button/get_button_state", 1.0)
        if not states:
            out.append("/sensor/button/get_button_state did not resume once enabled")
        elif any(m.get("data") is not True for m in states):
            out.append(f"/sensor/button/get_button_state reads {states[-1]} with the button "
                       f"released (the pull-up pin reads true)")
        for node in ("/color_detection", "/face_detect"):
            rb.call(f"{node}/enter", {})
            try:
                frames = collect(rb, f"{node}/image_result", 3.0, throttle_rate=1000,
                                 queue_length=1)
            finally:
                rb.call(f"{node}/exit", {})
            # the recorded frame (the node's name string, amended 2026-10-03)
            want = next(t for t in iface["topics"]
                        if t["name"] == f"{node}/image_result").get("frame_id")
            if not frames:
                out.append(f"{node}/image_result not published after {node}/enter")
            elif (frames[-1]["encoding"], frames[-1]["header"]["frame_id"]) != ("rgb8", want):
                out.append(f"{node}/image_result is {frames[-1]['encoding']} labelled "
                           f"{frames[-1]['header']['frame_id']!r}, not rgb8 {want!r}")
    finally:
        rb.close()
    return out


def rosmaster_camera_services(port, sim_port) -> list:
    """The ROSMASTER's Astra services answer as its driver does: get_color_camera_info
    gives the intrinsics /camera/rgb/camera_info publishes; toggle_color false stops
    /camera/rgb/image_raw and true restarts it; a toggle to the state the stream is in fails
    (the driver's callback returns false)."""
    out = []
    rb = Rosbridge("127.0.0.1", port)
    try:
        info = rb.call("/camera/get_color_camera_info", {}, timeout=20).get("info") or {}
        pub = _first(rb, "/camera/rgb/camera_info", timeout=10.0)
        k_srv, k_pub = info.get("K", info.get("k")), (pub or {}).get("K", (pub or {}).get("k"))
        if pub is None:
            out.append("/camera/rgb/camera_info not published")
        elif k_srv is None or [round(v, 9) for v in k_srv] != [round(v, 9) for v in k_pub]:
            out.append(f"/camera/get_color_camera_info K {k_srv} is not the published K {k_pub}")
        rb.call("/camera/toggle_color", {"data": False}, timeout=20)
        try:
            time.sleep(0.3)
            if count(rb, "/camera/rgb/image_raw", 1.0, throttle_rate=100, queue_length=1):
                out.append("/camera/rgb/image_raw still published after toggle_color false")
            try:
                rb.call("/camera/toggle_color", {"data": False}, timeout=20)
                out.append("toggle_color false on a stopped stream did not fail")
            except RuntimeError:
                pass
        finally:
            rb.call("/camera/toggle_color", {"data": True}, timeout=20)
        if not count(rb, "/camera/rgb/image_raw", 2.0, throttle_rate=100, queue_length=1):
            out.append("/camera/rgb/image_raw did not resume after toggle_color true")
        try:
            rb.call("/camera/toggle_color", {"data": True}, timeout=20)
            out.append("toggle_color true on a running stream did not fail")
        except RuntimeError:
            pass
    finally:
        rb.close()
    return out


#: Recorded command behaviours beyond the motions, per robot.
def mycobot_no_joint_feedback(port, sim_port) -> list:
    """The myCobot 280's recorded boot publishes no measured joint feedback: /joint_states
    is an input (of /robot_state_publisher and /slider_control_adaptive_gripper), so no node
    of the wire but a client's rosbridge publishes it, and /get_angles (optional) is not
    served."""
    rb = Rosbridge("127.0.0.1", port)
    try:
        pubs = rb.call("/rosapi/publishers", {"topic": "/joint_states"}, timeout=20)
        services = set(rb.call("/rosapi/services", {}, timeout=20).get("services", []))
    finally:
        rb.close()
    out = [f"/joint_states published by {n}" for n in pubs.get("publishers") or []
           if n != "/rosbridge_websocket"]
    if "/get_angles" in services:
        out.append("/get_angles (recorded optional) is served")
    return out


BEHAVIOUR_CHECKS = {"ainex": ainex_app_nodes, "rosmaster_x3_plus": rosmaster_camera_services,
                    "mycobot280": mycobot_no_joint_feedback}


def test_recorded_behaviours(robot):
    """Recorded endpoints answer with their recorded behaviour, and discovery lists every
    recorded internal publisher and subscriber."""
    iface = wirecheck.interface(robot.robot)
    problems = internal_endpoints(robot.port, iface)
    if robot.robot in BEHAVIOUR_CHECKS:
        problems += BEHAVIOUR_CHECKS[robot.robot](robot.port, robot.sim_port)
    assert problems == []


# ---------------------------------------------------------------- sensors


def _quat_mat(w, x, y, z) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _first(rb, topic, timeout=10.0, **kw):
    box = []
    rb.subscribe(topic, lambda m: box.append(m["msg"]), **kw)
    t0 = time.monotonic()
    while not box and time.monotonic() - t0 < timeout:
        time.sleep(0.1)
    rb.unsubscribe(topic)
    return box[0] if box else None


def camera_problems(rb, c, rid, cam) -> list:
    """The wire's image against the simulation's render from the same camera: colour as
    luminance, depth (16UC1, mm) where the wire reports one, IR (mono16) as the luminance
    the wire derives it from; and its CameraInfo: the recorded size and intrinsics, and a
    focal length that is the rendered camera's field of view."""
    out = []
    topic = cam["image_topic"]
    img = _first(rb, topic, throttle_rate=1000, queue_length=1)
    if img is None:
        return [f"{topic}: no image"]
    h, w = img["height"], img["width"]
    raw = base64.b64decode(img["data"])
    view = {"camera": f"{rid}/{cam['frame_id']}"}
    enc = cam["encoding"]
    if enc == "16UC1":
        mm = np.frombuffer(raw, "<u2").reshape(h, w).astype(float)
        rr = c.call("render", view=view, width=w, height=h, format="depth")
        ref = np.round(np.frombuffer(rr["_payload"], np.float32).reshape(h, w) * 1000.0)
        valid = mm > 0
        diff = np.abs(mm[valid] - ref[valid]).mean() if valid.any() else math.inf
        if not diff <= 1.0:
            out.append(f"{topic}: depth differs from the render by {diff:.2f} mm on average")
    elif enc in ("rgb8", "yuv422_yuy2", "mono16"):
        if enc == "rgb8":
            luma = np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(float).mean(axis=2)
        elif enc == "yuv422_yuy2":
            luma = np.frombuffer(raw, np.uint8).reshape(h, w * 2)[:, 0::2].astype(float)
        else:
            luma = np.frombuffer(raw, "<u2").reshape(h, w).astype(float) * 255.0 / 1023.0
        rr = c.call("render", view=view, width=w, height=h, format="rgb")
        ref = np.frombuffer(rr["_payload"], np.uint8).reshape(h, w, 3).astype(float)
        ref_luma = ref.mean(axis=2) if enc == "rgb8" else \
            0.299 * ref[:, :, 0] + 0.587 * ref[:, :, 1] + 0.114 * ref[:, :, 2]
        diff = np.abs(luma - ref_luma).mean()
        if not diff < 12.0:
            out.append(f"{topic}: differs from the render by {diff:.1f} grey levels")
    else:
        out.append(f"{topic}: no check for encoding {enc}")
    if [w, h] != [cam["width"], cam["height"]] or img.get("encoding") != enc:
        out.append(f"{topic}: {w}x{h} {img.get('encoding')}, recorded "
                   f"{cam['width']}x{cam['height']} {enc}")
    if cam.get("info_topic"):
        info = _first(rb, cam["info_topic"], throttle_rate=1000, queue_length=1)
        if info is None:
            return out + [f"{cam['info_topic']}: no CameraInfo"]
        intr = cam.get("intrinsics") or {}
        k = info.get("K") or info.get("k")
        d = list(info.get("D") or info.get("d") or [])
        want_k = [intr.get("fx", 0.0), intr.get("cx", 0.0), intr.get("fy", 0.0), intr.get("cy", 0.0)]
        if [info["width"], info["height"]] != [cam["width"], cam["height"]] or \
                [k[0], k[2], k[4], k[5]] != want_k or d != list(intr.get("d") or []) or \
                info.get("distortion_model") != (intr.get("distortion_model") or ""):
            out.append(f"{cam['info_topic']}: {info['width']}x{info['height']} K {k} D {d} "
                       f"{info.get('distortion_model')!r}, recorded {intr}")
        if k[4] > 0:   # calibrated: its focal length is the rendered camera's
            fovy = {x["name"]: x["fovy"] for x in c.call("describe", robot=rid)["cameras"]}
            implied = math.degrees(2 * math.atan(cam["height"] / 2 / k[4]))
            if abs(implied - fovy[cam["frame_id"]]) > 0.01:
                out.append(f"{cam['info_topic']}: fy gives a {implied:.4f} deg vertical field "
                           f"of view, the camera renders {fovy[cam['frame_id']]} deg")
    return out


def _lidar_tolerance(rid):
    """[(low, high, tolerance)] of the recorded lidar range figures ("lidar range, a-b m")."""
    out = []
    for t in wirecheck.interface(rid).get("tolerances") or []:
        m = re.match(r"lidar range, ([\d.]+)-([\d.]+) m", t["figure"])
        if m:
            out.append((float(m.group(1)), float(m.group(2)), t["tolerance"]))
    return out


#: A range that no recorded figure covers is reproduced exactly: the two scans compared are
#: taken at different instants of the robot at rest, so to within a millimetre.
EXACT_RANGE = {"absolute": 0.001}


def lidar_problems(rb, sim_port, rid, iface, lid) -> list:
    """The wire's scan against the simulation's own ray cast from the lidar's site, taken
    at the nearest instant: every return within the recorded range tolerance, and no
    return where the world has none or none where it has one (outside the driver's
    recorded ignore_array sectors)."""
    topic = lid["topic"]
    wire, sim = [], []
    rb.subscribe(topic, lambda m: wire.append(m["msg"]))

    def on_event(h, payload):
        if h.get("event") == "sample" and payload:
            sim.append((h["stamp"], np.frombuffer(payload, np.float32).astype(float)))
    c = protocol.Client("127.0.0.1", sim_port, on_event=on_event)
    try:
        c.call("subscribe", robot=rid, stream="lidar", rate=float(lid["scan_rate"]),
               site=lid["frame_id"].lstrip("/"), angle_min=float(lid["angle_min"]),
               angle_max=float(lid["angle_max"]), samples=int(lid["samples"]),
               range_min=float(lid["range_min"]), range_max=float(lid["range_max"]))
        time.sleep(2.0)
    finally:
        c.close()
        rb.unsubscribe(topic)
    if not wire or not sim:
        return [f"{topic}: {len(wire)} wire scans, {len(sim)} simulation scans"]
    s, stamp, ref = min(((s, *x) for s in wire for x in sim),
                        key=lambda p: abs(motion.stamp_of(p[0]) - p[1]))
    out = []
    n = int(lid["samples"])
    if len(s["ranges"]) != n:
        return [f"{topic}: {len(s['ranges'])} samples, recorded {n}"]
    if abs(s["angle_min"] - lid["angle_min"]) > 1e-4:
        out.append(f"{topic}: angle_min {s['angle_min']}")
    lo, hi = float(lid["range_min"]), float(lid["range_max"])
    got = np.array([math.inf if r is None else float(r) for r in s["ranges"]])
    ang = np.degrees(float(lid["angle_min"]) + np.arange(n) * (
        (float(lid["angle_max"]) - float(lid["angle_min"])) / max(n - 1, 1)))
    node = next(t for t in iface["topics"] if t["name"] == topic)["nodes"][0]
    ign = str(next((p.get("value") for p in iface.get("parameters") or []
                    if p["name"] == f"{node}/ignore_array"), "") or "")
    vals = [float(x) for x in ign.replace(" ", "").split(",") if x]
    ignored = np.zeros(n, bool)
    for a, b in zip(vals[0::2], vals[1::2]):
        ignored |= (ang >= a) & (ang <= b)
    have = np.isfinite(got) & (got >= lo) & (got <= hi)
    world = np.isfinite(ref) & ~ignored
    bands = _lidar_tolerance(rid)
    bad = []
    for i in np.nonzero(have | world)[0]:
        if have[i] != world[i]:
            bad.append((i, got[i], ref[i]))
            continue
        t = next((t for a, b, t in bands if a <= ref[i] <= b), EXACT_RANGE)
        if abs(got[i] - ref[i]) > max(t.get("absolute", 0.0), ref[i] * t.get("relative", 0.0)):
            bad.append((i, got[i], ref[i]))
    if bad:
        out.append(f"{topic}: {len(bad)} of {n} rays differ from the simulation's ray cast "
                   f"(index, wire, world): {bad[:5]}")
    if have.sum() < 0.3 * n:
        out.append(f"{topic}: only {int(have.sum())} valid ranges")
    return out


def imu_problems(rb, sim_port, rid, imu) -> list:
    """At rest, the IMU's mean specific force over a second is gravity in the IMU site's
    frame, and an orientation it publishes tilts the same way."""
    topic = imu["topic"]
    msgs = []
    rb.subscribe(topic, lambda m: msgs.append(m["msg"]))
    time.sleep(1.0)
    rb.unsubscribe(topic)
    if not msgs:
        return [f"{topic}: no message"]
    site = motion.readings(sim_port, rid)["sites"][imu["frame_id"].lstrip("/")]
    want = _quat_mat(*site["quat"]).T @ np.array([0.0, 0.0, G])
    acc = np.mean([[m["linear_acceleration"][k] for k in "xyz"] for m in msgs], axis=0)
    out = []
    if np.linalg.norm(acc - want) > 0.2:
        out.append(f"{topic}: specific force {acc.round(3).tolist()} at rest, gravity in its "
                   f"frame is {want.round(3).tolist()}")
    q = msgs[-1]["orientation"]
    if math.sqrt(sum(q[k] ** 2 for k in "xyzw")) > 0.5:   # all zero: no orientation
        up = _quat_mat(q["w"], q["x"], q["y"], q["z"]).T @ np.array([0.0, 0.0, 1.0])
        tilt = math.acos(max(-1.0, min(1.0, float(up @ want) / G)))
        if tilt > 0.05:
            out.append(f"{topic}: orientation tilted {tilt:.3f} rad from the simulated one")
    return out


def test_sensors(robot):
    """Cameras show what the simulation renders from the same camera, with their recorded
    CameraInfo; lidars report the simulation's ray cast; IMUs report gravity at rest."""
    rid = robot.robot
    iface = wirecheck.interface(rid)
    sensors = iface.get("sensors") or {}
    problems = []
    t0 = time.time()
    c = protocol.Client("127.0.0.1", robot.sim_port)
    rb = Rosbridge("127.0.0.1", robot.port)
    try:
        for cam in wirecheck.served(sensors.get("cameras")):
            problems += camera_problems(rb, c, rid, cam)
        for lid in wirecheck.served(sensors.get("lidars")):
            problems += lidar_problems(rb, robot.sim_port, rid, iface, lid)
        for imu in wirecheck.served(sensors.get("imus")):
            problems += imu_problems(rb, robot.sim_port, rid, imu)
    finally:
        rb.close()
        c.close()
    problems += real_time_problems(robot.sim_port, t0)
    assert problems == []


# ---------------------------------------------------------------- limits (keep last)


def so101_limits(robot, iface) -> list:
    """Each arm joint, then the gripper, commanded beyond its recorded maximum: none goes
    past it (within its recorded tolerance)."""
    port, sim_port = robot.port, robot.sim_port
    out = []
    arm = limited(motion_row(iface, "arm"))
    names = [field_joint(f) for f in arm]
    t_arm = motion.tol("so101", "arm joint")["absolute"]
    t_grip = motion.tol("so101", "gripper")["absolute"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        topic = motion_row(iface, "arm")["command"]["name"]
        rb.advertise(topic, motion_row(iface, "arm")["command"]["type"])
        time.sleep(0.8)
        for f, name in zip(arm, names):
            hi = float(f["max"])
            rb.publish(topic, trajectory(names, [motion.beyond(hi) if n == name else 0.0
                                                 for n in names]))
            peak = motion.peak_joint(sim_port, "so101", name, 3.0)
            if peak > hi + t_arm:
                out.append(f"{name} reached {peak:.3f} rad, recorded maximum {hi}")
        rb.publish(topic, trajectory(names, [0.0] * len(names)))
        motion.wait_joints(sim_port, "so101", dict.fromkeys(names, 0.0), t_arm, 8)
        grip = motion_row(iface, "gripper")
        f = limited(grip)[0]
        hi = float(f["max"])
        try:
            rb.action(grip["command"]["name"], grip["command"]["type"],
                      {"command": {"name": ["gripper_joint"], "position": [motion.beyond(hi)],
                                   "velocity": [], "effort": []}}, timeout=20)
        except TimeoutError:
            pass   # a goal past the jaw's travel may stall; the position is what counts
        peak = motion.peak_joint(sim_port, "so101", "gripper_joint", 2.0)
        if peak > hi + t_grip:
            out.append(f"gripper_joint reached {peak:.3f} rad, recorded maximum {hi}")
    finally:
        rb.close()
    return out


def rosmaster_limits(robot, iface) -> list:
    """An ArmJoint array with one servo beyond its recorded maximum: the library ignores
    the whole array, as recorded, so no joint moves."""
    out = []
    t_arm = motion.tol("rosmaster_x3_plus", "arm joint")["absolute"]
    rb = Rosbridge("127.0.0.1", robot.port)
    try:
        rb.advertise("/TargetAngle", "yahboomcar_msgs/ArmJoint")
        time.sleep(0.8)
        for row in (motion_row(iface, "arm"), motion_row(iface, "gripper")):
            for f in limited(row):
                m = re.match(r"joints\[(\d+)(?:\.\.(\d+))?\]", f["field"])
                if m is None:
                    continue   # the single-servo form and run_time
                for i in range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1):
                    servo = [90] * 6
                    servo[i] = int(motion.beyond(float(f["max"])))
                    before = motion.joints(robot.sim_port, "rosmaster_x3_plus")
                    rb.publish("/TargetAngle", {"id": 0, "angle": 0.0, "joints": servo,
                                                "run_time": 500})
                    time.sleep(1.5)
                    after = motion.joints(robot.sim_port, "rosmaster_x3_plus")
                    moved = {n: abs(after[n] - before[n]) for n in ROSMASTER_SERVOS}
                    if max(moved.values()) > t_arm:
                        out.append(f"joints {servo} (servo {i + 1} past its maximum "
                                   f"{f['max']}) moved the arm: {moved}")
    finally:
        rb.close()
    return out


def ainex_limits(robot, iface) -> list:
    """Each head joint commanded beyond its recorded maximum stays at it (the board clamps
    the servo pulse); each limited walking parameter set beyond its maximum is clamped by
    the controller, as /walking/get_param reports (the robot does not walk), except one
    whose row records that the controller does not clamp it (dsp_ratio)."""
    port, sim_port = robot.port, robot.sim_port
    out = []
    t_head = motion.tol("ainex", "head joint")["absolute"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        for row in [m for m in iface["motions"] if m["id"] == "head"]:
            topic = row["command"]["name"]
            joint = topic.strip("/").split("_controller")[0]
            hi = float(next(f for f in limited(row) if f["field"] == "position")["max"])
            rb.advertise(topic, row["command"]["type"])
            time.sleep(0.8)
            rb.publish(topic, {"position": motion.beyond(hi), "duration": 0.5})
            peak = motion.peak_joint(sim_port, "ainex", joint, 2.0)
            if peak > hi + t_head:
                out.append(f"{joint} reached {peak:.3f} rad, recorded maximum {hi}")
            rb.publish(topic, {"position": 0.0, "duration": 0.5})
            time.sleep(1.0)
        walk = motion_row(iface, "walk")
        prm = rb.call("/walking/get_param", {})["parameters"]
        rb.advertise(walk["command"]["name"], walk["command"]["type"])
        time.sleep(0.5)
        for f in limited(walk):
            if "not clamp" in (f.get("notes") or ""):
                continue        # a range the controller passes through (recorded so)
            hi = float(f["max"])
            rb.publish(walk["command"]["name"], dict(prm, **{f["field"]: motion.beyond(hi)}))
            time.sleep(0.3)
            got = rb.call("/walking/get_param", {})["parameters"][f["field"]]
            if got > hi + 1e-6:
                out.append(f"walking {f['field']} set to {motion.beyond(hi)} is {got}, "
                           f"recorded maximum {hi}")
        rb.publish(walk["command"]["name"], prm)
    finally:
        rb.close()
    return out


def mycobot_limits(robot, iface) -> list:
    """An arm angle beyond its recorded maximum: as recorded, pymycobot raises in the
    node's callback and the node ends, so the spawn reports the lost node and ends (the
    first such command ends the robot, so one field is exercised)."""
    arm = motion_row(iface, "arm")
    fields = limited(arm) + limited(motion_row(iface, "gripper"))
    names = arm["command"]["example"]["name"]
    now = motion.joints(robot.sim_port, robot.robot)
    pos = [min(max(now[field_joint(f)], float(f["min"])), float(f["max"])) for f in fields]
    pos[0] = motion.beyond(float(fields[0]["max"]))
    rb = Rosbridge("127.0.0.1", robot.port)
    try:
        rb.advertise(arm["command"]["name"], arm["command"]["type"])
        time.sleep(0.8)
        robot.ended_by_test = True
        rb.publish(arm["command"]["name"], {"header": {"stamp": {"sec": 0, "nanosec": 0},
                                                       "frame_id": ""},
                                            "name": names, "position": pos, "velocity": [],
                                            "effort": []})
    finally:
        rb.close()
    try:
        code = robot.wait_exit(90)
    except subprocess.TimeoutExpired:
        code = None
    if not code or "lost node" not in robot.text():
        return [f"{field_joint(fields[0])} at {pos[0]:.3f} rad, past its maximum: the spawn "
                f"ended with {code}, expected its node lost:\n{robot.text()[-1500:]}"]
    return []


#: How each robot's recorded command limits are exercised (drives through
#: motion.drive_limits); a robot with limited command fields and no check fails.
LIMIT_CHECKS = {"so101": so101_limits, "rosmaster_x3_plus": rosmaster_limits,
                "ainex": ainex_limits, "mycobot280": mycobot_limits}


def test_limits(robot):
    """Recorded commands beyond their recorded limits move no further than the limits, or
    answer as the record says (clamped, ignored, or ending the node). Kept last: a robot
    whose recorded answer is its node ending is gone afterwards."""
    rid = robot.robot
    iface = wirecheck.interface(rid)
    rows = [m for m in iface.get("motions") or [] if limited(m)]
    problems = []
    t0 = time.time()
    if any(m["id"] == "drive" for m in rows):
        problems += motion.drive_limits(robot.port, rid)
    if any(m["id"] != "drive" for m in rows):
        if rid in LIMIT_CHECKS:
            problems += LIMIT_CHECKS[rid](robot, iface)
        else:
            problems.append(f"no check of the limits of {[m['id'] for m in rows]}")
    problems += real_time_problems(robot.sim_port, t0)
    assert problems == []
