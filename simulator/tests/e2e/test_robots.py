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
    yield sp
    assert sp.stop() == 0
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
    commands): every recorded action server is up."""
    iface = wirecheck.interface(rid)
    if iface["dialect"] != "ros2":
        return []
    actions = wirecheck.served(iface.get("actions"))
    out = []
    try:
        rb = Rosbridge("127.0.0.1", port)
    except OSError as exc:
        return [f"no rosbridge at readiness: {exc}"]
    try:
        up = set(rb.call("/rosapi/action_servers", {}, timeout=30).get("action_servers", []))
        out += [f"action {a['name']} not served at readiness" for a in actions
                if a["name"] not in up]
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
    """The joint a command field names in parentheses: `positions[0] (joint1)`."""
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


def test_the_repository_reaches_the_wire_read_only(robot):
    """Spec §2.4: simulator code reaches a container through a read-only mount, never a
    copy -- the wire's container mounts the simulator and the robot specification
    read-only, and nothing else of the host."""
    import json
    import subprocess

    from conftest import REPO, SIM

    info = json.loads(subprocess.run(["docker", "inspect", container(robot)],
                                     stdout=subprocess.PIPE, text=True, check=True).stdout)[0]
    mounts = {m["Destination"]: m for m in info["Mounts"]}
    want = {"/opt/rsim/simulator": SIM, "/opt/rsim/robots_specs": REPO / "robots_specs"}
    assert set(mounts) == set(want), mounts
    for dest, src in want.items():
        m = mounts[dest]
        assert m["Type"] == "bind" and m["RW"] is False and "ro" in m["Mode"].split(","), m
        assert m["Source"] == str(src), (m["Source"], src)


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
    "rebot_b601_rs": {"arm joint final position": "test_motion",
                      "gripper motor final position": "test_motion",
                      "end-effector repeatability": "test_motion",
                      "camera and lidar mount poses": "here",
                      "camera horizontal field of view": "here"},
}


#: "Exactly", for a figure the record gives no tolerance (spec §5 Acceptance bounds): a
#: sensor's mount to within the precision a model file stores its poses with
EXACT_M = 1e-6
EXACT_RAD = 1e-6


def rpy_matrix(roll, pitch, yaw) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw (Rz Ry Rx)."""
    x, y, z = (_quat_mat(math.cos(a / 2), *(math.sin(a / 2) * e))
               for a, e in ((roll, np.eye(3)[0]), (pitch, np.eye(3)[1]), (yaw, np.eye(3)[2])))
    return z @ y @ x


#: a MuJoCo camera looks along its -z, x right, y up; in the frame a record names it by,
#: that is (columns) -y, z, -x of a link frame (x forward, z up: URDF's camera_link) or
#: x, -y, -z of an optical frame (REP 103 `_optical`: z forward, x right, y down)
CAMERA_AXES = {"link": np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
               "optical": np.diag([1.0, -1.0, -1.0])}


def mount_poses(robot, iface, row=None) -> list:
    """Every served camera, lidar and IMU of the compiled model (at its home pose) sits at
    its recorded mount relative to the mount's parent link, position and orientation: the
    position within the recorded "camera and lidar mount poses" tolerance where the record
    gives one (cameras and lidars), every other figure exactly (`EXACT_M`, `EXACT_RAD`)."""
    import mujoco

    import robot_model

    m = mujoco.MjModel.from_xml_path(str(robot_model.model_file(registry.get(robot.robot))))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    sensors = iface.get("sensors") or {}
    out = []
    for kind, obj, rows in (("camera", mujoco.mjtObj.mjOBJ_CAMERA, sensors.get("cameras")),
                            ("lidar", mujoco.mjtObj.mjOBJ_SITE, sensors.get("lidars")),
                            ("imu", mujoco.mjtObj.mjOBJ_SITE, sensors.get("imus"))):
        lim = row["tolerance"]["absolute"] if row is not None and kind != "imu" else EXACT_M
        for s in wirecheck.served(rows):
            mount = s["mount"]
            name = s["frame_id"].lstrip("/")
            i = mujoco.mj_name2id(m, obj, name)
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mount["parent"].lstrip("/"))
            if i < 0 or b < 0:
                out.append(f"{kind} {s['frame_id']} or its mount parent {mount['parent']} is "
                           "not on the model")
                continue
            if kind == "camera":
                pos = d.cam_xpos[i]
                rot = d.cam_xmat[i].reshape(3, 3) @ \
                    CAMERA_AXES["optical" if "optical" in name else "link"].T
            else:
                pos, rot = d.site_xpos[i], d.site_xmat[i].reshape(3, 3)
            parent = d.xmat[b].reshape(3, 3)
            local = parent.T @ (pos - d.xpos[b])
            err = float(np.linalg.norm(local - np.asarray(mount["xyz"], float)))
            if err > lim:
                out.append(f"{kind} {s['frame_id']} is {err * 1000:.3f} mm from its recorded "
                           f"mount (allowed {lim * 1000:.3f} mm)")
            want = rpy_matrix(*[float(v) for v in mount["rpy"]])
            cos = (np.trace(want.T @ (parent.T @ rot)) - 1) / 2
            ang = math.acos(max(-1.0, min(1.0, cos)))
            if ang > EXACT_RAD:
                out.append(f"{kind} {s['frame_id']} is turned {ang:.2e} rad from its recorded "
                           f"mount rpy {mount['rpy']}")
    return out


def stated(x: float) -> int:
    """The decimals a recorded figure is stated with (53.98: 2)."""
    text = repr(float(x))
    return len(text.split(".")[1].rstrip("0")) if "." in text and "e" not in text else 0


def fields_of_view(robot, iface, row=None) -> list:
    """Each served camera's horizontal field of view, from the rendered camera's fovy and
    from the wire's CameraInfo (where it carries a calibration): within the recorded
    "camera horizontal field of view" tolerance where the record gives one, else exactly
    -- the figure as the record states it (rounded to its decimals)."""
    cams = [c for c in wirecheck.served((iface.get("sensors") or {}).get("cameras"))
            if c.get("hfov_deg") is not None]
    if not cams:
        return []
    c = protocol.Client("127.0.0.1", robot.sim_port)
    try:
        fovy = {x["name"]: x["fovy"] for x in c.call("describe", robot=robot.robot)["cameras"]}
    finally:
        c.close()
    # the images with their CameraInfo: a lazy driver (the Astra's) publishes the CameraInfo
    # only while an image or point cloud of it has a subscriber
    infos = wirecheck.sample(robot.port, [x["info_topic"] for x in cams if x.get("info_topic")]
                             + [x["image_topic"] for x in cams])
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
        want = float(cam["hfov_deg"])
        for src, hfov in got.items():
            if row is not None:
                bad = abs(hfov - want) > row["tolerance"]["absolute"]
            else:
                bad = round(hfov, stated(want)) != round(want, stated(want))
            if bad:
                out.append(f"{cam['image_topic']}: {src} hfov {hfov:.4f} deg, recorded "
                           f"{cam['hfov_deg']}" + ("" if row is not None else " (no tolerance "
                                                    "recorded: exactly)"))
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


FIGURE_CHECKS = {"standing height": standing_height}
#: the sensor figures judged for every served sensor of every robot, with the recorded
#: tolerance row where there is one and exactly where there is none
SENSOR_CHECKS = {"camera and lidar mount poses": mount_poses,
                 "camera horizontal field of view": fields_of_view}


def test_recorded_figures(robot):
    """Every recorded tolerance is judged by some check (`JUDGED_BY`); the figures no other
    check judges are judged here, before any motion (the robot in its init pose). Every
    served sensor's mount (position and orientation) and every camera's field of view are
    judged here for every robot (spec §5 Sensors: the recorded sensor poses, calibration
    and characteristics), within the recorded tolerance or, with none, exactly."""
    iface = wirecheck.interface(robot.robot)
    judged = JUDGED_BY.get(robot.robot, {})
    problems = []
    rows = {}
    for row in iface.get("tolerances") or []:
        hits = [f for f in judged if f in row["figure"]]
        if len(hits) != 1:
            problems.append(f"no check judges the recorded figure {row['figure']!r}")
        elif hits[0] in SENSOR_CHECKS:
            rows[hits[0]] = row
        elif judged[hits[0]] == "here":
            problems += FIGURE_CHECKS[hits[0]](robot, iface, row)
    for figure, check in SENSOR_CHECKS.items():
        problems += check(robot, iface, rows.get(figure))
    assert problems == []


# ---------------------------------------------------------------- the interface


def test_contract(robot):
    snap = wirecheck.snapshot(robot.port, robot.robot)
    assert wirecheck.contract_problems(robot.robot, snap) == []


def test_frames_and_message_fields(robot):
    """Each recorded output topic with a frame (and a child frame) carries it, and a
    message of every periodic and every latched output topic matches its declared type
    field for field, through every nested message (spec §5 Contracts). Event outputs,
    service responses and action results the other checks receive are judged the same way
    where they are received (`typed`)."""
    iface = wirecheck.interface(robot.robot)
    periodic = wirecheck.periodic_topics(robot.robot)
    rows = [t for t in wirecheck.served(iface["topics"]) if t.get("direction") == "out"
            and (t["name"] in periodic or t.get("latched"))]
    got = wirecheck.sample(robot.port, [t["name"] for t in rows])
    problems = []
    rb = Rosbridge("127.0.0.1", robot.port)
    try:
        for t in rows:
            msg = got.get(t["name"])
            if msg is None:
                problems.append(f"{t['name']}: no message")
                continue
            problems += wirecheck.typed_problems(rb, "message", t["type"], msg, t["name"])
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


def typed(rb, rid, kind, name, value) -> list:
    """`wirecheck.typed_problems` of what a check received from robot `rid`: a message of
    the recorded topic `name` (`kind` "message"), the response of the recorded service
    `name` ("service_response") or the result of the recorded action `name`
    ("action_result"), against the type the record declares for it."""
    iface = wirecheck.interface(rid)
    section = {"message": "topics", "service_response": "services",
               "action_result": "actions"}[kind]
    typ = next(r["type"] for r in iface.get(section) or [] if r["name"] == name)
    if kind == "action_result":
        value = value.get("values") or {}
    return wirecheck.typed_problems(rb, kind, typ, value, f"{name} ({kind.replace('_', ' ')})")


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
    # point clouds (megabytes a message at camera rates) in a window of their own: every
    # topic is still measured whole, at its rate and against every bound. Where the tail
    # gaps come from (ROSMASTER, both windows together, measured): two probes of one topic
    # see each late message equally late, so the delay is never a probe's own; the late
    # frames' stamps are on time while they reach the wire late (up to 1.66 s, every topic of
    # that frame alike), and those intervals are the ones whose real-time factor falls
    # (0.80-0.91): the simulation, sharing the host's CPUs with the Docker VM that carries
    # some 400 MB/s of probed messages with every cloud probed too, renders and sends late.
    # The cloud window takes 290 MB/s of that load off the other topics' window; the gaps
    # are not loosened, and the same flakiness shows on the reBot at HEAD.
    types = {t["name"]: t["type"] for t in iface["topics"]}
    clouds = {t: r for t, r in rates.items() if types[t].endswith("PointCloud2")}
    for group in ({t: r for t, r in rates.items() if t not in clouds}, clouds):
        if not group:
            continue
        window = max(6.0, 5.5 / min(group.values()))
        t0 = time.time()
        m = wirecheck.measure_on_graph(container(robot), robot.robot, list(group), window)
        problems += wirecheck.rate_problems(m, group, wirecheck.multi_publisher(iface))
        problems += real_time_problems(robot.sim_port, t0)
    # each transform the record gives a rate of its own, on its own (spec §5 Acceptance
    # bounds), not only /tf as a whole
    edges = wirecheck.tf_rates(iface)
    if edges:
        window = max(6.0, 5.5 / min(edges.values()))
        t0 = time.time()
        m = wirecheck.measure_tf_on_graph(container(robot), robot.robot, window)
        problems += wirecheck.tf_rate_problems(m, iface)
        problems += real_time_problems(robot.sim_port, t0)
    assert problems == []


# ---------------------------------------------------------------- motions


REBOT = "rebot_b601_rs"
REBOT_ARM = [f"joint{i}" for i in range(1, 7)]
#: action_msgs/GoalStatus of a goal's result, as rosbridge reports it (`status`)
SUCCEEDED, ABORTED = 4, 6
REBOT_FINGERS = ["gripper_joint1", "gripper_joint2"]
#: how long after a goal the reBot is watched still holding it: the record has no command
#: watchdog (motions[arm|gripper].watchdog.present false), so nothing drops the hold
HOLD_S = 5.0


def rebot_motor(joints: dict) -> float:
    """The gripper motor angle (rad) of the model's fingers: its tendon, 62.5 (q1 + q2)."""
    return 62.5 * (joints["gripper_joint1"] + joints["gripper_joint2"])


def rebot_goal(positions, seconds) -> dict:
    """A FollowJointTrajectory goal of one point, as the recorded example has it."""
    return {"trajectory": trajectory(REBOT_ARM, positions, seconds)}


def rebot_follow(rb, positions, seconds) -> dict:
    return rb.action("/rebotarm/follow_joint_trajectory",
                     "control_msgs/action/FollowJointTrajectory",
                     rebot_goal(positions, seconds), timeout=seconds + 30)


def rebot_arm(port, sim_port):
    """The recorded examples: the arm trajectory (Seeed's ready point with joint1 at 0.3,
    4 s) succeeds and the joints settle at its point within the recorded tolerance, judged
    by the recorded feedback (/rebotarm/joint_states) and the simulation's joints, the
    gripper_end returning there to within the recorded repeatability after a round trip to
    the zero pose; then the gripper goal (5.0 rad, fully open) is reached, the result and
    /rebotarm/joint_states reporting it as recorded, and it closes again."""
    out = []
    iface = wirecheck.interface(REBOT)
    ex = motion_row(iface, "arm")["command"]["example"]["trajectory"]
    point = ex["points"][-1]
    target = [float(v) for v in point["positions"]]
    seconds = float(point["time_from_start"]["sec"]) + point["time_from_start"]["nanosec"] * 1e-9
    t_arm = motion.tol(REBOT, "arm joint final position")["absolute"]
    t_grip = motion.tol(REBOT, "gripper motor final position")["absolute"]
    t_rep = motion.tol(REBOT, "end-effector repeatability")["absolute"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        res = rb.action("/rebotarm/follow_joint_trajectory",
                        "control_msgs/action/FollowJointTrajectory", {"trajectory": ex},
                        timeout=seconds + 30)
        if res.get("status") != SUCCEEDED or (res.get("values") or {}).get("error_code") != 0:
            out.append(f"arm trajectory result: {res}")
        out += typed(rb, REBOT, "action_result", "/rebotarm/follow_joint_trajectory", res)
        err = motion.wait_joints(sim_port, REBOT, dict(zip(REBOT_ARM, target)), t_arm, 4)
        if max(err.values()) > t_arm:
            out.append(f"arm joints off the trajectory's point: {err}")
        if not motion.at_rest(sim_port, REBOT, REBOT_ARM):
            out.append("arm not at rest after its trajectory")
        # no command watchdog: HOLD_S on, the arm still holds the trajectory's point
        time.sleep(HOLD_S)
        j = motion.joints(sim_port, REBOT)
        drift = {n: round(j[n] - t, 4) for n, t in zip(REBOT_ARM, target)
                 if abs(j[n] - t) > t_arm}
        if drift:
            out.append(f"{HOLD_S:.0f} s after its goal the arm no longer holds the "
                       f"trajectory's point: {drift}")
        js = _first(rb, "/rebotarm/joint_states")
        fed = dict(zip(js["name"], js["position"])) if js else {}
        bad = {n: round(fed.get(n, math.inf) - t, 4) for n, t in zip(REBOT_ARM, target)
               if not abs(fed.get(n, math.inf) - t) <= t_arm}
        if bad:
            out.append(f"/rebotarm/joint_states off the trajectory's point: {bad}")
        before = np.asarray(motion.readings(sim_port, REBOT)["sites"]["gripper_end"]["pos"])
        rebot_follow(rb, [0.0] * 6, 3)
        rebot_follow(rb, target, 3)
        motion.at_rest(sim_port, REBOT, REBOT_ARM)
        time.sleep(1.0)
        again = np.asarray(motion.readings(sim_port, REBOT)["sites"]["gripper_end"]["pos"])
        if np.linalg.norm(again - before) > t_rep:
            out.append(f"gripper_end came back {np.linalg.norm(again - before) * 1000:.2f} mm "
                       f"from where it was, recorded repeatability {t_rep * 1000:.1f} mm")
        # the gripper: open fully, the recorded example
        grip = motion_row(iface, "gripper")["command"]
        res = rb.action(grip["name"], grip["type"], grip["example"], timeout=20)
        want = float(grip["example"]["command"]["position"])
        values = res.get("values") or {}
        if res.get("status") != SUCCEEDED or values.get("reached_goal") is not True or \
                not abs(float(values.get("position", math.inf)) - want) < t_grip:
            out.append(f"gripper command result: {res}")
        out += typed(rb, REBOT, "action_result", grip["name"], res)
        motor = rebot_motor(motion.joints(sim_port, REBOT))
        if abs(motor - want) > t_grip:
            out.append(f"gripper motor at {motor:.3f} rad, commanded {want}")
        # no command watchdog: HOLD_S on, the gripper still holds its goal
        time.sleep(HOLD_S)
        motor = rebot_motor(motion.joints(sim_port, REBOT))
        if abs(motor - want) > t_grip:
            out.append(f"{HOLD_S:.0f} s after its goal the gripper motor is at {motor:.3f} "
                       f"rad, commanded {want}")
        js = _first(rb, "/rebotarm/joint_states")
        fed = dict(zip(js["name"], js["position"])) if js else {}
        ratio = min(max(want / 5.0, 0.0), 1.0)
        for n, limit, t in (("gripper_joint1", 0.05, 0.0012), ("gripper_joint2", 0.0715, 0.0017)):
            if not abs(fed.get(n, math.inf) - ratio * limit) <= t:
                out.append(f"/rebotarm/joint_states {n} {fed.get(n)}, recorded "
                           f"{ratio * limit:.4f} m at {want} motor rad")
        if not motion.at_rest(sim_port, REBOT, REBOT_FINGERS):
            out.append("gripper not at rest after its goal")
        rb.action(grip["name"], grip["type"], {"command": {"position": 0.0, "max_effort": 0.0}},
                  timeout=20)
    finally:
        rb.close()
    return out


def trajectory(names, positions, seconds=2):
    return {"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}, "joint_names": names,
            "points": [{"positions": list(positions), "velocities": [], "accelerations": [],
                        "effort": [], "time_from_start": {"sec": seconds, "nanosec": 0}}]}


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
        reply = rb.call("/CurrentAngle", {"apply": ""}, timeout=20)
        out += typed(rb, "rosmaster_x3_plus", "service_response", "/CurrentAngle", reply)
        angles = reply.get("angles") or []
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
        out += typed(rb, "ainex", "service_response",
                     "/ros_robot_controller/bus_servo/get_position", reply)
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
        time.sleep(3.5)
        # no command watchdog (motions[walk].watchdog.present false): 3.5 s after the one
        # start the gait still runs and the legs still move
        state = rb.call("/walking/is_walking", {}, timeout=20)
        out += typed(rb, "ainex", "service_response", "/walking/is_walking", state)
        a = motion.joints(sim_port, "ainex")
        time.sleep(0.2)
        b = motion.joints(sim_port, "ainex")
        legs = max(abs(b[k] - a[k]) for k in a if k[2:].startswith(("hip", "knee", "ank")))
        if state.get("state") is not True or legs < 0.01:
            out.append(f"walk: 3.5 s after its start the gait has stopped (is_walking "
                       f"{state.get('state')}, legs moved {legs:.3f} rad in 0.2 s)")
        time.sleep(max(0.0, 4.0 - (time.monotonic() - t0)))
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


def _up(q):
    """The top body's z axis in the world, from its (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


#: the most the AiNex's top body may tilt, end to end, over a group that leaves it standing
#: (a fall tilts it by far more; the bows of the estimated groups tilt it ~0.3 rad only while
#: they play)
STANDING_TILT_RAD = 0.05


def ainex_action_group(port, sim_port):
    """The recorded vendor example, `left_shot`: no pinned source holds its .d6a, so it plays
    the record's stand-in under that name, a small kick on one foot --
    its legs move, then the robot stands where it stood, upright, at its standing height,
    every joint back at the init pose. Then the record's smoke example, one of its own
    estimated groups (robots_specs/ainex/action_groups.yml) -- it moves the right arm and ends
    back at the init pose."""
    out = []
    command = motion_row(wirecheck.interface("ainex"), "action_group")["command"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        r0 = motion.readings(sim_port, "ainex")
        j0 = {k: v["position"] for k, v in r0["joints"].items()}
        rb.advertise("/app/set_action", "std_msgs/String")
        time.sleep(0.5)
        t_ag = motion.tol("ainex", "action group")["absolute"]
        t_h = motion.tol("ainex", "standing height")["absolute"]
        example = command["example"]
        legs = [k for k in j0 if k[2:].startswith(("hip", "knee", "ank"))]
        rb.publish("/app/set_action", example)
        moved = 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 6:
            j = motion.joints(sim_port, "ainex")
            moved = max(moved, max(abs(j[k] - j0[k]) for k in legs))
            time.sleep(0.05)
        time.sleep(1.0)
        r1 = motion.readings(sim_port, "ainex")
        j1 = {k: v["position"] for k, v in r1["joints"].items()}
        if moved < 0.2:
            out.append(f"action group {example['data']!r}: the legs moved only {moved:.2f} rad")
        off = {k: round(abs(j1[k] - j0[k]), 3) for k in j0 if abs(j1[k] - j0[k]) > t_ag}
        if off:
            out.append(f"action group {example['data']!r}: not back at the init pose {off}")
        b0, b1 = r0["base"], r1["base"]
        tilt = math.acos(min(1.0, float(_up(b0["quat"]) @ _up(b1["quat"]))))
        dz = b1["pos"][2] - b0["pos"][2]
        dxy = math.hypot(b1["pos"][0] - b0["pos"][0], b1["pos"][1] - b0["pos"][1])
        if abs(dz) > t_h or tilt > STANDING_TILT_RAD or dxy > 0.02:
            out.append(f"action group {example['data']!r}: not standing where it stood "
                       f"(height {dz:+.4f} m, tilt {tilt:.3f} rad, moved {dxy:.4f} m)")
        j0 = j1
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
    "myagv": {("drive",): lambda port, sim_port: motion.drive(port, "myagv", sim_port)},
    "rosmaster_x3_plus": {
        ("drive",): lambda port, sim_port: motion.drive(port, "rosmaster_x3_plus",
                                                     sim_port),
        ("arm", "gripper"): rosmaster_arm},
    REBOT: {("arm", "gripper"): rebot_arm},
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
            out += typed(rb, "ainex", "service_response", "/app/set_running", reply)
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
        for m in states[-1:]:
            out += typed(rb, "ainex", "message", "/sensor/button/get_button_state", m)
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
            # the recorded frame (the node's name string)
            want = next(t for t in iface["topics"]
                        if t["name"] == f"{node}/image_result").get("frame_id")
            for m in frames[-1:]:
                out += typed(rb, "ainex", "message", f"{node}/image_result", m)
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
        reply = rb.call("/camera/get_color_camera_info", {}, timeout=20)
        out += typed(rb, "rosmaster_x3_plus", "service_response",
                     "/camera/get_color_camera_info", reply)
        info = reply.get("info") or {}
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
#: a calibration no recorded camera has, sent with set_camera_info
SENT_CALIBRATION = {"distortion_model": "plumb_bob", "D": [0.01, -0.02, 0.001, 0.002, 0.0],
                    "K": [600.5, 0.0, 321.25, 0.0, 601.5, 238.75, 0.0, 0.0, 1.0],
                    "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "P": [600.5, 0.0, 321.25, 0.0, 0.0, 601.5, 238.75, 0.0, 0.0, 0.0, 1.0, 0.0]}
CALIBRATION_FIELDS = ("width", "height", "distortion_model", "D", "K", "R", "P", "binning_x",
                      "binning_y", "roi")


def usb_cam_calibration(rid, image_topic=None):
    """usb_cam's set_camera_info (camera_info_manager): it answers success, and the
    calibration it was sent is the CameraInfo the node publishes from then on, field for
    field, its header the frame's; sending the recorded calibration back restores it.
    `image_topic`: the usb_cam camera's, where the robot has other cameras."""

    def check(port, sim_port) -> list:
        iface = wirecheck.interface(rid)
        cam = next(c for c in wirecheck.served(iface["sensors"]["cameras"])
                   if c.get("info_topic") and image_topic in (None, c["image_topic"]))
        node = next(t for t in iface["topics"] if t["name"] == cam["image_topic"])["nodes"][0]
        service = f"{node}/set_camera_info"
        out = []
        rb = Rosbridge("127.0.0.1", port)
        try:
            before = _first(rb, cam["info_topic"], throttle_rate=200, queue_length=1)
            if before is None:
                return [f"{cam['info_topic']}: no CameraInfo"]
            for sent in (dict(before, **SENT_CALIBRATION), before):
                reply = rb.call(service, {"camera_info": sent}, timeout=20)
                out += typed(rb, rid, "service_response", service, reply)
                if reply.get("success") is not True:
                    out.append(f"{service} answered {reply}, camera_info_manager answers "
                               "success once it stored the calibration")
                time.sleep(0.5)
                got = _first(rb, cam["info_topic"], throttle_rate=200, queue_length=1) or {}
                off = [f for f in CALIBRATION_FIELDS if got.get(f) != sent.get(f)]
                if off:
                    out.append(f"{cam['info_topic']} after {service}: {off} are not the "
                               f"calibration sent ({ {f: got.get(f) for f in off} })")
                if (got.get("header") or {}).get("frame_id") != cam["frame_id"]:
                    out.append(f"{cam['info_topic']} after {service}: frame "
                               f"{(got.get('header') or {}).get('frame_id')!r}, recorded "
                               f"{cam['frame_id']!r}")
        finally:
            rb.close()
        return out

    return check


def usb_cam_compressed(port, sim_port) -> list:
    """The ROSMASTER arm camera's image_transport compressed plugin: each frame as
    compressed_image_transport encodes it ("rgb8; jpeg compressed bgr8", a JPEG) and, after
    set_parameters format png, as PNG; restored to its recorded jpeg afterwards."""
    out = []
    topic = "/usb_cam/image_raw/compressed"
    rb = Rosbridge("127.0.0.1", port)
    try:
        for fmt, magic in (("jpeg", b"\xff\xd8"), ("png", b"\x89PNG"), ("jpeg", b"\xff\xd8")):
            reply = rb.call(f"{topic}/set_parameters",
                            {"config": {"strs": [{"name": "format", "value": fmt}]}}, timeout=20)
            out += typed(rb, "rosmaster_x3_plus", "service_response", f"{topic}/set_parameters",
                         reply)
            # the frames of the next second, judged by the last: rosbridge may hand a new
            # subscription the message its own subscriber last held, encoded before the change
            box = []
            rb.subscribe(topic, lambda m: box.append(m["msg"]), queue_length=1)
            time.sleep(1.0)
            rb.unsubscribe(topic)
            msg = box[-1] if box else None
            if msg is None:
                out.append(f"{topic}: no message")
                continue
            data = base64.b64decode(msg["data"])
            want = f"rgb8; {fmt} compressed bgr8"
            if msg.get("format") != want or not data.startswith(magic):
                out.append(f"{topic} with format {fmt}: {msg.get('format')!r} {data[:4]!r}, "
                           f"want {want!r}")
            if (msg.get("header") or {}).get("frame_id") != "usb_cam":
                out.append(f"{topic}: frame {(msg.get('header') or {}).get('frame_id')!r}")
    finally:
        rb.close()
    return out


def usb_cam_theora_and_depth(port, sim_port) -> list:
    """The arm camera's theora plugin: a new subscriber's first packets are the stream's three
    header packets (b_o_s on the first, "\\x80theora", "\\x81theora", "\\x82theora"), then
    video packets, all with the frame's header; its compressedDepth plugin, subscribed,
    publishes nothing for the rgb8 frames and logs the plugin's error on /rosout."""
    out = []
    rb = Rosbridge("127.0.0.1", port)
    got, logs, depth = [], [], []
    try:
        rb.subscribe("/usb_cam/image_raw/theora", lambda m: got.append(m["msg"]), queue_length=50)
        t0 = time.monotonic()
        while len(got) < 5 and time.monotonic() - t0 < 10:
            time.sleep(0.1)
        rb.unsubscribe("/usb_cam/image_raw/theora")
        heads = [base64.b64decode(m["data"])[:7] for m in got[:3]]
        if heads != [b"\x80theora", b"\x81theora", b"\x82theora"] or got[0].get("b_o_s") != 1:
            out.append(f"/usb_cam/image_raw/theora: first packets {heads}, b_o_s "
                       f"{got[0].get('b_o_s') if got else None}")
        if len(got) < 4 or got[3].get("packetno") != 3:
            out.append(f"/usb_cam/image_raw/theora: {len(got)} packets, no video packet after "
                       "the headers")
        if any((m.get("header") or {}).get("frame_id") != "usb_cam" for m in got):
            out.append("/usb_cam/image_raw/theora: a packet not in frame usb_cam")
        rb.subscribe("/rosout", lambda m: logs.append(m["msg"]), queue_length=100)
        rb.subscribe("/usb_cam/image_raw/compressedDepth", lambda m: depth.append(m["msg"]))
        time.sleep(1.5)
        rb.unsubscribe("/usb_cam/image_raw/compressedDepth")
        rb.unsubscribe("/rosout")
        if depth:
            out.append(f"/usb_cam/image_raw/compressedDepth: {len(depth)} messages for rgb8")
        if not any(m.get("name") == "/usb_cam" and m.get("level") == 8 and
                   "(input format is: rgb8)" in (m.get("msg") or "") for m in logs):
            out.append("/usb_cam: no compressedDepth error on /rosout")
    finally:
        rb.close()
    return out


def usb_cam_first_frame(port, sim_port) -> list:
    """A new subscriber's first /usb_cam/image_raw, on the wire's own graph
    (simulator/tests/first_frame_probe_ros1.py), five subscribers in turn, none left out:
    within five recorded periods (167 ms). The real usb_cam, capturing continuously, sends
    the next captured frame (at most one period); the simulation renders the camera only
    while it is listened to, which costs a new subscriber more (measured 36-48 ms when
    subscribing again within seconds, up to about 150 ms after a long idle) -- the gap the
    wire's docstring and README state."""
    import json
    import subprocess

    iface = wirecheck.interface("rosmaster_x3_plus")
    period = 1.0 / float(next(t for t in iface["topics"] if t["name"] == "/usb_cam/image_raw")["rate"])
    names = set(docker_containers("label=rsim.robot=rosmaster_x3_plus")) & \
        set(docker_containers(f"label=rsim.sim_port={sim_port}"))
    [name] = names
    cmd = ("source /opt/ros/noetic/setup.bash; "
           "python3 /opt/rsim/simulator/tests/first_frame_probe_ros1.py /usb_cam/image_raw 5")
    res = subprocess.run(["docker", "exec", name, "bash", "-c", cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, timeout=120)
    lines = [l for l in res.stdout.splitlines() if l.startswith("{")]
    if not lines:
        return [f"first-frame probe failed: {res.stderr[-1000:]}"]
    lat = json.loads(lines[-1])["latency"]
    bad = [x for x in lat if x is None or x > 5 * period]
    return [f"/usb_cam/image_raw: first frames after {lat} s, over 5 periods "
            f"({5 * period:.3f} s)"] if bad else []


def checks(*fns):
    """Several behaviour checks run as one."""
    return lambda port, sim_port: [p for fn in fns for p in fn(port, sim_port)]


BEHAVIOUR_CHECKS = {"ainex": checks(ainex_app_nodes, usb_cam_calibration("ainex")),
                    "myagv": usb_cam_calibration("myagv"),
                    "rosmaster_x3_plus": checks(rosmaster_camera_services,
                                                  usb_cam_calibration("rosmaster_x3_plus",
                                                                      "/usb_cam/image_raw"),
                                                  usb_cam_compressed, usb_cam_theora_and_depth,
                                                  usb_cam_first_frame)}


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
    # the recorded intensities: the valid value at each reported range, the invalid one
    # where there is none (sensors.lidars[].intensity)
    if lid.get("intensity") is not None:
        it = lid["intensity"]
        inten = np.array([float(v) for v in s.get("intensities") or []])
        want = np.where(have, float(it["valid"]), float(it["invalid"]))
        if inten.shape != want.shape or not np.array_equal(inten, want):
            out.append(f"{topic}: intensities {inten[:5].tolist()}... ({inten.size}), recorded "
                       f"{it['valid']} at a return and {it['invalid']} elsewhere")
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


def rebot_limits(robot, iface) -> list:
    """Each arm joint commanded beyond its recorded maximum (the driver passes it on
    unclamped, as recorded): none goes past it, within the recorded tolerance; a gripper
    position beyond its recorded maximum aborts the goal, as recorded, and the gripper
    stays where it was."""
    port, sim_port = robot.port, robot.sim_port
    out = []
    arm = limited(motion_row(iface, "arm"))
    names = [field_joint(f) for f in arm]
    assert names == REBOT_ARM, names
    t_arm = motion.tol(REBOT, "arm joint final position")["absolute"]
    t_grip = motion.tol(REBOT, "gripper motor final position")["absolute"]
    rb = Rosbridge("127.0.0.1", port)
    try:
        for f, name in zip(arm, names):
            hi = float(f["max"])
            goal = rebot_goal([motion.beyond(hi) if n == name else 0.0 for n in names], 2)
            rb.send({"op": "send_action_goal", "action": "/rebotarm/follow_joint_trajectory",
                     "action_type": "control_msgs/action/FollowJointTrajectory", "args": goal,
                     "id": f"limit:{name}", "feedback": False})
            peak = motion.peak_joint(sim_port, REBOT, name, 3.0)
            if peak > hi + t_arm:
                out.append(f"{name} reached {peak:.3f} rad, recorded maximum {hi}")
            rebot_follow(rb, [0.0] * len(names), 2)
        motion.wait_joints(sim_port, REBOT, dict.fromkeys(names, 0.0), t_arm, 8)
        grip = motion_row(iface, "gripper")
        hi = float(limited(grip)[0]["max"])
        before = rebot_motor(motion.joints(sim_port, REBOT))
        res = rb.action(grip["command"]["name"], grip["command"]["type"],
                        {"command": {"position": motion.beyond(hi), "max_effort": 0.0}},
                        timeout=20)
        if res.get("status") != ABORTED:
            out.append(f"a gripper goal beyond {hi} rad was not aborted: {res}")
        motor = rebot_motor(motion.joints(sim_port, REBOT))
        if motor > hi + t_grip or abs(motor - before) > t_grip:
            out.append(f"the gripper motor went from {before:.3f} to {motor:.3f} rad on a goal "
                       f"beyond {hi}")
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
        reply = rb.call("/walking/get_param", {})
        out += typed(rb, "ainex", "service_response", "/walking/get_param", reply)
        prm = reply["parameters"]
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


#: How each robot's recorded command limits are exercised (drives through
#: motion.drive_limits); a robot with limited command fields and no check fails.
LIMIT_CHECKS = {REBOT: rebot_limits, "rosmaster_x3_plus": rosmaster_limits,
                "ainex": ainex_limits}


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
