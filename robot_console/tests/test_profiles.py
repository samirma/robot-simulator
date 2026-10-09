"""The packaged profiles: complete, source-traced, typed in their dialect, bounded."""

import copy
import math
import re

import pytest

from robot_console import dialect as d
from robot_console.profiles import (
    SUPPORTED_IDS, ProfileError, check_namespace_allowed, load,
    load_all, parse_profile, select_id, teleop_ids,
)

EXPECTED = {  # console spec §1.1
    "myagv": ("ros1", "base"), "ainex": ("ros1", "walk"), "rosmaster_x3_plus": ("ros1", "base"),
    "rebot_b601_rs": ("ros2", "none"),
}
# Pinned revisions identified by the robot specification (robots_specs/<id>.md).
PINNED = {
    "myagv": {"c71f3cc574e5ed1973a925238eabe88662cfa701", "addab4a65fdf65c460fec2cc3eee8fab94699370"},
    "ainex": {"e8fe2a816797cf83054135160df5a82ec3596a69"},
    "rosmaster_x3_plus": {"9732c62247dfb57a899aea01e4fe72ed434bac6d"},
    "rebot_b601_rs": {"7506dc0fc5c2a7787dcb974d89d13f9f49d16d61", "6415d43130d1e143c70dc106096a857ac5556f81"},
}
TRACE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+):(?P<path>[^#\s]+)#L\d+")


def test_supported_ids_and_dialects():
    assert SUPPORTED_IDS == tuple(EXPECTED)
    for p in load_all():
        assert (p.dialect, p.teleop) == EXPECTED[p.id]
    assert teleop_ids() == ("myagv", "ainex", "rosmaster_x3_plus")


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_pinned_sources(pid):
    p = load(pid)
    revs = {str(s.get("revision", "")) for s in p.sources.values()}
    assert PINNED[pid] <= revs, f"{pid} must record its pinned revisions"
    assert p.boot, "authoritative boot recorded"


def _traced(row, where):
    src, pol = str(row.get("source") or ""), str(row.get("policy") or "")
    assert src or pol, f"{where}: no source trace or console policy"
    return src


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_every_entry_is_traced(pid):
    raw = load(pid).raw
    keys = set(raw["sources"])
    for sec in ("topics", "services", "actions", "cameras", "sensors", "stop", "controls", "groups"):
        for row in raw.get(sec) or []:
            src = _traced(row, f"{pid}.{sec}.{row.get('name', row.get('topic', row.get('id')))}")
            m = TRACE.match(src)
            if m:
                assert m["key"] in keys, f"{pid}: trace key {m['key']} is not a pinned source"
    for c in raw.get("controls") or []:
        if c.get("busy"):                       # an action server's documented handling of a goal while one runs
            m = TRACE.match(_traced(c["busy"], f"{pid}.controls.{c['id']}.busy"))
            assert m and m["key"] in keys
        for f in c.get("fields") or []:
            _traced(f, f"{pid}.controls.{c['id']}.{f['name']}")
            if f.get("to_wire"):
                m = TRACE.match(_traced(f["to_wire"], f"{pid}.controls.{c['id']}.{f['name']}.to_wire"))
                assert m and m["key"] in keys
    for sec in ("teleop_base", "teleop_walk"):
        for axis, a in ((raw.get(sec) or {}).get("axes") or {}).items():
            _traced(a, f"{pid}.{sec}.{axis}")


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_types_use_the_profile_dialect(pid):
    p = load(pid)
    for e in p.endpoints:
        assert d.type_dialect(e.type) == p.dialect, (e.name, e.type)
    for c in p.cameras:
        assert d.normalize_type(c.type) == "sensor_msgs/Image"


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_cameras_are_topics_with_finite_stale_thresholds(pid):
    p = load(pid)
    for c in p.cameras:
        e = p.endpoint(c.topic)
        assert e is not None and e.kind == "topic" and e.type == c.type
        assert 0 < c.stale_after_s < math.inf
        assert c.encodings


@pytest.mark.parametrize("pid", ["myagv", "ainex", "rosmaster_x3_plus"])
def test_mobile_speeds_within_documented_limits_and_stop(pid):
    p = load(pid)
    axes = (p.teleop_base or p.teleop_walk).axes
    assert set(axes) == {"x", "y", "yaw"}
    for a in axes.values():
        assert 0 < a.speed <= a.limit
    stop = p.teleop_walk.stop if p.teleop_walk else p.stop
    assert stop, "a documented stop"
    if p.teleop_base:
        assert p.stop[0].op == "publish" and p.stop[0].name == p.teleop_base.topic
        assert all(v == 0 for part in p.stop[0].msg.values() for v in part.values())


def test_camera_sets():
    """Every mobile profile has a required raw camera stream; the arm's boot starts no camera, so
    its D405 wrist camera's colour and depth streams are optional ones (an arm without a D405 validates)."""
    cams = {p.id: [(c.topic, c.optional) for c in p.cameras] for p in load_all()}
    assert cams == {
        "myagv": [("/usb_cam/image_raw", False)],
        "ainex": [("/camera/image_raw", False)],
        "rosmaster_x3_plus": [("/camera/rgb/image_raw", False)],
        "rebot_b601_rs": [("/camera/camera/color/image_raw", True),
                          ("/camera/camera/aligned_depth_to_color/image_raw", True)],
    }
    assert {c.topic: c.kind for p in load_all() for c in p.cameras if c.kind != "color"} == {
        "/camera/camera/aligned_depth_to_color/image_raw": "depth"}


D405_TOPICS = {  # the realsense2_camera topics of the D405 wrist camera, all optional rows
    "/camera/camera/color/image_raw": "sensor_msgs/msg/Image",
    "/camera/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/camera/aligned_depth_to_color/image_raw": "sensor_msgs/msg/Image",
    "/camera/camera/aligned_depth_to_color/camera_info": "sensor_msgs/msg/CameraInfo",
}


def test_rebot_d405_is_optional_and_global():
    """The D405's four topics are optional rows, outside the arm namespace (realsense2_camera is not the
    controller), and the page's colour camera is rgb8 at the documented 30 Hz."""
    p = load("rebot_b601_rs")
    for name, typ in D405_TOPICS.items():
        e = p.endpoint(name)
        assert e is not None and (e.kind, e.type, e.direction, e.optional) == ("topic", typ, "out", True), name
        assert p.resolve(name, "right_arm") == name == p.resolve(name, "rebotarm")
    assert not {e.name for e in p.required()} & set(D405_TOPICS)
    color, depth = p.cameras
    assert (color.kind, color.encodings, color.rate_hz, color.optional) == ("color", ("rgb8",), 30, True)
    assert color.range_m is None and color.unit_m is None
    assert (depth.topic, depth.kind, depth.encodings, depth.rate_hz, depth.optional) == (
        "/camera/camera/aligned_depth_to_color/image_raw", "depth", ("16UC1",), 30, True)
    assert (depth.unit_m, depth.range_m) == (0.001, (0.07, 0.5))     # millimetres; the D405's recorded range
    assert p.resolve("/joint_states", "right_arm") == "/right_arm/joint_states"


# The sensors of the reBot, written out here from the pinned controller and messages (rebotarm_msgs
# JointMotorState.msg, ArmStatus.msg; rebotarmcontroller/ros_publishers.py).
MOTOR_TOPICS = [f"/joints/joint{i}/state" for i in range(1, 7)] + ["/gripper/state"]
STATUS_BITS = {0: "undervoltage", 1: "overcurrent", 2: "overtemperature", 3: "encoder fault", 4: "stall",
               5: "uncalibrated"}


def test_only_the_rebot_has_sensors_and_they_are_its_manufacturer_topics():
    assert {p.id: [s.topic for s in p.sensors] for p in load_all()} == {
        "myagv": [], "ainex": [], "rosmaster_x3_plus": [],
        "rebot_b601_rs": ["/joint_states"] + MOTOR_TOPICS + ["/arm_status"]}


def test_rebot_sensor_fields_are_the_messages_fields():
    p = load("rebot_b601_rs")
    by = {s.topic: s for s in p.sensors}
    assert by["/joint_states"].type == "sensor_msgs/msg/JointState" and by["/joint_states"].stale_after_s == 1.0
    names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_joint1", "gripper_joint2"]
    js = {f.name: f for f in by["/joint_states"].fields}
    assert len(js) == 3 * len(names)
    for i, n in enumerate(names):          # the arrays' index is the joint's index in `name`
        for k, key in enumerate(("position", "velocity", "effort")):
            f = js[f"{n}_{key}"]
            assert (f.path, f.group, f.kind) == (f"{key}.{i}", n, "number")
    for t in MOTOR_TOPICS:
        s = by[t]
        assert s.type == "rebotarm_msgs/msg/JointMotorState" and s.stale_after_s == 1.0
        assert [(f.name, f.path, f.kind) for f in s.fields] == [
            ("joint_name", "joint_name", "text"), ("position", "position", "number"),
            ("velocity", "velocity", "number"), ("torque", "torque", "number"),
            ("status_code", "status_code", "bits")]
        assert {b.bit: b.label for b in s.fields[-1].bits} == STATUS_BITS
    a = by["/arm_status"]
    assert a.type == "rebotarm_msgs/msg/ArmStatus" and a.stale_after_s is None       # latched, not periodic
    assert [(f.name, f.kind) for f in a.fields] == [
        ("mode", "text"), ("enabled", "bool"), ("control_loop_active", "bool"), ("state_machine", "text"),
        ("joint_names", "list"), ("per_joint_status_code", "list"), ("error_codes", "list")]


def test_sensors_reach_the_page_and_are_read_only():
    from robot_console.profiles import profile_json
    p = load("rebot_b601_rs")
    j = profile_json(p)
    assert [s["topic"] for s in j["sensors"]] == [s.topic for s in p.sensors]
    assert j["sensors"][1]["fields"][4]["bits"][0] == {"bit": 0, "label": "undervoltage"}
    assert j["cameras"][0]["optional"] is True
    assert [(c["kind"], c["range_m"], c["unit_m"]) for c in j["cameras"]] == [("color", None, None),
                                                                              ("depth", [0.07, 0.5], 0.001)]
    assert all(profile_json(q)["sensors"] == [] for q in load_all() if q.id != p.id)
    assert not {c.name for c in p.controls} & {s.topic for s in p.sensors}      # no sensor is commanded


def test_profile_loader_refuses_inconsistent_cameras():
    """A depth camera is 16UC1 with its metres-per-value and its range; a colour camera has neither."""
    raw = load("rebot_b601_rs").raw
    for idx, change, error in (
            (1, lambda c: c.update(kind="thermal"), "is none of"),
            (1, lambda c: c.pop("range_m"), "range_m is"),
            (1, lambda c: c.update(range_m=[0.5, 0.07]), "range_m is"),
            (1, lambda c: c.update(range_m=[0.07]), "range_m is"),
            (1, lambda c: c.pop("unit_m"), "unit_m is"),
            (1, lambda c: c.update(unit_m=0), "unit_m is"),
            (1, lambda c: c.update(encodings=["rgb8"]), "only encoding is 16UC1"),
            (0, lambda c: c.update(range_m=[0.07, 0.5]), "belong to a depth camera"),
            (0, lambda c: c.update(unit_m=0.001), "belong to a depth camera")):
        bad = copy.deepcopy(raw)
        change(bad["cameras"][idx])
        with pytest.raises(ValueError, match=error):
            parse_profile(bad)
    assert parse_profile(raw).cameras == load("rebot_b601_rs").cameras


def test_profile_loader_refuses_inconsistent_sensors():
    raw = load("rebot_b601_rs").raw
    for change, error in (
            (lambda s: s.update(topic="/nope"), "is not an out topic"),
            (lambda s: s.update(type="std_msgs/msg/String"), "is not an out topic"),
            (lambda s: s.update(topic="/joints/joint1/cmd/mit", type="rebotarm_msgs/msg/JointMitCmd"), "is not an out topic"),
            (lambda s: s.update(topic="/camera/camera/color/image_raw", type="sensor_msgs/msg/Image"), "is a camera"),
            (lambda s: s.update(stale_after_s=0), "stale_after_s is positive"),
            (lambda s: s.update(fields=[]), "one or more fields"),
            (lambda s: s["fields"].append(dict(s["fields"][0])), "distinct names"),
            (lambda s: s["fields"][0].update(kind="blob"), "is none of"),
            (lambda s: s["fields"][0].update(decimals=-1), "non-negative integer"),
            (lambda s: s["fields"][0].update(bits=[{"bit": 0, "label": "x"}]), "bits are recorded for"),
            (lambda s: s["fields"][-1]["bits"].append({"bit": 0, "label": "again"}), "distinct")):
        bad = copy.deepcopy(raw)
        change(bad["sensors"][1])              # joint1's motor state
        with pytest.raises(ValueError, match=error):
            parse_profile(bad)
    bad = copy.deepcopy(raw)
    bad["sensors"].append(copy.deepcopy(bad["sensors"][0]))
    with pytest.raises(ValueError, match="sensor ids are distinct"):
        parse_profile(bad)
    assert parse_profile(raw).sensors == load("rebot_b601_rs").sensors


def test_rebot_controls_are_the_controllers_goals():
    """The arm goal names joint1..joint6 in the controller's order (it aborts any other list,
    ros_actions.py#L146-L152) with one point reached after whole seconds; the driver does not clamp,
    so the fields carry the URDF joint limits (ReBot_Arm_RS.urdf), written out here from the file,
    and default to the safe-home pose (all joints 0). The gripper goal's position is the gripper
    motor angle within the rs configuration's close 0.0 .. open 5.0 (the driver refuses others),
    max_effort the vendor's documented 0.0. Both are cancellable actions with joint-state feedback."""
    p = load("rebot_b601_rs")
    arm, grip = (next(c for c in p.controls if c.id == i) for i in ("arm_trajectory", "gripper"))
    assert [c.id for c in p.controls] == ["arm_trajectory", "gripper"]
    assert (arm.kind, arm.name, arm.type) == ("action", "/follow_joint_trajectory", "control_msgs/action/FollowJointTrajectory")
    t = arm.template["trajectory"]
    assert t["joint_names"] == [f"joint{i}" for i in range(1, 7)]
    (point,) = t["points"]
    assert point == {"positions": [f"$joint{i}" for i in range(1, 7)], "time_from_start": {"sec": "$duration", "nanosec": 0}}
    assert {f.name: (f.min, f.max) for f in arm.fields} == {
        "joint1": (-2.8, 2.8), "joint2": (0.0, 3.14), "joint3": (0.0, 3.14), "joint4": (-1.57, 1.57),
        "joint5": (-1.57, 1.57), "joint6": (-3.14, 3.14), "duration": (1.0, 10.0)}
    assert all(f.default == 0.0 and f.unit == "rad" for f in arm.fields if f.name != "duration")
    duration = arm.fields[-1]
    assert (duration.name, duration.integer, duration.unit) == ("duration", True, "s")
    assert (grip.kind, grip.name, grip.type) == ("action", "/gripper/command", "control_msgs/action/GripperCommand")
    assert grip.template == {"command": {"position": "$position", "max_effort": 0.0}}
    (f,) = grip.fields
    assert (f.min, f.max, f.default, f.unit) == (0.0, 5.0, 0.0, "rad")
    for c in (arm, grip):
        assert [o.op for o in c.stop] == ["cancel"] and c.prerequisites == (c.name, "/joint_states")
    assert p.stop == () and [c.topic for c in p.cameras] == [
        "/camera/camera/color/image_raw", "/camera/camera/aligned_depth_to_color/image_raw"]


def test_rebot_action_servers_do_not_preempt():
    """What the reBot's servers do with a goal that arrives while one runs (console spec §1.1,
    §2.2): the arm's rejects it (arm_goal_callback returns REJECT in TRAJ_RUNNING, which an
    accepted trajectory enters as it starts, ros_actions.py#L46-L59, hardware_manager.py#L412-L416);
    the gripper's accepts it and runs it beside the running one (its goal callback refuses only
    during gravity compensation or safe home, ros_actions.py#L51-L59). Neither preempts, so the
    page streams goals to neither."""
    p = load("rebot_b601_rs")
    assert {c.id: c.busy for c in p.controls} == {"arm_trajectory": "reject", "gripper": "parallel"}
    raw = {c["id"]: c["busy"] for c in p.raw["controls"]}
    assert raw["arm_trajectory"]["source"].endswith("ros_actions.py#L46-L59")
    assert "hardware_manager.py#L412-L416" in raw["arm_trajectory"]["note"]
    assert raw["gripper"]["source"].endswith("ros_actions.py#L51-L59")
    from robot_console.profiles import profile_json
    assert [c["busy"] for c in profile_json(p)["controls"]] == ["reject", "parallel"]     # handed to the page


def test_profile_loader_requires_busy_on_action_controls_only():
    raw = load("rebot_b601_rs").raw
    for change, error in ((lambda c: c.pop("busy"), "records busy.new_goal"),
                          (lambda c: c["busy"].update(new_goal="queue"), "records busy.new_goal"),
                          (lambda c: c.update(busy="reject"), "records busy.new_goal")):
        bad = copy.deepcopy(raw)
        change(bad["controls"][0])
        with pytest.raises(ValueError, match=error):
            parse_profile(bad)
    bad = copy.deepcopy(load("ainex").raw)
    bad["controls"][0]["busy"] = {"new_goal": "reject", "source": "x"}
    with pytest.raises(ValueError, match="busy is recorded for action controls only"):
        parse_profile(bad)
    for p in load_all():
        for c in p.controls:
            assert (c.busy != "") == (c.kind == "action"), (p.id, c.id)


def test_ainex_head_limits():
    p = load("ainex")
    assert set(p.head) == {"pan", "tilt"}
    for h in p.head.values():
        assert h.min < 0 < h.max and h.rate > 0


def test_ainex_head_position_read():
    """Teleop starts each head axis from its measured position (console spec §2.1), the page reads
    it on a click: one documented read, servo 23/24 through the board driver's service, mapped
    back with the controller's pulse<->rad relation (init 500, 180/pi/240*1000 pulses per rad,
    ainex_controller.py#L24-L25). A servo the board could not read is left out of the answer."""
    p = load("ainex")
    answer = {"success": True, "position": [{"id": 23, "position": 739}, {"id": 24, "position": 500}]}
    for axis, servo, want in (("pan", 23, 239 * math.radians(240) / 1000), ("tilt", 24, 0.0)):
        read, value = p.head_read(axis)
        assert (read.service, read.type) == (
            "/ros_robot_controller/bus_servo/get_position", "ros_robot_controller/GetBusServosPosition")
        assert servo in read.request["id"]
        e = p.endpoint(read.service)
        assert e is not None and e.kind == "service" and e.type == read.type and not e.optional
        assert read.parse(answer)[(value.control, value.field)] == pytest.approx(want, rel=1e-6, abs=1e-12)
        assert read.parse({"success": True, "position": []})[(value.control, value.field)] is None
        with pytest.raises(ValueError):
            read.parse({"success": False, "position": []})


def test_reads_parse_documented_invalid_answers():
    x3 = load("rosmaster_x3_plus").reads[0]
    got = x3.parse({"angles": [80.0, 120.0, 10.0, 45.0, -1, 60.0]})
    assert got[("arm_pose", "j2")] == 120.0 and got[("arm_pose", "j5")] is None and got[("gripper", "angle")] == 60.0


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_controls_are_bounded_and_never_drive_the_base(pid):
    p = load(pid)
    sustained = set()
    if p.teleop_base:
        sustained.add(p.teleop_base.topic)
    if p.teleop_walk:
        sustained |= {p.teleop_walk.param_topic} | {o.name for o in p.teleop_walk.start}
    for c in p.controls:
        assert c.name not in sustained, f"{pid}: control {c.id} would drive the base/walk"
        assert c.kind in ("publish", "call", "action")
        if c.kind == "action":
            assert c.stop and c.stop[0].op == "cancel"
            assert p.dialect == "ros2", "the page sends ROS 2 goals only (one connection each, spec §2.2)"
        for f in c.fields:
            if f.choices is None:
                assert f.min is not None and f.max is not None and f.min <= f.max
                assert f.min <= float(f.default) <= f.max
            else:
                assert f.default in f.choices
        # every "$field" of the template is a field of the control, and every field is sent
        assert placeholders(c.template) == {f.name for f in c.fields}, c.id
        for pre in c.prerequisites:
            assert p.endpoint(pre) is not None


def placeholders(template):
    if isinstance(template, dict):
        return set().union(*map(placeholders, template.values())) if template else set()
    if isinstance(template, list):
        return set().union(*map(placeholders, template)) if template else set()
    return {template[1:]} if isinstance(template, str) and template.startswith("$") else set()


def test_namespace_only_where_documented():
    """Only the reBot's hardware interface is namespace-configurable: the bringup's arm_namespace
    launch argument (default "rebotarm") prefixes every name the controller creates, while
    robot_state_publisher's /robot_description, /tf and /tf_static stay global; joint names are
    not prefixed."""
    for p in load_all():
        assert (p.namespace is not None) == (p.id == "rebot_b601_rs"), p.id
    p = load("rebot_b601_rs")
    assert p.namespace.default == "rebotarm" and p.namespace.joint_prefix == ""
    assert p.resolve("/follow_joint_trajectory", "rebotarm") == "/rebotarm/follow_joint_trajectory"
    assert p.resolve("/joints/joint3/state", "right_arm") == "/right_arm/joints/joint3/state"
    assert p.resolve("/gripper/command", "right_arm") == "/right_arm/gripper/command"
    for name in ("/tf", "/tf_static", "/robot_description"):
        assert p.resolve(name, "right_arm") == name
    assert {e.name for e in p.endpoints if e.name in p.namespace.global_names} == set(p.namespace.global_names)
    for c in p.controls:                                   # joint names carry no namespace
        assert "{joint_prefix}" not in str(c.template), c.id
    for other in load_all():
        if other.id != "rebot_b601_rs":
            with pytest.raises(ProfileError, match="no namespace override"):
                check_namespace_allowed(other, "robot1")
            check_namespace_allowed(other, None)
    check_namespace_allowed(p, "right_arm")


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_reads_are_documented_measurements_of_control_fields(pid):
    """A profile's `reads` (documented read services returning measured joint positions, which the
    page calls only on a click) are traced, name a service row of the profile with its type, and
    map each answered value onto a bounded numeric field of one of its controls."""
    p = load(pid)
    raw = p.raw
    for r in raw.get("reads") or []:
        t = TRACE.match(str(r.get("source", "")))
        assert t and t["key"] in p.sources, r["id"]
        e = p.endpoint(r["service"])
        assert e is not None and e.kind == "service" and e.type == r["type"], r["id"]
        assert r["values"], r["id"]
        for v in r["values"]:
            assert TRACE.match(str(v.get("source", ""))), (r["id"], v)
            c = next(c for c in p.controls if c.id == v["control"])
            f = next(f for f in c.fields if f.name == v["field"])
            assert f.choices is None and f.min is not None and f.max is not None
            assert float(v.get("scale", 1)) != 0
    ids = {pid: [r["id"] for r in load(pid).raw.get("reads") or []] for pid in SUPPORTED_IDS}
    assert ids == {"myagv": [], "ainex": ["head_position", "joint_positions"], "rosmaster_x3_plus": ["current_angle"],
                   "rebot_b601_rs": []}


# ------------------------------------------------------------------ AiNex: one control per servo

# The pinned vendor servo table, written out here from the files themselves: servo id by joint
# (ainex:src/ainex_driver/ainex_kinematics/scripts/ainex_controller.py#L42-L65), init pulse and
# direction (servo_controller.yaml: flipped where min > max), standing pose (init_pose.yaml).
AINEX_SERVOS = {  # joint: (servo id, init, flipped, init_pose rad)
    "l_ank_roll": (1, 500, False, 0.016), "r_ank_roll": (2, 500, False, -0.016),
    "l_ank_pitch": (3, 500, False, 0.625), "r_ank_pitch": (4, 500, False, -0.625),
    "l_knee": (5, 240, False, 1.192), "r_knee": (6, 760, False, -1.192),
    "l_hip_pitch": (7, 500, False, -0.828), "r_hip_pitch": (8, 500, False, 0.828),
    "l_hip_roll": (9, 500, False, 0.016), "r_hip_roll": (10, 500, False, -0.016),
    "l_hip_yaw": (11, 500, False, 0.0), "r_hip_yaw": (12, 500, False, 0.0),
    "l_sho_pitch": (13, 875, True, 0.0), "r_sho_pitch": (14, 125, True, 0.0),
    "l_sho_roll": (15, 500, False, 1.293), "r_sho_roll": (16, 500, False, -1.293),
    "l_el_pitch": (17, 500, False, -0.10), "r_el_pitch": (18, 500, False, 0.10),
    "l_el_yaw": (19, 500, False, -1.926), "r_el_yaw": (20, 500, False, 1.926),
    "l_gripper": (21, 500, False, 0.0), "r_gripper": (22, 500, False, 0.0),
    "head_pan": (23, 500, False, 0.0), "head_tilt": (24, 500, False, 0.0),
}
TICKS_PER_RAD = 180 / 3.1415926 / 240 * 1000        # ainex_controller.py#L25 (the vendor's 7-digit pi)
URDF_LIMIT = 2.09                                   # every AiNex joint, ainex.urdf.xacro (test_view_urdf.py)
SET_POSITION = "/ros_robot_controller/bus_servo/set_position"
GET_POSITION = "/ros_robot_controller/bus_servo/get_position"


def angle2pulse(joint, rad):
    """ainex_controller.py#L92-L104,#L320-L322: init + int(round(rad * +-ENCODER_TICKS_PER_RADIAN))."""
    sid, init, flipped, _ = AINEX_SERVOS[joint]
    return init + int(round(rad * (-TICKS_PER_RAD if flipped else TICKS_PER_RAD)))


def servo_limits(joint):
    """The servo's 0..1000 pulses in rad, within the URDF's +-2.09."""
    _, init, flipped, _ = AINEX_SERVOS[joint]
    coef = -TICKS_PER_RAD if flipped else TICKS_PER_RAD
    lo, hi = sorted(((0 - init) / coef, (1000 - init) / coef))
    return max(lo, -URDF_LIMIT), min(hi, URDF_LIMIT)


def test_ainex_every_servo_has_one_control_with_a_single_servo_id():
    """Every one of the 24 servos is commanded by exactly one control: the head through the
    controller's head topics, every other servo through the board driver's set_position topic with
    one servo id per message (dragging one joint never resends the others' targets)."""
    p = load("ainex")
    servo = {c.id: c for c in p.controls if c.name == SET_POSITION}
    assert set(servo) | {"head_pan", "head_tilt"} == set(AINEX_SERVOS)
    for joint, c in servo.items():
        sid = AINEX_SERVOS[joint][0]
        assert (c.kind, c.type) == ("publish", "ros_robot_controller/SetBusServosPosition")
        assert c.template == {"duration": "$duration", "position": [{"id": sid, "position": "$position"}]}, joint
        assert c.prerequisites == (SET_POSITION,) and p.endpoint(SET_POSITION).direction == "in"
        raw = next(r for r in p.raw["controls"] if r["id"] == joint)
        assert raw["source"].endswith(f"ainex_controller.py#L{42 + list(_ID_ORDER).index(joint)}"), joint


_ID_ORDER = ("r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll", "l_hip_yaw", "l_hip_roll",
             "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll", "r_sho_pitch", "l_sho_pitch", "l_sho_roll", "r_sho_roll",
             "l_el_pitch", "r_el_pitch", "l_el_yaw", "r_el_yaw", "l_gripper", "r_gripper", "head_pan", "head_tilt")


def test_ainex_servo_controls_are_labelled_and_grouped_by_limb():
    p = load("ainex")
    side = {"l": "Left", "r": "Right"}
    part = {"sho_pitch": "shoulder pitch", "sho_roll": "shoulder roll", "el_pitch": "elbow pitch", "el_yaw": "elbow yaw",
            "gripper": "hand (gripper)", "hip_yaw": "hip yaw", "hip_roll": "hip roll", "hip_pitch": "hip pitch",
            "knee": "knee", "ank_pitch": "ankle pitch", "ank_roll": "ankle roll"}
    assert [(g.id, g.label) for g in p.groups] == [("head", "Head"), ("left_arm", "Left arm"), ("right_arm", "Right arm"),
                                                  ("left_leg", "Left leg"), ("right_leg", "Right leg")]
    for c in p.controls:
        if c.id in ("head_pan", "head_tilt"):
            assert c.group == "head"
            continue
        if c.name != SET_POSITION:
            assert c.group == "", c.id                      # Stand and action groups stay cards
            continue
        limb = ("arm" if c.id[2:4] in ("sh", "el", "gr") else "leg")
        assert c.label == f"{side[c.id[0]]} {part[c.id[2:]]}" and c.group == f"{'left' if c.id[0] == 'l' else 'right'}_{limb}"
        assert c.fields[0].label == f"{c.id} (servo {AINEX_SERVOS[c.id][0]})"
    order = [c.group for c in p.controls if c.group]
    assert order == sorted(order, key=[g.id for g in p.groups].index), "a group's controls are listed together"
    for g in p.raw["groups"]:
        m = TRACE.match(g["source"])
        assert m and m["key"] in p.sources, g["id"]
    notes = {g.id: g.note for g in p.groups}
    assert all("overwrite these targets" in notes[g] for g in ("left_arm", "right_arm", "left_leg", "right_leg"))


def test_ainex_servo_positions_go_on_the_wire_as_the_controllers_angle2pulse():
    """The field is the joint's rad; the template gets the controller's pulse for it (init +
    round(rad * +-238.73)), and a value outside the documented limits is refused."""
    p = load("ainex")
    for c in p.controls:
        if c.name != SET_POSITION:
            continue
        f, dur = c.fields
        _, init, flipped, standing = AINEX_SERVOS[c.id]
        w = f.to_wire
        assert (w.offset, w.integer, w.min, w.max) == (init, True, 0, 1000), c.id
        assert w.scale == pytest.approx(-TICKS_PER_RAD if flipped else TICKS_PER_RAD, rel=1e-9), c.id
        for rad in (f.min, -0.5, -0.1234, 0.0, 0.3, 0.5, f.max, standing):
            if f.min <= rad <= f.max:
                assert f.wire(rad) == angle2pulse(c.id, rad), (c.id, rad)
                assert 0 <= f.wire(rad) <= 1000
        assert (dur.min, dur.max, dur.default, dur.unit, dur.to_wire) == (0.2, 2.0, 0.5, "s", None)
    by = {c.id: c for c in p.controls}
    assert by["l_el_pitch"].fields[0].wire(0.5) == 619                  # 500 + round(0.5 * 238.73)
    assert by["l_knee"].fields[0].wire(1.0) == 479                      # 240 + 239
    assert by["l_sho_pitch"].fields[0].wire(0.5) == 756                 # flipped: 875 - 119
    with pytest.raises(ValueError, match="outside"):
        by["l_knee"].fields[0].to_wire.apply(-1.01)                     # below pulse 0


def test_ainex_servo_limits_are_the_servo_range_within_the_urdf():
    """No console envelope: the limits are the servo's 0..1000 pulses in rad, intersected with the
    URDF's +-2.09, rounded inward by less than 0.001 rad; the default is the vendor's standing pose."""
    p = load("ainex")
    for c in p.controls:
        if c.name != SET_POSITION:
            continue
        f = c.fields[0]
        lo, hi = servo_limits(c.id)
        assert lo <= f.min < lo + 0.001 and hi - 0.001 < f.max <= hi, (c.id, f.min, f.max, lo, hi)
        assert f.unit == "rad" and f.default == pytest.approx(AINEX_SERVOS[c.id][3])
    lim = {c.id: (c.fields[0].min, c.fields[0].max) for c in p.controls if c.name == SET_POSITION}
    assert lim["l_knee"] == (-1.005, 2.09) and lim["r_knee"] == (-2.09, 1.005)
    assert lim["l_sho_pitch"] == (-0.523, 2.09) and lim["r_sho_pitch"] == (-2.09, 0.523)
    assert lim["l_el_pitch"] == (-2.09, 2.09)


def test_ainex_polled_read_measures_every_servo():
    """The documented read-only joint-position query the page polls (console spec §2.2): all 24 servos through get_position, each mapped back to its control's rad with
    the inverse of that control's wire map. No other profile polls."""
    for p in load_all():
        polled = [r for r in p.reads if r.poll_s]
        assert len(polled) == (1 if p.id == "ainex" else 0), p.id
    p = load("ainex")
    (r,) = [r for r in p.reads if r.poll_s]
    assert (r.service, r.type, r.ok_field, r.poll_s) == (GET_POSITION, "ros_robot_controller/GetBusServosPosition", "success", 1.0)
    assert sorted(r.request["id"]) == list(range(1, 25))
    assert "console policy" in p.raw["reads"][1]["poll_policy"]
    fields = {v.control: v for v in r.values}
    assert set(fields) == set(AINEX_SERVOS)
    rads = {"l_el_pitch": 0.5, "l_knee": 1.206, "r_knee": -1.206, "l_sho_pitch": 0.4, "r_sho_pitch": -0.3, "head_pan": 0.82}
    answer = {"success": True, "position": [{"id": AINEX_SERVOS[j][0], "position": angle2pulse(j, rads.get(j, 0.0))}
                                            for j in AINEX_SERVOS if j != "l_gripper"]}
    got = r.parse(answer)
    for j in AINEX_SERVOS:
        if j == "l_gripper":
            assert got[(j, "position")] is None                         # left out of the answer: no value
            continue
        assert got[(j, "position")] == pytest.approx(rads.get(j, 0.0), abs=0.5 / TICKS_PER_RAD + 1e-9), j
        c = next(c for c in p.controls if c.id == j)
        if c.fields[0].to_wire:                                         # the read is the wire map's inverse
            w = c.fields[0].to_wire
            assert (fields[j].offset, fields[j].scale) == (w.offset, w.scale), j


def test_profile_loader_refuses_inconsistent_maps_groups_and_polls():
    raw = copy.deepcopy(load("ainex").raw)
    bad = copy.deepcopy(raw)
    next(c for c in bad["controls"] if c["id"] == "l_knee")["fields"][0]["min"] = -1.2    # pulse < 0
    with pytest.raises(ValueError, match="l_knee.position: documented limit"):
        parse_profile(bad)
    bad = copy.deepcopy(raw)
    next(c for c in bad["controls"] if c["id"] == "l_knee")["group"] = "tail"
    with pytest.raises(ValueError, match="unknown group"):
        parse_profile(bad)
    bad = copy.deepcopy(raw)
    bad["reads"][0]["poll_s"] = 2.0
    with pytest.raises(ValueError, match="at most one read is polled"):
        parse_profile(bad)
    assert parse_profile(raw).controls == load("ainex").controls


@pytest.mark.parametrize("unknown", ["turtlebot", "so100"])
def test_id_refusals(unknown):
    with pytest.raises(ProfileError, match=f"unknown robot id '{unknown}'. Accepted ids: "
                                           "myagv, ainex, rosmaster_x3_plus, rebot_b601_rs$"):
        select_id(unknown)
    with pytest.raises(ProfileError, match="'rebot_b601_rs' is an arm.*myagv, ainex, rosmaster_x3_plus$"):
        select_id("rebot_b601_rs", teleop=True)

