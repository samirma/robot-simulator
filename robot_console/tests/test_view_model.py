"""The profiles' `model` sections behind the page's 3D model (console spec §2.2): traced to the
pinned vendor sources and bound to the catalog fields that command each joint with the
documented unit mapping (independence from the simulator: test_independence.py)."""

import dataclasses
import math
import re

import pytest

from robot_console.profiles import SUPPORTED_IDS, all_profiles_json, load, load_all

TRACE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+):(?P<path>[^#\s]+)#L\d+")

# Every catalog field that commands a joint, and that joint's name in the vendor URDF; listed
# here independently of the profiles' model sections.
_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
COMMANDED = {
    "so101": {**{("arm_trajectory", a): f"{a}_joint" for a in _ARM}, ("gripper", "position"): "gripper_joint"},
    # every AiNex servo: its control (named as the joint) commands the joint, in the joint's rad
    "ainex": {(j, "position"): j for j in (
        "head_pan", "head_tilt", "l_sho_pitch", "l_sho_roll", "l_el_pitch", "l_el_yaw", "l_gripper", "r_sho_pitch",
        "r_sho_roll", "r_el_pitch", "r_el_yaw", "r_gripper", "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee",
        "l_ank_pitch", "l_ank_roll", "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll")},
    "rosmaster_x3_plus": {**{("arm_pose", f"j{i}"): f"arm_joint{i}" for i in range(1, 6)},
                          ("arm_pose", "gripper"): "grip_joint", ("gripper", "angle"): "grip_joint"},
    "myagv": {},
}


def bindings(pid):
    m = load(pid).raw["model"]
    return {(b["control"], b["field"]): (j, b) for j in m["joints"] for b in j.get("command") or []}


def field(pid, cid, name):
    c = next(c for c in load(pid).controls if c.id == cid)
    return next(f for f in c.fields if f.name == name)


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_every_commanding_field_is_bound_to_its_joint(pid):
    assert {k: j["name"] for k, (j, _) in bindings(pid).items()} == COMMANDED[pid]
    for (cid, name), (j, _) in bindings(pid).items():
        assert field(pid, cid, name).choices is None and not j.get("mimic") and j["type"] == "revolute"


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_model_rows_trace_to_pinned_sources_and_form_one_tree(pid):
    p = load(pid)
    m = p.raw["model"]
    rows = (m["joints"] + (m.get("tips") or []) + (m.get("boxes") or []) + ([m["feedback"]] if m["feedback"] else [])
            + [b for j in m["joints"] for b in j.get("command") or [] if "scale" in b or "offset" in b])
    for r in rows:
        t = TRACE.match(str(r.get("source", "")))
        assert t and t["key"] in p.sources, r
    children = [j["child"] for j in m["joints"]]
    assert len(children) == len(set(children)) and m["root"] not in children
    links = {m["root"], *children}
    for j in m["joints"]:
        assert j["parent"] in links
        assert len(j["xyz"]) == 3 and len(j["rpy"]) == 3
        if j["type"] != "fixed":
            assert len(j["axis"]) == 3 and math.hypot(*j["axis"]) > 0
        if j.get("mimic"):
            assert j["mimic"]["joint"] in {x["name"] for x in m["joints"]}
    for t in m.get("tips") or []:
        assert t["link"] in links
    for b in m.get("boxes") or []:
        assert b["link"] in links and all(lo <= hi for lo, hi in zip(b["min"], b["max"]))


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_feedback_is_a_documented_joint_state_output(pid):
    p = load(pid)
    fb = p.raw["model"]["feedback"]
    reported = [j for j in p.raw["model"]["joints"] if j.get("reported")]
    if fb is None:
        assert not reported
        return
    e = p.endpoint(fb["topic"])
    assert e is not None and e.kind == "topic" and e.direction == "out" and not e.optional
    assert e.type == fb["type"] and e.type.replace("/msg/", "/") == "sensor_msgs/JointState"
    assert reported


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_unit_mappings_keep_field_limits_within_the_joint_limits(pid):
    for (cid, name), (j, b) in bindings(pid).items():
        f = field(pid, cid, name)
        scale, offset = float(b.get("scale", 1)), float(b.get("offset", 0))
        lo, hi = sorted((f.min * scale + offset, f.max * scale + offset))
        # 0.04 rad: the ROSMASTER driver maps its 30 deg gripper end to -pi/2 where the URDF stops at -1.54
        assert j["min"] - 0.04 <= lo <= hi <= j["max"] + 0.04, (cid, name, lo, hi, j["min"], j["max"])
        if "scale" not in b and "offset" not in b:
            assert f.unit == "rad"


def test_rosmaster_mapping_is_the_drivers_joint_states_mapping():
    """Mcnamu_X3plus.py joints_states_update: rad = (deg - 90) * pi/180, the gripper's 30..180
    deg first mapped linearly to 0..90."""
    def driver(deg, gripper):
        if gripper:
            deg = (deg - 30) * 90 / 150
        return (deg - 90) * math.pi / 180
    for (cid, name), (j, b) in bindings("rosmaster_x3_plus").items():
        f = field("rosmaster_x3_plus", cid, name)
        for deg in (f.min, (f.min + f.max) / 2, f.max):
            got = deg * float(b["scale"]) + float(b["offset"])
            assert got == pytest.approx(driver(deg, j["name"] == "grip_joint"), abs=1e-6), (cid, name, deg)


def test_ainex_head_pan_turns_about_minus_z_and_its_label_says_so():
    """Console spec §2.1 (amended): a negative head_pan turns the head to the robot's left."""
    pan = next(j for j in load("ainex").raw["model"]["joints"] if j["name"] == "head_pan")
    assert pan["axis"] == [0, 0, -1]
    label = field("ainex", "head_pan", "position").label
    assert "- = left" in label and "+ = left" not in label


def test_ainex_model_feedback_is_the_polled_read_of_every_commanded_field():
    """The AiNex boot publishes no joint states (feedback null); the profile's polled read measures
    the field that commands each movable joint, so the model can follow the robot (console spec
    §2.2, amended). Bindings are the identity: the controller's rad is the URDF joint's
    (ainex_controller.py#L196-L203 sends the same goal_position to Gazebo and, as pulses, to the servo)."""
    p = load("ainex")
    m = p.raw["model"]
    assert m["feedback"] is None
    (r,) = [r for r in p.reads if r.poll_s]
    measured = {(v.control, v.field) for v in r.values}
    movable = [j for j in m["joints"] if j["type"] != "fixed"]
    for j in movable:
        (b,) = j["command"]
        assert (b["control"], b["field"]) in measured and "scale" not in b and "offset" not in b, j["name"]
    assert len(movable) == 24


def test_the_page_receives_every_model():
    j = all_profiles_json()
    assert [p["id"] for p in j["profiles"]] == list(SUPPORTED_IDS)
    for p in j["profiles"]:
        assert p["model"] == load(p["id"]).raw["model"]
        assert [g["id"] for g in p["groups"]] == [g["id"] for g in load(p["id"]).raw.get("groups") or []]
    ainex = next(p for p in j["profiles"] if p["id"] == "ainex")
    knee = next(c for c in ainex["controls"] if c["id"] == "l_knee")
    assert knee["group"] == "left_leg" and knee["fields"][0]["to_wire"] == {
        "offset": 240.0, "scale": 238.7324187, "integer": True, "min": 0.0, "max": 1000.0}
    assert [r["poll_s"] for r in ainex["reads"]] == [None, 1.0]



def test_the_page_receives_teleops_base_motion_contract():
    """Console spec §3: the browser follows the same profile contracts. The page's base motion is
    built from teleop's own profile data and constants (teleop_core.page_contract): the recorded
    speeds and documented limits, the command and the stop teleop sends, its rates, head axes and
    head read; an arm gets none."""
    from robot_console import teleop_core as tc
    for p in load_all():
        got = next(x for x in all_profiles_json()["profiles"] if x["id"] == p.id)["teleop"]
        if p.is_arm:
            assert got is None, p.id
            continue
        src = p.teleop_base or p.teleop_walk
        assert got["kind"] == p.teleop
        assert {k: (a["speed"], a["limit"]) for k, a in got["axes"].items()} == {k: (a.speed, a.limit) for k, a in src.axes.items()}
        assert got["rate_hz"] == src.rate_hz
        assert got["stop"] == [dataclasses.asdict(o) for o in tc.stop_ops(p)]
        assert got["command_topics"] == [list(x) for x in tc.command_topics(p)]
        assert got["axis_keys"] == [list(x) for x in tc.AXIS_KEYS] and got["motion_keys"] == list(tc.MOTION_KEYS)
        assert (got["call_timeout_s"], got["head_period_s"], got["advertise_settle_s"]) == (
            tc.STOP_CALL_TIMEOUT_S, tc.HEAD_PERIOD_S, tc.ADVERTISE_SETTLE_S)
        assert got["reenable_limitation"] == tc.REENABLE_LIMITATION and got["no_delivery"] == tc.no_delivery_statement(p)
        if p.teleop_walk:
            w = got["walk"]
            assert w["param_template"] == dict(p.teleop_walk.param_template)
            assert [o["msg"] for o in w["start"]] == [o.msg for o in p.teleop_walk.start]
            assert [o["msg"] for o in w["enable"]] == [o.msg for o in p.teleop_walk.enable]
        assert (got["head"] is None) == (p.head is None)
        for axis, h in (p.head or {}).items():
            g = got["head"][axis]
            assert (g["topic"], g["min"], g["max"], g["rate"], g["template"]) == (h.topic, h.min, h.max, h.rate, dict(h.template))
            r, v = p.head_read(axis)
            assert g["read"] == {"read": r.id, "control": v.control, "field": v.field}
