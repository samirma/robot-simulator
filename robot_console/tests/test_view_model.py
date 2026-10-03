"""The profiles' `model` sections behind the page's 3D model (console spec §2.2): traced to the
pinned vendor sources and bound to the catalog fields that command each joint with the
documented unit mapping (independence from the simulator: test_independence.py)."""

import math
import re

import pytest

from robot_console.profiles import SUPPORTED_IDS, all_profiles_json, load

TRACE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+):(?P<path>[^#\s]+)#L\d+")

# Every catalog field that commands a joint, and that joint's name in the vendor URDF; listed
# here independently of the profiles' model sections.
_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_COBOT = ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3", "joint5_to_joint4", "joint6_to_joint5",
          "joint6output_to_joint6"]
COMMANDED = {
    "so101": {**{("arm_trajectory", a): f"{a}_joint" for a in _ARM}, ("gripper", "position"): "gripper_joint"},
    "mycobot280": {**{("arm_gripper_target", f"j{i + 1}"): n for i, n in enumerate(_COBOT)},
                   **{("set_angles", f"j{i + 1}"): n for i, n in enumerate(_COBOT)},
                   ("arm_gripper_target", "gripper"): "gripper_controller"},
    "ainex": {("head_pan", "position"): "head_pan", ("head_tilt", "position"): "head_tilt"},
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
    """Console spec §2.1 (amended 2026-10-02): a negative head_pan turns the head to the robot's left."""
    pan = next(j for j in load("ainex").raw["model"]["joints"] if j["name"] == "head_pan")
    assert pan["axis"] == [0, 0, -1]
    label = field("ainex", "head_pan", "position").label
    assert "- = left" in label and "+ = left" not in label


def test_the_page_receives_every_model():
    j = all_profiles_json()
    assert [p["id"] for p in j["profiles"]] == list(SUPPORTED_IDS)
    for p in j["profiles"]:
        assert p["model"] == load(p["id"]).raw["model"]

