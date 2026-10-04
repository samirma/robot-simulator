"""The page's 3D model follows the robot's URDF (console spec §2.2: "the robot urdf should be used
as reference"): each profile's `model` rows are the kinematics of the pinned vendor file that its
`model.source` names -- joint names, types, parent and child links, origins, axes, limits, mimic
joints, the root link and the inertial origins links are drawn to -- and every movable joint of that
file is drawn. The vendor side is verbatim excerpts of the pinned files packaged with these tests
(urdf/<id>.xml), read here with URDF semantics; live/test_live_urdf.py (opt in) checks the excerpts
against the pinned files themselves. Nothing here reads a sibling source tree (root spec §1.4)."""

import os
import re
import xml.etree.ElementTree as ET

import pytest

from robot_console.profiles import SUPPORTED_IDS, load

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-6        # m, rad: the profiles round the vendor numbers (CAD-export noise such as 4.18e-17 is 0)
TRACE = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+):(?P<path>[^#\s]+)#L(?P<a>\d+)(?:-L(?P<b>\d+))?")


def span(text):
    a, _, b = text.partition("-")
    return int(a), int(b or a)


def nums(text, n=3):
    v = [float(x) for x in text.split()]
    assert len(v) == n, text
    return v


def urdf_joint(el, sub):
    """A URDF <joint> with the URDF defaults: no <origin> or attribute = 0, no <axis> = 1 0 0,
    <mimic> multiplier 1 and offset 0. Origins are read when used (an undrawn frame may be xacro math)."""
    o, ax, lim, mim = (el.find(t) for t in ("origin", "axis", "limit", "mimic"))
    j = {"name": sub(el.get("name")), "type": el.get("type"), "parent": sub(el.find("parent").get("link")),
         "child": sub(el.find("child").get("link")),
         "origin": lambda: (nums(sub(o.get("xyz", "0 0 0"))) if o is not None else [0.0] * 3,
                            nums(sub(o.get("rpy", "0 0 0"))) if o is not None else [0.0] * 3)}
    if j["type"] != "fixed":
        j["axis"] = nums(sub(ax.get("xyz"))) if ax is not None else [1.0, 0.0, 0.0]
    if j["type"] in ("revolute", "prismatic"):
        j["limits"] = [float(sub(lim.get(k, "0"))) for k in ("lower", "upper")]
    if mim is not None:
        j["mimic"] = {"joint": sub(mim.get("joint")), "multiplier": float(mim.get("multiplier", 1)),
                      "offset": float(mim.get("offset", 0))}
    return j


def static_tf(el):
    """A launch file's tf static_transform_publisher: args x y z yaw pitch roll frame child period."""
    assert el.get("pkg") == "tf" and el.get("type") == "static_transform_publisher"
    a = el.get("args").split()
    assert len(a) == 9, a
    xyz, (yaw, pitch, roll) = [float(x) for x in a[:3]], (float(x) for x in a[3:6])
    return {"name": el.get("name"), "type": "fixed", "parent": a[6].lstrip("/"), "child": a[7].lstrip("/"),
            "origin": lambda: (xyz, [roll, pitch, yaw])}


def vendor(pid):
    """The excerpts of urdf/<pid>.xml: every joint of the vendor files, with the file, cited lines and
    whether the model draws it; and the inertial origins, by link."""
    files, joints, inertial = [], {}, {}
    for f in ET.parse(os.path.join(HERE, "urdf", f"{pid}.xml")).getroot().iter("file"):
        files.append(f)
        args = {a.get("name"): a.get("value") for a in f.findall("arg")}

        def sub(text, args=args):                    # the xacro arguments of the boot, nothing else
            out = re.sub(r"\$\{(\w+)\}", lambda m: args[m.group(1)] if m.group(1) in args else m.group(0), text)
            assert "${" not in out, f"{pid}: unexpanded xacro in {text!r}"
            return out
        for e in f.findall("excerpt"):
            (el,) = list(e)
            where = {"file": f, "lines": span(e.get("lines"))}
            if e.get("inertial"):
                assert el.tag == "origin"
                inertial[e.get("inertial")] = dict(where, xyz=nums(el.get("xyz", "0 0 0")))
                continue
            j = urdf_joint(el, sub) if el.tag == "joint" else static_tf(el)
            assert j["name"] not in joints, j["name"]
            joints[j["name"]] = dict(j, drawn=e.get("drawn") != "false", **where)
    return files, joints, inertial


def cites(pid, row, where):
    """The row's `source` names the excerpt's file at the excerpt's lines."""
    t = TRACE.match(str(row.get("source", "")))
    f = where["file"]
    assert t and (t["key"], t["path"]) == (f.get("source"), f.get("path")), (pid, row.get("source"))
    return int(t["a"]), int(t["b"] or t["a"])


def close(a, b):
    assert all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in a), a   # numbers, not YAML strings
    return len(a) == len(b) and all(abs(x - y) <= TOL for x, y in zip(a, b))


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_the_excerpts_are_of_the_model_source_at_the_pinned_revision(pid):
    p = load(pid)
    files, _, _ = vendor(pid)
    assert f'{files[0].get("source")}:{files[0].get("path")}' == p.raw["model"]["source"]
    for f in files:
        src = p.sources[f.get("source")]
        assert (f.get("revision"), f.get("archive_sha256")) == (src.get("revision"), src.get("sha256")), f.attrib


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_model_joints_are_the_vendor_joints(pid):
    m = load(pid).raw["model"]
    _, joints, _ = vendor(pid)
    assert sorted(r["name"] for r in m["joints"]) == sorted(n for n, j in joints.items() if j["drawn"])
    for r in m["joints"]:
        j = joints[r["name"]]
        assert (r["type"], r["parent"], r["child"]) == (j["type"], j["parent"], j["child"]), r["name"]
        xyz, rpy = j["origin"]()
        assert close(r["xyz"], xyz) and close(r["rpy"], rpy), (r["name"], r["xyz"], r["rpy"], xyz, rpy)
        assert ("axis" in r) == ("axis" in j) and ("axis" not in r or close(r["axis"], j["axis"])), r["name"]
        assert ("min" in r or "max" in r) == ("limits" in j), r["name"]
        if "limits" in j:
            assert close([r["min"], r["max"]], j["limits"]), (r["name"], r["min"], r["max"], j["limits"])
        assert r.get("mimic") == j.get("mimic"), r["name"]
        assert cites(pid, r, j) == j["lines"], r["name"]


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_only_fixed_frames_are_left_out_and_the_root_is_the_vendor_root(pid):
    """An undrawn joint is a fixed leaf frame (camera, imu), or the identity above the model's root
    (the boot's world link of the SO-101 bringup)."""
    m = load(pid).raw["model"]
    _, joints, _ = vendor(pid)
    children = {j["child"] for j in joints.values()}
    (link,) = {j["parent"] for j in joints.values()} - children
    above = set()
    while link != m["root"]:
        (j,) = [j for j in joints.values() if j["parent"] == link]
        assert not j["drawn"] and j["type"] == "fixed" and j["origin"]() == ([0.0] * 3, [0.0] * 3), j["name"]
        above.add(j["name"])
        link = j["child"]
    parents = {j["parent"] for j in joints.values()}
    for n, j in joints.items():
        if not j["drawn"] and n not in above:
            assert j["type"] == "fixed" and j["child"] not in parents, n


def test_every_ainex_urdf_joint_is_commanded_within_its_urdf_limits():
    """All 24 movable joints of the vendor URDF can be edited on the page (each through its servo's
    control, in the joint's own rad), and no control's limits leave the URDF's."""
    p = load("ainex")
    _, joints, _ = vendor("ainex")
    rows = {r["name"]: r for r in p.raw["model"]["joints"]}
    movable = {n: j for n, j in joints.items() if j["type"] != "fixed"}
    assert len(movable) == 24
    for n, j in movable.items():
        (b,) = rows[n]["command"]
        c = next(c for c in p.controls if c.id == b["control"])
        f = next(f for f in c.fields if f.name == b["field"])
        assert f.unit == "rad" and "scale" not in b and "offset" not in b, n
        lo, hi = j["limits"]
        assert lo - TOL <= f.min < f.max <= hi + TOL, (n, f.min, f.max, j["limits"])


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_tips_are_the_vendor_inertial_origins(pid):
    """Terminal links are drawn to their centre of mass. (The boxes bound the vendor meshes: drawing,
    not kinematics.)"""
    _, _, inertial = vendor(pid)
    tips = load(pid).raw["model"].get("tips") or []
    assert {t["link"] for t in tips} == set(inertial)
    for t in tips:
        i = inertial[t["link"]]
        assert close(t["xyz"], i["xyz"]), (t["link"], t["xyz"], i["xyz"])
        a, b = cites(pid, t, i)
        assert a <= i["lines"][0] and i["lines"][1] == b, (t["link"], t["source"], i["lines"])
