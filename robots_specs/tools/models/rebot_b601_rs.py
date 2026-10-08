"""Generate robots_specs/rebot_b601_rs/model.xml from ReBot_Arm_RS.urdf (derived model).

Deterministic; standard library only. See robots_specs/rebot_b601_rs/import.md for every step
and the source line of every figure below.
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[2] / "rebot_b601_rs"
URDF = FOLDER / "ReBot_Arm_RS.urdf"
OUT = FOLDER / "model.xml"
PKG = "package://rebotarm_bringup/"          # resolves to the robot folder (import.md, step 1)
MUJOCO_COLLISION = "description/RS/meshes/mujoco_collision/"

# --- arm actuation: the boot's MIT gains (rebotarm_hardware.yaml models.rs.overrides.control
# mit_kp / mit_kd, equal to the SDK's rebotarm_rs.yaml joints[].MIT); import.md, "Actuation"
MIT_KP = {"joint1": 50.0, "joint2": 150.0, "joint3": 150.0, "joint4": 50.0, "joint5": 50.0, "joint6": 50.0}
MIT_KD = {"joint1": 3.0, "joint2": 10.0, "joint3": 10.0, "joint4": 5.0, "joint5": 4.0, "joint6": 4.0}

# --- gripper: one RS-00 motor (SDK joint "gripper") turning one pinion between two racks
GRIPPER = "gripper"            # actuator and tendon name: the SDK's gripper joint
GRIPPER_UNIT_RAD = 1.0         # motor output rad per gripper command unit (the wire's unit: rad)
GRIPPER_CLOSE_RAD, GRIPPER_OPEN_RAD = 0.0, 5.0   # rebotarm_hardware.yaml rs gripper.position_limits
GRIPPER_KP, GRIPPER_KD = 50.0, 4.0               # SDK rebotarm_rs.yaml joint "gripper" MIT kp / kd
GRIPPER_TORQUE = 14.0          # N m, RS-00 (URDF effort of the RS-00 joints joint4..joint6)
PINION_R = 0.008               # m: module 1 x 16 teeth / 2 (hardware BOM), finger travel per motor rad
FINGERS = ("gripper_joint1", "gripper_joint2")
# Seeed's detailed MuJoCo finger contact (Rebot_Arm_description/README.md, "MuJoCo"): the parts are
# expressed in gripper_end with the gripper closed; the RS left finger uses the "right" files.
FINGER_COLLISION = {
    "gripper_left": ["right_finger_front", "right_finger_mid", "right_finger_rear",
                     "colored_right_finger_carriage_grey", "colored_right_finger_travel_stop_yellow"],
    "gripper_right": ["left_finger_front", "left_finger_mid", "left_finger_rear",
                      "colored_left_finger_carriage_grey", "colored_left_finger_travel_stop_yellow"],
}

# --- keyframe home: safe-home pose (every arm joint 0) with the gripper closed (import.md)
HOME_ARM = {j: 0.0 for j in MIT_KP}
HOME_GRIPPER_RAD = GRIPPER_CLOSE_RAD


def fmt(v):
    out = []
    for x in v:
        s = f"{float(x):.12g}"
        out.append("0" if s in ("-0", "0") else s)
    return " ".join(out)


def rpy_quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def quat_rotate(q, v):
    w, x, y, z = q
    r = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return [sum(r[i][k] * v[k] for k in range(3)) for i in range(3)]


def floats(s, d="0 0 0"):
    return [float(x) for x in (s or d).split()]


def origin(el):
    """(xyz text or None, quaternion or None) of an element's <origin>, None when identity."""
    o = el.find("origin")
    if o is None:
        return None, None
    xyz = o.get("xyz", "0 0 0")
    rpy = floats(o.get("rpy"))
    q = rpy_quat(*rpy) if any(rpy) else None
    return (xyz if any(floats(xyz)) else None), q


def mesh_rel(fn):
    assert fn.startswith(PKG), fn
    return fn[len(PKG):]


urdf = ET.parse(URDF).getroot()
links = {l.get("name"): l for l in urdf.findall("link")}
joints = {j.get("name"): j for j in urdf.findall("joint")}
children = {}
for j in urdf.findall("joint"):
    children.setdefault(j.find("parent").get("link"), []).append(j)
materials = [(m.get("name"), m.find("color").get("rgba")) for m in urdf.findall("material")]
root_link = (set(links) - {j.find("child").get("link") for j in joints.values()}).pop()
assert root_link == "base_link", root_link

meshes = {}  # file -> name (the file stem; every stem of the 42 RS meshes is unique)


def mesh_name(rel):
    if rel not in meshes:
        name = rel.rsplit("/", 1)[1].rsplit(".", 1)[0]
        assert name not in meshes.values(), rel
        meshes[rel] = name
    return meshes[rel]


root = ET.Element("mujoco", model="rebot_b601_rs")
root.append(ET.Comment(
    " DERIVED MODEL - not manufacturer-provided. Converted from Seeed's official ReBot_Arm_RS.urdf "
    "(Seeed-Projects/reBotArmController_ROS2 @ 7506dc0fc5c2a7787dcb974d89d13f9f49d16d61, "
    "src/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf; equal to reBot-DevArm @ 0dfb49b "
    "apart from mesh URIs) by robots_specs/tools/models/rebot_b601_rs.py, following "
    "robots_specs/rebot_b601_rs/import.md. Kinematics, inertials, meshes, materials and limits are "
    "the URDF's. Adaptations listed in import.md: Seeed's mujoco_collision finger parts as finger "
    "contact, the coupled rack-and-pinion gripper (tendon + equality, one actuator in motor rad), "
    "position actuators with the boot's MIT gains, the boot's gravity feed-forward as gravcomp, "
    "collision filtering, the gripper_end site and the home keyframe. "))
ET.SubElement(root, "compiler", angle="radian", meshdir=".", autolimits="true",
              inertiafromgeom="false")
default = ET.SubElement(root, "default")
dcls = ET.SubElement(default, "default", {"class": "rebot"})
ET.SubElement(ET.SubElement(dcls, "default", {"class": "visual"}), "geom",
              type="mesh", group="2", contype="0", conaffinity="0", density="0")
ET.SubElement(ET.SubElement(dcls, "default", {"class": "collision"}), "geom",
              type="mesh", group="3", contype="1", conaffinity="0", density="0",
              rgba="0.8 0.2 0.2 0.4")
ET.SubElement(ET.SubElement(dcls, "default", {"class": "finger"}), "geom",
              type="mesh", group="3", contype="1", conaffinity="0", density="0", condim="4",
              friction="1 0.005 0.0001", rgba="0.2 0.5 0.9 0.4")

asset = ET.SubElement(root, "asset")
world = ET.SubElement(root, "worldbody")

ARM = []      # (joint name, URDF <limit>)


def add_geom(body, cls, rel, pos=None, quat=None, material=None):
    g = {"class": cls, "mesh": mesh_name(rel)}
    if pos is not None:
        g["pos"] = pos
    if quat is not None:
        g["quat"] = fmt(quat)
    if material is not None:
        g["material"] = material
    ET.SubElement(body, "geom", g)


def fill_link(body, lname):
    link = links[lname]
    inert = link.find("inertial")
    xyz, q = origin(inert)
    assert q is None, f"{lname}: rotated inertial frame"
    i = inert.find("inertia")
    ET.SubElement(body, "inertial", pos=xyz or "0 0 0", mass=inert.find("mass").get("value"),
                  fullinertia=" ".join(i.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
    for part in link.findall("visual"):
        xyz, q = origin(part)
        add_geom(body, "visual", mesh_rel(part.find("geometry/mesh").get("filename")), xyz, q,
                 part.find("material").get("name"))
    if lname in FINGER_COLLISION:
        # the parts are in gripper_end with the finger at 0 (closed): place them by the inverse of
        # the finger joint's origin (import.md, adaptation A1)
        j = next(j for j in joints.values() if j.find("child").get("link") == lname)
        o = j.find("origin")
        p, (w, x, y, z) = floats(o.get("xyz")), rpy_quat(*floats(o.get("rpy")))
        inv = (w, -x, -y, -z)
        pos = fmt([-c for c in quat_rotate(inv, p)])
        for stem in FINGER_COLLISION[lname]:
            add_geom(body, "finger", f"{MUJOCO_COLLISION}{stem}.stl", pos, inv)
    else:
        for part in link.findall("collision"):
            xyz, q = origin(part)
            add_geom(body, "collision", mesh_rel(part.find("geometry/mesh").get("filename")), xyz, q)
    for j in children.get(lname, []):
        child = j.find("child").get("link")
        xyz, q = origin(j)
        attrs = {"name": child}
        if xyz is not None:
            attrs["pos"] = xyz
        if q is not None:
            attrs["quat"] = fmt(q)
        attrs["gravcomp"] = "1"      # the boot's gravity feed-forward (import.md, A3)
        cb = ET.SubElement(body, "body", attrs)
        jt = j.get("type")
        lim = j.find("limit")
        if jt == "revolute":
            name = j.get("name")
            ET.SubElement(cb, "joint", {
                "name": name, "type": "hinge", "axis": j.find("axis").get("xyz"),
                "range": f"{lim.get('lower')} {lim.get('upper')}",
                "damping": fmt([MIT_KD[name]]),
                "actuatorfrcrange": f"-{lim.get('effort')} {lim.get('effort')}",
                "actuatorgravcomp": "true"})
            ARM.append((name, lim))
        elif jt == "prismatic":
            name = j.get("name")
            assert name in FINGERS, name
            ET.SubElement(cb, "joint", {
                "name": name, "type": "slide", "axis": j.find("axis").get("xyz"),
                "range": f"{lim.get('lower')} {lim.get('upper')}",
                # the motor's MIT kd seen at each finger (two racks share it): kd / (2 r^2)
                "damping": fmt([GRIPPER_KD / (2 * PINION_R ** 2)]),
                "actuatorfrcrange": f"-{lim.get('effort')} {lim.get('effort')}"})
        elif jt != "fixed":
            sys.exit(f"unhandled joint type {jt}")
        fill_link(cb, child)
        if child == "gripper_end":
            ET.SubElement(cb, "site", name="gripper_end", size="0.005", group="4")


base = ET.SubElement(world, "body", name=root_link, childclass="rebot")
fill_link(base, root_link)

# rack and pinion: both fingers move together; tendon length = motor angle in command units
tendon = ET.SubElement(ET.SubElement(root, "tendon"), "fixed", name=GRIPPER)
coef = 1.0 / (2 * PINION_R * GRIPPER_UNIT_RAD)
for jn in FINGERS:
    ET.SubElement(tendon, "joint", joint=jn, coef=fmt([coef]))
eq = ET.SubElement(root, "equality")
ET.SubElement(eq, "joint", name="gripper_rack", joint1=FINGERS[1], joint2=FINGERS[0],
              polycoef="0 1 0 0 0", solref="0.005 1")

act = ET.SubElement(root, "actuator")
for jn, lim in ARM:
    ET.SubElement(act, "position", {"name": jn, "joint": jn, "kp": fmt([MIT_KP[jn]]),
                                    "ctrlrange": f"{lim.get('lower')} {lim.get('upper')}",
                                    "forcerange": f"-{lim.get('effort')} {lim.get('effort')}"})
ET.SubElement(act, "position", {
    "name": GRIPPER, "tendon": GRIPPER, "kp": fmt([GRIPPER_KP * GRIPPER_UNIT_RAD ** 2]),
    "ctrlrange": fmt([GRIPPER_CLOSE_RAD / GRIPPER_UNIT_RAD, GRIPPER_OPEN_RAD / GRIPPER_UNIT_RAD]),
    "forcerange": fmt([-GRIPPER_TORQUE * GRIPPER_UNIT_RAD, GRIPPER_TORQUE * GRIPPER_UNIT_RAD])})

for name, rgba in materials:
    ET.SubElement(asset, "material", name=name, rgba=rgba)
for rel, name in sorted(meshes.items(), key=lambda kv: kv[1]):
    ET.SubElement(asset, "mesh", name=name, file=rel)

# keyframe home: qpos in tree order (joint1..joint6, then the fingers), ctrl in actuator order
finger_q = HOME_GRIPPER_RAD * PINION_R
qpos = [HOME_ARM[jn] for jn, _ in ARM] + [finger_q for _ in FINGERS]
ctrl = [HOME_ARM[jn] for jn, _ in ARM] + [HOME_GRIPPER_RAD / GRIPPER_UNIT_RAD]
ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home", qpos=fmt(qpos), ctrl=fmt(ctrl))

ET.indent(root, "  ")
OUT.write_text(ET.tostring(root, encoding="unicode") + "\n")
print("wrote", OUT, "arm joints:", [j for j, _ in ARM], "meshes:", len(meshes))
