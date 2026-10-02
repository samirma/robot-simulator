"""Generate robots_specs/rosmaster_x3_plus/model.xml from yahboomcar_X3plus.urdf (derived model).

Deterministic; stdlib only. See robots_specs/rosmaster_x3_plus/import.md for every step.
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _visual import stl_parts  # noqa: E402

FOLDER = Path(__file__).resolve().parents[2] / "rosmaster_x3_plus"
URDF = FOLDER / "yahboomcar_X3plus.urdf"
OUT = FOLDER / "model.xml"
PKG = "package://yahboomcar_description/"

# --- mecanum adaptation (see import.md) ---
WHEEL_R = 0.040            # firmware MECANUM_MAX_CIRCLE_MM 251.327 mm / pi
WHEEL_W = 0.036            # base_link.STL wheel band |y| 0.086..0.122 (estimate from mesh)
# wheel centres in base_link, measured from the wheels inside X3plus/visual/base_link.STL
WHEELS = {  # name: (x, y, z, roller-axis sign at the ground contact, see import.md)
    "front_left": (0.1053, 0.1042, -0.0389, -1),
    "front_right": (0.1053, -0.1042, -0.0389, +1),
    "back_left": (-0.1146, 0.1042, -0.0393, +1),
    "back_right": (-0.1146, -0.1042, -0.0393, -1),
}
N_ROLLERS = 12
ROLLER_R = 0.011
ROLLER_HALF = 0.0095
ROLLER_MASS = 0.004
HUB_MASS = 0.08
WHEEL_MAX = 17.5           # rad/s = 0.7 m/s firmware per-wheel clamp / 0.04 m
WHEEL_TORQUE = 1.0         # N m, estimate


def fmt(v):
    out = []
    for x in v:
        s = f"{float(x):.6g}"
        out.append("0" if s in ("-0", "0") else s)
    return " ".join(out)


def rpy_quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def floats(s, d="0 0 0"):
    return [float(x) for x in (s or d).split()]


def origin(el):
    o = el.find("origin")
    if o is None:
        return [0, 0, 0], (1, 0, 0, 0)
    return floats(o.get("xyz")), rpy_quat(*floats(o.get("rpy")))


def mesh_rel(fn):
    assert fn.startswith(PKG), fn
    return fn[len(PKG):]


urdf = ET.parse(URDF).getroot()
links = {l.get("name"): l for l in urdf.findall("link")}
joints = urdf.findall("joint")
children = {}
for j in joints:
    children.setdefault(j.find("parent").get("link"), []).append(j)

meshes = {}  # file -> name
OVERSIZE = []


def stl_faces(path):
    import struct
    with open(path, "rb") as f:
        head = f.read(84)
    return struct.unpack("<I", head[80:84])[0]


def mesh_name(rel, kind):
    if rel not in meshes:
        stem = rel.rsplit("/", 1)[1].rsplit(".", 1)[0]
        base = f"{stem}_{kind}"
        name, k = base, 1
        while name in meshes.values():
            k += 1
            name = f"{base}{k}"
        meshes[rel] = name
    return meshes[rel]


root = ET.Element("mujoco", model="rosmaster_x3_plus")
root.append(ET.Comment(
    " DERIVED MODEL - not manufacturer-provided. Converted from Yahboom's yahboomcar_X3plus.urdf "
    "(ROSMASTER-X3Plus_ROS1_code.zip!/yahboomcar_ws.zip, URDF sha256 17c7c8fa...) by the steps in "
    "import.md. Mecanum wheels, rollers, chassis collision boxes, actuators and sensor sites are "
    "adaptations/estimates listed there. "))
ET.SubElement(root, "compiler", angle="radian", meshdir=".", autolimits="true",
              inertiafromgeom="false", balanceinertia="true")
default = ET.SubElement(root, "default")
dcls = ET.SubElement(default, "default", {"class": "x3plus"})
ET.SubElement(ET.SubElement(dcls, "default", {"class": "visual"}), "geom",
              type="mesh", group="2", contype="0", conaffinity="0", density="0")
ET.SubElement(ET.SubElement(dcls, "default", {"class": "collision"}), "geom",
              group="3", contype="1", conaffinity="0", density="0", condim="3",
              friction="0.8 0.005 0.0001", rgba="0.8 0.2 0.2 0.4")
roller_cls = ET.SubElement(dcls, "default", {"class": "roller"})
ET.SubElement(roller_cls, "geom", type="capsule", group="3", contype="1", conaffinity="0",
              density="0", condim="3", friction="1.0 0.005 0.0001", rgba="0.2 0.2 0.2 1")
ET.SubElement(roller_cls, "joint", type="hinge", damping="1e-5", armature="1e-6")
servo_cls = ET.SubElement(dcls, "default", {"class": "servo"})
ET.SubElement(servo_cls, "joint", armature="0.005", damping="0.05")
ET.SubElement(servo_cls, "position", kp="20", dampratio="1")
wheel_cls = ET.SubElement(dcls, "default", {"class": "wheel"})
ET.SubElement(wheel_cls, "joint", type="hinge", axis="0 1 0", armature="0.001", damping="0.001")
ET.SubElement(wheel_cls, "velocity", kv="0.5",
              ctrlrange=f"{-WHEEL_MAX} {WHEEL_MAX}", forcerange=f"{-WHEEL_TORQUE} {WHEEL_TORQUE}")

asset = ET.SubElement(root, "asset")
world = ET.SubElement(root, "worldbody")

SERVO_JOINTS = []
MIMIC = []


def add_link(parent_el, lname, pos=None, quat=None):
    attrs = {"name": lname}
    if pos is not None:
        attrs["pos"] = fmt(pos)
    if quat is not None and any(abs(a - b) > 1e-12 for a, b in zip(quat, (1, 0, 0, 0))):
        attrs["quat"] = fmt(quat)
    body = ET.SubElement(parent_el, "body", attrs)
    return body


def fill_link(body, lname):
    link = links[lname]
    inert = link.find("inertial")
    if inert is not None:
        xyz, q = origin(inert)
        i = inert.find("inertia")
        ET.SubElement(body, "inertial", pos=fmt(xyz), mass=inert.find("mass").get("value"),
                      fullinertia=fmt([float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]))
    for kind in ("visual", "collision"):
        if lname == "base_link" and kind == "collision":
            continue  # replaced by boxes (import.md: its hull would include the wheels)
        for part in link.findall(kind):
            m = part.find("geometry/mesh")
            rel = mesh_rel(m.get("filename"))
            rels = [rel]
            if kind == "visual" and stl_faces(FOLDER / rel) > 200000:
                # MuJoCo's STL decoder refuses > 200000 faces: the official visual mesh is used
                # as its lossless parts (fetch_meshes.py; import.md, adaptation A3)
                rels = stl_parts(FOLDER, rel)
                OVERSIZE.append(lname)
            xyz, q = origin(part)
            for r in rels:
                g = {"class": kind, "type": "mesh", "mesh": mesh_name(r, "vis" if kind == "visual" else "col")}
                if any(xyz):
                    g["pos"] = fmt(xyz)
                if kind == "visual":
                    c = part.find("material/color")
                    if c is not None:
                        g["rgba"] = c.get("rgba")
                ET.SubElement(body, "geom", g)
    for j in children.get(lname, []):
        child = j.find("child").get("link")
        xyz, q = origin(j)
        cb = add_link(body, child, xyz, q)
        jt = j.get("type")
        if jt in ("revolute", "continuous"):
            ja = {"name": j.get("name"), "axis": fmt(floats(j.find("axis").get("xyz")))}
            lim = j.find("limit")
            if jt == "revolute":
                ja["range"] = f"{lim.get('lower')} {lim.get('upper')}"
            mim = j.find("mimic")
            if mim is not None:
                ja["damping"] = "0.001"
                ja["armature"] = "1e-5"
                MIMIC.append((j.get("name"), mim.get("joint"), float(mim.get("multiplier", "1"))))
            else:
                ja["class"] = "servo"
                SERVO_JOINTS.append((j.get("name"), lim))
            ET.SubElement(cb, "joint", ja)
        elif jt != "fixed":
            sys.exit(f"unhandled joint type {jt}")
        fill_link(cb, child)


fp = ET.SubElement(world, "body", name="base_footprint", childclass="x3plus")
ET.SubElement(fp, "freejoint", name="root")
# base_footprint -> base_link is the fixed base_joint of the URDF
bj = next(j for j in joints if j.get("name") == "base_joint")
bxyz, bq = origin(bj)
base = add_link(fp, "base_link", bxyz, bq)

# chassis collision boxes (estimates fitted to X3plus/visual/base_link.STL; import.md)
for name, pos, size in (
        ("chassis_col", (0.0, 0.0, 0.0175), (0.142, 0.076, 0.0625)),
        ("deck_col", (0.04, 0.0, 0.115), (0.097, 0.04, 0.035)),
        ("tower_post_col", (-0.145, 0.0, 0.235), (0.008, 0.083, 0.155)),
        ("tower_top_col", (-0.095, 0.0, 0.335), (0.06, 0.083, 0.055)),
):
    ET.SubElement(base, "geom", {"class": "collision", "name": name, "type": "box",
                                 "pos": fmt(pos), "size": fmt(size)})

fill_link(base, "base_link")

# IMU and lidar sites (frames of the boot's IMU and scan messages)
def find_body(el, name):
    for b in el.iter("body"):
        if b.get("name") == name:
            return b


ET.SubElement(find_body(base, "imu_link"), "site", name="imu_link", size="0.005", group="4")


laser_body = find_body(base, "laser_link")
# /laser_link -> /laser is yaw 6.28 rad (laser_astrapro_bringup.launch L11), i.e. 6.28 - 2 pi = -0.0032 rad
ET.SubElement(laser_body, "site", name="laser", quat=fmt(rpy_quat(0, 0, 6.28)), size="0.005", group="4")
cam_body = find_body(base, "camera_link")
# camera_link -> camera_color_frame: device extrinsic, estimated identity; optical frame rpy(-pi/2,0,-pi/2).
# A MuJoCo camera looks along its -Z with +Y up: image right = -y_link, image up = +z_link.
ET.SubElement(cam_body, "camera", name="camera_color_optical_frame", pos="0 0 0",
              xyaxes="0 -1 0 0 0 1", fovy="40.2", resolution="640 480")

# mecanum wheels with passive rollers
c45 = math.cos(math.pi / 4)
for wname, (x, y, z, s) in WHEELS.items():
    wb = ET.SubElement(base, "body", name=f"{wname}_wheel", pos=fmt((x, y, z)))
    ET.SubElement(wb, "joint", {"class": "wheel", "name": f"{wname}_joint"})
    ET.SubElement(wb, "inertial", pos="0 0 0", mass=str(HUB_MASS),
                  diaginertia=fmt((HUB_MASS * 0.02 ** 2 / 2 + HUB_MASS * WHEEL_W ** 2 / 12,
                                   HUB_MASS * 0.02 ** 2,
                                   HUB_MASS * 0.02 ** 2 / 2 + HUB_MASS * WHEEL_W ** 2 / 12)))
    ET.SubElement(wb, "geom", {"class": "visual", "type": "cylinder", "size": fmt((0.02, WHEEL_W / 2)),
                               "zaxis": "0 1 0", "rgba": "0.15 0.15 0.15 1"})
    d = WHEEL_R - ROLLER_R
    for k in range(N_ROLLERS):
        th = 2 * math.pi * k / N_ROLLERS
        # radial direction in the wheel's x-z plane (axle along y); th=0 points down (-z)
        rad = (math.sin(th), 0.0, -math.cos(th))
        tan = (math.cos(th), 0.0, math.sin(th))          # rolling direction at that roller
        axis = tuple(c45 * tan[i] + s * c45 * (0, 1, 0)[i] for i in range(3))
        pos = tuple(d * r for r in rad)
        rb = ET.SubElement(wb, "body", name=f"{wname}_roller{k}", pos=fmt(pos))
        ET.SubElement(rb, "joint", {"class": "roller", "name": f"{wname}_roller{k}_joint", "axis": fmt(axis)})
        I_ax = ROLLER_MASS * ROLLER_R ** 2 / 2
        ET.SubElement(rb, "inertial", pos="0 0 0", mass=str(ROLLER_MASS), diaginertia=fmt((I_ax, I_ax, I_ax)))
        a = [ROLLER_HALF * v for v in axis]
        ET.SubElement(rb, "geom", {"class": "roller", "size": fmt((ROLLER_R,)),
                                   "fromto": fmt([-a[0], -a[1], -a[2], a[0], a[1], a[2]])})

# mimic joints -> equality constraints
eq = ET.SubElement(root, "equality")
for jn, driver, mult in MIMIC:
    ET.SubElement(eq, "joint", joint1=jn, joint2=driver, polycoef=fmt((0, mult, 0, 0, 0)),
                  solref="0.005 1")

# no self collision between the fingers' linkage bodies is already given by conaffinity=0

act = ET.SubElement(root, "actuator")
for jn, lim in SERVO_JOINTS:
    ET.SubElement(act, "position", {"class": "servo", "name": jn, "joint": jn,
                                    "ctrlrange": f"{lim.get('lower')} {lim.get('upper')}",
                                    "forcerange": f"-{lim.get('effort')} {lim.get('effort')}"})
for wname in WHEELS:
    ET.SubElement(act, "velocity", {"class": "wheel", "name": f"{wname}_joint", "joint": f"{wname}_joint"})

# assets (insert before worldbody content usage; order in file does not matter to MuJoCo)
for rel, name in sorted(meshes.items(), key=lambda kv: kv[1]):
    ET.SubElement(asset, "mesh", name=name, file=rel)

# home keyframe: boot arm pose [90,145,0,45,90,30] deg -> (deg-90) rad, gripper interp -> -pi/2
# clamped to the URDF grip_joint lower limit -1.54 (import.md)
ET.indent(root, "  ")
OUT.write_text(ET.tostring(root, encoding="unicode") + "\n")
print("oversize visuals split into lossless parts:", OVERSIZE)
print("wrote", OUT, "servo joints:", [j for j, _ in SERVO_JOINTS], "mimic:", MIMIC)

# ---- second pass: the home keyframe (boot pose), computed on the compiled model ----
import mujoco  # noqa: E402

HOME = {"arm_joint1": 0.0, "arm_joint2": math.radians(145 - 90), "arm_joint3": math.radians(0 - 90),
        "arm_joint4": math.radians(45 - 90), "arm_joint5": 0.0, "grip_joint": -1.54}
mm = mujoco.MjModel.from_xml_path(str(OUT))
q = mm.qpos0.copy()
q[2] = 0.0029  # resting height of base_footprint over a floor (settled value, import.md)
for jn, v in HOME.items():
    q[mm.jnt_qposadr[mm.joint(jn).id]] = v
for jn, driver, mult in MIMIC:
    q[mm.jnt_qposadr[mm.joint(jn).id]] = mult * HOME[driver]
ctrl = [HOME.get(mm.actuator(i).name, 0.0) for i in range(mm.nu)]
kf = ET.SubElement(root, "keyframe")
ET.SubElement(kf, "key", name="home", qpos=fmt(q), ctrl=fmt(ctrl))
ET.indent(root, "  ")
OUT.write_text(ET.tostring(root, encoding="unicode") + "\n")
print("keyframe home written")
