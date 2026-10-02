"""Generate robots_specs/ainex/model.xml from robots_specs/ainex/ainex.urdf (derived model).

    python gen_ainex_model.py            # writes model.xml

Rules (import.md, "Conversion steps"): every URDF link a body at its joint origin, every
revolute joint a hinge with the URDF axis and limits, every visual mesh a visual geom and
every collision mesh a collision geom at its URDF origin, every URDF inertial as
`fullinertia`. Added (documented adaptations): freejoint on base_link, sole boxes under
each foot, position actuators, the head camera, the IMU site and the `home` keyframe.
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "ainex"
urdf = ET.parse(R / "ainex.urdf").getroot()


def f(v):
    s = f"{float(v):.9g}"
    return "0" if s in ("-0", "0") else s


def fl(vals):
    return " ".join(f(v) for v in vals)


def rpy_quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def origin(el):
    o = el.find("origin") if el is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return xyz, rpy


links = {l.get("name"): l for l in urdf.findall("link")}
joints = urdf.findall("joint")
children = {}
for j in joints:
    children.setdefault(j.find("parent").get("link"), []).append(j)
materials = {m.get("name"): m.find("color").get("rgba") for m in urdf.findall("material")
             if m.find("color") is not None}

# --- servo classes (estimates; import.md "Actuators") --------------------------------
# HX-35H 35 kg.cm (3.43 N.m) legs + shoulder pitch; HX-35HM shoulder roll; HX-12H
# 12 kg.cm (1.18 N.m) elbows; grippers and head: servo model not documented -> HX-12H assumed.
LEG = {"hip_yaw", "hip_roll", "hip_pitch", "knee", "ank_pitch", "ank_roll"}
def servo_class(jname):
    base = jname[2:] if jname[:2] in ("l_", "r_") else jname
    if base in LEG or base in ("sho_pitch", "sho_roll"):
        return "servo35"
    return "servo12"

# --- boot init pose (ainex_kinematics/config/init_pose.yaml) -------------------------
INIT = {"l_ank_pitch": 0.625, "l_ank_roll": 0.016, "l_hip_pitch": -0.828, "l_hip_roll": 0.016,
        "l_hip_yaw": 0.0, "l_knee": 1.192, "l_sho_pitch": 0.0, "l_sho_roll": 1.293,
        "l_el_pitch": -0.10, "l_el_yaw": -1.926, "l_gripper": 0.0,
        "r_ank_pitch": -0.625, "r_ank_roll": -0.016, "r_hip_pitch": 0.828, "r_hip_roll": -0.016,
        "r_hip_yaw": 0.0, "r_knee": -1.192, "r_sho_pitch": 0.0, "r_sho_roll": -1.293,
        "r_el_pitch": 0.10, "r_el_yaw": 1.926, "r_gripper": 0.0, "head_pan": 0.0, "head_tilt": 0.0}

# Foot sole: ank_roll_link mesh AABB (meshes/?_ank_roll_link.STL, link frame):
#   x [-0.0379, 0.0972], y r:[-0.052, 0.0241] l:[-0.024, 0.052], z min -0.0231
SOLE = {"r_ank_roll_link": (0.02965, -0.01395), "l_ank_roll_link": (0.02965, 0.0140)}
SOLE_HALF = (0.06755, 0.038, 0.002)
SOLE_Z = -0.0231 + 0.002

# Camera: frame "camera" (usb_cam camera_frame_id). URDF camera_link sits on body_link at
# xyz (0.0430140359, 0, 0.1523561209); the camera is in the head, so it is mounted on
# head_tilt_link at the pose that coincides with camera_link when head_pan = head_tilt = 0.
HP = (-0.00410029363273529, 0.0, 0.107800000000251)       # head_pan origin in body_link
HT = (0.00904594549071587, -0.0185733521405788, 0.0281579437975029)  # head_tilt in head_pan
CAM = (0.0430140359009206, 0.0, 0.152356120938238)
CAM_IN_TILT = tuple(CAM[i] - HP[i] - HT[i] for i in range(3))
FY, H = 270.0, 480
FOVY = 2 * math.degrees(math.atan((H / 2) / FY))

mj = ET.Element("mujoco", model="ainex")
ET.SubElement(mj, "compiler", angle="radian", meshdir=".", autolimits="true")
default = ET.SubElement(mj, "default")
dcls = ET.SubElement(default, "default", {"class": "ainex"})
ET.SubElement(dcls, "joint", damping="0.1", armature="0.005", frictionloss="0.02")
vis = ET.SubElement(dcls, "default", {"class": "visual"})
ET.SubElement(vis, "geom", type="mesh", contype="0", conaffinity="0", group="2")
col = ET.SubElement(dcls, "default", {"class": "collision"})
ET.SubElement(col, "geom", group="3", condim="3", friction="1.0 0.005 0.0001")
sole = ET.SubElement(dcls, "default", {"class": "sole"})
ET.SubElement(sole, "geom", type="box", group="3", condim="4", friction="1.5 0.02 0.0001",
              priority="1", solref="0.004 1")
for name, kp, frc in (("servo35", 30, 3.43), ("servo12", 10, 1.18)):
    c = ET.SubElement(dcls, "default", {"class": name})
    ET.SubElement(c, "position", kp=str(kp), dampratio="1", forcerange=f"-{frc} {frc}")

asset = ET.SubElement(mj, "asset")
for lname, l in links.items():
    v = l.find("visual/geometry/mesh")
    if v is not None:
        fn = v.get("filename").replace("package://ainex_description/", "")
        ET.SubElement(asset, "mesh", name=lname, file=fn)
for mname, rgba in materials.items():
    ET.SubElement(asset, "material", name=mname, rgba=rgba)

world = ET.SubElement(mj, "worldbody")
HOME_Z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2029
HOME_PITCH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2609658522284916   # soles flat: torso leans forward
root = ET.SubElement(world, "body", name="base_link", childclass="ainex", pos=fl((0, 0, HOME_Z)),
                     quat=fl((math.cos(HOME_PITCH / 2), 0, math.sin(HOME_PITCH / 2), 0)))
ET.SubElement(root, "freejoint", name="root")
order = []


def emit(lname, body):
    l = links[lname]
    inertial = l.find("inertial")
    if inertial is not None:
        xyz, rpy = origin(inertial)
        i = inertial.find("inertia").attrib
        attrs = {"pos": fl(xyz), "mass": f(inertial.find("mass").get("value")),
                 "fullinertia": fl([i["ixx"], i["iyy"], i["izz"], i["ixy"], i["ixz"], i["iyz"]])}
        if any(rpy):
            attrs["quat"] = fl(rpy_quat(*rpy))
        ET.SubElement(body, "inertial", attrs)
    for kind in ("visual", "collision"):
        for part in l.findall(kind):
            m = part.find("geometry/mesh")
            if m is None:
                continue
            xyz, rpy = origin(part)
            a = {"class": kind, "type": "mesh", "mesh": lname}
            if any(xyz):
                a["pos"] = fl(xyz)
            if any(rpy):
                a["quat"] = fl(rpy_quat(*rpy))
            if kind == "visual":
                mat = part.find("material")
                if mat is not None and mat.get("name") in materials:
                    a["material"] = mat.get("name")
            else:
                a["name"] = f"{lname}_collision"
            ET.SubElement(body, "geom", a)
    if lname in SOLE:
        cx, cy = SOLE[lname]
        ET.SubElement(body, "geom", {"name": f"{lname[:2]}sole", "class": "sole",
                                     "pos": fl((cx, cy, SOLE_Z)), "size": fl(SOLE_HALF)})
    if lname == "head_tilt_link":
        ET.SubElement(body, "camera", name="camera", pos=fl(CAM_IN_TILT),
                      xyaxes="0 -1 0 0 0 1", fovy=f(FOVY), resolution="640 480")
    if lname == "imu_link":
        ET.SubElement(body, "site", name="imu_link", size="0.005", group="4")
    for j in children.get(lname, []):
        child = j.find("child").get("link")
        xyz, rpy = origin(j)
        a = {"name": child}
        if any(xyz):
            a["pos"] = fl(xyz)
        if any(rpy):
            a["quat"] = fl(rpy_quat(*rpy))
        sub = ET.SubElement(body, "body", a)
        if j.get("type") in ("revolute", "continuous"):
            lim = j.find("limit")
            ja = {"name": j.get("name"), "axis": fl(float(v) for v in j.find("axis").get("xyz").split())}
            if j.get("type") == "revolute":
                ja["range"] = fl((lim.get("lower"), lim.get("upper")))
            ET.SubElement(sub, "joint", ja)
            order.append((j.get("name"), ja.get("range")))
        emit(child, sub)


emit("base_link", root)

contact = ET.SubElement(mj, "contact")
# convex hulls of these non-adjacent links overlap in the URDF meshes themselves (import.md)
for a, b in (("r_hip_yaw_link", "r_hip_pitch_link"), ("l_hip_yaw_link", "l_hip_pitch_link"),
             ("r_knee_link", "r_ank_roll_link"), ("l_knee_link", "l_ank_roll_link")):
    ET.SubElement(contact, "exclude", body1=a, body2=b)

act = ET.SubElement(mj, "actuator")
for jn, rng in order:
    ET.SubElement(act, "position", {"name": jn, "joint": jn, "class": servo_class(jn),
                                     "ctrlrange": rng})

sensor = ET.SubElement(mj, "sensor")
ET.SubElement(sensor, "accelerometer", name="imu_accel", site="imu_link")
ET.SubElement(sensor, "gyro", name="imu_gyro", site="imu_link")
ET.SubElement(sensor, "magnetometer", name="imu_mag", site="imu_link")

# home keyframe: root height from the standing check (import.md), joints = init pose
kf = ET.SubElement(mj, "keyframe")
# the spawn pose: the boot's init pose with the arms relaxed down at the sides (the boot's
# own arm values, shoulder roll +-1.293 with the elbows yawed, hold them raised in the air)
ARMS_DOWN = {"l_sho_pitch": 0.0, "l_sho_roll": -1.45, "l_el_pitch": 0.0, "l_el_yaw": 0.0,
             "r_sho_pitch": 0.0, "r_sho_roll": 1.45, "r_el_pitch": 0.0, "r_el_yaw": 0.0}
HOME = {**INIT, **ARMS_DOWN}
qpos = [0, 0, HOME_Z, math.cos(HOME_PITCH / 2), 0, math.sin(HOME_PITCH / 2), 0] + [HOME[jn] for jn, _ in order]
ET.SubElement(kf, "key", name="home", qpos=fl(qpos), ctrl=fl(HOME[jn] for jn, _ in order))

ET.indent(mj, "  ")
body = ET.tostring(mj, encoding="unicode")
header = """<!--
  DERIVED MuJoCo model of the Hiwonder AiNex - NOT manufacturer-provided.
  Converted from robots_specs/ainex/ainex.urdf (expansion of the official xacro in
  https://github.com/Hiwonder/ainex @ e8fe2a816797cf83054135160df5a82ec3596a69).
  Adaptations and estimates (freejoint, foot soles, servo actuators, head camera, IMU site,
  home keyframe) are listed in import.md. Meshes: robots_specs/tools/fetch_meshes.py.
-->
"""
(R / "model.xml").write_text(header + body + "\n")
print("wrote", R / "model.xml", "fovy", FOVY, "cam_in_tilt", CAM_IN_TILT, "joints", len(order))
