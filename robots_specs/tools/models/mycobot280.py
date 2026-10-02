"""Generator for robots_specs/mycobot280/model.xml (steps recorded in import.md).

    python3 robots_specs/tools/models/mycobot280.py   (after fetch_meshes.py mycobot280)
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _visual import dae_parts  # noqa: E402

R = Path(__file__).resolve().parents[2] / "mycobot280"
urdf = ET.parse(R / "mycobot_280_pi_adaptive_gripper.urdf").getroot()
links = {l.get("name"): l for l in urdf.findall("link")}
joints = urdf.findall("joint")
children = {}
for j in joints:
    children.setdefault(j.find("parent").get("link"), []).append(j)


def fmt(vals):
    out = []
    for v in vals:
        s = f"{float(v):.6g}"
        out.append("0" if s in ("-0", "0") else s)
    return " ".join(out)


def quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def origin(el):
    o = el.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return xyz, rpy


def pose_attrs(xyz, rpy):
    a = {}
    if any(abs(v) > 0 for v in xyz):
        a["pos"] = fmt(xyz)
    if any(abs(v) > 0 for v in rpy):
        a["quat"] = fmt(quat(*rpy))
    return a


# Inertials of the moving arm links: official myCobot 280 JetsonNano MJCF
# (elephantrobotics/mycobot_mujoco 72b7a1b0, xml/mycobot_280jn_mujoco.xml), same link frames.
JN = {
    "joint2": dict(pos="-4.46252e-07 -0.0048982 -0.0203936", quat="0.988685 0.150004 1.75753e-05 -6.19332e-05",
                   mass="0.153266", diaginertia="0.00010997 0.000104609 4.89092e-05"),
    "joint3": dict(pos="-0.0548658 -3.79418e-06 0.0581352", quat="0.50338 0.49662 0.497294 0.502669",
                   mass="0.4", diaginertia="0.000569683 0.000554829 8.82752e-05"),
    "joint4": dict(pos="-0.0454441 4.51021e-07 0.00478273", quat="0.508392 0.491471 0.491339 0.508508",
                   mass="0.219069", diaginertia="0.000359328 0.000340078 6.35956e-05"),
    "joint5": dict(pos="1.49997e-05 -0.00761485 -0.00688111", quat="0.903244 0.429125 -0.00118023 0.00112811",
                   mass="0.0576212", diaginertia="1.88765e-05 1.81573e-05 1.0565e-05"),
    "joint6": dict(pos="-3.34552e-08 0.00487808 -0.00751659", quat="0.395895 0.918296 6.92557e-06 2.44425e-06",
                   mass="0.0904837", diaginertia="4.06947e-05 3.23387e-05 2.57661e-05"),
    "joint6_flange": dict(pos="-6.68861e-07 -1.54403e-07 -0.00514555", quat="0.276336 0.650875 -0.27633 0.650877",
                          mass="0.0121397", diaginertia="1.94827e-06 1.13494e-06 1.13491e-06"),
}
# Estimates (no official figure): mass assigned to the link's collision hull, inertia from its shape.
EST_MASS = {"g_base": 0.10, "joint1": 0.25,                      # base plate; Pi base housing
            "gripper_base": 0.072,                                # 110 g gripper total (manufacturer)
            "gripper_left1": 0.005, "gripper_right1": 0.005,
            "gripper_left2": 0.004, "gripper_right2": 0.004,
            "gripper_left3": 0.010, "gripper_right3": 0.010}

ARM_JOINTS = ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3", "joint5_to_joint4",
              "joint6_to_joint5", "joint6output_to_joint6"]
# Estimated actuator torque limits (N m): see import.md.
FORCE = {"joint2_to_joint1": 2.0, "joint3_to_joint2": 3.0, "joint4_to_joint3": 3.0,
         "joint5_to_joint4": 1.5, "joint6_to_joint5": 1.0, "joint6output_to_joint6": 1.0,
         "gripper_controller": 0.3}

mj = ET.Element("mujoco", {"model": "mycobot280"})
mj.append(ET.Comment(
    " DERIVED MODEL - not manufacturer-provided. Converted from the official myCobot 280 Pi +\n"
    "       adaptive gripper URDF (elephantrobotics/mycobot_ros2 d42ff61a, branch humble,\n"
    "       mycobot_description/urdf/mycobot_280_pi/mycobot_280_pi_adaptive_gripper.urdf).\n"
    "       Conversion steps, estimates and adaptations: import.md in this folder. "))
ET.SubElement(mj, "compiler", {"angle": "radian", "meshdir": ".", "autolimits": "true"})
default = ET.SubElement(mj, "default")
top = ET.SubElement(default, "default", {"class": "mycobot280"})
ET.SubElement(top, "joint", {"armature": "0.005", "damping": "0.05", "frictionloss": "0.01"})
ET.SubElement(top, "position", {"kp": "80", "dampratio": "1"})
d = ET.SubElement(top, "default", {"class": "visual"})
ET.SubElement(d, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2",
                          "density": "0"})
d = ET.SubElement(top, "default", {"class": "collision"})
ET.SubElement(d, "geom", {"type": "mesh", "group": "3", "condim": "3", "density": "0",
                          "friction": "0.8 0.005 0.0001"})
d = ET.SubElement(top, "default", {"class": "gripper_visual"})
ET.SubElement(d, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2",
                          "density": "0"})
d = ET.SubElement(top, "default", {"class": "gripper_collision"})
ET.SubElement(d, "geom", {"type": "mesh", "group": "3", "condim": "4", "density": "0",
                          "friction": "1.0 0.01 0.0001", "priority": "1"})
d = ET.SubElement(top, "default", {"class": "gripper"})
ET.SubElement(d, "position", {"kp": "5", "dampratio": "1"})

asset = ET.SubElement(mj, "asset")
meshes = {}


def mesh_name(filename):
    """Collision mesh: the whole COLLADA file as one STL (MuJoCo collides its convex hull)."""
    rel = filename.replace("package://mycobot_description/", "")
    stl = "derived_meshes/" + rel[:-4] + ".stl"
    name = Path(rel).stem
    if name not in meshes:
        meshes[name] = stl
        ET.SubElement(asset, "mesh", {"name": name, "file": stl})
    return name


visuals = {}


def visual_parts(filename):
    """Visual mesh: one OBJ per COLLADA material with that material's own colour/texture
    (fetch_meshes.py derives them; the URDF defines no <material>)."""
    rel = filename.replace("package://mycobot_description/", "")
    stem = Path(rel).stem
    if stem not in visuals:
        out = []
        for p in dae_parts(R, rel):
            name = f"{stem}_m{p['k']}"
            ET.SubElement(asset, "mesh", {"name": name, "file": p["obj"]})
            mat = {"name": name, "rgba": fmt(p["rgba"])}
            if p["texture"]:
                ET.SubElement(asset, "texture", {"name": name, "type": "2d", "file": p["texture"]})
                mat["texture"] = name
            ET.SubElement(asset, "material", mat)
            out.append(name)
        visuals[stem] = out
    return visuals[stem]


world = ET.SubElement(mj, "worldbody")
contact_excl = []


def emit(link, body):
    el = links[link]
    grip = link.startswith("gripper")
    if link in JN:
        ET.SubElement(body, "inertial", JN[link])
    for kind in ("visual", "collision"):
        for part in el.findall(kind):
            mesh = part.find("geometry/mesh")
            xyz, rpy = origin(part)
            cls = ("gripper_" if grip else "") + kind
            if kind == "visual":
                names = visual_parts(mesh.get("filename"))
                for n in names:
                    attrs = {"class": cls, "name": f"{link}_visual" if len(names) == 1 else f"{link}_visual_{n[-2:]}",
                             "mesh": n, "material": n}
                    attrs.update(pose_attrs(xyz, rpy))
                    ET.SubElement(body, "geom", attrs)
                continue
            attrs = {"class": cls, "name": f"{link}_{kind}", "mesh": mesh_name(mesh.get("filename"))}
            attrs.update(pose_attrs(xyz, rpy))
            if kind == "collision" and link in EST_MASS:
                attrs["mass"] = fmt([EST_MASS[link]])
            ET.SubElement(body, "geom", attrs)
    for j in children.get(link, []):
        child = j.find("child").get("link")
        xyz, rpy = origin(j)
        attrs = {"name": child}
        attrs.update(pose_attrs(xyz, rpy))
        sub = ET.SubElement(body, "body", attrs)
        if j.get("type") in ("revolute", "continuous"):
            lim = j.find("limit")
            ja = {"name": j.get("name"), "axis": fmt([float(v) for v in j.find("axis").get("xyz").split()])}
            if j.find("mimic") is None:
                ja["range"] = fmt([float(lim.get("lower")), float(lim.get("upper"))])
            ET.SubElement(sub, "joint", ja)
        emit(child, sub)


root_link = "g_base"
rb = ET.SubElement(world, "body", {"name": root_link, "childclass": "mycobot280"})
emit(root_link, rb)

contact = ET.SubElement(mj, "contact")
grip_links = [l for l in links if l.startswith("gripper")]
for i, a in enumerate(grip_links):
    for b in grip_links[i + 1:]:
        ET.SubElement(contact, "exclude", {"body1": a, "body2": b})
for a, b in [("g_base", "joint2"), ("joint1", "joint2"), ("joint1", "joint3"), ("joint2", "joint4"), ("joint3", "joint5"),
             ("joint4", "joint6"), ("joint5", "joint6_flange"), ("joint6", "gripper_base"),
             ("joint6_flange", "gripper_base")]:
    ET.SubElement(contact, "exclude", {"body1": a, "body2": b})

eq = ET.SubElement(mj, "equality")
for j in joints:
    m = j.find("mimic")
    if m is not None:
        ET.SubElement(eq, "joint", {"name": j.get("name"), "joint1": j.get("name"),
                                    "joint2": m.get("joint"),
                                    "polycoef": fmt([float(m.get("offset", 0)), float(m.get("multiplier", 1)), 0, 0, 0]),
                                    "solref": "0.005 1"})

act = ET.SubElement(mj, "actuator")
for n in ARM_JOINTS:
    lim = next(j for j in joints if j.get("name") == n).find("limit")
    ET.SubElement(act, "position", {"name": n, "joint": n, "class": "mycobot280",
                                    "ctrlrange": fmt([float(lim.get("lower")), float(lim.get("upper"))]),
                                    "forcerange": fmt([-FORCE[n], FORCE[n]])})
ET.SubElement(act, "position", {"name": "gripper_controller", "joint": "gripper_controller",
                                "class": "gripper", "ctrlrange": "-0.74 0.15",
                                "forcerange": fmt([-FORCE["gripper_controller"], FORCE["gripper_controller"]])})

kf = ET.SubElement(mj, "keyframe")
# the spawn pose is a natural ready pose, not the boot's all-zero upright stretch:
# joint3_to_joint2 (shoulder) 0, joint4_to_joint3 (elbow) -1.2, the rest 0
READY = {2: -1.2}
ET.SubElement(kf, "key", {"name": "home", "qpos": "0 " * 0 + "PLACEHOLDER",
                          "ctrl": "0 0 -1.2 0 0 0 0"})

ET.indent(mj, "  ")
text = ET.tostring(mj, encoding="unicode")
import mujoco  # noqa: E402
tmp = R / "model.xml"
tmp.write_text(text.replace(' qpos="PLACEHOLDER"', ""))
m = mujoco.MjModel.from_xml_path(str(tmp))
text = text.replace("PLACEHOLDER", " ".join(str(READY.get(i, 0)) for i in range(m.nq)))
tmp.write_text(text + "\n")
print("nq", m.nq, "nu", m.nu, "nbody", m.nbody, "mass", sum(m.body_mass))
