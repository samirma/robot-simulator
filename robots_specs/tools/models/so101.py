"""Derive robots_specs/so101/model.xml from the official so101_new_calib.xml.

Kinematics (joint origins, names, limits) follow the boot robot_description
(robots_specs/so101/boot/robot_description.urdf); everything else is copied from the official MJCF.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

F = Path(__file__).resolve().parents[2] / "so101"
OFFICIAL = F / "so101_new_calib.xml"
BOOT = F / "boot" / "robot_description.urdf"
OUT = F / "model.xml"

BODY = {"base": "base_link", "shoulder": "shoulder_link", "upper_arm": "upper_arm_link",
        "lower_arm": "lower_arm_link", "wrist": "wrist_link", "gripper": "gripper_link",
        "moving_jaw_so101_v1": "jaw_link"}
JOINT = {"shoulder_pan": "shoulder_pan_joint", "shoulder_lift": "shoulder_lift_joint",
         "elbow_flex": "elbow_flex_joint", "wrist_flex": "wrist_flex_joint",
         "wrist_roll": "wrist_roll_joint", "gripper": "gripper_joint"}


def fmt(vals):
    out = []
    for v in vals:
        s = f"{float(v):.9g}"
        out.append("0" if s in ("-0", "0") else s)
    return " ".join(out)


def urdf_joints():
    root = ET.parse(BOOT).getroot()
    j = {}
    for e in root.findall("joint"):
        o = e.find("origin")
        lim = e.find("limit")
        j[e.get("name")] = {
            "xyz": [float(v) for v in o.get("xyz").split()],
            "rpy": [float(v) for v in o.get("rpy").split()],
            "limit": (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None,
            "child": e.find("child").get("link"),
        }
    return j


def quat_wxyz(rpy):
    x, y, z, w = Rot.from_euler("xyz", rpy).as_quat()
    return [w, x, y, z]


def main():
    uj = urdf_joints()
    tree = ET.parse(OFFICIAL)
    root = tree.getroot()
    root.set("model", "so101")
    comp = root.find("compiler")
    comp.set("meshdir", ".")
    for m in root.find("asset").findall("mesh"):
        f = m.get("file")
        m.set("name", Path(f).stem)
        m.set("file", f"assets/{f}")
    # bodies and joints
    for body in root.iter("body"):
        old = body.get("name")
        body.set("name", BODY[old])
        j = body.find("joint")
        if j is not None:
            new = JOINT[j.get("name")]
            j.set("name", new)
            u = uj[new]
            body.set("pos", fmt(u["xyz"]))
            body.set("quat", fmt(quat_wxyz(u["rpy"])))
            j.set("range", fmt(u["limit"]))
    for a in root.find("actuator"):
        new = JOINT[a.get("joint")]
        a.set("name", new)
        a.set("joint", new)
        a.set("ctrlrange", fmt(uj[new]["limit"]))
    # gripper_frame_link as a site at the boot's fixed-joint pose; wrist camera
    gl = next(b for b in root.iter("body") if b.get("name") == "gripper_link")
    gf = uj["gripper_frame_joint"]
    site = ET.Element("site", {"name": "gripper_frame_link", "group": "3", "size": "0.004",
                               "pos": fmt(gf["xyz"]), "quat": fmt(quat_wxyz(gf["rpy"]))})
    idx = list(gl).index(next(e for e in gl if e.tag == "site")) + 1
    gl.insert(idx, site)
    # Estimated mount (import.md): lens on the hex-nut mount plate on the +y side of gripper_link,
    # centred over the jaw gap, looking along the fingers (-z) pitched 15 deg toward them.
    tilt = math.radians(15.0)
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, math.cos(tilt), -math.sin(tilt)])       # MuJoCo camera +y
    hfov = math.radians(70.0)
    fovy = 2 * math.degrees(math.atan(math.tan(hfov / 2) * 480 / 640))
    cam = ET.Element("camera", {"name": "default_cam", "mode": "fixed",
                                "pos": fmt([-0.008, 0.045, -0.01]),
                                "xyaxes": fmt(list(right) + list(up)),
                                "fovy": f"{fovy:.4f}", "resolution": "640 480"})
    gl.insert(idx + 1, cam)
    # keyframe: the pose it spawns in, a natural ready pose (shoulder lift -0.6, elbow 1.2:
    # upper arm back, forearm forward, gripper angled down) instead of the boot's
    # initial_position 0.0 on every joint, which holds the arm stretched out awkwardly
    ready = [0, -0.6, 1.2, 0, 0, 0]
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {"name": "home", "qpos": fmt(ready), "ctrl": fmt(ready)})

    ET.indent(tree, space="  ")
    body = ET.tostring(root, encoding="unicode")
    # the official file's comments are dropped by ElementTree; re-add provenance at the top
    header = f"""<?xml version="1.0" ?>
<!--
  DERIVED MODEL - not manufacturer-provided. See robots_specs/so101/import.md.

  Derived from the official TheRobotStudio MJCF so101_new_calib.xml (SO-ARM100 @
  aec17bbc256d1a7342d53aaa4950595d4c30b40d, Simulation/SO101), which is kept unmodified next to
  this file. Geometry, meshes (assets/), materials, inertials, joint dynamics classes and actuator
  gains are the official file's. Changes, each an adaptation listed in import.md:
    * body, joint and actuator names, joint origins (zero offsets) and joint ranges follow the
      authoritative boot's robot_description (ros2_so_arm @ e166df9, boot/robot_description.urdf);
    * site gripper_frame_link at the boot's gripper_frame_joint pose;
    * camera default_cam: the usb_cam wrist camera (frame_id default_cam), 640x480, at an
      ESTIMATED mount pose and ESTIMATED field of view (the boot publishes no calibration);
    * keyframe home: the spawn pose, a natural ready pose (shoulder lift -0.6, elbow 1.2, the
      rest 0.0) rather than the boot's initial_position 0.0 on every joint.
  Generated by the conversion steps in import.md.
-->
"""
    OUT.write_text(header + body + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
