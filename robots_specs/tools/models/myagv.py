"""Generate robots_specs/myagv/model.xml (derived myAGV MJCF). Deterministic.

python3 robots_specs/tools/models/myagv.py > robots_specs/myagv/model.xml
(after robots_specs/tools/fetch_meshes.py myagv)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _visual import dae_parts  # noqa: E402

ROBOT = Path(__file__).resolve().parents[2] / "myagv"
VIS = {name: dae_parts(ROBOT, f"urdf/{name}.dae") for name in ("myagv_base", "myagv_up")}

# ---- figures (see import.md for the source of each) --------------------------------
R = 0.040            # wheel radius (CAD mesh: wheel bounding box 0.080 m) - estimate from CAD
WHEEL_W = 0.0315     # wheel width (CAD mesh)
LX = 0.1069          # half wheelbase (CAD mesh wheel centres x = +0.1124 / -0.1014)
LY = 0.09875         # half track (CAD mesh wheel centres y = +0.09455 / -0.10295)
CX, CY = 0.0055, -0.0042   # wheel-centre centroid in base_footprint (CAD mesh)
N = 12               # rollers per wheel
RR = 0.012           # roller (capsule) radius
D = R - RR           # roller axis distance from the hub axis
HALF = D * math.tan(2 * math.pi / N / 4) * math.sqrt(2)   # capsule half-length (flat envelope at mid-gap)
M_TOTAL = 4.16       # manufacturer weight
M_HUB = 0.070        # estimate
M_ROLLER = 0.004     # estimate
M_UP = 0.30          # top plate (base_up) estimate
M_CHASSIS = M_TOTAL - M_UP - 4 * (M_HUB + N * M_ROLLER)
WMAX = 0.9 / R       # manufacturer max speed 0.9 m/s over wheel radius

# bottom-contact roller axis direction in the ground plane, (x, y) (see import.md)
WHEELS = [
    ("front_left_wheel", CX + LX, CY + LY, -1),
    ("front_right_wheel", CX + LX, CY - LY, +1),
    ("rear_left_wheel", CX - LX, CY + LY, +1),
    ("rear_right_wheel", CX - LX, CY - LY, -1),
]


def f(*v):
    out = []
    for x in v:
        s = f"{x:.6g}"
        out.append("0" if s in ("-0", "0") else s)
    return " ".join(out)


def box_inertia(m, a, b, c):
    return (m / 12 * (b * b + c * c), m / 12 * (a * a + c * c), m / 12 * (a * a + b * b))


lines = []
w = lines.append
w('<mujoco model="myagv">')
w('<!--')
w('  DERIVED MODEL - not manufacturer-provided. Converted from the official myAGV.urdf')
w('  (elephantrobotics/myagv_ros @ c71f3cc574e5ed1973a925238eabe88662cfa701, myagv_urdf/urdf/myAGV.urdf)')
w('  with the adaptations listed in import.md: mecanum wheels with explicit passive rollers,')
w('  sensor mounts from the boot launch, masses and collision geometry (estimates).')
w('  Generated deterministically by the procedure in import.md; do not edit by hand.')
w('-->')
w('  <compiler angle="radian" meshdir="." autolimits="true"/>')
w('')
w('  <default>')
w('    <default class="myagv">')
w('      <default class="visual">')
w('        <geom type="mesh" group="2" contype="0" conaffinity="0" density="0"/>')
w('      </default>')
w('      <default class="collision">')
w('        <geom group="3" contype="1" conaffinity="0" condim="3" friction="0.8 0.005 0.0001" density="0"/>')
w('      </default>')
w('      <default class="roller">')
w('        <geom type="capsule" group="3" contype="1" conaffinity="0" condim="3" friction="1.0 0.005 0.0001"/>')
w('        <joint type="hinge" damping="1e-5" armature="1e-6" frictionloss="0"/>')
w('      </default>')
w('      <default class="wheel">')
w('        <joint type="hinge" axis="0 1 0" damping="0.001" armature="1e-4"/>')
w('        <velocity kv="0.3" ctrlrange="%s" forcerange="-1.5 1.5"/>' % f(-WMAX, WMAX))
w('      </default>')
w('    </default>')
w('  </default>')
w('')
w('  <asset>')
# Visual meshes: the official COLLADA files, one OBJ per COLLADA material, each with that
# material's own colour (the meshes' own materials govern, as in RViz; the URDF <material>
# "black" rgba 0.7 0.7 0 is used by no mesh without a material, so it is not used).
for name, parts in VIS.items():
    for p in parts:
        w('    <mesh name="%s_m%d" file="%s"/>' % (name, p["k"], p["obj"]))
        w('    <material name="%s_m%d" rgba="%s"/>' % (name, p["k"], f(*p["rgba"])))
w('  </asset>')
w('')
w('  <worldbody>')
w('    <body name="base_footprint" childclass="myagv">')
w('      <freejoint name="root"/>')
ci = box_inertia(M_CHASSIS, 0.31, 0.20, 0.08)
w('      <inertial pos="%s" mass="%s" diaginertia="%s"/>' % (f(CX, CY, 0.055), f(M_CHASSIS), f(*ci)))
for p in VIS["myagv_base"]:
    w('      <geom class="visual" mesh="myagv_base_m%d" material="myagv_base_m%d"/>' % (p["k"], p["k"]))
w('      <geom class="collision" name="chassis" type="box" pos="%s" size="%s"/>' % (f(0.0065, -0.004, 0.0555), f(0.155, 0.080, 0.0355)))
w('      <site name="laser_frame" pos="0.065 0 0.08" euler="0 0 3.14159265" size="0.01" group="4"/>')
# static_transform_publisher args "0 0 0 0 3.14159 3.14159" = yaw 0, pitch 3.14159, roll 3.14159
w('      <site name="imu_link" pos="0 0 0" euler="3.14159 3.14159 0" size="0.01" group="4"/>')
w('      <body name="camera_link" pos="0.13 0 0.131">')
w('        <camera name="camera_link" xyaxes="0 -1 0 0 0 1" fovy="41.83" resolution="640 480"/>')
w('      </body>')
w('      <body name="base_up">')
w('        <joint name="base_up" type="hinge" axis="0 0 1"/>')
ui = box_inertia(M_UP, 0.31, 0.20, 0.04)
w('        <inertial pos="%s" mass="%s" diaginertia="%s"/>' % (f(0.0065, -0.004, 0.11), f(M_UP), f(*ui)))
for p in VIS["myagv_up"]:
    w('        <geom class="visual" mesh="myagv_up_m%d" material="myagv_up_m%d"/>' % (p["k"], p["k"]))
w('        <geom class="collision" name="top_plate" type="box" pos="%s" size="%s"/>' % (f(0.0065, -0.004, 0.1115), f(0.155, 0.100, 0.0205)))
w('      </body>')

hub_i = (M_HUB / 12 * (3 * 0.03 ** 2 + WHEEL_W ** 2), 0.5 * M_HUB * 0.03 ** 2, M_HUB / 12 * (3 * 0.03 ** 2 + WHEEL_W ** 2))
for name, x, y, s in WHEELS:
    w('      <body name="%s" pos="%s">' % (name, f(x, y, R)))
    w('        <joint name="%s_joint" class="wheel"/>' % name)
    w('        <inertial pos="0 0 0" mass="%s" diaginertia="%s"/>' % (f(M_HUB), f(*hub_i)))
    a0 = (1 / math.sqrt(2), s / math.sqrt(2), 0.0)   # roller axis when at the bottom
    for i in range(N):
        th = 2 * math.pi * i / N
        c, sn = math.cos(th), math.sin(th)
        # rotation about +y by th: x' = c x + s z ; z' = -s x + c z
        px, pz = sn * (-D), c * (-D)
        ax, ay, az = c * a0[0], a0[1], -sn * a0[0]
        p0 = (px - HALF * ax, -HALF * ay, pz - HALF * az)
        p1 = (px + HALF * ax, HALF * ay, pz + HALF * az)
        w('        <body name="%s_roller_%d" pos="%s">' % (name, i, f(px, 0, pz)))
        w('          <joint name="%s_roller_%d" class="roller" axis="%s"/>' % (name, i, f(ax, ay, az)))
        w('          <geom class="roller" size="%s" fromto="%s" mass="%s"/>' % (
            f(RR), f(p0[0] - px, p0[1], p0[2] - pz, p1[0] - px, p1[1], p1[2] - pz), f(M_ROLLER)))
        w('        </body>')
    w('      </body>')
w('    </body>')
w('  </worldbody>')
w('')
w('  <equality>')
w('    <!-- base_up is a continuous joint in the URDF; the real top plate is rigid (import.md). -->')
w('    <joint name="base_up_locked" joint1="base_up" polycoef="0 0 0 0 0"/>')
w('  </equality>')
w('')
w('  <actuator>')
for name, *_ in WHEELS:
    w('    <velocity name="%s_joint" joint="%s_joint" class="wheel"/>' % (name, name))
w('  </actuator>')
w('')
nq = 7 + 1 + 4 * (1 + N)
qpos = [0, 0, 0, 1, 0, 0, 0] + [0] * (nq - 7)
w('  <keyframe>')
w('    <key name="home" qpos="%s"/>' % " ".join(str(v) for v in qpos))
w('  </keyframe>')
w('</mujoco>')
print("\n".join(lines))
