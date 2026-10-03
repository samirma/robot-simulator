# ROSMASTER X3 PLUS — MuJoCo import notes

`model.xml` is a **derived** model, not manufacturer-provided. Yahboom publishes no MJCF for
the ROSMASTER X3 PLUS; the model is converted from the official URDF
`yahboomcar_X3plus.urdf` and extended only where the interface or the physics needs it. Every
change is listed below as an adaptation or an estimate.

## Sources

| What | Where | Revision / digest |
| --- | --- | --- |
| URDF (committed here) | `ROSMASTER-X3Plus_ROS1_code.zip!/yahboomcar_ws.zip!/yahboomcar_ws/src/yahboomcar_description/urdf/yahboomcar_X3plus.urdf` | sha256 `17c7c8fac92774f0d0cd093fb7500704d3b5149998cb8b60c958f5352646919c` (= spec) |
| Meshes (fetched, not committed) | same archive, `yahboomcar_description/meshes/` → `meshes/` here, paths preserved (`package://yahboomcar_description/` = this folder) | `yahboomcar_ws.zip` sha256 `13d752e04cba3e34116912c3903bbe194e5ca5185c304036fb0ddedb59937162`; per file in `robots_specs/meshes.sha256` |
| Code download | <https://drive.google.com/file/d/1SRg1aD_u8kyxxjm4vp0cFYxu8Ddj2sXU> `ROSMASTER-X3Plus_ROS1_code.zip` | 6343438429 bytes |
| Mecanum geometry, speed clamps | same archive, `Expansion board STM32 firmware/Rosmaster_V3.5.1.zip` (`app_mecanum.h`, `app_mecanum.c`, `app_motion.[ch]`) | zip sha256 `09bcd255830bd38da6858c81971d7afeb00f80d33c3fd3309c69b2db384003f6` |
| Servo ranges, protocol | same archive, `py_install_V3.3.9.zip` (`Rosmaster_Lib.py`) | zip sha256 `1761c5873b6d1407afe5b4f4c5c4fb05787e07730448e786243071fe9e8b6ce7` |
| Camera driver / frames | same archive, `software.zip!/software/library_ws/src/orbbec-ros-sdk` | per-file digests in `ros.yml` `sources[software_zip]` |
| Tutorials | <https://github.com/YahboomTechnology/ROSMASTERX3-PLUS> | `9732c62247dfb57a899aea01e4fe72ed434bac6d` |

The firmware and library members were range-read from the pinned archive on 2026-09-29;
their sizes match the archive's central directory. On 2026-09-30 Google Drive refused
further downloads (quota exceeded), so their CRCs were not re-checked in this session. On
2026-10-02 both members were range-read again and matched the sha256 digests above (read for
the /cmd_vel yaw-adjust path and the arm watchdog sources in `ros.yml`).

## Conversion steps

1. Parse `yahboomcar_X3plus.urdf`; every link becomes a body named after the link at its
   joint origin (`rpy` converted to a quaternion, fixed axes), every `revolute` or
   `continuous` joint a hinge with the URDF axis and, for `revolute`, the URDF range. Fixed
   joints become welded child bodies. `<compiler angle="radian" meshdir="." autolimits="true"
   inertiafromgeom="false" balanceinertia="true"/>`.
2. Every URDF `<inertial>` is copied as `<inertial pos mass fullinertia>` (ixx iyy izz ixy
   ixz iyz), unchanged.
3. Every URDF visual mesh becomes a `class="visual"` geom (group 2, no contact) with the URDF
   `<material><color>` of that visual (the STLs carry no colour, so the URDF's colours — green
   `0 0.7 0 1` and grey `0.7 0.7 0.7 1` — are the official appearance); every collision mesh a
   `class="collision"` geom (group 3). Mesh files are
   referenced at their upstream relative paths under `meshes/`.
4. The top body is `base_footprint` with `<freejoint name="root"/>`; `base_link` sits 0.076 m
   above it (URDF `base_joint`).
5. The five URDF `<mimic>` joints of the gripper linkage (`rlink_joint2` ×−1, `rlink_joint3`
   ×1, `llink_joint1` ×−1, `llink_joint2` ×1, `llink_joint3` ×−1, all driven by
   `grip_joint`) become `<equality><joint polycoef="0 k 0 0 0"/>` constraints.
6. Actuators: one `position` actuator per actuated URDF joint (`arm_joint1`–`arm_joint5`,
   `grip_joint`), named after the joint, `ctrlrange` = URDF limits (rad), `forcerange` =
   URDF `effort` (100); four `velocity` actuators on the added wheel joints (rad/s).
7. Adaptations A1–A7 below, then a `home` keyframe for the boot pose.

The conversion is `robots_specs/tools/models/rosmaster_x3_plus.py` (standard library + the
MuJoCo Python bindings for the keyframe); it reproduces `model.xml` byte for byte and is
fully determined by the steps and numbers in this file.

## Adaptations

**A1 — Mecanum wheels and rollers (contact the URDF lacks).** The URDF has no wheel links:
the four 80 mm mecanum wheels are drawn inside `X3plus/visual/base_link.STL`. The model
adds bodies `front_left_wheel`, `front_right_wheel`, `back_left_wheel`, `back_right_wheel`
with hinge joints `front_left_joint` … `back_right_joint` (axis +y of `base_link`; joint
names follow Yahboom's own X3 URDF, `yahboomcar_X3.urdf`). These joints are **not** in the
boot's `robot_description` and are never on the wire (the driver publishes no wheel
states). Each wheel carries 12 passive rollers: capsules of radius 11 mm and half-length
9.5 mm, centred 29 mm from the axle, each on its own hinge whose axis is tilted 45° from the
rolling direction toward the axle. At the ground contact the roller axes point along
(1, −1) for front-left and back-right and (1, 1) for front-right and back-left (top view,
x forward, y left), which reproduces the firmware's mixing
`v_FL = vx − vy − k·wz`, `v_BL = vx + vy − k·wz`, `v_FR = vx + vy + k·wz`,
`v_BR = vx − vy + k·wz`, `k = lx + ly` (`app_mecanum.c` L28–L67). Rollers: mass 4 g,
friction 1.0, joint damping 1e-5. The wheel hub is a visual cylinder only; the wheel visual
the user sees is the (non-spinning) wheel inside the base mesh.

| Figure | Value | Basis |
| --- | --- | --- |
| wheel radius | 0.040 m | firmware `MECANUM_MAX_CIRCLE_MM` 251.327 mm / π (manufacturer source) |
| lx + ly | 0.2141 m | firmware `MECANUM_MAX_APB` 214.1 mm (manufacturer source) |
| wheel centres (base_link) | front (0.1053, ±0.1042, −0.0389), back (−0.1146, ±0.1042, −0.0393) | **estimate**, measured from the wheels in `base_link.STL` (lx 0.110 + ly 0.104 = 0.214 m agrees with the firmware) |
| wheel width | 0.036 m | **estimate** from the mesh (|y| 0.086–0.122) |
| hub mass / roller mass | 0.08 kg / 0.004 kg | **estimate** (the URDF has no wheel mass) |
| wheel speed limit | ±17.5 rad/s | firmware per-wheel clamp 700 mm/s (`CAR_X3_PLUS_MAX_SPEED`) / 0.04 m |
| wheel torque limit | ±1.0 N·m, `kv` 0.5 N·m·s/rad, armature 0.001 kg·m² | **estimate** (520 motor, 56:1; no vendor torque curve) |

**A2 — Chassis collision.** MuJoCo collides with a mesh's convex hull; the hull of
`X3plus/collision/base_link.STL` includes the wheels and the camera tower, so it would drag
on the floor. It is replaced by four boxes fitted to the visual mesh (**estimates**):
`chassis_col` (x ±0.142, y ±0.076, z −0.045…0.080 in `base_link`, 31 mm ground clearance),
`deck_col`, `tower_post_col`, `tower_top_col`. The URDF collision meshes of the camera, lidar,
arm and gripper links are used as given.

**A3 — Oversized visual meshes.** MuJoCo's STL decoder refuses meshes over 200000 faces.
The official visual STLs of `base_link` (567561 faces), `arm_link2` (206810), `arm_link3`
(206810) and `arm_link4` (257460) are therefore shown as lossless parts:
`robots_specs/tools/fetch_meshes.py` copies each file's triangle records verbatim, in order,
into `derived_meshes/meshes/X3plus/visual/<name>.STL.part<k>.stl` of at most 200000 faces
(3, 2, 2 and 2 parts), and the link gets one visual geom per part at the same origin with the
same URDF colour. No triangle is changed or dropped (a test re-concatenates the parts);
collision geometry, frames and inertia are unchanged.
Appearance vs the product: the vendor photo (`Yahboom_ROSMASTERX3_PLUS.jpg` in the tutorial
repository) shows green PCB decks, black pillar, wheels and servos; the URDF gives each link a
single colour (the whole `base_link` green), so the model shows that and no per-part colours.

**A4 — Collision filtering.** Robot collision geoms have `contype="1" conaffinity="0"`: they
touch the world (floor, furniture, grasped objects) but not each other, so the closed
gripper linkage and the arm resting near the deck do not fight. This is a modelling choice;
the real arm can hit its own base.

**A5 — Servo model.** Arm and gripper joints are position servos (`kp` 20, `dampratio` 1),
joint armature 0.005 kg·m² and damping 0.05 (**estimates**; the URDF has no dynamics and
Yahboom gives no servo stiffness). The real bus servos move to a target over the commanded
`run_time` (0–2000 ms, `Rosmaster_Lib.py` L814–L818); interpolating the target over
`run_time` is the simulator's job, not the model's. Mimic joints: damping 0.001, armature
1e-5.

**A6 — Sensors.** `camera_color_optical_frame` is a MuJoCo `<camera>` in `camera_link`
(URDF `astra_joint`: (−0.043645, 0, 0.41955) from `base_link`), looking along +x of
`camera_link` (`xyaxes="0 -1 0 0 0 1"`: MuJoCo cameras look along −z with +y up, i.e. the
ROS optical frame rotated 180° about x), `resolution="640 480"`, `fovy="40.2"`. Pose
source: the URDF for `camera_link`; the driver's colour-sensor offset from `camera_link`
is the unit's factory extrinsic (`ob_camera_node.cpp` L694–L725), which is not in any
pinned file, so it is **estimated as zero**. `fovy` 40.2° is Orbbec's published RGB
vertical FoV (<https://store.orbbec.com/products/astra-pro-plus>, stated for 1920×1080),
kept for the 640×480 mode as an **estimate**; the robot's CameraInfo carries the factory
calibration instead. Depth and IR are published in the same optical frame (the boot
registers depth to colour), so the same camera renders them. Sites: `imu_link` (in body
`imu_link`, URDF `base_imu`) and `laser` (in `laser_link`, rotated by the boot's
`/laser_link → /laser` yaw 6.28 rad, `laser_astrapro_bringup.launch` L10–L11). The arm's
`mono_link` USB camera is not started by the boot and has no MuJoCo camera.

**A7 — Gripper range.** The driver publishes `grip_joint` = −π/2 for the open gripper
(servo 30°, `Mcnamu_X3plus.py` L205–L210) while the URDF limits `grip_joint` to
[−1.54, 0]. The boot's `robot_description` governs, so the model keeps [−1.54, 0]; a
simulator mapping servo 30° clamps to −1.54 (0.031 rad short of the wire value).

## Boot `robot_description` vs model

`robot_description` is `xacro --inorder yahboomcar_X3plus.urdf` (`bringup.launch` L8–L9);
the file has no xacro macros, so joint names, axes, limits and zero offsets are identical to
the model's. Wire convention: joint angle = (servo degrees − 90°), so servo 90° is the URDF
zero for `arm_joint1`–`arm_joint5`; `grip_joint` = (servo° − 30°)·90/150 − 90° in degrees.
Differences are only the model-only joints of A1 and the gripper range note of A7.

## Keyframe `home`

The driver commands the arm to servo angles `[90, 145, 0, 45, 90, 30]` at start
(`Mcnamu_X3plus.py` L53–L54): `arm_joint1..5` = 0, 0.9599, −1.5708, −0.7854, 0 rad,
`grip_joint` −1.54 (A7), mimic joints set by their multipliers, `base_footprint` 2.9 mm
above the floor (settled height, wheels touching).

## Model figures: manufacturer versus estimate

* Manufacturer / source: URDF geometry, masses and inertias; wheel radius and lx+ly;
  per-wheel speed clamp; servo ranges; camera resolution and rate; lidar angle and range
  limits (`TG.launch`).
* Estimates: wheel centre split lx/ly, wheel width, wheel and roller masses, motor torque and
  gains, servo gains/armature, chassis collision boxes, camera intrinsics (fovy) and the
  colour-sensor offset, lidar sample count (derived from launch settings).
* Mass: the URDF links total 1.544 kg; with the estimated wheels and rollers (4 × 0.128 kg)
  the model weighs 2.056 kg. No total robot mass appears in the pinned sources, so the URDF
  masses are kept as they are (a real X3 PLUS with battery is likely heavier; unverified).

## Estimates (not manufacturer data)

Interface values read from the expansion board at run time, which no pinned source fixes
(added 2026-10-03):

* **Battery** (`/voltage`): **12.3 V** (the pack is not modelled).
* **Firmware version** (`/edition`): **3.5**, the major.minor of the pinned firmware V3.5.1.
* **Magnetometer** (`/mag/mag_raw`): a fixed local field of **0.22 gauss horizontal (north)
  and 0.42 gauss down**, rotated into `imu_link`.
* **`orbbec_camera` message and service definitions**: the vendor's copy (`software.zip`)
  could not be re-read (Google Drive quota), so its `msg/`/`srv/` files are taken from
  Orbbec's public OrbbecSDK_ROS1. Basis: the vendor's cited `ob_camera_node.cpp`,
  `ros_setup.cpp` and `ros_service.cpp` equal OrbbecSDK_ROS1 `v1.2.9`
  (`9569cc31c14989d8008c7f1e99ec1ced12f3c138`) byte for byte (the sha256 prefixes in `ros.yml`
  `sources[software_zip]`), and every `msg/` and `srv/` file of `v1.2.9` is the same git blob at
  `v1.4.2` (`7ef885a4a168df02948ca97da216751a83d40471`), the revision the definitions are built
  from (checked 2026-10-03). The vendor's own msg/srv files are still unverified.

## Verification (2026-09-30, MuJoCo 3.14.0, scratch wrapper adding a floor)

* Loads with `mujoco.MjModel.from_xml_path` once `robots_specs/tools/fetch_meshes.py` has
  fetched `meshes/`: nq 70, nv 69, nu 10, 1 camera, 2 sites.
* At rest from `home` for 3 s (implicitfast, dt 0.002): base drift < 0.2 mm, |v| 6e-8.
* Wheel speeds from the firmware mixing, 2 s runs, dt 0.002 (dt 0.001 gives the same to
  ±1 mm): forward 0.2 m/s → +0.401 m (expected 0.400); back → −0.401 m; left 0.2 m/s →
  +0.390 m; right → −0.390 m; yaw 1.0 rad/s → +1.998 rad (expected 2.000), all with < 7 mm
  cross-track error; after the zero command the base stops within 3.4 mm.
* Arm targets (0.5, 0.3, −1.0, −0.5, 1.0 rad) reached within 0.008 rad after 2 s; gripper
  0 rad closes the finger pads (26 mm apart) and −1.54 rad opens them (85 mm).
