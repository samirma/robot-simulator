# reBot Arm B601-RS — MuJoCo import notes

`model.xml` in this folder is a **derived model, not manufacturer-provided**. Seeed publishes no
MJCF for the reBot Arm B601-RS: its description package ships the URDF and the STL meshes MuJoCo
needs but "does not contain a complete MJCF scene"
([`Rebot_Arm_description/README.md` L158-L165](https://github.com/Seeed-Projects/reBot-DevArm/blob/0dfb49b85bc146eb3741f414be9dd2b1662345e8/Rebot_Arm_description/README.md#L158-L165)).
Seeed's ROS 2 repository has a package, `rebotarm_mujoco_rs`, that wraps a community MJCF of the
arm, `LAN-GER/reBot-B601-RS-for-mujoco_sim`, a personal repository without a licence; Seeed does
not vendor it, and the pinned revision holds neither that model nor the
`scripts/setup_rs_workspace.sh` its README names to fetch it
([`src/rebotarm_mujoco_rs/README.md` L3-L9](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarm_mujoco_rs/README.md#L3-L9)).
That model is neither manufacturer-provided nor pinned, and it is not used. `model.xml` is
converted from the official URDF, the authoritative boot's own copy, and extended only where the
hardware or the boot needs it. This file records the sources, the conversion, every adaptation
and every estimate. The interface the model serves is `ros2.yml`, Seeed's own (manufacturer)
ROS 2 interface. No figure here was measured on a real robot.

## Sources

| What | Source | Revision |
| --- | --- | --- |
| URDF `ReBot_Arm_RS.urdf` (sha256 `978568c1…7b77b`) and the 42 RS meshes it and the MuJoCo finger contact use; the boot (`bringup.launch.py`, `rviz_urdf_compat.py`, `rebotarm_hardware.yaml`) and the controller node | <https://github.com/Seeed-Projects/reBotArmController_ROS2>, `src/rebotarm_bringup/`, `src/rebotarmcontroller/` | `7506dc0fc5c2a7787dcb974d89d13f9f49d16d61` |
| SDK: motor ids and models, MIT gains (`config/rebotarm_rs.yaml`), the 500 Hz MIT hold loop with its gravity feed-forward, safe home | <https://github.com/Seeed-Projects/reBotArm_control_py> | `6415d43130d1e143c70dc106096a857ac5556f81` |
| Hardware BOM (pinion, racks, rail, finger material), specifications (payload, reach, weight, repeatability), description package README (MuJoCo finger contact, left/right file mapping) | <https://github.com/Seeed-Projects/reBot-DevArm>, `hardware/reBot_B601_RS/README.md`, `README.md`, `Rebot_Arm_description/README.md` | `0dfb49b85bc146eb3741f414be9dd2b1662345e8` |
| The wrist camera: the D405 launch and topics of the B601-RS grasping guide | <https://wiki.seeedstudio.com/rebot_arm_b601_rs_grasping_demo/>, source `Seeed-Studio/wiki-documents`, `sites/en/docs/Robotics/Robot_Kits/reBot_Arm/B601_RS/reBot_Arm_B601_RS_Grasping_Demo.md` | `2d9c3d1225f1ab98b79e1382b72ca9e2148bb23a` (retrieved 2026-10-09) |
| The wrist camera's mount on `gripper_end` | <https://github.com/xiehuangbao888/rebot_visual_grasp>, `config/wrist_extrinsics_d405.yaml`, `description/urdf/rebotarm_rs_with_d405.urdf.xacro` (the community package the guide clones) | `dd28d65598deec767cf95fa45521d69b38155833` |
| The D405 driver: node, topics, frames, transforms, parameters | <https://github.com/IntelRealSense/realsense-ros>, `realsense2_camera/` (4.58.4) and `realsense2_description/urdf/_d405.urdf.xacro`; the guide's fork `xiehuangbao888/realsense-ros` is the same commit | `9a11121700cb4780e273e34141f6402fe184321d` |
| RobStride RS-06 / RS-00 position, velocity and torque ranges; RobStride parameter table with its stated defaults | <https://github.com/motorbridge/motorbridge> (v0.5.6), `motor_vendors/robstride/src/motor.rs`, `motor_vendors/robstride/src/registers.rs` | `c652ce420e7da1008fa1864cc7f47d7d7c57fdb6` |

reBot-DevArm's `Rebot_Arm_description/RS/urdf/ReBot_Arm_RS.urdf` equals the boot's copy apart from
its `../meshes/` mesh URIs, and its 42 RS meshes are the same git blobs as the boot's (checked
2026-10-08). Every further dependency release is pinned in `ros2.yml` → `sources`.

## Files in this folder

* `ReBot_Arm_RS.urdf` — the official URDF, the boot's copy
  (`src/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf`), unmodified.
* `description/RS/meshes/` — fetched, not committed (`robots_specs/tools/fetch_meshes.py`, digests
  in `robots_specs/meshes.sha256`), upstream file names and relative paths kept: `shared/` (10
  meshes, URDF visual and collision), `visual/` (22, URDF visual only) and `mujoco_collision/` (10
  finger contact parts that no URDF references, used by adaptation A1).
* `boot/robot_description.urdf` — the boot's `robot_description`: the stdout of the pinned
  `rviz_urdf_compat.py` run on the URDF (`ros2.yml` → `model.robot_description.content`, sha256
  `b45bd97e…6f64`).
* `model.xml` — the derived MJCF (below).
* `ros2.yml` — the ROS 2 interface (schema `robots_specs/SCHEMA.md`).

## Conversion steps

1. **Mesh URIs.** The URDF's meshes are
   `package://rebotarm_bringup/description/RS/meshes/<kind>/<name>`. The robot folder stands for
   the `rebotarm_bringup` package root, so they resolve inside it to
   `description/RS/meshes/<kind>/<name>`. Fetch them with
   `python3 robots_specs/tools/fetch_meshes.py rebot_b601_rs` (a sparse checkout of the pinned
   reBotArmController_ROS2, every file verified against `robots_specs/meshes.sha256`).
2. **Generate.** `python3 robots_specs/tools/models/rebot_b601_rs.py` (standard library only,
   deterministic, from any directory) writes `model.xml` from `ReBot_Arm_RS.urdf`; it reproduces
   the committed file byte for byte (`robots_specs/tests/test_robots.py`,
   `test_derived_model_is_reproducible`). Every figure it uses is a constant at its top, sourced
   in this file.
3. **What it does**, under
   `<compiler angle="radian" meshdir="." autolimits="true" inertiafromgeom="false"/>`:
   * The top body is `base_link`, the URDF root, with no freejoint (an arm is fixed where it is
     placed). Every other link becomes a body named after the link at its parent joint's
     `origin` (`rpy` converted to a quaternion, fixed axes; the URDF's own rounded angles such
     as 1.5708 are kept).
   * Revolute `joint1`–`joint6` become hinges with the URDF axis and `range` = the URDF limits;
     the fixed `j_gripper_end` welds body `gripper_end` to `link6`; the prismatic
     `gripper_joint1` and `gripper_joint2` become slides with the URDF axis and range (0–0.05 m
     and 0–0.0715 m).
   * Every URDF `<inertial>` becomes `<inertial pos mass fullinertia>` (ixx iyy izz ixy ixz iyz),
     unchanged; geoms add no mass (`density="0"`).
   * Every URDF visual becomes a geom of class `visual` (group 2, no contact) with its URDF
     material: the URDF's six `rs_*` colours are MuJoCo materials of the same names and rgba.
     Every URDF collision mesh becomes a geom of class `collision` (group 3), except the two
     fingers' (A1). Each mesh is named after its file stem (unique among the 42).
   * Then the actuation, adaptations A1–A7 and the `home` keyframe below.
4. **Validate:** see "Verification".

## Actuation

The URDF has no actuators or transmissions; the model adds them as the boot drives the hardware,
an adaptation. The boot runs the SDK's 500 Hz MIT loop (`ros2.yml` →
`motions[arm].command.servo_profile`): each arm motor computes
`tau = kp (q_target − q) + kd (0 − dq) + tau_gravity` with the `rs` gains of
[`rebotarm_hardware.yaml` L52-L55](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarm_bringup/config/rebotarm_hardware.yaml#L52-L55)
(equal to the SDK's
[`rebotarm_rs.yaml` `joints[].MIT`, L37-L134](https://github.com/Seeed-Projects/reBotArm_control_py/blob/6415d43130d1e143c70dc106096a857ac5556f81/config/rebotarm_rs.yaml#L37-L134))
and the SDK's gravity feed-forward
([`rebotarm_endpose_controller.py` L328-L358](https://github.com/Seeed-Projects/reBotArm_control_py/blob/6415d43130d1e143c70dc106096a857ac5556f81/reBotArm_control_py/controllers/rebotarm_endpose_controller.py#L328-L358)).
The model reproduces that loop term by term:

| Drive | Motor (CAN id) | MIT `kp` N·m/rad → actuator `kp` | MIT `kd` N·m·s/rad → `damping` | torque range N·m → `forcerange`, `actuatorfrcrange` |
| --- | --- | --- | --- | --- |
| `joint1` | RS-06 (0x01) | 50 | 3 | ±36 |
| `joint2` | RS-06 (0x02) | 150 | 10 | ±36 |
| `joint3` | RS-06 (0x03) | 150 | 10 | ±36 |
| `joint4` | RS-00 (0x04) | 50 | 5 | ±14 |
| `joint5` | RS-00 (0x05) | 50 | 4 | ±14 |
| `joint6` | RS-00 (0x06) | 50 | 4 | ±14 |
| `gripper` (tendon, A2) | RS-00 (0x07) | 50 | 4 (as finger damping, A2) | ±14 (actuator), 500 N per finger joint (URDF) |

* One `position` actuator per arm joint, named after the joint, commanded in rad (the wire's
  unit), `ctrlrange` = the URDF limits, which are the boot `robot_description`'s.
* MIT `kd` is the joint's passive `damping`: the boot always sends a zero velocity target (the
  hold loop and trajectories alike), so `kd (0 − dq)` is exactly a viscous damping.
* The torque range is the URDF `effort` (36 N·m on joint1–joint3, 14 N·m on joint4–joint6), which
  equals motorbridge's RS-06 and RS-00 torque limits
  ([`motor.rs` L17-L24, L60-L66](https://github.com/motorbridge/motorbridge/blob/c652ce420e7da1008fa1864cc7f47d7d7c57fdb6/motor_vendors/robstride/src/motor.rs#L17-L66)).
  Through `actuatorfrcrange` it clamps the position term together with the gravity
  compensation (A3); the damping term is outside that clamp.

## Adaptations

**A1 — Finger contact from Seeed's MuJoCo collision parts.** Seeed ships ten STL parts for MuJoCo
gripper contact and says not to replace them with a convex hull of the complete finger, which
"may reintroduce interpenetration during closing, overly large contact surfaces, or unstable
grasping"
([`Rebot_Arm_description/README.md` L158-L165](https://github.com/Seeed-Projects/reBot-DevArm/blob/0dfb49b85bc146eb3741f414be9dd2b1662345e8/Rebot_Arm_description/README.md#L158-L165)).
MuJoCo collides a mesh as its convex hull, so the model uses the ten parts in place of the URDF's
finger collision meshes `shared/gripper_left.STL` and `shared/gripper_right.STL` (fetched because
the URDF references them, unused by the model): per finger the front, mid and rear finger
segments, the grey carriage and the yellow travel stop. As Seeed states for the RS ("RS left maps
to the DM right file and vice versa",
[L66](https://github.com/Seeed-Projects/reBot-DevArm/blob/0dfb49b85bc146eb3741f414be9dd2b1662345e8/Rebot_Arm_description/README.md#L66)),
body `gripper_left` (joint `gripper_joint1`) carries the `right_finger_*` and
`colored_right_finger_*` files, and `gripper_right` (`gripper_joint2`) the `left_*` ones. The
parts are expressed in `gripper_end` with the gripper closed, so each is placed in its finger
body by the inverse of the finger joint's origin: at joint position 0 it sits where it is in
`gripper_end`, and it moves with its finger (checked, see "Verification"). Class `finger`:
group 3, `condim 4` (torsional friction, so a pinched object does not spin about the grasp
axis), friction `1 0.005 0.0001` (an estimate, below).

**A2 — The coupled rack-and-pinion gripper, one actuator in motor rad.** The real gripper is one
RS-00 motor (SDK joint `gripper`, CAN id 0x07,
[`rebotarm_rs.yaml` L122-L134](https://github.com/Seeed-Projects/reBotArm_control_py/blob/6415d43130d1e143c70dc106096a857ac5556f81/config/rebotarm_rs.yaml#L122-L134))
turning a module-1, 16-tooth pinion between two racks that drive both fingers on one MGN9 rail
([BOM L111, L136-L138](https://github.com/Seeed-Projects/reBot-DevArm/blob/0dfb49b85bc146eb3741f414be9dd2b1662345e8/hardware/reBot_B601_RS/README.md#L111-L138)):
the fingers always move together and by the same amount, and the wire commands the motor angle
in rad, 0.0 closed to 5.0 open
([`rebotarm_hardware.yaml` L48-L51](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarm_bringup/config/rebotarm_hardware.yaml#L48-L51)).
The URDF models the fingers as two independent prismatic joints and has no `<mimic>`. The model:

* couples them with the equality `gripper_rack`, `gripper_joint2 = gripper_joint1`
  (`<joint joint1="gripper_joint2" joint2="gripper_joint1" polycoef="0 1 0 0 0" solref="0.005 1"/>`);
* defines a fixed tendon `gripper` = 62.5 · (`gripper_joint1` + `gripper_joint2`), i.e.
  1 / (2 r) times the summed travel with r = 0.008 m the pinion's pitch radius (estimate, below),
  so the tendon's length is the motor angle in rad;
* drives that tendon with **one** `position` actuator `gripper`: `kp` 50 N·m/rad (the gripper's
  MIT kp), `ctrlrange 0 5` rad, `forcerange ±14` N·m (the RS-00's torque limit, equal to the URDF
  effort of the other RS-00 joints);
* gives each finger slide the gripper's MIT kd 4 N·m·s/rad as seen through the pinion,
  `damping` = kd / (2 r²) = 31250 N·s/m (both racks share the pinion: kd·θ̇² = c·(ẋ₁² + ẋ₂²)
  with ẋ = r·θ̇);
* keeps the URDF finger `effort` 500 as each finger joint's `actuatorfrcrange` (N). Through the
  pinion that caps the grip at a motor-equivalent 2 r · 500 N = 8 N·m, below the RS-00's
  14 N·m (875 N per finger); ros2.yml records the URDF value as having no motor counterpart.

This is SCHEMA.md's convention for a drive that moves several joints through a fixed coupling:
one actuator on a fixed tendon, named as the boot names the drive — the SDK's gripper joint
`gripper`, as in the `/rebotarm/gripper/...` endpoints and the `joint_name` of
`/rebotarm/gripper/state` — whose length is the drive's position in the wire's unit, so the
actuator takes the wire's `command.position` in motor rad unchanged; the equality ties the two
finger joints, and neither has an actuator of its own. The hardware has no independently driven
fingers commanded in metres, and the model has none either. The equality stands in for the
`<mimic>` the URDF lacks.

The wire does not report the model's finger positions: `/rebotarm/joint_states` reports the boot's
own mapping of the motor angle (D6 below), so a simulator computes it from the motor angle, the
length of tendon `gripper`.

**A3 — The boot's gravity feed-forward as gravity compensation.** In MIT mode the SDK adds the
Pinocchio gravity torque of the URDF at the measured joints to every arm motor command, every
2 ms, for as long as the hold loop runs
([`rebotarm_endpose_controller.py` L328-L345](https://github.com/Seeed-Projects/reBotArm_control_py/blob/6415d43130d1e143c70dc106096a857ac5556f81/reBotArm_control_py/controllers/rebotarm_endpose_controller.py#L328-L345)).
The model reproduces it with MuJoCo gravity compensation: `gravcomp="1"` on every body below
`base_link`, and `actuatorgravcomp="true"` on `joint1`–`joint6`, so the compensation is applied
through the joint's actuator and clamped with it at the motor's torque range. The model's masses
are the URDF's, so the compensation is the boot's feed-forward. Remaining differences:

* the SDK pads the finger joints with 0 (closed) in its Pinocchio model
  ([`robot_model.py` L94-L99](https://github.com/Seeed-Projects/reBotArm_control_py/blob/6415d43130d1e143c70dc106096a857ac5556f81/reBotArm_control_py/kinematics/robot_model.py#L94-L99));
  the model uses their actual positions. The two equal fingers move apart symmetrically, so
  their combined centre of mass stays put and the arm joints' compensation agrees to 1e-6 N·m
  (fingers at 0 and at 0.04 m, 2000 random arm poses);
* the gripper motor gets no feed-forward on the robot
  ([L353-L358](https://github.com/Seeed-Projects/reBotArm_control_py/blob/6415d43130d1e143c70dc106096a857ac5556f81/reBotArm_control_py/controllers/rebotarm_endpose_controller.py#L353-L358)),
  while the model's finger slides, which have no `actuatorgravcomp`, receive the compensation of
  their own weight as a passive force (at most 0.0752 kg × 9.81 m/s² = 0.74 N per finger, which
  the gripper's MIT loop holds on the robot within 0.001 rad of motor angle);
* the model always compensates; on the robot the feed-forward stops with the hold loop (a
  disabled arm).

**A4 — Self-collision, with Seeed's excluded body pairs.** Every robot collision geom (classes
`collision` and `finger`) has `contype="1" conaffinity="1"`: the robot collides with itself as it
does with the scene, except for the ten body pairs Seeed's own MuJoCo scene excludes
([`rs_grasp_scene.xml` L7-L18](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarm_mujoco_rs/models/rs_grasp_scene.xml#L7-L18)),
which the model's `<contact>` excludes the same way: `world`–`link1`, `base_link`–`link1`,
`link1`–`link2`, `link2`–`link3`, `link2`–`link5`, `link3`–`link4`, `link3`–`link5`,
`link4`–`link5`, `link5`–`link6` and `gripper_left`–`gripper_right`. Seeed's scene names the
bodies of the community MJCF it includes, whose arm bodies are the URDF's links under the same
names and in the same tree (`base_link`, `link1`–`link6`, `gripper_end`, `gripper_left`,
`gripper_right`), so each name is the model body of that name; `world` is MuJoCo's world body,
to which `base_link` is welded. The convex hulls MuJoCo builds from the URDF's whole-link
collision meshes overlap where the real parts do not, and the excluded pairs cover each such
overlap: `base_link` and `link1` overlap by 10 mm in every pose (MuJoCo's parent filter does not
apply to the child of a body welded to the world), `link2` and `link5` touch at `home`, and the
closed fingers touch. The parent–child pairs among them overlap by up to 69 mm, which MuJoCo's
parent filter already ignores, as it ignores each finger against `gripper_end`. In the model
alone `world`–`link1` has no geom to act on; in a scene it keeps `link1` from colliding with the
world body's own geoms (Seeed's floor under the base). Every other pair of robot bodies collides;
at `home` none touches, the closest being `link2`–`link4` (3.2 mm apart), `gripper_end`–`link5`
(9 mm), `link1`–`link3` (13 mm) and `link1`–`link5` (14 mm). The boot checks no self-collision.

**A5 — Site `gripper_end`.** A site (5 mm, group 4) at the origin of body `gripper_end` (URDF link
`gripper_end`, fixed to `link6` by `j_gripper_end`): the SDK's end-effector frame
(`end_effector_frame: gripper_end`,
[`rebotarm_hardware.yaml` L40](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarm_bringup/config/rebotarm_hardware.yaml#L40)),
the frame of the boot's Cartesian goals (`/rebotarm/move_to_pose`, `/rebotarm/move_to_pose_ik`,
in `base_link`) and of the repeatability tolerance in `ros2.yml`.

**A6 — Keyframe `home`: the safe-home pose.** The boot has no initial configuration of its own: on
start it holds the pose the arm is in
([`hardware_manager.py` L186-L195, L243-L249, L715-L724](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarmcontroller/rebotarmcontroller/hardware_manager.py#L186-L195)).
On shutdown it runs safe home, the gripper closed and then the arm to all-zero
([L207-L224, L295-L328](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarmcontroller/rebotarmcontroller/hardware_manager.py#L295-L328);
the SDK's `home_pos = np.zeros(...)`,
[`rebotarm_endpose_controller.py` L175-L193](https://github.com/Seeed-Projects/reBotArm_control_py/blob/6415d43130d1e143c70dc106096a857ac5556f81/reBotArm_control_py/controllers/rebotarm_endpose_controller.py#L175-L193)),
and disables the motors, so the arm rests in that pose when the boot next starts; the motor zero
is that pose (D5). `home` is therefore every arm joint 0.0 rad (`joint2` and `joint3` at their
URDF lower limit 0.0) with the gripper closed (motor 0.0 rad, both fingers 0.0 m); `ctrl` is the
same. The model holds it exactly (A3).

**A7 — The D405 wrist camera.** The authoritative arm boot starts no camera, but Seeed's B601-RS
grasping guide fits an Intel RealSense D405 on the gripper and runs it with realsense-ros beside the
arm boot (`ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
depth_module.depth_profile:=640x360x30 depth_module.color_profile:=640x360x30`,
[guide L886-L898](https://github.com/Seeed-Studio/wiki-documents/blob/2d9c3d1225f1ab98b79e1382b72ca9e2148bb23a/sites/en/docs/Robotics/Robot_Kits/reBot_Arm/B601_RS/reBot_Arm_B601_RS_Grasping_Demo.md#L886-L898)),
so the camera is served by an approved community camera boot (`camera_boot_authority` in `ros2.yml`).
The model gets one `<camera>` on `gripper_end`, named after the colour optical frame, see "Wrist
camera" below. It adds no body, no geom and no mass; the camera is a sensor only.

## Wrist camera

The streams, names and frames are realsense-ros 4.58.4's for a D405 with those arguments: node
`/camera/camera` (camera_name and camera_namespace `camera`), `color/image_raw` (rgb8) and
`color/camera_info`, `aligned_depth_to_color/image_raw` (16UC1, millimetres) and
`aligned_depth_to_color/camera_info`, 640x360 at 30 Hz, all in `camera_color_optical_frame`, under
`/camera/camera/` and so not under `/rebotarm`. The driver's static transforms are
`camera_link` → `camera_depth_frame` → `camera_depth_optical_frame` and `camera_link` →
`camera_color_frame` → `camera_color_optical_frame` (`tfs.cpp`), the frame to optical frame step
being rpy −π/2, 0, −π/2; the D405's colour and depth are one sensor, so the colour frame is taken at
`camera_link` (the RealSense xacro's nominal colour offset is 0).

**Mount.** The community package the guide clones places the camera by three fixed joints
(`config/wrist_extrinsics_d405.yaml`, equal to the `mount_*` and `camera_*` properties of
`rebotarm_rs_with_d405.urdf.xacro`): `gripper_end` → `camera_mount_link` at xyz −0.1201 0.0003 0.0507,
rpy 1.5827 0.0024 1.5515; → `camera_bottom_screw_frame` at xyz −0.0003 0.0443 −0.0093, rpy −0.0225
−1.0448 −1.5460; → `camera_link` at xyz 0.01085 0.009 0.021, rpy 0 (RealSense's `_d405.urdf.xacro`: the
mesh offset 0.01465 − 0.0001 − 0.0037, the depth offset 0.009 and half the case height 0.021). The
composition (`gripper_end` → `camera_link`) is xyz −0.10873 0.00908 0.10761 m, rpy 0.003575 0.537753
−0.006233 rad (rounded to 5 and 6 decimals; `D405_LINK_XYZ`, `D405_LINK_RPY` in the generator): the
camera sits 0.109 m behind and 0.108 m above `gripper_end` looking along its x axis, 31° down toward
the fingertips, which are in view. The colour optical frame is `camera_link` turned by the optical
rotation, rpy −2.108552138 0.0030704291 −1.5788604741 relative to `gripper_end` (the `mount` of
`sensors.cameras`). The wire serves the composition as one static transform, `/camera_mount_tf`
(the stock `tf2_ros` static transform publisher), where the community package publishes the three
joints through its own `robot_state_publisher` and xacro; the arm boot's `robot_description` stays the
boot's.

**Model camera.** A MuJoCo camera looks along its −z with y up, which is the optical frame (z forward,
x right, y down) turned 180° about x: `<camera name="camera_color_optical_frame" pos quat fovy
resolution="640 360">` on body `gripper_end`. `fovy` 56.19° is the vertical field of view of the
estimated intrinsics below.

**Depth.** The rendered depth is reported in millimetres as 16UC1, 0 outside Intel's ideal range of
0.07–0.5 m (`sensors.cameras[d405_depth].range_m`, `range_basis` manufacturer): where the stereo pair
cannot match the sensor reads no depth, and the simulator reports no depth beyond those limits.

## Boot `robot_description` vs model

The boot's `robot_description` is the URDF passed through `rviz_urdf_compat.py`
(`bringup.launch.py` L59-L75 with `model:=rs`). Each difference from the model is recorded in
`ros2.yml` → `model.robot_description.differences_from_model`, and each is an adaptation here:

* **D1 — Compat split.** Every link with several visuals keeps only its first in
  `robot_description`; each further visual moves into a new child link
  `<link>__rviz_visual_<mesh stem>` on a fixed identity joint, without inertial or collision (15
  such links). The model keeps the official URDF's links with all their visuals; the 15 extra
  links exist only on the wire, as identity transforms on `/tf_static` (`ros2.yml` → `tf`).
* **D2 — Materials.** `robot_description` lightens `rs_base_black`, `rs_motor_black` and
  `rs_pla_black` to rgba `0.140 0.160 0.150 1` for RViz; the model keeps the official colours.
* **D3 — Serialisation.** `robot_description` drops the XML declaration and every comment;
  attribute values are unchanged. No effect on the model.
* **D4 — Names, axes, origins, limits, zero offsets.** Identical to the official URDF; the model
  uses them as they are.
* **D5 — Arm command frame.** The motors' positions are the URDF joint angles with no sign, offset
  or ratio, and the motor zero (the `set_zero` position) is the URDF zero, so the actuators take
  the wire's joint targets unchanged. The driver does not clamp arm targets to the URDF limits
  (motorbridge clamps MIT position targets to ±4π rad); the model clamps the actuator target to
  them (`ctrlrange`) and limits the joints (`range`), so a target outside the URDF limits is held
  at the limit, while the real arm follows it as far as its mechanics allow (no pinned source
  gives the mechanical stops). `joint2` and `joint3` have their lower limit 0.0 at the zero pose.
  The same clamp bounds the torque of the simulated motors ("Estimates"), which apply a motor's
  torque τ through the joint's position actuator by setting its target to
  q + (τ − g + kd·q̇) / kp (q and q̇ the sampled joint, g the model's gravity compensation at the
  joint, kp and kd as in "Actuation"), so that actuator force + compensation − damping = τ.
  While the joint is within |τ − g + kd·q̇| / kp of a limit and that target lies beyond it,
  `ctrlrange` holds the target at the limit, and the joint receives, and the motor reports,
  kp·(limit − q) + g − kd·q̇ instead of τ. A disabled motor (τ = 0) in the `home` pose shows it:
  `joint2` and `joint3` rest at their lower limit 0.0 under gravity torques g of 1.54 and
  6.76 N·m, their targets −g / kp (−0.010 and −0.045 rad) are held at 0.0, and the model goes on
  applying the gravity compensation and reports it as the motor's torque, where the robot's
  disabled motors apply none and the arm rests on its own body. The pose is the same; the
  reported torque is not.
* **D6 — Gripper.** The boot commands one motor in rad (A2) and reports it on the two finger
  joints as ratio = clamp(motor / 5.0, 0, 1), `gripper_joint1` = 0.05 · ratio m and
  `gripper_joint2` = 0.0715 · ratio m (the URDF upper limits): 0.0100 and 0.0143 m per motor rad,
  asymmetric, while the rack and pinion moves both fingers equally. The model moves both by
  0.008 m per motor rad (estimate, below). A simulator publishes the boot's mapping of the motor
  angle (the length of tendon `gripper`), not the model's finger positions. Positive finger
  travel opens in both (each finger axis points away from the jaw centre).
* **D7 — Velocity and effort limits.** Efforts as in "Actuation" (equal to the motors' torque
  limits). MuJoCo has no joint velocity limit, so the URDF velocities (50 rad/s on joint1–joint3,
  40 rad/s on joint4–joint6, 10 on the fingers) are not enforced; 40 rad/s exceeds the RS-00's
  33 rad/s (motorbridge). The boot's trajectories interpolate positions with no velocity limit
  either, so the commands pace the motion in the model as on the robot. With the simulated
  motors' torque target (D5), the actuator's `forcerange` clamps τ − g + kd·q̇ and the joint's
  `actuatorfrcrange` clamps τ + kd·q̇ at the motor's torque limit, where the motor clamps τ: near
  its torque limit a gravity-loaded or moving joint receives, and reports, less than the
  commanded torque.

## Model figures: manufacturer, source, estimate

| Figure | Model | Basis |
| --- | --- | --- |
| Link geometry, joint origins, axes, types and ranges, masses, centres of mass, inertias, visual meshes and colours | URDF values | official URDF (the boot's copy) |
| Total mass | 6.009 kg (URDF sum) | URDF. Seeed's specification gives ≈ 6.7 kg ([`README.md` L156](https://github.com/Seeed-Projects/reBot-DevArm/blob/0dfb49b85bc146eb3741f414be9dd2b1662345e8/README.md#L156)), 11 % more; not adjusted: the boot's gravity feed-forward uses the URDF masses |
| Arm `kp`, `kd`; gripper `kp` 50, `kd` 4; 500 Hz loop | as "Actuation" | source (`rebotarm_hardware.yaml`, `rebotarm_rs.yaml`) |
| Torque ranges 36 / 14 N·m | URDF effort | source; equal to motorbridge's RS-06 / RS-00 limits |
| Gripper range 0–5 rad | `ctrlrange` | source (`rebotarm_hardware.yaml` L48-L51) |
| Finger travel 0.008 m per motor rad | tendon coefficient 62.5 | **estimate** (below) |
| Finger damping 31250 N·s/m | finger `damping` | source kd through the estimated pinion radius |
| Finger contact geometry | Seeed's ten MuJoCo parts | manufacturer (A1) |
| Finger contact friction, condim 4 | `1 0.005 0.0001` | **estimate** (below) |
| Link contact | convex hulls of the URDF collision meshes, MuJoCo default contact parameters; self-collision except Seeed's excluded body pairs (A4) | URDF meshes; the hull is MuJoCo's approximation; the excluded pairs manufacturer (Seeed's MuJoCo scene) |
| Rotor inertia (armature), joint friction | none | missing in the pinned sources; **estimate** 0 |
| Simulated motors: power-off damping, holding before enable, PP / CSP / velocity profiles, unstated power-up values, faults, readings, 250 Hz bus sampling | the simulator's drop-in `motorbridge` | **estimate** (below); RobStride ranges and stated parameter defaults from motorbridge (source) |
| Payload 2.5 kg, reach 754 mm, repeatability < 0.2 mm | not model parameters | manufacturer ([`README.md` L153-L157](https://github.com/Seeed-Projects/reBot-DevArm/blob/0dfb49b85bc146eb3741f414be9dd2b1662345e8/README.md#L153-L157)); the repeatability is `ros2.yml`'s end-effector tolerance |
| Wrist camera mount on `gripper_end`: xyz −0.10873 0.00908 0.10761 m, rpy 0.003575 0.537753 −0.006233 rad (camera_link) | `<camera>` pose | source: the community package's mount (one unit's hand-eye result), composed (A7, "Wrist camera") |
| Wrist camera: 640x360 at 30 Hz, rgb8 and 16UC1 mm, frames and topics | `<camera>` `resolution`, wire | source: the guide's arguments and realsense-ros 4.58.4 |
| Wrist camera intrinsics fx = fy 337.2, cx 319.5, cy 179.5, no distortion; field of view 87° (56.2° vertical) | `fovy` 56.19, `hfov_deg` 87.0 | **estimate** (below); the 87° is Intel's published figure (manufacturer) |
| Lidar, IMU | none | no source records one |

## Estimates (not manufacturer data)

* **Finger travel per motor rad: 0.008 m** (`PINION_R` in the generator): the pitch radius of the
  BOM's pinion, module 1 × 16 teeth / 2
  ([BOM L138](https://github.com/Seeed-Projects/reBot-DevArm/blob/0dfb49b85bc146eb3741f414be9dd2b1662345e8/hardware/reBot_B601_RS/README.md#L138)).
  The BOM gives the module and the tooth count, not the travel; the estimate is that the pinion
  sits directly on the gripper motor's output (no reduction is listed) and drives both racks at
  its pitch radius. It gives 0.040 m per finger at the 5.0 rad open limit (an 80 mm jaw opening),
  inside the URDF finger ranges. The pinned sources disagree and none states the figure as the
  hardware's: the boot's `/rebotarm/joint_states` implies 0.0100 and 0.0143 m per rad (D6); Seeed's
  fake driver reports 0.045 m per finger at 5.0 rad, 0.009 m per rad
  ([`fake_rs_driver.py` L33](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarmcontroller/rebotarmcontroller/fake_rs_driver.py#L33));
  its MuJoCo wrapper reads 0.045 m as fully open and opens the community model's fingers 0.05 m,
  0.010 m per rad
  ([`mujoco_sync.py` L20-L22, L213-L224](https://github.com/Seeed-Projects/reBotArmController_ROS2/blob/7506dc0fc5c2a7787dcb974d89d13f9f49d16d61/src/rebotarm_mujoco_rs/rebotarm_mujoco_rs/mujoco_sync.py#L20-L22)).
* **Finger joint damping 31250 N·s/m**: kd / (2 r²), the gripper's MIT kd (source) through the
  estimated pinion radius.
* **Finger contact friction**: sliding 1, torsional 0.005, rolling 0.0001 (MuJoCo's defaults) with
  `condim 4`. The fingers are printed ABS
  ([BOM L58](https://github.com/Seeed-Projects/reBot-DevArm/blob/0dfb49b85bc146eb3741f414be9dd2b1662345e8/hardware/reBot_B601_RS/README.md#L58));
  no source gives their friction against grasped objects.
* **Rack coupling** `solref 0.005 1` of the equality `gripper_rack`: a numerical value for a stiff
  coupling; the real racks are rigidly geared.
* **Armature and friction loss: none.** The pinned sources give no rotor inertia and no gear or
  bearing friction for the RS-06 and RS-00; the model takes them as zero.
* **Link contact parameters**: MuJoCo's defaults; the sources give none.
* **Wrist camera intrinsics**: on the robot CameraInfo carries the unit's factory calibration, read
  from the device at run time (`base_realsense_node.cpp` L794-L852); no pinned source has it. The
  estimate keeps Intel's published D405 field of view, 87° × 58°
  ([product page](https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d405.html)), as
  87° across the 640 px of the guide's 640x360 mode: fx = fy = 320 / tan 43.5° = 337.2 px, square
  pixels (56.2° over the 360 rows), principal point at the image centre (319.5, 179.5), no distortion,
  colour and aligned depth alike. The depth-to-colour extrinsic is taken as identity. The mount is the
  community package's own figure, not an estimate of this record, but it is the hand-eye result for
  one unit of one arm and the position of the sensor inside the D405 case is the device's factory
  extrinsic, so `ros2.yml` allows 2 mm on it. The depth range limits 0.07 and 0.5 m are Intel's ideal
  range.
* The interface's own estimates (the arm joint final-position tolerance of 0.03 rad, the `/tf`
  rate) are labelled `estimate` in `ros2.yml`.

The simulated motors. A simulator serves the boot unmodified over a drop-in `motorbridge` whose
CAN bus and RobStride motors are simulated
(`simulator/shared/wire/robots/sim_libs/motorbridge.py`, whose docstring lists these figures
too). It reproduces what motorbridge's RobStride driver and parameter table fix (`motor.rs`,
[`registers.rs` L163-L191](https://github.com/motorbridge/motorbridge/blob/c652ce420e7da1008fa1864cc7f47d7d7c57fdb6/motor_vendors/robstride/src/registers.rs#L163-L191),
which states most parameters' defaults); the motor behaviour that neither they nor any other
pinned source fixes is estimated:

* **Power-off damping.** A disabled motor applies no torque, and its joint keeps its `damping`
  (the MIT kd of "Actuation"; the gripper's through the pinion, A2) as RobStride's power-off
  back-drive damping, which `damper` (0x702A, 0 at power-up) switches off when written 1; its
  magnitude is unpublished, and the simulated motor keeps it whatever `damper` holds.
* **Holding before enable.** Until the host first enables a motor, its actuator holds the
  spawn pose, `home` (A6), with the gains of "Actuation" and the gravity compensation. The
  robot's motors are disabled then and its folded arm rests on its own body in that pose. The
  model's convex hulls do not hold it there: limp from `home`, the model's wrist sags
  (`joint4` carries 2.0 N·m of gravity in `home`) by 0.22 rad before `link5` comes to rest on
  `link1` (A4), while `joint2` and `joint3` stay at their lower limits. After a disable the
  model's joints are limp, as the robot's are: the arm falls until its own body (A4), the scene
  or a joint limit (D5) stops it. A motor enabled in MIT mode holds the
  position it was enabled at, the same way, until its first MIT frame.
* **PP, CSP and velocity modes.** The motor's own loops (`loc_kp`, `spd_kp`, `spd_ki`, the
  current loop) are not simulated: each mode moves a setpoint along its profile, and the joint's
  position actuator (gains as in "Actuation", with the gravity compensation) tracks it.
  * PP (`run_mode` 1): a trapezoidal profile to `loc_ref` (0x7016) at up to `vel_max` (0x7024),
    accelerating and decelerating at `acc_set` (0x7025), defaults 10 rad/s and 10 rad/s². The
    PP deceleration `dcc_set` (0x702E) is stored and not used, and `limit_spd`, which
    motorbridge's `send_pos_vel` writes with every PP goal, does not bound a PP profile.
  * CSP (`run_mode` 5): the setpoint moves to `loc_ref` at up to `limit_spd` (0x7017), with no
    acceleration limit.
  * Velocity (`run_mode` 2): the setpoint's speed ramps to `spd_ref` (0x700A) at `acc_rad`
    (0x7022, default 20 rad/s²).
  * Current (`run_mode` 3), and any `run_mode` RobStride does not define: the motor applies no
    torque (the target cancels the gravity compensation and the joint's damping), its behaviour
    at `iq_ref`'s power-up value of 0. `iq_ref` is stored and not acted on: no pinned source gives
    the motors' torque constant. The vendor code never selects this mode.

  A limit of 0 is taken as its default (`limit_spd`: 10 rad/s). A `run_mode` write that changes
  the mode, an enable of a disabled motor and a disable start the next profile from rest where
  the joint is; a `run_mode` write of the current mode (motorbridge's `send_pos_vel` writes
  `run_mode` PP on every call) leaves a running profile as it is, so a new PP goal continues
  from the setpoint's position and speed.
* **Power-up values `registers.rs` does not state.** `run_mode` 0 (MIT), `limit_spd` 10 rad/s,
  `limit_cur` 23 A, `limit_torque` the motor's torque limit (36 / 14 N·m), and 0 for `iq_ref`,
  `spd_ref`, `loc_ref`, `iqf`, `damper`, `alveolous_open` and `iq_test`. Apart from `run_mode`,
  `limit_spd` and the profile targets, the parameters are stored and read back, not acted on.
* **No faults.** Undervoltage, overcurrent, overtemperature, encoder faults, stall and
  calibration faults are not simulated: every status frame's fault bits are clear and the
  fault report is (0, 0).
* **Readings.** Motor temperature 30.0 °C (`t_rotor` 0.0); bus voltage `VBUS` 48.0 V, the
  recorded DC 48 V supply.
* **Bus timing.** A motor answers at once (no CAN latency). The joints are sampled, and the
  actuator targets updated, at 250 Hz, half the SDK's 500 Hz control loop: every second status
  frame repeats the previous sample, the feedback the SDK reads is up to about 4 ms old, and a
  command acts from the next sample.
* **Torque at a joint limit and at the torque limit.** A motor's torque is applied as a
  position target, which the URDF range and the force ranges clamp (D5, D7).

## Verification (2026-10-09, MuJoCo 3.5.0, the model alone, timestep 0.002 s)

* `mujoco.MjModel.from_xml_path("robots_specs/rebot_b601_rs/model.xml")` loads once the meshes are
  fetched: 10 bodies besides the world, nq 8 (`joint1`–`joint6`, `gripper_joint1`,
  `gripper_joint2`), nu 7, 1 tendon, 1 equality, 1 site, 40 meshes, 43 geoms, 10 contact
  exclusions; total mass 6.009 kg.
* Self-collision (A4): at `home` there is no contact between robot geoms.
* From `home` with `ctrl` = `home`, 2 s: no joint moves (|q| < 1e-15) and no contact arises; the
  gravity compensation holds the folded pose exactly. MuJoCo 3.3.1 gives the same.
* `ctrl` of `joint1`–`joint6` = [0.3, 0.75, 0.55, 0, 0, 0] (`ros2.yml`'s arm example): reached
  within 1e-5 rad after 4 s. Neither that step nor a 3 s linear ramp to the same target, nor the
  gripper opened 0 → 5 rad and closed again (each over 3 s) at `home` and at that target, brings
  two robot geoms into contact.
* `ctrl` of `gripper` = 5.0: both fingers at 0.0400 m and tendon `gripper` at 5.000 rad after 3 s.
  With the fingers at 0 and `ctrl` 5.0, the actuator force is clamped at 14 N·m and the force on
  each finger joint at 500 N (A2).
* A1 placement, in the `home` pose and the base frame (site `gripper_end` at x 0.3017 m, y 0,
  z 0.2177 m): with the fingers at 0 the `gripper_left` parts span y −0.0345…0 m and the
  `gripper_right` parts 0…0.0343 m, meeting at the jaw centre plane, with the x and z extent of
  the visual fingers to within 1 mm, apart from the visual parts reaching 6 mm further toward the
  wrist; with both fingers at 0.02 m each set has moved 0.02 m outward with its finger.
* Limp from `home` (no actuator force, no gravity compensation): `joint4` sags 0.22 rad and
  `link5` comes to rest on `link1` within 1 s (0.3 mm contact penetration), `joint2` and
  `joint3` held at their lower limits (MuJoCo 3.3.1: the same); without self-collision the
  wrist would keep falling (−1.03 rad after 3 s, still moving).
* The wrist camera (A7), rendered offscreen at `home` over a plane 0.22 m below the gripper: it looks
  forward and down along `gripper_end`'s x axis and sees both fingertips with the gripper open
  (finger joints 0.04 m), the plane filling the rest of the frame; `camera_color_optical_frame` is at
  (0.1930, 0.0091, 0.3254) m in the base frame, 640x360, `fovy` 56.19°.
* `robots_specs/tests` (`test_models_load_in_mujoco`, `test_model_conventions`,
  `test_model_settles_on_a_floor`, `test_derived_model_is_reproducible`) pass.

## Open items

* Not measured on a real B601-RS: the command watchdogs (derived from source, `ros2.yml`), the
  finger travel per motor rad, the grip force, joint friction and rotor inertia.
* About 0.7 kg separates Seeed's stated weight from the URDF masses, and no pinned source explains
  it. The real arm's feed-forward also uses the URDF masses, so on the robot any gravity the URDF
  misses is left to the MIT loop (a static error of residual torque / kp); the model, compensating
  its own masses exactly, shows no such error.
* The community MJCF `LAN-GER/reBot-B601-RS-for-mujoco_sim` is unlicensed and not pinned by Seeed;
  it is not used (see the introduction).
