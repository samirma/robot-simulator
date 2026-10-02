# myCobot 280 Pi + Adaptive Gripper — MuJoCo import notes

`model.xml` in this folder is a **derived model, not manufacturer-provided**. Elephant Robotics
publishes no MJCF for the 280 Pi. The one in
[elephantrobotics/mycobot_mujoco](https://github.com/elephantrobotics/mycobot_mujoco)
(`xml/mycobot_280jn_mujoco.xml`) is for the JetsonNano variant (different base: `joint1_jet`,
joint2 at z = 0.15756 m instead of the Pi's 0.13956 m, other J3/J5/J6 ranges) and is not used as
this robot's model; only its link inertials are borrowed, as labelled estimates (below).

## Sources

| what | source | revision |
| --- | --- | --- |
| URDF `mycobot_280_pi_adaptive_gripper.urdf` (sha256 `8c8fbea8…7f12`) | `elephantrobotics/mycobot_ros2`, `mycobot_description/urdf/mycobot_280_pi/` | `d42ff61a78122c79246623391540d75738b03b23` (branch `humble`) |
| meshes `urdf/mycobot_280_pi/*.dae`, `urdf/adaptive_gripper/*.dae` (+ `joint2..7.png` textures) | same | same |
| arm link inertials (estimate) | `elephantrobotics/mycobot_mujoco`, `xml/mycobot_280jn_mujoco.xml` | `72b7a1b01800304d90a33ef2c17776a3e9687472` |
| joint limits, gripper mapping | `pymycobot` 4.0.7 wheel (sha256 `cbcde962…adc4`), `robot_info.py` `MyCobot280` | 4.0.7 |
| weight, payload, reach, repeatability, joint ranges | [product parameters](https://docs.elephantrobotics.com/docs/mycobot_280_pi_en/1-ProductInformation/2.ProductParameter/2-ProductParameters.html) | retrieved 2026-09-30 |
| gripper weight, range, force, 100 = open | [adaptive gripper](https://docs.elephantrobotics.com/docs/mycobot_280_pi_en/4-SupportAndService/Accessories/AdaptiveGripper.html) | retrieved 2026-09-30 |

## Conversion steps

`robots_specs/tools/models/mycobot280.py` (standard library + the MuJoCo bindings for the
load check) performs steps 2–5 and reproduces `model.xml` byte for byte.

1. `robots_specs/tools/fetch_meshes.py mycobot280` fetches the COLLADA meshes at the pinned
   revision into `urdf/…` (the URDF's `package://mycobot_description/` resolves to this folder)
   and derives from each, under `derived_meshes/urdf/…` (every file is in
   `robots_specs/meshes.sha256`): `<name>.stl`, every triangle in one binary STL (the
   **collision** mesh), and `<name>.dae.m<k>.obj` + `<name>.dae.mtl`, the **visual** mesh split
   into one OBJ per COLLADA material with its own normals and texture coordinates, and each
   material's own diffuse colour, opacity and texture image (derived adaptation: MuJoCo applies
   one material per mesh geom). The conversion applies each file's `<unit meter>` (0.001 for
   `G_base`, `joint1_pi`, `joint5` and the gripper parts, 1 for `joint2..4,6,7`) and its node
   transforms; all files are `Z_UP`.
   Materials as pinned (each file has exactly one): `G_base` and every gripper part
   `a0000000-…` diffuse 0.980392 (near-white); `joint1_pi` diffuse 0.6 (grey); `joint2`,
   `joint3`, `joint4`, `joint6`, `joint7` a texture, `jointN.png` (fetched beside the `.dae`);
   `joint5` binds `a0000000-…`, which the file does not define, so it is white without texture
   (its library's `Material_001` → `joint5.png` is bound to no geometry). The URDF defines no
   `<material>`, so every colour is the mesh's own.
2. Every URDF link becomes a body of the same name, nested along the URDF joints; every joint
   `origin` becomes the child body's `pos`/`quat` (URDF rpy converted to a quaternion). The root
   link `g_base` is the top body, with no freejoint (an arm is fixed where it is placed).
3. Every revolute joint becomes a hinge of the same name, axis and range. Every `<visual>` and
   `<collision>` mesh becomes geoms at the URDF origin: visual (group 2, no contact) one geom per
   material part with a MuJoCo material (and texture) of that part's colour; collision (group 3)
   the whole-file STL, which MuJoCo collides as its convex hull.
4. The five `<mimic>` joints become `<equality><joint>` constraints
   (`polycoef="offset multiplier 0 0 0"`) to `gripper_controller`.
5. One position actuator per commanded joint (the six arm joints and `gripper_controller`),
   named after its joint, in rad, `ctrlrange` = the URDF limits. Keyframe `home` = the spawn
   pose, a natural ready pose: `joint4_to_joint3` (elbow) -1.2 rad, every other joint 0
   (upper arm up, forearm folded forward and down), chosen over all zeros, which holds the arm
   stretched straight up (changed 2026-10-02; the gripper stays at 0 rad, i.e. value 83 of
   100 — the boot's first command decides the real start pose; the robot has no defined
   power-on pose).
6. `<contact><exclude>` pairs: every pair of gripper bodies (their convex hulls overlap by
   construction), each arm body with its grandchild, and `joint1`/`joint2`, `g_base`/`joint2`
   (`g_base` and `joint1` are welded to the world when the arm stands alone, so MuJoCo's
   parent filter does not apply to them).

The model was checked with MuJoCo 3.14.0: it loads, has no contacts at `home`, and each
actuator reaches its target within 0.0075 rad in 3 s of simulation (timestep 0.002 s) for
several arm/gripper targets across the ranges, with no self-contact on the way.

## Boot `robot_description` vs model

The boot publishes `xacro mycobot_280_pi_adaptive_gripper.urdf`, element-for-element this URDF.
Joint names, axes, zero offsets and arm/gripper limits are identical in the model. Adaptations:

* **Mimic joint limits dropped.** The URDF gives the mimic joints their own limits, which the
  mimic relation exceeds over `gripper_controller`'s range [-0.74, 0.15]
  (`gripper_left3_to_gripper_left1`, `gripper_right3_to_gripper_right1` [-0.5, 0.5],
  `gripper_base_to_gripper_right3` [-0.15, 0.7]). robot_state_publisher ignores them; the model
  leaves the mimic joints unlimited so they follow the relation over the whole command range.
* **Effort/velocity.** The URDF's `effort="1000.0"` and `velocity="0"` are placeholders.
* **Closed gripper linkage.** The adaptive gripper is a four-bar per finger; the URDF models it
  as an open chain with mimics and the model keeps that open chain (fingers stay parallel through
  the mimic relations, not a closed loop).

## Estimates (not manufacturer figures)

* **Arm link inertials** (`joint2` … `joint6_flange`): taken from the official myCobot 280
  JetsonNano MJCF, whose link frames equal the Pi URDF's (checked: same joint-origin rotations),
  masses 0.153, 0.400, 0.219, 0.058, 0.090, 0.012 kg. The arm links are the same design on both
  variants, but these are not Pi figures.
* **Base and gripper masses**: `g_base` 0.10 kg and `joint1` (Pi base housing with the Raspberry
  Pi) 0.25 kg; gripper 0.110 kg in all (manufacturer weight), split as base 0.072 and per side
  0.010 / 0.004 / 0.005 kg (links 3 / 2 / 1). Inertia from each link's collision hull.
* **Conflict:** the model's arm mass is 1.28 kg (+0.11 kg gripper), while the manufacturer gives
  860 g for the 280 Pi; the JetsonNano MJCF's moving links alone already weigh 0.93 kg. No
  per-link Pi figure resolves this; the base links only matter when the arm rides the myAGV.
* **Actuators**: servo stiffness `kp` 80 N·m/rad (arm) and 5 N·m/rad (gripper), critically
  damped; torque limits ±2 (J1), ±3 (J2, J3), ±1.5 (J4), ±1 (J5, J6) and ±0.3 N·m (gripper).
  The manufacturer publishes no servo torque; the limits hold the 250 g payload at the 280 mm
  reach (≈1.3 N·m at J2) with margin. Joint armature 0.005, damping 0.05, friction loss 0.01.
* **Joint speed**: the boot commands `send_angles(…, 25)` (speed 25 of 1..100) and gripper speed
  80; the manufacturer does not document the deg/s these map to, so the model has no velocity
  limit — the simulator must pace motions, and no joint speed figure is claimed here.
* **Contact**: friction 0.8 (arm), 1.0 with condim 4 (gripper fingers); estimates.

## Manufacturer figures preserved

Joint ranges J1 ±168°, J2 ±140°, J3 ±150°, J4 ±150°, J5 −155…+160°, J6 ±180° (= URDF limits,
= pymycobot limits); gripper command 0..100 with 100 = open, mapped from
`gripper_controller` ∈ [−0.74, 0.15] rad (0.15 = open). Payload 250 g, reach 280 mm,
repeatability ±0.5 mm, gripper range 20–45 mm, gripping force 150 g.
