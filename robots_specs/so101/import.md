# SO-101 — MuJoCo import notes

`model.xml` in this folder is a **derived model, not manufacturer-provided**. The official
TheRobotStudio MJCF `so101_new_calib.xml` is kept next to it **unmodified**; `model.xml` exists
because the authoritative boot differs from the official model (names, zero offsets, limits)
and publishes a wrist camera the official MJCF lacks (robot specification §1, "MuJoCo import").
The ROS interface this model serves is `ros2.yml` — an approved community interface
(`ros2_so_arm` + `usb_cam`), not a manufacturer one.

## Sources and revisions

| What | Source | Revision |
| --- | --- | --- |
| Official URDF, MJCF, meshes (`so101_new_calib.urdf`, `so101_new_calib.xml`, `assets/`) | <https://github.com/TheRobotStudio/SO-ARM100>, `Simulation/SO101` | `aec17bbc256d1a7342d53aaa4950595d4c30b40d` |
| Authoritative arm boot and its `robot_description` | <https://github.com/ros-physical-ai/ros2_so_arm>, `so_arm101_description` | `e166df9d51f43b24da9b99047c6c51c306bda74f` |
| Wrist-camera driver | <https://github.com/ros-drivers/usb_cam> (0.8.1) | `cb2ae6bc0a312de629f84fef933fb3e07dac1ef8` |
| Hardware plugin (servo tick conversion, speed/acceleration written) | <https://github.com/ros-physical-ai/feetech_ros2_driver> 0.2.2 (Jazzy release) | `e424839f0fb23cc0b0ade8d6e5ea166e30811356` |
| Wrist-camera mount hardware | SO-ARM100 `Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module/`, photo `media/UVC_cam_mount_so101.jpg` | `aec17bbc256d1a7342d53aaa4950595d4c30b40d` |

Every further dependency release (ros2_control 4.48.1, ros2_controllers 4.42.1, …) is pinned in
`ros2.yml` → `sources`.

File digests (sha256) at the time of conversion:

* `so101_new_calib.urdf` `3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c` (official, unmodified)
* `so101_new_calib.xml` `d75253eb568e8a7214db9c631ab7bed4217f608a26f7276ebe9a7636cac82580` (official, unmodified)
* `boot/robot_description.urdf` — see `ros2.yml` → `model.robot_description.content.sha256_after_banner`
  (digest of everything after the xacro banner comment)
* meshes: `robots_specs/meshes.sha256` (`so101/assets/*`)

## Files in this folder

* `so101_new_calib.urdf`, `so101_new_calib.xml`, `assets/` — official, unmodified (meshes are
  fetched by `robots_specs/tools/fetch_meshes.py`, not committed).
* `boot/robot_description.urdf` — the authoritative boot's `robot_description`, generated
  (step 1 below). Its meshes are `package://so_arm101_description/meshes/<name>.stl`; those
  thirteen STL files are byte-identical (sha256) to `assets/<name>.stl` of SO-ARM100 at the
  pinned revision, so `package://so_arm101_description/meshes/` resolves to `assets/`.
* `model.xml` — derived MJCF (step 2 below).
* `ros2.yml` — the ROS 2 interface (schema `robots_specs/SCHEMA.md`).

## Conversion steps

1. **Boot `robot_description`.** Expand the bringup xacro exactly as
   `controllers_bringup.launch.py` does (`load_xacro` mappings), with xacro 2.1.1:

   ```python
   import xacro
   B = "<ros2_so_arm checkout>/so_arm101_description"
   doc = xacro.process_file(f"{B}/urdf/so_arm101.urdf.xacro", mappings={
       "prefix": "", "ros2_control_file": f"{B}/control/so_arm101.ros2_control.xacro",
       "ros2_control_hardware_type": "real", "usb_port": "/dev/LeRobotFollower",
       "urdf_file": f"{B}/urdf/so_arm101_macro.xacro"})   # = the xacro:arg default
   open("boot/robot_description.urdf", "w").write(doc.toxml())
   ```

   The first comment (xacro banner) names the xacro file's absolute path; it varies by
   installation and is written here as `<so_arm101_description share>/urdf/so_arm101.urdf.xacro`.

2. **Derived MJCF.** Start from the official `so101_new_calib.xml` and apply, with
   `robots_specs/tools/models/so101.py` (Python 3, numpy, scipy):
   * `model="so101"`; `meshdir="."`, every mesh named after its file stem and read from `assets/`;
   * bodies, joints and actuators renamed to the boot names (table below);
   * every jointed body's `pos`/`quat` replaced by the boot joint origin (URDF `rpy` → quaternion);
   * joint `range` and actuator `ctrlrange` replaced by the boot joint limits;
   * site `gripper_frame_link` added at the boot's `gripper_frame_joint` pose (the official
     `gripperframe` site is kept; it has a different orientation, see below);
   * camera `default_cam` added in `gripper_link` (wrist camera, below);
   * keyframe `home`: the spawn pose, a natural ready pose (shoulder lift -0.6 rad, elbow 1.2
     rad, every other joint 0.0), chosen over the boot's `initial_position` of 0.0 on every
     joint, which holds the arm stretched out awkwardly (changed 2026-10-02).

   Everything else — meshes and their poses, materials, inertials (mass, CoM, full inertia),
   the `sts3215` joint class (damping 0.60, frictionloss 0.052, armature 0.028), the backlash
   class, the actuator gains (`kp` 998.22, `kv` 2.731, `forcerange` ±3.35) and the collision
   geoms — is the official file's.

## Adaptations: boot `robot_description` vs official model

The boot's `robot_description` governs the wire and `model.xml`.

| Official body / joint | Boot link / joint (model.xml) |
| --- | --- |
| `base` / — | `base_link` / (`world_to_base_joint`, fixed, identity to `world`) |
| `shoulder` / `shoulder_pan` | `shoulder_link` / `shoulder_pan_joint` |
| `upper_arm` / `shoulder_lift` | `upper_arm_link` / `shoulder_lift_joint` |
| `lower_arm` / `elbow_flex` | `lower_arm_link` / `elbow_flex_joint` |
| `wrist` / `wrist_flex` | `wrist_link` / `wrist_flex_joint` |
| `gripper` / `wrist_roll` | `gripper_link` / `wrist_roll_joint` |
| `moving_jaw_so101_v1` / `gripper` | `jaw_link` / `gripper_joint` |
| site `gripperframe` | site `gripper_frame_link` added (boot fixed joint `gripper_frame_joint`) |

Joint limits (rad; every joint effort 10, velocity 10 in both):

| Joint | Official | Boot / model.xml |
| --- | --- | --- |
| shoulder_pan_joint | ±1.91986 | ±1.91986 |
| shoulder_lift_joint | ±1.74533 | ±1.74533 |
| elbow_flex_joint | [-1.69, 1.69] | **[-1.69, 1.54]** |
| wrist_flex_joint | ±1.65806 | **±1.6** |
| wrist_roll_joint | [-2.74385, 2.84121] | **±2.3** |
| gripper_joint | [-0.174533, 1.74533] | **[0.0, 1.70]** |

Zero offsets. Axes are identical (`0 0 1` in every child frame); three joint origins differ by a
rotation about the joint axis, so `q_official = q_boot + delta`:

| Joint | Boot origin rpy | Official origin rpy | delta (rad) |
| --- | --- | --- | --- |
| wrist_roll_joint | -1.57 3.14 0 | 1.5708 0.0486795 3.14159 | +0.04709 |
| gripper_joint | 1.5708 0.18 0 | 1.5708 -5.24e-08 0 | -0.18 |
| wrist_flex_joint | yaw -1.57 | yaw -1.5708 | +0.0008 |

(The boot's rounded 3.14/1.57 values leave a further 7.9e-4 rad rotation off the joint axis in
`gripper_link`; the model reproduces the boot values exactly.) The boot's `gripper_joint` 0.0
(its lower limit, jaw closed) is the official model's -0.18 rad, slightly past the official lower
limit -0.1745.

The official `gripperframe` site is rotated 90° about y; the boot's `gripper_frame_link` is
rotated 3.14159 rad about y. Both are kept; only `gripper_frame_link` is on the wire (`/tf_static`).

The boot's root link is `world`; MuJoCo's world body stands for it. `base_link` is the top body
and has no free joint: the arm is fixed where it is placed (SCHEMA.md model conventions).

Checks performed (MuJoCo 3.14.0): `model.xml` loads with `mujoco.MjModel.from_xml_path`; body
poses equal a forward-kinematics evaluation of `boot/robot_description.urdf` to 1.7e-9 over 50
random configurations; at `q_boot` every body equals the official model at `q_boot + delta` (≤ 2.3e-5
m, ≤ 7.9e-4 rad); position actuators track a step target on all six joints to within 4.3e-4 rad
after 3 s under gravity; no contacts at `home`.

## Actuation (estimates and source figures)

* The official actuator gains model an STS3215 at servo P coefficient 16 (comment in
  `so101_new_calib.xml`, derived by a third party, not TheRobotStudio measurement). The boot writes
  P coefficient 16 to servos 2–6 but **8 to shoulder_pan** (`p_cofficient` in the ros2_control
  xacro). `model.xml` keeps the official gain on every joint; the shoulder_pan stiffness
  difference is an unmodelled estimate.
* The hardware plugin writes every goal with speed 2400 ticks/s (3.68 rad/s at 4096 ticks/rev)
  and acceleration register 50 (`ros2.yml` → `motions[arm].command.servo_profile`). A MuJoCo
  position actuator has no speed limit, so a simulator must rate-limit the servo targets to
  reproduce it; the model itself does not encode it.
* Commands and feedback are in rad in the boot's joint frames (0 = the per-joint `offset` tick
  of the ros2_control xacro), which are the model's joint frames.

## Wrist camera

* **Wire:** `usb_cam` 0.8.1 started as the pinned README documents first,
  `ros2 run usb_cam usb_cam_node_exe` with default settings: node `/usb_cam`, `/image_raw`
  (640×480, `yuv422_yuy2`, 30 Hz), `/camera_info`, `frame_id` `default_cam`. (An earlier revision
  of these records put the node in a `/wrist` namespace; no pinned source defines that namespace,
  so it is not used. `camera.launch.py` would instead load the example `params_1.yaml`, a
  `camera1` node and a calibration of an unrelated test camera; it is not the boot.)
* **Calibration:** `camera_info_url` is empty, so the published CameraInfo is uncalibrated
  (only width, height and frame_id set; K, D, R, P zero). No calibration exists to reproduce.
* **Hardware:** a 32×32 mm USB UVC camera module on the SO101 hex-nut wrist-camera mount
  (SO-ARM100 `Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module`, whose README recommends a
  module of at least 720p/30 fps and running it at 640×480, 30 fps).
* **Field of view — estimate:** horizontal 70°, so `fovy` = 2·atan(tan 35° · 480/640) = 55.41°.
  Not a manufacturer figure for any specific module.
* **Mount pose — estimate:** camera `default_cam` in `gripper_link` at
  `pos -0.008 0.045 -0.01`, `xyaxes 1 0 0  0 0.966 -0.259` (looking along the fingers, −z,
  pitched 15° toward them; image right = +x). In ROS optical-frame terms (z forward, x right,
  y down) that is xyz `[-0.008, 0.045, -0.01]`, rpy `[2.8798, 0, 0]` relative to `gripper_link`.
  Estimated from the pinned photo `media/UVC_cam_mount_so101.jpg`: the lens sits on a plate rising
  from the Wrist_Roll_Follower on the side where, viewed from the fingertips, the moving jaw is on
  the left (gripper_link +y), centred over the jaw gap (x of `gripper_frame_link`), about 4.5 cm
  off the jaw plane, near the wrist. Not measured; the real mount may differ by about 2 cm and
  10° (stated in `ros2.yml` → `sensors.cameras[wrist].mount.notes`; no acceptance tolerance is
  recorded for it, so a simulator reproduces the recorded pose exactly).
* **Image orientation — derived (checked 2026-10-01):** the roll about the optical axis puts the
  jaws at the bottom of the image, side by side (moving jaw on the right, from the photo below),
  the plate side up. This matches the pinned photos (`media/UVC_cam_mount_so101.jpg`,
  `media/Wrist_Cam_Mount_32x32_UVC_module_1.jpg`/`_2.jpg`: lens on the plate above the jaws,
  jaws opening across the image) and recorded SO-101 wrist-camera footage (for example the
  Hugging Face dataset `youliangtan/so101-table-cleanup`, `observation.images.wrist`: jaws at the
  bottom). At the boot's home (every joint 0.0) the gripper is rolled so the jaws open in the
  **vertical** plane, moving jaw on top: the official model at qpos 0 and the LeRobot SO-101
  calibration video's "middle of range" pose
  (`huggingface/documentation-images` `lerobot/calibrate_so101_2.mp4`, about 9 s) agree. The camera
  then sits beside the gripper on the robot's left, so **at home the image is rolled 90°: world up
  is image right**, and a vertical edge appears horizontal. This is what the mounted camera sees,
  not a model error. The image is upright (world up = image up, jaws at the bottom) at
  `wrist_roll_joint` = −π/2 (the jaws open horizontally and the camera is above them); at +π/2 it
  is upside down.
* The boot publishes **no transform** for `default_cam` (the `robot_description` has no camera
  link); the pose above is used only to render the image.

## Contact

Collision geometry is the official file's: each part's mesh (convex hull in MuJoCo), group 3,
default contact parameters. MuJoCo filters contacts between parent and child bodies, so the
moving jaw does not collide with the fixed jaw (as in the official model). No contact modelling
was added.

## Open items

* Command watchdogs are derived from source (none present), not measured on the real robot.
* Wrist-camera mount pose and field of view are estimates; the camera has no calibration.
* The feetech plugin revision is not pinned by `ros2_so_arm`; 0.2.2 (Jazzy release) is used
  because it is the release whose parameter names the bringup's xacro uses (`offset`,
  `p_cofficient`). A source build of feetech_ros2_driver `main` ignores `offset` (deprecated,
  uses `homing_offset` in servo EEPROM) and `p_cofficient`, which would move the joint zeros.

## Conversion script (step 2)

`robots_specs/tools/models/so101.py` (Python 3, numpy, scipy) writes `model.xml` from the
official `so101_new_calib.xml` and `boot/robot_description.urdf`; run it from any directory
once the meshes are fetched. It reproduces `model.xml` byte for byte.
