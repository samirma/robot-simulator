# Robot Specifications — High-Level Specification

## 1. Purpose and layout

This document describes the robots and the files needed to import them into
MuJoCo, reproducing the real hardware's geometry, physical properties, motion,
sensors and control interfaces as documented by their manufacturers. It is
self-contained. Each robot or assembly has its own section below.

Each section defines a stable **robot id**, its identity and kind, its **embodiment**
(the exact hardware variant, its controller, and its camera and lidar hardware), its
authoritative sources and boot, and its required files. Each individual robot's folder is always
`robots_specs/<id>/`, with the folder name exactly matching its robot id. A composite
robot references its component robots' distinct folders, rather than duplicating
their assets or interface descriptions in a composite folder. Paths below are relative to the
repository root. These are required asset paths, not a statement that the files
have already been downloaded: meshes are not committed, and the simulator's
`run.sh setup` fetches them from the pinned sources, verified against
`robots_specs/meshes.sha256`.

Each robot folder must contain:

* The official URDF and every mesh or other asset it references, preserving upstream
  filenames and relative paths. If the manufacturer provides no composite URDF,
  use the component robots' official URDFs; the mounting transform is recorded in the
  assembly's section of this document.
* Exactly one of `ros.yml` (ROS 1) or `ros2.yml` (ROS 2), for the ROS dialect of the
  recorded boot; the filename identifies that dialect.
* An official MJCF and its assets where one exists for the same hardware embodiment.
  A model of a different hardware variant must not be presented as that robot's official model.

The ROS interface descriptions record every authoritative node, topic, service,
action and parameter, including message types, frames, rates, camera streams,
optional features, each command's units and limits, and the stop command. The stop
command is recorded for every velocity-driven motion (driving, walking); every other
motion ends on reaching its goal or end state. A topic published only on events or on demand
is recorded as non-periodic and has no rate; every other topic records its rate. Record whether each command has a watchdog or timeout and its interval,
established by measurement on the real robot running the authoritative boot, with the
method and date of the measurement. Record the ROS distribution, hardware boot
configuration and pinned source revisions used to establish the interface. Do not
invent a manufacturer ROS interface when none exists; identify an approved community
interface explicitly. Describe the robot's real-hardware interface, separately
from transport or discovery services added by a simulation application.

Each robot section names its **authoritative boot**: the launch and parameter files, at
the pinned revision, that the vendor documentation starts for normal operation, and the
vendor data files they load (such as action groups and gait parameters). The interface
description records that boot. Endpoints that exist only when an optional plugin is
installed or through a launch other than the authoritative boot, such as mapping and
navigation, are recorded as `optional`. Each camera's raw image stream is marked as a
camera.

### MuJoCo import and real-hardware fidelity

Use the official MJCF for the exact hardware variant when available. Otherwise,
import the official URDF and retain a reproducible MJCF conversion in the robot's
folder as `model.xml`, with all referenced assets. Label converted models as
derived, not manufacturer-provided. Record conversion steps, source revisions
and any adaptations in `import.md` in the same folder.

Where the authoritative boot's `robot_description` and the model disagree on joint
names, axes, limits or zero offsets, the boot's `robot_description` governs; `import.md`
lists each difference as an adaptation. An official MJCF is never modified: when it
lacks an element the interface requires, such as a camera the boot publishes, a derived
`model.xml` adds it, labelled as derived, with the element's pose source.

The imported model must preserve:

* Link dimensions, mesh scales, joint origins and axes, joint types, ranges,
  velocity and effort limits, and the manufacturer's coordinate frames.
* Masses, centres of mass and inertia tensors; collision geometry and contact
  properties appropriate to the real wheels, feet, grippers and body.
* Actuator mappings, gearing, command units, control modes and limits needed to
  reproduce real arm, gripper, wheel or walking behaviour.
* Sensor poses and documented characteristics, including camera resolution and
  field of view, lidar geometry, and feedback frames and rates.

Use pinned manufacturer models, real-hardware software and documentation as the
sources for these properties. Document missing values, conflicting sources and
necessary estimates in `import.md`; distinguish estimates from manufacturer
specifications. Conversion must not silently change the robot's embodiment or
exceed its documented capabilities. For an assembly, record its mounting transform
and source in its section of this document; component import notes may only refer to
it. Preserve both components' physical properties and interfaces.

## 2. myAGV

* **Robot id:** `myagv`
* **Folder:** `robots_specs/myagv/`
* **Manufacturer:** Elephant Robotics
* **Kind:** `mobile_base`.
* **Official product:** <https://www.elephantrobotics.com/en/myagv-2023-pi-en/>
* **Official URDF:** `robots_specs/myagv/myAGV.urdf`
* **ROS interface:** `robots_specs/myagv/ros.yml` (ROS 1).
* **Official source:** <https://github.com/elephantrobotics/myagv_ros>, branch
  `myagv_ros_2023Pi`, revision `c71f3cc574e5ed1973a925238eabe88662cfa701`.
* **Camera boot:** <https://github.com/ros-drivers/usb_cam>, revision
  `addab4a65fdf65c460fec2cc3eee8fab94699370`.
* **Documentation:** <https://docs.elephantrobotics.com/docs/myagv_pi23_en/>

The camera boot is an approved community camera boot, not manufacturer-provided, and the
camera streams it publishes are required. Preserve the official base geometry, wheel joints and sensor mounting. The interface
description must cover driving and its stop, odometry, transforms, lidar and camera
streams. Mapping and navigation endpoints come from separate launches and are recorded
as `optional`.

## 3. SO-101

* **Robot id:** `so101`
* **Folder:** `robots_specs/so101/`
* **Manufacturer:** TheRobotStudio
* **Kind:** `arm`.
* **Official project:** <https://github.com/TheRobotStudio/SO-ARM100>
* **Official URDF:** `robots_specs/so101/so101_new_calib.urdf`
* **Official MJCF:** `robots_specs/so101/so101_new_calib.xml`
* **ROS interface:** `robots_specs/so101/ros2.yml` (ROS 2).
* **Official model source:** `SO-ARM100`, revision
  `aec17bbc256d1a7342d53aaa4950595d4c30b40d`, under `Simulation/SO101`.
* **Community ROS bringup:** <https://github.com/ros-physical-ai/ros2_so_arm>,
  revision `e166df9d51f43b24da9b99047c6c51c306bda74f`.
* **Wrist-camera source:** <https://github.com/ros-drivers/usb_cam>, revision
  `cb2ae6bc0a312de629f84fef933fb3e07dac1ef8`. The wrist-camera boot is the README's
  default invocation, `ros2 run usb_cam usb_cam_node_exe` with no parameters file: node
  `/usb_cam`, no namespace, image stream `/image_raw`.
* **Documentation:** <https://huggingface.co/docs/lerobot/so101>

The manufacturer supplies the physical models but no ROS interface. `ros2.yml`
describes the pinned community real-hardware bringup and wrist-camera boot,
including arm and gripper commands and feedback; it is not a manufacturer-provided
ROS interface. Preserve the upstream model assets under `assets/`. The wire's names,
frames, joint conventions and `robot_description` follow the pinned `ros2_so_arm` and
`usb_cam` boot; `import.md` records how they map onto the SO-ARM100 model and the wrist
camera's hardware, mount and pose source.

## 4. AiNex

* **Robot id:** `ainex`
* **Folder:** `robots_specs/ainex/`
* **Manufacturer:** Hiwonder
* **Kind:** `humanoid`.
* **Official product:** <https://www.hiwonder.com/products/ainex>
* **Official URDF:** `robots_specs/ainex/ainex.urdf`
* **ROS interface:** `robots_specs/ainex/ros.yml` (ROS 1).
* **Official source:** <https://github.com/Hiwonder/ainex>, branch `main`, revision
  `e8fe2a816797cf83054135160df5a82ec3596a69`.
* **Documentation:** <https://docs.hiwonder.com/projects/AiNex/en/raspberry-pi5-version/>

Preserve the official humanoid joints and meshes. The interface description must
cover walking and its stop, head control, action-group playback and camera streams.
Record sensor calibration sources and any estimates explicitly.

## 5. myCobot 280

* **Robot id:** `mycobot280`
* **Folder:** `robots_specs/mycobot280/`
* **Manufacturer:** Elephant Robotics
* **Kind:** `arm`.
* **Embodiment:** myCobot 280 Pi with the myCobot Adaptive Gripper.
* **Official product:** <https://www.elephantrobotics.com/en/mycobot-280-pi-2023-en/>
* **Official URDF:** `robots_specs/mycobot280/mycobot_280_pi_adaptive_gripper.urdf`
* **ROS interface:** `robots_specs/mycobot280/ros2.yml` (ROS 2).
* **Official description source:** <https://github.com/elephantrobotics/mycobot_ros2>,
  branch `humble`, revision `d42ff61a78122c79246623391540d75738b03b23`.
* **ROS interface source:** the same `mycobot_ros2` revision, whose real-hardware boot is
  this robot's authoritative boot (§1).
* **Gripper:** <https://shop.elephantrobotics.com/collections/mycobot/products/adaptive-gripper>
* **Documentation:** <https://docs.elephantrobotics.com/docs/mycobot_280_pi_en/>

Preserve the official arm and gripper geometry and their referenced assets. Record
the real-hardware ROS 2 boot from that source in `ros2.yml`, including arm and gripper
commands and the feedback actually published; do not invent measured feedback.
The model published in <https://github.com/elephantrobotics/mycobot_mujoco> as
`mycobot_280jn_mujoco.xml` is for the JetsonNano variant, so it is not an official
model of this Pi embodiment. Use the Pi URDF for a derived MuJoCo model unless an
official MJCF for the exact Pi embodiment is available.

## 6. myAGV + myCobot 280

* **Robot id:** `myagv_mycobot280`
* **Base folder:** `robots_specs/myagv/` — the `myagv` robot's official URDF,
  meshes and ROS 1 interface description.
* **Arm folder:** `robots_specs/mycobot280/` — the `mycobot280` robot's official
  URDF, meshes and ROS 2 interface description, including its adaptive gripper.
* **Manufacturer:** Elephant Robotics
* **Kind:** `mobile_manipulator`.
* **`base`:** `myagv` (§2).
* **`arm`:** `mycobot280` (§5), including its adaptive gripper.
* **Official product/example:** <https://www.elephantrobotics.com/en/myagv-2023-pi-en/>
* **Official component URDFs:** `robots_specs/myagv/myAGV.urdf` and
  `robots_specs/mycobot280/mycobot_280_pi_adaptive_gripper.urdf`.
  Elephant Robotics publishes no official composite URDF; the arm URDF comes from
  the pinned `mycobot_ros2` description source in §5.
* **ROS interface descriptions:** `robots_specs/myagv/ros.yml` describes
  the myAGV ROS 1 wire; `robots_specs/mycobot280/ros2.yml` describes the
  myCobot 280 ROS 2 wire. The assembly uses these component interfaces unchanged.
* **Official mounting/communication documentation:**
  <https://docs.elephantrobotics.com/docs/myagv_pi23_en/7-ExamplesRobotsUsing/7.1-myagvAnd280Pi/7.1.1-SocketCommunication.html>
* **Mounting transform:** `mycobot280` link `g_base` relative to `myagv` link
  `base_footprint`: xyz = (0.0, 0.0, 0.161) m, rpy = (0, 0, 0) rad. This entry is the only
  copy. Source: an **estimate**, not a measurement. The official mounting documentation
  above (§1 "Hardware installation") mounts the arm on top of the myAGV with four M4×8
  screws, "in the front or rear", with its X axis along the AGV's, and gives no hole
  position; the arm is taken as centred with its axes aligned. The height puts the arm
  base plate's underside (`G_base.dae`, 0.030 m below `g_base`) on the myAGV's top deck
  (`myagv_up.dae`, z = 0.131 m in `base_footprint`), both from the meshes at the pinned
  revisions.

`myagv_mycobot280` is the assembly id, not a third component robot or an asset
folder. Its two component folders are the same ones used by the standalone robots.
Mount the official arm and gripper on the official base as one physical body at the
recorded mounting transform. Keep the base and arm control interfaces distinct. Each
component's own recorded stop command applies to that component's motions, on that
component's wire.

## 7. ROSMASTER X3 PLUS

* **Robot id:** `rosmaster_x3_plus`
* **Folder:** `robots_specs/rosmaster_x3_plus/`
* **Manufacturer:** Yahboom
* **Kind:** `mobile_manipulator`.
* **Official product:** <https://category.yahboom.net/products/rosmaster-x3-plus>
* **Official URDF:** `robots_specs/rosmaster_x3_plus/yahboomcar_X3plus.urdf`
* **ROS interface:** `robots_specs/rosmaster_x3_plus/ros.yml` (ROS 1).
* **Official tutorial source:** <https://github.com/YahboomTechnology/ROSMASTERX3-PLUS>,
  branch `main`, revision `9732c62247dfb57a899aea01e4fe72ed434bac6d`.
* **Official code download:**
  <https://drive.google.com/file/d/1SRg1aD_u8kyxxjm4vp0cFYxu8Ddj2sXU>,
  `ROSMASTER-X3Plus_ROS1_code.zip`; the URDF and meshes are in `yahboomcar_ws.zip`.
* **URDF SHA-256:** `17c7c8fac92774f0d0cd093fb7500704d3b5149998cb8b60c958f5352646919c`.
* **Workspace archive SHA-256:**
  `13d752e04cba3e34116912c3903bbe194e5ca5185c304036fb0ddedb59937162`.
* **Documentation:** <http://www.yahboom.net/study/ROSMASTER-X3-PLUS>

The tutorial repository is not the robot's complete ROS source; use the pinned
vendor code download to establish the interface. Preserve the official base, arm,
gripper and sensor geometry. The interface description must cover driving and
its stop, odometry, arm and gripper control, transforms and camera/lidar streams.
