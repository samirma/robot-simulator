# reBot Arm B601-RS

* **Robot id:** `rebot_b601_rs`
* **Folder:** `robots_specs/rebot_b601_rs/`
* **Manufacturer:** Seeed Studio
* **Kind:** `arm`.
* **Embodiment:** the reBot Arm B601-RS (RobStride motor version of reBot-DevArm), 6 DOF plus a
  rack-and-pinion parallel gripper. Seven RobStride quasi-direct-drive motors on one CAN bus: RS-06 on
  joint1-joint3 (CAN ids 0x01-0x03), RS-00 on joint4-joint6 (0x04-0x06) and on the gripper (0x07), all
  answering host id 0xFD (SDK `config/rebotarm_rs.yaml`). The gripper motor turns a module-1, 16-tooth
  pinion between two racks driving both fingers on an MGN9 rail. Supply: DC 48 V (Seeed recommends a
  MeanWell LRS-600-48, 48 V 12.5 A) through an XT30 2+2 power separation board. Controller: no on-arm
  computer; the host PC (Ubuntu 24.04, ROS 2 Jazzy, Python 3.12) drives the bus through the kit's
  USB-to-CAN adapter (PCAN-USB firmware, Linux `peak_usb` SocketCAN) as `can0` at 1 Mbit/s, with
  `motorbridge` 0.5.6 as the CAN driver. Camera: none; the authoritative boot starts no camera driver
  (Seeed only publishes optional wrist-camera mounts). No lidar, no IMU.
* **Official product:** <https://www.seeedstudio.com/reBot-Arm-B601-RS-Bundle-p-6898.html>
* **Official URDF:** `robots_specs/rebot_b601_rs/ReBot_Arm_RS.urdf`, the authoritative boot's copy
  (`src/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf`), kept unmodified; its
  `package://rebotarm_bringup/` mesh URIs resolve inside the robot folder.
* **URDF SHA-256:** `978568c1b73310ec134b8f2285295f46d0c32c904d397bd097e3bcc81507b77b`.
* **MuJoCo model:** `robots_specs/rebot_b601_rs/model.xml`, derived from the official URDF (Seeed publishes no
  MJCF for the B601-RS); conversion steps, adaptations and estimates in
  `robots_specs/rebot_b601_rs/import.md`.
* **ROS interface:** `robots_specs/rebot_b601_rs/ros2.yml` (ROS 2).
* **Official project source:** <https://github.com/Seeed-Projects/reBot-DevArm>, branch `main`, revision
  `0dfb49b85bc146eb3741f414be9dd2b1662345e8` (hardware BOM `hardware/reBot_B601_RS/`, URDF and meshes
  `Rebot_Arm_description/RS/`).
* **Official ROS 2 source:** <https://github.com/Seeed-Projects/reBotArmController_ROS2>, branch `main`,
  revision `7506dc0fc5c2a7787dcb974d89d13f9f49d16d61`.
* **Official SDK source:** <https://github.com/Seeed-Projects/reBotArm_control_py>, branch `main`, revision
  `6415d43130d1e143c70dc106096a857ac5556f81`, cloned by the wiki into `third_party/reBotArm_control_py`.
* **Motor driver:** <https://github.com/motorbridge/motorbridge>, release `v0.5.6` (PyPI `motorbridge`
  0.5.6, the newest release when the ROS 2 source was pinned), revision
  `c652ce420e7da1008fa1864cc7f47d7d7c57fdb6`.
* **Authoritative boot:** with SocketCAN `can0` up at 1 Mbit/s, `ros2 launch rebotarm_bringup
  bringup.launch.py model:=rs channel:=can0` at the pinned ROS 2 revision:
  `src/rebotarm_bringup/launch/bringup.launch.py` starts `reBotArmController` and
  `robot_state_publisher` (RViz only with `use_rviz:=true`), builds `robot_description` by running
  `src/rebotarm_bringup/launch/rviz_urdf_compat.py` on `src/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf`,
  and the controller loads the vendor data files `src/rebotarm_bringup/config/rebotarm_hardware.yaml` (its
  `rs` overrides) and, from the pinned SDK revision, `config/rebotarm_rs.yaml` (`sdk_config`); the SDK's
  Pinocchio model reads the same `ReBot_Arm_RS.urdf`. It loads no other vendor data files
  (`config/driver_params.yaml` is installed but unused).
* **Documentation:** <https://wiki.seeedstudio.com/rebot_arm_b601_rs_ros2_integration/> (source
  `Seeed-Studio/wiki-documents`, branch `docusaurus-version`, revision
  `2d9c3d1225f1ab98b79e1382b72ca9e2148bb23a`, retrieved on 2026-10-08); getting started:
  <https://wiki.seeedstudio.com/rebot_b601_rs_getting_started/>

Seeed publishes and documents this ROS 2 interface itself, so `ros2.yml` describes a manufacturer
interface: the pinned `rebotarm_bringup` real-hardware boot with the `rs` model, its controller node
(arm trajectory and Cartesian goals, gripper commands, per-motor raw commands, state and status
topics), `robot_state_publisher` and the transforms it publishes. The boot starts no camera, lidar or
IMU. Arm joints are commanded and reported in the motors' own frame, which is the URDF joint frame; the
gripper is commanded as one motor angle and reported on the URDF's two finger joints. The wire's names,
frames, joint conventions and `robot_description` follow the pinned boot; `import.md` records how they
map onto the model derived from the official URDF.
