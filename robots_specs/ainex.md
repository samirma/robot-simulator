# AiNex

* **Robot id:** `ainex`
* **Folder:** `robots_specs/ainex/`
* **Manufacturer:** Hiwonder
* **Kind:** `humanoid`.
* **Embodiment:** the AiNex humanoid, Raspberry Pi 5 version (the documented version; older
  kits carry a Pi 4B), running Ubuntu 20.04 with ROS Noetic in the Docker container "AiNex".
  Controller: Hiwonder's STM32 controller board on `/dev/rrc` (1 000 000 baud), driving 24
  serial bus servos (HX-35H / HX-35HM / HX-12H; legs 1-12, shoulders 13-16, elbows 17-20,
  grippers 21-22, head pan/tilt 23-24) and reporting the board IMU, battery, buttons and
  gamepad. Camera: one USB 2.0 head camera on the 2-DOF pan-tilt head, 640x480 (docs: FOV
  170°), `/dev/usb_cam`. No lidar.
* **Official product:** <https://www.hiwonder.com/products/ainex>
* **Official URDF:** `robots_specs/ainex/ainex.urdf` (the xacro expansion of the official
  `ainex_description/urdf/ainex.urdf.xacro`; see `import.md`).
* **MuJoCo model:** `robots_specs/ainex/model.xml`, derived (Hiwonder publishes no MJCF);
  conversion steps, adaptations and estimates in `robots_specs/ainex/import.md`.
* **ROS interface:** `robots_specs/ainex/ros.yml` (ROS 1).
* **Official source:** <https://github.com/Hiwonder/ainex>, branch `main`, revision
  `e8fe2a816797cf83054135160df5a82ec3596a69`.
* **Authoritative boot:** the systemd unit `src/ainex_bringup/service/start_app_node.service`
  (`sudo systemctl start start_app_node.service`), which runs `roslaunch ainex_bringup
  bringup.launch` at the pinned revision: `src/ainex_bringup/launch/bringup.launch` with the
  launches it includes (`usb_cam_with_calib.launch`, `base.launch`, `joystick_control.launch`,
  `rosbridge.launch`, `start.launch` and theirs; every one is listed in `ros.yml` `boot`).
  Parameter and data files it loads: `ainex_kinematics/config/init_pose.yaml`,
  `servo_controller.yaml` and `walking_param.yaml`, the gait engine `walking_module.so` and
  leg IK `kinematics.so`, `ainex_calibration/config/imu_calib.yaml` and `mag_calib.yaml`,
  `ainex_example/config/color_track_pid.yaml` and `calib.yaml`; and, on the robot image and in
  no pinned source, the action groups `/home/ubuntu/software/ainex_controller/ActionGroups/*.d6a`,
  the LAB thresholds `lab_config.yaml` and the camera calibration `head_camera.yaml`.
* **Estimated action groups:** `robots_specs/ainex/action_groups.yml`: playable groups that
  are estimates, not vendor data, since no vendor group is in a pinned source.
* **Documentation:** <https://docs.hiwonder.com/projects/AiNex/en/raspberry-pi5-version/>

Preserve the official humanoid joints and meshes. The interface description must
cover walking and its stop, head control, action-group playback and camera streams.
Record sensor calibration sources and any estimates explicitly.
