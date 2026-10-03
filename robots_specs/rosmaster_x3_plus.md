# ROSMASTER X3 PLUS

* **Robot id:** `rosmaster_x3_plus`
* **Folder:** `robots_specs/rosmaster_x3_plus/`
* **Manufacturer:** Yahboom
* **Kind:** `mobile_manipulator`.
* **Embodiment:** ROSMASTER X3 PLUS: aluminium chassis with four 80 mm mecanum wheels on 520
  geared motors with Hall encoders, and a 6-DOF bus-servo arm (five joints and the gripper
  servo). Controller: Yahboom's ROS expansion board V3 (STM32F103, firmware Rosmaster V3.5.1,
  `/dev/myserial` at 115200 baud, ICM20948 or MPU9250 IMU), driven from the main computer
  (Jetson Nano 4GB, Jetson Orin NX/Nano SUPER or Raspberry Pi 5, ROS Noetic). Lidar: YDLIDAR
  4ROS (ToF), `/dev/ydlidar`. Camera: Orbbec Astra Pro Plus RGB-D camera (colour, depth and IR
  streams). The arm's `mono_link` USB camera is not started by the authoritative boot.
* **Official product:** <https://category.yahboom.net/products/rosmaster-x3-plus>
* **Official URDF:** `robots_specs/rosmaster_x3_plus/yahboomcar_X3plus.urdf`
* **MuJoCo model:** `robots_specs/rosmaster_x3_plus/model.xml`, derived (Yahboom publishes no
  MJCF); conversion steps, adaptations and estimates in
  `robots_specs/rosmaster_x3_plus/import.md`.
* **ROS interface:** `robots_specs/rosmaster_x3_plus/ros.yml` (ROS 1).
* **Official tutorial source:** <https://github.com/YahboomTechnology/ROSMASTERX3-PLUS>,
  branch `main`, revision `9732c62247dfb57a899aea01e4fe72ed434bac6d`.
* **Official code download:**
  <https://drive.google.com/file/d/1SRg1aD_u8kyxxjm4vp0cFYxu8Ddj2sXU>,
  `ROSMASTER-X3Plus_ROS1_code.zip`; the URDF and meshes are in `yahboomcar_ws.zip`.
* **URDF SHA-256:** `17c7c8fac92774f0d0cd093fb7500704d3b5149998cb8b60c958f5352646919c`.
* **Workspace archive SHA-256:**
  `13d752e04cba3e34116912c3903bbe194e5ca5185c304036fb0ddedb59937162`.
* **Authoritative boot:** with `ROBOT_TYPE=X3plus` (and `RPLIDAR_TYPE=4ROS`) exported as the
  tutorials set them, `roslaunch yahboomcar_nav laser_astrapro_bringup.launch`, which the
  tutorials start before every lidar, camera, mapping and navigation course. From
  `yahboomcar_ws.zip`: `yahboomcar_ws/src/yahboomcar_nav/launch/laser_astrapro_bringup.launch`,
  the included `yahboomcar_bringup/launch/bringup.launch` with its parameter file
  `yahboomcar_bringup/param/robot_localization.yaml`, `yahboomcar_ctrl/launch/yahboom_joy.launch`,
  and `robot_description` from `yahboomcar_description/urdf/yahboomcar_X3plus.urdf`; from the
  same download's `software.zip`: `ydlidar_ros_driver/launch/TG.launch` and
  `orbbec-ros-sdk/launch/astraproplus.launch` with `astra_frames.launch`.
* **Documentation:** <http://www.yahboom.net/study/ROSMASTER-X3-PLUS>

The tutorial repository is not the robot's complete ROS source; use the pinned
vendor code download to establish the interface. Preserve the official base, arm,
gripper and sensor geometry. The interface description must cover driving and
its stop, odometry, arm and gripper control, transforms and camera/lidar streams.
