# myAGV

* **Robot id:** `myagv`
* **Folder:** `robots_specs/myagv/`
* **Manufacturer:** Elephant Robotics
* **Kind:** `mobile_base`.
* **Embodiment:** myAGV 2023 Pi: Raspberry Pi 4B (4 GB) running Elephant Robotics' Ubuntu
  20.04 image with ROS Noetic. Controller: the base MCU (motor drivers, IMU, battery monitor;
  firmware not published) on `/dev/ttyAMA2` at 115200 baud, driving four mecanum wheels on
  planetary brushless DC motors. Lidar: YDLIDAR X2L on `/dev/ttyAMA0`, powered through GPIO20.
  Camera: the built-in CSI camera, 5 MP, 65° viewing angle, `/dev/video0`.
* **Official product:** <https://www.elephantrobotics.com/en/myagv-2023-pi-en/>
* **Official URDF:** `robots_specs/myagv/myAGV.urdf`
* **MuJoCo model:** `robots_specs/myagv/model.xml`, derived (Elephant Robotics publishes no
  MJCF); conversion steps, adaptations and estimates in `robots_specs/myagv/import.md`.
* **ROS interface:** `robots_specs/myagv/ros.yml` (ROS 1).
* **Official source:** <https://github.com/elephantrobotics/myagv_ros>, branch
  `myagv_ros_2023Pi`, revision `c71f3cc574e5ed1973a925238eabe88662cfa701`.
* **Camera boot:** <https://github.com/ros-drivers/usb_cam>, revision
  `addab4a65fdf65c460fec2cc3eee8fab94699370`.
* **Authoritative boot:** the documented bring-up (docs 6.2.4 "Basic Control Based on ROS"):
  `myagv_odometry/scripts/start_ydlidar.sh` (lidar power), then `roslaunch myagv_odometry
  myagv_active.launch` at the pinned `myagv_ros` revision
  (`myagv_odometry/launch/myagv_active.launch`, which includes
  `ydlidar_ros_driver/launch/X2.launch` and loads `myagv_urdf/urdf/myAGV.urdf` as
  `robot_description`); beside it the camera boot `rosrun usb_cam usb_cam_node
  _camera_frame_id:=camera_link` (usb_cam 0.3.7 at the pinned revision, every other setting at
  the driver's defaults).
* **Documentation:** <https://docs.elephantrobotics.com/docs/myagv_pi23_en/>

The camera boot is an approved community camera boot, not manufacturer-provided, and the
camera streams it publishes are required. Preserve the official base geometry, wheel joints and sensor mounting. The interface
description must cover driving and its stop, odometry, transforms, lidar and camera
streams. Mapping and navigation endpoints come from separate launches and are recorded
as `optional`.
