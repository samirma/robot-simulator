# myAGV

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
