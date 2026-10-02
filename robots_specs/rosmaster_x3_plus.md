# ROSMASTER X3 PLUS

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
