# myCobot 280

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
  this robot's authoritative boot ([robot specification](high_level_spec.md) §1).
* **Gripper:** <https://shop.elephantrobotics.com/collections/mycobot/products/adaptive-gripper>
* **Documentation:** <https://docs.elephantrobotics.com/docs/mycobot_280_pi_en/>

Preserve the official arm and gripper geometry and their referenced assets. Record
the real-hardware ROS 2 boot from that source in `ros2.yml`, including arm and gripper
commands and the feedback actually published; do not invent measured feedback.
The model published in <https://github.com/elephantrobotics/mycobot_mujoco> as
`mycobot_280jn_mujoco.xml` is for the JetsonNano variant, so it is not an official
model of this Pi embodiment. Use the Pi URDF for a derived MuJoCo model unless an
official MJCF for the exact Pi embodiment is available.
