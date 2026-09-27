# The Robot Studio SO101 Description (MJCF)

> [!IMPORTANT]
> Requires MuJoCo 3.1.3 or later.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for a full history of changes.

## Overview

The simulator's SO-101 is [The Robot Studio SO101 robot](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)'s
official MJCF, `so101_new_calib.xml` (SO-ARM100 revision
aec17bbc256d1a7342d53aaa4950595d4c30b40d), plus what simulation needs and the official
file does not have. None of the additions moves a frame, a joint or a limit.

## Where the files are

The official URDF, MJCF and meshes are in `robots_specs/so101/` at the workspace root,
as `robots_specs/robots.yml` records them. This folder holds only what the simulator
adds to them:

* `model.xml` — **generated** from `robots_specs/so101/so101_new_calib.xml` by
  `molmospaces/robots/so101/make_model.py`; do not edit it by hand. It loads the
  official meshes from `robots_specs/so101/assets/`;
* `assets/` — the three gripper collision meshes (from mujoco_menagerie's
  `robotstudio_so101`); the official meshes are not copied here;
* `LICENSE` — Apache 2.0, the licence of the menagerie collision set.

## MJCF derivation steps

1. Parse `so101_new_calib.xml`. Every body, inertial, joint, visual geom, site
   (`baseframe`, `gripperframe`), default class and actuator is kept as it is, and so
   is `<option>` (none: MuJoCo's defaults; the engines graft the robot into their own
   scene, whose solver settings step it).
2. Replace the official collision geoms, which are convex hulls of the visual meshes
   (a jaw's hull fills the gap between the fingers), with mujoco_menagerie's
   `robotstudio_so101` collision set: its primitive boxes (group 3), its named
   `collision_gripper` jaw geoms (`fixed_jaw_*`, `moving_jaw_*`, group 3) and its
   three gripper-part meshes (`collision_gripper_mesh`, group 4), with those two
   contact classes. Menagerie's `camera_mount` body, its mesh and its two collision
   boxes are not official geometry and are not added.
3. Add a group-3 site `tcp` in the `gripper` body at the grasp centre (the jaw-tip
   midpoint), +z along the approach and +y along the finger axis, measured from the
   jaw-tip geoms: MolmoSpaces' `SO101RobotView` resolves it by name.
4. Add the camera `wrist_cam` directly in the `gripper` body at menagerie's pose
   (`pos="0 0.055 -0.045" euler="-0.57 0 0"`; menagerie's `camera_mount` body sits at
   the gripper origin), as a 640x480 capture: `sensorsize="0.00576 0.00432"`,
   `focal="0.0036 0.0036"` -- menagerie's sensor width and focal length with a 4:3
   height, so fx = fy = 400 px and fovy is 61.9 degrees. 640x480 is usb_cam's default
   mode, which is what the ROS wrist topic publishes.
5. Verify: `python molmospaces/robots/so101/make_model.py --check` (run after every
   generation) compiles both files and asserts identical bodies, inertials, joints,
   official sites, actuators and visual geoms, no added body and no visible addition,
   and compares forward kinematics with the official URDF at random configurations
   (agreement to ~2e-6 m / 2e-5 rad; the URDF's rpy is rounded to 1.5708).

## License

This model is released under the [Apache License 2.0](LICENSE).
