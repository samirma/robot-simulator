# Changelog – SO-ARM100 Description

All notable changes to this model will be documented in this file.

## [2025-12-18]

- Initial release.

## [2026-09-27]

**`model.xml` is now derived from the official `so101_new_calib.xml`, not from
mujoco_menagerie's `robotstudio_so101`.** `molmospaces/robots/so101/make_model.py` reads
`robots_specs/so101/so101_new_calib.xml` and keeps its bodies, inertials, joints, visual
geoms, sites, defaults and actuators unchanged; it adds menagerie's collision set, the
`tcp` site and `wrist_cam`, and `make_model.py --check` asserts that nothing else moved.
`so101.xml` (menagerie's file), the camera-mount mesh and menagerie's unused
`moving_jaw_so101_gripper_v1.stl` are gone; the collision geoms are transcribed into
`make_model.py` with their provenance. What changed against the menagerie model:

- **`gripperframe` is the official site**, at `-0.0079 -0.000218121 -0.0981274` (the
  fixed-jaw tip). Menagerie had moved it +19.9 mm along the gripper body's x. The
  console's `kinematics.py` follows it, with `JAW_CENTER_OFFSET` and the preflight's
  reach point grown by the same 19.9 mm so the world points it aims at are unchanged.
- **Actuators are the official ones**: `forcerange="-3.35 3.35"` on all six (menagerie
  used the `sts3215` class's +/-2.94).
- **`wrist_roll` is the official -2.7438..+2.8412 rad** (menagerie: symmetric
  +/-2.7438); the other ranges agree to rounding.
- **No `camera_mount` body**: `wrist_cam` sits directly in `gripper`, at the same pose.
  Its 12 g mount mesh -- which, having no `<inertial>`, added mass to the model -- and
  its two collision boxes are gone. The camera is now **640x480** (4:3, fovy 61.9
  degrees), not menagerie's 1920x1080 16:9 sensor.
- **No `<option>`, `<visual>` or mesh `maxhullvert` default**: the official file has
  none. Menagerie's `implicitfast`, 5 ms, elliptic-cone, `impratio=10` options never
  reached a simulation (both engines graft the robot's bodies into a scene that brings
  its own options); they only affect compiling `model.xml` on its own.
- The official default class is `so101_new_calib` (was `so101`) and the model name is
  `so101_new_calib`.

The `tcp` site is where it was (`0.0113261 -0.0001 -0.0978094`): it is measured from
the jaw-tip geoms, which did not move.

## [2026-09-06]

**The model is now mujoco_menagerie's `robotstudio_so101` and nothing else** --
rendering, joints, actuators and camera positions alike. `model.xml` is still generated
by `molmospaces/robots/so101/make_model.py`, but its whole delta from `so101.xml` is one
group-3 `<site name="tcp">`: no visual, no collision, no dynamics. It stays because
`SO101RobotView` resolves it by name, and removing it would break the MolmoSpaces engine
at load without changing anything the simulator renders or steps.

Removed, all three of them measured and all three now recorded in `make_model.py` rather
than in the model:

- **`exo_camera`.** Not upstream. Both engines' `_pick_camera` falls through to
  `wrist_cam`; RoboCasa already preferred its own scene camera to it.
- **Gripper `forcerange="-0.30 0.30"`.** The gripper actuator is back on the `sts3215`
  class default of +/-2.94 N.m, which is the servo's real figure and far too strong for
  a 20 g apple: at the equivalent +/-3.35 the rig measured peak jaw force 96 N and the
  apple slipping 397.9 mm out of the fingers. **Expect the apple-on-plate grasp to
  regress**; that is what the official actuator does, not a fault.
- **The `wrist` camera.** The ROS wrist topic now renders upstream's `wrist_cam`, 119 mm
  and 53.7 degrees from where the removed camera sat, at 48.5 deg fovy against 75 and
  16:9 against 1:1. The removed camera was placed by ray-casting a 4 x 7 x 3 grid to
  find a line to the grasp point that the printed wrist_roll_follower housing does not
  block; the official pose is one of the poses that search rejected, so the housing is
  in frame.

## [2026-09-03]

`model.xml` is generated; both changes are in
`molmospaces/robots/so101/make_model.py`, which is where the delta from
mujoco_menagerie's `robotstudio_so101` belongs. **This file is the spec both engines
load, so an edit here changes both.**

- **Gripper `forcerange` -0.30 0.30**, overriding the `sts3215` class default of
  +/-2.94 N.m. Right for the servo, far too strong for grasping something light:
  measured on the equivalent rig at +/-3.35, peak jaw force 96 N, mean 56 N, the object
  slipping 397.9 mm and ending extruded onto the table; at +/-0.30, peak 12.4 N, mean
  4.8 N, slip 19.6 mm, held at z 0.1003. Below about +/-1.0 the commanded jaw width
  stops mattering because the actuator saturates on force before reaching it -- which is
  the regime a compliant grasp wants. The five arm actuators are untouched.
- **A second wrist camera, `wrist`** (fovy 75, 256x256), added *beside* upstream's
  `wrist_cam` rather than replacing it. `wrist_cam` is menagerie's photographic camera
  (1920x1080, specified by sensorsize/focal) and stays as the one to look through;
  `wrist` is the geometry a policy was calibrated against. Its position is 45 mm
  off-axis because the printed wrist_roll_follower housing occludes every on-axis mount,
  and its `resolution` is mandatory -- a camera without one renders zero-sized images
  and reports nothing.
