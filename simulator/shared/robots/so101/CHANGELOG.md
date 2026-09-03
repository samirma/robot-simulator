# Changelog – SO-ARM100 Description

All notable changes to this model will be documented in this file.

## [2025-12-18]

- Initial release.

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
