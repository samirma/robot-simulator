# Out-of-tree robots

The robots this engine spawns are the ones `robots_specs/robots.yml` marks `simulated`;
their URDF, MJCF and meshes are read from `robots_specs/<id>/` through
`shared/robots_spec.py`. The adapters here live entirely outside the upstream clone:
`BaseRobotConfig.robot_dir` accepts an external directory, so nothing in `molmospaces/`
needs patching.

Reference: `molmospaces/docs/tutorials/add_robot.md`, worked example in
`molmospaces/examples/add_robot/` (xarm7).

| Robot | Status |
|---|---|
| `so101` | spawn, view, joint control (local and over the bridge) |
| `myagv` | spawn, view, holonomic drive, camera stream, keyboard teleop |
| `ainex` | spawn, view, animated-gait locomotion, two arms + claws, head, action groups |

## What each robot needs

1. **An MJCF** under `robots/<name>/`, adjusted to MolmoSpaces conventions:
   base frame at the origin, and a TCP site whose **+z is the approach direction**
   with the fingers opening along **y** (README → "Robot Conventions"; the tutorial
   prose says "+x away from the robot", which does not match the shipped Franka —
   trust the Franka's `gripper/grasp_site`).
2. **Move groups** — `MocapRobotBaseGroup` for the mount, `MJCFFrameMixin` +
   `SimplyActuatedMoveGroup` for the arm, `GripperGroup` for the gripper.
3. **A `RobotView`** collecting those groups.
4. **A `Robot`** wiring a controller per group, plus **a `BaseRobotConfig`** pointing
   `robot_dir` at the robot's directory.

Mark the robot `simulated` in `robots_specs/robots.yml` and register its adapter in
`tools/spawn_robot.py::ADAPTERS` to make it available to `run.sh view --robot <id>`.

## so101

TheRobotStudio / LeRobot SO-101: a 5-DoF tabletop arm with a single hinged jaw.
`shared/robots/so101/model.xml` is **generated** by `make_model.py` from the official
`robots_specs/so101/so101_new_calib.xml`, whose meshes it loads from
`robots_specs/so101/assets/`. Every body, inertial, joint, visual geom, site, default and
actuator is the official file's; the generator adds only what simulation needs:

* **collision**: the official convex hulls (a hull of a jaw fills the gap between the
  fingers) are replaced by mujoco_menagerie's `robotstudio_so101` primitives and gripper
  collision meshes (Apache 2.0; the meshes are in `shared/robots/so101/assets/`), in
  groups 3 and 4, invisible;
* a `tcp` site in MolmoSpaces convention, which `SO101RobotView` resolves by name:
  positioned at the true grasp centre (the midpoint between the jaw tips, ~19 mm from
  the official `gripperframe`, which sits on the fixed-jaw tip) and oriented with +z
  along the approach. Both axes are *measured from the model* at generation time;
* the wrist camera `wrist_cam`, directly in the official `gripper` body at menagerie's
  pose, as a 640x480 capture (usb_cam's default mode, square pixels).

`python robots/so101/make_model.py --check` (also run after every generation) compiles
the official file and `model.xml` and asserts the kinematics, inertials, limits,
actuators and visual geoms are identical, and compares forward kinematics against the
official URDF. An `exo_camera`, a second wrist camera and a softened gripper
`forcerange` lived here until 2026-09-06; see `shared/robots/so101/CHANGELOG.md`.

Re-run `python robots/so101/make_model.py` after `robots_specs/so101/` changes.

### Verified

`python robots/so101/test_attach.py` (add `--scene <house.xml>` to run it in a house):

* attaches to an empty world and to an iTHOR house
* move groups resolve; gripper spans 0.007 m closed to 0.127 m open, `is_open` agrees
* arm tracks joint-position commands to <0.001 rad
* holds its rest pose with 0.0004 rad drift over 3000 steps

### Known limitations (so101)

* **5 DoF, so arbitrary 6-DoF Cartesian poses are unreachable.** The arm Jacobian has
  rank 5; IK is under-determined in orientation. Joint-space control is exact.
* **No grasp library** — per-gripper grasp sets exist upstream only for `droid`
  (Franka) and `rum`. Use `run.sh view --robot so101` plus the external control
  server instead.
* **The gripper is a single hinged jaw**, not a parallel-jaw. `inter_finger_dist` is
  measured between the jaw tips and varies non-linearly with the joint angle; the
  finger axis is only exactly perpendicular to the approach at one opening
  (make_model.py measures it at ~4 cm, where it is 7.4° off).
* `parallel_kinematics` raises `NotImplementedError` — it is a CUDA-only batched IK
  path and this install is CPU-only.
* The pedestal (`base_size`) is a plain box; placement by `tools/spawn_robot.py`
  maximises floor clearance and does not reason about reachability of any target.

## myagv

Elephant Robotics [myAGV Pi 2023](https://shop.elephantrobotics.com/collections/myagv-smart-navigation-robot/products/myagv-pi):
a 311 × 230 × 110 mm, 4.16 kg Mecanum-wheeled mobile base. No arm.

```bash
./run.sh view --robot myagv                          # spawn it in a house
python robots/myagv/test_attach.py                   # self-test (empty world)
python robots/myagv/test_attach.py --scene <house>   # self-test (in a house)
```

Drive it by keyboard with a live camera feed from [../../robot_console](../../robot_console).

### The model is authored, not converted

The only official model
([`elephantrobotics/myagv_ros`](https://github.com/elephantrobotics/myagv_ros), branch
`myagv_ros_2023Pi`) is visualisation-only: 47 lines, two links joined by a dummy
`continuous` joint, three COLLADA meshes, and **no wheels, collision geometry, inertia
or usable scale**. `make_model.py` therefore generates `shared/robots/myagv/model.xml`,
reusing the meshes in `robots_specs/myagv/urdf/` only for appearance:

* **DAE → OBJ**, since MuJoCo does not read COLLADA. The meshes carry no real scale
  (~12.3 × 9.0 × 5.2 units), so the scale is derived by matching the published
  footprint — x and y independently imply 0.02522 and 0.02551, and the mean is used
  rather than distorting the model with a non-uniform fit. Result: 0.313 × 0.229 m
  against a spec of 0.311 × 0.230.
* **A box collision hull** and solid-box inertia, since upstream supplies neither.
  It is lifted 5 mm off the floor: the base has no vertical DoF so it cannot fall, and
  this stops it scraping the floor and fighting the actuators while still colliding
  with walls and furniture.
* **A forward-facing `front_camera`** on the top deck.

Re-run `python robots/myagv/make_model.py` after changing anything upstream.

### The drive is holonomic, not simulated Mecanum

The four wheels are **decorative**. Motion comes from three virtual joints — slide-X,
slide-Y, hinge-Z — driven by position actuators, reusing MolmoSpaces'
`HoloJointsRobotBaseGroup` (the same class RB-Y1 uses; `rby1m` ships exactly this
arrangement for its own Mecanum base). A holonomic planar base reproduces precisely the
motion envelope a Mecanum drive has, without the fragility of four-roller contact.

Consequences worth knowing:

* **The robot must be attached at the origin with identity rotation**, because the
  slide joints are world-aligned. Placement is done by writing the joints
  (`robot_view.base.pose = ...`), which is what `tools/spawn_robot.py` does. Attaching
  it anywhere else raises rather than silently producing a base whose "forward" is wrong.
* No wheel slip and no traction limits.
* Gains use `dampratio` rather than an explicit `kv`. The yaw inertia is only
  ~0.05 kg·m², and a hand-set `kv` large enough to look critically damped violates the
  explicit-integration stability bound (`kv·dt/I < 2`) and makes the simulation diverge.

### The laser is ray-cast, not a sensor in the MJCF

The real 2023 Pi AGV carries a **YDLidar X2** publishing `/scan` and `/point_cloud`; the
model has no `<sensor>` element at all. Rather than regenerate `model.xml` with a ring of
rangefinders, `shared/mujoco_bridge.py::laser_scan_ranges` casts `mujoco.mj_ray` in a fan,
and the myAGV's surface (`shared/ros_surfaces/myagv.py::scan_ranges`) publishes it as
`sensor_msgs/LaserScan`.

Every parameter is the robot's interface, not a flag: `ros_surfaces/myagv.py` transcribes
them from `robots_specs/myagv/ros.yml` -- `ydlidar_ros_driver/launch/X2.launch` and the
`base2laser_link` static transform in `myagv_odometry/launch/myagv_active.launch`, on the
[`myagv_ros_2023Pi`](https://github.com/elephantrobotics/myagv_ros/tree/myagv_ros_2023Pi)
branch -- and no launcher flag changes one:

| | value | where |
|---|---|---|
| `frame_id` | `laser_frame` | `FRAME_LASER` |
| `range_min` / `range_max` | 0.1 / 12.0 m | `SCAN_RANGE_MIN` / `SCAN_RANGE_MAX` |
| rate | the ROS file's, for `/scan` and `/point_cloud` | `rate_of(TOPIC_SCAN, NODE_LIDAR)` |
| mount, off `base_footprint` | x +0.065 m, z +0.08 m, yaw pi | `STATIC_TRANSFORMS[base2laser_link]` |
| points per sweep | `sample_rate` (3 kHz) over the scan rate | `SCAN_BEAMS` |
| blind wedge | `ignore_array` -50..50 deg, reported as 0.0 | `SCAN_IGNORE_DEG` |
| a miss | 0.0 (`invalid_range_is_inf: false`) | `SCAN_INVALID` |

Details that are load-bearing:

* the rays exclude every body of the robot's own, or every beam returns its chassis at
  11 cm -- only its own: a neighbouring robot is something to see;
* the fan is cast from the **laser** origin, not the base origin. 65 mm is a whole cell
  at the 5 cm resolution `myagv_navigation`'s gmapping uses, and casting from the base
  centre instead is invisible in a viewer and ruins a map;
* the scan runs **counter-clockwise from `-pi`** in `laser_frame`, which the launch turns a
  half-turn about z from `base_footprint`: the beam at 180 deg points along the base's +x.
  The X2 is launched `inverted: true` because it is mounted upside down; do not change the
  mount without the scan direction.

The myAGV has no depth camera. Its one camera is `usb_cam` at 640x480 on `/camera/*`.

`../../robot_console` uses these to map a house autonomously — see its README.

### Velocity means velocity

`serve_ros` integrates the commanded `cmd_vel` into a position setpoint that is carried
forward between steps and clamped to a short lead ahead of the measured pose. It used to
re-derive the setpoint from the measured pose each step, which left it one 14 mm increment
ahead of a robot that was chasing it: the base settled at roughly a **sixth** of the
commanded speed. The clamp keeps the property that made the old version tempting — a robot
held up by a wall stops advancing its target instead of winding up a lunge.

### Gotcha when testing in a house

The house origin is usually *inside* furniture — in FloorPlan1 it is inside the kitchen
island — and a robot embedded in geometry cannot move, which looks exactly like a broken
actuator. `test_attach.py --scene` places itself on open floor first and drives toward
the middle of the room for the same reason: "drove into a wall and stopped" is correct
behaviour that would otherwise read as a failure.

## ainex

[Hiwonder AiNex](https://www.hiwonder.com/products/ainex): a 24-DoF biped humanoid,
193 × 135 × 415 mm, 2.45 kg, walking at 21 cm/s on HX-series serial bus servos. Two 5-DoF
arms ending in a single hinged claw, a 2-DoF pan/tilt head carrying the only camera, a
9-axis IMU, and **no lidar, no depth sensor and no wheels**.

```bash
./run.sh view --robot ainex                          # spawn it in a house
../kitchen.sh serve --robots ainex                   # ...on its own vendor ROS topics
python robots/ainex/test_attach.py                   # self-test (empty world)
python robots/ainex/test_attach.py --scene <house>   # self-test (in a house)
python robots/ainex/test_ros.py                      # self-test of the ROS surface
```

It is the first robot here that walks rather than rolls, and the first whose ROS contract
has nothing in common with the myAGV's. Both facts drive everything below. Provenance of
the vendor files (`robots_specs/ainex/`) is in
[shared/robots/ainex/PROVENANCE.md](../../shared/robots/ainex/PROVENANCE.md).

### It does not actually walk

The real robot's gait engine is `walking_module.so` — a precompiled ARM binary with no
source — so there is nothing to port, and authoring a balance controller for a 2.35 kg
biped is a research project rather than an integration. So the torso rides **the same
virtual slide-X / slide-Y / hinge-Z base `myagv` uses for its Mecanum drive**, and the
twelve leg joints are animated over the top at a phase matched to the distance covered.

That trade buys a robot that navigates reliably and never falls. It costs balance
entirely: there is no ZMP, no push recovery, and `WalkingParam`'s balance gains are
accepted on the wire and ignored because nothing exists for them to act on.

The animation is still made to be honest where it is cheap to be. `gait.py` solves a
2-link sagittal IK to a commanded foot position rather than writing per-joint sinusoids,
which buys two properties across the *whole* envelope rather than at one tuned point:

* the stance foot stays at a constant height under the hip, so ride height is a constant;
* **the stance foot does not skate.** Stance is deliberately linear in phase, because the
  base advances at a constant velocity and only a foot moving at constant speed cancels
  against it exactly. A cosine stance matches over the half-cycle but lags mid-stance,
  which measured ~8 mm of skate at the top of the envelope; `test_attach.py` checks this.

### Where the walking speed comes from

`WalkingParam` is field-for-field the ROBOTIS preview-control module's, in which
`period_time` T is a full cycle and `x_move_amplitude` A is a foot's body-frame half-sweep.
A foot in stance is fixed to the ground, so the body advances 2A per stance and 4A per
cycle:

    v = 4A / T

At the vendor's default `period_time: 400` ms and the envelope maximum A = 0.02 m that is
**0.200 m/s** against Hiwonder's published **0.21 m/s** — 4.8% low. The other two readings
of the amplitude give 0.100 and 0.050 m/s. That agreement is the entire argument for the
factor of four, and it is why the constant is written as a derivation rather than a number.

The lateral and yaw factors reuse the same geometry but are **not** calibrated: Hiwonder
publish a walking speed and no sidestep or turn rate.

Worth knowing: the published figure is not reachable through `/app/set_walking_param`. That
interface uses the requested x/y/angle only for their **sign** and replaces the magnitude
with a per-tier constant, so its fastest setting is 0.16 m/s. Full-envelope amplitudes
arrive only over `/walking/set_param`. That is the vendor's behaviour, not a simplification
— and note its `speed` field is 1-based with **4 as the fastest**, the opposite of the
ordering `gait_manager.move()` uses for its own preset list.

### Grasping is replayed trajectories, not IK

The real AiNex has no inverse-kinematics service and no Cartesian arm interface at all.
Every manipulation it performs is a recorded servo trajectory replayed open-loop by
`MotionManager.run_action`, triggered over ROS by `/app/set_action`. This follows that
model: `shared/ros_surfaces/ainex/action_groups/` holds a small set of keyframed
poses, and `actions.py` beside it also
reads Hiwonder's own `.d6a` (SQLite) format so `--action-dir` can point straight at a real
robot's `ActionGroups` directory.

**The hands reach between roughly 0.25 m and 0.43 m above the floor, and never lower.**
The arms are short relative to the robot's 0.46 m height, and because the torso is bolted
to planar joints it **cannot pitch** — so unlike the real robot it cannot bend forward over
its feet. It grasps from a surface at its own chest height. This is the sharpest
consequence of the planar base; `test_attach.py` asserts the band so it cannot regress
silently.

### One correction to the vendor description

The URDF fixes `camera_link` to `body_link`. That is wrong about the hardware — Hiwonder's
own README calls it a "2-DOF HD camera" and it sits on the pan/tilt head — and harmless in
RViz, where nobody looks through it. Here the camera *is* the sensor, so leaving it on the
torso would make `/head_pan_controller/command` and every look-at behaviour untestable:
panning the head would not move the view. It is reparented to `head_tilt_link` at a pose
measured from the vendor's own numbers, so the neutral view is unchanged and only its
behaviour under head motion differs.

Note that `head_pan`'s axis is `0 0 -1`, as every joint's is, so a positive command yaws
the head **clockwise**. That is the robot's convention; "fixing" it would mean disagreeing
with the hardware.

### Faithful, and not

**Faithful:** the interface is exactly `robots_specs/ainex/ros.yml` -- every topic,
service and parameter it lists, with its type, the node that provides it and, for a
periodic topic, its rate, and nothing else (`shared/ros_surfaces/ainex/topics.py`, held to
the file by `shared/contracts/test_ainex_contract.py`). So there is no `/cmd_vel`, `/odom`,
`/scan`, `/joint_states`, `/tf` or `robot_description`, and of the 24
`/<joint>_controller/command` topics only the head pair, which are the only two the real
`ainex_controller` subscribes to (the rest are `gazebo_sim` only). Also faithful: the
24-joint topology and the servo-id map; the count<->radian mapping, including the per-joint
`init` offset and the two deliberately inverted `sho_pitch` servos; `/walking/command`'s
control gate, its blocking `stop` and the absence of any watchdog; `/app/set_action` and
`/walking/init_pose` ending in the init pose; grasping as action-group replay with no IK;
the gait envelope and the 0.20 m/s it yields at the default gait. Servo angles are read the
vendor's way, through `bus_servo/get_position`.

**Deliberate departures**, each of them load-bearing somewhere:

* Locomotion is a planar base plus a cosmetic gait: no balance, and the torso cannot
  roll. It *can* pitch -- `base_pitch`, a joint of its own that action groups author,
  because the base rides the torso and the real robot's forward lean when it crawls has
  nowhere else to come from -- and it *can* fall: `ground.py` solves the torso's height
  every tick so the stance sole sits on the surface a ray-cast finds beneath it, and
  integrates a fall when it finds none. The fall is the setpoint's, not gravity's.
* `gravcomp` is on, so it does not sag the way a 2.35 kg robot on hobby servos really does.
  Without it every limb pose, including the replayed grasps, would land somewhere other
  than commanded.
* The feet do not collide with the floor. Colliding feet grip at default friction and fight
  the world-aligned position servos, which shows up as an undershooting base picking up
  uncommanded yaw. Only one torso hull and the two hands collide with the world; the
  ground-follow above is what stands the robot on a surface instead.
* The IMU pipeline reports the torso's real yaw, lean and their rates, with roll
  identically zero because the base has no roll degree of freedom, and a fixed simulated
  Earth field on the magnetometer. Each periodic topic runs at its declared rate on its own
  clock; the values behind it, and the camera frames, refresh at the engine's control rate.
* The board's peripherals have nothing to drive: `set_led`, `set_buzzer`, `set_oled`,
  `set_rgb`, the motor and PWM-servo topics and `enable_reception` are accepted and do
  nothing, and `joy`, `sbus`, `button` and `battery` are declared but never published --
  a simulated board has no receiver, radio, button press or battery reading to report.
* The app's modes set up exactly as `app_node.py` does (init pose, head, control gate,
  detector enter/start), but their autonomous behaviours are not simulated, and the vision
  nodes publish their result frames with no detections.
* The shipped action groups are ours, not Hiwonder's — see the licence note below.

### Two URDF-import behaviours specific to this robot

* **MuJoCo merges *two* links into the worldbody, not one.** `base_link` is jointless and
  `body_link` hangs off it by a fixed joint, so compiling the vendor file untouched yields
  **five disconnected root bodies** and drops 0.743 kg of the 2.3475 out of the tree.
  Adding the virtual planar joints to `body_link` is what makes it a body at all.
* **`discardvisual` defaults to true for URDF**, and step 3 of the surgery turns almost the
  whole robot non-colliding on purpose. Leave the flag alone and the AiNex compiles down to
  two hand meshes and a hull — a robot that drives correctly and renders as nothing.
  A robot loaded from an MJCF needs no such guard: the flag only defaults on for URDF.

### Licence

**`Hiwonder/ainex` carries no LICENSE file** despite being published as open source, and
that covers the URDF, the 25 meshes and `servo_controller.yaml` — not just the action
groups. This is the only robot here whose vendor files are not under an identified licence;
[URDF.md](URDF.md) records it explicitly. No Hiwonder action groups are redistributed here
for the same reason: the format is read so that an owner supplies their own.
