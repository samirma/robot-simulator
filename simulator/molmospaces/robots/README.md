# Out-of-tree robots

MolmoSpaces ships only Franka, RB-Y1, YAM and a floating gripper. Robots added here
live entirely outside the upstream clone: `BaseRobotConfig.robot_dir` accepts an
external directory, so nothing in `molmospaces/` needs patching.

Reference: `molmospaces/docs/tutorials/add_robot.md`, worked example in
`molmospaces/examples/add_robot/` (xarm7).

| Robot | Status |
|---|---|
| `so101` | spawn, view, joint control (local and over the bridge) |
| `myagv` | spawn, view, holonomic drive, camera stream, keyboard teleop |
| `rebot_b601` | spawn, view, arm + gripper (no self-collision) |
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

Register the robot in `tools/spawn_robot.py::ROBOTS` to make it available to
`run.sh view --robot <name>`.

## so101

TheRobotStudio / LeRobot SO-101: a 5-DoF tabletop arm with a single hinged jaw.
Model from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
`robotstudio_so101`, kept verbatim as `so101.xml` (Apache 2.0, see `so101/LICENSE`).

`model.xml` is **generated** from it by `make_model.py`, which adds exactly one thing:

* a `tcp` site in MolmoSpaces convention, positioned at the true grasp centre (the
  midpoint between the jaw tips, ~12 mm from the stock `gripperframe`) and oriented
  with +z along the approach. Both axes are *measured from the model* at generation
  time rather than hardcoded, so the file stays correct if the jaw geometry changes.

It is in group 3 -- no visual, no collision, no dynamics -- and it stays only because
`SO101RobotView` resolves it by name. **Everything the simulator renders or steps is
upstream's**: geoms, joints, actuators and the one camera, `wrist_cam`, which is what
the ROS wrist topic streams. An `exo_camera`, a second wrist camera and a softened
gripper `forcerange` lived here until 2026-09-06; `make_model.py` records what each was
and what removing it costs.

Re-run `python robots/so101/make_model.py` after changing `so101.xml`.

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
or usable scale**. `make_model.py` therefore generates `model.xml`, reusing the meshes
only for appearance:

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

The real 2023 Pi AGV carries a **YDLidar X2** publishing `/scan`; the model has no
`<sensor>` element at all. Rather than regenerate `model.xml` with a ring of
rangefinders, `tools/spawn_robot.py::laser_scan_ranges` casts `mujoco.mj_ray` in a fan
and publishes `sensor_msgs/LaserScan` over rosbridge. 360 beams cost about 1 ms, against
a 100 ms scan period.

The parameters are the hardware's, not invented — from `ydlidar_ros_driver/launch/X2.launch`
and the `base_footprint -> laser_frame` static transform in `myagv_odometry/launch/myagv_active.launch`,
both on the [`myagv_ros_2023Pi`](https://github.com/elephantrobotics/myagv_ros/tree/myagv_ros_2023Pi)
branch:

| | value | flag |
|---|---|---|
| `frame_id` | `laser_frame` | — |
| `range_min` / `range_max` | 0.1 / 12.0 m | `--scan-min-range` / `--scan-range` |
| rate | 10 Hz, independent of `--control-hz` | `--scan-hz` |
| mount, off `base_footprint` | x +0.065 m, z +0.08 m | `--scan-offset` |
| beams | 360 (the X2's 3 kHz at 10 Hz is ~300) | `--scan-beams` |

Details that are load-bearing:

* the rays exclude the robot's own root body, or every beam returns its chassis at 11 cm;
* the fan is cast from the **laser** origin, not the base origin. 65 mm is a whole cell
  at the 5 cm resolution `myagv_navigation`'s gmapping uses, and casting from the base
  centre instead is invisible in a viewer and ruins a map;
* the scan runs **counter-clockwise from `-pi`** in the base frame. The X2 is launched
  `inverted: true` because it is mounted upside down, and the static transform then rolls
  `laser_frame` by pi; the two mirrors cancel. Do not change one without the other.

Two deliberate departures from the hardware, both of which a correct client tolerates
anyway — it should be testing `range_min <= r <= range_max`, which is true under every
convention:

* a miss is sent as `range_max + 1` rather than `inf`, because JSON has no infinity. The
  real driver runs `invalid_range_is_inf: false` and reports `0.0`;
* the X2's `ignore_array: "-50,50"` blind wedge is not modelled. Its orientation cannot
  be confirmed without the hardware, and guessing wrong would carve free space out of a
  real obstacle.

The depth image (`--depth-hz`, default 5 Hz, 320×240 uint16 millimetres) comes from a
second `mujoco.Renderer` in depth mode; it is deliberately slower and smaller than the
colour stream, since 640×480 float over a JSON websocket is 1.2 MB a frame.

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

## rebot_b601

[Seeed Studio reBot Arm B601-DM](https://www.seeedstudio.com/reBot-Arm-B601-DM-p-6740.html):
6-DoF arm, 767 mm reach, plus a two-finger prismatic gripper. Built from the vendor
URDF at [`vectorBH6/reBotArm_control_py`](https://github.com/vectorBH6/reBotArm_control_py),
a raw SolidWorks export — real geometry, limits and inertias, but no actuators, no TCP
frame and no MuJoCo tuning.

A convenient accident of the CAD: **link6's frame already matches the MolmoSpaces
gripper convention** (+z approach, fingers separating along y), so the TCP site needs
only an offset, no rotation.

Three URDF-import behaviours worth knowing:

* MuJoCo **strips the directory from mesh filenames**, so `meshdir` must be set to
  `../meshes` — and a patched copy elsewhere on disk would silently fail to find them.
* MuJoCo **merges the URDF root link into the worldbody** when it has no joint, so
  `base_link` vanishes and `link1` becomes the root. The mocap pedestal replaces its
  visual, as for the SO-101.
* **`spec.compile()` clears the name table of elements added before it.** Anything named
  must be added *after* the measurement pass, or it comes out anonymous and cannot be
  looked up.

Gains are sized per joint from two competing requirements: stiffness from the URDF's own
effort limits (36 N·m shoulder/elbow, 14 N·m wrist), clamped by the explicit-integrator
stability bound `kv·dt/I < 2` against the inertia measured from `dof_M0`. Hardcoding a
`kv` is exactly what fails — the wrist inertias are ~1e-3 kg·m², so a value that looks
sensible beside the shoulder's makes joints 4–6 oscillate instead of hold.

### Known limitations

* **Self-collision is disabled.** MuJoCo collides the convex hull of each visual mesh,
  and on a raw CAD export those hulls overlap wherever parts nest — the gripper fingers
  worst of all, driving joint6 a full radian off target with 81 N·m fighting them.
  Excluding only the pairs touching at rest was not enough, since more collide as the arm
  moves. The arm can therefore pass through itself. Collision with the world is
  unaffected. A proper fix is convex decomposition (`coacd`) of the collision meshes.
* `add_robot_to_scene` forces the implicit integrator: the gains assume it, and a bare
  `MjSpec` defaults to Euler, where the wrist goes NaN. Every MolmoSpaces house already
  uses `implicitfast`, so this only matters for standalone scenes.
* Collision uses per-link convex hulls, which are coarser than the visual meshes.

## ainex

[Hiwonder AiNex](https://www.hiwonder.com/products/ainex): a 24-DoF biped humanoid,
193 × 135 × 415 mm, 2.45 kg, walking at 21 cm/s on HX-series serial bus servos. Two 5-DoF
arms ending in a single hinged claw, a 2-DoF pan/tilt head carrying the only camera, a
9-axis IMU, and **no lidar, no depth sensor and no wheels**.

```bash
./run.sh view --robot ainex                          # spawn it in a house
./run.sh view --robot ainex --ros-port 9090          # ...on its own vendor ROS topics
python robots/ainex/test_attach.py                   # self-test (empty world)
python robots/ainex/test_attach.py --scene <house>   # self-test (in a house)
python robots/ainex/test_ros.py                      # self-test of the ROS surface
```

It is the first robot here that walks rather than rolls, and the first whose ROS contract
has nothing in common with the myAGV's. Both facts drive everything below. Provenance and
the vendored files are in [ainex/urdf/PROVENANCE.md](ainex/urdf/PROVENANCE.md).

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
model: `robots/ainex/actions/` holds a small set of keyframed poses, and `actions.py` also
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

**Faithful:** the 24-joint topology and the servo-id map; the count↔radian mapping,
including the per-joint `init` offset and the two deliberately inverted `sho_pitch` servos;
the native topics and services, with no `/cmd_vel`, no `/odom` and no `/tf`; grasping as
action-group replay with no IK; the gait envelope and the 0.20 m/s it yields at the default
gait; `/camera/image_raw/compressed` as `image_transport`'s standard companion topic.

**Deliberate departures**, each of them load-bearing somewhere:

* **`/scan` is a virtual lidar and the real AiNex has none.** The mount pose is invented,
  not transcribed — mid-torso, which is above the leg swing. Enabled by default because
  it is what makes the robot navigable; `--no-scan` turns it off.
* Locomotion is a planar base plus a cosmetic gait: no balance, no falling, and the torso
  cannot pitch or roll.
* `gravcomp` is on, so it does not sag the way a 2.35 kg robot on hobby servos really does.
  Without it every limb pose, including the replayed grasps, would land somewhere other
  than commanded.
* The feet do not collide with the floor. Colliding feet grip at default friction and fight
  the world-aligned position servos, which shows up as an undershooting base picking up
  uncommanded yaw. Only one torso hull and the two hands collide with the world.
* `/imu` reports real yaw and yaw rate with roll and pitch identically zero, because the
  base has no roll or pitch degree of freedom. Covariances are `-1`, the ROS convention for
  "not reported".
* The shipped action groups are ours, not Hiwonder's — see the licence note below.
* Scan misses are `range_max + 1` rather than `inf`, inherited from the existing convention
  (JSON has no infinity).

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
