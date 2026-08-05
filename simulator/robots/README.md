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
| `lekiwi` | spawn, view, holonomic drive, arm + gripper |
| `rebot_b601` | spawn, view, arm + gripper (no self-collision) |

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

`model.xml` is **generated** from it by `make_model.py`, which adds:

* a `tcp` site in MolmoSpaces convention, positioned at the true grasp centre (the
  midpoint between the jaw tips, ~12 mm from the stock `gripperframe`) and oriented
  with +z along the approach. Both axes are *measured from the model* at generation
  time rather than hardcoded, so the file stays correct if the jaw geometry changes.
* an `exo_camera` on the base. The upstream `wrist_cam` is kept as the wrist camera.

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
* **No grasp library**, so the scripted pick/place planners and the datagen pipeline
  (`run.sh sim`) do not work with it — those need per-gripper grasp sets, which exist
  only for `droid` (Franka) and `rum`. Use `run.sh view --robot so101` plus the
  bridge instead.
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

## lekiwi

SO-ARM100 arm on a three-wheel kiwi base, from
[Ekumen-OS/lekiwi](https://github.com/Ekumen-OS/lekiwi). Upstream is a genuine omni
drive — three hinge joints with velocity actuators — which this adapter **replaces with
the same virtual holonomic base as `myagv`**, since MolmoSpaces has no kiwi
inverse-kinematics controller (`controllers/base_pose.py` sketches a swerve variant but
is marked untested and its `pose` handling is broken).

The upstream file is loaded untouched and the spec is edited in memory, which keeps
`upstream/` pristine and avoids rewriting the relative `meshdir` and `<attach file=...>`
paths a copied XML would break.

Four things upstream does that had to be undone, each of which broke the base until fixed:

1. **Contact pairs naming a `floor` geom** that only exists in upstream's own
   `scene.xml` — the model will not even compile standalone with them.
2. **A free joint on the chassis.** A body may carry at most 6 DoF, so it cannot coexist
   with the three virtual joints; and with it gone the chassis sits at its declared z=0,
   buried in the floor, hence the ride-height offset on attach.
3. **Wheel and plate colliders.** Those contact pairs were what gave the omni wheels
   their sideways slip. Without them the chassis grips the floor at default friction and
   fights the virtual joints — the base under-shot by half and picked up ~20° of
   uncommanded yaw. Every chassis collider is replaced by one lifted box hull.
4. The arm is **SO-ARM100**, the SO-101's predecessor, with joints
   `Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll` and a `Jaw` gripper.

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
