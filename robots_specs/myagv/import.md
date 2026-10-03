# myAGV — MuJoCo import notes

`model.xml` in this folder is a **derived model, not manufacturer-provided**. Elephant
Robotics publishes no MJCF for the myAGV (Raspberry Pi 2023); the model is converted from the
official URDF and extended with the elements the URDF lacks but the real robot and its
interface need. This file records the sources, the conversion, every adaptation and every
estimate.

## Sources

| What | Source | Revision |
| --- | --- | --- |
| URDF `myAGV.urdf` (sha256 `afdac8fa…43e7`) and COLLADA meshes `urdf/myagv_base.dae`, `urdf/myagv_up.dae` (`urdf/myagv_all.dae` is referenced only inside a commented-out block, so it is not fetched) | <https://github.com/elephantrobotics/myagv_ros> `myagv_urdf/urdf/` | `c71f3cc574e5ed1973a925238eabe88662cfa701` (branch `myagv_ros_2023Pi`) |
| Sensor mounts (camera, IMU, laser) | same repo, `myagv_odometry/launch/myagv_active.launch` L9-11 | same |
| Weight 4.16 kg, max speed 0.9 m/s, mecanum wheels, camera 5 MP / 65°, dimensions 331.15 × 230 mm | <https://github.com/elephantrobotics/myAGV-docs> `MYAGV_PI_2023_EN/2-ProductFeature/2.1-MachineSpecification.md`, `2.3-MechanicalStructureParameter.md` (drawing `resources/2-ProductFeature/2.3/structure_param.png`) | `2510f8a3bd399823dd82088a92b1b2ee62d6d130` |
| Lidar YDLIDAR X2L figures | same docs repo, `3-UserNotes/3_hardware.md` L3-9 and `resources/3-UserNotes/FAQ/hardware_1.png` | same |

`package://myagv_urdf/` resolves to this folder (`robots_specs/myagv/`), so the URDF's meshes
live at `robots_specs/myagv/urdf/*.dae`. MuJoCo reads no COLLADA:
`robots_specs/tools/fetch_meshes.py` derives, under `derived_meshes/urdf/` (units applied: the
files are in inches, `<unit meter="0.0254">`, `Z_UP`; every node transform composed):

* `myagv_base.dae.m<k>.obj` / `myagv_up.dae.m<k>.obj` + `*.dae.mtl` — the **visual** meshes,
  one OBJ per COLLADA material with the file's own normals and texture coordinates, and an MTL
  with each material's own diffuse colour, opacity and texture (derived adaptation: MuJoCo
  applies one material per mesh geom, so a multi-material mesh becomes one geom per material).
  No collision STL is derived: this model's collision geometry is boxes.

All are hashed in `robots_specs/meshes.sha256`.

## Conversion steps

1. Fetch the meshes: `python3 robots_specs/tools/fetch_meshes.py myagv`.
2. Generate the model: `python3 robots_specs/tools/models/myagv.py >
   robots_specs/myagv/model.xml`. It is deterministic; all figures are constants at its top.
3. What it does:
   * URDF link `base_footprint` → top body `base_footprint` with `<freejoint name="root"/>`,
     visual mesh `myagv_base` at the URDF visual origin (identity).
   * URDF joint `base_up` (continuous, axis z, origin identity) → hinge `base_up`; link
     `base_up` → body `base_up` with visual mesh `myagv_up`.
   * **Appearance (derived adaptation):** each visual geom uses its COLLADA material's own
     colour, as RViz shows a mesh that carries materials. Both pinned `.dae` files carry a
     single material, `a0000000-…` (diffuse `0.980392 0.980392 0.980392 1`, opaque, no
     texture), so each is one OBJ part in near-white. The URDF `<material name="black">`
     (rgba `0.7 0.7 0 1`, `0.7 0.5 0 0.5` in the commented-out block) applies only to meshes
     without materials, so it is not used; the earlier yellow appearance came from it. The
     real myAGV Pi 2023 is a mid-grey chassis with dark mecanum rollers and white hubs
     (docs `resources/1-ProductIntroduction/README/PI-main.png`); the official meshes carry no
     such colours, and none are invented here.
   * The URDF has no inertials, collision geometry, wheels or sensors: all added below.
4. Validate: `mujoco.MjModel.from_xml_path("robots_specs/myagv/model.xml")` loads
   (nq 60, nv 59, nu 4, 56 bodies, 1 camera, total mass 4.160 kg), and the drive check below.

## Adaptations

### robot_description vs model

The boot's `robot_description` is `myAGV.urdf` unchanged. Differences:

* **`base_up` locked.** The URDF makes the top plate a *continuous* joint; the real plate is
  rigid and `joint_state_publisher` only ever publishes 0 for it. The model keeps the joint
  (same name, axis, zero) so the wire's `/joint_states` and `/tf` have a joint to report, and
  locks it at 0 with `<equality><joint joint1="base_up" polycoef="0 0 0 0 0"/>`.
* **Added bodies** (not in `robot_description`, never on the wire): four wheels with rollers,
  `camera_link` body with the camera, `laser_frame` and `imu_link` sites.

### Wheels and mecanum rollers (contact the source lacks)

The URDF has no wheels; the official mesh `myagv_base.dae` contains four identical wheel
instances. Measured on that mesh (converted to metres, in `base_footprint`):

| wheel | centre x | centre y | centre z |
| --- | --- | --- | --- |
| front left | 0.1124 | 0.09455 | 0.0400 |
| front right | 0.1124 | −0.10295 | 0.0400 |
| rear left | −0.1014 | 0.09455 | 0.0417 |
| rear right | −0.1014 | −0.10295 | 0.0417 |

Diameter 0.080 m, width 0.0315 m. Hence wheel radius r = 0.040, half wheelbase
lx = 0.1069, half track ly = 0.09875, wheel-centre centroid (0.0055, −0.0042). The CAD
assembly is tilted by ~0.45° (rear wheels 1.7 mm higher); the model puts all four hubs at
z = r so the chassis stands level — the visual mesh therefore sits 0.45° nose-down.
These are **estimates from the CAD mesh**; the manufacturer publishes no wheel figures, and
the drawing gives only overall dimensions (331.15 × 230 mm; mesh 314 × 229 mm, the difference
being the rear screen bracket not in the mesh).

Each wheel is a body at its hub with hinge `<wheel>_joint` (axis +y, positive = rolling
forward) and a `<velocity>` actuator of the same name (rad/s). Mecanum behaviour is modelled
physically, not kinematically: each wheel carries **12 passive rollers**, each a body with a
free hinge about the roller axis and a capsule (radius 0.012 m) whose axis is tilted 45°
between the rolling direction and the axle, at 0.028 m from the hub axis. The capsule
half-length, 0.028·tan(7.5°)·√2 = 5.2 mm, makes the rolling envelope flat to within 0.24 mm
(the capsule tips carry the wheel between rollers). Roller handedness (roller axis at the
ground contact, in the base x-y plane): front-left and rear-right along (1, −1), front-right
and rear-left along (1, 1) — the "X" arrangement that realises the standard mixing recorded
in `ros.yml` `motions[drive].kinematics`:

    w_fl = (vx − vy − (lx+ly)·wz)/r   w_fr = (vx + vy + (lx+ly)·wz)/r
    w_rl = (vx + vy − (lx+ly)·wz)/r   w_rr = (vx − vy + (lx+ly)·wz)/r

The MCU firmware that performs this mixing on the real robot is not published; the mixing
and its constants are estimates. The base moves only through roller–ground contact.

Actuators (estimates): `kv = 0.3 N·m·s/rad`, `forcerange ±1.5 N·m`, `ctrlrange ±22.5 rad/s`
(= manufacturer max speed 0.9 m/s / r). Wheel joint damping 0.001, armature 1e-4; roller
damping 1e-5, armature 1e-6 (numerical, for stability).

### Mass and inertia (estimates except the total)

| body | mass kg | basis |
| --- | --- | --- |
| whole robot | 4.160 | manufacturer (docs 2.1) |
| each wheel hub | 0.070 | estimate |
| each roller (×48) | 0.004 | estimate |
| `base_up` top plate | 0.300 | estimate |
| `base_footprint` chassis | 3.388 | remainder of 4.16 |

Inertias are solid-box/cylinder approximations of the mesh extents (chassis box
0.31 × 0.20 × 0.08 m centred at (0.0055, −0.0042, 0.055); top plate 0.31 × 0.20 × 0.04 m).

### Collision geometry (estimate)

The URDF has none. Two boxes fitted to the meshes: chassis `(0.0065, −0.004, 0.0555)` half
sizes `(0.155, 0.080, 0.0355)` (bottom 20 mm above the floor, like the mesh's lower shell),
top plate `(0.0065, −0.004, 0.1115)` half `(0.155, 0.100, 0.0205)`. Only the rollers touch
the floor. Robot collision geoms are `contype="1" conaffinity="0"`: they collide with the
scene but never with each other (no roller–roller or roller–chassis contact).

### Sensors

* **Camera** `camera_link`: pose from the boot's `base2camera_link` static transform
  (`xyz 0.13 0 0.131`, rpy 0; myagv_active.launch L9) — the frame the image carries
  (`_camera_frame_id:=camera_link`). The camera looks along +x of `camera_link` (MuJoCo
  `xyaxes="0 -1 0 0 0 1"`). The vendor launch starts no camera driver; the camera streams come
  from the approved community camera boot, the pinned usb_cam 0.3.7 (commit `addab4a6…`) run
  beside it — approved, not manufacturer-provided, and required (the ROS camera rows in
  `ros.yml` are not optional). `resolution="640 480"` follows that boot's driver defaults
  (640x480, 30 Hz, rgb8); `fovy = 41.83°` is an **estimate**: the manufacturer states a 65° viewing angle
  without an axis; taken as the image diagonal it gives f = 627.9 px (hfov 53.98°), consistent
  with the vendor's own approximation f = image width in `aruco_detector.py`. The CSI module's
  real intrinsics are not published.
* **Lidar** site `laser_frame`: `base2laser_link` (`xyz 0.065 0 0.08`, yaw π; launch L11).
  The mesh's lidar head centre (x ≈ 0.071, z ≈ 0.105 top) agrees within a few mm in x.
* **IMU** site `imu_link`: `base2imu_link` (origin, roll π, pitch π; launch L10).

## Model figures: manufacturer versus estimate

Manufacturer: total mass 4.16 kg, max speed 0.9 m/s, mecanum wheels, 65° camera, X2L lidar
figures, overall dimensions. Source (exact): URDF joint/visuals, sensor mount poses. Estimates:
wheel radius/positions (from the official mesh), mixing, masses split, inertias, collision
boxes, actuator gains and limits, camera field-of-view axis, roller geometry.

## Estimates (not manufacturer data)

Interface values the pinned sources leave to the hardware (added 2026-10-03):

* **`/Voltage` = 12.0 V**: the battery level is not modelled; a nominal 12 V pack is taken.
* **`/voltage_backup` = 0.0**: no backup battery is modelled.
* The units of the raw MCU IMU values are those `ros.yml` records as estimates (deg/s,
  m/s²).

## Drive check (scratch scene: this model + a floor plane, friction 1.0, implicitfast)

Settle 1 s, command the recorded mixing for 2 s, then zero wheel speeds. Displacement in the
start body frame (timestep 0.002 s):

| command | expected | got | after stop |
| --- | --- | --- | --- |
| vx +0.2 m/s | dx +0.400 | dx +0.389, dy 0.000, dyaw 0.000 | coasts 3 mm, 0.0008 m/s after 0.5 s |
| vx −0.2 m/s | dx −0.400 | dx −0.390 | coasts 3 mm |
| vy +0.2 m/s | dy +0.400 | dy +0.383, dx −0.003, dyaw +0.008 | coasts 3 mm |
| vy −0.2 m/s | dy −0.400 | dy −0.383, dx −0.003, dyaw −0.008 | coasts 3 mm |
| wz +0.5 rad/s | dyaw +1.000 | dyaw +0.977, drift 6 mm | 0.004 rad/s after 0.5 s |
| wz −0.5 rad/s | dyaw −1.000 | dyaw −0.977 | |
| vx +0.5 m/s, 1 s | dx +0.500 | dx +0.475 | coasts 20 mm |
| vx=vy=0.2 m/s | (0.400, 0.400) | (0.387, 0.385) | |

At rest the chassis stays level (root z −0.4 mm contact penetration); while driving the height
varies by at most 1.5 mm. Results at timestep 0.001 and 0.004 s agree within 1.5%. All within
the tolerances recorded in `ros.yml` (10% + 0.02 m / 0.05 rad).

## Generator

The generator is `robots_specs/tools/models/myagv.py` (standard library only):
`python3 robots_specs/tools/models/myagv.py > robots_specs/myagv/model.xml` reproduces
`model.xml` byte for byte.
