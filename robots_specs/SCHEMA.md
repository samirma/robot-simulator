# ROS interface file schema (`ros.yml` / `ros2.yml`)

Every robot folder `robots_specs/<id>/` holds exactly one interface file: `ros.yml` for a
ROS 1 boot, `ros2.yml` for a ROS 2 boot. All of them follow this schema (version 1), so a
consumer such as the simulator can read every robot the same way.
`robots_specs/tests/schema.py` checks it (`validate(path)`); `robots_specs/tests/` runs it
over every robot.

Conventions:

* **Names** are fully qualified ROS names (`/cmd_vel`, `/camera/image_raw`, node
  `/robot_state_publisher`), exactly as they appear on the real robot's graph.
* **Types** use the dialect's own spelling, as rosapi reports it: ROS 1 `pkg/Type`
  (`geometry_msgs/Twist`); ROS 2 `pkg/msg/Type`, `pkg/srv/Type`, `pkg/action/Type`.
* **Units** are SI unless a row states otherwise (m, rad, s, m/s, rad/s, Hz).
* **Poses** are `xyz: [x, y, z]` in m and `rpy: [roll, pitch, yaw]` in rad (URDF
  convention, fixed axes).
* **`source`** is a URL pinned to a revision (`.../blob/<sha>/path#Lnn`), or for the
  ROSMASTER's Drive archive a `zip-member` reference `<archive>!/<path>#Lnn`. It may be a
  single string or a list.
* **`basis`** says how a figure was established: `source` (read from the pinned code or
  launch files), `manufacturer` (vendor documentation or datasheet), `measured` (real
  hardware; must then carry `method` and `date`), `estimate` (stated as such, with a
  reason), or `uncalibrated` (the boot publishes no calibration). Nothing is ever marked
  `measured` unless it was measured on the real robot.
* **`optional: true`** marks any row (node, topic, service, action, parameter, tf,
  camera) that exists only when an optional plugin is installed or when a launch other
  than the authoritative boot runs. The simulator does not serve optional rows. Absent
  means `false`.
* ROS infrastructure (rosbridge, rosapi, `/rosout`, `/rosout_agg`, `/parameter_events`,
  master parameters, client-library logger/parameter/type-description services, ROS 2
  action-internal topics and services) is never listed.

## Top level

| key | type | meaning |
| --- | --- | --- |
| `schema_version` | int | `1` |
| `robot_id` | str | the robot id; equals the folder name |
| `name` | str | display name as in the robot file `<id>.md` |
| `dialect` | `ros1` \| `ros2` | must match the filename (`ros.yml` = `ros1`, `ros2.yml` = `ros2`) |
| `ros_distribution` | str | e.g. `noetic`, `melodic`, `humble`, `jazzy` |
| `interface_authority` | `manufacturer` \| `approved_community` | who defines the interface |
| `camera_boot_authority` | `approved_community` | optional: the camera rows come from an approved community camera boot while the rest is `interface_authority` (myAGV) |
| `sources` | list | pinned sources (below) |
| `boot` | map | the authoritative boot (below) |
| `model` | map | URDF / MJCF files and `robot_description` (below) |
| `nodes` | list | every node of the boot |
| `topics` | list | every topic |
| `services` | list | every service (may be empty) |
| `actions` | list | every action (may be empty) |
| `parameters` | list | every parameter |
| `tf` | list | every transform the boot publishes |
| `motions` | list | every commanded motion, its units, limits, stop and watchdog |
| `sensors` | map | `cameras`, `lidars`, `imus` (lists, possibly empty) |
| `tolerances` | list | per-figure acceptance tolerances (simulator spec §5) |
| `notes` | str | optional free text |

### `sources[]`

`{id, url, revision, branch?, role, sha256?, retrieved?}` — `id` is a short key other rows
may cite; `revision` is a commit hash, a release tag, or for a download the file name.
`sha256` records a download's digest. `role` is prose (e.g. `official source`,
`community ROS bringup`, `dependency (release used by the boot's distribution)`).

### `boot`

`{summary, commands[], launch_files[], data_files[], hardware}` —
`commands` are the shell commands the vendor documentation runs for normal operation;
`launch_files`/`data_files` are `{source, path, role?}` rows (no other keys) at the pinned revision
(`source` is a `sources[].id`); `hardware` is prose: computer, OS image, serial devices,
camera and lidar devices, controller firmware.

### `model`

`{urdf, urdf_sha256, urdf_source, mjcf, mjcf_official, robot_description}`

* `urdf` — file name inside the robot folder; `urdf_sha256` its digest;
  `urdf_source` a `{source, path}` row (or prose if it was generated, e.g. from xacro).
* `mjcf` — the MuJoCo model to load: the official MJCF where one exists and is complete,
  else the derived `model.xml`. `mjcf_official` is `true` only for a manufacturer MJCF.
  `official_mjcf` (optional) names the official MJCF when `mjcf` is a derived model that
  extends it.
* `robot_description` — `{published_as, content, differences_from_model[]}`: how the boot
  publishes it (parameter/topic), what it contains (the URDF file, or how it is
  generated) and each difference from the model (joint names, axes, limits, zero
  offsets); `import.md` lists the same differences as adaptations.

### `nodes[]`

`{name, package, executable, source, optional?, notes?}`

### `topics[]`

`{name, type, direction, nodes, rate, rate_basis, source, ...}`

* `direction`: `out` (the robot publishes) or `in` (the robot subscribes).
* `nodes`: publishing nodes for `out`, subscribing nodes for `in`.
* `rate`: a number in Hz for a periodic topic — the aggregate rate on the wire when several
  nodes publish it — or the string `non_periodic` for a topic published only on events, on
  demand, latched/transient-local once, or a command input. Every `out` topic that is not
  `non_periodic` has a rate.
* `rate_basis`: `source` | `manufacturer` | `measured` | `estimate` (see conventions);
  `rate_note` explains how the value follows from the source.
* `frame_id`: the `header.frame_id` the publisher sets — required on every `out` row whose
  message carries a header (images, camera info, scans, point clouds, IMU, magnetic field,
  odometry, joint states, maps, paths and stamped geometry), `""` when the publisher leaves
  it empty; `child_frame_id` likewise for odometry.
* optional: `latched` (bool), `qos` (ROS 2 map: `reliability`, `durability`, `depth`),
  `camera: true` on each camera's **raw image stream**, `optional`, `notes`;
  `internal_publishers` on an `in` row lists boot nodes that also publish that command
  topic (a teleop or app node feeding the controller), `internal_subscribers` on an `out`
  row the boot nodes that consume it.

### `services[]`, `actions[]`

`{name, type, node, source, optional?, notes?}`

### `parameters[]`

`{name, node?, value, source, optional?, notes?}` — ROS 1: `name` is the full global name
(`/robot_pose_ekf/freq`), `node` optional. ROS 2: `name` is the parameter name and `node`
the fully qualified node. `value` is the literal boot value; for large values
(`robot_description`) a map `{file: <name in the folder>}` or `{generated: <prose>}`.

### `tf[]`

`{parent, child, static, publisher, rate, xyz?, rpy?, source, optional?}` — `static`
is `true` for fixed transforms; `rate` is Hz or `non_periodic` (latched `/tf_static`);
`xyz`/`rpy` are given for fixed transforms.

### `motions[]`

One row per motion the robot performs (`drive`, `walk`, `arm`, `gripper`, `head`,
`action_group`):

```yaml
- id: drive
  velocity_driven: true          # drive and walk are; every other motion is not
  command:
    interface: topic             # topic | service | action
    name: /cmd_vel
    type: geometry_msgs/Twist
    fields:                      # every commanded field: unit and limits
      - {field: linear.x, unit: m/s, min: -1.0, max: 1.0, source: ...}
    example: {linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}
  stop:                          # required when velocity_driven
    interface: topic
    name: /cmd_vel
    type: geometry_msgs/Twist
    message: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}
    source: ...
  watchdog:
    present: false               # true | false
    interval_s: null             # number when present
    basis: source                # source | measured
    method: "..."                # how it was established
    date: 2026-09-30             # when it was established
    source: ...
  feedback:                      # topics/fields that show the motion, or [] with a note
    - {name: /odom, field: pose.pose.position}
  end_state: "..."               # non-velocity motions: what "goal reached" means
```

A `watchdog` with `basis: source` has been derived from the pinned code only; its
`method` says so, and the real-hardware measurement the robot specification asks for is
still outstanding.

`feedback` rows are `{name, field, notes?}` with `name` a recorded topic, service or action;
`feedback_note` explains feedback or its absence. Further optional motion keys: `joint`
(the joint a row drives when one motion has several rows, e.g. the AiNex head's pan and
tilt), `kinematics` (a drive's wheel mixing and its figures), `gait` (a walk's gait engine,
files and defaults); in `command`: `notes`, `source`, `sequence` (a multi-step command
flow), `alternative` (another vendor command path for the same motion), `also` (further
endpoints accepting the same command), `servo_profile` (servo speed/acceleration the
hardware layer writes), `step_definition` with `step_definition_source` (what one step of
a walk command is, and so the nominal displacement it commands), and `estimated_groups`
with `smoke_example` and `smoke_example_basis: estimate` (an action-group motion whose vendor
groups are in no pinned source: the file of estimated groups and the playable example); in
`stop`: `notes`.

### `sensors`

* `cameras[]`: `{id, image_topic, info_topic?, compressed_topic?, frame_id, width, height,
  rate, encoding, hfov_deg?, intrinsics{fx, fy, cx, cy, distortion_model, d, basis,
  source}, mount{parent, xyz, rpy, basis, source}, hardware, optional?}`
  (`intrinsics.basis: uncalibrated` records what the boot publishes when it loads no
  calibration; the simulator then renders with the model camera's `fovy` and publishes
  the recorded CameraInfo content). Optional: `hfov_basis`/`hfov_note` (how `hfov_deg`
  was established), `range_m` with `range_basis` (a depth stream's working range, m),
  `notes` in `intrinsics` and `mount`.
* `lidars[]`: `{id, topic, frame_id, model, angle_min, angle_max, samples, range_min,
  range_max, scan_rate, mount{...}, basis, source}` (angles in rad).
* `imus[]`: `{id, topic, frame_id, rate, mount{...}, source, filtered_topic?}`
  (`filtered_topic`: the boot's filtered IMU output, when it has one).

### `tolerances[]`

`{figure, applies_to, tolerance{absolute?, relative?, unit?}, basis, source?, notes?}` —
the acceptance tolerance for a physical or sensor figure the simulator reproduces (for
example a commanded drive displacement read back from odometry, or a joint reaching its
commanded position). `basis: manufacturer` when the vendor states the accuracy,
otherwise `estimate` with the reason. Periodic rates are not listed here: the simulator
spec fixes them at ±10%.

## Estimated action groups (`action_groups.yml`)

A robot whose vendor action groups are in no pinned source may record playable groups,
labelled as estimates, in `robots_specs/<id>/action_groups.yml`, named by its action-group
motion's `command.estimated_groups`. `tests/schema.py` checks it (`validate_action_groups`).

| key | meaning |
| --- | --- |
| `schema_version`, `robot_id` | `1`; the folder name |
| `basis`, `reason` | always `estimate`, and why |
| `player` | `{path, file_format, source}`: where and in what format the vendor player reads a group |
| `frames_are`, `pulse_source` | how a frame becomes servo commands, and the files that mapping uses |
| `smoke_example` | a group name; equals the motion's `command.smoke_example.data` |
| `groups` | `{<name>: {description, frames: [{time_ms, offsets_rad: {<joint>: rad}}]}}` — offsets from the recorded init pose; every target stays inside the URDF joint limits |

## MuJoCo model conventions (`model.xml`)

A derived `model.xml` (and the official `so101_new_calib.xml`) is loaded with
`mujoco.MjModel.from_xml_path("robots_specs/<id>/<model.mjcf>")` once
`robots_specs/tools/fetch_meshes.py` has fetched the meshes. Derived models follow:

* Mesh `file`s are relative to the robot folder (`<compiler meshdir="."/>`): the fetched
  upstream meshes (`assets/`, `meshes/`, `urdf/`) and the files `fetch_meshes.py` derives
  under `derived_meshes/` (same relative path): `<x>.stl` (a COLLADA file as one STL, for
  collision), `<x>.dae.m<k>.obj` + `<x>.dae.mtl` (a COLLADA visual, one OBJ per material with
  its own colour/texture) and `<x>.STL.part<k>.stl` (lossless parts of an STL over MuJoCo's
  200000-face limit).
* Appearance comes from the official assets: a mesh's own materials (COLLADA colours and
  textures) where it has them, else the URDF `<material>` colour of that visual, else the
  official MJCF's materials. Visual geoms show the official visual meshes, never collision
  meshes.
* The model holds the robot only: no floor, lights or scene. The top body is named after
  the URDF root link. A mobile robot's top body carries `<freejoint name="root"/>`; an
  arm has none (it is fixed where it is placed).
* Body names are URDF link names; joint names are the boot `robot_description`'s joint
  names; each actuated joint has one actuator of the same name, in the command units the
  interface uses (position actuators in rad or m, wheel velocity actuators in rad/s).
  URDF `<mimic>` joints are `<equality><joint>` constraints.
* Visual geoms: `group="2" contype="0" conaffinity="0"`; collision geoms: `group="3"`.
* Each camera the interface publishes is a `<camera>` named after its image `frame_id`,
  at the recorded mount, with `fovy` and `resolution` matching the recorded image size
  and intrinsics. Each lidar and IMU is a `<site>` named after its `frame_id`.
* A `<keyframe>` named `home` holds the boot's initial configuration.
* Contact the URDF lacks (mecanum rollers, feet pads) is modelled explicitly and listed as
  an adaptation in `import.md`.
