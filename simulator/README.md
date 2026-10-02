# Simulator

Runs a MuJoCo household scene in one of two **engines** (MolmoSpaces or RoboCasa), lets
you spawn any robot of [`robots_specs/high_level_spec.md`](../robots_specs/high_level_spec.md)
into it by id, and serves each spawned robot's **vendor ROS interface** on its own
rosbridge websocket, from a Docker container running the robot's real ROS distribution.
The requirements are [`high_level_spec.md`](high_level_spec.md).

## Quick start

```sh
simulator/molmospaces/run.sh setup          # once (or robocasa/run.sh setup); also fetches
                                            # the robot meshes and the worktop objects
simulator/kitchen.sh start                  # MolmoSpaces iTHOR house ithor:1, headless
simulator/spawn.sh myagv --placement floor  # in another terminal: ws://127.0.0.1:9090
```

```
simulator/kitchen.sh start [--engine molmospaces|robocasa] [--scene <source>:<id>] [--mujoco] [--sim-port <p>]
simulator/<engine>/run.sh setup | assets [<source>] | start [--scene ..] [--sim-port ..] [--mujoco] | repair
simulator/spawn.sh <id> [--placement worktop|floor] [--sim-port <p>] [--port <p>] [--arm-port <p>]
```

| Engine | Scene sources | Default |
| --- | --- | --- |
| `molmospaces` | `ithor:<n>`, `procthor:<n>`, `test:1` | `ithor:1` |
| `robocasa` | `robocasa:<layout>-<style>` (1-60 each), `test:1` | `robocasa:1-1` |

`test:1` is a flat 12 x 12 m floor with one 1.2 x 0.8 m worktop whose top is at 0.75 m,
centred 1.5 m along +x -- the same MJCF on both engines.

Robot ids (`spawn.sh --help` prints them from the robot specification):

| id | wire(s) | mobile? | default evidence placement |
| --- | --- | --- | --- |
| `myagv` | ROS 1 Noetic on `--port` | yes | floor |
| `so101` | ROS 2 Jazzy on `--port` | arm | worktop |
| `ainex` | ROS 1 Noetic on `--port` | yes (humanoid) | floor |
| `mycobot280` | ROS 2 Humble on `--port` | arm | worktop |
| `myagv_mycobot280` | base: ROS 1 on `--port`, arm: ROS 2 on `--arm-port` (default `--port`+1) | yes | floor |
| `rosmaster_x3_plus` | ROS 1 Noetic on `--port` | yes | floor |

`spawn.sh` prints one readiness line when every wire serves its recorded interface, e.g.

```
spawn ready: myagv_mycobot280 (myAGV + myCobot 280) in molmospaces ithor:1 on the floor; wire(s): base (myagv) ws://127.0.0.1:9090 [ROS 1 noetic]; arm (mycobot280) ws://127.0.0.1:9091 [ROS 2 humble]
```

and stays in the foreground. `Ctrl-C` (SIGINT) or SIGTERM removes the robot and its
containers and exits 0. Every refusal, failed startup, wire failure, or the simulation
ending exits non-zero. If `spawn.sh` is killed (even `SIGKILL`) the simulation removes the
robot and releases its id and placement; its containers exit and remove themselves.

## Layout

| Path | What |
| --- | --- |
| `kitchen.sh`, `spawn.sh` | entry points |
| `<engine>/run.sh`, `<engine>/env.sh` | per-engine launcher and environment |
| `molmospaces/tools/` | the reference project's scene resolution and asset relinking, unchanged |
| `robocasa/tools/fetch_assets.py` | builds `robocasa/assets/` (generated; the upstream checkout is never written) |
| `shared/scenes.py` | scene sources: naming, validation, loading (shared by both engines) |
| `shared/simulation.py` | the simulation process: physics, rendering, the control port |
| `shared/world.py` | the world: add/remove robots by recompiling with state kept; physics loop |
| `shared/placement.py` | the one placement function (floor and worktop) |
| `shared/worktop_survey.py` | the reference project's worktop survey: which surface is the worktop, the spots a worktop robot is tried at |
| `shared/worktop_objects.py` | the six worktop objects staged around a worktop robot, and the clearing of its working area |
| `shared/objects/` | `ycb.sha256` and `LICENSES.md` of the worktop objects' YCB meshes; `ycb/` is fetched by `run.sh setup` (git-ignored) |
| `shared/tools/fetch_objects.py` | fetches and verifies `shared/objects/ycb/` from its pinned source |
| `shared/robot_model.py` | a registry robot as a MuJoCo spec (composites assembled at the recorded mount) |
| `shared/sensors.py` | render pool, camera/lidar/joint/IMU sampling |
| `shared/spawn.py` | `spawn.sh`: admission, Docker wires, readiness, watching, cleanup |
| `shared/registry.py` | the robot registry, parsed from `robots_specs/high_level_spec.md` |
| `shared/protocol.py`, `shared/simport.py` | the control-port protocol and a CLI client |
| `shared/rosbridge_client.py` | stdlib rosbridge client (spawn health checks, tests) |
| `shared/wire/` | everything that runs inside the wire containers (mounted read-only) |
| `shared/wire/docker/` | wire images: `noetic`, `humble`, `jazzy` bases and per-robot layers |
| `shared/wire/robots/<id>.py` | each robot's wire plan and simulated drivers (one per robot) |
| `tests/` | unit, integration and end-to-end checks (see below) |

## Scenes

The household scenes are loaded exactly as the prior version of this project (the
reference) loaded them, with nothing added (no light, object or robot; a worktop robot
brings its own objects when it is spawned, below):

* **MolmoSpaces**: the pinned `allenai/molmospaces` checkout (`713fd12a`), assets at the
  versions it pins (iTHOR scenes `20251217_with_occupancy`, THOR objects `20251117`, ...),
  the house resolved and installed by the reference's `tools/resolve_scene.py`
  (`get_scenes` + `install_scene_with_objects_and_grasps_from_path`) and read with
  `mujoco.MjSpec.from_file`.
* **RoboCasa**: robosuite `5ce6643f` + robocasa `v1.0`; the kitchen built from
  `KitchenArena(layout, style, rng=default_rng(0))` with an empty robot list inside a
  robosuite `ManipulationTask` (multi-CCD on, sleeping islands off), fixtures only. Its
  asset files are the same downloads the reference extracted into its checkout; here they
  go to the generated `robocasa/assets/` tree and `robocasa.models.assets_root` points
  there. RoboCasa's collision hulls (group 0) are hidden, visuals (groups 1, 2) shown.

Scene equivalence with the reference was checked by building `ithor:1` and `robocasa:1-1`
with the reference's own loader and with this one: identical model counts and arrays
(body/geom poses, sizes, meshes, textures, lights, materials) and pixel-identical renders
from three viewpoints each. The only difference is the random colours RoboCasa gives 11
hidden collision hulls, which also differ between two runs of the reference itself.

The **worktop** is the surface the reference project's worktop survey ranks first
(`shared/worktop_survey.py`, a port of the reference's `scene_placement.py`,
`place_arm_on_table` and RoboCasa `find_counter_mount`): on iTHOR/ProcTHOR and `test`, the
table-height (0.35-1.30 m) top faces of the scene's bodies, those holding a plate or an
apple (and reachable loose objects) first, or with nothing graspable the largest; on
RoboCasa the roomiest counter region of `Counter.get_reset_regions`. `ithor:1`: the island
(`StandardIslandHeight`) at 1.100 m; `robocasa:1-1`: the counter run
`counter_main_main_group/geom_1` at 0.920 m; `test:1`: the block at 0.75 m.

## Worktop robots and their objects

The arms (`so101`, `mycobot280`) spawned on the worktop stand where the reference project
stood its arm, and bring the reference's six task objects with them.

**Where.** The survey ranks spots, the reference's own choice first; `shared/placement.py`
tries them in order and takes the first one that is supported, clear (the robot's
collision geometry against the scene, after clearing, in a trial compile), camera-clear
and that leaves every object standing on the surface:

| Scene | so101 | mycobot280 |
| --- | --- | --- |
| `ithor:1` | island (-0.382, 0.218), z 1.100, yaw 0 deg -- the reference's spot, exactly | the same cell, yaw 30 deg (the reference's rule with its own reach, (0.14, 0.32) m) |
| `robocasa:1-1` | counter (2.231, -0.200), z 0.920, yaw -90 deg -- the reference's spot, exactly | counter (2.231, -0.580), z 0.920, yaw 90 deg: the front edge, facing the wall |

On RoboCasa the reference's back-edge spot puts the myCobot 280's upright arm 16.7 mm into
the wall cabinet, and every spot of the reference's tabletop search over the counter tops
still has it under the cabinet; the survey's last tier (not the reference's) stands it
against a counter's front edge facing the wall, its objects on the counter in front of it.

**What.** `shared/worktop_objects.py` stages, in the robot's base frame (4 mm above the
top face), the reference's apple (a 20 mm sphere with its tuned contact, under a YCB mesh),
plate (fixed; a cylinder and a 24-box rim) and the bowl, mug, banana and lemon (meshes,
condim 6), at the reference's poses. The scene's loose objects within 0.55 m of the base
(0.15 m below to 0.45 m above it) are cleared: their free joint removed, parked 50 m
under the scene (on `ithor:1`: the apple, bread and book on the island). Nothing else of
the reference's task comes along (no rig cameras, `/reset`, truth log or solver changes),
and nothing reaches a wire. The objects settle about 4 mm onto the surface after the spawn.

**Removal.** Ending the spawn removes the robot and its objects in one recompile and
returns each cleared object with the pose and velocity it had when cleared; every other
robot and object keeps its state.

**Assets.** The YCB meshes are not committed: `run.sh setup` (either engine) runs
`shared/tools/fetch_objects.py`, which downloads them from elpis-lab/YCB_Dataset at
`9e8c6488a2ff673d9aa48a91492fb89423c1b106` (byte-identical to the reference's vendored
files) into `shared/objects/ycb/`, verifies them against `shared/objects/ycb.sha256` and
refuses on a mismatch. `python3 simulator/shared/tools/fetch_objects.py --check` verifies
only. A worktop spawn without them is refused, naming `run.sh setup`.

## Control port (`--sim-port`, default 9080) -- simulator-private

One TCP connection carries framed messages: `u32 big-endian header length | UTF-8 JSON
header | payload` (a payload of `nbytes` bytes follows when the header has `nbytes`).
Requests carry `op` and `id`; replies echo `id` with `ok` (and `error`). Nothing here
reaches a ROS wire. `shared/protocol.py` has a stdlib client (`protocol.Client`), and
`shared/simport.py` a CLI:

```sh
python3 simulator/shared/simport.py hello | robots | scene
python3 simulator/shared/simport.py readings <robot id>
python3 simulator/shared/simport.py render --out scene.png --robot <id>           # frames the robot
python3 simulator/shared/simport.py render --out cam.png --camera <id>/<camera> --width 640 --height 480
python3 simulator/shared/simport.py render --out v.png --pos X Y Z --target X Y Z [--fovy 60]
python3 simulator/shared/simport.py render --out v.png --lookat X Y Z --distance D --azimuth A --elevation E
```

Operations (all fields JSON):

| op | request | reply |
| --- | --- | --- |
| `hello` | `role?` | `engine`, `scene`, `sim_time`, `floor_z`, `worktop` (`name`, `z`, `area_m2`, `centroid`, `bounds`) or null, `robots`, `starting`, `rtf` |
| `robots` | | `robots`: `[{id, state (pending/running), placement, xyz, yaw, ports, prefix, components, staged, cleared}]` -- `staged`: `{object: {body, pos, quat}}` as staged around a worktop robot (empty for others), `cleared`: the scene bodies cleared for them |
| `scene` | | model counts (`nbody`, `ngeom`, `nmesh`, ...), `scene_nbody`, `free_bodies` (loose objects, staged ones included), `staging` (`{robot: {staged, cleared}}`), `worktop`, `sim_time`, `rtf` |
| `readings` | `robot` | `joints`: `{name: {position, velocity, effort}}` (a composite's arm joints are `arm/<name>`); `base`: `{pos, quat (w,x,y,z), linvel_world, angvel_world, linvel_local, angvel_local}` of the robot's top body; `bodies`: `{name: {pos, quat}}` of every body of the robot; `sites`; `ctrl` (actuator targets); `stamp` (wall time), `sim_time` |
| `bodies` | `names: [..]` | world poses of any named bodies (scene objects included) |
| `render` | `view`, `width`, `height`, `format` (`png` default, `rgb`, `depth`) | payload: PNG bytes, raw RGB (`height*width*3`), or float32 metres (`height*width`) |
| `describe` | `robot`, `component?` | joints, actuators (`position`/`velocity`), cameras (name, fovy, resolution), sites |
| `spawn` | `robot`, `placement`, `ports` | `robot`, `token`, `prefix`, `xyz`, `yaw`, `surface_z`, `placement`, `components`, `staged`, `cleared` (as in `robots`); the connection holds the robot until `remove` or until it closes |
| `commit`, `remove`, `precheck`, `wire`, `ctrl`, `subscribe` | | used by `spawn.sh` and the wires |

`view` is one of: `{"frame_robot": id, "elevation": -30}` (in front of the robot, turning
until the line of sight is clear), `{"camera": "<robot id>/<camera>"}` (a robot camera
exactly as its wire renders it; cameras are named after their image frame ids:
`myagv/camera_link`, `so101/default_cam`, `ainex/camera`,
`rosmaster_x3_plus/camera_color_optical_frame`), `{"lookat": [x,y,z], "distance": d,
"azimuth": deg, "elevation": deg}`, or `{"pos": [x,y,z], "target": [x,y,z], "fovy": deg?}`.

Example (Python, any python3):

```python
import sys; sys.path.insert(0, "simulator/shared")
import protocol
c = protocol.Client("127.0.0.1", 9080)
before = c.call("readings", robot="ainex")["base"]["pos"]
png = c.call("render", view={"frame_robot": "ainex"}, width=1280, height=720)["_payload"]
open("scene.png", "wb").write(png)
```

## Wires

Each wire is a container (`rsim-<sim port>-<id>-<role>`, rosbridge listening on the same
port number inside as it is published on outside) built on first use from pinned
sources (base images by digest; rosbridge_suite 0.11.17 on Noetic, 2.0.8 on Humble; the
jazzy image's ros2_controllers 4.42.1) and started with the simulator code mounted
read-only. Its ROS graph holds every non-optional node of the robot's interface file with
the recorded names, types, frames, parameters and rates, plus ROS infrastructure
(rosbridge_websocket, rosapi, rosout). Where the recorded boot runs vendor or stock code
that needs no hardware, that code runs unchanged; hardware is simulated underneath it:

| Robot | Real code on the wire | Simulated |
| --- | --- | --- |
| myAGV | joint_state_publisher, robot_state_publisher, tf static publishers, the vendor's modified robot_pose_ekf (built from myagv_ros `c71f3cc5`) | `/myagv_odometry_node` (myAGV.cpp's behaviour: 0.01 m/s MCU resolution, 0.1 deg yaw dead band, held command, no watchdog), `/ydlidar_lidar_publisher` (X2L: 400 samples, ignore sector reported 0), `/usb_cam` |
| SO-101 | robot_state_publisher, ros2_control_node with joint_trajectory_controller, parallel gripper controller, joint_state_broadcaster (spawned by the stock spawner) | the `feetech_ros2_driver/FeetechHardwareInterface` plugin (same name, simulated STS3215 bus: 2400 ticks/s, acceleration register 50, 4096-tick readback), `/usb_cam` (YUYV) |
| AiNex | the vendor's `ros_robot_controller_node.py` and `ainex_controller.py` with its gait engine (`walking_module.so`, `kinematics.so`), imu_calib, imu_complementary_filter, the nodelet manager with image_proc/rectify, web_video_server | the controller board (bus servos 0-1000 pulses over their move time, IMU in g and deg/s), `/camera`, `/joystick` (an idle receiver), `/sensor`, `/joystick_control`, `/color_detection`, `/face_detect`, `/app` (endpoints with default behaviour) |
| myCobot 280 | robot_state_publisher (with `xacro` of the boot URDF) | `/slider_control_adaptive_gripper` (pymycobot's send_angles/set_gripper_value semantics, including ending on an invalid command) |
| ROSMASTER X3 PLUS | the vendor's `Mcnamu_X3plus.py` driver, `base_node`, `yahboom_joy.py` (built from yahboomcar_ws.zip), robot_state_publisher, imu_filter_madgwick, robot_localization EKF, joy_node, tf publishers | the expansion board (`Rosmaster_Lib`: firmware mecanum mixing, 0.7 m/s wheel clamp, servo run times), `/ydlidar_lidar_publisher` (4ROS), `/camera/camera` (Astra Pro Plus colour, 16UC1 depth, IR, point clouds, TF) |

Robots move only through contact of their own wheels (mecanum rollers modelled in the
robot models), feet and grippers. `spawn.sh` watches every wire (container state and a
rosapi node check every 2 s) and ends the spawn, removing everything, when one fails. A ROS 1
node killed without unregistering stays on the master's list, so a ROS 1 wire also asks
each recorded node's XML-RPC server every 2 s and ends itself, reporting
`lost node(s) ...`, when one stops answering.

### Adaptations and estimates (simulator side)

* ROSMASTER `orbbec_camera` messages/services: the vendor's copy (software.zip in the
  pinned Drive archive) could not be fetched (Drive quota); definitions come from Orbbec's
  public OrbbecSDK_ROS1 v1.4.2 (`7ef885a4`), whose v1 service set the interface records.
  Its IR image is the rendered luminance scaled to 10 bits (IR is not rendered). The
  expansion board's IMU yaw-hold correction while driving is not reproduced. Battery
  12.3 V, firmware version 3.5 and the magnetometer's field are estimates. The driver's
  dynamic_reconfigure server is the vendor's own (built from its cfg).
* myAGV `/Voltage` 12.0 V and `/voltage_backup` 0.0 are estimates; IMU raw units as the
  interface file estimates (deg/s, m/s^2).
* myCobot 280 motion pacing: speed 25 is taken as 40 deg/s (25 % of ~160 deg/s) and the
  gripper crosses its range in ~0.5 s -- estimates (the manufacturer gives no figure).
* SO-101 servo acceleration register 50 = 5000 ticks/s^2 (Feetech's documented unit;
  the pinned source does not state it).
* AiNex action groups: the vendor's `.d6a` files are in no pinned source; the simulator
  writes its own groups (`wave`, `raise_hands`, `nod`; arm and head motions from the init
  pose) with `shared/wire/robots/ainex_actions.py`. The nodes needing the robot's GPIO,
  camera pipeline or app stack present their endpoints with default behaviour.
* ros2_control on Jazzy is the apt release 4.48.0 (the interface file cites 4.48.1).

Differences between a wire and its interface file that come from the unchanged stock code
the boot runs (so the real robot shows them too) are listed in
`tests/wirecheck.py::INTERFACE_FILE_GAPS` and were raised with robots_specs: AiNex `/tf`
(advertised, never sent, by imu_complementary_filter), the ROSMASTER's
`/imu_filter_madgwick/{gain,zeta,mag_bias_*}` (written by its dynamic_reconfigure server),
and SO-101 `robot_state_publisher` not declaring
`qos_overrides./joint_states.subscription.durability`.

## Timing

Physics runs against the wall clock on its own thread; periodic streams are sampled by
elapsed wall time and stamped with their acquisition time. When the real-time factor over
a 10 s window falls below 0.90 the simulation prints a warning on stderr. Camera streams
render on their own GL contexts; lidar rays are cast on a mirror of the world, off the
physics thread.

## Verification

```sh
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/unit simulator/tests/integration
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/e2e      # Docker, minutes
# reference parity of the worktop spot and objects, against a checkout of the reference
RSIM_REF_DIR=<checkout of github.com/samirma/robot-simulator at 34547ae> \
  simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/integration/test_reference_parity.py
```

See `tests/README.md` for what each suite checks.
