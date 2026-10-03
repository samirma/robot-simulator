# Simulator

Runs a MuJoCo household scene in one of two **engines** (MolmoSpaces or RoboCasa), lets
you spawn any robot of the robot specification
([`robots_specs/high_level_spec.md`](../robots_specs/high_level_spec.md) and its robot files
`robots_specs/<id>.md`) into it by id, and serves each spawned robot's **vendor ROS
interface** on its own rosbridge websocket, from a Docker container running the robot's
real ROS distribution.
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
simulator/spawn.sh <id> [--placement worktop|floor] [--sim-port <p>] [--port <p>]
```

| Engine | Scene sources | Default |
| --- | --- | --- |
| `molmospaces` | `ithor:<n>`, `procthor:<n>`, `test:1` | `ithor:1` |
| `robocasa` | `robocasa:<layout>-<style>` (1-60 each), `test:1` | `robocasa:1-1` |

`test:1` is a flat 12 x 12 m floor with one 1.2 x 0.8 m worktop whose top is at 0.75 m,
centred 1.5 m along +x -- the same MJCF on both engines.

Robot ids (`spawn.sh --help` prints them from the robot specification). Every robot is a
single body with one interface, served on `--port`:

| id | wire | mobile? | default evidence placement |
| --- | --- | --- | --- |
| `myagv` | ROS 1 Noetic | yes | floor |
| `so101` | ROS 2 Jazzy | arm | worktop |
| `ainex` | ROS 1 Noetic | yes (humanoid) | floor |
| `mycobot280` | ROS 2 Humble | arm | worktop |
| `rosmaster_x3_plus` | ROS 1 Noetic | yes | floor |

`spawn.sh` prints one readiness line when the wire serves its recorded interface -- every
recorded node, topic, service and action on its graph, its boot's one-shot steps done
and every recorded periodic output delivering samples -- e.g.

```
spawn ready: myagv (myAGV) in molmospaces ithor:1 on the floor; wire(s): myagv ws://127.0.0.1:9090 [ROS 1 noetic]
```

and stays in the foreground. `Ctrl-C` (SIGINT) or SIGTERM removes the robot and its
container and exits 0. Every refusal, failed startup, wire failure, or the simulation
ending exits non-zero. If `spawn.sh` is killed (even `SIGKILL`) the simulation removes the
robot and releases its id and placement; its container exits and removes itself.

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
| `shared/worktop_objects.py` | the six worktop objects (staged with the scene, or around an arm placed elsewhere while it is there) and the clearing of their area |
| `shared/robocasa_objects.py` | RoboCasa's own kitchen objects on the counters of a RoboCasa scene |
| `shared/objects/` | `ycb.sha256` and `LICENSES.md` of the worktop objects' YCB meshes; `ycb/` is fetched by `run.sh setup` (git-ignored) |
| `shared/tools/fetch_objects.py` | fetches and verifies `shared/objects/ycb/` from its pinned source |
| `shared/robot_model.py` | a registry robot as a MuJoCo spec |
| `shared/mjutil.py` | the MuJoCo helpers the modules share (heading, subtree, geom boxes, spec edits) |
| `shared/sensors.py` | render pool, camera/lidar/joint/IMU sampling |
| `shared/spawn.py` | `spawn.sh`: admission, Docker wires, readiness, watching, cleanup |
| `shared/registry.py` | the robot registry, parsed from the robot files `robots_specs/<id>.md` |
| `shared/protocol.py`, `shared/simport.py` | the control-port protocol and a CLI client |
| `shared/rosbridge_client.py` | stdlib rosbridge client (spawn health checks, tests) |
| `shared/wire/` | everything that runs inside the wire containers (mounted read-only) |
| `shared/wire/docker/` | wire images: `noetic`, `humble`, `jazzy` bases and per-robot layers |
| `shared/wire/robots/<id>.py` | each robot's wire plan and simulated drivers (one per robot) |
| `tests/` | unit, integration and end-to-end checks (see below) |

## Scenes

The household scenes' fixtures are loaded exactly as the prior version of this project
(the reference) loaded them, with no light and no robot added. `start` adds the six worktop
objects to every scene with a worktop (below), and a RoboCasa scene also carries RoboCasa's
own kitchen objects on its counters:

* **MolmoSpaces**: the pinned `allenai/molmospaces` checkout (`713fd12a`), assets at the
  versions it pins (iTHOR scenes `20251217_with_occupancy`, THOR objects `20251117`, ...),
  the house resolved and installed by the reference's `tools/resolve_scene.py`
  (`get_scenes` + `install_scene_with_objects_and_grasps_from_path`) and read with
  `mujoco.MjSpec.from_file`.
* **RoboCasa**: robosuite `5ce6643f` + robocasa `v1.0`; the kitchen built from
  `KitchenArena(layout, style, rng=default_rng(0))` with an empty robot list inside a
  robosuite `ManipulationTask` (multi-CCD on, sleeping islands off), its fixtures placed
  as RoboCasa's kitchen environment places them when it loads a kitchen
  (`scenes.place_fixtures`: the fixtures a layout sets on others -- a toaster, toaster
  oven, coffee machine, knife block, paper towel, plant -- sampled onto their counter with
  RoboCasa's own fixture samplers, from the arena's random stream; the reference skipped
  that step and left them at the world origin, inside the corner walls, or on the floor). Its asset files are
  the same downloads the reference extracted into its checkout; here they go to the
  generated `robocasa/assets/` tree and `robocasa.models.assets_root` points there.
  RoboCasa's collision hulls (group 0) are hidden, visuals (groups 1, 2) shown. Its bare
  kitchen holds fixtures only, so `shared/robocasa_objects.py` adds RoboCasa's own objects
  the way its task environments do: graspable objects drawn from its object library with
  its own sampler (the Lightwheel set `setup` installs, plus any registry `run.sh assets`
  fetched), each upright on a free spot of a counter top (`Counter.get_reset_regions()`),
  about eight per square metre of counter and at most 24, not touching one another -- the
  same draw for the same layout and style. They are loose objects like any other, and they
  stay where they are set (spec §2.1, amended 2026-10-03):
  * RoboCasa v1.0 pins MuJoCo 3.3.1, whose default solver takes every contact of the scene
    as one problem. RoboCasa's lightest objects (a 1.2 g straw and sugar cube, 3-7 g shrimp,
    marshmallows, cookie-dough balls) came off their first contacts with the counters at
    up to 5.6 m/s and knocked the others over: on `robocasa:3-5` 23 of 24 objects moved,
    six onto the floor, and on `robocasa:5-2` the solve diverged. The same scene XML is
    still under MuJoCo 3.3.6 and later (MolmoSpaces runs 3.5.0), which solve by constraint
    island by default, and unstable again there with islands disabled. So a RoboCasa scene
    is solved by island (`scenes.solve_by_island`: on 3.3.1 the island flag, which works
    with the CG solver). Timestep, cone, `impratio` and the objects' own figures are
    RoboCasa's.
  * An object that cannot stand still upright -- a dish brush or whisk set on its end tips
    over, a marshmallow keeps rolling -- is drawn again: `robocasa_objects.stands_still`
    sets each candidate alone on a flat top the way the counters get it, under the scene's
    solver, and keeps it only if it has tilted less than 10 degrees after 1 s and moves
    less than 2 mm in the 2 s after.

Scene equivalence with the reference is checked by
`tests/integration/test_reference_parity.py` against a checkout of the reference: the
MolmoSpaces scene `start` loads, with no robot, has the reference's bodies, geoms and
meshes, its six objects at the reference's poses and pixel-identical renders; a RoboCasa
scene has the reference kitchen's fixtures, its six objects at the reference's poses, and
RoboCasa's own objects besides.

The **worktop** is the surface the reference project's worktop survey ranks first
(`shared/worktop_survey.py`, a port of the reference's `scene_placement.py`,
`place_arm_on_table` and RoboCasa `find_counter_mount`): on iTHOR/ProcTHOR and `test`, the
table-height (0.35-1.30 m) top faces of the scene's bodies, those holding a plate or an
apple (and reachable loose objects) first, or with nothing graspable the largest; on
RoboCasa the roomiest counter region of `Counter.get_reset_regions`. `ithor:1`: the island
(`StandardIslandHeight`) at 1.100 m; `robocasa:1-1`: the counter run
`counter_main_main_group/geom_1` at 0.920 m; `test:1`: the block at 0.75 m.

## The worktop objects and the arms

The six worktop objects are part of the scene: `start` stages them on the worktop, around
the spot the SO-101 is placed at, before any robot is spawned, and they stay there through
every spawn and removal. A scene with no worktop has none.

**Where.** The survey ranks spots, the reference's own choice first; `shared/placement.py`
tries them in order and takes the first one that is supported, clear (the robot's own
collision geometry against the scene, after clearing, in a trial compile), camera-clear
and that leaves every object standing on the surface. The SO-101's spot frames the
scene's objects. An arm spawned on the worktop is first tried there, among them; both arms
fit there on the default scenes and then leave the objects untouched:

| Scene | so101 and mycobot280 |
| --- | --- |
| `ithor:1` | island (-0.382, 0.218), z 1.100, yaw 0 deg -- the reference's spot, exactly |
| `robocasa:1-1` | counter (2.231, -0.200), z 0.920, yaw -90 deg -- the reference's spot, exactly |

An arm that does not fit among them is placed at its own survey spot instead, and for as
long as it is there the six objects stand around it at the same poses in its base frame,
the loose objects in that area cleared as well.

**What.** `shared/worktop_objects.py` stages, in that spot's base frame (4 mm above the
top face), the reference's apple (a 20 mm sphere with its tuned contact, under a YCB mesh),
plate (fixed; a cylinder and a 24-box rim) and the bowl, mug, banana and lemon (meshes,
condim 6), at the reference's poses. The scene's loose objects within 0.55 m of the base
(0.15 m below to 0.45 m above it) are cleared: their free joint removed, parked 50 m
under the scene (on `ithor:1`: the apple, bread and book on the island; on a RoboCasa
scene, the RoboCasa objects in that area). Nothing else of the reference's task comes
along (no rig cameras, `/reset`, truth log or solver changes), and nothing reaches a wire.
The objects settle about 4 mm onto the surface after staging. The spawn's reply and the
control port's `robots` and `scene` answers name the objects an arm stands among and the
scene objects cleared for them.

**Removal.** Ending an arm's spawn removes the robot in one recompile. The six objects
stay where the scene staged them -- or, after an arm placed elsewhere, are back there --
and each object cleared for that arm returns with the pose and velocity it had when
cleared; every other robot and object keeps its state.

**Assets.** The YCB meshes are not committed: `run.sh setup` (either engine) runs
`shared/tools/fetch_objects.py`, which downloads them from elpis-lab/YCB_Dataset at
`9e8c6488a2ff673d9aa48a91492fb89423c1b106` (byte-identical to the reference's vendored
files) into `shared/objects/ycb/`, verifies them against `shared/objects/ycb.sha256` and
refuses on a mismatch. `python3 simulator/shared/tools/fetch_objects.py --check` verifies
only. On a scene with a worktop, `start` refuses (exit 2, before its ready line) when
these meshes or the SO-101's files are missing, with a message naming `run.sh setup`. If
no spot on the worktop fits the objects, `start` runs without them and logs `no worktop
objects staged`.

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
| `hello` | | `engine`, `scene`, `sim_time`, `floor_z`, `worktop` (`name`, `z`, `area_m2`, `centroid`, `bounds`) or null, `robots`, `starting`, `rtf` |
| `robots` | | `robots`: `[{id, state (pending/running), placement, xyz, yaw, ports, prefix, staged, cleared}]` -- `staged`: `{object: {body, pos, quat}}`, the worktop objects an arm stands among (empty for others), `cleared`: the scene bodies cleared for them |
| `scene` | | model counts (`nbody`, `ngeom`, `nmesh`, ...), `scene_nbody`, `free_bodies` (loose objects, staged ones included), `staging` (`{robot: {staged, cleared}}` for each arm on the worktop, and `scene`: the scene's own six objects as staged at start and what was cleared for them), `worktop`, `sim_time`, `rtf` |
| `readings` | `robot` | `joints`: `{name: {position, velocity, effort}}`; `base`: `{pos, quat (w,x,y,z), linvel_world, angvel_world, linvel_local, angvel_local}` of the robot's top body; `bodies`: `{name: {pos, quat}}` of every body of the robot; `sites`; `ctrl` (actuator targets); `stamp` (wall time), `sim_time` |
| `bodies` | `names: [..]` | world poses of any named bodies (scene objects included) |
| `rtf` | `since?` (epoch s) | `rtf` (the last window's), `window_s` (10), `warn_below` (0.9), `windows`: `[[wall start, wall end, rtf], ...]` of every completed window ending at or after `since` (the last hour) |
| `render` | `view`, `width`, `height`, `format` (`png` default, `rgb`, `depth`) | payload: PNG bytes, raw RGB (`height*width*3`), or float32 metres (`height*width`) |
| `describe` | `robot` | joints, actuators (`position`/`velocity`), cameras (name, fovy, resolution), sites |
| `spawn` | `robot`, `placement`, `ports` | `robot`, `token`, `prefix`, `xyz`, `yaw`, `surface_z`, `placement`, `staged`, `cleared` (as in `robots`); the connection holds the robot until `remove` or until it closes |
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

Each wire is a container (`rsim-<sim port>-<id>-main`, rosbridge listening on the same
port number inside as it is published on outside) built on first use from pinned
sources (base images by digest; every ROS package from a dated snapshots.ros.org snapshot
of the ROS repository, never the live one: noetic 2025-05-29, humble 2026-09-14, jazzy
2026-09-11; rosbridge_suite 0.11.17 on Noetic, 2.0.8 on Humble, 2.7.1 on Jazzy; the stock
packages the boots run also pinned by the releases their interface files cite, and on
Jazzy ros2_control 4.48.1 and geometry2's tf2/tf2_ros 0.36.23, which no snapshot carries,
built from their recorded revisions) and started with the
simulator code mounted read-only. Its ROS graph holds every non-optional node of the robot's interface file with
the recorded names, types, frames, parameters and rates, plus ROS infrastructure
(rosbridge_websocket, rosapi, rosout). Where the recorded boot runs vendor or stock code
that needs no hardware, that code runs unchanged; hardware is simulated underneath it:

| Robot | Real code on the wire | Simulated |
| --- | --- | --- |
| myAGV | joint_state_publisher, robot_state_publisher, tf static publishers, the vendor's modified robot_pose_ekf (built from myagv_ros `c71f3cc5`) | `/myagv_odometry_node` (myAGV.cpp's behaviour: 0.01 m/s MCU resolution, 0.1 deg yaw dead band, held command, no watchdog), `/ydlidar_lidar_publisher` (X2L: 400 samples, ignore sector reported 0), `/usb_cam` |
| SO-101 | robot_state_publisher, ros2_control_node with joint_trajectory_controller, parallel gripper controller, joint_state_broadcaster (spawned by the stock spawner) | the `feetech_ros2_driver/FeetechHardwareInterface` plugin (same name, simulated STS3215 bus: 2400 ticks/s, acceleration register 50, 4096-tick readback), `/usb_cam` (YUYV) |
| AiNex | the vendor's `ros_robot_controller_node.py` and `ainex_controller.py` with its gait engine (`walking_module.so`, `kinematics.so`), imu_calib, imu_complementary_filter, the nodelet manager with image_proc/rectify, web_video_server | the controller board (bus servos 0-1000 pulses over their move time, IMU in g and deg/s), `/camera`, `/joystick` (an idle receiver), `/sensor` (the user button, published while enabled), `/color_detection` and `/face_detect` (the undrawn camera frame on `<node>/image_result`; nothing detected), `/app` in its boot state `idle` (set_running true refused; false stops the gait), `/joystick_control` (its endpoints; the gamepad is idle) |
| myCobot 280 | robot_state_publisher (with `xacro` of the boot URDF) | `/slider_control_adaptive_gripper` (pymycobot's send_angles/set_gripper_value semantics, including ending on an invalid command) |
| ROSMASTER X3 PLUS | the vendor's `Mcnamu_X3plus.py` driver, `base_node`, `yahboom_joy.py` (built from yahboomcar_ws.zip), robot_state_publisher, imu_filter_madgwick, robot_localization EKF, joy_node, tf publishers | the expansion board (`Rosmaster_Lib`: the firmware's motion path -- vx/vy clamped to 0.7 m/s, mecanum mixing, 0.7 m/s wheel clamp, no yaw-hold, which the firmware applies only to commands the driver never sends -- and servo run times), `/ydlidar_lidar_publisher` (4ROS), `/camera/camera` (Astra Pro Plus colour, 16UC1 depth, IR, point clouds, TF; its camera services answer with the intrinsics its streams publish, and `toggle_<stream>` stops or restarts that stream) |

Robots move only through contact of their own wheels (mecanum rollers modelled in the
robot models), feet and grippers. `spawn.sh` watches its wire (container state and a
rosapi node check every 2 s) and ends the spawn, removing everything, when it fails. A ROS 1
node killed without unregistering stays on the master's list, so a ROS 1 wire also asks
each recorded node's XML-RPC server every 2 s and ends itself, reporting
`lost node(s) ...`, when one stops answering.

### Adaptations and estimates

Every figure the wires use that no pinned source fixes is the robot specification's,
labelled there as an estimate: each robot's `robots_specs/<id>/import.md`, section
"Estimates (not manufacturer data)" -- the myAGV's battery voltages and IMU units, the
myCobot 280's motion pacing, the SO-101's servo acceleration unit, the ROSMASTER's battery,
firmware version, magnetometer field and `orbbec_camera` definitions (built from
OrbbecSDK_ROS1 `v1.4.2`, `7ef885a4`, whose `msg/` and `srv/` files are the same git blobs as
`v1.2.9`, `9569cc31`, the release the vendor's driver sources equal), and the AiNex board's
IMU report rate -- and the AiNex's playable action groups, `robots_specs/ainex/action_groups.yml`
(the vendor's `.d6a` files are in no pinned source; `shared/wire/robots/ainex_actions.py`
writes the record's groups `wave`, `raise_hands` and `nod` where the vendor player reads
them, and a vendor name such as the record's example `left_shot` answers as the controller
answers an unknown name: 'can not find aciton file', then back to the init pose).

What the simulator itself adapts, in wire behaviour:

* ROSMASTER Astra Pro Plus: its IR image is the rendered luminance scaled to 10 bits (IR is
  not rendered). The driver's dynamic_reconfigure server is the vendor's own (built from
  its cfg).
* AiNex `/color_detection` and `/face_detect`: their LAB thresholds and models are in no
  pinned source, so between `<node>/enter` and `<node>/exit` each publishes the undrawn
  camera frame on `<node>/image_result` (rgb8, `header.frame_id` the node's name, stamped
  when published, as the vendor's `cv2_image2ros` does) and never publishes
  `/object/pixel_coords`.
* AiNex `/app` stays in its boot state `idle`: `/app/set_running` true is refused; false
  stops the gait (`/walking/command` `stop`) and answers success false.
* AiNex board: it has the SDK methods the vendor node calls, under the SDK's own names; the
  node's `bus_servo/get_state` voltage and torque requests call methods the pinned SDK lacks
  and fail as they do on the robot.

## Timing

Physics runs against the wall clock on its own thread; periodic streams are sampled by
elapsed wall time and stamped with their acquisition time. When the real-time factor over
a 10 s window falls below 0.90 the simulation prints a warning on stderr; the control
port's `rtf` answer lists every window, and the checks claim no rate or physical bound
for an interval a window below 0.90 overlaps (they fail, saying so). Camera streams
render on their own GL contexts; lidar rays are cast on a mirror of the world, off the
physics thread.

## Verification

```sh
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/unit simulator/tests/integration
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/e2e      # Docker, minutes
# reference parity of the scenes, the worktop spot and objects (part of the integration
# suite): the reference is extracted from this repository's history (34547ae), or taken
# from RSIM_REF_DIR=<checkout> when set
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/integration/test_reference_parity.py
```

See `tests/README.md` for what each suite checks.
