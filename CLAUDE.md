# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this workspace is

Two **independent** projects that talk over network protocols, never Python imports:

| Project | Responsibility |
|---|---|
| `simulator/` | Simulate robots with a choice of **engine**, exposing generic network servers for observations and actuator targets. |
| `robot_console/` | Drive, map and navigate compatible simulated or physical robots; all control policy lives here. |

The separation is a hard constraint, not a preference: **`robot_console` must stay
installable and runnable on a machine with no MuJoCo, no MolmoSpaces, and no
`simulator/` checkout.** Its base runtime dependencies are `numpy`, `opencv-python`, and
`roslibpy`; arm/Inspect integrations are optional extras and still must not depend on
MuJoCo or simulator code. Never import from `simulator/` in console code or its tests — the console's
test double for the bridge (`robot_console/tests/fake_bridge.py`) is deliberately a
separate reimplementation for this reason. `.venv/bin/python -c "import mujoco"` failing
inside `robot_console` is a feature worth preserving.

Each project has its own `uv` venv and its own launcher script. There is no top-level
build.

### The multi-engine split (the central invariant)

`simulator/` is not one simulator but **two interchangeable engines plus a shared
layer**:

```
simulator/
  shared/       cross-engine resources: the wire bridge and the robot specs
    contracts/    rosbridge_server.py — pure transport, no MuJoCo: ROS 1 builders for the
                  myAGV's vendor topics, ROS 2 builders for the SO-101's, and a small
                  rosapi shim so browser clients can discover topics
    ros_surfaces/ per-robot topic sets + the loop that feeds them (myagv.py, so101.py) —
                  one copy, every engine
    tasks/        what an engine can stage into a scene: the objects, the cameras and the
                  definition of done (apple_on_plate.py) — engine-neutral
    robots/       so101/ + myagv/ hardware specs (MJCF, meshes, URDF) — engine-neutral
    mujoco_bridge.py  MuJoCo→wire helpers shared by the MuJoCo engines (imports mujoco)
  molmospaces/  engine #1 — MuJoCo + MolmoSpaces (iTHOR / procthor houses)
  robocasa/     engine #2 — MuJoCo + robosuite + RoboCasa (kitchens)
  kitchen_arm.sh  the SO-101 in both engines at once, around the same objects
```

Each engine has its own `run.sh`, `env.sh`, `tools/spawn_robot.py`, and `uv` venv, and each
must spawn at least `so101` and `myagv`. **`robot_console` connects and controls a robot
identically regardless of which engine hosts it — because every engine presents the *real
hardware's* interface** by feeding the one shared bridge in `simulator/shared/contracts/`:
the myAGV on its vendor `elephantrobotics/myagv_ros` rosbridge topics, the SO-101 on the
ros2_control topic set a real bringup for that arm presents. An engine change that makes the console
able to tell the engines apart is a regression. Per-engine specifics live in each engine's
`README.md`; the sections below cover MolmoSpaces (the reference engine) and then the ROS
contract every engine obeys.

"Cannot tell them apart" is a testable claim, not an aspiration, and it is easy to break
by accident: the SO-101 in MolmoSpaces exposes a `base` move group — the unactuated mocap
mount it is bolted to, zero controls — and a `base_pose` observation, so the RoboCasa
engine advertises both too even though it bolts the arm straight into the worldbody. The
same goes for the depth stream, which nothing consumes: both engines publish it, because
one that did not could be identified by its topic list. To check, run the same client
against both (`kitchen_arm.sh serve`) and compare the metadata and the observation keys.

A **ROS surface** — the topic set one robot's vendor stack presents, plus the loop that
feeds it — belongs to the robot, not to an engine, and so lives in `shared/ros_surfaces/`.
Each engine's adapter is only the part that knows how *that* engine names a base: the
MolmoSpaces one pulls a move group out of a `RobotView`, and the RoboCasa one builds
`mujoco_bridge.PlanarJointBase` from raw joints. Two copies of the loop itself would be
two chances for the engines to drift.

---

## simulator/molmospaces/  (the reference engine)

Everything goes through `./run.sh` (from inside `simulator/molmospaces/`); it sources
`env.sh` for asset paths and `MUJOCO_GL`. The other engine (`robocasa/`)
mirrors this launcher surface — see its README.

```bash
cd simulator/molmospaces
./run.sh setup                      # venv + install molmospaces[mujoco] + default assets
./run.sh assets ithor               # pre-fetch iTHOR houses/objects/grasps (~13 GB)
./run.sh view --scene ithor:1       # a house in the viewer
./run.sh view --robot myagv --scene ithor:1 --ros-port 9090   # + robot, as a ROS robot
./run.sh view --robot so101 --ros-port 9090            # the arm on its ROS topics
./run.sh view --robot so101 --scene ithor:1 --target bowl,apple  # ...facing those objects
./run.sh shell                      # interactive shell in the venv
./run.sh help
```

`./run.sh assets ithor` is optional for the arm task: `tools/resolve_scene.py` installs
a house on demand, so `--scene ithor:1` works from a bare `setup` and the 13 GB pull is
only worth it for offline use or a sweep across many houses.

`--target CAT,CAT` ranks the scene's grasp targets by category, which is how an arm is
set up around *specific* objects rather than whatever `find_grasp_targets` liked best. It
ranks rather than filters: a surface holding one of the two is still better than one
holding neither, and filtering would turn a near miss into an error.

`--ros-port PORT` puts a robot on its own vendor topics; `--task NAME` additionally
stages a task's objects, cameras and success predicate into the scene (see
`shared/tasks/`), and `--wrist-camera` adds the eye-in-hand stream. Cameras are rendered
inside the physics loop, so each one costs control rate for every client — which is why
the wrist view is opt-in rather than always on.

Robot self-tests are standalone scripts, not pytest — a failure points at the robot
definition:

```bash
python robots/myagv/test_attach.py [--scene /path/to/house.xml]
python tools/render_robots.py --outdir /tmp/robots   # render/load test for every robot
```

### One bridge, two robots, two ROS dialects — do not conflate the dialects

`shared/contracts/rosbridge_server.py` is the only transport: plain rosbridge JSON over a
websocket, served in-process, wired up by `serve_ros()` in each engine's
`tools/spawn_robot.py`. Every robot is on it, and `robot_console` speaks nothing else.

What differs per robot is the **dialect**, because each one mirrors a different piece of
real hardware:

- the **myAGV** is a ROS 1 stack, so it gets single-slash type strings
  (`geometry_msgs/Twist`) and a `secs`/`nsecs` stamp;
- the **SO-101** is a ROS 2 ros2_control bringup, so it gets `pkg/msg/Type` and a
  `sec`/`nanosec` stamp.

Both sets of message builders live side by side in that file, clearly labelled. A client
that had to guess which dialect a stamp was in would be a worse thing than a little
duplication.

**The SO-101 used to speak a bespoke msgpack protocol (`molmospaces-control-v1`) on its
own port, and no longer does.** The rationale for that choice was that a real SO-101 does
not speak ROS — true of the servo bus, but not of how anyone deploys one: the arm ships
with a `ros2_so_arm` description and is driven through `joint_trajectory_controller` and
a `forward_command_controller` on the jaw. Presenting *that* is what makes the simulated
arm and a real one interchangeable, and it is what lets the `inspect-robots` ROS
embodiment drive either without a line of adapter code. `control_server.py`,
`arm_client.py`, `so101_driver.py`, `inspect_so101.py` and `arm_task.py` are gone rather
than deprecated; two transports for one robot is two things to keep in step.

Both MuJoCo engines additionally share `simulator/shared/mujoco_bridge.py` (holonomic
`cmd_vel` integration, ray-cast `/scan`, camera encode) — it imports `mujoco`.

### Out-of-tree robots

Registered in the `ROBOTS` dict at the top of `molmospaces/tools/spawn_robot.py` (name →
module, config class, robot class; imported lazily). The shared MJCF and meshes live in
`simulator/shared/robots/<name>/`; the MolmoSpaces-coupled `RobotView`/`Robot`/`BaseRobotConfig`
trio and `make_model.py` live in `molmospaces/robots/<name>/` as thin adapters over that
shared spec, alongside a `test_attach.py`. `BaseRobotConfig.robot_dir` accepts an external
directory, so none of this requires forking the upstream clone — **never modify
`molmospaces/upstream/`** (the vendored allenai/molmospaces checkout).

Three sets in `spawn_robot.py` drive placement and are the thing to check when a robot
spawns wrong: `HOLONOMIC_BASE_ROBOTS`, `TABLETOP_ROBOTS`, `ARM_REACH`.

`assets/` is a generated MolmoSpaces symlink tree that gets force-refreshed — curated
files belong in `robots/<name>/`, never there. See `robots/README.md` and
`robots/URDF.md`.

**Pass scenes by their `assets/` path, not their realpath.** Scene MJCFs reference meshes
relatively (`../../objects/thor/...`), which resolves through the symlink tree; handing
MuJoCo the resolved `data/mujoco/scenes/...` path makes those lookups land in a directory
that has no `objects/` and the compile dies on a missing `.obj`. `run.sh view --scene`
goes through `tools/resolve_scene.py`, which gets this right — the trap is only there when
calling `tools/spawn_robot.py` directly.

Multi-room houses live in `assets/scenes/procthor-10k-*`; `assets/scenes/ithor` holds
single rooms (FloorPlan1-30 kitchens, 201+ living rooms, 301+ bedrooms, 401+ bathrooms).

Only out-of-tree robots load with `view`; the MolmoSpaces built-ins (`franka`, `droid`,
`rum`, `rby1`, `yam`, `bimanual_yam`) live upstream and are not wired into the launcher.

### macOS constraints (these explain otherwise-baffling code)

- The venv **must** be a Homebrew framework Python 3.11, not uv's standalone CPython:
  `mjpython` needs a shared `libpython3.11.dylib`. The MuJoCo passive viewer must own
  the main thread, which is what `mjpython` provides — `run.sh` routes viewer commands
  through it and everything else through plain `python`.
- `mjpython -m mujoco.viewer` **does not work**; the viewer must be launched from a
  script (`tools/view_scene.py`).
- `env.sh`/`run.sh` avoid `$(cd ... && pwd)` on purpose: a shell with a `chpwd`/`precmd`
  hook writing a terminal-title escape would get that escape captured into the path.
  `bin/teleop.sh` copies the same idiom.

---

## simulator/robocasa/  (engine #2)

Mirrors the reference engine's launcher surface. `view` grows a `--robot` that is
deliberately overloaded: `myagv`/`so101` go to `tools/spawn_robot.py` and get the vendor
wire contracts, anything else is a robosuite robot name for `tools/view_kitchen.py`.

```bash
./run.sh view --robot myagv --ros-port 9090            # myAGV on its vendor ROS topics
./run.sh view --robot so101 --control 127.0.0.1:8000   # SO-101 control protocol
./run.sh view --robot so101 --objects bowl,apple       # ...with objects inside its reach
./run.sh view --robot myagv --headless --ros-port 9090 # displayless; what checks run
```

**RoboCasa is a scene provider here, not a robot stack.** The kitchen is built from
`KitchenArena` with `mujoco_robots=[]` — 44 fixtures, 825 geoms, *zero* actuators — and
the shared robot MJCF is grafted into that spec and stepped by plain MuJoCo. Going
through `robosuite.make` instead drags in a robosuite robot with its own controller
stack, action space and observation dict, which would then have to be cut back out of
the compiled model, and would stand a Panda in the middle of every camera frame and
every map. The robots here are not robosuite robots and must not become them: the myAGV
is a vendor ROS device and the SO-101 speaks the ros2_control topic set.

Three traps, all of which produce results that look like bugs somewhere else:

- **Geom groups are inverted from the MolmoSpaces convention.** RoboCasa collision hulls
  are group 0 — 501 of them in layout 1, painted in random translucent colours — and the
  visual meshes are group 1. Anything that renders must go through `visual_only()`, or
  the camera streams a kitchen full of red and green boxes. The shared robot MJCFs use
  the opposite convention (2 visual, 3 collision), so the mask has to pass both.
- **Worktops come from RoboCasa, never from geometry.** `Counter.get_reset_regions()`
  returns the free rectangles the dataset itself places objects on. Inferring them from
  collision AABBs fails in one specific, repeatable way: a sink basin's floor is "a flat
  surface at counter height" whose centre is a clean 0.22 m clear of anything, so it
  outscores every real worktop and the arm gets mounted in the sink.
- **Clearance is measured to geom surfaces, not centres** (`world_boxes`). A kitchen is
  four long wall boxes; the centre of a 5 m wall is metres from a robot pressed against
  it, so a centre-distance search parks the robot inside the wall.

A RoboCasa kitchen contains no loose objects at all — everything is a fixture — so an arm
has nothing to reach for until `--objects` spawns some from RoboCasa's own registry. That
is the one real asymmetry with MolmoSpaces, where iTHOR houses come with graspables and
their metadata, and it is why **arm-with-objects work belongs in MolmoSpaces**:
`tools/scene_placement.py` there finds a surface that already has graspable objects on it
and mounts the arm so they land in its working annulus. `simulator/kitchen_arm.sh` sets
both engines up around the same object pair for comparison.

---

## robot_console/

The arm task, from `simulator/`:

```bash
./kitchen_arm.sh inspect                       # sim + inspect-robot, reports PASS/FAIL
./kitchen_arm.sh inspect --policy molmoact2    # the VLA instead of the scripted plan
./kitchen_arm.sh inspect --episodes 8          # a pass count, which is the only useful unit
./kitchen_arm.sh serve                         # both engines on rosbridge, drive them yourself
./kitchen_arm.sh cameras                       # the live camera page against a running serve
```

`inspect` is the front door and it does four things in order: stage the task into the
engine's kitchen, wait for the *topics* (not the port — a listening socket says nothing
about whether the scene compiled), reset and **verify** the world, then run the episode
and report a verdict. Each of those exists because its absence produced a confusing
failure; the comments in the script say which.

**Report pass counts, never a single run.** The scripted policy has passed every attempt
so far, but a VLA on this task is a coin toss on the reference rig, and one episode of it
tells you nothing.

The rest of the console, from `robot_console/`:

```bash
./bin/teleop.sh                         # first run creates .venv and installs
./bin/teleop.sh --host 192.168.1.42     # a real myAGV
./bin/teleop.sh --record runs/drive1    # feed.mp4 + commands.jsonl
./bin/teleop.sh --no-preflight          # skip the reachability check
./bin/teleop.sh --reinstall

./bin/slam.sh explore  --out runs/house # autonomous frontier exploration
./bin/slam.sh map      --out runs/house # teleop with the map building live
./bin/slam.sh navigate --map runs/house # click a point, drive there

uv pip install -e '.[dev]'
.venv/bin/python -m pytest                       # offline; no robot, no display
.venv/bin/python -m pytest tests/test_teleop.py::test_motion_expires_when_the_key_stops_repeating
.venv/bin/python -m pytest -m live               # opt into anything needing port 9090
.venv/bin/python -m robot_console.smoke          # live check; DRIVES the robot ~0.3 m
```

`bin/teleop.sh` reinstalls itself when `pyproject.toml` is newer than the venv stamp.
The default pytest run is `-m 'not live'`; no test may call `cv2.imshow`/`namedWindow`.

### Structure

Behaviour lives in pure, directly testable modules; `app.py` is wiring:

- `topics.py` — topic names and ROS type strings, the contract in one place
- `teleop.py` — keymap, hold/release state machine, speed model (stdlib only)
- `camera.py` — CompressedImage decode + `LatestFrame`, the thread hand-off
- `bridge.py` — `RobotLink` (roslibpy) and pure `parse_odom`
- `hud.py`, `recorder.py`, `preflight.py`, `cli.py`, `smoke.py`
- `arm/` — the SO-101 task, and the only part of the console that drives an arm. It
  registers itself with the `inspect-robots` framework through the
  `inspect_robots.{tasks,policies,embodiments,scorers}` entry points in
  `pyproject.toml`, which is what makes `inspect-robot run --task apple_on_plate
  --policy molmoact2 --embodiment so101_ros -E url=ws://…` work with no console script
  of its own. Inside: `ros_client.py` (the header-stamping shim, below), `ros_settings.py`
  (every topic name and camera size in one place), `kinematics.py` (MuJoCo-free FK/IK),
  `waypoints.py`/`policy.py` (the scripted plan), `molmoact.py` (the MolmoAct2-SO100_101
  VLA), `task.py`/`success.py`/`scorer.py`, `embodiment.py`, and `preflight.py`
  (reset-and-verify). Torch lives behind a function-local import so the offline suite
  stays torch-free; `bin/` has no arm launcher any more because `kitchen_arm.sh inspect`
  is the front door.

- `slam/` — occupancy-grid SLAM on `/scan`, the same sensor `myagv_slam_laser.launch`
  uses. `scan.py` (message → base-frame points), `grid.py` (log-odds map), `mapio.py`
  (`map_server` pgm/yaml + npz sidecar), `matcher.py`/`pose.py` (correlative scan
  matching, keyframed), `planner.py` (inflate + A*), `frontier.py`, `controller.py`
  (path → holonomic `Command`), `explorer.py` (the give-up ladder and goal commitment),
  `mapview.py`, `app.py`, `cli.py`. Everything but the two `app.py`/`mapview.py` files is
  pure and tested offline — including a whole exploration run, via `tests/simworld.py`.
- `explore.py`, `mapping.py`, `navigate.py` — thin entry points over `slam/cli.py`

### SLAM invariants

- **Scan matching is keyframed, not per-scan.** It runs after 0.15 m or 10° of motion,
  capped by `--slam-hz`, because it shares the thread with the 20 Hz `/cmd_vel` stream.
  `slam/app.py::_Budget` measures tick time against the publish period and warns on
  overrun — a robot that drives fine and maps badly leaves no other trace.
- The map is **grown, never shifted**: `OccupancyGrid.grow_to_include` only pads outward
  and moves `origin` to match, so a pose already computed against the map stays valid.
- Maps are saved in `map_server` format because that is what `navigation_active.launch`
  loads. The `.npz` sidecar is what makes a reloaded map *continuable*; the yaml is the
  authority on geometry, and a sidecar that disagrees is discarded.
- `navigate` plans with unknown space **blocked**; `explore` plans with it **free**. The
  same costmap for both would either forbid exploring or route a navigation run through
  a region no sensor has seen.
- Invalid laser returns arrive as `0.0` (real driver), `inf` (stock driver) or
  `range_max + 1` (simulator). Always test `range_min <= r <= range_max`.
- **The obstacle brake looks along the direction of travel, not straight ahead.** The
  base is Mecanum; checking `+x` while strafing brakes for things it is moving away from
  and wedges the robot anywhere something happens to sit in front of it. `nearest_obstacle`
  takes a `bearing` for this reason.
- A `blocked` result reroutes; only `is_stuck` blacklists the goal. Blacklisting on a
  local obstruction burns through every frontier in the house in seconds while the robot
  stands still.
- **Frontiers are ranked geodesically, off one `planner.distance_field`.** Straight-line
  distance puts a frontier a metre away through a wall ahead of one three metres down an
  open corridor. One wavefront scores every cluster at once, so there is no shortlist to
  cap — and a capped shortlist is what used to make "the map is finished" and "my best
  twelve guesses all failed" the same answer.
- **The goal is a cell near the cluster, never the centroid.** A cluster that wraps a
  corner has its centroid inside the wall it wraps, so the biggest frontiers were the
  likeliest to be discarded as unreachable. It is picked as the cheapest reachable cell
  in a standoff-wide collar, which also stops the base driving its own centre onto the
  boundary and into its own obstacle brake.
- **The progress watchdog must be armed before any early return in `PathFollower.step`.**
  The obstacle brake returns early; when it did so before arming, `is_stuck` answered
  `False` forever and a braked robot re-routed to the same goal every tick, never
  blacklisting it and never finishing. That is a hang, not an inefficiency.
- **Running out of frontiers is not the same as being finished.** `slam/explorer.py`
  walks a ladder — relax `min_cells`, flush the suppression list once, enclosed sensor
  holes, one sweep — and only then says `explored`. Each rung re-scores the *same*
  wavefront, which is what makes trying again affordable.
- Suppression decays and counts strikes; the radius is 0.25 m because a doorway is about
  0.8 m and a wider one sealed a room's only entrance from a single bad approach.
- **Frontier detection filters unknown regions thinner than a cell or two.** A mapped
  wall is a dashed line — grazing beams skip cells — so the slivers between the dashes
  look like frontiers and can never be resolved. Left in, they are what a run spends its
  endgame driving at. Same reason `unknown_pockets` ignores anything touching a wall.

### Invariants worth knowing before editing

- **One loop, on the main thread.** `cv2.imshow`/`waitKey` must own the main thread on
  macOS. A separate publisher thread would need a lock on `TeleopState` and would keep
  the robot driving while the UI was wedged; with one loop a UI stall stops feeding the
  command stream, so a freeze degrades into a stop. Do not add a publisher thread.
- The roslibpy callback stores the raw dict and a timestamp only — no base64, no
  `imdecode`. Blocking that reactor thread stalls `/odom` and `/cmd_vel` too.
- **roslibpy's Twisted reactor is process-global and single-shot.** It cannot be
  restarted after `terminate()`. Hence: preflight is a plain TCP probe, not a rosbridge
  connect; and every roslibpy-touching test lives in `test_link_roundtrip.py` behind one
  session-scoped fixture that never calls `terminate()`.
- The recorder writes the **raw** camera frame; `hud.draw_overlay` returns a copy and
  must never mutate its input.
- `decode_compressed_image` returns `None` on any corruption rather than raising — the
  loop that decodes is also the loop keeping the robot's command stream alive.

---

## The ROS contracts (both sides must agree)

### The myAGV — a ROS 1 mobile base

| Direction | Topic | Type | Fields used |
|---|---|---|---|
| console → robot | `/cmd_vel` | `geometry_msgs/Twist` | `linear.x`, `linear.y`, `angular.z` |
| robot → console | `/odom` | `nav_msgs/Odometry` | pose, twist; `odom` → `base_footprint` |
| robot → console | `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | base64 JPEG |
| robot → console | `/scan` | `sensor_msgs/LaserScan` | `ranges`, angles, range limits |

ROS1 single-slash type strings. Body frame: `+x` forward, `+y` left, `+z` CCW. The base
is holonomic (the myAGV is Mecanum), so `linear.y` is a real strafe.

`/scan` follows the **YDLidar X2** — `ydlidar_ros_driver/launch/X2.launch`: frame
`laser_frame`, 0.1–12.0 m, 10 Hz, CCW from `-pi`, mounted at
`base_footprint + (0.065, 0, 0.08)` per `myagv_active.launch`'s static transform. Both
sides encode that mount offset; 65 mm is more than a map cell at 5 cm, and dropping it
smears every wall by a cell. The driver's `inverted: true` and the transform's roll of
π cancel, so no sign flip is needed anywhere — changing one without the other is the
easy mistake. `ranges` is a plain JSON float array; only `uint8[]` is base64.

Bridge quirks that clients must not rely on: no status handshake on connect, `id` fields
ignored and never echoed, no loopback of published topics, `advertise`/`unadvertise` are
no-ops. `data` on CompressedImage is a base64 **string**, not an int array. Match
`format` on containing `jpeg` — the simulator sends `"jpeg"`, real `image_transport`
sends `"rgb8; jpeg compressed bgr8"`.

**Watchdogs differ between sim and hardware.** The simulator's bridge stops the base
0.5 s after commands stop, so the console publishes at 20 Hz. The **real myAGV has no
watchdog at all** — `myagv_odometry_node` stores the last Twist in a global and writes
it to the motors at 100 Hz forever. That is why the console publishes a zero Twist on
every exit path (Esc, window close, exception, `SIGINT`/`SIGTERM`), and why that must
not be weakened to best-effort.

Vendor reference, when changing anything on either side:
[`elephantrobotics/myagv_ros`](https://github.com/elephantrobotics/myagv_ros), branch
`myagv_ros_2023Pi` — `myagv_odometry/src/myAGV.cpp` (odom fields and covariances),
`myagv_teleop/scripts/myagv_teleop.py` (speed 0.25 m/s, turn 0.5 rad/s, 0.52 s key
timeout — the source of the console's speed cap and turn ratio). Simulator changes that
make it *less* like the real robot are regressions even when nothing fails.

### The SO-101 — a ROS 2 ros2_control arm

| Direction | Topic / service | Type |
|---|---|---|
| console → robot | `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` |
| console → robot | `/gripper_controller/commands` | `std_msgs/msg/Float64MultiArray` |
| robot → console | `/joint_states` | `sensor_msgs/msg/JointState` |
| robot → console | `/free_joint_publisher/free_joint_states` | `mujoco_ros2_control_msgs/msg/FreeJointStateArray` |
| robot → console | `/task_success` | `std_msgs/msg/Bool` |
| robot → console | `/overhead/color/compressed`, `/side/color/compressed`, `/wrist/color/compressed` | `sensor_msgs/msg/CompressedImage` |
| console → robot | `/reset`, `/mujoco_ros2_control_node/reset_world` | Trigger-shaped `{success, message}` |

ROS 2 `pkg/msg/Type` strings and a `sec`/`nanosec` stamp. Joint order is fixed everywhere
(`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, then the
gripper), all `_joint`-suffixed, all radians except the gripper, which is 0 (closed) to
1 (open).

Five things that are silent when wrong, and each cost a debugging session:

- **`/joint_states` comes back alphabetically sorted**, which for this arm shares *no*
  index with the contract order — `elbow_flex_joint` first, `shoulder_pan_joint` fourth.
  Index by name, never by position. The simulator sorts deliberately, so the hazard the
  real broadcaster presents is exercised here rather than discovered on hardware.
- **A `JointTrajectory` with no `header` is accepted and ignored** by a real
  `joint_trajectory_controller`: it holds its pose and reports zero error. The gripper is
  a `ForwardCommandController` and needs no header, so the jaw keeps working and the
  episode looks alive while the arm never moves. `arm/ros_client.py` exists only for this.
- **The gripper must be a topic, not an action.** The client's ROS adapter has no action
  client at all, so a `GripperActionController` — which the stock SO-ARM controller config
  declares — is simply undrivable.
- **Stamps are simulated time**, not the wall clock. The success predicate holds for
  ≥ 1.0 s *of simulated time*, the client refuses to start if simulated time is not
  advancing against the wall clock, and the offline scorer re-derives the hold from these
  stamps. All three read the same number.
- **The MJCF jaw hinge is not the contract gripper.** The model runs −0.174533…1.745329
  rad and the contract runs 0…1; the map is an exact *offset*, verified by measuring tip
  separation against the curve the grasp tuning was fitted on (agreement to ~0.1 mm).
  Rescaling instead would move the aperture at every value.

Poses on `free_joint_states` are in the **arm base frame** with the work surface at
z = 0 — not the engine's world frame. That is what lets one client read the same numbers
whether the arm is bolted to a kitchen island or standing on a bare table.

---

## The arm task, and what is actually known about it

`shared/tasks/apple_on_plate.py` is grafted onto whichever kitchen an engine compiled: it
brings a 20 mm apple, a white plate, two scene cameras and its own arbiter, all placed
relative to wherever the arm got mounted. The engines still supply the room — a task that
replaced the scene would make them bystanders.

**Success is computed three independent ways and disagreement is the signal.** The
simulator's own `/task_success`, a live geometric verdict in `arm/success.py`, and an
offline re-derivation from the recorded log in `arm/scorer.py`. Never make one read
another's answer; `kitchen_arm.sh` prints a warning when the first two disagree, which
they legitimately can at the margin — the episode terminates the instant its own hold
passes 1.0 s and the simulator's hold starts a beat later.

Constants that look arbitrary and are not. Each was measured, and each was wrong first:

- **`condim="6"` on the apple.** Rolling friction is the *third* entry of `friction` and
  only exists at condim 6; below that the declared value is parsed and discarded, so a
  fruit nudged at 7 mm/s coasts for half a minute and episodes end with it on the floor.
- **The plate's rim is 24 boxes**, because MuJoCo has no torus. Without it apples
  delivered to within a millimetre of the centre roll straight off, and the "at rest"
  clause never fires on a placement that looked perfect.
- **`grasp_gripper = 0.40`, not 0.50.** The jaw is force-limited, so it only squeezes
  while *stalled* short of its target. A 40 mm apple blocks anything under ~0.52, so 0.50
  reaches its target and holds the apple with nothing but compliance — it creeps out
  during the carry. Sweeping the whole plan offline, 0.35-0.45 place the apple and 0.50
  drops it in transit.
- **`JAW_CENTER_OFFSET = 0.002`,** found by sweeping it through an offline pick rather
  than derived from geometry: a plausible geometric proxy (the jaw-tip midpoint) is 15 mm
  out, and the working window is only 4 mm wide.
- **Arrival ignores the jaw except when the jaw is what is moving.** Counting the jaw's
  error makes arrival impossible for every carrying waypoint, because a jaw holding
  something deliberately stalls short; ignoring it makes `close` a step count, and the
  jaw is far slower than the arm. So the jaw counts as arrived when it reaches its target
  *or* stops moving — a stalled jaw is either shut or pressing on something.

**Report pass counts, never a single run.** The scripted policy has passed every attempt;
a VLA on this task is a coin toss even on the rig it was tuned for, and one episode of
`--policy molmoact2` tells you nothing. `--episodes N` exists for this.

---

## Workspace conventions

- `specs.json` describes the current, verified state and **is tracked**.
  `target_specs.json` and `implementation_plan.json` are per-run working documents and
  are gitignored. Both validate against `spec.schema.json` / `plan.schema.json`.
  `goal_prompt.md` describes that spec-driven workflow.
- Comments in this codebase explain *why*, especially where a choice looks wrong without
  the constraint behind it (macOS threading, the reactor, key auto-repeat, codec
  choices). Match that when adding code; do not narrate what the line already says.
