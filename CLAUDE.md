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
  kitchen.sh  the SO-101 in both engines at once, around the same objects
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
against both (`kitchen.sh serve`) and compare the metadata and the observation keys.

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
python tools/test_placement.py                       # where a tabletop arm gets bolted
```

### One bridge, N robots, one namespace each

`shared/contracts/rosbridge_server.py` is the only transport: plain rosbridge JSON over a
websocket, served in-process, wired up by `RobotFleet` in each engine's
`tools/spawn_robot.py`. Every robot is on it, and `robot_console` speaks nothing else.

**Several robots share one server, one port and one graph.** That is ROS's own answer,
not a shortcut: `ROS_NAMESPACE=robot1` / `<group ns="robot1">` in ROS 1, `-r __ns:=/robot1`
in ROS 2, with the matching prefix on every `frame_id`. rosbridge sits above that -- one
websocket exposes the whole graph -- so a fleet is one port whose topic list reads
`/so101/joint_states` beside `/myagv/cmd_vel`, and a client picks its robot by prefix.
Two bridges on two ports would be two graphs: the robots could not see each other, and
nothing could discover the fleet or drive it over one connection.

    ./kitchen.sh serve --robots so101,myagv       # both, on ws://127.0.0.1:9090
    ../robot_console/run_task.sh --robots so101,myagv   # ...and grade the arm task in it

Three rules, in `shared/contracts/namespace.py`, each of them a test:

- **Topics get a leading slash, frames do not.** `/myagv/cmd_vel` is an absolute graph
  path; `myagv/base_footprint` is a tf node joined by `tf_prefix`. Composing a frame the
  way a topic is composed gives `/myagv/odom`, which no real stack emits and nothing will
  connect a tf tree to.
- **An empty namespace is the identity**, so the bare single-robot contract -- the tables
  further down, the thing this simulator claims to be indistinguishable from -- stays
  expressible and therefore testable. `--ros-namespace ''` on either engine, and
  `namespace=""` on `RosSettings`, reproduce it exactly.
- **Composition is idempotent**, which is what lets a client pass a namespace *and* name a
  topic explicitly without being prefixed twice.

The namespace defaults to the robot's own name and is **on even for one robot**, so a
lone SO-101 is on `/so101/*`. The topic constants in `ros_surfaces/` and in the console
stay bare regardless: they are the record of what each vendor's stack presents, and the
prefix is applied where a name reaches the wire (`NamespacedBus` on one side,
`RosSettings.topic()` on the other). Prefixing them in place would make the record
disagree with itself.

Two things that were silent before and now are not:

- **`RosBridgeServer.on` refuses a second handler for a topic.** It used to overwrite,
  so an unnamespaced second robot took the first one's `/cmd_vel` and both published
  `/odom` onto one topic -- the first robot simply stopped responding, which reads as a
  physics fault and is a naming one.
- **`serve_rosapi()` is called once per server, by the fleet.** It used to be called from
  inside the SO-101's surface, which meant a myAGV-only run had no topic discovery at all.

**The MJCF prefix and the ROS namespace are different things.** `robot_0/` prefixes
bodies, joints, actuators and cameras inside one compiled MuJoCo model; `/so101` prefixes
topics and services on the wire. Conflating them would put an engine's model layout onto
the wire, where a client could see it -- which is also why `CameraStreams` derives a
camera's `frame_id` from its topic and not from the MJCF camera name, after the wrist view
was found shipping `robot_0/wrist` as its frame.

**Two robots in one engine is not the `--engine both` that was removed.** That was two
*engines*, two ports and two kitchens sharing one GPU. This is one engine, one port, one
scene -- and the cost is real but different: see the rate note below.

**Rate is the cost of a fleet, and it is the camera.** Cameras render inside the physics
loop. Measured on an iTHOR kitchen at `--control-hz 10`: the SO-101 alone publishes at
9.8 Hz; adding a myAGV takes it to **5.7 Hz**; disabling only the AGV's colour camera puts
it back to 8.4 Hz, while dropping the AGV's lidar entirely is worth just 1.0 Hz, and depth
0.3 Hz. So the render is the cost and the lidar is not, which is the opposite of the
intuition. It matters: at 5.7 Hz the scripted policy still passes the camera verdict 3/3,
but `reference_success` -- the pose-based scorer that checks the height clause -- goes
3/3 to **0/3**, and comes back to 1/1 the moment the AGV's camera is off. `--camera-hz N`
caps a base's frame rate independently of the control rate (`--camera-hz 2` recovers
7.5 Hz), which is also what real hardware does: the lidar and depth streams have had
their own clocks in `SensorStreams` all along and the colour camera did not.

What differs per robot is the **dialect**, because each one mirrors a different piece of
real hardware:

- the **myAGV** is a ROS 1 stack, so it gets single-slash type strings
  (`geometry_msgs/Twist`) and a `secs`/`nsecs` stamp;
- the **SO-101** is a ROS 2 ros2_control bringup, so it gets `pkg/msg/Type` and a
  `sec`/`nanosec` stamp.

Both sets of message builders live side by side in that file, clearly labelled. A client
that had to guess which dialect a stamp was in would be a worse thing than a little
duplication.

The `rosapi` shim answers `/rosapi/topics` from **both** what a surface has published and
what it declared to `on()` -- which is why `on()` takes a message type it does not need in
order to route anything. A real `rosapi` lists a node's subscriptions beside its
publications, so a client can discover how to *command* a robot as well as how to watch
it; answering from publications alone left the two command topics invisible and the arm
looking like a sensor. `topics_for_type` still answers from publications alone, on
purpose: its one caller asks what publishes CompressedImage and subscribes to the answer,
so a subscription in that list could only ever be a topic to wait on that nothing sends.

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

**A tabletop arm's heading is chosen by what is under its workspace, not by where the
target object is.** `find_tabletop_mount` scores every candidate cell at 24 headings by
*coverage* — the fraction of the arm's forward half-disc, out to the reach annulus plus
`WORKSPACE_MARGIN`, with worktop underneath — and that is the first sort key, ahead of
objects in reach, the rim, and elbow room. It has to be, because a task lays its objects
out in the **arm's base frame**: a heading that runs off the surface takes the plate, the
mug and both cameras with it. Facing the target object instead put the arm on the corner
of FloorPlan1's island looking diagonally off it, which reads perfectly in the placement
log — the apple was 0.23 m away and in reach — while the staged mug hung in mid-air over
the floor and a third of the overhead camera's frame was floorboards. Coverage went from
that pose to 100 %, and the arm now sits at the rim looking in, which is what the RoboCasa
engine's own `find_counter_mount` has always done and what a clamp-mounted arm looks like.
`--mount-centre` (`edge_bias=False`) now only drops a tie-break and on most surfaces
changes nothing. `tools/test_placement.py` pins the rule on synthetic surfaces, including
an L-shaped counter whose notch is open floor.

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

- **The MuJoCo viewer and offscreen camera rendering do coexist in one process**, which
  is what `kitchen.sh serve --viewer` relies on and was not safe to assume. Under
  `MUJOCO_GL=glfw` each `mujoco.Renderer` opens a *hidden* GLFW window -- a real
  `NSWindow` -- while under `mjpython` the script runs off the Cocoa main thread, and
  GLFW documents window creation as main-thread-only there. Verified working before the
  flag was added; if it ever regresses, both `env.sh` files name `MUJOCO_GL=cgl` (no
  window, thread-agnostic contexts) as the escape hatch.
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
and mounts the arm so they land in its working annulus. `simulator/kitchen.sh` sets
both engines up around the same object pair for comparison.

---

## robot_console/

The arm task takes two terminals, because it is two projects. The simulator hosts the
world; the console runs the task against it. From `simulator/`:

```bash
./kitchen.sh serve                         # stage the task, serve it on ws://127.0.0.1:9090
./kitchen.sh serve --viewer                # ...plus the engine's own MuJoCo window
./kitchen.sh serve --engine robocasa       # the other engine (one engine per run)
./kitchen.sh serve --robots so101,myagv    # ...with a myAGV in the same kitchen, one port
./kitchen.sh cameras                       # the live camera page against a running serve
```

and from `robot_console/`:

```bash
./run_task.sh                              # MolmoAct2 over ROS against the default port
./run_task.sh --episodes 8 --label robocasa   # a pass count, named after the engine serving
./run_task.sh --instruction "..."          # a different instruction (scorers unchanged)
```

**The scripted `so101_waypoint` policy is gone**, deleted rather than deprecated: the VLA
is the only policy this console runs. What it used to provide -- a transport check that
passed every time, and the preflight's "does the plan solve from here" reach gate -- is
covered by the fleet check and by `arm/preflight.py` solving a top-down grasp at the
apple and a release over the plate with the console's own IK (`_within_reach`). Its
first waypoint was also the record of the start pose; that is now `task.START_ARM_QPOS`,
held to the simulator's by a test.

`kitchen.sh serve` stages the task into the engine's kitchen -- on the engine's own
counter, without the reference rig's wooden slab (`--reference-table` puts it back; it
overlapped the island visibly and, measured, does not move the VLA's pass count) -- and
prints whether the staged plate and apple are inside the arm's reach annulus, read from
the compiled model, along with the layout, the plate centre, the apple's measured radius
and the resting height the success gate will use.

**The apple and the plate are each engine's own, by default.** `--task-objects` brings
back the task's measured YCB pair; without it MolmoSpaces adopts the iTHOR house's
nearest apple and plate (`adopt_native_objects`) and RoboCasa spawns `apple_10` and
`plate_4` from its registry (`make_task_objects`; `NATIVE_MODELS` records why those
two -- the reddest apple and the only pure-white plate, by measured texture). Both
engines scale the apple to the contract's 20 mm radius, give its colliders the task's
`APPLE_CONTACT` block (`condim 6` is what makes rolling friction exist at all), and hand
the bodies to the task under `task_apple`/`task_plate` -- the `task_` prefix is what
keeps the workspace clearing from sinking them. The arbiter then *measures* the plate
centre, the apple radius and the resting height off the compiled model rather than
reading the constants, which is what lets one predicate grade a YCB sphere on a rimmed
cylinder and an iTHOR hull on an iTHOR plate alike. One trap, measured: **MjSpec keeps
compiled mesh data across compiles**, so scaling an existing mesh asset after the
launcher's first compile changes `mesh_scale` and nothing else -- the apple came out
full-sized with the metadata claiming 0.4x. Scaled *copies* under new names are compiled
fresh; that is what `adopt_native_objects` does.

RoboCasa's registry objects needed the task's *physics* as well as its sizes, and the wire
said so before any episode ran. `apple_10` decomposes into 12 convex pieces, some 1 mm
thin, and under the task's soft contact block they bounce: after a `/reset` the apple
was 2 cm off its spawn within half a second, its speed spiked to 0.4-1.1 m/s in bursts
with nothing touching it, and it had drifted 14 cm in 15 s -- so the preflight never saw
it still and refused every episode. The free-jointed `plate_4` crept 1 cm across the
counter in 12 s under the arm's start pose. So on RoboCasa the registry meshes stay for
looks and the physics is the task's own: the apple's collision is one 20 mm sphere (the
task's own recipe -- "a textured mesh for looks, a primitive for physics"), and the plate
is static, as the task's plate and the reference rig's are. Measured afterwards: apple at
(0.2260, -0.2260), plate at (0.300, 0.100), zero velocity before and after a reset. The
iTHOR apple's hull needed neither. One more trap: the two engines' venvs run **MuJoCo 3.5
and 3.3.1**, whose spec-editing APIs disagree exactly (`spec.delete(el)` only in 3.5,
`el.delete()` only in 3.3.1); `apple_on_plate.spec_delete` is the one rule both use.

On RoboCasa the **dressing is the registry's too** (`NATIVE_DRESSING`: the reference
rig's bowl, mug, banana and lemon, plus orange, pear, bread and kiwi), and it is placed by
`layout_native_dressing` rather than at the task's fixed positions, because those were
laid out for the standard layout: with the plate at the apple's spawn the task's banana
reached into the plate's footprint. The placer takes each object's own
`horizontal_radius`, keeps it off the apple, the plate, the arm's footprint, everything
placed before it, the worktop's edges and the apple-to-plate carry line, and picks the
free cell nearest the reference position (or a far corner, for the extras). **The
dressing is static**, like the plate, and for the same measured reason: left free, the
registry hulls bounce on their own under the task's solver settings -- the banana, lemon
and mug drifted 8-14 cm in 20 s with 0.4-1.2 m/s speed spikes while nothing touched
them, visible as scenery wandering on the live camera page -- and static they read
exactly zero. It means the arm cannot knock them over, which is the lesser evil. Nothing red
besides the apple, and that was measured rather than eyeballed: both registry cups are
red (so there is a pear instead), and the sampler's default mug and bowl were too (so
`mug_7` and `bowl_11` are pinned). What remains is a 4 mm contact between the jaw at its
start pose -- jaw centre (0.241, 0.014, 0.044) -- and the rim of a plate parked where
the arm's home position looks; the plate is static, so it costs a warning and nothing
else.

**One engine swaps the objects.** `kitchen.sh serve --engine robocasa` stages the plate
at the apple's spawn and the apple where the plate was (`--swap-objects`, on by default
there and off for MolmoSpaces; `--no-swap-objects` reverts). The console is never told:
`arm/preflight.py` reads which layout is on the wire off the apple's reset position
(`task.layout_of`), writes it to `scene_reset.json`, and `run_task.sh` hands it to the
task as `-T layout=…`, so the pose-derived `reference_success` column grades the world
that exists. The camera verdict never needed telling -- it finds both objects in the
frame.
`run_task.sh` is the front door for the task and does four things in order: pick the venv
the policy needs, wait for the *topics* (not the port — a listening socket says nothing
about whether the scene compiled), check every expected robot is on the wire, then per
episode reset and **verify** the world, run `inspect-robot`, and grade the log with
`arm/verdict.py`. Each of those exists because its absence produced a confusing failure;
the comments in the script say which. `kitchen.sh inspect` used to do both halves in one
place and no longer exists: grading is the console's business, and the shell heredoc it
lived in had a bug nobody had hit (its `*.json` glob matched inspect-robot's live snapshot).

**Report pass counts, never a single run.** A VLA on this task is a coin toss on the
reference rig, and one episode of it tells you nothing. (The scripted policy that used
to pass every attempt is gone; every figure below that cites it is history.)

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
- `fleet.py` — "which robots are on this rosbridge, and are they the ones expected?" One
  `/rosapi/topics` call, checked against the console's **own** contract constants rather
  than a list typed into a shell script, which would drift from both sides. Built on
  roslibpy, not the arm extra's client, so a fleet check works on a console installed with
  nothing but numpy/OpenCV/roslibpy. `--dump` prints the sorted topic list, which is how
  the two engines are compared: identical lists is the "a client cannot tell them apart"
  invariant checked instead of eyeballed across two terminals.
- `arm/` — the SO-101 task, and the only part of the console that drives an arm. It
  registers itself with the `inspect-robots` framework through the
  `inspect_robots.{tasks,policies,embodiments,scorers}` entry points in
  `pyproject.toml`, which is what makes `inspect-robot run --task apple_on_plate
  --policy molmoact2 --embodiment so101_ros -E url=ws://…` work with no console script
  of its own. Inside: `ros_client.py` (the header-stamping shim, below), `ros_settings.py`
  (every topic name and camera size in one place), `kinematics.py` (MuJoCo-free FK/IK,
  which the preflight's reach gate solves with), `molmoact.py` (the MolmoAct2-SO100_101
  VLA), `task.py`/`success.py`/`scorer.py`, `embodiment.py`, and `preflight.py`
  (reset-and-verify). Torch lives behind a function-local import so the offline suite
  stays torch-free. `run_task.sh` at the console's top level is the arm launcher: it
  picks `.venv-vla` for `molmoact2` and `.venv` for everything else, and grades with
  `arm/verdict.py` (stdlib-only, tested, and the only reader of an episode's log).

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

**The tables below are the bare contract** — what a single robot's vendor stack presents,
and what `--ros-namespace ''` / `namespace=""` reproduce exactly. On the wire every name
is under the robot's namespace, `myagv` and `so101` by default: `/myagv/cmd_vel`,
`/so101/joint_states`, and frames `myagv/odom → myagv/base_footprint`. See "One bridge, N
robots" above for the rule and why the constants stay bare on both sides.

### The myAGV — a ROS 1 mobile base

| Direction | Topic | Type | Fields used |
|---|---|---|---|
| console → robot | `/cmd_vel` | `geometry_msgs/Twist` | `linear.x`, `linear.y`, `angular.z` |
| robot → console | `/odom` | `nav_msgs/Odometry` | pose, twist; `odom` → `base_footprint` |
| robot → console | `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | base64 JPEG |
| robot → console | `/scan` | `sensor_msgs/LaserScan` | `ranges`, angles, range limits |

Namespaced: `/<ns>/cmd_vel` and friends, with `<ns>/odom → <ns>/base_footprint` and the
lidar on `<ns>/laser_frame`. `--namespace myagv` on `bin/teleop.sh`, `bin/slam.sh` and
`robot_console.smoke` sets all four topics at once; naming a topic explicitly still wins.
**`smoke.py` is the only code anywhere that reads a `frame_id`,** so its odom-frame check
is what keeps the frame half of this verified rather than merely emitted.

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
| robot → console | `/overhead/color/compressed`, `/side/color/compressed`, `/wrist/color/compressed` | `sensor_msgs/msg/CompressedImage` |
| console → robot | `/reset`, `/mujoco_ros2_control_node/reset_world` | Trigger-shaped `{success, message}` |

Namespaced: `/<ns>/joint_states`, `/<ns>/reset` and the rest, `so101` by default.
`JointState.frame_id` is the one exception and stays **empty** — a real
`joint_state_broadcaster` publishes no frame there, and inventing `so101/` would be a
difference from hardware rather than a fidelity to it. `-E namespace=so101` is how
`inspect-robot` is pointed at it; `arm/ros_settings.py` applies the prefix in
`base_kwargs()` and `cameras()`, never to the fields.

ROS 2 `pkg/msg/Type` strings and a `sec`/`nanosec` stamp. Joint order is fixed everywhere
(`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, then the
gripper), all `_joint`-suffixed, all radians except the gripper, which is 0 (closed) to
1 (open).

Six things that are silent when wrong, and each cost a debugging session:

- **The header-stamping shim must be keyed on the topic as it goes on the wire.** It is
  keyed on `settings.topic(command_topic)`, not on the bare constant. Keyed on the bare
  one under a namespace it matches nothing, every trajectory ships without a header, and
  `joint_trajectory_controller` accepts and ignores all of them — which is exactly the
  next failure in this list, reintroduced by a rename. A test pins it.

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
- **There is no success topic, deliberately.** The reference container publishes
  `/task_success`; these simulators do not, and a test pins the absence. It is the one
  place this contract diverges from the container on purpose rather than by oversight.
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

`shared/tasks/apple_on_plate.py` stages the reference rig's whole table into whichever
kitchen an engine compiled: a 0.92 m wood work surface, the apple and plate, and the
bowl, mug, banana and lemon it keeps as scenery. **Bringing a table into a kitchen is
not redundant** -- a camera's framing is a property of the surface under it, and with
the slab staged all four of its corners land in the overhead frame at a worst normalised
radius of 0.930, the reference's own figure, where a bare counter gave a diagonal worktop
with a third of the frame floor. `--render-framing` prints that number, the exposure, and
how the slab sits on the counter, so none of it has to be judged by eye.

Two engine traps that the same geom cannot satisfy at once, which is why every task
object splits a visual geom from a collider:

- RoboCasa renders through a mask showing **groups 1-2** (its own collision hulls are
  group 0), so a visual mesh in group 0 is invisible in every frame there -- silently.
- RoboCasa also sets **`inertiagrouprange = [0, 0]`**, so only group-0 geoms carry
  inertia. A collider outside group 0 leaves its body massless and the kitchen refuses
  to compile, with an error that names no body.

So: visual geoms group 2, colliders group 0. On MolmoSpaces neither constraint applies
and the split costs nothing.

A third disagreement, and the one that actually decided whether the task passed: an
iTHOR house ships `noslip_iterations = 4` and a RoboCasa kitchen ships **0**. MuJoCo's
no-slip solver is what stops a held object creeping out from between the fingers, so it
is a requirement of *grasping* and not a scene preference -- the task now asserts it
alongside the friction cone and `impratio`. Without it, RoboCasa executed all 37
waypoints, closed the jaw on the apple, stalled at the apple-between-the-fingers width,
and finished with the apple back at its spawn point having never travelled: a slip, not
a miss, and indistinguishable from a policy failure from the outside. With it, 2/2.

Lighting does **not** transfer wholesale, and the way that turned out is the most
useful thing in this file. On an iTHOR kitchen's overhead frame: the kitchen's own
lighting clips 5.7 % of pixels to white, adding the reference's headlight and shadowclip
gives 3.1 % against the reference scene's own 3.0 %, and adding its two directional lamps
as well gives 75.2 %. So the exposure block is a near-perfect photometric match to the rig
MolmoAct2 was tuned on -- and it was on by default, and it cost the policy the task.

Seven six-episode runs, varying the slab, the distractors and the exposure block
independently: **0/24 episodes passed with the block on, 6/18 with it off**, Fisher
one-tailed p ~ 0.004. The approach distances separate more cleanly than the pass counts do
-- with it off the policy's best episodes put the apple 3, 8, 12 and 50 mm from the plate
centre, while 24 episodes with it on never once got inside 150 mm. Neither the slab nor
the distractors moved the result; only the lighting did.

The lesson is not about lighting. Clipped-pixel fraction was a proxy for "the policy can
see this", it was optimised until it matched the reference to 0.1 %, and the thing it
stood in for got worse the whole time. A scene statistic agreeing with the reference rig
is not evidence a policy can act in it, and nothing here should be tuned on one again
without an episode count beside it. `--reference-lighting` still stages the block, for
comparing exposure; `--extra-lights` still adds the lamps. Both are off.


`shared/tasks/apple_on_plate.py` is grafted onto whichever kitchen an engine compiled: it
brings a 20 mm apple, a white plate, two scene cameras and its own arbiter, all placed
relative to wherever the arm got mounted. The engines still supply the room — a task that
replaced the scene would make them bystanders.

**Success is inferred from the overhead camera, and nothing on the wire answers the
task's question.** The simulator used to publish its own verdict on `/task_success` and
does not any more: grading on it means grading on state no camera can see and no real
SO-101 emits, which leaves the grader better informed than the policy it grades.
`arm/vision_success.py` finds the apple and the plate in the same overhead frame the
policy is handed, back-projects both through the known camera pose, and applies the
contract's own gate. The embodiment takes that frame out of the observation rather than
subscribing separately, so grader and policy cannot end up judging different moments.

**Finding the white plate on a white worktop is the detector's hard half, and a flat
threshold does not do it.** As first written the plate mask was `sat < 40 and val > 150`,
which on both engines returns one white region containing the counter *and* the plate:
`findContours(RETR_EXTERNAL)` then hands back the counter's outline with the plate inside
it, `find_plate` answers None for every frame of the episode, and a placement the pose
scorer confirms is graded FAIL at "0.9035 m from plate centre". It reproduced at both mount
positions, so it is the threshold and not the framing.

Labelling plate pixels by projecting the staged plate's known position — rather than by eye
— the two engines fail it in opposite directions, which is why no pair of constants fixes
both:

| engine | plate sat | counter sat | plate val | counter val |
|---|---|---|---|---|
| MolmoSpaces (iTHOR marble) | 4–5 | 15–243 | 240–255 | 85–255 |
| RoboCasa | 0–5 | 3–245 | 252–255 | 83–249 |

Saturation separates them on marble and value separates them on RoboCasa's worktop, and
each engine's separator is useless on the other: sweeping the flat thresholds, MolmoSpaces
needs `sat < 10` and finds nothing above `val > 245`, while RoboCasa needs `val > 250` and
finds nothing below it. What holds on both is that the plate is the *brightest* thing in
the white field, so `find_plate` now takes the flat mask first and falls back to the 95th
percentile of that field's own value distribution (`_bright_core`) — the same statement
without the constant. The fallback only runs on frames that would have returned None, so it
can add detections but never move one that already succeeded.

**What that recovery still costs, on RoboCasa, is about 15 mm of plate centre.** The
percentile eats the dimmer side of the disc, so the fitted centre is pulled 7.5 px off where
the geometry puts it, against 2.6 px on MolmoSpaces where the plain mask resolves it. The
size is right — semi-major 46 px on both — and the bias is one-sided. Measured consequence:
the scripted plan, which `reference_success` confirms places the apple, grades on RoboCasa
at **0.0807 m against the 0.080 m gate** — a miss by 0.7 mm that is detector bias, not
policy. Do not close that gap by moving the percentile until an episode passes; that is the
lighting mistake again. A local Otsu refit around the located plate was tried and is worse
(it takes the whole window: 32.6 px, semi-major 189). The plate's rim is the cue that has
not been tried.

Two more things about that module are load-bearing and were measured, not assumed:

- **Perspective has to be undone properly.** Normalising distances against the plate's own
  ellipse — the obvious shortcut — reads 24 mm too far at the plate and 43 mm too far at
  the spawn point, which silently turns the 80 mm gate into a 56 mm one and fails genuine
  placements. Back-projecting through the camera pose reproduces the poses to 3.0 mm.
- **The height clause is NOT enforced, by decision, and this bounds what a pass means.**
  A verdict means the apple came to rest inside the plate's outline and stayed there. It
  does not mean the apple was on the plate rather than above it. Projection onto the
  resting plane plus `REST_RADIUS_M` rejects an apple being *lowered* — a descent sweeps
  the projected position ~30 mm against 5.1 mm of settled jitter — but nothing rejects one
  held perfectly still, and MolmoAct2 parks the apple over the plate without opening the
  jaw in about half its episodes. Apparent size was measured as a discriminator and the
  resting and held populations interleave (0.949–1.013 against 1.007–1.056), so there is
  no threshold to find; the side camera would settle it by triangulation but the arm
  occludes the apple there too often. **Any rate this produces is an upper bound for a
  policy that might not let go** — measured, the difference between MolmoAct2 reading 3/6
  and 1/6 — and `reference_success`, which sees the pose and does check height, is what
  tells the two apart.

**The free-joint poses stay, and the same predicate is still computed from them** as the
`reference_success` scorer — recorded every step, grading nothing. It earns its keep twice
over: it separates "the policy failed" from "the detector stopped seeing", since a drifted
detector produces exactly the run of failures a broken policy does, and it is the only
thing that checks the height clause the camera gives up on. `kitchen.sh` prints a warning
when the two disagree, and that warning is expected on an episode where the apple was never
released. Every serious defect in the camera verdict was found by measuring against this
column over labelled frames — never by reading the code.

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

**Report pass counts, never a single run.** A VLA on this task is a coin toss even on
the rig it was tuned for, and one episode of `--policy molmoact2` tells you nothing.
`--episodes N` exists for this.

### Where MolmoAct2 stands here, and what moved the needle

Everything in this section was measured with the task's own YCB apple and plate in the
standard layout, before the engines switched to their native objects and RoboCasa to the
swapped layout (see the console section above); the figures are the record of that rig.
The scripted `so101_waypoint` plan it is compared against passed every attempt and has
since been deleted. `molmoact2` passed **3 in 6**, and the pass
count is the least interesting number in this section — what changed under it matters
more. (It briefly passed **0 in 24**, because the reference exposure block was on by
default; see the lighting section above, which is the single largest effect anything in
this repo has had on this policy.)

| | before | after |
|---|---|---|
| episodes where the apple never left spawn (~0.33 m) | 3 of 4 | 1 of 6 |
| episodes that engaged the task at all | 1 of 4 | 5 of 6 |
| passes | 1/4 | 3/6 |

**Namespacing the wire did not move this policy, and that was checked rather than
assumed.** A matched-pair control, 30 episodes, the only difference being
`--ros-namespace ''` + `namespace=` (which reproduce the pre-namespacing wire byte for
byte) against the default `/so101/*`:

| engine | wire | passes | never left spawn |
|---|---|---|---|
| molmospaces | bare (pre-change) | 2/6 | 4/6 |
| molmospaces | namespaced | 2/6 | 4/6 |
| molmospaces | fleet, `so101,myagv` | 2/6 | 3/6 |
| robocasa | bare (pre-change) | 0/6 | 5/6 |
| robocasa | namespaced | 0/6 | 6/6 |

Two things fall out of that table, and neither is about namespacing:

- **RoboCasa is 0/12 for this policy on either wire.** The 3/6 above is a MolmoSpaces
  figure -- the reference engine -- and nothing here has ever recorded a RoboCasa one.
  The gap is real, pre-existing, and unexplained; the scripted policy passed on both
  engines while it existed, so it is the *policy's* transfer between kitchens, not the
  contract.
- **The engagement row above no longer reproduces.** 4 of 6 never left spawn on both
  wires, against the 1 of 6 recorded when that table was written. Whatever drifted, it
  drifted on the pre-change wire too, so it is the rig or the record and not this work.

Worth running the control before reading anything into a VLA pass count: this policy is a
coin toss at n = 6, and the wire is the one variable that can be held fixed exactly.
`/private/tmp` scripts do not survive, so the shape is the thing to keep --
`--ros-namespace ''` on the engine, `-E namespace=` on the embodiment, same episode count.

Images were the one path the scripted policy did **not** exercise -- a blind waypoint
plan passes 3/3 with a broken camera stream while the VLA quietly starves -- which is why
both views are checked directly and still are, now that only the VLA runs: present,
distinct, 640x480, and framed `so101/overhead` / `so101/side` with no MJCF prefix
leaking through.

So the policy went from mostly *not attempting* the task to mostly attempting it and
missing. The pass counts are not distinguishable at this sample size and should not be
read as one; the engagement change is large enough to be worth acting on. **Report pass
counts over many episodes, never a single run** -- `--episodes N` exists for this, and a
VLA on this task is a coin toss even on the rig it was tuned for.

Three things were wrong, all found by measuring the checkpoint rather than the code:

- **The arm started outside the trained state band.** A VLA conditions on measured joint
  state, and this one's processor bins that state into 256 buckets and clips *silently* --
  so an out-of-band start is not an error, it is a total failure of conditioning on every
  step, invisibly. MolmoAct2-SO100_101's `wrist_roll` band maps to +47..+153 degrees in
  our frame; the engine's stock rest pose sits at 0. `shared/tasks/apple_on_plate.py` now
  starts the arm at +1.62 rad, which maps to the middle of that band.
- **Two channels were clipping.** `ros2_so_arm` narrows `wrist_flex` to 1.6 and
  `wrist_roll` to 2.3, below both the mechanism's own range *and* the checkpoint's action
  band, which reaches +2.715 rad on `wrist_roll`. This arm is the mujoco_menagerie model
  with the full ranges, so that truncation was pure loss. `JOINT_LIMITS` now uses the
  MJCF's own limits and nothing clips.
- **The scene had opinions.** The kitchen's own apple landed 0.10 m from the task's
  spawn; see the workspace-clearing note above.

**The two wrist branches were not a bug to tidy away, and the lesson outlives the plan.**
`level_jaw_roll` always has two solutions half a turn apart. The VLA needs the +1.62 one
because that is where its state band lives; the scripted plan, while it existed, needed
the -1.52 one because rolled the other way its jaw fouled the plate rim while lowering --
measured, the arm error grew 0.057 -> 0.087 -> 0.118 rad across the last three
sub-waypoints with each running its full step budget. The start pose is the task's
(`START_ARM_QPOS`, mirrored in the console's `task.py`), and nothing else picks a branch
any more.

That white-plate-on-white-marble worry is now answered, and the answer was no. Staging
the reference's brown wood slab under the plate -- the fix that worry implied -- moves the
pass count not at all: 1/6 with the slab and the distractors, 2/6 with a bare counter, the
same distances either way. What did move it was the exposure block above. The contrast
hypothesis was reasonable and wrong, and it was wrong in a way that a rendered frame would
never have shown, because the frame with the slab looks *better*.

One real defect did come out of looking: the slab was sunk so its top face landed exactly
at z = 0, which is exactly where the counter the arm is bolted to already is. Two coplanar
faces have no depth-test winner, and the counter's marble tore through the wood in
hard-edged patches, one across the corner of the plate. It reads as a broken texture
rather than as geometry, which is how it survived being looked at. `TABLE_TOP_LIFT` stands
the slab a millimetre proud, far below every tolerance that reads that plane. Note `apple_plate_distance` is the **closest**
approach across an episode, not where the apple ended; its explanation string carries the
final distance, and reading the first as the second makes a fly-past look like a near-miss.

---

## Workspace conventions

- `specs.json` describes the current, verified state and **is tracked**.
  `target_specs.json` and `implementation_plan.json` are per-run working documents and
  are gitignored. Both validate against `spec.schema.json` / `plan.schema.json`.
  `goal_prompt.md` describes that spec-driven workflow.
- Comments in this codebase explain *why*, especially where a choice looks wrong without
  the constraint behind it (macOS threading, the reactor, key auto-repeat, codec
  choices). Match that when adding code; do not narrate what the line already says.
