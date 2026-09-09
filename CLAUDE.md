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
  ainex_model.py    the AiNex's model, built from the vendor URDF; both engines call it
  molmospaces/  engine #1 — MuJoCo + MolmoSpaces (iTHOR / procthor houses)
  robocasa/     engine #2 — MuJoCo + robosuite + RoboCasa (kitchens)
  kitchen.sh  the SO-101 on a kitchen work surface, in one engine at a time
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

**The vendor topics are served on 9090 unless told otherwise, on both engines.** A robot
spawned with no wire is a robot nothing can drive or watch, and having to remember
`--ros-port 9090` to get the thing every client here already assumes made the default the
wrong way round; `--ros-port 0` is how a run that wants no server — a render, a placement
check — says so. The port is busy if something else is already serving on it, and that now
fails a second `view` where it used to be silently portless.

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

The shared ones run under either engine's venv, from `simulator/`, and the AiNex's three
are the only checks on a robot whose feet do not collide and whose description carries no
licence:

```bash
molmospaces/.venv/bin/python shared/tests/ainex_ground_check.py       # or robocasa/.venv
molmospaces/.venv/bin/python shared/tests/ainex_grasp_check.py
molmospaces/.venv/bin/python shared/tests/tf_frames_check.py  # every robot's tf vs its URDF
python shared/tests/ainex_provenance_check.py    # the vendor files are still the vendor's
```

`ainex_provenance_check.py` recomputes the git blob hash of all 25 STLs and the xacro
sources against the table in `shared/robots/ainex/urdf/PROVENANCE.md`, which used to name
the upstream commit in prose and verify nothing. This robot is where that matters most:
`Hiwonder/ainex` states no licence, so "verbatim" is the entire basis on which its
description is vendored, and a re-export or a nudged mesh would be indistinguishable from
the vendor's own file afterwards.

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
  `namespace=""` on `RosSettings`, reproduce it exactly for everything the *robot*
  presents. The worktop's camera rig is not that and sits at `/scene/…` either way: it is
  not a robot, so no robot's namespace -- empty or otherwise -- is its.
- **Composition is idempotent**, which is what lets a client pass a namespace *and* name a
  topic explicitly without being prefixed twice.

A namespace holds what that robot presents, and nothing else: the worktop's camera rig
is published under `scene` (see the SO-101 table below), and it is a **fleet member of
its own** — `shared/ros_surfaces/scene.py`, attached by the engine under `SCENE_NAMESPACE`
whenever the task's cameras are in the model. It used to be rendered inside the SO-101's
surface loop through a `NamespacedBus.sibling(ns)`, which meant a kitchen with no arm in
it compiled both cameras and published neither; `sibling` is gone with that. One server,
one graph, and the rig's name belongs to the scene because the rig does.

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

**Those two used to be the whole of `rosapi` here, and a client could tell.** Asked
`/rosapi/message_details` for `ainex_interfaces/HeadState`, or `topic_type` for any topic,
the bridge answered `no service` -- not an empty list, a refusal -- which is the largest
single way this simulator could be told apart from the robot it claims to be
indistinguishable from. A real rosbridge ships `rosapi_node` beside the websocket server;
a real AiNex's launch (`UruBots/ainex-robot-code`,
`ros_ws_src/ainex_app/launch/rosbridge.launch`) starts it unconditionally with every glob
`[*]`. `serve_rosapi()` now answers all sixteen, from the most honest source each has:

- `topics`, `topic_type`, `topics_for_type`, `services`, `service_type` -- the server's own
  tables. `service()` takes a type now, as `on()` did first and for the same reason.
- `message_details`, `service_{request,response}_details` -- `contracts/message_schemas.py`,
  **transcribed from each manufacturer's definition files** with the source recorded in its
  `PROVENANCE` block: Hiwonder's `ainex_interfaces`/`ros_robot_controller`, the
  `mujoco_ros2_control_msgs` plugin for the SO-101's free-joint stream, the ROS
  distributions' own files for the standard packages. Both dialects resolve to one
  definition; `typedefs()` returns the transitive closure, as `rosapi` does.
- `publishers`, `subscribers`, `nodes`, `node_details` -- the **namespace** that declared
  the name (`NamespacedBus` claims every name it composes). A real robot returns node names
  (`/ainex_controller`); there are no nodes here, and a namespace -- one robot's surface --
  is the closest true statement. No client in this project uses node names.
- `get_param_names`, `get_param` -- the parameters a surface set, which in practice means
  one `robot_description` per robot (see "The transform tree", below). Values come back
  **JSON-encoded**, as the real `rosapi_node` returns them, because roslibpy's `Param.get`
  runs the answer through `json.loads`; handing back raw XML makes a robot's description
  arrive at the client as a parse error. A real parameter server holds a great deal more
  than one URDF per robot and none of the rest is here -- a difference, stated.
- `action_servers` -- empty, which is true of this simulator: none, by design.
- `get_ros_version` -- 1, the dialect of the rosapi surface itself. The graph carries two
  dialects by design; a client reads per-topic types from `topics` and never infers them.

`contracts/test_fleet.py` calls every one over the wire and checks the ones with structure
for content -- the closure on `sensor_msgs/Imu`, `HeadState` as `float64 position` /
`float64 duration`, ownership kept apart on a two-robot fleet -- plus a **drift** check
that every message a builder produces matches its declared schema field for field. That
check is what found `get_bus_servos_position` returning `{position}` where the vendor's
`.srv` says `bool success, BusServoPosition[] position`; it carries `success` now. Two
things the same comparison turned up and did not fix: the real robot publishes
`/ros_robot_controller/battery` and `bus_servo/{get,set}_state`, which this contract lacks.

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
  is what a `kitchen.sh serve` window relies on and was not safe to assume. Under
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
- **`find_counter_mount`'s height is not the worktop.** It returns the surface plus
  `SO101_BASE_LIFT`, 4 mm of clearance for the arm's base-plate meshes, which hang below
  its body origin. That is a fact about one robot's meshes. A legged robot whose ride
  height is measured to the *sole* has to stand on the surface itself, and inheriting the
  arm's clearance left the AiNex 4 mm in the air. MolmoSpaces mounts on the chosen
  support's own top face and never had the offset to take back off — the two engines
  disagreed about what "the mount height" meant, and only one of them was wrong.

**The AiNex's feet do not collide, and its height is the ground's to decide.** The torso
rides five position-actuated joints (`ainex_model`, step 2, in the load-bearing order
x, y, theta, z, pitch). Colliding feet would grip the surface at default friction and
fight the x/y actuators — measured as a base that undershoots and picks up yaw it was
never commanded — so the feet are decorative in precisely the sense the myAGV's wheels
are. What keeps them on the surface is `ros_surfaces/ainex/ground.py`: every control tick
it ray-casts straight down from the lowest sole, and drives `base_z` so the stance sole
sits exactly on whatever it finds — a worktop, an uneven iTHOR counter, the floor — for
whatever the legs and the lean are doing. A crouch lowers the body because the legs fold
and the sole stays put. **When no sole finds a surface within reach, the robot falls**:
the z setpoint integrates g and lands the tick a sole would cross the surface below. It
is a fall of the setpoint, not gravity on the torso — `gravcomp` stays at 1 and the limbs
hold their pose on the way down — and it is what turns "walked off the worktop" from a
robot hanging in the air where it left into a robot standing on the floor beneath.
`shared/tests/ainex_ground_check.py` measures all four on a plane: standing at 0.00 mm,
holding that height over 40 consecutive ticks to 0.0174 mm, a crouch that drops the torso
26.7 mm with the sole still on the table, and a robot teleported past the edge landing on
the floor at exactly its ride height (0.2023 m).

**The probe walks through the robot's own foot, and how far it steps to do that is the
whole robot's stability.** `RECAST_STEP_M` is 1 µm because the sole's lowest geom stands
**26 µm** above the surface it rests on: a step over that face restarts the ray *inside*
the worktop and `mj_ray` returns the box's underside, which is exactly the trap `PROBE_M`
is written against, reintroduced by the skip meant to clear the foot. It was 1 mm, so a
standing robot read its surface 100 mm below its sole on every tick a foot geom poked
above it — a gap wider than `SUPPORT_GAP_M`, so "walked off the edge", so a 67 mm fall and
four ticks climbing back. Measured: a permanent **69 mm bob at 2 Hz**, which on the wire is
0.97° of torso pitch and 2.9° of shoulder swing and in the viewer is a robot juddering
where it stands. An infinite ground plane is immune — no far face, and the next surface is
0.2 m down — so it bit only on worktops, counters and iTHOR floor boxes, which is
everywhere this robot is staged. `MAX_RECAST` went 4 → 12 with it: a 1 µm step costs up to
two casts per own geom, and the measured worst case is 5 while walking against a budget of
five.

**And the check passed the whole time, at "gap +0.02 mm".** The cycle's period is 5 control
ticks, `tick(30)` is 30 ≡ 0 (mod 5), and that phase is the one where the sole is right — so
a single sample after a settle read the one good tick in five. This is the file's own rule
about a check sharing its measurement's method, in its third form: **a settled reading is
not a stable one, and only consecutive ticks tell them apart.** The standing check now
asserts the span over 40 of them, which reads 66.44 mm on the old constant and 0.0174 mm
on this one.

**An apple is not a floor, and the probe used to think it was.** `_surface_below` accepted
any geom that was not the robot, so the tick a sole's probe point crossed the task's 20 mm
apple the surface came back as its crown — measured 0.5392 against the worktop's 0.5000 —
and the robot was lifted 39 mm onto it, while the apple itself did not move by a
millimetre. It climbed the fruit. The rule now is `mujoco_bridge.is_loose`: **a body is
loose iff its weld root hangs off the world on a free joint**, and loose geoms are skipped
exactly as the robot's own are.

`body_weldid == 0` — "is this welded to the world" — is *not* that test, and it is the
obvious wrong answer: iTHOR hinges every cabinet door and slides every oven drawer, so it
calls **1608 of FloorPlan1's 2116 geoms** movable and a robot obeying it falls through a
third of the kitchen. The free joint puts the island the AiNex mounts on
(`standardislandheight_…`) with the floor, and the 37 things actually lying about — apple,
bowl, cup, bread, bottle, book, pan, knife — on the other side.

That budget then had to grow, and the give-up warning is what said so, from the island edge
of a real kitchen: an iTHOR object is a decomposed hull, ~30 convex pieces each, and every
piece in the column costs a cast. Sweeping 62 500 downward columns of FloorPlan1, 94 %
reach ground in one cast and the worst needs **37**; `MAX_RECAST` is 64.

**And the feet meet what they walk into, which they did not.** Every geom but the hands and
the torso hull was made non-colliding because contact with the *ground* fights the base —
true of the floor, applied to everything, so the robot passed through an apple without
disturbing it. `ainex_model.enable_foot_contacts` gives the feet a class of their own:
they meet loose bodies and pass through the world. Measured on the check rig, the apple
goes from 0.00 mm to 1571 mm, peaking at 0.76 m/s off a 0.2 m/s walk — about what an
elastic rebound from a servo-held swing foot gives.

Three traps in that, each of which produced a foot that touched nothing while every mask
read correctly:

- **MuJoCo builds a mesh's convex hull only for meshes some collidable geom uses.** A foot
  compiled at contype/conaffinity 0 comes out with `mesh_graphadr == -1` and can never
  collide, whatever the masks are set to afterwards. So `build_spec` gives the feet
  `FOOT_CONTACT_BIT` at build time and `enable_foot_contacts` only re-points it.
- **`body_contype`/`body_conaffinity` are computed once at compile and prune whole bodies
  in broadphase.** Edit the geom masks at runtime without recomputing them and the pair is
  discarded before any geom-level test: measured, with the apple at the exact centroid of a
  foot mesh, MuJoCo reported it touching the table and nothing else.
- **The bit lives on the feet's `conaffinity` and the scene's `contype`, not the reverse.**
  A contact needs the bit in one side's `contype`; iTHOR's `contype` values are only 0, 1
  and 8, so a robot grafted by some path that never calls the function has inert feet.
  Mirrored, the untagged case is the dangerous one — and the bit must be *cleared* from
  everything not loose, because iTHOR writes `conaffinity` 7 and 15 and 892 of FloorPlan1's
  geoms already carry the first free one.

**And the sole is the whole sole, which it was not.** The vendor's `init_pose` leg chain
sums to **−14.95°** rather than to zero — that is their `hip_pitch_offset`, and on the
real robot it tips the *body* forward over feet that stay flat, because the hips carry the
body. Our torso is on planar joints, so with `base_pitch` at zero those 14.95° landed on
the feet instead: both soles toe-up, **the toe 37.6 mm off the surface**, the robot
balanced on two heel corners with 5 of 426 foot vertices touching. `gait.py` had predicted
it in as many words — "the robot would walk on its heels" — and corrected it only for the
walking gait, which solves its own flat-sole IK. Standing had nothing.

`ainex_model.stance_lean` measures that angle off the compiled model and `rest_pose()`
gives it to `base_pitch`, so the vendor's joint values stay exactly as shipped and the
lean goes where the hardware puts it; `gait.leg_joint_targets` states `base_pitch = 0`
explicitly, because a sole solved flat against a leaning torso would tilt the other way.
`ainex_model.stand()` is what both engines apply at spawn — one function, so a client
cannot tell them apart by the pose of a robot's legs — and the ride height it stands at
fell from 0.2114 m (a heel corner) to 0.2023 m (a sole).

**Every check agreed it was standing perfectly, and every check was measuring the heel.**
`ride_height`, `sole_z`, `report_sole_contact` and `test_attach.py`'s "feet rest on the
floor" all take the single lowest vertex, which really was on the surface. That is the
same rule this file records two sections down — *a check of a measurement must not share
its method* — broken a second time in the same file, so the checks now assert the sole's
**tilt** (from `xmat`, a frame reading) *and* its heel-to-toe span (from the mesh, an
independent one): 0.00° and 34.3 mm flat against 14.95° and 53.1 mm on its heels.

Before z existed a robot grafted at the wrong height did not fall, did not warn and
produced no wrong number anywhere — it simply hung there, and at 4 mm that read as a
rendering artefact. `report_sole_contact` still prints the spawn-time gap on both engines
(`soles on the worktop at z 0.9200 (gap +0.00 mm)` on RoboCasa, `z 1.1000` on
MolmoSpaces), and it is now the record of the *graft arithmetic*: the follow would
correct a bad graft on the first tick, so the printed number is what catches the
arithmetic going wrong, not the robot.

**`base_pitch` is the torso's lean, and it exists because the base rides the torso.**
Measured with the legs folded flat and the torso level, the claw tip is still 72 mm above
the sole; a 20° lean brings it to 20 mm. On the real robot the hip chain tips the body
over planted feet when it crawls; here folding the hips alone lifts the legs, so the lean
is a joint of its own, authored as a 25th channel in action-group YAML — and it is what
the standing pose leans by too, which is the section above. `crawl_left`/`crawl_right` — the vendor's names for its bend-down grasp
— are solved by `shared/tools/author_ainex_crawl.py` rather than typed, and
`shared/tests/ainex_grasp_check.py` stages the task's apple where their `reach:` says
and asserts a hand geom touches it and it moves (measured: 46 and 51 mm).

**That check reported 0.00 mm while the robot floated 42.7 mm, and the reason is the most
useful thing in this section.** `ainex_model._mesh_points` offset each mesh's vertices by
`geom_pos` and never rotated them by `geom_quat` -- and every one of the URDF's 25 mesh
geoms carries one, because MuJoCo folds a mesh's principal-axes re-orientation into it at
compile. So `ride_height` came out 0.2541 m against a true 0.2114, the robot was grafted
that much too high on both engines, and the sole check -- which went through the same
function -- measured the same wrong number and agreed. So did `test_attach.py`, whose
`mesh_world_z` had copied the shortcut. Three measurements, one method, unanimous and
wrong. The rule that fell out: **a check of a measurement must not share its method.**
`sole_z` and the test now go through MuJoCo's own `geom_xpos`/`geom_xmat`, which is the
independent frame; `_mesh_points` is fixed too, and the 0.4581 / 0.1901 / 0.2541 figures
both engines quoted in comments were all wrong the same way and are re-measured.

**The side camera is 1.233 m from its look-at, not 0.733 m,** moved back along its own
axis on 2026-09-08 with orientation unchanged so the whole robot is in the side view: at
0.733 m the crown projected at 1.68x the frame height and the robot ran off one edge. It
is not a grading camera -- `vision_success` reads the overhead frame alone -- so the
verdict does not move; it *is* one of the two views the VLA sees, so it is a scene change
of the kind the lighting section warns about and wants a pass count beside it. The pose
is mirrored in `ros_settings.SCENE_CAMERA_POSES` with the contract test holding the two
copies equal. The rig publishes whenever its cameras are in the model, arm or no arm:
`--robots ainex --cameras both` puts `/scene/overhead`, `/scene/side` and the head camera
on the wire, and the console's fleet check asks for the rig behind `--humanoid` as well
as `--arm`. Measured on both engines with `--robots so101,myagv,ainex`: **54 topics,
byte-identical `--dump`s**, and the same three `robot_description` parameters at the same
three sizes (43654 / 16231 / 1380 chars). That was 37 before every robot grew a transform
tree; the count is only worth quoting beside the pair of engines it was equal across.

A RoboCasa kitchen contains no loose objects at all — everything is a fixture — so an arm
has nothing to reach for until `--objects` spawns some from RoboCasa's own registry. That
is the one real asymmetry with MolmoSpaces, where iTHOR houses come with graspables and
their metadata, and it is why **arm-with-objects work belongs in MolmoSpaces**:
`tools/scene_placement.py` there finds a surface that already has graspable objects on it
and mounts the arm so they land in its working annulus. `simulator/kitchen.sh` stages the
task's own apple and plate on either engine's counter, which is what makes the two
comparable without either engine's registry having a say.

---

## robot_console/

The arm task takes two terminals, because it is two projects. The simulator hosts the
world; the console runs the task against it. From `simulator/`:

```bash
./kitchen.sh serve                         # the task on ws://127.0.0.1:9090, headless
./kitchen.sh serve --engine robocasa       # the other engine (one engine per run)
./kitchen.sh serve --robots so101,myagv    # ...with a myAGV in the same kitchen, one port
./kitchen.sh serve --robots so101,ainex    # ...or the humanoid, on /ainex/*
./kitchen.sh serve --cameras robot         # each robot's own camera and nothing else
./kitchen.sh serve --mujoco                # ...and a MuJoCo window on the world it serves
./kitchen.sh serve --engine robocasa --robots ainex --cameras robot --mujoco
./kitchen.sh view                          # the camera page on whatever is serving
./kitchen.sh view --port 9091              # ...on the serve there instead
./kitchen.sh serve --help                  # per-command help; `view --help` too
```

and from `robot_console/`:

```bash
./run_task.sh                              # MolmoAct2 over ROS against the default port
./run_task.sh --episodes 8 --label robocasa   # a pass count, named after the engine serving
./run_task.sh --instruction "..."          # a different instruction (scorers unchanged)
```

**`serve` loads a world, serves it, and optionally shows it; `view` is the camera page and
nothing else.** Each command refuses the other's flags by name, and each `--help` prints
the shared header plus its own section, cut out of that header by `#:` markers, so neither
can drift from what it parses.

**The window belongs to `serve` because the physics does.** `mujoco.viewer` offers
`launch`, `launch_from_path` and `launch_passive` and no `connect` (checked on 3.5.0 and
3.3.1, the two engines' versions), so a viewer is built from the model and data objects in
memory and can only exist inside the process holding them. There is therefore no way to
open a window onto an engine already running, and a `view` in a second terminal that
claimed to was either erroring on a busy port or quietly compiling a *second* kitchen and
showing you that one instead. `--mujoco` is a `serve` flag for that reason: it is the run
that owns the world, so it is the run that can draw it. `serve` is still headless without
it -- a window costs control rate for every client on the port, and closing the window
ends the run, because the two are one process. A serve already running cannot grow a
window; restart it with `--mujoco`.

**`view` holds nothing.** It is a websocket client on `--port`, so it shows whatever a
`serve` is publishing there, and it takes none of the flags that shape a world --
`--engine`, `--robots`, `--scene`, `--cameras` and the staging flags are all refused by
name, because a view has no world to apply them to and `view --engine robocasa` used to be
accepted and change nothing. It is not told what is on the port either: the page asks
`rosapi` (see the live page, below). `--live` still names what `view` already is and is
accepted for that alone.

`--cameras` chooses what renders: `both` (default -- the worktop rig and every robot's
own), `scene` (the rig alone) or `robot` (each robot's own official camera alone: the
SO-101's `wrist_cam`, a myAGV's or an AiNex's `front_camera`). No per-robot syntax is
needed behind those words, because each engine already resolves a base's camera against
that robot's own MJCF prefix and the arm's is one flag. Both narrowing settings take
topics off the wire that something expects -- `robot` drops the rig the arm task is graded
from, `scene` drops a base's camera, which is one of the four rows of its vendor
contract -- so the console's fleet check refuses them by name.

**`apple_on_plate` is staged for the robots that stand at a worktop -- the SO-101 bolted
to one and the AiNex standing on one -- and for no others.** A myAGV takes the floor and
has neither a work surface nor a gripper, so a kitchen holding only bases gets the room,
the robots and their cameras and no task; staging it anyway bound the task to whichever
robot happened to be first in the list. The objects land in front of whichever robot got
mounted because the staging transform is built from the mount point and the worktop
height, not from the robot, so it is the same apple and the same plate on the same counter
either way.

Three places assumed that robot was the arm, and each says so now rather than failing:

- The arbiter finds the robot's **root body by name**, given by the engine. The SO-101's
  is `base`; the AiNex roots at `body_link`, the torso its vendor URDF roots at. Guessing
  from a list would work until two robots in one scene had different roots.
- The task's **start pose is the SO-101's** and is applied only to it. `START_ARM_QPOS`
  exists because MolmoAct2 bins measured joint state into 256 buckets and clips silently,
  so an arm outside the trained band is not conditioned at all -- a statement about five
  named joints on one robot. Another robot has no such pose, and its engine has already
  stood it up in the vendor's `init_pose`. The flag is explicit rather than "skip joints
  that are missing", which for the arm would fail invisibly in exactly that way.
- The **contact check** finds the gripper by geom name (`fixed_jaw`/`moving_jaw`) for the
  arm and by owning body for the AiNex, whose hands carry URDF mesh geoms with generated
  names and are the only bodies `ainex_model` leaves collidable at all. Measured: 16 jaw
  geoms for the SO-101 on either engine, 4 hand geoms for the AiNex, all colliding with
  the task's objects.

What this does **not** do is make the AiNex an SO-101. The reach report still prints the
arm's annulus, and at ~0.32 m the objects sit outside the AiNex's own 0.11-0.25 m; the
success predicate is still written around a jaw closing on an apple. The objects are on
the counter in front of it, which is what was asked for.

`shot`, `inspect` and `cameras` used to be commands here and are gone: `shot` rendered a
screenshot per engine back when two could run at once, grading is the console's half of
the split, and `cameras` is what `view` now is.

### The live page reads the wire, and is told nothing

`live_cameras.html` is one static file served by a stdlib `http.server`; it speaks the
rosbridge JSON protocol by hand and has no build step. It used to be handed `?ns=so101` by
the launcher and drew that arm's sliders whatever was actually running. Both halves are
discovered now, and both had to be:

- **Cameras come from `topics_for_type`, asked twice** -- once for
  `sensor_msgs/msg/CompressedImage` and once for `sensor_msgs/CompressedImage`. Two
  dialects share one graph: the SO-101 is a ROS 2 bringup and the myAGV and AiNex are
  ROS 1 stacks, and `topics_for_type` matches the string exactly. Asking once found one
  dialect, so `--robots ainex` or `--robots myagv` showed an **empty grid** -- which reads
  as a simulator that failed to render rather than as a client asking the wrong question.
  Tile names come off the topic, which is `/color/compressed` behind a RealSense-style
  node and `/image_raw/compressed` behind `usb_cam`; both shapes are stripped.
- **Robots come from `/rosapi/topics`**, grouped by namespace and identified by a
  signature *command* topic -- `joint_trajectory_controller/joint_trajectory` for the
  SO-101, `walking/set_param` for the AiNex, `cmd_vel` for the myAGV. Command topics
  because a robot whose first camera frame has not been encoded is still identifiable, and
  because `rosapi` keeps declared subscriptions in that answer precisely so a client can
  discover how to *drive* a robot. `scene` is not a robot and gets no panel. `?ns=` is a
  filter to one robot, not the setting it used to be.

Each robot found gets its own **Enable control** switch and publishes nothing until it is
ticked; that is per robot, so arming the arm cannot start the humanoid walking. The AiNex
panel is the vendor's shape rather than a drive pad, because the robot has no `/cmd_vel`:
walk state machine, gait parameter block, action groups, head. Two traps in it, both
measured against `shared/ros_surfaces/ainex/surface.py` rather than assumed:

- **`enable`, `start` and `stop` are ignored unless the robot considers itself
  initialised, and only `enable_control` sets that.** The enable button sends both, in
  that order, as the vendor's app layer does. Without it every button on the panel does
  nothing, with no error anywhere.
- **The gait block on the wire is not the one `gait.py` holds.** `period_time` is
  milliseconds there and seconds inside; the yaw amplitude is degrees on the wire and
  radians inside; and the fields are the vendor's `x_move_amplitude`/`y_move_amplitude`/
  `angle_move_amplitude`. Every slider republishes the whole block, because the handler
  defaults any field it is not given and sending one would silently reset the other three.
  The forward slider shows `4A/T` beside it -- an amplitude is half a foot's sweep and
  does not read as a speed.

The AiNex panel also carries **one slider per joint, in the body's groups** — left arm,
right arm, each hand, head, legs — publishing to the manufacturer's own per-joint
controllers, `/<joint>_controller/command` for all 24 (`ainex_gazebo`'s
`position_controller.yaml` layout; the head pair were already two of them). The head
takes `ainex_interfaces/HeadState`, the rest `std_msgs/Float64`, and the surface accepts
either shape on any of them. The console's `ainex_topics.py` lists the same 24 in
`CONTRACT_TOPICS`, held equal to the simulator's table by its contract test, so the
fleet check requires them.

`myagv` is identified and labelled but deliberately gets no drive pad: the base has no
watchdog on real hardware, so anything driving it must publish continuously *and*
guarantee a zero Twist on every exit path, and a browser tab cannot promise the second
half. `robot_console/bin/teleop.sh` does both and is the tested path.

The page also re-sends every panel's `subscribe` and `advertise` after a reconnect.
rosbridge keeps no state across a dropped websocket, and without that a reconnected page
looks entirely alive -- cameras stream, sliders move -- while no readout updates and no
command lands.


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
./bin/teleop.sh                         # drive whatever robot is on ws://127.0.0.1:9090
./bin/teleop.sh --namespace myagv       # ...the one on /myagv/*, without asking
./bin/teleop.sh --namespace ''          # ...the bare contract, on purpose
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

### The launchers ask the wire which robot is on it

Two defaults that are each right and did not meet. The simulator names every robot after
itself, so a lone myAGV is on `/myagv/*`; the console's constants are the **bare** vendor
contract, `/cmd_vel` and `/odom`, because that is what one real bringup presents. Run
`./run.sh view --robot myagv --ros-port 9090` and then `bin/teleop.sh` and the console
published into a void and subscribed to topics nobody fed — and **nothing errored**.
roslibpy subscribes happily to a name that does not exist and the bridge acks nothing, so
it read as a black camera window and a robot ignoring every key. The TCP preflight cannot
see it: it never reads a topic name.

So `--robot` and `--namespace` now default to *not given*, which means "ask", and
`discovery.py` asks — `/rosapi/topics`, grouped by the namespace that composes a signature
**command** topic: `/cmd_vel` for the myAGV, `/walking/set_param` for the AiNex. Command
topics because a robot whose first frame has not been encoded is still identifiable, and
because rosapi keeps declared subscriptions in that answer precisely so a client can
discover how to *drive* something. It is the same table and the same question
`live_cameras.html` uses, duplicated rather than shared because the console must install
with no simulator checkout. `bin/slam.sh` does the same, always wanting the myAGV: it maps
`/scan` and dead-reckons `/odom`, and a walking robot has neither.

Four things worth knowing before touching it:

- **`--namespace ''` is still how the bare contract is asked for**, and it is a decision
  rather than a gap. Those two were the same thing when the default was `''`, which is the
  whole bug.
- **Discovery must `close()` and never `terminate()`.** roslibpy's Twisted reactor is
  process-global and single-shot, and the link that drives the robot connects *after* the
  discovery pass, in the same process. `fleet.list_topics` already had this right;
  `test_link_roundtrip.py` now pins it, because it is the one part of this a pure test
  cannot reach.
- **A wire that cannot be asked is not an error.** No rosapi node, an old bridge, a
  timeout: the console says so and falls back to the myAGV on the bare contract, which is
  what it assumed before it could ask. Two robots of the same kind, or a named robot that
  is not there, *is* an error — those are questions only the user can answer.
- **The signature is a command topic, so the worktop rig is not a robot.** `/scene/*` has
  two cameras and nothing to drive, and an SO-101 has a signature of its own and no place
  in teleop; a `so101,myagv` fleet resolves to the base. Measured on both.
- **Every name a link uses takes the namespace, not just the camera.** `AiNexLink`
  composed the camera under the discovered namespace and left `/walking/set_param` and
  `/walking/command` bare, so on either engine — both namespace every robot after itself —
  the view streamed at 20 Hz while every walk and turn command went to a topic nobody
  subscribed to. A robot that shows you its camera and ignores the keyboard reads as a
  broken gait, not as a naming bug.

**The AiNex is driven through a service, and the vendor's handshake comes first.** There
is no `/cmd_vel` here: `publish_cmd_vel` writes a `WalkingParam` block to
`/walking/set_param` and calls `/walking/command` with `start`/`stop` on the *transitions*
only, because the vendor node restarts its gait phase on every `start`. `connect()` sends
`enable_control` then `enable`, in that order, as the vendor's app layer and the browser
panel do — the simulator seeds `initialised = True`, so their absence was invisible here,
while on hardware every command this link sends would be accepted and ignored, silently,
with the rejection discarded by a fire-and-forget service call.

`tests/test_ainex_live.py` (`-m live`) is what joins the two halves, and nothing did
before: the console's gait arithmetic was tested with no wire, the simulator's
`robots/ainex/test_ros.py` drives the *other* command topic
(`/app/set_walking_param`, the tiered preset one), so the topic teleop actually publishes
to had no end-to-end check at all. It holds a Q, an E and a W and reads yaw back off
`/imu` — the only pose this robot's vendor contract carries. Measured, identically on both
engines: **+68.75° and −68.76°** for a three-second turn, 0.01° of drift walking forward.

**The arrow keys point the AiNex's head, over the vendor's own per-joint controllers.**
`publish_head` writes `ainex_interfaces/HeadState` to `/<ns>/head_pan_controller/command`
and `/<ns>/head_tilt_controller/command` — two of the 24 the contract already lists, not a
topic of teleop's own — and only when the pose changes, because a head is a position with
no watchdog to feed where the base's Twist needs re-sending at 20 Hz. `RobotProfile` gains
`has_head` beside `has_odom`, so the myAGV neither grows the keys nor needs a method to
ignore them, and the camera really is on the head (`ainex_model` step 4 moved it there),
which is what the arrows are for.

Two things that are wrong in a way no error reports:

- **An arrow's low byte is a letter.** `action_for_key` masks to the low byte on purpose,
  and GTK/Qt reports Left as `0xFF51`, whose low byte is `0x51` → `q` → `ROT_LEFT`: the
  left arrow would *turn the robot*. So the full code is matched first, against all three
  backends' tables (Cocoa `63232-63235`, GTK/Qt `65361-65364`, Win32), and `app.py` reads
  keys with `cv2.waitKeyEx`, which returns the value untruncated and is identical for ASCII.
- **`head_pan`'s axis is `[0, 0, -1]`, so +pan looks *right*** — the opposite of the base's
  `+z` counter-clockwise yaw, while `head_tilt`'s `[0, -1, 0]` does give +tilt = up. Written
  the intuitive way round, the left arrow pointed the camera right. Measured off the
  compiled model and then confirmed on the wire by phase-correlating the head camera's own
  frames: ← moves the scene +67.6 px right, → −67.7 px left, ↑ +65.6 px down, ↓ −65.6 px up.
  The HUD badge reads `head 8L 8U` rather than signed degrees for exactly this reason.

That confirmation is also a lesson about the method: the same check at a 33° swing reported
tilt *inverted*, because phase correlation loses lock when two frames of a cluttered
kitchen barely overlap. The small step is the trustworthy one, and the offline render
agreed with the vector all along.

### Structure

Behaviour lives in pure, directly testable modules; `app.py` is wiring:

- `topics.py` — topic names and ROS type strings, the contract in one place
- `teleop.py` — keymap, hold/release state machine, speed model (stdlib only)
- `camera.py` — CompressedImage decode + `LatestFrame`, the thread hand-off
- `bridge.py` — `RobotLink` (roslibpy) and pure `parse_odom`
- `hud.py`, `recorder.py`, `preflight.py`, `cli.py`, `smoke.py`
- `discovery.py` — "which robot is on this rosbridge, and under what name?" Pure over the
  `{topic: type}` dict rosapi answers with, so what the console concludes about a wire is
  tested with no wire; the transport is `fleet.list_topics`, so there is one
  `/rosapi/topics` implementation and not two. See the section above for why it exists.
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

### The transform tree, and the description it is read against

**Every robot publishes `/tf`, `/tf_static` and a `robot_description`, and until it did,
every `frame_id` in these tables named a node of a tree that was never published.** A real
bringup runs `robot_state_publisher` beside whatever produces `/joint_states`: it reads
the URDF out of the `robot_description` parameter and turns joint angles into frames, and
that is what lets a client put a scan in the base frame, ask where the gripper is, or draw
the robot at all. This contract emitted the labels and not the substance -- and the
absence went unnoticed for so long precisely because nothing here consumes it: `slam/`
dead-reckons `/odom` and hardcodes the lidar's 65 mm mount rather than looking a transform
up.

Where each number comes from, in `contracts/tf.py`, `mujoco_bridge.TransformTree` and
`ros_surfaces/tf_stream.py` — one copy of the loop, three robots, both engines:

- **Moving links come from MuJoCo's own `xpos`/`xmat`**, not from composing joint angles.
  A second kinematics implementation is a second thing to keep in step, and this file
  already records what that costs.
- **Frames are the *description's* names, through a per-robot map.** They differ on every
  link of the SO-101 — menagerie's MJCF says `shoulder`, TheRobotStudio's URDF says
  `shoulder_link`, and a client renders from the URDF. Menagerie's `camera_mount` has no
  link at all and is deliberately dropped: publishing it would put an engine's model
  layout into a client's tf tree, the same rule that keeps a camera's `frame_id` off its
  MJCF camera name. A skipped body's children reattach to its nearest named ancestor, so
  dropping one cannot break the chain.
- **Links MuJoCo merged away come from the URDF's fixed joints.** A fixed-jointed link
  carries no body, so `imu_link` and `gripper_frame_link` exist in no compiled model. A
  real `robot_state_publisher` reads those out of the URDF too.
- **A camera frame is converted back to the link convention.** MuJoCo cameras look down
  `-z` with `+y` up; a URDF camera link is `+x` forward. Publishing the MuJoCo frame raw
  is the failure that looks like a working system: every transform resolves and the camera
  is drawn on its side. The AiNex is the case that pins it — the vendor bolts `camera_link`
  to the torso, `ainex_model` step 4 moves it to the head where the hardware's camera is,
  and the tree follows the model; `shared/tests/tf_frames_check.py` composes the published
  head-relative frame back into the torso and gets the vendor's own `(0.043, 0, 0.1524)`
  to 0.00e+00.
- **The root's own transform is never published**, because a robot's root has no parent
  inside the robot. The myAGV's `odom → base_footprint` is supplied by the surface from
  the same x/y/yaw that just went out on `/odom`, and it is the only transform in any of
  these trees that is a measurement rather than a reading of the robot's own geometry.
  **On real hardware it is `robot_pose_ekf`'s, not the odometry node's**: `myAGV.cpp:317`
  builds the transform and then does not send it — `//odomBroadcaster.sendTransform(...)
  // robot_pose_ekf ros package instead` — so the real robot emits it only once the EKF
  has odom *and* IMU and its filter has updated, where this one emits from the first tick.
  A difference in *when*. The SO-101 and the AiNex have no root parent at all, which is
  what a real bringup of either presents.
- **`/tf_static` is a ROS 2 topic, and only the SO-101 has one.** The myAGV's three static
  transforms come from `pkg="tf"` publishers, and tf1's node re-publishes onto `/tf` on a
  period; its `robot_state_publisher` has nothing for `/tf_static` either, because the
  vendor URDF's only joint (`base_up`) is `continuous` and there are no fixed joints at
  all. So the ROS 1 robots put their static half on `/tf` and never advertise
  `/tf_static`, and `fleet.py` requires it of an arm and not of a base.
- **The static half is repeated at 1 Hz, because rosbridge has no latching.** For the
  SO-101 that is a departure — on hardware `/tf_static` is latched and a client that
  connects an hour later still receives it. For the ROS 1 robots it is not one: repeating
  on a period is what tf1's own publisher does, at 10–50 ms rather than 1 s.
- **`/tf` takes the namespace, like every other topic.** The tf topic is a *relative* name
  in ROS, so `<group ns="myagv">` publishes `/myagv/tf` with the matching `tf_prefix` on
  the frames — and `--ros-namespace ''` gives back the bare `/tf` a single-robot bringup
  presents.

`shared/tests/tf_frames_check.py` is the check, and it is deliberately made a *different*
way than the tree is: the tree reads MuJoCo's forward kinematics, the check reads the
URDF's own joint origins through an independent rpy conversion — the rule about a check
not sharing its measurement's method, applied in advance for once. Measured: all 26 AiNex
link transforms agree to 2.86e-17 m, all 7 SO-101 ones to 9e-09 m and 2.6e-06 of
quaternion (the MJCF writes its quaternions as decimal text; a tolerance tighter than the
file format is a check nobody can act on).

Two frames are **not** links of any description and are not meant to be: a camera driver
names its own frame (`camera`, `wrist`), and the lidar's mount is a
`static_transform_publisher` line in `myagv_active.launch` rather than a joint in the
chassis URDF.

**The meshes are not served and cannot be.** rosbridge is a JSON websocket; a real client
resolves `package://` against its own filesystem or an out-of-band web server, and that is
true of real rosbridge too — so this is not a divergence from hardware. A client with no
copy of the meshes still gets every frame, every joint limit and the whole link tree.

**Three robots, three different true answers, and the console must not average them.**
Each was checked against the vendor's own source rather than against ROS convention:

| | real robot | required by `fleet.py`? |
|---|---|---|
| myAGV | `myagv_active.launch` starts `robot_state_publisher`, `joint_state_publisher`, `robot_pose_ekf` and 3 tf1 static publishers, and loads the URDF by `textfile` | `/tf` yes, `/tf_static` no |
| SO-101 | **no vendor ROS package exists at all** — `TheRobotStudio/SO-ARM100` ships description files and points at LeRobot. Every ROS 2 bringup is third-party, and all three that ship a launch run `robot_state_publisher` beside the controller manager | both yes |
| AiNex | the vendor's own `display.launch` and `ainex_gazebo/position_controller.launch` run one off this same URDF; the **shipped boot chain** (`start_app_node.service` → `bringup.launch`) starts neither | neither |

The AiNex line is the one that was written wrong first. The claim used to be "a sweep of
`Hiwonder/ainex` finds no `robot_state_publisher` and no `/tf` anywhere", and an exhaustive
sweep of both that repo (485 paths) and `UruBots/ainex-robot-code` (1299) finds three
launch files running one, plus `ainex_peripherals/scripts/tf_broadcaster_imu.py` behind a
`debug` flag that defaults false, plus `/tf` from the apriltag demos on the physical robot
(`ainex_example/config/settings.yaml`: `publish_tf: true`). So giving a simulated AiNex a
tree **matches the vendor's own simulation bringups**, off the vendor's own description; it
differs only from what the shipped robot exposes over rosbridge at boot. Those are two
claims, and conflating them is what made a narrow difference look like an invention.
`robot_console/ainex_topics.py` still leaves the pair out of `CONTRACT_TOPICS`, because the
boot chain is what a client actually meets.

### The myAGV — a ROS 1 mobile base

| Direction | Topic | Type | Fields used |
|---|---|---|---|
| console → robot | `/cmd_vel` | `geometry_msgs/Twist` | `linear.x`, `linear.y`, `angular.z` |
| robot → console | `/odom` | `nav_msgs/Odometry` | pose, twist; `odom` → `base_footprint` |
| robot → console | `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | base64 JPEG |
| robot → console | `/scan` | `sensor_msgs/LaserScan` | `ranges`, angles, range limits |
| robot → console | `/tf` | `tf2_msgs/TFMessage` | `odom → base_footprint`, the lidar and camera mounts |

Namespaced: `/<ns>/cmd_vel` and friends, with `<ns>/odom → <ns>/base_footprint` and the
lidar on `<ns>/laser_frame`. `--namespace myagv` on `bin/teleop.sh`, `bin/slam.sh` and
`robot_console.smoke` sets all four topics at once; naming a topic explicitly still wins.
On the first two the flag is optional — left out, the namespace is read off the wire (see
"The launchers ask the wire", above); `robot_console.smoke` still defaults to bare.
**`smoke.py` is the only code anywhere that reads a `frame_id`,** so its odom-frame check
is what keeps the frame half of this verified rather than merely emitted.

ROS1 single-slash type strings. Body frame: `+x` forward, `+y` left, `+z` CCW. The base
is holonomic (the myAGV is Mecanum), so `linear.y` is a real strafe.

`/scan` follows the **YDLidar X2** — `ydlidar_ros_driver/launch/X2.launch`: frame
`laser_frame`, 0.1–12.0 m, 10 Hz, CCW from `-pi`, mounted at
`base_footprint + (0.065, 0, 0.08)` per `myagv_active.launch`'s static transform. Both
sides encode that mount offset; 65 mm is more than a map cell at 5 cm, and dropping it
smears every wall by a cell. `ranges` is a plain JSON float array; only `uint8[]` is
base64.

**The π in that static transform is on yaw, not roll, and this paragraph used to say
otherwise.** The line is `args="0.065 0.0 0.08 3.14159265 0.0 0.0"`, and `tf`'s nine-
argument form is `x y z yaw pitch roll` — so it is a half-turn about z, not an
upside-down mount. The old claim here was that the driver's `inverted: true` and "the
transform's roll of π" cancel so no sign flip is needed anywhere; roll π reverses a scan's
angular direction and yaw π shifts it by 180° while preserving direction, so whatever is
true of the real robot, that *reasoning* is not. The simulated lidar is a ray-cast that
starts in the base frame and sweeps CCW from `-pi` with no mount rotation, so its own
`base_footprint → laser_frame` transform is identity and its ranges are consistent with
it. **The open question is the real robot**, and it wants a scan captured off hardware
rather than another argument from the launch file — nothing here consumes the tree, so
nothing has been forced to answer it.

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
| robot → console | `/wrist/color/compressed` | `sensor_msgs/msg/CompressedImage` |
| robot → console | `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` |
| scene → console | `/overhead/color/compressed`, `/side/color/compressed` | `sensor_msgs/msg/CompressedImage` |
| console → robot | `/reset`, `/mujoco_ros2_control_node/reset_world` | Trigger-shaped `{success, message}` |

Namespaced: `/<ns>/joint_states`, `/<ns>/reset` and the rest, `so101` by default.
`-E namespace=so101` is how `inspect-robot` is pointed at it; `arm/ros_settings.py`
applies the prefix in `base_kwargs()` and `cameras()`, never to the fields.

**"A real ros2_control bringup for this arm" means a community one — there is no vendor
ROS package.** `TheRobotStudio/SO-ARM100` has no `package.xml` and no `launch/` anywhere;
`Simulation/SO101/` is description files and the README points at LeRobot, which is a
Python/Feetech-serial stack. The reference this contract is written against is
`ros-physical-ai/ros2_so_arm`, which is also where `JOINT_LIMITS` was found narrowing two
channels and which carries a `mujoco_ros2_control` backend of its own — the closest thing
to a canonical answer, and still third-party. Two consequences worth holding: its
`ros2_controllers.yaml` declares the gripper as a
`parallel_gripper_action_controller/GripperActionController`, which is the action-vs-topic
constraint recorded below arrived at independently; and **the vendor URDF's joints carry
no `_joint` suffix** (`shoulder_pan`, not `shoulder_pan_joint`) — the suffixed order in
this contract is `ros2_so_arm`'s, so anyone diffing against TheRobotStudio's file directly
will find a mismatch that is not a bug.

**The last row is not the robot's, and does not take its namespace.** The overhead and
side views are the worktop's fixed camera rig: they watch the work surface, they would
still be there with the arm unbolted, and on real hardware they are a camera driver
launched outside any robot's namespace. They go out under `scene` —
`/scene/overhead/color/compressed`, frame `scene/overhead` — while the eye-in-hand view,
which really is the arm's, stays at `/<ns>/wrist/color/compressed`. Putting them under
`/so101` said the arm owned a view of itself, and with a second robot around the same
worktop it is worse than untidy: whichever robot happened to be asked to render would
lend the scene its name. `SCENE_NAMESPACE` is defined once on each side
(`shared/ros_surfaces/so101.py`, `arm/ros_settings.py`) and a contract test holds the two
equal. The console composes it in one place, `ros_settings.camera_topic` — the preflight's
reachability check reads that too, because a camera under the wrong namespace fails as an
eight-second timeout that blames the simulator.

Two exceptions that look like this one and are not. `JointState.frame_id` stays **empty**:
a real `joint_state_broadcaster` publishes no frame there, and inventing `so101/` would be
a difference from hardware rather than a fidelity to it. And
`/<ns>/free_joint_publisher/free_joint_states` carries the *task objects'* poses but keeps
the robot's namespace, because that is where the publishing plugin puts it on the
reference rig — the name is the contract, not a claim about ownership.

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
- **The arm is mujoco_menagerie's `robotstudio_so101` and nothing else.** Verified by git
  blob SHA against upstream: all 20 meshes, `so101.xml`, `scene.xml`, `scene_box.xml`,
  the README, the LICENSE and the render byte-identical, and `urdf/` likewise against
  TheRobotStudio's `SO-ARM100/Simulation/SO101`. `model.xml` is generated from
  `so101.xml` and its whole delta is one group-3 `tcp` site with no visual, collision or
  dynamics, kept because `SO101RobotView` resolves it by name. Three project edits were
  removed on 2026-09-06 and each has a measured cost recorded in `make_model.py`: an
  `exo_camera`; a softened gripper `forcerange` (the actuator is back on the `sts3215`
  class default of ±2.94 N·m, which is the servo's real figure and **far too strong for a
  20 g apple** — expect the grasp to regress); and a second wrist camera. `/wrist/…` now
  renders upstream's `wrist_cam` at 640x360 — its own 16:9 aspect, not the declared
  1920x1080, which is 2 Mpx inside the physics loop. That pose is 119 mm and 53.7° from
  the removed camera, its half-FOV is 24.2° against the grasp point's 15.9° offset, and
  the printed wrist_roll_follower housing is in frame: the removed camera existed
  precisely because a ray-cast found no unoccluded on-axis mount. A test pins the MJCF
  camera by name, because a rename is how the model would quietly stop being upstream's.

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
distinct, 640x480, and framed `scene/overhead` / `scene/side` with no MJCF prefix
leaking through. (They were `so101/overhead` / `so101/side` when that was measured; the
rig has since moved out of the robot's namespace, which is where it never belonged.)

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
