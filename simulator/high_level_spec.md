# Simulator — High-Level Specification

This document defines only the simulator's requirements. Its sole external specification
is [`../robots_specs/high_level_spec.md`](../robots_specs/high_level_spec.md) together with
its robot files `../robots_specs/<id>.md` (amended 2026-10-02: the robot sections moved into
those files), referred to below as the **robot specification**. Robot ids, identities,
kinds, embodiments, sources, models, sensors and control interfaces are defined there
rather than repeated here.

**Terms.** A robot's **vendor interface** is every node, topic, service, action and parameter
that its authoritative sources, as recorded in the robot specification, define. **ROS
infrastructure** — the `rosbridge_websocket` and `rosapi` nodes and their stock endpoints and
parameters (including `/client_count`, `/connected_clients` and `/rosapi_params`), `/rosout`,
`/rosout_agg`, `/parameter_events`, the ROS master's own parameters (`/rosdistro`,
`/rosversion`, `/run_id`, `/roslaunch/*`) and each node's client-library services (logger,
parameter and type-description services) — is not part of it. Infrastructure may appear on a
wire but never alters or replaces a vendor-owned name or behaviour. The **reference
project** is the prior version of this workspace, github.com/samirma/robot-simulator at
revision `34547ae`. A **worktop robot** is a robot of `kind` `arm` (the SO-101 and the
myCobot 280); its **worktop objects** are the six objects of the reference project's
apple-on-plate scene — an apple, a plate, a bowl, a mug, a banana and a lemon (YCB scans)
— that are part of every scene with a worktop, staged on it when the simulation starts and
independent of any robot (§2.2, §2.3, amended 2026-10-01). An **evidence case** is
one robot spawned with `spawn.sh` into one engine's default scene — an `arm` on the worktop,
every other kind on the floor — with a picture of the scene (for an arm, showing its
worktop objects), a picture from each of the
robot's cameras not marked `optional` and a passing smoke run, through its vendor interface, of each of its motion
commands — drive or walk, arm, gripper, head and action group, as the robot has them — a
drive or walk followed by its recorded stop command and every other motion ending on its
reaching its goal or end state, judged as §5 **Evidence** states.

## 1. Goal

Run a MuJoCo simulation of a realistic household scene in a choice of physics **engine**,
and spawn into it, one at a time and by id, every robot defined in the robot specification.
Every spawned robot shares the one simulation and serves its recorded control interface
on **its own rosbridge websocket** (amended 2026-10-02: robots are single bodies; the
myAGV + myCobot 280 composite is no longer supported). Each simulated robot reproduces the
recorded real-hardware behaviour within the acceptance bounds in §5 on every engine. Scene content differs
between engines by design.

| Entry point | Responsibility |
| --- | --- |
| `simulator/<engine>/run.sh` | Per-engine setup, asset fetching and repair (`setup`, `assets`, `repair`), and `start`: start this engine's simulation headless by default, or in a MuJoCo window with `--mujoco`. |
| `simulator/kitchen.sh start` | Start one engine's simulation, headless or in a MuJoCo window. |
| `simulator/spawn.sh` | Spawn one robot by id into a running simulation and serve its vendor interface on its own rosbridge websocket. The only entry point that serves a wire. |

## 2. Components

### 2.1 Engines — `simulator/<engine>/run.sh`
 
```sh
run.sh setup                            # venv, upstream checkout, default assets, robot meshes,
                                        # worktop objects
run.sh assets [<source>]                # optional bulk pre-fetch for offline use
run.sh start [--scene <s>] [--sim-port <p>] [--mujoco]   # headless unless --mujoco is supplied
run.sh repair                           # re-point venv and assets after a move
```

Two interchangeable engines exist, `molmospaces` and `robocasa`, each with its own launcher
and venv. `assets` fetches every asset source the engine knows, or
only the named `<source>`, so later runs need no network. `setup` also fetches the robot
meshes the robot specification's folders require but the repository does not commit,
from each robot's pinned sources into `robots_specs/<id>/`, verifies them against
`robots_specs/meshes.sha256` and refuses on a mismatch, leaving no unverified file behind;
it fetches the worktop objects' meshes the same way (§4).

* **MolmoSpaces** (the default engine) — MuJoCo with iTHOR and ProcTHOR houses: scene
  sources `ithor` and `procthor`, default scene `ithor:1`. Scenes install on demand, so a
  bare `setup` suffices for `--scene ithor:1`.
* **RoboCasa** — MuJoCo with RoboCasa kitchens, a **scene provider only**: no robosuite
  robot, controller or observation stack enters the model. Scene source `robocasa`, whose
  scene id is `<layout>-<style>`; default scene `robocasa:1-1`. Its fixtures stand where
  RoboCasa's kitchen environment stands them (amended 2026-10-03): the fixtures a layout
  sets on others -- a toaster, toaster oven, coffee machine, knife block, paper towel or
  plant on a counter -- are placed with RoboCasa's own fixture samplers, as its environment
  does when it loads a kitchen (the reference built the arena without that step and left
  them at the world origin, inside the corner walls, or on the floor). Physics is
  RoboCasa's, solved by constraint island (below). **RoboCasa's own kitchen
  objects** (amended 2026-10-02) stand on the counters of every RoboCasa scene as well: its
  bare kitchen holds fixtures only, its objects come from its task environments, so the
  simulator draws graspable objects from RoboCasa's object library with RoboCasa's own
  sampler (the Lightwheel set `setup` installs, plus any registry `assets` fetched) and sets
  each upright on a free spot of a counter top (`Counter.get_reset_regions()`), about eight
  per square metre of counter and at most 24, not touching one another. The draw is a
  deterministic function of the layout and style. They are loose objects like any other:
  those within the six worktop objects' area are cleared (§2.2). Like MolmoSpaces' objects
  they stay where they are set (amended 2026-10-03): a RoboCasa scene is solved by
  constraint island, as MuJoCo does by default from 3.3.6 on (RoboCasa pins 3.3.1, whose
  whole-scene solve threw its lightest objects off the counters), and an object that does
  not stand still upright on a flat top -- one that tips over or keeps rolling -- is drawn
  again.
* Both engines also provide the scene source `test`: a flat floor with one worktop at a
  fixed height, identical on both engines, used for cross-engine comparison (§5).

* Headless by default. `--mujoco` opens a MuJoCo window **in the simulation process** —
  only the process holding the physics can draw it. Ending the simulation, by closing the
  window or stopping the process, ends every spawned robot with it, with the cleanup of
  §2.3.

Each engine must accept every robot defined in the robot specification through the
shared spawn path (§2.3). Only the engine actually used needs setup. A moved or
copied checkout repairs itself on the next run of any entry point.

### 2.2 Starting a simulation — `kitchen.sh start`

```sh
kitchen.sh start [--engine molmospaces|robocasa] [--scene <source>:<id>] [--mujoco]
                 [--sim-port <p>]
```

* One engine per run (default `molmospaces`). The simulation starts with the scene alone
  and no robot. `start` refuses any flag its `--help` does not list.
* **`--scene <source>:<id>`** is the one scene flag, with the same form on every engine.
  Each engine names its own sources and default scene (§2.1), and `--help` lists them. A
  source the chosen engine does not have is refused with a message naming the engine that
  has it, and an id outside the source's range is refused with that range.
* The simulation process owns the physics and all rendering. It accepts spawn requests on
  the local port `--sim-port` (default `9080`), refusing to start when that port is taken.
  The same port also returns, on request, an offscreen render of the current scene from a
  given viewpoint and a spawned robot's joint and pose readings, headless or not. Both are
  simulator-private and never reach a wire.
* The **worktop** is the surface the reference project's worktop survey ranks first
  (amended 2026-10-01; it was the largest level top 0.6-1.1 m above the floor). On iTHOR
  and ProcTHOR scenes, and on `test`: the fixed, table-height (0.35-1.30 m) top surfaces
  of the scene's bodies, those holding the task's own categories (plate, apple) and
  reachable loose objects first, as the reference's `find_grasp_targets` ranks them, and
  with nothing graspable anywhere the largest such top. On RoboCasa: the roomiest counter
  region RoboCasa's `get_reset_regions` reports. It is a deterministic function of the
  scene; a scene with none has no worktop. On the default scenes it is FloorPlan1's island
  (top 1.100 m) and RoboCasa layout 1's counter run (top 0.920 m).
* **The worktop objects are part of the scene** (amended 2026-10-01): `start`, headless or
  with `--mujoco`, on either engine, loads the six worktop objects of §2.3 on the worktop
  with the scene — the objects the reference project's `kitchen.sh` showed with its default
  arm — staged around the spot the SO-101 is placed at, with the loose scene objects in
  their area cleared. They do not depend on any robot being spawned, and exist before,
  while and after one. A scene with no worktop has none.
* Headless by default. `--mujoco` opens a MuJoCo window **in the simulation process** —
  only the process holding the physics can draw it. Ending the simulation, by closing the
  window or stopping the process, ends every spawned robot with it, with the cleanup of
  §2.3.
* There is no reset service: a world is reset by ending the simulation and starting it
  again.

### 2.3 Spawning a robot — `spawn.sh`

```sh
spawn.sh <id> [--placement worktop|floor] [--sim-port <p>] [--port <p>]
```

* `<id>` is one robot id in the robot specification. `spawn.sh --help` lists every
  accepted id with its name, and gives no example id that list lacks. `spawn.sh` refuses
  any flag its `--help` does not list. An unknown id is refused with the same list of
  accepted ids and names. The robot specification is the robot registry. A robot whose
  required files are missing is refused with a message naming `run.sh setup`.
* It adds the robot to the simulation running on `--sim-port` (default `9080`) without
  restarting it, and refuses, with a message, when no simulation is running there.
  Successful addition and removal preserve simulation time and the current poses, joint
  states and velocities of surviving robots and scene objects, as well as ongoing
  controller and watchdog state. Model changes must not artificially reinitialize them;
  subsequent normal physical interaction may change their state. The one exception is the
  loose scene objects a worktop robot's staging clears (**Worktop objects** below): they
  leave the scene while that robot is there, and its removal returns each with the pose and
  velocity it had when it was cleared.
* **Admission.** Each robot id may have at most one pending or running instance in a
  simulation. Refuse a duplicate id, and refuse a new spawn request while another startup
  is in progress, with a diagnostic. Admission reserves the id, placement and required
  wire ports against conflicting requests. A failed attempt releases only its own
  reservations; ending a successful spawn makes its id available again.
* **All-or-nothing startup.** A spawn succeeds only when the physical robot and all its
  required wires are ready to accept their recorded commands and provide their required
  outputs. Once all are ready, `spawn.sh` prints one readiness line naming each wire and its
  port and stays in the foreground. Every refusal, failed startup or wire-failure
  termination exits with a non-zero status after cleanup, as does a spawn ended because
  the simulation ended; a spawn ended by `SIGINT` or `SIGTERM` delivered to it exits with
  status zero. A failed attempt removes the robot and all resources created by that attempt,
  releases its wire ports and placement reservation, and does not disturb existing robots.
* It serves the robot's vendor interface on its own rosbridge websocket on `--port`
  (default `9090`), refusing a port that is taken. Each wire has its own ROS graph;
  robots in one simulation share only the physics. The interface may be manufacturer-
  provided or an explicitly identified community interface, as the robot specification
  defines.
* The interface is bare, as on the real robot; there is no namespace option. Every spawn
  runs its own ROS graph, so robots sharing one simulation never share or collide on names.
* **Placement.** `--placement` (default `worktop`) stands the robot on the worktop or on
  the floor. On the worktop it stands on a spot clear of the objects there; spawning onto
  the worktop of a scene with none, one already holding a robot, or one with no clear
   spot, is refused. A worktop robot is tried, in order, at the spots the reference
   project's survey ranks (amended 2026-10-01): on iTHOR, ProcTHOR and `test`, the
   reference's tabletop search — the robot at the rim of the worktop facing in, with the
   most of its forward workspace over the worktop — first; on RoboCasa, the reference's
   spot against the back edge of the roomiest counter, facing the room, then the
   reference's tabletop search over the counter tops, then spots against a counter's front
   edge facing the wall. The first spot is the reference's own choice, and it is taken
   whenever it meets every requirement below. With its worktop objects staged and the loose
   objects in its working area cleared, a worktop robot's clearance is judged on its own
   collision geometry, not on its bounding box. A robot on the floor faces open floor. The
   spot and heading are a
  deterministic function of the scene, the placement, the robot id and the world as it is
  (the robots already present and where the loose objects are). On either placement, camera
   clearance means that a deterministic 7-column by 5-row grid of rays evenly spanning each
   camera's image frustum has no hit on furniture, walls or earlier robots within 0.8 m of
   the lens;
   exclude the spawned robot's own body, its worktop objects and the surface it stands on
   (the worktop top or the floor) from this test. This is a near-field clearance
   criterion, not a requirement that the rendered scene contain no visible furniture.
    A mobile robot (every `kind` but `arm`) **on the floor** can drive forward at least
    0.5 m, back and to each side at least 0.25 m, and turn in place, without unintended
    contact, with its entire support footprint remaining supported throughout those paths
    and clear of unsupported edges. Intended support contacts are allowed. On the worktop
    (amended 2026-10-02) a mobile robot has no travel requirement: it is placed where its
    footprint is supported and clear, even on a top too small to drive or walk on, and its
    motions are then its own to keep on the surface.
  The entire initial robot configuration must be supported and free of unintended
  interpenetration with the scene or earlier robots; intended support contacts are
  allowed. Each worktop object must stand on the surface the robot stands on
  and be free of interpenetration deeper than 1 mm with the scene, the robot and the other
  objects. If no placement satisfies all applicable requirements, spawning
  refuses with a diagnostic and leaves no robot or wire resources behind.
* **Worktop objects** (amended 2026-10-01: the scene is no longer only what its loader
  builds; its six worktop objects are the scene's own, staged at `start` and independent of
  robots — they are never brought, removed or counted by a robot's spawn).
  They are staged on both engines exactly as the reference project staged them: at the
  reference's poses in a base frame (that frame 4 mm above the top face), the frame of the
  spot the SO-101 is placed at on the worktop (§2.2), with the reference's physics (the apple a
  20 mm sphere with its contact tuning, the plate a fixed cylinder with a 24-box rim, the
  bowl, mug, banana and lemon colliding on their own meshes) and its visuals (the YCB
  scans). Every loose scene object (a body with a free joint, not a robot's) any part of
  which lies within 0.55 m of that frame's origin, horizontally, and between 0.15 m below
  and 0.45 m above it is cleared at start: it is held still 50 m under the scene. No
  camera, reset, solver setting or anything else of the reference's task comes with them,
  and nothing about them reaches a wire: the vendor interface is unchanged.
  **Arms and the objects.** A worktop robot is placed where the scene's objects are, when
  it fits there under every requirement above; the objects are then untouched. An arm that
  does not fit there (both arms do, in their natural ready pose, on the default scenes) is placed at its own
  survey spot instead, and for as long as it is there the six objects stand staged around it
  at the same reference poses in its base frame, the loose scene objects in that area
  cleared too; when it is removed the six objects are back where the scene staged them and
  the objects cleared for it return with the pose and velocity they had when cleared. A
  robot on the floor, and a mobile robot on the worktop, moves none of them. The spawn's
  reply and the control port's `robots` and `scene` answers name the objects an arm stands
  among and the scene objects cleared for them; `scene` also names the scene's own.
* **One wire per robot** (amended 2026-10-02: the myAGV + myCobot 280 composite is no longer
  supported): every robot is a single body with one interface, served on `--port` in its
  recorded ROS dialect; there is no assembly and no `--arm-port`.
* Every spawned robot presents every camera required by its recorded interface;
  no flag removes one.
* The spawn runs in the foreground. Ending it removes its robot from the simulation,
  removes all its wire containers and releases their host ports and its placement
  reservation, without disturbing other robots.
  If any required wire unexpectedly loses its ability to serve the recorded interface
  after startup, including when its container exits or its serving process fails while
  the container remains running, the owning spawn reports the failure and terminates,
  removing its robot and all remaining wires and
  releasing its id, ports and placement reservation. Other robots and surviving world
  and controller state are preserved under the same removal guarantees.
  If the spawn process itself exits abnormally, including by `SIGKILL`, the simulation
  detects it, removes its robot and releases its id and placement reservation, and its
  wire containers are removed and their host ports released, without disturbing other
  robots. When the simulation ends,
  every spawn reports it, removes all its wire containers, releases their host ports and
  exits.

### 2.4 Wires

* Each wire is a Docker container running the ROS distribution specified for its robot
  interface, with rosbridge_suite, published on the host at the wire's port. Its ROS graph
  holds every node of the robot's interface; the simulation process holds the physics and
  rendering and knows nothing of ROS.
* `spawn.sh` builds the images on first use, from pinned sources, and refuses, with a
  message, to run when Docker is not running. Nothing else needs Docker.
* Simulator code reaches a container through a read-only mount, never a copy.

## 3. Wire interfaces

This section defines the ROS interfaces produced by the simulator.

* **Robot authority** — load the models, physical properties, sensor characteristics
  and control-interface definitions designated by the robot specification. Simulator
  interface contracts reproduce the recorded names, types, frames, parameters and
  periodic rates, using the recorded bare interface names. Robot-specific definitions and source metadata belong to the robot specification.
  Where the recorded boot's `robot_description` and the designated model disagree on joint
  names, axes, limits or zero offsets, the boot's `robot_description` governs both the wire
  and the compiled model; the robot specification documents each difference as an
  adaptation.
* **Sensor calibration** — use the recorded measured calibration or explicitly documented
  estimates identically on both engines. Intrinsics must be internally consistent with
  the rendered camera. Estimates must never be represented as measured physical calibration.
* **Timing** — physics targets real-time operation. Hardware-facing periodic rates and
  command watchdog intervals use elapsed wall time, preserving any interface-specific
  timing semantics recorded by the robot authority. Sensor and feedback timestamps
  represent sample acquisition time in the clock expected by the recorded ROS interface;
  stale samples must not be restamped as newly acquired. When execution cannot keep up,
  warn on the simulation process's standard error when the real-time factor over a
  10 s window falls below 0.90, preserve watchdog deadlines, and do not
  claim that the required rates or physical acceptance bounds were met for that interval.
* **No simulator-private state** — no engine name, contact state or other
  simulator-private state is published.
* **Real-robot fidelity** — every simulated robot reproduces the hardware and behaviour
  defined in the robot specification:
   * it presents exactly the recorded interface: every name, type, frame,
    parameter and periodic rate, except rows marked `optional`, which it does not serve;
  * it answers every command according to the recorded behaviour, including the stop
    command and the presence or absence of a command watchdog;
   * its compiled model reproduces the authoritative geometry, joints and limits, subject
     only to adaptations and estimates explicitly documented under the robot specification;
   * it preserves recorded dimensions, joint axes and limits, masses, inertias, collision
     geometry, actuator mappings, command units, control modes and sensor poses;
   * it moves only through physical contact of its own wheels, feet and grippers, driven
     by its recorded command semantics; no simulator-added base or kinematic motion moves
     it. Contact modelling the source models lack, such as mecanum rollers, is a documented
     adaptation.

  Interface names, types, frames, parameters, command semantics and authoritative model
  properties must match exactly, subject to the documented adaptations permitted by the
  robot authority. Physical motion and sensor accuracy are assessed using the acceptance
  bounds in §5. A violation is a regression even when nothing fails.
* **Engine consistency** — the same robot spawned with the same flags on either engine
  presents:
  * identical node, topic, service and action names and types, including depth topics;
  * identical frame ids and transform-tree structure;
  * identical parameter names and values, `robot_description` included, excluding ROS
    infrastructure as defined under **Terms**.
* **Discovery answers** — `/rosapi` is rosbridge_suite's own, answering as it does on the
  real robot; beside the recorded robot interface it lists only ROS infrastructure as
  defined under **Terms**. Infrastructure must never alter or replace a robot-owned
  name or behaviour.

## 4. Constraints

* Each engine has one venv.
* On macOS the MuJoCo viewer must own the main thread.
* Vendored upstreams (`molmospaces/upstream/`, the robosuite and robocasa checkouts) are
  never modified; out-of-tree robots and scenes are added from outside them. each engine's
  `assets/` is generated and never holds curated files.
* Shared logic exists once: one wire-serving implementation per robot, one spawn path for both engines,
  and one function that stands a robot on the floor or the worktop. Two copies are two
  chances to drift.
* Curated object assets — the worktop objects' meshes and textures — live under
  `simulator/shared/objects/`, never in an engine's generated `assets/` and never committed:
  each engine's `run.sh setup` fetches them from their pinned source (elpis-lab/YCB_Dataset
  at the commit whose files are byte-identical to the reference project's), verifies them
  against `simulator/shared/objects/ycb.sha256`, refuses on a mismatch and leaves no
  unverified file behind. Their licences are in `simulator/shared/objects/LICENSES.md`.

## 5. Verification

The project must have unit, integration and end-to-end acceptance checks aligning all
components and their observable behaviour with this specification. Physical and sensor
accuracy checks use the acceptance bounds below.

**Acceptance bounds.** A periodic topic's measured rate is within ±10% of its recorded rate
and no gap between consecutive messages exceeds three recorded periods. Every other
recorded physical or sensor figure is reproduced within the tolerance the robot
specification records for it, or exactly where it records none. A periodic topic
not marked `optional` that is absent from the wire, a served periodic topic of the vendor
interface for which the robot specification records no rate, or a rate measurement whose
observation window is shorter than five periods of the topic measured fails the check; it is
never skipped. Rows marked `optional`, ROS infrastructure and topics the robot specification
records as non-periodic are outside this paragraph. Every reference to acceptance bounds
in this document means these.

* **Robot loading** — every robot id in the robot specification can be spawned on each
  engine with its recorded embodiment, physical properties and sensors.
* **Contracts** — each simulator interface contract matches the interface designated by
  the robot specification (names, types, frames, parameters and rates), and every
  published message matches its declared type field for field.
* **Motion and timing** — recorded commands produce the required motion and respect
  limits, stopping and watchdog behaviour. Check rates and timestamps during normal
  operation and the specified watchdog and diagnostic behaviour during overruns.
* **Sensors** — outputs correspond to the simulated world and the recorded sensor poses,
  calibration and characteristics, within the documented acceptance bounds.
* **Lifecycle** — verify successful spawn, rejected and partially failed startup,
  foreground termination, robot removal and whole-simulation shutdown, including wire
  cleanup and isolation of existing robots. Verify duplicate-id and busy-startup refusal,
  reservation release, and preservation of surviving world and controller state when a
  robot is added or removed while another is operating and a scene object has moved.
  Verify that the six worktop objects are in the scene at start with no robot, and after
  every robot is removed; that an arm placed away from them takes them along and removing
  it puts them back where the scene staged them, returning every scene object cleared for
  it with the pose and velocity it had when cleared, while the other robots and scene
  objects keep their state bit for bit; and that a failed addition rolls back the robot and
  its staging together.
  Disable a required wire after successful startup and verify whole-spawn cleanup,
  including container exit and serving-process failure while its container remains
  running. Kill a spawn process, including
  with `SIGKILL`, and verify its robot, containers, ports, id and placement reservation
  are released. Verify the readiness line and the exit status of every termination path.
* **Placement** — verify valid placements and refusal when support, non-interpenetration,
  camera clearance or, on the floor, supported forward, back, side and in-place-turn
  travel cannot be satisfied, and that a mobile robot is placed on a worktop that supports
  its initial pose but not its full travel paths; the support surface is not counted as a camera-clearance hit. Also
  verify the refusals for no worktop, an occupied worktop and no clear worktop spot,
  open-floor facing, repeatable spot and heading for identical inputs, and that every
  refusal leaves no robot or wire resources. For a worktop robot also verify the staging:
  the six objects at their poses, standing on the surface, the working area's loose
  objects cleared, and refusal when no spot leaves every object supported and clear.
* **Reference parity** — on each engine's default scene the SO-101 stands where the
  reference project's own code stands it, run from a checkout of it: xy and surface height
  within 1 µm, heading within 1e-9 rad, base height within 3 mm; its six worktop objects
  at the reference's poses within 1 µm; the same scene objects cleared, by name. The
  MolmoSpaces scene as `start` loads it, with no robot, equals the reference's default
  scene with its robot left out (its apple-on-plate staging included): the same bodies,
  geoms and meshes, the same six objects at the same poses, and pixel-identical renders.
  A RoboCasa scene is the reference's kitchen (the same fixtures, every one where the
  reference has it but those RoboCasa's own fixture placement sets on a counter (§2.1,
  amended 2026-10-03), the same six objects at the same poses) plus RoboCasa's own kitchen
  objects (§2.1), which the reference did not load; verify they stand on the counters, are the same on every start, and that every
  RoboCasa object lies on a counter top and, from the first second on, stays where it is
  (amended 2026-10-03).
* **Evidence** — every robot id has a passing evidence case (see **Terms**) on each engine.
  A motion passes when the recorded feedback shows the commanded displacement within the
  acceptance bounds and the robot comes to rest after its stop command (a drive or walk) or,
  for every other motion, once it has reached its goal or end state. Where the recorded
  interface publishes no measured feedback for a motion, joint or pose readings from
  `--sim-port` taken before and after judge it; the commands always go through the wire.
  Base motions stay within the travel the placement guarantees (§2.3).
* **Engine consistency** — verify exact interface equality as required by §3 and compare
  physical robot behaviour within the same acceptance bounds under equivalent controlled
  conditions on both engines, using the `test` scene; differing household scenes are not
  equivalent conditions.
* As a result, `run.sh` and `kitchen.sh` allow me to spawn a robot there using `spawn.sh` with a selected robot id.