# Robot Simulator — High-Level Specification

## 1. Goal

Operate real robots and their simulated counterparts with the same tools. The workspace
simulates the robots recorded in `robots_specs/high_level_spec.md` (§2) — mobile bases,
mobile manipulators, a humanoid and arms — inside realistic household scenes, and presents
each simulated robot through **the same network interface the real hardware presents**. A
client written against a robot's real hardware, such as the robot console, therefore works
against the simulated robot unchanged.

### 1.1 What to expect from the workspace

* Every recorded robot can be spawned into each engine's default household scene and
  commanded through its vendor interface (§1.3) on its own rosbridge websocket (amended
  2026-10-02: robots are single bodies; the myAGV + myCobot 280 composite is no longer
  supported).
* The robot console teleoperates a supported mobile robot, and views the cameras and
  operates the bounded controls of any supported robot, whose wire matches one of its
  profiles, whether the simulator or the physical robot serves that wire.
* `evidences/` records, for every robot on every engine, a picture of the scene, a picture
  from each camera, a passing motion smoke run and a passing console check (§3).
* What the cameras see — the scene itself — is content, not interface: it differs between
  simulator and hardware and between engines by design.

### 1.2 What to expect from each part

These expectations are binding: each part's own specification defines the detail and must
meet them.

**Robot records — `robots_specs/`.** The robot registry: one robot file per robot,
`robots_specs/<id>.md`, giving its id, identity, `kind`, embodiment, pinned authoritative
sources, authoritative boot and the paths of its URDF, its official MJCF where one exists,
and its ROS interface file, which live under `robots_specs/<id>/` (amended 2026-10-02: the
robot sections moved into those files). Expect a
self-contained, source-traceable description of each real robot, normative for the
simulator and a reference for the console. It holds robot data, plus only the tooling that
fetches and verifies that data or regenerates its derived models (`robots_specs/tools/`)
and the tests that check it (`robots_specs/tests/`); no simulator or console code.

**Simulator — `simulator/`.** Expect to:

* start a MuJoCo scene — iTHOR or ProcTHOR houses on MolmoSpaces, kitchens on RoboCasa, or
  the shared `test` scene on either — headless or in a MuJoCo window;
* spawn any recorded robot by id onto the floor or the worktop of the running scene without
  restarting it, where a valid placement exists, and remove it again by ending its spawn;
* reach each spawned robot's vendor interface on its own rosbridge websocket, presenting
  exactly the recorded names, types, frames, parameters and periodic rates, except rows marked `optional`, and reproducing the
  recorded geometry and physical behaviour within the simulator's acceptance bounds.

The simulator publishes no simulator-private state on a wire and has no reset service; a
world is reset by restarting the simulation.

**Robot console — `robot_console/`.** Expect to:

* point it with `--url` at a ROS 1 or ROS 2 rosbridge websocket, simulated or physical,
  serving one of its supported profiles;
* teleoperate a supported mobile robot from the keyboard with `teleop.sh`, its cameras shown
  live, with a stop requested when the last motion key is released, on `Space`, on input or
  focus loss, on exit, and at every start once its target is validated; a lost connection
  ends `teleop.sh`;
* open `view.sh` for a browser page showing the selected robot's cameras and the bounded
  controls its profile lists, live once the target validates: changed values are sent to
  the robot immediately (streamed while dragging), and nothing is stopped automatically,
  so a goal keeps running if the tab closes;
* check a wire non-interactively with `python -m robot_console.fleet [--expect <id>]`, which
  validates the wire's robots against the profiles and their non-`optional` cameras, exits
  non-zero on failure and publishes nothing.

The console ships its own robot profiles, needs neither the simulator nor robot firmware,
and uses the same profile unchanged against matching physical and simulated wires.
Mapping, navigation, grading and recording are out of its scope.

### 1.3 Interfaces

A simulated robot presents exactly the interface its authoritative source defines, bare as on
the hardware and without the rows marked `optional` (§2). Every spawned robot runs its own
ROS graph, so the simulator offers no namespacing. The console supports namespace selection
only where the robot's hardware interface documents it.

A robot's **vendor interface** is every node, topic, service, action and parameter its
authoritative sources define. For the SO-101, whose manufacturer defines no ROS interface,
the approved interface is the pinned `ros2_so_arm` real-hardware bringup plus the pinned
`usb_cam` wrist-camera boot, recorded in `robots_specs/so101/ros2.yml` for the simulator and
independently in the console's `so101` profile. This is an approved
community interface, not a claim that it is manufacturer-provided.
The myAGV's approved camera boot is the pinned `usb_cam` boot recorded in
`robots_specs/myagv/ros.yml` for the simulator and independently in the console's `myagv`
profile; it is an approved community interface, not manufacturer-provided.
**ROS infrastructure** — the `rosbridge_websocket` and `rosapi` nodes and their stock
endpoints and parameters (including `/client_count`, `/connected_clients` and
`/rosapi_params`), `/rosout`, `/rosout_agg`, `/parameter_events`, the ROS master's own
parameters (`/rosdistro`, `/rosversion`, `/run_id`, `/roslaunch/*`) and each node's
client-library services (logger, parameter and type-description services) — is not part of it.
Infrastructure may appear on any wire, simulated or physical, but never alters or replaces a
vendor-owned name or behavior.

### 1.4 Authority and independence

This document states the shared goal, the expectations above and the project boundary.
Everything specific to one project — its components, entry points, interfaces, constraints
and checks — is in that project's specification, which must not contradict this one. No
other document or code in the workspace may contradict either level; where code does, the
code is wrong.

The robot records in `robots_specs/` (§2) are normative for the simulator, which takes its
interface facts from them. The console independently derives the interface facts it needs
from the pinned authoritative sources those records identify and from real-hardware
behaviour; neither project's source, tests or specification is authoritative for the other.
Workspace tests in `tests/` may read both source trees and exercise both projects over
rosbridge, but may not use one project's code or data as the oracle for the other, import
one project into the other, or make either depend on the other's source tree. Installed
projects communicate only through standard ROS interfaces.

## 2. Projects

The workspace is two **independent** projects whose installed production code talks only
over rosbridge and never imports the other project. Each project has its own high-level specification, which lists
its user-facing entry points:

| Project | Responsibility | Specification |
| --- | --- | --- |
| `simulator/` | Run a simulation in a choice of physics **engine** (MolmoSpaces or RoboCasa; simulator spec §2.1), spawn robots into it by id, and serve each one's vendor ROS interface on its own rosbridge websocket. | [`simulator/high_level_spec.md`](simulator/high_level_spec.md) |
| `robot_console/` | Connect to compatible ROS 1 or ROS 2 systems to drive robots, view cameras and operate bounded controls without depending on their hardware or simulation provider. | [`robot_console/high_level_spec.md`](robot_console/high_level_spec.md) |

Simulator robot embodiments are recorded under `robots_specs/`:

* [`robots_specs/high_level_spec.md`](robots_specs/high_level_spec.md) and its robot files,
  one per robot, `robots_specs/<id>.md` (amended 2026-10-02: the robot sections moved into
  those files) — each robot's **id**, identity, `kind`, embodiment, official URL, pinned
  sources, authoritative boot,
  documentation URLs, and the paths of its URDF, its official MJCF where one exists, and
  its ROS interface file. The simulator hosts every robot recorded there. Every
  robot id argument in the simulator takes these ids; its `--help` lists the ids it
  accepts, each with its name, and the same list appears in the message that refuses an
  unknown id. These files are the robot registry; there is no separate YAML registry. The console owns
  its supported robot profiles independently and may use the same stable ids where they
  identify the same vendor robot.
* `robots_specs/<id>/` — the robot's URDF, its official MJCF where one exists (else a
  derived `model.xml` with `import.md`), their meshes, and its authoritative ROS interface as `ros.yml` (ROS 1) or `ros2.yml` (ROS 2): every
  node, topic, service, action and parameter the authoritative sources define, with types,
  frames, rates and the robot's stop command. Rows that exist only when an optional plugin
  is installed or a launch other than the robot's authoritative boot runs are marked
  `optional`; each camera's raw image stream is marked as a camera.
* Every robot is a single body: no robot is assembled from other robots (amended
  2026-10-02: the myAGV + myCobot 280 composite is no longer supported).
* Robot meshes are not committed. Each engine's `simulator/<engine>/run.sh setup` fetches
  them into `robots_specs/<id>/` from each robot's pinned sources, verifies them against
  `robots_specs/meshes.sha256` and refuses on a mismatch; `spawn.sh` refuses a robot whose
  required files are missing, naming `run.sh setup`.

## 3. Evidence

`./evidences`, produced by the workspace evidence script in `tests/`, holds evidence that
the workspace works, with an index,
`evidences/index.md`, listing every case and its result. A **case** is one recorded robot
spawned with `simulator/spawn.sh` into one engine's simulation, started by
`simulator/<engine>/run.sh start` or by `simulator/kitchen.sh start`, on the engine's
default scene. Every robot has a case on each engine; a mobile robot (every `kind` but
`arm`) is spawned on the floor, an arm on the worktop. Every case has:

* a picture of the scene with the robot in it — an offscreen render of the simulation
  stands in for a window. For an arm on the worktop the picture also shows the worktop
  objects staged around it (simulator spec §2.3), and the case records which objects were
  staged and which scene objects were cleared for them (amended 2026-10-01);
* a picture from each of the robot's cameras not marked `optional`;
* a passing smoke run, through the robot's vendor interface on its own wire, of each
  motion listed below using its recorded command. A drive or walk is followed by the
  robot's recorded stop command; every other motion ends
  on reaching its goal or end state. A motion passes when the recorded feedback shows the
  commanded displacement within the simulator's acceptance bounds (simulator spec §5) and
  the robot comes to rest after the stop command or, for other motions, once it has reached
  its goal or end state. Where the recorded interface publishes no measured feedback for a
  motion, joint or pose readings taken through the simulation's private control port
  before and after judge it; the commands themselves always go through the vendor wire.
  Base motions stay within the travel the placement guarantees:
  * a wheeled base drives forward, back, sideways and turns, confirmed by its odometry;
  * the ROSMASTER X3 PLUS also moves its arm and its gripper;
  * the SO-101 moves its arm and its gripper;
  * the AiNex turns its head, walks and stops, and plays an action group;
  * the myCobot 280 moves its arm and its gripper;
* for the console profile whose id names this robot, if any, a passing
  `python -m robot_console.fleet --expect <id>` over the case's wire:
  the profile's typed validation passes and every profile camera not marked `optional` shows
  live images. A
  failure is resolved against the pinned sources; neither project is the other's oracle.

A missing or failing case is a defect. Workspace tests, in `tests/`, check that the index
has a passing case, with its pictures, for every robot id in the robot registry (§2) on
every engine, and that every worktop case records its
six staged objects.
