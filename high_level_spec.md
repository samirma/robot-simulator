# Robot Simulator — High-Level Specification

## 1. Goal

Simulate real robots — the ones `robots_specs/robots.yml` records (§2): mobile bases,
mobile manipulators, a humanoid and an arm — inside realistic household scenes, and present
each one to clients through **the same
network interface the real hardware presents**: a simulated robot presents exactly the
interface its authoritative source defines.
One console can then teleoperate, map and navigate a simulated robot or a physical one
unchanged, and task a simulated one: a client written against the authoritative sources works on either
(applying standard ROS namespace composition on a namespaced simulator wire). What the cameras see —
the scene itself — is content, not interface, and differs between simulator and hardware
and between engines by design.

Here, a robot's **vendor interface** excludes workspace-owned facilities around it. The
simulator may additionally expose rosbridge discovery, the fixed worktop camera rig and
the simulation-only reset service that `simulator/high_level_spec.md` §3 defines. Those
extensions have distinct owners, are discoverable rather than assumed, and may identify a
wire as simulated; they must not alter or replace any vendor-owned name or behavior.

This document states the shared goal and project boundary only. Everything specific to one
project — its components, entry points, interfaces, constraints and checks — is in that
project's specification, which must not contradict this one. No other document or code in
the workspace may contradict either level; where code does, the code is wrong.

The robot records in `robots_specs/` (§2) and the source-precedence rules in
[`simulator/high_level_spec.md`](simulator/high_level_spec.md) §3 are normative for this
workspace. The console keeps an independent copy of only the contract facts it consumes,
and both projects' verification holds those copies equal. This is a source-level agreement
mechanism, not a runtime dependency: installed projects still communicate only through
rosbridge.

## 2. Projects

The workspace is two **independent** projects whose installed production code talks only
over rosbridge and never imports the other project. Workspace-only conformance tests may
inspect both source trees. Each project has its own high-level specification, which lists
its user-facing entry points:

| Project | Responsibility | Specification |
| --- | --- | --- |
| `simulator/` | Host robots in a choice of physics **engine** (MolmoSpaces or RoboCasa; simulator spec §2.1) and serve each one's vendor ROS interface on one rosbridge websocket. | [`simulator/high_level_spec.md`](simulator/high_level_spec.md) |
| `robot_console/` | Drive, map, navigate and grade compatible robots, simulated or physical. All control policy lives here. | [`robot_console/high_level_spec.md`](robot_console/high_level_spec.md) |

Robot embodiments are recorded once, under `robots_specs/`, and both projects'
specifications refer to them rather than restating them:

* [`robots_specs/robots.yml`](robots_specs/robots.yml) — one entry per robot: its **id**,
  identity, placement, whether the simulator hosts it (`simulated`), official URL, pinned sources, documentation URLs, and the paths of
  its URDF, MJCF and ROS interface files. Every `--robot`/`--robots` flag in either project takes
  these ids, and a robot's id is its default ROS namespace on a simulator wire. Every such
  flag's `--help` lists the ids it accepts, each with its name, read from `robots.yml` at
  run time rather than typed into the code, and the same list appears in the message that
  refuses an unknown id.
* `robots_specs/<id>/` — the robot's URDF, its official MJCF where one exists, their
  meshes, and its official ROS interface as
  `ros.yml` (ROS 1) or `ros2.yml` (ROS 2): every topic, service, action and parameter the
  official sources define, with types, rates and the robot's stop command. A composite
  robot's file `extends` its base robot's and lists only the additions.
