# Simulator — High-Level Specification

Part of the workspace specification in [`../high_level_spec.md`](../high_level_spec.md),
which defines the shared goal and project boundary this document builds on. This document
states the simulator's requirements only; no other document or code in the simulator may
contradict this document or the workspace specification.

## 1. Goal

Host every robot marked `simulated` in
[`../robots_specs/robots.yml`](../robots_specs/robots.yml) inside realistic household
scenes, in a choice of physics **engine**, and serve them on one rosbridge websocket. Each
simulated robot behaves as the real robot its manufacturer describes, as `robots_specs/`
records it (§3, real-robot fidelity), identically on every engine. Scene content differs
between engines by design.

The simulator contains no control policy; that belongs to `robot_console/`
([`../robot_console/high_level_spec.md`](../robot_console/high_level_spec.md)).

| Entry point | Responsibility |
| --- | --- |
| `simulator/<engine>/run.sh` | Per-engine setup, asset fetching, repair, and viewing a single robot in a scene without serving a wire (`setup`, `assets`, `view`, `repair`). |
| `simulator/kitchen.sh serve` | Load one engine's scene, stage `apple_on_plate` when a worktop robot is present, spawn a fleet and serve it on one port. The only entry point that serves a wire. |

## 2. Components

### 2.1 Engines — `simulator/<engine>/run.sh`

```sh
run.sh setup                                         # venv, upstream checkout, default assets
run.sh assets [<source>]                             # optional bulk pre-fetch for offline use
run.sh view --robot <id>   [<engine's scene flags>]  # a viewer window; serves no wire
run.sh repair                                        # re-point venv and assets after a move
```

Two interchangeable engines must exist, `molmospaces` and `robocasa`. Each has its own
launcher (`run.sh`), venv, environment script (`env.sh`) and robot-spawning tool
(`tools/spawn_robot.py`). An engine's **scene flags** are the ones §2.3 marks as that
engine's. `assets` fetches every asset source the engine knows, or only the named
`<source>`, so that later runs need no network.

* **MolmoSpaces** (the default engine) — MuJoCo with iTHOR and ProcTHOR houses. Scenes are
  installed on demand, so a bare `setup` suffices for `--scene ithor:1`.
* **RoboCasa** — MuJoCo with RoboCasa kitchens, used as a **scene provider only**: no
  robosuite robot, controller or observation stack enters the model.

Each engine must spawn every robot marked `simulated` in `robots_specs/robots.yml`, and present all of them through the
shared layer (§2.2). Only the engine actually used needs to be set up. A moved or copied
checkout must repair itself on the next run of any entry point.

### 2.2 Shared layer — `simulator/shared/`

Everything that is not specific to one engine exists exactly once:

* **Transport** — one in-process rosbridge server (`contracts/rosbridge_server.py`), with
  no MuJoCo dependency. It accepts every ROS 1 and ROS 2 rosbridge operation required by
  the served interfaces, and carries ROS 1 and ROS 2 message dialects side by side.
  * A latched topic is delivered as latching delivers it: each publisher's last message,
    once to each new subscriber, never republished.
  * A client's action goal is cancelled only by `cancel_action_goal`, or aborted by
    `/reset`; a client that disconnects leaves its goals running.
* **ROS surfaces** (`ros_surfaces/`) — per member, the authoritative vendor interface it
  presents and the loop that feeds it. An engine supplies only an adapter that places
  each robot in its scene.
* **Tasks** (`tasks/`) — the objects and the rig a task stages into a scene,
  engine-neutral. Grading belongs to the console.
* **Robot models** (`robots/`) — the engine-loadable model of each robot, built from the
  URDF, MJCF and meshes in its `robots_specs/<id>/` folder. What the simulator adds (an
  MJCF where no official one exists, actuators, collision simplifications) derives from
  those files and never changes their geometry, joints or limits.
* **Transform trees** — `/tf`, `/tf_static` and `/robot_description` exactly where a
  member's interface has them, each produced as its source's node produces it:
  `robot_state_publisher` over the published description and `/joint_states`, the myAGV's
  static transforms as its boot launch's `static_transform_publisher`s send them, and its
  `odom`→`base_footprint` from `robot_pose_ekf`'s fusion. MuJoCo's forward kinematics
  feeds only the simulated sensors and joint states. The rig's static frames come from its
  source's camera geometry.
* **The rig's mount** — the overhead and side cameras are staged at the mount poses over
  the worktop recorded in both projects' contract constants.

### 2.3 Serving a world — `kitchen.sh serve`

```sh
kitchen.sh serve [--engine molmospaces|robocasa] [--robots <id>[,<id>…]]
                 [--ros-namespace <ns>] [--mujoco] [--port <p>]
                 [--scene <s>]                  # MolmoSpaces scene flag
                 [--layout N --style N]         # RoboCasa scene flags
                 [<staging flags>]
```

`--robots` takes a non-empty set of ids of `simulated` robots in `robots_specs/robots.yml` (default `so101`);
`--port` defaults to `9090`. `--scene` takes `ithor:<n>` or `procthor:<n>`. An engine
refuses the other engine's scene flags by name, and `serve` refuses any flag its `--help`
does not list. Each
engine's **default scene** is MolmoSpaces `ithor:1` or RoboCasa layout 1 style 1.

* One engine per run (default `molmospaces`). Every robot in `--robots` shares **one scene,
  one port and one ROS graph**, each under its own ROS namespace.
* The **worktop** is the counter or table surface the task is staged on: the one the
  engine's scene marks as the task surface. The **worktop robots** are those whose
  `placement` in `robots.yml` is `worktop`, placed so the staged objects are within their
  reach; the others stand on the floor.
* When a worktop robot is in `--robots`, `apple_on_plate` (an apple and a plate) is staged
  on the worktop in front of them, and the rig is served. A fleet with no worktop robot
  gets the room, its robots and their cameras, with no task and no rig. A worktop robot in a scene
  with no worktop is a start-up error.
* The rig is presented exactly as the shared SO-101 and rig contract constants define it,
  under the rig's own namespace rather than a robot's.
* `/reset` is served whenever the SO-101 is. A fleet with the task but
  no SO-101 has no `/reset`.
* Every served member presents all its cameras; no flag removes one. Cameras render inside
  the physics loop, so each one costs rate for every topic on the port. A swept-set
  configuration (§4) that fails the rate gate in §5 is a defect.
* `--ros-namespace <ns>` puts a lone robot under `<ns>` instead of its name, and refuses an
  invalid ROS namespace or one that collides with another provider. `''` serves the bare
  vendor interface. Both are refused with more than one robot; the rig does not
  count.
* Headless by default. `--mujoco` opens a MuJoCo window **in the serving process**, since
  only the process holding the physics can draw it; closing the window ends the run.
* Every flag `serve --help` lists that the synopsis above does not name is a **staging
  flag**, e.g.
  `--reference-table`, `--no-dressing`, `--reference-lighting`, `--extra-lights`,
  `--swap-objects`. Each has a default, which `--help` shows, and changes
  only the staged world, never the wire. No staging flag changes the task objects' sizes
  or colours: every engine stages the apple and plate the shared task constants describe.
  `--swap-objects` exchanges the two objects' staging poses, and the
  constants record the poses for both of its settings.
* At start-up, when the SO-101 is served, `serve` reports whether each staged object is inside the SO-101's reach: a
  top grasp at its position has an inverse-kinematics solution within the joint ranges of
  the compiled model.

## 3. Wire interfaces

This section is the workspace's normative provider-side wire record. The console duplicates
only the facts it consumes and may not redefine them.

* **Authority and precedence** — a robot's interface record is its ROS file in
  `robots_specs/<id>/`, and its sources and revisions are its entry in `robots.yml`. The
  contract modules (`ros_surfaces/myagv.py`, `ros_surfaces/so101.py` for the SO-101 and
  rig, and `ros_surfaces/ainex/topics.py` under `simulator/shared/`) transcribe the names,
  types, frames and periodic rates the simulator serves from those files, as bare vendor
  names composed only where a name reaches the wire; they do not restate source metadata.
  Authority is resolved in this order:
  1. names, types and behaviour captured from the physical robot's complete boot launch
     at the revision `robots.yml` records;
  2. that revision's launch files, node sources and controller configuration, and the
     manufacturer's documentation;
  3. the robot's URDF and MJCF in `robots_specs/<id>/` for geometry, joints and
     transforms not settled above;
  4. this specification for workspace-owned `/reset`, the rig and task constants.

  A lower source never overrides a higher one. Where `robots_specs/` disagrees with a
  higher source, `robots_specs/` is corrected first and the contract modules follow;
  changing a recorded revision requires a reviewed interface diff.
* **Transport protocol** — one websocket carries rosbridge protocol 2.0 JSON. The server
  accepts `advertise`, `unadvertise`, `publish`, `subscribe`, `unsubscribe`,
  `call_service`, `advertise_service`, `unadvertise_service`, `set_level` and `status`.
  For ROS 2 actions it also accepts `advertise_action`, `unadvertise_action`,
  `send_action_goal` and `cancel_action_goal`, and emits action feedback, status and
  result messages. Unsupported or malformed operations receive a rosbridge `status`
  error. ROS 1 names and headers retain ROS 1 spelling and shape; ROS 2 names and headers
  retain ROS 2 spelling and shape.
* **Workspace-owned extensions** — these are outside every vendor interface and are the
  only additions a client may use to identify simulation:
  * `/reset`, composed with the SO-101 namespace, is a `std_srvs/srv/Trigger` service
    provided by `/simulator`. It restores the complete staged world and controller state,
    aborts outstanding goals, clears partial task timers and returns only after reset
    observations are available.
  * The fixed rig owns namespace `/scene`. It publishes
    `/scene/overhead/color/compressed` and `/scene/side/color/compressed` as
    `sensor_msgs/msg/CompressedImage`, `/scene/overhead/color/camera_info` and
    `/scene/side/color/camera_info` as `sensor_msgs/msg/CameraInfo`, and its calibrated
    transforms on `/scene/tf_static`.
    Camera poses, intrinsics, dimensions and rates are the constants in
    `shared/tasks/apple_on_plate.py`, duplicated exactly by the console.
  * No task verdict, engine name, contact state or other simulator-private state is
    published. In particular, there is no task-success topic.
* **Nothing engine-internal on the wire** — every name on the wire is in a member's
  interface, is a `/rosapi/*` service, is `/reset` or its
  node `/simulator`, or is one of the protocol-defined ROS runtime names
  (`/rosapi`, `/rosbridge_websocket` and clients' advertisements). In particular, a model's MJCF
  prefix never does.
* **One provider per name among members** — no two members provide the same service or
  action name, and a fleet that would is a start-up error. Multiple publishers may share
  a topic only where the authoritative source graph does. Clients' own advertisements
  follow the transport operations above.
* **Real-robot fidelity** — every simulated robot follows the behaviour of the real robot
  as its manufacturer describes it and `robots_specs/` records it:
  * it presents exactly the official interface in its ROS file: every name, type, frame,
    parameter and periodic rate;
  * it answers every command as the manufacturer's sources say the real robot does,
    including its `stop_command` and the absence of any command watchdog;
  * its compiled model reproduces the geometry, joints and limits of its URDF and MJCF,
    and the manufacturer's published physical figures.

  A violation is a regression even when nothing fails.
* **Engine indistinguishability** — a client must not be able to tell from the interface
  which engine hosts a member. Run with the same `--robots` and `--ros-namespace`, both
  engines must present:
  * identical node, topic, service and action names and types, including topics no console
    component consumes, such as depth;
  * identical frame ids and transform-tree structure;
  * identical parameter names and values, `robot_description` included;
  * in the swept set (§4), every periodic topic passing the rate gate in §5.
* **Discovery** — the bridge answers these 31 `rosapi` services:
  * the ROS 1 `rosapi_node`'s 25: `topics`, `topics_for_type`, `topics_and_raw_types`,
    `topic_type`, `services`, `services_for_type`, `service_type`, `service_providers`,
    `service_node`, `service_host`, `nodes`, `node_details`, `publishers`, `subscribers`,
    `action_servers`, `message_details`, `service_request_details`,
    `service_response_details`, `get_param_names`, `get_param`, `set_param`, `has_param`,
    `search_param`, `delete_param`, `get_time`;
  * the ROS 2 `rosapi_node`'s 6 more: `interfaces`, `action_type`, `action_goal_details`,
    `action_result_details`, `action_feedback_details`, `get_ros_version`.

  Topic, service and action answers come from the server's own tables. Message, service
  and action details come from the vendors' definitions, with recorded provenance. Node
  answers name the node that provides each name in the member's source, composed. The
  `/rosapi/*` services' node is `/rosapi`, and clients' advertisements belong to
  `/rosbridge_websocket`. Parameter values are JSON-encoded.

## 4. Constraints

The simulator has these constraints:

* Each engine has one venv, built on Homebrew's framework Python 3.11, so that MuJoCo's
  `mjpython` launcher, which the viewer needs on macOS, works.
* **Real time** — the **swept set** is every `kitchen.sh serve` run that:
  * combines one engine with a non-empty set of `simulated` robot ids;
  * uses that engine's default scene;
  * runs headless, with default namespaces and every staging flag at its default.

  Each of those runs in real time on the **reference host**, a
  MacBook Pro with an Apple M4 Max and 36 GB. Anything else is best-effort: another host,
  another scene, a changed flag, `--mujoco`. `serve` warns when its real-time factor over
  a 10 s window falls below 0.90.
* On macOS the MuJoCo viewer must own the main thread.
* Vendored upstreams (`molmospaces/upstream/`, the robosuite and robocasa checkouts) are
  never modified; out-of-tree robots and scenes are added from outside them. `assets/` is
  generated and never holds curated files.
* Shared logic exists once. There is one transport, one surface loop per member, one task
  definition, one transform-tree loop, and one function that stands a robot on the floor or
  the worktop. Two copies are two chances to drift.

## 5. Verification

* **Contracts** — `simulator/shared/contracts/test_fleet.py` checks every contract module
  against its robot's ROS file in `robots_specs/` (names, types, frames and every periodic
  rate), exercises every
  transport operation in §3, calls every `rosapi` service over the wire, checks namespaces
  and providers on a multi-member fleet, and checks every published message against its
  declared type field for field. Workspace parity tests compare every contract fact the
  console duplicates, including namespace composition, task geometry and rig calibration.
* **Engine indistinguishability (needs both halves)** — `python -m robot_console.fleet
  --dump` against the same `kitchen.sh serve --robots …` on both engines yields identical
  output.
* **Robot models** — these checks:
  * `shared/tests/tf_frames_check.py`: every published robot tree against MuJoCo forward
    kinematics of the compiled model (myAGV,
    SO-101);
  * `shared/tests/ainex_ground_check.py`: sole on the surface, stable over consecutive
    physics steps, falls when unsupported;
  * `shared/tests/ainex_grasp_check.py`: the simulator's AiNex action groups `crawl_left`
    and `crawl_right`, played as `/app/set_action` plays them, bring a hand geom into
    contact with the staged apple and move it, so the AiNex can act on the task;
  * `shared/tests/ainex_provenance_check.py`: vendored files are byte-for-byte the
    vendor's;
  * each engine's `robots/<robot>/test_attach.py`: the robot's model attaches to the
    engine's scene and compiles;
  * each engine's `tools/test_placement.py`: every robot stands at its placement, on the
    floor or the worktop, without interpenetration;
  * a physical-figures check: the compiled model reproduces every physical figure in the
    contract constants within its tolerance.
* **Rate (needs both halves)** — `python -m robot_console.fleet --rates --gate` measures every
  periodic topic and the real-time factor on the reference host over the swept set (§4).
  After a 5 s warm-up it observes each run for the greater of 30 s or five periods of its
  slowest declared topic. For each topic, the expected Hz comes from its robot's ROS file
  (§3); measured Hz must be within ±10% and no inter-message gap may exceed
  three expected periods. The mean real-time factor must be in `[0.90, 1.10]` and no
  rolling 10 s window may fall below `0.90`. A missing rate declaration, missing periodic
  topic or insufficient observation window fails rather than being skipped.
* **Scene changes (needs both halves)** — any change to the scene, the staging, the
  lighting or a camera is reported with a matched pair of `run_task.sh` pass counts
  produced by the console. A matching image statistic alone is not evidence that the policy can act in the
  scene.
