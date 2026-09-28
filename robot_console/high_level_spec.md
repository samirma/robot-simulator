# Robot Console — High-Level Specification

Part of the workspace specification in [`../high_level_spec.md`](../high_level_spec.md),
which defines the shared goal and project boundary this document builds on. This document
states the console's requirements only; no other document or code in the console may
contradict this document or the workspace specification.

## 1. Goal

Drive, map and navigate the compatible robots `robots_specs/robots.yml` records over
rosbridge,
identically whether they are simulated or physical and whichever engine hosts them. Grade
the SO-101 task in simulation (§2.3). **All control policy lives here.**

The console never starts a simulator. It connects to whatever serves the contract, which
may be `simulator/` ([`../simulator/high_level_spec.md`](../simulator/high_level_spec.md))
or real hardware.

Every entry point takes `--url ws://<host>:<port>` to choose its rosbridge wire,
defaulting to `ws://127.0.0.1:9090`. The **launchers** are the shell entry points
(`bin/teleop.sh`, `bin/slam.sh`, `bin/view.sh`, `run_task.sh`).

| Entry point | Responsibility |
| --- | --- |
| `robot_console/bin/teleop.sh` | Keyboard teleoperation of a mobile robot (every `kind` but `arm` in `robots.yml`), with live camera and optional recording. |
| `robot_console/bin/slam.sh` | 2D occupancy-grid mapping (`explore`, `map`) and goal navigation (`navigate`) for a myAGV. |
| `robot_console/run_task.sh` | Run the SO-101 `apple_on_plate` task (§2.3) with a VLA policy over N episodes, and grade it. |
| `robot_console/bin/view.sh` | Browser page showing the cameras of whatever is on a wire, with per-robot controls. |

## 2. Components

### 2.1 Teleoperation — `bin/teleop.sh`

```sh
teleop.sh [--robot <id>] [--namespace <ns>] [--url ws://…] [--record <dir>]
          [--speed <m/s>] [--max-speed <m/s>] [--latch]
```

* **Keys.** The **motion keys** are:
  * `W`/`S`: forward and back;
  * `A`/`D`: strafe left and right (the wheeled bases are omnidirectional);
  * `Q`/`E`: rotate.

  The other keys are: Space stops, Esc quits, `+`/`-` change speed, `H` shows help. For
  the AiNex, the arrow keys turn the head and `0` centres it.
* **Speeds.** `--speed` is the initial linear speed and `--max-speed` its cap.
  * The myAGV defaults to 0.15 m/s with a 0.28 m/s cap, and `+`/`-` step by 0.05 m/s.
    Rotation runs at `TURN_RATIO` (2.0) rad per metre of the linear speed.
  * The myAGV + myCobot 280 drives with the myAGV's envelope: it is a myAGV underneath.
  * The ROSMASTER X3 PLUS defaults to 0.20 m/s with a 0.70 m/s cap (its board's input
    range), in steps of 0.05 m/s. Rotation runs at a `TURN_RATIO` of 5.0, capped at
    3.2 rad/s.
  * The AiNex defaults to 0.10 m/s with a 0.20 m/s cap, in steps of 0.02 m/s. The speed
    maps linearly onto the step amplitude in `/walking/set_param`. Rotation runs at a
    `TURN_RATIO` of 4.0, capped at 1.0 rad/s.
* **Latching.** With `--latch`, a motion key keeps the robot moving until another motion
  key, Space or Esc. Without it, teleop sends the stop command once no motion-key event
  has arrived for 0.6 s (the OS's auto-repeat keeps a held key's events coming).
* **Safety supervisor.** A separate supervisor process owns the rosbridge connection and
  every motion publication. The UI sends desired commands plus a heartbeat
  over local IPC. If the heartbeat, parent process or IPC disappears for the **safety
  timeout** of 0.25 s, the supervisor sends the robot's stop command three
  times, 50 ms apart, before closing. This applies with and without `--latch`; a frozen UI
  therefore cannot leave the last command running. Before enabling a non-zero command,
  teleop requires the operator to confirm that an independent physical emergency stop or
  motor-power dead-man is armed. This independent device is the required protection for
  host failure or network loss, which software on the failed path cannot stop.
* Robot and namespace default to **discovered from `/rosapi`**.
  `--namespace ''` asks for the bare contract on purpose. `--robot` takes a robot id from
  [`../robots_specs/robots.yml`](../robots_specs/robots.yml); teleop drives every robot
  there whose `kind` is not `arm`. After `--robot` and `--namespace` narrow the discovered
  robots, exactly one must remain. Robots sharing a command topic (`/cmd_vel`) are told
  apart by a topic only one of them has. None, or more than one, is an error that names
  what was found.
* Each robot is driven, and the AiNex's head moved, only through the official interface
  in its ROS file (`robots_specs/<id>/ros.yml` or `ros2.yml`).
* A robot's **stop command** is the `stop_command` in its ROS file. Teleop sends it on
  **every** exit path (Esc, window close, exception, `SIGINT`/`SIGTERM`), because none of
  these robots has a watchdog.
* `--record` writes every decoded camera frame as received (`feed.mp4`) and every command
  (`commands.jsonl`).
* Self-installs its venv on first run and when `pyproject.toml` changes.

### 2.2 Mapping and navigation — `bin/slam.sh`

```sh
slam.sh explore  --out <map-dir> [--namespace <ns>] [--url ws://…]  # autonomous frontier exploration
slam.sh map      --out <map-dir> [--namespace <ns>] [--url ws://…]  # teleop with the map building live
slam.sh navigate --map <map-dir> [--namespace <ns>] [--url ws://…]  # click a goal, drive there
```

`explore` additionally accepts `--max-duration <s>` (default 3600) and `--max-goals <n>`
(default 500).

* Occupancy-grid SLAM on `/scan` and `/odom` of a myAGV. After `--namespace` narrows the
  myAGVs discovered through `/rosapi`, exactly one must remain. Other robots on the wire are
  ignored.
* Maps are saved as `map.pgm` + `map.yaml` in `map_server` format, loadable by the vendor's
  `myagv_navigation/launch/navigation_active.launch`. Beside them goes `map.npz`, a sidecar
  holding the grid's occupancy values at full precision and the robot's last pose. A later
  `map` or `explore` run whose `--out` names the same directory continues the saved map. It
  assumes the robot starts at that stored pose.
* The grid's resolution is 0.05 m per cell.
* `explore` chases **frontiers**: clusters of free cells bordering unknown space. A goal
  **progresses** when the robot comes 0.45 m closer to it than its closest approach so
  far, or uncovers 0.25 m² of map since the goal's last progress. A goal is blacklisted
  after 90 s without progress or 300 s total, and when it is reached without uncovering any map. A goal
  reached without uncovering any map is never chosen again. Each progress either shortens
  its distance or grows the observed map. When no frontier qualifies, it climbs a **give-up
  ladder**, trying each rung only if the previous one found nothing:
  1. frontiers of ≥ 6 cells;
  2. frontiers of ≥ 3 cells;
  3. once per run: clear the blacklisted goals that timed out (reached-but-fruitless goals
     stay), and frontiers of ≥ 6 cells again;
  4. frontiers of ≥ 1 cell;
  5. unknown holes fully enclosed by observed cells;
  6. once per run: one rotation on the spot, then back to rung 1.

  A goal from any rung restarts the ladder from rung 1. `explore` ends with `explored` when
  rung 5 finds nothing and rung 6 has already run. It ends with `limit` when either hard
  run limit is reached, saves the current map in both cases, and reports elapsed time and
  attempted goals. A goal's 300 s deadline plus the two hard run limits make termination
  an enforceable bound rather than relying on assumptions about map finiteness.
* `navigate` plans with unknown space treated as blocked; `explore` treats it as free.
* Every `slam.sh` mode uses the safety supervisor from §2.1, requires the same independent
  emergency stop, and sends the myAGV's stop command on every controlled exit path.

### 2.3 The arm task — `run_task.sh`

```sh
run_task.sh [--episodes N] [--label <engine>] [--url ws://…] [--robot <id>]
            [--namespace <ns>] [--instruction <text> | --instruction-file <f>] [-- <inspect-robot args>]
```

* **`apple_on_plate`** is the task the simulator stages (simulator spec §2.3): a red apple
  and a white plate on the worktop in front of the SO-101. An episode passes when, for at
  least 1.0 s at its end, the camera-verdict scorer measures all of these conditions: the
  apple centre is within 0.08 m of the plate centre horizontally and within 0.015 m of its
  resting height; its speed is at most 0.01 m/s; and every gripper finger is at least
  0.05 m from the apple centre. The last condition establishes release for this observable
  acceptance criterion rather than accepting an apple held at the expected pose. An episode ends when the policy reports the task
  done to `inspect-robot`, or after 220 policy steps.
* Runs against a simulator someone else started; it launches no engine. It requires the
  simulation-only `/reset` service and the rig.
  It refuses, with a message, a wire lacking either, so the arm task does not run on
  hardware.
* Flags:
  * `--episodes` defaults to 1. A one-episode run is a smoke run, not a result.
  * `--robot` names, by the id of a `simulated` robot in `robots_specs/robots.yml`, the
    member expected besides the SO-101 and the rig (default `so101`, which expects the
    SO-101 alone). The SO-101 is always required, since `/reset` and the `so101_ros`
    embodiment exist only with it. Any other id is refused with a message listing the
    accepted ones.
  * `--namespace` is the SO-101's, defaulting to discovered.
  * `--label` names the engine in the report, and the log subdirectory
    `runs/task/<label>/`, one run directory per episode holding the framework's log and
    `episode.log`.
  * `--instruction` is the natural-language instruction the policy is given, and
    `--instruction-file` reads it from a file. The default is "Move the arm towards the red
    apple, grasp it, lift it up, and place it on the white plate."
  * A one-episode run prints that episode's verdict, labelled a smoke run.
  * Arguments after `--` go to `inspect-robot` unchanged.
* In order:
  1. pick the venv the policy needs (§3);
  2. wait for the members' **topics**, not just the port;
  3. check every expected member's typed interface is on the wire;
  4. per episode, call the SO-101 namespace's composed `/reset` (on `success: false`, abort the run with its message and
     report the episodes completed so far), run `inspect-robot` with the `molmoact2` policy on the
     `so101_ros` embodiment, and grade that episode's log with the camera-verdict scorer.
* **The framework.** `inspect-robots` is the evaluation framework the console depends on,
  and `inspect-robot` is its command-line tool. An **embodiment** is the framework's
  binding of one robot's observations and actions to a transport. `molmoact2` is a VLA
  (vision-language-action) policy. The console registers four entry points with the
  framework, in its groups `inspect_robots.tasks`, `inspect_robots.policies`,
  `inspect_robots.embodiments` and `inspect_robots.scorers`: the task `apple_on_plate`, the
  policy `molmoact2`, the embodiment `so101_ros`, and the camera-verdict scorer
  `apple_on_plate`. It ships no console script of its own. The framework's packages
  are pinned to exact versions in the console's `pyproject.toml`.
* **The `so101_ros` embodiment** observes the rig's overhead and side colour images, the
  SO-101's wrist camera and `/joint_states`, and commands the arm and gripper through the
  controllers in `robots_specs/so101/ros2.yml`.
* The **camera-verdict scorer** applies the pass criterion above using synchronized
  overhead and side colour frames, both cameras' `camera_info`, the rig mount poses and
  `/joint_states`. All are public observations available to the policy; nothing on the
  wire answers the task's question directly.
  * **Apple pose.** The scorer triangulates the segmented apple centroid from the two
    calibrated views. It derives speed from timestamped consecutive poses rather than
    image stillness alone.
  * **Release.** Forward kinematics from `/joint_states` places both fingers in the same
    rig frame. The scorer requires the finger clearance above throughout the 1.0 s hold,
    so an apple held stationary over the plate fails.
  * **Decision.** Missing synchronization, calibration, segmentation, joint state or any
    part of the full hold interval fails closed. Thresholds and synchronization tolerance
    are named constants in `robot_console/arm/vision_success.py` and are duplicated in
    scorer tests, not inferred from simulator-private state.
* Results are pass counts over N episodes; a single episode is never reported as a result.

### 2.4 Camera and control page — `bin/view.sh`

```sh
view.sh [--url ws://…]
```

* One static page speaking rosbridge; it is told nothing about the wire but its URL.
  Members and cameras are discovered from `rosapi`, cameras in both
  dialects' image types.
* Each robot's controls publish nothing until the user enables that robot's control on the
  page. They command the robot only through its interface (topics, services and actions).
* The page offers no control that starts sustained motion: no myAGV drive and no AiNex
  walk. A browser tab cannot guarantee a stop command on every exit path, and
  `bin/teleop.sh` can. It offers only bounded commands: a head position, an arm trajectory,
  a gripper position command and any bounded action the discovered vendor interface offers.
* On unload, the page cancels every unfinished action goal it sent. After a reconnect, it
  re-sends every subscription and advertisement. A goal whose client disconnected without
  cancelling it runs on.

## 3. Constraints

The console has these constraints:

* **Installs and runs with no MuJoCo, no MolmoSpaces and no `simulator/` checkout.** Base
  runtime dependencies are `numpy`, `opencv-python` and `roslibpy`. Optional dependency
  groups (extras) add the test tools (`dev`: `pytest` and the fake server's `websockets`),
  the arm's `inspect-robots` support (`arm`) and the VLA's dependencies (`vla`). None
  imports anything from the simulator.
* **Venvs.** The console has two:
  * `.venv` — the base plus the `dev` and `arm` extras; it runs everything except the VLA;
  * `.venv-vla` — the base plus the `arm` and `vla` extras; it runs `run_task.sh` with the
    `molmoact2` policy.
* Tests use an in-process fake rosbridge server of the console's own.
* In the interactive components (`teleop.sh` and every `slam.sh` mode), the UI loop owns
  the main thread and never publishes motion directly. The safety supervisor in §2.1 is
  the sole motion publisher and treats missing UI heartbeats as a stop request.
* The offline test suite runs with no robot, no display and no network; tests needing a
  live rosbridge are opt-in (`-m live`), and no test opens a window.

## 4. Verification

The project must have comprehensive unit tests ensuring that all required components and
their behaviour align with this specification.

* **Contract parity** — workspace tests compare every console-owned topic, service, action,
  type, namespace composition rule, joint definition, camera calibration and task constant
  with the normative simulator contract modules, which transcribe the ROS files in
  `robots_specs/`. These comparisons may read a sibling
  checkout by path, but the installed console and its ordinary tests do not require one.
* **Discovery** — fake-bridge tests cover a lone bare robot, a namespaced robot, a mixed
  fleet (every `/cmd_vel` base on one wire among them), missing distinguishing topics,
  wrong types, duplicate candidates and an unreachable `/rosapi`. Every ambiguous case must fail with the candidates found.
* **Motion safety** — subprocess tests freeze the UI, close its IPC, terminate it normally,
  raise an exception and send `SIGINT` and `SIGTERM`. A fake bridge must receive the first
  stop no later than the safety timeout plus 100 ms and three stops in total. An opt-in live
  drill verifies that the independent emergency stop halts motion after network loss.
* **SLAM** — deterministic scan/odometry fixtures verify map persistence, planning modes,
  each give-up rung and goal blacklisting. Separate tests hold a goal in continuous
  progress and continuously generate frontiers; both runs must still finish with `limit`
  at the configured duration or goal count and save a loadable map.
* **Task grading** — recorded synchronized observations cover a valid placement and, as
  mandatory negative cases, an apple held by the gripper at the target pose, an apple
  moving through the target, the wrong height, incomplete hold time, missing frames and
  stale joint states. Only the valid released placement passes. A separate audit compares
  camera verdicts with simulator ground truth without exposing that truth to the scorer.
* **View page** — browser tests verify typed discovery, opt-in controls, bounded commands,
  cancellation on unload and complete re-advertisement and re-subscription after reconnect.
* **Compatibility** — opt-in live tests run the same discovery, command, stop and
  observation checks against each simulator engine and available physical hardware. No
  client branch may select behavior from an engine identity.
