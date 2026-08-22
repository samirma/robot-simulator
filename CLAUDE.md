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

`simulator/` is not one simulator but **three interchangeable engines plus a shared
layer**:

```
simulator/
  shared/       cross-engine resources: the wire bridge and the robot specs
    contracts/    rosbridge_server.py (myAGV topics) + control_server.py (SO-101) — pure transport, no MuJoCo
    robots/       so101/ + myagv/ hardware specs (MJCF, meshes, URDF) — engine-neutral
    mujoco_bridge.py  MuJoCo→wire helpers shared by the MuJoCo engines (imports mujoco)
  molmospaces/  engine #1 — MuJoCo + MolmoSpaces (iTHOR / procthor houses)
  robocasa/     engine #2 — MuJoCo + robosuite + RoboCasa (kitchens)
  coppeliasim/  engine #3 — CoppeliaSim (ZMQ remote API)
```

Each engine has its own `run.sh`, `env.sh`, `tools/spawn_robot.py`, and `uv` venv, and each
must spawn at least `so101` and `myagv`. **`robot_console` connects and controls a robot
identically regardless of which engine hosts it — because every engine presents the *real
hardware's* interface** by feeding the one shared bridge in `simulator/shared/contracts/`:
the myAGV on its vendor `elephantrobotics/myagv_ros` rosbridge topics, the SO-101 on the
`molmospaces-control-v1` msgpack-numpy contract. An engine change that makes the console
able to tell the engines apart is a regression. Per-engine specifics live in each engine's
`README.md`; the sections below cover MolmoSpaces (the reference engine) and then the ROS
contract every engine obeys.

---

## simulator/molmospaces/  (the reference engine)

Everything goes through `./run.sh` (from inside `simulator/molmospaces/`); it sources
`env.sh` for asset paths and `MUJOCO_GL`. The other engines (`robocasa/`, `coppeliasim/`)
mirror this launcher surface — see their READMEs.

```bash
cd simulator/molmospaces
./run.sh setup                      # venv + install molmospaces[mujoco] + default assets
./run.sh assets ithor               # pre-fetch iTHOR houses/objects/grasps (~13 GB)
./run.sh view --scene ithor:1       # a house in the viewer
./run.sh view --robot myagv --scene ithor:1 --ros-port 9090   # + robot, as a ROS robot
./run.sh view --robot so101 --control 127.0.0.1:8000   # generic arm control server
./run.sh shell                      # interactive shell in the venv
./run.sh help
```

`--control HOST:PORT` is the current spelling (the console's `robot-console-arm --control
HOST:PORT` mirrors it); `--control-port PORT` is kept as an alias.

Robot self-tests are standalone scripts, not pytest — a failure points at the robot
definition:

```bash
python robots/myagv/test_attach.py [--scene /path/to/house.xml]
python tools/render_robots.py --outdir /tmp/robots   # render/load test for every robot
```

### Two unrelated bridges — do not conflate them

Both now live in `simulator/shared/contracts/` and are reused by every engine:

- `shared/contracts/control_server.py` — the **arm-robot transport**. msgpack-numpy
  over binary websocket frames, hosted by `view --control`. It only sends
  observations and applies returned targets. All clients and action-selection code live
  in `robot_console` (`arm_client.py`, `inspect_so101.py`, `so101_driver.py`).
- `shared/contracts/rosbridge_server.py` — the **mobile-base** bridge, wired up by
  `serve_ros()` in each engine's `tools/spawn_robot.py`. Plain rosbridge JSON over a
  websocket, served in-process. This is what `robot_console` talks to. ROS-only mode:
  mutually exclusive with `--control`, and only for robots with a mobile base (`myagv`).

The MuJoCo engines additionally share `simulator/shared/mujoco_bridge.py` (holonomic
`cmd_vel` integration, ray-cast `/scan`, camera encode) — it imports `mujoco`, so the
CoppeliaSim engine does **not** use it and sources its observations over the ZMQ API
instead.

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

## robot_console/

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
- `arm_client.py` — generic msgpack-numpy simulator client; `so101_driver.py` and
  `inspect_so101.py` adapt it to the optional unmodified Inspect Robots SO-101 stack
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

## The ROS contract (both sides must agree)

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

---

## Workspace conventions

- `specs.json` describes the current, verified state and **is tracked**.
  `target_specs.json` and `implementation_plan.json` are per-run working documents and
  are gitignored. Both validate against `spec.schema.json` / `plan.schema.json`.
  `goal_prompt.md` describes that spec-driven workflow.
- Comments in this codebase explain *why*, especially where a choice looks wrong without
  the constraint behind it (macOS threading, the reactor, key auto-repeat, codec
  choices). Match that when adding code; do not narrate what the line already says.
