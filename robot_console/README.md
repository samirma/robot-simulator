# robot_console

Keyboard teleoperation, mapping and navigation for a [myAGV], over rosbridge. Teleop
also drives a Hiwonder AiNex — see [`--robot`](#--robot).

The console is an independent project. Its base dependencies are `numpy`,
`opencv-python`, and `roslibpy`, so it installs and runs on a machine that has never
seen MuJoCo, MolmoSpaces, or the `simulator/` checkout. Everything it drives, it drives
over rosbridge: the myAGV on its vendor ROS 1 topics, the SO-101 on the ROS 2
ros2_control topic set a real bringup for that arm presents. Optional arm and Inspect
Robots dependencies do not introduce a simulator import; `import mujoco` failing inside
`.venv` is a property worth preserving, and the test suite checks it.

## Quick start

Start a robot in one terminal:

```bash
cd ../simulator
./run.sh view --robot myagv --scene ithor:1 --ros-port 9090
```

Drive it from another:

```bash
cd robot_console
./bin/teleop.sh                        # ws://127.0.0.1:9090
./bin/teleop.sh --host 192.168.1.42    # a real myAGV on the network
./bin/teleop.sh --record runs/drive1   # ...writing feed.mp4 + commands.jsonl
./bin/teleop.sh --robot ainex          # a different robot; see below
```

The first run creates `.venv` and installs the package; after that `bin/teleop.sh` is
just a launcher. It re-installs by itself when `pyproject.toml` changes, and
`--reinstall` forces it.

Before connecting it checks that something is listening, and prints how to start each
kind of robot when nothing is. `--no-preflight` skips the check.

## The arm

The SO-101 is driven over **rosbridge**, on the ros2_control topic set a real bringup for
that arm presents — `joint_trajectory_controller` for the five arm joints, a
`forward_command_controller` for the jaw, one `joint_state_broadcaster` covering both.
There is no simulator-specific protocol: the same client drives the simulated arm and a
real one, which is the whole reason for presenting that interface rather than an easier
one.

Everything arm-related lives in `src/robot_console/arm/` and registers itself with the
[Inspect Robots](https://pypi.org/project/inspect-robots/) framework through the
`inspect_robots.{tasks,policies,embodiments,scorers}` entry points in `pyproject.toml`.
That is what makes the framework's own `inspect-robot` CLI able to find it — this project
ships no arm console script of its own:

```bash
uv pip install -e '.[arm]'        # numpy + inspect-robots + inspect-robots-ros
.venv/bin/inspect-robot list tasks        # apple_on_plate
.venv/bin/inspect-robot list policies     # molmoact2, so101_waypoint
.venv/bin/inspect-robot list embodiments  # so101_ros
```

Registered pieces:

| kind | name | what it is |
|---|---|---|
| task | `apple_on_plate` | pick a 20 mm apple off the work surface, place it on the plate, hold it there |
| policy | `so101_waypoint` | the scripted plan: a task-space pick-and-place lowered through IK, closed-loop on proprioception |
| policy | `molmoact2` | the `allenai/MolmoAct2-SO100_101` VLA, from two camera views and the task text |
| embodiment | `so101_ros` | the arm behind rosbridge |
| scorer | `apple_on_plate_success`, `reference_success`, `apple_plate_distance` | the camera's verdict, plus a pose-derived one that grades nothing and exists to audit it |

The usual way to run it is from the simulator, which starts the world and the client
together and reports PASS/FAIL:

```bash
cd ../simulator && ./kitchen.sh inspect                    # scripted baseline
cd ../simulator && ./kitchen.sh inspect --policy molmoact2 --episodes 8
```

Manually, against a simulator someone else started:

```bash
.venv/bin/python -m robot_console.arm.preflight --url ws://127.0.0.1:9090   # reset + verify
.venv/bin/inspect-robot run --task apple_on_plate --policy so101_waypoint \
    --embodiment so101_ros -E url=ws://127.0.0.1:9090 -T max_steps=400 \
    --max-action-delta 0.65
```

Three flags there are load-bearing rather than decorative:

- **`-T max_steps=400`.** The task's own default is smaller, and a short budget cuts the
  episode off with the apple still in the air over the plate — a working plan scored as a
  failure.
- **`--max-action-delta 0.65`.** The framework applies its own per-step limiter, derived
  from the action space, which lands near 0.03 and halves the policy's 0.06 rad step. The
  policy is already the rate limiter and it is the one holding the measured constants;
  this raises the framework's limit above the jaw's largest intended move and leaves the
  bounds clamp in place.
- **`preflight` before the episode.** `/reset` says the world was restored; this checks
  that it *was* — the apple measured back at spawn over three consecutive samples, and the
  plan still solving from there. A drifted apple otherwise scores zero looking exactly
  like a policy failure, and a failed episode really does leave the apple on the floor.

### torch stays out of `.venv`

`molmoact2` needs torch, transformers and a ~22 GB checkpoint; the scripted policy needs
none of it. So the VLA extra installs into a **separate** venv:

```bash
uv venv --python 3.12 .venv-vla && VIRTUAL_ENV=.venv-vla uv pip install -e '.[vla]'
```

`kitchen.sh inspect` picks the venv from `--policy`, so this is only worth knowing
when running the client by hand. `arm/molmoact.py` imports torch inside `_load()` rather
than at module scope, which is what lets `inspect-robot list policies` work — and the
offline test suite run — in the torch-free venv.


## Mapping and navigation

`bin/slam.sh` is the same launcher with three modes, all built on `/scan`:

```bash
./bin/slam.sh explore  --out runs/house    # drive around by itself until the map is done
./bin/slam.sh map      --out runs/house    # drive by hand, map builds live
./bin/slam.sh navigate --map runs/house    # click a point, the robot drives there
```

- **explore** picks the boundary between mapped and unmapped space, plans to it, drives
  there, and repeats. It terminates because a map with no frontiers left is finished by
  definition. `Space` pauses; any drive key takes over.
- **map** is `teleop.sh` with a map window beside the camera. `M` saves without quitting.
- **navigate** loads a saved map and localizes into it by scan matching, so the robot
  does not have to be put back where the mapping run started. Left click sets a goal,
  right click or `Space` cancels. The live map keeps updating -- furniture that has moved
  gets planned around -- but the saved map is only overwritten if you press `M`.

Maps are written as a **`map_server` pgm/yaml pair**, which is what
`myagv_navigation`'s `navigation_active.launch` loads, so a map built in the simulator
can be handed to the real robot and back. A `map.npz` sidecar carries the raw log-odds
so a map can be reloaded and *kept building* rather than only navigated.

Nothing here knows which simulator is on the other end of the socket, and that is worth
stating because it has been checked rather than assumed: the same
`robot-console-explore` builds a map of a MolmoSpaces iTHOR house and of a RoboCasa
kitchen, over the same four topics, with no flag telling it which.

This is the same sensor choice Elephant Robotics makes -- `myagv_slam_laser.launch` runs
gmapping on the YDLidar X2 -- reimplemented in numpy and OpenCV so it needs no ROS. It
is occupancy-grid SLAM with correlative scan matching: **no loop closure, no global
optimiser**. On an identical trajectory with 6 % odometry drift, scan matching roughly
halves the map smear (median wall offset 0.28 m -> 0.13 m, p90 0.68 m -> 0.28 m) at
about 2 ms per keyframe. It does not correct global drift after a long loop.

`--no-match` trusts `/odom` and skips matching, which is reasonable against the
simulator, where odometry is ground truth, and is not on hardware.

## Controls

The OpenCV camera window owns keyboard input, which avoids a global keyboard hook.
**The window must have focus for keys to register.**

| Key | Action |
|---|---|
| `W` / `S` | Forward / back |
| `A` / `D` | Strafe left / right |
| `Q` / `E` | Rotate left / right |
| `Space` | Stop |
| `+` / `-` | Adjust speed |
| `H` or `?` | Show/hide the on-screen hints |
| `Esc` | Quit |

The same list is drawn over the live feed, so the keys are visible while driving. `H`
collapses it to a small badge.

**Hold a key to drive; let go and the robot stops.** There is no key-up event to work
with -- `cv2.waitKey` reports key-down only, and a real one would mean the global hook
this design rules out. What the OS does give is auto-repeat: holding `W` delivers `w`
over and over. So a motion is armed by a key press and expires `--hold-timeout` seconds
(0.6 by default) after the last repeat, which is what a release looks like.

That timeout has to clear the OS's *initial* repeat delay or a held key would stutter:
move, expire, then resume once repeat kicks in. macOS ships 375 ms before the first
repeat and 90 ms between them, so 0.6 s has margin while costing about 9 cm of coast at
the default speed. The vendor's own teleop makes the same trade at 0.52 s. If your
keyboard repeat is disabled or unusually slow, raise `--hold-timeout`, or pass `--latch`
to keep the old behaviour where a direction persists until `Space` or another key.

`+`/`-` step the speed by 0.05 m/s between 0.05 and 0.28; `=` and `_` work too, since
`+` and `_` need shift on most layouts.

Closing the window with its close button quits as cleanly as `Esc` does.

## Speeds

| | Value | Source |
|---|---|---|
| Default | 0.15 m/s | conservative indoor pace |
| Range | 0.05 – 0.28 m/s | 0.28 is the real myAGV's top speed |
| Turn rate | `speed x 2`, capped at 1.0 rad/s | the vendor teleop pairs 0.25 m/s with 0.5 rad/s and caps turn at 1.0 |

One knob scales the whole envelope, so a drive rehearsed in the simulator behaves the
same on hardware. `--max-speed` raises the cap and warns when it goes above the
hardware limit.

## Recording

`--record <dir>` writes two files.

`feed.mp4` is the raw camera feed, without the key hints drawn on it. The writer opens
lazily: the first 20 frames are
buffered, the true frame rate is measured from their arrival times (median interval, so
one network stall does not halve the playback speed), and only then is the file created
with the real size and rate. If no camera frames ever arrive, no `feed.mp4` is written
-- an empty file would be worse than an absent one.

`commands.jsonl` is one JSON object per line: a `meta` line, then `cmd`, `frame`, and
`odom` lines carrying `t` in seconds from the start, then a `summary` line.

```jsonc
{"type":"meta","schema":1,"host":"127.0.0.1","port":9090,"speed":0.15,...}
{"type":"cmd","t":0.05,"seq":1,"key":"w","action":"FORWARD","speed":0.15,
 "linear":{"x":0.15,"y":0.0,"z":0.0},"angular":{"x":0.0,"y":0.0,"z":0.0}}
{"type":"frame","t":0.06,"index":0,"header_seq":5880,"width":640,"height":480}
{"type":"odom","t":0.07,"x":4.68,"y":0.62,"yaw":1.03,"vx":0.15,"vy":0.0,"wz":0.0}
{"type":"summary","t":9.01,"duration":9.01,"commands":181,"frames":178,"fps":19.8}
```

`cmd` lines carry the literal `geometry_msgs/Twist` sub-dicts, so a replayer can feed a
line straight back to `/cmd_vel` without translating anything. `frame.header_seq` and
`frame.index` line the video up against the commands.

## ROS contract

| Direction | Topic | Type | Fields used |
|---|---|---|---|
| console -> robot | `/cmd_vel` | `geometry_msgs/Twist` | `linear.x`, `linear.y`, `angular.z` |
| robot -> console | `/odom` | `nav_msgs/Odometry` | pose, twist; `odom` -> `base_footprint` |
| robot -> console | `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | base64 JPEG |
| robot -> console | `/scan` | `sensor_msgs/LaserScan` | `ranges`, angles, `range_min`/`range_max` |

`/scan` is the YDLidar X2's topic: `laser_frame`, 0.1-12.0 m, 10 Hz, mounted 65 mm ahead
of and 80 mm above `base_footprint` (`myagv_active.launch`'s static transform, which
`slam/scan.py` applies -- 65 mm is more than a whole map cell). Only the teleop console
ignores it; the SLAM commands need it.

Unlike `CompressedImage`, `ranges` arrives as a **plain JSON float array** -- rosbridge
base64-encodes `uint8[]` only. A no-return is reported three different ways depending on
who is publishing: `0.0` by the real driver (`invalid_range_is_inf: false`), `inf` by a
stock one, and `range_max + 1` by the simulator, which cannot express infinity in JSON.
The console tests `range_min <= r <= range_max`, which rejects all three and `NaN` too.

Body frame, ROS convention: `+x` forward, `+y` left, `+z` counter-clockwise. The base is
holonomic -- the myAGV is Mecanum-wheeled, so `linear.y` is a real strafe, not a
no-op.

Topic names are sent absolute. The simulator's bridge normalises relative names, but
stock `rosbridge_suite` resolves them against the node namespace, where a relative name
silently misses.

`format` is matched on containing `jpeg`, because the simulator sends `"jpeg"` while a
real `image_transport` republisher sends `"rgb8; jpeg compressed bgr8"`.

### Running against a real myAGV

On the AGV:

```bash
roslaunch myagv_odometry myagv_active.launch      # odometry + the YDLidar: /odom, /scan
roslaunch rosbridge_server rosbridge_websocket.launch
# plus a camera publisher for /camera/image_raw/compressed
```

`myagv_active.launch` already starts the lidar (it includes
`ydlidar_ros_driver/launch/X2.launch`) and publishes the `base_footprint -> laser_frame`
transform, so `/scan` needs nothing extra. Then `./bin/teleop.sh --host <agv-ip>`, or
`./bin/slam.sh explore --host <agv-ip>`.

**The real myAGV has no command watchdog.** `myagv_odometry_node` stores the last Twist
it received in a global and writes it to the motors at 100 Hz forever, so a robot told
to move keeps moving until it is told otherwise -- the vendor's own teleop guards
against this with a 0.52 s client-side key timeout. The console therefore treats
stopping as part of quitting rather than as best-effort: it publishes a zero Twist on
`Esc`, on window close, on an exception, and on `SIGINT`/`SIGTERM`. The simulator's
0.5 s bridge watchdog makes this invisible there; on hardware it is the only thing that
stops the robot.

## Checks

```bash
uv pip install -e '.[dev]'
.venv/bin/python -m pytest              # offline; no robot, no display

.venv/bin/python -m robot_console.smoke # live; drives the robot ~0.3 m each way
```

The offline suite covers the keymap and latch semantics, the speed model, JPEG decode
and its corruption cases, the frame mailbox under concurrency, odometry quaternion
maths, the recorder schema and video output, preflight, and CLI parsing. It also
round-trips `RobotLink` against `tests/fake_bridge.py`, a small independent rosbridge
implementation, which proves the bytes `roslibpy` emits are the bytes the server
accepts -- without needing the simulator checkout.

`smoke` is the live version: it connects, measures the `/odom` and camera rates, decodes
a frame and checks it is not a flat buffer, then drives forward, back, sideways and
around, checking the pose moved each time. It also stops publishing for 1.5 s to prove
the simulator's watchdog fires (skipped against a robot that has none). `--json` emits
the same results as one object.

## Layout

```
bin/teleop.sh              venv bootstrap + launcher
bin/slam.sh                the same, dispatching explore | map | navigate
src/robot_console/
  topics.py                topic names and type strings -- the contract in one place
  teleop.py                keymap, latch state, speed model   (pure)
  camera.py                CompressedImage decode + the frame mailbox
  bridge.py                RobotLink, and pure odometry parsing
  robots.py                --robot: one RobotProfile per robot, resolved lazily
  ainex_link.py            the AiNex's gait, behind a RobotLink-shaped API
  recorder.py              feed.mp4 + commands.jsonl
  preflight.py             reachability probe + startup instructions
  arm/                     the SO-101, over rosbridge -- see "The arm" above
    ros_client.py          the header-stamping shim (without it the arm never moves)
    ros_settings.py        every topic name, type and camera size, in one place
    kinematics.py          FK/IK for the SO-101                  (pure, MuJoCo-free)
    waypoints.py policy.py the scripted pick-and-place plan      (pure)
    molmoact.py            the MolmoAct2-SO100_101 VLA (torch imported inside _load)
    task.py                the task, its scene and its instruction
    success.py             the live geometric verdict            (pure)
    scorer.py              the offline re-derivation from a log  (pure)
    embodiment.py          the arm behind rosbridge
    preflight.py           reset the world and verify it took
    steptrace.py wire_trace.py   observational only; never affect an action
  app.py                   the teleop loop
  cli.py                   argument parsing
  smoke.py                 live integration check
  explore.py mapping.py navigate.py    the three SLAM entry points (thin)
  slam/
    scan.py                LaserScan -> base-frame points        (pure)
    grid.py                the log-odds occupancy map            (pure)
    mapio.py               map.pgm + map.yaml + npz sidecar
    matcher.py             correlative scan matching             (pure)
    pose.py                odom propagation + keyframed matching (pure)
    planner.py             obstacle inflation and A*             (pure)
    frontier.py            where the map stops, and what to fill (pure)
    controller.py          path -> /cmd_vel for a holonomic base (pure)
    mapview.py             rendering, and clicks back into metres
    app.py                 the loop the three commands are modes of
    cli.py                 argument parsing
tests/                     offline suite + fake_bridge.py + synthetic.py
```

Everything with interesting behaviour is pure and tested; both `app.py` files are
wiring. The loop runs on the main thread and does everything -- read a key, publish
`/cmd_vel` at 20 Hz, fold in a scan, draw the frame. `cv2.imshow` must own the main
thread on macOS, and a separate publisher thread would keep the robot driving while the
UI was wedged. With one loop, a stalled UI stops feeding the command stream, so a freeze
degrades into a stop. That matters more in an autonomous mode than in teleop, because
nobody is watching the window.

Which is why scan matching is keyframed rather than run on every scan: it happens only
after 0.15 m or 10 deg of motion and never faster than `--slam-hz`. The loop measures
its own tick time against the publish period and says so when it overruns, because a
robot that drives fine and maps badly otherwise leaves nothing in the log to explain it.

## Limitations

- Releasing a key is inferred from OS auto-repeat stopping, so the robot coasts for up
  to `--hold-timeout` after you let go. A keyboard with repeat disabled will stutter;
  `--latch` is the fallback.
- Real myAGV hardware has not been tested here; only the simulator path has been run.
- No TF, services, or parameters -- the console uses four topics. The
  `base_footprint -> laser_frame` transform is applied from a constant in `slam/scan.py`
  rather than read from `/tf`, because the bridge does not carry TF; a robot with the
  lidar mounted elsewhere needs that constant changed.
- **SLAM has no loop closure.** Local consistency is maintained by scan matching, but
  drift accumulated around a long loop is not corrected when the robot returns to a
  place it has already mapped. Big spaces will not close perfectly.
- Exploration reports "explored" only after a ladder of increasingly desperate retries
  has come back empty -- a lower frontier threshold, one flush of the suppression list,
  the enclosed sensor holes, and a last look round. `tests/test_exploration_coverage.py`
  drives a four-room floorplan offline and asserts it maps every reachable square metre
  and leaves no frontier behind; on that house it does, including the room whose only
  entrance is a 0.7 m doorway. A run that genuinely gets nowhere ends as `stalled`
  (`--stall-timeout`) rather than pretending to be finished.
- Unknown space behind a wall is not chased. Walls are recorded a cell thick and grazing
  beams skip cells, so a mapped wall has unknown slivers along it that look exactly like
  frontiers; they are filtered by thickness, because they can never be resolved. A real
  opening thinner than two cells at the working resolution would be filtered with them.
- The map grows without bound, and there is no downsampling. At 5 cm a large house is
  fine; a warehouse would want a coarser `--resolution`.
- `mp4v` is the recording codec; `avc1` is missing from many `opencv-python` builds. If
  the writer cannot open, the drive continues and `commands.jsonl` is still written.

[myAGV]: https://shop.elephantrobotics.com/collections/myagv-smart-navigation-robot/products/myagv-pi
