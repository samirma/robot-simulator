# Robot Simulation and Console Specification

This workspace contains two independent projects that communicate over network protocols,
not Python imports.

| Project | Responsibility |
|---|---|
| `simulator/` | Simulate robots in furnished houses with MuJoCo and MolmoSpaces. |
| `robot_console/` | Drive a compatible simulated or physical robot through ROS/rosbridge. |

`robot_console` must remain usable on a machine without MuJoCo, MolmoSpaces, or the
simulator checkout.

## Simulator

### Scope

`simulator/` is based on [allenai/molmospaces](https://github.com/allenai/molmospaces).
It loads furnished houses, provides the upstream scripted task and data-generation pipeline,
and adds out-of-tree robot adapters without modifying the upstream clone.

All normal simulator operations use `simulator/run.sh`:

```bash
./run.sh setup
./run.sh assets ithor
./run.sh view --scene ithor:1
./run.sh view --robot myagv --scene ithor:1 --ros-port 9090
./run.sh sim --robot droid
./run.sh help
```

### Installation and assets

- `./run.sh setup` creates `simulator/.venv`, installs MolmoSpaces with MuJoCo support,
  and initializes the default asset index.
- The simulator venv is managed with `uv` and must use a framework Python 3.11 on macOS.
  The passive MuJoCo viewer runs through `mjpython`, which requires the framework shared
  library.
- `./run.sh assets ithor` downloads the iTHOR scenes, objects, and DROID grasps needed for
  offline iTHOR use. Other supported asset sources can be listed or fetched through
  `./run.sh assets`.
- `assets/` is a generated MolmoSpaces symlink tree. Curated robot source files belong under
  `robots/<name>/`, not under `assets/`.

### Operating modes

| Command | Function |
|---|---|
| `view` | Open a house in the MuJoCo viewer, optionally with an out-of-tree robot. |
| `sim` | Run an upstream MolmoSpaces scripted task and data-generation pipeline. |
| `bridge` | Run the scripted pipeline with actions supplied by an external websocket controller. |
| `serve` | Start the external websocket controller server used by `bridge`. |
| `shell` | Open an interactive shell using the simulator venv. |

`sim` and `bridge` support MolmoSpaces' built-in robots only:
`franka`, `droid`, `rum`, `rby1`, `yam`, and `bimanual_yam`. Out-of-tree robots are loaded
with `view`; they do not have the grasp libraries required by the scripted task planners.

### Out-of-tree robots

The following adapters are registered in `tools/spawn_robot.py` and load with
`./run.sh view --robot <name>`:

| Robot | Supported behavior | Important boundary |
|---|---|---|
| `so101` | Spawn, view, joint control, websocket control. | Five degrees of freedom; no general 6-DoF IK or grasp library. |
| `myagv` | Spawn, view, holonomic drive, front camera, ROS/rosbridge control. | Motion is a virtual holonomic base; wheels are visual only. |
| `lekiwi` | Spawn, view, holonomic base, arm, and gripper. | Uses a virtual holonomic base rather than wheel-contact control. |
| `rebot_b601` | Spawn, view, arm, and gripper. | Self-collision is disabled; world collision remains enabled. |

Each adapter, its model-generation code, and its reference robot description live under
`simulator/robots/<name>/`. `robots/URDF.md` records the provenance and loading role of the
vendor descriptions. `tools/render_robots.py` is the repeatable renderer/load test for all
registered robots.

### External controller bridge

The general controller bridge uses binary WebSocket frames containing msgpack-numpy data.
The simulator connects to a controller started by `./run.sh serve`.

```text
server -> simulator   metadata once at connection
simulator -> server   observation: qpos, qvel, robot_base_pose, cameras, task
server -> simulator   action: arm and gripper targets
```

Controllers can use the built-in `hold` or `wave` policies, or provide an importable
`observation -> action` callable. This bridge is the control path for arm robots.

### ROS/rosbridge mobile-base bridge

`./run.sh view --robot myagv --ros-port 9090` exposes the simulated robot as a rosbridge
server. The implementation is intentionally in-process: it avoids mixing macOS ROS and
MuJoCo Python runtimes while remaining compatible with standard rosbridge clients.

The ROS mode is available only to robots with a mobile base (`myagv` and `lekiwi`) and cannot
be combined with the general `--control` websocket mode. It applies a 0.5-second command
watchdog by default, stopping the simulated base when commands stop arriving.

## Robot Console

`robot_console/` is an independent Python project for keyboard teleoperation with a live
camera feed. It has its own `uv` venv, `pyproject.toml`, and standard `src/` package layout.
Its only runtime dependencies are `numpy`, `opencv-python`, and `roslibpy`.

Start a simulated myAGV in one terminal:

```bash
cd simulator
./run.sh view --robot myagv --scene ithor:1 --ros-port 9090
```

Then drive it from another terminal:

```bash
cd robot_console
./bin/teleop.sh
./bin/teleop.sh --host 192.168.1.42
./bin/teleop.sh --record runs/drive1
```

On its first run, `bin/teleop.sh` creates the console venv and installs the package. Before
connecting, it checks whether rosbridge is reachable and prints the simulator or real-robot
startup instructions when it is not. `--no-preflight` bypasses that check.

The OpenCV camera window owns keyboard input, avoiding a global keyboard hook. The window
must have focus for controls to work.

| Key | Action |
|---|---|
| `W` / `S` | Forward / back |
| `A` / `D` | Strafe left / right |
| `Q` / `E` | Rotate left / right |
| `Space` | Stop |
| `+` / `-` | Adjust speed |
| `Esc` | Quit |

`--record <directory>` writes `feed.mp4` and `commands.jsonl`. The console also includes a
scripted live integration check at `python -m robot_console.smoke`; offline tests are run with
`uv pip install -e '.[dev]'` followed by `.venv/bin/python -m pytest`.

## ROS Contract

The console and the simulator mobile-base bridge follow the myAGV ROS interface over the
rosbridge WebSocket protocol. The same contract is intended for a real myAGV running
`ros-noetic-rosbridge-suite`.

| Direction | Topic | ROS type | Semantics |
|---|---|---|---|
| Console to robot | `/cmd_vel` | `geometry_msgs/Twist` | Body velocity: `linear.x`, `linear.y`, `angular.z`. |
| Robot to console | `/odom` | `nav_msgs/Odometry` | Pose and velocity using `odom` to `base_footprint`. |
| Robot to console | `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | JPEG camera frame, base64 encoded by rosbridge. |

The bridge accepts relative and absolute topic names as equivalent. It implements the
rosbridge `advertise`, `unadvertise`, `publish`, `subscribe`, and `unsubscribe` operations.
Services, TF, and parameters are outside its scope.

For real hardware, the AGV must run its odometry launch file, a rosbridge websocket server,
and a camera publisher for `/camera/image_raw/compressed`. Real hardware remains untested in
this workspace.

## Current Limitations

- Out-of-tree robots are not supported by `run.sh sim` or its scripted pick/place planners.
- `rebot_b601` can self-intersect because its self-collision is disabled.
- The myAGV and LeKiwi bases model holonomic motion rather than physical wheel contact.
- Keyboard teleoperation has been exercised in scripted demo mode; real physical keyboard
  operation and the real myAGV path have not been validated here.
