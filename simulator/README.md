# MolmoSpaces simulator

A MuJoCo-based simulation of fully-furnished, interactable houses, installed from
[allenai/molmospaces](https://github.com/allenai/molmospaces), plus a launcher and
an external control bridge.

## Quick start

```bash
./run.sh view                      # open a house in the viewer, no robot
./run.sh view --robot myagv        # ...with an out-of-tree robot spawned in it
./run.sh help                      # all commands and flags
```

## Layout

| Path | What it is |
|---|---|
| `run.sh` | launcher; all commands go through it |
| `env.sh` | environment variables, sourced by `run.sh` |
| `molmospaces/` | upstream clone (git-pullable) |
| `.venv/` | Python 3.11 virtualenv, managed by `uv` |
| `data/mujoco/` | `MLSPACES_CACHE_DIR` — versioned downloaded assets |
| `assets/` | `MLSPACES_ASSETS_DIR` — unversioned symlink tree that MuJoCo loads from |
| `bridge/` | simulator-hosted generic control servers |
| `tools/resolve_scene.py` | scene reference → loadable MJCF path |
| `robots/` | out-of-tree robot definitions (so101, myagv, rebot_b601, ainex) |
| `tools/render_robots.py` | render every loadable robot; doubles as a load test |

## Commands

```
./run.sh setup                     venv + install + default assets (idempotent)
./run.sh assets [ithor|list|<src>] bulk pre-fetch for offline use
./run.sh view [--scene ithor:1]    a house in the MuJoCo viewer, no robot
./run.sh shell                     interactive shell in the venv
```

## External control

The simulator hosts a generic websocket server that publishes observations and applies
the actuator targets returned by one external client. It contains no action-selection
or robot-control policy; controllers live in `robot_console` and can target this server
without importing MuJoCo or MolmoSpaces.

```bash
./run.sh view --robot so101 --control-port 8000         # terminal 1: simulator server
cd ../robot_console
.venv/bin/robot-console-arm --controller wave --port 8000  # terminal 2: control client
```

Protocol (msgpack-numpy over binary websocket frames):

```
sim -> client    metadata dict, once, on connect
sim -> client    observation  {"qpos", "qvel", "actions/joint_pos", <cameras>}
client -> sim    action       {"arm": ndarray, "gripper": ndarray}
```

Action semantics follow the robot's `command_mode`
(`molmospaces/molmo_spaces/configs/robot_configs.py`); for joint-position
robots both are absolute joint targets.

Write your own controller in `robot_console` as any `obs -> action` callable and point
the client at it:

```bash
.venv/bin/robot-console-arm --controller mypkg.mymodule:my_controller
```

The console provides `hold` (the default) and `wave` clients for protocol checks.

### Inspect Robots SO-101

The Inspect Robots adapter and policy live in `robot_console`; the simulator only
provides this scene and its generic control server.

Install the console's optional Inspect Robots dependencies, then run both projects:

```bash
cd ../robot_console
uv pip install -e '.[inspect]'  # requires Python 3.10+

# terminal 1: simulator-hosted generic control endpoint
cd ../simulator
./run.sh view --robot so101 --scene robots/so101/inspect_scene.xml \
  --control-port 8000 --control-hz 20 --timeout 300

# terminal 2: real SOArmEmbodiment, with a safe no-LLM smoke motion
cd ../robot_console
.venv/bin/robot-console-inspect-so101 --smoke --port 8000
```

For the local Ollama model, omit `--smoke`; defaults are
`qwen3.8:27b-mlx` at `http://127.0.0.1:11434/v1` and the instruction
`Pick up the blue cup and place it inside the open orange box.`. Inspect Robots and its
SO-101 package are never imported by the simulator process and are not modified by this
integration.

### ROS

Mobile robots can instead be presented on **their own manufacturer's ROS topics**, so the
same client drives the simulated robot and the real one:

```bash
./run.sh view --robot myagv --ros-port 9090   # cmd_vel / odom
./run.sh view --robot ainex --ros-port 9090   # /walking/*, /app/*, bus servos
```

Each robot's contract lives beside it in `robots/<name>/ros_surface.py`, because the
contract is part of the robot definition — and the two that exist share **no topic at
all**. What they do share (renderers, the depth stream, the ray-cast lidar, and the
velocity-to-setpoint integration) is in `tools/spawn_robot.py`.

#### myagv

Subscribes `cmd_vel` (`geometry_msgs/Twist`) and publishes:

| topic | type | notes |
|---|---|---|
| `/odom` | `nav_msgs/Odometry` | `odom` → `base_footprint`, `myAGV.cpp`'s covariances |
| `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | base64 JPEG, `--camera-size`, `--jpeg-quality` |
| `/scan` | `sensor_msgs/LaserScan` | ray-cast stand-in for the YDLidar X2, 10 Hz |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | `16UC1` millimetres, `--depth-hz` (5 Hz) |
| `/camera/rgb/camera_info` | `sensor_msgs/CameraInfo` | intrinsics for the depth image |

`/scan` follows the real lidar's parameters — `laser_frame`, 0.1–12.0 m, 10 Hz, mounted
65 mm ahead of and 80 mm above `base_footprint`. `--scan-beams`, `--scan-range`,
`--scan-min-range`, `--scan-offset`, `--scan-hz` and `--no-scan` adjust it;
[robots/README.md](robots/README.md) records where each default comes from and the two
places it deliberately departs from the hardware.

#### ainex

The Hiwonder AiNex is a 24-DoF biped, and **its interface has no `cmd_vel`, no `odom` and
no `tf`** — none of the three appears anywhere in `Hiwonder/ainex`. It is commanded as a
state machine:

| direction | name | type |
|---|---|---|
| sub | `/walking/set_param`, `/app/set_walking_param` | `ainex_interfaces/WalkingParam`, `AppWalkingParam` |
| sub | `/app/set_action` | `std_msgs/String` — replays a recorded action group |
| sub | `/ros_robot_controller/bus_servo/set_position` | raw servo counts by id |
| sub | `/head_pan_controller/command`, `/head_tilt_controller/command` | `ainex_interfaces/HeadState` |
| srv | `/walking/command` | `SetWalkingCommand` — enable, disable, start, stop, enable_control, disable_control |
| srv | `/walking/get_param`, `/walking/is_walking`, `/walking/init_pose` | |
| pub | `/joint_states`, `/walking/is_walking`, `/imu`, `/camera/image_raw/compressed`, `/scan` | |

Walking is `/walking/command start` after setting a parameter block; grasping is
`/app/set_action clamp_left` and friends. `--action-dir` points the action player at a
real robot's `ActionGroups` directory, whose `.d6a` files it reads directly.
[robots/README.md](robots/README.md) lists what is faithful and what is not — notably that
`/scan` is a virtual lidar on a robot that carries none, and that the gait is animated
rather than balanced.

`bridge/rosbridge_server.py` serves the rosbridge protocol in-process. ROS has no
official macOS build, and MuJoCo needs the Homebrew framework Python while ROS on macOS
comes from conda — so serving the protocol costs far less than migrating the stack, and
any rosbridge client works against it unchanged. Drive it from
[../robot_console](../robot_console).

The base integrates the commanded velocity itself (that is what `cmd_vel` means) and
stops if no command arrives for 0.5 s. The websocket bridge above stays for the arm
robots, which have no ROS contract.

## Platform notes (macOS)

- **The venv must be built on a framework Python**, not uv's standalone CPython.
  `mjpython` (which the viewer requires, because the MuJoCo passive viewer must
  own the main thread) needs a shared `libpython3.11.dylib` that the standalone
  build does not ship. `run.sh setup` looks for Homebrew's `python@3.11`.
- **`mjpython -m mujoco.viewer` does not work** — it re-executes the module and
  drops the handle mjpython stamps onto `mujoco.viewer` at startup, failing with
  `RuntimeError: Caught an unknown exception!`. The viewer must be launched from
  a *script*, which is what `tools/view_scene.py` is for.
- `MUJOCO_GL=glfw` drives both the viewer and offscreen rendering. There is no
  EGL/OSMesa on macOS; use `MUJOCO_GL=cgl` for pure headless rendering.
- The `mujoco-filament` extra is a Linux-x86_64-only wheel and cannot be used here.
- `curobo` (GPU planning, used by the RB-Y1 upstream) is CUDA-only and not installed.
- `mujoco-warp` installs but runs on CPU.

`env.sh` and `run.sh` deliberately avoid `$(cd ... && pwd)`: an interactive shell
with a `chpwd`/`precmd` hook that writes a terminal-title escape will otherwise
have that escape captured into the path.

## Assets

Objects and scenes stream on demand by default. `./run.sh assets ithor`
pre-fetches the hand-crafted iTHOR houses (48 in the `train` split), the ~2k THOR
objects and the DROID grasp set for offline use — currently ~13 GB. It deliberately
skips Objaverse (~129k objects) and the ProcTHOR/Holodeck scene sets (~110k houses
each), which remain on-demand.

Note that a bulk archive fetch alone is *not* enough to use a house offline:
scenes are installed per-file, so `tools/prefetch_scenes.py` walks every house and
pulls its meshes and grasps. `./run.sh assets ithor` does this for you.

Asset installs take an exclusive lock, so a `view` launched while a prefetch
is running will block until it finishes rather than fail.

`./run.sh assets list` shows every available source;
`./run.sh assets mujoco/objects/objaverse/<version>` fetches one explicitly.

## Robots

**Built into MolmoSpaces** (upstream, not wired into the launcher): `franka` /
`droid` (Franka FR3), `rby1` (Rainbow RB-Y1), `yam` / `bimanual_yam` (I2RT YAM),
`rum` (floating gripper).

**Out-of-tree**, in `robots/`: `so101`, `myagv`, `rebot_b601`, `ainex`.

```bash
./run.sh view --robot so101                      # spawn it in a house, interactive
./run.sh view --robot myagv                      # ...or any other out-of-tree robot
python tools/render_robots.py --outdir /tmp/robots     # render them all
./run.sh view --robot so101 --control-port 8000       # terminal 1: simulator server
../robot_console/.venv/bin/robot-console-arm --controller wave --port 8000  # terminal 2
python robots/so101/test_attach.py               # self-test in an empty world
```

See [robots/README.md](robots/README.md) for how each was added and what its
limitations are — every one of them has at least one.

Drive the mobile robots by keyboard with a live camera feed from
[../robot_console](../robot_console).
