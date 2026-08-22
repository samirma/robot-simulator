# RoboCasa engine

RoboCasa (robosuite + MuJoCo) as a **simulator engine** for this workspace: it hosts a
kitchen scene with one of the shared robots dropped in, and presents that robot on the
*exact same wire contract* as the MolmoSpaces engine and the real hardware. `robot_console`
connects to a RoboCasa-hosted robot with no changes — that is the whole point of the
`simulator/shared/` split.

## Quick start

```bash
./run.sh setup                                            # venv + robosuite + robocasa + ~10 GB assets
./run.sh view --robot myagv --scene kitchen:1 --ros-port 9090
./run.sh view --robot so101 --scene kitchen:3/5 --control 127.0.0.1:8000
./run.sh view --robot myagv --scene empty --ros-port 9090 --headless
.venv/bin/python tools/screenshot.py --scene kitchen:1   # headless render of each robot
./run.sh shell
```

`--scene`:
- `kitchen:<layout>[/<style>]` — a RoboCasa kitchen. Layouts `1..60`, styles `1..60`
  (see `robocasa/models/scenes/scene_registry.py`). Needs the downloaded assets.
- `empty` — the bare kitchen room shell (floor + walls, no fixtures). Always loads even
  before the asset download finishes; handy for SLAM/drive testing.

`--robot`: `so101` (arm) or `myagv` (holonomic mobile base). Both come from
`simulator/shared/robots/`, so they are byte-for-byte the same models the MolmoSpaces
engine uses.

The wire surface is mutually exclusive, mirroring MolmoSpaces:
- `--ros-port N` — present the myAGV vendor ROS topics (`/cmd_vel`, `/odom`, `/scan`,
  `/camera/image_raw/compressed`) over rosbridge. Mobile base only.
- `--control HOST:PORT` — present the generic msgpack-numpy arm control server
  (`molmospaces-control-v1`). Used by `robot_console`'s arm client / SO-101 driver.
- `--headless` — run the sim + servers with no viewer window (needed on displayless
  hosts and for automated checks; set `MUJOCO_GL=cgl`).

## How it works

`robosuite.make(...)` would build a whole RL task with a robosuite-native robot,
controllers and rewards — none of which we want. Instead `tools/spawn_robot.py`:

1. Builds the kitchen with RoboCasa's own arena + fixture merger
   (`KitchenArena` + `ManipulationTask` with an **empty** robot list), and takes the
   composed MJCF.
2. Grafts the shared robot in with `mujoco.MjSpec.attach(..., prefix="robot_")`.
3. Runs a plain MuJoCo step loop at the control rate, feeding the **shared** bridge in
   `simulator/shared/`:
   - `mujoco_bridge.py` — holonomic `cmd_vel` integration (`PlanarSetpoint`), the
     YDLidar-X2-shaped ray-cast `/scan` (`laser_scan_ranges` / `SensorStreams`), camera
     JPEG encode. Engine-neutral: it takes a raw `mujoco` model/data.
   - `contracts/rosbridge_server.py` and `contracts/control_server.py` — the wire
     transports themselves, shared with every engine.

Because the observation sourcing and the wire transport are shared code, a RoboCasa
myAGV publishes `/odom`+`/scan`+camera and obeys `/cmd_vel` identically to a MolmoSpaces
myAGV or the real 2023 Pi myAGV; a RoboCasa SO-101 speaks the identical control protocol.

## Layout

```
robocasa/
  run.sh env.sh
  tools/spawn_robot.py     kitchen build + robot attach + shared-bridge serve loop
  tools/screenshot.py      headless render of each robot in a kitchen (a smoke check)
  upstream/                robosuite (master) + robocasa clones (gitignored)
  .venv/ data/             (gitignored)
```

There is deliberately no per-robot adapter package here (unlike MolmoSpaces, which needs
`RobotView` trios): the shared MJCF attaches directly, so the robot definitions live only
in `simulator/shared/robots/`.

## Notes

- **macOS:** the venv is built on Homebrew framework Python 3.11 so `mjpython` (the
  viewer's main-thread host) has a shared `libpython`. `MUJOCO_GL=glfw` for the viewer,
  `cgl` for headless offscreen. Never `egl` on macOS.
- **Assets:** `run.sh setup` downloads ~10 GB into the robocasa package tree
  (`upstream/robocasa/robocasa/models/assets`). The `empty` scene works without them.
- robosuite is pinned to its **master** branch (the docs are explicit — not a tag);
  robocasa pins `mujoco==3.3.1`. robocasa's full dependency set pulls heavy RL/learning
  packages that are irrelevant here and can fail to resolve on macOS arm64, so `setup`
  falls back to a `--no-deps` install plus the handful the models code actually imports.
