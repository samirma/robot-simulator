# Robot Simulator

A robot simulation workspace with two independent projects that talk over network protocols (rosbridge), never Python imports:

- **`simulator/`** — simulates robots (myAGV mobile base, SO-101 arm, AiNex humanoid) with a choice of two interchangeable MuJoCo engines: MolmoSpaces (iTHOR houses) and RoboCasa (kitchens). Both engines expose each robot's real vendor ROS interface over one rosbridge websocket.
- **`robot_console/`** — drives, maps and navigates the simulated (or physical) robots: teleop, SLAM exploration, and the SO-101 arm task. It connects identically regardless of which engine hosts the robot.

## Setup

Each project has its own `uv` venv and its own launcher; there is no top-level build.
Tested on macOS (Apple Silicon); Linux works headless with `MUJOCO_GL=egl`.

### 1. Prerequisites

```bash
brew install python@3.11 uv git
```

- **Homebrew's Python 3.11 is required for the simulator**, not uv's standalone CPython:
  the MuJoCo window runs under `mjpython`, which needs a shared `libpython3.11.dylib`.
  The setup scripts find it at `/opt/homebrew/opt/python@3.11` (or `/usr/local/opt/...`).
- The console is happy on any Python; its launchers let `uv` fetch 3.12.
- Disk: MolmoSpaces ~13 GB with every iTHOR house (houses otherwise download on demand);
  RoboCasa ~10 GB of kitchen assets; the MolmoAct2 checkpoint ~22 GB.

### 2. Robot assets

`robots_specs/` holds each robot's official URDF, MuJoCo model and ROS interface; the
meshes they reference (~220 MB) are not in git but fetched from each robot's pinned
upstream and checked against `robots_specs/meshes.sha256`. Each engine's `run.sh setup`
also does this.

```bash
./fetch_robot_assets.sh
```

### 3. Simulator — MolmoSpaces (the default engine)

```bash
cd simulator/molmospaces
./run.sh setup            # clones allenai/molmospaces into upstream/, builds .venv, fetches default assets
./run.sh assets ithor     # optional: every iTHOR house for offline use (~13 GB)
```

`--scene ithor:1` (the `kitchen.sh` default) is installed on first use if you skip
`assets ithor`.

### 4. Simulator — RoboCasa (optional, second engine)

Only needed for `./kitchen.sh serve --engine robocasa`.

```bash
cd simulator/robocasa
./run.sh setup            # clones robosuite + robocasa into upstream/, builds .venv
./run.sh assets           # kitchen assets (~10 GB)
```

### 5. Console

Nothing to do up front: `bin/teleop.sh`, `bin/slam.sh` and `run_task.sh` create their venv
on first run (`.venv`, and `.venv-vla` for the MolmoAct2 policy, which pulls torch). To
install eagerly and run the offline tests:

```bash
cd robot_console
./bin/teleop.sh --reinstall
.venv/bin/python -m pytest
```

### 6. Check it works

```bash
cd simulator
./kitchen.sh serve --mujoco              # MuJoCo window on the kitchen, served on ws://127.0.0.1:9090
```

In a second terminal, from the repo root — every topic on the wire, as the console sees it:

```bash
robot_console/.venv/bin/python -m robot_console.fleet --dump
```

Closing the MuJoCo window (or Ctrl-C) ends the serve. Drop `--mujoco` to run headless,
which is faster for every client on the port.

### Moved or copied the checkout?

uv venvs and MolmoSpaces' `assets/` tree hold absolute paths, so after moving the repo
`mjpython` fails on a stale shebang and scenes "fail to download". Every launcher
(`run.sh`, `kitchen.sh`, the console scripts) repairs this in place on start; to do it
by hand:

```bash
simulator/molmospaces/run.sh repair
simulator/robocasa/run.sh repair
```

### Troubleshooting

- `port 9090 is already in use` — another serve is running; stop it or pass `--port`.
- `<engine> is not set up yet` — run that engine's `./run.sh setup`.
- The viewer window does not open or crashes on macOS — try `MUJOCO_GL=cgl` for the
  offscreen cameras (see the engine's `env.sh`).

## Quick start

Serve a simulated kitchen with robots on `ws://127.0.0.1:9090`:

```bash
cd simulator
./kitchen.sh serve                       # headless, SO-101 arm task
./kitchen.sh serve --engine robocasa     # the other engine
./kitchen.sh serve --robots so101,myagv  # a fleet, one port
./kitchen.sh serve --mujoco              # ...with a MuJoCo window
```

Drive and task the robots from the console:

```bash
cd robot_console
./bin/teleop.sh          # keyboard teleop
./bin/slam.sh explore    # autonomous mapping
./run_task.sh            # run the arm task with the VLA policy
```

See `CLAUDE.md` for the full architecture, commands and invariants, and each engine's own `README.md` for per-engine details.
