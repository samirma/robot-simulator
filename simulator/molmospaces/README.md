# MolmoSpaces simulator

A MuJoCo-based simulation of fully-furnished, interactable houses, installed from
[allenai/molmospaces](https://github.com/allenai/molmospaces), plus a launcher and
an external control bridge.

## Quick start

```bash
../kitchen.sh serve --robots myagv --mujoco   # a simulated robot in a house, in a window
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
| `../shared/contracts/` | the rosbridge transport, shared by every engine |
| `../shared/ros_surfaces/` | per-robot topic sets: `myagv.py`, `so101.py` |
| `../shared/tasks/` | what a task stages into a scene, and its success predicate |
| `tools/resolve_scene.py` | scene reference → loadable MJCF path |
| `robots/` | this engine's adapter for each simulated robot in `robots_specs/robots.yml` |
| `tools/test_placement.py` | self-test: the worktop mount rule, and every robot standing at its placement, alone and in the fleet |

## Commands

```
./run.sh setup                     venv + install + default assets (idempotent)
./run.sh assets [ithor|list|<src>] bulk pre-fetch for offline use
./run.sh repair                    re-point the venv and assets/ after a move
```

## Driving a robot

Every robot here is on rosbridge; nothing speaks a bespoke protocol any more. The arm
presents the ros2_control topic set a real SO-101 bringup presents, so the same client
drives the simulated arm and a real one:

```bash
# terminal 1: the simulator, with the task staged (it is whenever a worktop robot is)
../kitchen.sh serve --robots so101 --scene ithor:1

# terminal 2: the client
cd ../../robot_console
.venv-vla/bin/inspect-robot run --task apple_on_plate --policy molmoact2 \
    --embodiment so101_ros -E url=ws://127.0.0.1:9090 -T max_steps=400 \
    --max-action-delta 0.65
```

`../../robot_console/run_task.sh` does both of those, plus the readiness gate and the
reset-and-verify step, and reports PASS/FAIL — it is the front door and this is the
manual version of it.

The topic set, and the traps in it, are in the repo's `CLAUDE.md` under "The ROS
contracts". The two that bite hardest: `/joint_states` comes back **alphabetically
sorted**, so index by name; and a `JointTrajectory` with no `header` is accepted and
silently ignored by a real controller.

Policies live in `robot_console`, never here. With a worktop robot in the fleet the
task's objects, cameras and arbiter are staged into whatever scene the engine compiled —
see `../shared/tasks/`.

## Platform notes (macOS)

- **The venv must be built on a framework Python**, not uv's standalone CPython.
  `mjpython` (which the viewer requires, because the MuJoCo passive viewer must
  own the main thread) needs a shared `libpython3.11.dylib` that the standalone
  build does not ship. `run.sh setup` looks for Homebrew's `python@3.11`.
- **`mjpython -m mujoco.viewer` does not work** — it re-executes the module and
  drops the handle mjpython stamps onto `mujoco.viewer` at startup, failing with
  `RuntimeError: Caught an unknown exception!`. The viewer must be launched from
  a *script*, which is what `kitchen.sh serve --mujoco` runs (`tools/spawn_robot.py`).
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

The robots are the ones `robots_specs/robots.yml` marks `simulated`, each with an
adapter in `robots/<id>/`; their descriptions and meshes are read from
`robots_specs/<id>/`.

```bash
../kitchen.sh serve --robots so101 --mujoco      # spawn it in a house, in a window
../kitchen.sh serve --robots so101                # ...on its ROS topics (see above)
python robots/so101/test_attach.py               # self-test in an empty world
python tools/test_placement.py                   # every robot stands at its placement
python ../shared/tests/physical_figures_check.py --engine molmospaces  # descriptions and published figures
python ../shared/tests/staging_check.py --engine molmospaces  # the six task objects, and nothing else
```

See [robots/README.md](robots/README.md) for how each was added and what its
limitations are — every one of them has at least one.

Drive the mobile robots by keyboard with a live camera feed from
[../robot_console](../robot_console).
