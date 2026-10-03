# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Simulated robots that present the **same vendor ROS interface as the real hardware**, so a client written for the hardware (the robot console) works against the simulation unchanged. Three directories, each with its own `high_level_spec.md` (the root one is the workspace spec), plus the workspace's own `tests/`:

* `robots_specs/` — the robot registry and data: one robot file per robot, `robots_specs/<id>.md` (the registry; every robot is a single body), and under `robots_specs/<id>/` its URDF/MJCF models, meshes and recorded ROS interface (`ros.yml` ROS 1, `ros2.yml` ROS 2). **Normative for the simulator.**
* `simulator/` — MuJoCo scenes in two engines (`molmospaces`, `robocasa`), robot spawning, and one rosbridge websocket per spawned robot, served from a Docker container running the robot's real ROS distro.
* `robot_console/` — teleop (`teleop.sh`), camera/controls page (`view.sh`) and the `fleet` wire checker. Independent of the simulator: it derives its own profiles (`src/robot_console/profiles/*.yaml`) and never imports it.
* `tests/` — the workspace evidence harness (`evidence.sh` → `evidence.py`, `evidence_io.py`; writes `evidences/`, which is not committed) and the checks that span both projects (`test_evidence_*.py`, `test_console_stop_outcome.py`). The harness exercises both projects as installed, over rosbridge and the documented `--sim-port`, and imports neither; the opt-in stop-outcome test imports the console only to run its own `TeleopApp` against a simulated wire.

The specs are the source of truth: where code contradicts a spec, the code is wrong. Spec changes are made deliberately and dated ("amended 2026-MM-DD"). Production code of the simulator and console talk only over rosbridge; `tests/` (workspace) may read both trees but must not use one project's code or data as the oracle for the other.

## Commands

```sh
simulator/molmospaces/run.sh setup        # or robocasa/run.sh setup; once per engine (venv, upstream checkouts, assets, robot meshes, worktop objects)
simulator/kitchen.sh start [--engine molmospaces|robocasa] [--scene ithor:1|robocasa:50-1|test:1] [--mujoco] [--sim-port 9080]
simulator/spawn.sh <robot-id> [--placement worktop|floor]   # another terminal; wire on ws://127.0.0.1:9090
robot_console/teleop.sh --url ws://127.0.0.1:9090           # also view.sh; fleet: robot_console/python.sh -m robot_console.fleet --expect <id>
tests/evidence.sh [--engine E] [--robot ID] [--index-only]  # scene + camera pictures, smoke run, console check per robot/engine -> evidences/index.md
```

Tests run with an engine venv's python (it has MuJoCo, NumPy, PyYAML); one test: add `path::name` or `-k`.

```sh
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/unit simulator/tests/integration
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/e2e -k ainex   # needs Docker; slow
(cd robots_specs && ../simulator/molmospaces/.venv/bin/python -m pytest tests)
robot_console/python.sh -m pytest robot_console/tests                          # test_view_page.py needs Playwright
uv run --no-project --with pytest --with pyyaml --with numpy --with pillow pytest tests/test_evidence_script.py tests/test_evidence_index.py   # the index test fails until tests/evidence.sh has written evidences/
TELEOP_WIRES=myagv=ws://127.0.0.1:9090 robot_console/python.sh -m pytest tests/test_console_stop_outcome.py   # opt in; needs the robot spawned
```

Integration/e2e/evidence start real simulations and are timing-sensitive: a loaded host drops the real-time factor below 0.90 and fails smoke runs (the evidence harness waits for low load and retries). A leftover `simulation.py`/`spawn.py` holds ports 9080/9090 and fails the next run.

## Simulator architecture

* **One simulation process** (`simulator/shared/simulation.py`) owns the physics, all rendering and the MuJoCo window; it listens on `--sim-port` (framing and client in `protocol.py`; operations `hello`, `robots`, `spawn`, `scene`, `readings`, `bodies`, `render`... in `simulation.py`, documented in `simulator/README.md`). That port is simulator-private and never reaches a robot wire. There is no reset: restart to reset.
* **Scenes** (`scenes.py`) load each source the way the reference project did (`MjSpec.from_file` for iTHOR/ProcTHOR; RoboCasa's `KitchenArena` in a `ManipulationTask` with an empty robot list, its counter fixtures placed by RoboCasa's own fixture samplers as its environment does). `world.py` holds the live `MjSpec` and recompiles in one step when a robot enters or leaves, preserving every other state.
* **Placement** (`placement.py`, `worktop_survey.py`) is one function for both engines, measured by ray casting. The worktop is the surface the reference's survey ranks first; arms try the survey's spots in order. Arms need support/clearance/camera clearance; mobile robots on the floor also need guaranteed travel (not on the worktop).
* **Scene objects**: the six reference worktop objects (apple, plate, bowl, mug, banana, lemon; `worktop_objects.py`) are staged at start around the SO-101's spot, independent of robots; an arm that doesn't fit there takes them to its own spot while it is present. RoboCasa scenes also get RoboCasa's own objects on counters (`robocasa_objects.py`, sampled with RoboCasa's sampler; only objects that stand still upright are kept), and are solved by constraint island (`scenes.solve_by_island`): RoboCasa pins MuJoCo 3.3.1, whose whole-scene solve threw its lightest objects off the counters. Loose objects in the six objects' area are "cleared" (free joint removed, sunk 50 m).
* **Robots**: models come from `robots_specs/<id>/` (`model.xml` generated by `robots_specs/tools/models/<id>.py`); the spawn pose is the `home` keyframe. `spawn.sh` builds a Docker image per ROS distro and runs `shared/wire/` inside it with the repo **mounted read-only** (code edits apply without rebuilding); `wire/robots/<id>.py` is the per-robot plan (vendor nodes unmodified, simulated hardware). Served parameters/topics must equal the recorded `ros.yml`/`ros2.yml`; `simulator/tests/e2e/test_robots.py::test_contract` enforces it.

## Console architecture

`teleop_core.py` is the window-independent state machine (held-key motion, stop on release/Space/focus loss/exit, commands disabled after a failed stop); `teleop.py` is the pygame window around it. `view.py` serves the camera/controls page (`web/view.html`, `web/console.js`), `fleet.py` is the non-interactive wire check; `discovery.py` selects and type-checks the target against the profiles. Profiles in `profiles/*.yaml` carry each robot's topics, cameras, stop commands, speeds, head limits and bounded controls with source references.

## Conventions worth knowing

* Reference scenes: `kitchen.sh` must stay faithful to github.com/samirma/robot-simulator (rev `34547ae`): MolmoSpaces `ithor:1` and RoboCasa `robocasa:1-1` fixtures, plus the objects described above. Re-check parity after scene changes.
* When a recorded interface value is deliberately changed (e.g. AiNex arm init pose), change the `ros.yml` value with a note rather than overriding it in the wire, or the contract test fails.
* Meshes and object assets are fetched by `run.sh setup` and verified against `robots_specs/meshes.sha256` / `simulator/shared/objects/ycb.sha256`; they are not committed.
