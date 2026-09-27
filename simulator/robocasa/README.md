# RoboCasa engine (simulator engine #2)

MuJoCo + robosuite + RoboCasa kitchens. Mirrors the `simulator/molmospaces/`
launcher surface; see `../molmospaces/README.md` and the repo `CLAUDE.md` for
the shared architecture (engines feed the one wire bridge in
`../shared/contracts/`).

```bash
./run.sh setup                    # clone upstream robosuite/robocasa + editable install
./run.sh assets                   # kitchen assets (~10 GB) into upstream/robocasa/.../assets
./run.sh shell

# A simulated robot (robots_specs/robots.yml) in a kitchen, on the real hardware's interface:
./run.sh view --robot so101 --layout 2 --style 7     # that kitchen in the MuJoCo viewer
./run.sh view --robot myagv --render /tmp/kitchen.png    # just a PNG, headless
../kitchen.sh serve --engine robocasa --robots myagv     # on the wire: that is serve's job
python tools/test_placement.py                       # every robot stands at its placement
python ../shared/tests/attach_check.py --engine robocasa  # every robot attaches and moves
```

- `--layout` 1-60, `--style` 1-60 (1-10 are the "test" set; see
  `upstream/robocasa/robocasa/models/scenes/scene_registry.py`). Both default
  to 1.
- `./run.sh <flags>` without a subcommand is shorthand for `view <flags>`.
- `view` always takes `--robot <id>`, an id `robots_specs/robots.yml` marks `simulated`,
  and serves no wire. RoboCasa is a scene provider only: no robosuite robot is ever loaded.
- On macOS the viewer runs under `mjpython` (main-thread constraint, same as
  the molmospaces engine); everything else runs under plain `python`.

## Layout

- `upstream/` — pinned clones: robosuite `master` (robocasa v1.0 passes
  `lite_physics`/`load_model_on_init`, which no v1.5.x tag accepts) and
  robocasa `v1.0`. The venv installs both as *editable* packages, so these
  directories must stay put. **Never modify `upstream/`.**
- `tools/download_lightwheel_assets.py` — fetches the fixture/object assets
  the v1.0 downloader misses (renamed nvidia HF repo, base `fixtures.zip`);
  run by `./run.sh assets`.
- `env.sh` — venv/upstream paths, `MUJOCO_GL`, plus the
  `DYLD_FALLBACK_LIBRARY_PATH` fix that numba/llvmlite needs under
  `mjpython`; sourced by `run.sh`.

## Shared robots in a kitchen

`tools/spawn_robot.py` is this engine's half of `../shared/spawn.py`, the spawn tool both
engines run: how a kitchen is built, which counter is the worktop and where the task
robot is mounted on it, where the open floor is, and how a shared robot model
(`../shared/robots/`, `../shared/ainex_model.py`) is grafted in. Everything else --
the command line, where each robot stands, the task, the ROS interfaces
(`../shared/ros_surfaces/`), rendering and the loop -- is the shared code, so
`robot_console` drives a robot here with the same client it uses against engine #1 and
cannot tell them apart.

**RoboCasa is a scene provider here, not a robot stack.** The kitchen comes from
`KitchenArena` with `mujoco_robots=[]` — 44 fixtures, 825 geoms, zero actuators —
and the shared robot MJCF is grafted into that spec and stepped by plain MuJoCo.
Going through `robosuite.make` would drag in a robosuite robot with its own
controller stack and action space, which would then have to be cut back out of
the compiled model, and would stand a Panda in the middle of every camera frame.

Two RoboCasa-specific traps are documented at length in the file, because both
produce results that look like bugs elsewhere:

- **Geom groups are inverted from the MolmoSpaces convention** — collision hulls
  are group 0 (painted in random translucent colours, 501 of them in layout 1)
  and visual meshes are group 1. Everything that renders goes through
  `visual_only()`; skip it and the camera streams a scene full of red and green
  boxes.
- **Worktops come from RoboCasa, not from geometry.** `Counter.get_reset_regions()`
  returns the free worktop rectangles the dataset itself places objects on.
  Inferring them from collision AABBs instead fails in a specific and repeatable
  way: a sink basin's floor is "a flat surface at counter height" whose centre is
  a clean 0.22 m clear of anything, so it beats every real worktop on score and
  the arm gets mounted in the sink.

For the SO-101 in both engines side by side, see `../kitchen.sh`.
