# RoboCasa engine (simulator engine #2)

MuJoCo + robosuite + RoboCasa kitchens. Mirrors the `simulator/molmospaces/`
launcher surface; see `../molmospaces/README.md` and the repo `CLAUDE.md` for
the shared architecture (engines feed the one wire bridge in
`../shared/contracts/`).

```bash
./run.sh setup                    # clone upstream robosuite/robocasa + editable install
./run.sh assets                   # kitchen assets (~10 GB) into upstream/robocasa/.../assets
./run.sh --layout 1 --style 3     # open that kitchen in the MuJoCo viewer
./run.sh view --layout 2 --style 7 --robot PandaMobile --timeout 10
./run.sh shell

# A shared robot in the kitchen, on the real hardware's interface:
./run.sh view --robot myagv --ros-port 9090          # myAGV vendor ROS topics
./run.sh view --robot so101 --control 127.0.0.1:8000 # SO-101 control protocol
./run.sh view --robot so101 --objects bowl,apple     # ...with objects in its reach
./run.sh view --robot myagv --headless --ros-port 9090   # no window
./run.sh view --robot myagv --render /tmp/kitchen.png    # just a PNG
```

- `--layout` 1-60, `--style` 1-60 (1-10 are the "test" set; see
  `upstream/robocasa/robocasa/models/scenes/scene_registry.py`). Both default
  to 1.
- `./run.sh <flags>` without a subcommand is shorthand for `view <flags>`.
- On macOS the viewer runs under `mjpython` (main-thread constraint, same as
  the molmospaces engine); everything else runs under plain `python`.

## Layout

- `upstream/` — pinned clones: robosuite `master` (robocasa v1.0 passes
  `lite_physics`/`load_model_on_init`, which no v1.5.x tag accepts) and
  robocasa `v1.0`. The venv installs both as *editable* packages, so these
  directories must stay put. **Never modify `upstream/`.**
- `tools/view_kitchen.py` — builds the `Kitchen` env for one layout/style pair
  and opens the passive viewer. `--timeout N` makes it smoke-testable.
- `tools/download_lightwheel_assets.py` — fetches the fixture/object assets
  the v1.0 downloader misses (renamed nvidia HF repo, base `fixtures.zip`);
  run by `./run.sh assets`.
- `env.sh` — venv/upstream paths, `MUJOCO_GL`, plus the
  `DYLD_FALLBACK_LIBRARY_PATH` fix that numba/llvmlite needs under
  `mjpython`; sourced by `run.sh`.

## Shared robots in a kitchen

`tools/spawn_robot.py` is the RoboCasa counterpart of the MolmoSpaces tool of the
same name: it puts a robot from `../shared/robots/` into a kitchen and presents
it on the *hardware's* interface, so `robot_console` drives it with the same
client and the same flags it uses against engine #1.

- `--robot myagv --ros-port 9090` — the vendor `myagv_ros` topics (`/cmd_vel` in;
  `/odom`, camera, `/scan`, depth out), served by `../shared/ros_surfaces/myagv.py`.
- `--robot so101 --control HOST:PORT` — the `molmospaces-control-v1` protocol.
- `--objects bowl,apple` — RoboCasa objects spawned inside the arm's working
  annulus. A RoboCasa kitchen is *fixtures*; it ships with no loose objects at
  all, so an arm gets nothing to reach for until this puts something there.
- `--headless`, `--render out.png`, `--timeout N` for automated checks.

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

For the SO-101 in both engines side by side, see `../kitchen_arm.sh`.
