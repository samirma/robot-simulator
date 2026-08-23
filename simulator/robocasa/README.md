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

## Not yet wired (molmospaces parity, TODO)

Robot spawning (`so101`, `myagv`), the rosbridge/control servers from
`../shared/contracts/`, and out-of-tree `robots/` adapters are not implemented
in this engine yet.
