# Vendor URDFs

Each robot keeps its manufacturer robot description under `robots/<name>/urdf/`, with
the meshes it references.

These are **reference copies**, not what the simulator loads — with one exception noted
below. The simulator runs from MJCF, because URDF carries no actuators, sites, cameras,
collision classes or solver settings, so a URDF-sourced robot needs all of that
reconstructed by hand. Where a maintained MJCF exists it already has tuned gains and
calibrated keyframes; discarding that to re-derive it worse would be a poor trade. See
[README.md](README.md) for how each robot is actually built.

They are kept because they are the authoritative vendor description: useful for export
to other tools (Isaac, PyBullet, ROS/RViz), for checking dimensions and joint limits
against what we model, and as provenance for the numbers in each `make_model.py`.

Deliberately **not** placed in `simulator/assets/` — that is `MLSPACES_ASSETS_DIR`, a
symlink tree the MolmoSpaces resource manager generates and force-refreshes, so
hand-curated files there could be pruned by a later `./run.sh assets`.

| Robot | Source | Used to load? |
|---|---|---|
| `so101` | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `Simulation/SO101` — `so101_new_calib.urdf`, `so101_old_calib.urdf` | no — loads Menagerie MJCF |
| `myagv` | [elephantrobotics/myagv_ros](https://github.com/elephantrobotics/myagv_ros) `myagv_ros_2023Pi` — `myAGV.urdf` + COLLADA meshes | **meshes only** — `make_model.py` converts the DAE files; the URDF itself is visualisation-only (no wheels, collision or inertia) |
| `rebot_b601` | [vectorBH6/reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py) `00-arm-rs_asm-v3.urdf` + STL meshes, plus the `reBot-DevArm_fixend_description` variant | **yes** — loaded from URDF, since no MJCF exists for it |
| `ainex` | [Hiwonder/ainex](https://github.com/Hiwonder/ainex) `ainex_description` — `ainex.urdf.xacro` flattened to `ainex.urdf`, plus 25 STL meshes. **No licence stated** — see below | **yes** — no MJCF exists that is not itself a derivative of this |

## Licences

Every source above is under an identified licence **except `ainex`**. The Hiwonder
repository carries no LICENSE file despite describing itself as fully open source, and
that covers the URDF, the meshes and `servo_controller.yaml` alike — not merely the action
groups. It is vendored anyway, as a deliberate exception with the risk recorded rather than
hidden; `ainex/urdf/PROVENANCE.md` repeats the note beside the files themselves. The
third-party MuJoCo ports of the same description (`Glowing-Torch/ainex_rl` and others)
carry no licence either, so they are not a way around it — which is one of the reasons the
vendor description is used directly rather than one of them.

Consequence for anything derived from vendor data: no Hiwonder action group is
redistributed here. `robots/ainex/actions.py` reads their `.d6a` format so that an owner
points `--action-dir` at their own robot and supplies the licensed data themselves.

## Loading one of these in MuJoCo

Two import behaviours bite every time (both learned the hard way on `rebot_b601`):

- MuJoCo **strips the directory from URDF mesh filenames**, so `meshdir` must point at
  the mesh folder — and a copy elsewhere on disk silently fails to find its meshes.
- MuJoCo **merges the URDF root link into the worldbody** when it carries no joint, so
  the first real body becomes the root.

Two more that `ainex` added, both of which produce a model that compiles and looks wrong
rather than one that fails:

- The merge above can happen **more than once**. The AiNex has a jointless `base_link`
  *and* a `body_link` attached to it by a fixed joint, so both merge and the robot comes
  out as five disconnected root bodies with a third of its mass missing. Giving the
  intended root a joint is what stops it.
- **`discardvisual` defaults to true for URDF** (and false for MJCF). Any surgery that
  makes a geom non-colliding therefore *deletes* it, so a robot deliberately reduced to a
  single collision hull renders as nothing at all. Set `spec.compiler.discardvisual =
  False` before compiling.
