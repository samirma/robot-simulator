# Vendor URDFs

Each robot's manufacturer description — its URDF, its official MJCF where one exists, and
the meshes they reference — is recorded once, under `robots_specs/<id>/` at the workspace
root, with its paths in `robots_specs/robots.yml`. The simulator reads them from there
through `shared/robots_spec.py`; nothing under `simulator/` keeps a copy.

What the simulator builds from them lives in `shared/robots/<id>/` (a generated MJCF,
converted meshes, menagerie's additions) or, for the AiNex, in `shared/ainex_model.py`.
See [README.md](README.md) for how each robot is actually built.

Deliberately **not** placed in `simulator/assets/` — that is `MLSPACES_ASSETS_DIR`, a
symlink tree the MolmoSpaces resource manager generates and force-refreshes, so
hand-curated files there could be pruned by a later `./run.sh assets`.

| Robot | Source | Used to load? |
|---|---|---|
| `so101` | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `Simulation/SO101` — `so101_new_calib.urdf`, `so101_new_calib.xml` + STL meshes | **meshes** — `model.xml` is menagerie's MJCF, loading the official meshes; the URDF is served as `robot_description` |
| `myagv` | [elephantrobotics/myagv_ros](https://github.com/elephantrobotics/myagv_ros) `myagv_ros_2023Pi` — `myAGV.urdf` + COLLADA meshes | **meshes only** — `make_model.py` converts the DAE files; the URDF itself is visualisation-only (no wheels, collision or inertia) |
| `ainex` | [Hiwonder/ainex](https://github.com/Hiwonder/ainex) `ainex_description` — `ainex.urdf.xacro` flattened to `ainex.urdf`, plus 25 STL meshes. **No licence stated** — see below | **yes** — no MJCF exists that is not itself a derivative of this |

## Licences

Every source above is under an identified licence **except `ainex`**. The Hiwonder
repository carries no LICENSE file despite describing itself as fully open source, and
that covers the URDF, the meshes and `servo_controller.yaml` alike — not merely the action
groups. It is used anyway, as a deliberate exception with the risk recorded rather than
hidden; `../../shared/robots/ainex/PROVENANCE.md` repeats the note. The third-party MuJoCo
ports of the same description (`Glowing-Torch/ainex_rl` and others) carry no licence
either, so they are not a way around it — which is one of the reasons the vendor
description is used directly rather than one of them.

Consequence for anything derived from vendor data: no Hiwonder action group is
redistributed here. `shared/ros_surfaces/ainex/actions.py` reads their `.d6a` format so that an owner
points `--action-dir` at their own robot and supplies the licensed data themselves.

## Loading one of these in MuJoCo

Two import behaviours bite every time:

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
