# Third-party assets

| Asset | Source | License |
|---|---|---|
| `ycb/*` meshes + textures | [elpis-lab/ycb_dataset](https://github.com/elpis-lab/ycb_dataset), wrapping the [YCB Object and Model Set](https://www.ycbbenchmarks.com/object-set/) | MIT (wrapper); YCB meshes free for research |

Objects vendored: apple, plate, bowl, mug, banana, lemon — the full set the reference
apple-on-plate scene puts on its table (see `tasks/apple_on_plate.py`).

How they are used differs by object, and the difference is deliberate. The apple and the
plate are *visual* meshes only: their physics runs on invisible primitives (a sphere, a
cylinder plus a rim of boxes) that carry every tuned contact parameter, so a mesh can be
swapped for looks without touching the grasp tuning. The bowl, mug, banana and lemon are
scenery the arm never has to grasp, and for them the textured mesh *is* the collider.

---

*This file is copied from the reference project (github.com/samirma/robot-simulator rev
34547ae, `simulator/shared/tasks/assets/LICENSES.md`). Here the six objects are not
committed: each engine's `run.sh setup` fetches them (`simulator/shared/tools/fetch_objects.py`)
from elpis-lab/YCB_Dataset at commit `9e8c6488a2ff673d9aa48a91492fb89423c1b106`, whose
files are byte-identical to the reference's, into `ycb/`, verified against `ycb.sha256`.
They are staged by `simulator/shared/worktop_objects.py`.*
