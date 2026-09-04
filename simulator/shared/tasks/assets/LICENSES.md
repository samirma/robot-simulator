# Third-party assets

| Asset | Source | License |
|---|---|---|
| `ycb/*` meshes + textures | [elpis-lab/ycb_dataset](https://github.com/elpis-lab/ycb_dataset), wrapping the [YCB Object and Model Set](https://www.ycbbenchmarks.com/object-set/) | MIT (wrapper); YCB meshes free for research |
| `textures/light-wood.png` | [robosuite](https://github.com/ARISE-Initiative/robosuite) `models/assets/textures` | Apache-2.0 |

Objects vendored: apple, plate, bowl, mug, banana, lemon — the full set the reference
apple-on-plate scene puts on its table (see `tasks/apple_on_plate.py`).

How they are used differs by object, and the difference is deliberate. The apple and the
plate are *visual* meshes only: their physics runs on invisible primitives (a sphere, a
cylinder plus a rim of boxes) that carry every tuned contact parameter, so a mesh can be
swapped for looks without touching the grasp tuning. The bowl, mug, banana and lemon are
scenery the arm never has to grasp, and for them the textured mesh *is* the collider.
