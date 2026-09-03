# Third-party assets

| Asset | Source | License |
|---|---|---|
| `ycb/*` meshes + textures | [elpis-lab/ycb_dataset](https://github.com/elpis-lab/ycb_dataset), wrapping the [YCB Object and Model Set](https://www.ycbbenchmarks.com/object-set/) | MIT (wrapper); YCB meshes free for research |

Objects vendored: apple, plate. They are the *visual* geometry of the apple-on-plate
task only -- physics runs on invisible primitives (see `tasks/apple_on_plate.py`), so a
mesh here can be swapped for looks without touching the grasp tuning.
