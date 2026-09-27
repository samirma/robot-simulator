"""The simulator's own record of where the task's things really were, for an offline audit.

The console grades `apple_on_plate` from public observations only (the rig's images and
the arm's joint states). `robot_console/src/robot_console/arm/audit.py` is the separate
check that those verdicts are right, and it needs the truth: where the apple, the plate
and the fingers actually were. That truth never goes on the wire -- a task-success topic or
anything like one is exactly what spec §3 forbids -- so it goes to a **local file**, and
only when asked:

    SIMULATOR_TRUTH_LOG=simulator/runs/truth.jsonl ./kitchen.sh serve

appends JSON lines to that path (created with its folder; `simulator/runs/` is ignored by
git), in the format `audit.py` documents:

* `{"kind": "reset", "wall": <unix s>, "stamp": <sim s>}` once each `/reset` has completed
  and its observations are out;
* `{"kind": "state", "stamp": <sim s>, "apple": [x, y, z], "plate": [x, y, z],
  "fingers": [[[x, y, z], [x, y, z]], ...]}` for each rig frame, `stamp` the frame's own
  `header.stamp` (the `data.time` it was rendered from), positions in the rig's root frame
  `scene/worktop` -- the arm base frame the rig's constants are written in, the task
  arbiter's -- as the apple body's and plate body's centres and each jaw's gripping segment
  from the root of its pad to its tip.

The fingers are read off the compiled model's jaw geometry, not the console's constants,
so the audit compares two independent statements: the fixed jaw from its innermost pad box
(`fixed_jaw_box7`) to its tip sphere (`fixed_jaw_sph_tip1`), the moving jaw likewise
(`moving_jaw_box3` to `moving_jaw_sph_tip1`). With no SO-101 in the fleet the list is
empty.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

#: The environment variable `kitchen.sh serve` reads (the console names it too, as
#: `audit.TRUTH_ENV`).
TRUTH_ENV = "SIMULATOR_TRUTH_LOG"

#: (body, pad-root geom, tip geom) per finger, bare MJCF names of the SO-101 model.
FINGERS = (
    ("gripper", "fixed_jaw_box7", "fixed_jaw_sph_tip1"),
    ("moving_jaw_so101_v1", "moving_jaw_box3", "moving_jaw_sph_tip1"),
)


def _named(model, kind, name: str, prefix: str) -> int:
    """`prefix + name`, or the one object whose name ends in `/name`; -1 when neither."""
    for candidate in (f"{prefix}{name}", name):
        found = mujoco.mj_name2id(model, kind, candidate)
        if found >= 0:
            return found
    count = {mujoco.mjtObj.mjOBJ_GEOM: model.ngeom, mujoco.mjtObj.mjOBJ_BODY: model.nbody}[kind]
    matches = [i for i in range(count)
               if (mujoco.mj_id2name(model, kind, i) or "").endswith("/" + name)]
    return matches[0] if len(matches) == 1 else -1


class TruthLog:
    """Appends the truth to one local file; safe to call from the loop and server threads."""

    def __init__(self, path: str | os.PathLike, model, task, *, arm_prefix: str | None) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._task = task
        self._model = model
        # Kinematics of the frame's own state, on a buffer of this log's: the same qpos
        # the render worker draws the frame from, not the loop's last step.
        self._scratch = mujoco.MjData(model)
        self._fingers: list[tuple[int, int]] = []
        if arm_prefix is not None:
            for _body, root, tip in FINGERS:
                ids = (_named(model, mujoco.mjtObj.mjOBJ_GEOM, root, arm_prefix),
                       _named(model, mujoco.mjtObj.mjOBJ_GEOM, tip, arm_prefix))
                if min(ids) < 0:
                    raise SystemExit(f"truth log: the SO-101's {root}/{tip} geoms are not in "
                                     "this model; the fingers cannot be recorded")
                self._fingers.append(ids)
        print(f"truth log: appending to {self.path} ({TRUTH_ENV})", file=sys.stderr)

    def _write(self, row: dict) -> None:
        line = json.dumps(row, separators=(",", ":"))
        with self._lock:
            if not self._file.closed:
                self._file.write(line + "\n")

    def reset(self, stamp_s: float) -> None:
        """`/reset` has completed and its observations are out."""
        self._write({"kind": "reset", "wall": time.time(), "stamp": float(stamp_s)})

    def state(self, data, stamp_s: float) -> None:
        """One rig frame's truth, from the `data` the frame is rendered from.

        Called on the loop thread as the frame is handed to its render worker; the
        kinematics are recomputed from that state exactly as the worker recomputes them.
        """
        own = self._scratch
        own.qpos[:] = data.qpos
        own.mocap_pos[:] = data.mocap_pos
        own.mocap_quat[:] = data.mocap_quat
        mujoco.mj_kinematics(self._model, own)
        to_base = self._task.base_from_world

        def local(world) -> list[float]:
            p = to_base[:3, :3] @ np.asarray(world, dtype=np.float64) + to_base[:3, 3]
            return [round(float(v), 6) for v in p]

        where = self._task.object_positions(own)
        self._write({
            "kind": "state", "stamp": float(stamp_s),
            "apple": local(where["apple"]), "plate": local(where["plate"]),
            "fingers": [[local(own.geom_xpos[root]), local(own.geom_xpos[tip])]
                        for root, tip in self._fingers],
        })

    def close(self) -> None:
        with self._lock:
            self._file.close()


def from_environment(model, task, *, arm_prefix: str | None) -> TruthLog | None:
    """A `TruthLog` when `SIMULATOR_TRUTH_LOG` names a path and a task is staged."""
    path = os.environ.get(TRUTH_ENV, "").strip()
    if not path:
        return None
    if task is None:
        print(f"truth log: {TRUTH_ENV} is set but no task is staged; nothing to record",
              file=sys.stderr)
        return None
    return TruthLog(path, model, task, arm_prefix=arm_prefix)
