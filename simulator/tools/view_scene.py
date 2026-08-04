#!/usr/bin/env python
"""Open an MJCF in the MuJoCo passive viewer.

Must be run under ``mjpython`` on macOS. Note that ``mjpython -m mujoco.viewer``
does *not* work: mjpython imports ``mujoco.viewer`` and stamps a module-level
handle on it before handing control to the user program, and ``-m`` re-executes
the module from scratch, dropping that handle -- the viewer then dies with
"RuntimeError: Caught an unknown exception!". Running a script like this one
keeps mjpython's already-initialised module.

    mjpython tools/view_scene.py /path/to/scene.xml
"""

import argparse
import sys
import time

import mujoco
import mujoco.viewer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mjcf", help="absolute path to an MJCF .xml")
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="close automatically after N seconds (for smoke tests)",
    )
    args = ap.parse_args()

    print(f"loading {args.mjcf}", file=sys.stderr)
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    print(
        f"{model.nbody} bodies, {model.ngeom} geoms, {model.njnt} joints, {model.nv} dof",
        file=sys.stderr,
    )

    deadline = None if args.timeout is None else time.monotonic() + args.timeout
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            if deadline is not None and time.monotonic() > deadline:
                print("timeout reached, closing viewer", file=sys.stderr)
                break
            # Keep wall-clock roughly in step with simulated time.
            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
