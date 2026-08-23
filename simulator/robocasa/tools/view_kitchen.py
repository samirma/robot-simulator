#!/usr/bin/env python
"""Open a RoboCasa kitchen in the MuJoCo passive viewer.

Must be run under ``mjpython`` on macOS: the passive viewer must own the main
thread (``mjpython -m mujoco.viewer`` does not work either -- mjpython stamps
a module-level handle on ``mujoco.viewer`` before handing over, and ``-m``
re-executes the module from scratch, dropping it). Running a script like this
one keeps mjpython's already-initialised module.

    mjpython tools/view_kitchen.py --layout 1 --style 3 [--robot PandaOmron]
"""

import argparse
import sys
import time

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

import robocasa  # noqa: F401  -- registers the Kitchen envs with robosuite
from robocasa.models.scenes.scene_registry import LayoutType, StyleType


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", type=int, default=1, help="kitchen layout id (1-60)")
    ap.add_argument("--style", type=int, default=1, help="kitchen style id (1-60)")
    ap.add_argument("--robot", default="PandaOmron", help="robosuite robot name")
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="close automatically after N seconds (for smoke tests)",
    )
    args = ap.parse_args()

    for name, value, enum in (
        ("layout", args.layout, LayoutType),
        ("style", args.style, StyleType),
    ):
        if value not in [e.value for e in enum if e.value > 0]:
            print(f"error: {name} id {value} out of range (1-60)", file=sys.stderr)
            return 1

    print(f"layout {args.layout}, style {args.style}, robot {args.robot}", file=sys.stderr)
    env = robosuite.make(
        env_name="Kitchen",
        robots=args.robot,
        controller_configs=load_composite_controller_config(robot=args.robot),
        layout_ids=args.layout,
        style_ids=args.style,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
        translucent_robot=False,
    )
    env.reset()

    action = np.zeros(env.action_dim)
    deadline = None if args.timeout is None else time.monotonic() + args.timeout
    while True:
        step_start = time.time()
        env.step(action)
        env.render()
        # MjviewerRenderer wraps the passive viewer handle as .viewer.
        viewer = getattr(env.viewer, "viewer", None)
        if viewer is not None and not viewer.is_running():
            break
        if deadline is not None and time.monotonic() > deadline:
            print("timeout reached, closing viewer", file=sys.stderr)
            break
        # Keep wall-clock roughly in step with simulated time.
        slack = env.control_timestep - (time.time() - step_start)
        if slack > 0:
            time.sleep(slack)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
