#!/usr/bin/env python
"""Run the simulator with the robot driven by an external control server.

Same scene/robot/task setup as ``scripts/datagen/run_pipeline.py`` -- its
``setup_config`` is reused verbatim -- except the scripted planner policy is
replaced by :class:`bridge.policy.BridgePolicy`, which forwards observations to
``bridge/server.py`` and applies the actions it returns.

    ./run.sh serve --controller wave     # terminal 1
    ./run.sh bridge --robot droid        # terminal 2

Start the server first: the simulator retries the connection but gives up after
``--connect-timeout`` seconds.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# The launcher puts both the repo root and scripts/datagen on PYTHONPATH, but
# make direct `python bridge/run_bridge_sim.py` work too.
SIM_ROOT = Path(__file__).resolve().parent.parent
for extra in (SIM_ROOT, SIM_ROOT / "molmospaces", SIM_ROOT / "molmospaces" / "scripts" / "datagen"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from bridge.policy import BridgePolicyConfig  # noqa: E402

log = logging.getLogger("bridge.sim")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Scene / robot / task selection -- mirrors run_pipeline.py.
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--robot", default="droid", help="franka, droid, rum, rby1, yam, bimanual_yam")
    ap.add_argument("--task_type", default="pick")
    ap.add_argument("--scene_dataset", default="ithor")
    ap.add_argument("--data_split", default="train")
    ap.add_argument("--house_inds", type=int, default=1)
    ap.add_argument("--target_types", default=None)
    ap.add_argument("--samples_per_house", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--single_step", action="store_true")
    ap.add_argument("--filter_for_successful_trajectories", action="store_true")
    ap.add_argument("--randomize_lighting", type=bool, default=False)
    ap.add_argument("--randomize_textures", type=bool, default=False)
    ap.add_argument("--randomize_dynamics", type=bool, default=False)
    ap.add_argument("--randomize_scene", type=bool, default=False)
    ap.add_argument("--run_name_prefix", default="bridge")
    ap.add_argument("--eval", default=None)
    ap.add_argument("--config", default=None)

    # Bridge-specific.
    ap.add_argument("--host", default="127.0.0.1", help="control server host")
    ap.add_argument("--port", type=int, default=8000, help="control server port")
    ap.add_argument(
        "--connect-timeout",
        type=float,
        default=120.0,
        dest="connect_timeout",
        help="seconds to wait for the control server before giving up",
    )
    ap.add_argument(
        "--task_horizon", type=int, default=1000, help="max steps before the episode ends"
    )
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_parser().parse_args()

    from run_pipeline import MyRolloutRunner, get_output_dir, setup_config

    # run_pipeline.setup_config reads `.policy` off the namespace in some paths.
    args.policy = "bridge"

    exp_config = setup_config(args)
    exp_config.num_workers = 1
    exp_config.use_passive_viewer = args.viewer
    exp_config.task_horizon = args.task_horizon
    # External control is interactive; don't throw away episodes it didn't finish.
    exp_config.filter_for_successful_trajectories = False

    exp_config.policy_config = BridgePolicyConfig(
        remote_config=dict(host=args.host, port=args.port),
        connection_timeout=args.connect_timeout,
    )
    policy = exp_config.policy_config.policy_factory(exp_config, None)

    exp_config.output_dir = get_output_dir(args, exp_config)
    exp_config.save_config()

    log.info("connecting to control server at ws://%s:%d", args.host, args.port)
    MyRolloutRunner(exp_config).run(preloaded_policy=policy)


if __name__ == "__main__":
    main()
