"""`arm_task.sh` -- a VLA model performs a task on the SO-101 in each simulator.

    python -m robot_console.arm_task                     # default task, both sims
    python -m robot_console.arm_task "push the bowl left"
    python -m robot_console.arm_task --dry-run --ports 8000

The policy is MolmoAct2-SO100_101 (see molmoact.py); the transport is the same
`molmospaces-control-v1` protocol `robot-console-arm` speaks, because that is the
SO-101's hardware contract. ROS was considered and rejected: the simulators' rosbridge
carries the myAGV's vendor topics only, and a real SO-101 does not speak ROS -- serving
the arm over ROS would be an interface no hardware presents, which is exactly what the
engines are forbidden to do.

Everything in this module is deliberately importable without torch: the model lives in
`molmoact.py` and is imported inside `main()` after argument parsing, so the offline
test suite (and `--help`) never pays for it. The split mirrors how `inspect_so101.py`
keeps the Inspect stack out of module scope.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import numpy as np

from robot_console.so101_driver import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    JOINT_HIGH,
    JOINT_LOW,
    MOTOR_KEYS,
)

# Matches the kitchen_arm.sh staging (a bowl and an apple inside the arm's reach) and
# deliberately mirrors the checkpoint card's own sample phrasing ("Move the arm towards
# the lemon, grasp it, lift it up, and drop it into the red bowl.") -- language close to
# the training distribution is the cheapest accuracy there is.
DEFAULT_TASK = "move the arm towards the apple, grasp it, lift it up, and drop it into the bowl"

DEFAULT_MODEL = "allenai/MolmoAct2-SO100_101"
DEFAULT_PORTS = "8000,8001"

# How far one control step may move a joint. The checkpoint was trained on real
# teleoperation, where 8 deg per 50 ms step is already brisk; anything asking for more
# is a mapping error, not a plan worth executing faster.
MAX_DELTA_DEG = 8.0
MAX_DELTA_GRIPPER = 15.0

# A correct absolute-joint-pose policy predicts its first step near the state it was
# fed. A first step further away than this is the signature of a frame mismatch.
PASSTHROUGH_WARN_DEG = 15.0


@dataclasses.dataclass(frozen=True)
class Convention:
    """Affine map between the driver's frame and the checkpoint's "robot scale".

    The driver speaks the LeRobot-calibrated convention (5 joints in +-100-ish degrees,
    gripper 0-100 %). The SO-100/101 training mixture is raw-er: the card's sample state
    carries 189/181 on joints 2-3 and a gripper value of 1.097. Rather than hardcode a
    guess, the map is a per-joint affine plus a gripper mode, exposed as flags -- and
    `action_to_driver` is the exact inverse of `state_to_model`, so a wrong-but-
    consistent setting round-trips instead of commanding a lunge.
    """

    joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    joint_scales: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    gripper_mode: str = "percent"  # percent | radians | fraction

    def state_to_model(self, driver_state: np.ndarray) -> np.ndarray:
        """Driver frame (deg x5, percent) -> model frame."""
        values = np.asarray(driver_state, dtype=np.float64).reshape(6).copy()
        values[:5] = values[:5] * np.asarray(self.joint_scales) + np.asarray(self.joint_offsets)
        percent = values[5]
        if self.gripper_mode == "radians":
            span = GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
            values[5] = GRIPPER_CLOSED_RAD + percent * span / 100.0
        elif self.gripper_mode == "fraction":
            values[5] = percent / 100.0
        elif self.gripper_mode != "percent":
            raise ValueError(f"unknown gripper mode {self.gripper_mode!r}")
        # float64 here; the float32 the model wants is cast at the policy boundary, so
        # the round trip through `action_to_driver` stays exact.
        return values

    def action_to_driver(self, model_action: np.ndarray) -> np.ndarray:
        """Model frame -> driver frame. Exact inverse of `state_to_model`."""
        values = np.asarray(model_action, dtype=np.float64).reshape(6).copy()
        values[:5] = (values[:5] - np.asarray(self.joint_offsets)) / np.asarray(self.joint_scales)
        raw = values[5]
        if self.gripper_mode == "radians":
            span = GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
            values[5] = (raw - GRIPPER_CLOSED_RAD) * 100.0 / span
        elif self.gripper_mode == "fraction":
            values[5] = raw * 100.0
        elif self.gripper_mode != "percent":
            raise ValueError(f"unknown gripper mode {self.gripper_mode!r}")
        return values


def observation_to_state(obs: Mapping[str, Any]) -> np.ndarray:
    """The 6-vector state in the driver's frame, in MOTOR_KEYS order."""
    return np.array([float(obs[key]) for key in MOTOR_KEYS], dtype=np.float64)


def action_to_driver_dict(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(6)
    return {key: float(v) for key, v in zip(MOTOR_KEYS, values)}


def clamp_step(current: np.ndarray, target: np.ndarray,
               max_delta_deg: float = MAX_DELTA_DEG,
               max_delta_gripper: float = MAX_DELTA_GRIPPER) -> np.ndarray:
    """Limit one control step's travel. The driver's absolute clip still applies after.

    Two layers on purpose: the absolute clip bounds *where* the arm can be told to go,
    this bounds *how fast* it is allowed to get there. Under a wrong unit convention the
    worst case is then a slow drift toward a limit, never a whip.
    """
    current = np.asarray(current, dtype=np.float64).reshape(6)
    target = np.asarray(target, dtype=np.float64).reshape(6)
    limit = np.array([max_delta_deg] * 5 + [max_delta_gripper])
    return current + np.clip(target - current, -limit, limit)


def parse_ports(spec: str) -> list[int]:
    try:
        ports = [int(p) for p in spec.split(",") if p.strip()]
    except ValueError:
        raise SystemExit(f"--ports: expected comma-separated integers, got {spec!r}")
    if not ports:
        raise SystemExit("--ports: no ports given")
    return ports


def chunk_warnings(chunk: np.ndarray, state_model: np.ndarray, conv: Convention) -> list[str]:
    """Sanity-check a predicted chunk before anything moves.

    These are the three signatures of a broken driver<->model mapping, and they are
    checked every prediction because a mapping that is right at the rest pose can still
    be wrong once the arm moves.
    """
    warnings: list[str] = []
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 6:
        return [f"chunk has shape {chunk.shape}, expected (N, 6)"]
    if not np.all(np.isfinite(chunk)):
        warnings.append("chunk contains NaN/inf")
        return warnings

    first_step_gap = np.abs(chunk[0, :5] - np.asarray(state_model[:5], dtype=np.float64))
    if np.any(first_step_gap > PASSTHROUGH_WARN_DEG):
        worst = int(np.argmax(first_step_gap))
        warnings.append(
            f"passthrough check: chunk step 0 is {first_step_gap[worst]:.1f} deg from the "
            f"fed state on joint {worst} (> {PASSTHROUGH_WARN_DEG:.0f}); the frame mapping "
            "is probably wrong"
        )

    in_driver = np.array([conv.action_to_driver(step) for step in chunk])
    below = in_driver < np.asarray(JOINT_LOW) - 1e-9
    above = in_driver > np.asarray(JOINT_HIGH) + 1e-9
    if below.any() or above.any():
        n = int(below.sum() + above.sum())
        warnings.append(
            f"{n} chunk values map outside the driver's joint limits (they will be clipped)"
        )

    if np.allclose(chunk, chunk[0], atol=1e-6) and len(chunk) > 1:
        warnings.append("chunk is constant across all steps")
    return warnings


def format_prediction(state_driver: np.ndarray, state_model: np.ndarray,
                      chunk: np.ndarray, conv: Convention) -> str:
    """The dry-run table: what was fed, what came back, what would be sent."""
    lines = [
        "  joint            state(driver)  state(model)  chunk[0](model)  chunk[0]->driver",
    ]
    first_driver = conv.action_to_driver(chunk[0])
    names = [key.split(".")[0] for key in MOTOR_KEYS]
    for i, name in enumerate(names):
        lines.append(
            f"  {name:<16} {state_driver[i]:>12.2f} {state_model[i]:>13.2f} "
            f"{float(chunk[0][i]):>16.2f} {first_driver[i]:>17.2f}"
        )
    lines.append(f"  chunk: {len(chunk)} steps")
    return "\n".join(lines)


def run_episode(
    policy: Any,
    driver: Any,
    task: str,
    conv: Convention,
    *,
    max_steps: int = 20,
    seconds: Optional[float] = None,
    execute_steps: int = 10,
    max_delta_deg: float = MAX_DELTA_DEG,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Predict-and-execute until the step or time budget runs out.

    `policy` needs `predict_chunk(image_rgb, task, state) -> (N, 6)`; `driver` is the
    `SimulatorSO101` surface (`get_observation`, `send_action`). Both are duck-typed so
    the offline tests can pass stubs. Returns the number of predictions made.

    Not sending an action *is* holding still: the ArmClient thread answers every
    observation with the last commanded target, so a dry run predicts and prints while
    the arm stays parked.
    """
    deadline = None if seconds is None else clock() + seconds
    predictions = 0
    while predictions < max_steps:
        if deadline is not None and clock() >= deadline:
            break
        obs = driver.get_observation()
        state_driver = observation_to_state(obs)
        state_model = conv.state_to_model(state_driver)
        chunk = np.asarray(policy.predict_chunk(obs["front"], task, state_model))
        predictions += 1

        for warning in chunk_warnings(chunk, state_model, conv):
            log(f"warning: {warning}")
        log(f"prediction {predictions}:")
        log(format_prediction(state_driver, state_model, chunk, conv))
        if dry_run:
            continue

        current = state_driver
        for step in chunk[:execute_steps]:
            if deadline is not None and clock() >= deadline:
                break
            target = clamp_step(current, conv.action_to_driver(step), max_delta_deg)
            sent = driver.send_action(action_to_driver_dict(target))
            current = np.array([sent[key] for key in MOTOR_KEYS])
            # Re-observe between steps: get_observation blocks on a fresh frame, so
            # this paces execution at the simulator's control rate.
            obs = driver.get_observation()
    return predictions


def parse_float_csv(spec: str, name: str, n: int) -> tuple[float, ...]:
    parts = [p for p in spec.split(",") if p.strip()]
    if len(parts) != n:
        raise SystemExit(f"{name}: expected {n} comma-separated values, got {len(parts)}")
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise SystemExit(f"{name}: expected numbers, got {spec!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="arm_task",
        description="A VLA model performs a task on the SO-101 in each simulator.",
    )
    ap.add_argument("task", nargs="*", help=f"task text (default: {DEFAULT_TASK!r})")
    ap.add_argument("--task", dest="task_flag", default=None,
                    help="task text as one flag (overrides positional words)")
    ap.add_argument("--ports", default=DEFAULT_PORTS,
                    help="control ports to run against, in order (default %(default)s "
                         "= the two kitchen_arm.sh engines)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--connect-timeout", type=float, default=300.0)
    ap.add_argument("--seconds", type=float, default=None,
                    help="wall-clock budget per simulator (default: until --max-steps)")
    ap.add_argument("--max-steps", type=int, default=20,
                    help="model predictions per simulator (default %(default)s)")
    ap.add_argument("--execute-steps", type=int, default=10,
                    help="steps of each predicted chunk to execute before re-observing "
                         "(default %(default)s of the 30-step chunk)")
    ap.add_argument("--dry-run", action="store_true",
                    help="predict and print; the arm holds still")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="mps", choices=("mps", "cpu"))
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    # Tuned against the live checkpoint, not guessed: with these offsets the model's
    # first predicted step lands within ~3 deg of the fed state (the passthrough
    # check), where identity left shoulder_lift saturated 80 deg away at the training
    # distribution's edge. The SO-100/101 mixture uses the old LeRobot raw-motor frame,
    # where the lift and elbow rest near 180 and 90 rather than zero.
    ap.add_argument("--joint-offsets", default="0,180,90,0,0", metavar="A,B,C,D,E",
                    help="degrees added per joint going driver->model (default "
                         "%(default)s, the SO-100/101 mixture's raw-motor frame)")
    ap.add_argument("--joint-scales", default="1,1,1,1,1", metavar="A,B,C,D,E")
    ap.add_argument("--gripper-mode", default="percent",
                    choices=("percent", "radians", "fraction"))
    ap.add_argument("--max-delta-deg", type=float, default=MAX_DELTA_DEG,
                    help="per-step joint travel limit (default %(default)s)")
    return ap


def resolve_task(args: argparse.Namespace) -> str:
    if args.task_flag:
        return args.task_flag
    if args.task:
        return " ".join(args.task)
    return DEFAULT_TASK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    task = resolve_task(args)
    ports = parse_ports(args.ports)
    conv = Convention(
        joint_offsets=parse_float_csv(args.joint_offsets, "--joint-offsets", 5),
        joint_scales=parse_float_csv(args.joint_scales, "--joint-scales", 5),
        gripper_mode=args.gripper_mode,
    )

    # Torch enters here and not before: --help and the tests never pay for it.
    from robot_console.molmoact import MolmoActPolicy

    print(f"task: {task!r}")
    policy = MolmoActPolicy(args.model, device=args.device, dtype=args.dtype)

    import threading

    from robot_console.arm_client import ArmClient
    from robot_console.so101_driver import SimulatorSO101

    exit_code = 0
    for port in ports:
        print(f"\n=== simulator on {args.host}:{port} ===")
        driver = SimulatorSO101()
        client = ArmClient(driver, args.host, port, args.connect_timeout)
        client_error: list[BaseException] = []

        def serve_observations() -> None:
            try:
                client.run()
            except Exception as exc:  # noqa: BLE001 - forward transport failures.
                client_error.append(exc)
            finally:
                driver.disconnect()

        thread = threading.Thread(target=serve_observations, daemon=True)
        thread.start()
        try:
            run_episode(
                policy,
                driver,
                task,
                conv,
                max_steps=args.max_steps,
                seconds=args.seconds,
                execute_steps=args.execute_steps,
                max_delta_deg=args.max_delta_deg,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001 - keep going to the next simulator.
            print(f"error on port {port}: {exc}", file=sys.stderr)
            exit_code = 1
        finally:
            driver.disconnect()
            client.close()
            thread.join(timeout=5.0)
        if client_error:
            print(f"transport error on port {port}: {client_error[0]}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
