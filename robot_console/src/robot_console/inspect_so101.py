"""Control a simulated SO-101 with the unmodified Inspect Robots embodiment."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from .arm_client import ArmClient
from .so101_driver import JOINT_HIGH, JOINT_LOW, SimulatorSO101, _decode_front


class RecordingSO101(SimulatorSO101):
    """A SimulatorSO101 that also writes every camera frame to a Recorder.

    The tee lives in ``__call__`` deliberately: the ArmClient thread sees every
    simulator observation at the control rate whether or not the policy is thinking,
    so the video shows the whole episode -- including the seconds the arm holds still
    waiting on an LLM -- rather than only the frames the embodiment happened to poll.
    """

    def __init__(self, recorder) -> None:
        super().__init__()
        self._recorder = recorder

    def __call__(self, obs):
        action = super().__call__(obs)
        try:
            import cv2

            frame = _decode_front(obs)
            self._recorder.add_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception:
            # A frame that will not decode is a lost video frame, never a lost
            # command: this thread's real job is answering observations.
            pass
        return action


def run_eval(args: argparse.Namespace, driver: SimulatorSO101) -> None:
    from inspect_robots import Scene, Task, eval
    from inspect_robots.approver import ClampApprover
    from inspect_robots_agent import LLMAgentPolicy
    from inspect_robots_so101 import SOArmConfig, SOArmEmbodiment
    from inspect_robots_so101.operator import OperatorIO

    config = SOArmConfig(
        control_hz=args.control_hz,
        joint_low=JOINT_LOW,
        joint_high=JOINT_HIGH,
        cameras=("front",),
        cam_width=640,
        cam_height=480,
        use_degrees=True,
    )
    embodiment = SOArmEmbodiment(
        config,
        driver_factory=lambda _config: driver,
        operator=OperatorIO(input_fn=lambda _prompt: "", output_fn=lambda _text: None),
        poll_end=lambda: False,
    )
    policy = LLMAgentPolicy(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        effort=args.effort,
        max_llm_calls=args.max_llm_calls,
        max_speed_frac=args.max_speed_frac,
        images="always",
        prior_learnings=args.prior_learnings,
    )
    task = Task(
        name="so101-simulator",
        scenes=[Scene(id="tabletop", instruction=args.instruction)],
        scorer="episode_length",
        max_steps=args.max_steps,
    )
    try:
        logs = eval(
            task,
            policy,
            embodiment,
            log_dir=args.log_dir,
            approver=ClampApprover(embodiment.info.action_space),
            store_frames=True,
        )
        print(f"Inspect Robots run status: {logs[0].status}")
    finally:
        embodiment.close()


def run_smoke(args: argparse.Namespace, driver: SimulatorSO101) -> None:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_so101 import SOArmConfig, SOArmEmbodiment
    from inspect_robots_so101.operator import OperatorIO

    config = SOArmConfig(
        control_hz=args.control_hz,
        joint_low=JOINT_LOW,
        joint_high=JOINT_HIGH,
        cameras=("front",),
        cam_width=640,
        cam_height=480,
    )
    embodiment = SOArmEmbodiment(
        config,
        driver_factory=lambda _config: driver,
        operator=OperatorIO(input_fn=lambda _prompt: "", output_fn=lambda _text: None),
        poll_end=lambda: False,
    )
    try:
        observation = embodiment.reset(Scene(id="smoke", instruction="small safe joint motion"))
        start = observation.state["joint_pos"].copy()
        target = start.copy()
        target[4] = min(target[4] + 5.0, JOINT_HIGH[4])
        target[5] = 75.0
        result = embodiment.step(Action(data=target))
        if result.observation.images["front"].shape != (480, 640, 3):
            raise RuntimeError("SO-101 front camera did not return 640x480 RGB")
        embodiment.step(Action(data=start))
        print("Inspect Robots SO-101 live smoke: ok")
    finally:
        embodiment.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--connect-timeout", type=float, default=300.0)
    ap.add_argument("--model", default="qwen3.8:27b-mlx")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    ap.add_argument("--effort", default="none")
    ap.add_argument(
        "--instruction",
        default="Pick up the blue cup and place it inside the open orange box.",
    )
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-llm-calls", type=int, default=50)
    ap.add_argument("--max-speed-frac", type=float, default=0.05)
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--log-dir", default="runs/so101-inspect")
    ap.add_argument("--prior-learnings", default=None)
    ap.add_argument("--smoke", action="store_true", help="test without calling an LLM")
    ap.add_argument("--record", metavar="DIR", default=None,
                    help="also write feed.mp4 + commands.jsonl of the whole episode")
    args = ap.parse_args()

    recorder = None
    if args.record:
        from .recorder import Recorder

        recorder = Recorder(Path(args.record))
        recorder.start({"source": "inspect_so101", "port": args.port,
                        "instruction": args.instruction, "model": args.model})
        driver = RecordingSO101(recorder)
    else:
        driver = SimulatorSO101()
    client = ArmClient(driver, args.host, args.port, args.connect_timeout)
    client_error: list[BaseException] = []

    def serve_observations() -> None:
        try:
            client.run()
        except Exception as exc:  # noqa: BLE001 - forward background transport failures.
            client_error.append(exc)
        finally:
            driver.disconnect()

    thread = threading.Thread(target=serve_observations, daemon=True)
    thread.start()
    print(
        "Start the simulator in another terminal:\n"
        "  cd ../simulator\n"
        "  ./run.sh view --robot so101 --scene robots/so101/inspect_scene.xml "
        f"--control-port {args.port} --control-hz {args.control_hz:g} --timeout 300"
    )
    try:
        if args.smoke:
            run_smoke(args, driver)
        else:
            run_eval(args, driver)
    finally:
        driver.disconnect()
        client.close()
        thread.join(timeout=5.0)
        if recorder is not None:
            summary = recorder.close()
            print(f"recorded: {summary}")
    if client_error:
        raise client_error[0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
