#!/usr/bin/env python
"""The visual-SLAM go/no-go gate: does stella_vslam track this footage, and close a loop?

Runs entirely offline against frames on disk -- no robot, no rosbridge, no console loop,
no engine protocol. That is the point: the riskiest assumption behind replacing
MapAnything with stella is that a feature tracker works at all on simulator imagery, and
that assumption is worth falsifying before any of the protocol, alignment or packaging
work that depends on it. A gate you can re-run against the same bytes is also the only
kind that can tell a binding regression from a scene that was always too textureless.

Needs `_stella`, so it runs under `.venv-vslam`, not the console venv:

    .venv-vslam/bin/python tools/stella_offline.py runs/gate0

Input is a directory written by the simulator's `tools/capture_trajectory.py`:
`frames/*.jpg`, `camera_info.json`, `poses.jsonl` (ground-truth cam->world, OpenCV axes).

## What it measures, and why each one

- **M1 init**: frames until tracking starts. Monocular initialization needs parallax; a
  robot that only rotates cannot provide it, which is why the capture drives before it
  spins and why this number is reported rather than assumed.
- **M2 tracking**: fraction of frames with a pose, and the longest unbroken lost run. A
  tracker that recovers instantly is usable; one that loses a room is not.
- **M3 loop closure**: whether old keyframes ever move *together*, and whether trajectory
  error drops when they do. This is the capability MapAnything never had and the entire
  reason for the swap -- if it never fires, the case for stella is much weaker.
- **M4 landmarks**: median landmarks per keyframe, the input to the sparse-vs-dense
  decision.
- **ATE**: RMSE against ground truth *after* a similarity fit, because monocular SLAM is
  up to scale. Fitting first is not cheating -- comparing an unscaled trajectory to metres
  would measure the missing scale factor and nothing else.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

# The camera/ORB config and the similarity fit are the engine's own, imported rather
# than copied: a threshold tuned here and not there would make this gate's PASS a
# statement about a tracker the console never runs.
from robot_console.vslam.engines import stella_config
from robot_console.vslam.engines.stella_align import umeyama_sim3

# A loop closure moves many *already-settled* keyframes at once. A single keyframe
# drifting, or the newest few being refined by local BA, is ordinary and must not count --
# hence both a displacement floor and a "how many moved" floor, and hence ignoring the
# most recent keyframes entirely.
LOOP_MOVE_M = 0.05
LOOP_MIN_KEYFRAMES = 5
LOOP_IGNORE_NEWEST = 3


def ate(estimated: np.ndarray, truth: np.ndarray):
    """(rmse, scale) of `estimated` against `truth` after a similarity fit."""
    scale, rot, trans = umeyama_sim3(estimated, truth)
    aligned = (scale * (rot @ estimated.T)).T + trans
    return float(np.sqrt(((aligned - truth) ** 2).sum(axis=1).mean())), scale


def load_capture(path: Path):
    info = json.loads((path / "camera_info.json").read_text())
    frames = sorted((path / "frames").glob("*.jpg"))
    truth, stamps = {}, {}
    with (path / "poses.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            i = int(row["i"])
            truth[i] = np.asarray(row["T_wc"], dtype=np.float64).reshape(4, 4)
            # The capture's own sim-time clock, not a synthesized one: stella keys
            # keyframe spacing off timestamps, so feeding a clock that disagrees with the
            # actual motion between frames misprices every insertion decision.
            stamps[i] = float(row["t"])
    if not frames:
        raise SystemExit(f"{path}: no frames")
    return info, frames, truth, stamps


def run(session, frames, stamps, *, verbose: bool) -> dict:
    """Feed every frame, watching for the first pose, lost runs and loop closures."""
    poses: dict[int, np.ndarray] = {}
    first_tracked = None
    lost_run = longest_lost = 0
    loops = []
    settled: dict[int, np.ndarray] = {}
    # How the run actually failed matters more than that it failed: a tracker that
    # initializes and immediately loses is a different problem from one that never
    # initializes, and they have different fixes.
    states: list[tuple[int, object]] = []
    peak_keyframes = 0
    max_settled_move = 0.0

    for i, jpg in enumerate(frames):
        image = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"could not read {jpg}")
        T = session.feed_monocular(image, stamps.get(i, i / 30.0))
        state = session.tracking_state
        if not states or states[-1][1] != state:
            states.append((i, state))
            if verbose:
                print(f"  frame {i}: state -> {state}", file=sys.stderr)

        if T is None:
            lost_run += 1
            longest_lost = max(longest_lost, lost_run)
        else:
            lost_run = 0
            # `feed_monocular` returns the world->camera pose, the opposite of what
            # `keyframes()` reports; comparing the un-inverted one against ground truth
            # scores a rotated reflection of the trajectory, which a similarity fit is
            # happy to give a small number to. (Verified: inverted, these agree with
            # the keyframe poses to 0.011 m; un-inverted, to 1.08 m.)
            poses[i] = np.linalg.inv(np.asarray(T, dtype=np.float64).reshape(4, 4))
            if first_tracked is None:
                first_tracked = i

        ids, _, kf_poses = session.keyframes()
        current = {
            int(k): np.asarray(p, dtype=np.float64).reshape(4, 4)[:3, 3]
            for k, p in zip(ids, kf_poses)
        }
        peak_keyframes = max(peak_keyframes, len(current))
        if settled:
            recent = set(sorted(current)[-LOOP_IGNORE_NEWEST:])
            moved = [
                float(np.linalg.norm(current[k] - settled[k]))
                for k in current.keys() & settled.keys()
                if k not in recent
            ]
            # Tracked unthresholded: on a trajectory with little drift a genuine loop
            # closure corrects by millimetres, so "no closure above 5 cm" and "the
            # optimizer never ran" are different claims and must not look the same.
            if moved:
                max_settled_move = max(max_settled_move, max(moved))
            big = [m for m in moved if m > LOOP_MOVE_M]
            if len(big) >= LOOP_MIN_KEYFRAMES:
                loops.append({"frame": i, "keyframes": len(big), "max_move": max(big)})
                if verbose:
                    print(f"  loop closure at frame {i}: {len(big)} keyframes moved, "
                          f"max {max(big):.3f}", file=sys.stderr)
        settled = current

        if verbose and i % 100 == 0:
            print(f"  frame {i}/{len(frames)}  keyframes={len(current)}  "
                  f"tracked={'yes' if T is not None else 'NO'}", file=sys.stderr)

    return {
        "poses": poses,
        "first_tracked": first_tracked,
        "longest_lost": longest_lost,
        "loops": loops,
        "states": states,
        "peak_keyframes": peak_keyframes,
        "max_settled_move": max_settled_move,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture", help="directory from tools/capture_trajectory.py")
    ap.add_argument("--vocab", default=None,
                    help="orb_vocab.fbow (default: .vendor/stella/share/orb_vocab.fbow)")
    ap.add_argument("--ini-fast", type=int, default=stella_config.INI_FAST, dest="ini_fast",
                    help="ORB initial FAST threshold; lower finds more corners on the "
                         "flat, low-texture walls a rendered house is full of")
    ap.add_argument("--min-fast", type=int, default=stella_config.MIN_FAST, dest="min_fast")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", default=None, help="also write the metrics here")
    args = ap.parse_args()

    import _stella

    vocab = stella_config.vocab_path(args.vocab)

    capture = Path(args.capture)
    info, frames, truth, stamps = load_capture(capture)
    print(f"{len(frames)} frames, {info['width']}x{info['height']}, "
          f"fx={info['fx']:.1f}", file=sys.stderr)

    config_path = stella_config.write_config(stella_config.render(
        fx=info["fx"], fy=info["fy"], cx=info["cx"], cy=info["cy"],
        fps=args.fps, cols=info["width"], rows=info["height"],
        ini_fast=args.ini_fast, min_fast=args.min_fast,
        name="robot_console capture",
    ))

    session = _stella.Session(str(config_path), str(vocab))
    session.startup()
    try:
        result = run(session, frames, stamps, verbose=not args.quiet)
        ids, _, kf_poses = session.keyframes()
        landmarks = [int(np.asarray(session.landmarks_of(int(k))).shape[0]) for k in ids]
    finally:
        session.shutdown()

    poses = result["poses"]
    tracked = sorted(poses.keys() & truth.keys())
    metrics = {
        "frames": len(frames),
        "keyframes": len(ids),
        "m1_frames_to_init": result["first_tracked"],
        "m2_tracked_fraction": len(poses) / len(frames),
        "m2_longest_lost_run": result["longest_lost"],
        "m3_loop_closures": len(result["loops"]),
        "m3_loops": result["loops"],
        "m4_median_landmarks": statistics.median(landmarks) if landmarks else 0,
        "peak_keyframes": result["peak_keyframes"],
        "max_settled_move_m": result["max_settled_move"],
        "state_changes": [[i, str(s)] for i, s in result["states"]],
    }
    if len(tracked) >= 3:
        est = np.array([poses[i][:3, 3] for i in tracked])
        gt = np.array([truth[i][:3, 3] for i in tracked])
        metrics["ate_rmse_m"], metrics["fitted_scale"] = ate(est, gt)
        metrics["truth_path_m"] = float(
            np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()
        )

    print("\n--- gate 0 ---", file=sys.stderr)
    print(f"  states                   "
          f"{' '.join(f'{i}:{s}' for i, s in result['states'][:12])}", file=sys.stderr)
    for key in ("frames", "keyframes", "peak_keyframes", "m1_frames_to_init",
                "m2_tracked_fraction", "m2_longest_lost_run", "m3_loop_closures",
                "m4_median_landmarks", "max_settled_move_m", "ate_rmse_m", "fitted_scale",
                "truth_path_m"):
        if key in metrics and not isinstance(metrics[key], list):
            value = metrics[key]
            print(f"  {key:24} {value:.4f}" if isinstance(value, float)
                  else f"  {key:24} {value}", file=sys.stderr)

    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2) + "\n")

    # The gate itself. Deliberately loose: this asks "is stella viable here at all", not
    # "is it tuned". A tighter bar belongs in the benchmark that decides sparse vs dense.
    verdict = []
    if result["first_tracked"] is None:
        verdict.append("never initialized")
    elif result["first_tracked"] > 5 * args.fps:
        verdict.append(f"slow init ({result['first_tracked']} frames)")
    if metrics["m2_tracked_fraction"] < 0.8:
        verdict.append(f"tracked only {metrics['m2_tracked_fraction']:.0%} of frames")
    if not result["loops"]:
        verdict.append("no loop closure")
    print(f"\nRESULT: {'PASS' if not verdict else 'FAIL -- ' + '; '.join(verdict)}",
          file=sys.stderr)
    return 0 if not verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
