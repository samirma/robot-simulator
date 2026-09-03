"""MolmoAct2 as a native in-process Inspect Robots policy for the SO-101.

MolmoAct2 emits **absolute joint pose in degrees**, in the LeRobot SO-100/101
frame, 30 steps per inference. Our embodiment wants absolute joint positions in
radians in the MuJoCo frame, in the same joint order. The whole adaptation is a
units-and-frame transform -- no IK retarget -- which is why this checkpoint is
the closest fit of the candidates.

    state_model = joint_signs * state_deg + joint_offsets
    action_deg  = joint_signs * (action_model - joint_offsets)

``joint_signs``/``joint_offsets`` are published in
``lerobot/MolmoAct2-SO100_101-LeRobot``.

Three things here are load-bearing and each fails *silently* if changed:

1. **float16, not bfloat16.** bf16 has an 8-bit mantissa, so joint values in the
   170-190 degree range land on a 1.0-degree grid and collapse: measured over one
   30-step chunk, ``shoulder_lift`` spread 0.00 deg / 1 distinct value in bf16
   against 45.88 deg / 30 values in fp16. Shape, dtype and finiteness all still
   pass, so a check that only asserts "the output is finite" ships a frozen arm.
2. **The gripper channel is a jaw angle in DEGREES**, not 0..1 and not radians.
   See ``GRIPPER_ACTION_OPEN_DEG``.
3. **The state is binned into 256 buckets and out-of-range values clip with no
   error.** ``assert_state_in_distribution`` exists so that clipping is loud.

``predict_action`` returns actions that are **already un-normalised** into robot
scale. Do not apply ``norm_stats`` a second time.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    CameraSpec,
    Observation,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots.spaces import ObservationSpace
from robot_console.arm.kinematics import ARM_JOINTS, JOINT_LIMITS, JOINT_ORDER

# The one source of truth for the task text. This module used to redefine it,
# colourless ("pick up the apple and place it on the plate"), which was dead
# only because ``reset(scene)`` overrides it -- a latent trap if that ever
# changed. ``task`` does not import this module, so there is no cycle.
from robot_console.arm.task import INSTRUCTION

REPO = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"

#: LeRobot SO-100/101 calibration, from ``lerobot/MolmoAct2-SO100_101-LeRobot``,
#: with ``wrist_roll`` (index 4) corrected against measurement -- see below.
#:
#: WRIST_ROLL, corrected 2026-08-30. The published table gave index 4 sign +1 and
#: offset 0. Measured against a *physically correct* execution of this exact task
#: -- 140,392 ``/joint_states`` samples from 7 arbiter-PASSING scripted runs,
#: pushed through ``to_model_state`` and compared to the checkpoint's own
#: ``norm_stats.json`` q01..q99 state band for tag ``so100_so101_molmoact2`` --
#: that mapping put ``wrist_roll`` OUTSIDE the trained band for 98.6% of the run,
#: median +90.05 against a training q50 of -11.04, about 47 deg past q99 (42.94).
#: A correct execution must look like training data; that one did not.
#:
#: ``model_wrist_roll = 90 - ours_in_degrees``: the axis is reversed and the zero
#: is shifted a quarter turn. Our ``wrist_roll = +pi/2`` is precisely the "level
#: jaw" pose (``kinematics.level_jaw_roll``), which is the natural zero for the
#: LeRobot convention -- an ordinary frame-convention difference, not a fudge.
#: Measured effect: wrist_roll in-band 1.4% -> 98.4%, median -0.05 against q50
#: -11.04; HOME becomes in-band on that channel; mean in-band across the five arm
#: channels 64.3% -> 83.7%.
#:
#: Three independent confirmations that the old mapping was wrong:
#: 1. The shipped guard fired live: ``wrist_roll: 86.75 outside [-63.45, 42.94]``.
#: 2. The recorded failure signature matches the un-normalisation rails: a prior
#:    rosbridge run froze ``shoulder_pan`` at -0.7352 rad = -42.13 deg = action
#:    q01 to the decimal and ``wrist_roll`` at 0.7598 rad = +43.53 deg = action
#:    q99 to the decimal. Normalised output pinned at -1/+1 is what a model does
#:    when its conditioning state lies far outside the trained band.
#: 3. A baseline trace showed live joint-limit saturation: ``wrist_flex`` sat at
#:    its +1.600 rad limit for 15.9% of 13,082 samples.
#:
#: THE DISCLOSED COST: with this calibration the model's wrist_roll action q01
#: (-65.576 deg) maps to +155.6 deg, past our +131.8 deg (2.3 rad) MJCF limit, so
#: some fraction of commanded wrist_roll now saturates: it is the model's LOW end
#: that clips, and because the sign is flipped it lands on OUR high limit, so it
#: is tallied as ``clipped_high`` on ``wrist_roll_joint``. That is what
#: ``CLIP_COUNTS`` exists to measure. Do NOT "fix" it by widening the joint
#: limit: the limit is the MJCF's and it belongs to the simulator.
SIGNS = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
OFFSETS = np.array([0.0, 90.0, 90.0, 0.0, 90.0, 0.0])

#: The checkpoint's gripper channel is a **jaw-opening angle in degrees**, not
#: 0..1 and not radians. From ``norm_stats.json`` (tag ``so100_so101_molmoact2``):
#:
#:     action ch5   q01 -0.302   q50  4.867   q90 32.123   q99 44.746   max 119.408
#:     state  ch5   q01  0.940   q50  9.240   q99 44.140
#:
#: Our jaw is the MJCF ``ctrlrange`` 0..1, so both directions are a linear
#: rescale between "our 0..1" and "the model's 0..q99 degrees".
#:
#: Getting this wrong is silent and it was: clipping the *raw* action value to
#: [0, 1] saturated everything >= 1 to fully open, and ``q50 = 4.867`` means most
#: of the distribution did exactly that. The model was closing the jaw and the
#: adapter was reopening it, every step, all episode.
#: ``tests/test_molmoact.py`` pins both constants against ``norm_stats.json``.
GRIPPER_ACTION_OPEN_DEG = 44.746
GRIPPER_STATE_OPEN_DEG = 44.140

#: A floor under the jaw *closing* command. **OFF (0.0) since 2026-08-30**: the
#: actuator defect it worked around has been fixed, and at 0.5 it now forbids the
#: only apertures that work.
#:
#: Our jaw is an MJCF ``ctrlrange`` 0..1 **position** actuator. The floor existed
#: because closing below it asked the jaw to close to a gap narrower than the
#: ball, and the solver resolved that by ejecting the apple -- MolmoAct2 drove the
#: jaw to 0.000 at contact and the apple left at 0.862 m/s. That ejection was the
#: ``gripper_joint`` actuator's +-3.35 N.m ``forcerange`` driving 18.6 mm through
#: a 20 g apple. **The simulator role has since cut that forcerange to +-0.30
#: N.m** (H3), verified by actuator NAME in the MJCF, in the rebuilt install
#: space, and in the compiled model, with the five arm actuators and the
#: ``sts3215`` class default untouched. The same full close now peaks at 2.9 N
#: with 2.8 mm of penetration and HOLDS. Removing the cause is what licenses
#: removing the workaround; this is the completion of H3, not a new hypothesis.
#:
#: THE MEASURED HOLDING WINDOW, solver-validated by the simulator role (at every
#: aperture a sphere 2 mm smaller gives ``ncon=0`` and 2 mm larger contacts both
#: fingers, 17/17):
#:
#:     command g   forcerange +-3.35        forcerange +-0.30
#:     0.00        extruded 0.27 m @ 20.6 N HOLDS, lifts the apple +58.4 mm
#:     0.00-0.35   none hold                ALL HOLD
#:     0.40+       none hold                none hold
#:
#: The entire working window is **g in [0.00, 0.35]**, which lies ENTIRELY BELOW
#: the old 0.50 floor. The floor had gone from being the workaround to being the
#: thing blocking the fix. It also destroyed proprioception across that window:
#: ``to_model_state`` undoes the floor, so at 0.5 every command in [0.00, 0.50]
#: reported the same state, 0.0 deg -- the model could not tell a full close from
#: 0.35. Off, that axis is injective again (g=0.25 -> 11.04 deg, 0.35 -> 15.45).
#:
#: TWO NUMBERS THAT ARE WRONG AND MUST NOT BE CITED AGAIN:
#: * The "96 N on a 20 g apple" figure is a FIXTURE ARTEFACT. That measurement
#:   seated the apple 28.4 mm inside the fixed finger's hull, ~60 mm behind the
#:   fingertips, and returned 27.376 N *identically at +-3.35 and +-0.30*. A
#:   number that does not move under an 11x change in actuator authority is not
#:   measuring the actuator. At the true grasp site: 20.640 N (+-3.35) -> 2.919 N
#:   (+-0.30) at g=0, against the 49 mN actually needed to hold the apple.
#: * The aperture curve is CONCAVE, not linear: measured
#:   ``gap(g) ~= -23.24 g^2 + 92.54 g - 0.84`` mm, R^2 = 0.99947. gap(0.25) =
#:   21.55, gap(0.50) = 39.58, gap(0.75) = 55.38 mm, against a 40.0 mm apple. So
#:   the old floor left 0.42 mm of interference -- a knife edge, which is why a
#:   scripted policy could sit there and a learned one could not. Linear
#:   interpolation of the endpoints is right at 0.5 only by coincidence and is
#:   wrong by 11-14 mm either side.
#:
#: At 0.0 both mapping directions are exactly the plain rescale: ``np.maximum(x,
#: 0.0)`` is a no-op on a value already clipped to [0, 1], and ``to_model_state``
#: reduces to ``jaw * GRIPPER_STATE_OPEN_DEG``. The constant is kept, rather than
#: deleted, because it IS the rollback path -- set it back to 0.5 to restore the
#: old behaviour exactly. ``tests/test_molmoact.py`` pins both directions.
GRIPPER_MIN_COMMAND = 0.0

#: Views the checkpoint expects, in order. ``camera_keys`` is ``[]`` in the
#: checkpoint metadata, so views are **positional**: slot 0 and slot 1, not
#: names. They must be two *different* static third-person views.
#:
#: ``overhead`` and ``side`` are the only two cameras the simulator publishes
#: (``/overhead/color/compressed``, ``/side/color/compressed``), so this is the
#: only pair that can be subscribed at all -- and it is the pair the three
#: passing runs (``rim_molmo_1``/``6``/``7``) used.
#:
#: History, for anyone tempted to reintroduce a training-framing pair: the
#: ``trainlow``/``trainhigh`` cameras framed the apple closer to the
#: checkpoint's own reference frames but measured strictly worse (0/5, apple
#: travel 0.000 m), and were deleted from the simulator on 2026-08-31.
DEFAULT_VIEWS: tuple[str, ...] = ("overhead", "side")

LOW = np.asarray([JOINT_LIMITS[name][0] for name in JOINT_ORDER], dtype=np.float64)
HIGH = np.asarray([JOINT_LIMITS[name][1] for name in JOINT_ORDER], dtype=np.float64)

#: How close a raw model action must come to its own un-normalisation rail before
#: it is counted as railed. ``predict_action`` un-normalises with
#: ``q01 + 0.5 * (n + 1) * (q99 - q01)``, so a normalised output pinned at -1 or
#: +1 lands EXACTLY on q01 / q99; this tolerance only absorbs the fp16 -> float64
#: round trip. Kept tight on purpose: the point is to detect the pinned case, not
#: to flag ordinary values that happen to be near the edge of the distribution.
RAIL_TOLERANCE_DEG = 1e-3


@lru_cache(maxsize=1)
def norm_stats(repo: str = REPO, tag: str = NORM_TAG) -> dict[str, Any]:
    """Load the checkpoint's own ``norm_stats.json`` from the local HF snapshot."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, allow_patterns=["norm_stats.json"]))
    return json.loads((root / "norm_stats.json").read_text())["metadata_by_tag"][tag]


#: Diagnostic hook for the vision-ablation probes. OFF unless the environment asks.
#:
#: ``MOLMOACT_VIEWMODE`` accepts:
#:   ``normal`` (default) -- untouched, and the ONLY behaviour when the var is unset
#:   ``blind``            -- every view replaced by a black frame of identical size
#:   ``swap``             -- the two view slots exchanged for the whole episode
#:
#: Exists to resolve a direct contradiction in the evidence: forward kinematics says
#: this checkpoint puts the jaw within 8-21 mm of the apple in every run, while a
#: pointing probe says it cannot locate the apple at all (155 px error). Either the
#: reach is visually guided and the pointing probe is unrepresentative, or the reach
#: is a motor prior that happens to land on a fixed spawn. Blacking the frames
#: distinguishes those.
#:
#: CONFOUND, which no result here can remove: the apple is at a FIXED spawn in every
#: episode, so a model that perceives perfectly and a model that replays a motor
#: prior produce near-identical trajectories across repeats. This probe can show that
#: vision is NOT being used; it cannot prove that it IS.
VIEWMODE_ENV = "MOLMOACT_VIEWMODE"
_VIEWMODE_WARNED = False


def _apply_viewmode(images: list[Any], views: tuple[str, ...]) -> list[Any]:
    """Apply the diagnostic view ablation, if one is requested. Default: identity."""
    import os

    from PIL import Image

    mode = os.environ.get(VIEWMODE_ENV, "normal").strip().lower()
    if mode in ("", "normal"):
        return images

    global _VIEWMODE_WARNED
    if not _VIEWMODE_WARNED:
        _VIEWMODE_WARNED = True
        print(
            f"*** {VIEWMODE_ENV}={mode!r} -- CAMERA INPUT IS BEING ALTERED. "
            f"This is a diagnostic ablation, NOT a normal run. ***",
            file=sys.stderr,
            flush=True,
        )

    if mode == "blind":
        return [Image.new("RGB", im.size, (0, 0, 0)) for im in images]
    if mode == "swap":
        if len(images) != 2:
            raise ValueError(
                f"{VIEWMODE_ENV}=swap needs exactly two views, got {len(images)} ({list(views)})"
            )
        return [images[1], images[0]]
    raise ValueError(f"{VIEWMODE_ENV}={mode!r} is not recognised; use normal, blind or swap")


@lru_cache(maxsize=1)
def _action_rails() -> tuple[np.ndarray, np.ndarray] | None:
    """The checkpoint's own action q01/q99, or ``None`` if the snapshot is absent.

    Failure is cached, so an offline run does not re-attempt a download inside a
    10 Hz control loop. Rail counting is diagnostic; it must never be able to
    break a run or slow one down.
    """
    try:
        block = norm_stats()["action_stats"]
        return (
            np.asarray(block["q01"], dtype=np.float64),
            np.asarray(block["q99"], dtype=np.float64),
        )
    except Exception:  # noqa: BLE001 -- deliberate: see the docstring
        # Offline, no snapshot, or a changed schema. Rail counting is a
        # diagnostic that runs inside the control loop; nothing it can hit is
        # worth failing an episode over, so every failure degrades to "not
        # countable" and ``rails_available`` reports that fact.
        return None


class ClipCounter:
    """Per-channel tally of saturation, so a wrong frame mapping is LOUD.

    ``to_joint_actions`` ends in ``np.clip(radians, LOW, HIGH)``. That clip is
    correct -- commanding past an MJCF limit is not allowed -- but it is silent,
    so a frame-convention error shows up as a subtly lazy joint rather than as an
    error. Two things are counted, and they mean different things:

    * **clipped_low / clipped_high** -- our command hit an MJCF joint limit. This
      is *our* mapping meeting *our* robot. The wrist_roll recalibration
      (see ``SIGNS``) knowingly pushes the model's action q01 to +155.6 deg
      against a +131.8 deg limit, so a nonzero ``clipped_low`` on wrist_roll is
      an EXPECTED, disclosed cost -- its size is the thing to watch.
    * **rail_q01 / rail_q99** -- the *raw model action* arrived already at or past
      its own un-normalisation rail, i.e. the network's normalised output was
      pinned at (or beyond) -1 or +1. That is a statement about the model, not our
      limits, and it is the signature of conditioning the policy on a state far
      outside its trained band. Requires the HF snapshot; ``rails_available``
      says whether it was countable.

    This object accumulates across a whole episode. It does not change any
    returned value, so ``to_joint_actions`` stays a pure function of its input
    as far as every caller and every existing test is concerned.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Zero every tally. Eval scripts call this before an episode."""
        size = len(JOINT_ORDER)
        self.chunks = 0
        self.steps = 0
        self.clipped_low = np.zeros(size, dtype=np.int64)
        self.clipped_high = np.zeros(size, dtype=np.int64)
        self.rail_q01 = np.zeros(size, dtype=np.int64)
        self.rail_q99 = np.zeros(size, dtype=np.int64)
        #: Worst distance past a limit, in the channel's own command units
        #: (radians for the arm, 0..1 for the jaw). Zero means never clipped.
        self.worst_below_low = np.zeros(size, dtype=np.float64)
        self.worst_above_high = np.zeros(size, dtype=np.float64)
        self.rails_available = False

    def record(self, model_actions: np.ndarray, commanded: np.ndarray, clipped: np.ndarray) -> None:
        """Tally one chunk. ``commanded`` is pre-clip, ``clipped`` is post-clip."""
        raw = np.asarray(model_actions, dtype=np.float64).reshape(-1, len(JOINT_ORDER))
        self.chunks += 1
        self.steps += int(raw.shape[0])

        below = commanded < LOW
        above = commanded > HIGH
        self.clipped_low += below.sum(axis=0).astype(np.int64)
        self.clipped_high += above.sum(axis=0).astype(np.int64)
        self.worst_below_low = np.maximum(
            self.worst_below_low, np.where(below, LOW - commanded, 0.0).max(axis=0)
        )
        self.worst_above_high = np.maximum(
            self.worst_above_high, np.where(above, commanded - HIGH, 0.0).max(axis=0)
        )

        rails = _action_rails()
        if rails is not None:
            self.rails_available = True
            q01, q99 = rails
            self.rail_q01 += (raw <= q01 + RAIL_TOLERANCE_DEG).sum(axis=0).astype(np.int64)
            self.rail_q99 += (raw >= q99 - RAIL_TOLERANCE_DEG).sum(axis=0).astype(np.int64)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot, keyed by joint name."""
        names = list(JOINT_ORDER)

        def by_name(values: list[Any]) -> dict[str, Any]:
            return {
                name: float(v) if isinstance(v, float) else int(v)
                for name, v in zip(names, values, strict=True)
            }

        steps = max(self.steps, 1)
        return {
            "chunks": self.chunks,
            "steps": self.steps,
            "joint_order": names,
            "joint_limits": {name: [float(LOW[i]), float(HIGH[i])] for i, name in enumerate(names)},
            "clipped_low": by_name(self.clipped_low.tolist()),
            "clipped_high": by_name(self.clipped_high.tolist()),
            "clipped_low_fraction": by_name((self.clipped_low / steps).tolist()),
            "clipped_high_fraction": by_name((self.clipped_high / steps).tolist()),
            "worst_below_low": by_name(self.worst_below_low.tolist()),
            "worst_above_high": by_name(self.worst_above_high.tolist()),
            "action_rail_q01": by_name(self.rail_q01.tolist()),
            "action_rail_q99": by_name(self.rail_q99.tolist()),
            "action_rails_available": self.rails_available,
            "rail_tolerance_deg": RAIL_TOLERANCE_DEG,
        }

    def summary(self) -> str:
        """A compact table for the end of a run."""
        if self.steps == 0:
            return "clip counter: no actions recorded"
        header = (
            f"  {'joint':<22} {'clip@LOW':>9} {'clip@HIGH':>10} "
            f"{'worst_past':>11} {'raw@q01':>8} {'raw@q99':>8}"
        )
        lines = [
            f"commanded-action saturation over {self.steps} steps ({self.chunks} chunks):",
            header,
        ]
        for i, name in enumerate(JOINT_ORDER):
            low_n = int(self.clipped_low[i])
            high_n = int(self.clipped_high[i])
            worst = max(float(self.worst_below_low[i]), float(self.worst_above_high[i]))
            lines.append(
                f"  {name:<22} {low_n:>4} {low_n / self.steps:>5.1%} "
                f"{high_n:>4} {high_n / self.steps:>5.1%} "
                f"{worst:>11.4f} "
                f"{int(self.rail_q01[i]):>8} {int(self.rail_q99[i]):>8}"
            )
        if not self.rails_available:
            lines.append("  (raw@q01/raw@q99 unavailable: norm_stats.json not loadable)")
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        """Write ``as_dict()`` to ``path``, creating parent directories."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), indent=2))
        return out


#: Episode-scoped, module-level accumulator. Eval scripts ``reset()`` it before a
#: run and read it after; nothing here feeds back into a returned action.
CLIP_COUNTS = ClipCounter()


def to_model_state(measured: np.ndarray) -> np.ndarray:
    """Map our six measured joints (radians, jaw 0..1) into the model's frame."""
    measured = np.asarray(measured, dtype=np.float64).reshape(-1)[:6]
    state = SIGNS * np.degrees(measured) + OFFSETS
    # The jaw is not an angle we can just convert: our 0..1 has to be rescaled
    # onto the model's 0..44.14 degrees, or it lands outside q01..q99 and clips.
    #
    # With GRIPPER_MIN_COMMAND = 0.0 (the live value) this whole block reduces to
    # that plain rescale, ``jaw * GRIPPER_STATE_OPEN_DEG``, and the floor-undo
    # below is inert. It is kept because it is half of the rollback path, and
    # because it records what a nonzero floor costs the OBSERVATION side, which is
    # easy to forget when arguing about the command side:
    #
    # A floor leaks into proprioception. The commanded jaw cannot go below
    # GRIPPER_MIN_COMMAND, so the measured jaw cannot either, so at 0.5 state ch5
    # could never report below 0.50 * 44.140 = 22.05 deg -- about the 73rd
    # percentile of the checkpoint's own state distribution (q50 = 9.24 deg).
    # Undoing it restores the axis, but only as a rescale: at 0.5 every command in
    # [0.00, 0.50] -- i.e. the ENTIRE measured holding window g in [0.00, 0.35] --
    # reported the same 0.0 deg, so the model could not tell a full close from a
    # 0.35 grasp. Measured: samples below the training median fell from 6.7-9.6%
    # (no floor) to 0.1% (floor), and during the one successful grasp the reported
    # state sat flat at 22.07 deg for all 139 frames the apple was aloft -- while
    # physically holding the apple the model was told it had a moderately OPEN
    # hand, with no proprioceptive signature of the grasp at all.
    #
    # This affects only what the policy observes; the arbiter judges from the
    # apple's pose and never reads gripper state, so it cannot affect scoring.
    span = 1.0 - GRIPPER_MIN_COMMAND
    if span > 0.0:
        jaw = (float(np.clip(measured[5], GRIPPER_MIN_COMMAND, 1.0)) - GRIPPER_MIN_COMMAND) / span
    else:  # GRIPPER_MIN_COMMAND == 1.0 would collapse the axis; degrade gracefully.
        jaw = float(np.clip(measured[5], 0.0, 1.0))
    state[5] = jaw * GRIPPER_STATE_OPEN_DEG
    return state


def to_joint_actions(model_actions: np.ndarray) -> np.ndarray:
    """Map a chunk of model-frame actions (degrees) into our radians, jaw 0..1."""
    arr = np.asarray(model_actions, dtype=np.float64).reshape(-1, 6)
    radians = np.radians(SIGNS * (arr - OFFSETS))
    # Channel 5 is a degree scale; rescale rather than clip. Clipping the raw
    # value to [0, 1] is defect 1 and it forces the jaw open for the episode.
    radians[:, 5] = np.clip(arr[:, 5] / GRIPPER_ACTION_OPEN_DEG, 0.0, 1.0)
    # The rollback hook for the crush-aperture floor, INERT at the live value of
    # 0.0: the operand is already clipped to [0, 1], so np.maximum(x, 0.0) is the
    # identity. Retained because setting GRIPPER_MIN_COMMAND back to 0.5 is the
    # documented way to restore the old behaviour exactly. Monotone and one-sided
    # whatever the value: opening commands are never touched, so the policy always
    # keeps full authority to open and release. See GRIPPER_MIN_COMMAND.
    radians[:, 5] = np.maximum(radians[:, 5], GRIPPER_MIN_COMMAND)
    # Clipping to the MJCF limits is correct but SILENT, and silence is exactly
    # how a wrong frame mapping hides. Tally it. The counter is module-level and
    # write-only from here: the returned values are unchanged, so this function
    # is still a pure function of its argument for every caller.
    clipped = np.clip(radians, LOW, HIGH)
    CLIP_COUNTS.record(arr, radians, clipped)
    return clipped


def assert_state_in_distribution(
    measured: np.ndarray, *, stats: dict[str, Any] | None = None, tolerance: float = 0.0
) -> dict[str, tuple[float, float, float]]:
    """Raise if the mapped state falls outside the checkpoint's q01..q99 band.

    The processor bins state into 256 buckets and clips out-of-range values with
    no error, so a bad mapping degrades quality invisibly. Returns the per-joint
    ``(mapped, q01, q99)`` so a caller can report the margins.
    """
    block = (stats or norm_stats())["state_stats"]
    q01 = np.asarray(block["q01"], dtype=np.float64)
    q99 = np.asarray(block["q99"], dtype=np.float64)
    mapped = to_model_state(measured)
    report = {
        name: (float(mapped[i]), float(q01[i]), float(q99[i]))
        for i, name in enumerate(block["names"])
    }
    outside = [
        f"{name}: {value:.2f} outside [{low:.2f}, {high:.2f}]"
        for name, (value, low, high) in report.items()
        if value < low - tolerance or value > high + tolerance
    ]
    if outside:
        raise ValueError(
            "mapped state is outside the checkpoint's q01..q99 band and will be "
            "silently clipped by the 256-bucket state encoder: " + "; ".join(outside)
        )
    return report


class MolmoAct2Policy:
    """MolmoAct2-SO100_101 driving the SO-101 through absolute joint poses."""

    def __init__(
        self,
        *,
        device: str = "mps",
        dtype: str = "float16",
        views: tuple[str, ...] = DEFAULT_VIEWS,
        chunk_steps: int = 30,
        num_steps: int = 10,
        control_hz: float = 10.0,
        check_state_range: str = "warn",
        allow_single_view: bool = False,
        replan_interval: int | None = None,
        ensemble: int = 1,
        seed: int | None = None,
        name: str = "molmoact2",
    ) -> None:
        if dtype != "float16":
            raise ValueError(
                f"dtype must be 'float16', got {dtype!r}. bfloat16 collapses "
                "shoulder_lift and elbow_flex to a single value across the chunk "
                "while still passing every shape and finiteness check."
            )
        if len(set(views)) != len(views):
            raise ValueError(
                f"duplicate view in {views!r}. MolmoAct2 takes views positionally and "
                "was trained on two *different* third-person views; feeding the same "
                "frame into both slots is off-distribution input, so it is refused "
                "rather than padded."
            )
        if len(views) < 2 and not allow_single_view:
            raise ValueError(
                f"MolmoAct2 is trained on two different third-person views and takes "
                f"them positionally; got {views!r}. Pass allow_single_view=True to run "
                "degraded on one -- which is not the same as duplicating a frame, and "
                "will be reported as a deviation."
            )
        self.device = device
        self.views = tuple(views)
        self.num_steps = int(num_steps)
        if check_state_range not in ("warn", "raise", "off"):
            raise ValueError(
                f"check_state_range must be 'warn', 'raise' or 'off', got {check_state_range!r}"
            )
        self.check_state_range = check_state_range
        if int(ensemble) < 1:
            raise ValueError(f"ensemble must be >= 1, got {ensemble!r}")
        # The checkpoint's flow-matching sampler starts from ``torch.randn``
        # and ``predict_action`` accepts a ``generator`` that this repo never
        # passed, so every chunk was an unseeded single draw. ``ensemble`` draws
        # K chunks per decision and takes the per-step MEDIAN (see
        # ``aggregate_samples`` for why not the mean); ``seed`` makes the draws
        # reproducible and labels them in the log.
        self.ensemble = int(ensemble)
        self.seed = None if seed is None else int(seed)
        self._model = None
        self._processor = None
        self._instruction = INSTRUCTION
        self._checked_state = False
        self._inferences = 0
        self.inference_latencies: list[float] = []

        self.info = PolicyInfo(
            name=name,
            action_space=Box(
                shape=(6,),
                low=LOW,
                high=HIGH,
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                    dim_labels=(*ARM_JOINTS, "gripper"),
                ),
            ),
            # Declaring the cameras makes a missing view a loud compatibility
            # failure instead of a silently black frame. Sizes are what the
            # simulator publishes; the processor resizes to 378x378 itself.
            observation_space=ObservationSpace(
                # Sizes are placeholders until ``bind`` adopts the embodiment's
                # own specs; only the *names* take part in the compatibility
                # check, and the processor resizes to 378x378 regardless.
                cameras=tuple(CameraSpec(name=view, height=480, width=640) for view in self.views),
                state_keys=frozenset({"joint_pos"}),
            ),
            control_hz=float(control_hz),
        )
        # One inference produces 30 absolute poses; by default play them all
        # before re-inferring, because inference costs ~3 s against a 10 Hz
        # loop. ``replan_interval`` < chunk_steps closes the loop more often at
        # proportionally more inference time (the controller truncates the
        # chunk to that many actions).
        replan = int(replan_interval) if replan_interval else int(chunk_steps)
        if replan < 1 or replan > int(chunk_steps):
            raise ValueError(
                f"replan_interval must be in [1, {chunk_steps}], got {replan_interval!r}"
            )
        self.config = PolicyConfig(action_horizon=int(chunk_steps), replan_interval=replan)

    # -- lifecycle ---------------------------------------------------------
    def bind(self, embodiment_info: Any) -> "MolmoAct2Policy":
        """Adopt the arm's real action space so the compatibility check passes."""
        import dataclasses

        space = getattr(embodiment_info, "action_space", None)
        if space is not None and getattr(space, "shape", None) == (6,):
            self.info = dataclasses.replace(self.info, action_space=space)

        # Adopt the embodiment's real camera specs so the declared resolution is
        # the as-built one rather than a constant that drifts when the simulator
        # changes a camera.
        provided = getattr(getattr(embodiment_info, "observation_space", None), "cameras", ())
        by_name = {camera.name: camera for camera in provided}
        if all(view in by_name for view in self.views):
            self.info = dataclasses.replace(
                self.info,
                observation_space=dataclasses.replace(
                    self.info.observation_space,
                    cameras=tuple(by_name[view] for view in self.views),
                ),
            )
        return self

    def reset(self, scene: Scene) -> None:
        """Take this scene's instruction; the model is otherwise stateless."""
        instruction = getattr(scene, "instruction", None)
        if instruction:
            self._instruction = instruction
        self._checked_state = False
        self._inferences = 0
        self.inference_latencies = []

    # -- model -------------------------------------------------------------
    def _load(self) -> None:
        """Load lazily so ``inspect-robots list`` stays fast."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            REPO, trust_remote_code=True, dtype=torch.float16
        )
        self._model = model.to(self.device).eval()
        self._torch = torch

    def _view_images(self, observation: Observation) -> list[Any]:
        """Return the configured views as PIL images, in declaration order.

        Missing or duplicated views raise: the checkpoint consumes views
        positionally, so silently substituting one for another changes what the
        model is looking at without changing anything observable downstream.
        """
        from PIL import Image

        images = dict(getattr(observation, "images", None) or {})
        missing = [view for view in self.views if view not in images]
        if missing:
            raise KeyError(
                f"observation is missing camera view(s) {missing}; it has "
                f"{sorted(images)}. MolmoAct2 needs {list(self.views)} in that order."
            )
        out = []
        for view in self.views:
            array = np.asarray(images[view])
            if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
                array = np.transpose(array, (1, 2, 0))
            if array.dtype != np.uint8:
                array = (
                    (np.clip(array, 0, 1) * 255).astype(np.uint8)
                    if array.max() <= 1.0
                    else array.astype(np.uint8)
                )
            out.append(Image.fromarray(array[..., :3]))
        return _apply_viewmode(out, self.views)

    # -- inference ---------------------------------------------------------
    def act(self, observation: Observation) -> ActionChunk:
        """Run one inference and return its 30 absolute joint poses."""
        self._load()
        measured = np.asarray(observation.state["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        if self.check_state_range != "off" and not self._checked_state:
            self._checked_state = True
            try:
                assert_state_in_distribution(measured)
            except ValueError as exc:
                if self.check_state_range == "raise":
                    raise
                # Loud, but not fatal: the arm's rest pose is legitimately
                # outside the demonstrated band on two channels, and refusing to
                # start would be worse than starting slightly off-distribution.
                print(f"warning: {exc}", file=sys.stderr, flush=True)

        images = self._view_images(observation)
        state = to_model_state(measured).tolist()
        samples: list[np.ndarray] = []
        seeds: list[int] = []
        started = time.perf_counter()
        for k in range(self.ensemble):
            generator = None
            if self.seed is not None:
                # Distinct per (run, inference, sample) so K draws differ and a
                # re-run with the same seed replays the same noise.
                sample_seed = self.seed * 1000 + self._inferences * self.ensemble + k
                generator = self._generator(sample_seed)
                seeds.append(sample_seed)
            with self._torch.no_grad():
                out = self._model.predict_action(
                    processor=self._processor,
                    images=images,
                    task=self._instruction,
                    state=state,
                    norm_tag=NORM_TAG,
                    inference_action_mode="continuous",
                    enable_depth_reasoning=False,
                    num_steps=self.num_steps,
                    normalize_language=True,
                    enable_cuda_graph=False,
                    generator=generator,
                )
            raw = out.actions if hasattr(out, "actions") else out
            if hasattr(raw, "detach"):
                raw = raw.detach().float().cpu().numpy()
            raw = np.asarray(raw, dtype=np.float64)
            if raw.ndim == 3:
                raw = raw[0]
            samples.append(raw)
        latency = time.perf_counter() - started
        self._inferences += 1
        self.inference_latencies.append(latency)

        joints = to_joint_actions(aggregate_samples(samples))
        return ActionChunk(
            actions=[Action(data=row) for row in joints],
            control_hz=self.info.control_hz,
            inference_latency_s=latency,
            meta={"ensemble": self.ensemble, "seeds": seeds, "inference": self._inferences},
        )

    def _generator(self, seed: int) -> Any:
        """A seeded generator on the model's device, or global seeding if unsupported.

        The checkpoint draws its initial noise with ``torch.randn(...,
        generator=generator)`` on the model device; a CPU generator with an
        MPS tensor raises, so the generator must live on the same device.
        """
        torch = self._torch
        try:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            return generator
        except (RuntimeError, TypeError):
            torch.manual_seed(int(seed))
            return None


def aggregate_samples(samples: list[np.ndarray]) -> np.ndarray:
    """Combine K sampled chunks of shape ``(30, 6)`` into one, per step.

    The per-step MEDIAN, not the mean, for the jaw's sake: averaging one
    sample that closes (0 deg) with one that stays open (44 deg) gives ~22 deg,
    which after scaling lands at g ~ 0.5 -- above the measured holding window
    (g <= 0.35), i.e. exactly the aperture that lets the apple go. The median
    of an odd K picks a jaw value one sample actually proposed. For K = 1 it
    is the identity.
    """
    if len(samples) == 1:
        return np.asarray(samples[0], dtype=np.float64)
    stacked = np.stack([np.asarray(s, dtype=np.float64) for s in samples])
    if stacked.ndim != 3:
        raise ValueError(f"expected K x steps x 6 samples, got shape {stacked.shape}")
    return np.median(stacked, axis=0)


def molmoact2(**kwargs: Any) -> MolmoAct2Policy:
    """Registry entry point; CLI ``-P key=value`` strings are coerced here."""
    integers = {"chunk_steps", "num_steps", "replan_interval", "ensemble", "seed"}
    floats = {"control_hz"}
    booleans: set[str] = set()
    coerced: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in integers:
            coerced[key] = int(value)
        elif key in floats:
            coerced[key] = float(value)
        elif key in booleans:
            coerced[key] = (
                value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
            )
        elif key == "views" and isinstance(value, str):
            coerced[key] = tuple(part.strip() for part in value.split(",") if part.strip())
        else:
            coerced[key] = value
    return MolmoAct2Policy(**coerced)
