"""A deterministic waypoint policy that picks the red apple and places it.

This is a real Inspect Robots policy: it implements the
[`Policy`][inspect_robots.policy.Policy] contract, reads the embodiment's
``joint_pos`` proprioception out of every
[`Observation`][inspect_robots.types.Observation], and emits six **absolute**
joint-position commands per step in ``joint_pos`` control mode.

It is scripted rather than learned on purpose. No public VLA checkpoint
transfers to this domain (a MuJoCo SO-101 with a 60 mm sphere, no wrist camera,
and a base-frame joint-position interface), so a scripted policy is what makes
the rest of the stack — embodiment, scorer, success polling, logging —
verifiable end to end. See ``docs/smolvla.md`` for the optional local VLA path
and what it does *not* establish.

The scripting is closed-loop on proprioception: a waypoint is held until the
measured joints are within tolerance and have settled, so the policy reacts to
an arm that lags or stalls instead of playing a fixed-length tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    Observation,
    PolicyBase,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots.spaces import ObservationSpace
from robot_console.arm.kinematics import ARM_JOINTS, JOINT_LIMITS, JOINT_ORDER
from robot_console.arm.waypoints import PickPlaceConfig, Plan, build_plan

#: Default per-step joint change, radians. At 10 Hz this is 0.6 rad/s. It was
#: 0.12 (1.2 rad/s), which the actuators track fine but which moves the tool
#: about 25 mm per control period near the apple -- more than the ~4 mm of
#: clearance the open jaws leave around it, so the descent caught the fruit and
#: flicked it away before the jaws ever closed. Measured: at 0.12 the apple is
#: never lifted; at 0.06 it is lifted to z = 0.146 m. Above 0.08 the grasp fails
#: again, and below 0.04 the descent cannot converge inside its step budget.
DEFAULT_MAX_DELTA = 0.06

#: Per-step limit on *opening* the jaw, radians. Closing is deliberately not
#: limited (see ``act``). At 10 Hz this withdraws the fingers over 1.0 s for the
#: 0.50 rad release swing, which measured 0.074 m/s of ejection when done in one
#: step -- the fingers sweep outward through whatever they were holding.
DEFAULT_MAX_GRIPPER_OPEN_DELTA = 0.05

#: The jaw counts as having reached its commanded width within this much.
JAW_TOLERANCE = 0.02

#: Below this much movement in one control period the jaw counts as stalled. The jaw
#: travels about 0.24 contract units per second under its own force limit, so a moving
#: jaw covers ~0.04 per period at 5-10 Hz and a stalled one covers essentially nothing;
#: 0.005 sits an order of magnitude below the former and above solver noise.
JAW_STALL_RATE = 0.005


def _action_box() -> Box:
    low = np.asarray([JOINT_LIMITS[name][0] for name in JOINT_ORDER], dtype=np.float64)
    high = np.asarray([JOINT_LIMITS[name][1] for name in JOINT_ORDER], dtype=np.float64)
    return Box(
        shape=(len(JOINT_ORDER),),
        low=low,
        high=high,
        semantics=ActionSemantics(
            control_mode="joint_pos",
            rotation_repr="none",
            gripper="continuous",
            frame="base",
            dim_labels=(*ARM_JOINTS, "gripper"),
        ),
    )


@dataclass
class _Progress:
    """Where the policy is in the plan and how long it has been there."""

    index: int = 0
    steps_in_waypoint: int = 0
    settled_steps: int = 0
    #: Last measured jaw value, for the stall test in `act`.
    last_jaw: float = 0.0
    finished: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


class SO101WaypointPolicy(PolicyBase):
    """Drive the SO-101 through a solved pick-and-place plan on proprioception."""

    def __init__(
        self,
        *,
        config: PickPlaceConfig | None = None,
        max_delta: float = DEFAULT_MAX_DELTA,
        max_gripper_open_delta: float = DEFAULT_MAX_GRIPPER_OPEN_DELTA,
        name: str = "so101_waypoint",
        control_hz: float = 10.0,
        use_scene_geometry: bool = True,
    ) -> None:
        if not np.isfinite(max_delta) or max_delta <= 0:
            raise ValueError(f"max_delta must be finite and > 0, got {max_delta!r}")
        if not np.isfinite(max_gripper_open_delta) or max_gripper_open_delta <= 0:
            raise ValueError(
                f"max_gripper_open_delta must be finite and > 0, got {max_gripper_open_delta!r}"
            )
        self._base_config = config or PickPlaceConfig()
        self.max_delta = float(max_delta)
        self.max_gripper_open_delta = float(max_gripper_open_delta)
        self.use_scene_geometry = use_scene_geometry
        self.info = PolicyInfo(
            name=name,
            action_space=_action_box(),
            observation_space=ObservationSpace(state_keys=frozenset({"joint_pos"})),
            control_hz=control_hz,
        )
        self.config = PolicyConfig(action_horizon=1, replan_interval=1)
        self._plan: Plan = build_plan(self._base_config)
        self._progress = _Progress()

    # -- lifecycle ---------------------------------------------------------
    def reset(self, scene: Scene) -> None:
        """Re-solve the plan for this scene's geometry and rewind progress."""
        self._plan = build_plan(self._scene_config(scene))
        self._progress = _Progress()

    def _scene_config(self, scene: Scene) -> PickPlaceConfig:
        """Override apple/plate geometry from the scene target when it supplies it.

        Scenes stay the single source of truth for where the objects are, so a
        second scene with a different layout needs no policy change. A scene
        that says nothing falls back to the constructor's configuration.
        """
        if not self.use_scene_geometry or scene.target is None:
            return self._base_config
        spec = scene.target.spec
        updates: dict[str, Any] = {}
        for key in ("apple_xyz", "plate_xyz", "release_xy"):
            if key in spec:
                updates[key] = tuple(float(value) for value in spec[key])
        if not updates:
            return self._base_config
        return replace_config(self._base_config, **updates)

    # -- inference ---------------------------------------------------------
    def act(self, observation: Observation) -> ActionChunk:
        """Return one absolute six-joint command for the current waypoint."""
        measured = self._measured_joints(observation)
        waypoint = self._plan.waypoints[self._progress.index]
        target = self._plan.joint_targets[self._progress.index]

        # Arm joints only. The gripper is force-limited and, once it is holding
        # something, deliberately stalls short of its commanded width -- that is what
        # "gripping" means for a servo that saturates on force. Including it in "have we
        # arrived" makes arrival impossible for every waypoint after the grasp: measured
        # here, `lift_1`, `lift_2`, `lift`, `transit_1` and `transit_2` each burned their
        # full 40-step budget on a joint_error of 0.092 that was entirely the jaw
        # (commanded 0.40, measured stalled at ~0.49 with the apple between the fingers),
        # and the episode ran out of steps in transit having never released. The arm's
        # own error is reported separately because it is the one that says whether the
        # tool actually got where the plan sent it.
        arm_error = float(np.max(np.abs(target[:-1] - measured[:-1])))
        error = arm_error

        # A waypoint has arrived when the joints it is *moving* have arrived, and the
        # jaw is one of those on the grasp waypoint. Two separate hazards, one rule:
        #
        #  - counting the jaw's error as "not arrived" makes arrival impossible for
        #    every carrying waypoint, because a force-limited jaw holding something
        #    deliberately stalls short of its commanded width;
        #  - *ignoring* the jaw makes `close` a step count, and the jaw is much slower
        #    than the arm. At +/-0.30 N.m it needs about 2.5 s to travel from open to
        #    the grasp width; `close_steps` is 8, which at the ~5.4 Hz this scene
        #    actually achieves is 1.5 s. Measured: the jaw was still at 0.76 when the
        #    lift began, the apple stayed on the table, and the run scored zero having
        #    executed all 37 waypoints perfectly.
        #
        # So the jaw counts as arrived when it reaches its target *or* stops moving --
        # a stalled jaw is either shut or pressing on something, and both are the
        # condition to lift on.
        jaw_error = float(abs(target[-1] - measured[-1]))
        jaw_travel = abs(float(measured[-1]) - self._progress.last_jaw)
        self._progress.last_jaw = float(measured[-1])
        jaw_ready = jaw_error <= JAW_TOLERANCE or (
            jaw_travel < JAW_STALL_RATE and self._progress.steps_in_waypoint > 0
        )

        if error <= waypoint.tolerance and jaw_ready:
            self._progress.settled_steps += 1
        else:
            self._progress.settled_steps = 0
        self._progress.steps_in_waypoint += 1

        command = measured + np.clip(target - measured, -self.max_delta, self.max_delta)
        # The gripper is a jaw command, not a tracked trajectory. *Closing* is
        # still instantaneous: rate-limiting it only delays the grasp, and a
        # partially closed jaw slips off the sphere. *Opening* is rate-limited,
        # because the fingers sweep outward through whatever is between them --
        # measured live, releasing 0.50 -> 0.99 in one 100 ms period launched the
        # apple off the plate at 0.074 m/s, and nothing on a flat plate with
        # 0.001 rolling friction ever damps that.
        jaw, measured_jaw = float(target[-1]), float(measured[-1])
        if jaw > measured_jaw:
            jaw = min(jaw, measured_jaw + self.max_gripper_open_delta)
        command[-1] = jaw
        command = np.clip(command, self.info.action_space.low, self.info.action_space.high)

        self._progress.history.append(
            {
                "waypoint": waypoint.name,
                "index": self._progress.index,
                "joint_error": round(error, 5),
                "jaw_error": round(jaw_error, 5),
                "steps_in_waypoint": self._progress.steps_in_waypoint,
            }
        )
        self._advance(waypoint)
        return ActionChunk(
            actions=[
                Action(
                    data=command,
                    meta={
                        "waypoint": waypoint.name,
                        "waypoint_index": self._progress.index,
                        "joint_error": error,
                    },
                )
            ],
            control_hz=self.info.control_hz,
        )

    def _advance(self, waypoint: Any) -> None:
        arrived = self._progress.settled_steps >= waypoint.settle_steps
        exhausted = self._progress.steps_in_waypoint >= waypoint.max_steps
        if not (arrived or exhausted):
            return
        if self._progress.index + 1 >= len(self._plan):
            self._progress.finished = True
            return
        self._progress.index += 1
        self._progress.steps_in_waypoint = 0
        self._progress.settled_steps = 0

    def _measured_joints(self, observation: Observation) -> npt.NDArray[np.float64]:
        raw = observation.state.get("joint_pos")
        if raw is None:
            raise KeyError(
                "observation is missing 'joint_pos'; this policy needs the arm's measured "
                f"joint positions in the order {JOINT_ORDER}"
            )
        measured = np.asarray(raw, dtype=np.float64).reshape(-1)
        if measured.size != len(JOINT_ORDER):
            raise ValueError(
                f"observation 'joint_pos' has {measured.size} entries, expected "
                f"{len(JOINT_ORDER)} for {JOINT_ORDER}"
            )
        return measured

    # -- audit -------------------------------------------------------------
    @property
    def finished(self) -> bool:
        """Whether the final waypoint has been reached or timed out."""
        return self._progress.finished

    @property
    def current_waypoint(self) -> str:
        """Name of the waypoint currently being driven towards."""
        return self._plan.waypoints[self._progress.index].name

    def transcript(self) -> Any | None:
        """Return the per-step waypoint trace for this trial's log."""
        return {
            "plan": [
                {
                    "name": waypoint.name,
                    "xyz": list(waypoint.xyz),
                    "pitch": waypoint.pitch,
                    "gripper": waypoint.gripper,
                    "ik_error_m": round(solve.position_error, 6),
                }
                for waypoint, solve in zip(self._plan.waypoints, self._plan.solves, strict=True)
            ],
            "steps": self._progress.history,
            "finished": self._progress.finished,
        }


def replace_config(config: PickPlaceConfig, **updates: Any) -> PickPlaceConfig:
    """Return a copy of ``config`` with the given fields replaced."""
    import dataclasses

    return dataclasses.replace(config, **updates)


def so101_waypoint(**kwargs: Any) -> SO101WaypointPolicy:
    """Registry entry point for the waypoint policy."""
    numeric = {"max_delta", "control_hz"}
    coerced = {key: (float(value) if key in numeric else value) for key, value in kwargs.items()}
    return SO101WaypointPolicy(**coerced)
