"""The deterministic pick-and-place waypoint plan for the apple-on-plate task.

Shared by the ROS policy and the offline MuJoCo validator so both play exactly
the same joint targets. The plan is expressed in task space (TCP xyz plus an
approach pitch) and lowered to six absolute joint positions through
[`robot_console.arm.kinematics`][], seeding each solve from the previous one so the
arm stays on a single IK branch across the whole episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from robot_console.arm.kinematics import (
    ARM_JOINTS,
    IKResult,
    approach_pitch,
    fk,
    JAW_CENTER_OFFSET,
    ik_position,
    jaw_axis,
    jaw_center,
    level_jaw_roll,
)

#: Native ``gripper_joint`` radians. The MJCF range is 0..1 and the measured jaw
#: aperture grows monotonically with it, so 0 is fully closed and 1 fully open.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0


@dataclass(frozen=True)
class Waypoint:
    """One task-space pose the arm should reach, plus how long to hold it."""

    name: str
    xyz: tuple[float, float, float]
    pitch: float
    gripper: float
    settle_steps: int = 2
    #: Roll the wrist so the jaws open horizontally. Needed wherever the tool is
    #: near the table: a tilted jaw axis drives one finger into the surface.
    level_jaws: bool = True
    #: Joint-space tolerance (radians, max abs) that counts as "arrived".
    tolerance: float = 0.05
    #: Hard step budget so an unreachable pose cannot stall the episode.
    max_steps: int = 40


@dataclass(frozen=True)
class Plan:
    """A sequence of waypoints already lowered to absolute joint targets."""

    waypoints: tuple[Waypoint, ...]
    joint_targets: tuple[npt.NDArray[np.float64], ...]
    solves: tuple[IKResult, ...] = field(repr=False, default=())

    def __post_init__(self) -> None:
        if len(self.waypoints) != len(self.joint_targets):
            raise ValueError("plan must have one joint target per waypoint")

    def __len__(self) -> int:
        return len(self.waypoints)

    @property
    def worst_position_error(self) -> float:
        """Largest IK residual over the plan, in metres."""
        return max((solve.position_error for solve in self.solves), default=0.0)


@dataclass(frozen=True)
class PickPlaceConfig:
    """Geometry and clearances the plan is built from, all in metres/radians.

    Defaults describe the simulator's ``task_scene.xml`` measured in world
    coordinates: the apple is a 20 mm sphere resting at ``(0.30, 0.10, 0.020)``
    and the plate is centred at ``(0.226, -0.226)`` with its collision top face
    at ``z = 0.0204``. The apple centre was 0.03 here until contract v2; that
    described a 60 mm ball the SO-101's 43 mm jaws could not close on, and it
    put every grasp waypoint 10 mm too high.
    """

    apple_xyz: tuple[float, float, float] = (0.30, 0.10, 0.020)
    plate_xyz: tuple[float, float, float] = (0.226, -0.226, 0.020)
    #: Where over the plate to release, re-derived for the moved plate rather
    #: than translated from the old point. Two gates bound it, both from
    #: ``CONTRACT.md`` section 5: clause 1 wants the apple within 0.080 m of the
    #: plate centre, clause 5 wants 0.25 m of travel from the spawn at
    #: ``(0.30, 0.10, 0.020)``.
    #:
    #: The new plate centre is 0.3343 m from the spawn horizontally and 0.3349 m
    #: in 3-D once the apple rests at z = 0.0404, so the clause-1 disc spans
    #: 0.2551 m to 0.4148 m of travel. Clause 5 is therefore *implied* by clause
    #: 1 here -- but only by 5.1 mm at the near edge. At the old centre
    #: ``(0.40, -0.20)`` the near edge sat at 0.2371 m and failed clause 5
    #: outright, which is why the old point was biased 0.0424 m to the NEAR side
    #: and why it had to be. That reasoning does not carry over and its sign is
    #: now wrong.
    #:
    #: Reach no longer binds either. The old centre was 0.4472 m from the base
    #: and its far half left the workspace; the new one is 0.3196 m out, and
    #: every candidate from 40 mm near to 60 mm far of it solves to an IK
    #: residual under 0.1 mm at both the transit and the release pose.
    #:
    #: So this point is biased 0.0247 m to the FAR side along the spawn->plate
    #: axis. It is 0.0247 m from the plate centre (gate 0.080), 0.3596 m of
    #: travel from the spawn (gate 0.25) and 0.3330 m from the base. The bias is
    #: deliberate asymmetry, not a gate requirement: it widens the tolerable
    #: near-ward settling error from 80 mm to 105 mm and narrows the far-ward one
    #: from 80 mm to 55 mm. Near-ward is the direction to protect, because a
    #: reach that falls short at this bearing moves the apple 53% toward the
    #: spawn, and because the near edge is where the 5.1 mm of clause-5 slack is.
    release_xy: tuple[float, float] = (0.220, -0.250)
    approach_height: float = 0.12
    #: TCP height at the grasp, relative to the apple centre.
    grasp_z_offset: float = -0.002
    lift_height: float = 0.14
    #: Tool height while crossing to the plate and while lowering onto it. The
    #: lift off the table needs ``lift_height``; this does not. It was 0.14 while
    #: the plate was at ``(0.40, -0.20)``, and carrying the tool that high that
    #: far out is what pushed the release point out of the arm's reach. At the
    #: moved plate, 0.3196 m from the base, reach no longer forces the issue --
    #: 0.09 is kept because it is what the placing runs were measured at, and
    #: because the carried apple's bottom sits at 0.070 and only has to clear the
    #: plate rim at 0.0204, which it does by 0.0496 m.
    transit_height: float = 0.09
    #: Jaw-centre height at the moment the jaws open. The jaw centre carries the
    #: apple centre, and an apple resting on the plate has its centre at
    #: 0.0404 (plate top 0.0204 + radius 0.020), so this must be just above that
    #: -- not 0.105, which opened the jaws with the apple 64.6 mm in the air and
    #: dropped it. A dropped apple bounces and rolls, and never banks the 1.0 s
    #: at-rest hold that CONTRACT.md section 5 clause 4 requires.
    release_height: float = 0.045
    grasp_pitch: float = -1.15
    lift_pitch: float = -1.0
    transit_pitch: float = -0.75
    release_pitch: float = -0.5
    #: Waypoint poses name the midpoint between the jaws, not the ``gripperframe``
    #: site; this is the offset between them along the jaw axis.
    jaw_center_offset: float = JAW_CENTER_OFFSET
    #: Largest IK residual tolerated before ``build_plan`` refuses the geometry.
    max_position_error: float = 0.005
    home_xyz: tuple[float, float, float] = (0.26, 0.0, 0.18)
    home_pitch: float = -0.5
    #: Jaw command while lifting and carrying, contract units (0 closed, 1 open).
    #:
    #: **0.40 on this arm, and the number is about the servo, not the aperture.** The
    #: gripper actuator is force-limited to +/-0.30 N.m, so it only squeezes while it is
    #: *stalled* short of its target: command a width the apple physically blocks and the
    #: full 0.30 N.m is applied for as long as the apple is there; command a width the
    #: jaw can actually reach and the servo arrives, its error goes to zero, and the apple
    #: is held by nothing but its own compliance. A 40 mm apple blocks anything under
    #: ~0.52, so 0.50 -- which a softer jaw model held with -- reaches its target here
    #: and the apple creeps out during the carry (measured live: jaw/apple contacts
    #: 6 -> 5 -> 4 -> 3 -> 2 -> 0 over five seconds, dropped a quarter of the way to the
    #: plate). Sweeping the whole scripted plan offline, ramped exactly as the policy
    #: ramps it:
    #:
    #:     grasp_gripper   0.20  0.25  0.30  0.35  0.40  0.45  0.50
    #:     final position  off   off   off   ON    ON    ON    dropped in transit
    #:                    (over-squeezed and thrown at release)
    #:
    #: 0.40 is the middle of the window. A measured jaw that stalls around 0.50 while
    #: this is commanded is the apple between the fingers, not a fault: the `close`
    #: waypoint runs to its step budget rather than settling, by design.
    grasp_gripper: float = 0.40

    #: Which branch of the wrist roll to solve on, or None to let `level_jaw_roll`
    #: choose. There are always **two** branches, half a turn apart, since a jaw rolled
    #: by pi is the same level jaw -- and on this arm they are *not* interchangeable in
    #: practice, which is worth knowing before anyone "tidies" this back up.
    #:
    #: Unseeded the solver picks -1.52, and that is what ships. Seeding +1.62 was tried,
    #: because it is the branch the VLA's trained state band lives in, and it made the
    #: *release* fail: lowering the held apple onto the plate, the arm error grew
    #: 0.057 -> 0.087 -> 0.118 rad across the last three sub-waypoints and each ran its
    #: full 70-step budget. That is a blocked arm, not a slow one -- the flipped jaw
    #: fouls the plate's rim on the way down. The episode still reached the plate (the
    #: apple landed 7-10 mm from centre, better than the 12-13 mm of the shipped branch)
    #: but ran out of budget before the 1.0 s hold could complete.
    #:
    #: The VLA's need is met elsewhere and does not require this: the *start pose* it
    #: conditions on is the task's business, and `shared/tasks/apple_on_plate.py` sets
    #: that in-band independently. The scripted plan simply swings the wrist once on its
    #: way to `home`, which costs it steps it has to spare.
    wrist_roll_preference: float | None = None
    settle_steps: int = 3
    close_steps: int = 8
    #: Steps the arm holds still while opening the jaws and then waiting --
    #: 3.0 s at 10 Hz. The jaws now withdraw over ~1.0 s (see
    #: ``policy.DEFAULT_MAX_GRIPPER_OPEN_DELTA``), so this leaves ~2.0 s of
    #: genuinely still holding, against the 1.0 s the contract needs. CONTRACT.md section 5 clause 4 needs the apple at rest for
    #: 1.0 s, and measured runs at release_steps=6 banked only 0.20-0.50 s before
    #: `retreat` lifted the open jaws back through the apple and knocked it off.
    #: The retreat is not required by the task; the hold is.
    release_steps: int = 30
    #: Joint-space tolerance (rad) for the descent onto the apple and for the
    #: close. The open jaws clear the 40 mm apple by only ~4 mm a side, so the
    #: tool has to be centred to a few millimetres before the fingers come down
    #: past the equator; ``_ramp``'s general-purpose 0.05 rad is more than 10 mm
    #: at this arm's lever and let the descent catch the fruit. Everything above
    #: the apple keeps the looser tolerance, because precision there costs steps
    #: and the grasp is only good for a few seconds once made.
    descend_tolerance: float = 0.012

    #: Joint-space tolerance for *lowering the held apple onto the plate*, which used to
    #: share `descend_tolerance` and should not. 12 mm is sized for straddling a 40 mm
    #: apple with jaws that clear it by ~4 mm a side; putting that apple down on a 200 mm
    #: plate is a far coarser job, and the success gate agrees -- 80 mm horizontally.
    #:
    #: Sharing the tight one is only free while the arm can actually hit it. Measured
    #: after the wrist moved to its in-band roll branch: `over_plate_5`, `_6` and `_7`
    #: each ran their full 70-step budget at an arm error of 0.02-0.09 rad, burning 176
    #: steps, and the episode reached the plate but ran out of budget before the 1.0 s
    #: hold could complete. The apple was landing 7-10 mm from the plate centre at the
    #: time -- better than the 12-13 mm the tighter tolerance was buying.
    lower_tolerance: float = 0.05
    #: Step budget per descend sub-waypoint. A tight tolerance needs room to
    #: converge; without this the descent would simply time out at the old 40.
    descend_max_steps: int = 70
    #: How many straight-line sub-waypoints each leg is split into.
    descend_segments: int = 8
    lift_segments: int = 3
    #: The apple sags out of the jaws while carried, so the transit is split
    #: finely to keep tool speed down; see the note on ``grasp_gripper``.
    transit_segments: int = 6
    #: The descent onto the plate is 45 mm -- ``transit_height`` 0.09 down to
    #: ``release_height`` 0.045 -- so it is split finely for the same reason the
    #: approach is: a big step at the end of a reach arrives with enough lateral
    #: error to catch the plate rim. This comment claimed 95 mm, which was the
    #: descent from ``lift_height`` 0.14 and has been stale since
    #: ``transit_height`` was introduced. It does not depend on the plate's xy.
    lower_segments: int = 8


def _ramp(
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    pitch_start: float,
    pitch_end: float,
    gripper: float,
    *,
    segments: int,
    settle_steps: int,
    tolerance: float = 0.05,
    max_steps: int = 40,
) -> list[Waypoint]:
    """Split a move into straight-line sub-waypoints in task space.

    Interpolating in *joint* space between two IK solutions bows the tool
    through an arc, which is how a descent aimed beside an object still knocks
    it over. Emitting intermediate task-space poses keeps the path straight
    enough that the arc never leaves the corridor the clearances allow.
    """
    out: list[Waypoint] = []
    for index in range(1, segments + 1):
        fraction = index / segments
        xyz = tuple(
            float(a + (b - a) * fraction) for a, b in zip(start, end, strict=True)
        )
        pitch = pitch_start + (pitch_end - pitch_start) * fraction
        final = index == segments
        out.append(
            Waypoint(
                name=name if final else f"{name}_{index}",
                xyz=(xyz[0], xyz[1], xyz[2]),
                pitch=pitch,
                gripper=gripper,
                settle_steps=settle_steps if final else max(1, settle_steps // 2),
                tolerance=tolerance,
                max_steps=max_steps,
            )
        )
    return out


def build_waypoints(config: PickPlaceConfig | None = None) -> tuple[Waypoint, ...]:
    """Return the task-space waypoint sequence for one pick-and-place episode.

    Every pose names the **midpoint between the jaws**, not the tool frame
    origin; ``build_plan`` handles the offset between them.
    """
    cfg = config or PickPlaceConfig()
    apple_x, apple_y, apple_z = cfg.apple_xyz
    grasp_height = apple_z + cfg.grasp_z_offset
    release_x, release_y = cfg.release_xy

    approach = (apple_x, apple_y, cfg.approach_height)
    grasp = (apple_x, apple_y, grasp_height)
    lifted = (apple_x, apple_y, cfg.lift_height)
    over = (release_x, release_y, cfg.transit_height)
    drop = (release_x, release_y, cfg.release_height)

    waypoints: list[Waypoint] = [
        Waypoint("home", cfg.home_xyz, cfg.home_pitch, GRIPPER_OPEN, cfg.settle_steps),
        Waypoint("pre_grasp", approach, cfg.grasp_pitch, GRIPPER_OPEN, cfg.settle_steps),
    ]
    waypoints += _ramp(
        "descend", approach, grasp, cfg.grasp_pitch, cfg.grasp_pitch, GRIPPER_OPEN,
        segments=cfg.descend_segments, settle_steps=cfg.settle_steps,
        tolerance=cfg.descend_tolerance, max_steps=cfg.descend_max_steps,
    )
    waypoints.append(
        Waypoint("close", grasp, cfg.grasp_pitch, cfg.grasp_gripper, cfg.close_steps,
                 tolerance=cfg.descend_tolerance, max_steps=cfg.descend_max_steps)
    )
    waypoints += _ramp(
        "lift", grasp, lifted, cfg.grasp_pitch, cfg.lift_pitch, cfg.grasp_gripper,
        segments=cfg.lift_segments, settle_steps=cfg.settle_steps, tolerance=0.08,
    )
    waypoints += _ramp(
        "transit", lifted, over, cfg.lift_pitch, cfg.transit_pitch, cfg.grasp_gripper,
        segments=cfg.transit_segments, settle_steps=cfg.settle_steps, tolerance=0.08,
    )
    waypoints += _ramp(
        "over_plate", over, drop, cfg.transit_pitch, cfg.release_pitch, cfg.grasp_gripper,
        segments=cfg.lower_segments, settle_steps=cfg.settle_steps,
        tolerance=cfg.lower_tolerance, max_steps=cfg.descend_max_steps,
    )
    waypoints.append(
        Waypoint("release", drop, cfg.release_pitch, GRIPPER_OPEN, cfg.release_steps,
                 tolerance=0.08)
    )
    waypoints += _ramp(
        "retreat", drop, over, cfg.release_pitch, cfg.release_pitch, GRIPPER_OPEN,
        segments=cfg.lower_segments, settle_steps=cfg.settle_steps,
    )
    return tuple(waypoints)


def build_plan(
    config: PickPlaceConfig | None = None,
    *,
    seed: npt.ArrayLike | None = None,
    pitch_weight: float = 1.0,
) -> Plan:
    """Lower the waypoint sequence to six-element absolute joint targets."""
    cfg = config or PickPlaceConfig()
    waypoints = build_waypoints(cfg)
    targets: list[npt.NDArray[np.float64]] = []
    solves: list[IKResult] = []
    previous_roll: float | None = cfg.wrist_roll_preference
    current = (
        np.zeros(len(ARM_JOINTS), dtype=np.float64)
        if seed is None
        else np.asarray(seed, dtype=np.float64).reshape(-1)[: len(ARM_JOINTS)].copy()
    )
    offset = cfg.jaw_center_offset if cfg.jaw_center_offset else 0.0
    for waypoint in waypoints:
        goal = np.asarray(waypoint.xyz, dtype=np.float64)
        solve = ik_position(
            goal, seed=current, pitch=waypoint.pitch, pitch_weight=pitch_weight,
            max_iterations=600,
        )
        roll = level_jaw_roll(solve.joints, prefer=previous_roll) if waypoint.level_jaws else 0.0
        # The jaw axis depends on the pose being solved for, so aim the site at
        # "goal minus half an aperture" and re-solve until the *jaw centre*, not
        # the site, lands on the waypoint. Two refinements are ample: the jaw
        # axis barely turns over a 22 mm correction.
        for _ in range(3):
            aim = goal - offset * jaw_axis(fk(solve.joints))
            solve = ik_position(
                aim, seed=solve.joints, pitch=waypoint.pitch, pitch_weight=pitch_weight,
                roll=roll, max_iterations=600,
            )
        current = solve.joints
        solve = IKResult(
            joints=current,
            position_error=float(np.linalg.norm(goal - jaw_center(current, offset=offset))),
            converged=True,
        )
        previous_roll = float(current[4])
        solves.append(solve)
        targets.append(np.concatenate((current, [waypoint.gripper])))
        if solve.position_error > cfg.max_position_error:
            raise ValueError(
                f"waypoint {waypoint.name!r} at {waypoint.xyz} with pitch "
                f"{waypoint.pitch:.3f} is {solve.position_error:.4f} m out of reach for the "
                f"SO-101 (limit {cfg.max_position_error} m); relax the pitch or move the pose "
                "closer to the base"
            )
    return Plan(waypoints=waypoints, joint_targets=tuple(targets), solves=tuple(solves))


def describe_plan(plan: Plan) -> str:
    """Render a human-readable table of the solved plan, for logs and docs."""
    lines = [
        f"{'waypoint':<10} {'target xyz':<26} {'jaw centre':<26} "
        f"{'err(m)':>7} {'pitch':>7} {'roll':>6} {'tilt':>6} {'grip':>5}"
    ]
    for waypoint, joints, solve in zip(plan.waypoints, plan.joint_targets, plan.solves, strict=True):
        pose = fk(joints[: len(ARM_JOINTS)])
        lines.append(
            f"{waypoint.name:<10} {np.round(waypoint.xyz, 4)!s:<26} "
            f"{np.round(jaw_center(joints[: len(ARM_JOINTS)]), 4)!s:<26} "
            f"{solve.position_error:>7.4f} "
            f"{approach_pitch(pose):>7.3f} {joints[4]:>6.2f} "
            f"{jaw_axis(pose)[2]:>6.3f} {joints[-1]:>5.2f}"
        )
    return "\n".join(lines)
