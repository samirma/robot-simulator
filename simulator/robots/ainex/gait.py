"""The AiNex walking gait: parameters, the planar velocity they mean, and leg angles.

The real robot's gait engine is `walking_module.so` -- a precompiled ARM binary with no
source, so it cannot be ported. What *is* open is its parameter block
(`ainex_interfaces/WalkingParam`), the envelope `gait_manager.py` documents, and the
app-layer preset table in `ainex_controller.py`. All three are reproduced here.

The simulated robot does not balance: its torso rides three virtual planar joints (see
`ainex.py`), exactly as `myagv`'s Mecanum base does. So this
module has two jobs, and they are deliberately separate:

* `planar_velocity` -- what a given parameter set *means* as a body-frame velocity, which
  is what actually moves the robot;
* `leg_joint_targets` / `arm_joint_targets` -- what the limbs should be doing while it
  moves, so the animation matches the distance covered.

Pure: no MuJoCo, no ROS.

The velocity derivation
-----------------------
`WalkingParam` is field-for-field the ROBOTIS preview-control walking module's, and in
that module `period_time` T is one full gait cycle (each foot steps once) while
`x_move_amplitude` A is the *body-frame* sagittal half-sweep of a foot -- each foot
travels from -A to +A.

A foot in stance is fixed to the ground, so while it is in stance the body advances by
that foot's body-frame sweep, 2A. Each foot stances once per cycle, so per cycle the body
advances 4A:

    v = 4A / T

Calibration, which is the only reason to believe the factor of 4: at the vendor's default
`period_time: 400` ms and the envelope maximum `x_amplitude` 0.02 m this gives
4*0.02/0.400 = **0.200 m/s**, against Hiwonder's published walking speed of **0.21 m/s**
-- 4.8% low. The two other candidate readings of the amplitude give 2A/T = 0.100 and
A/T = 0.050 m/s, half and a quarter of published. Nothing else lands close.

Note that the published figure is *not* reachable through `/app/set_walking_param`: the
app layer overrides the requested amplitude with a per-tier preset (see APP_PRESETS), so
its fastest setting is 0.16 m/s. Full-envelope amplitudes only arrive over
`/walking/set_param`. That is the vendor's behaviour, not a simplification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .servos import INIT_POSE, clamp

# --- the envelope, from gait_manager.py's own comments and range lists ----------------

PERIOD_RANGE = (0.150, 1.000)  # s; gait_manager only forbids negative, presets span .3-.6
BODY_HEIGHT_RANGE = (0.015, 0.06)  # m below fully upright
STEP_HEIGHT_RANGE = (0.01, 0.04)  # m of foot lift
X_AMPLITUDE_RANGE = (-0.02, 0.02)  # m; gait_manager checks abs() against 0.02
Y_AMPLITUDE_RANGE = (-0.02, 0.02)  # m
ANGLE_AMPLITUDE_RANGE = (-10.0, 10.0)  # degrees
ARM_SWAP_RANGE = (0.0, math.radians(60))
Y_SWAP_RANGE = (0.0, 0.05)

# Body advance per gait cycle, in units of x_move_amplitude. See the module docstring --
# this is the calibrated one.
STRIDE_FACTOR = 4.0
# The same geometry sideways and in yaw. Hiwonder publish a walking speed and no sidestep
# or turn rate, so unlike STRIDE_FACTOR these are carried over for consistency rather
# than measured. They are separate constants so that a future measurement can retune one
# without disturbing the derivation the forward speed rests on.
LATERAL_FACTOR = 4.0
YAW_FACTOR = 4.0


@dataclass(frozen=True)
class WalkingParam:
    """The subset of `ainex_interfaces/WalkingParam` that changes what the robot does.

    Field names and defaults are the vendor's `walking_param.yaml`. `period_time` is in
    **seconds** here while the ROS message carries milliseconds; `ros_surface.py` converts
    at the boundary, so nothing inside this module has to remember which unit it holds.

    The balance block (`balance_enable`, `balance_hip_roll_gain` and friends) is accepted
    on the wire and deliberately not represented: the torso is on position-controlled
    planar joints and cannot tip, so there is nothing for a balance gain to act on.
    """

    period_time: float = 0.400
    dsp_ratio: float = 0.2
    x_amplitude: float = 0.0
    y_amplitude: float = 0.0
    angle_amplitude: float = 0.0  # radians
    body_height: float = 0.025  # init_z_offset: m below fully upright
    step_height: float = 0.02  # z_move_amplitude
    y_swap: float = 0.02
    z_swap: float = 0.006
    arm_swing_gain: float = 0.5
    hip_pitch_offset: float = math.radians(15.0)
    period_times: int = 0  # 0 = walk until stopped; N = walk N steps

    def clamped(self) -> "WalkingParam":
        """Fold the parameters into the envelope, the way gait_manager validates them.

        gait_manager raises on an out-of-range value; a bridge that raised would drop the
        client's message with no way to report why, so this clamps instead. The values a
        well-behaved client sends are unaffected either way.
        """
        return replace(
            self,
            period_time=_clamp(self.period_time, *PERIOD_RANGE),
            dsp_ratio=_clamp(self.dsp_ratio, 0.0, 1.0),
            x_amplitude=_clamp(self.x_amplitude, *X_AMPLITUDE_RANGE),
            y_amplitude=_clamp(self.y_amplitude, *Y_AMPLITUDE_RANGE),
            angle_amplitude=_clamp(
                self.angle_amplitude, *map(math.radians, ANGLE_AMPLITUDE_RANGE)
            ),
            body_height=_clamp(self.body_height, *BODY_HEIGHT_RANGE),
            step_height=_clamp(self.step_height, *STEP_HEIGHT_RANGE),
            y_swap=_clamp(self.y_swap, *Y_SWAP_RANGE),
            arm_swing_gain=_clamp(self.arm_swing_gain, *ARM_SWAP_RANGE),
        )

    @property
    def is_moving(self) -> bool:
        return bool(self.x_amplitude or self.y_amplitude or self.angle_amplitude)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


# --- /app/set_walking_param ------------------------------------------------------------

# ainex_controller.py::set_app_walking_param_callback, transcribed.
#
# Two things about this interface surprise everyone who meets it:
#
#   * `speed` is 1-based and **4 is the fastest**, not the slowest. (gait_manager.move()
#     indexes its own preset list the other way round, so 1 is fastest there. The two
#     vendor entry points genuinely disagree; this table follows the ROS topic, since
#     that is the interface being served.)
#   * the requested x/y/angle are used only for their **sign**. The magnitude is replaced
#     by a per-tier constant -- a larger one when a single axis is commanded alone, a
#     smaller one when axes are mixed. So `/app/set_walking_param` is a direction-and-tier
#     interface, not a continuous velocity one.
#
# speed -> (period_time s, pure-axis x, pure-axis y, pure-axis angle deg,
#           mixed x, mixed y, mixed angle deg)
APP_PRESETS: dict[int, tuple[float, float, float, float, float, float, float]] = {
    4: (0.300, 0.012, 0.015, 10.0, 0.010, 0.010, 8.0),
    3: (0.400, 0.013, 0.015, 10.0, 0.010, 0.010, 8.0),
    2: (0.500, 0.015, 0.015, 10.0, 0.010, 0.010, 10.0),
    1: (0.600, 0.015, 0.012, 10.0, 0.010, 0.010, 10.0),
}
APP_DEFAULT_SPEED = 3  # the 400 ms default period, matching walking_param.yaml


def from_app_params(speed: int, height: float, x: float, y: float, angle_deg: float
                    ) -> WalkingParam:
    """`ainex_interfaces/AppWalkingParam` -> a WalkingParam, as the vendor node maps it."""
    if speed not in APP_PRESETS:
        speed = APP_DEFAULT_SPEED
    period, pure_x, pure_y, pure_a, mix_x, mix_y, mix_a = APP_PRESETS[speed]

    axes = (bool(x), bool(y), bool(angle_deg))
    alone = sum(axes) == 1

    def magnitude(value: float, pure: float, mixed: float) -> float:
        if not value:
            return 0.0
        return math.copysign(pure if alone else mixed, value)

    return WalkingParam(
        period_time=period,
        x_amplitude=magnitude(x, pure_x, mix_x),
        y_amplitude=magnitude(y, pure_y, mix_y),
        angle_amplitude=math.radians(magnitude(angle_deg, pure_a, mix_a)),
        body_height=_clamp(height, *BODY_HEIGHT_RANGE),
        # Fixed by the callback regardless of what the client asks for.
        step_height=0.02,
        hip_pitch_offset=math.radians(15.0),
        arm_swing_gain=0.5,
    ).clamped()


def planar_velocity(param: WalkingParam) -> tuple[float, float, float]:
    """Body-frame (vx m/s, vy m/s, wz rad/s) this gait would produce on the real robot."""
    if param.period_time <= 0:
        return (0.0, 0.0, 0.0)
    return (
        STRIDE_FACTOR * param.x_amplitude / param.period_time,
        LATERAL_FACTOR * param.y_amplitude / param.period_time,
        YAW_FACTOR * param.angle_amplitude / param.period_time,
    )


# --- leg animation ---------------------------------------------------------------------

# Sign taking a rotation expressed about the world +y axis (pitch, nose-down positive) to
# the joint's own value. Read off the `<axis>` elements of the vendor URDF: the left leg's
# hip_pitch and knee turn about +y and its ank_pitch about -y, and the right leg mirrors
# all three. This is why `init_pose.yaml` lists left and right with opposite signs.
PITCH_SIGN = {
    "l_hip_pitch": +1.0, "l_knee": +1.0, "l_ank_pitch": -1.0,
    "r_hip_pitch": -1.0, "r_knee": -1.0, "r_ank_pitch": +1.0,
}
# hip_roll turns about -x on both legs and ank_roll about +x on both, but init_pose gives
# the two legs opposite roll values, i.e. the pelvis is levelled by splaying them. Rolling
# both the same way is what shifts weight sideways, so a swap is applied with these signs.
ROLL_SIGN = {"l_hip_roll": +1.0, "r_hip_roll": +1.0,
             "l_ank_roll": -1.0, "r_ank_roll": -1.0}
YAW_SIGN = {"l_hip_yaw": +1.0, "r_hip_yaw": +1.0}


@dataclass(frozen=True)
class LegGeometry:
    """Thigh and shank lengths, measured from the model rather than typed in.

    `hip_pitch -> knee` and `knee -> ank_pitch`. `ainex.py` measures these off the
    compiled model at load, so the gait stays correct if the description changes.
    """

    thigh: float
    shank: float

    @property
    def max_extension(self) -> float:
        return self.thigh + self.shank


def leg_joint_targets(param: WalkingParam, phase: float, geom: LegGeometry
                      ) -> dict[str, float]:
    """Joint angles for both legs at `phase` in [0, 1) of the gait cycle.

    Sagittal placement is solved as a 2-link IK to a commanded foot position, rather than
    written as per-joint sinusoids, and that buys two properties a hand-tuned set of
    sinusoids cannot hold across the whole envelope:

    * the **stance foot stays at a constant height under the hip**, so the torso's ride
      height is a constant and does not have to be re-measured per parameter set -- which
      matters because the torso is on planar joints and cannot rise or fall;
    * **no foot skate.** With the foot placed at x = +/-A*cos(2*pi*phase), the stance
      foot's backward travel in the body frame is 2A per stance, which is exactly the
      distance `planar_velocity` moves the base over the same interval. The animation
      therefore matches the distance covered at every point of the envelope instead of at
      one tuned operating point.

    The legs are cosmetic in the sense that nothing here carries the robot's weight -- but
    a foot that slides while the robot moves is the thing that makes a walk look wrong,
    and it is free to avoid.
    """
    lift = param.step_height
    stride = param.x_amplitude
    sway = param.y_amplitude
    out: dict[str, float] = {}

    for side, offset in (("l", 0.0), ("r", 0.5)):
        # The two legs are half a cycle apart. `p` in [0,1): the leg is in **stance** over
        # the first half and in swing over the second.
        p = (phase + offset) % 1.0
        swinging = p >= 0.5

        # The planted foot travels *backwards* through the body frame while the body moves
        # forwards over it, from +A to -A, and the lifted foot returns it to +A. Getting
        # that the other way round is not a cosmetic error: it makes the stance foot skate
        # forwards at twice the walking speed.
        #
        # Stance is deliberately **linear** in p, not a cosine. The base advances at a
        # constant velocity (`planar_velocity`), so only a foot that also travels at
        # constant speed cancels against it exactly; a sinusoidal stance matches over the
        # whole half-cycle but lags mid-stance, which showed up as ~8 mm of skate at the
        # top of the envelope. The swing half is cosine-eased instead, because there the
        # foot is off the ground and only smoothness matters.
        sweep = _sweep(p)
        x_foot = stride * sweep
        y_foot = sway * sweep
        # Only the swing foot leaves the ground, and it is re-based onto the swing half so
        # the lift is zero at both touch-down and lift-off rather than discontinuous.
        z_lift = lift * math.sin(2 * math.pi * (p - 0.5)) if swinging else 0.0

        # Extension measured from the hip. body_height is how far below fully upright the
        # torso sits, so the stance leg is shortened by it; the swing leg additionally
        # lifts. The torso itself never moves vertically -- it is on planar joints.
        extension = geom.max_extension - param.body_height - z_lift
        hip, knee = _sagittal_ik(x_foot, extension, geom)
        # Sole parallel to the floor: hip + knee + ankle == 0.
        #
        # The vendor's `hip_pitch_offset` (15 degrees) does *not* appear here, and that is
        # deliberate. On the real robot it leans the torso forward over the feet, which is
        # visible in init_pose.yaml as a leg chain whose rotations sum to -14.95 degrees
        # rather than to zero. Our torso is bolted to three planar joints and cannot
        # pitch, so applying the offset here would rotate the *feet* by 15 degrees instead
        # of the body -- the robot would walk on its heels. It joins the balance gains as
        # a parameter accepted on the wire with nothing to act on.
        ankle = -(hip + knee)

        out[f"{side}_hip_pitch"] = PITCH_SIGN[f"{side}_hip_pitch"] * hip
        out[f"{side}_knee"] = PITCH_SIGN[f"{side}_knee"] * knee
        out[f"{side}_ank_pitch"] = PITCH_SIGN[f"{side}_ank_pitch"] * ankle

        # Lateral weight shift, half a cycle out of phase with the step, plus whatever
        # sideways stride was asked for.
        roll = param.y_swap * math.sin(2 * math.pi * p) + y_foot
        hip_roll = math.atan2(roll, max(extension, 1e-6))
        out[f"{side}_hip_roll"] = (
            INIT_POSE[f"{side}_hip_roll"] + ROLL_SIGN[f"{side}_hip_roll"] * hip_roll
        )
        out[f"{side}_ank_roll"] = (
            INIT_POSE[f"{side}_ank_roll"] + ROLL_SIGN[f"{side}_ank_roll"] * hip_roll
        )
        out[f"{side}_hip_yaw"] = (
            YAW_SIGN[f"{side}_hip_yaw"] * param.angle_amplitude * math.cos(2 * math.pi * p)
        )

    # The envelope's corners are reachable individually but not together: the deepest
    # crouch (body_height 0.06) plus the highest lift (step_height 0.04) shortens a
    # 0.186 m leg by 0.10 m, which folds the knee past the servo's own travel. A real
    # servo simply stops at its limit, so clamping here matches it -- and stops the model
    # driving an actuator into a joint limit it can only fight.
    return {name: clamp(name, value) for name, value in out.items()}


def _sweep(p: float) -> float:
    """Fore-aft position of one foot over its cycle, in units of the stride amplitude.

    +1 at touch-down, running linearly to -1 at lift-off, then eased back to +1 while the
    foot is in the air. Continuous and periodic; see `leg_joint_targets` for why stance is
    the straight-line half.
    """
    if p < 0.5:  # stance: constant ground speed, so it cancels the base's velocity
        return 1.0 - 4.0 * p
    return -math.cos(2 * math.pi * (p - 0.5))  # swing: -1 -> +1, zero slope at both ends


def _sagittal_ik(x_foot: float, extension: float, geom: LegGeometry
                 ) -> tuple[float, float]:
    """Hip and knee pitch, about +y, putting the foot at (x_foot, -extension) from the hip.

    Standard planar two-link. The reach is clamped rather than allowed to raise a domain
    error, because a parameter set that asks for a stride longer than the leg should look
    like a robot at full stretch, not like a crashed simulator.
    """
    reach = math.hypot(x_foot, extension)
    reach = min(max(reach, 1e-6), geom.max_extension - 1e-6)

    cos_interior = (geom.thigh**2 + geom.shank**2 - reach**2) / (2 * geom.thigh * geom.shank)
    interior = math.acos(min(max(cos_interior, -1.0), 1.0))
    knee = math.pi - interior

    # Angle of the leg vector from straight down, less the offset the bent knee introduces.
    knee_offset = math.asin(min(max(geom.shank * math.sin(knee) / reach, -1.0), 1.0))
    hip = math.atan2(x_foot, extension) - knee_offset
    return hip, knee


def arm_joint_targets(param: WalkingParam, phase: float) -> dict[str, float]:
    """Shoulder swing, opposite the leg on the same side.

    `arm_swing_gain` of 0 means "send the arms nothing at all" on the real robot
    (gait_manager documents it, and ainex_controller skips both sho_pitch joints when it
    is zero), so this returns an empty dict rather than zeros -- letting whatever the arms
    were doing, such as holding a grasped object, continue.
    """
    if not param.arm_swing_gain:
        return {}

    out = {}
    for side, offset in (("l", 0.0), ("r", 0.5)):
        p = (phase + offset) % 1.0
        # Opposite the leg of the same side: the leg swings forward on cos(2*pi*p), so
        # negating gives the counter-swing a walk actually has.
        swing = -param.arm_swing_gain * math.cos(2 * math.pi * p)
        name = f"{side}_sho_pitch"
        out[name] = clamp(name, INIT_POSE[name] + swing)
    return out


def rest_pose() -> dict[str, float]:
    """The vendor's init pose -- what `walking/init_pose` drives to and walking starts from."""
    return dict(INIT_POSE)
