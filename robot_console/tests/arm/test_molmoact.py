"""The MolmoAct2 adapter's units, pinned against the checkpoint's own norm_stats.

Two defects lived here and both were silent -- no exception, no shape change, no
NaN, just a robot that behaved slightly wrongly:

1. The gripper channel is a jaw angle in **degrees**. Clipping the raw action to
   [0, 1] saturated everything >= 1 to fully open, and the action median is
   4.867, so most of the distribution did exactly that. The model closed the jaw
   and the adapter reopened it.
2. Sending ``degrees(jaw)`` as state put it at 0..57.3 against a model range of
   0.94..44.14. State is binned into 256 buckets and clips without error.

These tests read ``norm_stats.json`` from the real snapshot, so they fail if a
future edit changes the constants *or* if the checkpoint's own statistics move.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_console.arm.kinematics import JOINT_LIMITS, JOINT_ORDER
from robot_console.arm.molmoact import (
    CLIP_COUNTS,
    GRIPPER_ACTION_OPEN_DEG,
    GRIPPER_MIN_COMMAND,
    GRIPPER_STATE_OPEN_DEG,
    HIGH,
    LOW,
    OFFSETS,
    RAIL_TOLERANCE_DEG,
    SIGNS,
    assert_state_in_distribution,
    norm_stats,
    to_joint_actions,
    to_model_state,
)

pytest.importorskip("huggingface_hub", reason="norm_stats.json comes from the HF snapshot")

REST = np.zeros(6)


@pytest.fixture(scope="module")
def home() -> np.ndarray:
    """The pose every episode actually starts from: the task's start pose, jaw open.

    ``REST`` (all zeros) is a convenient constant, not the initial condition: the
    simulator puts the arm at ``START_ARM_QPOS`` before it snapshots the spawn state,
    and the console mirrors that tuple in ``task.py``. wrist_roll there is +1.62 rad,
    chosen so the corrected calibration maps it to the middle of the trained band
    (model -2.8 deg) -- see the simulator's ``apple_on_plate.START_ARM_QPOS``.
    """
    from robot_console.arm.task import START_ARM_QPOS

    return np.asarray([*START_ARM_QPOS, 1.0], dtype=np.float64)

#: The floor this suite pins, kept as a literal so a silent edit to the constant
#: fails here rather than passing by comparing the value to itself. It is 0.0
#: since 2026-08-30: the H3 actuator fix (gripper_joint forcerange +-3.35 ->
#: +-0.30 N.m) removed the crush that the floor worked around, and the measured
#: holding window g in [0.00, 0.35] lies ENTIRELY BELOW the old 0.50.
EXPECTED_FLOOR = 0.0

#: The value it used to hold, still exercised below as the documented rollback.
LEGACY_FLOOR = 0.50

#: The solver-validated holding window under the fixed actuator: every command in
#: this range holds the 40 mm apple, and 0.40 and above does not. From the
#: simulator role, 17/17 on a 2 mm-smaller / 2 mm-larger sphere control.
HOLDING_WINDOW = (0.00, 0.35)


@pytest.fixture
def no_floor(monkeypatch):
    """Run a test against the raw degree rescale, with the floor disabled.

    Since the floor went to 0.0 this is the LIVE configuration, not a special
    case. It is kept, and kept explicit, so the tests that describe the mapping
    underneath the floor still state their premise rather than depending on the
    module constant happening to be zero today.
    """
    from robot_console.arm import molmoact
    monkeypatch.setattr(molmoact, "GRIPPER_MIN_COMMAND", 0.0)
    return molmoact


@pytest.fixture
def legacy_floor(monkeypatch):
    """Restore the pre-H3 0.50 floor, so the rollback path stays pinned."""
    from robot_console.arm import molmoact
    monkeypatch.setattr(molmoact, "GRIPPER_MIN_COMMAND", LEGACY_FLOOR)
    return molmoact


@pytest.fixture(scope="module")
def stats():
    try:
        return norm_stats()
    except Exception as exc:  # offline, or the snapshot is not present
        pytest.skip(f"MolmoAct2 snapshot unavailable: {type(exc).__name__}: {exc}")


def test_the_checkpoint_agrees_on_joint_order(stats) -> None:
    """Our contract order and the checkpoint's channel order must be the same."""
    names = stats["action_stats"]["names"]
    assert names == [name.replace("_joint", "") for name in JOINT_ORDER]
    assert stats["control_mode"] == "absolute joint pose"


def test_gripper_constants_come_from_norm_stats(stats) -> None:
    """Pin both scales to the checkpoint's own q99, so an edit cannot drift."""
    # The constants are the published q99 rounded to 2 dp; pin them to within
    # half of that last place so a real change to the checkpoint fails here.
    assert GRIPPER_ACTION_OPEN_DEG == pytest.approx(stats["action_stats"]["q99"][5], abs=5e-3)
    assert GRIPPER_STATE_OPEN_DEG == pytest.approx(stats["state_stats"]["q99"][5], abs=5e-3)


def test_the_gripper_channel_is_degrees_not_a_unit_interval(stats, no_floor) -> None:
    """The defect this file exists for: ch5 is a degree scale, so >1 is normal."""
    action = stats["action_stats"]
    assert action["q50"][5] > 1.0, "a 0..1 channel could not have a median of 4.867"
    assert action["q99"][5] > 40.0
    # The old code did np.clip(raw, 0, 1). Show what that did to the median.
    median = action["q50"][5]
    assert np.clip(median, 0.0, 1.0) == 1.0, "the median saturated to fully open"
    assert no_floor.to_joint_actions(np.tile([0, 90, 90, 0, 0, median], (1, 1)))[0, 5] < 0.2, (
        "the correct rescale must leave a median jaw command nearly closed"
    )


def test_action_jaw_rescales_across_the_full_range(no_floor) -> None:
    chunk = np.zeros((3, 6))
    chunk[:, 1] = 90.0
    chunk[:, 2] = 90.0
    chunk[:, 5] = [0.0, GRIPPER_ACTION_OPEN_DEG / 2.0, GRIPPER_ACTION_OPEN_DEG]
    jaw = no_floor.to_joint_actions(chunk)[:, 5]
    assert jaw == pytest.approx([0.0, 0.5, 1.0], abs=1e-6)


def test_state_jaw_rescales_into_the_model_band() -> None:
    assert to_model_state(np.r_[np.zeros(5), 0.0])[5] == pytest.approx(0.0)
    assert to_model_state(np.r_[np.zeros(5), 1.0])[5] == pytest.approx(GRIPPER_STATE_OPEN_DEG)
    # The old mapping sent degrees(1.0) = 57.3, well past the q99 of 44.14.
    assert np.degrees(1.0) > GRIPPER_STATE_OPEN_DEG


def test_arm_channels_use_the_published_calibration() -> None:
    """signs/offsets, with the measured wrist_roll correction, and they round-trip.

    Channels 0-3 are the published lerobot/MolmoAct2-SO100_101-LeRobot table.
    Channel 4 is NOT: see the wrist_roll block on ``SIGNS`` in molmoact.py. The
    published mapping put wrist_roll outside the checkpoint's own trained band for
    98.6% of seven arbiter-PASSING scripted runs -- median +90.05 against a
    training q50 of -11.04, about 47 deg past q99 -- and a physically correct
    execution of the task has to look like training data.
    """
    assert SIGNS.tolist() == [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
    assert OFFSETS.tolist() == [0.0, 90.0, 90.0, 0.0, 90.0, 0.0]
    measured = np.array([0.2, -0.4, 0.3, 0.1, -0.5, 0.25])
    model = to_model_state(measured)
    back = to_joint_actions(model.reshape(1, 6))[0]
    # Channels 0..4 only: ch5 is deliberately not a round trip once the crush
    # floor applies, and that asymmetry is the point of GRIPPER_MIN_COMMAND.
    assert back[:5] == pytest.approx(measured[:5], abs=1e-9)


def test_wrist_roll_is_reversed_and_offset_by_a_quarter_turn() -> None:
    """The one behavioural change of this round, stated as an equation.

    ``model_wrist_roll = 90 - ours_in_degrees``. Our +pi/2 is the "level jaw" pose
    (``kinematics.level_jaw_roll``), which is the natural zero for the LeRobot
    convention -- an ordinary frame-convention difference, not a tuned constant.
    """
    for ours_deg, expected in [
        (90.0, 0.0),       # level jaw -> the model's zero
        (0.0, 90.0),       # what the shipped mapping wrongly called zero
        (45.0, 45.0),
        (-45.0, 135.0),
        (131.78, -41.78),  # our +2.3 rad limit
    ]:
        measured = np.r_[np.zeros(4), np.radians(ours_deg), 0.0]
        assert to_model_state(measured)[4] == pytest.approx(expected, abs=1e-6), ours_deg


def test_the_round_trip_is_an_involution_over_random_poses() -> None:
    """to_joint_actions(to_model_state(q)) == q for every reachable arm pose.

    Sampled over the full MJCF range of all five arm joints, not just near zero:
    a sign or offset error on ONE channel is invisible at that channel's fixed
    point, and index 4's old offset of 0 was exactly such a fixed point at q = 0.
    """
    rng = np.random.default_rng(20260830)
    low = np.array([JOINT_LIMITS[name][0] for name in JOINT_ORDER[:5]])
    high = np.array([JOINT_LIMITS[name][1] for name in JOINT_ORDER[:5]])
    arms = rng.uniform(low, high, size=(2000, 5))
    jaws = rng.uniform(0.0, 1.0, size=(2000, 1))  # the full jaw range, floor off
    poses = np.hstack([arms, jaws])

    model = np.vstack([to_model_state(pose) for pose in poses])
    back = to_joint_actions(model)
    assert back[:, :5] == pytest.approx(poses[:, :5], abs=1e-9)
    # And nothing in that sweep hit an MJCF limit, i.e. the involution is real
    # and not an artefact of both sides being clipped to the same rail.
    assert np.all(back[:, :5] > LOW[:5]) and np.all(back[:, :5] < HIGH[:5])


def test_the_gripper_round_trips_up_to_its_two_scales(no_floor) -> None:
    """Channel 5 is not an involution, and neither the floor nor a bug is why.

    The state axis is scaled by the checkpoint's *state* q99 (44.140 deg) and the
    action axis by its *action* q99 (44.746 deg); they are different numbers
    because they are different distributions. So the jaw round trip is exact only
    up to that fixed ratio, which is 1.4%. Pinned so the ratio cannot drift into
    something that is a bug.
    """
    rng = np.random.default_rng(7)
    poses = np.hstack([rng.uniform(-1.0, 1.0, size=(200, 5)), rng.uniform(0, 1, size=(200, 1))])
    model = np.vstack([no_floor.to_model_state(pose) for pose in poses])
    back = no_floor.to_joint_actions(model)
    assert back[:, :5] == pytest.approx(poses[:, :5], abs=1e-9)
    ratio = GRIPPER_STATE_OPEN_DEG / GRIPPER_ACTION_OPEN_DEG
    assert ratio == pytest.approx(0.9865, abs=1e-3)
    assert back[:, 5] == pytest.approx(poses[:, 5] * ratio, abs=1e-9)


def test_home_is_in_band_on_wrist_roll(stats, home) -> None:
    """The first state the model sees every episode, on the recalibrated channel.

    Under the shipped mapping HOME's wrist_roll landed at +90.05 against a q99 of
    +42.94 -- 47.1 deg past the top of the trained band, on the very first
    inference of every run, and ``assert_state_in_distribution`` said so out loud.
    """
    index = stats["state_stats"]["names"].index("wrist_roll")
    q01 = stats["state_stats"]["q01"][index]
    q99 = stats["state_stats"]["q99"][index]
    assert home[4] == pytest.approx(1.62, abs=1e-6), "home is the task's start roll"
    mapped = to_model_state(home)[index]
    assert q01 <= mapped <= q99, f"wrist_roll {mapped:.2f} outside [{q01:.2f}, {q99:.2f}]"
    # It is not merely inside; it sits near the model's zero (-2.8 deg by design, the
    # trained median being -11) rather than near either edge of the band.
    assert abs(mapped) < 5.0
    # And the mapping this replaces would have been out the top, by ~47 deg.
    assert np.degrees(home[4]) - q99 > 45.0


#: REST is all-zeros. It is a convenient constant, NOT the pose an episode starts
#: from -- see the ``home`` fixture. At zero the arm has the jaw fully closed, the
#: wrist flat, and the jaw axis rolled a quarter turn away from level, and the
#: SO-100/101 demonstrations never did any of those. Three channels are therefore
#: out of band there, and wrist_roll is out of band at zero *because* of the
#: corrected calibration, not in spite of it: ours = 0 is the model's +90.
REST_OUT_OF_BAND = {"wrist_flex", "wrist_roll", "gripper"}


def test_rest_pose_is_out_of_band_on_exactly_three_channels(stats) -> None:
    """Measured, not assumed: the all-zeros pose is outside the demonstrated band.

    This is a property of that pose, not a mapping error -- the calibration
    round-trips exactly (see the involution test), and the pose we actually start
    from is in band on wrist_roll (see ``test_home_is_in_band_on_wrist_roll``).
    Pinned here so the deviation stays visible and cannot grow unnoticed.
    """
    with pytest.raises(ValueError, match="silently clipped"):
        assert_state_in_distribution(REST, stats=stats)

    q01 = stats["state_stats"]["q01"]
    q99 = stats["state_stats"]["q99"]
    mapped = to_model_state(REST)
    outside = {
        name: (mapped[i], q01[i], q99[i])
        for i, name in enumerate(stats["state_stats"]["names"])
        if not (q01[i] <= mapped[i] <= q99[i])
    }
    assert set(outside) == REST_OUT_OF_BAND
    # wrist_flex and the jaw fall out the BOTTOM, by under a tenth of a band.
    for name in ("wrist_flex", "gripper"):
        value, low, high = outside[name]
        shortfall = (low - value) / (high - low)
        assert 0.0 < shortfall < 0.10, f"{name} is {shortfall:.1%} below q01"
    # wrist_roll falls out the TOP at zero, and by about half a band. That is the
    # direction the corrected calibration inverts: at our +pi/2 it reads ~0.
    value, low, high = outside["wrist_roll"]
    assert value > high
    assert 0.3 < (value - high) / (high - low) < 0.6


def test_the_three_in_band_channels_stay_in_band(stats) -> None:
    q01 = stats["state_stats"]["q01"]
    q99 = stats["state_stats"]["q99"]
    mapped = to_model_state(REST)
    for i, name in enumerate(stats["state_stats"]["names"]):
        if name in REST_OUT_OF_BAND:
            continue
        assert q01[i] <= mapped[i] <= q99[i], f"{name} {mapped[i]} left the band"


def test_the_old_jaw_mapping_would_have_been_refused(stats) -> None:
    """degrees(1.0) = 57.3 is past the q99 of 44.14 -- defect 2, now caught."""
    assert np.degrees(1.0) > stats["state_stats"]["q99"][5]
    with pytest.raises(ValueError, match="silently clipped"):
        assert_state_in_distribution(np.r_[np.zeros(5), 5.0], stats=stats)


def test_fp16_is_required() -> None:
    from robot_console.arm.molmoact import MolmoAct2Policy

    with pytest.raises(ValueError, match="bfloat16 collapses"):
        MolmoAct2Policy(dtype="bfloat16")


def test_two_distinct_views_are_required() -> None:
    from robot_console.arm.molmoact import MolmoAct2Policy

    with pytest.raises(ValueError, match="two different third-person views"):
        MolmoAct2Policy(views=("overhead",))


def test_the_policy_declares_its_cameras() -> None:
    from robot_console.arm.molmoact import MolmoAct2Policy

    policy = MolmoAct2Policy()
    # The only pair the simulator publishes. Order is load-bearing: the
    # checkpoint consumes views positionally.
    assert policy.info.observation_space.camera_names == frozenset({"overhead", "side"})
    assert [c.name for c in policy.info.observation_space.cameras] == ["overhead", "side"]


def test_a_missing_view_raises_rather_than_going_black() -> None:
    from inspect_robots import Observation

    from robot_console.arm.molmoact import MolmoAct2Policy

    policy = MolmoAct2Policy()
    only_overhead = Observation(images={"overhead": np.zeros((720, 1280, 3), np.uint8)})
    with pytest.raises(KeyError, match="missing camera view"):
        policy._view_images(only_overhead)


# --- Gate A: the crush-aperture floor, now OFF ----------------------------------------
#
# Our jaw is an MJCF ``ctrlrange`` 0..1 POSITION actuator. The floor existed because
# closing below it asked the jaw to close to a gap narrower than the ball, and the
# solver resolved that by ejecting the apple: MolmoAct2 drove the jaw to 0.000 at
# contact and the apple left at 0.862 m/s.
#
# That ejection was the gripper_joint actuator's +-3.35 N.m forcerange driving 18.6 mm
# through a 20 g apple. The simulator role has cut that forcerange to +-0.30 N.m (H3);
# the same full close now peaks at 2.9 N with 2.8 mm of penetration and HOLDS. The
# measured, solver-validated holding window is g in [0.00, 0.35] -- ENTIRELY BELOW the
# old 0.50 floor, which had therefore stopped being a workaround and started being the
# thing forbidding every aperture that works.
#
# These tests now pin the floor OFF, and pin the 0.50 behaviour as the rollback.


def test_the_gripper_floor_is_off() -> None:
    assert GRIPPER_MIN_COMMAND == pytest.approx(EXPECTED_FLOOR, abs=1e-9)
    assert GRIPPER_MIN_COMMAND == 0.0


def test_the_old_floor_sat_entirely_above_the_holding_window() -> None:
    """Why it had to go, as an interval comparison rather than an argument."""
    low, high = HOLDING_WINDOW
    assert high < LEGACY_FLOOR, (
        f"the whole working window [{low}, {high}] is below the old floor {LEGACY_FLOOR}"
    )
    # And the aperture curve is CONCAVE, so a straight line is not a substitute
    # for it: measured gap(g) = -23.24 g^2 + 92.54 g - 0.84 mm, R^2 = 0.99947,
    # against a 40.0 mm apple. The fit and the quoted sample points agree to
    # within 0.71 mm, which is the fit residual, so the tolerance here is 0.8 mm
    # rather than a precision the fit does not have.
    def gap(g: float) -> float:
        return -23.24 * g * g + 92.54 * g - 0.84

    assert gap(LEGACY_FLOOR) == pytest.approx(39.58, abs=0.8)
    assert 40.0 - gap(LEGACY_FLOOR) < 1.0, "the old floor was a knife edge on the apple"
    assert gap(0.25) == pytest.approx(21.55, abs=0.8), "well closed on the apple"
    # NOTE, deliberately not asserted as a mechanism: the window's upper edge at
    # 0.35 is NOT the aperture crossing the apple diameter. gap(0.35) = 29.6 mm
    # and gap(0.40) = 32.5 mm are BOTH narrower than the 40 mm apple, and the
    # curve does not reach 40 mm until g ~= 0.50. So the edge is force-limited,
    # not geometry-limited, and 0.35 is taken from the simulator role's measured
    # sweep rather than derived from this curve.
    assert gap(HOLDING_WINDOW[1]) < 40.0
    assert gap(0.40) < 40.0
    # Concave, so a straight line through the endpoints is nearly right in the
    # middle and wrong either side -- the shape of the claim, with the numbers
    # rederived here rather than quoted.
    assert -23.24 < 0.0, "negative quadratic coefficient == concave"
    def linear(g: float) -> float:
        return gap(0.0) + g * (gap(1.0) - gap(0.0))

    assert abs(linear(0.25) - gap(0.25)) > 4.0
    assert abs(linear(0.75) - gap(0.75)) > 4.0


@pytest.mark.parametrize(
    ("model_degrees", "expected", "why"),
    [
        (-0.30, 0.0000, "q01: the extreme closing command now reaches a full close"),
        (4.87, 0.1088, "q50: over half of all commands, now inside the holding window"),
        (15.65, 0.3498, "the top of the holding window, now reachable"),
        (32.12, 0.7178, "q90: an OPENING command, and it must pass through untouched"),
        (44.75, 1.0000, "q99: fully open, unchanged"),
        (119.41, 1.0000, "max: still clipped to fully open, unchanged"),
    ],
)
def test_gate_a_channel_five_mapping(model_degrees, expected, why) -> None:
    """Closing commands reach the working window; opening commands are untouched."""
    row = np.array([[0.0, 90.0, 90.0, 0.0, 90.0, model_degrees]])
    assert to_joint_actions(row)[0, 5] == pytest.approx(expected, abs=2e-3), why


def test_the_whole_holding_window_is_reachable() -> None:
    """Every aperture that holds the apple must be commandable. It was not.

    Under the 0.50 floor NO command in [0.00, 0.35] was reachable -- the policy
    was structurally forbidden from ever closing enough to hold the apple.
    """
    low, high = HOLDING_WINDOW
    targets = np.linspace(low, high, 15)
    # The model command that asks for each aperture, in the checkpoint's degrees.
    rows = np.zeros((targets.size, 6))
    rows[:, 1] = 90.0
    rows[:, 2] = 90.0
    rows[:, 4] = 90.0
    rows[:, 5] = targets * GRIPPER_ACTION_OPEN_DEG
    got = to_joint_actions(rows)[:, 5]
    assert got == pytest.approx(targets, abs=1e-9)
    assert got.max() <= high + 1e-9 and got.min() >= low - 1e-9


def test_the_old_floor_forbade_the_entire_holding_window(legacy_floor) -> None:
    """The same sweep against the rollback value, so the contrast is pinned."""
    low, high = HOLDING_WINDOW
    targets = np.linspace(low, high, 15)
    rows = np.zeros((targets.size, 6))
    rows[:, 1] = 90.0
    rows[:, 2] = 90.0
    rows[:, 4] = 90.0
    rows[:, 5] = targets * GRIPPER_ACTION_OPEN_DEG
    got = legacy_floor.to_joint_actions(rows)[:, 5]
    assert np.all(got == pytest.approx(LEGACY_FLOOR)), "every one collapsed onto the floor"
    assert LEGACY_FLOOR > high


def test_channel_five_is_now_the_plain_rescale_everywhere() -> None:
    """With the floor off, the whole axis is the untouched degree rescale.

    ``np.maximum(x, 0.0)`` on a value already clipped to [0, 1] is the identity,
    so this asserts the change is exactly a no-op against the reference mapping
    across the full command range -- not merely at a few sampled points.
    """
    degrees = np.linspace(-10.0, 130.0, 400)
    rows = np.zeros((degrees.size, 6))
    rows[:, 1] = 90.0
    rows[:, 2] = 90.0
    rows[:, 4] = 90.0
    rows[:, 5] = degrees
    got = to_joint_actions(rows)[:, 5]
    raw = np.clip(degrees / GRIPPER_ACTION_OPEN_DEG, 0.0, 1.0)
    assert got == pytest.approx(raw, abs=0.0), "identical to the un-floored rescale"
    assert np.all(np.diff(got) >= -1e-12), "must stay monotone in the model's command"
    assert got.min() == pytest.approx(0.0), "a full close is now commandable"


def test_the_legacy_floor_never_reduced_a_command(legacy_floor) -> None:
    """The rollback stays monotone and one-sided: it may only ever OPEN the jaw."""
    degrees = np.linspace(-10.0, 130.0, 400)
    rows = np.zeros((degrees.size, 6))
    rows[:, 1] = 90.0
    rows[:, 2] = 90.0
    rows[:, 4] = 90.0
    rows[:, 5] = degrees
    floored = legacy_floor.to_joint_actions(rows)[:, 5]
    raw = np.clip(degrees / GRIPPER_ACTION_OPEN_DEG, 0.0, 1.0)
    assert np.all(floored >= raw - 1e-12), "the floor must never close the jaw further"
    assert np.all(floored >= LEGACY_FLOOR - 1e-12)
    assert np.all(np.diff(floored) >= -1e-12)
    above = degrees > LEGACY_FLOOR * GRIPPER_ACTION_OPEN_DEG + 1e-6
    assert np.allclose(floored[above], raw[above])


def test_opening_commands_are_untouched_at_either_floor(legacy_floor) -> None:
    """The one thing the floor never changed, pinned across the change itself."""
    rows = np.zeros((3, 6))
    rows[:, 1] = 90.0
    rows[:, 2] = 90.0
    rows[:, 4] = 90.0
    rows[:, 5] = [32.123, 44.746, 119.408]  # q90, q99, max
    expected = [0.7178, 1.0, 1.0]
    assert to_joint_actions(rows)[:, 5] == pytest.approx(expected, abs=2e-3)
    assert legacy_floor.to_joint_actions(rows)[:, 5] == pytest.approx(expected, abs=2e-3)


def test_the_live_mapping_is_the_un_floored_one(monkeypatch) -> None:
    """What used to be the rollback test is now the LIVE path.

    robot_console.arm is not a git repository, so this constant IS the revert path in
    both directions and it is pinned here rather than trusted. Setting it to 0.0
    explicitly must be indistinguishable from the shipped module.
    """
    from robot_console.arm import molmoact

    chunk = np.zeros((3, 6))
    chunk[:, 1] = 90.0
    chunk[:, 2] = 90.0
    chunk[:, 4] = 90.0
    chunk[:, 5] = [0.0, GRIPPER_ACTION_OPEN_DEG / 2.0, GRIPPER_ACTION_OPEN_DEG]
    shipped = to_joint_actions(chunk)[:, 5]
    assert shipped == pytest.approx([0.0, 0.5, 1.0], abs=1e-6)

    monkeypatch.setattr(molmoact, "GRIPPER_MIN_COMMAND", 0.0)
    assert molmoact.to_joint_actions(chunk)[:, 5] == pytest.approx(shipped, abs=0.0)


def test_restoring_the_legacy_floor_is_a_one_constant_change(legacy_floor) -> None:
    """The rollback now runs the other way: 0.0 -> 0.50 restores the old mapping."""
    chunk = np.zeros((3, 6))
    chunk[:, 1] = 90.0
    chunk[:, 2] = 90.0
    chunk[:, 4] = 90.0
    chunk[:, 5] = [0.0, GRIPPER_ACTION_OPEN_DEG / 2.0, GRIPPER_ACTION_OPEN_DEG]
    assert legacy_floor.to_joint_actions(chunk)[:, 5] == pytest.approx(
        [LEGACY_FLOOR, 0.5, 1.0], abs=1e-6
    )


def test_the_floor_leaves_the_five_arm_channels_alone() -> None:
    """Channel 5 only: a floor that perturbed an arm joint would be a real bug."""
    measured = np.array([0.2, -0.4, 0.3, 0.1, -0.5, 0.25])
    model = to_model_state(measured)
    back = to_joint_actions(model.reshape(1, 6))[0]
    assert back[:5] == pytest.approx(measured[:5], abs=1e-9)


def test_the_median_command_now_lands_inside_the_holding_window(stats) -> None:
    """The checkpoint's own median gripper command, against the measured window.

    q50 = 4.867 deg -> 0.109, which under the fixed actuator HOLDS the apple. The
    old floor lifted that same command to 0.500, outside the window entirely --
    over half of everything the model asked for was being overridden.
    """
    low, high = HOLDING_WINDOW
    median_deg = stats["action_stats"]["q50"][5]
    row = np.array([[0.0, 90.0, 90.0, 0.0, 90.0, median_deg]])
    commanded = to_joint_actions(row)[0, 5]
    assert commanded == pytest.approx(median_deg / GRIPPER_ACTION_OPEN_DEG, abs=1e-9)
    assert low <= commanded <= high, f"median command {commanded:.4f} must hold the apple"
    assert not (low <= LEGACY_FLOOR <= high), "the old floor did not"


def test_the_state_map_spans_the_full_trained_gripper_range(stats) -> None:
    """Proprioception: a closed jaw must read at the BOTTOM of the model's band.

    The state axis is a plain rescale of our 0..1 onto the model's 0..44.140 deg
    now that the floor is off, so a full close reports 0.0 -- at or just under the
    checkpoint's own q01 of 0.940, i.e. the bottom of the trained range -- and a
    full open reports q99. Both ends must be reachable or the model never sees a
    closed hand.
    """
    q01 = stats["state_stats"]["q01"][5]
    q50 = stats["state_stats"]["q50"][5]
    q99 = stats["state_stats"]["q99"][5]

    closed = to_model_state(np.r_[np.zeros(5), 0.0])[5]
    assert closed == pytest.approx(0.0, abs=1e-9)
    assert closed < q01, "a full close must sit at the very bottom of the band"
    assert (q01 - closed) / (q99 - q01) < 0.03, "and only just under it, not far off"
    assert to_model_state(np.r_[np.zeros(5), 1.0])[5] == pytest.approx(GRIPPER_STATE_OPEN_DEG)
    assert GRIPPER_STATE_OPEN_DEG == pytest.approx(q99, abs=5e-3)

    # Monotone and injective across the whole command range, 0.0 included.
    grid = np.linspace(0.0, 1.0, 51)
    vals = np.array([to_model_state(np.r_[np.zeros(5), v])[5] for v in grid])
    assert np.all(np.diff(vals) > 0.0), "the axis must be strictly increasing"
    assert vals == pytest.approx(grid * GRIPPER_STATE_OPEN_DEG, abs=1e-9)

    # The bottom of the trained distribution is reachable again: the model's own
    # median state, q50 = 9.24 deg, corresponds to a jaw of about 0.209.
    assert q50 / GRIPPER_STATE_OPEN_DEG == pytest.approx(0.209, abs=5e-3)


def test_the_legacy_floor_collapsed_the_entire_holding_window(monkeypatch, stats) -> None:
    """What the floor cost the OBSERVATION side, pinned as measurement.

    ``to_model_state`` undoes the floor, so at 0.50 every command in [0.00, 0.50]
    -- which contains the ENTIRE measured holding window g in [0.00, 0.35] --
    reported the identical state 0.0 deg. The model could not tell a full close
    from a 0.35 grasp. Measured live: during the one successful grasp the reported
    state sat flat at 22.07 deg for all 139 frames the apple was aloft, i.e. while
    physically holding the apple the model was told it had a moderately open hand.
    """
    from robot_console.arm import molmoact

    low, high = HOLDING_WINDOW
    window = np.linspace(low, high, 15)

    # Live first: the fixture below patches the module global that to_model_state
    # reads, so the two halves cannot be measured with the floor in one state.
    new_states = np.array([to_model_state(np.r_[np.zeros(5), v])[5] for v in window])
    assert len(np.unique(np.round(new_states, 9))) == window.size, "now injective"
    assert new_states[-1] == pytest.approx(high * GRIPPER_STATE_OPEN_DEG, abs=1e-9)

    monkeypatch.setattr(molmoact, "GRIPPER_MIN_COMMAND", LEGACY_FLOOR)
    old_states = np.array([molmoact.to_model_state(np.r_[np.zeros(5), v])[5] for v in window])
    assert np.all(old_states == pytest.approx(0.0, abs=1e-9)), "all 15 collapsed to one value"
    assert len(np.unique(np.round(old_states, 9))) == 1
    # The floored axis reported 22.07 deg at its own floor: the 73rd percentile of
    # the checkpoint's state distribution, i.e. "a moderately open hand".
    assert LEGACY_FLOOR * GRIPPER_STATE_OPEN_DEG == pytest.approx(22.07, abs=1e-2)
    assert LEGACY_FLOOR * GRIPPER_STATE_OPEN_DEG > stats["state_stats"]["q50"][5]


def test_the_rollback_covers_both_directions_of_the_mapping(monkeypatch) -> None:
    """Flipping the constant back must move the state map too, not only actions."""
    from robot_console.arm import molmoact

    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Live: the plain linear rescale, every value distinct.
    live = [to_model_state(np.r_[np.zeros(5), v])[5] for v in grid]
    assert live == pytest.approx([v * GRIPPER_STATE_OPEN_DEG for v in grid], abs=1e-9)

    monkeypatch.setattr(molmoact, "GRIPPER_MIN_COMMAND", LEGACY_FLOOR)
    for v in grid:
        # Rolled back: everything at or below the floor reports fully closed.
        expected = max(v - LEGACY_FLOOR, 0.0) / (1.0 - LEGACY_FLOOR) * GRIPPER_STATE_OPEN_DEG
        assert molmoact.to_model_state(np.r_[np.zeros(5), v])[5] == pytest.approx(
            expected, abs=1e-9
        )


def test_the_viewmode_hook_is_inert_unless_asked(monkeypatch) -> None:
    """The vision-ablation hook must be a strict no-op by default.

    A diagnostic that can silently alter what the model sees is worse than no
    diagnostic, so this pins that an unset (or 'normal') env var returns the very
    same objects, and that an unrecognised value raises rather than guessing.
    """
    from PIL import Image

    from robot_console.arm.molmoact import VIEWMODE_ENV, _apply_viewmode

    views = ("overhead", "side")
    imgs = [Image.new("RGB", (8, 6), (10, 20, 30)), Image.new("RGB", (8, 6), (40, 50, 60))]

    monkeypatch.delenv(VIEWMODE_ENV, raising=False)
    assert _apply_viewmode(imgs, views) is imgs, "unset must be an exact no-op"

    monkeypatch.setenv(VIEWMODE_ENV, "normal")
    assert _apply_viewmode(imgs, views) is imgs

    monkeypatch.setenv(VIEWMODE_ENV, "blind")
    blind = _apply_viewmode(imgs, views)
    assert [im.size for im in blind] == [im.size for im in imgs], "size must be preserved"
    assert all(im.getextrema() == ((0, 0), (0, 0), (0, 0)) for im in blind), "must be black"

    monkeypatch.setenv(VIEWMODE_ENV, "swap")
    swapped = _apply_viewmode(imgs, views)
    assert swapped[0] is imgs[1] and swapped[1] is imgs[0]

    monkeypatch.setenv(VIEWMODE_ENV, "nonsense")
    with pytest.raises(ValueError, match="not recognised"):
        _apply_viewmode(imgs, views)


# --- the clip / rail counter ----------------------------------------------------------
#
# ``to_joint_actions`` ends in ``np.clip(radians, LOW, HIGH)``. The clip is correct --
# commanding past an MJCF limit is not allowed -- but it is SILENT, so a wrong frame
# mapping shows up as a slightly lazy joint rather than as an error. That is precisely
# how the wrist_roll miscalibration survived. The counter makes it loud, and these tests
# pin that it counts the right things and changes no returned value.


def test_the_counter_records_a_low_clip() -> None:
    CLIP_COUNTS.reset()
    # shoulder_pan LOW is -1.9199 rad = -110.0 deg. Ask for 150 deg past it.
    row = np.array([[-260.0, 90.0, 90.0, 0.0, 90.0, 4.867]])
    out = to_joint_actions(row)
    assert out[0, 0] == pytest.approx(LOW[0])
    counts = CLIP_COUNTS.as_dict()
    assert counts["steps"] == 1 and counts["chunks"] == 1
    assert counts["clipped_low"]["shoulder_pan_joint"] == 1
    assert counts["clipped_high"]["shoulder_pan_joint"] == 0
    assert counts["clipped_low_fraction"]["shoulder_pan_joint"] == pytest.approx(1.0)
    assert counts["worst_below_low"]["shoulder_pan_joint"] == pytest.approx(
        abs(np.radians(-260.0)) - abs(LOW[0]), abs=1e-9
    )
    # No other arm channel moved.
    assert sum(counts["clipped_low"][n] for n in JOINT_ORDER) == 1


def test_the_counter_records_the_disclosed_wrist_roll_cost(stats) -> None:
    """The known, accepted cost of this calibration, measured rather than asserted away.

    The model's wrist_roll action q01 is -65.576 deg. Under the corrected mapping
    that is ``-1 * (-65.576 - 90)`` = +155.58 deg = 2.715 rad, past our +2.3 rad
    MJCF limit, so it saturates -- at OUR high limit, because the sign is flipped.
    The fix for this is NOT to widen the limit: the limit is the MJCF's.
    """
    q01 = stats["action_stats"]["q01"][4]
    assert q01 == pytest.approx(-65.576, abs=1e-2)
    CLIP_COUNTS.reset()
    row = np.array([[0.0, 90.0, 90.0, 0.0, q01, 4.867]])
    out = to_joint_actions(row)
    assert out[0, 4] == pytest.approx(HIGH[4])
    counts = CLIP_COUNTS.as_dict()
    assert counts["clipped_high"]["wrist_roll_joint"] == 1
    assert counts["worst_above_high"]["wrist_roll_joint"] == pytest.approx(
        np.radians(155.576) - HIGH[4], abs=1e-4
    )
    # And the model's q99 end is comfortably inside our range, so this is one-sided.
    CLIP_COUNTS.reset()
    q99 = stats["action_stats"]["q99"][4]
    to_joint_actions(np.array([[0.0, 90.0, 90.0, 0.0, q99, 4.867]]))
    assert CLIP_COUNTS.as_dict()["clipped_high"]["wrist_roll_joint"] == 0


def test_the_counter_finds_raw_actions_sitting_on_their_own_rails(stats) -> None:
    """The failure signature from the prior run: normalised output pinned at -1/+1.

    ``shoulder_pan`` froze at -0.7352 rad = -42.13 deg = action q01 to the decimal
    and ``wrist_roll`` at 0.7598 rad = +43.53 deg = action q99 to the decimal.
    That is a statement about the MODEL, not about our joint limits, so it is
    counted separately from the clips.
    """
    q01 = np.asarray(stats["action_stats"]["q01"], dtype=np.float64)
    q99 = np.asarray(stats["action_stats"]["q99"], dtype=np.float64)
    CLIP_COUNTS.reset()
    to_joint_actions(np.vstack([q01, q99, 0.5 * (q01 + q99)]))
    counts = CLIP_COUNTS.as_dict()
    assert counts["action_rails_available"] is True
    for name in JOINT_ORDER:
        assert counts["action_rail_q01"][name] == 1, name
        assert counts["action_rail_q99"][name] == 1, name
    # The recorded shoulder_pan freeze, to the decimal.
    assert np.degrees(-0.7352) == pytest.approx(q01[0], abs=1e-2)
    assert np.degrees(0.7598) == pytest.approx(q99[4], abs=1e-2)


def test_a_midband_action_rails_nothing(stats) -> None:
    """The tolerance is tight on purpose: near the edge is not on the rail."""
    q01 = np.asarray(stats["action_stats"]["q01"], dtype=np.float64)
    q99 = np.asarray(stats["action_stats"]["q99"], dtype=np.float64)
    CLIP_COUNTS.reset()
    # One full tolerance inside each rail must not count.
    inside = np.vstack([q01 + 10 * RAIL_TOLERANCE_DEG, q99 - 10 * RAIL_TOLERANCE_DEG])
    to_joint_actions(inside)
    counts = CLIP_COUNTS.as_dict()
    assert sum(counts["action_rail_q01"].values()) == 0
    assert sum(counts["action_rail_q99"].values()) == 0


def test_the_counter_accumulates_and_resets() -> None:
    CLIP_COUNTS.reset()
    assert CLIP_COUNTS.as_dict()["steps"] == 0
    rows = np.tile([-260.0, 90.0, 90.0, 0.0, 90.0, 4.867], (4, 1))
    to_joint_actions(rows)
    to_joint_actions(rows)
    counts = CLIP_COUNTS.as_dict()
    assert counts["chunks"] == 2 and counts["steps"] == 8
    assert counts["clipped_low"]["shoulder_pan_joint"] == 8
    CLIP_COUNTS.reset()
    assert CLIP_COUNTS.as_dict()["clipped_low"]["shoulder_pan_joint"] == 0
    assert CLIP_COUNTS.as_dict()["steps"] == 0


def test_counting_does_not_change_a_single_returned_value() -> None:
    """``to_joint_actions`` must stay a pure function of its argument."""
    rng = np.random.default_rng(11)
    chunk = np.hstack([
        rng.uniform(-200.0, 200.0, size=(30, 5)),
        rng.uniform(-1.0, 120.0, size=(30, 1)),
    ])
    CLIP_COUNTS.reset()
    first = to_joint_actions(chunk)
    second = to_joint_actions(chunk)  # counter now non-empty; output must not move
    assert first == pytest.approx(second, abs=0.0)
    # And the reference implementation, with no counter at all, agrees exactly.
    expected = np.radians(SIGNS * (chunk - OFFSETS))
    expected[:, 5] = np.maximum(
        np.clip(chunk[:, 5] / GRIPPER_ACTION_OPEN_DEG, 0.0, 1.0), GRIPPER_MIN_COMMAND
    )
    assert first == pytest.approx(np.clip(expected, LOW, HIGH), abs=0.0)


def test_the_counter_writes_json_and_a_summary(tmp_path) -> None:
    import json

    CLIP_COUNTS.reset()
    to_joint_actions(np.array([[-260.0, 90.0, 90.0, 0.0, -65.576, 4.867]]))
    out = CLIP_COUNTS.write_json(tmp_path / "nested" / "clip_counts.json")
    payload = json.loads(out.read_text())
    assert payload["clipped_low"]["shoulder_pan_joint"] == 1
    assert payload["clipped_high"]["wrist_roll_joint"] == 1
    assert payload["joint_order"] == list(JOINT_ORDER)
    assert payload["joint_limits"]["wrist_roll_joint"] == [float(LOW[4]), float(HIGH[4])]
    text = CLIP_COUNTS.summary()
    assert "wrist_roll_joint" in text and "clip@LOW" in text
    assert len(text.splitlines()) == 2 + len(JOINT_ORDER) or not payload[
        "action_rails_available"
    ]


def test_the_eval_script_takes_its_views_from_the_module_default() -> None:
    """The wiring is stated in one place, and the script and policy agree.

    The script used to hard-code ``("overhead", "side")`` while ``DEFAULT_VIEWS``
    and ``RosSettings`` both said ``("trainlow", "trainhigh")``.
    """
    import subprocess
    import sys

    from robot_console.arm.molmoact import DEFAULT_VIEWS
    from robot_console.arm.ros_settings import settings_for_views

    help_text = subprocess.run(
        [sys.executable, "scripts/molmoact_eval.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert ",".join(DEFAULT_VIEWS) in help_text
    assert "--views" in help_text
    # And the wiring derived from those views subscribes to exactly them, in order.
    assert list(settings_for_views(DEFAULT_VIEWS).cameras()) == list(DEFAULT_VIEWS)
