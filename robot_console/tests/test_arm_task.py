"""Offline tests for the arm_task module. No torch, no network, no display."""

import sys

import numpy as np
import pytest

from robot_console.arm_task import (
    DEFAULT_TASK,
    Convention,
    action_to_driver_dict,
    build_parser,
    chunk_warnings,
    clamp_step,
    observation_to_state,
    parse_float_csv,
    parse_ports,
    resolve_task,
    run_episode,
)
from robot_console.so101_driver import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    MOTOR_KEYS,
)


def test_importing_arm_task_does_not_drag_in_torch():
    # The whole point of the arm_task/molmoact split: this module must be importable
    # (and this suite runnable) in a venv that has never seen torch.
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_observation_to_state_reads_motor_keys_in_order():
    obs = {key: float(i) for i, key in enumerate(MOTOR_KEYS)}
    obs["front"] = np.zeros((480, 640, 3), dtype=np.uint8)
    state = observation_to_state(obs)
    assert state.shape == (6,)
    assert list(state) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_action_to_driver_dict_round_trips():
    values = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 60.0])
    mapping = action_to_driver_dict(values)
    assert list(mapping) == list(MOTOR_KEYS)
    assert observation_to_state(mapping).tolist() == values.tolist()


@pytest.mark.parametrize("mode", ["percent", "radians", "fraction"])
def test_convention_round_trips_in_every_gripper_mode(mode):
    conv = Convention(
        joint_offsets=(10.0, 180.0, -90.0, 0.0, 5.0),
        joint_scales=(1.0, -1.0, 1.0, 2.0, 1.0),
        gripper_mode=mode,
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = np.concatenate((rng.uniform(-90, 90, 5), rng.uniform(0, 100, 1)))
        back = conv.action_to_driver(conv.state_to_model(state))
        np.testing.assert_allclose(back, state, atol=1e-9)


def test_convention_radians_mode_matches_driver_gripper_calibration():
    conv = Convention(gripper_mode="radians")
    closed = conv.state_to_model(np.array([0, 0, 0, 0, 0, 0.0]))[5]
    opened = conv.state_to_model(np.array([0, 0, 0, 0, 0, 100.0]))[5]
    assert closed == pytest.approx(GRIPPER_CLOSED_RAD)
    assert opened == pytest.approx(GRIPPER_OPEN_RAD)


def test_convention_rejects_unknown_gripper_mode():
    conv = Convention(gripper_mode="parsecs")
    with pytest.raises(ValueError):
        conv.state_to_model(np.zeros(6))


def test_clamp_step_limits_travel_but_not_small_moves():
    current = np.zeros(6)
    target = np.array([100.0, -100.0, 1.0, 0.0, 0.0, 90.0])
    clamped = clamp_step(current, target, max_delta_deg=8.0, max_delta_gripper=15.0)
    assert clamped.tolist() == [8.0, -8.0, 1.0, 0.0, 0.0, 15.0]


def test_parse_ports():
    assert parse_ports("8000,8001") == [8000, 8001]
    assert parse_ports("9000") == [9000]
    with pytest.raises(SystemExit):
        parse_ports("eight thousand")
    with pytest.raises(SystemExit):
        parse_ports(",")


def test_parse_float_csv():
    assert parse_float_csv("1,2,3,4,5", "--x", 5) == (1.0, 2.0, 3.0, 4.0, 5.0)
    with pytest.raises(SystemExit):
        parse_float_csv("1,2", "--x", 5)
    with pytest.raises(SystemExit):
        parse_float_csv("a,b,c,d,e", "--x", 5)


def test_task_resolution_prefers_flag_then_positional_then_default():
    parser = build_parser()
    assert resolve_task(parser.parse_args([])) == DEFAULT_TASK
    assert resolve_task(parser.parse_args(["push", "the", "bowl"])) == "push the bowl"
    args = parser.parse_args(["ignored", "--task", "grab the apple"])
    assert resolve_task(args) == "grab the apple"


# ---------------------------------------------------------------- chunk_warnings

def _sane_chunk(state, n=10):
    chunk = np.tile(np.asarray(state, dtype=np.float64), (n, 1))
    chunk += np.linspace(0, 2.0, n)[:, None]  # a small, plausible drift
    return chunk


def test_chunk_warnings_quiet_on_a_sane_chunk():
    conv = Convention()
    state = np.array([0, -30, 40, 10, 0, 50.0])
    assert chunk_warnings(_sane_chunk(state), state, conv) == []


def test_chunk_warnings_flags_passthrough_violation():
    conv = Convention()
    state = np.zeros(6)
    chunk = _sane_chunk(state)
    chunk[0, 1] = 120.0  # first step far from the fed state
    warnings = chunk_warnings(chunk, state, conv)
    assert any("passthrough" in w for w in warnings)


def test_chunk_warnings_flags_out_of_limits_and_nan_and_constant():
    conv = Convention()
    state = np.zeros(6)

    out_of_range = _sane_chunk(state)
    out_of_range[:, 0] = 500.0
    assert any("limits" in w for w in chunk_warnings(out_of_range, state, conv))

    with_nan = _sane_chunk(state)
    with_nan[3, 2] = np.nan
    assert any("NaN" in w for w in chunk_warnings(with_nan, state, conv))

    constant = np.tile(state, (5, 1))
    assert any("constant" in w for w in chunk_warnings(constant, state, conv))


def test_chunk_warnings_rejects_bad_shape():
    conv = Convention()
    assert chunk_warnings(np.zeros((10, 7)), np.zeros(6), conv)


# ---------------------------------------------------------------- run_episode

class StubDriver:
    """Records send_action calls and hands back canned observations."""

    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.float64)
        self.sent = []

    def get_observation(self):
        obs = dict(zip(MOTOR_KEYS, self.state))
        obs["front"] = np.zeros((480, 640, 3), dtype=np.uint8)
        return obs

    def send_action(self, action):
        self.sent.append(dict(action))
        self.state = np.array([action[key] for key in MOTOR_KEYS])
        return dict(action)


class StubPolicy:
    """Predicts a fixed absolute chunk and counts calls."""

    def __init__(self, chunk):
        self.chunk = np.asarray(chunk, dtype=np.float64)
        self.calls = 0

    def predict_chunk(self, image, task, state):
        assert image.shape == (480, 640, 3)
        self.calls += 1
        return self.chunk


def test_run_episode_dry_run_sends_nothing():
    state = np.array([0, -30, 40, 10, 0, 50.0])
    driver = StubDriver(state)
    policy = StubPolicy(_sane_chunk(state))
    n = run_episode(policy, driver, "task", Convention(),
                    max_steps=3, dry_run=True, log=lambda _line: None)
    assert n == 3
    assert policy.calls == 3
    assert driver.sent == []


def test_run_episode_executes_clamped_steps():
    state = np.zeros(6)
    chunk = np.tile(np.array([50.0, 0, 0, 0, 0, 0.0]), (10, 1))  # far target
    driver = StubDriver(state)
    policy = StubPolicy(chunk)
    run_episode(policy, driver, "task", Convention(),
                max_steps=1, execute_steps=4, max_delta_deg=8.0,
                log=lambda _line: None)
    assert len(driver.sent) == 4
    # Per-step travel limited to 8 deg: 8, 16, 24, 32.
    first_joint = [s[MOTOR_KEYS[0]] for s in driver.sent]
    assert first_joint == [8.0, 16.0, 24.0, 32.0]


def test_run_episode_respects_the_time_budget():
    state = np.zeros(6)
    driver = StubDriver(state)
    policy = StubPolicy(_sane_chunk(state))
    now = [0.0]

    def clock():
        now[0] += 10.0
        return now[0]

    n = run_episode(policy, driver, "task", Convention(),
                    max_steps=100, seconds=25.0, dry_run=True,
                    log=lambda _line: None, clock=clock)
    # deadline = 10 + 25 = 35; budget checks pass at t=20 and t=30 (two predictions)
    # and fail at t=40.
    assert n == 2
