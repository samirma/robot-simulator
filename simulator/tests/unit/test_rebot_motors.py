"""The reBot Arm B601-RS's simulated bus (`wire/robots/sim_libs/motorbridge.py`), the drop-in
`motorbridge` the vendor SDK drives, answers as the RobStride behaviour recorded in
robots_specs/rebot_b601_rs/ros2.yml: motors report only in reply to commands, MIT frames
are clamped and quantised as motorbridge encodes them, a mode change disables the motor
(and only a changed mode restarts a running profile), set_zero moves the zero, canTimeout
is written unscaled, and the gripper motor is the model's tendon in motor rad; a motor not
on the bus takes the frames motorbridge sends without waiting and times out the rest.
Each motor's torque reaches the model through its position actuator."""

import importlib.util
import math
import sys
import types

import pytest

import common
from conftest import SHARED

ARM = [f"joint{i}" for i in range(1, 7)]
KP = {"joint1": 50.0, "joint2": 150.0, "joint3": 150.0, "joint4": 50.0, "joint5": 50.0,
      "joint6": 50.0, "gripper": 50.0}      # the model's actuator gains (the boot's MIT kp)
KD = {"joint1": 3.0, "joint2": 10.0, "joint3": 10.0, "joint4": 5.0, "joint5": 4.0,
      "joint6": 4.0, "gripper": 4.0}        # its joint damping, the gripper's in motor units
LSB = 8 * math.pi / 0x7FFF                    # one 16-bit step of +-4 pi rad


@pytest.fixture
def mb(monkeypatch):
    joints = {j: [0.0, 0.0, 0.0] for j in ARM + ["gripper_joint1", "gripper_joint2"]}
    grav = {j: 0.0 for j in joints}

    class Link:
        sent = {}

        def subscribe(self, stream, rate, callback, **params):
            assert stream == "state" and params.get("gravcomp") and params.get("joints")
            self.callback = callback
            callback({"joints": joints, "gravcomp": grav}, None)
            return 1

        def ctrl(self, values):
            Link.sent = dict(values)

    monkeypatch.setenv("RSIM_ROBOT", "rebot_b601_rs")
    monkeypatch.setattr(common, "SimLink", Link)
    monkeypatch.setattr(common, "exit_when_lost", lambda link, cleanup=None: None)
    path = SHARED / "wire" / "robots" / "sim_libs" / "motorbridge.py"
    spec = importlib.util.spec_from_file_location("motorbridge_under_test", path)
    lib = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, lib)
    spec.loader.exec_module(lib)
    lib.joints, lib.grav, lib.Link = joints, grav, Link
    lib.feed = lambda: lib._Bus.get().link.callback({"joints": joints, "gravcomp": grav}, None)
    return lib


def arm_bus(mb):
    """The SDK's set-up for the RS model (rebotarm_rs.yaml): one controller on can0, motors
    1-7 answering host 0xFD."""
    c = mb.Controller("can0")
    models = ["rs-06"] * 3 + ["rs-00"] * 4
    return c, [c.add_robstride_motor(i + 1, 0xFD, models[i]) for i in range(7)]


def bus_clock(mb):
    """State samples every 4 ms (the bus's 250 Hz) from now on, sleeps instant: the bus's
    last sample is taken as now, so the next one is 4 ms on, whatever real time the calls
    before took (`ensure_mode` sleeps 90 ms)."""
    t0 = mb.time.monotonic()
    clock = iter(t0 + 0.004 * k for k in range(1, 10000))
    mb.time = types.SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda s: None)
    mb._Bus.get().last = t0


def test_the_package_surface_the_sdk_imports(mb):
    assert mb.__version__ == "0.5.6"
    assert [m.value for m in mb.Mode] == [1, 2, 3, 4, 5]
    assert issubclass(mb.CallError, mb.MotorBridgeError) and issubclass(mb.CallError, RuntimeError)
    c, motors = arm_bus(mb)
    with pytest.raises(mb.CallError):
        mb.Controller.from_dm_serial("/dev/ttyACM0", 921600)
    with pytest.raises(mb.CallError):
        c.add_damiao_motor(1, 0x11, "4340P")


def test_motors_report_only_in_reply_to_commands(mb):
    c, motors = arm_bus(mb)
    m = motors[1]
    assert m.get_state() is None                      # nothing sent yet
    mb.joints["joint2"][0] = 0.4
    mb.feed()
    m.request_feedback()                               # a no-op for RobStride
    c.poll_feedback_once()
    assert m.get_state() is None
    assert m.robstride_get_param_f32(0x7019) == pytest.approx(0.4)   # mechPos, read live
    c.enable_all()
    st = m.get_state()
    assert st.pos == pytest.approx(0.4, abs=LSB) and st.status_code == 0 and st.can_id == 2
    mb.joints["joint2"][0] = 0.9                      # it moves; no command, no new frame
    mb.feed()
    assert m.get_state().pos == pytest.approx(0.4, abs=LSB)
    m.send_mit(0.9, 0.0, 150.0, 10.0, 0.0)
    assert m.get_state().pos == pytest.approx(0.9, abs=LSB)


def test_mit_with_the_boot_gains_and_gravity_feed_forward_holds_the_target(mb):
    c, motors = arm_bus(mb)
    c.enable_all()
    mb.grav["joint3"] = 6.76                           # the model's gravity compensation
    mb.joints["joint3"][:2] = [0.30, 0.2]
    motors[2].send_mit(0.55, 0.0, 150.0, 10.0, 6.76)   # the SDK's hold: Pinocchio gravity as tau
    mb.feed()
    assert mb.Link.sent["joint3"] == pytest.approx(0.55, abs=2 * LSB)
    # another gain acts through the sampled state: tau = kp (q_t - q) + kd (0 - dq) + tau_ff
    motors[2].send_mit(0.55, 0.0, 2.0, 1.0, 6.76)
    mb.feed()
    tau = 2.0 * (0.55 - 0.30) - 1.0 * 0.2 + 6.76
    # the model applies kp_model (target - q) + gravcomp - damping dq
    model = KP["joint3"] * (mb.Link.sent["joint3"] - 0.30) + 6.76 - KD["joint3"] * 0.2
    assert model == pytest.approx(tau, abs=0.02)


def test_mit_fields_are_clamped_and_quantised_as_motorbridge_encodes_them(mb):
    c, motors = arm_bus(mb)
    c.enable_all()
    motors[0].send_mit(20.0, 0.0, 1e6, 1e6, 0.0)      # past +-4 pi rad, kp 5000, kd 100 (RS-06)
    hw = mb._Bus.get().motor(1)
    pos, vel, kp, kd, tau = hw.mit
    assert pos == pytest.approx(4 * math.pi, abs=LSB) and kp == pytest.approx(5000.0, rel=1e-3)
    assert kd == pytest.approx(100.0, rel=1e-3)
    motors[3].send_mit(0.0, 0.0, 1e6, 1e6, -100.0)    # RS-00: kp 500, kd 5, torque +-14
    _, _, kp, kd, tau = mb._Bus.get().motor(4).mit
    assert kp == pytest.approx(500.0, rel=1e-3) and kd == pytest.approx(5.0, rel=1e-3)
    assert tau == pytest.approx(-14.0, abs=1e-3)


def test_a_disabled_motor_applies_no_torque(mb):
    c, motors = arm_bus(mb)
    mb.joints["joint4"][0] = 0.1
    mb.feed()
    assert "joint4" not in mb.Link.sent             # before the host enables it: untouched
    c.enable_all()
    mb.feed()
    assert mb.Link.sent["joint4"] == pytest.approx(0.1)   # enabled: holds where it was
    c.disable_all()
    mb.grav["joint4"] = 2.0
    mb.feed()
    # the actuator cancels the model's gravity compensation: kp (target - q) + g = 0
    assert KP["joint4"] * (mb.Link.sent["joint4"] - 0.1) + 2.0 == pytest.approx(0.0, abs=1e-9)
    motors[3].send_mit(1.0, 0.0, 50.0, 5.0, 0.0)     # a disabled motor ignores MIT frames
    mb.feed()
    assert KP["joint4"] * (mb.Link.sent["joint4"] - 0.1) + 2.0 == pytest.approx(0.0, abs=1e-9)


def test_a_mode_change_disables_and_pos_vel_does_not_enable(mb):
    c, motors = arm_bus(mb)
    c.enable_all()
    m = motors[0]
    m.ensure_mode(mb.Mode.MIT, 1000)                  # already MIT: nothing happens
    assert mb._Bus.get().motor(1).enabled
    m.ensure_mode(mb.Mode.POS_VEL, 1000)              # RobStride PP: disabled first
    hw = mb._Bus.get().motor(1)
    assert m.robstride_get_param_i8(0x7005) == 1 and not hw.enabled
    m.send_pos_vel(0.5, 2.0)                           # run_mode PP, limit_spd, loc_ref; no enable
    assert not hw.enabled and hw.params[0x7016] == 0.5 and hw.params[0x7017] == 2.0
    m.enable()
    bus_clock(mb)
    for _ in range(400):                               # 1.6 s at vel_max 10, acc_set 10
        mb.feed()
    assert mb.Link.sent["joint1"] == pytest.approx(0.5, abs=1e-9)


def test_set_zero_moves_the_zero_and_the_gripper_is_the_tendon_in_motor_rad(mb):
    c, motors = arm_bus(mb)
    mb.joints["joint6"][0] = 0.25
    mb.feed()
    motors[5].set_zero_position()
    assert motors[5].get_state().pos == pytest.approx(0.0, abs=LSB)
    assert motors[5].robstride_get_param_u8(0x7029) == 1
    c.enable_all()
    motors[5].send_mit(0.1, 0.0, 50.0, 4.0, 0.0)
    mb.feed()
    assert mb.Link.sent["joint6"] == pytest.approx(0.35, abs=2 * LSB)
    # gripper: motor rad = 62.5 (q1 + q2), the rack-and-pinion's coupled fingers
    mb.joints["gripper_joint1"][0] = mb.joints["gripper_joint2"][0] = 0.02
    mb.feed()
    motors[6].send_mit(5.0, 0.0, 50.0, 4.0, 0.0)
    assert motors[6].get_state().pos == pytest.approx(2.5, abs=LSB)
    mb.feed()
    assert mb.Link.sent["gripper"] == pytest.approx(5.0, abs=2 * LSB)


def test_a_motor_not_on_the_bus_never_answers(mb):
    c = mb.Controller("can0")
    ghost = c.add_robstride_motor(9, 0xFD, "rs-00")
    with pytest.raises(mb.CallError):
        ghost.enable()
    with pytest.raises(mb.CallError):
        ghost.robstride_get_param_f32(0x7019)
    assert ghost.get_state() is None


def test_can_timeout_is_written_unscaled_and_zero_sta_starts_at_zero(mb):
    c, motors = arm_bus(mb)
    m = motors[0]
    # motorbridge's RobStride path writes canTimeout (0x7028) as given (x20 is Damiao's)
    assert m.robstride_get_param_u32(0x7028) == 0
    m.set_can_timeout_ms(50)
    assert m.robstride_get_param_u32(0x7028) == 50
    # zero_sta (0x7029) is 0 at power-up; motorbridge's set zero writes 1
    assert m.robstride_get_param_u8(0x7029) == 0
    m.set_zero_position()
    assert m.robstride_get_param_u8(0x7029) == 1


def test_repeated_pos_vel_keeps_the_running_profile(mb):
    c, motors = arm_bus(mb)
    m = motors[0]
    m.ensure_mode(mb.Mode.POS_VEL, 1000)
    m.enable()
    bus_clock(mb)
    hw = mb._Bus.get().motor(1)
    for _ in range(50):                                # the SDK resends every loop
        m.send_pos_vel(2.0, 10.0)
        mb.feed()
    # accelerating at acc_set 10 rad/s^2 over 50 samples 4 ms apart, not restarted at rest
    # (a restart at every resend would leave one sample's 0.04 rad/s)
    assert hw.speed == pytest.approx(50 * 10.0 * 0.004)
    assert hw.setpoint == pytest.approx(sum(k * 10.0 * 0.004 * 0.004 for k in range(1, 51)))
    speed = hw.speed
    m.send_pos_vel(2.0, 10.0)                          # run_mode PP again: unchanged mode
    assert hw.speed == speed
    m.robstride_write_param_i8(0x7005, 5)              # a new mode (CSP) starts from rest
    assert hw.setpoint is None and hw.speed == 0.0


def test_an_absent_motor_takes_unacknowledged_frames_and_times_out_the_rest(mb):
    c = mb.Controller("can0")
    ghost = c.add_robstride_motor(9, 0xFD, "rs-00")
    # motorbridge sends these without waiting for an answer: they return
    ghost.send_mit(0.1, 0.0, 10.0, 1.0, 0.0)
    ghost.send_pos_vel(0.1, 1.0)
    ghost.send_vel(0.5)
    ghost.robstride_write_param_f32(0x7017, 2.0)
    ghost.set_can_timeout_ms(10)
    ghost.request_feedback()
    assert ghost.robstride_get_fault_report() == (0, 0)
    assert ghost.get_state() is None
    # and waits on these: no answer, a timeout
    for call in (ghost.enable, ghost.disable, ghost.clear_error, ghost.set_zero_position,
                 ghost.store_parameters, ghost.robstride_ping,
                 lambda: ghost.ensure_mode(mb.Mode.POS_VEL, 1000),
                 lambda: ghost.robstride_get_param_i8(0x7005),
                 lambda: ghost.robstride_send_pos_vel_csp(0.1, 1.0)):
        with pytest.raises(mb.CallError):
            call()
    with pytest.raises(mb.CallError, match="type mismatch"):   # rejected before sending
        ghost.robstride_write_param_u8(0x7017, 1)


def test_current_mode_applies_no_torque(mb):
    """run_mode 3 (current), written directly (the vendor code never selects it): the motor
    applies the torque of iq_ref at its power-up 0 -- none -- and runs no position profile
    toward loc_ref; iq_ref itself is stored, not acted on."""
    c, motors = arm_bus(mb)
    m = motors[0]
    m.robstride_write_param_f32(0x7016, 0.5)           # a loc_ref a PP profile would run to
    m.robstride_write_param_i8(0x7005, 3)
    m.enable()
    m.robstride_write_param_f32(0x7006, 2.0)           # iq_ref
    hw = mb._Bus.get().motor(1)
    bus_clock(mb)
    mb.joints["joint1"][1] = 0.4                       # moving, and under gravity
    mb.grav["joint1"] = 1.5
    for _ in range(100):
        mb.feed()
    # the actuator cancels the gravity compensation and the joint damping: no torque
    target = mb.Link.sent["joint1"]
    assert KP["joint1"] * (target - 0.0) + 1.5 - KD["joint1"] * 0.4 == pytest.approx(0.0, abs=1e-9)
    assert hw.setpoint is None and m.robstride_get_param_f32(0x7006) == 2.0


def test_the_bus_and_its_supply_are_the_recorded_ones(mb):
    """The simulated bus's motor map and supply voltage are the robot specification's,
    read from the record here (robots_specs/rebot_b601_rs/ros2.yml: each motor state topic
    names its motor model and CAN id, `boot.hardware` the host id and the DC supply; the
    robot file robots_specs/rebot_b601_rs.md states the same supply)."""
    import re

    import registry

    iface = common.interface(registry.get("rebot_b601_rs"))
    bus = {}
    for t in iface["topics"]:
        m = re.fullmatch(r"/rebotarm/(?:joints/(\w+)|(gripper))/state", t["name"])
        if not m:
            continue
        got = re.search(r"\((?:RobStride )?(RS-\d+), CAN id (\d+)\)", t["notes"])
        assert got, t["notes"]
        bus[int(got.group(2))] = (m.group(1) or "gripper", got.group(1).lower())
    assert bus == mb.BUS
    hw = iface["boot"]["hardware"]
    assert int(re.search(r"host \(feedback\) id (0x[0-9A-Fa-f]+)", hw).group(1), 16) == mb.HOST_ID
    volts = float(re.search(r"Power: DC (\d+(?:\.\d+)?) V", hw).group(1))
    robot_file = (registry.get("rebot_b601_rs").folder_path.parent / "rebot_b601_rs.md").read_text()
    assert float(re.search(r"Supply: DC (\d+(?:\.\d+)?) V", robot_file).group(1)) == volts
    assert mb.VBUS == volts and mb.PARAMS[0x701C] == ("VBUS", "f32", volts)
    # and what each motor answers when the SDK reads its bus voltage
    c, motors = arm_bus(mb)
    assert [m.robstride_get_param_f32(0x701C) for m in motors] == [volts] * 7
