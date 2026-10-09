"""The reBot Arm B601-RS's CAN bus and RobStride motors, simulated: a drop-in `motorbridge`
(0.5.6, motorbridge/motorbridge c652ce4, the Python package the vendor SDK imports), put
first on the vendor controller's PYTHONPATH, so Seeed's reBotArmController and its SDK
(reBotArm_control_py) run unchanged.

Where the real package drives SocketCAN `can0` through its Rust ABI, this one talks to the
simulation. It provides, with the package's signatures, the part of its Python API that
the vendor code uses -- the SDK (reBotArm_control_py 6415d43, `actuator/rebotarm.py`;
the controller reaches the motors only through it): `Controller` (`Controller(channel)`,
`from_dm_serial` (refused: no serial device), `add_robstride_motor` (the other vendors'
`add_*_motor` refused), `enable_all`, `disable_all`, `poll_feedback_once`, `shutdown`,
`close`), `Motor` (`ensure_mode`, `send_mit`, `send_pos_vel`, `send_vel`,
`request_feedback`, `get_state`, `set_zero_position`, `robstride_get_param_f32`,
`robstride_write_param_f32`), `Mode` and `CallError` -- plus the package's other RobStride
motor calls (`enable`, `disable`, `clear_error`, the other `robstride_*` parameter reads
and writes, ping, fault report, PP/CSP sends, `set_can_timeout_ms`, `store_parameters`;
`send_force_pos` refused, as motorbridge refuses it for RobStride), the other
`Controller` constructors (`from_socketcanfd` on the same bus; `from_mcu_serial` and
`from_dm_device` refused: no such device), `MotorState` and the `MotorBridgeError`
family. `robstride_set_active_report` is there but always raises `CallError` (active
reporting is not simulated), where motorbridge sends the frame without waiting and
returns. It does not provide `Motor.robstride_set_device_id` (a motor's CAN id is the
robot's), the package's Damiao register access (`write_register_*`, `get_register_*`,
`damiao_*`, `RegisterSpec`, `get_damiao_register_spec`), `ensure_dm_device_runtime`,
`abi_version` / `abi_capabilities`, its `RID_*` / `MODE_*` / `DAMIAO_*` exports, or its
submodules (the ABI loader, CLI and gateway).
It reproduces the RobStride behaviour the robot specification records
(robots_specs/rebot_b601_rs/ros2.yml) from motorbridge's RobStride driver:

* the bus: seven RobStride motors answering host id 0xFD, CAN ids 1-3 RS-06 (joint1-joint3),
  4-6 RS-00 (joint4-joint6) and 7 RS-00 (the gripper), as the robot file records. A motor
  id not on the bus never answers: the calls motorbridge waits on (enable, disable,
  clear_error, set zero, ensure_mode, parameter reads, ping, store_parameters) time out;
  the frames it sends without waiting (MIT, pos_vel, velocity and parameter writes, with
  MOTORBRIDGE_ROBSTRIDE_WRITE_ACK_TIMEOUT_MS unset as the boot leaves it) go unanswered
  and return.
* feedback: a motor sends a status frame only in reply to a command (operation control,
  enable, disable, set zero, a parameter write); `request_feedback` is a no-op, as in
  motorbridge, so while no command is sent the state repeats the last frame. A status
  frame carries position, velocity and torque in the motor's own 16-bit ranges
  (+-4 pi rad, +-50 / +-33 rad/s, +-36 / +-14 N m for RS-06 / RS-00) and the fault bits
  (all clear: no undervoltage, overcurrent, overtemperature, encoder fault, stall or
  calibration fault is simulated); `MotorState.status_code` packs them as motorbridge
  does. A parameter read answers with the live value (mechPos 0x7019, mechVel 0x701B) or
  the stored one.
* modes (run_mode 0x7005): MIT (0, at power-up), PP position (1), velocity (2) and CSP
  position (5). Current mode (3), which neither the SDK nor motorbridge's `ensure_mode`
  selects, is not simulated: written directly, it leaves the motor applying no torque (the
  torque of iq_ref 0x7006 at its power-up 0) whatever iq_ref holds -- no pinned source
  gives the motors' torque constant -- and so does a run_mode RobStride does not define.
  `ensure_mode` changes the mode as motorbridge does: when it differs it
  disables the motor first, writes run_mode and reads it back, leaving the motor
  disabled. `send_pos_vel` is motorbridge's RobStride mapping: run_mode PP, limit_spd
  (0x7017), loc_ref (0x7016), no enable.
* MIT (operation control): the frame's fields are clamped and quantised as motorbridge
  encodes them (position +-4 pi rad, velocity +-vmax, kp 0-5000 / 0-500, kd 0-100 / 0-5,
  torque +-tmax; 16 bits each) and the motor applies tau = kp (pos - q) + kd (vel - dq) +
  tau_ff until the next frame. Nothing times it out: motorbridge never writes canTimeout
  (0x7028, default 0, off). `set_can_timeout_ms` stores its value in 0x7028 unscaled, as
  motorbridge's RobStride path does, but the timeout is not simulated: neither the SDK nor
  the controller calls it, and the boot leaves the register at 0 (ros2.yml, watchdog).
* set zero: the current position becomes the motor's zero (positions are reported and
  commanded from it), and zero_sta (0x7029, 0 at power-up) becomes 1, as motorbridge
  writes it after zeroing.

The motors drive the simulated robot (robots_specs/rebot_b601_rs/model.xml) through its
position actuators -- joint1..joint6 and `gripper`, the latter on the gripper tendon in
motor rad -- whose gains are the boot's MIT gains (actuator kp, joint damping kd) and which
apply the model's gravity compensation. Each actuator target is set, at every state sample
(RATE), so that the actuator's force plus that compensation minus the joint damping is the
motor's torque: target = q + (tau - g + kd_model dq) / kp_model, with q, dq and g (the
model's gravity compensation at the joint) from the sample. With the boot's own gains and
its gravity feed-forward the target is the commanded position exactly; other gains (the
gravity compensation mode's kp 2, kd 1) and a disabled motor (no torque) act through the
sampled state. Reported torque is the actuator force less the damping.

The target is clamped to the actuator's `ctrlrange`, the URDF joint range, which the
robot specification keeps as the joint stop (robots_specs/rebot_b601_rs/import.md, D5).
While a joint is within |tau - g + kd_model dq| / kp_model of a URDF limit and the target
lies beyond it, the target is held at the limit, and the joint receives, and the status
frame reports, kp_model (limit - q) + g - kd_model dq instead of the motor's torque: a
disabled motor with joint2 or joint3 at 0, their lower limit (the `home` pose), still
carries the model's gravity compensation there (1.54 and 6.76 N m) and reports it as its
torque, where the robot's disabled motors apply none and the arm rests on its own body.
The robot specification records this in D5 and among its estimates (import.md,
"Estimates (not manufacturer data)": torque at a joint limit).

Estimates (the pinned sources state none; the robot specification's, in
robots_specs/rebot_b601_rs/import.md, "Estimates (not manufacturer data)"):
* a disabled motor keeps its damping (the model's joint damping) as RobStride's power-off
  back-drive damping (damper 0x702A, on by default), whose magnitude is unpublished;
* until the host first enables a motor it holds the robot's spawn pose (the folded zero
  pose rests on the arm's own body, which the model's collision filtering leaves out:
  limp, its wrist would sag), and a motor enabled in MIT mode holds where it was enabled
  until its first MIT frame;
* PP, CSP and velocity modes follow their profile (PP: vel_max 0x7024 and acc_set 0x7025,
  defaults 10 rad/s and 10 rad/s^2; CSP: limit_spd 0x7017, default 10 rad/s; velocity:
  spd_ref 0x700A with acc_rad 0x7022, default 20 rad/s^2) with the model's position servo
  in place of the motor's own loc_kp / spd_kp / spd_ki loops; a run_mode write that
  changes the mode starts the new profile from rest where the motor is, and rewriting the
  current mode (`send_pos_vel` writes run_mode PP on every call) leaves a running profile
  as it is;
* a motor's temperature reads 30.0 deg C and the bus voltage 48.0 V (the recorded supply);
* motor and bus timing: a reply is at once; the state is sampled, and the actuator
  targets updated, at RATE (250) Hz, half the SDK's 500 Hz control loop, so every second
  status frame repeats the previous sample and the feedback the SDK reads is up to about
  4 ms old.

No CAN interface exists in the simulation, so the boot's `ip link` steps are not run; the
channel name is accepted as given.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

_WIRE = Path(__file__).resolve().parents[2]
for _p in (str(_WIRE), str(_WIRE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402

VERSION = "0.5.6"
__version__ = VERSION

#: state samples from the simulation per second, and so actuator target updates
RATE = 250.0
#: the robot's bus (robots_specs/rebot_b601_rs.md): CAN id -> (model actuator, motor model)
BUS = {1: ("joint1", "rs-06"), 2: ("joint2", "rs-06"), 3: ("joint3", "rs-06"),
       4: ("joint4", "rs-00"), 5: ("joint5", "rs-00"), 6: ("joint6", "rs-00"),
       7: ("gripper", "rs-00")}
HOST_ID = 0xFD
#: motorbridge's RobStride catalogue: model -> (pmax rad, vmax rad/s, tmax N m, kp max, kd max)
MODELS = {"rs-00": (4 * math.pi, 33.0, 14.0, 500.0, 5.0),
          "rs-01": (4 * math.pi, 44.0, 17.0, 500.0, 5.0),
          "rs-02": (4 * math.pi, 44.0, 17.0, 500.0, 5.0),
          "rs-03": (4 * math.pi, 20.0, 60.0, 5000.0, 100.0),
          "rs-04": (4 * math.pi, 15.0, 120.0, 5000.0, 100.0),
          "rs-05": (4 * math.pi, 50.0, 5.5, 500.0, 5.0),
          "rs-06": (4 * math.pi, 50.0, 36.0, 5000.0, 100.0)}
TEMPERATURE_C = 30.0
VBUS = 48.0

# RobStride run_mode values simulated (3, current, is not), and motorbridge's unified mode
# numbers mapped onto them
RUN_MIT, RUN_PP, RUN_VEL, RUN_CSP = 0, 1, 2, 5
_UNIFIED = {1: RUN_MIT, 2: RUN_PP, 3: RUN_VEL, 5: RUN_CSP}

#: RobStride's runtime parameters (motorbridge registers.rs): id -> (name, type, default)
PARAMS = {
    0x7005: ("run_mode", "i8", 0), 0x7006: ("iq_ref", "f32", 0.0),
    0x700A: ("spd_ref", "f32", 0.0), 0x700B: ("limit_torque", "f32", None),
    0x7010: ("cur_kp", "f32", 0.17), 0x7011: ("cur_ki", "f32", 0.012),
    0x7014: ("cur_filter_gain", "f32", 0.1), 0x7016: ("loc_ref", "f32", 0.0),
    0x7017: ("limit_spd", "f32", 10.0), 0x7018: ("limit_cur", "f32", 23.0),
    0x7019: ("mechPos", "f32", None), 0x701A: ("iqf", "f32", 0.0),
    0x701B: ("mechVel", "f32", None), 0x701C: ("VBUS", "f32", VBUS),
    0x701E: ("loc_kp", "f32", 60.0), 0x701F: ("spd_kp", "f32", 6.0),
    0x7020: ("spd_ki", "f32", 0.02), 0x7021: ("spd_filter_gain", "f32", 0.1),
    0x7022: ("acc_rad", "f32", 20.0), 0x7024: ("vel_max", "f32", 10.0),
    0x7025: ("acc_set", "f32", 10.0), 0x7026: ("EPScan_time", "u16", 1),
    0x7028: ("canTimeout", "u32", 0), 0x7029: ("zero_sta", "u8", 0),
    0x702A: ("damper", "u8", 0), 0x702B: ("add_offset", "f32", 0.0),
    0x702C: ("alveolous_open", "u8", 0), 0x702D: ("iq_test", "u8", 0),
    0x702E: ("dcc_set", "f32", 10.0),
}


# ---------------------------------------------------------------- the package's types


class MotorBridgeError(RuntimeError):
    """Base error for motorbridge Python SDK."""


class AbiLoadError(MotorBridgeError):
    """Raised when libmotor_abi cannot be found or loaded."""


class CallError(MotorBridgeError):
    """Raised when ABI call returns non-zero status."""


class Mode(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    FORCE_POS = 4
    ROBSTRIDE_POS_VEL_CSP = 5


@dataclass(frozen=True)
class MotorState:
    can_id: int
    arbitration_id: int
    status_code: int
    pos: float
    vel: float
    torq: float
    t_mos: float
    t_rotor: float


def get_version() -> str:
    return VERSION


# ---------------------------------------------------------------- 16-bit fields


def _u16(x: float, lo: float, hi: float, bits: int = 16) -> int:
    """motorbridge's encoding of a field in [lo, hi] (truncating, as its `as u16`)."""
    x = min(max(float(x), lo), hi)
    full = (1 << bits) - 1
    if lo == 0.0:
        return int(x / hi * full)
    return int((x / hi + 1.0) * 0x7FFF)


def _from_u16(u: int, lo: float, hi: float) -> float:
    if lo == 0.0:
        return u / 0xFFFF * hi
    return (u / 0x7FFF - 1.0) * hi


# ---------------------------------------------------------------- the simulated hardware


def _model_facts(robot) -> dict:
    """Per position actuator of the robot's model: its transmission (a joint, or a tendon's
    joints and coefficients), kp, the damping it sees in its own units, and whether the
    joint's gravity compensation is an actuator force."""
    root = ET.parse(robot.folder_path / "model.xml").getroot()
    joints = {j.get("name"): j for j in root.iter("joint")}
    tendons = {t.get("name"): [(c.get("joint"), float(c.get("coef", "1")))
                               for c in t.iter("joint")] for t in root.iter("fixed")}
    out = {}
    for a in root.iter("position"):
        kp = float(a.get("kp", "1"))
        if a.get("joint"):
            j = joints[a.get("joint")]
            parts = [(a.get("joint"), 1.0)]
            damping = float(j.get("damping", "0"))
            gravcomp = (j.get("actuatorgravcomp") or "false").lower() == "true"
        else:
            # a tendon over joints the model couples to move together (the gripper's two
            # racks): its length is sum(c q), its damping sum(d) / sum(c)^2
            parts = tendons[a.get("tendon")]
            damping = sum(float(joints[n].get("damping", "0")) for n, _ in parts) / \
                sum(c for _, c in parts) ** 2
            gravcomp = False
        out[a.get("name")] = {"parts": parts, "kp": kp, "damping": damping,
                              "gravcomp": gravcomp}
    return out


class _HardwareMotor:
    """One motor on the bus: its mode, enable state, last command and zero."""

    def __init__(self, can_id: int, actuator: str, model: str, facts: dict):
        self.can_id = can_id
        self.actuator = actuator
        self.model = model
        self.pmax, self.vmax, self.tmax, self.kp_max, self.kd_max = MODELS[model]
        self.kp_m = facts["kp"]
        self.d_m = facts["damping"]
        self.parts = facts["parts"]
        self.gravcomp = facts["gravcomp"]
        self.enabled = False
        self.ever_enabled = False  # before its first enable it holds the spawn pose
        self.hold = 0.0            # where it was enabled (physical)
        self.params = {pid: v for pid, (_n, _t, v) in PARAMS.items() if v is not None}
        self.params[0x700B] = self.tmax
        self.offset = 0.0
        self.mit = None            # (pos, vel, kp, kd, tau), motor frame
        self.setpoint = None       # profile setpoint of PP / CSP / velocity, physical
        self.speed = 0.0           # profile velocity
        self.handles = []          # host handles that receive this motor's frames

    # -- the physical joint (from the latest sample)
    def physical(self, sample) -> tuple:
        """(position, velocity, actuator effort, gravity compensation carried by the
        actuator) in the actuator's own units."""
        joints = sample.get("joints") or {}
        q = sum(c * joints[n][0] for n, c in self.parts if n in joints)
        dq = sum(c * joints[n][1] for n, c in self.parts if n in joints)
        name, c0 = self.parts[0]
        effort = joints[name][2] / c0 if name in joints else 0.0
        g = 0.0
        if self.gravcomp:
            g = float((sample.get("gravcomp") or {}).get(name, 0.0))
        return q, dq, effort, g

    def frame(self, sample) -> tuple:
        """The status frame this motor would send now: 16-bit position, velocity, torque."""
        q, dq, effort, _g = self.physical(sample)
        torque = effort - self.d_m * dq
        return (_u16(q - self.offset, -self.pmax, self.pmax),
                _u16(dq, -self.vmax, self.vmax),
                _u16(torque, -self.tmax, self.tmax),
                int(round(TEMPERATURE_C * 10)))

    # -- the actuator target that makes the model apply this motor's torque
    def target(self, sample, dt: float) -> float:
        q, dq, _e, g = self.physical(sample)
        mode = self.params[0x7005]
        if not self.enabled:
            return q - g / self.kp_m
        if mode not in (RUN_MIT, RUN_PP, RUN_VEL, RUN_CSP):
            # current mode (or a run_mode RobStride does not define): iq_ref's torque is
            # not simulated, the motor applies none -- as at iq_ref's power-up 0
            return q + (-g + self.d_m * dq) / self.kp_m
        if mode == RUN_MIT:
            if self.mit is None:
                return self.hold
            pos, vel, kp, kd, tau = self.mit
            qt = pos + self.offset
            return qt + ((kp - self.kp_m) * (qt - q) + kd * vel - (kd - self.d_m) * dq
                         + tau - g) / self.kp_m
        if self.setpoint is None:
            self.setpoint, self.speed = q, 0.0
        if mode == RUN_VEL:
            want = float(self.params[0x700A])
            acc = abs(float(self.params[0x7022])) or 20.0
            self.speed += max(-acc * dt, min(acc * dt, want - self.speed))
            self.setpoint += self.speed * dt
            return self.setpoint
        goal = float(self.params[0x7016]) + self.offset
        err = goal - self.setpoint
        if mode == RUN_CSP:
            vmax = abs(float(self.params[0x7017])) or 10.0
            self.setpoint += max(-vmax * dt, min(vmax * dt, err))
            return self.setpoint
        # PP: a trapezoidal profile at vel_max, acc_set
        vmax = abs(float(self.params[0x7024])) or 10.0
        acc = abs(float(self.params[0x7025])) or 10.0
        v_want = math.copysign(min(vmax, math.sqrt(2 * acc * abs(err))), err)
        self.speed += max(-acc * dt, min(acc * dt, v_want - self.speed))
        step = self.speed * dt
        if abs(step) >= abs(err):
            self.setpoint, self.speed = goal, 0.0
        else:
            self.setpoint += step
        return self.setpoint


class _Bus:
    """The CAN bus with the robot's motors, and the link to the simulation."""

    _instance = None
    _ilock = threading.Lock()

    @classmethod
    def get(cls) -> "_Bus":
        with cls._ilock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.lock = threading.RLock()
        facts = _model_facts(common.owner())
        self.motors = {cid: _HardwareMotor(cid, act, model, facts[act])
                       for cid, (act, model) in BUS.items()}
        self.sample = None
        self.have = threading.Event()
        self.last = None
        self.link = common.SimLink()
        common.exit_when_lost(self.link)
        self.link.subscribe("state", RATE, self._on_state, joints=True, gravcomp=True)
        if not self.have.wait(10):
            raise CallError("new_socketcan failed: the simulated bus sent no state")

    def _on_state(self, header, _payload):
        now = time.monotonic()
        with self.lock:
            self.sample = header
            dt = 0.0 if self.last is None else min(now - self.last, 0.05)
            self.last = now
            targets = {m.actuator: m.target(header, dt) for m in self.motors.values()
                       if m.ever_enabled}
        self.have.set()
        self.link.ctrl(targets)

    def motor(self, can_id: int):
        return self.motors.get(int(can_id))

    def reply(self, m: _HardwareMotor) -> None:
        """A status frame from motor `m`, received by every host handle of its id."""
        frame = m.frame(self.sample)
        for h in list(m.handles):
            h._frame = frame


# ---------------------------------------------------------------- the package's API


def _ok_timeout(what: str):
    raise CallError(f"{what} failed: control ack timeout: no RobStride motor answers")


class Motor:
    def __init__(self, bus: _Bus, motor_id: int, feedback_id: int, model: str,
                 vendor: str = "robstride"):
        self._bus = bus
        self._open = True
        self.motor_id = int(motor_id)
        self.feedback_id = int(feedback_id)
        self.vendor = vendor
        self._limits = MODELS[model]
        self._hw = bus.motor(motor_id)
        self._frame = None
        if self._hw is not None:
            self._hw.handles.append(self)

    def _require_open(self):
        if not self._open:
            raise CallError("motor handle is closed")

    def _hardware(self, what: str) -> _HardwareMotor:
        """The motor, for a call motorbridge waits on: none on the bus times out."""
        self._require_open()
        if self._hw is None:
            _ok_timeout(what)
        return self._hw

    def _unanswered(self) -> bool:
        """For a frame motorbridge sends without waiting: True when no motor is there to
        receive it (the send returns, as on the bus)."""
        self._require_open()
        return self._hw is None

    def close(self) -> None:
        if self._open and self._hw is not None and self in self._hw.handles:
            self._hw.handles.remove(self)
        self._open = False

    # -- lifecycle
    def enable(self) -> None:
        with self._bus.lock:
            hw = self._hardware("enable")
            if not hw.enabled:
                hw.enabled = hw.ever_enabled = True
                hw.hold = hw.physical(self._bus.sample)[0]
                hw.mit, hw.setpoint, hw.speed = None, None, 0.0
            self._bus.reply(hw)

    def disable(self) -> None:
        with self._bus.lock:
            hw = self._hardware("disable")
            hw.enabled = False
            hw.mit, hw.setpoint, hw.speed = None, None, 0.0
            self._bus.reply(hw)

    def clear_error(self) -> None:
        self.disable()

    def set_zero_position(self) -> None:
        with self._bus.lock:
            hw = self._hardware("set_zero_position")
            q, _dq, _e, _g = hw.physical(self._bus.sample)
            hw.offset = q
            hw.params[0x7029] = 1
            self._bus.reply(hw)

    def ensure_mode(self, mode, timeout_ms: int = 1000) -> None:
        mode = int(mode)
        if mode not in _UNIFIED:
            raise CallError("ensure_mode failed: RobStride mode must be 1(MIT) / "
                            "2(POSITION-PP) / 3(VELOCITY) / 5(POSITION-CSP)")
        want = _UNIFIED[mode]
        hw = self._hardware("ensure_mode")
        if hw.params[0x7005] == want:
            return
        # motorbridge disables first (some firmware ignores run_mode while enabled)
        self.disable()
        time.sleep(0.06)
        self._write(0x7005, want, "i8")
        time.sleep(0.03)

    # -- commands
    def send_mit(self, pos: float, vel: float, kp: float, kd: float, tau: float) -> None:
        pmax, vmax, tmax, kp_max, kd_max = self._limits
        raw = (_u16(pos, -pmax, pmax), _u16(vel, -vmax, vmax), _u16(kp, 0.0, kp_max),
               _u16(kd, 0.0, kd_max), _u16(tau, -tmax, tmax))
        if self._unanswered():
            return
        with self._bus.lock:
            hw = self._hw
            if hw.enabled and hw.params[0x7005] == RUN_MIT:
                # the motor decodes the frame in its own ranges
                hw.mit = (_from_u16(raw[0], -hw.pmax, hw.pmax),
                          _from_u16(raw[1], -hw.vmax, hw.vmax),
                          _from_u16(raw[2], 0.0, hw.kp_max), _from_u16(raw[3], 0.0, hw.kd_max),
                          _from_u16(raw[4], -hw.tmax, hw.tmax))
            self._bus.reply(hw)

    def send_pos_vel(self, pos: float, vlim: float) -> None:
        self._write(0x7005, RUN_PP, "i8")
        v = abs(float(vlim))
        if math.isfinite(v) and v > 0.0:
            self._write(0x7017, v, "f32")
        self._write(0x7016, float(pos), "f32")

    def robstride_send_pos_vel_csp(self, pos: float, vlim: float) -> None:
        self._write(0x7005, RUN_CSP, "i8")
        self.enable()
        v = abs(float(vlim))
        if math.isfinite(v) and v > 0.0:
            self._write(0x7017, v, "f32")
        self._write(0x7016, float(pos), "f32")

    def robstride_send_pos_vel_pp(self, pos: float, vel_max: float, acc_set: float) -> None:
        self._write(0x7005, RUN_PP, "i8")
        self.enable()
        for pid, v in ((0x7024, vel_max), (0x7025, acc_set)):
            v = abs(float(v))
            if math.isfinite(v) and v > 0.0:
                self._write(pid, v, "f32")
        self._write(0x7016, float(pos), "f32")

    def send_vel(self, vel: float) -> None:
        self._write(0x700A, float(vel), "f32")

    def send_force_pos(self, pos: float, vlim: float, ratio: float) -> None:
        raise CallError("send_force_pos failed: send_force_pos is not supported for RobStride")

    def request_feedback(self) -> None:
        # RobStride has no single-shot status request: a no-op, as in motorbridge
        self._require_open()

    def set_can_timeout_ms(self, timeout_ms: int) -> None:
        # motorbridge's RobStride path writes the value unscaled (its x20 is Damiao's);
        # stored, not enforced: the canTimeout is not simulated
        self._write(0x7028, int(timeout_ms), "u32")

    def store_parameters(self) -> None:
        self._hardware("store_parameters")

    def get_state(self) -> MotorState | None:
        self._require_open()
        f = self._frame
        if f is None:
            return None
        pmax, vmax, tmax, _kp, _kd = self._limits
        return MotorState(can_id=self.motor_id,
                          arbitration_id=(2 << 24) | (self.motor_id << 8) | self.feedback_id,
                          status_code=0, pos=_from_u16(f[0], -pmax, pmax),
                          vel=_from_u16(f[1], -vmax, vmax), torq=_from_u16(f[2], -tmax, tmax),
                          t_mos=f[3] * 0.1, t_rotor=0.0)

    # -- RobStride parameters
    def _write(self, pid: int, value, kind: str) -> None:
        info = PARAMS.get(int(pid))
        what = f"robstride_write_param_{kind}"
        if info is None:
            raise CallError(f"{what} failed: unknown RobStride parameter 0x{int(pid):04X}")
        if info[1] != kind:
            raise CallError(f"{what} failed: type mismatch for parameter 0x{int(pid):04X}")
        if self._unanswered():
            return
        with self._bus.lock:
            hw = self._hw
            if info[0] in ("mechPos", "mechVel", "VBUS", "iqf"):
                self._bus.reply(hw)          # read-only: the motor ignores it
                return
            if int(pid) == 0x7005 and hw.params[0x7005] != value:
                hw.setpoint, hw.speed = None, 0.0    # a new mode starts from rest
            hw.params[int(pid)] = value
            self._bus.reply(hw)

    def _read(self, pid: int, kind: str):
        info = PARAMS.get(int(pid))
        what = f"robstride_get_param_{kind}"
        with self._bus.lock:
            hw = self._hardware(what)
            if info is None:
                raise CallError(f"{what} failed: parameter 0x{int(pid):04X} not received")
            if info[1] != kind:
                raise CallError(f"{what} failed: parameter 0x{int(pid):04X} is not {kind}")
            q, dq, _e, _g = hw.physical(self._bus.sample)
            if info[0] == "mechPos":
                return float(q - hw.offset)
            if info[0] == "mechVel":
                return float(dq)
            return hw.params.get(int(pid), 0)

    def robstride_write_param_i8(self, param_id: int, value: int) -> None:
        self._write(param_id, int(value), "i8")

    def robstride_write_param_u8(self, param_id: int, value: int) -> None:
        self._write(param_id, int(value), "u8")

    def robstride_write_param_u16(self, param_id: int, value: int) -> None:
        self._write(param_id, int(value), "u16")

    def robstride_write_param_u32(self, param_id: int, value: int) -> None:
        self._write(param_id, int(value), "u32")

    def robstride_write_param_f32(self, param_id: int, value: float) -> None:
        self._write(param_id, float(value), "f32")

    def robstride_get_param_i8(self, param_id: int, timeout_ms: int = 1000) -> int:
        return int(self._read(param_id, "i8"))

    def robstride_get_param_u8(self, param_id: int, timeout_ms: int = 1000) -> int:
        return int(self._read(param_id, "u8"))

    def robstride_get_param_u16(self, param_id: int, timeout_ms: int = 1000) -> int:
        return int(self._read(param_id, "u16"))

    def robstride_get_param_u32(self, param_id: int, timeout_ms: int = 1000) -> int:
        return int(self._read(param_id, "u32"))

    def robstride_get_param_f32(self, param_id: int, timeout_ms: int = 1000) -> float:
        return float(self._read(param_id, "f32"))

    def robstride_get_param_f32_host_id(self, param_id: int, host_id: int,
                                        timeout_ms: int = 1000) -> float:
        return self.robstride_get_param_f32(param_id, timeout_ms)

    def robstride_ping(self) -> tuple[int, int]:
        self._hardware("robstride_ping")
        return self.motor_id, HOST_ID

    def robstride_ping_host_id(self, host_id: int, timeout_ms: int = 500) -> tuple[int, int]:
        self._hardware("robstride_ping_host_id")
        return self.motor_id, int(host_id)

    def robstride_get_fault_report(self) -> tuple[int, int]:
        # motorbridge reads its cached report, sending nothing: none, as no fault is simulated
        self._require_open()
        return 0, 0

    def robstride_set_active_report(self, enabled: bool) -> None:
        # active reporting is not simulated: the motors answer commands only
        raise CallError("robstride_set_active_report failed: not simulated")


class Controller:
    """A SocketCAN controller on the simulated bus."""

    def __init__(self, channel: str = "can0") -> None:
        self.channel = channel
        self._bus = _Bus.get()
        self._motors: list[Motor] = []
        self._open = True

    @classmethod
    def from_socketcanfd(cls, channel: str = "can0") -> "Controller":
        return cls(channel)

    @classmethod
    def from_dm_serial(cls, serial_port: str = "/dev/ttyACM0", baud: int = 921600):
        raise CallError(f"new_dm_serial failed: no serial device {serial_port} (the "
                        "simulated reBot's motors are on CAN)")

    @classmethod
    def from_mcu_serial(cls, serial_port: str = "/dev/ttyUSB0", baud: int = 921600):
        raise CallError(f"new_mcu_serial failed: no serial device {serial_port}")

    @classmethod
    def from_dm_device(cls, dm_device_type: str = "usb2canfd-dual", dm_channel: str = "0"):
        raise CallError("new_dm_device failed: no Damiao USB device")

    def _require_open(self):
        if not self._open:
            raise CallError("controller is closed")

    def close(self) -> None:
        for m in self._motors:
            m.close()
        self._open = False

    def shutdown(self) -> None:
        self._require_open()
        for m in self._motors:
            if m._hw is not None:
                m.disable()

    def close_bus(self) -> None:
        self._require_open()

    def enable_all(self) -> None:
        self._require_open()
        for m in sorted(self._motors, key=lambda m: m.motor_id):
            m.enable()

    def disable_all(self) -> None:
        self._require_open()
        for m in sorted(self._motors, key=lambda m: m.motor_id):
            m.disable()

    def poll_feedback_once(self) -> None:
        # replies are received as they are sent (motorbridge's polling thread)
        self._require_open()

    def add_robstride_motor(self, motor_id: int, feedback_id: int, model: str) -> Motor:
        self._require_open()
        if not 1 <= int(motor_id) <= 255:
            raise ValueError(f"RobStride motor_id must be in 1..255, got {motor_id}")
        if not 0 <= int(feedback_id) <= 255:
            raise ValueError(f"RobStride feedback_id/host_id must be in 0..255, got {feedback_id}")
        if model not in MODELS:
            raise CallError(f"add_robstride_motor failed: unknown RobStride model: {model}")
        m = Motor(self._bus, motor_id, feedback_id, model)
        self._motors.append(m)
        return m

    def _absent(self, vendor: str):
        raise CallError(f"add_{vendor}_motor failed: no {vendor} motor on the simulated "
                        "reBot's bus")

    def add_damiao_motor(self, motor_id: int, feedback_id: int, model: str) -> Motor:
        self._absent("damiao")

    def add_hexfellow_motor(self, motor_id: int, feedback_id: int, model: str) -> Motor:
        self._absent("hexfellow")

    def add_myactuator_motor(self, motor_id: int, feedback_id: int, model: str) -> Motor:
        self._absent("myactuator")

    def add_hightorque_motor(self, motor_id: int, feedback_id: int, model: str) -> Motor:
        self._absent("hightorque")

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.shutdown()
        finally:
            self.close()


__all__ = ["__version__", "get_version", "Controller", "Motor", "Mode", "MotorState",
           "MotorBridgeError", "AbiLoadError", "CallError"]
