"""The AiNex controller board, simulated: a drop-in for the vendor's
`ros_robot_controller.ros_robot_controller_sdk` (its `Board` class), injected by
`run_with_fakes.py` so the vendor's ros_robot_controller_node runs unchanged.

Where the real SDK exchanges packets with the STM32 board over /dev/rrc, this one talks
to the simulation:

* bus servos 1-24 (Hiwonder serial servos, 0-1000 pulses over 240 deg): a servo moves to
  its pulse linearly over the commanded move time, or at its top speed (0.2 s per 60 deg,
  the figure ainex_controller itself uses) for a move time of 0; each drives the position
  actuator of its joint. Servo id -> joint and pulse <-> radian come from the controller's
  own `servo_controller.yaml` (the /ainex_controller/controllers parameter): radian =
  (pulse - init) / (+-1000 / 240 deg), the sign flipped where min > max. Position reads
  return the simulated joint in pulses.
* the IMU reports from the IMU site at 200 Hz (estimate: the firmware's report rate is in
  no pinned source; the record takes the node's 100 Hz loop cap as the published rate, so
  the board reports faster than the loop polls and every cycle finds the newest sample):
  acceleration in g, rates in deg/s, and a magnetometer (a fixed local field, estimate);
  battery 11.1 V as millivolts every second
  (estimate: the pack is not simulated). No gamepad receiver, SBUS or button events.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from pathlib import Path

_WIRE = Path(__file__).resolve().parents[2]
for _p in (str(_WIRE), str(_WIRE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402

TICKS_PER_RAD = 180 / 3.1415926 / 240 * 1000     # ainex_controller's ENCODER_TICKS_PER_RADIAN
SERVO_SPEED = math.radians(60) / 0.2               # rad/s: 0.2 s per 60 deg
RATE = 200.0
GRAVITY = 9.80665
IMU_REPORT_HZ = 200.0     # estimate, faster than the node's 100 Hz loop (module docstring)


class Board:
    def __init__(self, device="/dev/rrc", baudrate=1000000, timeout=5):
        import rospy

        self.enable_recv = False
        ctl = rospy.get_param("/ainex_controller/controllers", {})
        self.joint = {}     # servo id -> (joint, init pulse, ticks per rad with sign)
        for c in ctl.values():
            if c.get("type") != "JointPositionController":
                continue
            s = c["servo"]
            sign = -1.0 if s["min"] > s["max"] else 1.0
            self.joint[int(s["id"])] = (c["joint_name"], float(s["init"]), sign * TICKS_PER_RAD)
        self.link = common.SimLink()
        common.exit_when_lost(self.link)
        self.lock = threading.Lock()
        self.state = {}
        # the board's IMU report is a queue of 1, polled once per node cycle: the pending
        # sample is replaced by each newer one, so the node never takes a stale sample
        self.imu_q: queue.Queue = queue.Queue(maxsize=1)
        self.have = threading.Event()
        self.link.subscribe("state", IMU_REPORT_HZ, self._on_state, joints=True,
                            imu_sites=["imu_link"])
        self.have.wait(10)
        with self.lock:
            self.cmd = {sid: self.state[j][0] for sid, (j, _, _) in self.joint.items()
                        if j in self.state}
        self.moves = {}     # servo id -> (start rad, target rad, t0, duration)
        self.servo_position = {}
        self.last_battery = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ sim side

    def _on_state(self, h, _):
        with self.lock:
            self.state.update(h["joints"])
        self.have.set()
        imu = (h.get("imu") or {}).get("imu_link")
        if imu is None:
            return
        ax, ay, az = (a / GRAVITY for a in imu["accel"])
        gx, gy, gz = (math.degrees(g) for g in imu["gyro"])
        mx, my, mz = common.world_to_sensor(imu["quat"], common.MAG_FIELD_WORLD)
        try:
            self.imu_q.put_nowait((ax, ay, az, gx, gy, gz, mx, my, mz))
        except queue.Full:
            try:
                self.imu_q.get_nowait()
            except queue.Empty:
                pass
            self.imu_q.put_nowait((ax, ay, az, gx, gy, gz, mx, my, mz))

    def _loop(self):
        dt = 1.0 / RATE
        while True:
            t = time.monotonic()
            with self.lock:
                for sid, (a, b, t0, dur) in list(self.moves.items()):
                    if dur > 0:
                        f = min(1.0, (t - t0) / dur)
                        self.cmd[sid] = a + (b - a) * f
                        done = f >= 1.0
                    else:  # move time 0: the servo's top speed
                        step = SERVO_SPEED * dt
                        cur = self.cmd.get(sid, b)
                        self.cmd[sid] = cur + max(-step, min(step, b - cur))
                        done = self.cmd[sid] == b
                    if done:
                        del self.moves[sid]
                cmd = {self.joint[sid][0]: v for sid, v in self.cmd.items() if sid in self.joint}
            self.link.ctrl(cmd)
            time.sleep(max(0.0, dt - (time.monotonic() - t)))

    def _pulse_to_rad(self, sid, pulse):
        _, init, k = self.joint[sid]
        return (float(pulse) - init) / k

    def _rad_to_pulse(self, sid, rad):
        _, init, k = self.joint[sid]
        return int(round(init + rad * k))

    # ------------------------------------------------------------------ SDK API

    def enable_reception(self, enable=True):
        self.enable_recv = enable

    def bus_servo_set_position(self, duration, positions):
        now = time.monotonic()
        with self.lock:
            for sid, pulse in positions:
                sid = int(sid)
                if sid not in self.joint:
                    continue
                pulse = max(0, min(1000, int(pulse)))
                self.servo_position[str(sid)] = pulse
                target = self._pulse_to_rad(sid, pulse)
                start = self.cmd.get(sid, target)
                self.moves[sid] = (start, target, now, float(duration))

    def bus_servo_read_position(self, servo_id, fake=False):
        if fake:
            return self.servo_position.get(str(servo_id))
        sid = int(servo_id)
        with self.lock:
            j = self.joint.get(sid)
            if j is None or j[0] not in self.state:
                return None
            return [self._rad_to_pulse(sid, self.state[j[0]][0])]

    def bus_servo_stop(self, servo_id):
        with self.lock:
            for sid in servo_id:
                self.moves.pop(int(sid), None)

    def bus_servo_read_id(self, servo_id=254):
        return [servo_id]

    def bus_servo_read_offset(self, servo_id):
        return [0]

    # The pinned node's get_state also asks for `bus_servo_read_voltage` and
    # `bus_servo_read_torque`, which the pinned SDK does not have (its names are
    # bus_servo_read_vin / bus_servo_read_torque_state): as on the robot, those requests
    # raise in the node. The fake has the SDK methods the node calls, under the SDK's own
    # names -- nothing the SDK lacks.

    def bus_servo_read_temp(self, servo_id):
        return [35]

    def bus_servo_read_temp_limit(self, servo_id):
        return [85]

    def bus_servo_read_angle_limit(self, servo_id):
        return [0, 1000]

    def bus_servo_read_vin_limit(self, servo_id):
        return [4500, 14000]

    def bus_servo_enable_torque(self, servo_id, enable):
        pass

    def bus_servo_set_id(self, a, b):
        pass

    def bus_servo_set_offset(self, servo_id, offset):
        pass

    def bus_servo_save_offset(self, servo_id):
        pass

    def bus_servo_set_angle_limit(self, servo_id, limit):
        pass

    def bus_servo_set_vin_limit(self, servo_id, limit):
        pass

    def bus_servo_set_temp_limit(self, servo_id, limit):
        pass

    def get_imu(self):
        if not self.enable_recv:
            return None
        try:
            return self.imu_q.get_nowait()
        except queue.Empty:
            return None

    def get_battery(self):
        if not self.enable_recv:
            return None
        now = time.monotonic()
        if now - self.last_battery >= 1.0:
            self.last_battery = now
            return 11100
        return None

    def get_button(self):
        return None

    def get_gamepad(self):
        return None

    def get_sbus(self):
        return None

    def set_led(self, *a, **k):
        pass

    def set_buzzer(self, *a, **k):
        pass

    def set_motor_speed(self, *a, **k):
        pass

    def set_oled_text(self, *a, **k):
        pass

    def set_rgb(self, *a, **k):
        pass

    def set_motor_duty(self, *a, **k):
        pass

    def pwm_servo_set_position(self, *a, **k):
        pass

    def pwm_servo_set_offset(self, *a, **k):
        pass

    def pwm_servo_read_offset(self, servo_id):
        return [0]

    def pwm_servo_read_position(self, servo_id):
        return [1500]
