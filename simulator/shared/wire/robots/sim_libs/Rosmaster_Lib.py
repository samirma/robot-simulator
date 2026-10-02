"""The ROSMASTER expansion board, simulated: a drop-in `Rosmaster_Lib` for the vendor's
unchanged `Mcnamu_X3plus.py` driver (yahboomcar_bringup), put first on its PYTHONPATH.

Where the real library talks to the STM32 board over serial, this one talks to the
simulation. It reproduces the board firmware behaviour recorded in the robot
specification's interface file (robots_specs/rosmaster_x3_plus/ros.yml):

* `set_car_motion(vx, vy, wz)`: the firmware's mecanum mixing (motor order L1, L2, R1, R2),
  each wheel clamped to +-0.7 m/s at the rim, all-zero brakes; the four wheels' velocity
  servos in the simulation are the motors. No watchdog. (The firmware's IMU yaw-hold
  correction while driving is not reproduced -- a documented simplification.)
* `get_motion_data()`: the chassis velocity from the wheel encoders (the simulated wheels'
  speeds through the inverse mixing).
* `set_uart_servo_angle_array(angles, run_time)` / `set_uart_servo_angle(id, angle, t)`:
  bus servos 1-6 move to their angles over run_time ms (servo deg 90 = joint zero;
  servo 6, the gripper, 30 deg open .. 180 deg closed); an out-of-range array is ignored
  with the library's message. `get_uart_servo_angle_array()` reads the simulated joints
  back in servo degrees.
* accelerometer (m/s^2) and gyroscope (rad/s) from the IMU site; magnetometer from a fixed
  local geomagnetic field (estimate: 0.22 horizontal north, 0.42 down, in gauss);
  battery 12.3 V and firmware version 3.5 (estimates: not simulated quantities).
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

_WIRE = Path(__file__).resolve().parents[2]
for _p in (str(_WIRE), str(_WIRE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402

WHEELS = ["front_left_joint", "back_left_joint", "front_right_joint", "back_right_joint"]
SERVOS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint"]
SERVO_RANGE = [(0, 180), (0, 180), (0, 180), (0, 180), (0, 270), (0, 180)]
RATE = 100.0
MAG_WORLD = (0.22, 0.0, -0.42)


def _drive_row():
    iface = common.interface()
    return next(m for m in iface["motions"] if m["id"] == "drive")["kinematics"]


def servo_to_rad(i: int, deg: float) -> float:
    if i == 5:  # gripper: 30..180 deg -> 0..90 deg (np.interp clamps), then zero at 90
        deg = (min(max(deg, 30.0), 180.0) - 30.0) * 90.0 / 150.0
    return math.radians(deg - 90.0)


def rad_to_servo(i: int, rad: float) -> float:
    deg = math.degrees(rad) + 90.0
    if i == 5:
        deg = 30.0 + deg * 150.0 / 90.0
    return deg


class Rosmaster:
    def __init__(self, car_type=1, com="/dev/myserial", delay=0.002, debug=False):
        k = _drive_row()
        self.r = float(k["wheel_radius"])
        self.s = float(k.get("lx_plus_ly") or (float(k["lx"]) + float(k["ly"])))
        self.vmax = float(k.get("wheel_speed_limit", 0.7))
        self.link = common.SimLink()
        common.exit_when_lost(self.link)
        self.lock = threading.Lock()
        self.state = {}
        self.imu = {}
        self.have = threading.Event()
        self.link.subscribe("state", 50.0, self._on_state, joints=True, imu_sites=["imu_link"])
        self.have.wait(10)
        self.servo_cmd = [self.state[j][0] for j in SERVOS] if self.have.is_set() else [0.0] * 6
        self.moves = [None] * 6    # per servo: (start value, target, t0, duration)
        self.wheel_cmd = [0.0] * 4
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ sim side

    def _on_state(self, h, _):
        with self.lock:
            self.state.update(h["joints"])
            self.imu = h.get("imu", {}).get("imu_link", self.imu)
        self.have.set()

    def _loop(self):
        dt = 1.0 / RATE
        while True:
            t = time.monotonic()
            with self.lock:
                for i, mv in enumerate(self.moves):
                    if mv is None:
                        continue
                    a, b, t0, dur = mv
                    f = 1.0 if dur <= 0 else min(1.0, (t - t0) / dur)
                    self.servo_cmd[i] = a + (b - a) * f
                    if f >= 1.0:
                        self.moves[i] = None
                cmd = dict(zip(SERVOS, self.servo_cmd))
                cmd.update(zip(WHEELS, self.wheel_cmd))
            self.link.ctrl(cmd)
            time.sleep(max(0.0, dt - (time.monotonic() - t)))

    # ------------------------------------------------------------------ board API

    def set_car_type(self, car_type):
        pass

    def create_receive_threading(self):
        pass

    def set_car_motion(self, v_x, v_y, v_z):
        vx, vy, wz = float(v_x), float(v_y), float(v_z)
        s = self.s
        rim = [vx - vy - s * wz, vx + vy - s * wz, vx + vy + s * wz, vx - vy + s * wz]
        rim = [max(-self.vmax, min(self.vmax, v)) for v in rim]
        with self.lock:
            self.wheel_cmd = [v / self.r for v in rim]

    def get_motion_data(self):
        with self.lock:
            w = [self.state.get(j, [0, 0, 0])[1] * self.r for j in WHEELS]
        fl, bl, fr, br = w
        vx = (fl + bl + fr + br) / 4.0
        vy = (-fl + bl + fr - br) / 4.0
        wz = (-fl - bl + fr + br) / (4.0 * self.s)
        return round(vx, 3), round(vy, 3), round(wz, 3)

    def _servo_move(self, i, deg, run_time):
        target = servo_to_rad(i, deg)
        dur = max(0.0, float(run_time)) / 1000.0
        with self.lock:
            while len(self.moves) < 6:
                self.moves.append(None)
            self.moves[i] = (self.servo_cmd[i], target, time.monotonic(), dur)

    def set_uart_servo_angle_array(self, angle_s=[90, 90, 90, 90, 90, 180], run_time=500):
        if len(angle_s) != 6 or any(not lo <= a <= hi for a, (lo, hi) in zip(angle_s, SERVO_RANGE)):
            print("angle_s input error!")
            return
        for i, a in enumerate(angle_s):
            self._servo_move(i, a, run_time)

    def set_uart_servo_angle(self, s_id, s_angle, run_time=500):
        i = int(s_id) - 1
        if not 0 <= i < 6 or not SERVO_RANGE[i][0] <= s_angle <= SERVO_RANGE[i][1]:
            print("angle input error!")
            return
        self._servo_move(i, s_angle, run_time)

    def get_uart_servo_angle_array(self):
        with self.lock:
            return [int(round(rad_to_servo(i, self.state[j][0]))) if j in self.state else -1
                    for i, j in enumerate(SERVOS)]

    def get_uart_servo_angle(self, s_id):
        return self.get_uart_servo_angle_array()[int(s_id) - 1]

    def get_accelerometer_data(self):
        with self.lock:
            a = self.imu.get("accel", [0.0, 0.0, 9.81])
        return tuple(round(x, 4) for x in a)

    def get_gyroscope_data(self):
        with self.lock:
            g = self.imu.get("gyro", [0.0, 0.0, 0.0])
        return tuple(round(x, 4) for x in g)

    def get_magnetometer_data(self):
        with self.lock:
            q = self.imu.get("quat", [1.0, 0.0, 0.0, 0.0])
        w, x, y, z = q
        # rotate the world field into the IMU frame (inverse rotation)
        R = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        m = [sum(R[r][c] * MAG_WORLD[r] for r in range(3)) for c in range(3)]
        return tuple(round(v, 4) for v in m)

    def get_battery_voltage(self):
        return 12.3

    def get_version(self):
        return 3.5

    def set_colorful_effect(self, effect, speed=255, parm=255):
        pass

    def set_colorful_lamps(self, led_id, red, green, blue):
        pass

    def set_beep(self, on_time):
        pass

    def set_pid_param(self, kp, ki, kd, forever=False):
        pass
