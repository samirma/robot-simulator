"""The simulated ROSMASTER expansion board (`wire/robots/sim_libs/Rosmaster_Lib.py`) answers
the driver's `set_car_motion` as the recorded firmware path does
(robots_specs/rosmaster_x3_plus/ros.yml, drive motion): vx and vy clamped to the speed
limit, the recorded mecanum mixing, each wheel clamped at the rim -- and no IMU yaw-hold,
which the firmware applies only to commands the driver never sends (amended 2026-10-02)."""

import importlib.util
import math
import time

import pytest

import common
import wirecheck
from conftest import SHARED


@pytest.fixture
def board(monkeypatch):
    joints = {}
    imu = {"imu_link": {"accel": [0.0, 0.0, 9.81], "gyro": [0.0, 0.0, 0.0],
                        "quat": [1.0, 0.0, 0.0, 0.0]}}

    class Link:
        def subscribe(self, stream, rate, callback, **params):
            self.callback = callback
            callback({"joints": joints, "imu": imu}, None)
            return 1

        def ctrl(self, values):
            self.sent = dict(values)

    monkeypatch.setenv("RSIM_ROBOT", "rosmaster_x3_plus")
    monkeypatch.setattr(common, "SimLink", Link)
    monkeypatch.setattr(common, "exit_when_lost", lambda link, cleanup=None: None)
    path = SHARED / "wire" / "robots" / "sim_libs" / "Rosmaster_Lib.py"
    spec = importlib.util.spec_from_file_location("Rosmaster_Lib_under_test", path)
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    for j in lib.WHEELS + lib.SERVOS:
        joints[j] = [0.0, 0.0, 0.0]
    b = lib.Rosmaster()
    b.feed = lambda yaw: b.link.callback(
        {"joints": joints, "imu": {"imu_link": dict(imu["imu_link"], quat=[
            math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])}}, None)
    return b


def mixed(vx, vy, wz):
    k = next(m for m in wirecheck.interface("rosmaster_x3_plus")["motions"]
             if m["id"] == "drive")["kinematics"]
    s, r, vmax = k["lx_plus_ly"], k["wheel_radius"], k["wheel_speed_limit"]
    vx, vy = (max(-vmax, min(vmax, v)) for v in (vx, vy))
    rim = [vx - vy - s * wz, vx + vy - s * wz, vx + vy + s * wz, vx - vy + s * wz]
    return [max(-vmax, min(vmax, v)) / r for v in rim]


def test_set_car_motion_is_the_recorded_mixing_with_no_yaw_hold(board):
    for cmd in ((0.2, 0.0, 0.0), (0.0, -0.3, 0.5), (1.0, 0.0, 0.0), (1.0, 0.5, 0.0)):
        board.set_car_motion(*cmd)
        assert board.wheel_cmd == pytest.approx(mixed(*cmd), abs=1e-12), cmd
    # the heading drifts off by 0.1 rad while driving straight: nothing corrects it
    board.set_car_motion(0.2, 0.0, 0.0)
    board.feed(0.1)
    time.sleep(0.1)                  # the board's loop sends the wheels a few times
    sent = [board.link.sent[j] for j in ("front_left_joint", "back_left_joint",
                                         "front_right_joint", "back_right_joint")]
    assert sent == pytest.approx(mixed(0.2, 0.0, 0.0), abs=1e-12)
