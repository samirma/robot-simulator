#!/usr/bin/env python3
"""The SO-101's simulated STS3215 servo bus (not a ROS node).

The simulator's feetech_ros2_driver plugin sends every goal the real plugin would write
to the bus (UDP localhost:47001) and reads each servo's present position and speed
(UDP localhost:47002). Each servo moves to its goal the way the plugin commands the real
one: at speed 2400 ticks/s (3.682 rad/s) with acceleration register 50, driving the
position servo of its joint in the simulation. Present position and speed are the
simulated joint's.

Acceleration register 50: Feetech's STS unit is 100 ticks/s^2 per count, i.e. 5000
ticks/s^2 = 7.67 rad/s^2 -- an estimate (the pinned source does not state the unit).
"""

from __future__ import annotations

import math
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import common  # noqa: E402

SPEED = 2400 * 2 * math.pi / 4096          # rad/s
ACCEL = 50 * 100 * 2 * math.pi / 4096      # rad/s^2 (estimate, see above)
RATE = 200.0
BUS_PORT, STATE_PORT = 47001, 47002


def main():
    link = common.SimLink()
    common.exit_when_lost(link)
    joints = [a for a in link.actuators()]
    state = {}
    have = threading.Event()
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", BUS_PORT))
    rx.setblocking(False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def on_state(h, _):
        state.update({k: v for k, v in h["joints"].items() if k in joints})
        have.set()
        msg = "state " + " ".join(f"{state[j][0]:.9f} {state[j][1]:.9f}" for j in order)
        tx.sendto(msg.encode(), ("127.0.0.1", STATE_PORT))

    order = joints
    link.subscribe("state", RATE, on_state, joints=True)
    have.wait(10)
    sp = {j: state[j][0] for j in joints}      # profile setpoint
    vel = {j: 0.0 for j in joints}
    goal = dict(sp)
    dt = 1.0 / RATE
    while True:
        t0 = time.monotonic()
        while True:
            try:
                data = rx.recv(8192).decode().split()
            except BlockingIOError:
                break
            if data and data[0] == "goal":
                for name, v in zip(data[1::2], data[2::2]):
                    if name in goal:
                        goal[name] = float(v)
        for j in joints:
            err = goal[j] - sp[j]
            # trapezoidal profile: accelerate/decelerate at ACCEL, cruise at SPEED
            v_stop = math.sqrt(2 * ACCEL * abs(err))
            v_want = math.copysign(min(SPEED, v_stop), err)
            dv = max(-ACCEL * dt, min(ACCEL * dt, v_want - vel[j]))
            vel[j] += dv
            step = vel[j] * dt
            if abs(step) >= abs(err):
                sp[j], vel[j] = goal[j], 0.0
            else:
                sp[j] += step
        link.ctrl(sp)
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
