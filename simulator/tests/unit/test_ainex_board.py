"""The AiNex's simulated controller board (wire/robots/sim_libs/ainex_board.Board), without
ROS: its IMU report is a queue of 1 (robots_specs/ainex/ros.yml), so the node, which stamps
what it reads with the time it reads it, always gets the newest sample (simulator spec §3
Timing: stale samples are not restamped as newly acquired)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

SHARED = Path(__file__).resolve().parents[3] / "simulator" / "shared"
for p in (str(SHARED), str(SHARED / "wire"), str(SHARED / "wire" / "robots" / "sim_libs")):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_imu_report_is_a_queue_of_one(monkeypatch):
    import common

    controllers = {"head_pan_controller": {"type": "JointPositionController",
                                           "joint_name": "head_pan",
                                           "servo": {"id": 23, "init": 500, "min": 0,
                                                     "max": 1000}}}
    monkeypatch.setitem(sys.modules, "rospy",
                        types.SimpleNamespace(get_param=lambda name, default=None: controllers))

    class Link:
        def __init__(self, *a, **kw):
            self.lost = types.SimpleNamespace(wait=lambda: None, is_set=lambda: False)

        def subscribe(self, stream, rate, callback, **params):
            callback({"joints": {"head_pan": [0.0, 0.0, 0.0]}}, b"")
            return 1

        def ctrl(self, values):
            pass

    monkeypatch.setattr(common, "SimLink", Link)
    monkeypatch.setattr(common, "exit_when_lost", lambda link, cleanup=None: None)
    monkeypatch.syspath_prepend(str(Path(common.__file__).resolve().parent / "robots" / "sim_libs"))
    import ainex_board

    # no servo loop: only the IMU path is under test
    monkeypatch.setattr(ainex_board.threading, "Thread",
                        lambda *a, **kw: types.SimpleNamespace(start=lambda: None))
    board = ainex_board.Board()
    board.enable_reception(True)

    def state(ax):
        return {"joints": {}, "imu": {"imu_link": {"accel": [ax, 0.0, 9.80665],
                                                   "gyro": [0.0, 0.0, 0.0],
                                                   "quat": [1.0, 0.0, 0.0, 0.0]}}}

    board._on_state(state(1.0 * ainex_board.GRAVITY), b"")
    board._on_state(state(2.0 * ainex_board.GRAVITY), b"")
    first = board.get_imu()
    assert first is not None and abs(first[0] - 2.0) < 1e-9   # the newer sample
    assert board.get_imu() is None                              # nothing older is left
