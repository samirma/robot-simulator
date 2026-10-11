"""The simulated ydlidar_ros_driver (wire/robots/_ros1_drivers.YdLidar), without ROS: the
ROSMASTER's YDLIDAR 4ROS reports the intensities its record states -- the SDK's default
quality, 10, at each reported range and 0 where there is none, and 10 for each
/point_cloud point (robots_specs/rosmaster_x3_plus/ros.yml sensors.lidars[].intensity);
the myAGV's X2L its recorded 0.0 (simulator spec §3 Robot authority)."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SHARED = Path(__file__).resolve().parents[3] / "simulator" / "shared"
for p in (str(SHARED), str(SHARED / "wire")):
    if p not in sys.path:
        sys.path.insert(0, p)


class Msg:
    def __init__(self, *a, **kw):
        self.header = types.SimpleNamespace(stamp=None, frame_id="")
        self.args = a
        self.__dict__.update(kw)


@pytest.fixture
def lidar(monkeypatch, request):
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    module("geometry_msgs")
    module("geometry_msgs.msg", Point32=lambda x, y, z: (x, y, z))
    module("sensor_msgs")
    module("sensor_msgs.msg", ChannelFloat32=lambda name, values: (name, values))
    monkeypatch.setenv("RSIM_ROBOT", request.param)

    import common
    from robots._ros1_drivers import YdLidar

    class Node:
        name = "/ydlidar_lidar_publisher"
        iface = common.interface()

        def __init__(self):
            self.sent = {}
            self.pub = {"/scan": None, "/point_cloud": None}

        def new(self, topic, stamp=None):
            return Msg()

        def publish(self, topic, msg):
            self.sent.setdefault(topic, []).append(msg)

    node = Node()
    lid = YdLidar(node)
    lid.n = int(lid.row["samples"])
    return node, lid


def scan(lid, node):
    n = lid.n
    ranges = np.full(n, 2.0, np.float32)
    ranges[::3] = np.inf                                   # no return
    lid._scan({"stamp": 1.0}, ranges.tobytes())
    return ranges, node.sent["/scan"][-1], node.sent["/point_cloud"][-1]


@pytest.mark.parametrize("lidar", ["rosmaster_x3_plus"], indirect=True)
def test_rosmaster_intensities_are_the_recorded_default_quality(lidar):
    node, lid = lidar
    it = lid.row["intensity"]
    assert (it["valid"], it["invalid"]) == (10.0, 0.0)
    ranges, msg, pc = scan(lid, node)
    assert len(msg.intensities) == len(msg.ranges) == lid.n
    for r, i in zip(msg.ranges, msg.intensities):
        assert i == (10.0 if math.isfinite(r) else 0.0)
    (name, values), (sname, _) = pc.channels
    assert name == "intensities" and sname == "stamps"
    assert len(values) == len(pc.points) == int(np.isfinite(ranges).sum())
    assert set(values) == {10.0}


@pytest.mark.parametrize("lidar", ["myagv"], indirect=True)
def test_myagv_intensities_are_its_records(lidar):
    """The myAGV's X2L, as its record states (an estimate: the recorded SDK revision settles
    the X2L's intensity width at 0 bits): intensities all 0.0, as many as ranges, and 0.0
    per /point_cloud point."""
    node, lid = lidar
    it = lid.row["intensity"]
    assert (it["valid"], it["invalid"], it["basis"]) == (0.0, 0.0, "estimate")
    ranges, msg, pc = scan(lid, node)
    assert msg.intensities == [0.0] * len(msg.ranges) and len(msg.ranges) == lid.n
    (name, values), _ = pc.channels
    assert name == "intensities" and values == [0.0] * len(pc.points)


@pytest.mark.parametrize("lidar", ["myagv"], indirect=True)
def test_a_record_without_intensity_gets_none(lidar):
    """Only a lidar no ydlidar_ros_driver publishes may lack the record (the schema refuses
    it for one that does); the wire then states no intensities."""
    node, lid = lidar
    lid.intensity = None
    _, msg, pc = scan(lid, node)
    assert msg.intensities == []
