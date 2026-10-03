"""The ROSMASTER's simulated Astra Pro Plus (wire/robots/rosmaster_x3_plus.AstraProPlus),
without ROS: its depth range is the record's (range_m, a manufacturer figure), and its
camera services answer with the intrinsics its streams publish (simulator spec §3 Robot
authority, Sensor calibration, Real-robot fidelity)."""

from __future__ import annotations

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
    """A message with a header and whatever fields the code sets."""

    def __init__(self, **kw):
        self.header = types.SimpleNamespace(stamp=None, frame_id="")
        self.__dict__.update(kw)


class CameraInfo(Msg):
    def __init__(self):
        super().__init__(width=0, height=0, distortion_model="", D=[], K=[0.0] * 9,
                         R=[0.0] * 9, P=[0.0] * 12)


class PointField:
    FLOAT32 = 7

    def __init__(self, name, offset, datatype, count):
        self.name, self.offset, self.datatype, self.count = name, offset, datatype, count


class ServiceException(Exception):
    pass


@pytest.fixture
def astra(monkeypatch):
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    module("sensor_msgs")
    module("sensor_msgs.msg", PointField=PointField, CameraInfo=CameraInfo)
    module("std_srvs")
    module("std_srvs.srv", SetBoolResponse=lambda **kw: Msg(**kw))
    module("orbbec_camera")
    module("orbbec_camera.srv", GetCameraInfoResponse=lambda **kw: Msg(**kw),
           GetCameraParamsResponse=lambda **kw: Msg(**kw))
    module("rospy", ServiceException=ServiceException)
    monkeypatch.setenv("RSIM_ROBOT", "rosmaster_x3_plus")

    import common
    from robots.rosmaster_x3_plus import AstraProPlus

    class Node:
        name = "/camera/camera"
        iface = common.interface()

        def __init__(self):
            self.sent = {}

        def new(self, topic, stamp=None):
            m = CameraInfo() if topic.endswith("camera_info") else Msg()
            self.fill_header(m, topic, stamp)
            return m

        def fill_header(self, msg, topic, stamp=None):
            msg.header.stamp = stamp

        def publish(self, topic, msg):
            self.sent.setdefault(topic, []).append(msg)

    node = Node()
    return node, AstraProPlus(node)


def frame(cam, depth_m):
    w, h = int(cam.rgb["width"]), int(cam.rgb["height"])
    rgb = np.full((h, w, 3), 128, np.uint8).tobytes()
    depth = np.full((h, w), depth_m, np.float32).tobytes()
    return {"width": w, "height": h, "stamp": 12.5}, rgb + depth


def recorded_camera(node, image_topic):
    return next(c for c in node.iface["sensors"]["cameras"] if c["image_topic"] == image_topic)


def test_depth_range_is_the_recorded_one(astra):
    node, cam = astra
    rec = recorded_camera(node, "/camera/depth/image_raw")
    assert rec["range_basis"] == "manufacturer"
    assert cam.depth_range == tuple(float(x) for x in rec["range_m"]) == (0.6, 8.0)
    assert not hasattr(cam, "DEPTH_RANGE")

    hd, payload = frame(cam, 0.4)        # nearer than the sensor's 0.6 m: no return
    cam._frame(hd, payload)
    img = node.sent["/camera/depth/image_raw"][-1]
    assert img.encoding == "16UC1"
    assert not np.frombuffer(img.data, np.uint16).any()
    assert node.sent["/camera/depth/points"][-1].width == 0
    assert node.sent["/camera/depth_registered/points"][-1].width == 0

    hd, payload = frame(cam, 1.0)
    cam._frame(hd, payload)
    mm = np.frombuffer(node.sent["/camera/depth/image_raw"][-1].data, np.uint16)
    assert (mm == 1000).all()
    n = hd["width"] * hd["height"]
    assert node.sent["/camera/depth/points"][-1].width == n
    assert node.sent["/camera/depth_registered/points"][-1].width == n


def test_camera_info_services_answer_the_published_intrinsics(astra):
    node, cam = astra
    hd, payload = frame(cam, 1.0)
    cam._frame(hd, payload)
    for stream, info_topic in (("color", "/camera/rgb/camera_info"),
                               ("depth", "/camera/depth/camera_info"),
                               ("ir", "/camera/ir/camera_info")):
        res = cam.on_service(f"/camera/get_{stream}_camera_info", Msg())
        assert res.success is True
        published = node.sent[info_topic][-1]
        assert res.info.K == published.K and res.info.K[0] > 0, stream
        assert (res.info.width, res.info.height) == (published.width, published.height)
    i = recorded_camera(node, "/camera/depth/image_raw")["intrinsics"]
    k = cam.on_service("/camera/get_depth_camera_info", Msg()).info.K
    assert (k[0], k[4], k[2], k[5]) == (i["fx"], i["fy"], i["cx"], i["cy"])

    p = cam.on_service("/camera/get_camera_params", Msg())
    c = recorded_camera(node, "/camera/rgb/image_raw")["intrinsics"]
    assert p.success is True
    assert p.l_intr_p == [i["fx"], i["fy"], i["cx"], i["cy"]]
    assert p.r_intr_p == [c["fx"], c["fy"], c["cx"], c["cy"]]
    assert p.r2l_r == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] and p.r2l_t == [0.0] * 3
    assert cam.on_service("/camera/get_color_exposure", Msg()) is None   # default answer


def test_toggle_stops_and_restarts_a_stream(astra):
    node, cam = astra
    hd, payload = frame(cam, 1.0)
    assert cam.on_service("/camera/toggle_color", Msg(data=False)).success is True
    with pytest.raises(ServiceException, match="color Already OFF"):
        cam.on_service("/camera/toggle_color", Msg(data=False))
    cam._frame(hd, payload)
    assert "/camera/rgb/image_raw" not in node.sent
    assert "/camera/rgb/camera_info" not in node.sent
    assert "/camera/depth_registered/points" not in node.sent   # needs colour
    assert len(node.sent["/camera/depth/points"]) == 1
    assert len(node.sent["/camera/depth/image_raw"]) == 1

    assert cam.on_service("/camera/toggle_depth", Msg(data=False)).success is True
    cam._frame(hd, payload)
    assert len(node.sent["/camera/depth/image_raw"]) == 1
    assert len(node.sent["/camera/depth/points"]) == 1
    assert len(node.sent["/camera/ir/image_raw"]) == 2

    for stream in ("color", "depth"):
        assert cam.on_service(f"/camera/toggle_{stream}", Msg(data=True)).success is True
    cam._frame(hd, payload)
    for topic in ("/camera/rgb/image_raw", "/camera/depth_registered/points"):
        assert len(node.sent[topic]) == 1, topic
    assert len(node.sent["/camera/depth/image_raw"]) == 2
