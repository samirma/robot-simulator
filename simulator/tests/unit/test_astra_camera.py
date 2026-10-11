"""The ROSMASTER's simulated Astra Pro Plus (wire/robots/rosmaster_x3_plus.AstraProPlus),
without ROS: its depth range is the record's (range_m, a manufacturer figure), its camera
services answer with the intrinsics its streams publish, and its pipeline is lazy as the
driver's is -- rendered only while an image or point-cloud topic has a subscriber
(simulator spec §3 Robot authority, Sensor calibration, Real-robot fidelity)."""

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

    class Link:
        def __init__(self):
            self.subs, self.ended, self.next = {}, [], 1

        def subscribe(self, stream, rate, callback, **params):
            sid, self.next = self.next, self.next + 1
            self.subs[sid] = (stream, rate, params)
            return sid

        def unsubscribe(self, sid):
            self.ended.append(sid)
            self.subs.pop(sid)

    class Node:
        name = "/camera/camera"
        iface = common.interface()

        def __init__(self):
            self.sent = {}
            self.link = Link()

        def new(self, topic, stamp=None):
            m = CameraInfo() if topic.endswith("camera_info") else Msg()
            self.fill_header(m, topic, stamp)
            return m

        def fill_header(self, msg, topic, stamp=None):
            msg.header.stamp = stamp

        def publish(self, topic, msg):
            self.sent.setdefault(topic, []).append(msg)

    node = Node()
    cam = AstraProPlus(node)
    cam._pipeline(True)       # a subscriber listens (test_lazy_pipeline)
    return node, cam


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


def test_lazy_pipeline_follows_the_drivers_subscriber_callbacks():
    """ob_camera_node.cpp L535-L617 with enable_pipeline, one connect/disconnect event at a
    time: any image or point-cloud subscriber starts the pipeline; an image topic losing one
    stops it once no image topic has any; a point cloud losing its last one stops it too,
    even while the other cloud is subscribed (the driver checks only the image publishers)."""
    from robots.rosmaster_x3_plus import lazy_pipeline

    images = ["rgb", "depth", "ir"]
    clouds = ["pts", "reg"]
    zero = dict.fromkeys(images + clouds, 0)

    def go(running, event, topic, **counts):
        return lazy_pipeline(running, event, topic, dict(zero, **counts), images, clouds)

    assert go(False, "subscribe", "rgb", rgb=1) is True
    assert go(False, "subscribe", "pts", pts=1) is True
    assert go(True, "subscribe", "depth", rgb=1, depth=1) is True
    assert go(True, "unsubscribe", "rgb", ir=1) is True
    assert go(True, "unsubscribe", "rgb") is False
    assert go(True, "unsubscribe", "pts") is False
    assert go(True, "unsubscribe", "pts", pts=1) is True
    # the driver's quirk: the last image subscriber leaving stops it under a cloud subscriber,
    # and so does one cloud's last subscriber while the other cloud keeps one
    assert go(True, "unsubscribe", "rgb", reg=1) is False
    assert go(True, "unsubscribe", "pts", reg=1) is False
    # a leave and a join in quick succession: the join (re)starts it, as on the driver
    assert go(go(True, "unsubscribe", "rgb"), "subscribe", "rgb", rgb=1) is True
    assert go(False, "unsubscribe", "rgb") is False


def test_subscriber_events_start_and_stop_the_render_stream(astra):
    node, cam = astra
    link = node.link
    cam.on_subscriber_event("unsubscribe", "/camera/rgb/image_raw", dict.fromkeys(
        cam.images + cam.clouds, 0))
    assert not cam.running and not link.subs
    cam.on_subscriber_event("subscribe", "/camera/depth/points", dict(
        dict.fromkeys(cam.images + cam.clouds, 0), **{"/camera/depth/points": 1}))
    assert cam.running and len(link.subs) == 1


def test_lazy_pipeline_renders_only_while_running(astra):
    node, cam = astra
    link = node.link
    [(sid, (stream, rate, params))] = link.subs.items()
    assert stream == "camera" and params["camera"] == "camera_color_optical_frame"
    assert params["rgb"] and params["depth"] and rate == 30.0
    cam._pipeline(False)
    assert link.ended == [sid] and not link.subs
    hd, payload = frame(cam, 1.0)
    cam._frame(hd, payload)            # rendered before the stop: dropped
    assert node.sent == {}
    cam._pipeline(True)
    assert len(link.subs) == 1
    cam._frame(hd, payload)
    assert len(node.sent["/camera/rgb/image_raw"]) == 1
    assert len(node.sent["/camera/rgb/camera_info"]) == 1


def test_a_point_cloud_is_built_only_while_subscribed(astra):
    """A cloud nobody subscribes to is never seen: the wire builds each cloud only while it
    has a subscriber (the images are published regardless, as the running pipeline does)."""
    node, cam = astra

    class Pub:
        def __init__(self, n):
            self.n = n

        def get_num_connections(self):
            return self.n

    node.pub = {"/camera/depth/points": Pub(0), "/camera/depth_registered/points": Pub(1)}
    hd, payload = frame(cam, 1.0)
    cam._frame(hd, payload)
    assert "/camera/depth/points" not in node.sent
    assert len(node.sent["/camera/depth_registered/points"]) == 1
    assert len(node.sent["/camera/depth/image_raw"]) == 1
    node.pub["/camera/depth_registered/points"].n = 0
    cam._frame(hd, payload)
    assert len(node.sent["/camera/depth_registered/points"]) == 1
    assert len(node.sent["/camera/rgb/image_raw"]) == 2


def test_point_clouds_have_the_drivers_padded_layout(astra):
    """setPointCloud2FieldsByString(1, "xyz") pads xyz to 16 bytes, and the coloured cloud
    adds rgb after the padding (the clouds' notes in ros.yml)."""
    node, cam = astra
    hd, payload = frame(cam, 1.0)
    cam._frame(hd, payload)
    for topic, step, names in (("/camera/depth/points", 16, ["x", "y", "z"]),
                               ("/camera/depth_registered/points", 20, ["x", "y", "z", "rgb"])):
        pc = node.sent[topic][-1]
        assert pc.point_step == step and pc.row_step == step * pc.width, topic
        assert [f.name for f in pc.fields] == names
        assert [f.offset for f in pc.fields] == [0, 4, 8, 16][:len(names)]
        pts = np.frombuffer(pc.data, np.uint8).reshape(pc.width, step)
        assert not pts[:, 12:16].any()                     # the padding stays zero
        assert np.allclose(pts[:, 8:12].copy().view(np.float32), 1.0)   # z
    pts = np.frombuffer(node.sent["/camera/depth_registered/points"][-1].data,
                        np.uint8).reshape(-1, 20)
    assert (pts[:, 16:20] == [128, 128, 128, 0]).all()     # b, g, r, 0
