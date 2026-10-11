"""usb_cam's set_camera_info, on the myAGV's and the ROSMASTER's /usb_cam and the AiNex's /camera (ros1_node.py
with the robots' wire modules, without ROS): camera_info_manager stores the calibration the
call sends and the node publishes it with every frame from then on, header set to the
frame's (simulator spec §3 Real-robot fidelity: it answers every command according to the
recorded behaviour)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from test_ainex_app_nodes import FakeRospy, Msg

CAMERAS = {"myagv": "/usb_cam", "ainex": "/camera", "rosmaster_x3_plus": "/usb_cam"}


@pytest.fixture(params=sorted(CAMERAS))
def camera(request, monkeypatch):
    return _make(request.param, monkeypatch)


def _make(rid, monkeypatch):
    rospy = FakeRospy()
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    srv_cls = type("Srv", (), {"_response_class": Msg})
    roslib = types.ModuleType("roslib")
    roslib.message = types.SimpleNamespace(get_message_class=lambda t: Msg,
                                           get_service_class=lambda t: srv_cls)
    monkeypatch.setitem(sys.modules, "roslib", roslib)
    monkeypatch.setitem(sys.modules, "roslib.message", roslib.message)
    servers = []

    class Server:
        """dynamic_reconfigure's Server: the callback with the config at start."""

        def __init__(self, cfg, callback, namespace="~"):
            self.cfg, self.callback, self.namespace = cfg, callback, namespace
            servers.append(self)
            callback(dict(getattr(cfg, "defaults", {})), 0)

    class TheoraPublisher:
        """libtheora is not on the host: the encoder's packets, as the wire publishes them."""

        def __init__(self):
            self.stream_header, self.configs = [], []

        def configure(self, config):
            self.configs.append(dict(config))
            return config

        def encode(self, header, rgb):
            out = []
            if not self.stream_header:
                self.stream_header = [dict(header=header, data=b"\x80theora", b_o_s=1, e_o_s=0,
                                           granulepos=0, packetno=0)]
                out = list(self.stream_header)
            return out + [dict(header=header, data=b"frame", b_o_s=0, e_o_s=0, granulepos=1,
                               packetno=len(out) + 1)]

    def cfg_module(name, defaults):
        m = types.ModuleType(name)
        m.defaults = defaults
        return m

    rospy.servers = servers
    for mod, attrs in {"sensor_msgs": {}, "sensor_msgs.srv": {"SetCameraInfoResponse": Msg},
                       "std_srvs": {}, "std_srvs.srv": {"EmptyResponse": Msg},
                       "dynamic_reconfigure": {}, "dynamic_reconfigure.server": {"Server": Server},
                       "compressed_image_transport": {}, "compressed_image_transport.cfg": {},
                       "compressed_depth_image_transport": {},
                       "compressed_depth_image_transport.cfg": {},
                       "theora_image_transport": {}, "theora_image_transport.cfg": {}}.items():
        m = types.ModuleType(mod)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, mod, m)
    for name, defaults in {
            "compressed_image_transport.cfg.CompressedPublisherConfig": {"format": "jpeg"},
            "compressed_depth_image_transport.cfg.CompressedDepthPublisherConfig": {"format": "png"},
            "theora_image_transport.cfg.TheoraPublisherConfig": {
                "optimize_for": 1, "target_bitrate": 800000, "quality": 31,
                "keyframe_frequency": 64}}.items():
        monkeypatch.setitem(sys.modules, name, cfg_module(name, defaults))
    import robots._theora

    monkeypatch.setattr(robots._theora, "TheoraPublisher", TheoraPublisher)
    monkeypatch.setenv("RSIM_ROBOT", rid)
    monkeypatch.delitem(sys.modules, "ros1_node", raising=False)
    import ros1_node

    class Link:
        def __init__(self):
            self.subs, self.ended, self.next = {}, [], 1

        def subscribe(self, *a, **kw):
            sid, self.next = self.next, self.next + 1
            self.subs[sid] = (a, kw)
            return sid

        def unsubscribe(self, sid):
            self.ended.append(sid)
            self.subs.pop(sid, None)

    link = Link()
    rospy.link = link
    monkeypatch.setattr(ros1_node.Node, "link", property(lambda self: link))
    node = ros1_node.Node(CAMERAS[rid])
    return rid, node, rospy


def published(rospy, topic):
    [pub] = rospy.pubs[topic]
    return pub.sent[-1]


def test_set_camera_info_is_published_from_then_on(camera):
    rid, node, rospy = camera
    cam = node.behaviour
    row = cam.cam
    name = CAMERAS[rid] + "/set_camera_info"
    assert name in rospy.services
    h, w = int(row["height"]), int(row["width"])
    cam._frame({"height": h, "width": w, "stamp": 1.0}, np.zeros((h, w, 3), np.uint8).tobytes())
    before = published(rospy, row["info_topic"])
    want_k = [float((row["intrinsics"] or {}).get(k, 0.0)) for k in ("fx", "cx", "fy", "cy")]
    k0 = before.k
    assert [k0[0], k0[2], k0[4], k0[5]] == want_k          # the recorded calibration
    sent = Msg(width=w, height=h, distortion_model="plumb_bob",
               d=[0.1, -0.2, 0.001, 0.002, 0.0],
               k=[600.0, 0.0, 321.5, 0.0, 601.0, 238.5, 0.0, 0.0, 1.0],
               r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
               p=[600.0, 0.0, 321.5, 0.0, 0.0, 601.0, 238.5, 0.0, 0.0, 0.0, 1.0, 0.0],
               binning_x=0, binning_y=0)
    sent.header.frame_id = "elsewhere"
    res = rospy.services[name](Msg(camera_info=sent))
    assert res.success is True and res.status_message == ""
    for stamp in (2.0, 3.0):
        cam._frame({"height": h, "width": w, "stamp": stamp},
                   np.zeros((h, w, 3), np.uint8).tobytes())
        info = published(rospy, row["info_topic"])
        for field in ("width", "height", "distortion_model", "d", "k", "r", "p", "binning_x",
                      "binning_y"):
            assert getattr(info, field) == getattr(sent, field), field
        # the header is the frame's, not the request's
        assert info.header.stamp == stamp and info.header.frame_id == row["frame_id"]
    # the stored calibration is a copy: changing the request afterwards changes nothing
    sent.k[0] = 1.0
    cam._frame({"height": h, "width": w, "stamp": 4.0}, np.zeros((h, w, 3), np.uint8).tobytes())
    assert published(rospy, row["info_topic"]).k[0] == 600.0



def _rosmaster(request, monkeypatch):
    return _make("rosmaster_x3_plus", monkeypatch)


def test_arm_camera_compressed_plugins(monkeypatch):
    """The ROSMASTER's /usb_cam carries image_transport's compressed, compressedDepth and
    theora plugins, each with its stock dynamic_reconfigure server in its namespace;
    compressed publishes each frame while subscribed as compressed_image_transport encodes
    it."""
    import cv2

    rid, node, rospy = _rosmaster(None, monkeypatch)
    cam = node.behaviour
    assert {s.namespace for s in rospy.servers} == {"/usb_cam/image_raw/compressed",
                                                    "/usb_cam/image_raw/compressedDepth",
                                                    "/usb_cam/image_raw/theora"}
    for name in cam.serves:       # left to the dynamic_reconfigure servers
        assert name not in rospy.services
    h, w = int(cam.cam["height"]), int(cam.cam["width"])
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, : w // 2] = (200, 30, 10)
    payload = rgb.tobytes()
    cam._frame({"height": h, "width": w, "stamp": 1.0}, payload)
    assert "/usb_cam/image_raw/compressed" not in {t for t, ps in rospy.pubs.items() if ps[0].sent}
    [pub] = rospy.pubs["/usb_cam/image_raw/compressed"]
    pub.connections = 1
    cam._frame({"height": h, "width": w, "stamp": 2.0}, payload)
    msg = pub.sent[-1]
    assert msg.format == "rgb8; jpeg compressed bgr8"
    assert msg.header.stamp == 2.0 and msg.header.frame_id == "usb_cam"
    bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(bgr[:, :, ::-1].astype(int) - rgb).mean() < 3
    cam._configurer("compressed")({"format": "png", "png_level": 3}, 0)
    cam._frame({"height": h, "width": w, "stamp": 3.0}, payload)
    msg = pub.sent[-1]
    assert msg.format == "rgb8; png compressed bgr8"
    assert (cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1] == rgb).all()
    assert not rospy.pubs["/usb_cam/image_raw/compressedDepth"][0].sent   # rgb8: nothing


def _until(pred, timeout=5.0):
    import time

    t0 = time.monotonic()
    while not pred() and time.monotonic() - t0 < timeout:
        time.sleep(0.01)
    return pred()


def test_arm_camera_plugins_are_the_records(monkeypatch):
    """The arm camera's plugins are the record's served ones on the image topic, each with
    its recorded initial config; compressedDepth logs the plugin's error for an rgb8 frame
    while subscribed and publishes nothing; theora publishes the encoder's packets, its stream
    header first and again to each new subscriber on connecting."""
    rid, node, rospy = _rosmaster(None, monkeypatch)
    cam = node.behaviour
    assert set(cam.plugins) == {"compressed", "compressedDepth", "theora"}
    assert cam.config["compressed"]["jpeg_quality"] == 80
    assert cam.config["theora"] == {"optimize_for": 1, "target_bitrate": 800000,
                                    "quality": 31, "keyframe_frequency": 64}
    assert cam.theora.configs[-1]["quality"] == 31
    h, w = int(cam.cam["height"]), int(cam.cam["width"])
    frame = {"height": h, "width": w, "stamp": 5.0}
    payload = bytes(h * w * 3)
    [depth] = rospy.pubs["/usb_cam/image_raw/compressedDepth"]
    depth.connections = 1
    cam._frame(frame, payload)
    assert not depth.sent and rospy.errors and "(input format is: rgb8)" in rospy.errors[-1]
    depth.connections = 0
    [theora] = rospy.pubs["/usb_cam/image_raw/theora"]
    cam._frame(frame, payload)
    assert not theora.sent                             # nobody listens: nothing encoded
    theora.connections = 1
    cam._frame(frame, payload)
    assert [m.data for m in theora.sent] == [b"\x80theora", b"frame"]
    assert theora.sent[0].header.frame_id == "usb_cam" and theora.sent[0].b_o_s == 1
    peer = []
    for listener in theora.listeners:                  # a later subscriber: the header alone
        listener.peer_subscribe("/usb_cam/image_raw/theora", None, peer.append)
    assert [m.data for m in peer] == [b"\x80theora"]


def test_arm_camera_renders_while_capturing_and_listened(monkeypatch):
    """Rendered at start to warm the stream, then only while usb_cam captures and one of its
    topics has a subscriber: a subscriber's connection starts it at once, its leaving and
    stop_capture stop it; the render thread uses the node's own link throughout."""
    import time

    rid, node, rospy = _rosmaster(None, monkeypatch)
    cam, link = node.behaviour, rospy.link
    assert cam.link is link
    assert _until(lambda: len(link.subs) == 1)         # warming
    h, w = int(cam.cam["height"]), int(cam.cam["width"])
    for i in range(cam.WARM_FRAMES):
        cam._frame({"height": h, "width": w, "stamp": 1.0 + i}, bytes(h * w * 3))
    assert _until(lambda: not link.subs)               # nobody listens
    [pub] = rospy.pubs["/usb_cam/image_raw"]
    pub.connections = 1
    t0 = time.monotonic()
    for listener in pub.listeners:
        listener.peer_subscribe("/usb_cam/image_raw", None, None)
    assert _until(lambda: len(link.subs) == 1, 0.5) and time.monotonic() - t0 < 0.1
    rospy.services["/usb_cam/stop_capture"](types.SimpleNamespace())
    assert _until(lambda: not link.subs)               # usb_cam stopped capturing
    rospy.services["/usb_cam/start_capture"](types.SimpleNamespace())
    assert _until(lambda: len(link.subs) == 1)
    pub.connections = 0
    for listener in pub.listeners:
        listener.peer_unsubscribe("/usb_cam/image_raw", 0)
    assert _until(lambda: not link.subs)
