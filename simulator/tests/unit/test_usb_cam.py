"""usb_cam's set_camera_info, on the myAGV's /usb_cam and the AiNex's /camera (ros1_node.py
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

CAMERAS = {"myagv": "/usb_cam", "ainex": "/camera"}


@pytest.fixture(params=sorted(CAMERAS))
def camera(request, monkeypatch):
    rid = request.param
    rospy = FakeRospy()
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    srv_cls = type("Srv", (), {"_response_class": Msg})
    roslib = types.ModuleType("roslib")
    roslib.message = types.SimpleNamespace(get_message_class=lambda t: Msg,
                                           get_service_class=lambda t: srv_cls)
    monkeypatch.setitem(sys.modules, "roslib", roslib)
    monkeypatch.setitem(sys.modules, "roslib.message", roslib.message)
    for mod, attrs in {"sensor_msgs": {}, "sensor_msgs.srv": {"SetCameraInfoResponse": Msg},
                       "std_srvs": {}, "std_srvs.srv": {"EmptyResponse": Msg}}.items():
        m = types.ModuleType(mod)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, mod, m)
    monkeypatch.setenv("RSIM_ROBOT", rid)
    monkeypatch.delitem(sys.modules, "ros1_node", raising=False)
    import ros1_node

    class Link:
        def subscribe(self, *a, **kw):
            return 1

    monkeypatch.setattr(ros1_node.Node, "link", property(lambda self: Link()))
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
