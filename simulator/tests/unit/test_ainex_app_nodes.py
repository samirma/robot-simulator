"""The AiNex's app-stack nodes on the wire (ros1_node.py with wire/robots/ainex.py), without
ROS: each presents the endpoints the record gives it, internal publishers and subscribers
included, and answers as the vendor node does (simulator spec §3 Real-robot fidelity,
Discovery answers)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

SHARED = Path(__file__).resolve().parents[3] / "simulator" / "shared"
for p in (str(SHARED), str(SHARED / "wire")):
    if p not in sys.path:
        sys.path.insert(0, p)


class Msg:
    _slots = ()

    def __init__(self, *a, **kw):
        self.header = types.SimpleNamespace(stamp=None, frame_id="")
        self.data = None
        self.__dict__.update(kw)


class FakeRospy(types.ModuleType):
    def __init__(self, name="rospy"):
        super().__init__(name)
        self.pubs, self.subs, self.services, self.timers, self.proxies = {}, {}, {}, [], []
        self.params = {"/camera": {"camera_name": "camera", "image_topic": "image_raw"}}
        self.node_name = None
        test = self

        class Publisher:
            def __init__(self, topic, cls, queue_size=10, latch=False):
                self.topic, self.sent = topic, []
                test.pubs.setdefault(topic, []).append(self)

            def publish(self, msg):
                self.sent.append(msg)

        class Subscriber:
            def __init__(self, topic, cls, cb, callback_args=None, queue_size=None):
                self.topic, self.cb, self.args, self.alive = topic, cb, callback_args, True
                test.subs.setdefault(topic, []).append(self)

            def unregister(self):
                self.alive = False

        class Service:
            def __init__(self, name, cls, handler):
                test.services[name] = handler

        class Timer:
            def __init__(self, period, cb):
                test.timers.append((period, cb))

        class Duration:
            def __init__(self, s):
                self.s = s

        class ServiceProxy:
            def __init__(self, name, cls):
                self.name = name

            def __call__(self, *a):
                test.proxies.append((self.name, a))

        self.Publisher, self.Subscriber, self.Service = Publisher, Subscriber, Service
        self.Timer, self.Duration, self.ServiceProxy = Timer, Duration, ServiceProxy
        self.Time = types.SimpleNamespace(from_sec=lambda t: t)
        self.ServiceException = type("ServiceException", (Exception,), {})

    def init_node(self, name, **kw):
        self.node_name = "/" + name

    def get_name(self):
        return self.node_name

    def get_param(self, name, default=None):
        return self.params.get(name, default)


@pytest.fixture
def make_node(monkeypatch):
    rospy = FakeRospy()
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    srv_cls = type("Srv", (), {"_response_class": Msg})
    roslib = types.ModuleType("roslib")
    roslib.message = types.SimpleNamespace(get_message_class=lambda t: Msg,
                                           get_service_class=lambda t: srv_cls)
    monkeypatch.setitem(sys.modules, "roslib", roslib)
    monkeypatch.setitem(sys.modules, "roslib.message", roslib.message)
    for mod, attrs in {"std_srvs": {}, "std_srvs.srv": {"SetBoolResponse": Msg,
                                                        "EmptyResponse": Msg},
                       "sensor_msgs": {}, "sensor_msgs.msg": {"Image": Msg},
                       "ainex_interfaces": {},
                       "ainex_interfaces.srv": {"SetWalkingCommand": object}}.items():
        m = types.ModuleType(mod)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, mod, m)
    monkeypatch.setenv("RSIM_ROBOT", "ainex")
    monkeypatch.delitem(sys.modules, "ros1_node", raising=False)
    import ros1_node

    def make(name):
        node = ros1_node.Node(name)
        return node, rospy
    return make


def test_app_set_running_false_stops_the_gait(make_node):
    node, rospy = make_node("/app")
    handler = rospy.services["/app/set_running"]
    res = handler(Msg(data=False))
    assert rospy.proxies == [("/walking/command", ("stop",))]
    assert res.success is False                 # the node answers success = req.data
    assert handler(Msg(data=True)).success is False   # refused in `idle`
    assert len(rospy.proxies) == 1
    for topic in ("/walking/set_param", "/ros_robot_controller/bus_servo/set_position",
                  "/ros_robot_controller/set_buzzer", "/ros_robot_controller/set_rgb",
                  "/color_detection/update_detect"):
        assert topic in node.pub, topic          # recorded internal publishers


def test_sensor_button_stream_follows_enable(make_node):
    node, rospy = make_node("/sensor")
    [(period, fire)] = rospy.timers
    assert abs(period.s - 1 / 50) < 1e-9
    [pub] = rospy.pubs["/sensor/button/get_button_state"]
    fire(None)
    assert len(pub.sent) == 1 and pub.sent[0].data is True     # released (pull-up)
    res = rospy.services["/sensor/button/enable"](Msg(data=False))
    assert res.success is True and res.message == "set_button_enable"
    fire(None)
    assert len(pub.sent) == 1
    rospy.services["/sensor/button/enable"](Msg(data=True))
    fire(None)
    assert len(pub.sent) == 2


def test_detection_image_result_after_enter(make_node):
    """<node>/image_result after enter: the undrawn frame, rgb8, with the recorded frame id
    (the node's name) and stamped with its publish time, as the vendor's cv2_image2ros
    stamps it (the record, amended 2026-10-03); nothing on /object/pixel_coords."""
    import time

    import common

    rows = {t["name"]: t for t in common.interface(common.owner())["topics"]}
    for name in ("/color_detection", "/face_detect"):
        node, rospy = make_node(name)
        assert "/camera/image_raw" not in rospy.subs
        rospy.services[f"{name}/enter"](Msg())
        [sub] = rospy.subs["/camera/image_raw"]
        frame = Msg(height=2, width=3, encoding="rgb8", data=b"\x01" * 18)
        frame.header.stamp = 42.0
        t0 = time.time()
        sub.cb(frame)
        [pub] = rospy.pubs[f"{name}/image_result"]
        out = pub.sent[-1]
        assert (out.height, out.width, out.encoding, out.step) == (2, 3, "rgb8", 9)
        assert out.data == frame.data and t0 <= out.header.stamp <= time.time()
        assert out.header.frame_id == rows[f"{name}/image_result"]["frame_id"] == name.lstrip("/")
        assert not any(p.sent for p in rospy.pubs.get("/object/pixel_coords", []))
        rospy.services[f"{name}/exit"](Msg())
        assert not sub.alive
        rospy.subs.clear()
        rospy.pubs.clear()


def test_joystick_control_discovery(make_node):
    node, rospy = make_node("/joystick_control")
    assert "/walking/set_param" in node.pub
    assert "/joy" in rospy.subs
