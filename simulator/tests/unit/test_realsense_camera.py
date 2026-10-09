"""The reBot Arm B601-RS's simulated Intel RealSense D405 (wire/ros2_realsense.RealSenseD405 and
its plan in wire/robots/rebot_b601_rs.py), without ROS: it publishes exactly the recorded streams
(colour rgb8 and depth aligned to colour, 16UC1 in mm with 0 outside the recorded range, both
with the recorded CameraInfo, all in the colour optical frame), stamped with the sample's
acquisition time, and its plan runs the stock static transform publisher for the recorded mount
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

COLOR = "/camera/camera/color/image_raw"
DEPTH = "/camera/camera/aligned_depth_to_color/image_raw"


class Msg:
    """A message with a header and whatever fields the code sets."""

    def __init__(self, **kw):
        self.header = types.SimpleNamespace(stamp=None, frame_id="")
        self.__dict__.update(kw)


class Image(Msg):
    pass


class CameraInfo(Msg):
    def __init__(self):
        super().__init__(width=0, height=0, distortion_model="", d=[], k=[0.0] * 9,
                         r=[0.0] * 9, p=[0.0] * 12)


class Time:
    def __init__(self, sec=0, nanosec=0):
        self.sec, self.nanosec = sec, nanosec


class Node:
    """An rclpy node's publishing surface."""

    def __init__(self):
        self.sent = {}
        self.created = {}

    def create_publisher(self, typ, topic, qos):
        self.created[topic] = (typ, qos)
        node = self

        class Pub:
            def publish(self, msg):
                node.sent.setdefault(topic, []).append(msg)
        return Pub()


@pytest.fixture
def d405(monkeypatch):
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    qos = lambda **kw: types.SimpleNamespace(**kw)  # noqa: E731
    enum = lambda *names: types.SimpleNamespace(**{n: n for n in names})  # noqa: E731
    module("rclpy")
    module("rclpy.qos", QoSProfile=qos, HistoryPolicy=enum("KEEP_LAST"),
           ReliabilityPolicy=enum("RELIABLE"), DurabilityPolicy=enum("VOLATILE"))
    module("sensor_msgs")
    module("sensor_msgs.msg", Image=Image, CameraInfo=CameraInfo)
    module("builtin_interfaces")
    module("builtin_interfaces.msg", Time=Time)
    monkeypatch.setenv("RSIM_ROBOT", "rebot_b601_rs")

    import common
    import ros2_realsense

    node = Node()
    return node, ros2_realsense.RealSenseD405(node, common.interface()), common.interface()


def frame(depth_m, w=640, h=360, stamp=12.5):
    rgb = np.full((h, w, 3), 128, np.uint8).tobytes()
    depth = np.full((h, w), depth_m, np.float32).tobytes()
    return {"width": w, "height": h, "stamp": stamp}, rgb + depth


def test_publishes_the_recorded_streams_in_the_recorded_frame(d405):
    node, cam, iface = d405
    assert set(node.created) == {COLOR, "/camera/camera/color/camera_info", DEPTH,
                                 "/camera/camera/aligned_depth_to_color/camera_info"}
    for topic, qos in ((t, q[1]) for t, q in node.created.items()):
        assert qos.reliability == "RELIABLE" and qos.durability == "VOLATILE"
    hd, payload = frame(0.3)
    cam.on_frame(hd, payload)
    for topic, encoding, step in ((COLOR, "rgb8", 640 * 3), (DEPTH, "16UC1", 640 * 2)):
        img = node.sent[topic][-1]
        assert (img.height, img.width, img.encoding, img.step) == (360, 640, encoding, step)
        assert img.header.frame_id == "camera_color_optical_frame"
        assert (img.header.stamp.sec, img.header.stamp.nanosec) == (12, 500_000_000)
        assert len(img.data) == step * 360
    assert bytes(node.sent[COLOR][-1].data) == payload[:640 * 360 * 3]
    for topic in ("/camera/camera/color/camera_info",
                  "/camera/camera/aligned_depth_to_color/camera_info"):
        info = node.sent[topic][-1]
        assert info.header.frame_id == "camera_color_optical_frame"
        assert (info.width, info.height) == (640, 360)
        assert (info.k[0], info.k[4], info.k[2], info.k[5]) == (337.2, 337.2, 319.5, 179.5)
        assert info.k[8] == 1.0 and info.r[0] == info.r[4] == info.r[8] == 1.0
        assert info.p[0] == 337.2 and info.p[5] == 337.2 and info.p[10] == 1.0
        assert info.header.stamp.sec == 12


def test_depth_is_millimetres_and_zero_outside_the_recorded_range(d405):
    node, cam, iface = d405
    row = next(c for c in iface["sensors"]["cameras"] if c["image_topic"] == DEPTH)
    assert row["range_basis"] == "manufacturer"
    assert cam.depth_range == tuple(float(x) for x in row["range_m"]) == (0.07, 0.5)
    for metres, want in ((0.3, 300), (0.07, 70), (0.5, 500), (0.05, 0), (0.6, 0),
                         (float("inf"), 0)):
        hd, payload = frame(metres)
        cam.on_frame(hd, payload)
        mm = np.frombuffer(bytes(node.sent[DEPTH][-1].data), np.uint16)
        assert (mm == want).all(), metres


def test_a_frame_goes_out_once_on_every_stream(d405):
    node, cam, _ = d405
    for _ in range(3):
        cam.on_frame(*frame(0.3))
    assert {t: len(m) for t, m in node.sent.items()} == {t: 3 for t in node.created}


def test_camera_helpers_are_shared_by_the_ros1_drivers():
    from robots import _camera, _ros1_drivers

    for name in ("camera_row", "fill_camera_info", "encode"):
        assert getattr(_ros1_drivers, name) is getattr(_camera, name)
    assert _camera.depth_mm(np.array([[0.0, 0.1, 1.2]], np.float32), (0.07, 0.5)).tolist() == \
        [[0, 100, 0]]


def test_the_plan_runs_the_camera_and_the_stock_mount_publisher(monkeypatch):
    monkeypatch.setenv("RSIM_ROBOT", "rebot_b601_rs")
    import common
    import registry
    from robots import rebot_b601_rs

    iface = common.interface()
    plan = rebot_b601_rs.plan(registry.get("rebot_b601_rs"), iface, {})
    procs = dict(plan.procs)
    assert set(procs) == {"bringup", "/camera_mount_tf", "/camera/camera"}
    assert procs["/camera/camera"][-1].endswith("ros2_realsense.py")
    row = next(r for r in iface["tf"] if r["publisher"] == "/camera_mount_tf")
    cmd = procs["/camera_mount_tf"]
    assert cmd[:4] == ["ros2", "run", "tf2_ros", "static_transform_publisher"]
    flags = dict(zip(cmd[4:], cmd[5:]))
    for key, value in zip(("--x", "--y", "--z", "--roll", "--pitch", "--yaw"),
                          list(row["xyz"]) + list(row["rpy"])):
        assert float(flags[key]) == float(value)
    assert flags["--frame-id"] == "gripper_end" and flags["--child-frame-id"] == "camera_link"
    assert cmd[-3:] == ["--ros-args", "-r", "__node:=camera_mount_tf"]


def test_the_model_camera_is_the_recorded_one():
    import mujoco

    import registry

    r = registry.get("rebot_b601_rs")
    model = mujoco.MjModel.from_xml_path(str(r.folder_path / "model.xml"))
    cam = model.camera("camera_color_optical_frame")
    assert [int(v) for v in model.cam_resolution[cam.id]] == [640, 360]
    # the recorded 337.2 px focal length over 360 rows
    assert np.degrees(2 * np.arctan(180 / 337.2)) == pytest.approx(model.cam_fovy[cam.id], abs=1e-6)
    assert model.body(int(model.cam_bodyid[cam.id])).name == "gripper_end"
