"""Simulated ROS 1 drivers shared by several robots: a USB camera (usb_cam 0.3.x) and a
YDLIDAR (ydlidar_ros_driver). Each is one node's behaviour for `ros1_node.py`, fed by the
simulation's camera and lidar streams and publishing with the sample's acquisition time.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

import common


def camera_row(iface: dict, image_topic: str) -> dict:
    for c in (iface.get("sensors") or {}).get("cameras") or []:
        if c.get("image_topic") == image_topic:
            return c
    raise KeyError(image_topic)


def fill_camera_info(info, cam: dict):
    info.width, info.height = int(cam["width"]), int(cam["height"])
    intr = cam.get("intrinsics") or {}
    fx, fy = float(intr.get("fx", 0.0)), float(intr.get("fy", 0.0))
    cx, cy = float(intr.get("cx", 0.0)), float(intr.get("cy", 0.0))
    info.distortion_model = intr.get("distortion_model", "") or ""
    d = [float(x) for x in intr.get("d") or []]
    kname = "K" if hasattr(info, "K") else "k"
    setattr(info, "D" if hasattr(info, "D") else "d", d)
    setattr(info, kname, [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 if fx else 0.0])
    setattr(info, "R" if hasattr(info, "R") else "r",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] if fx else [0.0] * 9)
    setattr(info, "P" if hasattr(info, "P") else "p",
            [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0 if fx else 0.0, 0.0])
    return info


def encode(rgb: np.ndarray, encoding: str):
    """(bytes, step) of an RGB frame in the recorded encoding."""
    h, w, _ = rgb.shape
    if encoding == "rgb8":
        return rgb.tobytes(), w * 3
    if encoding == "bgr8":
        return rgb[:, :, ::-1].tobytes(), w * 3
    if encoding in ("mono8",):
        g = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
        return g.tobytes(), w
    if encoding in ("yuv422_yuy2", "yuyv"):
        f = rgb.astype(np.float32)
        y = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
        u = -0.169 * f[:, :, 0] - 0.331 * f[:, :, 1] + 0.5 * f[:, :, 2] + 128
        v = 0.5 * f[:, :, 0] - 0.419 * f[:, :, 1] - 0.081 * f[:, :, 2] + 128
        out = np.empty((h, w * 2), np.uint8)
        out[:, 0::4] = np.clip(y[:, 0::2], 0, 255)
        out[:, 1::4] = np.clip((u[:, 0::2] + u[:, 1::2]) / 2, 0, 255)
        out[:, 2::4] = np.clip(y[:, 1::2], 0, 255)
        out[:, 3::4] = np.clip((v[:, 0::2] + v[:, 1::2]) / 2, 0, 255)
        return out.tobytes(), w * 2
    raise ValueError(f"unsupported encoding {encoding}")


class StaticTransformPublisher:
    """tf's static_transform_publisher (ROS 1): the node's recorded transform on /tf at its
    recorded period, frame ids exactly as given (leading slash kept), each stamp dated one
    period ahead as the stock program does. Its loop is timed on a fixed schedule so the
    recorded rate holds (the stock loop sleeps a full period after each send)."""

    publishes = ("/tf",)

    def __init__(self, node):
        self.node = node
        self.rows = [r for r in node.iface.get("tf", []) if r.get("publisher") == node.name
                     and not r.get("optional")]

    def start(self):
        import rospy
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage

        rate = float(self.rows[0]["rate"])
        period = 1.0 / rate

        def fire(_):
            msg = TFMessage()
            stamp = rospy.Time.from_sec(time.time() + period)
            for r in self.rows:
                ts = TransformStamped()
                ts.header.stamp = stamp
                ts.header.frame_id, ts.child_frame_id = r["parent"], r["child"]
                x, y, z = r.get("xyz", [0, 0, 0])
                ts.transform.translation.x, ts.transform.translation.y = x, y
                ts.transform.translation.z = z
                qx, qy, qz, qw = common.rpy_to_quat(*r.get("rpy", [0, 0, 0]))
                ts.transform.rotation.x, ts.transform.rotation.y = qx, qy
                ts.transform.rotation.z, ts.transform.rotation.w = qz, qw
                msg.transforms.append(ts)
            self.node.publish("/tf", msg)

        self._timer = rospy.Timer(rospy.Duration(period), fire)


class UsbCam:
    """usb_cam: the recorded camera's raw image and CameraInfo, from the simulation's
    rendering of the model camera named after the image frame id."""

    image_topic = None

    def __init__(self, node, image_topic=None):
        self.node = node
        cams = [c for c in (node.iface.get("sensors") or {}).get("cameras") or []
                if not c.get("optional")]
        self.cam = camera_row(node.iface, image_topic or self.image_topic) if (
            image_topic or self.image_topic) else cams[0]
        self.publishes = {self.cam["image_topic"], self.cam.get("info_topic")}
        self.capturing = True
        self.info = fill_camera_info(node.new(self.cam["info_topic"]), self.cam) \
            if self.cam.get("info_topic") else None

    def start(self):
        c = self.cam
        self.node.link.subscribe("camera", float(c["rate"]), self._frame, camera=c["frame_id"],
                                 width=int(c["width"]), height=int(c["height"]), rgb=True)

    def _frame(self, h, payload):
        if not self.capturing:
            return
        if common.env("RSIM_PROFILE"):
            import time as _t
            self._n = getattr(self, "_n", 0) + 1
            if self._n % 60 == 0:
                now = _t.time()
                print(f"[usb_cam] 60 frames in {now - getattr(self, '_t0', now):.2f} s, "
                      f"latency {now - h['stamp']:.3f} s", flush=True)
                self._t0 = now
        c = self.cam
        rgb = np.frombuffer(payload, np.uint8).reshape(h["height"], h["width"], 3)
        data, step = encode(rgb, c["encoding"])
        img = self.node.new(c["image_topic"], stamp=h["stamp"])
        img.header.frame_id = c["frame_id"]
        img.height, img.width, img.encoding, img.is_bigendian, img.step = \
            h["height"], h["width"], c["encoding"], 0, step
        if getattr(self.node, "dialect", "ros1") == "ros2":
            import array

            img.data = array.array("B", data)
        else:
            img.data = data
        self.node.publish(c["image_topic"], img)
        if self.info is not None:
            self.node.fill_header(self.info, c["info_topic"], h["stamp"])
            self.node.publish(c["info_topic"], self.info)

    def on_service(self, name, req):
        if name.endswith("stop_capture"):
            self.capturing = False
        elif name.endswith("start_capture"):
            self.capturing = True
        return None


def _param(node, name, default=None):
    for p in node.iface.get("parameters") or []:
        if p.get("name") == name:
            return p.get("value")
    return default


class YdLidar:
    """ydlidar_ros_driver: /scan and /point_cloud from the simulation's planar ray
    cast at the lidar site, with the driver's recorded geometry and conventions
    (ignore_array sectors and invalid returns reported as recorded)."""

    def __init__(self, node):
        self.node = node
        rows = [r for r in (node.iface.get("sensors") or {}).get("lidars") or [] if not r.get("optional")]
        self.row = rows[0]
        self.publishes = {self.row["topic"], "/point_cloud"}
        me = node.name
        self.inf_invalid = bool(_param(node, f"{me}/invalid_range_is_inf", False))
        ign = str(_param(node, f"{me}/ignore_array", "") or "")
        vals = [float(x) for x in ign.replace(" ", "").split(",") if x]
        self.ignore = [(math.radians(vals[i]), math.radians(vals[i + 1]))
                       for i in range(0, len(vals) - 1, 2)]
        self.scanning = True

    def start(self):
        r = self.row
        self.n = int(r["samples"])
        self.node.link.subscribe("lidar", float(r["scan_rate"]), self._scan,
                                 site=r["frame_id"].lstrip("/"), angle_min=float(r["angle_min"]),
                                 angle_max=float(r["angle_max"]), samples=self.n,
                                 range_min=float(r["range_min"]), range_max=float(r["range_max"]))

    def _scan(self, h, payload):
        if not self.scanning:
            return
        r = self.row
        ranges = np.frombuffer(payload, np.float32).astype(np.float64)
        amin, amax = float(r["angle_min"]), float(r["angle_max"])
        inc = (amax - amin) / max(self.n - 1, 1)
        ang = amin + inc * np.arange(self.n)
        invalid = ~np.isfinite(ranges)
        out = np.where(invalid, math.inf if self.inf_invalid else 0.0, ranges)
        for lo, hi in self.ignore:
            out[(ang >= lo) & (ang <= hi)] = 0.0
        period = 1.0 / float(r["scan_rate"])
        msg = self.node.new(r["topic"], stamp=h["stamp"])
        msg.header.frame_id = r["frame_id"]
        msg.angle_min, msg.angle_max, msg.angle_increment = amin, amax, inc
        msg.scan_time = period
        msg.time_increment = period / self.n
        msg.range_min, msg.range_max = float(r["range_min"]), float(r["range_max"])
        msg.ranges = out.astype(np.float32).tolist()
        msg.intensities = []
        self.node.publish(r["topic"], msg)
        if "/point_cloud" in self.node.pub:
            self._cloud(h, ang, out)

    def _cloud(self, h, ang, ranges):
        r = self.row
        ok = np.isfinite(ranges) & (ranges >= float(r["range_min"])) & (ranges <= float(r["range_max"]))
        pc = self.node.new("/point_cloud", stamp=h["stamp"])
        pc.header.frame_id = r["frame_id"]
        from geometry_msgs.msg import Point32
        from sensor_msgs.msg import ChannelFloat32

        xs, ys = ranges[ok] * np.cos(ang[ok]), ranges[ok] * np.sin(ang[ok])
        pc.points = [Point32(float(x), float(y), 0.0) for x, y in zip(xs, ys)]
        period = 1.0 / float(r["scan_rate"])
        idx = np.nonzero(ok)[0]
        pc.channels = [ChannelFloat32("intensities", [0.0] * len(idx)),
                       ChannelFloat32("stamps", [float(i * period / self.n) for i in idx])]
        self.node.publish("/point_cloud", pc)

    def on_service(self, name, req):
        if name == "/stop_scan":
            self.scanning = False
        elif name == "/start_scan":
            self.scanning = True
        return None
