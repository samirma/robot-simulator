"""Simulated ROS 1 drivers shared by several robots: a USB camera (usb_cam 0.3.x) and a
YDLIDAR (ydlidar_ros_driver). Each is one node's behaviour for `ros1_node.py`, fed by the
simulation's camera and lidar streams and publishing with the sample's acquisition time.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

import common
from robots._camera import camera_row, encode, fill_camera_info  # noqa: F401  (re-exported)


def tf_message(rows, stamp: float):
    """A tf2_msgs/TFMessage of the recorded fixed transforms `rows` (parent, child, xyz,
    rpy), every one stamped `stamp` (s)."""
    import rospy
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage

    msg = TFMessage()
    for r in rows:
        ts = TransformStamped()
        ts.header.stamp = rospy.Time.from_sec(stamp)
        ts.header.frame_id, ts.child_frame_id = r["parent"], r["child"]
        x, y, z = r.get("xyz", [0, 0, 0])
        ts.transform.translation.x, ts.transform.translation.y = x, y
        ts.transform.translation.z = z
        qx, qy, qz, qw = common.rpy_to_quat(*r.get("rpy", [0, 0, 0]))
        ts.transform.rotation.x, ts.transform.rotation.y = qx, qy
        ts.transform.rotation.z, ts.transform.rotation.w = qz, qw
        msg.transforms.append(ts)
    return msg


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

        period = 1.0 / float(self.rows[0]["rate"])

        def fire(_):
            self.node.publish("/tf", tf_message(self.rows, time.time() + period))

        self._timer = rospy.Timer(rospy.Duration(period), fire)


class UsbCam:
    """usb_cam: the recorded camera's raw image and CameraInfo, from the simulation's
    rendering of the model camera named after the image frame id.

    Its CameraInfo is camera_info_manager's: the recorded calibration at start, and from a
    `set_camera_info` call on, the calibration that call sent -- camera_info_manager stores
    the request's camera_info (setCameraInfoService), answers success once it has saved it,
    and usb_cam publishes it with every frame, header set to the frame's."""

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
        c = self.cam
        rgb = np.frombuffer(payload, np.uint8).reshape(h["height"], h["width"], 3)
        data, step = encode(rgb, c["encoding"])
        img = self.node.new(c["image_topic"], stamp=h["stamp"])
        img.header.frame_id = c["frame_id"]
        img.height, img.width, img.encoding, img.is_bigendian, img.step = \
            h["height"], h["width"], c["encoding"], 0, step
        img.data = data
        self.node.publish(c["image_topic"], img)
        info = self.info
        if info is not None:
            self.node.fill_header(info, c["info_topic"], h["stamp"])
            self.node.publish(c["info_topic"], info)

    def on_service(self, name, req):
        if name.endswith("stop_capture"):
            self.capturing = False
        elif name.endswith("start_capture"):
            self.capturing = True
        elif name.endswith("/set_camera_info") and self.info is not None:
            return self._set_camera_info(req)
        return None

    def _set_camera_info(self, req):
        """camera_info_manager's setCameraInfoService: the request's camera_info becomes
        the published calibration, every field of it (the header is the frame's when
        published); the answer is success, as once the calibration is saved."""
        from sensor_msgs.srv import SetCameraInfoResponse

        info = self.node.new(self.cam["info_topic"])
        sent = req.camera_info
        for field in getattr(sent, "__slots__", None) or vars(sent):
            if field != "header":
                setattr(info, field, copy.deepcopy(getattr(sent, field)))
        self.info = info      # one reference swap: a frame publishes the old or the new one
        return SetCameraInfoResponse(success=True, status_message="")


class YdLidar:
    """ydlidar_ros_driver: /scan and /point_cloud from the simulation's planar ray
    cast at the lidar site, with the driver's recorded geometry and conventions
    (ignore_array sectors and invalid returns reported as recorded).

    Intensities as the record's lidar row states them (`intensity`: the value at a return
    and where there is none): `intensities` as many as `ranges`, the valid value where a
    range within range_min..range_max is reported, and each /point_cloud point's
    `intensities` channel entry the valid value; a record that states none gets none."""

    def __init__(self, node):
        self.node = node
        rows = [r for r in (node.iface.get("sensors") or {}).get("lidars") or [] if not r.get("optional")]
        self.row = rows[0]
        self.publishes = {self.row["topic"], "/point_cloud"}
        me = node.name
        self.inf_invalid = bool(common.param(node.iface, f"{me}/invalid_range_is_inf", False))
        ign = str(common.param(node.iface, f"{me}/ignore_array", "") or "")
        vals = [float(x) for x in ign.replace(" ", "").split(",") if x]
        self.ignore = [(math.radians(vals[i]), math.radians(vals[i + 1]))
                       for i in range(0, len(vals) - 1, 2)]
        self.scanning = True
        self.intensity = self.row.get("intensity")

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
        if self.intensity is None:
            msg.intensities = []
        else:
            ok = self._returns(out)
            msg.intensities = np.where(ok, float(self.intensity["valid"]),
                                       float(self.intensity["invalid"])).astype(np.float32).tolist()
        self.node.publish(r["topic"], msg)
        if "/point_cloud" in self.node.pub:
            self._cloud(h, ang, out)

    def _returns(self, ranges):
        """The indices where the scan reports a range (within range_min..range_max)."""
        r = self.row
        return np.isfinite(ranges) & (ranges >= float(r["range_min"])) & \
            (ranges <= float(r["range_max"]))

    def _cloud(self, h, ang, ranges):
        r = self.row
        ok = self._returns(ranges)
        pc = self.node.new("/point_cloud", stamp=h["stamp"])
        pc.header.frame_id = r["frame_id"]
        from geometry_msgs.msg import Point32
        from sensor_msgs.msg import ChannelFloat32

        xs, ys = ranges[ok] * np.cos(ang[ok]), ranges[ok] * np.sin(ang[ok])
        pc.points = [Point32(float(x), float(y), 0.0) for x, y in zip(xs, ys)]
        period = 1.0 / float(r["scan_rate"])
        idx = np.nonzero(ok)[0]
        level = float(self.intensity["valid"]) if self.intensity is not None else 0.0
        pc.channels = [ChannelFloat32("intensities", [level] * len(idx)),
                       ChannelFloat32("stamps", [float(i * period / self.n) for i in idx])]
        self.node.publish("/point_cloud", pc)

    def on_service(self, name, req):
        if name == "/stop_scan":
            self.scanning = False
        elif name == "/start_scan":
            self.scanning = True
        return None
