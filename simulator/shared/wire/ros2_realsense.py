#!/usr/bin/env python3
"""Intel RealSense D405 wrist camera (ROS 2): the node `/camera/camera` of realsense-ros'
`ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
depth_module.depth_profile:=640x360x30 depth_module.color_profile:=640x360x30`
(robots_specs/rebot_b601_rs/ros2.yml, `camera_boot_authority`).

Serves exactly the recorded, non-optional endpoints of that node: the colour image (rgb8) and
its CameraInfo, the depth aligned to colour (16UC1, millimetres, 0 where the sensor reports no
depth, i.e. outside the recorded working range) and its CameraInfo, all in the colour optical
frame, and the driver's static camera transforms on /tf_static; the recorded parameters with
their recorded values. The rendered model camera (named after the colour optical frame) provides
colour and depth, which are aligned by construction. Stamps are the samples' acquisition times.

    ros2_realsense.py            (inside a wire container, started by robots/rebot_b601_rs.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
from robots._camera import camera_row, depth_mm, encode, fill_camera_info  # noqa: E402

NODE = "/camera/camera"
COLOR_IMAGE = "/camera/camera/color/image_raw"
DEPTH_IMAGE = "/camera/camera/aligned_depth_to_color/image_raw"


def _time(stamp: float):
    from builtin_interfaces.msg import Time

    sec = int(stamp)
    return Time(sec=sec, nanosec=int(round((stamp - sec) * 1e9)) % 1_000_000_000)


class RealSenseD405:
    """The camera's behaviour on an rclpy `node` (created as `/camera/camera`) for the interface
    `iface`: publishers for the recorded streams, one callback per rendered frame."""

    def __init__(self, node, iface: dict):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        self.node = node
        self.color = camera_row(iface, COLOR_IMAGE)
        self.depth = camera_row(iface, DEPTH_IMAGE)
        self.depth_range = tuple(float(x) for x in self.depth["range_m"])
        # the driver's image QoS is SYSTEM_DEFAULT and its camera info DEFAULT: reliable, volatile
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE)
        info_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                              reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        self.pubs = {}
        self.infos = {}
        for cam in (self.color, self.depth):
            self.pubs[cam["image_topic"]] = node.create_publisher(Image, cam["image_topic"], image_qos)
            self.pubs[cam["info_topic"]] = node.create_publisher(CameraInfo, cam["info_topic"], info_qos)
            self.infos[cam["info_topic"]] = fill_camera_info(CameraInfo(), cam)
            self.infos[cam["info_topic"]].header.frame_id = cam["frame_id"]

    def start(self, link):
        c = self.color
        link.subscribe("camera", float(c["rate"]), self.on_frame, camera=c["frame_id"],
                       width=int(c["width"]), height=int(c["height"]), rgb=True, depth=True)

    def _publish(self, cam, stamp, data, encoding, step, w, h):
        from sensor_msgs.msg import Image

        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = cam["frame_id"]
        img.height, img.width, img.encoding, img.is_bigendian, img.step = h, w, encoding, 0, step
        img.data = data
        self.pubs[cam["image_topic"]].publish(img)
        info = self.infos[cam["info_topic"]]
        info.header.stamp = stamp
        self.pubs[cam["info_topic"]].publish(info)

    def on_frame(self, hd, payload):
        """One rendered frame: width x height rgb (uint8) then depth (float32, m)."""
        w, h = hd["width"], hd["height"]
        n = w * h
        stamp = _time(hd["stamp"])
        rgb = np.frombuffer(payload[:n * 3], np.uint8).reshape(h, w, 3)
        depth = np.frombuffer(payload[n * 3:n * 3 + n * 4], np.float32).reshape(h, w)
        data, step = encode(rgb, self.color["encoding"])
        self._publish(self.color, stamp, data, self.color["encoding"], step, w, h)
        self._publish(self.depth, stamp, depth_mm(depth, self.depth_range).tobytes(),
                      self.depth["encoding"], w * 2, w, h)


def declare_parameters(node, iface: dict) -> None:
    """The recorded parameters of the node with their recorded values (the QoS override
    parameters of the /tf_static broadcaster are declared by its publisher)."""
    for row in common.params_of(iface, NODE):
        if row["name"].startswith("qos_overrides."):
            continue
        node.declare_parameter(row["name"], row["value"])


def static_transforms(node, iface: dict):
    """The driver's static transforms (every recorded non-optional `tf` row whose publisher is
    this node) on /tf_static, once, latched (transient local, as tf2_ros'
    StaticBroadcasterQoS); the publisher declares the QoS override parameters of the
    broadcaster."""
    from geometry_msgs.msg import TransformStamped
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)
    from rclpy.qos_overriding_options import QoSOverridingOptions, QoSPolicyKind
    from tf2_msgs.msg import TFMessage

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(TFMessage, "/tf_static", qos, qos_overriding_options=
                                QoSOverridingOptions(policy_kinds=(
                                    QoSPolicyKind.DEPTH, QoSPolicyKind.HISTORY,
                                    QoSPolicyKind.RELIABILITY)))
    stamp = node.get_clock().now().to_msg()
    msg = TFMessage()
    for row in common.served(iface.get("tf")):
        if row.get("publisher") != NODE:
            continue
        ts = TransformStamped()
        ts.header.stamp = stamp
        ts.header.frame_id, ts.child_frame_id = row["parent"], row["child"]
        x, y, z = row["xyz"]
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = x, y, z
        qx, qy, qz, qw = common.rpy_to_quat(*row["rpy"])
        ts.transform.rotation.x, ts.transform.rotation.y = qx, qy
        ts.transform.rotation.z, ts.transform.rotation.w = qz, qw
        msg.transforms.append(ts)
    pub.publish(msg)
    return pub


def main() -> int:
    import rclpy

    iface = common.interface()
    rclpy.init()
    node = rclpy.create_node("camera", namespace="camera")
    declare_parameters(node, iface)
    static_transforms(node, iface)
    cam = RealSenseD405(node, iface)
    link = common.SimLink(on_lost=lambda: None)
    common.exit_when_lost(link)
    cam.start(link)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
