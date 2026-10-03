"""ROSMASTER X3 PLUS (ROS 1 Noetic): the wire of yahboomcar_nav/laser_astrapro_bringup.launch.

Real, unchanged: the vendor's driver_node (Mcnamu_X3plus.py, against the simulated
expansion board `sim_libs/Rosmaster_Lib.py`), odometry_publisher (base_node) and
yahboom_joy, built from the pinned yahboomcar_ws.zip; the stock robot_state_publisher,
imu_filter_madgwick, robot_localization EKF, joy_node and the static transform
publishers, remapped and parameterised as the launch files do. Simulated: the YDLIDAR
4ROS (/ydlidar_lidar_publisher) and the Orbbec Astra Pro Plus (/camera/camera).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

import common
from robots import _plan
from robots._ros1_drivers import YdLidar, camera_row, encode, fill_camera_info, tf_message

LIBS = Path(__file__).resolve().parent / "sim_libs"


def _vendor(script: str, package: str, name: str, *remaps: str, libs: bool = False) -> list:
    pre = f"PYTHONPATH={LIBS}:$PYTHONPATH " if libs else ""
    args = " ".join([f"__name:={name}", *remaps])
    return ["bash", "-c", f"{pre}exec python3 $(rospack find {package})/scripts/{script} {args}"]


def plan(robot, iface, describe):
    p = _plan.Plan()
    # The EKF's covariance matrices are recorded as generated from the vendor's
    # robot_localization.yaml (loaded by bringup.launch with rosparam): load that file.
    import yaml

    rl = Path("/opt/rsim_ws/src/yahboomcar_bringup/param/robot_localization.yaml")
    if rl.is_file():
        values = yaml.safe_load(rl.read_text()) or {}
        for key in ("process_noise_covariance", "initial_estimate_covariance"):
            if key in values:
                p.params[f"/ekf_localization/{key}"] = values[key]
    p.add("/robot_state_publisher", ["rosrun", "robot_state_publisher", "robot_state_publisher",
                                     *_plan.ros1_name("/robot_state_publisher")])
    p.add("/driver_node", _vendor("Mcnamu_X3plus.py", "yahboomcar_bringup", "driver_node",
                                  "/pub_vel:=/vel_raw", "/pub_imu:=/imu/data_raw",
                                  "/pub_mag:=/mag/mag_raw", libs=True))
    p.add("/odometry_publisher", ["rosrun", "yahboomcar_bringup", "base_node",
                                  "__name:=odometry_publisher", "/sub_vel:=/vel_raw",
                                  "/pub_odom:=/odom_raw"])
    p.add("/imu_filter_madgwick", ["rosrun", "imu_filter_madgwick", "imu_filter_node",
                                   "__name:=imu_filter_madgwick", "/sub_imu:=/imu/imu_raw",
                                   "/sub_mag:=/mag/mag_raw", "/pub_imu:=/imu/imu_data",
                                   "/pub_mag:=/mag/mag_field"])
    p.add("/ekf_localization", ["rosrun", "robot_localization", "ekf_localization_node",
                                "__name:=ekf_localization", "odometry/filtered:=odom",
                                "/imu0:=/imu/data", "/odom0:=odom_raw"])
    p.add("/joy_node", ["rosrun", "joy", "joy_node", "__name:=joy_node"])
    p.add("/yahboom_joy", _vendor("yahboom_joy.py", "yahboomcar_ctrl", "yahboom_joy"))
    _plan.ros1_static_tf(p, iface)
    for node in ("/ydlidar_lidar_publisher", "/camera/camera"):
        _plan.ros1_emulated(p, node)
    return p


class AstraProPlus:
    """/camera/camera (orbbec_camera_node): colour, registered depth (16UC1, mm) and IR
    (mono16) at 30 Hz with their CameraInfo, the depth and coloured point clouds, and
    the driver's own camera transforms on /tf at tf_publish_rate. The rendered model
    camera (named after the colour frame) provides colour and depth, with depth returns
    outside the recorded range reported as 0; IR is the scene's luminance scaled to the
    sensor's 10-bit range (an estimate: IR is not rendered).

    Its services answer as the driver's do (ros_service.cpp): get_<stream>_camera_info
    returns the CameraInfo the stream publishes, get_camera_params the same recorded
    intrinsics (depth on the left, colour on the right; with depth registered to colour,
    as recorded, the extrinsic between them is identity), and toggle_<stream> stops or
    restarts that stream (and the point clouds built from it); toggling a stream to the
    state it is in fails with the driver's message. The other services keep their default
    answers."""

    def __init__(self, node):
        self.node = node
        iface = node.iface
        self.rgb = camera_row(iface, "/camera/rgb/image_raw")
        self.depth = camera_row(iface, "/camera/depth/image_raw")
        self.ir = camera_row(iface, "/camera/ir/image_raw")
        self.depth_range = tuple(float(x) for x in self.depth["range_m"])
        self.publishes = {"/camera/rgb/image_raw", "/camera/rgb/camera_info",
                          "/camera/depth/image_raw", "/camera/depth/camera_info",
                          "/camera/ir/image_raw", "/camera/ir/camera_info",
                          "/camera/depth/points", "/camera/depth_registered/points", "/tf"}
        self.infos = {}
        for cam in (self.rgb, self.depth, self.ir):
            self.infos[cam["info_topic"]] = fill_camera_info(node.new(cam["info_topic"]), cam)
        self.streams = {"color": self.rgb, "depth": self.depth, "ir": self.ir}
        self.enabled = {s: True for s in self.streams}
        self.tf_rows = [r for r in iface.get("tf", []) if r.get("publisher") == node.name
                        and not r.get("optional")]
        self.tf_rate = float(common.param(iface, "/camera/camera/tf_publish_rate", 10.0))
        intr = self.rgb["intrinsics"]
        w, h = int(self.rgb["width"]), int(self.rgb["height"])
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.ray_x = (u - float(intr["cx"])) / float(intr["fx"])
        self.ray_y = (v - float(intr["cy"])) / float(intr["fy"])

    def start(self):
        c = self.rgb
        self.node.link.subscribe("camera", float(c["rate"]), self._frame, camera=c["frame_id"],
                                 width=int(c["width"]), height=int(c["height"]), rgb=True,
                                 depth=True)
        threading.Thread(target=self._tf_loop, daemon=True).start()

    def _tf_loop(self):
        import rospy

        period = 1.0 / self.tf_rate
        while not rospy.is_shutdown():
            t0 = time.time()
            self.node.publish("/tf", tf_message(self.tf_rows, t0))
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _image(self, topic, cam, stamp, data, encoding, step, w, h):
        img = self.node.new(topic, stamp=stamp)
        img.header.frame_id = cam["frame_id"]
        img.height, img.width, img.encoding, img.is_bigendian, img.step = h, w, encoding, 0, step
        img.data = data
        self.node.publish(topic, img)
        info = self.infos[cam["info_topic"]]
        self.node.fill_header(info, cam["info_topic"], stamp)
        self.node.publish(cam["info_topic"], info)

    def _frame(self, hd, payload):
        w, h = hd["width"], hd["height"]
        stamp = hd["stamp"]
        n = w * h
        on = dict(self.enabled)
        rgb = np.frombuffer(payload[:n * 3], np.uint8).reshape(h, w, 3)
        depth = np.frombuffer(payload[n * 3:n * 3 + n * 4], np.float32).reshape(h, w)
        lo, hi = self.depth_range
        valid = (depth >= lo) & (depth <= hi)
        if on["color"]:
            data, step = encode(rgb, "rgb8")
            self._image("/camera/rgb/image_raw", self.rgb, stamp, data, "rgb8", step, w, h)
        if on["depth"]:
            mm = np.where(valid, np.round(depth * 1000.0), 0).astype(np.uint16)
            self._image("/camera/depth/image_raw", self.depth, stamp, mm.tobytes(), "16UC1",
                        w * 2, w, h)
        if on["ir"]:
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            ir = (lum * (1023.0 / 255.0)).astype(np.uint16)
            self._image("/camera/ir/image_raw", self.ir, stamp, ir.tobytes(), "mono16", w * 2, w, h)
        if not on["depth"]:
            return
        z = np.where(valid, depth, 0.0).astype(np.float32)
        x, y = self.ray_x * z, self.ray_y * z
        sel = valid.ravel()
        xyz = np.stack([x.ravel()[sel], y.ravel()[sel], z.ravel()[sel]], axis=1).astype(np.float32)
        self._cloud("/camera/depth/points", "camera_depth_optical_frame", stamp, xyz, None)
        if not on["color"]:
            return
        cols = rgb.reshape(-1, 3)[sel].astype(np.uint32)
        packed = ((cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2]).view(np.float32)
        self._cloud("/camera/depth_registered/points", "camera_color_optical_frame", stamp,
                    xyz, packed)

    def on_service(self, name, req):
        base = name.rpartition("/")[2]
        if base.startswith("toggle_") and base[len("toggle_"):] in self.streams:
            import rospy
            from std_srvs.srv import SetBoolResponse

            stream, want = base[len("toggle_"):], bool(req.data)
            if self.enabled[stream] == want:
                # the driver's callback returns false: the call fails
                raise rospy.ServiceException(f"{stream} Already {'ON' if want else 'OFF'}")
            self.enabled[stream] = want
            return SetBoolResponse(success=True, message="")
        if base.startswith("get_") and base.endswith("_camera_info") and \
                base[len("get_"):-len("_camera_info")] in self.streams:
            from orbbec_camera.srv import GetCameraInfoResponse
            from sensor_msgs.msg import CameraInfo

            # the driver answers from the device's colour intrinsics for colour and its
            # depth intrinsics for depth and IR, with no header
            cam = self.rgb if base == "get_color_camera_info" else self.depth
            return GetCameraInfoResponse(info=fill_camera_info(CameraInfo(), cam),
                                         success=True, message="")
        if base == "get_camera_params":
            from orbbec_camera.srv import GetCameraParamsResponse

            def intr(cam):
                i = cam["intrinsics"]
                return [float(i["fx"]), float(i["fy"]), float(i["cx"]), float(i["cy"])]

            # depth (left) and colour (right) intrinsics; depth is registered to colour
            # (both recorded in camera_color_optical_frame), so the extrinsic is identity
            return GetCameraParamsResponse(
                l_intr_p=intr(self.depth), r_intr_p=intr(self.rgb),
                r2l_r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], r2l_t=[0.0, 0.0, 0.0],
                success=True, message="")
        return None

    def _cloud(self, topic, frame, stamp, xyz, rgb):
        from sensor_msgs.msg import PointField

        pc = self.node.new(topic, stamp=stamp)
        pc.header.frame_id = frame
        pc.height, pc.width = 1, int(xyz.shape[0])
        fields = [PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1),
                  PointField("z", 8, PointField.FLOAT32, 1)]
        if rgb is not None:
            fields.append(PointField("rgb", 12, PointField.FLOAT32, 1))
            arr = np.empty((xyz.shape[0], 4), np.float32)
            arr[:, :3] = xyz
            arr[:, 3] = rgb
        else:
            arr = xyz
        pc.fields = fields
        pc.is_bigendian = False
        pc.point_step = 16 if rgb is not None else 12
        pc.row_step = pc.point_step * pc.width
        pc.is_dense = True
        pc.data = arr.tobytes()
        self.node.publish(topic, pc)


BEHAVIOURS = {
    "/ydlidar_lidar_publisher": YdLidar,
    "/camera/camera": AstraProPlus,
}
