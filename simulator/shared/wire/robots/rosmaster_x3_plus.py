"""ROSMASTER X3 PLUS (ROS 1 Noetic): the wire of yahboomcar_nav/laser_astrapro_bringup.launch
and its camera boot, the arm camera's usb_cam/usb_cam-test.launch as Yahboom's tutorials write it.

Real, unchanged: the vendor's driver_node (Mcnamu_X3plus.py, against the simulated
expansion board `sim_libs/Rosmaster_Lib.py`), odometry_publisher (base_node) and
yahboom_joy, built from the pinned yahboomcar_ws.zip; the stock robot_state_publisher,
imu_filter_madgwick, robot_localization EKF, joy_node, web_video_server and the static
transform publishers, remapped and parameterised as the launch files do. Simulated: the
YDLIDAR 4ROS (/ydlidar_lidar_publisher), the Orbbec Astra Pro Plus (/camera/camera) and
the arm's USB camera (/usb_cam).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

import common
from robots import _plan
from robots._camera import camera_row, encode, fill_camera_info
from robots._ros1_drivers import UsbCam, YdLidar, tf_message

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
    # the camera boot, usb_cam-test.launch as the tutorials write it: usb_cam and
    # web_video_server (an HTTP server inside the wire's container; its port is not a ROS
    # interface and is not published), image_view only with open_view:=true (optional)
    p.add("/web_video_server", ["rosrun", "web_video_server", "web_video_server",
                                *_plan.ros1_name("/web_video_server")])
    for node in ("/ydlidar_lidar_publisher", "/camera/camera", "/usb_cam"):
        _plan.ros1_emulated(p, node)
    return p


class ArmCamera(UsbCam):
    """/usb_cam (usb_cam 0.3.7): the arm's wrist camera, the model camera `usb_cam` on
    mono_link, rgb8 640x480 at its 30 Hz loop, with camera_info_manager's uncalibrated
    CameraInfo; start_capture / stop_capture and set_camera_info as usb_cam answers them.
    The arm camera is taken to be the launch's /dev/video0: an ESTIMATE, as no source
    establishes which device that is while the Astra is attached (ros.yml camera row).

    usb_cam captures whether or not anyone listens; a frame nobody subscribes to reaches no
    one, so the simulation renders this camera only while usb_cam captures and one of its
    topics has a subscriber (rendering it continuously costs the simulation process about
    80% of a CPU core at 640x480, measured, and pushes the timing checks over their bounds
    on a shared host). Rendering starts on a subscriber's connection (rospy's
    SubscribeListener) on a stream warmed at start. A new subscriber's first frame comes
    later than from the real camera, which sends the next captured frame (at most one period,
    33 ms): measured 36-48 ms (1.1-1.4 periods) when subscribing again within seconds, and up
    to about 150 ms (4.4 periods) after a long idle.

    Its CameraPublisher carries the image_transport plugins the record serves on
    `<image>/<plugin>` (compressed, compressedDepth, theora), each with its stock
    dynamic_reconfigure server in its namespace, its initial config the record's values, and
    each encoding only while its own topic has a subscriber, as image_transport publishes:
    compressed the frame as compressed_image_transport encodes it ("rgb8; jpeg compressed
    bgr8", or png); compressedDepth nothing for an rgb8 frame but the plugin's ROS_ERROR on
    /rosout (codec.cpp L390-L393); theora libtheora's packets as theora_image_transport makes
    them (`_theora.TheoraPublisher`), its stream header to each new subscriber on connecting
    (connectCallback)."""

    image_topic = "/usb_cam/image_raw"
    CONFIGS = {"compressed": ("compressed_image_transport.cfg", "CompressedPublisherConfig"),
               "compressedDepth": ("compressed_depth_image_transport.cfg",
                                   "CompressedDepthPublisherConfig"),
               "theora": ("theora_image_transport.cfg", "TheoraPublisherConfig")}

    def __init__(self, node):
        super().__init__(node)
        base = self.cam["image_topic"]
        rows = {t["name"]: t for t in common.served(node.iface.get("topics"))}
        # the plugins the record serves on this camera, and their recorded initial config
        self.plugins = {p: f"{base}/{p}" for p in self.CONFIGS if f"{base}/{p}" in rows}
        self.config = {p: {r["name"].rsplit("/", 1)[1]: r["value"]
                           for r in common.served(node.iface.get("parameters"))
                           if r["name"].rpartition("/")[0] == t}
                       for p, t in self.plugins.items()}
        self.publishes = set(self.publishes) | set(self.plugins.values()) | {
            f"{t}/{leaf}" for t in self.plugins.values()
            for leaf in ("parameter_descriptions", "parameter_updates")}
        self.serves = {f"{t}/set_parameters" for t in self.plugins.values()}
        self.theora = None

    def start(self):
        import importlib

        from dynamic_reconfigure.server import Server

        self.servers = []
        for plugin, topic in self.plugins.items():
            module, name = self.CONFIGS[plugin]
            cfg = importlib.import_module(f"{module}.{name}")   # the generated config module
            self.servers.append(Server(cfg, self._configurer(plugin), namespace=topic))
        if "theora" in self.plugins:
            topic = self.plugins["theora"]
            self.node.pub[topic].impl.add_subscriber_listener(_TheoraHeaders(self))
        import queue

        self.sub = None
        self.link = self.node.link        # this node's link, kept: never a new one later
        self.warmed = threading.Event()
        self.frames = 0
        self.events = queue.Queue()
        c = self.cam
        self.listened = [t for t in (c["image_topic"], c.get("info_topic"),
                                     *self.plugins.values()) if t]
        for t in self.listened:
            self.node.pub[t].impl.add_subscriber_listener(_Listener(self.events, t))
        self.events.put(("warm", None))
        threading.Thread(target=self._events, daemon=True, name="usb_cam-render").start()

    #: frames rendered to warm the stream at start: half a second, so that each of the
    #: stream's render threads (two renders in flight) has built its GL context
    WARM_FRAMES = 15

    def _render(self, on: bool):
        c = self.cam
        if on and self.sub is None:
            self.sub = self.link.subscribe(
                "camera", float(c["rate"]), self._frame, camera=c["frame_id"],
                width=int(c["width"]), height=int(c["height"]), rgb=True)
        elif not on and self.sub is not None:
            sub, self.sub = self.sub, None
            self.link.unsubscribe(sub)

    def _events(self):
        """Subscriber connections and capture changes, in order: rendered while usb_cam
        captures and someone listens."""
        while True:
            event, _topic = self.events.get()
            try:
                if event == "warm":
                    self._render(True)
                    self.warmed.wait(60.0)
                listened = any(self.node.pub[t].get_num_connections() for t in self.listened)
                self._render(self.capturing and listened)
            except (Exception, SystemExit) as exc:   # SimLink exits when the simulation is gone
                print(f"usb_cam: render stream change failed: {exc!r}", file=sys.stderr,
                      flush=True)
                if isinstance(exc, SystemExit):
                    return

    def on_service(self, name, req):
        res = super().on_service(name, req)
        if name.endswith("/stop_capture") or name.endswith("/start_capture"):
            self.events.put(("capture", None))
        return res

    def _configurer(self, plugin):
        def configure(config, level):
            self.config[plugin].update({k: config[k] for k in self.config[plugin] if k in config})
            if plugin == "theora":
                if self.theora is None:
                    from robots._theora import TheoraPublisher

                    self.theora = TheoraPublisher()
                config = self.theora.configure(config)
            return config
        return configure

    def _frame(self, h, payload):
        self.frames += 1
        if self.frames >= self.WARM_FRAMES:
            self.warmed.set()
        super()._frame(h, payload)
        if not self.capturing:
            return
        rgb = None
        for plugin, topic in self.plugins.items():
            if not self.node.pub[topic].get_num_connections():
                continue
            if rgb is None:
                rgb = np.frombuffer(payload, np.uint8).reshape(h["height"], h["width"], 3)
            if plugin == "compressed":
                self.node.publish(topic, compressed_image(self.node.new(topic, stamp=h["stamp"]),
                                                          self.cam, rgb, self.config[plugin]))
            elif plugin == "compressedDepth":
                import rospy

                # codec.cpp L390-L393: no depth encoding, an error and nothing published
                rospy.logerr("Compressed Depth Image Transport - Compression requires "
                             "single-channel 32bit-floating point or 16bit raw depth images "
                             f"(input format is: {self.cam['encoding']}).")
            elif plugin == "theora":
                self._publish_theora(topic, h, rgb)

    def _publish_theora(self, topic, h, rgb):
        header = self.node.new(self.cam["image_topic"], stamp=h["stamp"]).header
        header.frame_id = self.cam["frame_id"]
        for p in self.theora.encode(header, rgb):
            self.node.publish(topic, self._packet_msg(topic, p))

    def _packet_msg(self, topic, p):
        msg = self.node.new(topic)
        msg.header = p["header"]
        msg.data = p["data"]
        msg.b_o_s, msg.e_o_s = p["b_o_s"], p["e_o_s"]
        msg.granulepos, msg.packetno = p["granulepos"], p["packetno"]
        return msg


class _TheoraHeaders:
    """theora_image_transport's connectCallback: the stream's header packets to each new
    subscriber alone (rospy's per-peer publish)."""

    def __init__(self, cam):
        self.cam = cam

    def peer_subscribe(self, topic_name, topic_publish, peer_publish):
        th = self.cam.theora
        for p in (th.stream_header if th is not None else []):
            peer_publish(self.cam._packet_msg(self.cam.plugins["theora"], p))

    def peer_unsubscribe(self, topic_name, num_peers):
        pass


def compressed_image(msg, cam: dict, rgb, config: dict):
    """compressed_image_transport's encoding of an rgb8 frame: the frame's header, format
    "<encoding>; <jpeg|png> compressed bgr8" and the frame converted to bgr8, as JPEG with
    the configured quality (progressive, optimize, restart interval) or as PNG at
    png_level."""
    import cv2

    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if config.get("format") == "png":
        kind = "png"
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(config["png_level"])])
    else:
        kind = "jpeg"
        ok, buf = cv2.imencode(".jpg", bgr, [
            cv2.IMWRITE_JPEG_QUALITY, int(config["jpeg_quality"]),
            cv2.IMWRITE_JPEG_PROGRESSIVE, int(bool(config["jpeg_progressive"])),
            cv2.IMWRITE_JPEG_OPTIMIZE, int(bool(config["jpeg_optimize"])),
            cv2.IMWRITE_JPEG_RST_INTERVAL, int(config["jpeg_restart_interval"])])
    if not ok:
        raise RuntimeError(f"{kind} encoding failed")
    msg.header.frame_id = cam["frame_id"]
    msg.format = f"{cam['encoding']}; {kind} compressed bgr8"
    msg.data = buf.tobytes()
    return msg


def lazy_pipeline(running: bool, event: str, topic: str, counts: dict, images, clouds) -> bool:
    """Whether orbbec_camera's pipeline runs after one subscriber event of its publishers
    (enable_pipeline true; the driver's connect/disconnect callbacks, ob_camera_node.cpp
    L535-L617, ros_setup.cpp L219-L243), `counts` the subscriber counts at that moment:
    a subscriber to an image or point-cloud topic starts it; an image topic losing one stops
    it when no image topic has any left; a point cloud losing its last one does too
    (whether or not the other cloud still has subscribers, as the driver does)."""
    if event == "subscribe":
        return True
    no_image = all(counts[t] == 0 for t in images)
    if topic in images:
        return running and not no_image
    if topic in clouds and counts[topic] == 0:
        return running and not no_image
    return running


class _Listener:
    """rospy's SubscribeListener on one publisher: its connect and disconnect events, in
    order, to the camera's event queue."""

    def __init__(self, events, topic):
        self.events, self.topic = events, topic

    def peer_subscribe(self, topic_name, topic_publish, peer_publish):
        self.events.put(("subscribe", self.topic))

    def peer_unsubscribe(self, topic_name, num_peers):
        self.events.put(("unsubscribe", self.topic))


class AstraProPlus:
    """/camera/camera (orbbec_camera_node): colour, registered depth (16UC1, mm) and IR
    (mono16) at 30 Hz with their CameraInfo, the depth and coloured point clouds, and
    the driver's own camera transforms on /tf at tf_publish_rate. The rendered model
    camera (named after the colour frame) provides colour and depth, with depth returns
    outside the recorded range reported as 0; IR is the scene's luminance scaled to the
    sensor's 10-bit range (an estimate: IR is not rendered).

    The streams are lazy, as the driver's pipeline is (`lazy_pipeline`): the camera is
    rendered, and frames and CameraInfo published, only from a subscription to one of its
    image or point-cloud topics on; the latched CameraInfo keeps the last one. The
    transforms go out from the start.

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
        self.images = [c["image_topic"] for c in (self.rgb, self.depth, self.ir)]
        self.clouds = ["/camera/depth/points", "/camera/depth_registered/points"]
        self.running = False
        self.sub = None

    def start(self):
        import queue

        threading.Thread(target=self._tf_loop, daemon=True).start()
        self.events = queue.Queue()
        # the driver's per-connection subscribe/unsubscribe callbacks: rospy's
        # SubscribeListener on each image and point-cloud publisher, handled in order
        for t in (*self.images, *self.clouds):
            self.node.pub[t].impl.add_subscriber_listener(_Listener(self.events, t))
        threading.Thread(target=self._events, daemon=True).start()

    def _counts(self) -> dict:
        return {t: self.node.pub[t].get_num_connections() for t in (*self.images, *self.clouds)}

    def _events(self):
        while True:
            event, topic = self.events.get()
            self.on_subscriber_event(event, topic, self._counts())

    def on_subscriber_event(self, event, topic, counts):
        want = lazy_pipeline(self.running, event, topic, counts, self.images, self.clouds)
        if want == self.running:
            return
        try:
            self._pipeline(want)
        except (Exception, SystemExit) as exc:   # SimLink exits when the simulation is gone
            self.running, self.sub = False, None
            print(f"camera: pipeline {'start' if want else 'stop'} failed: {exc!r}",
                  file=sys.stderr, flush=True)

    def _pipeline(self, on: bool):
        link, c = self.node.link, self.rgb
        self.running = on
        if on:
            self.sub = link.subscribe("camera", float(c["rate"]), self._frame,
                                      camera=c["frame_id"], width=int(c["width"]),
                                      height=int(c["height"]), rgb=True, depth=True)
        elif self.sub is not None:
            sub, self.sub = self.sub, None
            link.unsubscribe(sub)

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
        if not self.running:
            return  # a frame rendered before the pipeline stopped
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
        # each point cloud is built only while it has a subscriber (a cloud nobody receives
        # is never seen); the driver publishes them from the same depth frame
        want_pts = on["depth"] and self._listened("/camera/depth/points")
        want_reg = on["depth"] and on["color"] and self._listened("/camera/depth_registered/points")
        if not (want_pts or want_reg):
            return
        z = np.where(valid, depth, 0.0).astype(np.float32)
        x, y = self.ray_x * z, self.ray_y * z
        sel = valid.ravel()
        xyz = np.stack([x.ravel()[sel], y.ravel()[sel], z.ravel()[sel]], axis=1).astype(np.float32)
        if want_pts:
            self._cloud("/camera/depth/points", "camera_depth_optical_frame", stamp, xyz, None)
        if not want_reg:
            return
        cols = rgb.reshape(-1, 3)[sel].astype(np.uint32)
        packed = ((cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2]).view(np.float32)
        self._cloud("/camera/depth_registered/points", "camera_color_optical_frame", stamp,
                    xyz, packed)

    def _listened(self, topic) -> bool:
        pub = (getattr(self.node, "pub", None) or {}).get(topic)
        return pub is None or pub.get_num_connections() > 0

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
        # The driver's layout (the topic rows' notes in ros.yml): setPointCloud2FieldsByString
        # pads "xyz" to 16 bytes, and the coloured cloud adds "rgb" after that padding.
        fields = [PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1),
                  PointField("z", 8, PointField.FLOAT32, 1)]
        arr = np.zeros((xyz.shape[0], 5 if rgb is not None else 4), np.float32)
        arr[:, :3] = xyz
        if rgb is not None:
            fields.append(PointField("rgb", 16, PointField.FLOAT32, 1))
            arr[:, 4] = rgb
        pc.fields = fields
        pc.is_bigendian = False
        pc.point_step = arr.shape[1] * 4
        pc.row_step = pc.point_step * pc.width
        pc.is_dense = True
        pc.data = arr.tobytes()
        self.node.publish(topic, pc)


BEHAVIOURS = {
    "/ydlidar_lidar_publisher": YdLidar,
    "/camera/camera": AstraProPlus,
    "/usb_cam": ArmCamera,
}
