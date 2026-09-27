"""The worktop's fixed camera rig, a fleet member of its own under `/scene` (spec §3).

The rig -- `overhead` and `side`, staged into the model by `tasks/apple_on_plate.py` --
watches the work surface and is not a robot's, so it has its own namespace. It presents:

    /scene/overhead/color/compressed    sensor_msgs/msg/CompressedImage   SCENE_CAMERA_HZ
    /scene/overhead/color/camera_info   sensor_msgs/msg/CameraInfo        SCENE_CAMERA_HZ
    /scene/side/color/compressed        sensor_msgs/msg/CompressedImage   SCENE_CAMERA_HZ
    /scene/side/color/camera_info       sensor_msgs/msg/CameraInfo        SCENE_CAMERA_HZ
    /scene/tf_static                    tf2_msgs/msg/TFMessage            latched

Poses, intrinsics, sizes and the rate are the constants in `tasks/apple_on_plate.py`
(`SCENE_CAMERAS`, `SCENE_CAMERA_HZ`). Each image is stamped with the camera's optical
frame (`scene/overhead`: x right, y down, z forward), and `/scene/tf_static` places those
frames in `scene/worktop`, the arm base frame the constants are written in.
"""

from __future__ import annotations

import math
import sys

# Defined in the SO-101's contract module, where the workspace parity tests read them.
from ros_surfaces.so101 import (  # noqa: F401
    SCENE_CAMERA_INFO_TOPICS,
    SCENE_CAMERA_TOPICS,
    SCENE_NAMESPACE,
    SCENE_ROOT_FRAME,
    SCENE_TF_STATIC,
    TYPE_SCENE_CAMERA_INFO,
    TYPE_SCENE_IMAGE,
)


def probe_scene_cameras(model, cameras: dict[str, tuple[str, int, int]] | None = None
                        ) -> dict[str, tuple[str, int, int]]:
    """The subset of `cameras` whose MJCF camera exists in `model` (a kitchen with no task
    staged has no rig, which is not an error)."""
    import mujoco

    wanted = SCENE_CAMERA_TOPICS if cameras is None else cameras
    return {
        topic: spec for topic, spec in wanted.items()
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, spec[0]) >= 0
    }


def _optical_quat(xyaxes) -> tuple[float, float, float, float]:
    """(w, x, y, z) of the optical frame: x = image right, y = image down, z = forward."""
    import numpy as np

    right = np.asarray(xyaxes[:3], dtype=float)
    up = np.asarray(xyaxes[3:], dtype=float)
    right /= np.linalg.norm(right)
    up -= right * float(right @ up)
    up /= np.linalg.norm(up)
    back = np.cross(right, up)
    rot = np.column_stack([right, -up, -back])
    quat = np.zeros(4)
    import mujoco

    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot).flatten())
    return tuple(float(v) for v in quat)


def camera_info(width: int, height: int, fovy_deg: float, frame_id: str, stamp_s: float) -> dict:
    """ROS 2 CameraInfo for an ideal pinhole with MuJoCo's square pixels and centred axis."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx, cx, cy = fy, width / 2.0, height / 2.0
    sec = math.floor(stamp_s)
    return {
        "header": {"stamp": {"sec": int(sec), "nanosec": int(round((stamp_s - sec) * 1e9))},
                   "frame_id": frame_id},
        "height": int(height), "width": int(width), "distortion_model": "plumb_bob",
        "d": [0.0] * 5,
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "binning_x": 0, "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }


def attach_scene_rig(bus, *, model, cameras, jpeg_quality: int = 70, scene_option=None,
                     world_reset=None, truth=None):
    """Put the rig on `bus`; return the per-step callback, which carries `rate_hz`.

    `truth` (a `tasks.truth_log.TruthLog`, when `SIMULATOR_TRUTH_LOG` asks for one)
    records the task's true state for every frame the rig renders, stamped as that frame
    is; it writes a local file and puts nothing on the wire.
    """

    from contracts.rosbridge_server import compressed_image_ros2
    from contracts.tf import TYPE_TF_MESSAGE_ROS2, tf_message
    from mujoco_bridge import RenderWorker, _camera_frame, _encode_jpeg
    from tasks.apple_on_plate import SCENE_CAMERA_HZ, SCENE_CAMERAS

    del world_reset  # the rig holds no state a reset could restore
    constants = {name: (pos, xyaxes, fovy, res) for name, pos, xyaxes, fovy, res in SCENE_CAMERAS}
    streams = []
    statics = []
    for topic, (mjcf_name, width, height) in dict(cameras).items():
        view = _camera_frame(topic)
        pos, xyaxes, fovy, resolution = constants[view]
        if tuple(resolution) != (width, height):
            raise SystemExit(f"scene rig: {topic} is {width}x{height} but the task's "
                             f"constants say {resolution}")
        frame = bus.frame(view)
        streams.append((topic, SCENE_CAMERA_INFO_TOPICS[topic], mjcf_name, width, height,
                        fovy, frame))
        statics.append((bus.frame(SCENE_ROOT_FRAME), frame, pos, _optical_quat(xyaxes)))
        bus.advertise(topic, TYPE_SCENE_IMAGE)
        bus.advertise(SCENE_CAMERA_INFO_TOPICS[topic], TYPE_SCENE_CAMERA_INFO)
    # Rendered off the control loop, like the wrist camera; see `RenderWorker`.
    worker = RenderWorker(model, name="scene rig") if streams else None
    if statics:
        bus.publish(SCENE_TF_STATIC, tf_message(statics, stamp_s=0.0, ros2=True),
                    TYPE_TF_MESSAGE_ROS2, latched=True)
    print(f"scene rig under namespace {bus.ns}\n  pub "
          f"{', '.join(bus.topic(s[0]) for s in streams)}", file=sys.stderr)

    def frame_done(topic, info_topic, width, height, fovy, frame):
        def publish(rgb, stamp):
            bus.publish(topic, compressed_image_ros2(0, _encode_jpeg(rgb, jpeg_quality), stamp,
                                                     frame_id=frame), TYPE_SCENE_IMAGE)
            if bus.has_subscribers(info_topic):
                bus.publish(info_topic, camera_info(width, height, fovy, frame, stamp),
                            TYPE_SCENE_CAMERA_INFO)
        return publish

    def step(data):
        if data is None:
            if worker is not None:
                worker.close()
            return
        stamp = float(data.time)
        jobs = []
        for topic, info_topic, mjcf_name, width, height, fovy, frame in streams:
            if bus.has_subscribers(topic):
                jobs.append((mjcf_name, width, height, scene_option,
                             frame_done(topic, info_topic, width, height, fovy, frame)))
            elif bus.has_subscribers(info_topic):
                bus.publish(info_topic, camera_info(width, height, fovy, frame, stamp),
                            TYPE_SCENE_CAMERA_INFO)
        if jobs and worker is not None and worker.submit(data, stamp, jobs) \
                and truth is not None:
            truth.state(data, stamp)

    step.rate_hz = SCENE_CAMERA_HZ
    return step
