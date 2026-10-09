"""Camera helpers shared by the wires of every ROS dialect: a recorded camera row, the
CameraInfo it publishes, and a frame in the recorded image encoding. Standard library and
NumPy only, so ROS 1 and ROS 2 wires (and the unit tests, without ROS) use the same code."""

from __future__ import annotations

import numpy as np


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


def depth_mm(depth: np.ndarray, depth_range) -> np.ndarray:
    """A rendered depth (m, float32) as a 16UC1 image in millimetres, 0 outside the recorded
    working range `(near, far)` (m), the sensor's no-return value."""
    lo, hi = depth_range
    valid = (depth >= lo) & (depth <= hi)
    return np.where(valid, np.round(depth * 1000.0), 0).astype(np.uint16)
