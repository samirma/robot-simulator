"""Synchronized rig observations for the scorer's tests, rendered from known poses.

Each sample is exactly what the embodiment records per rig pair: both views as JPEG
bytes with their stamps, both cameras' ``camera_info`` as the simulator publishes it (an
ideal pinhole from the rig's field of view), and the ``/joint_states`` message nearest in
stamp. The frames are drawn by projecting the scene through the rig's calibrated mounts:
a grey worktop, the white plate as the projected outline of its top face, the red apple
as a disc of its projected radius. Because every pose is known, each fixture is a labelled
case: the scorer must reach the stated verdict from pixels, calibration and joint state
alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from robot_console.arm.kinematics import ARM_JOINTS, GRIPPER_JOINT, JAW_CENTER_OFFSET, ik_position
from robot_console.arm.ros_settings import SCENE_CAMERAS
from robot_console.arm.task import START_ARM_QPOS
from robot_console.arm.vision_success import CameraModel

PLATE_XY = (0.226, -0.226)
PLATE_TOP_Z = 0.0204
PLATE_RADIUS = 0.10
APPLE_RADIUS = 0.020
RESTING = (PLATE_XY[0], PLATE_XY[1], 0.040)
#: The arm parked upright, well clear of the plate.
AWAY = (*START_ARM_QPOS, 1.0)


def camera_info(name: str) -> dict:
    _pos, _xy, fovy, (width, height) = SCENE_CAMERAS[name]
    fy = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return {"k": [fy, 0.0, width / 2.0, 0.0, fy, height / 2.0, 0.0, 0.0, 1.0],
            "width": width, "height": height, "frame_id": f"scene/{name}"}


CAMERA_INFO = {name: camera_info(name) for name in SCENE_CAMERAS}
MODELS = {name: CameraModel.from_camera_info(name, CAMERA_INFO[name]) for name in SCENE_CAMERAS}


def _outline(cam: CameraModel, radius: float) -> np.ndarray:
    rim = [cam.project((PLATE_XY[0] + radius * math.cos(a),
                        PLATE_XY[1] + radius * math.sin(a), PLATE_TOP_Z))
           for a in np.linspace(0, 2 * math.pi, 96, endpoint=False)]
    return np.round(np.asarray([p for p in rim if p is not None], np.float32)).astype(np.int32)


def render(name: str, apple: Sequence[float] | None, *, bowl: bool = False,
           shaded: bool = False, white_worktop: bool = False) -> bytes:
    """One view; ``bowl`` adds a dark magenta-red bowl touching the apple in the side
    view, which is what the plate's neighbour does on the MolmoSpaces kitchen. With
    ``shaded`` the apple's side-view red has the bowl's own hue (177), only brighter, as
    it reads there when it comes to rest turned the other way.

    ``white_worktop`` is the RoboCasa kitchen's white marble, measured on its overhead
    view: a worktop as white as the plate by saturation (S ~3) at V 234-249, with thin
    veins as bright as the plate; the plate's rim at V ~251 and its well at 255, but for
    the well's lower edge, in the rim's shadow (V ~230)."""
    cam = MODELS[name]
    if white_worktop:
        rng = np.random.default_rng(7)
        cloud = cv2.GaussianBlur(rng.normal(0.0, 1.0, (cam.height, cam.width)), (0, 0), 12)
        base = np.clip(244 + 3.0 * cloud / cloud.std(), 0, 255)
        image = np.stack([base - 3, base - 1, base], axis=-1).astype(np.uint8)
        for _ in range(120):   # the marble's veins: thin, and as bright as the plate
            x, y = int(rng.integers(0, cam.width)), int(rng.integers(0, cam.height))
            dx, dy = (int(v) for v in rng.integers(-40, 41, 2))
            cv2.line(image, (x, y), (x + dx, y + dy), (254, 255, 255), 1)
        cv2.fillPoly(image, [_outline(cam, PLATE_RADIUS)], (250, 251, 251))
        cv2.fillPoly(image, [_outline(cam, 0.70 * PLATE_RADIUS)], (255, 255, 255))
        well = _outline(cam, 0.66 * PLATE_RADIUS)
        low = well[well[:, 1] > np.median(well[:, 1])]
        cv2.polylines(image, [low[np.argsort(low[:, 0])]], False, (229, 230, 230), 7)
    else:
        image = np.full((cam.height, cam.width, 3), (120, 128, 132), np.uint8)   # BGR worktop
        cv2.fillPoly(image, [_outline(cam, PLATE_RADIUS)], (250, 252, 252))
    if apple is not None:
        centre = cam.project(apple)
        if centre is not None:
            distance = float(np.linalg.norm(np.asarray(apple) - cam.eye))
            radius = cam.fy * APPLE_RADIUS / distance
            colour = (20, 35, 205)
            if bowl and name == "side":
                cv2.ellipse(image, (int(centre[0]) - 25, int(centre[1] + radius + 8)),
                            (40, 14), 0, 0, 360, (13, 5, 80) if shaded else (45, 12, 115), -1)
                if shaded:
                    colour = (18, 10, 115)
            cv2.circle(image, (int(round(centre[0] * 16)), int(round(centre[1] * 16))),
                       int(round(radius * 16)), colour, -1, cv2.LINE_AA, shift=4)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def holding(apple: Sequence[float], gripper: float = 0.40) -> tuple[float, ...]:
    """Joints putting the jaw centre on ``apple``, closed on it: a held apple."""
    solve = ik_position(apple, pitch=-1.3, pitch_weight=1.0, max_iterations=600,
                        offset=JAW_CENTER_OFFSET)
    assert solve.position_error < 2e-3
    return (*[float(v) for v in solve.joints], gripper)


def sample(stamp: float, apple: Sequence[float] | None, joints: Sequence[float] = AWAY, *,
           side_offset: float = 0.0, joint_offset: float = 0.0, side: bool = True,
           camera_info: dict | None = CAMERA_INFO, bowl: bool = False,
           shaded: bool = False, white_worktop: bool = False) -> dict:
    names = (*ARM_JOINTS, GRIPPER_JOINT)
    order = sorted(range(len(names)), key=lambda i: names[i])   # the wire sorts by name
    return {
        "overhead": {"stamp": stamp, "format": "jpeg",
                     "data": render("overhead", apple, white_worktop=white_worktop)},
        "side": ({"stamp": stamp + side_offset, "format": "jpeg",
                  "data": render("side", apple, bowl=bowl, shaded=shaded)} if side else None),
        "camera_info": camera_info,
        "joint_state": {"stamp": stamp + joint_offset,
                        "name": [names[i] for i in order],
                        "position": [float(joints[i]) for i in order]},
    }


def episode(apple_at: Callable[[float], Sequence[float] | None], *, seconds: float = 3.0,
            hz: float = 10.0, joints_at: Callable[[float], Sequence[float]] = lambda t: AWAY,
            **kwargs) -> list[dict]:
    n = int(round(seconds * hz)) + 1
    return [sample(100.0 + i / hz, apple_at(i / hz), joints_at(i / hz), **kwargs)
            for i in range(n)]
