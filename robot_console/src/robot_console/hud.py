"""The key-hint panel drawn over the live feed.

Every function returns a new array. The recorder writes the *raw* camera frame, so the
overlay must never touch the image it was handed -- a recording with the hints burned
in would be useless as a record of what the camera saw.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_ALPHA = 0.62

HINTS: Sequence[Tuple[str, str]] = (
    ("W / S", "forward / back"),
    ("A / D", "strafe left / right"),
    ("Q / E", "rotate left / right"),
    ("Space", "stop"),
    ("+ / -", "speed"),
    ("H", "hide these hints"),
    ("Esc", "quit"),
)

FOOTER = "hold a key to drive"
COLLAPSED = "H  keys"


def placeholder(width: int = 640, height: int = 480, message: str = "") -> np.ndarray:
    """A stand-in until the first camera frame lands.

    The window is the input device, so it has to exist and hold focus from the start --
    the operator must be able to drive before the first JPEG arrives.
    """
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    if message:
        size = cv2.getTextSize(message, FONT, 0.55, 1)[0]
        origin = ((width - size[0]) // 2, height // 2)
        cv2.putText(canvas, message, origin, FONT, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
    return canvas


def _panel(frame: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    """Darken a rectangle in place, so light text stays readable over any scene."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, frame.shape[1]), min(y + height, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(region, 1.0 - PANEL_ALPHA, np.zeros_like(region), PANEL_ALPHA, 0)


def draw_overlay(
    frame: np.ndarray,
    *,
    show_help: bool = True,
    speed: Optional[float] = None,
    speed_max: Optional[float] = None,
    moving: bool = False,
) -> np.ndarray:
    """Return a copy of `frame` with the key hints drawn on it."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    if not show_help:
        _badge(canvas, COLLAPSED, 12, 12)
        if speed is not None:
            _speed_badge(canvas, width, speed, speed_max, moving)
        return canvas

    scale = 0.44 if width < 520 else 0.5
    line_height = int(22 * (scale / 0.5))
    key_column = 12
    text_column = key_column + int(74 * (scale / 0.5))
    panel_width = min(width - 24, int(250 * (scale / 0.5)))
    panel_height = line_height * (len(HINTS) + 1) + 18

    _panel(canvas, 12, 12, panel_width, panel_height)

    y = 12 + line_height
    for key, description in HINTS:
        cv2.putText(canvas, key, (12 + key_column, y), FONT, scale, (120, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, description, (12 + text_column, y), FONT, scale, (235, 235, 235), 1, cv2.LINE_AA)
        y += line_height
    cv2.putText(canvas, FOOTER, (12 + key_column, y + 2), FONT, scale * 0.86, (150, 150, 150), 1, cv2.LINE_AA)

    if speed is not None:
        _speed_badge(canvas, width, speed, speed_max, moving)
    return canvas


def _badge(frame: np.ndarray, text: str, x: int, y: int, colour=(200, 200, 200)) -> None:
    size = cv2.getTextSize(text, FONT, 0.44, 1)[0]
    _panel(frame, x, y, size[0] + 16, size[1] + 14)
    cv2.putText(frame, text, (x + 8, y + size[1] + 7), FONT, 0.44, colour, 1, cv2.LINE_AA)


def _speed_badge(
    frame: np.ndarray, width: int, speed: float, speed_max: Optional[float], moving: bool
) -> None:
    text = f"{speed:.2f} m/s"
    if speed_max is not None and speed >= speed_max - 1e-9:
        text += "  max"
    # Green while the robot is actually being driven, so a stuck key is obvious.
    colour = (140, 240, 140) if moving else (200, 200, 200)
    size = cv2.getTextSize(text, FONT, 0.44, 1)[0]
    _badge(frame, text, width - size[0] - 28, 12, colour)
