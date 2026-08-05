"""Drawing the map, and turning a click on it back into a world coordinate.

The pixel<->world round trip is the part worth being careful about. The grid's row 0 is
the lowest y, an image's row 0 is the top, and the view scales by an integer zoom -- three
chances to be off by a flip or a factor, each of which sends the robot somewhere the user
did not click.

Like `hud.draw_overlay`, `render` never mutates its input.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from robot_console.slam.grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR. Deliberately close to the greys map_server's own viewer uses, so a map looks the
# same here as it does in rviz.
COLOUR_UNKNOWN = (70, 70, 70)
COLOUR_FREE = (225, 225, 225)
COLOUR_OCCUPIED = (25, 25, 25)
COLOUR_ROBOT = (60, 200, 60)
COLOUR_PATH = (230, 160, 40)
COLOUR_GOAL = (60, 60, 235)
COLOUR_SCAN = (235, 120, 200)
COLOUR_FRONTIER = (200, 200, 40)
COLOUR_TRAIL = (140, 110, 70)

PANEL_ALPHA = 0.62


class MapView:
    """Renders an `OccupancyGrid` at an integer zoom and maps clicks back to metres.

    Integer zoom, not arbitrary: `cv2.resize` with INTER_NEAREST at an integer factor
    makes every map cell an exact square block of pixels, so the inverse transform is
    exact too and a click lands in the cell it looks like it landed in.
    """

    def __init__(self, *, zoom: int = 4, max_side: int = 900) -> None:
        self.zoom = max(1, int(zoom))
        self.max_side = int(max_side)
        self._origin = np.zeros(2)
        self._resolution = 0.05
        self._height = 0
        self._scale = self.zoom
        self.click: Optional[np.ndarray] = None
        self.cancelled = False

    # ------------------------------------------------------------------ transforms

    def world_to_pixel(self, point: Sequence[float]) -> Tuple[int, int]:
        cell_x = (float(point[0]) - self._origin[0]) / self._resolution
        cell_y = (float(point[1]) - self._origin[1]) / self._resolution
        px = int(round(cell_x * self._scale))
        py = int(round((self._height - cell_y) * self._scale))
        return px, py

    def world_to_pixels(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) world XY -> (N, 2) int32 pixels, in one pass.

        The scalar version in a Python loop is what a long trail and a 360-beam scan turn
        into thousands of interpreter round-trips per frame -- which is most of the render
        cost once a map gets big.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        cells = (pts - self._origin) / self._resolution
        px = np.rint(cells[:, 0] * self._scale)
        py = np.rint((self._height - cells[:, 1]) * self._scale)
        return np.column_stack((px, py)).astype(np.int32)

    def pixel_to_world(self, px: float, py: float) -> np.ndarray:
        cell_x = float(px) / self._scale
        cell_y = self._height - float(py) / self._scale
        return self._origin + np.array([cell_x, cell_y]) * self._resolution

    # ------------------------------------------------------------------ mouse

    def on_mouse(self, event: int, x: int, y: int, flags: int, userdata=None) -> None:
        """`cv2.setMouseCallback` handler. Left click sets a goal, right click cancels."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = self.pixel_to_world(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.cancelled = True

    def take_click(self) -> Optional[np.ndarray]:
        click, self.click = self.click, None
        return click

    def take_cancel(self) -> bool:
        cancelled, self.cancelled = self.cancelled, False
        return cancelled

    # ------------------------------------------------------------------ rendering

    def render(
        self,
        grid: OccupancyGrid,
        *,
        pose: Optional[Sequence[float]] = None,
        path: Optional[np.ndarray] = None,
        goal: Optional[Sequence[float]] = None,
        scan_points: Optional[np.ndarray] = None,
        frontiers: Optional[Sequence] = None,
        trail: Optional[Sequence] = None,
        status: Sequence[str] = (),
        hints: Sequence[str] = (),
    ) -> np.ndarray:
        self._origin = grid.origin.copy()
        self._resolution = grid.resolution
        self._height = grid.height
        # Shrink the zoom rather than the map when a big house would overflow the screen;
        # cropping would hide exactly the frontier the user is waiting on.
        longest = max(grid.width, grid.height, 1)
        self._scale = max(1, min(self.zoom, self.max_side // longest or 1))

        classes = grid.classify()
        canvas = np.empty((grid.height, grid.width, 3), dtype=np.uint8)
        canvas[:] = COLOUR_UNKNOWN
        canvas[classes == FREE] = COLOUR_FREE
        canvas[classes == OCCUPIED] = COLOUR_OCCUPIED
        # Flip once here so everything below is in image coordinates.
        canvas = np.flipud(canvas)
        canvas = cv2.resize(
            canvas,
            (grid.width * self._scale, grid.height * self._scale),
            interpolation=cv2.INTER_NEAREST,
        )

        if trail is not None and len(trail) > 1:
            cv2.polylines(canvas, [self.world_to_pixels(np.asarray(trail))], False,
                          COLOUR_TRAIL, 1, cv2.LINE_AA)

        if frontiers:
            for f in frontiers:
                centre = getattr(f, "centroid", f)
                cv2.circle(canvas, self.world_to_pixel(centre), 3, COLOUR_FRONTIER, -1)

        if scan_points is not None and len(scan_points):
            pixels = self.world_to_pixels(scan_points)
            inside = (
                (pixels[:, 0] >= 0) & (pixels[:, 0] < canvas.shape[1])
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < canvas.shape[0])
            )
            pixels = pixels[inside]
            canvas[pixels[:, 1], pixels[:, 0]] = COLOUR_SCAN

        if path is not None and len(path) > 1:
            cv2.polylines(canvas, [self.world_to_pixels(path)], False,
                          COLOUR_PATH, 2, cv2.LINE_AA)

        if goal is not None:
            gx, gy = self.world_to_pixel(goal)
            cv2.drawMarker(canvas, (gx, gy), COLOUR_GOAL, cv2.MARKER_CROSS, 14, 2)
            cv2.circle(canvas, (gx, gy), 8, COLOUR_GOAL, 1, cv2.LINE_AA)

        if pose is not None:
            self._draw_robot(canvas, pose)

        if status:
            _panel(canvas, 0, 0, canvas.shape[1], 16 * len(status) + 10)
            for i, line in enumerate(status):
                cv2.putText(canvas, line, (8, 20 + 16 * i), FONT, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        if hints:
            top = canvas.shape[0] - (16 * len(hints) + 10)
            _panel(canvas, 0, top, canvas.shape[1], 16 * len(hints) + 10)
            for i, line in enumerate(hints):
                cv2.putText(
                    canvas, line, (8, top + 20 + 16 * i), FONT, 0.4, (215, 215, 215), 1, cv2.LINE_AA
                )
        return canvas

    def _draw_robot(self, canvas: np.ndarray, pose: Sequence[float]) -> None:
        px, py = self.world_to_pixel(pose)
        radius = max(4, int(0.16 / self._resolution * self._scale))
        cv2.circle(canvas, (px, py), radius, COLOUR_ROBOT, 2, cv2.LINE_AA)
        yaw = float(pose[2])
        # Screen y grows downward, so the heading's y component is negated to draw.
        tip = (
            int(round(px + math.cos(yaw) * radius * 1.9)),
            int(round(py - math.sin(yaw) * radius * 1.9)),
        )
        cv2.line(canvas, (px, py), tip, COLOUR_ROBOT, 2, cv2.LINE_AA)


def _panel(canvas: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + width), min(canvas.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = (region * (1.0 - PANEL_ALPHA)).astype(np.uint8)


def placeholder(message: str, width: int = 480, height: int = 360) -> np.ndarray:
    frame = np.full((height, width, 3), COLOUR_UNKNOWN, dtype=np.uint8)
    cv2.putText(frame, message, (16, height // 2), FONT, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return frame
