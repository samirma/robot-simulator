"""Map rendering and the click-to-world transform."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import mapped_room

import cv2

from robot_console.slam.grid import OccupancyGrid
from robot_console.slam.mapview import (
    COLOUR_FREE,
    COLOUR_OCCUPIED,
    COLOUR_UNKNOWN,
    MapView,
    placeholder,
)


@pytest.fixture
def view_of_a_room():
    grid = mapped_room()
    view = MapView(zoom=3)
    view.render(grid)
    return view, grid


def test_render_produces_a_bgr_image_at_the_requested_zoom():
    grid = OccupancyGrid(0.05, width=40, height=30)
    image = MapView(zoom=4).render(grid)
    assert image.shape == (30 * 4, 40 * 4, 3)
    assert image.dtype == np.uint8


def test_render_never_mutates_the_grid():
    grid = mapped_room()
    before = grid.data.copy()
    MapView(zoom=2).render(grid, pose=(1.0, 1.0, 0.0), goal=(3.0, 2.0), trail=[(1.0, 1.0), (2.0, 1.0)])
    assert np.array_equal(grid.data, before)


def test_the_three_cell_classes_get_their_three_colours():
    grid = OccupancyGrid(0.1, width=3, height=1, origin=(0.0, 0.0))
    grid.data[0, 0] = -5.0   # free
    grid.data[0, 1] = 0.0    # unknown
    grid.data[0, 2] = 5.0    # occupied
    image = MapView(zoom=1).render(grid)
    assert tuple(image[0, 0]) == COLOUR_FREE
    assert tuple(image[0, 1]) == COLOUR_UNKNOWN
    assert tuple(image[0, 2]) == COLOUR_OCCUPIED


def test_the_image_is_flipped_so_higher_y_is_higher_up():
    grid = OccupancyGrid(0.1, width=1, height=4, origin=(0.0, 0.0))
    grid.data[0, 0] = 5.0    # lowest y
    image = MapView(zoom=1).render(grid)
    assert tuple(image[-1, 0]) == COLOUR_OCCUPIED, "lowest y belongs at the bottom"
    assert tuple(image[0, 0]) == COLOUR_UNKNOWN


@pytest.mark.parametrize(
    "point", [(0.1, 0.1), (2.0, 1.5), (5.0, 3.0), (-1.0, 2.0), (3.33, 0.77)]
)
def test_a_click_round_trips_back_to_the_point_it_looks_like(view_of_a_room, point):
    """Grid rows run one way, image rows the other; getting this wrong sends the robot
    to the mirror image of wherever the user clicked."""
    view, grid = view_of_a_room
    px, py = view.world_to_pixel(point)
    back = view.pixel_to_world(px, py)
    assert back == pytest.approx(point, abs=grid.resolution)


def test_a_left_click_becomes_a_goal(view_of_a_room):
    view, _ = view_of_a_room
    px, py = view.world_to_pixel((2.0, 1.5))
    view.on_mouse(cv2.EVENT_LBUTTONDOWN, px, py, 0)
    goal = view.take_click()
    assert goal == pytest.approx([2.0, 1.5], abs=0.05)
    assert view.take_click() is None, "a click is consumed once"


def test_a_right_click_cancels(view_of_a_room):
    view, _ = view_of_a_room
    view.on_mouse(cv2.EVENT_RBUTTONDOWN, 10, 10, 0)
    assert view.take_cancel()
    assert not view.take_cancel()


def test_mouse_motion_does_nothing(view_of_a_room):
    view, _ = view_of_a_room
    view.on_mouse(cv2.EVENT_MOUSEMOVE, 10, 10, 0)
    assert view.take_click() is None and not view.take_cancel()


def test_a_huge_map_is_scaled_down_rather_than_cropped():
    """Cropping would hide exactly the frontier the user is waiting on."""
    grid = OccupancyGrid(0.05, width=2000, height=2000)
    image = MapView(zoom=4, max_side=800).render(grid)
    assert max(image.shape[:2]) <= 2000, "scaled, not cropped"
    assert image.shape[0] == image.shape[1] == 2000, "one pixel per cell at minimum zoom"


def test_a_scaled_down_map_still_round_trips_clicks():
    grid = OccupancyGrid(0.05, width=2000, height=1500)
    view = MapView(zoom=6, max_side=800)
    view.render(grid)
    px, py = view.world_to_pixel((10.0, 20.0))
    assert view.pixel_to_world(px, py) == pytest.approx((10.0, 20.0), abs=grid.resolution)


def test_overlays_draw_without_complaint(view_of_a_room):
    view, grid = view_of_a_room
    from robot_console.slam.frontier import find_frontiers

    image = view.render(
        grid,
        pose=(1.0, 1.0, 0.5),
        path=np.array([[1.0, 1.0], [3.0, 2.0]]),
        goal=(3.0, 2.0),
        scan_points=np.array([[2.0, 2.0], [2.5, 2.5], [-99.0, -99.0]]),
        frontiers=find_frontiers(grid),
        trail=[(1.0, 1.0), (1.5, 1.2), (2.0, 1.5)],
        status=["line one", "line two"],
        hints=["press M to save"],
    )
    assert image.shape[2] == 3


def test_offscreen_scan_points_are_dropped_not_wrapped(view_of_a_room):
    view, grid = view_of_a_room
    plain = view.render(grid)
    with_offscreen = view.render(grid, scan_points=np.array([[-500.0, -500.0]]))
    assert np.array_equal(plain, with_offscreen)


def test_placeholder_is_a_readable_image():
    image = placeholder("waiting for /scan ...")
    assert image.shape == (360, 480, 3)
    assert image.dtype == np.uint8


def test_vectorised_and_scalar_pixel_transforms_agree(view_of_a_room):
    """The vectorised path exists for speed; if it disagrees, clicks land elsewhere."""
    view, _ = view_of_a_room
    points = np.array([[0.1, 0.1], [2.0, 1.5], [5.0, 3.0], [-1.0, 2.0], [3.33, 0.77]])
    scalar = np.array([view.world_to_pixel(p) for p in points], dtype=np.int32)
    assert np.array_equal(view.world_to_pixels(points), scalar)


def test_vectorised_transform_handles_an_empty_cloud(view_of_a_room):
    view, _ = view_of_a_room
    assert view.world_to_pixels(np.empty((0, 2))).shape == (0, 2)


def test_rendering_a_big_map_with_a_long_trail_is_not_quadratic():
    """A trail grows for the whole run; a Python loop over it per frame was most of the
    render cost by the time a house was half mapped."""
    import time

    grid = OccupancyGrid(0.05, width=400, height=400)
    view = MapView(zoom=2, max_side=800)
    trail = [(i * 0.01, i * 0.01) for i in range(4000)]
    scan = np.random.default_rng(0).uniform(0, 15, size=(360, 2))

    view.render(grid, pose=(1.0, 1.0, 0.0), trail=trail, scan_points=scan)  # warm up
    started = time.perf_counter()
    for _ in range(5):
        view.render(grid, pose=(1.0, 1.0, 0.0), trail=trail, scan_points=scan)
    per_frame = (time.perf_counter() - started) / 5
    assert per_frame < 0.05, f"{per_frame * 1000:.0f} ms per frame"
