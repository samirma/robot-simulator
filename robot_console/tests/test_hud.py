import numpy as np
import pytest

from robot_console.hud import HINTS, draw_overlay, placeholder


@pytest.fixture
def frame():
    rng = np.random.default_rng(4)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def test_placeholder_shape():
    canvas = placeholder(320, 240, "waiting")
    assert canvas.shape == (240, 320, 3)
    assert canvas.dtype == np.uint8


def test_overlay_never_mutates_the_recorded_frame(frame):
    # --record writes the raw frame, so burning hints into it would corrupt the
    # recording of what the camera actually saw.
    original = frame.copy()
    draw_overlay(frame, show_help=True, speed=0.15)
    assert np.array_equal(frame, original)


def test_overlay_returns_a_same_shaped_image(frame):
    out = draw_overlay(frame, show_help=True, speed=0.15, speed_max=0.28)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_overlay_actually_draws_something(frame):
    out = draw_overlay(frame, show_help=True, speed=0.15)
    assert not np.array_equal(out, frame)


def test_hidden_hints_draw_less_than_shown_hints(frame):
    shown = draw_overlay(frame, show_help=True, speed=0.15)
    hidden = draw_overlay(frame, show_help=False, speed=0.15)
    changed = lambda img: int(np.count_nonzero(np.any(img != frame, axis=-1)))
    assert changed(shown) > changed(hidden) > 0


def test_overlay_handles_a_small_frame():
    small = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_overlay(small, show_help=True, speed=0.15, speed_max=0.28)
    assert out.shape == small.shape


def test_overlay_without_speed(frame):
    assert draw_overlay(frame, show_help=True).shape == frame.shape


def test_hints_cover_every_documented_key():
    keys = " ".join(key for key, _ in HINTS)
    for fragment in ("W", "S", "A", "D", "Q", "E", "Space", "+", "-", "Esc", "H"):
        assert fragment in keys
