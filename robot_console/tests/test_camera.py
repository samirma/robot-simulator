"""Raw image decoding and live/stale/failed stream states (console spec §2.3)."""

import base64

import numpy as np
import pytest

from robot_console import camera as cam


def img(enc, data, w=2, h=1, step=None):
    return {"width": w, "height": h, "encoding": enc, "is_bigendian": 0,
            "step": step if step is not None else len(data) // h,
            "data": base64.b64encode(bytes(data)).decode()}


def test_rgb_bgr_mono():
    assert cam.decode_image(img("rgb8", [10, 20, 30, 40, 50, 60])).tolist() == [[[10, 20, 30], [40, 50, 60]]]
    assert cam.decode_image(img("bgr8", [10, 20, 30, 40, 50, 60])).tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert cam.decode_image(img("mono8", [7, 9])).tolist() == [[[7, 7, 7], [9, 9, 9]]]
    assert cam.decode_image(img("rgba8", [1, 2, 3, 4, 5, 6, 7, 8])).tolist() == [[[1, 2, 3], [5, 6, 7]]]
    m16 = cam.decode_image(img("mono16", [0, 0, 0xFF, 0xFF]))
    assert m16[0, 0, 0] == 0 and m16[0, 1, 0] == 255


def test_row_padding_is_respected():
    # step 8 > width*3 = 6: two padding bytes per row
    out = cam.decode_image(img("rgb8", [1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0], w=2, h=2, step=8))
    assert out.tolist() == [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]


@pytest.mark.parametrize("enc, data", [("yuv422", [128, 100, 128, 200]), ("yuv422_yuy2", [100, 128, 200, 128])])
def test_yuv422_grey(enc, data):
    out = cam.decode_image(img(enc, data))
    assert out.shape == (1, 2, 3)
    assert np.all(out[0, 0] == 100) and np.all(out[0, 1] == 200)


def test_failures():
    with pytest.raises(cam.FrameError, match="unsupported encoding"):
        cam.decode_image(img("bayer_rggb8", [0, 0]))
    with pytest.raises(cam.FrameError, match="truncated"):
        cam.decode_image(img("rgb8", [1, 2, 3], w=2, step=6))
    with pytest.raises(cam.FrameError, match="profile documents rgb8"):
        cam.decode_image(img("bgr8", [1] * 6), allowed=("rgb8",))
    with pytest.raises(cam.FrameError):
        cam.decode_image({"width": 2})


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def test_stream_states():
    clk = Clock()
    s = cam.CameraStream(cam.StreamSpec("/c", "sensor_msgs/Image", 1.0, ("rgb8",)), clock=clk)
    assert s.state() == cam.WAITING
    clk.t += 1.5
    assert s.state() == cam.STALE and "no frame within" in s.status_text()
    s.offer(img("rgb8", [1] * 6))
    s.poll()
    assert s.state() == cam.LIVE and s.frame is not None
    clk.t += 0.9
    assert s.state() == cam.LIVE
    clk.t += 0.2
    assert s.state() == cam.STALE, "a frozen frame is never presented as live"
    s.offer(img("bgr8", [1] * 6))
    s.poll()
    assert s.state() == cam.UNSUPPORTED and s.frame is None
    s.offer(img("rgb8", [1, 2], step=6))
    s.poll()
    assert s.state() == cam.FAILED and "truncated" in s.status_text()
    s.offer(img("rgb8", [1] * 6))
    s.poll()
    assert s.state() == cam.LIVE
    s.missing = True
    assert s.state() == cam.MISSING


def test_depth_is_a_false_colour_map_scaled_to_the_camera_range():
    """A depth camera's 16UC1 frame (millimetres) is drawn over its range, not stretched to its own
    maximum; 0 is black; values nearer or farther than the range take the end colours."""
    lut = cam.depth_lut()
    assert lut.shape == (256, 3)
    assert lut[0].tolist() == list(cam.DEPTH_STOPS[0][1]) and lut[255].tolist() == list(cam.DEPTH_STOPS[-1][1])
    vals = [0, 30, 70, 285, 500, 900]
    scale = (0.001, (0.07, 0.5))
    little = [b for v in vals for b in int(v).to_bytes(2, "little")]
    out = cam.decode_image(img("16UC1", little, w=6), ("16UC1",), scale)
    assert out.shape == (1, 6, 3)
    assert out[0, 0].tolist() == [0, 0, 0]                                  # no depth
    assert out[0, 1].tolist() == out[0, 2].tolist() == lut[0].tolist()      # at or nearer than near
    assert out[0, 3].tolist() == lut[128].tolist()                          # the middle of the range
    assert out[0, 4].tolist() == out[0, 5].tolist() == lut[255].tolist()    # at or beyond far
    big = dict(img("16UC1", [b for v in vals for b in int(v).to_bytes(2, "big")], w=6), is_bigendian=1)
    assert cam.decode_image(big, None, scale).tolist() == out.tolist()
    # a 16-bit stream with no scale (an untied one) is stretched to its own maximum, as before
    assert cam.decode_image(img("16UC1", [0, 0, 0xFF, 0xFF])).tolist() == [[[0, 0, 0], [255, 255, 255]]]


def test_target_specs_carry_a_depth_cameras_scale():
    from robot_console.profiles import load
    p = load("rebot_b601_rs")

    class T:
        profile = p
        wire = staticmethod(lambda n: n)
    specs = {s.topic: s for s in cam.target_specs(T)}
    assert specs["/camera/camera/color/image_raw"].depth is None
    assert specs["/camera/camera/aligned_depth_to_color/image_raw"].depth == (0.001, (0.07, 0.5))
