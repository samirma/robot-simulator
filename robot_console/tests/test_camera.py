import base64
import threading

import cv2
import numpy as np
import pytest

from robot_console.camera import (
    LatestFrame,
    compressed_image_bytes,
    decode_compressed_image,
    decode_depth_image,
    header_seq,
    is_jpeg,
)


def jpeg_message(width=64, height=48, fmt="jpeg", seq=7):
    # A smooth gradient rather than noise: JPEG is a frequency-domain codec, so random
    # pixels lose ~45 grey levels per channel and would say nothing about correctness.
    ys, xs = np.mgrid[0:height, 0:width]
    image = np.stack(
        [
            (xs * 255 // max(width - 1, 1)).astype(np.uint8),
            (ys * 255 // max(height - 1, 1)).astype(np.uint8),
            np.full((height, width), 128, dtype=np.uint8),
        ],
        axis=-1,
    )
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return image, {
        "header": {"seq": seq, "stamp": {"secs": 1, "nsecs": 2}, "frame_id": "camera"},
        "format": fmt,
        "data": base64.b64encode(buffer.tobytes()).decode("ascii"),
    }


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("jpeg", True),
        ("JPEG", True),
        # What a real image_transport republisher emits, as opposed to the simulator.
        ("rgb8; jpeg compressed bgr8", True),
        ("bgr8; jpeg compressed bgr8", True),
        ("png", False),
        ("", False),
        (None, False),
        (123, False),
    ],
)
def test_is_jpeg(fmt, expected):
    assert is_jpeg(fmt) is expected


def test_decode_round_trip():
    original, message = jpeg_message()
    frame = decode_compressed_image(message)
    assert frame is not None
    assert frame.shape == original.shape
    assert frame.dtype == np.uint8
    # JPEG is lossy, so compare loosely rather than exactly.
    assert np.abs(frame.astype(int) - original.astype(int)).mean() < 12


def test_decode_accepts_real_hardware_format():
    _, message = jpeg_message(fmt="rgb8; jpeg compressed bgr8")
    assert decode_compressed_image(message) is not None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(format="png"),
        lambda m: m.update(data=""),
        lambda m: m.update(data="not base64 at all !!!"),
        lambda m: m.update(data=m["data"][:20]),  # truncated
        lambda m: m.update(data=[1, 2, 3]),  # int array instead of base64
        lambda m: m.pop("data"),
        lambda m: m.pop("format"),
    ],
)
def test_decode_returns_none_on_corruption(mutate):
    _, message = jpeg_message()
    mutate(message)
    # A bad frame must not raise: the loop that decodes it is also the loop that keeps
    # the robot's command stream alive.
    assert decode_compressed_image(message) is None


def test_decode_rejects_non_mapping():
    assert decode_compressed_image(None) is None
    assert decode_compressed_image("nope") is None


def test_header_seq():
    _, message = jpeg_message(seq=42)
    assert header_seq(message) == 42
    assert header_seq({}) is None
    assert header_seq({"header": {"seq": "x"}}) is None


def test_latest_frame_is_newest_wins():
    slot = LatestFrame()
    slot.offer({"n": 1})
    slot.offer({"n": 2})
    slot.offer({"n": 3})
    taken = slot.take()
    assert taken is not None and taken[0] == {"n": 3}
    assert slot.dropped == 2
    assert slot.received == 3
    assert slot.take() is None


def test_latest_frame_starts_empty():
    slot = LatestFrame()
    assert slot.take() is None
    assert slot.rate_hz == 0.0
    assert slot.last_arrival is None


def test_latest_frame_is_thread_safe():
    slot = LatestFrame()
    stop = threading.Event()

    def producer(tag):
        while not stop.is_set():
            slot.offer({"tag": tag})

    threads = [threading.Thread(target=producer, args=(i,), daemon=True) for i in range(4)]
    for thread in threads:
        thread.start()
    taken = 0
    for _ in range(2000):
        if slot.take() is not None:
            taken += 1
    stop.set()
    for thread in threads:
        thread.join(timeout=2)

    assert taken > 0
    # Nothing is created or lost: every offer either was taken or was dropped.
    assert slot.received >= taken + slot.dropped


# ------------------------------------------------------------------ raw bytes


def test_compressed_image_bytes_returns_a_decodable_jpeg():
    """The camera mapper forwards these bytes to its engine without re-encoding."""
    frame = np.zeros((8, 8, 3), np.uint8)
    frame[2:6, 2:6] = 200
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    msg = {"format": "jpeg", "data": base64.b64encode(buf.tobytes()).decode()}
    raw = compressed_image_bytes(msg)
    assert isinstance(raw, bytes) and raw
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None


@pytest.mark.parametrize("msg", [
    None, {}, {"format": "png", "data": "x"}, {"format": "jpeg"},
    {"format": "jpeg", "data": ""}, {"format": "jpeg", "data": "!!!not base64!!!"},
])
def test_compressed_image_bytes_never_raises(msg):
    assert compressed_image_bytes(msg) is None


def test_decoding_agrees_with_the_raw_bytes_it_is_built_on():
    frame = np.full((6, 6, 3), 120, np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    msg = {"format": "jpeg", "data": base64.b64encode(buf.tobytes()).decode()}
    assert decode_compressed_image(msg) is not None
    assert compressed_image_bytes(msg) == buf.tobytes()


# ------------------------------------------------------------------ depth


def depth_message(mm: np.ndarray) -> dict:
    return {
        "encoding": "16UC1",
        "height": mm.shape[0],
        "width": mm.shape[1],
        "step": mm.shape[1] * 2,
        "data": base64.b64encode(mm.astype(np.uint16).tobytes()).decode(),
    }


def test_depth_comes_back_in_metres():
    out = decode_depth_image(depth_message(np.array([[1500, 250], [3000, 10]])))
    np.testing.assert_allclose(out[0], [1.5, 0.25])
    np.testing.assert_allclose(out[1], [3.0, 0.01])
    assert out.dtype == np.float32


def test_a_zero_depth_reading_is_no_return_not_zero_metres():
    """0 means the sensor saw nothing; left as 0.0 it is a wall against the lens."""
    out = decode_depth_image(depth_message(np.array([[0, 2000]])))
    assert np.isnan(out[0, 0])
    assert out[0, 1] == pytest.approx(2.0)


@pytest.mark.parametrize("msg", [
    None, {}, {"encoding": "32FC1", "height": 1, "width": 1, "data": "AA=="},
    {"encoding": "16UC1", "height": 0, "width": 0, "data": "AA=="},
    {"encoding": "16UC1", "height": 4, "width": 4, "data": "AA=="},   # truncated
])
def test_decode_depth_image_never_raises(msg):
    assert decode_depth_image(msg) is None
