import json

import cv2
import numpy as np
import pytest

from robot_console.bridge import Odom
from robot_console.recorder import FEED_NAME, LOG_NAME, Recorder, estimate_fps
from robot_console.teleop import Command


def frames(n=30, width=64, height=48):
    rng = np.random.default_rng(1)
    return [rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(n)]


def read_log(directory):
    return [json.loads(line) for line in (directory / LOG_NAME).read_text().splitlines()]


def test_estimate_fps_clean_stream():
    arrivals = [i * 0.05 for i in range(40)]
    assert estimate_fps(arrivals) == pytest.approx(20.0, abs=0.1)


def test_estimate_fps_survives_a_stall():
    # A median shrugs off one 2 s hiccup; a mean would report ~9 fps and the recording
    # would play back in slow motion.
    arrivals = [i * 0.05 for i in range(20)]
    arrivals += [arrivals[-1] + 2.0 + i * 0.05 for i in range(20)]
    assert estimate_fps(arrivals) == pytest.approx(20.0, abs=0.5)


def test_estimate_fps_falls_back():
    assert estimate_fps([]) == 20.0
    assert estimate_fps([1.0]) == 20.0
    assert estimate_fps([1.0, 1.0]) == 20.0


def test_records_video_and_log(tmp_path):
    recorder = Recorder(tmp_path, t0=0.0)
    recorder.start({"host": "127.0.0.1", "port": 9090})
    for i, frame in enumerate(frames(30)):
        recorder.add_frame(frame, t=i * 0.05, seq=1000 + i)
        recorder.add_command(Command(vx=0.15), speed=0.15, action="FORWARD", key="w", t=i * 0.05)
    recorder.add_odom(Odom(x=1.0, y=2.0, yaw=0.5), t=1.0)
    recorder.add_event("speed", speed=0.2, t=1.1)
    summary = recorder.close(t=2.0)

    assert summary["frames"] == 30
    assert summary["commands"] == 30
    assert (tmp_path / FEED_NAME).exists()

    capture = cv2.VideoCapture(str(tmp_path / FEED_NAME))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0, abs=1.0)
    capture.release()


def test_log_schema(tmp_path):
    recorder = Recorder(tmp_path, t0=0.0)
    recorder.start({"host": "h", "port": 9090, "speed": 0.15})
    recorder.add_command(Command(vx=0.15, wz=0.3), speed=0.15, action="FORWARD", key="w", t=0.5)
    recorder.add_frame(frames(1)[0], t=0.6, seq=5)
    recorder.add_odom(Odom(x=1.0, y=2.0, yaw=0.5, vx=0.15), t=0.7)
    recorder.add_event("speed", speed=0.2, t=0.8)
    recorder.close(t=1.0)

    records = read_log(tmp_path)
    assert records[0]["type"] == "meta"
    assert records[0]["schema"] == 1
    assert records[-1]["type"] == "summary"
    assert {r["type"] for r in records} == {"meta", "cmd", "frame", "odom", "event", "summary"}

    command = next(r for r in records if r["type"] == "cmd")
    # The literal Twist sub-dicts, so a replayer can feed a line back to /cmd_vel.
    assert command["linear"] == {"x": 0.15, "y": 0.0, "z": 0.0}
    assert command["angular"] == {"x": 0.0, "y": 0.0, "z": 0.3}
    assert command["key"] == "w"
    assert command["action"] == "FORWARD"
    assert command["t"] == pytest.approx(0.5)

    frame = next(r for r in records if r["type"] == "frame")
    assert frame["header_seq"] == 5
    assert (frame["width"], frame["height"]) == (64, 48)

    odom = next(r for r in records if r["type"] == "odom")
    assert (odom["x"], odom["y"], odom["yaw"]) == (1.0, 2.0, 0.5)

    # Every line stands alone.
    for record in records[1:]:
        assert "t" in record and "type" in record


def test_no_camera_writes_no_video(tmp_path):
    recorder = Recorder(tmp_path, t0=0.0)
    recorder.start({"host": "h", "port": 9090})
    recorder.add_command(Command(), speed=0.15, action="NONE", t=0.1)
    summary = recorder.close(t=1.0)

    # An empty or placeholder-filled mp4 is worse than an absent one.
    assert not (tmp_path / FEED_NAME).exists()
    assert summary["frames"] == 0
    assert summary["feed"] is None
    assert any(r.get("kind") == "no_camera" for r in read_log(tmp_path))


def test_short_drive_still_produces_video(tmp_path):
    # Fewer frames than the fps probe wants.
    recorder = Recorder(tmp_path, t0=0.0, probe_frames=20)
    recorder.start({})
    for i, frame in enumerate(frames(3)):
        recorder.add_frame(frame, t=i * 0.05)
    summary = recorder.close(t=1.0)
    assert summary["frames"] == 3
    assert (tmp_path / FEED_NAME).exists()


def test_fixed_fps_opens_writer_immediately(tmp_path):
    recorder = Recorder(tmp_path, fps=10.0, t0=0.0)
    recorder.start({})
    recorder.add_frame(frames(1)[0], t=0.0)
    assert recorder.frames_written == 1
    assert recorder.fps == 10.0
    recorder.close(t=1.0)


def test_resizes_a_changed_frame_size(tmp_path):
    # Real image_transport can change resolution mid-stream.
    recorder = Recorder(tmp_path, fps=20.0, t0=0.0)
    recorder.start({})
    recorder.add_frame(np.zeros((48, 64, 3), dtype=np.uint8), t=0.0)
    recorder.add_frame(np.zeros((96, 128, 3), dtype=np.uint8), t=0.05)
    summary = recorder.close(t=1.0)
    assert summary["frames"] == 2


def test_close_is_idempotent(tmp_path):
    recorder = Recorder(tmp_path, t0=0.0)
    recorder.start({})
    recorder.close(t=1.0)
    assert recorder.close(t=2.0) == {}
