"""Raw image decoding and live/stale/failed stream states (console spec §2.3)."""

import base64
import re

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


def test_infrared_is_grey_stretched_from_the_frames_minimum_to_its_maximum():
    """An ir camera's frame is drawn as the vendor IR viewer normalises it (IR_transform.cpp: NORM_MINMAX
    to 0..255): its darkest value black, its brightest white, the rest rounded in between; a flat frame
    is black. Without ``ir`` a 16-bit frame is stretched from 0 to its maximum (an untied stream)."""
    le = lambda *vs: [b for v in vs for b in v.to_bytes(2, "little")]       # noqa: E731
    got = cam.decode_image(img("mono16", le(1000, 1500, 2000, 3000), w=4), ["mono16"], ir=True)
    assert got[0, :, 0].tolist() == [0, 64, 128, 255] and (got[..., 0] == got[..., 2]).all()
    assert cam.decode_image(img("mono16", le(700, 700), w=2), ["mono16"], ir=True).tolist() == [[[0, 0, 0], [0, 0, 0]]]
    assert cam.decode_image(img("mono8", [10, 20, 30], w=3), ir=True)[0, :, 0].tolist() == [0, 128, 255]
    assert cam.decode_image(img("mono16", le(1000, 3000), w=2))[0, :, 0].tolist() == [85, 255]
    with pytest.raises(cam.FrameError, match="unsupported encoding"):
        cam.decode_image(img("rgb8", [1, 2, 3, 4, 5, 6]), ["mono16"], ir=True)


def test_target_specs_mark_an_infrared_camera():
    from robot_console.profiles import load
    p = load("rosmaster_x3_plus")

    class T:
        profile = p
        wire = staticmethod(lambda n: n)
    specs = {s.topic: s for s in cam.target_specs(T)}
    assert [(t, s.ir, s.depth is not None, s.optional, s.on_demand, s.throttle_ms) for t, s in specs.items()] == [
        ("/camera/rgb/image_raw", False, False, False, False, 500), ("/camera/depth/image_raw", False, True, False, False, 500),
        ("/camera/ir/image_raw", True, False, True, False, 500), ("/usb_cam/image_raw", False, False, True, False, 500),
        ("/usb_cam/image_raw/compressed", False, False, True, False, 200)]     # every profile camera starts by itself
    assert specs["/camera/depth/image_raw"].depth == (0.001, (0.6, 8.0))
    assert "laser_usb_bringup" in specs["/usb_cam/image_raw"].missing_note
    assert "restarted without it" in specs["/camera/ir/image_raw"].silent_note


class FakeRb:
    def __init__(self):
        self.subs, self.unsubs = [], []

    def subscribe(self, topic, type_, cb, *, throttle_rate=0, queue_length=1):
        self.subs.append((topic, throttle_rate))
        return f"sub:{topic}"

    def unsubscribe(self, sid):
        self.unsubs.append(sid)


def test_camera_set_caps_each_stream_and_leaves_untied_previews_off():
    """Every stream is subscribed at its rate cap; an untied preview (on_demand) stays off, a neutral state; a stream
    advertised but never delivering is explained when it has a note; any stream can be stopped."""
    specs = [cam.StreamSpec("/a", "sensor_msgs/Image", 2.0, throttle_ms=100),
             cam.StreamSpec("/b", "sensor_msgs/Image", 2.0, throttle_ms=200, on_demand=True),
             cam.StreamSpec("/c", "sensor_msgs/Image", 0.5, silent_note="the driver may run without it.")]
    rb = FakeRb()
    cs = cam.CameraSet(rb, specs)
    assert rb.subs == [("/a", 100), ("/c", cam.DEFAULT_THROTTLE_MS)]
    assert cs.streams["/b"].state() == cam.OFF and "on request" in cs.streams["/b"].status_text()
    s = cs.streams["/c"]
    assert s.state(s.started + 1.0) == cam.SILENT and "may run without it" in s.status_text(s.started + 1.0)
    cs.stop("/a")
    assert rb.unsubs == ["sub:/a"] and cs.streams["/a"].sid is None
    assert [s.throttle_ms for s in cam.untied_specs(type("G", (), {"cameras": lambda self: {"/x": "sensor_msgs/Image"}})())] == [500]


def test_full_size_frames_decode():
    """640x480 frames as the Astra and the USB camera send them (rgb8, 16UC1 mm, mono16) decode at full size."""
    import base64 as b64
    w, h = 640, 480
    rgb = np.arange(w * h * 3, dtype=np.uint32).astype(np.uint8)
    msg = {"width": w, "height": h, "encoding": "rgb8", "is_bigendian": 0, "step": 3 * w,
           "data": b64.b64encode(rgb.tobytes()).decode()}
    assert cam.decode_image(msg, ["rgb8"]).shape == (h, w, 3)
    mm = np.linspace(0, 9000, w * h).astype("<u2")
    depth = dict(msg, encoding="16UC1", step=2 * w, data=b64.b64encode(mm.tobytes()).decode())
    img = cam.decode_image(depth, ["16UC1"], depth=(0.001, (0.6, 8.0)))
    lut = cam.depth_lut()
    assert img.shape == (h, w, 3) and img[0, 0].tolist() == [0, 0, 0] and img[-1, -1].tolist() == lut[255].tolist()
    ir = cam.decode_image(dict(depth, encoding="mono16"), ["mono16"], ir=True)
    assert ir[0, 0].tolist() == [0, 0, 0] and ir[-1, -1].tolist() == [255, 255, 255]


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def synced(ref, clock, robot_minus_console, rtt=0.02, delay_back=0.0):
    """One /rosapi/get_time sample: sent now, read by the robot after rtt/2, answered after rtt/2 + delay_back."""
    t0 = clock.t
    robot = t0 + rtt / 2 + robot_minus_console
    clock.t += rtt + delay_back
    ref.add(t0, clock.t, robot)


def test_clock_reference_bounds_the_offset_ntp_style():
    """[T - t1, T - t0] always holds the true offset; samples are intersected; a delayed answer only widens its
    own bounds; a sample outside the estimate is a clock step, and only it is kept."""
    c = Clock()
    ref = cam.ClockRef(clock=c)
    assert ref.offset() is None and "measuring" in ref.reason
    synced(ref, c, 1234.5, rtt=0.02)
    lo, hi = ref.offset()
    assert lo <= 1234.5 <= hi and hi - lo == pytest.approx(0.02)
    synced(ref, c, 1234.5, rtt=0.004)
    lo, hi = ref.offset()
    assert lo <= 1234.5 <= hi and hi - lo == pytest.approx(0.004)
    synced(ref, c, 1234.5, rtt=0.004, delay_back=3.0)      # answer held behind a backlog: estimate unchanged
    assert ref.offset() == (lo, hi) and ref.steps == 0
    synced(ref, c, 1229.5, rtt=0.01)                        # the robot's clock stepped back 5 s
    lo2, hi2 = ref.offset()
    assert ref.steps == 1 and lo2 <= 1229.5 <= hi2
    c.t += cam.CLOCK_STALE_S + 1
    assert ref.offset() is None and "older than" in ref.reason


def frame_stream(clock, ref):
    s = cam.CameraStream(cam.StreamSpec("/c", "sensor_msgs/Image", 5.0), clock=clock)
    s.ref = ref
    return s


def show(s, clock, stamp):
    import base64 as b64
    sec = int(stamp)
    s.offer({"header": {"stamp": {"secs": sec, "nsecs": int(round((stamp - sec) * 1e9))}}, "width": 1, "height": 1,
             "encoding": "mono8", "is_bigendian": 0, "step": 1, "data": b64.b64encode(b"\x07").decode()})
    s.poll()


def test_lag_comes_from_the_clock_reference_and_survives_clock_steps():
    """Lag = arrival in robot time - stamp: a current frame is live whatever the clock offset. A frame is shown only
    once a time answer sent after it arrived confirms it; a robot clock step widens the age bounds (every answer of
    the last 10 s counts) and never reads as LIVE an image that may be old; a backlog is LAGGING once confirmed."""
    c = Clock()
    ref = cam.ClockRef(clock=c)
    s = frame_stream(c, ref)
    synced(ref, c, 500.0)
    c.t += 0.1
    show(s, c, c.t + 500.0 - 0.05)
    assert s.frames == 0 and s.needs_sync              # held: no answer was asked after it arrived
    synced(ref, c, 500.0)
    assert s.state() == cam.LIVE and s.frames == 1 and "image 0.07" in s.status_text()
    # the robot's clock steps back 5 s: the new frame looks 5 s late against the old reading, so the clock is asked
    # again before anything is called late; then its age spans both readings
    c.t += 0.3
    show(s, c, c.t + 495.0 - 0.05)
    assert s.state() == cam.UNKNOWN and s.needs_sync and "re-checking" in s.status_text()
    synced(ref, c, 495.0)
    assert s.state() == cam.UNKNOWN and "disagree by 5.0 s" in s.status_text()
    assert s.frames == 2 and ref.steps == 1
    c.t += cam.CLOCK_STALE_S                            # the readings before the step leave the window
    synced(ref, c, 495.0)
    show(s, c, c.t + 495.0 - 0.05)
    synced(ref, c, 495.0)
    assert s.state() == cam.LIVE
    # and forward 5 s: the bounds widen downwards only, the image is still known young
    c.t += 0.5
    show(s, c, c.t + 500.0 - 0.05)
    synced(ref, c, 500.0)
    assert s.state() == cam.LIVE
    # a backlog present from the first frame (no baseline is learnt from the frames themselves)
    c.t += cam.CLOCK_STALE_S
    synced(ref, c, 500.0)
    s2 = frame_stream(c, ref)
    c.t += 0.5
    show(s2, c, c.t + 500.0 - 2.5)
    assert s2.state() == cam.UNKNOWN and "re-checking" in s2.status_text()   # asked again before called late
    synced(ref, c, 500.0, delay_back=2.0)              # that sample's answer is itself held back: wide bounds
    assert s2.state() == cam.STALE and "no newer frame" in s2.status_text()
    c.t += cam.CLOCK_STALE_S
    synced(ref, c, 500.0)
    show(s2, c, c.t + 500.0 - 3.0)
    synced(ref, c, 500.0)
    assert s2.state() == cam.LAGGING and "not current" in s2.status_text()


def test_without_a_clock_reference_lag_is_unknown_never_zero():
    c = Clock()
    s = frame_stream(c, cam.ClockRef(clock=c))
    s.ref.fail("no clock reference: /rosapi/get_time did not answer (Service /rosapi/get_time does not exist)")
    show(s, c, 1.0)
    assert s.state() == cam.UNKNOWN and "does not exist" in s.status_text() and "lag 0" not in s.status_text()
    ref = cam.ClockRef(clock=c)
    synced(ref, c, 0.0)
    s = frame_stream(c, ref)
    s.offer({"header": {"stamp": {"secs": 0, "nsecs": 0}}, "width": 1, "height": 1, "encoding": "mono8",
             "is_bigendian": 0, "step": 1, "data": "Bw=="})
    s.poll()
    assert s.state() == cam.UNKNOWN and "no header stamp" in s.status_text()
    ref.add(c.t, c.t + 0.01, None)                          # the robot's clock reads 0 (simulated time not started)
    assert "reads 0" in ref.reason
    assert cam.stamp_s({"header": {"stamp": {"secs": 3, "nsecs": 500_000_000}}}) == 3.5
    assert cam.stamp_s({"header": {"stamp": {"sec": 2, "nanosec": 0}}}) == 2.0
    assert cam.stamp_s({}) is None


def test_a_set_without_a_reference_reports_delivery_only():
    """The wire check passes no clock: its streams are live on delivery and say nothing about lag."""
    c = Clock()
    s = cam.CameraStream(cam.StreamSpec("/c", "sensor_msgs/Image", 2.0), clock=c)
    show(s, c, 1.0)
    assert s.state() == cam.LIVE and s.status_text() == "live"


def stream_at(clock, ref, throttle_ms=500, stale=2.0):
    s = cam.CameraStream(cam.StreamSpec("/c", "sensor_msgs/Image", stale, throttle_ms=throttle_ms), clock=clock)
    s.ref = ref
    return s


class SimRb:
    """A rosbridge on a fake clock: /rosapi/get_time reads the robot's clock (console + offset at that moment) half a
    round trip ``rtt`` after it is asked and is answered after the other half; get_param says /use_sim_time is false."""

    def __init__(self, clock, offset=0.0, rtt=0.02):
        import concurrent.futures as cf
        self.cf, self.clock, self.offset, self.rtt, self.pending, self.n = cf, clock, offset, rtt, [], 0
        self.sim_time, self.param_rtt, self.asked = False, None, []

    def subscribe(self, topic, type_, cb, *, throttle_rate=0, queue_length=1):
        self.n += 1
        return f"s{self.n}"

    def unsubscribe(self, sid):
        pass

    def call_service_async(self, name, args=None, type=None):
        f = self.cf.Future()
        rtt = self.param_rtt if name.endswith("get_param") and self.param_rtt is not None else self.rtt
        self.pending.append([self.clock.t + rtt / 2, self.clock.t + rtt, name, f, None])
        self.asked.append(name)
        return f

    def step(self):
        now = self.clock.t + 1e-9
        for p in self.pending:
            if p[4] is None and p[0] <= now:
                p[4] = p[0] + self.offset                 # the robot reads its clock
        for p in [p for p in self.pending if p[1] <= now]:
            self.pending.remove(p)
            if p[2].endswith("get_param"):
                p[3].set_result({"value": "true" if self.sim_time else "false"})
            else:
                T = p[4]
                p[3].set_result({"time": {"secs": int(T), "nsecs": int(round((T - int(T)) * 1e9))}})


def sim_set(throttle_ms=500, rtt=0.02):
    c = Clock(1000.0)
    rb = SimRb(c, rtt=rtt)
    ref = cam.ClockRef(clock=c, dialect="ros1")
    spec = cam.StreamSpec("/camera/rgb/image_raw", "sensor_msgs/Image", 5.0, encodings=("rgb8",),
                          throttle_ms=throttle_ms)
    cs = cam.CameraSet(rb, [spec], clock=ref)
    s = cs.streams[spec.topic]
    s._clock = c
    return c, rb, ref, cs, s


def run_frames(c, rb, cs, s, seconds, lag, period=0.5, dt=0.01, emit=True, watch=None, jitter=False):
    """Frames every ``period`` (plus up to 2/30 s jitter), taken now and arriving ``lag`` later, stamped with the
    robot's clock at capture; polled every ``dt``. ``watch(true_age, state)`` sees each check."""
    end, k = c.t + seconds, 0
    nxt = c.t
    taken = s.__dict__.setdefault("in_flight", [])      # (arrival, stamp, capture console time)
    caps = s.__dict__.setdefault("captures", {})        # stamp -> console time it was taken
    while c.t < end - 1e-9:
        c.t += dt
        rb.step()
        if emit and c.t >= nxt - 1e-9:
            nxt = c.t + period + ((k % 3) / 30 if jitter else 0.0)
            k += 1
            taken.append((c.t + lag, c.t + rb.offset, c.t))
        while taken and taken[0][0] <= c.t + 1e-9:
            _, stamp, cap = taken.pop(0)
            s.offer(rgb_frame(stamp))
            caps[round(stamp, 6)] = cap
        cs.poll()
        if watch is not None and s.frame_at is not None and s.last_stamp is not None:
            watch(c.t - caps[round(s.last_stamp, 6)], s.state())


@pytest.mark.parametrize("lag", [0.3, 0.45, 0.6, 0.9])
def test_a_steady_lag_below_the_limit_never_flickers(lag):
    """At 2 frames/s the shown image ages up to a frame period (plus a time answer) between frames: with a steady
    transport lag of 0.3, 0.45, 0.6 or 0.9 s (a 1.2 MB frame takes 0.25-0.5 s at 20-40 Mbit/s) every check over 25 s
    of the real poll loop reads LIVE once the first frame is shown."""
    c, rb, ref, cs, s = sim_set()
    run_frames(c, rb, cs, s, 3.0, lag, jitter=True)
    seen = []
    run_frames(c, rb, cs, s, 25.0, lag, jitter=True, watch=lambda age, st: seen.append(st))
    assert set(seen) == {cam.LIVE}, sorted(set(seen))
    assert not cs.paused


def test_no_image_older_than_its_bound_reads_live():
    """Frames 0.9 s late are live; the stream then stops. The image is live only up to its bound, 1 s + 1.5 frame
    periods (1.75 s at 2 frames/s, 1.15 s at 10; shown in the status), then stale "no newer frame"."""
    for throttle, bound in ((500, 1.75), (100, 1.15)):
        c, rb, ref, cs, s = sim_set(throttle_ms=throttle)
        run_frames(c, rb, cs, s, 4.0, 0.9, period=throttle / 1000)
        assert s.state() == cam.LIVE and f"live while at most {bound:.2f} s" in s.status_text()
        ages = []
        run_frames(c, rb, cs, s, 3.0, 0.9, emit=False, watch=lambda age, st: ages.append((age, st)))
        assert max(a for a, st in ages if st == cam.LIVE) <= bound + 1e-6
        assert ages[-1][1] == cam.STALE and "no newer frame" in s.status_text()
        assert cam.live_bound_s(throttle) == pytest.approx(bound)
    c = Clock()                               # no clock reference: never live, and stale past the bound
    s2 = stream_at(c, cam.ClockRef(clock=c))
    show(s2, c, 1.0)
    assert s2.state() == cam.UNKNOWN
    c.t += cam.live_bound_s(500) + 0.1
    assert s2.state() == cam.STALE


def step_after_answer(delta, lag, phase, rtt=0.02, throttle_ms=500):
    """Frames ``lag`` late at the cap; ``phase`` s after a time answer lands the robot's clock steps by ``delta``;
    one more frame is taken, then the camera stalls. Returns the oldest true image age seen while LIVE."""
    c, rb, ref, cs, s = sim_set(throttle_ms=throttle_ms, rtt=rtt)
    run_frames(c, rb, cs, s, 6.0, lag, period=throttle_ms / 1000)
    assert s.state() == cam.LIVE or rtt > 0.3
    last = ref.last_ok
    while ref.last_ok == last:
        run_frames(c, rb, cs, s, 0.01, lag, period=throttle_ms / 1000)
    run_frames(c, rb, cs, s, phase, lag, period=throttle_ms / 1000)
    rb.offset += delta
    worst = [0.0]

    def watch(age, st):
        if st == cam.LIVE:
            worst[0] = max(worst[0], age)
    run_frames(c, rb, cs, s, 0.5, lag, period=throttle_ms / 1000, watch=watch)   # one more frame, after the step
    run_frames(c, rb, cs, s, 4.0, lag, emit=False, watch=watch)                    # the camera stalls
    return worst[0]


def test_a_clock_step_with_a_real_delay_never_shows_an_old_image_as_live():
    """The critic's case: a real delay of 0.9 s, a forward robot clock step of 1.1 s just after a time answer, then the
    camera stalls: no image older than its bound (1.75 s) ever reads LIVE."""
    assert step_after_answer(1.1, 0.9, 0.0) <= cam.live_bound_s(500) + 1e-6


@pytest.mark.parametrize("delta", [-1.5, -0.6, -0.2, 0.2, 0.6, 1.1, 1.5, 2.5])
@pytest.mark.parametrize("lag", [0.1, 0.9])
@pytest.mark.parametrize("phase", [0.0, 0.01, 0.3, 1.2])
def test_no_clock_step_makes_an_old_image_live(delta, lag, phase):
    """The worst case, swept: steps either way up to 2.5 s, at any point between answers, with a short or long real
    delay, with 0.02 s or 0.3 s answers: the oldest image ever read LIVE is within the bound itself, no margin."""
    for rtt in (0.02, 0.3):
        assert step_after_answer(delta, lag, phase, rtt=rtt) <= cam.live_bound_s(500) + 1e-6


def test_a_steady_offset_reads_lagging_with_its_note():
    """Frames that sit a steady 7 s behind the robot's clock on a quick link: a real delay or stamps on another clock
    cannot be told apart, so the stream is LAGGING (never live) with that note."""
    c, rb, ref, cs, s = sim_set()
    seen = []
    for _ in range(15):
        s.offer(rgb_frame(c.t + rb.offset - 7.0))
        for _ in range(40):
            c.t += 0.01
            rb.step()
            cs.poll()
        seen.append(s.state())
    assert cam.LIVE not in seen and seen[-1] == cam.LAGGING
    txt = s.status_text()
    assert "steady offset of +7.0 s" in txt and "another clock" in txt


def test_a_steady_offset_never_pauses_the_cameras():
    """The critic's thrash: frames stamped a steady 3 s behind (another clock) on a healthy link; time answers then
    take 0.35 s (one big frame ahead of them). A steady offset is not a backlog: R1 skips it, so the cameras are not
    paused, then resumed, then paused again."""
    c, rb, ref, cs, s = sim_set()
    end = c.t + 30.0
    states = set()
    while c.t < end:
        s.offer(rgb_frame(c.t + rb.offset - 3.1))
        for _ in range(50):
            c.t += 0.01
            rb.step()
            cs.poll()
        if c.t > 1010.0:
            rb.rtt = 0.35
            states.add(s.state())
        assert not cs.paused, cs.paused
    assert states == {cam.LAGGING} and "steady offset" in s.status_text()


def test_an_unknown_round_trip_is_not_quick():
    """After a resume (round trip forgotten) the steady offset is not claimed until a new answer is timed, as on the
    page; R1 names the round trip that triggered it."""
    c, rb, ref, cs, s = sim_set()
    for _ in range(14):
        s.offer(rgb_frame(c.t + rb.offset - 7.0))
        for _ in range(50):
            c.t += 0.01
            rb.step()
            cs.poll()
    assert s.steady_offset() is not None
    ref.last_rtt = None
    assert s.steady_offset() is None
    ref.last_rtt = 0.9                                 # well under the 7 s lag: still an offset, not a shared queue
    assert s.steady_offset() is not None
    ref.last_rtt = 1.2                                 # a slow answer: maybe a queue the frames share
    assert s.steady_offset() is None


def test_a_backward_step_hidden_by_a_wide_answer_is_still_a_step():
    """An answer delayed by a backlog has wide bounds that may still overlap the old estimate after a step; one whose
    bounds exclude the estimate's midpoint is taken as a step."""
    c = Clock()
    ref = cam.ClockRef(clock=c)
    synced(ref, c, 0.0, rtt=0.004)
    synced(ref, c, -0.6, rtt=0.004, delay_back=1.0)       # stepped back 0.6 s, answer held 1 s: bounds [-1.6, -0.6]
    assert ref.steps == 1 and ref.offset()[0] <= -0.6 <= ref.offset()[1]


class FakeRb2:
    """Records subscriptions; answers rosapi calls through futures the test settles."""

    def __init__(self):
        import concurrent.futures as cf
        self.cf, self.subs, self.unsubs, self.calls = cf, [], [], []

    def subscribe(self, topic, type_, cb, *, throttle_rate=0, queue_length=1):
        self.subs.append(topic)
        return f"sub:{topic}:{len(self.subs)}"

    def unsubscribe(self, sid):
        self.unsubs.append(sid)

    def call_service_async(self, service, args=None, type=None):
        f = self.cf.Future()
        self.calls.append((service, args, f))
        return f


def test_ros1_clock_reads_use_sim_time_first_and_retries_until_known():
    c = Clock()
    rb = FakeRb2()
    ref = cam.ClockRef(clock=c, dialect="ros1")
    ref.maybe_sync(rb)
    assert rb.calls[-1][0] == "/rosapi/get_param" and rb.calls[-1][1]["name"] == "/use_sim_time"
    rb.calls[-1][2].set_exception(RuntimeError("no answer"))
    assert ref.sim is None and ref.offset() is None and "use_sim_time" in ref.reason
    c.t += cam.CLOCK_SYNC_S
    ref.maybe_sync(rb)
    assert rb.calls[-1][0] == "/rosapi/get_param"           # asked again: not assumed false
    rb.calls[-1][2].set_result({"value": "true"})
    assert ref.sim is True and ref.offset() is None and "simulated time" in ref.reason
    ref2 = cam.ClockRef(clock=c, dialect="ros1")
    ref2.maybe_sync(rb)
    rb.calls[-1][2].set_result({"value": "false"})
    ref2.maybe_sync(rb)
    assert rb.calls[-1][0] == "/rosapi/get_time"
    rb.calls[-1][2].set_result({"time": {"secs": 100, "nsecs": 0}})
    assert ref2.offset() is not None


def test_congestion_pauses_every_stream_until_resumed():
    """A LAGGING stream, a slow time answer, or none for 3 s while frames keep arriving: every stream is unsubscribed,
    the set says why, and resume subscribes them again."""
    c = Clock()
    rb = FakeRb2()
    ref = cam.ClockRef(clock=c)
    specs = [cam.StreamSpec("/a", "sensor_msgs/Image", 5.0), cam.StreamSpec("/b", "sensor_msgs/Image", 5.0)]
    cs = cam.CameraSet(rb, specs, clock=ref)
    for st in cs.streams.values():
        st._clock = c
    synced(ref, c, 0.0, rtt=0.004)
    a = cs.streams["/a"]
    for i in range(7):                                      # a backlog building: 2.5 s late and growing
        c.t += 0.5
        show(a, c, c.t - 2.5 - 0.3 * i)
        c.t += 0.01
        synced(ref, c, 0.0, rtt=0.004)                     # confirmed by a fresh answer: LAGGING
        assert a.state() == cam.LAGGING
    assert a.steady_offset() is None and cs.congestion() == ""    # quick time answers: not the link
    ref.last_rtt = 0.6                                      # the time answers slow down too: the link backs up
    assert "arrive late while the robot's time answer took 0.60 s" in cs.congestion()
    cs.pause(cs.congestion())
    assert set(rb.unsubs) == {"sub:/a:1", "sub:/b:2"} and {s.state() for s in cs.streams.values()} == {cam.PAUSED}
    cs.resume()
    assert rb.subs[-2:] == ["/a", "/b"] and not cs.paused
    ref.last_rtt = 1.4                                      # a time answer took 1.4 s
    assert "took 1.4 s" in cs.congestion()
    ref.last_rtt = 0.01
    ref._inflight, ref._last_sent = True, c.t - 3.5        # unanswered for 3.5 s ...
    show(cs.streams["/b"], c, c.t)                          # ... while frames keep arriving
    assert "late while frames keep arriving" in cs.congestion()


def test_compressed_frames_decode_and_other_formats_are_unsupported():
    import base64 as b64
    import io
    import pygame
    surf = pygame.Surface((4, 2))
    surf.fill((200, 10, 10))
    buf = io.BytesIO()
    pygame.image.save(surf, buf, "x.png")
    msg = {"format": "rgb8; png compressed bgr8", "data": b64.b64encode(buf.getvalue()).decode()}
    assert cam.decode_image(msg, ["png"]).tolist() == [[[200, 10, 10]] * 4] * 2
    with pytest.raises(cam.FrameError, match="unsupported format"):
        cam.decode_image(msg, ["jpeg"])
    with pytest.raises(cam.FrameError, match="unsupported format"):
        cam.decode_image(dict(msg, format="theora"), None)


class TimeRb:
    """A rosbridge whose /rosapi/get_time answers (robot = console + 5000 s) can be held back: ``answer`` False keeps
    every answer pending; ``late`` answers after that many seconds of the fake clock (checked at each poll)."""

    def __init__(self, clock):
        import concurrent.futures as cf
        self.cf, self.clock, self.answer, self.late, self.pending, self.n = cf, clock, True, 0.0, [], 0

    def subscribe(self, topic, type_, cb, *, throttle_rate=0, queue_length=1):
        self.n += 1
        return f"s{self.n}"

    def unsubscribe(self, sid):
        pass

    def call_service_async(self, name, args=None, type=None):
        f = self.cf.Future()
        if name.endswith("get_param"):
            f.set_result({"value": "false"})
        elif self.answer and not self.late:
            f.set_result({"time": {"secs": int(self.clock.t + 5000), "nsecs": 0}})
        else:
            self.pending.append((self.clock.t, f))
        return f

    def settle(self):
        for t0, f in list(self.pending):
            if self.answer and self.late and self.clock.t - t0 >= self.late:
                f.set_result({"time": {"secs": int(t0 + 5000), "nsecs": 0}})
                self.pending.remove((t0, f))


def rgb_frame(stamp):
    import base64 as b64
    sec = int(stamp)
    return {"header": {"stamp": {"secs": sec, "nsecs": int((stamp - sec) * 1e9)}}, "width": 8, "height": 6, "step": 24,
            "encoding": "rgb8", "is_bigendian": 0, "data": b64.b64encode(bytes(8 * 6 * 3)).decode()}


def congestion_set():
    c = Clock()
    rb = TimeRb(c)
    ref = cam.ClockRef(clock=c, dialect="ros1")
    spec = cam.StreamSpec("/camera/rgb/image_raw", "sensor_msgs/Image", 2.0, encodings=("rgb8",), throttle_ms=500)
    cs = cam.CameraSet(rb, [spec], clock=ref)
    s = cs.streams[spec.topic]
    s._clock = c
    for _ in range(10):                       # a healthy link
        c.t += 0.5
        s.offer(rgb_frame(c.t + 5000 - 0.1))
        cs.poll()
    assert s.state() == cam.LIVE and not cs.paused
    return c, rb, cs, s


def test_teleop_pauses_when_time_answers_stop_while_frames_keep_coming():
    """The critic's probe through the real poll() loop: the time answers stop (the link backed up) while frames keep
    arriving, 2 s old and growing. R3 pauses the cameras within a few seconds, before the reference expires."""
    c, rb, cs, s = congestion_set()
    rb.answer = False
    for i in range(60):
        c.t += 0.5
        s.offer(rgb_frame(c.t + 5000 - 2.0 - i * 0.2))
        cs.poll()
        if cs.paused:
            break
    assert cs.paused and "late while frames keep arriving" in cs.paused, (i, s.status_text())
    assert c.t - 1005.0 <= 5.0 and s.state() == cam.PAUSED


def test_teleop_pauses_on_a_late_time_answer():
    """R2 through poll(): answers arriving 1.5 s late (kept even past the 3 s give-up) pause the cameras."""
    c, rb, cs, s = congestion_set()
    rb.late = 1.5
    for _ in range(20):
        c.t += 0.5
        rb.settle()
        s.offer(rgb_frame(c.t + 5000 - 0.1))
        cs.poll()
        if cs.paused:
            break
    assert cs.paused and "took 1.5 s" in cs.paused


def test_a_growing_backlog_pauses_through_poll_naming_the_round_trip():
    """R1 through the real poll loop: frames falling further behind (1.5 s and growing) while the time answers take
    0.4 s. The first late-looking frame is re-checked against a fresh answer, then the stream is LAGGING, and once its
    lags span 2.4 s with no steady offset the cameras pause, the warning naming the round trip."""
    c, rb, ref, cs, s = sim_set(rtt=0.4)
    run_frames(c, rb, cs, s, 3.0, 0.1)
    assert s.state() == cam.LIVE
    lag = 1.5
    for _ in range(80):
        s.offer(rgb_frame(c.t + rb.offset - lag))
        lag += 0.05
        for _ in range(10):
            c.t += 0.01
            rb.step()
            cs.poll()
        if cs.paused:
            break
    assert re.search(r"arrive late while the robot's time answer took 0\.\d\d s \(at least 0\.3 s\)", cs.paused), cs.paused


def test_sim_time_keeps_timing_the_answers_and_counts_unanswered_questions():
    """As on the page: a ROS 1 robot on simulated time has no reference (lag unknown), but its time is still asked and
    the answers timed, so 1.4 s answers pause the cameras (R2); and while /use_sim_time is not known yet, a question
    unanswered for 3 s while frames keep arriving is congestion (R3)."""
    c, rb, ref, cs, s = sim_set()
    rb.sim_time = True
    run_frames(c, rb, cs, s, 5.0, 0.1)
    assert ref.sim is True and s.state() == cam.UNKNOWN and "simulated time" in s.status_text()
    assert rb.asked.count("/rosapi/get_time") >= 2 and ref.last_rtt is not None
    rb.rtt = 1.4
    run_frames(c, rb, cs, s, 6.0, 0.1)
    assert "took 1.4 s" in cs.paused
    c, rb, ref, cs, s = sim_set()
    rb.param_rtt = 8.0
    run_frames(c, rb, cs, s, 5.0, 0.1)
    assert ref.sim is None and "late while frames keep arriving" in cs.paused
