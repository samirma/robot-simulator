"""``python -m robot_console.fleet`` (console spec §2.4, §4)."""

import re
import socket
import subprocess
import sys

import pytest

from robot_console.profiles import SUPPORTED_IDS, load
from wirespec import drop, merge, retype, wire_spec


def fleet(*args, timeout=60):
    r = subprocess.run([sys.executable, "-m", "robot_console.fleet", *args],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_passes_on_matching_wire(fake, pid):
    p = load(pid)
    srv = fake(wire_spec(p))
    code, out = fleet("--url", srv.url)
    assert code == 0, out
    label = pid + (f" (namespace /{p.namespace.default})" if p.namespace else "")    # the hardware default
    assert f"robot {label}: typed validation PASS" in out and out.rstrip().endswith("PASS")
    for c in p.cameras:
        if c.optional and p.endpoint(c.topic).optional:   # not on this wire (its optional rows are absent): reported, not failed
            assert f"camera {c.topic} (optional): MISSING" in out
        else:
            assert f"camera {c.topic}{' (optional)' if c.optional else ''}: LIVE" in out
    if not p.cameras:
        assert f"cameras: none in profile '{pid}'" in out
    code, out = fleet("--url", srv.url, "--expect", pid)
    assert code == 0, out
    assert srv.command_ops() == [], "the check publishes nothing"


def test_optional_wrist_camera_is_live_when_present(fake):
    """The reBot's D405 colour and depth streams (global names, optional): live when the wire has them
    (the 16UC1 depth frames decode with the profile's scale), and its sensors' topics are plain profile
    rows that the check validates and never subscribes to."""
    p = load("rebot_b601_rs")
    srv = fake(wire_spec(p, include_optional=True))
    code, out = fleet("--url", srv.url, "--expect", "rebot_b601_rs")
    assert code == 0, out
    assert "camera /camera/camera/color/image_raw (optional): LIVE" in out and out.rstrip().endswith("PASS"), out
    assert "camera /camera/camera/aligned_depth_to_color/image_raw (optional): LIVE" in out, out
    assert srv.command_ops() == []
    subscribed = {o["topic"] for o in srv.ops_of("subscribe")}
    assert subscribed == {"/camera/camera/color/image_raw", "/camera/camera/aligned_depth_to_color/image_raw"}, subscribed


def test_x3_cameras_are_sampled_briefly_at_their_rate_cap(fake):
    """The ROSMASTER's Astra colour and depth (required), infrared (its frames optional) and the arm's USB camera
    (optional): on a wire with them all four are live. The check subscribes to the cameras only (never to the
    sensor topics, so never to the point clouds), each at its profile rate cap, samples the two frames that show a
    stream live and unsubscribes it at once, and sends nothing. Without the USB camera it is reported missing with
    the launch that starts it; a silent infrared stream is explained, not failed."""
    p = load("rosmaster_x3_plus")
    srv = fake(wire_spec(p, include_optional=True))
    code, out = fleet("--url", srv.url, "--expect", "rosmaster_x3_plus")
    assert code == 0 and out.rstrip().endswith("PASS"), out
    for line in ("camera /camera/rgb/image_raw: LIVE", "camera /camera/depth/image_raw: LIVE",
                 "camera /camera/ir/image_raw (optional): LIVE", "camera /usb_cam/image_raw (optional): LIVE"):
        assert line in out, out
    assert srv.command_ops() == []
    subs = srv.ops_of("subscribe")
    assert {o["topic"] for o in subs} == {c.topic for c in p.cameras}
    assert {o["topic"]: o["throttle_rate"] for o in subs} == {c.topic: c.throttle_ms() for c in p.cameras}
    unsub = {o["id"] for o in srv.ops_of("unsubscribe")}
    assert all(o["id"] in unsub for o in subs)
    # one stream at a time: each is unsubscribed before the next is subscribed, so the check never takes more of the
    # link than one stream (the profile's link budget holds)
    spans = []
    for o in subs:
        off = next(u for u in srv.ops_of("unsubscribe") if u["id"] == o["id"])
        spans.append((o["_t"], off["_t"]))
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), spans
    assert max(c.nominal_mbps() for c in p.cameras) <= p.link_budget_mbps
    for c in p.cameras:          # held only for its two frames: unsubscribed well within the window
        sub = next(o for o in subs if o["topic"] == c.topic)
        off = next(o for o in srv.ops_of("unsubscribe") if o["id"] == sub["id"])
        assert off["_t"] - sub["_t"] < 2 * c.throttle_ms() / 1000 + 1.5, c.topic
    srv.frame_encoding["/camera/ir/image_raw"] = "rgb8"            # not a single-channel infrared frame
    code, out = fleet("--url", srv.url, "--expect", "rosmaster_x3_plus")
    assert "camera /camera/ir/image_raw (optional): UNSUPPORTED" in out, out
    srv2 = fake(wire_spec(p))
    srv2.paused.add("/camera/ir/image_raw")                       # the driver ran without IR: advertised, silent
    code, out = fleet("--url", srv2.url, "--expect", "rosmaster_x3_plus")
    assert code == 0 and "camera /usb_cam/image_raw (optional): MISSING" in out and "laser_usb_bringup.launch" in out, out
    assert "camera /camera/ir/image_raw (optional): SILENT" in out and "restarted without it" in out, out


def test_depth_camera_frame_of_another_encoding_fails(fake):
    """A depth camera documents 16UC1 only: a depth frame in another encoding is reported unsupported."""
    p = load("rebot_b601_rs")
    srv = fake(wire_spec(p, include_optional=True))
    srv.frame_encoding["/camera/camera/aligned_depth_to_color/image_raw"] = "mono8"
    code, out = fleet("--url", srv.url, "--expect", "rebot_b601_rs")
    assert "camera /camera/camera/aligned_depth_to_color/image_raw (optional): UNSUPPORTED" in out, out
    assert "camera /camera/camera/color/image_raw (optional): LIVE" in out, out


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_wrong_type_fails(fake, pid):
    p = load(pid)
    row = next(e for e in p.required() if e.kind == "topic")
    bad = "std_msgs/Bool" if p.dialect == "ros1" else "std_msgs/msg/Bool"
    wire = p.resolve(row.name, p.namespace.default if p.namespace else "")
    srv = fake(retype(wire_spec(p), wire, bad))
    code, out = fleet("--url", srv.url, "--expect", pid)
    assert code != 0 and f"wrong type on topic {wire}" in out, out
    code, out = fleet("--url", srv.url)
    assert code != 0, out


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_missing_required_endpoint_fails(fake, pid):
    p = load(pid)
    row = p.required()[0]
    wire = p.resolve(row.name, p.namespace.default if p.namespace else "")
    srv = fake(drop(wire_spec(p), wire))
    code, out = fleet("--url", srv.url, "--expect", pid)
    assert code != 0 and wire in out, out


def test_failed_validation_still_reports_cameras(fake):
    """Every discovered robot's profile cameras are reported, also when it fails validation."""
    srv = fake(retype(wire_spec(load("ainex")), "/walking/command", "std_srvs/Empty"))
    code, out = fleet("--url", srv.url)
    assert code == 1 and "robot ainex: typed validation FAIL" in out, out
    assert "camera /camera/image_raw: LIVE" in out, out


def test_ambiguous_candidates_are_named(fake):
    srv = fake(merge(wire_spec(load("myagv")), wire_spec(load("rosmaster_x3_plus"))))
    code, out = fleet("--url", srv.url)
    assert code != 0 and "ambiguous" in out and "myagv" in out and "rosmaster_x3_plus" in out, out
    # each candidate's cameras, once (both list /usb_cam/image_raw: the myAGV's camera, the X3's arm camera)
    for topic in {c.topic for c in load("myagv").cameras + load("rosmaster_x3_plus").cameras}:
        assert len(re.findall(rf"camera {re.escape(topic)}( \(optional\))?:", out)) == 1, out
    code, out = fleet("--url", srv.url, "--expect", "myagv")
    assert code != 0 and "rosmaster_x3_plus" in out, out


def test_namespaced_candidates_are_named(fake):
    p = load("rebot_b601_rs")
    srv = fake(merge(wire_spec(p, namespace="arm1"), wire_spec(p, namespace="arm2")))
    code, out = fleet("--url", srv.url, "--expect", "rebot_b601_rs")
    assert code != 0 and "arm1" in out and "arm2" in out, out


def test_no_supported_robot(fake):
    srv = fake({"dialect": "ros2", "topics": [{"name": "/chatter", "type": "std_msgs/msg/String"}],
                "services": [], "actions": []})
    code, out = fleet("--url", srv.url)
    assert code != 0 and "no supported robot" in out, out


def test_unreachable_wire():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    code, out = fleet("--url", f"ws://127.0.0.1:{port}", timeout=30)
    assert code != 0 and "unreachable" in out, out


def test_stale_camera_fails(fake):
    p = load("ainex")
    srv = fake(wire_spec(p))
    srv.paused.add("/camera/image_raw")
    code, out = fleet("--url", srv.url, "--expect", "ainex")
    assert code != 0 and "camera /camera/image_raw: STALE" in out, out


def test_unsupported_encoding_is_not_live(fake):
    p = load("ainex")
    srv = fake(wire_spec(p))
    srv.frame_encoding["/camera/image_raw"] = "bgr8"      # the profile documents rgb8 only
    code, out = fleet("--url", srv.url, "--expect", "ainex")
    assert code != 0 and "UNSUPPORTED" in out, out


def test_missing_camera_reported(fake):
    srv = fake(drop(wire_spec(load("rosmaster_x3_plus")), "/camera/rgb/image_raw"))
    code, out = fleet("--url", srv.url, "--expect", "rosmaster_x3_plus")
    assert code != 0 and "camera /camera/rgb/image_raw: MISSING" in out, out


@pytest.mark.parametrize("unknown", ["turtlebot", "so100"])
def test_unknown_id(unknown):
    code, out = fleet("--url", "ws://127.0.0.1:1", "--expect", unknown)
    assert code == 2 and "unknown robot id" in out and "Accepted ids: myagv, ainex, rosmaster_x3_plus, rebot_b601_rs\n" in out, out
