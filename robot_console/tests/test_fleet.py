"""``python -m robot_console.fleet`` (console spec §2.4, §4)."""

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
    srv = fake(wire_spec(load(pid)))
    code, out = fleet("--url", srv.url)
    assert code == 0, out
    assert f"robot {pid}: typed validation PASS" in out and out.rstrip().endswith("PASS")
    for c in load(pid).cameras:
        assert f"camera {c.topic}: LIVE" in out
    code, out = fleet("--url", srv.url, "--expect", pid)
    assert code == 0, out
    assert srv.command_ops() == [], "the check publishes nothing"


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_wrong_type_fails(fake, pid):
    p = load(pid)
    row = next(e for e in p.required() if e.kind == "topic")
    bad = "std_msgs/Bool" if p.dialect == "ros1" else "std_msgs/msg/Bool"
    srv = fake(retype(wire_spec(p), row.name, bad))
    code, out = fleet("--url", srv.url, "--expect", pid)
    assert code != 0 and f"wrong type on topic {row.name}" in out, out
    code, out = fleet("--url", srv.url)
    assert code != 0, out


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_missing_required_endpoint_fails(fake, pid):
    p = load(pid)
    row = p.required()[0]
    srv = fake(drop(wire_spec(p), row.name))
    code, out = fleet("--url", srv.url, "--expect", pid)
    assert code != 0 and row.name in out, out


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
    for c in load("myagv").cameras + load("rosmaster_x3_plus").cameras:
        assert out.count(f"camera {c.topic}:") == 1, out     # each candidate's cameras, once
    code, out = fleet("--url", srv.url, "--expect", "myagv")
    assert code != 0 and "rosmaster_x3_plus" in out, out


def test_namespaced_candidates_are_named(fake):
    p = load("so101")
    srv = fake(merge(wire_spec(p, namespace="arm1"), wire_spec(p, namespace="arm2")))
    code, out = fleet("--url", srv.url, "--expect", "so101")
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


@pytest.mark.parametrize("unknown", ["turtlebot", "myagv_mycobot280"])
def test_unknown_id(unknown):
    code, out = fleet("--url", "ws://127.0.0.1:1", "--expect", unknown)
    assert code == 2 and "unknown robot id" in out and "Accepted ids" in out, out
