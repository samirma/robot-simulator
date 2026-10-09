"""The console against REAL rosbridge_suite servers in Docker (opt in: ``pytest -m live``).

Each container runs stock rosbridge_websocket + rosapi and a small node (``nodes/``) that
presents a profile's required interface with the documented types. The reBot's vendor message
package (rebotarm_msgs) is built from its pinned source into the Jazzy image; the AiNex and
ROSMASTER vendor packages live in the vendors' ROS 1 workspaces and are not installable in stock
images, so those profiles are not covered here.

    docker build -t robot-console-test:noetic -f tests/live/docker/Dockerfile.noetic tests/live/docker
    docker build -t robot-console-test:jazzy  -f tests/live/docker/Dockerfile.jazzy  tests/live/docker
"""

import contextlib
import dataclasses
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import pygame as pg
import pytest

from robot_console import camera as cam
from robot_console.discovery import fetch_graph
from robot_console.profiles import load, load_all, profile_json
from robot_console.rosbridge import Rosbridge
from robot_console.teleop import TeleopApp
from wirespec import retype, wire_spec

pytestmark = pytest.mark.live
HERE = os.path.dirname(os.path.abspath(__file__))


def _docker_ok():
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


if not shutil.which("docker") or not _docker_ok():
    pytest.skip("docker is not available", allow_module_level=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Wire:
    def __init__(self, image, spec, tmp):
        self.port = free_port()
        self.url = f"ws://127.0.0.1:{self.port}"
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "spec.json"), "w") as f:
            json.dump(spec, f)
        self.name = f"rc-live-{self.port}"
        r = subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name, "-e", f"PORT={self.port}",
                            "-p", f"{self.port}:{self.port}", "-v", f"{HERE}/nodes:/nodes:ro",
                            "-v", f"{tmp}:/spec:ro", image, "/nodes/run_wire.sh", "/spec/spec.json"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        self.wait_ready(spec)

    def wait_ready(self, spec, timeout=90):
        want = {t["name"] for t in spec["topics"]}
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if '"ev": "ready"' in self.logs():
                try:
                    rb = Rosbridge(self.url).connect(3)
                    try:
                        topics = set(rb.call_service("/rosapi/topics", timeout=3)["topics"])
                        if want <= topics:
                            time.sleep(1.0)
                            return
                    finally:
                        rb.close()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.5)
        raise AssertionError("wire not ready:\n" + self.logs()[-3000:])

    def logs(self):
        return subprocess.run(["docker", "logs", self.name], capture_output=True, text=True).stdout

    def events(self):
        out = []
        for line in self.logs().splitlines():
            if line.startswith("WIRE "):
                out.append(json.loads(line[5:]))
        return out

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


@pytest.fixture
def wire(tmp_path):
    made = []

    def start(image, spec):
        w = Wire(image, spec, str(tmp_path / f"w{len(made)}"))
        made.append(w)
        return w
    yield start
    for w in made:
        w.close()


def fleet(*args):
    r = subprocess.run([sys.executable, "-m", "robot_console.fleet", *args], capture_output=True, text=True,
                       timeout=120)
    return r.returncode, r.stdout + r.stderr


# ------------------------------------------------------------------ ROS 1 (Noetic)

def test_noetic_myagv_fleet_teleop_and_camera(wire):
    p = load("myagv")
    spec = wire_spec(p)
    spec["topics"].append({"name": "/test_cam/image_raw", "type": "sensor_msgs/Image", "direction": "out",
                           "camera": True, "rate_hz": 10, "encoding": "bgr8"})
    w = wire("robot-console-test:noetic", spec)
    code, out = fleet("--url", w.url, "--expect", "myagv")
    print(out)
    assert code == 0 and "robot myagv: typed validation PASS" in out, out
    assert "camera /usb_cam/image_raw: LIVE" in out, out
    code, out = fleet("--url", w.url)
    assert code == 0, out
    # a raw image stream decodes live over real ROS 1 rosbridge
    rb = Rosbridge(w.url).connect()
    g = fetch_graph(rb)
    assert g.dialect == "ros1"
    cs = cam.CameraSet(rb, [s for s in cam.untied_specs(g) if s.topic == "/test_cam/image_raw"])
    end = time.monotonic() + 10
    s = cs.streams["/test_cam/image_raw"]
    while time.monotonic() < end and s.frames < 3:
        cs.poll()
        time.sleep(0.05)
    assert s.state() == cam.LIVE and s.frame.shape == (48, 64, 3)
    cs.close()
    rb.close()
    # teleop through the real window event path: hold W, release, Esc
    steps = [(3, [pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN)]), (5, [pg.event.Event(pg.KEYDOWN, key=pg.K_w)]),
             (20, [pg.event.Event(pg.KEYUP, key=pg.K_w)]), (25, [pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE)])]

    class Script:
        n = 0

        def __call__(self):
            self.n += 1
            while steps and steps[0][0] <= self.n:
                for e in steps.pop(0)[1]:
                    pg.event.post(e)
            return pg.event.get()
    code = TeleopApp(w.url, p, None, events=Script(), max_seconds=30).run()
    assert code == 0
    time.sleep(1.0)
    twists = [e["msg"] for e in w.events() if e.get("topic") == "/cmd_vel"]
    xs = [t["linear"]["x"] for t in twists]
    speed = p.teleop_base.axes["x"].speed
    assert xs[0] == 0.0, "start-up stop first"
    assert speed in xs and xs[-1] == 0.0 and all(abs(x) <= p.teleop_base.axes["x"].limit for x in xs)
    # a handled signal stops too
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    proc = subprocess.Popen([sys.executable, "-m", "robot_console.teleop", "--url", w.url, "--robot", "myagv"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    n0 = len(twists)
    end = time.monotonic() + 20
    while time.monotonic() < end and len([e for e in w.events() if e.get("topic") == "/cmd_vel"]) <= n0:
        time.sleep(0.2)
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 130, out
    time.sleep(1.0)
    assert len([e for e in w.events() if e.get("topic") == "/cmd_vel"]) == n0 + 2


def test_noetic_wrong_type_fails(wire):
    w = wire("robot-console-test:noetic", retype(wire_spec(load("myagv")), "/odom", "std_msgs/String"))
    code, out = fleet("--url", w.url, "--expect", "myagv")
    assert code != 0 and "wrong type on topic /odom" in out, out


# ------------------------------------------------------------------ ROS 2 (Jazzy)

ARM = "/rebotarm/follow_joint_trajectory"
ARM_GOAL = {"trajectory": {"joint_names": [f"joint{i}" for i in range(1, 7)],
                           "points": [{"positions": [0.0] * 6, "time_from_start": {"sec": 1, "nanosec": 0}}]}}


def preempting_variant(p):
    """The profile as if its action servers preempted a running goal with a new one, as a
    ros2_control controller does: no packaged profile's server preempts, so the page's streaming
    of preempting goals is checked on this variant (tests only)."""
    return dataclasses.replace(p, controls=tuple(dataclasses.replace(c, busy="preempt") for c in p.controls))


@contextlib.contextmanager
def page_on(url):
    """The view.sh page for `url` in headless Chromium, ready (target validated or refused)."""
    playwright = pytest.importorskip("playwright.sync_api")
    from robot_console.view import make_server
    http = make_server(url, None, None, 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        with playwright.sync_playwright() as pw:
            b = pw.chromium.launch()
            try:
                page = b.new_page(viewport={"width": 1400, "height": 1000})
                page.goto(f"http://127.0.0.1:{http.server_address[1]}/")
                page.wait_for_function("window.RC && RC.ready === true", timeout=20000)
                yield page
            finally:
                b.close()
    finally:
        http.shutdown()


def drag_joint1(page, steps=60):
    """Drag the arm goal's joint1 slider with the mouse from its middle rightwards, one step per
    50 ms (60 steps: a 3 s drag); the value it ends on."""
    field = page.locator('[data-control="arm_trajectory"] input[data-field="joint1"]')
    r = field.locator("xpath=ancestor::div[contains(concat(' ', @class, ' '), ' field ')][1]").locator('input[type="range"]')
    r.scroll_into_view_if_needed()
    box = r.bounding_box()
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + box["width"] / 2, y)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(box["x"] + box["width"] * (0.5 + 0.006 * i), y)
        page.wait_for_timeout(50)
    page.mouse.up()
    return float(field.input_value())


def wait_final(events, final, done, timeout=20):
    """Wait until the latest arm goal executed carries joint1 `final` and `done(kinds)` holds for
    the arm events' kinds; the arm events."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        ev = events()
        execs = [e["goal"] for e in ev if e["ev"] == "goal_exec"]
        if execs and execs[-1]["trajectory"]["points"][0]["positions"][0] == final and done([e["ev"] for e in ev]):
            break
        time.sleep(0.3)
    return events()


def client_counts(url):
    """rosbridge's /client_count as it changes (the page's goal connections among them)."""
    counts = []
    rb = Rosbridge(url).connect()
    rb.subscribe("/client_count", "std_msgs/msg/Int32", lambda m: counts.append(m.get("data")))
    return rb, counts


def test_jazzy_rebot_fleet_and_page_goals_one_at_a_time(wire):
    """Over stock Jazzy rosbridge, against an arm action server that rejects a goal while one
    runs, as the reBot's controller does (console spec §2.2): each goal on its own connection
    (rosbridge blocks a client's ops while its goal runs); a drag keeps one goal outstanding, its
    held value going out after the running goal's result, so the server rejects none of them;
    the final value delivered, no cancel sent and no connection leaked. While another client's
    goal runs, the page's goal is rejected, which rosbridge reports and the page shows as
    rejected. The wire is the reBot's at its default namespace /rebotarm, with the vendor's
    rebotarm_msgs types. Its optional D405 colour and depth streams are on the wire (live over rosbridge, the depth drawn as its false-colour map), and its
    manufacturer sensors, published in sensor-data (best effort) QoS and, for arm_status, transient
    local, reach the page's read-only sensor panel."""
    p = load("rebot_b601_rs")
    spec = wire_spec(p, include_optional=True)
    for a in spec["actions"]:
        a["exec_s"] = 1.5
    assert {a["name"]: a["busy"] for a in spec["actions"]}[ARM] == "reject"
    w = wire("robot-console-test:jazzy", spec)

    def arm_events():
        return [e for e in w.events() if e.get("action") == ARM]
    code, out = fleet("--url", w.url, "--expect", "rebot_b601_rs")
    print(out)
    assert code == 0 and "robot rebot_b601_rs (namespace /rebotarm): typed validation PASS" in out, out
    assert "camera /camera/camera/color/image_raw (optional): LIVE" in out, out   # the D405, a global name
    assert "camera /camera/camera/aligned_depth_to_color/image_raw (optional): LIVE" in out, out   # and its depth
    assert "type not verifiable" in out
    assert "/rebotarm/arm_status" not in out, "present with its vendor type, not missing or unexpected"
    code2, out2 = fleet("--url", w.url)
    assert code2 == 0, out2
    rb, counts = client_counts(w.url)
    try:
        with page_on(w.url) as page:
            assert page.locator('[data-testid="target"]').inner_text() == "rebot_b601_rs (namespace /rebotarm)"
            page.wait_for_function("[...document.querySelectorAll('[data-testid=camera]')].length === 2 && "
                                   "[...document.querySelectorAll('[data-testid=camera]')].every((e) => e.dataset.state === 'live')", timeout=15000)
            assert page.locator('[data-testid="camera"]').evaluate_all("els => els.map(e => e.dataset.topic)") == [
                "/camera/camera/color/image_raw", "/camera/camera/aligned_depth_to_color/image_raw"]
            # the depth stream (16UC1 over real rosbridge) is drawn as the false-colour map: pixel 0 has no
            # depth (black), pixel 1 is at the camera's near end of the map
            px = page.locator('[data-kind="depth"] canvas').first.evaluate(
                "c => Array.from(c.getContext('2d').getImageData(0, 0, 2, 1).data)")
            assert px[:3] == [0, 0, 0] and px[4:7] == list(cam.DEPTH_STOPS[0][1]), px
            page.wait_for_function("[...document.querySelectorAll('[data-testid=sensor]')].length === 9 && "
                                   "[...document.querySelectorAll('[data-testid=sensor]')].every((e) => ['live', 'latched'].includes(e.dataset.state))",
                                   timeout=15000)
            value = lambda sid, f: page.locator(f'[data-sensor="{sid}"] [data-field="{f}"] [data-testid="sensor-value"]').get_attribute("data-value")
            assert [value("joint2_state", f) for f in ("joint_name", "position", "torque", "status_code")] == ["joint2", "0.5", "3", "5"]
            assert value("joint_states", "joint1_position") == "0.1"
            assert [value("arm_status", f) for f in ("mode", "enabled", "state_machine")] == ["mit", "true", "IDLE"]
            assert page.locator('[data-sensor="arm_status"] [data-field="per_joint_status_code"] .sf-value').inner_text() == "0, 0, 0, 0, 0, 0"
            assert page.locator('[data-sensor="joint2_state"] [data-testid="sensor-bit"][data-set="true"]').evaluate_all(
                "els => els.map(e => e.textContent)") == ["undervoltage", "overtemperature"]
            time.sleep(0.5)
            base = counts[-1]
            final = drag_joint1(page)
            wait_final(arm_events, final, lambda k: k[-1] == "goal_succeeded")   # the held final value, after the running goal
            time.sleep(1.0)
            peak, settled = max(counts), counts[-1]
            ev = arm_events()
            # Another client's goal runs: the page's next goal is rejected, and shown so.
            # (Its first feedback says it runs; the server rejects goals for exec_s from then.)
            other, running = Rosbridge(w.url).connect(), threading.Event()
            seen = other._dispatch
            other._dispatch = lambda m: (running.set() if m.get("op") == "action_feedback" else None, seen(m))
            other._send({"op": "send_action_goal", "id": "other-client", "action": ARM, "feedback": True,
                         "action_type": "control_msgs/action/FollowJointTrajectory", "args": ARM_GOAL})
            assert running.wait(10), "the other client's goal did not start"
            f2 = page.locator('[data-control="arm_trajectory"] input[data-field="joint2"]')
            f2.fill("0.5")
            f2.press("Enter")
            page.wait_for_function("document.querySelector('[data-control=arm_trajectory]').dataset.phase === 'rejected'",
                                   timeout=10000)
            shown = page.locator('[data-control="arm_trajectory"] [data-testid="status"]').inner_text()
            other.close()
    finally:
        rb.close()
    kinds = [e["ev"] for e in ev]
    execs = [e["goal"] for e in ev if e["ev"] == "goal_exec"]
    print(kinds, counts, shown)
    assert kinds.count("goal") >= 2 and "cancel" not in kinds and "goal_canceled" not in kinds, kinds
    assert "goal_rejected" not in kinds and "goal_preempted" not in kinds, kinds    # one goal at a time
    assert kinds.count("goal_succeeded") == kinds.count("goal"), kinds
    assert execs[-1]["trajectory"]["points"][0]["positions"][0] == final > 1.0
    assert execs[-1]["trajectory"]["joint_names"] == [f"joint{i}" for i in range(1, 7)]
    assert peak <= base + 3, counts                                   # goal connections bounded
    assert settled == base, counts                                    # every finished goal's connection closed
    assert "goal_rejected" in [e["ev"] for e in arm_events()]
    assert shown.startswith("rejected:") and "Action goal was rejected" in shown, shown


def test_jazzy_page_streams_preempting_goals(wire, monkeypatch):
    """Over stock Jazzy rosbridge, against action servers that preempt a running goal with a new
    one (the reBot's profile with every action control's busy set to preempt, served by
    nodes/ros2_wire.py's preempting server; console spec §2.2, §4): a drag streams goals at about
    5 per second, each on its own connection and each preempting the previous one; the final value
    delivered and still running, no cancel sent, the goal connections bounded and only the
    running goal's left open."""
    import robot_console.view as view_mod
    p = preempting_variant(load("rebot_b601_rs"))
    packaged = view_mod.all_profiles_json
    monkeypatch.setattr(view_mod, "all_profiles_json", lambda: dict(
        packaged(), profiles=[profile_json(p if x.id == p.id else x) for x in load_all()]))
    spec = wire_spec(p)
    for a in spec["actions"]:
        a["exec_s"] = 20.0                                       # goals outlive the test unless preempted
    assert {a["busy"] for a in spec["actions"]} == {"preempt"}
    w = wire("robot-console-test:jazzy", spec)

    def arm_events():
        return [e for e in w.events() if e.get("action") == ARM]
    rb, counts = client_counts(w.url)
    try:
        with page_on(w.url) as page:
            assert page.locator('[data-testid="target"]').inner_text() == "rebot_b601_rs (namespace /rebotarm)"
            assert "preempts" in page.locator('[data-control="arm_trajectory"] .ctl-kind').inner_text()
            time.sleep(0.5)
            base = counts[-1]
            t0 = time.monotonic()
            final = drag_joint1(page)
            dur = time.monotonic() - t0
            wait_final(arm_events, final, lambda k: k.count("goal_preempted") == k.count("goal") - 1)
            page.wait_for_function("document.querySelector('[data-control=arm_trajectory]').dataset.phase === 'running'",
                                   timeout=10000)
            time.sleep(1.5)                                       # superseded goals' connections close
            peak, settled = max(counts), counts[-1]
            shown = page.locator('[data-control="arm_trajectory"] [data-testid="status"]').inner_text()
    finally:
        rb.close()
    ev = arm_events()
    kinds = [e["ev"] for e in ev]
    execs = [e["goal"] for e in ev if e["ev"] == "goal_exec"]
    print(kinds, counts, shown)
    assert 5 <= kinds.count("goal") <= dur / 0.2 + 3, (kinds.count("goal"), dur)   # about 5 per s
    assert "cancel" not in kinds and "goal_canceled" not in kinds and "goal_rejected" not in kinds, kinds
    assert kinds.count("goal_preempted") == kinds.count("goal") - 1, kinds        # each preempts the previous
    assert "goal_succeeded" not in kinds, kinds                                    # the latest still runs
    assert execs[-1]["trajectory"]["points"][0]["positions"][0] == final > 1.0
    assert peak <= base + 3, counts                                   # goal connections bounded
    assert settled == base + 1, counts                                # only the running goal's connection
    assert "goal running" in shown, shown
