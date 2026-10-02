"""The console against REAL rosbridge_suite servers in Docker (opt in: ``pytest -m live``).

Each container runs stock rosbridge_websocket + rosapi and a small node (``nodes/``) that
presents a profile's required interface with the documented stock types. Profiles whose
interface uses vendor message packages (ainex, rosmaster_x3_plus) are not covered here: those
packages are not installable in stock images.

    docker build -t robot-console-test:noetic -f tests/live/docker/Dockerfile.noetic tests/live/docker
    docker build -t robot-console-test:humble -f tests/live/docker/Dockerfile.humble tests/live/docker
    docker build -t robot-console-test:jazzy  -f tests/live/docker/Dockerfile.jazzy  tests/live/docker
"""

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
from robot_console.profiles import load
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


# ------------------------------------------------------------------ ROS 2 (Humble, Jazzy)

def test_humble_mycobot280_fleet_and_page_publish(wire):
    p = load("mycobot280")
    w = wire("robot-console-test:humble", wire_spec(p))
    code, out = fleet("--url", w.url, "--expect", "mycobot280")
    print(out)
    assert code == 0, out
    playwright = pytest.importorskip("playwright.sync_api")
    from robot_console.view import make_server
    http = make_server(w.url, None, None, 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    with playwright.sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page()
        page.goto(f"http://127.0.0.1:{http.server_address[1]}/")
        page.wait_for_function("window.RC && RC.ready === true", timeout=20000)
        assert page.locator('[data-testid="target"]').inner_text() == "mycobot280"
        c = page.locator('[data-control="arm_gripper_target"]')
        c.locator('input[data-field="j1"]').fill("0.5")
        c.locator('input[data-field="j1"]').press("Enter")              # a committed value is sent once
        page.wait_for_timeout(1500)
        b.close()
    http.shutdown()
    msgs = [e["msg"] for e in w.events() if e.get("topic") == "/joint_states"]
    assert len(msgs) == 1 and msgs[0]["position"][0] == 0.5 and len(msgs[0]["position"]) == 7


def test_jazzy_so101_fleet_and_page_streamed_goals(wire):
    """Dragging a slider streams goals over stock Jazzy rosbridge: each goal on its own
    connection (rosbridge blocks a client's ops while its goal runs), each preempting the
    previous one, the final value delivered, no cancel sent and no connection leaked."""
    p = load("so101")
    spec = wire_spec(p)
    for a in spec["actions"]:
        a["exec_s"] = 20.0
    w = wire("robot-console-test:jazzy", spec)
    code, out = fleet("--url", w.url, "--expect", "so101")
    print(out)
    assert code == 0 and "camera /image_raw: LIVE" in out, out
    assert "type not verifiable" in out
    assert "/usb_cam/set_camera_info" not in out, "present with its type, not missing or unexpected"
    code2, out2 = fleet("--url", w.url)
    assert code2 == 0, out2
    playwright = pytest.importorskip("playwright.sync_api")
    from robot_console.view import make_server
    counts = []
    rb = Rosbridge(w.url).connect()
    rb.subscribe("/client_count", "std_msgs/msg/Int32", lambda m: counts.append(m.get("data")))
    http = make_server(w.url, None, None, 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        with playwright.sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1400, "height": 1000})
            page.goto(f"http://127.0.0.1:{http.server_address[1]}/")
            page.wait_for_function("window.RC && RC.ready === true", timeout=20000)
            assert page.locator('[data-testid="target"]').inner_text() == "so101"
            page.wait_for_function("document.querySelector('[data-testid=camera]').dataset.state === 'live'", timeout=10000)
            time.sleep(0.5)
            base = counts[-1]
            r = page.locator('[data-control="arm_trajectory"] input[data-field="shoulder_pan"]').locator(
                "xpath=ancestor::div[contains(concat(' ', @class, ' '), ' field ')][1]").locator('input[type="range"]')
            r.scroll_into_view_if_needed()
            box = r.bounding_box()
            y = box["y"] + box["height"] / 2
            page.mouse.move(box["x"] + box["width"] / 2, y)
            page.mouse.down()
            for i in range(1, 61):                                     # a 3 s drag
                page.mouse.move(box["x"] + box["width"] * (0.5 + 0.006 * i), y)
                page.wait_for_timeout(50)
            page.mouse.up()
            final = float(page.locator('[data-control="arm_trajectory"] input[data-field="shoulder_pan"]').input_value())
            page.wait_for_function("document.querySelector('[data-control=arm_trajectory] [data-testid=status]')"
                                   ".innerText.includes('running')", timeout=10000)
            time.sleep(1.5)
            peak, settled = max(counts), counts[-1]
            b.close()
    finally:
        http.shutdown()
        rb.close()
    ev = [e for e in w.events() if e.get("action") == "/joint_trajectory_controller/follow_joint_trajectory"]
    kinds = [e["ev"] for e in ev]
    execs = [e["goal"] for e in ev if e["ev"] == "goal_exec"]
    print(kinds, counts)
    assert kinds.count("goal") >= 5 and "cancel" not in kinds and "goal_canceled" not in kinds, kinds
    assert kinds.count("goal_preempted") == kinds.count("goal") - 1, kinds
    assert execs[-1]["trajectory"]["points"][0]["positions"][0] == final > 1.0
    assert execs[-1]["trajectory"]["joint_names"][0] == "shoulder_pan_joint"
    assert peak <= base + 3, counts                                   # goal connections bounded
    assert settled == base + 1, counts                                # only the running goal's connection
