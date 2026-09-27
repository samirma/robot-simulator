"""The camera and control page (`live_cameras.html`, served by `bin/view.sh`), in a browser.

Console spec §4, "View page": typed discovery, opt-in controls, bounded commands,
cancellation on unload, and complete re-advertisement and re-subscription after a
reconnect. Each is checked by loading the real page into a headless Chromium
(`tests/browser.py`) against `ViewBridge`, a fake rosbridge that records every op each
connection sends and answers the `rosapi` calls the page makes. What the page *sends* is
the evidence; its DOM is only the way the test presses its buttons.

The parity tests at the bottom need no browser: they hold the page's `CONTRACT` block
equal to the console's own constants and to the ROS files.
"""

from __future__ import annotations

import http.server
import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

import pytest
from websockets.sync.server import serve

from browser import Browser, find_browser

CONSOLE = Path(__file__).resolve().parents[1]
PAGE = CONSOLE / "live_cameras.html"
SPECS = CONSOLE.parent / "robots_specs"

ROS2_IMAGE = "sensor_msgs/msg/CompressedImage"
ROS1_IMAGE = "sensor_msgs/CompressedImage"

#: A fleet like `kitchen.sh serve --robots so101,myagv,ainex`, plus decoys: a namespace
#: whose `/cmd_vel` has the wrong type, and image topics that are not viewable streams.
WIRE = {
    "/so101/joint_trajectory_controller/joint_trajectory": "trajectory_msgs/msg/JointTrajectory",
    "/so101/joint_states": "sensor_msgs/msg/JointState",
    "/so101/robot_description": "std_msgs/msg/String",
    "/so101/wrist/image_raw": "sensor_msgs/msg/Image",
    "/so101/wrist/image_raw/compressed": ROS2_IMAGE,
    "/so101/wrist/image_raw/compressedDepth": ROS2_IMAGE,
    "/ainex/walking/set_param": "ainex_interfaces/WalkingParam",
    "/ainex/walking/is_walking": "std_msgs/Bool",
    "/ainex/head_pan_controller/command": "ainex_interfaces/HeadState",
    "/ainex/head_tilt_controller/command": "ainex_interfaces/HeadState",
    "/ainex/app/set_action": "std_msgs/String",
    "/ainex/camera/image_raw/compressed": ROS1_IMAGE,
    "/myagv/cmd_vel": "geometry_msgs/Twist",
    "/myagv/camera/image_raw/compressed": ROS1_IMAGE,
    "/scene/overhead/color/compressed": ROS2_IMAGE,
    "/scene/side/color/compressed": ROS2_IMAGE,
    "/decoy/cmd_vel": "std_msgs/String",
    "/decoy/walking/set_param": "std_msgs/String",
}
ACTIONS = {
    "/so101/joint_trajectory_controller/follow_joint_trajectory":
        "control_msgs/action/FollowJointTrajectory",
    "/so101/gripper_controller/gripper_cmd": "control_msgs/action/ParallelGripperCommand",
}
EXPECTED_CAMERAS = sorted([
    "/ainex/camera/image_raw/compressed", "/myagv/camera/image_raw/compressed",
    "/scene/overhead/color/compressed", "/scene/side/color/compressed",
    "/so101/wrist/image_raw/compressed",
])
#: What a watching, never-enabled page may send: subscriptions and rosapi questions.
PASSIVE_OPS = {"subscribe", "unsubscribe", "call_service"}


class ViewBridge:
    """A fake rosbridge that records every op, per connection, in arrival order."""

    def __init__(self, topics: dict[str, str], actions: dict[str, str]) -> None:
        self.topics, self.actions = dict(topics), dict(actions)
        self.connections: list[list[dict]] = []
        self._live: list = []
        self._lock = threading.Lock()
        self._server = serve(self._handle, "127.0.0.1", 0, compression=None, max_size=None)
        self.port = self._server.socket.getsockname()[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.shutdown()

    def _handle(self, connection) -> None:
        log: list[dict] = []
        with self._lock:
            self.connections.append(log)
            self._live.append(connection)
        try:
            for raw in connection:
                message = json.loads(raw)
                with self._lock:
                    log.append(message)
                if message.get("op") == "call_service":
                    connection.send(json.dumps(self._answer(message)))
        except Exception:
            pass
        finally:
            with self._lock:
                if connection in self._live:
                    self._live.remove(connection)

    def _answer(self, message: dict) -> dict:
        service, args = message.get("service"), message.get("args") or {}
        values: dict = {}
        if service == "/rosapi/topics":
            values = {"topics": list(self.topics), "types": list(self.topics.values())}
        elif service == "/rosapi/action_servers":
            values = {"action_servers": sorted(self.actions)}
        elif service == "/rosapi/action_type":
            values = {"type": self.actions.get(args.get("action"), "")}
        return {"op": "service_response", "id": message.get("id"), "service": service,
                "values": values, "result": True}

    # -- what the test does to the page

    def finish_goal(self, goal_id: str) -> None:
        for connection in list(self._live):
            connection.send(json.dumps({"op": "action_result", "id": goal_id, "status": 4,
                                        "values": {}, "result": True}))

    def drop_all(self) -> None:
        for connection in list(self._live):
            connection.close()

    def publish(self, topic: str, msg: dict) -> None:
        for connection in list(self._live):
            connection.send(json.dumps({"op": "publish", "topic": topic, "msg": msg}))

    # -- what the test reads back

    def ops(self, index: int = -1, op: Optional[str] = None) -> list[dict]:
        with self._lock:
            log = list(self.connections[index]) if self.connections else []
        return [m for m in log if op is None or m.get("op") == op]

    def all_ops(self) -> list[dict]:
        with self._lock:
            return [m for log in self.connections for m in log]

    def wait(self, predicate, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError("the page never sent what was expected")
            time.sleep(0.05)


class _PageServer:
    def __init__(self) -> None:
        body = PAGE.read_bytes()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def url(self, ws_url: str) -> str:
        return f"http://127.0.0.1:{self.port}/?url={ws_url}"

    def stop(self) -> None:
        self._server.shutdown()


@pytest.fixture(scope="module")
def browser():
    executable = find_browser()
    if executable is None:
        pytest.skip("no Chromium-family browser found; set VIEW_TEST_BROWSER to one")
    b = Browser(executable)
    try:
        yield b
    finally:
        b.close()


@pytest.fixture
def page(browser):
    """The page loaded against a fresh `ViewBridge`, discovery settled."""
    bridge = ViewBridge(WIRE, ACTIONS)
    server = _PageServer()
    browser.navigate(server.url(bridge.url))
    browser.wait_for("window.viewPage && viewPage.members().length === 4 "
                     "&& viewPage.cameras().length === 5")
    browser.wait_for("document.querySelector('.control[data-kind=so101] .note')"
                     ".textContent.includes('trajectory goals')")
    try:
        yield browser, bridge
    finally:
        browser.navigate("about:blank")
        time.sleep(0.2)
        bridge.stop()
        server.stop()


def _set(browser, kind: str, key: str, value: float) -> None:
    """Move a slider as a user releasing it would: set the value, fire `change`."""
    browser.evaluate(
        f"(() => {{ const i = document.querySelector('.control[data-kind={kind}] "
        f"input[data-key={key}]'); i.value = {value}; "
        "i.dispatchEvent(new Event('change')); })()")


def _enable(browser, kind: str) -> None:
    browser.evaluate(f"document.querySelector('.control[data-kind={kind}] .enable').click()")


def _commands(ops: list[dict]) -> list[dict]:
    return [m for m in ops if m.get("op") not in PASSIVE_OPS
            or (m.get("op") == "call_service" and not m["service"].startswith("/rosapi/"))]


# ------------------------------------------------------------------ discovery


def test_members_and_cameras_are_discovered_by_type(page) -> None:
    browser, bridge = page
    members = browser.evaluate("viewPage.members()")
    assert sorted((m["ns"], m["kind"]) for m in members) == [
        ("ainex", "ainex"), ("myagv", "myagv"), ("scene", "scene"), ("so101", "so101")]
    # Both dialects' streams, and nothing that is not an image_transport `compressed` one.
    assert sorted(browser.evaluate("viewPage.cameras()")) == EXPECTED_CAMERAS
    subscribed = {m["topic"]: m["type"] for m in bridge.ops(op="subscribe")}
    for topic in EXPECTED_CAMERAS:
        assert subscribed[topic] == WIRE[topic]
    # The decoy namespace named two signatures with the wrong types: no panel.
    assert browser.evaluate("document.querySelector('.control[data-ns=decoy]')") is None


def test_classify_is_typed_and_namespace_exact(browser) -> None:
    browser.navigate("about:blank")
    browser.evaluate("1")
    server = _PageServer()
    bridge = ViewBridge({}, {})
    try:
        browser.navigate(server.url(bridge.url))
        browser.wait_for("window.viewPage")
        result = browser.evaluate("""viewPage.classify({
            '/cmd_vel': 'geometry_msgs/Twist',
            '/walking/set_param': 'ainex_interfaces/WalkingParam',
            '/robot_2/joint_trajectory_controller/joint_trajectory': 'trajectory_msgs/JointTrajectory',
            '/scene/overhead/color/compressed': 'sensor_msgs/CompressedImage',
            '/cam/image_raw/compressed': 'sensor_msgs/Image',
        })""")
        # A bare AiNex wins its namespace over the bare myAGV signature beside it; a ROS 1
        # spelling of the arm's ROS 2 type is not the arm; a rig overhead in the wrong
        # dialect is not the rig, but is still a camera.
        assert result["members"] == [{"kind": "ainex", "ns": ""}]
        assert [c["topic"] for c in result["cameras"]] == ["/scene/overhead/color/compressed"]
    finally:
        browser.navigate("about:blank")
        bridge.stop()
        server.stop()


# ------------------------------------------------------------------ opt-in


def test_nothing_is_sent_until_a_robot_is_enabled(page) -> None:
    browser, bridge = page
    time.sleep(0.5)
    # Sliders moved on disabled panels are ignored, not queued.
    _set(browser, "so101", "shoulder_pan_joint", 0.5)
    _set(browser, "ainex", "head_pan", 0.5)
    browser.evaluate("document.querySelector('.control[data-kind=ainex] button[data-key=\"action:wave\"]').click()")
    time.sleep(0.5)
    assert _commands(bridge.all_ops()) == []
    assert not bridge.ops(op="advertise")
    # The myAGV cannot be enabled at all.
    assert browser.evaluate(
        "document.querySelector('.control[data-kind=myagv] .enable').disabled") is True

    _enable(browser, "ainex")
    bridge.wait(lambda: len(bridge.ops(op="advertise")) == 3)
    _set(browser, "ainex", "head_pan", 0.5)
    bridge.wait(lambda: bridge.ops(op="publish"))
    # Enabling one robot enabled that robot only.
    _set(browser, "so101", "shoulder_pan_joint", 0.5)
    time.sleep(0.3)
    assert not bridge.ops(op="send_action_goal")


# ------------------------------------------------------------------ bounded commands


def test_every_command_is_bounded_and_on_the_official_interface(page) -> None:
    browser, bridge = page
    _enable(browser, "so101")
    _enable(browser, "ainex")
    bridge.publish("/so101/joint_states", {"name": ["shoulder_pan_joint"], "position": [0.0]})
    _set(browser, "so101", "shoulder_pan_joint", 99)      # past the range: the input clamps
    _set(browser, "so101", "wrist_roll_joint", -99)
    _set(browser, "so101", "gripper_joint", 99)
    _set(browser, "ainex", "head_pan", 99)
    _set(browser, "ainex", "head_tilt", -99)
    browser.evaluate("document.querySelector('.control[data-kind=ainex] button[data-key=\"action:wave\"]').click()")
    bridge.wait(lambda: len(bridge.ops(op="send_action_goal")) >= 3
                and len(bridge.ops(op="publish")) >= 3)

    limits = json.loads(re.search(r'"fallback_limits": (\{.*?\})', PAGE.read_text(), re.S).group(1))
    for goal in bridge.ops(op="send_action_goal"):
        assert goal["action"] in ACTIONS and goal["action_type"] == ACTIONS[goal["action"]]
        if goal["action"].endswith("follow_joint_trajectory"):
            trajectory = goal["args"]["trajectory"]
            assert len(trajectory["points"]) == 1, "one point: a move that ends by itself"
            point = trajectory["points"][0]
            seconds = point["time_from_start"]["sec"] + point["time_from_start"]["nanosec"] * 1e-9
            assert 1.0 <= seconds <= 6.0
            for joint, value in zip(trajectory["joint_names"], point["positions"]):
                lo, hi = limits[joint]
                assert lo - 1e-9 <= value <= hi + 1e-9, (joint, value)
        else:
            (value,) = goal["args"]["command"]["position"]
            lo, hi = limits["gripper_joint"]
            assert lo <= value <= hi
    heads = [m for m in bridge.ops(op="publish") if "head_" in m["topic"]]
    assert {m["topic"] for m in heads} == {"/ainex/head_pan_controller/command",
                                           "/ainex/head_tilt_controller/command"}
    for m in heads:
        assert abs(m["msg"]["position"]) <= 2.09 and m["msg"]["duration"] > 0
    assert {"topic": "/ainex/app/set_action", "msg": {"data": "wave"}} in [
        {"topic": m["topic"], "msg": m["msg"]} for m in bridge.ops(op="publish")]

    # Nothing anywhere starts sustained motion, and no control exists that could.
    for m in _commands(bridge.all_ops()):
        name = m.get("topic") or m.get("action") or m.get("service")
        assert not name.endswith(("/cmd_vel", "/walking/command", "/walking/set_param",
                                  "/app/set_walking_param")), m
    labels = browser.evaluate("[...document.querySelectorAll('button')].map(b => b.textContent)")
    assert not [label for label in labels if re.search(r"\b(walk|drive|start)\b", label, re.I)
                and label != "walk_ready"]


# ------------------------------------------------------------------ unload


def test_unload_cancels_every_unfinished_goal(page) -> None:
    browser, bridge = page
    _enable(browser, "so101")
    _set(browser, "so101", "shoulder_pan_joint", 0.3)
    _set(browser, "so101", "gripper_joint", 0.5)
    _set(browser, "so101", "shoulder_pan_joint", 0.2)
    bridge.wait(lambda: len(bridge.ops(op="send_action_goal")) == 3)
    sent = [g["id"] for g in bridge.ops(op="send_action_goal")]
    bridge.finish_goal(sent[0])
    browser.wait_for(f"!viewPage.goals().includes({json.dumps(sent[0])})")

    browser.navigate("about:blank")
    bridge.wait(lambda: len(bridge.ops(0, op="cancel_action_goal")) == 2)
    cancels = bridge.ops(0, op="cancel_action_goal")
    assert sorted(c["id"] for c in cancels) == sorted(sent[1:])
    by_id = {g["id"]: g["action"] for g in bridge.ops(0, op="send_action_goal")}
    assert all(c["action"] == by_id[c["id"]] for c in cancels)


def test_disabling_a_robot_cancels_its_goals(page) -> None:
    browser, bridge = page
    _enable(browser, "so101")
    _set(browser, "so101", "gripper_joint", 0.5)
    bridge.wait(lambda: bridge.ops(op="send_action_goal"))
    _enable(browser, "so101")          # the checkbox again: off
    bridge.wait(lambda: bridge.ops(op="cancel_action_goal"))
    assert bridge.ops(op="cancel_action_goal")[0]["id"] == bridge.ops(op="send_action_goal")[0]["id"]


# ------------------------------------------------------------------ reconnect


def test_a_reconnect_resends_every_subscription_and_advertisement(page) -> None:
    browser, bridge = page
    _enable(browser, "ainex")
    bridge.wait(lambda: len(bridge.ops(op="advertise")) == 3)
    time.sleep(0.3)

    def wire_state(index: int) -> set[tuple]:
        return {(m["op"], m["topic"], m.get("type")) for m in bridge.ops(index)
                if m["op"] in ("subscribe", "advertise")}

    before = wire_state(0)
    assert len([s for s in before if s[0] == "subscribe"]) >= 8
    bridge.drop_all()
    bridge.wait(lambda: len(bridge.connections) == 2, timeout=10)
    bridge.wait(lambda: wire_state(1) >= before, timeout=5)
    assert wire_state(1) == before
    # And the controls still work on the new connection.
    _set(browser, "ainex", "head_tilt", 0.2)
    bridge.wait(lambda: bridge.ops(1, op="publish"))


# ------------------------------------------------------------------ parity (no browser)


def _contract() -> dict:
    return json.loads(re.search(r'<script type="application/json" id="contract">(.*?)</script>',
                                PAGE.read_text(), re.S).group(1))


def test_the_pages_signatures_are_discoverys() -> None:
    from robot_console import discovery

    contract = _contract()
    assert [tuple(s) for s in contract["member_signatures"]] == list(discovery.MEMBER_SIGNATURES)
    assert contract["rig"] == {"kind": discovery.RIG_KIND, "namespace": discovery.RIG_NAMESPACE,
                               "signature": list(discovery.RIG_SIGNATURE)}
    assert set(contract["camera_types"]) == set(discovery.CAMERA_TYPES)


def test_the_pages_ainex_names_are_the_consoles() -> None:
    from robot_console import ainex_topics as at

    ainex = _contract()["ainex"]
    assert ainex["head_pan"] == [at.TOPIC_HEAD_PAN, at.TYPE_HEAD_STATE]
    assert ainex["head_tilt"] == [at.TOPIC_HEAD_TILT, at.TYPE_HEAD_STATE]
    assert ainex["head_limit_rad"] == at.HEAD_PAN_LIMIT == at.HEAD_TILT_LIMIT
    assert ainex["set_action"] == [at.TOPIC_APP_ACTION, at.TYPE_STRING]
    assert ainex["is_walking"] == [at.TOPIC_IS_WALKING, at.TYPE_BOOL]


def _ros_file_entries(path: Path, section: str) -> dict[str, str]:
    """`{name: type}` of one top-level list in a ROS file, read as text (no YAML parser)."""
    entries, name, inside = {}, None, False
    for line in path.read_text().splitlines():
        if re.match(r"^\S", line):
            inside = line.startswith(f"{section}:")
            continue
        if not inside:
            continue
        m = re.match(r"^  - name:\s*(\S+)", line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r"^    type:\s*(\S+)", line)
        if m and name:
            entries.setdefault(name, m.group(1))
    return entries


@pytest.mark.skipif(not SPECS.is_dir(), reason="no robots_specs/ beside this checkout")
def test_the_pages_names_are_in_the_official_interfaces() -> None:
    contract = _contract()
    so101 = SPECS / "so101" / "ros2.yml"
    actions = _ros_file_entries(so101, "actions")
    topics = _ros_file_entries(so101, "topics")
    assert actions[contract["so101"]["trajectory_action"][0]] == contract["so101"]["trajectory_action"][1]
    assert actions[contract["so101"]["gripper_action"][0]] == contract["so101"]["gripper_action"][1]
    for key in ("joint_states", "robot_description"):
        name, kind = contract["so101"][key]
        assert topics[name] == kind
    joints = re.search(r"^joints:\s*\[(.*?)\]", so101.read_text(), re.M).group(1)
    names = [j.strip() for j in joints.split(",")]
    assert contract["so101"]["arm_joints"] + [contract["so101"]["gripper_joint"]] == names

    ainex_topics = _ros_file_entries(SPECS / "ainex" / "ros.yml", "topics")
    for key in ("head_pan", "head_tilt", "set_action", "is_walking"):
        name, kind = contract["ainex"][key]
        assert ainex_topics[name] == kind
    signatures = {"so101": topics, "ainex": ainex_topics,
                  "myagv": _ros_file_entries(SPECS / "myagv" / "ros.yml", "topics")}
    for kind, topic, topic_type in contract["member_signatures"]:
        assert signatures[kind][topic] == topic_type


@pytest.mark.skipif(not SPECS.is_dir(), reason="no robots_specs/ beside this checkout")
def test_the_fallback_limits_are_the_urdfs() -> None:
    """Used only until the robot's own `robot_description` arrives; the URDF's arm limits,
    and the jaw's shifted by the bringup's 0.174533 rad offset (0 = closed)."""
    urdf = (SPECS / "so101" / "so101_new_calib.urdf").read_text()
    limits = _contract()["so101"]["fallback_limits"]
    for joint, (lo, hi) in limits.items():
        m = re.search(rf'<joint name="{joint.removesuffix("_joint")}" type="revolute">.*?'
                      r'lower="([-\d.]+)" upper="([-\d.]+)"', urdf, re.S)
        urdf_lo, urdf_hi = float(m.group(1)), float(m.group(2))
        if joint == "gripper_joint":
            urdf_lo, urdf_hi = urdf_lo + 0.174533, urdf_hi + 0.174533
        assert (lo, hi) == pytest.approx((urdf_lo, urdf_hi), abs=1e-5)


def test_view_sh_refuses_what_it_cannot_serve(tmp_path) -> None:
    import subprocess

    script = CONSOLE / "bin" / "view.sh"
    bad = subprocess.run([str(script), "--url", "http://x:1", "--no-open"],
                         capture_output=True, text=True, timeout=10)
    assert bad.returncode == 2 and "ws://" in bad.stderr
    unknown = subprocess.run([str(script), "--host", "x"], capture_output=True, text=True,
                             timeout=10)
    assert unknown.returncode == 2
    helped = subprocess.run([str(script), "--help"], capture_output=True, text=True, timeout=10)
    assert helped.returncode == 0 and "--url" in helped.stdout


def test_view_sh_serves_the_page_with_the_url() -> None:
    import subprocess
    import urllib.request

    proc = subprocess.Popen([str(CONSOLE / "bin" / "view.sh"), "--url", "ws://127.0.0.1:9",
                             "--no-open"], stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        address = re.search(r"(http://\S+)", line).group(1)
        assert "url=ws%3A%2F%2F127.0.0.1%3A9" in address
        body = urllib.request.urlopen(address, timeout=5).read()
        assert body == PAGE.read_bytes()
        with pytest.raises(Exception):
            urllib.request.urlopen(address.split("?")[0] + "pyproject.toml", timeout=5)
    finally:
        proc.terminate()
        proc.wait(10)
