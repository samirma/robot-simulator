"""The view.sh page in headless Chromium against the fake rosbridge (console spec §2.2, §2.3, §4).

The page speaks rosbridge itself; the fake server records every op it sends, so the tests
check what reached the wire, not only what the page displays. Contract: controls are live as
soon as the target validates; changed values are sent at once (topic publishes and action
goals streamed while dragging, throttled with the last value always delivered; services and
action-group buttons once per click); limits are never exceeded; nothing is ever stopped or
cancelled by the page. The 3D model is posed from the joint positions the robot reports (where
the profile documents them) and its commanded joints edit their controls through the same path.
"""

import math
import threading
import time

import numpy as np
import pytest

import robot_console.view as view_mod
from robot_console.discovery import SelectionError, fetch_graph, select_target
from robot_console.profiles import SUPPORTED_IDS, load
from robot_console.rosbridge import Rosbridge
from robot_console.view import make_server
from wirespec import drop, merge, retype, wire_spec

pytestmark = pytest.mark.browser
playwright = pytest.importorskip("playwright.sync_api")

SO_ARM = "/joint_trajectory_controller/follow_joint_trajectory"
SO_GRIP = "/gripper_controller/gripper_cmd"
HEAD_PAN = "/head_pan_controller/command"
HEAD_TILT = "/head_tilt_controller/command"
GOAL_CONNS_MAX = 3          # console.js: open goal connections per control


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def view(fake, browser):
    https, pages = [], []

    def open_page(spec=None, srv=None, robot=None, namespace=None):
        srv = srv or fake(spec)
        http = make_server(srv.url, robot, namespace, 0)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        https.append(http)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        pages.append(page)
        page.goto(f"http://127.0.0.1:{http.server_address[1]}/")
        page.wait_for_function("window.RC && window.RC.ready === true", timeout=15000)
        return srv, page
    yield open_page
    for p in pages:
        if not p.is_closed():
            p.close()
    for h in https:
        h.shutdown()


def so101_spec(exec_s=20.0, namespace=None):
    spec = wire_spec(load("so101"), namespace=namespace) if namespace else wire_spec(load("so101"))
    for a in spec["actions"]:
        a["exec_s"] = exec_s                 # goals outlive the test unless preempted
    return spec


def control(page, cid):
    return page.locator(f'[data-control="{cid}"]')


def disabled(page, cid):
    """The control's inputs and buttons (one fieldset) are disabled."""
    return control(page, cid).locator("fieldset").evaluate("e => e.disabled")


def status(page, cid):
    return control(page, cid).locator('[data-testid="status"]').inner_text()


def num(page, cid, field):
    return control(page, cid).locator(f'input[data-field="{field}"]')


def rng(page, cid, field):
    return num(page, cid, field).locator(
        "xpath=ancestor::div[contains(concat(' ', @class, ' '), ' field ')][1]").locator('input[type="range"]')


def drag(page, cid, field, start=0.5, end=0.9, steps=40, step_ms=40, during=None):
    """Drag a slider's thumb with the mouse from fraction `start` to `end` of its track."""
    r = rng(page, cid, field)
    r.scroll_into_view_if_needed()
    box = r.bounding_box()
    y = box["y"] + box["height"] / 2
    x = lambda f: box["x"] + 10 + f * (box["width"] - 20)          # noqa: E731
    page.mouse.move(x(start), y)
    page.mouse.down()
    for i in range(1, steps + 1):
        page.mouse.move(x(start + (end - start) * i / steps), y)
        page.wait_for_timeout(step_ms)
        if during:
            during()
    page.mouse.up()
    return float(num(page, cid, field).input_value())


def goals(srv, action):
    return [o for o in srv.ops_of("send_action_goal") if o["action"] == action]


def no_stop_ops(srv):
    """Nothing the page sent stops or cancels anything."""
    assert srv.ops_of("cancel_action_goal") == []
    assert [c for c in srv.calls() if "cancel_goal" in c["service"]] == []
    assert [p for p in srv.published() if p["topic"].endswith("/cancel")] == []


def gaps(ops):
    return [b["_t"] - a["_t"] for a, b in zip(ops, ops[1:])]


# ------------------------------------------------------------------ cameras and catalogue

def test_cameras_live_and_only_profile_controls(view):
    srv, page = view(wire_spec(load("ainex")))
    assert page.locator('[data-testid="target"]').inner_text() == "ainex"
    cam = page.locator('[data-testid="camera"]')
    assert cam.count() == 1 and cam.get_attribute("data-topic") == "/camera/image_raw"
    page.wait_for_function("document.querySelector('[data-testid=camera]').dataset.state === 'live'")
    ids = page.eval_on_selector_all('[data-testid="control"]', "els => els.map(e => e.dataset.control)")
    assert ids == [c.id for c in load("ainex").controls]
    text = page.locator('[data-testid="controls"]').inner_text()
    assert "/walking/set_param" not in text and "/walking/command" not in text and "/cmd_vel" not in text
    # Controls take only bounded numbers and listed choices: no free topic/message entry.
    assert page.locator('#controls input:not([type="number"]):not([type="range"])').count() == 0
    assert page.locator("#controls textarea, #controls [contenteditable]").count() == 0
    for r in page.locator('#controls input[type="range"]').all():      # sliders span the documented limits
        n = r.locator("xpath=ancestor::div[contains(concat(' ', @class, ' '), ' field ')][1]").locator('input[type="number"]')
        assert (r.get_attribute("min"), r.get_attribute("max")) == (n.get_attribute("min"), n.get_attribute("max"))


@pytest.mark.parametrize("pid", [i for i in SUPPORTED_IDS if load(i).cameras])
def test_every_profile_camera_is_shown_live_then_stale(view, pid):
    """Console spec §4, for each profile: every supported stream is displayed under its resolved
    topic name, live while frames arrive and visibly stale (never live) once they stop."""
    p = load(pid)
    srv, page = view(wire_spec(p))
    cams = page.locator('[data-testid="camera"]')
    assert sorted(cams.evaluate_all("els => els.map(e => e.dataset.topic)")) == sorted(c.topic for c in p.cameras)
    page.wait_for_function("[...document.querySelectorAll('[data-testid=camera]')].every(e => e.dataset.state === 'live')",
                           timeout=8000)
    srv.paused.add(p.cameras[0].topic)
    page.wait_for_function(f"document.querySelector('[data-testid=camera][data-topic=\"{p.cameras[0].topic}\"]')"
                           ".dataset.state === 'stale'", timeout=8000)
    assert "STALE" in cams.first.locator('[data-testid="camera-state"]').inner_text()
    assert srv.command_ops() == []


def cam_state(page):
    return page.locator('[data-testid="camera"]').get_attribute("data-state")


def test_unsupported_and_malformed_frames_are_visible_states(view):
    """An encoding the profile does not document shows UNSUPPORTED; a frame whose row step is
    shorter than its width (camera.py refuses it too) shows FAILED and is never drawn as live."""
    srv, page = view(wire_spec(load("ainex")))
    live = "document.querySelector('[data-testid=camera]').dataset.state === '%s'"
    page.wait_for_function(live % "live")
    srv.frame_encoding["/camera/image_raw"] = "bgr8"                     # the profile documents rgb8
    page.wait_for_function(live % "unsupported", timeout=5000)
    assert "UNSUPPORTED" in page.locator('[data-testid="camera-state"]').inner_text()
    del srv.frame_encoding["/camera/image_raw"]
    page.wait_for_function(live % "live", timeout=5000)
    send = srv._broadcast

    def short_rows(topic, msg):
        if topic == "/camera/image_raw" and "step" in msg:
            msg = dict(msg, step=msg["width"] * 3 - 3)
        send(topic, msg)
    srv._broadcast = short_rows
    page.wait_for_function(live % "failed", timeout=5000)
    assert "row step" in page.locator('[data-testid="camera"] .ov-detail').inner_text()
    page.wait_for_timeout(400)
    assert cam_state(page) == "failed"
    assert srv.command_ops() == []


def test_missing_profile_camera_is_a_visible_state(view, monkeypatch):
    """A profile camera absent from a validated target's wire shows MISSING. No packaged profile
    has an optional camera, so the page is handed an AiNex profile whose camera row is optional."""
    packaged = view_mod.all_profiles_json

    def optional_camera():
        j = packaged()
        p = next(x for x in j["profiles"] if x["id"] == "ainex")
        for row in p["endpoints"] + p["cameras"]:
            if row.get("name", row.get("topic")) == "/camera/image_raw":
                row["optional"] = True
        return j
    monkeypatch.setattr(view_mod, "all_profiles_json", optional_camera)
    srv, page = view(drop(wire_spec(load("ainex")), "/camera/image_raw"))
    assert page.locator('[data-testid="target"]').inner_text() == "ainex"
    page.wait_for_function("document.querySelector('[data-testid=camera]').dataset.state === 'missing'")
    assert "MISSING" in page.locator('[data-testid="camera-state"]').inner_text()


def test_no_cameras_is_a_visible_state(view):
    srv, page = view(wire_spec(load("mycobot280")))
    assert page.locator('[data-testid="target"]').inner_text() == "mycobot280"
    page.wait_for_function("getComputedStyle(document.querySelector('[data-testid=no-cameras]')).display !== 'none'")
    assert "No cameras" in page.locator('[data-testid="no-cameras"]').inner_text()
    assert page.locator('[data-testid="control"]').count() == len(load("mycobot280").controls)


def test_mobile_base_page_has_camera_and_no_controls(view):
    srv, page = view(wire_spec(load("myagv")))
    assert page.locator('[data-testid="target"]').inner_text() == "myagv"
    page.wait_for_function("document.querySelector('[data-testid=camera]').dataset.state === 'live'")
    assert page.locator('[data-testid="camera"]').get_attribute("data-topic") == "/usb_cam/image_raw"
    assert page.locator('[data-testid="no-controls"]').count() == 1
    assert "Camera only" in page.locator('[data-testid="live-card"]').inner_text()
    assert srv.command_ops() == []


# ------------------------------------------------------------------ validation gates the controls; nothing on load

def test_nothing_is_sent_before_validation(view):
    srv, page = view(merge(wire_spec(load("myagv")), wire_spec(load("rosmaster_x3_plus"))))
    reason = page.locator('[data-testid="reason"]').inner_text()
    assert "myagv" in reason and "rosmaster_x3_plus" in reason
    assert page.locator('[data-testid="control"]').count() == 0
    assert page.locator('[data-testid="untied"]').count() == 2      # both robots' streams, untied
    assert page.locator('[data-testid="live-card"]').get_attribute("data-state") == "unavailable"
    n = len(srv.clients)
    page.select_option('[data-testid="robot"]', "rosmaster_x3_plus")
    page.click('[data-testid="select"]')
    page.wait_for_function("document.querySelector('[data-testid=reason]').innerText.includes('ambiguous')")
    # Explicit selection on an ambiguous wire is still refused: the typed evidence is shared.
    assert len(srv.clients) == n + 1 and page.locator('[data-testid="target"]').inner_text() == "none"
    assert page.locator('[data-testid="control"]').count() == 0
    assert page.evaluate("RC.state.model") is None                     # no target, no model
    assert srv.command_ops() == []


@pytest.mark.parametrize("robot,name,bad", [("ainex", HEAD_PAN, "std_msgs/Float64"),
                                            ("so101", "/joint_states", "sensor_msgs/msg/Imu")])
def test_explicit_selection_cannot_bypass_type_checks(view, robot, name, bad):
    srv, page = view(retype(wire_spec(load(robot)), name, bad), robot=robot)
    assert page.locator('[data-testid="target"]').inner_text() == "none"
    reason = page.locator('[data-testid="reason"]').inner_text()
    assert "wrong type" in reason and name in reason and bad in reason, reason
    assert page.locator('[data-testid="control"]').count() == 0
    assert page.evaluate("RC.state.model") is None
    assert srv.command_ops() == []


@pytest.mark.parametrize("robot", ["so101", "ainex", "mycobot280", "rosmaster_x3_plus"])
def test_nothing_is_sent_on_load_or_reload(view, robot):
    srv, page = view(wire_spec(load(robot), include_optional=True))
    assert page.locator('[data-testid="live-card"]').get_attribute("data-state") == "live"
    assert not disabled(page, load(robot).controls[0].id)               # live at once
    page.wait_for_timeout(800)
    page.reload()
    page.wait_for_function("window.RC && window.RC.ready === true")
    page.wait_for_timeout(800)
    assert srv.command_ops() == []


@pytest.mark.parametrize("robot", ["so101", "ainex"])
def test_page_states_the_sending_contract_and_has_no_arm_or_stop(view, robot):
    srv, page = view(so101_spec() if robot == "so101" else wire_spec(load(robot)))
    note = page.locator('[data-testid="send-note"]').inner_text()
    assert "Changes are sent to the robot immediately, and nothing is stopped automatically." in note
    assert "keeps running" in note
    assert f"Controls live on {robot}" in page.locator('[data-testid="live-card"]').inner_text()
    for sel in ("#enable", '[data-testid="enable"]', '[data-testid="stop-all"]', '[data-testid="armed-pill"]',
                '[data-testid="hold"]', '[data-testid="stop-status"]', '[data-testid="startup"]'):
        assert page.locator(sel).count() == 0, sel
    assert "STOP" not in page.locator("body").inner_text()
    page.keyboard.press("Escape")                                        # Esc does nothing
    page.wait_for_timeout(300)
    assert srv.command_ops() == []


# ------------------------------------------------------------------ topic publishes stream while dragging

def test_ros2_publish_drag_streams_throttled_and_delivers_final_value(view):
    srv, page = view(wire_spec(load("mycobot280")))
    t0 = time.monotonic()
    final = drag(page, "arm_gripper_target", "j1", 0.5, 0.95, steps=40, step_ms=40)
    dur = time.monotonic() - t0
    page.wait_for_timeout(400)
    pubs = srv.published("/joint_states")
    assert 5 <= len(pubs) <= dur / 0.1 + 3, len(pubs)                  # streamed, about 10 per s
    assert min(gaps(pubs)) > 0.07, gaps(pubs)                           # throttled
    assert pubs[-1]["msg"]["position"][0] == final > 2.0                # the final value is delivered
    assert len(pubs[-1]["msg"]["position"]) == 7
    assert "sent" in status(page, "arm_gripper_target")
    no_stop_ops(srv)


def test_ros1_head_slider_and_pad_stream(view):
    srv, page = view(wire_spec(load("ainex")))
    final = drag(page, "head_pan", "position", 0.5, 0.1, steps=30, step_ms=40)
    page.wait_for_timeout(300)
    pubs = srv.published(HEAD_PAN)
    assert len(pubs) >= 5 and min(gaps(pubs)) > 0.07
    assert pubs[-1]["msg"] == {"position": final, "duration": 0.5}
    # The head pad streams both axes while dragged. Its left is the robot's left: a negative
    # head_pan on the model's -Z axis (console spec §2.1, amended 2026-10-02), as teleop's Left key.
    pad = page.locator(".pad")
    pad.scroll_into_view_if_needed()
    labels = pad.locator(".pad-lbl").all_inner_texts()
    assert "← left −1.57" in labels and "right 1.57 →" in labels, labels
    box = pad.bounding_box()
    n_pan, n_tilt = len(srv.published(HEAD_PAN)), len(srv.published(HEAD_TILT))
    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    page.mouse.down()
    for i in range(1, 26):
        page.mouse.move(box["x"] + box["width"] * (0.5 - 0.016 * i), box["y"] + box["height"] * (0.5 - 0.016 * i))
        page.wait_for_timeout(40)
    page.mouse.up()
    page.wait_for_timeout(300)
    pan = float(num(page, "head_pan", "position").input_value())
    tilt = float(num(page, "head_tilt", "position").input_value())
    assert len(srv.published(HEAD_PAN)) - n_pan >= 4 and len(srv.published(HEAD_TILT)) - n_tilt >= 4
    assert srv.published(HEAD_PAN)[-1]["msg"]["position"] == pan < -0.5           # up-left: pan negative
    assert srv.published(HEAD_TILT)[-1]["msg"]["position"] == tilt > 0.1
    # ArrowLeft on the pad turns the head further to the robot's left (a lower pan).
    pad.focus()
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(300)
    assert srv.published(HEAD_PAN)[-1]["msg"]["position"] == pytest.approx(pan - 0.05)
    assert "- = left" in control(page, "head_pan").locator("label").first.inner_text()
    no_stop_ops(srv)


def test_typed_number_is_sent_on_commit_only(view):
    srv, page = view(wire_spec(load("ainex")))
    f = num(page, "head_pan", "position")
    f.fill("0.3")                                                        # typing alone sends nothing
    page.wait_for_timeout(300)
    assert srv.published(HEAD_PAN) == []
    f.press("Enter")
    page.wait_for_timeout(300)
    assert [p["msg"] for p in srv.published(HEAD_PAN)] == [{"position": 0.3, "duration": 0.5}]


# ------------------------------------------------------------------ limits

def test_out_of_range_values_are_never_sent(view):
    srv, page = view(wire_spec(load("ainex")))
    num(page, "head_pan", "position").fill("3.0")
    num(page, "head_pan", "position").press("Enter")
    page.wait_for_timeout(300)
    assert srv.published(HEAD_PAN) == []
    assert "above documented maximum" in status(page, "head_pan")
    srv2, page2 = view(so101_spec())
    num(page2, "arm_trajectory", "shoulder_pan").fill("2.5")
    num(page2, "arm_trajectory", "shoulder_pan").press("Tab")
    page2.wait_for_timeout(400)
    assert srv2.ops_of("send_action_goal") == []
    assert "above documented maximum" in status(page2, "arm_trajectory")


def test_slider_cannot_leave_documented_limits(view):
    srv, page = view(so101_spec())
    f = num(page, "arm_trajectory", "shoulder_pan")
    r = rng(page, "arm_trajectory", "shoulder_pan")
    r.evaluate("r => { r.value = '99'; r.dispatchEvent(new Event('input', {bubbles: true})); }")
    assert float(f.input_value()) == 1.91986
    r.evaluate("r => { r.value = '-99'; r.dispatchEvent(new Event('input', {bubbles: true})); }")
    assert float(f.input_value()) == -1.91986
    final = drag(page, "arm_trajectory", "shoulder_pan", 0.5, 1.3, steps=20, step_ms=30)   # past the end
    assert final == 1.91986
    page.wait_for_timeout(1200)
    sent = [g["args"]["trajectory"]["points"][0]["positions"][0] for g in goals(srv, SO_ARM)]
    assert sent and all(-1.91986 <= v <= 1.91986 for v in sent) and sent[-1] == 1.91986


# ------------------------------------------------------------------ action goals stream and preempt

def test_action_drag_streams_preempting_goals_with_bounded_connections(view):
    srv, page = view(so101_spec())
    base = srv.open_clients()
    seen = []
    t0 = time.monotonic()
    final = drag(page, "arm_trajectory", "shoulder_pan", 0.5, 0.1, steps=50, step_ms=40,
                 during=lambda: seen.append(srv.open_clients()))
    dur = time.monotonic() - t0                                          # longer than 2 s on a loaded host
    page.wait_for_timeout(800)
    gs = goals(srv, SO_ARM)
    assert 5 <= len(gs) <= dur / 0.2 + 3, (len(gs), dur)                 # about 5 per s
    assert min(gaps(gs)) > 0.1, gaps(gs)                         # throttled (each goal has its own connection: arrival jitter)
    g = gs[-1]
    assert g["args"]["trajectory"]["points"][0]["positions"][0] == final < -1.0
    assert g["args"]["trajectory"]["joint_names"][0] == "shoulder_pan_joint" and g["feedback"] is True
    # Each goal preempted the previous one; only the latest still runs.
    assert [srv.goals[x["id"]]["done"] for x in gs] == [True] * (len(gs) - 1) + [False]
    assert max(seen) <= base + GOAL_CONNS_MAX, seen
    assert srv.open_clients() == base + 1                                # the latest goal's connection
    assert "goal running" in status(page, "arm_trajectory")
    no_stop_ops(srv)


def test_gripper_slider_streams_goals(view):
    srv, page = view(so101_spec())
    final = drag(page, "gripper", "position", 0.0, 0.6, steps=25, step_ms=40)
    page.wait_for_timeout(800)
    gs = goals(srv, SO_GRIP)
    assert len(gs) >= 3 and gs[-1]["args"]["command"]["position"] == [final] and final > 0.8
    assert gs[-1]["args"]["command"]["name"] == ["gripper_joint"]


def test_silent_action_server_bounds_goals_and_connections(view):
    """A server that never acknowledges: about one goal per second, and never more than
    GOAL_CONNS_MAX goal connections at any moment (sampled every 5 ms through a 5 s drag)."""
    srv, page = view(so101_spec())
    srv.action_behaviour[SO_ARM] = "silent"                              # never acknowledges a goal
    base = srv.open_clients()
    seen, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            seen.append(srv.open_clients())
            time.sleep(0.005)
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        final = drag(page, "arm_trajectory", "shoulder_pan", 0.5, 0.9, steps=130, step_ms=40)
        page.wait_for_timeout(1500)                                      # the trailing goal after the ack timeout
    finally:
        stop.set()
        th.join()
    gs = goals(srv, SO_ARM)
    assert 4 <= len(gs) <= 10, len(gs)                                   # about one per second
    assert gs[-1]["args"]["trajectory"]["points"][0]["positions"][0] == final
    assert max(seen) <= base + GOAL_CONNS_MAX, (base, max(seen))


def test_use_measured_and_reset_send_their_values(view):
    srv, page = view(so101_spec())
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_flex_joint", "wrist_flex_joint", "wrist_roll_joint", "gripper_joint"]
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            srv._broadcast("/joint_states", {"name": names, "position": [0.25, -0.5, 0.75, 0.0, 0.1, 0.4]})
            time.sleep(0.05)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        page.wait_for_function("document.querySelector('[data-control=arm_trajectory] .measured').dataset.state === 'live'", timeout=5000)
        assert "+0.250" in control(page, "arm_trajectory").locator(".measured").first.inner_text()
        assert srv.command_ops() == []                                   # feedback is read-only
        control(page, "arm_trajectory").get_by_role("button", name="Use measured").click()
        assert float(num(page, "arm_trajectory", "elbow_flex").input_value()) == 0.75
        assert float(num(page, "gripper", "position").input_value()) == 0       # per control
        page.wait_for_timeout(400)
        assert goals(srv, SO_ARM)[-1]["args"]["trajectory"]["points"][0]["positions"] == [0.25, -0.5, 0.75, 0.0, 0.1]
        control(page, "gripper").get_by_role("button", name="Use measured").click()
        assert float(num(page, "gripper", "position").input_value()) == 0.4
        control(page, "arm_trajectory").get_by_role("button", name="Reset to defaults").click()
        page.wait_for_timeout(400)
        assert goals(srv, SO_ARM)[-1]["args"]["trajectory"]["points"][0]["positions"] == [0.0] * 5
    finally:
        stop.set()
    page.wait_for_function("document.querySelector('[data-control=arm_trajectory] .measured').dataset.state === 'none'", timeout=5000)
    assert srv.published() == []


# ------------------------------------------------------------------ measured reads: once per click, never on load

def read_value(page, cid, field):
    return num(page, cid, field).locator(
        "xpath=ancestor::div[contains(concat(' ', @class, ' '), ' field ')][1]").locator('[data-testid="read-value"]')


def test_x3_measured_servo_angles_are_read_per_click_and_copied(view):
    """ROSMASTER: /CurrentAngle reads the six servo angles back (a measurement, unlike the
    /joint_states echo). Console spec §2.2: the page shows measured positions where the profile
    documents them and copies them into the targets. It calls the service only on a click,
    with the vendor's request; -1 (unread) is shown as no value."""
    srv, page = view(wire_spec(load("rosmaster_x3_plus")))
    srv.set_service("/CurrentAngle", lambda args: {"angles": [80.0, 120.0, 10.0, 45.0, -1, 60.0]})
    page.wait_for_timeout(600)
    assert srv.calls("/CurrentAngle") == [] and srv.command_ops() == []      # nothing on load
    control(page, "arm_pose").locator('[data-testid="read"]').click()
    page.wait_for_function("document.querySelector('[data-control=arm_pose] [data-testid=read-value]').dataset.state === 'read'")
    assert [c["args"] for c in srv.calls("/CurrentAngle")] == [{"apply": "GetArmJoints"}]
    assert "+120.0 deg" in read_value(page, "arm_pose", "j2").inner_text()
    assert read_value(page, "arm_pose", "j5").get_attribute("data-state") == "invalid"
    assert "+60.0 deg" in read_value(page, "gripper", "angle").inner_text()  # one read serves both controls
    assert "read from /CurrentAngle" in control(page, "arm_pose").locator('[data-testid="read-status"]').inner_text()
    assert srv.published("/TargetAngle") == []                               # reading sends no command
    control(page, "arm_pose").get_by_role("button", name="Use measured").click()
    page.wait_for_timeout(400)
    joints = srv.published("/TargetAngle")[-1]["msg"]["joints"]
    assert joints == [80, 120, 10, 45, 90, 60]                               # j5 unread: its target is kept
    assert len(srv.calls("/CurrentAngle")) == 1


def test_ainex_measured_head_position_is_read_per_click(view):
    srv, page = view(wire_spec(load("ainex")))
    pulses = {23: 739, 24: 400}                                              # pan ~ +1.0 rad, tilt ~ -0.42 rad
    srv.set_service("/ros_robot_controller/bus_servo/get_position",
                    lambda args: {"success": True, "position": [{"id": i, "position": pulses[i]} for i in args["id"]]})
    page.wait_for_timeout(500)
    assert srv.calls() == []
    control(page, "head_pan").locator('[data-testid="read"]').click()
    page.wait_for_function("document.querySelector('[data-control=head_tilt] [data-testid=read-value]').dataset.state === 'read'")
    assert [c["args"] for c in srv.calls()] == [{"id": [23, 24]}]
    assert "+1.001 rad" in read_value(page, "head_pan", "position").inner_text()
    assert "−0.419 rad" in read_value(page, "head_tilt", "position").inner_text()
    control(page, "head_tilt").get_by_role("button", name="Use measured").click()
    page.wait_for_timeout(400)
    assert srv.published(HEAD_TILT)[-1]["msg"]["position"] == pytest.approx(-0.42)
    assert srv.published(HEAD_PAN) == []
    srv.set_service("/ros_robot_controller/bus_servo/get_position", lambda args: {"success": False, "position": []})
    control(page, "head_pan").locator('[data-testid="read"]').click()
    page.wait_for_function("document.querySelector('[data-control=head_pan] [data-testid=read-status]').innerText.includes('failed')")
    assert srv.calls("/walking/command") == []
    no_stop_ops(srv)


def test_mycobot_all_zero_answer_is_no_reading(view):
    """listen_real_service answers the response defaults (all zeros) when it cannot read the arm."""
    srv, page = view(wire_spec(load("mycobot280"), include_optional=True))
    srv.set_service("/get_angles", lambda args: {f"joint_{i}": 0.0 for i in range(1, 7)})
    control(page, "set_angles").locator('[data-testid="read"]').click()
    page.wait_for_function("document.querySelector('[data-control=set_angles] [data-testid=read-value]').dataset.state === 'invalid'")
    assert control(page, "set_angles").get_by_role("button", name="Use measured").is_disabled()
    srv.set_service("/get_angles", lambda args: {f"joint_{i}": 10.0 * i for i in range(1, 7)})
    control(page, "arm_gripper_target").locator('[data-testid="read"]').click()
    page.wait_for_function("document.querySelector('[data-control=arm_gripper_target] [data-testid=read-value]').dataset.state === 'read'")
    assert "+0.17 rad" in read_value(page, "arm_gripper_target", "j1").inner_text()           # 10 deg
    assert "+30.0 deg" in read_value(page, "set_angles", "j3").inner_text()
    assert len(srv.calls("/get_angles")) == 2 and srv.calls("/set_angles") == []


# ------------------------------------------------------------------ sending, done and failed states

def phase(page, cid):
    return control(page, cid).get_attribute("data-phase")


def test_each_control_shows_its_sending_done_and_failed_state(view):
    srv, page = view(wire_spec(load("mycobot280"), include_optional=True))
    assert phase(page, "set_angles") is None                                  # nothing sent yet
    srv.set_service("/set_angles", "hang")
    control(page, "set_angles").locator('[data-testid="send"]').click()
    page.wait_for_timeout(200)
    assert phase(page, "set_angles") == "sending" and status(page, "set_angles").startswith("sending")
    # The documented response field `flag` is false when the arm refused (listen_real_service.py)
    srv.set_service("/set_gripper", lambda args: {"flag": False})
    control(page, "gripper_close").locator('[data-testid="send"]').click()
    page.wait_for_function("document.querySelector('[data-control=gripper_close]').dataset.phase === 'failed'")
    srv.set_service("/set_gripper", lambda args: {"flag": True})
    control(page, "gripper_open").locator('[data-testid="send"]').click()
    page.wait_for_function("document.querySelector('[data-control=gripper_open]').dataset.phase === 'done'")
    drag(page, "arm_gripper_target", "j1", 0.5, 0.6, steps=5, step_ms=40)
    page.wait_for_timeout(300)
    assert phase(page, "arm_gripper_target") == "done" and "sent" in status(page, "arm_gripper_target")
    num(page, "arm_gripper_target", "j1").fill("9")
    num(page, "arm_gripper_target", "j1").press("Enter")
    page.wait_for_timeout(200)
    assert phase(page, "arm_gripper_target") == "invalid" and "not sent" in status(page, "arm_gripper_target")


def test_aborted_goal_is_a_visible_failure(view):
    srv, page = view(so101_spec(exec_s=0.6))
    srv.action_behaviour[SO_GRIP] = "abort"
    num(page, "gripper", "position").fill("0.5")
    num(page, "gripper", "position").press("Enter")
    page.wait_for_function("document.querySelector('[data-control=gripper]').dataset.phase === 'failed'", timeout=8000)
    assert "aborted" in status(page, "gripper")
    page.wait_for_timeout(300)
    assert len(goals(srv, SO_GRIP)) == 1                                      # a failure is not retried
    no_stop_ops(srv)


# ------------------------------------------------------------------ services and action groups: once per click

def test_service_is_called_once_per_click_never_streamed(view):
    srv, page = view(wire_spec(load("mycobot280"), include_optional=True))
    drag(page, "set_angles", "j1", 0.5, 0.8, steps=15, step_ms=40)       # editing a service's values sends nothing
    page.wait_for_timeout(300)
    assert srv.calls("/set_angles") == []
    send = control(page, "set_angles").locator('[data-testid="send"]')
    for _ in range(3):
        send.click()
    page.wait_for_timeout(300)
    calls = srv.calls("/set_angles")
    assert len(calls) == 3
    assert calls[-1]["args"]["joint_1"] == float(num(page, "set_angles", "j1").input_value()) and calls[-1]["args"]["speed"] == 50
    assert "done" in status(page, "set_angles")
    control(page, "gripper_open").locator('[data-testid="send"]').click()
    page.wait_for_timeout(300)
    assert [c["args"] for c in srv.calls("/set_gripper")] == [{"status": True}]


def test_call_control_pending_and_failure_states(view):
    srv, page = view(wire_spec(load("mycobot280"), include_optional=True))
    srv.set_service("/set_angles", "hang")
    control(page, "set_angles").locator('[data-testid="send"]').click()
    page.wait_for_timeout(200)
    assert "pending" in status(page, "set_angles")
    srv.set_service("/set_gripper", "fail")
    control(page, "gripper_close").locator('[data-testid="send"]').click()
    page.wait_for_function("document.querySelector('[data-control=gripper_close] .status').innerText.includes('failed')")
    assert len(srv.calls("/set_gripper")) == 1


def test_ros1_call_control_pending_and_failure_states(view):
    srv, page = view(wire_spec(load("ainex")))
    srv.set_service("/walking/init_pose", "hang")
    control(page, "init_pose").locator('[data-testid="send"]').click()
    page.wait_for_timeout(200)
    assert "pending" in status(page, "init_pose")
    srv.set_service("/walking/init_pose", "fail")
    control(page, "init_pose").locator('[data-testid="send"]').click()
    page.wait_for_function("document.querySelector('[data-control=init_pose] .status').innerText.includes('failed')")
    assert len(srv.calls("/walking/init_pose")) == 2
    assert srv.calls("/walking/command") == []
    no_stop_ops(srv)


def test_ainex_init_pose_and_action_group_once_per_click(view):
    srv, page = view(wire_spec(load("ainex")))
    control(page, "init_pose").locator('[data-testid="send"]').click()
    page.locator('[data-control="action_group"] [data-choice="greet"]').click()
    page.wait_for_timeout(300)
    assert len(srv.calls("/walking/init_pose")) == 1
    assert [p["msg"] for p in srv.published("/app/set_action")] == [{"data": "greet"}]
    page.locator('[data-control="action_group"] [data-choice="twist"]').click()
    page.wait_for_timeout(300)
    assert [p["msg"] for p in srv.published("/app/set_action")] == [{"data": "greet"}, {"data": "twist"}]
    no_stop_ops(srv)


def test_unsupported_controls_remain_unavailable(view):
    srv, page = view(wire_spec(load("mycobot280")))       # boot only: no optional services
    for cid in ("set_angles", "gripper_open", "gripper_close"):
        assert "is not on the wire" in control(page, cid).locator('[data-testid="unavailable"]').inner_text()
        assert disabled(page, cid)
    assert control(page, "set_angles").locator('[data-testid="send"]').is_disabled()
    assert not disabled(page, "arm_gripper_target")
    assert control(page, "set_angles").locator('[data-testid="read"]').count() == 0      # /get_angles absent too


def test_incompatible_prerequisite_leaves_the_control_unavailable(view):
    """myCobot 280: the slider-control boot executes every /joint_states message, so the page's
    target needs the boot with gui:=false. While another node (the slider GUI, or a measured
    joint-state publisher) publishes /joint_states the control is unavailable and names it;
    rosbridge's own node (an earlier page's publication) is infrastructure."""
    def spec_with(pubs):
        s = wire_spec(load("mycobot280"))
        next(r for r in s["topics"] if r["name"] == "/joint_states")["publishers"] = pubs
        return s
    srv, page = view(spec_with(["/joint_state_publisher"]))
    assert page.locator('[data-testid="target"]').inner_text() == "mycobot280"
    reason = control(page, "arm_gripper_target").locator('[data-testid="unavailable"]').inner_text()
    assert "/joint_state_publisher" in reason and "re-command" in reason
    assert disabled(page, "arm_gripper_target")
    assert srv.command_ops() == []
    srv2, page2 = view(spec_with(["/rosbridge_websocket"]))
    assert not disabled(page2, "arm_gripper_target")
    assert control(page2, "arm_gripper_target").locator('[data-testid="unavailable"]').count() == 0


@pytest.mark.parametrize("pid", SUPPORTED_IDS)
def test_controls_without_their_prerequisites_are_unavailable(view, pid):
    """Console spec §4, for each profile: a control whose prerequisite is an optional row absent
    from the wire is shown unavailable with its reason; a missing required prerequisite fails the
    target's validation, so no control is offered at all. Either way nothing is sent."""
    p = load(pid)
    if not p.controls:
        srv, page = view(wire_spec(p))
        assert page.locator('[data-testid="control"]').count() == 0
        assert page.locator('[data-testid="no-controls"]').count() == 1
        return
    optional = {e.name for e in p.endpoints if e.optional}
    gated = [c for c in p.controls if set(c.prerequisites) & optional]
    if gated:
        srv, page = view(wire_spec(p))                    # the optional rows are absent
        for c in gated:
            assert "is not on the wire" in control(page, c.id).locator('[data-testid="unavailable"]').inner_text()
            assert disabled(page, c.id)
    c = p.controls[0]
    required = next(n for n in c.prerequisites if n not in optional)
    srv, page = view(drop(wire_spec(p, include_optional=True), required), robot=pid)
    assert page.locator('[data-testid="target"]').inner_text() == "none"
    assert required in page.locator('[data-testid="reason"]').inner_text()
    assert page.locator('[data-testid="control"]').count() == 0
    assert srv.command_ops() == []


# ------------------------------------------------------------------ nothing is stopped automatically

@pytest.mark.parametrize("robot", ["so101", "ainex"])
def test_release_escape_focus_hide_and_close_send_no_stop(view, browser, robot):
    srv, page = view(so101_spec() if robot == "so101" else wire_spec(load(robot)))
    cid, field = ("arm_trajectory", "shoulder_pan") if robot == "so101" else ("head_pan", "position")
    drag(page, cid, field, 0.5, 0.6, steps=5, step_ms=40)               # release ends the drag
    page.wait_for_timeout(500)
    n = len(srv.command_ops())
    assert n >= 1
    page.keyboard.press("Escape")
    page.evaluate("window.dispatchEvent(new Event('blur'))")
    page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    page.evaluate("window.dispatchEvent(new Event('pagehide'))")
    page.wait_for_timeout(300)
    page.close()                                                          # closing the tab
    time.sleep(0.5)
    assert len(srv.command_ops()) == n
    no_stop_ops(srv)
    assert srv.calls("/walking/command") == []                           # the AiNex walking stop
    if robot == "so101":
        latest = goals(srv, SO_ARM)[-1]
        assert srv.goals[latest["id"]]["done"] is False                  # the goal keeps running


def test_target_change_sends_nothing_and_rebinds_controls(view):
    p = load("so101")
    spec = merge(so101_spec(namespace="arm1"), so101_spec(namespace="arm2"))
    srv, page = view(spec, robot="so101", namespace="arm1")
    assert page.locator('[data-testid="target"]').inner_text() == "so101 (namespace /arm1)"
    drag(page, "arm_trajectory", "shoulder_pan", 0.5, 0.7, steps=8, step_ms=40)
    page.wait_for_timeout(500)
    arm1 = goals(srv, f"/arm1{SO_ARM}")
    assert arm1 and arm1[-1]["args"]["trajectory"]["joint_names"][0] == "arm1/shoulder_pan_joint"
    before = len(srv.clients)
    page.fill('[data-testid="namespace"]', "arm2")
    page.click('[data-testid="select"]')
    page.wait_for_function("document.querySelector('[data-testid=target]').innerText === 'so101 (namespace /arm2)'",
                           timeout=8000)
    assert len(srv.clients) == before + 1                               # a new connection for the new target
    page.wait_for_timeout(500)
    no_stop_ops(srv)
    assert srv.goals[arm1[-1]["id"]]["done"] is False                   # the old target's goal keeps running
    assert len(goals(srv, f"/arm1{SO_ARM}")) == len(arm1)
    drag(page, "arm_trajectory", "shoulder_pan", 0.5, 0.3, steps=5, step_ms=40)
    page.wait_for_timeout(500)
    arm2 = goals(srv, f"/arm2{SO_ARM}")
    assert arm2 and arm2[-1]["args"]["trajectory"]["joint_names"][0] == "arm2/shoulder_pan_joint"
    assert len(goals(srv, f"/arm1{SO_ARM}")) == len(arm1)
    assert p.id == "so101"


def test_ros1_target_change_revalidates_and_sends_only_to_the_new_target(view):
    """Every pair of ROS 1 profiles shares required names and none is namespace-capable, so two
    validated ROS 1 targets cannot share one wire: the change goes through a refused selection
    and back. Each selection validates afresh on a new connection and sends nothing."""
    srv, page = view(wire_spec(load("ainex")))
    num(page, "head_pan", "position").fill("0.2")
    num(page, "head_pan", "position").press("Enter")
    page.wait_for_timeout(300)
    assert [p["msg"]["position"] for p in srv.published(HEAD_PAN)] == [0.2]
    n_clients, n_ops = len(srv.clients), len(srv.command_ops())
    page.select_option('[data-testid="robot"]', "myagv")
    page.click('[data-testid="select"]')
    page.wait_for_function("document.querySelector('[data-testid=reason]').innerText.includes('myagv')", timeout=8000)
    assert page.locator('[data-testid="target"]').inner_text() == "none"
    assert page.locator('[data-testid="control"]').count() == 0
    assert len(srv.clients) == n_clients + 1
    page.select_option('[data-testid="robot"]', "ainex")
    page.click('[data-testid="select"]')
    page.wait_for_function("document.querySelector('[data-testid=target]').innerText === 'ainex'", timeout=8000)
    assert len(srv.clients) == n_clients + 2
    page.wait_for_timeout(500)
    assert len(srv.command_ops()) == n_ops                              # the changes sent nothing
    no_stop_ops(srv)
    assert srv.calls("/walking/command") == []
    num(page, "head_pan", "position").fill("0.3")
    num(page, "head_pan", "position").press("Enter")
    page.wait_for_timeout(400)
    assert [p["msg"]["position"] for p in srv.published(HEAD_PAN)] == [0.2, 0.3]     # once, from the new target


@pytest.mark.parametrize("robot", ["so101", "ainex"])
def test_connection_loss_disables_controls_sends_no_stop_and_never_reconnects(view, robot):
    srv, page = view(so101_spec() if robot == "so101" else wire_spec(load(robot)))
    cid, field = ("arm_trajectory", "shoulder_pan") if robot == "so101" else ("head_pan", "position")
    drag(page, cid, field, 0.5, 0.6, steps=5, step_ms=40)
    page.wait_for_timeout(500)
    n_ops = len(srv.command_ops())
    assert n_ops >= 1
    n = len(srv.clients)
    srv.drop_all()
    page.wait_for_function("RC.state.lost === true", timeout=8000)
    assert "reload" in page.locator('[data-testid="conn"]').inner_text().lower()
    banner = page.locator('[data-testid="banner"]').inner_text()
    assert "lost" in banner and "Controls disabled" in banner and "nothing was stopped" in banner
    assert disabled(page, cid)
    assert page.locator('[data-testid="live-card"]').get_attribute("data-state") == "fault"
    assert page.eval_on_selector_all("#controls fieldset", "els => els.every(e => e.disabled)")
    if robot == "ainex":
        assert "disabled" in (page.locator(".pad").get_attribute("class") or "")
    page.keyboard.press("Escape")
    time.sleep(1.5)
    assert len(srv.clients) == n, "the page connects again only when reloaded"
    assert len(srv.command_ops()) == n_ops                              # nothing sent on or after the loss
    no_stop_ops(srv)


# ------------------------------------------------------------------ reported joint values; the 3D model

SO_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_flex_joint", "wrist_flex_joint",
             "wrist_roll_joint", "gripper_joint"]
X3_JOINTS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint"]


def joint_states(srv, names, positions, topic="/joint_states", period=0.05):
    """Publish a JointState from the robot every `period` s until the returned event is set."""
    stop = threading.Event()

    def run():
        while not stop.is_set():
            srv._broadcast(topic, {"name": names, "position": positions})
            time.sleep(period)
    threading.Thread(target=run, daemon=True).start()
    return stop


def readout(page, cid, field):
    return num(page, cid, field).locator(
        "xpath=ancestor::div[contains(concat(' ', @class, ' '), ' field ')][1]").locator(".measured:not(.read)")


def model(page):
    return page.evaluate("RC.state.model")


def model_point(page, joint):
    """Page coordinates of a commanded joint drawn on the model canvas."""
    cv = page.locator('[data-testid="model-canvas"]')
    cv.scroll_into_view_if_needed()
    page.wait_for_timeout(100)
    box = cv.bounding_box()
    x, y = page.evaluate("j => RC.state.model.points[j]", joint)
    return box["x"] + x, box["y"] + y


def select_in_model(page, joint):
    """Click the joint on the canvas (again, where joints are drawn on top of each other)."""
    x, y = model_point(page, joint)
    for _ in range(len(model(page)["clickable"]) + 1):
        page.mouse.click(x, y)
        if model(page)["selected"] == joint:
            return x, y
    raise AssertionError(f"{joint} cannot be selected in the model: {model(page)['selected']}")


def test_reported_arm_angles_are_shown_in_the_controls_units(view):
    """ROSMASTER: /joint_states is the driver's echo of its last commanded servo angles, mapped to
    rad as (deg - 90) * pi/180 (gripper 30..180 deg first mapped to 0..90); the page shows it back
    in the controls' degrees, labelled as reported, not measured. Displaying sends nothing."""
    srv, page = view(wire_spec(load("rosmaster_x3_plus")))
    stop = joint_states(srv, X3_JOINTS, [0.0, 0.5, 0.0, 0.0, 0.0, -1.5708], period=0.1)
    try:
        page.wait_for_function("document.querySelector('[data-control=arm_pose] .measured').dataset.state === 'live'", timeout=5000)
        assert readout(page, "arm_pose", "j1").inner_text().replace("\n", " ").split()[-2:] == ["+90.0", "deg"]
        assert "+118.6 deg" in readout(page, "arm_pose", "j2").inner_text()
        assert "+30.0 deg" in readout(page, "arm_pose", "gripper").inner_text()       # 30 = open
        page.wait_for_function("document.querySelector('[data-control=gripper] .measured').dataset.state === 'live'", timeout=5000)
        assert "+30.0 deg" in readout(page, "gripper", "angle").inner_text()
        assert "reported" in readout(page, "arm_pose", "j1").inner_text()
        # the echo is not a measurement: it cannot be copied into the targets (only a /CurrentAngle read can)
        assert control(page, "arm_pose").get_by_role("button", name="Use measured").is_disabled()
        m = model(page)
        assert m["sources"]["arm_joint2"] == "reported" and m["joints"]["arm_joint2"] == 0.5
    finally:
        stop.set()
    assert srv.command_ops() == []


def test_model_follows_reported_joint_positions(view):
    """SO-101: each joint of the model is posed from /joint_states; when a joint moves, the same
    joint of the model turns by the same angle. Without fresh reports the page says so."""
    srv, page = view(so101_spec())
    m = model(page)
    assert m["rendered"] and set(m["clickable"]) == set(SO_JOINTS)
    assert {m["sources"][j] for j in SO_JOINTS} == {"target"}
    assert "not reported" in page.locator('[data-testid="model-pose"]').inner_text()
    pose = [0.5, -0.25, 0.75, 0.1, -0.2, 0.3]
    stop = joint_states(srv, SO_JOINTS, pose)
    try:
        page.wait_for_function("RC.state.model.sources.shoulder_pan_joint === 'reported'", timeout=5000)
        m = model(page)
        assert [m["joints"][j] for j in SO_JOINTS] == pose
        assert {m["sources"][j] for j in SO_JOINTS} == {"reported"}
        assert "reported on /joint_states" in page.locator('[data-testid="model-pose"]').inner_text()
        before = m["world"]
    finally:
        stop.set()
    stop = joint_states(srv, SO_JOINTS, [0.5 - 0.4] + pose[1:])
    try:
        page.wait_for_function("Math.abs(RC.state.model.joints.shoulder_pan_joint - 0.1) < 1e-9", timeout=5000)
        after = model(page)["world"]
    finally:
        stop.set()
    # The links after shoulder_pan turned about its axis (vertical, to the URDF's 5-decimal pi in
    # its origin rpy) by the same 0.4 rad.
    pan = before["shoulder_pan_joint"]
    ang = lambda p: math.atan2(p[1] - pan[1], p[0] - pan[0])           # noqa: E731
    for j in ("elbow_flex_joint", "wrist_flex_joint"):
        turn = (ang(after[j]) - ang(before[j]) + math.pi) % (2 * math.pi) - math.pi
        assert abs(turn) == pytest.approx(0.4, abs=1e-4), j
        assert after[j][2] == pytest.approx(before[j][2], abs=1e-5)
    page.wait_for_function("RC.state.model.sources.shoulder_pan_joint === 'target'", timeout=5000)   # stale: says so
    assert srv.command_ops() == []


def urdf_forward(model, q):
    """Joint origins in the model's root frame, computed here from the URDF rows (origin xyz/rpy
    then a rotation about the axis; mimic joints follow their joint)."""
    def rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])

    def about(a, t):
        a = np.asarray(a, float) / np.linalg.norm(a)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k
    frames, out, joints = {model["root"]: (np.eye(3), np.zeros(3))}, {}, list(model["joints"])
    while joints:
        j = next(j for j in joints if j["parent"] in frames)
        joints.remove(j)
        R, t = frames[j["parent"]]
        Rc, tc = R @ rpy(*j["rpy"]), R @ np.array(j["xyz"], float) + t
        if j["type"] != "fixed":
            m = j.get("mimic")
            angle = m["multiplier"] * q.get(m["joint"], 0.0) + m["offset"] if m else q.get(j["name"], 0.0)
            Rc = Rc @ about(j["axis"], angle)
        frames[j["child"]] = (Rc, tc)
        out[j["name"]] = tc
    return out


@pytest.mark.parametrize("robot,names,pose", [
    ("so101", SO_JOINTS, [0.5, -0.25, 0.75, 0.1, -0.2, 0.3]),
    ("rosmaster_x3_plus", X3_JOINTS, [0.3, -0.6, 0.9, -0.4, 1.2, -1.0]),
])
def test_model_pose_is_the_urdf_pose_of_the_reported_joints(view, robot, names, pose):
    srv, page = view(wire_spec(load(robot)))
    stop = joint_states(srv, names, pose)
    try:
        page.wait_for_function(f"RC.state.model.sources['{names[0]}'] === 'reported'", timeout=5000)
        world = model(page)["world"]
    finally:
        stop.set()
    want = urdf_forward(load(robot).raw["model"], dict(zip(names, pose)))
    assert set(world) == set(want)
    for j, p in want.items():
        assert world[j] == pytest.approx(list(p), abs=1e-9), j


def test_dragging_a_model_joint_changes_its_control_and_sends_it(view):
    srv, page = view(so101_spec())
    x, y = select_in_model(page, "shoulder_pan_joint")
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 16):
        page.mouse.move(x + 4 * i, y)
        page.wait_for_timeout(40)
    page.mouse.up()
    page.wait_for_timeout(800)
    v = float(num(page, "arm_trajectory", "shoulder_pan").input_value())
    assert v == pytest.approx(60 * 2 * 1.91986 / 300, abs=0.011)          # 300 px span the documented range
    assert goals(srv, SO_ARM)[-1]["args"]["trajectory"]["points"][0]["positions"][0] == v
    assert model(page)["joints"]["shoulder_pan_joint"] == pytest.approx(v)
    page.keyboard.press("ArrowRight")                                    # the selected joint, by key
    page.wait_for_timeout(600)
    v2 = float(num(page, "arm_trajectory", "shoulder_pan").input_value())
    assert v2 == pytest.approx(v + 2 * 1.91986 / 100, abs=0.011)
    assert goals(srv, SO_ARM)[-1]["args"]["trajectory"]["points"][0]["positions"][0] == v2
    for _ in range(60):                                                  # never past the documented limit
        page.keyboard.press("Shift+ArrowRight")
    page.wait_for_timeout(800)
    assert float(num(page, "arm_trajectory", "shoulder_pan").input_value()) == 1.91986
    assert goals(srv, SO_ARM)[-1]["args"]["trajectory"]["points"][0]["positions"][0] == 1.91986
    no_stop_ops(srv)


def test_ainex_head_joints_in_the_model_publish_the_head_controls(view):
    srv, page = view(wire_spec(load("ainex")))
    m = model(page)
    assert set(m["clickable"]) == {"head_pan", "head_tilt"}
    assert m["sources"]["head_pan"] == "target" and m["sources"]["r_knee"] == "zero"
    assert "documents no joint-position topic" in page.locator('[data-testid="model-pose"]').inner_text()
    x, y = select_in_model(page, "head_pan")
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 16):
        page.mouse.move(x - 4 * i, y)                                    # to the left: a negative pan
        page.wait_for_timeout(40)
    page.mouse.up()
    page.wait_for_timeout(400)
    v = float(num(page, "head_pan", "position").input_value())
    assert v < -0.5
    assert srv.published(HEAD_PAN)[-1]["msg"]["position"] == v
    assert model(page)["joints"]["head_pan"] == pytest.approx(v)
    assert srv.published(HEAD_TILT) == []
    no_stop_ops(srv)


@pytest.mark.parametrize("robot", SUPPORTED_IDS)
def test_model_renders_and_every_commanded_joint_is_clickable(view, robot):
    srv, page = view(wire_spec(load(robot), include_optional=True))
    expected = {j["name"] for j in load(robot).raw["model"]["joints"] if j.get("command") and not j.get("mimic")}
    m = model(page)
    assert m["rendered"] and set(m["clickable"]) == expected
    drawn = page.evaluate("""() => { const c = document.querySelector('[data-testid=model-canvas]');
        return c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some((v, i) => i % 4 === 3 && v); }""")
    assert drawn
    for j in sorted(expected):
        select_in_model(page, j)
    assert srv.command_ops() == []                                       # selecting sends nothing


# ------------------------------------------------------------------ same contracts as Python

CASES = {
    "so101": lambda: wire_spec(load("so101")),
    "so101-arm1": lambda: wire_spec(load("so101"), namespace="arm1"),
    "two-so101": lambda: merge(wire_spec(load("so101"), namespace="arm1"), wire_spec(load("so101"), namespace="arm2")),
    "ambiguous": lambda: merge(wire_spec(load("myagv")), wire_spec(load("rosmaster_x3_plus"))),
    "missing": lambda: drop(wire_spec(load("ainex")), "/walking/command"),
    "mycobot": lambda: wire_spec(load("mycobot280")),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_page_selection_matches_python(view, case):
    spec = CASES[case]()
    srv, page = view(spec)
    rb = Rosbridge(srv.url).connect()
    try:
        g = fetch_graph(rb)
    finally:
        rb.close()
    try:
        expected = select_target(g, None, None).label
    except SelectionError:
        expected = None
    shown = page.locator('[data-testid="target"]').inner_text()
    assert (shown if shown != "none" else None) == expected
    assert srv.command_ops() == []
