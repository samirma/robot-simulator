"""The view.sh page in headless Chromium against the fake rosbridge (console spec §2.2, §2.3, §4).

The page speaks rosbridge itself; the fake server records every op it sends, so the tests
check what reached the wire, not only what the page displays. Contract: controls are live as
soon as the target validates; changed values are sent at once (topic publishes and action
goals streamed while dragging, throttled with the last value always delivered; services and
action-group buttons once per click); limits are never exceeded; no bounded control is ever
stopped or cancelled by the page. The 3D model is posed from the joint positions the robot
reports (where the profile documents them) and its commanded joints edit their controls through
the same path. For a mobile profile the page also offers teleop.sh's hold-to-move base motion
(console spec §2.2 as amended), tested at the end of this file; elsewhere, loading such
a page sends its start-up stop and nothing else.
"""

import dataclasses
import math
import threading
import time

import numpy as np
import pytest

import robot_console.discovery as discovery_mod
import robot_console.view as view_mod
from robot_console.discovery import SelectionError, fetch_graph, select_target
from robot_console.profiles import SUPPORTED_IDS, Field, load, load_all, profile_json
from robot_console.rosbridge import Rosbridge
from robot_console.teleop_core import command_topics, stop_ops
from robot_console.view import make_server
from test_discovery import without_camera
from test_profiles import AINEX_SERVOS, GET_POSITION, SET_POSITION, angle2pulse
from wirespec import drop, merge, retype, wire_spec

pytestmark = pytest.mark.browser
playwright = pytest.importorskip("playwright.sync_api")

SO_ARM = "/joint_trajectory_controller/follow_joint_trajectory"
SO_GRIP = "/gripper_controller/gripper_cmd"
HEAD_PAN = "/head_pan_controller/command"
HEAD_TILT = "/head_tilt_controller/command"
GOAL_CONNS_MAX = 3          # console.js: open goal connections per control
# The profiles' documented read-only position queries, which the page polls once a target is
# validated (console spec §2.2, amended): not commands (FakeRosbridge.read_only).
POLLED = {r.service for p in load_all() for r in p.reads if r.poll_s}


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def view(fake, browser):
    https, contexts = [], []

    def open_page(spec=None, srv=None, robot=None, namespace=None, viewport=(1400, 1000), locale=None):
        srv = srv or fake(spec)
        srv.read_only |= POLLED
        http = make_server(srv.url, robot, namespace, 0)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        https.append(http)
        ctx = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, **({"locale": locale} if locale else {}))
        contexts.append(ctx)
        page = ctx.new_page()
        page.goto(f"http://127.0.0.1:{http.server_address[1]}/")
        page.wait_for_function("window.RC && window.RC.ready === true", timeout=15000)
        return srv, page
    yield open_page
    for c in contexts:
        c.close()
    for h in https:
        h.shutdown()


def serve_variants(monkeypatch, *variants):
    """Hand the page (and Python's discovery) these profiles in place of the packaged profiles of
    the same id, or in addition to them: test variants for behaviour no packaged profile has.
    Patch before opening the page (the server reads the profiles when it starts)."""
    packaged_json, packaged = view_mod.all_profiles_json, discovery_mod.load_all
    by_id = {v.id: v for v in variants}

    def profiles():
        ps = packaged()
        return [by_id.get(p.id, p) for p in ps] + [v for v in variants if v.id not in {p.id for p in ps}]

    def profiles_json():
        return dict(packaged_json(), profiles=[profile_json(p) for p in profiles()])
    monkeypatch.setattr(view_mod, "all_profiles_json", profiles_json)
    monkeypatch.setattr(discovery_mod, "load_all", profiles)


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


def is_stop(p, o):
    """`o` is base motion's documented stop (teleop's stop: the walking stop for the AiNex)."""
    return any(o["op"] == ("publish" if s.op == "publish" else "call_service")
               and (o.get("topic") or o.get("service")) == s.name and (o.get("msg") if s.op == "publish" else o.get("args")) == s.msg
               for s in stop_ops(p))


def stops(srv, p):
    return [o for o in srv.command_ops() if is_stop(p, o)]


def unprompted(srv, pid):
    """The command ops apart from base motion's start-up stops (teleop's advertisements and the
    documented stop, console spec §2.2): empty when the page sent nothing else."""
    p = load(pid)
    if p.is_arm:
        return srv.command_ops()
    adv = {t for t, _ in command_topics(p)}
    return [o for o in srv.command_ops() if not is_stop(p, o) and not (o["op"] == "advertise" and o["topic"] in adv)]


def started(page):
    """Wait until base motion's start-up stop has been attempted."""
    page.wait_for_function("RC.motion() && RC.motion().started === true", timeout=8000)


def only_startup(srv, page, pid, loads=1):
    """Nothing was sent but the read-only poll and, for a mobile profile, one start-up stop per
    load (and, on a reload, the stops requested as the old page was hidden and unloaded)."""
    if not load(pid).is_arm:
        started(page)
        n = len(stops(srv, load(pid)))
        assert n == 1 if loads == 1 else n >= loads, n
    assert unprompted(srv, pid) == []


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
    # Controls take only bounded numbers and listed choices: no free topic/message entry. (Numbers
    # are text fields, so they read the same in every locale; each is a bounded field.)
    assert page.locator('#controls input:not(input.num[data-field]):not([type="range"])').count() == 0
    assert page.locator("#controls textarea, #controls [contenteditable]").count() == 0
    for r in page.locator('#controls input[type="range"]').all():      # sliders span the documented limits
        n = r.locator("xpath=ancestor::div[contains(concat(' ', @class, ' '), ' field ')][1]").locator("input.num")
        assert (r.get_attribute("min"), r.get_attribute("max")) == (n.get_attribute("data-min"), n.get_attribute("data-max"))


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
    only_startup(srv, page, pid)


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
    only_startup(srv, page, "ainex")


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


def test_no_cameras_is_a_visible_state(view, monkeypatch):
    """A validated target without cameras shows that state, and its controls (console spec §2.3).
    Every packaged profile has a camera, so the page is handed an SO-101 booted without one."""
    serve_variants(monkeypatch, without_camera(load("so101")))
    srv, page = view(drop(so101_spec(), "/image_raw"))
    assert page.locator('[data-testid="target"]').inner_text() == "so101"
    page.wait_for_function("getComputedStyle(document.querySelector('[data-testid=no-cameras]')).display !== 'none'")
    assert "No cameras" in page.locator('[data-testid="no-cameras"]').inner_text()
    assert page.locator('[data-testid="camera"]').count() == 0
    assert page.locator('[data-testid="control"]').count() == len(load("so101").controls)
    assert not disabled(page, "arm_trajectory")
    assert srv.command_ops() == []


def test_mobile_base_page_has_camera_base_motion_and_no_bounded_controls(view):
    """The myAGV's profile lists no bounded control: the page shows its camera and its base motion."""
    srv, page = view(wire_spec(load("myagv")))
    assert page.locator('[data-testid="target"]').inner_text() == "myagv"
    page.wait_for_function("document.querySelector('[data-testid=camera]').dataset.state === 'live'")
    assert page.locator('[data-testid="camera"]').get_attribute("data-topic") == "/usb_cam/image_raw"
    assert page.locator('[data-testid="no-controls"]').count() == 1
    assert "No bounded controls" in page.locator('[data-testid="live-card"]').inner_text()
    assert page.locator('[data-testid="motion"]').is_visible()
    only_startup(srv, page, "myagv")


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


@pytest.mark.parametrize("robot", ["so101", "ainex", "rosmaster_x3_plus"])
def test_nothing_is_sent_on_load_or_reload(view, robot):
    """Apart from the read-only poll and, for a mobile profile, the start-up stop of every load."""
    srv, page = view(wire_spec(load(robot), include_optional=True))
    assert page.locator('[data-testid="live-card"]').get_attribute("data-state") == "live"
    assert not disabled(page, load(robot).controls[0].id)               # live at once
    page.wait_for_timeout(800)
    page.reload()
    page.wait_for_function("window.RC && window.RC.ready === true")
    page.wait_for_timeout(800)
    only_startup(srv, page, robot, loads=2)


@pytest.mark.parametrize("robot", ["so101", "ainex"])
def test_page_states_the_sending_contract_and_has_no_arm_switch(view, robot):
    """No arm switch, no Esc behaviour, no stop for a bounded control; only a mobile profile's
    base motion has a STOP button (console spec §2.2 as amended)."""
    srv, page = view(so101_spec() if robot == "so101" else wire_spec(load(robot)))
    note = page.locator('[data-testid="send-note"]').inner_text()
    assert "Changes are sent to the robot immediately, and only base motion is stopped automatically." in note
    assert "keeps running" in note
    assert f"Controls live on {robot}" in page.locator('[data-testid="live-card"]').inner_text()
    for sel in ("#enable", '[data-testid="enable"]', '[data-testid="stop-all"]', '[data-testid="armed-pill"]',
                '[data-testid="stop-status"]', '[data-testid="startup"]'):
        assert page.locator(sel).count() == 0, sel
    assert page.locator('#controls button:has-text("stop"), #controls button:has-text("cancel")').count() == 0
    if robot == "so101":
        assert "STOP" not in page.locator("body").inner_text()
        assert not page.locator('[data-testid="motion"]').is_visible()
    else:
        assert page.locator('[data-testid="motion"] [data-testid="stop"]').is_visible()
    page.keyboard.press("Escape")                                        # Esc does nothing
    page.wait_for_timeout(300)
    only_startup(srv, page, robot)


# ------------------------------------------------------------------ topic publishes stream while dragging

def test_publish_drag_streams_throttled_and_delivers_final_value(view):
    """A multi-field topic publish (the ROSMASTER's six-servo arm pose): every message carries the
    whole target, about 10 per second while dragging, the last one the final value."""
    srv, page = view(wire_spec(load("rosmaster_x3_plus")))
    started(page)
    t0 = time.monotonic()
    final = drag(page, "arm_pose", "j1", 0.5, 0.95, steps=40, step_ms=40)
    dur = time.monotonic() - t0
    page.wait_for_timeout(400)
    pubs = srv.published("/TargetAngle")
    assert 5 <= len(pubs) <= dur / 0.1 + 3, len(pubs)                  # streamed, about 10 per s
    assert min(gaps(pubs)) > 0.07, gaps(pubs)                           # throttled
    assert pubs[-1]["msg"]["joints"][0] == final > 150                  # the final value is delivered
    assert pubs[-1]["msg"]["joints"][1:] == [145, 0, 45, 90, 30]        # with the other fields' targets
    assert "sent" in status(page, "arm_pose")
    assert unprompted(srv, "rosmaster_x3_plus") == [o for o in srv.command_ops() if o.get("topic") == "/TargetAngle"]
    no_stop_ops(srv)


def test_ros1_head_slider_and_pad_stream(view):
    srv, page = view(wire_spec(load("ainex")))
    final = drag(page, "head_pan", "position", 0.5, 0.1, steps=30, step_ms=40)
    page.wait_for_timeout(300)
    pubs = srv.published(HEAD_PAN)
    assert len(pubs) >= 5 and min(gaps(pubs)) > 0.07
    assert pubs[-1]["msg"] == {"position": final, "duration": 0.5}
    # The head pad streams both axes while dragged. Its left is the robot's left: a negative
    # head_pan on the model's -Z axis (console spec §2.1, amended), as teleop's Left key.
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
    assert srv.calls("/CurrentAngle") == []                                    # nothing on load
    only_startup(srv, page, "rosmaster_x3_plus")
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


ALL_SERVOS = list(range(1, 25))
# A standing AiNex (bent knees, arms down at the sides, head turned right): rad per joint.
STANDING = {"head_pan": 0.82, "l_knee": 1.206, "r_knee": -1.206, "l_hip_pitch": -0.83, "r_hip_pitch": 0.83,
            "l_ank_pitch": 0.62, "r_ank_pitch": -0.62, "l_sho_roll": -1.449, "r_sho_roll": 1.449}


def serve_servos(srv, rads, missing=()):
    """The board answers get_position with the pulses of `rads` (rad per joint, 0 elsewhere), as
    the controller maps them (angle2pulse); servos in `missing` are left out of the answer. Returns
    the pulse table, which the test may change (a servo that moved)."""
    pulses = {AINEX_SERVOS[j][0]: angle2pulse(j, rads.get(j, 0.0)) for j in AINEX_SERVOS if j not in missing}
    srv.set_service(GET_POSITION, lambda a: {"success": True, "position": [
        {"id": i, "position": pulses[i]} for i in a["id"] if i in pulses]})
    return pulses


def polls(srv):
    return srv.calls(GET_POSITION)


def test_ainex_pose_is_polled_once_validated_and_shown_without_a_click(view, fake):
    """Console spec §2.2 (amended): once the target is validated the page asks the
    documented read-only query about once a second, one call at a time, with every servo; the model
    shows the robot's pose (head turned, knees bent, arms down) within about 2 s, the controls show
    the measured values in rad and degrees, and targets not yet changed follow them. Nothing is
    commanded."""
    srv = fake(wire_spec(load("ainex")))
    serve_servos(srv, STANDING)
    t0 = time.monotonic()
    srv, page = view(srv=srv)
    page.wait_for_function("RC.state.model && RC.state.model.sources.r_knee === 'measured'", timeout=3000)
    assert time.monotonic() - t0 < 2.5
    m = model(page)
    for j in ("head_pan", "r_knee", "l_knee", "l_sho_roll", "r_sho_roll"):
        assert m["joints"][j] == pytest.approx(STANDING[j], abs=0.005), j
        assert m["sources"][j] == "measured", j
    assert {m["sources"][j] for j in AINEX_SERVOS} == {"measured"}
    assert "Pose measured" in page.locator('[data-testid="model-pose"]').inner_text()
    assert "s ago" in page.locator('[data-testid="model-pose"]').inner_text()
    page.wait_for_function("document.querySelector('[data-control=head_pan] [data-testid=read-value]').dataset.state === 'live'")
    assert "+0.82 rad (47°)" in read_value(page, "head_pan", "position").inner_text()
    assert "−1.21 rad (−69°)" in read_value(page, "r_knee", "position").inner_text()
    assert num(page, "r_knee", "position").input_value() == "-1.21"            # untouched targets follow
    assert num(page, "head_pan", "position").input_value() == "0.82"
    assert control(page, "head_pan").locator(".deg").inner_text() == "47°"
    assert "measured" in page.locator('[data-group="left_leg"] .jg-meta').inner_text()
    assert control(page, "head_pan").locator('[data-testid="read"]').count() == 0   # polled: no read button
    page.wait_for_timeout(3200)
    calls = polls(srv)
    assert 4 <= len(calls) <= 7, len(calls)                                     # about one per second
    assert all(c["args"] == {"id": ALL_SERVOS} for c in calls)
    assert min(gaps(calls)) > 0.8, gaps(calls)
    only_startup(srv, page, "ainex")
    assert srv.published() == []


def test_ainex_poll_waits_for_validation_and_stops_on_loss_and_target_change(view, fake):
    srv, page = view(retype(wire_spec(load("ainex")), HEAD_PAN, "std_msgs/Float64"), robot="ainex")
    page.wait_for_timeout(1500)
    assert page.locator('[data-testid="target"]').inner_text() == "none" and polls(srv) == []   # never before validation
    # Connection loss stops it.
    srv, page = view(wire_spec(load("ainex")))
    serve_servos(srv, STANDING)
    srv.wait_for(lambda: len(polls(srv)) >= 2, timeout=4)
    srv.drop_all()
    page.wait_for_function("RC.state.lost === true", timeout=8000)
    n = len(polls(srv))
    page.wait_for_timeout(2500)
    assert len(polls(srv)) == n
    page.wait_for_function("RC.state.model.sources.r_knee === 'target'", timeout=5000)   # no fresh measurement
    # A target change stops it: the old target is asked nothing more, and so is a refused one.
    srv, page = view(wire_spec(load("ainex")))
    serve_servos(srv, STANDING)
    srv.wait_for(lambda: len(polls(srv)) >= 2, timeout=4)
    page.select_option('[data-testid="robot"]', "myagv")
    page.click('[data-testid="select"]')
    page.wait_for_function("document.querySelector('[data-testid=reason]').innerText.includes('myagv')", timeout=8000)
    n = len(polls(srv))
    page.wait_for_timeout(2500)
    assert len(polls(srv)) == n
    assert unprompted(srv, "ainex") == []         # the start-up stop and the target change's stop only


def test_ainex_servo_joint_sends_one_servo_in_pulses_and_the_model_follows(view, fake):
    """The critic's acceptance on a fake wire: "Left elbow pitch" to +0.5 rad is exactly one message
    with servo 17 at pulse 619; the model draws the target until a measurement asked for after the
    change arrives, then the measurement. 2.2 rad is invalid and never sent. A leg joint streams
    while dragged, every message one servo."""
    srv = fake(wire_spec(load("ainex")))
    pulses = serve_servos(srv, STANDING)
    srv, page = view(srv=srv)
    page.wait_for_function("RC.state.model.sources.l_el_pitch === 'measured'", timeout=3000)
    f = num(page, "l_el_pitch", "position")
    f.fill("0.5")
    f.press("Enter")
    page.wait_for_timeout(300)
    assert [p["msg"] for p in srv.published(SET_POSITION)] == [{"duration": 0.5, "position": [{"id": 17, "position": 619}]}]
    m = model(page)
    assert m["joints"]["l_el_pitch"] == 0.5 and m["sources"]["l_el_pitch"] == "target" and "l_el_pitch" in m["clickable"]
    assert m["labels"]["l_el_pitch"] == "Left elbow pitch"
    pulses[17] = 619                                                         # the servo got there
    page.wait_for_function("RC.state.model.sources.l_el_pitch === 'measured'", timeout=3000)
    assert model(page)["joints"]["l_el_pitch"] == pytest.approx(0.5, abs=0.5 / 238.73)
    assert "+0.50 rad (29°)" in read_value(page, "l_el_pitch", "position").inner_text()
    f.fill("2.2")
    f.press("Enter")
    page.wait_for_timeout(300)
    assert len(srv.published(SET_POSITION)) == 1
    assert phase(page, "l_el_pitch") == "invalid" and "above documented maximum 2.09" in status(page, "l_el_pitch")
    assert "1 invalid" in page.locator('[data-group="left_arm"] .jg-state').inner_text()
    page.locator('[data-group="left_arm"] > summary').click()                 # collapsed: the state stays visible
    assert not page.locator('[data-group="left_arm"]').evaluate("e => e.open")
    assert page.locator('[data-group="left_arm"] .jg-state').is_visible()
    # A leg joint: dragging streams single-servo messages, the last one the final value.
    final = drag(page, "l_knee", "position", 0.6, 0.8, steps=20, step_ms=40)
    page.wait_for_timeout(300)
    knee = [p["msg"] for p in srv.published(SET_POSITION)][1:]
    assert len(knee) >= 3 and all(len(k["position"]) == 1 and k["position"][0]["id"] == 5 for k in knee)
    assert knee[-1]["position"][0]["position"] == angle2pulse("l_knee", final) and 0 <= min(
        k["position"][0]["position"] for k in knee)
    assert srv.published(HEAD_PAN) == [] and srv.calls("/walking/command") == [srv.calls("/walking/command")[0]]
    assert is_stop(load("ainex"), srv.calls("/walking/command")[0])           # the start-up stop only
    no_stop_ops(srv)


def test_ainex_model_click_highlights_and_reveals_the_joints_control(view, fake):
    """Clicking a joint in the model selects its control row (labelled as on the page) and scrolls
    the controls to it; the page itself does not move, so the model stays under the pointer."""
    srv = fake(wire_spec(load("ainex")))
    serve_servos(srv, STANDING)
    srv, page = view(srv=srv)
    page.wait_for_function("RC.state.model.sources.r_knee === 'measured'", timeout=3000)
    started(page)
    for joint, label in (("r_ank_roll", "Right ankle roll"), ("head_tilt", "Head tilt"), ("l_gripper", "Left hand (gripper)")):
        select_in_model(page, joint)
        scroll = page.evaluate("window.scrollY")
        row = control(page, joint)
        assert "hl" in row.get_attribute("class")
        assert page.locator(".control.hl").count() == 1
        page.wait_for_function("""j => { const r = document.querySelector(`[data-control=${j}]`).getBoundingClientRect(),
            p = document.getElementById('control-panel').getBoundingClientRect();
            return r.top >= p.top - 1 && r.bottom <= Math.min(p.bottom, innerHeight) + 1; }""", arg=joint, timeout=3000)
        assert abs(page.evaluate("window.scrollY") - scroll) <= 2           # (sub-pixel rounding)
        assert page.locator('[data-testid="model-sel"]').inner_text().startswith(f"{label} ({joint})")
        assert model(page)["labels"][joint] == label
    only_startup(srv, page, "ainex")


def test_model_is_large_beside_the_controls_and_marks_the_front(view, fake):
    """1440x900: the model sits in the left column under the camera, at least 560 px wide, the
    robot drawn over at least 60 % of its height. On a phone the page does not scroll sideways."""
    srv = fake(wire_spec(load("ainex")))
    serve_servos(srv, STANDING)
    srv, page = view(srv=srv, viewport=(1440, 900))
    page.wait_for_function("RC.state.model.sources.r_knee === 'measured'", timeout=3000)
    cv = page.locator('[data-testid="model-canvas"]').bounding_box()
    cam = page.locator('[data-testid="camera"]').bounding_box()
    panel = page.locator("#control-panel").bounding_box()
    assert cv["width"] >= 560 and cv["height"] >= 450, cv
    assert cv["x"] + cv["width"] <= panel["x"] and cv["y"] >= cam["y"] + cam["height"]
    x0, y0, x1, y1 = model(page)["drawn"]
    assert (y1 - y0) >= 0.6 * cv["height"], (y1 - y0, cv["height"])
    assert "front" in page.locator('[data-testid="model-canvas"]').get_attribute("aria-label")
    srv2 = fake(wire_spec(load("ainex")))
    serve_servos(srv2, STANDING)
    srv2, phone = view(srv=srv2, viewport=(390, 844))
    assert phone.evaluate("document.scrollingElement.scrollWidth") <= 390
    assert phone.locator('[data-testid="model-canvas"]').bounding_box()["width"] <= 390 - 2 * 12
    started(phone)
    assert phone.evaluate("document.scrollingElement.scrollWidth") <= 390          # with the motion pad
    only_startup(srv, page, "ainex")
    only_startup(srv2, phone, "ainex")


def test_numbers_read_the_same_in_every_locale(view):
    """A German browser shows "0,5" in number inputs; the page's fields show "0.5" like its
    readouts, and a typed comma is read as the decimal separator."""
    srv, page = view(wire_spec(load("ainex")), locale="de-DE")
    assert num(page, "head_pan", "position").input_value() == "0"
    assert num(page, "head_pan", "duration").input_value() == "0.5"
    num(page, "head_pan", "position").fill("0,3")
    num(page, "head_pan", "position").press("Enter")
    page.wait_for_timeout(300)
    assert [p["msg"] for p in srv.published(HEAD_PAN)] == [{"position": 0.3, "duration": 0.5}]
    page.locator('[data-control="head_pan"] input[data-field="position"]').press("ArrowUp")   # steps and sends
    page.wait_for_timeout(300)
    assert num(page, "head_pan", "position").input_value() == "0.31"
    assert srv.published(HEAD_PAN)[-1]["msg"]["position"] == 0.31
    assert ",3" not in page.locator(".pad-readout").inner_text()


# ------------------------------------------------------------------ sending, done and failed states

def phase(page, cid):
    return control(page, cid).get_attribute("data-phase")


def ainex_stand_with_a_field():
    """No packaged call control takes a value (the AiNex's Stand is a std_srvs/Empty call), so
    these tests hand the page an AiNex whose Stand call takes a bounded `duration`."""
    p = load("ainex")
    duration = Field(name="duration", label="Duration", min=0.2, max=2.0, default=0.5, unit="s")
    return dataclasses.replace(p, controls=tuple(
        dataclasses.replace(c, template={"duration": "$duration"}, fields=(duration,))
        if c.id == "init_pose" else c for c in p.controls))


def test_each_control_shows_its_sending_done_and_failed_state(view):
    srv, page = view(wire_spec(load("ainex")))
    started(page)
    assert phase(page, "init_pose") is None                                   # nothing sent yet
    srv.set_service("/walking/init_pose", "hang")
    control(page, "init_pose").locator('[data-testid="send"]').click()
    page.wait_for_timeout(200)
    assert phase(page, "init_pose") == "sending" and status(page, "init_pose").startswith("sending")
    srv.set_service("/walking/init_pose", "fail")                             # a service error: the call failed
    control(page, "init_pose").locator('[data-testid="send"]').click()
    page.wait_for_function("document.querySelector('[data-control=init_pose]').dataset.phase === 'failed'")
    srv.set_service("/walking/init_pose", lambda args: {})
    control(page, "init_pose").locator('[data-testid="send"]').click()
    page.wait_for_function("document.querySelector('[data-control=init_pose]').dataset.phase === 'done'")
    drag(page, "head_pan", "position", 0.5, 0.6, steps=5, step_ms=40)
    page.wait_for_timeout(300)
    assert phase(page, "head_pan") == "done" and "sent" in status(page, "head_pan")
    num(page, "head_pan", "position").fill("9")
    num(page, "head_pan", "position").press("Enter")
    page.wait_for_timeout(200)
    assert phase(page, "head_pan") == "invalid" and "not sent" in status(page, "head_pan")


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

def test_service_is_called_once_per_click_never_streamed(view, monkeypatch):
    serve_variants(monkeypatch, ainex_stand_with_a_field())
    srv, page = view(wire_spec(load("ainex")))
    started(page)
    drag(page, "init_pose", "duration", 0.5, 0.8, steps=15, step_ms=40)  # editing a service's values sends nothing
    page.wait_for_timeout(300)
    assert srv.calls("/walking/init_pose") == []
    send = control(page, "init_pose").locator('[data-testid="send"]')
    for _ in range(3):
        send.click()
    page.wait_for_timeout(300)
    calls = srv.calls("/walking/init_pose")
    assert len(calls) == 3
    assert calls[-1]["args"] == {"duration": float(num(page, "init_pose", "duration").input_value())} != {"duration": 0.5}
    assert "done" in status(page, "init_pose")
    no_stop_ops(srv)


def test_ros1_call_control_pending_and_failure_states(view):
    srv, page = view(wire_spec(load("ainex")))
    started(page)
    srv.set_service("/walking/init_pose", "hang")
    control(page, "init_pose").locator('[data-testid="send"]').click()
    page.wait_for_timeout(200)
    assert "pending" in status(page, "init_pose")
    srv.set_service("/walking/init_pose", "fail")
    control(page, "init_pose").locator('[data-testid="send"]').click()
    page.wait_for_function("document.querySelector('[data-control=init_pose] .status').innerText.includes('failed')")
    assert len(srv.calls("/walking/init_pose")) == 2
    assert [c["args"] for c in srv.calls("/walking/command")] == [{"command": "stop"}]    # the start-up stop
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


def with_optional(p, *names):
    """Profile ``p`` with the endpoint rows ``names`` optional (test variant)."""
    return dataclasses.replace(p, endpoints=tuple(dataclasses.replace(e, optional=True) if e.name in names else e
                                                  for e in p.endpoints))


def test_unsupported_controls_remain_unavailable(view, monkeypatch):
    """A control or read whose optional prerequisite is absent is unavailable with its reason; the
    rest of the target stays live. No packaged control or read depends on an optional row, so the
    page is handed an AiNex whose Stand service is optional and a ROSMASTER whose /CurrentAngle is."""
    serve_variants(monkeypatch, with_optional(load("ainex"), "/walking/init_pose"),
                   with_optional(load("rosmaster_x3_plus"), "/CurrentAngle"))
    srv, page = view(drop(wire_spec(load("ainex")), "/walking/init_pose"))
    assert page.locator('[data-testid="target"]').inner_text() == "ainex"
    assert "is not on the wire" in control(page, "init_pose").locator('[data-testid="unavailable"]').inner_text()
    assert disabled(page, "init_pose") and control(page, "init_pose").locator('[data-testid="send"]').is_disabled()
    assert not disabled(page, "head_pan") and not disabled(page, "action_group")
    only_startup(srv, page, "ainex")
    srv2, page2 = view(drop(wire_spec(load("rosmaster_x3_plus")), "/CurrentAngle"))
    assert page2.locator('[data-testid="target"]').inner_text() == "rosmaster_x3_plus"
    assert not disabled(page2, "arm_pose")
    assert control(page2, "arm_pose").locator('[data-testid="read"]').count() == 0     # its read is absent
    only_startup(srv2, page2, "rosmaster_x3_plus")


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


# ------------------------------------------------------------------ no bounded control is stopped automatically

@pytest.mark.parametrize("robot", ["so101", "ainex", "rosmaster_x3_plus"])
def test_release_escape_focus_hide_and_close_send_no_stop_for_bounded_controls(view, browser, robot):
    """Console spec §2.2/§4: release, focus loss, page hide or close send no stop or cancel for a
    bounded control. A mobile profile's base motion requests its own documented stop on focus loss
    and page hide (and nothing else); an arm's page sends nothing at all."""
    srv, page = view(so101_spec() if robot == "so101" else wire_spec(load(robot)))
    p = load(robot)
    cid, field = {"so101": ("arm_trajectory", "shoulder_pan"), "ainex": ("head_pan", "position"),
                  "rosmaster_x3_plus": ("arm_pose", "j1")}[robot]
    if not p.is_arm:
        started(page)
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
    new = srv.command_ops()[n:]
    if p.is_arm:
        assert new == []
    else:
        assert new and all(is_stop(p, o) for o in new), new                # base motion's stops only
    no_stop_ops(srv)
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
    and back. Each selection validates afresh on a new connection; the change sends nothing to the
    old target but base motion's stop, before leaving it (console spec §2.2 as amended),
    and the new validated target gets its start-up stop."""
    srv, page = view(wire_spec(load("ainex")))
    p = load("ainex")
    started(page)
    num(page, "head_pan", "position").fill("0.2")
    num(page, "head_pan", "position").press("Enter")
    page.wait_for_timeout(300)
    assert [p["msg"]["position"] for p in srv.published(HEAD_PAN)] == [0.2]
    n_clients, n_ops, n_all = len(srv.clients), len(srv.command_ops()), len(srv.ops)
    page.select_option('[data-testid="robot"]', "myagv")
    page.click('[data-testid="select"]')
    page.wait_for_function("document.querySelector('[data-testid=reason]').innerText.includes('myagv')", timeout=8000)
    assert page.locator('[data-testid="target"]').inner_text() == "none"
    assert page.locator('[data-testid="control"]').count() == 0
    assert len(srv.clients) == n_clients + 1
    after = srv.ops[n_all:]
    (stop,) = srv.command_ops()[n_ops:]                                 # only base motion's stop...
    assert is_stop(p, stop)
    first_rosapi = next(i for i, o in enumerate(after) if o.get("service", "").startswith("/rosapi/"))
    assert after.index(stop) < first_rosapi                             # ...before the new validation
    assert page.locator('[data-testid="banner"]').is_hidden()           # it did not fail
    page.select_option('[data-testid="robot"]', "ainex")
    page.click('[data-testid="select"]')
    page.wait_for_function("document.querySelector('[data-testid=target]').innerText === 'ainex'", timeout=8000)
    assert len(srv.clients) == n_clients + 2
    started(page)
    page.wait_for_timeout(300)
    assert len(stops(srv, p)) == 3                                      # start-up, target change, start-up
    assert unprompted(srv, "ainex") == srv.published(HEAD_PAN)
    no_stop_ops(srv)
    num(page, "head_pan", "position").fill("0.3")
    num(page, "head_pan", "position").press("Enter")
    page.wait_for_timeout(400)
    assert [p["msg"]["position"] for p in srv.published(HEAD_PAN)] == [0.2, 0.3]     # once, from the new target


@pytest.mark.parametrize("robot", ["so101", "ainex"])
def test_connection_loss_disables_controls_sends_no_stop_and_never_reconnects(view, robot):
    srv, page = view(so101_spec() if robot == "so101" else wire_spec(load(robot)))
    cid, field = ("arm_trajectory", "shoulder_pan") if robot == "so101" else ("head_pan", "position")
    if robot == "ainex":
        started(page)
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
    only_startup(srv, page, "rosmaster_x3_plus")


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
    """Child link origins in the model's root frame, computed here from the URDF rows (origin xyz/rpy,
    then a rotation about the axis, or a slide along it for a prismatic joint; mimic joints follow
    their joint)."""
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
            if j["type"] == "prismatic":
                tc = tc + Rc @ (np.asarray(j["axis"], float) / np.linalg.norm(j["axis"]) * angle)
            else:
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


def test_a_prismatic_joint_slides_along_its_axis(view, monkeypatch):
    """No packaged model has a prismatic joint, so the page is handed a myAGV model with a reported
    prismatic joint (tilted origin, non-unit axis) and a frame after it: it slides, it does not turn."""
    packaged = view_mod.all_profiles_json
    extra = [{"name": "lift", "type": "prismatic", "parent": "base_up", "child": "lift_link", "xyz": [0.05, 0.02, 0.1],
              "rpy": [0.3, -0.2, 1.1], "axis": [0, 3, 4], "min": 0, "max": 0.2, "reported": "lift"},
             {"name": "lift_tool", "type": "fixed", "parent": "lift_link", "child": "tool", "xyz": [0.1, 0, 0.05],
              "rpy": [0, 0, 0]}]

    stub = dict(load("myagv").raw["model"], joints=load("myagv").raw["model"]["joints"] + extra)

    def with_prismatic():
        j = packaged()
        next(x for x in j["profiles"] if x["id"] == "myagv")["model"] = stub
        return j
    monkeypatch.setattr(view_mod, "all_profiles_json", with_prismatic)
    srv, page = view(wire_spec(load("myagv")))
    stop = joint_states(srv, ["base_up", "lift"], [0.7, 0.15])
    try:
        page.wait_for_function("RC.state.model && RC.state.model.sources.lift === 'reported'", timeout=5000)
        world = model(page)["world"]
    finally:
        stop.set()
    want = urdf_forward(stub, {"base_up": 0.7, "lift": 0.15})
    for j in ("lift", "lift_tool"):
        assert world[j] == pytest.approx(list(want[j]), abs=1e-9), j
    only_startup(srv, page, "myagv")


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
    srv, page = view(wire_spec(load("ainex")))                              # the board answers no positions
    started(page)
    m = model(page)
    assert set(m["clickable"]) == set(AINEX_SERVOS)                         # every servo's joint
    assert {m["sources"][j] for j in AINEX_SERVOS} == {"target"}
    assert m["joints"]["r_knee"] == pytest.approx(-1.192)                   # the standing default, not 0
    assert "Pose not measured" in page.locator('[data-testid="model-pose"]').inner_text()
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
    only_startup(srv, page, robot)                                       # selecting sends nothing


# ------------------------------------------------------------------ same contracts as Python

CASES = {
    "so101": lambda: wire_spec(load("so101")),
    "so101-arm1": lambda: wire_spec(load("so101"), namespace="arm1"),
    "two-so101": lambda: merge(wire_spec(load("so101"), namespace="arm1"), wire_spec(load("so101"), namespace="arm2")),
    "ambiguous": lambda: merge(wire_spec(load("myagv")), wire_spec(load("rosmaster_x3_plus"))),
    "missing": lambda: drop(wire_spec(load("ainex")), "/walking/command"),
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


def test_page_does_not_select_a_subsumed_profile(view, monkeypatch):
    """As discovery.py (test_discovery.py): a wire presenting every name one profile requires, and
    more, is the larger profile's, and the smaller one is refused even when selected explicitly.
    No two packaged profiles nest, so the page's catalog also holds a profile of the SO-101 booted
    without its camera."""
    serve_variants(monkeypatch, without_camera(load("so101"), "so101_nocam"))
    for robot, shown in ((None, "so101"), ("so101_nocam", "none")):
        srv, page = view(so101_spec(), robot=robot)
        rb = Rosbridge(srv.url).connect()
        try:
            g = fetch_graph(rb)
        finally:
            rb.close()
        assert page.locator('[data-testid="target"]').inner_text() == shown
        assert select_target(g, None, None).label == "so101"
        if robot:
            reason = page.locator('[data-testid="reason"]').inner_text()
            assert "presents so101, whose interface contains all of so101_nocam" in reason, reason
            assert page.locator('[data-testid="control"]').count() == 0
        assert srv.command_ops() == []
    srv, page = view(drop(so101_spec(), "/image_raw"))                  # without the camera: the smaller one
    assert page.locator('[data-testid="target"]').inner_text() == "so101_nocam"


# ------------------------------------------------------------------ base motion (console spec §2.2, §4 as amended)

MOBILE = ["myagv", "ainex", "rosmaster_x3_plus"]
KEY_SIGN = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0), "q": (0, 0, 1), "e": (0, 0, -1)}
WALK = "/walking/command"


def speeds(p):
    ax = (p.teleop_base or p.teleop_walk).axes
    return tuple(ax[k].speed for k in ("x", "y", "yaw"))


def limits(p):
    ax = (p.teleop_base or p.teleop_walk).axes
    return tuple(ax[k].limit for k in ("x", "y", "yaw"))


def axes_of(p, o):
    """(x, y, yaw) of a base-motion command (a Twist, or the AiNex gait amplitudes), else None."""
    if o["op"] != "publish":
        return None
    m = o["msg"]
    if p.teleop_base and o["topic"] == p.teleop_base.topic:
        return (m["linear"]["x"], m["linear"]["y"], m["angular"]["z"])
    if p.teleop_walk and o["topic"] == p.teleop_walk.param_topic:
        ax = p.teleop_walk.axes
        return tuple(m[ax[k].field] for k in ("x", "y", "yaw"))
    return None


def motion_cmds(p, ops):
    """The motion commands among `ops` (a stop is not one)."""
    return [a for o in ops if not is_stop(p, o) and (a := axes_of(p, o)) is not None]


def walk_calls(ops):
    return [o["args"]["command"] for o in ops if o.get("op") == "call_service" and o.get("service") == WALK]


def mo(page):
    return page.evaluate("RC.motion()")


def settled(page):
    """Every motion event handled (and its sends answered)."""
    page.wait_for_function("RC.motion() && RC.motion().pending === 0", timeout=8000)


def mobile(view, pid, srv=None, **kw):
    srv, page = view(wire_spec(load(pid)) if srv is None else None, srv=srv, **kw)
    started(page)
    return srv, page, load(pid)


def blur(page):
    page.evaluate("window.dispatchEvent(new Event('blur'))")


def hide(page):
    page.evaluate("Object.defineProperty(document, 'visibilityState', {value: 'hidden', configurable: true});"
                  "document.dispatchEvent(new Event('visibilitychange'))")


def refuse_stop(args):
    """An AiNex controller whose walking stop errs (the call answers result false)."""
    if args.get("command") == "stop":
        raise RuntimeError("the gait refused to stop")
    return {}


# The transport rejects a base's stop on a usable connection: the page's websocket cannot send a
# zero Twist on /cmd_vel while window.__rejectStop is set (the ROSMASTER's and myAGV's stop is a
# publish, which no server acknowledges, so only the transport can fail it).
REJECT_STOP_JS = """() => {
    const send = WebSocket.prototype.send;
    WebSocket.prototype.send = function (d) {
        const m = JSON.parse(d);
        const zero = (v) => v && Object.values(v).every((p) => Object.values(p).every((x) => x === 0));
        if (window.__rejectStop && m.op === 'publish' && m.topic === '/cmd_vel' && zero(m.msg)) throw new Error('the transport rejected the stop');
        return send.call(this, d);
    };
}"""


@pytest.mark.parametrize("pid", MOBILE)
def test_base_motion_startup_stop_after_validation_without_resumed_motion(view, pid):
    """Once the target validates the page advertises teleop's command topics, lets them settle,
    attempts the documented stop once and shows the outcome, then enables base motion; a reload
    with a key still held stops again and resumes nothing."""
    srv, page, p = mobile(view, pid)
    ops = srv.command_ops()
    adv = [o for o in ops if o["op"] == "advertise"]
    assert [o["topic"] for o in adv] == [t for t, _ in command_topics(p)]
    rest = [o for o in ops if o["op"] != "advertise"]
    assert len(rest) == len(stop_ops(p)) and all(is_stop(p, o) for o in rest), rest
    validation = [o for o in srv.ops if o.get("service") in ("/rosapi/topics", "/rosapi/services")]
    assert max(o["_t"] for o in validation) < adv[0]["_t"] <= rest[0]["_t"] - 0.5
    state = page.locator('[data-testid="motion-state"]')
    assert state.get_attribute("data-state") == "stopped" and "stop requested (start-up, after validation)" in state.inner_text()
    assert mo(page)["enabled"] and not page.locator('[data-testid="reenable"]').is_visible()
    page.keyboard.down("w")
    page.wait_for_timeout(400)
    assert motion_cmds(p, srv.command_ops())
    n = len(srv.command_ops())
    page.reload()
    page.wait_for_function("window.RC && window.RC.ready === true")
    started(page)
    page.keyboard.up("w")                                                 # the old page's key
    page.wait_for_timeout(600)
    new = [o for o in srv.command_ops()[n:] if o["op"] != "advertise"]
    first = next(i for i, o in enumerate(new) if is_stop(p, o))           # the unload's stop, or the start-up stop
    assert motion_cmds(p, new[first:]) == [] and mo(page)["held"] == []   # nothing resumed
    assert mo(page)["outcomes"][0]["text"].startswith("stop requested (start-up, after validation)")


@pytest.mark.parametrize("pid", MOBILE)
def test_base_motion_key_direction_recorded_speeds_and_documented_limits(view, pid):
    """Each motion key commands its axis at the profile's recorded speed, re-sent at the profile's
    rate while held; release requests the stop. Keys combine per axis, opposite keys cancel, and
    nothing exceeds the documented limits. The AiNex walks through its gait: the walking enable
    with the first press after enabling, then the parameters and start, as teleop does."""
    srv, page, p = mobile(view, pid)
    sx, sy, sw = speeds(p)
    for i, (key, sign) in enumerate(KEY_SIGN.items()):
        n = len(srv.command_ops())
        page.keyboard.down(key)
        page.wait_for_timeout(450)
        assert mo(page)["state"] == "moving"
        page.keyboard.up(key)
        settled(page)
        new = srv.command_ops()[n:]
        assert set(motion_cmds(p, new)) == {(sign[0] * sx, sign[1] * sy, sign[2] * sw)}, (key, motion_cmds(p, new))
        assert is_stop(p, new[-1]), "release of the last motion key requests the stop"
        if p.teleop_walk:
            assert walk_calls(new) == (["enable_control", "enable"] if i == 0 else []) + ["start", "stop"], walk_calls(new)
            assert new[len(walk_calls(new)) - 2]["op"] == "publish"          # the parameters before start
    # held: re-sent at the profile's rate
    n = len(srv.command_ops())
    page.keyboard.down("w")
    t0 = time.monotonic()
    page.wait_for_timeout(1500)
    page.keyboard.up("w")
    dur = time.monotonic() - t0
    settled(page)
    sent = [o for o in srv.command_ops()[n:] if axes_of(p, o) and not is_stop(p, o)]
    rate = (p.teleop_base or p.teleop_walk).rate_hz
    assert dur * rate * 0.6 <= len(sent) <= dur * rate + 2, (len(sent), dur, rate)
    assert min(gaps(sent)) > 0.8 / rate, gaps(sent)
    # combined keys, opposite keys cancel; within the documented limits
    n = len(srv.command_ops())
    for k in ("w", "a", "q"):
        page.keyboard.down(k)
        page.wait_for_timeout(250)
    assert "x +" in page.locator('[data-testid="motion-cmd"]').inner_text()
    page.keyboard.down("s")
    page.wait_for_timeout(300)
    page.keyboard.down("d")
    page.keyboard.down("e")
    page.wait_for_timeout(300)
    cmds = motion_cmds(p, srv.command_ops()[n:])
    assert (sx, sy, sw) in cmds and (0, sy, sw) in cmds, cmds
    assert all(abs(v) <= lim for c in cmds for v, lim in zip(c, limits(p)))
    assert all(abs(v) <= lim for c in motion_cmds(p, srv.command_ops()) for v, lim in zip(c, limits(p)))
    m = len(srv.command_ops())
    for k in ("w", "a", "q", "s", "d"):
        page.keyboard.up(k)
    settled(page)
    assert not any(is_stop(p, o) for o in srv.command_ops()[m:])          # E still held
    page.keyboard.up("e")
    settled(page)
    assert is_stop(p, srv.command_ops()[-1])


@pytest.mark.parametrize("pid", MOBILE)
def test_base_motion_stops_on_release_space_stop_button_focus_loss_and_page_hide(view, pid):
    """Release of a hold button (also a touch cancelled), Space, the STOP button, focus loss and
    page hide clear the intent and request the documented stop; a key still held afterwards moves
    nothing (movement needs a fresh press)."""
    srv, page, p = mobile(view, pid)
    hold_w = page.locator('[data-testid="hold-w"]')
    hold_w.scroll_into_view_if_needed()
    # a hold button: pointer capture, release on pointerup (even off the button)
    n = len(srv.command_ops())
    box = hold_w.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.wait_for_timeout(350)
    assert hold_w.get_attribute("data-held") == "true" and mo(page)["held"] == ["w"]
    page.mouse.move(box["x"] - 120, box["y"] - 80)
    page.wait_for_timeout(250)
    assert mo(page)["held"] == ["w"]                                      # captured: still held
    page.mouse.up()
    settled(page)
    new = srv.command_ops()[n:]
    assert motion_cmds(p, new) and is_stop(p, new[-1])
    assert "last motion button released" in mo(page)["outcomes"][-1]["text"]
    # a touch that the browser cancels
    n = len(srv.command_ops())
    page.evaluate("""() => { const b = document.querySelector('[data-testid=hold-a]');
        b.dispatchEvent(new PointerEvent('pointerdown', {pointerId: 7, pointerType: 'touch', isPrimary: true, bubbles: true})); }""")
    page.wait_for_timeout(350)
    assert mo(page)["held"] == ["a"]
    page.evaluate("""() => document.querySelector('[data-testid=hold-a]').dispatchEvent(
        new PointerEvent('pointercancel', {pointerId: 7, pointerType: 'touch', bubbles: true}))""")
    settled(page)
    new = srv.command_ops()[n:]
    assert motion_cmds(p, new) and is_stop(p, new[-1])

    def held_then(action, reason):
        n = len(srv.command_ops())
        page.keyboard.down("w")
        page.wait_for_timeout(350)
        assert motion_cmds(p, srv.command_ops()[n:]), reason
        action()
        settled(page)
        page.wait_for_timeout(450)                                        # W is still down
        new = srv.command_ops()[n:]
        assert is_stop(p, new[-1]) and motion_cmds(p, new[new.index(new[-1]):]) == [], reason
        assert reason in mo(page)["log"][-1] and mo(page)["held"] == []
        m = len(srv.command_ops())
        page.keyboard.up("w")                                             # releasing the stale key sends nothing
        page.wait_for_timeout(300)
        assert srv.command_ops()[m:] == [], reason

    held_then(lambda: page.keyboard.press(" "), "Space")
    assert page.locator('[data-testid="motion-state"]').get_attribute("data-state") == "stopped"
    held_then(lambda: page.locator('[data-testid="stop"]').click(), "Stop button")
    held_then(lambda: blur(page), "window lost keyboard focus")
    held_then(lambda: hide(page), "page hidden")
    # page hide or unload: the stop goes out at once, best effort, and the page says it is not guaranteed
    n = len(srv.command_ops())
    page.keyboard.down("d")
    page.wait_for_timeout(350)
    page.evaluate("window.dispatchEvent(new Event('pagehide'))")
    page.wait_for_timeout(500)
    new = srv.command_ops()[n:]
    assert motion_cmds(p, new) and is_stop(p, new[-1])
    assert "not guaranteed" in mo(page)["log"][-1]
    assert "not guaranteed" in page.locator('[data-testid="motion"]').inner_text()


@pytest.mark.parametrize("pid", MOBILE)
def test_failed_stop_disables_base_motion_until_reenable_and_fresh_input(view, pid):
    """An explicitly failed stop (the AiNex walking stop answers an error; a base's stop the
    transport rejects) disables base motion with §2.1's limitation shown; keys and buttons move
    nothing; re-enabling (the button, or Enter as in teleop) needs fresh input: a key held across it
    does not resume motion."""
    srv, page, p = mobile(view, pid)
    if p.teleop_walk:
        srv.set_service(WALK, refuse_stop)
    else:
        page.evaluate(REJECT_STOP_JS)
        page.evaluate("window.__rejectStop = true")
    page.keyboard.down("w")
    page.wait_for_timeout(350)
    page.keyboard.up("w")
    page.wait_for_function("RC.motion().state === 'failed'", timeout=8000)
    text = page.locator('[data-testid="motion-state"]').inner_text()
    assert "Stop failed" in text and "STOP FAILED (last motion key released)" in text
    assert "Re-enabling does not check or confirm that the robot stopped" in text
    assert page.locator('[data-testid="motion-limit"]').is_visible() and page.locator('[data-testid="reenable"]').is_visible()
    assert page.locator('[data-testid="hold-w"]').get_attribute("aria-disabled") == "true"
    n = len(srv.command_ops())
    page.keyboard.down("s")                                               # disabled: nothing moves
    page.wait_for_timeout(300)
    page.keyboard.up("s")
    page.locator('[data-testid="hold-q"]').click(force=True)            # (aria-disabled)
    settled(page)
    assert motion_cmds(p, srv.command_ops()[n:]) == []
    assert "disabled after a failed stop" in mo(page)["log"][-1]
    # re-enable with a key held across it: not a fresh press
    if p.teleop_walk:
        srv.set_service(WALK, lambda args: {})
    else:
        page.evaluate("window.__rejectStop = false")
    page.keyboard.down("w")
    page.wait_for_timeout(200)
    m = len(srv.command_ops())
    if pid == "myagv":
        page.evaluate("document.activeElement && document.activeElement.blur && document.activeElement.blur()")
        page.keyboard.press("Enter")                                      # teleop's re-enable key
    else:
        page.locator('[data-testid="reenable"]').click()
    page.wait_for_function("RC.motion().enabled === true", timeout=8000)
    page.wait_for_timeout(500)
    assert motion_cmds(p, srv.command_ops()[m:]) == [] and mo(page)["held"] == []
    if p.teleop_walk:
        assert walk_calls(srv.command_ops()[m:]) == ["enable_control", "enable"]   # teleop's enable, on the explicit action
    assert "ENABLED" in mo(page)["log"][-1]
    page.keyboard.up("w")
    page.wait_for_timeout(200)
    page.keyboard.down("w")                                               # fresh input moves again
    page.wait_for_timeout(350)
    page.keyboard.up("w")
    settled(page)
    new = srv.command_ops()[m:]
    assert motion_cmds(p, new) and is_stop(p, new[-1]) and mo(page)["outcomes"][-1]["ok"]


def test_ainex_walking_stop_timeout_is_a_failure(view):
    """A walking stop call that does not answer within teleop's 3 s fails explicitly."""
    srv, page, p = mobile(view, "ainex")
    page.keyboard.down("w")
    page.wait_for_timeout(350)
    srv.set_service(WALK, "hang")
    page.keyboard.press(" ")
    page.wait_for_timeout(1000)
    assert mo(page)["state"] == "stopping"
    page.wait_for_function("RC.motion().state === 'failed'", timeout=6000)
    assert "did not answer within 3 s" in mo(page)["outcomes"][-1]["text"]
    page.keyboard.up("w")


@pytest.mark.parametrize("pid", MOBILE)
def test_connection_loss_clears_motion_intent_and_claims_no_stop(view, pid):
    srv, page, p = mobile(view, pid)
    page.keyboard.down("w")
    page.wait_for_timeout(350)
    srv.drop_all()
    page.wait_for_function("RC.state.lost === true", timeout=8000)
    m = mo(page)
    assert m["state"] == "lost" and m["held"] == [] and not m["enabled"]
    assert m["log"][-1].startswith("CONNECTION LOST") and "stop requested" not in m["log"][-1]
    text = page.locator('[data-testid="motion-state"]').inner_text()
    assert "No connection" in text and "No stop could be delivered" in text and "stopping is NOT guaranteed" in text
    banner = page.locator('[data-testid="banner"]').inner_text()
    assert "nothing was stopped" in banner and "NOT guaranteed" in banner
    assert page.locator('[data-testid="stop"]').is_disabled()
    assert page.locator('[data-testid="hold-w"]').get_attribute("aria-disabled") == "true"
    page.keyboard.up("w")
    page.keyboard.press(" ")
    page.wait_for_timeout(300)
    assert mo(page)["state"] == "lost"


@pytest.mark.parametrize("pid", MOBILE)
def test_motion_keys_typed_into_editable_fields_move_nothing(view, pid):
    """Keys act only while the page, not an editable field, has keyboard focus: typing in a number
    field, the robot list or a slider sends no base motion or head command; a key held when focus
    moves into a field is released (its stop requested)."""
    srv, page, p = mobile(view, pid)
    fields = ['[data-testid="robot"]']
    if p.controls:
        cid = next(c.id for c in p.controls if any(f.choices is None for f in c.fields))
        fields += [f'[data-control="{cid}"] input.num', f'[data-control="{cid}"] input[type="range"]']
    n_all = len(srv.ops)
    n = len(srv.command_ops())
    for sel in fields:
        page.locator(sel).first.focus()
        # (Space and Enter would open the list's popup)
        for k in ("w", "s", "a", "d", "q", "e") + (() if "robot" in sel else (" ", "Enter")):
            page.keyboard.down(k)
            page.wait_for_timeout(40)
            page.keyboard.up(k)
    page.locator('[data-testid="robot"]').focus()
    for k in ("ArrowLeft", "ArrowUp"):
        page.keyboard.press(k)
    settled(page)
    page.wait_for_timeout(300)
    new = srv.command_ops()[n:]
    assert motion_cmds(p, new) == [] and not any(is_stop(p, o) for o in new) and walk_calls(new) == []
    assert not [o for o in srv.ops[n_all:] if o.get("args") == {"id": [23, 24]}]    # no head-key read
    assert mo(page)["log"][-1].startswith("base motion ENABLED")         # nothing happened since start-up
    # with Meta (⌘) down the browser may not report key releases: the keys' holds end there
    page.evaluate("document.activeElement.blur()")
    n = len(srv.command_ops())
    page.keyboard.down("w")
    page.wait_for_timeout(300)
    page.keyboard.down("Meta")
    settled(page)
    page.keyboard.up("Meta")
    page.keyboard.up("w")
    new = srv.command_ops()[n:]
    assert motion_cmds(p, new) and is_stop(p, new[-1]) and "keyboard input lost" in mo(page)["outcomes"][-1]["text"]
    # a held key whose focus moves into a field is released
    page.evaluate("document.activeElement.blur()")
    page.keyboard.down("w")
    page.wait_for_timeout(350)
    page.locator('[data-testid="robot"]').focus()
    settled(page)
    page.wait_for_timeout(300)
    new = srv.command_ops()[n:]
    assert motion_cmds(p, new) and is_stop(p, new[-1])
    assert "keyboard focus moved to an editable field" in mo(page)["outcomes"][-1]["text"]
    page.keyboard.up("w")


def test_arm_profile_offers_no_base_motion(view):
    assert [p.id for p in load_all() if p.is_arm] == ["so101"]
    srv, page = view(so101_spec())
    assert page.evaluate("RC.motion()") is None
    assert not page.locator('[data-testid="motion"]').is_visible()
    assert page.locator('[data-testid^="hold-"]').count() == 0 and page.locator('[data-testid="stop"]').count() == 0
    for k in ("w", "s", "a", "d", "q", "e", " ", "Enter", "ArrowLeft", "ArrowUp"):
        page.keyboard.press(k)
    page.keyboard.down("w")
    blur(page)
    hide(page)
    page.evaluate("window.dispatchEvent(new Event('pagehide'))")
    page.wait_for_timeout(600)
    assert srv.command_ops() == []


def test_ainex_head_keys_start_from_the_measured_head_and_stay_within_limits(view, fake):
    """The arrow keys move the head as teleop's do: the head's position is read (the documented
    read) when the first arrow of an axis is pressed, then the target slews at the profile's rate
    in the key's direction (Left = a negative pan), published at most every 0.1 s with teleop's
    head message, clamped to the documented limits; release stops the target changing and
    requests no walking stop. An unknown position ignores the key. The page's head target shows
    what the keys sent."""
    srv = fake(wire_spec(load("ainex")))
    pulses = serve_servos(srv, STANDING)                                   # head_pan 0.82 rad, tilt 0
    srv, page, p = mobile(view, "ainex", srv=srv)
    hp = p.head["pan"]
    n_all, n = len(srv.ops), len(srv.published(HEAD_PAN))
    page.keyboard.down("ArrowLeft")
    page.wait_for_timeout(800)
    page.keyboard.up("ArrowLeft")
    settled(page)
    reads = [o for o in srv.ops[n_all:] if o.get("service") == GET_POSITION and o["args"] == {"id": [23, 24]}]
    assert len(reads) == 1                                                # read once, at the first press
    pubs = srv.published(HEAD_PAN)[n:]
    vals = [o["msg"]["position"] for o in pubs]
    assert all(o["msg"] == dict(hp.template, position=o["msg"]["position"]) for o in pubs)   # teleop's message
    start = (pulses[23] - 500) / 238.7324187                              # the measured pan
    assert vals[0] < start and all(b < a for a, b in zip(vals, vals[1:])), (start, vals)
    assert 0.4 <= start - vals[-1] <= 0.95, vals                          # ~1 rad/s for ~0.8 s
    assert min(gaps(pubs)) >= 0.09, gaps(pubs)
    assert float(num(page, "head_pan", "position").input_value()) == pytest.approx(vals[-1], abs=0.006)
    page.wait_for_timeout(400)
    assert len(srv.published(HEAD_PAN)) == n + len(pubs)                  # released: the target stops
    assert walk_calls(srv.ops[n_all:]) == []                              # head keys never stop walking
    # Right, held long enough to reach the documented limit: clamped there, never beyond
    n = len(srv.published(HEAD_PAN))
    page.keyboard.down("ArrowRight")
    page.wait_for_timeout(1300)
    page.keyboard.up("ArrowRight")
    settled(page)
    vals = [o["msg"]["position"] for o in srv.published(HEAD_PAN)[n:]]
    assert max(vals) == hp.max and all(v <= hp.max for v in vals)
    # Up tilts up
    n = len(srv.published(HEAD_TILT))
    page.keyboard.down("ArrowUp")
    page.wait_for_timeout(400)
    page.keyboard.up("ArrowUp")
    settled(page)
    tilt = [o["msg"]["position"] for o in srv.published(HEAD_TILT)[n:]]
    assert tilt and tilt[-1] > 0 and all(v <= p.head["tilt"].max for v in tilt)
    # an unknown head position ignores the key
    del pulses[23]
    n = len(srv.published(HEAD_PAN))
    page.keyboard.down("ArrowLeft")
    page.wait_for_timeout(400)
    page.keyboard.up("ArrowLeft")
    settled(page)
    assert len(srv.published(HEAD_PAN)) == n
    assert "head position unknown (pan)" in mo(page)["log"][-1] and "key ignored" in mo(page)["log"][-1]
    # arrow keys keep their own meaning on the head pad, a joint chip and the model
    pulses[23] = 500
    n_all = len(srv.ops)
    pad = page.locator(".pad")
    pad.focus()
    pad_before = float(num(page, "head_pan", "position").input_value())
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(300)
    assert srv.published(HEAD_PAN)[-1]["msg"] == {"position": pytest.approx(pad_before - 0.05), "duration": 0.5}   # the pad's step
    assert not [o for o in srv.ops[n_all:] if o.get("args") == {"id": [23, 24]}]
    no_stop_ops(srv)


def test_page_motion_pad_layout_has_legend_and_state(view):
    """The motion pad sits with the cameras, above the model: a D-pad of hold buttons around STOP,
    the AiNex head buttons, the key legend with the recorded speeds and limits, and a state readout."""
    srv, page, p = mobile(view, "ainex", viewport=(1440, 900))
    card = page.locator('[data-testid="motion"]')
    cam = page.locator('[data-testid="camera"]').bounding_box()
    mdl = page.locator('[data-testid="model"]').bounding_box()
    box = card.bounding_box()
    assert cam["y"] + cam["height"] <= box["y"] and box["y"] + box["height"] <= mdl["y"]
    assert box["x"] + box["width"] <= page.locator("#control-panel").bounding_box()["x"]
    for k in ("w", "s", "a", "d", "q", "e", "up", "down", "left", "right"):
        assert card.locator(f'[data-testid="hold-{k}"]').is_visible(), k
    stop = card.locator('[data-testid="stop"]').bounding_box()
    w, s = card.locator('[data-testid="hold-w"]').bounding_box(), card.locator('[data-testid="hold-s"]').bounding_box()
    assert w["y"] < stop["y"] < s["y"]                                    # STOP in the middle of the pad
    legend = card.inner_text()
    for t in ("x_move_amplitude 0.01 m (documented limit 0.02 m)", "angle_move_amplitude 5 deg (documented limit 10 deg)",
              "head pan left / right", "Space", "Enter"):
        assert t in legend, t
    assert page.locator('[data-testid="motion-state"]').get_attribute("data-state") == "stopped"
    srv2, page2, _ = mobile(view, "myagv")
    legend = page2.locator('[data-testid="motion"]').inner_text()
    assert "forward / back: 0.2 m/s (documented limit 0.9 m/s)" in legend and "rotate" in legend
    assert page2.locator('[data-testid="hold-left"]').count() == 0          # no head
