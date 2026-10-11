"""The view.sh page in headless Chromium against the fake rosbridge (console spec §2.2, §2.3, §4).

The page speaks rosbridge itself; the fake server records every op it sends, so the tests
check what reached the wire, not only what the page displays. Contract: controls are live as
soon as the target validates; changed values are sent at once (topic publishes and action
goals streamed while dragging, throttled with the last value always delivered; to an action
server that does not preempt, one goal at a time, the latest value held until the running
goal's result; services and action-group buttons once per click); limits are never exceeded; no bounded control is ever
stopped or cancelled by the page. The 3D model is posed from the joint positions the robot
reports (where the profile documents them) and its commanded joints edit their controls through
the same path. For a mobile profile the page also offers teleop.sh's hold-to-move base motion
(console spec §2.2), tested at the end of this file; elsewhere, loading such
a page sends its start-up stop and nothing else.
"""

import dataclasses
import json
import math
import re
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
from test_discovery import SUBSET_DROPPED, ns_camera_arm, without_row
from test_profiles import AINEX_SERVOS, GET_POSITION, SET_POSITION, angle2pulse
from wirespec import drop, merge, retype, wire_spec, with_hidden_action_endpoints

pytestmark = pytest.mark.browser
playwright = pytest.importorskip("playwright.sync_api")

REBOT = "rebot_b601_rs (namespace /rebotarm)"     # the reBot's target at its documented default namespace
ARM = "/rebotarm/follow_joint_trajectory"
GRIP = "/rebotarm/gripper/command"
REBOT_JS = "/rebotarm/joint_states"
HEAD_PAN = "/head_pan_controller/command"
HEAD_TILT = "/head_tilt_controller/command"
GOAL_CONNS_MAX = 3          # console.js: open goal connections per control
# The profiles' documented read-only position queries, which the page polls once a target is
# validated (console spec §2.2): not commands (FakeRosbridge.read_only).
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


def rebot_spec(exec_s=20.0, namespace=None, profile=None, include_optional=False):
    """The reBot's wire: its arm server rejects a goal while one runs, its gripper's runs goals
    side by side (the profile's `busy`), each goal running `exec_s`; its sensors publish samples,
    and its optional D405 topics are on the wire only with ``include_optional``."""
    spec = wire_spec(profile or load("rebot_b601_rs"), namespace=namespace, include_optional=include_optional)
    for a in spec["actions"]:
        a["exec_s"] = exec_s                 # goals outlive the test unless preempted
    return spec


def preempting_rebot():
    """The reBot as if its action servers preempted a running goal with a new one (as a
    ros2_control controller does): no packaged profile's server preempts, so the page's streaming
    of preempting goals is tested on this variant (tests only)."""
    p = load("rebot_b601_rs")
    return dataclasses.replace(p, controls=tuple(dataclasses.replace(c, busy="preempt") for c in p.controls))


def preempting(monkeypatch, exec_s=20.0):
    """Serve the preempting variant in place of the reBot; its wire, with preempting servers."""
    p = preempting_rebot()
    serve_variants(monkeypatch, p)
    spec = rebot_spec(exec_s, profile=p)
    assert {a["busy"] for a in spec["actions"]} == {"preempt"}
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
    srv, page = view(wire_spec(p, include_optional=True))     # an optional camera is on the wire only with this
    cams = page.locator('[data-testid="camera"]')
    assert sorted(cams.evaluate_all("els => els.map(e => e.dataset.topic)")) == sorted(c.topic for c in p.cameras)
    for c in p.cameras:                                       # an on-demand stream is live once its user starts it
        if c.on_demand:
            page.locator(f'[data-testid="camera"][data-topic="{c.topic}"] [data-testid="camera-toggle"]').click()
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


def test_optional_wrist_camera_missing_is_visible_and_live_when_present(view):
    """The reBot's boot starts no camera: its D405 colour and depth streams are optional profile cameras.
    On a wire without them the target still validates and both cameras show MISSING (console spec
    §2.3); on a wire with them the page shows both live, at their global names."""
    both = ["/camera/camera/color/image_raw", "/camera/camera/aligned_depth_to_color/image_raw"]
    srv, page = view(rebot_spec())
    assert page.locator('[data-testid="target"]').inner_text() == REBOT
    page.wait_for_function("[...document.querySelectorAll('[data-testid=camera]')].every(e => e.dataset.state === 'missing')")
    cams = page.locator('[data-testid="camera"]')
    assert cams.evaluate_all("els => els.map(e => e.dataset.topic)") == both
    assert all("optional camera" in t for t in page.locator('[data-testid="camera"] .ov-detail').all_inner_texts())
    assert not disabled(page, "arm_trajectory")
    srv, page = view(rebot_spec(include_optional=True))
    assert page.locator('[data-testid="target"]').inner_text() == REBOT
    page.wait_for_function("[...document.querySelectorAll('[data-testid=camera]')].every(e => e.dataset.state === 'live')")
    assert page.locator('[data-testid="camera"]').evaluate_all("els => els.map(e => e.dataset.topic)") == both
    assert {o["topic"] for o in srv.ops_of("subscribe")} >= set(both)
    assert unprompted(srv, "rebot_b601_rs") == []


def depth_card(page):
    return page.locator('[data-testid="camera"][data-kind="depth"]')


def card_pixels(page, points):
    """RGB of the depth card's canvas at (x, y) points."""
    return depth_card(page).locator("canvas").first.evaluate(
        "(c, pts) => pts.map(([x, y]) => Array.from(c.getContext('2d').getImageData(x, y, 1, 1).data).slice(0, 3))", points)


def test_depth_camera_card_is_a_false_colour_map_over_the_cameras_range(view):
    """The reBot's D405 depth stream (16UC1 millimetres) is its second camera card: drawn as the
    false-colour map scaled to the profile's 0.07-0.5 m, 0 black, with a legend that names the range."""
    from fake_rosbridge import depth_mm
    from robot_console.camera import depth_lut
    srv, page = view(rebot_spec(include_optional=True))
    page.wait_for_function("[...document.querySelectorAll('[data-testid=camera]')].every(e => e.dataset.state === 'live')")
    assert page.locator('[data-testid="camera"][data-kind="color"]').count() == 1
    card = depth_card(page)
    assert card.count() == 1 and card.get_attribute("data-topic") == "/camera/camera/aligned_depth_to_color/image_raw"
    assert "16UC1" in card.locator(".cam-meta").inner_text()
    w, h = 8, 6                                             # the fake wire's frame (fake_rosbridge.image_msg)
    mid = next(i for i in range(1, w * h) if depth_mm(i, w * h) == 285)
    lut = depth_lut()
    got = card_pixels(page, [[0, 0], [1, 0], [mid % w, mid // w], [w - 1, h - 1]])
    want = [[0, 0, 0], lut[0].tolist(), lut[128].tolist(), lut[255].tolist()]
    assert got == want, (got, want)                          # no depth is black; near, middle, far of the range
    legend = card.locator('[data-testid="depth-legend"]')
    assert legend.locator('[data-testid="depth-near"]').inner_text() == "0.07 m"
    assert legend.locator('[data-testid="depth-far"]').inner_text() == "0.50 m"
    assert "no depth" in legend.inner_text()
    ends = legend.locator('[data-testid="depth-bar"]').evaluate(
        "c => [0, 255].map((x) => Array.from(c.getContext('2d').getImageData(x, 0, 1, 1).data).slice(0, 3))")
    assert ends == [lut[0].tolist(), lut[255].tolist()]      # the legend is the map the image is drawn with
    assert page.locator('[data-testid="camera"][data-kind="color"] [data-testid="depth-legend"]').count() == 0
    srv.paused.add("/camera/camera/aligned_depth_to_color/image_raw")
    page.wait_for_function("document.querySelector('[data-kind=depth]').dataset.state === 'stale'", timeout=8000)
    assert page.locator('[data-testid="camera"][data-kind="color"]').get_attribute("data-state") == "live"
    assert unprompted(srv, "rebot_b601_rs") == []


def test_depth_camera_frames_in_another_encoding_are_unsupported(view):
    """The depth camera documents 16UC1 only: any other encoding shows UNSUPPORTED, never a colour map."""
    srv, page = view(rebot_spec(include_optional=True))
    srv.frame_encoding["/camera/camera/aligned_depth_to_color/image_raw"] = "mono8"
    page.wait_for_function("document.querySelector('[data-kind=depth]').dataset.state === 'unsupported'", timeout=8000)
    assert "UNSUPPORTED" in depth_card(page).locator('[data-testid="camera-state"]').inner_text()


X3 = "rosmaster_x3_plus"
X3_CAMS = ["/camera/rgb/image_raw", "/camera/depth/image_raw", "/camera/ir/image_raw", "/usb_cam/image_raw",
           "/usb_cam/image_raw/compressed"]
X3_ON_DEMAND = X3_CAMS[1:]


def cam_card(page, topic):
    return page.locator(f'[data-testid="camera"][data-topic="{topic}"]')


def cam_states(page):
    return page.locator('[data-testid="camera"]').evaluate_all("els => els.map(e => e.dataset.state)")


def start_streams(page, topics):
    for t in topics:
        cam_card(page, t).locator('[data-testid="camera-toggle"]').click()


def camera_subs(srv, topic):
    return [o for o in srv.ops_of("subscribe") if o["topic"] == topic]


def test_x3_shows_astra_colour_depth_infrared_and_arm_camera(view):
    """ROSMASTER X3 PLUS: the Astra Pro Plus' colour, depth and infrared streams and the arm's USB camera (raw and as
    JPEG) are five cards. The colour camera is live at once, capped at 2 frames/s; the others are off (a neutral
    state, no subscription) until the user starts them, then live at their caps, and off again when stopped.
    Every subscription asks rosbridge for queue_length 1, so no backlog of frames builds up on the server.
    The depth card is a false-colour map over 0.6-8 m, the infrared one grey with its note; each card names its
    CameraInfo topic. Showing them sends nothing but the start-up stop."""
    srv, page = view(wire_spec(load(X3), include_optional=True))
    assert page.locator('[data-testid="target"]').inner_text() == X3
    cams = page.locator('[data-testid="camera"]')
    assert cams.evaluate_all("els => els.map(e => [e.dataset.topic, e.dataset.kind])") == [
        [X3_CAMS[0], "color"], [X3_CAMS[1], "depth"], [X3_CAMS[2], "ir"], [X3_CAMS[3], "color"], [X3_CAMS[4], "color"]]
    page.wait_for_function(f"document.querySelector('[data-topic=\"{X3_CAMS[0]}\"]').dataset.state === 'live'", timeout=8000)
    page.wait_for_timeout(500)
    assert cam_states(page) == ["live", "off", "off", "off", "off"]
    assert [camera_subs(srv, t) for t in X3_ON_DEMAND] == [[], [], [], []]       # nothing is sent for an off stream
    assert [(o["throttle_rate"], o["queue_length"]) for o in camera_subs(srv, X3_CAMS[0])] == [(500, 1)]
    off = cam_card(page, X3_CAMS[1])
    assert "OFF" in off.locator('[data-testid="camera-state"]').inner_text()
    assert "only while you have it started" in off.locator(".ov-detail").inner_text()
    assert "dim" in off.locator('[data-testid="camera-state"]').get_attribute("class")      # not an alarm
    assert [cam_card(page, t).locator('[data-testid="camera-rate"]').inner_text() for t in X3_CAMS] == [
        "≤ 2 fps", "≤ 2 fps", "≤ 2 fps", "≤ 2 fps", "≤ 5 fps"]
    start_streams(page, X3_ON_DEMAND)
    page.wait_for_function("[...document.querySelectorAll('[data-testid=camera]')].every(e => e.dataset.state === 'live')", timeout=8000)
    assert [[(o["throttle_rate"], o["queue_length"]) for o in camera_subs(srv, t)] for t in X3_ON_DEMAND] == [[(500, 1)]] * 3 + [[(200, 1)]]
    assert {o["queue_length"] for o in srv.ops_of("subscribe")} == {1}                 # sensors too
    depth = cam_card(page, X3_CAMS[1]).locator('[data-testid="depth-legend"]')
    assert (depth.locator('[data-testid="depth-near"]').inner_text(), depth.locator('[data-testid="depth-far"]').inner_text()) == ("0.60 m", "8.00 m")
    ir = cam_card(page, X3_CAMS[2])
    assert "darkest" in ir.locator('[data-testid="ir-legend"]').inner_text() and "INFRARED" in ir.locator(".cam-kind").inner_text().upper()
    assert [cam_card(page, t).locator('[data-testid="camera-info"]').inner_text() for t in X3_CAMS[:4]] == [
        "info /camera/rgb/camera_info", "info /camera/depth/camera_info", "info /camera/ir/camera_info", "info /usb_cam/camera_info"]
    assert "mono16" in ir.locator(".cam-meta").inner_text()
    # the fake honours throttle_rate as rosbridge does: an on-demand stream arrives at most twice a second
    n0 = page.evaluate(f"RC.state.streams.find(s => s.topic === '{X3_CAMS[1]}').frames")
    page.wait_for_timeout(2000)
    assert page.evaluate(f"RC.state.streams.find(s => s.topic === '{X3_CAMS[1]}').frames") - n0 <= 5
    cam_card(page, X3_CAMS[1]).locator('[data-testid="camera-toggle"]').click()
    page.wait_for_function(f"document.querySelector('[data-topic=\"{X3_CAMS[1]}\"]').dataset.state === 'off'", timeout=3000)
    sid = camera_subs(srv, X3_CAMS[1])[0]["id"]
    assert sid in {o["id"] for o in srv.ops_of("unsubscribe")}
    assert cam_card(page, X3_CAMS[0]).get_attribute("data-state") == "live"
    only_startup(srv, page, X3)


def test_x3_full_size_colour_and_infrared_frames(view):
    """640x480 frames, the size the Astra and the USB camera send, are decoded at full size; an infrared frame is
    drawn as camera.py does (the vendor IR viewer's min-max normalisation)."""
    import base64
    from robot_console.camera import decode_image
    from fake_rosbridge import image_msg
    srv, page = view(wire_spec(load(X3), include_optional=True, camera_rate=4))
    srv.frame_size[X3_CAMS[0]] = (640, 480)
    page.wait_for_function(f"(() => {{ const s = RC.state.streams.find(s => s.topic === '{X3_CAMS[0]}'); return s && s.w === 640 && s.h === 480; }})()", timeout=10000)
    assert cam_card(page, X3_CAMS[0]).locator("canvas").evaluate("c => [c.width, c.height]") == [640, 480]
    want = decode_image(image_msg("ros1", "rgb8", w=640, h=480, seq=0), ["rgb8"])
    px = cam_card(page, X3_CAMS[0]).locator("canvas").evaluate("c => Array.from(c.getContext('2d').getImageData(0, 0, 1, 1).data).slice(0, 3)")
    assert len(px) == 3 and want.shape == (480, 640, 3)
    start_streams(page, [X3_CAMS[2]])
    page.wait_for_function(f"document.querySelector('[data-topic=\"{X3_CAMS[2]}\"]').dataset.state === 'live'", timeout=8000)
    srv.paused.add(X3_CAMS[2])
    w, h = 640, 480
    vals = [(1000 + 37 * i) % 65536 for i in range(w * h)]
    msg = {"header": {"stamp": {"secs": 1, "nsecs": 0}, "frame_id": "camera_color_optical_frame"}, "height": h, "width": w,
           "encoding": "mono16", "is_bigendian": 0, "step": 2 * w,
           "data": base64.b64encode(b"".join(v.to_bytes(2, "little") for v in vals)).decode()}
    page.wait_for_timeout(400)
    srv._broadcast(X3_CAMS[2], msg)
    want = decode_image(msg, ["mono16"], ir=True)
    pts = [[0, 0], [1, 0], [320, 240], [639, 479]]
    page.wait_for_function(
        "([t, v]) => { const c = document.querySelector(`[data-topic=\"${t}\"] canvas`); "
        "return c.width === 640 && c.getContext('2d').getImageData(320, 240, 1, 1).data[0] === v; }",
        arg=[X3_CAMS[2], int(want[240, 320, 0])], timeout=8000)
    got = cam_card(page, X3_CAMS[2]).locator("canvas").first.evaluate(
        "(c, pts) => pts.map(([x, y]) => Array.from(c.getContext('2d').getImageData(x, y, 1, 1).data).slice(0, 3))", pts)
    assert got == [want[y, x].tolist() for x, y in pts]


def test_x3_silent_infrared_is_explained_and_absent_arm_camera_names_its_launch(view, fake):
    """On the authoritative boot's wire (no USB camera) the target validates. An infrared stream started but
    never delivering (the Astra driver restarted without IR, keeping the topic advertised) is shown as NO FRAMES
    with that explanation, a neutral state, not a red STALE; the USB camera is MISSING and names the launch that
    starts it."""
    srv = fake(wire_spec(load(X3)))
    srv.paused.add(X3_CAMS[2])
    srv, page = view(srv=srv)
    assert page.locator('[data-testid="target"]').inner_text() == X3
    assert cam_card(page, X3_CAMS[3]).get_attribute("data-state") == "missing"
    assert cam_card(page, X3_CAMS[3]).locator('[data-testid="camera-toggle"]').count() == 0
    detail = cam_card(page, X3_CAMS[3]).locator(".ov-detail").inner_text()
    assert "optional camera is not on the wire" in detail and "laser_usb_bringup.launch" in detail
    start_streams(page, [X3_CAMS[2]])
    page.wait_for_function(f"document.querySelector('[data-topic=\"{X3_CAMS[2]}\"]').dataset.state === 'silent'", timeout=6000)
    ir = cam_card(page, X3_CAMS[2])
    assert "NO FRAMES" in ir.locator('[data-testid="camera-state"]').inner_text()
    assert "restarted without it" in ir.locator(".ov-detail").inner_text()
    assert "dim" in ir.locator('[data-testid="camera-state"]').get_attribute("class")
    only_startup(srv, page, X3)


def test_x3_depth_colours_follow_the_cameras_range(view):
    """The X3 depth stream (16UC1 mm, here 0 then 600 to 8000 mm across the frame) is drawn over the profile's
    0.6-8 m: no depth black, 0.6 m the near end of the map, the middle of the range its middle, 8 m its far end."""
    from fake_rosbridge import depth_mm
    from robot_console.camera import depth_lut
    srv, page = view(wire_spec(load(X3)))
    start_streams(page, [X3_CAMS[1]])
    page.wait_for_function(f"document.querySelector('[data-topic=\"{X3_CAMS[1]}\"]').dataset.state === 'live'", timeout=8000)
    w, h = 8, 6
    vals = [depth_mm(i, w * h, 600, 8000) for i in range(w * h)]
    assert vals[1] == 600 and vals[-1] == 8000
    mid = min(range(1, w * h), key=lambda i: abs(vals[i] - 4300))      # nearest pixel to the middle, 4.3 m
    lut = depth_lut()
    want_mid = lut[int((vals[mid] / 1000 - 0.6) / 7.4 * 255 + 0.5)].tolist()
    got = cam_card(page, X3_CAMS[1]).locator("canvas").first.evaluate(
        "(c, pts) => pts.map(([x, y]) => Array.from(c.getContext('2d').getImageData(x, y, 1, 1).data).slice(0, 3))",
        [[0, 0], [1, 0], [mid % w, mid // w], [w - 1, h - 1]])
    assert got == [[0, 0, 0], lut[0].tolist(), want_mid, lut[255].tolist()], got
    assert "registered to the colour image" in cam_card(page, X3_CAMS[1]).locator('[data-testid="camera-caption"]').inner_text().lower()


def cam_state_is(topic, state):
    return f"document.querySelector('[data-topic=\"{topic}\"]').dataset.state === '{state}'"


def watch_states(page, topic, seconds):
    """The camera's states, sampled every 50 ms for `seconds`."""
    return page.evaluate("""([t, ms]) => new Promise((done) => { const seen = []; const end = performance.now() + ms;
        const f = () => { seen.push(document.querySelector(`[data-topic="${t}"]`).dataset.state);
          if (performance.now() < end) setTimeout(f, 50); else done(seen); }; f(); })""", [topic, int(seconds * 1000)])


def test_lag_comes_from_the_robot_clock_through_rosapi(view):
    """Link health (console spec §2.3): the page asks /rosapi/get_time (read-only, never a command) and judges the shown
    image's age in robot time. A robot clock far off the console's is no lag; frames stamped 2.5 s before they arrive
    are LAGGING, not live (the link answering quickly, so no pause); kept at that steady offset they stay LAGGING,
    noted as a real delay or stamps on another clock; current again, live again."""
    srv, page = view(wire_spec(load(X3)))
    srv.clock_skew = 500.0                                  # the robot's clock runs 500 s ahead of the console's
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    assert re.search(r"age 0\.\d\d ± 0\.\d\d s · live to 1\.75 s", cam_card(page, X3_CAMS[0]).locator(".cam-meta").inner_text())
    assert any(o.get("service") == "/rosapi/get_time" for o in srv.ops_of("call_service"))
    srv.stamp_lag[X3_CAMS[0]] = 2.5                         # frames now arrive 2.5 s after they were taken
    page.wait_for_function(cam_state_is(X3_CAMS[0], "lagging"), timeout=8000)
    card = cam_card(page, X3_CAMS[0])
    assert "LAGGING" in card.locator('[data-testid="camera-state"]').inner_text() and "not current" in card.locator(".ov-detail").inner_text()
    assert page.locator('[data-testid="congestion"]').is_hidden()
    page.wait_for_function("document.querySelector('[data-topic=\"/camera/rgb/image_raw\"] .ov-detail').textContent.includes('steady offset')", timeout=8000)
    assert card.get_attribute("data-state") == "lagging" and "+2.5 s" in card.locator(".ov-detail").inner_text()
    assert "another clock" in card.locator(".ov-detail").inner_text() and page.locator('[data-testid="congestion"]').is_hidden()
    srv.stamp_lag[X3_CAMS[0]] = 0.0
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    only_startup(srv, page, X3)


def watch_ages(page, topic, seconds):
    """[state, age_lo, age_hi] of the camera every 50 ms for `seconds` (ages null when not judged)."""
    return page.evaluate("""([t, ms]) => new Promise((done) => { const out = []; const end = performance.now() + ms;
        const f = () => { const s = RC.state.streams.find((x) => x.topic === t), l = s.lagNow;
          out.push([s.state, l && l[0] === "known" ? l[3] : null, l && l[0] === "known" ? l[4] : null]);
          if (performance.now() < end) setTimeout(f, 50); else done(out); }; f(); })""", [topic, int(seconds * 1000)])


def test_an_image_stops_being_live_past_its_bound(view):
    """The critic's case: frames arrive 0.9 s after they were taken (live), then the stream stops. The image on screen
    is live only while its age is within its bound (1 s + 1.5 frame periods = 1.75 s at 2 frames/s), judged on every
    refresh, then stale."""
    srv, page = view(wire_spec(load(X3)))
    srv.stamp_lag[X3_CAMS[0]] = 0.9
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    srv.paused.add(X3_CAMS[0])
    seen = watch_ages(page, X3_CAMS[0], 2.0)
    live = [hi for st, lo, hi in seen if st == "live"]
    assert live and max(live) <= 1.75 + 0.06, max(live)
    assert seen[-1][0] == "stale", seen[-5:]


@pytest.mark.parametrize("lag", [0.3, 0.45, 0.6, 0.9])
def test_a_steady_lag_below_the_limit_never_flickers_on_the_page(view, lag):
    """At the X3's 2 frames/s, a steady transport lag of 0.3, 0.45, 0.6 or 0.9 s reads LIVE throughout: never STALE
    between frames."""
    srv, page = view(wire_spec(load(X3)))
    srv.stamp_lag[X3_CAMS[0]] = lag
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    seen = watch_states(page, X3_CAMS[0], 3.0)
    assert set(seen) == {"live"}, seen


def test_a_stopped_stream_shows_no_current_age(view):
    srv, page = view(wire_spec(load(X3)))
    start_streams(page, [X3_CAMS[1]])
    page.wait_for_function(cam_state_is(X3_CAMS[1], "live"), timeout=8000)
    assert "age " in cam_card(page, X3_CAMS[1]).locator(".cam-meta").inner_text()
    start_streams(page, [X3_CAMS[1]])                       # stop it
    page.wait_for_function(cam_state_is(X3_CAMS[1], "off"), timeout=3000)
    page.wait_for_timeout(300)
    meta = cam_card(page, X3_CAMS[1]).locator(".cam-meta").inner_text()
    assert "age" not in meta and "last frame" in meta


def congested(page):
    page.wait_for_function("!document.querySelector('[data-testid=congestion]').hidden", timeout=10000)
    return page.locator('[data-testid="congestion-text"]').inner_text()


def test_a_congested_link_pauses_the_cameras_until_resumed(view):
    """Late frames while the robot's time answers take over 1 s (the link backing up): every camera stream is
    unsubscribed, the page says so prominently, as a warning; the operator resumes them."""
    srv, page = view(wire_spec(load(X3)))
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    sub = camera_subs(srv, X3_CAMS[0])[0]
    srv.time_reply_delay = 1.3
    srv.stamp_lag[X3_CAMS[0]] = 2.5
    text = congested(page)
    assert "Cameras paused" in text and "congested" in text
    drive = page.locator('[data-testid="congestion-drive"]')        # repeated where the operator drives
    assert drive.is_visible() and "Cameras paused" in drive.inner_text()
    assert srv.wait_for(lambda: sub["id"] in {o["id"] for o in srv.ops_of("unsubscribe")}, 3.0)
    card = cam_card(page, X3_CAMS[0])
    page.wait_for_function(cam_state_is(X3_CAMS[0], "paused"), timeout=3000)
    assert "warn" in card.locator('[data-testid="camera-state"]').get_attribute("class")
    srv.time_reply_delay = 0.0
    srv.stamp_lag[X3_CAMS[0]] = 0.0
    page.wait_for_timeout(1500)
    page.locator('[data-testid="resume-cameras"]').click()
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    assert len(camera_subs(srv, X3_CAMS[0])) == 2 and page.locator('[data-testid="congestion"]').is_hidden()
    only_startup(srv, page, X3)


def test_an_operator_sample_does_not_pause_the_cameras(view):
    """A point-cloud sample's own burst delays the time answers: questions sent while the sample is expected on the
    link (a bounded grace window) are not counted, so the operator's click does not pause the cameras; a slow link
    after that window still does."""
    srv, page = view(wire_spec(load(X3)))
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    srv.time_reply_delay = 1.4                              # the sample's burst holds the time answers back
    sensor(page, "depth_points").locator('[data-testid="sensor-sample"]').click()
    page.wait_for_function("document.querySelector('[data-sensor=depth_points]').dataset.state === 'sampled'", timeout=8000)
    grace = page.evaluate("RC.clock().grace")
    assert 1.0 <= grace[1] - grace[0] <= 5.0
    page.wait_for_function(f"(performance.timeOrigin + performance.now()) / 1000 > {grace[1]}", timeout=8000)
    srv.time_reply_delay = 0.0
    page.wait_for_timeout(2500)
    assert page.locator('[data-testid="congestion"]').is_hidden()
    srv.time_reply_delay = 1.4                              # a slow link after the window: congestion
    assert "took 1.4 s" in congested(page)


def test_time_unanswered_while_frames_arrive_is_congestion(view):
    """The worst backlog: the time answer never comes within 3 s while frames keep arriving. That is congestion (a
    warning and a pause), never a neutral LAG UNKNOWN."""
    srv, page = view(wire_spec(load(X3)))
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    srv.time_reply_delay = 6.0
    assert "late while frames keep arriving" in congested(page)
    page.wait_for_function(cam_state_is(X3_CAMS[0], "paused"), timeout=3000)
    assert "warn" in cam_card(page, X3_CAMS[0]).locator('[data-testid="camera-state"]').get_attribute("class")


def test_a_backlog_present_before_a_stream_starts_is_never_live(view):
    """No baseline is learnt from the frames: an on-demand stream started while its frames are already 2.5 s late
    (with the link backed up) is never live; the cameras are paused."""
    srv, page = view(wire_spec(load(X3)))
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    srv.stamp_lag[X3_CAMS[1]] = 2.5
    srv.time_reply_delay = 1.3
    start_streams(page, [X3_CAMS[1]])
    seen = watch_states(page, X3_CAMS[1], 5.0)
    assert "live" not in seen and seen[-1] == "paused", seen


@pytest.mark.parametrize("step", [-5.0, 5.0])
def test_a_robot_clock_step_is_never_read_as_lag(view, step):
    """The robot's clock steps back or forward 5 s (stamps and its time alike): the page re-asks its time and sees the
    step; the stream is never LAGGING. Forward, the age bounds only widen downwards: it stays LIVE. Back, the readings
    of the last 10 s disagree by 5 s, so a new image may be up to 5 s old as far as the page can tell: LAG UNKNOWN,
    saying so, until the readings before the step leave the window, then LIVE again."""
    srv, page = view(wire_spec(load(X3)))
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    page.wait_for_timeout(500)
    srv.clock_skew += step
    seen = watch_states(page, X3_CAMS[0], 4.0)
    assert "lagging" not in seen, seen
    assert page.evaluate("RC.clock().steps") >= 1
    if step > 0:
        assert seen[-1] == "live", seen
        return
    assert seen[-1] == "unknown" and "disagree by 5.0 s" in cam_card(page, X3_CAMS[0]).inner_text(), seen
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=10000)


@pytest.mark.parametrize("why", ["no_time_service", "sim_time"])
def test_without_a_clock_reference_lag_is_unknown_never_zero(view, fake, why):
    """With no /rosapi/get_time, or a ROS 1 robot on simulated time, frames that arrive are LAG UNKNOWN (a neutral
    state) with the reason, never LIVE with a lag of zero."""
    srv = fake(wire_spec(load(X3)))
    if why == "no_time_service":
        srv.time_service = False
    else:
        srv.params["/use_sim_time"] = True
    srv, page = view(srv=srv)
    page.wait_for_function(cam_state_is(X3_CAMS[0], "unknown"), timeout=8000)
    seen = watch_states(page, X3_CAMS[0], 2.0)
    assert set(seen) == {"unknown"}, seen
    card = cam_card(page, X3_CAMS[0])
    assert "LAG UNKNOWN" in card.locator('[data-testid="camera-state"]').inner_text()
    assert "age unknown" in card.locator(".cam-meta").inner_text() and "age 0" not in card.locator(".cam-meta").inner_text()
    reason = "/rosapi/get_time" if why == "no_time_service" else "simulated time"
    assert reason in card.locator(".ov-detail").inner_text()
    assert "warn" in card.locator('[data-testid="camera-state"]').get_attribute("class")      # a caution, not neutral
    assert page.locator('[data-testid="congestion"]').is_hidden()


def test_the_stop_never_waits_behind_full_size_frames(view, monkeypatch):
    """With every X3 stream sending full-size 640x480 frames (the link budget lifted for this stress), releasing the
    motion key delivers the stop at once: frames are only stored when they arrive and decoded later, so input is
    handled between them."""
    serve_variants(monkeypatch, dataclasses.replace(load(X3), link_budget_mbps=None))
    srv, page = view(wire_spec(load(X3), include_optional=True, camera_rate=30))
    for t in X3_CAMS:
        srv.frame_size[t] = (640, 480)
    start_streams(page, X3_ON_DEMAND)
    page.wait_for_function("[...document.querySelectorAll('[data-testid=camera]')].every(e => e.dataset.state === 'live')", timeout=15000)
    started(page)
    p = load(X3)
    worst = 0.0
    for _ in range(3):
        m = len(motion_cmds(p, srv.command_ops()))
        page.keyboard.down("w")
        assert srv.wait_for(lambda: len(motion_cmds(p, srv.command_ops())) > m, 3.0)
        n = len(stops(srv, p))
        t0 = time.monotonic()
        page.keyboard.up("w")
        assert srv.wait_for(lambda: len(stops(srv, p)) > n, 3.0)
        worst = max(worst, stops(srv, p)[-1]["_t"] - t0)
        page.wait_for_timeout(300)
    assert worst < 0.25, worst


def test_x3_link_budget_refuses_a_stream_beyond_it(view, fake):
    """The X3's camera streams may take 40 Mbit/s together: with colour (about 20 Mbit/s at 640x480, 2 frames/s) and
    depth (about 13) running, starting infrared (about 13) is refused with the numbers and nothing is subscribed;
    after depth is stopped, infrared starts. The page shows the link's measured load and the cameras' share."""
    srv = fake(wire_spec(load(X3), include_optional=True))
    for t in X3_CAMS:
        srv.frame_size[t] = (640, 480)
    srv, page = view(srv=srv)
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    start_streams(page, [X3_CAMS[1]])
    page.wait_for_function(cam_state_is(X3_CAMS[1], "live"), timeout=8000)
    start_streams(page, [X3_CAMS[2]])
    page.wait_for_timeout(400)
    ir = cam_card(page, X3_CAMS[2])
    assert ir.get_attribute("data-state") == "off" and camera_subs(srv, X3_CAMS[2]) == []
    detail = ir.locator(".ov-detail").inner_text()
    assert "link budget of 40 Mbit/s" in detail and "Stop another stream first" in detail, detail
    rate = page.locator('[data-testid="link-rate"]')
    page.wait_for_function("parseFloat(document.querySelector('[data-testid=link-rate]').dataset.mbps) > 10", timeout=6000)
    assert "of 40 Mbit/s budget" in rate.inner_text()
    start_streams(page, [X3_CAMS[1]])                     # stop depth
    page.wait_for_function(cam_state_is(X3_CAMS[1], "off"), timeout=3000)
    start_streams(page, [X3_CAMS[2]])
    page.wait_for_function(cam_state_is(X3_CAMS[2], "live"), timeout=8000)
    assert len(camera_subs(srv, X3_CAMS[2])) == 1


def test_camera_cards_do_not_overflow_or_overlap(view, fake):
    """At 1500 px with all X3 streams started (budget lifted), no text of a camera card overflows the card, and the
    meta line overlaps neither the caption nor the legend."""
    srv = fake(wire_spec(load(X3), include_optional=True))
    for t in X3_CAMS:
        srv.frame_size[t] = (640, 480)
    srv, page = view(srv=srv, viewport=(1500, 1000))
    page.evaluate("RC.state.target.profile.link_budget_mbps = null")
    start_streams(page, X3_ON_DEMAND)
    page.wait_for_function("[...document.querySelectorAll('[data-testid=camera]')].every(e => ['live','lagging','unknown'].includes(e.dataset.state))", timeout=15000)
    page.wait_for_timeout(600)
    bad = page.evaluate("""() => { const out = [];
      for (const card of document.querySelectorAll('[data-testid=camera]')) {
        const cb = card.getBoundingClientRect();
        const parts = [...card.querySelectorAll('.cam-caption > *, .cam-caption-note, .depth-legend, .ir-legend')];
        for (const e of parts) { const r = e.getBoundingClientRect();
          if (r.width && (r.right > cb.right + 1 || r.left < cb.left - 1)) out.push(['overflow', card.dataset.topic, e.className]); }
        const meta = card.querySelector('.cam-meta').getBoundingClientRect();
        for (const e of card.querySelectorAll('.cam-caption-note, .depth-legend, .cam-caption code')) { const r = e.getBoundingClientRect();
          const hit = !(r.right <= meta.left || r.left >= meta.right || r.bottom <= meta.top || r.top >= meta.bottom);
          if (hit && r.width && meta.width) out.push(['overlap', card.dataset.topic, e.className || e.tagName]); }
      } return out; }""")
    assert bad == [], bad


def test_x3_usb_camera_as_jpeg_is_decoded_and_cheap_on_the_budget(view, fake):
    """The arm's USB camera through image_transport's compressed plugin (sensor_msgs/CompressedImage, JPEG): the
    browser decodes it, and it counts for a tenth of the raw stream against the link budget, so it starts beside
    colour and depth where the raw USB stream would not; a frame in another format is unsupported."""
    srv = fake(wire_spec(load(X3), include_optional=True))
    for t in X3_CAMS[:4]:
        srv.frame_size[t] = (640, 480)
    srv, page = view(srv=srv)
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    start_streams(page, [X3_CAMS[1]])
    page.wait_for_function(cam_state_is(X3_CAMS[1], "live"), timeout=8000)
    start_streams(page, [X3_CAMS[3]])                       # raw USB: over the budget
    page.wait_for_timeout(300)
    assert cam_card(page, X3_CAMS[3]).get_attribute("data-state") == "off" and camera_subs(srv, X3_CAMS[3]) == []
    jpeg = cam_card(page, X3_CAMS[4])
    assert jpeg.get_attribute("data-compressed") == "true" and "COMPRESSED" in jpeg.inner_text().upper()
    start_streams(page, [X3_CAMS[4]])
    page.wait_for_function(cam_state_is(X3_CAMS[4], "live"), timeout=8000)
    sub = camera_subs(srv, X3_CAMS[4])[0]
    assert (sub["type"], sub["throttle_rate"], sub["queue_length"]) == ("sensor_msgs/CompressedImage", 200, 1)
    px = jpeg.locator("canvas").first.evaluate("c => [c.width, c.height, ...Array.from(c.getContext('2d').getImageData(c.width - 1, 0, 1, 1).data).slice(0, 3)]")
    assert px[:2] == [8, 6] and px[2] > 180 and px[3] < 60      # the fake's red ramp, decoded
    assert "jpeg" in jpeg.locator(".cam-meta").inner_text()
    srv.frame_encoding[X3_CAMS[4]] = "rgb8; theora"
    page.wait_for_function(cam_state_is(X3_CAMS[4], "unsupported"), timeout=8000)


def test_a_point_cloud_sample_counts_against_the_link_budget(view, fake):
    """A cloud is megabytes in one burst: with colour and depth taking about 33 of the X3's 40 Mbit/s it is refused,
    saying why; with only colour (about 20, half the budget) it is sampled. The budget line warns near the budget."""
    srv = fake(wire_spec(load(X3)))
    for t in X3_CAMS[:2]:
        srv.frame_size[t] = (640, 480)
    srv, page = view(srv=srv)
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    start_streams(page, [X3_CAMS[1]])
    page.wait_for_function(cam_state_is(X3_CAMS[1], "live"), timeout=8000)
    page.wait_for_function("document.querySelector('[data-testid=link-rate]').dataset.level === 'near'", timeout=6000)
    sensor(page, "depth_points").locator('[data-testid="sensor-sample"]').click()
    page.wait_for_timeout(300)
    assert [o for o in srv.ops_of("subscribe") if o["topic"] == "/camera/depth/points"] == []
    meta = sensor(page, "depth_points").locator('[data-testid="sensor-meta"]').inner_text()
    assert "Not sampled" in meta and "MB in one burst" in meta and "Stop the depth" in meta
    start_streams(page, [X3_CAMS[1]])                       # stop depth
    page.wait_for_function(cam_state_is(X3_CAMS[1], "off"), timeout=3000)
    sensor(page, "depth_points").locator('[data-testid="sensor-sample"]').click()
    page.wait_for_function("document.querySelector('[data-sensor=depth_points]').dataset.state === 'sampled'", timeout=8000)


def test_use_sim_time_unknown_until_rosapi_answers(view, fake):
    """On ROS 1 the page reads /use_sim_time before trusting the robot's time; while rosapi does not answer that, the
    age is unknown (never live) and it asks again until it knows."""
    srv = fake(wire_spec(load(X3)))
    srv.param_failures = 2
    srv, page = view(srv=srv)
    page.wait_for_function(cam_state_is(X3_CAMS[0], "unknown"), timeout=8000)
    assert "use_sim_time" in cam_card(page, X3_CAMS[0]).locator(".ov-detail").inner_text()
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=12000)
    assert len([o for o in srv.ops_of("call_service") if o.get("service") == "/rosapi/get_param"]) >= 3


def test_non_finite_numbers_are_read_as_null_outside_strings_only(view):
    """Python's json writes NaN/Infinity bare: the page reads those tokens as null, and leaves strings untouched."""
    srv, page = view(wire_spec(load(X3)))
    text = '{"r":[1,NaN,-Infinity,Infinity,2.5e3],"s":"NaN \\"Infinity\\" -Infinity","t":"x"}'
    got = page.evaluate("t => RC.nonFiniteToNull(t)", text)
    assert json.loads(got) == {"r": [1, None, None, None, 2500.0], "s": 'NaN "Infinity" -Infinity', "t": "x"}


def test_untied_streams_are_previewed_on_request_one_at_a_time(view):
    """Before a target is validated (two candidates on the wire) every raw image stream is listed, untied, but none is
    streamed by itself (no profile says what it costs the link): a preview starts on request, at most 2 frames/s, and
    starting another stops the first."""
    srv, page = view(merge(wire_spec(load("myagv")), wire_spec(load(X3))))
    page.wait_for_function("document.querySelectorAll('[data-testid=untied]').length > 0", timeout=8000)
    page.wait_for_timeout(500)
    raw = [t for t in X3_CAMS if not t.endswith("/compressed")]
    assert set(raw) <= set(page.locator('[data-testid="camera"]').evaluate_all("els => els.map(e => e.dataset.topic)"))
    assert not {o["topic"] for o in srv.ops_of("subscribe")} & set(X3_CAMS)
    assert set(cam_states(page)) == {"off"} and "one at a time" in cam_card(page, raw[0]).locator(".ov-detail").inner_text()
    start_streams(page, [raw[0]])
    page.wait_for_function(cam_state_is(raw[0], "unknown") + " || " + cam_state_is(raw[0], "live"), timeout=8000)
    first = camera_subs(srv, raw[0])
    assert [(o["throttle_rate"], o["queue_length"]) for o in first] == [(500, 1)]
    start_streams(page, [raw[1]])
    assert srv.wait_for(lambda: first[0]["id"] in {o["id"] for o in srv.ops_of("unsubscribe")}, 3.0)
    page.wait_for_function(cam_state_is(raw[0], "off"), timeout=3000)
    assert len(camera_subs(srv, raw[1])) == 1
    assert srv.command_ops() == []


def test_no_cameras_is_a_visible_state(view, monkeypatch):
    """A validated target without cameras shows that state, and its controls (console spec §2.3).
    No packaged profile lacks a camera, so the page is handed the reBot profile without its camera."""
    bare = dataclasses.replace(load("rebot_b601_rs"), cameras=())
    serve_variants(monkeypatch, bare)
    srv, page = view(rebot_spec(profile=bare))
    assert page.locator('[data-testid="target"]').inner_text() == REBOT
    page.wait_for_function("getComputedStyle(document.querySelector('[data-testid=no-cameras]')).display !== 'none'")
    assert "No cameras" in page.locator('[data-testid="no-cameras"]').inner_text()
    assert page.locator('[data-testid="camera"]').count() == 0
    assert page.locator('[data-testid="control"]').count() == len(load("rebot_b601_rs").controls)
    assert not disabled(page, "arm_trajectory")
    assert srv.command_ops() == []


# ------------------------------------------------------------------ sensor panel (console spec §2.5)

SENSOR_TOPICS = [s.topic for s in load("rebot_b601_rs").sensors]


def sensor(page, sid):
    return page.locator(f'[data-testid="sensor"][data-sensor="{sid}"]')


def sensor_state(page, sid):
    return sensor(page, sid).get_attribute("data-state")


def sval(page, sid, field):
    """The raw value the sensor panel shows for a field (data-value), as text."""
    return sensor(page, sid).locator(f'[data-field="{field}"] [data-testid="sensor-value"]').get_attribute("data-value")


def sbit(page, sid, bit):
    return sensor(page, sid).locator(f'[data-testid="sensor-bit"][data-bit="{bit}"]').get_attribute("data-set")


def wait_sensors(page, state="live", sid=None, timeout=8000):
    sel = f'[data-testid="sensor"]' + (f'[data-sensor="{sid}"]' if sid else "")
    page.wait_for_function(
        "([sel, st]) => { const e = [...document.querySelectorAll(sel)]; "
        "return e.length > 0 && e.every((x) => x.dataset.state === st); }", arg=[sel, state], timeout=timeout)


def test_sensor_panel_shows_the_profile_sensors_read_only(view):
    """The reBot's page lists each of its profile sensors under its resolved topic; every one reports
    (a latched topic as its last value, a periodic one live), and the page only subscribes to them:
    nothing is published, advertised or called for a sensor."""
    srv, page = view(rebot_spec())
    assert page.locator('[data-testid="target"]').inner_text() == REBOT
    panel = page.locator('[data-testid="sensors"]')
    assert panel.is_visible() and "sensors" in panel.inner_text().lower()
    ids = [s.id for s in load("rebot_b601_rs").sensors]
    assert page.locator('[data-testid="sensor"]').evaluate_all("els => els.map(e => e.dataset.sensor)") == ids
    assert page.locator('[data-testid="sensor"]').evaluate_all("els => els.map(e => e.dataset.topic)") == \
        [f"/rebotarm{t}" for t in SENSOR_TOPICS]
    wait_sensors(page, "live", sid="joint_states")
    page.wait_for_function("[...document.querySelectorAll('[data-testid=sensor]')].every((e) => ['live','latched'].includes(e.dataset.state))")
    assert sensor_state(page, "arm_status") == "latched"
    assert page.locator('[data-testid="sensor-summary"]').inner_text() == f"{len(ids)} of {len(ids)} reporting"
    subscribed = {o["topic"] for o in srv.ops_of("subscribe")}
    assert {f"/rebotarm{t}" for t in SENSOR_TOPICS} <= subscribed
    assert unprompted(srv, "rebot_b601_rs") == []                       # read-only: nothing was sent
    assert srv.ops_of("advertise") == [] and srv.ops_of("publish") == []


def test_sensor_values_fields_and_status_bits(view):
    """Each shown field is the message's own value: a joint's position, velocity and torque, the
    arrays of joint_states by joint, the documented bits of status_code set or clear, and the
    arm status' booleans and lists."""
    srv, page = view(rebot_spec())
    wait_sensors(page, "live", sid="joint3_state")
    assert sval(page, "joint3_state", "joint_name") == "joint3"
    assert [sval(page, "joint3_state", f) for f in ("position", "velocity", "torque")] == ["0.75", "-0.375", "4.5"]
    shown = sensor(page, "joint3_state").locator('[data-field="position"] .sf-value').inner_text()
    assert shown == "0.750"
    assert sensor(page, "joint3_state").locator('[data-field="velocity"] .sf-value').inner_text() == "−0.375"
    assert sval(page, "joint3_state", "status_code") == "5"
    assert [sbit(page, "joint3_state", b) for b in range(6)] == ["true", "false", "true", "false", "false", "false"]
    assert sensor(page, "joint3_state").locator('[data-testid="sensor-bit"][data-set="true"]').evaluate_all(
        "els => els.map(e => e.textContent)") == ["undervoltage", "overtemperature"]
    assert sval(page, "gripper_state", "joint_name") == "gripper"
    wait_sensors(page, "live", sid="joint_states")
    assert sval(page, "joint_states", "joint1_position") == "0.1"
    assert sval(page, "joint_states", "joint6_effort") == "3"
    assert sval(page, "joint_states", "gripper_joint2_velocity") == "-0.08"
    assert sensor(page, "joint_states").locator('[data-group="gripper_joint1"] [data-testid="sensor-field"]').count() == 3
    page.wait_for_function("document.querySelector('[data-sensor=arm_status]').dataset.state === 'latched'")
    assert [sval(page, "arm_status", f) for f in ("mode", "enabled", "control_loop_active", "state_machine")] == \
        ["mit", "true", "true", "IDLE"]
    assert sval(page, "arm_status", "joint_names") == "joint1,joint2,joint3,joint4,joint5,joint6"
    assert sensor(page, "arm_status").locator('[data-field="error_codes"] .sf-value').inner_text() == "(none)"
    assert sensor(page, "arm_status").locator('[data-field="per_joint_status_code"] .sf-value').inner_text() == "0, 0, 0, 0, 0, 0"


def test_sensor_values_follow_new_messages(view):
    """A new message changes the shown values; a status word without flags clears every bit; a
    latched arm status republished (a service call, a trajectory goal) changes at once; a byte array
    that rosbridge sends as base64 shows its bytes."""
    srv, page = view(rebot_spec())
    wait_sensors(page, "live", sid="joint1_state")
    page.wait_for_function("document.querySelector('[data-sensor=arm_status]').dataset.state === 'latched'")
    from wirespec import sensor_sample
    msg = dict(sensor_sample("/joints/joint1/state", "rebotarm_msgs/msg/JointMotorState"),
               position=-1.25, torque=0.0, status_code=0b110010)
    srv.set_sample("/rebotarm/joints/joint1/state", msg)
    page.wait_for_function("document.querySelector('[data-sensor=joint1_state] [data-field=position] [data-testid=sensor-value]').dataset.value === '-1.25'")
    assert [sbit(page, "joint1_state", b) for b in range(6)] == ["false", "true", "false", "false", "true", "true"]
    srv.set_sample("/rebotarm/joints/joint1/state", dict(msg, status_code=0))
    page.wait_for_function("document.querySelector('[data-sensor=joint1_state] [data-field=status_code] [data-testid=sensor-value]').dataset.value === '0'")
    assert {sbit(page, "joint1_state", b) for b in range(6)} == {"false"}
    arm = sensor_sample("/arm_status", "rebotarm_msgs/msg/ArmStatus")
    srv.set_sample("/rebotarm/arm_status", dict(arm, state_machine="TRAJ_RUNNING", enabled=False,
                                                per_joint_status_code="AAECAwQF", error_codes=["E1", "E2"]))
    page.wait_for_function("document.querySelector('[data-sensor=arm_status] [data-field=state_machine] [data-testid=sensor-value]').dataset.value === 'TRAJ_RUNNING'")
    assert sval(page, "arm_status", "enabled") == "false"
    assert sensor(page, "arm_status").locator('[data-field="per_joint_status_code"] .sf-value').inner_text() == "0, 1, 2, 3, 4, 5"
    assert sensor(page, "arm_status").locator('[data-field="error_codes"] .sf-value').inner_text() == "E1, E2"
    assert sensor_state(page, "arm_status") == "latched"


def test_sensor_without_message_is_waiting_and_a_silent_one_goes_stale(view):
    """No message yet is its own state (a latched topic never delivered stays so, not stale); a periodic
    sensor that stops is stale after its threshold with its values marked frozen, and live again when
    it resumes. The others are unaffected."""
    srv, page = view(rebot_spec())
    wait_sensors(page, "live", sid="joint2_state")
    srv.paused.add("/rebotarm/joints/joint2/state")
    page.wait_for_function("document.querySelector('[data-sensor=joint2_state]').dataset.state === 'stale'", timeout=6000)
    assert "STALE" in sensor(page, "joint2_state").locator('[data-testid="sensor-state"]').inner_text()
    assert "frozen" in sensor(page, "joint2_state").locator('[data-testid="sensor-meta"]').inner_text()
    assert sensor_state(page, "joint1_state") == "live" and sensor_state(page, "arm_status") == "latched"
    srv.paused.discard("/rebotarm/joints/joint2/state")
    page.wait_for_function("document.querySelector('[data-sensor=joint2_state]').dataset.state === 'live'", timeout=6000)


def test_sensors_that_never_published_are_waiting(view, fake):
    srv = fake(rebot_spec())
    srv.paused |= {"/rebotarm/joints/joint5/state", "/rebotarm/arm_status"}
    srv, page = view(srv=srv)
    wait_sensors(page, "live", sid="joint4_state")
    assert sensor_state(page, "joint5_state") == "waiting" and sensor_state(page, "arm_status") == "waiting"
    assert "NO MESSAGE YET" in sensor(page, "joint5_state").locator('[data-testid="sensor-state"]').inner_text()
    assert "no message received yet" in sensor(page, "arm_status").locator('[data-testid="sensor-meta"]').inner_text()
    page.wait_for_timeout(1600)                       # longer than a periodic threshold: waiting, never stale
    assert sensor_state(page, "joint5_state") == "waiting"
    assert sval(page, "joint5_state", "position") == ""
    assert sensor(page, "joint5_state").locator('[data-field="position"] .sf-value').inner_text() == "—"
    assert page.locator('[data-testid="sensor-summary"]').inner_text() == f"{len(SENSOR_TOPICS) - 2} of {len(SENSOR_TOPICS)} reporting"


def test_sensors_show_disconnected_after_connection_loss(view):
    srv, page = view(rebot_spec())
    wait_sensors(page, "live", sid="joint_states")
    srv.drop_all()
    page.wait_for_function("[...document.querySelectorAll('[data-testid=sensor]')].every((e) => e.dataset.state === 'lost')", timeout=6000)
    assert "DISCONNECTED" in sensor(page, "joint1_state").locator('[data-testid="sensor-state"]').inner_text()


def test_profiles_without_sensors_show_no_sensor_panel(view):
    srv, page = view(wire_spec(load("myagv")))
    assert page.locator('[data-testid="target"]').inner_text() == "myagv"
    assert page.locator('[data-testid="sensors"]').count() == 1 and not page.locator('[data-testid="sensors"]').is_visible()
    assert page.locator('[data-testid="sensor"]').count() == 0


def x3_sensors(view, **kw):
    srv, page = view(wire_spec(load(X3), **kw))
    assert page.locator('[data-testid="target"]').inner_text() == X3
    return srv, page


X3_CLOUDS = ("/camera/depth/points", "/camera/depth_registered/points")


def test_x3_sensor_panel_shows_every_vendor_sensor_read_only(view):
    """ROSMASTER X3 PLUS: every vendor sensor topic of the profile in profile order, each periodic one reporting, the
    battery voltage also as the headline above the cameras. The arm echo and gamepad state are ordinary event
    publishers (no latching): they wait in a neutral EVENT-ONLY state, counted apart, and show their last event
    once one comes. The point clouds, which the Astra driver computes only while they have a subscriber, are never
    subscribed on load. The page only subscribes, and sends nothing but base motion's start-up stop."""
    srv, page = x3_sensors(view)
    ids = [s.id for s in load(X3).sensors]
    assert page.locator('[data-testid="sensor"]').evaluate_all("els => els.map(e => e.dataset.sensor)") == ids
    clouds = ("depth_points", "registered_points")
    page.wait_for_function("[...document.querySelectorAll('[data-testid=sensor]')].every((e) => "
                           "['live','event','off'].includes(e.dataset.state))", timeout=15000)
    assert sensor_state(page, "arm_angle_update") == "event" and sensor_state(page, "joy_state") == "event"
    assert "EVENT-ONLY" in sensor(page, "joy_state").locator('[data-testid="sensor-state"]').inner_text()
    assert "(normal)" in sensor(page, "joy_state").locator('[data-testid="sensor-meta"]').inner_text()
    assert [sensor_state(page, c) for c in clouds] == ["off", "off"]
    assert page.locator('[data-testid="sensor-summary"]').inner_text() == "11 of 11 reporting · 2 event-only (0 seen) · 2 on demand"
    srv.emit("/JoyState")
    page.wait_for_function("document.querySelector('[data-sensor=joy_state]').dataset.state === 'lastevent'", timeout=5000)
    assert "LAST EVENT" in sensor(page, "joy_state").locator('[data-testid="sensor-state"]').inner_text()
    assert sval(page, "joy_state", "active") == "true"
    assert page.locator('[data-testid="sensor-summary"]').inner_text() == "11 of 11 reporting · 2 event-only (1 seen) · 2 on demand"
    assert "NOT SAMPLED" in sensor(page, clouds[0]).locator('[data-testid="sensor-state"]').inner_text()
    head = page.locator('[data-testid="headline"]')
    assert head.count() == 1 and head.get_attribute("data-sensor") == "battery"
    assert head.locator('[data-testid="headline-value"]').inner_text() == "12.3" and "V" in head.inner_text()
    assert head.get_attribute("data-state") == "live" and head.get_attribute("data-warn") == ""
    assert page.evaluate("document.querySelector('[data-testid=headlines]').compareDocumentPosition("
                         "document.querySelector('[data-testid=cameras]')) & Node.DOCUMENT_POSITION_FOLLOWING")
    page.wait_for_timeout(500)
    subs = {o["topic"]: o for o in srv.ops_of("subscribe")}
    assert {s.topic for s in load(X3).sensors if not s.on_demand} <= set(subs)
    assert not set(X3_CLOUDS) & set(subs)                                   # never on load
    assert subs["/point_cloud"]["throttle_rate"] == 1000 and subs["/voltage"]["throttle_rate"] == 100
    only_startup(srv, page, X3)


def test_x3_point_clouds_are_sampled_once_on_request(view):
    """'Sample once' subscribes to a cloud, keeps its first message and unsubscribes at once, so the robot computes
    it only for that moment; the card then shows a snapshot (SAMPLED, never live or stale) of the cloud's size as
    the driver publishes it (height 1, width = the valid points, dense)."""
    srv, page = x3_sensors(view)
    for sid, topic, frame, step in (("depth_points", X3_CLOUDS[0], "camera_depth_optical_frame", "16"),
                                    ("registered_points", X3_CLOUDS[1], "camera_color_optical_frame", "20")):
        sensor(page, sid).locator('[data-testid="sensor-sample"]').click()
        page.wait_for_function(f"document.querySelector('[data-sensor={sid}]').dataset.state === 'sampled'", timeout=8000)
        subs = [o for o in srv.ops_of("subscribe") if o["topic"] == topic]
        # one message only: rosbridge may send no second one before the unsubscribe (a cloud is megabytes)
        assert len(subs) == 1 and (subs[0]["throttle_rate"], subs[0]["queue_length"]) == (3_600_000, 1)
        assert srv.wait_for(lambda: subs[0]["id"] in {o["id"] for o in srv.ops_of("unsubscribe")}, 3.0)
        assert [sval(page, sid, f) for f in ("height", "is_dense", "point_step", "frame_id")] == ["1", "true", step, frame]
        assert int(sval(page, sid, "width")) > 200_000
        assert "SAMPLED" in sensor(page, sid).locator('[data-testid="sensor-state"]').inner_text()
        assert "not live" in sensor(page, sid).locator('[data-testid="sensor-meta"]').inner_text()
    page.wait_for_timeout(1500)
    assert [len([o for o in srv.ops_of("subscribe") if o["topic"] == t]) for t in X3_CLOUDS] == [1, 1]   # nothing re-subscribed
    # the clouds publish continuously (10 Hz on this wire, 30 on the robot) while subscribed: exactly one reached the page
    assert [srv.delivered.get(t, 0) for t in X3_CLOUDS] == [1, 1]
    assert page.evaluate("RC.state.sensors.filter(s => s.def.on_demand).map(s => s.count)") == [1, 1]
    assert sensor_state(page, "depth_points") == "sampled"
    only_startup(srv, page, X3)


def test_x3_cloud_sample_without_a_message_unsubscribes_and_says_so(view, fake, monkeypatch):
    srv = fake(wire_spec(load(X3)))
    srv.paused.add(X3_CLOUDS[0])
    srv, page = view(srv=srv)
    page.evaluate("RC.sampleTimeoutMs = 1500")
    sensor(page, "depth_points").locator('[data-testid="sensor-sample"]').click()
    page.wait_for_function("document.querySelector('[data-sensor=depth_points]').dataset.state === 'nomessage'", timeout=8000)
    sub = next(o for o in srv.ops_of("subscribe") if o["topic"] == X3_CLOUDS[0])
    assert srv.wait_for(lambda: sub["id"] in {o["id"] for o in srv.ops_of("unsubscribe")}, 3.0)
    assert "unsubscribed" in sensor(page, "depth_points").locator('[data-testid="sensor-meta"]').inner_text()


def test_x3_sensor_values_are_formatted_honestly(view):
    """Vectors as x, y, z components kept whole, with no unit where no source establishes one (the IMU on an
    ICM20948 board, the magnetometer), said in a caption; quaternions as roll, pitch, yaw in degrees; arrays by
    count; the firmware version to one decimal; a missing orientation (the raw IMU's zero quaternion) is not shown."""
    srv, page = x3_sensors(view)
    wait_sensors(page, "live", sid="imu")
    wait_sensors(page, "live", sid="odom")
    field = lambda sid, f: sensor(page, sid).locator(f'[data-field="{f}"] .sf-value').text_content()      # noqa: E731
    assert field("imu", "orientation") == "roll 0.0° · pitch 0.0° · yaw 90.0°"
    assert abs(float(sval(page, "imu", "orientation").split(",")[2]) - 90) < 1e-9
    # the same orientation re-expressed for base_link through the URDF's imu mount (rpy 0, 3.1415, 1.5707)
    def qmul(a, b):
        return (a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2])
    ax = lambda v, ang: tuple(c * math.sin(ang / 2) for c in v) + (math.cos(ang / 2),)          # noqa: E731
    mount = qmul(qmul(ax((0, 0, 1), 1.5707), ax((0, 1, 0), 3.1415)), ax((1, 0, 0), 0.0))
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    x, y, z, w = qmul(q, (-mount[0], -mount[1], -mount[2], mount[3]))
    want = [math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
            math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x))))),
            math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))]
    got = [float(v) for v in sval(page, "imu", "orientation_base").split(",")]
    assert all(abs(g - e) < 1e-6 for g, e in zip(got, want)), (got, want)
    assert abs(abs(got[0]) - 180) < 0.1                    # the mount flips the frame: a level imu_link is a flipped base_link
    assert "base_link" in sensor(page, "imu").locator('[data-field="orientation_base"]').inner_text()
    assert field("imu_raw", "linear_acceleration") == "x 0.125 · y −0.250 · z 0.500"
    assert sensor(page, "imu_raw").locator('[data-field="linear_acceleration"] .sf-unit').count() == 0
    assert sensor(page, "imu").locator('.sf-unit').count() == 0
    assert "ICM20948" in sensor(page, "imu_raw").locator('[data-testid="sensor-caption"]').inner_text()
    assert sensor(page, "imu_raw").locator('[data-field="linear_acceleration"] .vc').all_text_contents() == ["x 0.125", "y −0.250", "z 0.500"]
    assert sval(page, "mag", "magnetic_field") == "12.5,-3.25,40"
    assert field("odom", "orientation").endswith("yaw 30.0°") and sval(page, "odom", "x") == "1.5"
    assert field("vel_raw", "wz") == "0.500"
    assert field("battery", "voltage") == "12.3" and field("edition", "version") == "1.1"
    wait_sensors(page, "live", sid="scan")
    assert sval(page, "scan", "returns") == "945" and sval(page, "scan", "frame_id") == "laser"
    wait_sensors(page, "live", sid="lidar_points")
    assert sval(page, "lidar_points", "points") == "3"
    srv.emit("/ArmAngleUpdate")
    page.wait_for_function("document.querySelector('[data-sensor=arm_angle_update]').dataset.state === 'lastevent'", timeout=5000)
    assert sval(page, "arm_angle_update", "angle") == "30"
    assert field("arm_angle_update", "joints") == "(none)"
    from wirespec import sensor_sample
    srv.set_sample("/imu/data", dict(sensor_sample("/imu/data", "sensor_msgs/Imu"), orientation={"x": 0, "y": 0, "z": 0, "w": 0}))
    page.wait_for_function("document.querySelector('[data-sensor=imu] [data-field=orientation] .sf-value').textContent === '(no orientation)'", timeout=5000)
    assert re.search(r"\d+(\.\d)?/s", sensor(page, "battery").locator('[data-testid="sensor-meta"]').inner_text())


def test_x3_values_that_mean_no_reading_are_not_shown_as_readings(view):
    """0.0 V is the library's value before the board reports a voltage, and -1 the firmware version when the board
    does not answer: shown as 'no reading', never as a reading, and an unread battery is never marked low."""
    srv, page = x3_sensors(view)
    wait_sensors(page, "live", sid="battery")
    srv.set_sample("/voltage", {"data": 0.0})
    srv.set_sample("/edition", {"data": -1.0})
    page.wait_for_function("document.querySelector('[data-testid=headline]').dataset.noreading === 'true'", timeout=5000)
    head = page.locator('[data-testid="headline"]')
    assert "no reading" in head.locator('[data-testid="headline-value"]').inner_text()
    assert head.get_attribute("data-warn") == "" and not head.locator(".hl-warn").is_visible()
    assert sval(page, "battery", "voltage") == "" and "no reading" in sensor(page, "battery").locator(".sf-value").inner_text()
    page.wait_for_function("document.querySelector('[data-sensor=edition] .sf-value').dataset.noreading === 'true'", timeout=5000)
    assert "did not answer" in sensor(page, "edition").locator(".sf-value").inner_text()
    srv.set_sample("/voltage", {"data": 11.9})
    page.wait_for_function("document.querySelector('[data-testid=headline-value]').textContent === '11.9'", timeout=5000)
    assert head.get_attribute("data-noreading") == ""


def test_x3_battery_below_the_documented_alarm_level_is_marked_low(view):
    """Below 9.6 V (the board's low-battery alarm, vendor battery precautions) the battery reading is marked
    low in its card and in the headline; above it, not."""
    srv, page = x3_sensors(view)
    wait_sensors(page, "live", sid="battery")
    srv.set_sample("/voltage", {"data": 9.4})
    page.wait_for_function("document.querySelector('[data-testid=headline]').dataset.warn === 'low'", timeout=5000)
    assert sensor(page, "battery").locator('[data-field="voltage"]').get_attribute("data-warn") == "low"
    assert sensor(page, "battery").locator('[data-testid="sensor-warn"]').is_visible()
    assert "below 9.6 V" in page.locator('[data-testid="headline"]').inner_text()
    assert page.locator('[data-testid="headline-value"]').inner_text() == "9.4"
    srv.set_sample("/voltage", {"data": 11.6})
    page.wait_for_function("document.querySelector('[data-testid=headline]').dataset.warn === ''", timeout=5000)
    assert not sensor(page, "battery").locator('[data-testid="sensor-warn"]').is_visible()
    srv.paused.add("/voltage")
    page.wait_for_function("document.querySelector('[data-testid=headline]').dataset.state === 'stale'", timeout=5000)
    assert "STALE" in page.locator('[data-testid="headline-state"]').inner_text()


def scan_plot(page):
    return sensor(page, "scan").locator('[data-testid="scan-plot"]')


def test_x3_lidar_scan_is_drawn_top_down_with_its_nearest_return(view):
    """The /scan sensor, as TG.launch configures the 4ROS (-85..85 deg, 945 slots, range 0.01-50 m, +inf where there
    is no return, sent by Python's json as bare Infinity), is drawn as a top-down plot (forward up): only returns
    within range_min..range_max, the nearest named with its bearing."""
    srv, page = x3_sensors(view)
    wait_sensors(page, "live", sid="scan")
    page.wait_for_function("document.querySelector('[data-testid=scan-plot]').dataset.points === '844'", timeout=5000)
    plot = scan_plot(page)
    assert plot.get_attribute("data-nearest") == "0.5" and plot.get_attribute("data-radius") == "2"
    text = plot.locator('[data-testid="scan-text"]').inner_text()
    assert "nearest 0.50 m at 0°" in text and "forward is up" in text and "844 of 945" in text
    c = plot.locator("canvas")
    px = c.evaluate("c => { const g = c.getContext('2d'), k = (c.width / 2 - 14) / 2, x = c.width / 2, y = c.height / 2 - 0.5 * k;"
                    " return Array.from(g.getImageData(Math.round(x), Math.round(y), 1, 1).data); }")
    assert px[3] == 255 and px[:3] != [0, 0, 0]
    from wirespec import SCAN_N, sensor_sample
    scan = sensor_sample("/scan", "sensor_msgs/LaserScan")
    scan["ranges"] = [float("inf")] * 472 + [1.25] + [float("nan")] * (SCAN_N - 473)
    srv.set_sample("/scan", scan)
    page.wait_for_function("document.querySelector('[data-testid=scan-plot]').dataset.points === '1'", timeout=5000)
    assert scan_plot(page).get_attribute("data-nearest") == "1.25" and scan_plot(page).get_attribute("data-radius") == "2"
    assert "at 0°" in scan_plot(page).locator('[data-testid="scan-text"]').inner_text()
    # a few far returns (a doorway, a window) do not shrink the plot: the radius follows the bulk of the returns
    scan["ranges"] = [2.0] * 900 + [40.0] * 45
    srv.set_sample("/scan", scan)
    page.wait_for_function("document.querySelector('[data-testid=scan-plot]').dataset.beyond === '45'", timeout=5000)
    assert scan_plot(page).get_attribute("data-radius") == "2"
    assert "45 beyond 2 m shown on the rim" in scan_plot(page).locator('[data-testid="scan-text"]').inner_text()
    page.locator('[data-testid="scan-zoom-out"]').click()
    assert scan_plot(page).get_attribute("data-radius") == "4"
    page.locator('[data-testid="scan-zoom-in"]').click()
    page.locator('[data-testid="scan-zoom-in"]').click()
    assert scan_plot(page).get_attribute("data-radius") == "1"
    page.locator('[data-testid="scan-zoom-auto"]').click()
    assert scan_plot(page).get_attribute("data-radius") == "2"
    only_startup(srv, page, X3)


def test_x3_page_on_a_phone(view):
    """At 400 px: an in-page jump reaches the sensors; vector components never break inside themselves."""
    srv, page = view(wire_spec(load(X3)), viewport=(400, 800))
    wait_sensors(page, "live", sid="imu_raw")
    assert page.locator('[data-testid="jump"]').is_visible()
    page.locator('[data-testid="jump-sensors"]').click()
    page.wait_for_timeout(400)
    top = page.evaluate("document.querySelector('[data-testid=sensors]').getBoundingClientRect().top")
    assert 0 <= top < 200, top
    page.evaluate("document.querySelector('[data-sensor=imu_raw]').scrollIntoView()")
    heights = sensor(page, "imu_raw").locator(".vc").evaluate_all("els => els.map(e => e.getClientRects().length)")
    assert heights and all(h == 1 for h in heights), heights
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")


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
    # both robots' streams, untied: the myAGV's /usb_cam and the X3's colour, depth and infrared
    assert page.locator('[data-testid="untied"]').count() == 4
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
                                            ("rebot_b601_rs", REBOT_JS, "sensor_msgs/msg/Imu")])
def test_explicit_selection_cannot_bypass_type_checks(view, robot, name, bad):
    srv, page = view(retype(wire_spec(load(robot)), name, bad), robot=robot)
    assert page.locator('[data-testid="target"]').inner_text() == "none"
    reason = page.locator('[data-testid="reason"]').inner_text()
    assert "wrong type" in reason and name in reason and bad in reason, reason
    assert page.locator('[data-testid="control"]').count() == 0
    assert page.evaluate("RC.state.model") is None
    assert srv.command_ops() == []


@pytest.mark.parametrize("robot", [None, "rebot_b601_rs"])
def test_hidden_action_services_type_check_the_action_as_python_does(view, robot):
    """A ROS 2 rosapi that lists the hidden `_action` services reveals an action's type through its
    send_goal service (discovery.py, dialect.ros2_action_from_send_goal). The page checks that type
    as Python does (console spec §3, §4): with the documented type the arm action is verified and
    the target validates; with another action's type it is refused, also when selected explicitly."""
    good = with_hidden_action_endpoints(rebot_spec(), ARM, "control_msgs/action/FollowJointTrajectory")
    bad = with_hidden_action_endpoints(rebot_spec(), ARM, "control_msgs/action/GripperCommand")
    p = load(robot) if robot else None
    for spec, ok in ((good, True), (bad, False)):
        srv, page = view(spec, robot=robot)
        rb = Rosbridge(srv.url).connect()
        try:
            g = fetch_graph(rb)
        finally:
            rb.close()
        assert g.actions[ARM] == ("control_msgs/action/FollowJointTrajectory" if ok else "control_msgs/action/GripperCommand")
        shown = page.locator('[data-testid="target"]').inner_text()
        if ok:
            t = select_target(g, p, None)
            assert ARM not in {n for _, n, _ in t.validation.unverified}         # verified, not only named
            assert shown == t.label == REBOT
            assert page.evaluate(f"RC.state.target.validation.unverified.includes('{ARM}')") is False
        else:
            with pytest.raises(SelectionError) as e:
                select_target(g, p, None)
            want = f"wrong type on action {ARM}: expected control_msgs/action/FollowJointTrajectory, found control_msgs/action/GripperCommand"
            assert want in e.value.reason, e.value.reason
            assert shown == "none"
            reason = page.locator('[data-testid="reason"]').inner_text()
            assert want in reason, reason
            assert page.locator('[data-testid="control"]').count() == 0
        assert srv.command_ops() == []


@pytest.mark.parametrize("robot", ["rebot_b601_rs", "ainex", "rosmaster_x3_plus"])
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


@pytest.mark.parametrize("robot", ["rebot_b601_rs", "ainex"])
def test_page_states_the_sending_contract_and_has_no_arm_switch(view, robot):
    """No arm switch, no Esc behaviour, no stop for a bounded control; only a mobile profile's
    base motion has a STOP button (console spec §2.2)."""
    srv, page = view(rebot_spec() if robot == "rebot_b601_rs" else wire_spec(load(robot)))
    note = page.locator('[data-testid="send-note"]').inner_text()
    assert "Changes are sent to the robot immediately, and only base motion is stopped automatically." in note
    assert "keeps running" in note
    # how an action control whose server does not preempt holds changes (console spec §2.2)
    assert "keeps one goal at a time, your latest change goes out when that goal ends" in " ".join(note.split())
    assert f"Controls live on {robot}" in page.locator('[data-testid="live-card"]').inner_text()
    for sel in ("#enable", '[data-testid="enable"]', '[data-testid="stop-all"]', '[data-testid="armed-pill"]',
                '[data-testid="stop-status"]', '[data-testid="startup"]'):
        assert page.locator(sel).count() == 0, sel
    assert page.locator('#controls button:has-text("stop"), #controls button:has-text("cancel")').count() == 0
    if robot == "rebot_b601_rs":
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
    # head_pan on the model's -Z axis (console spec §2.1), as teleop's Left key.
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
    srv2, page2 = view(rebot_spec())
    num(page2, "arm_trajectory", "joint1").fill("3.0")                  # joint1: -2.8..2.8 rad
    num(page2, "arm_trajectory", "joint1").press("Tab")
    page2.wait_for_timeout(400)
    assert srv2.ops_of("send_action_goal") == []
    assert "above documented maximum" in status(page2, "arm_trajectory")


def test_slider_cannot_leave_documented_limits(view):
    srv, page = view(rebot_spec(exec_s=0.3))
    f = num(page, "arm_trajectory", "joint1")
    r = rng(page, "arm_trajectory", "joint1")
    r.evaluate("r => { r.value = '99'; r.dispatchEvent(new Event('input', {bubbles: true})); }")
    assert float(f.input_value()) == 2.8
    r.evaluate("r => { r.value = '-99'; r.dispatchEvent(new Event('input', {bubbles: true})); }")
    assert float(f.input_value()) == -2.8
    final = drag(page, "arm_trajectory", "joint1", 0.5, 1.3, steps=20, step_ms=30)   # past the end
    assert final == 2.8
    page.wait_for_timeout(1200)
    sent = [g["args"]["trajectory"]["points"][0]["positions"][0] for g in goals(srv, ARM)]
    assert sent and all(-2.8 <= v <= 2.8 for v in sent) and sent[-1] == 2.8


# ------------------------------------------------------------------ action goals: preempting servers

def kind_text(page, cid):
    return control(page, cid).locator(".ctl-kind").inner_text()


def test_action_drag_streams_preempting_goals_with_bounded_connections(view, monkeypatch):
    """A server that preempts its running goal (the preempting variant): goals stream at about 5
    per second while dragging, each preempting the previous one, the final value delivered."""
    srv, page = view(preempting(monkeypatch))
    assert "each new one" in kind_text(page, "arm_trajectory") and "preempts" in kind_text(page, "arm_trajectory")
    base = srv.open_clients()
    seen = []
    t0 = time.monotonic()
    final = drag(page, "arm_trajectory", "joint1", 0.5, 0.1, steps=50, step_ms=40,
                 during=lambda: seen.append(srv.open_clients()))
    dur = time.monotonic() - t0                                          # longer than 2 s on a loaded host
    page.wait_for_timeout(800)
    gs = goals(srv, ARM)
    assert 5 <= len(gs) <= dur / 0.2 + 3, (len(gs), dur)                 # about 5 per s
    assert min(gaps(gs)) > 0.1, gaps(gs)                         # throttled (each goal has its own connection: arrival jitter)
    g = gs[-1]
    assert g["args"]["trajectory"]["points"][0]["positions"][0] == final < -1.0
    assert g["args"]["trajectory"]["joint_names"] == [f"joint{i}" for i in range(1, 7)] and g["feedback"] is True
    # Each goal preempted the previous one; only the latest still runs.
    assert [srv.goals[x["id"]]["done"] for x in gs] == [True] * (len(gs) - 1) + [False]
    assert max(seen) <= base + GOAL_CONNS_MAX, seen
    assert srv.open_clients() == base + 1                                # the latest goal's connection
    assert "goal running" in status(page, "arm_trajectory")
    no_stop_ops(srv)


@pytest.mark.parametrize("server", ["preempt", "reject"])
def test_silent_action_server_bounds_goals_and_connections(view, monkeypatch, server):
    """A server that never acknowledges a goal (and, where it rejects goals while one runs, never
    answers a rejected one, as rosbridge before 2.5 does not): a goal unacknowledged after 1 s holds
    nothing back, so about one goal per second, the final value delivered, and never more than
    GOAL_CONNS_MAX goal connections at any moment (sampled every 5 ms through a 5 s drag)."""
    srv, page = view(preempting(monkeypatch) if server == "preempt" else rebot_spec())
    srv.action_behaviour[ARM] = "silent"                                 # never acknowledges a goal
    base = srv.open_clients()
    seen, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            seen.append(srv.open_clients())
            time.sleep(0.005)
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        final = drag(page, "arm_trajectory", "joint1", 0.5, 0.9, steps=130, step_ms=40)
        page.wait_for_timeout(1500)                                      # the trailing goal after the ack timeout
    finally:
        stop.set()
        th.join()
    gs = goals(srv, ARM)
    assert 4 <= len(gs) <= 10, len(gs)                                   # about one per second
    assert gs[-1]["args"]["trajectory"]["points"][0]["positions"][0] == final
    assert max(seen) <= base + GOAL_CONNS_MAX, (base, max(seen))


def test_slow_preempting_server_bounds_connections_while_goals_retire(view, monkeypatch):
    """A preempting server that acknowledges slowly: the first two goals' acknowledgements (and
    results) arrive 2 s late, so after 1 s each holds nothing back and three goal connections
    are open; the third goal is acknowledged after 0.3 s, which retires the two older goals and
    lets the next goal go out at once. Their connections still count until they have closed:
    the next goal's connection opens only then, so never more than GOAL_CONNS_MAX are open
    (console spec §2.2), sampled every 5 ms through a 4 s drag; the final value is delivered."""
    srv, page = view(preempting(monkeypatch))
    srv.reply_delay[ARM] = 2.0
    base = srv.open_clients()
    seen, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            seen.append(srv.open_clients())
            if len(goals(srv, ARM)) >= 2:
                srv.reply_delay[ARM] = 0.3                               # from the third goal on
            time.sleep(0.005)
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        final = drag(page, "arm_trajectory", "joint1", 0.5, 0.9, steps=100, step_ms=40)
        page.wait_for_timeout(1000)
    finally:
        stop.set()
        th.join()
    gs = goals(srv, ARM)
    assert len(gs) >= 5, len(gs)                 # two slow goals, then one per acknowledgement
    assert gs[-1]["args"]["trajectory"]["points"][0]["positions"][0] == final
    assert max(seen) == base + GOAL_CONNS_MAX, (base, max(seen))         # bounded, and the bound was reached
    assert srv.wait_for(lambda: srv.open_clients() == base + 1, 2.0), srv.open_clients()   # the latest goal's
    no_stop_ops(srv)


# ------------------------------------------------------------------ action goals: servers that do not preempt

def joint1_of(g):
    return g["args"]["trajectory"]["points"][0]["positions"][0]


def finished_with(srv, action, value_of, final, timeout=6.0):
    """Wait until the latest goal of `action` carries `final` and has ended; its goals."""
    def ok():
        gs = goals(srv, action)
        return bool(gs) and value_of(gs[-1]) == final and srv.goals.get(gs[-1]["id"], {}).get("done") is True
    assert srv.wait_for(ok, timeout), [value_of(g) for g in goals(srv, action)]
    return goals(srv, action)


def one_at_a_time(srv, gs):
    """Every goal was sent after the previous goal's result."""
    for a, b in zip(gs, gs[1:]):
        done = srv.goals[a["id"]]["t_done"]
        assert done is not None and b["_t"] >= done, (a["id"], b["_t"] - (done or 0))


def test_rebot_arm_drag_holds_the_latest_value_until_the_running_goal_ends(view):
    """The reBot's arm server rejects a goal while one runs (ros_actions.py#L46-L59, profile `busy`),
    so a drag keeps one goal outstanding: changes made while it runs are held (only the latest)
    and go out when its result arrives; nothing is cancelled and the server rejects nothing; the
    final value is delivered after the running goal's result; each goal's connection closes after
    its result (console spec §2.2, §4)."""
    srv, page = view(rebot_spec(exec_s=0.5))
    assert "one at a time" in kind_text(page, "arm_trajectory")
    assert "rejects a new goal while one runs" in kind_text(page, "arm_trajectory")
    base = srv.open_clients()
    seen, outstanding, shown = [], [], []

    def during():
        seen.append(srv.open_clients())
        outstanding.append(sum(1 for g in list(srv.goals.values()) if g["action"] == ARM and not g["done"]))
        shown.append(status(page, "arm_trajectory"))
    t0 = time.monotonic()
    final = drag(page, "arm_trajectory", "joint1", 0.5, 0.1, steps=50, step_ms=40, during=during)
    dur = time.monotonic() - t0
    gs = finished_with(srv, ARM, joint1_of, final)
    assert final < -1.0 and 3 <= len(gs) <= dur / 0.5 + 2, (len(gs), dur)   # paced by the running goals
    one_at_a_time(srv, gs)
    assert max(outstanding) == 1 and srv.rejected == []
    assert any(s.startswith("running:") and "latest value is held" in s for s in shown), shown
    assert gs[-1]["args"]["trajectory"]["joint_names"] == [f"joint{i}" for i in range(1, 7)]
    page.wait_for_function("document.querySelector('[data-control=arm_trajectory]').dataset.phase === 'done'", timeout=3000)
    assert "goal succeeded" in status(page, "arm_trajectory")
    page.wait_for_timeout(300)
    assert len(goals(srv, ARM)) == len(gs)                               # an unchanged value is not resent
    assert max(seen) <= base + GOAL_CONNS_MAX, seen
    assert srv.open_clients() == base, (srv.open_clients(), base)        # every goal's connection closed
    no_stop_ops(srv)


def test_rebot_gripper_goals_go_one_at_a_time(view):
    """The gripper's server runs a goal beside the running one (ros_actions.py#L51-L59): the page
    still keeps one goal outstanding, so goals do not pile up on it, and delivers the final value.
    The goal carries the motor angle (0 closed .. 5 open rad) and the vendor's max_effort 0."""
    srv, page = view(rebot_spec(exec_s=0.3))
    assert "beside the running one" in kind_text(page, "gripper")
    base = srv.open_clients()
    seen = []
    final = drag(page, "gripper", "position", 0.0, 0.6, steps=25, step_ms=40,
                 during=lambda: seen.append(srv.open_clients()))
    gs = finished_with(srv, GRIP, lambda g: g["args"]["command"]["position"], final)
    assert len(gs) >= 2 and final > 2.0
    one_at_a_time(srv, gs)
    assert gs[-1]["args"]["command"]["max_effort"] == 0.0
    assert max(seen) <= base + GOAL_CONNS_MAX, seen
    assert srv.wait_for(lambda: srv.open_clients() == base, 2.0), (srv.open_clients(), base)   # every goal's connection closed
    no_stop_ops(srv)


def test_change_after_the_result_is_sent_even_when_it_equals_the_last_goal(view):
    """Goals of a control whose server does not preempt are paced at about 5 per second, as a
    preempting server's are. Here the running goal (joint2 0.5) ends before the pacing lets the
    value held meanwhile (0.6) go out, and the value is set back to 0.5 while no goal is
    outstanding: that change replaces the held value and is sent, although it equals the last
    goal's (console spec §2.2); 0.6 is never sent."""
    srv, page = view(rebot_spec(exec_s=0.03))
    r = page.evaluate("""async () => {
      const cs = RC.state.controls.get("arm_trajectory"), f = cs.inputs.joint2;
      const set = (v) => { f.value = String(v); f.dispatchEvent(new Event("input")); f.dispatchEvent(new Event("change")); };
      const until = async (p) => {
        const end = Date.now() + 3000;
        while (!p()) { if (Date.now() > end) throw new Error("timed out"); await new Promise((ok) => setTimeout(ok, 1)); }
      };
      set(0.5);
      await until(() => cs.goal && !cs.goal.done);
      set(0.6);                                        // held: the goal is outstanding
      const held = cs.th.pending && cs.th.held;
      await until(() => cs.goal.done);
      const paced = !!cs.th.timer && cs.th.held;       // the held value waits for the pacing
      set(0.5);                                        // no goal outstanding: back to the last goal's value
      return { held, paced, sinceSent: Date.now() - cs.th.last };
    }""")
    assert r["held"] and r["paced"] and r["sinceSent"] < 200, r
    joint2 = lambda g: g["args"]["trajectory"]["points"][0]["positions"][1]      # noqa: E731
    assert srv.wait_for(lambda: len(goals(srv, ARM)) >= 2, 3.0), [joint2(g) for g in goals(srv, ARM)]
    page.wait_for_timeout(500)
    gs = goals(srv, ARM)
    assert [joint2(g) for g in gs] == [0.5, 0.5], [joint2(g) for g in gs]
    assert gaps(gs)[0] > 0.1, gaps(gs)                                   # paced (arrival jitter aside)
    one_at_a_time(srv, gs)
    page.wait_for_function("document.querySelector('[data-control=arm_trajectory]').dataset.phase === 'done'", timeout=3000)
    no_stop_ops(srv)


def test_rejected_goal_is_shown_as_rejected_and_not_resent(view):
    """While another client's goal runs, the reBot's arm server rejects the page's goal; rosbridge
    reports it (result false, "Action goal was rejected") and the page shows it as rejected, not
    as a status code, and does not resend it. A later change goes out and is accepted. During a
    drag, each rejected goal's result releases the held latest value, so the final value is
    delivered once the server is free."""
    srv, page = view(rebot_spec(exec_s=0.5))
    srv.occupy(ARM, 1.5)                                                 # another client's goal runs
    f = num(page, "arm_trajectory", "joint2")
    f.fill("0.5")
    f.press("Enter")
    page.wait_for_function("document.querySelector('[data-control=arm_trajectory]').dataset.phase === 'rejected'", timeout=5000)
    shown = status(page, "arm_trajectory")
    assert shown.startswith("rejected: the action server rejected the goal") and "Action goal was rejected" in shown, shown
    assert "status" not in shown and "undefined" not in shown, shown
    assert control(page, "arm_trajectory").get_attribute("data-status") == "bad"
    assert len(goals(srv, ARM)) == 1 and len(srv.rejected) == 1
    page.wait_for_timeout(1600)                                          # the other goal ends: nothing is resent
    assert len(goals(srv, ARM)) == 1
    f.fill("0.6")
    f.press("Enter")
    gs = finished_with(srv, ARM, lambda g: g["args"]["trajectory"]["points"][0]["positions"][1], 0.6)
    assert len(gs) == 2 and gs[-1]["id"] in srv.goals                    # accepted
    # A drag while the server is busy: rejected goals, then the final value once it is free.
    srv.occupy(ARM, 1.0)
    n = len(goals(srv, ARM))
    final = drag(page, "arm_trajectory", "joint1", 0.5, 0.8, steps=40, step_ms=40)
    gs = finished_with(srv, ARM, joint1_of, final)[n:]
    assert len(srv.rejected) >= 2 and srv.goals.get(gs[-1]["id"]) is not None
    assert len(gs) <= 2.2 / 0.2 + 3, len(gs)                             # throttled, also while rejected
    no_stop_ops(srv)


def test_use_measured_and_reset_send_their_values(view):
    """The reBot's measured joints are copied into the arm goal; the gripper's measured motor angle is
    the controller's finger position mapped back (gripper_joint2 = motor / 5 * 0.0715 m)."""
    srv, page = view(rebot_spec(exec_s=0.3))
    stop = threading.Event()
    srv.paused.add(REBOT_JS)                 # this pump publishes the joint states, not the wire's sample

    def pump():
        while not stop.is_set():
            srv._broadcast(REBOT_JS, {"name": REBOT_JOINTS, "position": [0.25, 0.5, 0.75, 0.0, 0.1, -0.2, 0.02, 0.0286]})
            time.sleep(0.05)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        page.wait_for_function("document.querySelector('[data-control=arm_trajectory] .measured').dataset.state === 'live'", timeout=5000)
        assert "+0.250" in control(page, "arm_trajectory").locator(".measured").first.inner_text()
        assert srv.command_ops() == []                                   # feedback is read-only
        control(page, "arm_trajectory").get_by_role("button", name="Use measured").click()
        assert float(num(page, "arm_trajectory", "joint3").input_value()) == 0.75
        assert float(num(page, "gripper", "position").input_value()) == 0       # per control
        page.wait_for_timeout(400)
        assert goals(srv, ARM)[-1]["args"]["trajectory"]["points"][0]["positions"] == [0.25, 0.5, 0.75, 0.0, 0.1, -0.2]
        control(page, "gripper").get_by_role("button", name="Use measured").click()
        assert float(num(page, "gripper", "position").input_value()) == 2.0
        control(page, "arm_trajectory").get_by_role("button", name="Reset to defaults").click()
        page.wait_for_timeout(400)
        assert goals(srv, ARM)[-1]["args"]["trajectory"]["points"][0]["positions"] == [0.0] * 6
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
    """Console spec §2.2: once the target is validated the page asks the
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
    srv, page = view(rebot_spec(exec_s=0.6))
    srv.action_behaviour[GRIP] = "abort"
    num(page, "gripper", "position").fill("0.5")
    num(page, "gripper", "position").press("Enter")
    page.wait_for_function("document.querySelector('[data-control=gripper]').dataset.phase === 'failed'", timeout=8000)
    assert "aborted" in status(page, "gripper")
    page.wait_for_timeout(300)
    assert len(goals(srv, GRIP)) == 1                                         # a failure is not retried
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
    required = p.resolve(required, p.namespace.default if p.namespace else "")    # on the wire
    srv, page = view(drop(wire_spec(p, include_optional=True), required), robot=pid)
    assert page.locator('[data-testid="target"]').inner_text() == "none"
    assert required in page.locator('[data-testid="reason"]').inner_text()
    assert page.locator('[data-testid="control"]').count() == 0
    assert srv.command_ops() == []


# ------------------------------------------------------------------ no bounded control is stopped automatically

@pytest.mark.parametrize("robot", ["rebot_b601_rs", "ainex", "rosmaster_x3_plus"])
def test_release_escape_focus_hide_and_close_send_no_stop_for_bounded_controls(view, browser, robot):
    """Console spec §2.2/§4: release, focus loss, page hide or close send no stop or cancel for a
    bounded control. A mobile profile's base motion requests its own documented stop on focus loss
    and page hide (and nothing else); an arm's page sends nothing at all."""
    srv, page = view(rebot_spec() if robot == "rebot_b601_rs" else wire_spec(load(robot)))
    p = load(robot)
    cid, field = {"rebot_b601_rs": ("arm_trajectory", "joint1"), "ainex": ("head_pan", "position"),
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
    if robot == "rebot_b601_rs":
        latest = goals(srv, ARM)[-1]
        assert srv.goals[latest["id"]]["done"] is False                  # the goal keeps running


def test_target_change_sends_nothing_and_rebinds_controls(view):
    """Two reBots booted with arm_namespace arm1 and arm2: goals go to the selected one only (joint
    names are the same on both: the controller does not prefix them)."""
    spec = merge(rebot_spec(namespace="arm1"), rebot_spec(namespace="arm2"))
    srv, page = view(spec, robot="rebot_b601_rs", namespace="arm1")
    assert page.locator('[data-testid="target"]').inner_text() == "rebot_b601_rs (namespace /arm1)"
    drag(page, "arm_trajectory", "joint1", 0.5, 0.7, steps=8, step_ms=40)
    page.wait_for_timeout(500)
    arm1 = goals(srv, "/arm1/follow_joint_trajectory")
    assert arm1 and arm1[-1]["args"]["trajectory"]["joint_names"][0] == "joint1"
    assert goals(srv, "/arm2/follow_joint_trajectory") == []
    before = len(srv.clients)
    page.fill('[data-testid="namespace"]', "arm2")
    page.click('[data-testid="select"]')
    page.wait_for_function("document.querySelector('[data-testid=target]').innerText === 'rebot_b601_rs (namespace /arm2)'",
                           timeout=8000)
    assert len(srv.clients) == before + 1                               # a new connection for the new target
    page.wait_for_timeout(500)
    no_stop_ops(srv)
    assert srv.goals[arm1[-1]["id"]]["done"] is False                   # the old target's goal keeps running
    assert len(goals(srv, "/arm1/follow_joint_trajectory")) == len(arm1)
    drag(page, "arm_trajectory", "joint1", 0.5, 0.3, steps=5, step_ms=40)
    page.wait_for_timeout(500)
    arm2 = goals(srv, "/arm2/follow_joint_trajectory")
    assert arm2 and arm2[-1]["args"]["trajectory"]["joint_names"][0] == "joint1"
    assert len(goals(srv, "/arm1/follow_joint_trajectory")) == len(arm1)


def test_ros1_target_change_revalidates_and_sends_only_to_the_new_target(view):
    """Every pair of ROS 1 profiles shares required names and none is namespace-capable, so two
    validated ROS 1 targets cannot share one wire: the change goes through a refused selection
    and back. Each selection validates afresh on a new connection; the change sends nothing to the
    old target but base motion's stop, before leaving it (console spec §2.2),
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


@pytest.mark.parametrize("robot", ["rebot_b601_rs", "ainex"])
def test_connection_loss_disables_controls_sends_no_stop_and_never_reconnects(view, robot):
    srv, page = view(rebot_spec() if robot == "rebot_b601_rs" else wire_spec(load(robot)))
    cid, field = ("arm_trajectory", "joint1") if robot == "rebot_b601_rs" else ("head_pan", "position")
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

REBOT_JOINTS = [f"joint{i}" for i in range(1, 7)] + ["gripper_joint1", "gripper_joint2"]
X3_JOINTS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "grip_joint"]


def feedback_topic(pid):
    """The wire name of the profile model's joint-state feedback at the documented default namespace."""
    p = load(pid)
    return p.resolve(p.raw["model"]["feedback"]["topic"], p.namespace.default if p.namespace else "")


def joint_states(srv, names, positions, topic="/joint_states", period=0.05):
    """Publish a JointState from the robot every `period` s until the returned event is set (in
    place of the topic's own sample, which the wire stops publishing meanwhile)."""
    stop = threading.Event()
    srv.paused.add(topic)

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
    """reBot: each joint of the model, the arm's six and the two fingers, is posed from
    /rebotarm/joint_states; when a joint moves, the same joint of the model turns by the same angle.
    Without fresh reports the page says so."""
    srv, page = view(rebot_spec())
    m = model(page)
    assert m["rendered"] and set(m["clickable"]) == set(REBOT_JOINTS)
    assert {m["sources"][j] for j in REBOT_JOINTS} == {"target"}
    # an arm joint is named by its trajectory field; the one gripper field moves both fingers, so
    # each finger is named by the control and the joint
    assert m["labels"] == {**{f"joint{i}": f"Joint {i}" for i in range(1, 7)},
                           "gripper_joint1": "Gripper (gripper_joint1)", "gripper_joint2": "Gripper (gripper_joint2)"}
    assert "not reported" in page.locator('[data-testid="model-pose"]').inner_text()
    pose = [0.5, 0.6, 0.9, 0.1, -0.2, 0.3, 0.02, 0.0286]
    stop = joint_states(srv, REBOT_JOINTS, pose, topic=REBOT_JS)
    try:
        page.wait_for_function("RC.state.model.sources.joint1 === 'reported'", timeout=5000)
        m = model(page)
        assert [m["joints"][j] for j in REBOT_JOINTS] == pose
        assert {m["sources"][j] for j in REBOT_JOINTS} == {"reported"}
        assert f"reported on {REBOT_JS}" in page.locator('[data-testid="model-pose"]').inner_text()
        before = m["world"]
    finally:
        stop.set()
    stop = joint_states(srv, REBOT_JOINTS, [0.5 - 0.4] + pose[1:], topic=REBOT_JS)
    try:
        page.wait_for_function("Math.abs(RC.state.model.joints.joint1 - 0.1) < 1e-9", timeout=5000)
        after = model(page)["world"]
    finally:
        stop.set()
    # The links after joint1 turned about its axis (vertical: its origin has no rotation) by the
    # same 0.4 rad.
    base = before["joint1"]
    ang = lambda p: math.atan2(p[1] - base[1], p[0] - base[0])         # noqa: E731
    for j in ("joint3", "joint5", "gripper_joint1"):
        turn = (ang(after[j]) - ang(before[j]) + math.pi) % (2 * math.pi) - math.pi
        assert abs(turn) == pytest.approx(0.4, abs=1e-4), j
        assert after[j][2] == pytest.approx(before[j][2], abs=1e-5)
    page.wait_for_function("RC.state.model.sources.joint1 === 'target'", timeout=5000)   # stale: says so
    assert srv.command_ops() == []


@pytest.mark.parametrize("model_feedback", [True, False])
def test_joint_prefix_namespace_and_ros2_camera_on_the_page(view, monkeypatch, model_feedback):
    """No packaged profile composes joint names with its namespace or has a ROS 2 camera; the
    synthetic arm of test_discovery.ns_camera_arm does (joint_prefix "{ns}/"). Selected at namespace
    arm1 next to a second one at arm2: its sensor_msgs/msg/Image camera is shown live under
    /arm1; the joints it reports as "arm1/jointN" are shown measured (through the model's
    documented feedback, or else the trajectory's joint names) and pose the model; and its goals
    name "arm1/joint1".."arm1/joint6"."""
    p = ns_camera_arm(model_feedback)
    serve_variants(monkeypatch, p)
    spec = merge(rebot_spec(exec_s=0.3, namespace="arm1", profile=p), rebot_spec(exec_s=0.3, namespace="arm2", profile=p))
    srv, page = view(spec, robot="ns_camera_arm", namespace="arm1")
    assert page.locator('[data-testid="target"]').inner_text() == "ns_camera_arm (namespace /arm1)"
    assert page.locator('[data-testid="camera"]').evaluate_all("els => els.map(e => e.dataset.topic)") == ["/arm1/camera/image_raw"]
    page.wait_for_function("document.querySelector('[data-testid=camera]').dataset.state === 'live'", timeout=8000)
    names = [f"arm1/{j}" for j in REBOT_JOINTS]
    pose = [0.25, 0.5, 0.75, 0.0, 0.1, -0.2, 0.02, 0.0286]
    stop = joint_states(srv, [j for j in REBOT_JOINTS], [9.0] * 8, topic="/arm2/joint_states")    # the other arm
    stop1 = joint_states(srv, names, pose, topic="/arm1/joint_states")
    try:
        page.wait_for_function("document.querySelector('[data-control=arm_trajectory] .measured').dataset.state === 'live'",
                               timeout=5000)
        assert "+0.250" in readout(page, "arm_trajectory", "joint1").inner_text()
        assert "+0.750" in readout(page, "arm_trajectory", "joint3").inner_text()
        if model_feedback:
            page.wait_for_function("RC.state.model.sources.joint1 === 'reported'", timeout=5000)
            assert [model(page)["joints"][j] for j in REBOT_JOINTS] == pose
        else:
            assert model(page)["sources"]["joint1"] == "target"            # the model documents no feedback
    finally:
        stop.set()
        stop1.set()
    f = num(page, "arm_trajectory", "joint1")
    f.fill("0.4")
    f.press("Enter")
    assert srv.wait_for(lambda: len(goals(srv, "/arm1/follow_joint_trajectory")) == 1)
    (g,) = goals(srv, "/arm1/follow_joint_trajectory")
    assert g["args"]["trajectory"]["joint_names"] == [f"arm1/joint{i}" for i in range(1, 7)]
    assert g["args"]["trajectory"]["points"][0]["positions"][0] == 0.4
    assert goals(srv, "/arm2/follow_joint_trajectory") == []
    no_stop_ops(srv)


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
    ("rebot_b601_rs", REBOT_JOINTS, [0.5, 0.6, 0.9, 0.1, -0.2, 0.3, 0.02, 0.0286]),     # fingers slide (m)
    ("rosmaster_x3_plus", X3_JOINTS, [0.3, -0.6, 0.9, -0.4, 1.2, -1.0]),
])
def test_model_pose_is_the_urdf_pose_of_the_reported_joints(view, robot, names, pose):
    srv, page = view(wire_spec(load(robot)))
    stop = joint_states(srv, names, pose, topic=feedback_topic(robot))
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
    """The reBot's fingers are the packaged prismatic joints (posed above); they are leaves with unit
    axes, so the page is also handed a myAGV model with a reported prismatic joint (tilted origin,
    non-unit axis) and a frame after it: it slides, it does not turn."""
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


def test_dragging_a_model_joint_changes_its_control_and_sends_it(view, fake):
    srv = fake(rebot_spec(exec_s=0.3))
    srv.paused.add(REBOT_JS)                           # the robot reports no joint positions: the model follows its target
    srv, page = view(srv=srv)
    select_in_model(page, "joint1")
    page.wait_for_timeout(600)                         # the click brings joint1's control into view: let it settle
    x, y = model_point(page, "joint1")
    page.mouse.move(x, y)
    page.mouse.down()
    for i in range(1, 16):
        page.mouse.move(x + 4 * i, y)
        page.wait_for_timeout(40)
    page.mouse.up()
    page.wait_for_timeout(800)
    v = float(num(page, "arm_trajectory", "joint1").input_value())
    assert v == pytest.approx(60 * 2 * 2.8 / 300, abs=0.011)             # 300 px span the documented range
    assert goals(srv, ARM)[-1]["args"]["trajectory"]["points"][0]["positions"][0] == v
    assert model(page)["joints"]["joint1"] == pytest.approx(v)
    page.keyboard.press("ArrowRight")                                    # the selected joint, by key
    page.wait_for_timeout(600)
    v2 = float(num(page, "arm_trajectory", "joint1").input_value())
    assert v2 == pytest.approx(v + 2 * 2.8 / 100, abs=0.011)
    assert goals(srv, ARM)[-1]["args"]["trajectory"]["points"][0]["positions"][0] == v2
    for _ in range(60):                                                  # never past the documented limit
        page.keyboard.press("Shift+ArrowRight")
    page.wait_for_timeout(800)
    assert float(num(page, "arm_trajectory", "joint1").input_value()) == 2.8
    assert goals(srv, ARM)[-1]["args"]["trajectory"]["points"][0]["positions"][0] == 2.8
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
def test_model_renders_and_every_commanded_joint_is_clickable(view, fake, robot):
    srv = fake(wire_spec(load(robot), include_optional=True))
    srv.paused.add(REBOT_JS)                               # the reBot's wire reports no joint positions here: the model shows its targets
    srv, page = view(srv=srv)
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
    "rebot": lambda: wire_spec(load("rebot_b601_rs")),
    "rebot-arm1": lambda: wire_spec(load("rebot_b601_rs"), namespace="arm1"),
    "rebot-default-and-arm2": lambda: merge(wire_spec(load("rebot_b601_rs")), wire_spec(load("rebot_b601_rs"), namespace="arm2")),
    "two-rebots": lambda: merge(wire_spec(load("rebot_b601_rs"), namespace="arm1"),
                                wire_spec(load("rebot_b601_rs"), namespace="arm2")),
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
    No two packaged profiles nest, so the page's catalog also holds a profile of the reBot booted
    without its move_to_pose action."""
    serve_variants(monkeypatch, without_row(load("rebot_b601_rs"), SUBSET_DROPPED, "rebot_subset"))
    for robot, shown in ((None, REBOT), ("rebot_subset", "none")):
        srv, page = view(rebot_spec(), robot=robot)
        rb = Rosbridge(srv.url).connect()
        try:
            g = fetch_graph(rb)
        finally:
            rb.close()
        assert page.locator('[data-testid="target"]').inner_text() == shown
        assert select_target(g, None, None).label == REBOT
        if robot:
            reason = page.locator('[data-testid="reason"]').inner_text()
            assert f"presents {REBOT}, whose interface contains all of rebot_subset" in reason, reason
            assert page.locator('[data-testid="control"]').count() == 0
        assert srv.command_ops() == []
    srv, page = view(drop(rebot_spec(), "/rebotarm" + SUBSET_DROPPED))    # without that row: the smaller one
    assert page.locator('[data-testid="target"]').inner_text() == "rebot_subset (namespace /rebotarm)"


# ------------------------------------------------------------------ base motion (console spec §2.2, §4)

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
    assert [p.id for p in load_all() if p.is_arm] == ["rebot_b601_rs"]
    srv, page = view(rebot_spec())
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


def watch_stamps(page, topic, seconds):
    """[state, shown frame's stamp, wall time] of the camera every 20 ms for `seconds`."""
    return page.evaluate("""([t, ms]) => new Promise((done) => { const seen = []; const end = performance.now() + ms;
        const s = RC.state.streams.find((x) => x.topic === t);
        const f = () => { seen.push([document.querySelector(`[data-topic="${t}"]`).dataset.state, s.lastStamp, Date.now() / 1000]);
          if (performance.now() < end) setTimeout(f, 20); else done(seen); }; f(); })""", [topic, int(seconds * 1000)])


def test_a_clock_step_with_a_real_delay_never_shows_an_old_image_as_live(view):
    """The critic's case on the page: frames 0.9 s late (live), a forward robot clock step of 1.1 s just after a time
    answer lands, one more frame, then the camera stalls. The page holds each frame until a time answer has read the
    robot's clock after it was taken, so no image older than its bound (1.75 s at 2 frames/s) ever reads LIVE."""
    srv, page = view(wire_spec(load(X3)))
    srv.stamp_lag[X3_CAMS[0]] = 0.9
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    page.wait_for_timeout(1500)
    k = page.evaluate("RC.clock().lastOk")
    page.wait_for_function(f"RC.clock().lastOk !== {k}", timeout=5000, polling=5)
    t_step = time.time()
    srv.clock_skew += 1.1
    time.sleep(0.6)                                          # one more frame, taken after the step
    srv.paused.add(X3_CAMS[0])                               # the camera stalls
    rows = watch_stamps(page, X3_CAMS[0], 4.0)
    ages = [now - (stamp - (1.1 if stamp > t_step - 0.35 else 0.0)) for st, stamp, now in rows if st == "live"]
    assert rows[-1][0] == "stale", rows[-1]
    assert max(ages, default=0.0) <= 1.75 + 0.05, max(ages)


def test_sim_time_keeps_timing_the_robots_answers(view, fake):
    """A ROS 1 robot on simulated time has no clock reference, but the page keeps asking its time and timing the answers
    (as teleop.sh does), so a backed-up link still pauses the cameras (R2)."""
    srv = fake(wire_spec(load(X3)))
    srv.params["/use_sim_time"] = True
    srv, page = view(srv=srv)
    page.wait_for_function(cam_state_is(X3_CAMS[0], "unknown"), timeout=8000)
    asks = lambda: len([o for o in srv.ops_of("call_service") if o.get("service") == "/rosapi/get_time"])  # noqa: E731
    n0 = asks()
    page.wait_for_timeout(2500)
    assert asks() > n0 and page.evaluate("RC.clock().lastRtt") is not None
    srv.time_reply_delay = 1.4
    assert "took 1.4 s" in congested(page)


def test_an_unanswered_use_sim_time_question_counts_as_congestion(view, fake):
    """While the page cannot yet tell whether the robot runs on simulated time (the lag stays unknown), a
    /use_sim_time question unanswered for 3 s while frames keep arriving is congestion (R3), as in teleop.sh."""
    srv = fake(wire_spec(load(X3)))
    srv.param_reply_delay = 6.0
    srv, page = view(srv=srv)
    assert "late while frames keep arriving" in congested(page)


def test_a_steady_offset_never_thrashes_the_pause(view):
    """The critic's thrash: frames stamped a steady 3 s behind (another clock) on a healthy link, then time answers of
    0.35 s (one big frame ahead of them). The stream stays LAGGING with its note, and the cameras are never paused."""
    srv, page = view(wire_spec(load(X3)))
    srv.stamp_lag[X3_CAMS[0]] = 3.0
    page.wait_for_function(cam_state_is(X3_CAMS[0], "lagging"), timeout=8000)
    page.wait_for_timeout(4000)
    assert "steady offset" in cam_card(page, X3_CAMS[0]).locator(".ov-detail").inner_text()
    srv.time_reply_delay = 0.35
    seen = watch_states(page, X3_CAMS[0], 6.0)
    assert set(seen) == {"lagging"}, sorted(set(seen))
    assert page.locator('[data-testid="congestion"]').is_hidden()


def test_a_growing_backlog_pauses_naming_the_round_trip(view):
    """R1 names its trigger: frames falling further behind while the time answers take 0.4 s."""
    srv, page = view(wire_spec(load(X3)))
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    srv.time_reply_delay = 0.4
    stop = threading.Event()

    def grow():
        lag = 1.5
        while not stop.is_set():
            srv.stamp_lag[X3_CAMS[0]] = lag
            lag += 0.05
            time.sleep(0.1)
    th = threading.Thread(target=grow, daemon=True)
    th.start()
    try:
        text = congested(page)
    finally:
        stop.set()
    assert re.search(r"arrive late while the robot's time answer took 0\.\d\d s \(at least 0\.3 s\)", text), text


def test_a_refusal_is_cleared_when_the_cameras_pause(view, fake):
    """A budget refusal describes the link when it was given: when the cameras pause (and again on resume) it is
    cleared, so a paused page never shows an old refusal on an OFF card."""
    srv = fake(wire_spec(load(X3), include_optional=True))
    for t in X3_CAMS:
        srv.frame_size[t] = (640, 480)
    srv, page = view(srv=srv)
    page.wait_for_function(cam_state_is(X3_CAMS[0], "live"), timeout=8000)
    start_streams(page, [X3_CAMS[1]])
    page.wait_for_function(cam_state_is(X3_CAMS[1], "live"), timeout=8000)
    start_streams(page, [X3_CAMS[2]])
    page.wait_for_timeout(400)
    ir = cam_card(page, X3_CAMS[2])
    assert "link budget" in ir.locator(".ov-detail").inner_text()
    srv.time_reply_delay = 1.3
    congested(page)
    page.wait_for_timeout(400)
    detail = ir.locator(".ov-detail").inner_text()
    assert "link budget" not in detail and ir.get_attribute("data-state") == "off", detail
