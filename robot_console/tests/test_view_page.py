"""The view.sh page in headless Chromium against the fake rosbridge (console spec §2.2, §2.3, §4).

The page speaks rosbridge itself; the fake server records every op it sends, so the tests
check what reached the wire, not only what the page displays. Contract: controls are live as
soon as the target validates; changed values are sent at once (topic publishes and action
goals streamed while dragging, throttled with the last value always delivered; services and
action-group buttons once per click); limits are never exceeded; nothing is ever stopped or
cancelled by the page.
"""

import threading
import time

import pytest

from robot_console.discovery import SelectionError, fetch_graph, select_target
from robot_console.profiles import load
from robot_console.rosbridge import Rosbridge
from robot_console.view import make_server
from wirespec import drop, merge, wire_spec

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


def test_stale_camera_is_marked(view):
    srv, page = view(wire_spec(load("rosmaster_x3_plus")))
    page.wait_for_function("document.querySelector('[data-testid=camera]').dataset.state === 'live'")
    srv.paused.add("/camera/rgb/image_raw")
    page.wait_for_function("document.querySelector('[data-testid=camera]').dataset.state === 'stale'", timeout=8000)
    assert "STALE" in page.locator('[data-testid="camera-state"]').inner_text()


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
    assert srv.command_ops() == []


@pytest.mark.parametrize("robot", ["so101", "ainex", "mycobot280"])
def test_nothing_is_sent_on_load_or_reload(view, robot):
    srv, page = view(wire_spec(load(robot), include_optional=True))
    assert page.locator('[data-testid="live-card"]').get_attribute("data-state") == "live"
    assert not disabled(page, load(robot).controls[0].id)               # live at once
    page.wait_for_timeout(800)
    page.reload()
    page.wait_for_function("window.RC && window.RC.ready === true")
    page.wait_for_timeout(800)
    assert srv.command_ops() == []


def test_page_states_the_sending_contract_and_has_no_arm_or_stop(view):
    srv, page = view(so101_spec())
    note = page.locator('[data-testid="send-note"]').inner_text()
    assert "Changes are sent to the robot immediately, and nothing is stopped automatically." in note
    assert "keeps running" in note
    assert "Controls live on so101" in page.locator('[data-testid="live-card"]').inner_text()
    for sel in ("#enable", '[data-testid="enable"]', '[data-testid="stop-all"]', '[data-testid="armed-pill"]',
                '[data-testid="hold"]', '[data-testid="stop-status"]', '[data-testid="startup"]'):
        assert page.locator(sel).count() == 0, sel
    assert "STOP" not in page.locator("body").inner_text()


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
    # The head pad streams both axes while dragged.
    pad = page.locator(".pad")
    pad.scroll_into_view_if_needed()
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
    assert srv.published(HEAD_PAN)[-1]["msg"]["position"] == pan > 0.5
    assert srv.published(HEAD_TILT)[-1]["msg"]["position"] == tilt > 0.1
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
    final = drag(page, "arm_trajectory", "shoulder_pan", 0.5, 0.1, steps=50, step_ms=40,
                 during=lambda: seen.append(srv.open_clients()))
    page.wait_for_timeout(800)
    gs = goals(srv, SO_ARM)
    assert 5 <= len(gs) <= 2.2 / 0.2 + 3, len(gs)                        # about 5 per s
    assert min(gaps(gs)) > 0.15, gaps(gs)
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
    srv, page = view(so101_spec())
    srv.action_behaviour[SO_ARM] = "silent"                              # never acknowledges a goal
    base = srv.open_clients()
    seen = []
    final = drag(page, "arm_trajectory", "shoulder_pan", 0.5, 0.9, steps=60, step_ms=40,
                 during=lambda: seen.append(srv.open_clients()))
    page.wait_for_timeout(1500)                                          # the trailing goal after the ack timeout
    gs = goals(srv, SO_ARM)
    assert 2 <= len(gs) <= 6, len(gs)                                    # about one per second
    assert gs[-1]["args"]["trajectory"]["points"][0]["positions"][0] == final
    assert max(seen + [srv.open_clients()]) <= base + GOAL_CONNS_MAX, seen


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


# ------------------------------------------------------------------ nothing is stopped automatically

def test_release_escape_focus_hide_and_close_send_no_stop(view, browser):
    srv, page = view(so101_spec())
    drag(page, "arm_trajectory", "shoulder_pan", 0.5, 0.6, steps=5, step_ms=40)     # release ends the drag
    page.wait_for_timeout(500)
    n = len(srv.ops_of("send_action_goal"))
    assert n >= 1
    page.keyboard.press("Escape")
    page.evaluate("window.dispatchEvent(new Event('blur'))")
    page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    page.evaluate("window.dispatchEvent(new Event('pagehide'))")
    page.wait_for_timeout(300)
    page.close()                                                          # closing the tab
    time.sleep(0.5)
    assert len(srv.ops_of("send_action_goal")) == n
    no_stop_ops(srv)
    latest = goals(srv, SO_ARM)[-1]
    assert srv.goals[latest["id"]]["done"] is False                      # the goal keeps running


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
