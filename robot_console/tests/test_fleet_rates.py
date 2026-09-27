"""`python -m robot_console.fleet` without flags and with `--rates [--gate]`.

Simulator spec §5's rate gate and the discovery-driven validation that feeds it. The
judging (`fleet.evaluate`) is pure, so most of it is tested on synthetic arrival times;
two tests run the real stdlib websocket client against the fake bridge. The parity tests
hold the console's rate table equal to the ROS files, read as text with the stdlib.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from robot_console import fleet
from robot_console.discovery import Member
from robot_console.fleet import (
    APERIODIC,
    OPTIONAL,
    PERIODIC,
    Observation,
    Periodic,
    evaluate,
    parse_head,
    required_window_s,
    validate,
)
from robot_console.topics import namespaced

SPECS = Path(__file__).resolve().parents[2] / "robots_specs"
ROS_FILES = {"so101": "so101/ros2.yml", "myagv": "myagv/ros.yml", "ainex": "ainex/ros.yml"}


# ------------------------------------------------------------------ parity with the ROS files


def _ros_topics(path: Path) -> list[dict[str, str]]:
    """The `topics:` rows of a ROS file as `{field: text}`, read with the stdlib only.

    The files are regular: every row opens with `  - name:` and its fields sit at four
    spaces; trailing `# comments` are dropped. Deliberately not a YAML parser -- the
    console installs without one.
    """
    rows: list[dict[str, str]] = []
    inside = False
    for line in path.read_text().splitlines():
        if re.match(r"^\S", line):
            inside = line.startswith("topics:")
            continue
        if not inside:
            continue
        m = re.match(r"^  - name:\s*(\S+)", line)
        if m:
            rows.append({"name": m.group(1)})
            continue
        m = re.match(r"^    (\w+):\s*(.*?)\s*(?:#.*)?$", line)
        if m and rows:
            rows[-1].setdefault(m.group(1), m.group(2))
    return rows


def _declared_from_file(kind: str):
    rows = _ros_topics(SPECS / ROS_FILES[kind])
    assert rows, f"no topics read from {ROS_FILES[kind]}"
    periodic: dict[str, Periodic] = {}
    for row in rows:
        rate = row.get("rate_hz", "")
        if row.get("direction") == "out" and re.fullmatch(r"\d+(\.\d+)?", rate):
            hz = float(rate)
            old = periodic.get(row["name"])
            periodic[row["name"]] = Periodic(row["type"], (old.hz if old else 0.0) + hz,
                                             max(old.fastest_hz if old else 0.0, hz))
    aperiodic: dict[str, str] = {}
    for row in rows:
        if row["name"] not in periodic:
            aperiodic.setdefault(row["name"], row["type"])
    # image_transport plugin streams the file marks unverified, beside its `compressed`.
    optional = {r["name"] for r in rows if r.get("unverified") == "true"
                and re.search(r"/image_raw/(compressedDepth|theora|zstd)$", r["name"])}
    return periodic, aperiodic, optional


@pytest.mark.skipif(not SPECS.is_dir(), reason="no robots_specs/ beside this checkout")
@pytest.mark.parametrize("kind", sorted(ROS_FILES))
def test_the_rate_table_is_the_ros_files(kind) -> None:
    """Every periodic rate the console gates on is its robot's ROS file's, both ways."""
    periodic, aperiodic, optional = _declared_from_file(kind)
    assert PERIODIC[kind] == periodic
    assert APERIODIC[kind] == aperiodic
    assert set(OPTIONAL.get(kind, ())) == optional


def test_the_rigs_rate_is_ros_settings() -> None:
    from robot_console.arm import ros_settings as rs

    rig = PERIODIC["scene"]
    assert {p.hz for p in rig.values()} == {rs.SCENE_CAMERA_HZ}
    assert rig[rs.OVERHEAD_CAMERA_TOPIC].type == rs.OVERHEAD_CAMERA_TYPE
    assert rig[rs.SIDE_CAMERA_TOPIC].type == rs.SIDE_CAMERA_TYPE


def test_every_member_kind_has_a_rate_declaration() -> None:
    from robot_console.discovery import MEMBER_SIGNATURES, RIG_KIND

    assert {k for k, _, _ in MEMBER_SIGNATURES} | {RIG_KIND} == set(PERIODIC)


def test_the_consoles_older_constants_are_inside_the_contract() -> None:
    from robot_console.ainex_topics import CONTRACT_TOPICS as AINEX
    from robot_console.topics import CONTRACT_TOPICS as MYAGV

    assert MYAGV == {t: kind for t, kind in fleet.contract_of("myagv").items() if t in MYAGV}
    assert set(MYAGV) == set(fleet.contract_of("myagv"))
    assert set(AINEX) <= set(fleet.contract_of("ainex"))
    assert set(fleet.arm_topics()) <= set(fleet.contract_of("so101"))


# ------------------------------------------------------------------ validation


def _full_wire(*members: tuple[str, str]) -> dict[str, str]:
    wire: dict[str, str] = {}
    for kind, namespace in members:
        for bare, kind_type in fleet.contract_of(kind).items():
            wire[namespaced(bare, namespace)] = kind_type
    return wire


FLEET = (("so101", "so101"), ("myagv", "myagv"), ("ainex", "ainex"), ("scene", "scene"))


def test_a_complete_fleet_validates() -> None:
    members, problems = validate(_full_wire(*FLEET))
    assert problems == []
    assert [(m.kind, m.namespace) for m in members] == sorted(FLEET, key=lambda f: f[1])


def test_validation_names_a_missing_topic_a_wrong_type_and_an_undeclared_one() -> None:
    wire = _full_wire(*FLEET)
    del wire["/myagv/scan"]
    wire["/ainex/imu"] = "sensor_msgs/msg/Imu"            # the other dialect's spelling
    wire["/so101/secret_state"] = "std_msgs/msg/String"   # nothing declares it
    wire["/so101/gripper_controller/gripper_cmd/_action/status"] = "action_msgs/msg/GoalStatusArray"
    _, problems = validate(wire)
    assert any("/myagv/scan is missing" in p for p in problems)
    assert any("/ainex/imu is sensor_msgs/msg/Imu, not sensor_msgs/Imu" in p for p in problems)
    assert any("/so101/secret_state is not in its contract" in p for p in problems)
    assert len(problems) == 3, problems


def test_a_mistyped_signature_is_reported_not_counted() -> None:
    wire = _full_wire(("myagv", "myagv"))
    wire["/myagv/cmd_vel"] = "std_msgs/String"
    members, problems = validate(wire)
    assert members == []
    assert any("/myagv/cmd_vel is std_msgs/String, not geometry_msgs/Twist" in p for p in problems)
    assert any("no fleet member" in p for p in problems)


def test_the_optional_plugin_streams_are_not_required_or_timed() -> None:
    wire = _full_wire(("so101", "so101"))
    assert "/so101/wrist/image_raw/theora" not in wire
    members, problems = validate(wire)
    assert problems == []
    assert "/so101/wrist/image_raw/theora" not in fleet.expected_rates(members, wire)


def test_a_member_with_no_rate_declaration_fails(monkeypatch) -> None:
    monkeypatch.delitem(fleet.PERIODIC, "myagv")
    _, problems = validate(_full_wire(("myagv", "myagv")))
    assert any("no rate declaration for a myagv" in p for p in problems)


def test_members_are_clocked_by_namespace() -> None:
    members = [Member("so101", "so101"), Member("scene", "scene")]
    assert fleet.clocks_of(members, ["/so101/tf", "/scene/side/color/compressed"]) == {
        "/so101/tf": "so101", "/scene/side/color/compressed": "scene"}


# ------------------------------------------------------------------ judging


def test_the_window_is_thirty_seconds_or_five_slowest_periods() -> None:
    assert required_window_s({"/a": Periodic("t", 50, 50), "/b": Periodic("t", 1, 1)}) == 30.0
    assert required_window_s({"/a": Periodic("t", 0.1, 0.1)}) == pytest.approx(50.0)


@pytest.mark.parametrize("frame, expected", [
    (b'{"op": "publish", "topic": "/a/odom", "msg": {"header": {"seq": 1, "stamp": '
     b'{"secs": 12, "nsecs": 500000000}, "frame_id": "odom"}}}', ("/a/odom", 12.5)),
    (b'{"op": "publish", "topic": "/b/joint_states", "msg": {"header": {"stamp": '
     b'{"sec": 3, "nanosec": 250000000}, "frame_id": ""}}}', ("/b/joint_states", 3.25)),
    (b'{"op": "publish", "topic": "/c/Voltage", "msg": {"data": 12.1}}', ("/c/Voltage", None)),
    (b'{"op": "publish", "topic": "/d/x", "msg": {"header": {"stamp": {"sec": 0, '
     b'"nanosec": 0}}}}', ("/d/x", None)),
    (b'{"op": "service_response", "id": "x", "values": {}}', (None, None)),
])
def test_a_frame_head_gives_its_topic_and_stamp(frame, expected) -> None:
    assert parse_head(frame) == expected


def _synthetic(topics: dict[str, float], seconds: float = 30.0, rtf=lambda t: 1.0,
               drop: Optional[tuple[str, float, float]] = None) -> Observation:
    """Arrivals at each topic's rate over `seconds`, stamped by a clock running at `rtf(t)`."""
    step = 0.01
    clock = [100.0]
    for k in range(int(seconds / step) + 2):
        clock.append(clock[-1] + rtf(k * step) * step)
    arrivals: dict[str, list] = {}
    for topic, hz in topics.items():
        samples = []
        for i in range(int(seconds * hz)):
            t = (i + 0.5) / hz
            if drop and drop[0] == topic and drop[1] <= t < drop[2]:
                continue
            samples.append((t, clock[int(t / step)]))
        arrivals[topic] = samples
    return Observation(0.0, seconds, arrivals, wall_offset=1e9)


EXPECTED = {"/r/fast": Periodic("t", 50, 50), "/r/slow": Periodic("t", 10, 10)}
STEADY = {"/r/fast": 50, "/r/slow": 10}


def test_a_steady_fleet_passes() -> None:
    report = evaluate(EXPECTED, _synthetic(STEADY), clocks={"/r/fast": "r", "/r/slow": "r"})
    assert report.failures == []
    (clock,) = report.clocks
    assert clock.name == "r" and clock.mean == pytest.approx(1.0, abs=1e-2) and not clock.wall


def test_a_rate_outside_ten_percent_fails_and_inside_passes() -> None:
    assert evaluate(EXPECTED, _synthetic({"/r/fast": 45.5, "/r/slow": 10})).failures == []
    failures = evaluate(EXPECTED, _synthetic({"/r/fast": 44, "/r/slow": 10})).failures
    assert failures and all("/r/fast" in f for f in failures)
    assert any("outside +/-10%" in f for f in failures)


def test_a_gap_over_three_periods_fails_even_at_the_right_mean_rate() -> None:
    # 0.35 s missing from a 10 Hz topic: 3.5+ periods, while the count stays within 2%.
    report = evaluate(EXPECTED, _synthetic(STEADY, drop=("/r/slow", 12.0, 12.35)))
    assert report.failures and all("gap" in f and "/r/slow" in f for f in report.failures)
    assert evaluate(EXPECTED, _synthetic(STEADY, drop=("/r/slow", 12.0, 12.15))).failures == []


def test_a_topic_that_never_arrives_fails() -> None:
    report = evaluate(EXPECTED, _synthetic({"/r/fast": 50}))
    assert any(f.startswith("/r/slow: no message") for f in report.failures)


def test_the_mean_real_time_factor_is_gated_both_ways() -> None:
    slow = evaluate(EXPECTED, _synthetic(STEADY, rtf=lambda t: 0.85))
    assert any("mean 0.850" in f for f in slow.failures)
    fast = evaluate(EXPECTED, _synthetic(STEADY, rtf=lambda t: 1.15))
    assert any("mean 1.150" in f for f in fast.failures)


def test_one_slow_ten_second_window_fails_a_good_mean() -> None:
    # 0.80 for 8 s inside 30 s: the mean is ~0.95 -- inside the gate -- and a window is not.
    dip = evaluate(EXPECTED, _synthetic(STEADY, rtf=lambda t: 0.80 if 10 <= t < 18 else 1.0))
    assert 0.90 <= dip.clocks[0].mean <= 1.10
    assert dip.clocks[0].worst_window < 0.90
    assert dip.failures and all("window" in f for f in dip.failures)


def test_each_members_clock_is_judged_on_its_own() -> None:
    obs = _synthetic({"/a/x": 50, "/b/y": 50}, rtf=lambda t: 0.8)
    obs.arrivals["/a/x"] = [(t, 1e9 + t) for t, _ in obs.arrivals["/a/x"]]  # wall-clock stamps
    report = evaluate({"/a/x": Periodic("t", 50, 50), "/b/y": Periodic("t", 50, 50)}, obs,
                      clocks={"/a/x": "a", "/b/y": "b"})
    by_name = {c.name: c for c in report.clocks}
    assert by_name["a"].wall and by_name["a"].mean == pytest.approx(1.0)
    assert not by_name["b"].wall and by_name["b"].mean == pytest.approx(0.8, abs=1e-2)
    assert report.failures and all("(b)" in f for f in report.failures)


def test_an_unstamped_fleet_cannot_pass() -> None:
    obs = _synthetic(STEADY)
    obs.arrivals = {t: [(a, None) for a, _ in v] for t, v in obs.arrivals.items()}
    assert any("no clock was observed" in f for f in evaluate(EXPECTED, obs).failures)


def test_an_insufficient_window_fails() -> None:
    report = evaluate(EXPECTED, _synthetic(STEADY, seconds=12.0))
    assert any("insufficient observation window: 12.0 s observed, 30.0 s required" in f
               for f in report.failures)


def test_the_report_prints_every_topic_and_the_verdict() -> None:
    report = evaluate(EXPECTED, _synthetic({"/r/fast": 44, "/r/slow": 10}))
    text = fleet.format_report(report, "ws://x:1")
    assert "/r/fast" in text and "/r/slow" in text and "FAIL" in text
    assert fleet.format_report(evaluate(EXPECTED, _synthetic(STEADY)), "ws://x:1").endswith("PASS")


# ------------------------------------------------------------------ against a bridge


@pytest.fixture
def small_contract(monkeypatch):
    """A myAGV whose whole contract is two topics, so a real-time test can be short."""
    monkeypatch.setitem(fleet.PERIODIC, "myagv", {"/odom": Periodic("nav_msgs/Odometry", 20, 20)})
    monkeypatch.setitem(fleet.APERIODIC, "myagv", {"/cmd_vel": "geometry_msgs/Twist"})


@pytest.fixture
def odom_wire(bridge, small_contract):
    """The fake bridge carrying that myAGV, its `/odom` at 20 Hz stamped with the wall clock."""
    bridge.topics = {"/myagv/cmd_vel": "geometry_msgs/Twist", "/myagv/odom": "nav_msgs/Odometry"}
    stop = threading.Event()

    def publish() -> None:
        next_t = time.monotonic()
        while not stop.is_set():
            now = time.time()
            bridge.publish("/myagv/odom", {"header": {"seq": 0, "stamp": {
                "secs": int(now), "nsecs": int((now % 1) * 1e9)}, "frame_id": "myagv/odom"}})
            next_t += 0.05
            time.sleep(max(0.0, next_t - time.monotonic()))

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{bridge.port}"
    finally:
        stop.set()
        thread.join(2)


def test_measure_rates_times_a_live_wire(odom_wire, bridge) -> None:
    report = fleet.measure_rates(odom_wire, warmup_s=0.5, window_s=10.5, required_s=10.5)
    (row,) = report.rows
    assert row.topic == "/myagv/odom" and row.ok and row.count >= 200, row
    (clock,) = report.clocks
    assert clock.name == "myagv" and clock.wall and clock.mean == pytest.approx(1.0, abs=0.05)
    assert report.failures == []
    # Unthrottled, and let go of afterwards.
    assert not bridge.subscriptions


def test_the_gate_decides_the_exit_code(odom_wire, capsys) -> None:
    # One second against the required thirty: a report, and a failure in it.
    args = ["--url", odom_wire, "--rates", "--window", "1", "--warmup", "0.2"]
    assert fleet.main(args) == fleet.EXIT_OK
    out = capsys.readouterr().out
    assert "/myagv/odom" in out and "insufficient observation window" in out
    assert fleet.main([*args, "--gate"]) == fleet.EXIT_RATE


def test_an_unreachable_wire_is_a_transport_error(capsys) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert fleet.main(["--url", f"ws://127.0.0.1:{port}", "--rates", "--gate"]) == \
        fleet.EXIT_TRANSPORT
    assert "cannot reach" in capsys.readouterr().out


def test_gate_without_rates_is_refused() -> None:
    with pytest.raises(SystemExit):
        fleet.main(["--gate"])
