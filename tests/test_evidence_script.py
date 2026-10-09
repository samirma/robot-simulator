"""Unit checks of the workspace evidence script (`tests/evidence.py`) that start nothing:
its command line, its reading of the readiness line, the travel its base motions take, the
motions and scenes a case requires (workspace spec §3), the projection that checks a scene
picture shows the worktop objects, the AiNex action group's judgement against its record
(frames, init pose, standing height), the real-time check from the control port's `rtf`
operation and the harness heuristics the index states. Fakes stand in for the wire, the
control port and the clock.

    uv run --no-project --with pytest --with pyyaml --with numpy --with pillow pytest tests/test_evidence_script.py
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

try:
    import websocket  # noqa: F401  websocket-client, the script's rosbridge transport
except ImportError:      # not needed here: nothing connects
    sys.modules["websocket"] = types.ModuleType("websocket")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence  # noqa: E402

#: The travel a floor placement guarantees a mobile robot (simulator spec §2.3), m.
GUARANTEED = {"forward": 0.5, "back": 0.25, "side": 0.25}


def test_help_names_one_wire(capsys):
    """Every robot is a single body with one wire on `--port` (simulator spec §2.3)."""
    with pytest.raises(SystemExit) as exc:
        evidence.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # simulator spec §2.3: one wire on --port; there is no --arm-port
    assert "rosbridge port of the robot's wire" in " ".join(out.split()) and "--arm-port" not in out


@pytest.mark.parametrize("line", [
    "spawn ready: ainex wire(s): ainex ws://127.0.0.1:9090 [ROS 1 noetic]",
    "spawn ready: ainex (AiNex) in molmospaces ithor:1 on the floor; wire(s): "
    "ainex ws://127.0.0.1:9090 [ROS 1 noetic]",
])
def test_readiness_line_names_the_robots_wire(line):
    wires = evidence.parse_ready(line)
    assert len(wires) == 1
    w = wires[0]
    assert (w["robot"], w["port"], w["url"], w["ros"]) == \
        ("ainex", 9090, "ws://127.0.0.1:9090", "ROS 1 noetic")


def drive_tolerance(rid: str) -> dict:
    """The robot's recorded drive displacement tolerance."""
    return next(t for t in evidence.interface(rid)["tolerances"]
                if t["figure"].startswith("drive") and "displacement" in t["figure"])


DRIVEN = [r["id"] for r in evidence.registry() if r["kind"] != "arm"
          and any(m["id"] == "drive" for m in evidence.interface(r["id"]).get("motions") or [])]


def test_some_robot_drives():
    assert DRIVEN


@pytest.mark.parametrize("rid", DRIVEN)
def test_drives_stay_within_the_guaranteed_travel(rid):
    """Workspace spec §3: base motions stay within the travel the placement guarantees. After
    every leg of DRIVES the nominal pose, widened by that leg's acceptance bound, is inside it."""
    tol = drive_tolerance(rid)
    x = y = yaw = 0.0
    for name, vx, vy, wz, dur in evidence.DRIVES:
        dx, dy = vx * dur, vy * dur
        x += dx * math.cos(yaw) - dy * math.sin(yaw)
        y += dx * math.sin(yaw) + dy * math.cos(yaw)
        yaw += wz * dur
        b = evidence.bound_of(tol, math.hypot(dx, dy))
        assert x + b <= GUARANTEED["forward"] + 1e-9, (name, x, b)
        assert -x + b <= GUARANTEED["back"] + 1e-9, (name, x, b)
        assert abs(y) + b <= GUARANTEED["side"] + 1e-9, (name, y, b)


def test_ainex_walk_stays_within_the_guaranteed_travel():
    """Workspace spec §3: the AiNex walk stays within the forward travel a floor placement
    guarantees. A passing walk moves at most its commanded distance -- the recorded
    example's x_move_amplitude per step, one step per gait period as the record defines it
    (motions[walk].command.step_definition), over the walk plus the period the stop completes
    and one more for its start -- widened by the recorded acceptance bound."""
    walk = evidence.motion_row("ainex", "walk")
    ex = walk["command"]["example"]["set_param"]
    period = ex["period_time"] / 1000.0
    periods = math.ceil(evidence.AINEX_WALK_S / period) + 2
    commanded = ex["x_move_amplitude"] * periods / evidence.ainex_walk_step_periods(walk)
    reach = commanded + evidence.bound_of(evidence.tolerance("ainex", "walk displacement"), commanded)
    assert 0 < reach <= GUARANTEED["forward"], reach


def test_ainex_walk_step_is_read_from_the_record():
    """The walk's step definition comes from the record; a record that drops it fails loudly."""
    walk = evidence.motion_row("ainex", "walk")
    assert evidence.ainex_walk_step_periods(walk) == 1
    changed = {"command": dict(walk["command"], step_definition=None)}
    with pytest.raises(ValueError, match="one gait period"):
        evidence.ainex_walk_step_periods(changed)


def test_rebot_finger_scale_is_read_from_the_record():
    """The reBot gripper's finger metres per motor rad come from its record (the
    command.position notes), for both finger joints; a record that drops them fails loudly."""
    grip = evidence.motion_row("rebot_b601_rs", "gripper")
    per_rad = evidence.rebot_finger_m_per_motor_rad(grip)
    assert set(per_rad) == {"gripper_joint1", "gripper_joint2"} and all(v > 0 for v in per_rad.values())
    fields = [dict(f, notes="0.0 closed, 5.0 fully open") if f["field"] == "command.position" else f
              for f in grip["command"]["fields"]]
    with pytest.raises(ValueError, match="per gripper motor rad"):
        evidence.rebot_finger_m_per_motor_rad({"command": dict(grip["command"], fields=fields)})


#: Workspace spec §3's smoke-run motions, by robot: a wheeled base drives forward, back,
#: sideways and turns; the ROSMASTER X3 PLUS also moves its arm and gripper; the reBot Arm
#: B601-RS its arm and gripper; the AiNex turns its head (each head joint it records), walks
#: and stops, and plays an action group.
DRIVE = {"drive_forward", "drive_back", "drive_left", "drive_right", "drive_turn"}
SPEC_MOTIONS = {
    "myagv": DRIVE,
    "rosmaster_x3_plus": DRIVE | {"arm", "gripper"},
    "rebot_b601_rs": {"arm", "gripper"},
    "ainex": {"head_pan", "head_tilt", "walk", "action_group"},
}
REGISTRY = [r["id"] for r in evidence.registry()]


def test_every_recorded_robot_has_a_smoke_run():
    """A case per recorded robot (workspace spec §3): the script has a smoke run for every
    robot file, and §3 lists its motions."""
    assert REGISTRY and set(evidence.SMOKE) == set(REGISTRY) == set(SPEC_MOTIONS)


@pytest.mark.parametrize("rid", REGISTRY)
def test_required_motions_are_the_spec_list(rid):
    """The motions a case requires (from the robot's recorded motions) are §3's list."""
    assert set(evidence.required_motions(rid)) == SPEC_MOTIONS[rid]


def test_default_scenes():
    """Simulator spec §2.1: the engines' default scenes, on which every case runs."""
    assert evidence.DEFAULT_SCENES == {"molmospaces": "ithor:1", "robocasa": "robocasa:1-1"}
    assert evidence.ENGINES == ["molmospaces", "robocasa"]


def test_projection_of_a_free_view():
    """`project` places points in a render of a `{pos, target, fovy}` view as the simulation's
    free camera does (no roll, world z up, vertical field of view `fovy`)."""
    view = evidence.free_view([1.0, 2.0, 0.5], 2.0, 30.0, -25.0)
    W, H = 320, 180
    c, r, z = evidence.project(view, view["target"], W, H)
    assert (round(c, 6), round(r, 6), round(z, 6)) == (W / 2, H / 2, 2.0)
    eye = np.asarray(view["pos"])
    assert np.linalg.norm(eye - np.asarray(view["target"])) == pytest.approx(2.0)
    assert eye[2] > 0.5                                   # looking down: the eye is above
    # azimuth 30 deg looks along (cos 30, sin 30): its right is (sin 30, -cos 30)
    right = np.array([math.sin(math.radians(30)), -math.cos(math.radians(30)), 0.0])
    c2, r2, _ = evidence.project(view, np.asarray(view["target"]) + 0.2 * right, W, H)
    assert c2 > W / 2 and r2 == pytest.approx(H / 2, abs=1e-6)
    _, r3, _ = evidence.project(view, np.asarray(view["target"]) + [0, 0, 0.2], W, H)
    assert r3 < H / 2                                     # up in the world is up in the picture
    # the frame's top edge is fovy/2 above the axis, at the target's depth
    half = 2.0 * math.tan(math.radians(evidence.FOVY) / 2)
    up = np.array([-math.sin(math.radians(-25)) * math.cos(math.radians(30)),
                   -math.sin(math.radians(-25)) * math.sin(math.radians(30)),
                   math.cos(math.radians(-25))])
    _, r4, _ = evidence.project(view, np.asarray(view["target"]) + 0.99 * half * up, W, H)
    assert r4 == pytest.approx(0.005 * H, abs=0.01)
    assert evidence.project(view, eye - (np.asarray(view["target"]) - eye), W, H) is None   # behind
    assert evidence.project(view, np.asarray(view["target"]) + 3.0 * right, W, H) is None   # outside


def test_shown_needs_the_depth_render_to_reach_the_point():
    view = evidence.free_view([0.0, 0.0, 0.0], 2.0, 0.0, -30.0)
    depth = np.full((90, 160), 2.0, np.float32)
    assert evidence.shown(depth, view, [0.0, 0.0, 0.0])          # nothing in front
    depth[40:50, 75:85] = 1.5                                     # something 0.5 m nearer
    assert not evidence.shown(depth, view, [0.0, 0.0, 0.0])
    depth[40:50, 75:85] = 1.95                                    # the object's own surface
    assert evidence.shown(depth, view, [0.0, 0.0, 0.0])
    assert not evidence.shown(depth, view, [0.0, 5.0, 0.0])      # out of frame


# --------------------------------------------------------------------------- AiNex action group

AINEX_RECORD = evidence.SPECS / "ainex" / "ros.yml"


class FakeTime:
    """A clock that `sleep` advances: the evidence script's waits take no time here."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def smoke_group():
    """The record's smoke action group: its command row, frames (read here from the record,
    independently of the script) and the recorded init pose."""
    import yaml
    rec = yaml.safe_load(AINEX_RECORD.read_text())
    cmd = next(m for m in rec["motions"] if m["id"] == "action_group")["command"]
    init = next(p["value"] for p in rec["parameters"] if p["name"] == "/ainex_controller/init_pose")
    doc = yaml.safe_load((evidence.SPECS / "ainex" / cmd["estimated_groups"]).read_text())
    frames = [(f["time_ms"], dict(f.get("offsets_rad") or {}))
              for f in doc["groups"][cmd["smoke_example"]["data"]]["frames"]]
    return cmd, frames, {k: float(v) for k, v in init.items()}


def test_ainex_init_pose_is_the_recorded_parameter():
    """The action group's end state is the recorded init pose (ros.yml parameter
    /ainex_controller/init_pose), not the pose before the command."""
    _, _, init = smoke_group()
    assert evidence.ainex_init_pose() == init and len(init) == 24


def test_ainex_group_frames_are_the_records():
    cmd, frames, _ = smoke_group()
    assert evidence.ainex_group_frames(cmd, cmd["smoke_example"]["data"]) == \
        [(t, {k: float(v) for k, v in off.items()}) for t, off in frames]
    with pytest.raises(ValueError, match="has no group"):
        evidence.ainex_group_frames(cmd, "no_such_group")


def test_ainex_held_targets():
    """A target is held when a frame repeats the previous frame's non-zero offset for the
    joint; one passed through within a frame is not; a joint outside the init pose fails."""
    init = {"a": 0.1, "b": 0.0, "c": -0.2}
    frames = [(400, {"a": 0.5}), (400, {"a": 0.5, "b": 0.3}), (400, {"b": -0.3}),
              (400, {"b": -0.3, "c": 0.0}), (400, {})]
    assert evidence.ainex_held_targets(frames, init) == {"a": [pytest.approx(0.6)], "b": [-0.3]}
    with pytest.raises(ValueError, match="init pose lacks"):
        evidence.ainex_held_targets([(400, {"z": 0.1}), (400, {"z": 0.1})], init)


def test_ainex_smoke_group_holds_targets():
    """The record's smoke group holds frame targets (so its frames can show it played), and
    they are exactly the repeated non-zero offsets of consecutive frames."""
    _, frames, init = smoke_group()
    want = {}
    for (_, prev), (_, cur) in zip(frames, frames[1:]):
        for j, off in cur.items():
            if off != 0 and prev.get(j, 0.0) == off and init[j] + off not in want.get(j, []):
                want.setdefault(j, []).append(init[j] + off)
    held = evidence.ainex_held_targets(frames, init)
    assert want and held == {j: [pytest.approx(t) for t in ts] for j, ts in want.items()}


def follow(frames, init, clock, start, lag=0.2, end_offset=None):
    """Joint readings of a robot that plays `frames` from `start` (each frame's target reached
    linearly over its time, then the init pose), optionally ending `end_offset` away from it."""
    named = sorted({j for _, off in frames for j in off})
    targets = [{j: init[j] + off.get(j, 0.0) for j in named} for _, off in frames]

    def sample():
        t = clock.now - start - lag
        pos = dict(init)
        prev = {j: init[j] for j in named}
        for (ms, _), tg in zip(frames, targets):
            d = ms / 1000.0
            if t <= 0:
                break
            f = min(1.0, t / d)
            for j in named:
                pos[j] = prev[j] + f * (tg[j] - prev[j])
            prev = tg
            t -= d
        if t > 0 and end_offset:
            pos.update({j: init[j] + v for j, v in end_offset.items()})
        return pos
    return sample


def test_ainex_action_group_passes_when_played_to_the_init_pose(monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(evidence, "time", clock)
    _, frames, init = smoke_group()
    rec = evidence.ainex_play_and_judge(follow(frames, init, clock, clock.now), frames, init, 0.05)
    assert rec["problems"] == [] and rec["at_rest"], rec["problems"]
    assert all(v <= 0.05 for v in rec["held_targets_closest_rad"].values())
    assert set(rec["final_error_vs_init_pose_rad"]) == set(init)


def test_ainex_action_group_fails_when_not_played(monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(evidence, "time", clock)
    _, frames, init = smoke_group()
    rec = evidence.ainex_play_and_judge(lambda: dict(init), frames, init, 0.05)
    assert any("never came within" in p for p in rec["problems"]), rec["problems"]
    assert not rec["at_rest"]


def test_ainex_action_group_fails_off_the_recorded_init_pose(monkeypatch):
    """The end state is judged against the recorded init pose, within the tolerance."""
    clock = FakeTime()
    monkeypatch.setattr(evidence, "time", clock)
    _, frames, init = smoke_group()
    rec = evidence.ainex_play_and_judge(
        follow(frames, init, clock, clock.now, end_offset={"l_knee": 0.08}), frames, init, 0.05)
    assert any("not at the recorded init pose" in p and "l_knee" in p for p in rec["problems"]), \
        rec["problems"]


def test_ainex_action_group_without_held_targets_fails_loudly(monkeypatch):
    monkeypatch.setattr(evidence, "time", FakeTime())
    init = {"head_tilt": 0.0}
    nod = [(400, {"head_tilt": -0.4}), (400, {"head_tilt": 0.1}), (400, {})]
    with pytest.raises(ValueError, match="holds no frame target"):
        evidence.ainex_play_and_judge(lambda: dict(init), nod, init, 0.05)


def test_ainex_standing_height_is_read_from_the_record():
    """The body and its standing height at the init pose come from the record's figure;
    a record that drops them fails loudly."""
    tol = evidence.tolerance("ainex", "standing height")
    body, height = evidence.ainex_standing_height(tol)
    assert body == "body_link" and height == pytest.approx(0.203)
    assert tol["tolerance"]["absolute"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="standing height"):
        evidence.ainex_standing_height(dict(tol, notes="no manufacturer figure"))


# --------------------------------------------------------------------------- real-time factor


class FakeSimPort:
    """Answers the documented `rtf` operation (simulator/README.md) from scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def call(self, op, **fields):
        self.calls.append((op, fields))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_rtf_since_asks_the_documented_operation_until_the_end_is_covered(monkeypatch):
    clock = FakeTime()
    monkeypatch.setattr(evidence, "time", clock)
    a, b = clock.now - 30.0, clock.now
    early = {"rtf": 1.0, "window_s": 10, "warn_below": 0.9,
             "windows": [[a - 5, a + 5, 1.0], [a + 5, a + 15, 0.99], [a + 15, a + 25, 1.0]]}
    done = dict(early, windows=early["windows"] + [[a + 25, a + 35, 0.98]])
    sp = FakeSimPort([early, early, done])
    assert evidence.rtf_since(sp, a, b, wait=25.0) is done
    assert [c for c in sp.calls] == [("rtf", {"since": a})] * 3
    # without waiting: one call, whatever it lists
    sp = FakeSimPort([early])
    assert evidence.rtf_since(sp, a, b, wait=0.0) is early and len(sp.calls) == 1


def test_rtf_judgement_of_the_windows_over_the_collection():
    a = 1_700_000_000.0
    b = a + 20.0
    reply = {"rtf": 0.97, "window_s": 10, "warn_below": 0.9,
             "windows": [[a - 18, a, 0.5],          # ends at the start: not over it
                         [a - 8, a + 2, 0.98], [a + 2, a + 12, 0.85], [a + 12, a + 22, 0.99],
                         [a + 22, a + 32, 0.4]]}    # starts after the end: not over it
    j = evidence.rtf_judgement(reply, a, b)
    assert j["covered"] and len(j["windows"]) == 3
    assert [w["rtf"] for w in j["low"]] == [0.85]
    assert j["low"][0]["window_end"] == evidence.iso(a + 12)
    # the reply's own threshold is the one used
    assert [w["rtf"] for w in evidence.rtf_judgement(dict(reply, warn_below=0.99), a, b)["low"]] == [0.98, 0.85]
    # windows that stop before the collection's end do not cover it
    short = dict(reply, windows=reply["windows"][1:3])
    assert not evidence.rtf_judgement(short, a, b)["covered"]
    assert not evidence.rtf_judgement(dict(reply, windows=[]), a, b)["covered"]


def test_index_lists_the_harness_heuristics(tmp_path):
    """Thresholds with no recorded figure are stated in the index's 'How a case is judged'."""
    evidence.write_index(tmp_path)
    text = " ".join((tmp_path / "index.md").read_text().split())
    judged = text.split("## How a case is judged")[1].split("## Cases")[0]
    assert "Harness heuristics" in judged
    for h in evidence.HEURISTICS:
        assert " ".join(h.split()) in judged, h
    for v in (evidence.WALK_FALL_M, evidence.WALK_STILL_M, evidence.REST_SPEED_RADPS,
              evidence.STILL_RAD, evidence.AG_STILL_RAD):
        assert str(v) in judged, v
