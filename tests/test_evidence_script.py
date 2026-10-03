"""Unit checks of the workspace evidence script (`tests/evidence.py`) that start nothing:
its command line, its reading of the readiness line, the travel its base motions take, the
motions and scenes a case requires (workspace spec §3) and the projection that checks a
scene picture shows the worktop objects.

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


#: Workspace spec §3's smoke-run motions, by robot: a wheeled base drives forward, back,
#: sideways and turns; the ROSMASTER X3 PLUS also moves its arm and gripper; the SO-101 and
#: the myCobot 280 their arm and gripper; the AiNex turns its head (each head joint it
#: records), walks and stops, and plays an action group.
DRIVE = {"drive_forward", "drive_back", "drive_left", "drive_right", "drive_turn"}
SPEC_MOTIONS = {
    "myagv": DRIVE,
    "rosmaster_x3_plus": DRIVE | {"arm", "gripper"},
    "so101": {"arm", "gripper"},
    "mycobot280": {"arm", "gripper"},
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
