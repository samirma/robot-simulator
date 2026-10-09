"""The motion checks read the tolerance the robot specification records for each figure
(spec §5 Acceptance bounds): one row per figure, never a neighbouring one."""

import math

import pytest

import motion
import registry
import wirecheck


def test_a_tolerance_is_the_one_row_naming_the_figure():
    assert motion.tol("ainex", "head joint")["absolute"] == 0.035
    # 'head' names both the head joint and the walk's heading drift: refused, not guessed
    with pytest.raises(KeyError):
        motion.tol("ainex", "head")


#: The travel a floor placement guarantees (simulator spec §2.3).
GUARANTEED = {"forward": 0.5, "back": 0.25, "side": 0.25}
DRIVEN = [rid for rid in registry.ids() if registry.get(rid).kind != "arm"
          and any(m["id"] == "drive" for m in wirecheck.interface(rid).get("motions") or [])]


def test_some_robot_drives():
    assert DRIVEN


@pytest.mark.parametrize("rid", DRIVEN)
def test_the_drive_stays_within_the_guaranteed_travel(rid):
    """Workspace spec §3: base motions stay within the travel the placement guarantees.
    After every leg of `motion.LEGS` the nominal pose, widened by that leg's recorded
    acceptance bound, is inside it."""
    lin_t, _, _ = motion.drive_tolerances(rid)
    x = y = yaw = 0.0
    for name, (vx, vy, wz), seconds in motion.LEGS:
        dx, dy = vx * seconds, vy * seconds
        x += dx * math.cos(yaw) - dy * math.sin(yaw)
        y += dx * math.sin(yaw) + dy * math.cos(yaw)
        yaw += wz * seconds
        b = max(lin_t.get("relative", 0.0) * math.hypot(dx, dy), lin_t.get("absolute", 0.0))
        assert x + b <= GUARANTEED["forward"] + 1e-9, (name, x, b)
        assert -x + b <= GUARANTEED["back"] + 1e-9, (name, x, b)
        assert abs(y) + b <= GUARANTEED["side"] + 1e-9, (name, y, b)


def test_one_held_command_outlasts_a_watchdog():
    """Spec §5 Motion and timing: no recorded drive has a command watchdog, so one leg holds
    a single command for 4-5 s, and its base must still move at the end."""
    long = [(name, seconds) for name, _, seconds in motion.LEGS
            if seconds >= motion.WATCHDOG_LEG_S]
    assert long and all(4.0 <= s <= 5.0 for _, s in long)
    for rid in DRIVEN:
        drive = next(m for m in wirecheck.interface(rid)["motions"] if m["id"] == "drive")
        assert drive["watchdog"]["present"] is False, rid


@pytest.mark.parametrize("rid", DRIVEN)
def test_odometry_is_judged_against_the_simulated_base(rid, monkeypatch):
    """Spec §5 Sensors (outputs correspond to the simulated world): a leg whose odometry
    reports the commanded displacement while the simulated base moved otherwise -- a wheel
    spinning in the air, a base pushed by something else -- fails, as does a long leg whose
    base has stopped before its stop command."""
    lin_t, yaw_t, res = motion.drive_tolerances(rid)

    def legs(slip=1.0, moving=1.0):
        out = {}
        for name, (vx, vy, wz), seconds in motion.LEGS:
            fwd, lat, yaw = vx * seconds, vy * seconds, wz * seconds
            out[name] = {"cmd": (vx, vy, wz), "t_run": seconds, "fwd": fwd, "lat": lat,
                         "yaw": yaw, "sim": (fwd * slip, lat * slip, yaw * slip),
                         "moving": (moving * math.hypot(vx, vy)
                                    if seconds >= motion.WATCHDOG_LEG_S else None),
                         "residual": 0.0}
        return out

    monkeypatch.setattr(motion, "drive_legs", lambda *a, **kw: legs())
    assert motion.drive(0, rid, 0) == []
    monkeypatch.setattr(motion, "drive_legs", lambda *a, **kw: legs(slip=0.5))
    problems = motion.drive(0, rid, 0)
    assert any("the simulated base moved" in p for p in problems), problems
    assert any("the simulated base turned" in p for p in problems), problems
    monkeypatch.setattr(motion, "drive_legs", lambda *a, **kw: legs(moving=0.1))
    problems = motion.drive(0, rid, 0)
    assert problems and all("watchdog" in p for p in problems), problems


def test_a_leg_run_below_real_time_is_not_claimed(monkeypatch):
    """Spec §3 Timing: the odometry integrates in wall time and the base moves in simulated
    time, so a leg over which the simulation ran below 0.90 of real time claims nothing --
    it fails, saying so, instead of being judged."""
    rid = DRIVEN[0]
    legs = {}
    for name, (vx, vy, wz), seconds in motion.LEGS:
        legs[name] = {"cmd": (vx, vy, wz), "t_run": seconds, "fwd": vx * seconds,
                      "lat": vy * seconds, "yaw": wz * seconds,
                      "sim": (0.8 * vx * seconds, 0.8 * vy * seconds, 0.8 * wz * seconds),
                      "moving": None, "residual": 0.0, "rtf": 0.8}
    monkeypatch.setattr(motion, "drive_legs", lambda *a, **kw: legs)
    problems = motion.drive(0, rid, 0)
    assert len(problems) == len(motion.LEGS) and all("real-time factor 0.800" in p
                                                       for p in problems), problems
