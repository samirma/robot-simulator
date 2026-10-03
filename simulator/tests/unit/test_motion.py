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
