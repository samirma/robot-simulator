#!/usr/bin/env python
"""Self-test for `find_tabletop_mount`: the arm goes at the rim, looking in.

Standalone rather than pytest, like the robot self-tests -- a failure here points at the
placement rule and nothing else. It runs on synthetic surfaces (rectangles, no compiled
model), which is the whole point: a rule that only holds on FloorPlan1's island is not a
rule, and a scene big enough to exercise it is a 13 GB download.

    python tools/test_placement.py
"""

from __future__ import annotations

import sys

import numpy as np

from scene_placement import GraspTarget, _rects_covering, find_tabletop_mount

REACH = (0.15, 0.35)
FOOTPRINT = 0.12
BODY_RADIUS = 0.13


def surface(half_x: float, half_y: float, objects, top_z: float = 0.92) -> GraspTarget:
    """A single rectangular worktop centred on the origin, carrying `objects`."""
    lo = np.array([-half_x, -half_y])
    hi = np.array([half_x, half_y])
    rects = np.array([[lo[0], lo[1], hi[0], hi[1], top_z]], dtype=float)
    on_support = tuple(
        (np.array([x, y, top_z + 0.03]), np.array([0.06, 0.06, 0.06])) for x, y in objects
    )
    first = objects[0] if objects else (0.0, 0.0)
    return GraspTarget(
        support_name="worktop",
        support_category="counter",
        support_top_z=top_z,
        object_name="apple_1",
        object_category="apple",
        object_xyz=np.array([first[0], first[1], top_z + 0.03]),
        n_objects_on_support=len(on_support),
        reach_slack=0.0,
        support_xy_min=lo,
        support_xy_max=hi,
        objects_on_support=on_support,
        support_top_rects=rects,
    )


def mount(target: GraspTarget, **kwargs):
    return find_tabletop_mount(
        target,
        reach_range=REACH,
        footprint=FOOTPRINT,
        body_radius=BODY_RADIUS,
        obstacle_clearance=0.08,
        **kwargs,
    )


def heading(m) -> np.ndarray:
    return np.array([np.cos(m.yaw), np.sin(m.yaw)])


FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_narrow_counter_mounts_at_the_back():
    """A 0.70 m deep counter: the arm belongs against one long side, looking across it.

    In the middle it reaches the front lip and the air past it; the coverage score is what
    says so without anyone having to encode "back edge" as a rule.
    """
    # Objects along the middle of the run, so nothing biases the choice along y.
    target = surface(0.55, 0.35, [(-0.15, 0.0), (0.15, 0.0)])
    m = mount(target)
    # Inset is body_radius from the rim, so |y| near 0.22 is "against a long side".
    at_side = abs(m.xy[1]) > 0.15
    looks_across = abs(heading(m)[1]) > 0.9 and np.sign(heading(m)[1]) == -np.sign(m.xy[1])
    check(
        "narrow counter: mounted against a long side",
        at_side,
        f"xy=({m.xy[0]:.2f}, {m.xy[1]:.2f})",
    )
    check(
        "narrow counter: looking across the worktop, not along or off it",
        looks_across,
        f"yaw={np.degrees(m.yaw):.0f} deg",
    )
    check("narrow counter: workspace on the worktop", m.coverage > 0.9, f"{m.coverage:.0%}")


def test_island_corner_is_not_chosen_over_an_edge():
    """A free-standing island with one object on it.

    The old rule faced the object and took the rim cell nearest it, which on a corner
    meant a heading pointing diagonally off the surface. Coverage has to reject that.
    """
    target = surface(0.60, 0.60, [(0.40, 0.40)])
    m = mount(target)
    off_surface = (m.xy + 0.35 * heading(m))
    inside = (abs(off_surface) < np.array([0.60, 0.60])).all()
    check(
        "island: the heading keeps the working annulus on the surface",
        inside and m.coverage > 0.9,
        f"xy=({m.xy[0]:.2f}, {m.xy[1]:.2f}) yaw={np.degrees(m.yaw):.0f} deg "
        f"coverage={m.coverage:.0%}",
    )


def test_l_shaped_counter_ignores_its_own_notch():
    """The crook of an L is open floor, and the AABB the grid is cut from includes it.

    This is FloorPlan1's counter in miniature: MuJoCo's bounding box for it is a square
    covering the whole kitchen. Both the cell test and the coverage score have to work
    from the top-face rectangles, or the arm gets bolted into mid-air over the notch --
    or, more quietly, mounted correctly and pointed across it.
    """
    lo = np.array([-0.90, -0.70])
    hi = np.array([0.90, 0.70])
    rects = np.array(
        [
            [-0.90, 0.00, 0.90, 0.70, 0.92],   # the run along the wall
            [0.55, -0.70, 0.90, 0.00, 0.92],   # the return
        ],
        dtype=float,
    )
    target = surface(0.90, 0.70, [(0.0, 0.30)])
    target = GraspTarget(
        **{
            **target.__dict__,
            "support_xy_min": lo,
            "support_xy_max": hi,
            "support_top_rects": rects,
        }
    )
    m = mount(target)
    on_rect = _rects_covering(m.xy, rects).any()
    ahead = m.xy + 0.35 * heading(m)
    check(
        "L-shape: the base is over the worktop, not the notch",
        on_rect,
        f"xy=({m.xy[0]:.2f}, {m.xy[1]:.2f})",
    )
    check(
        "L-shape: it does not face across the notch",
        _rects_covering(ahead, rects).any() and m.coverage > 0.85,
        f"yaw={np.degrees(m.yaw):.0f} deg coverage={m.coverage:.0%}",
    )


def test_a_cell_that_reaches_nothing_is_not_preferred():
    """Coverage is the first key, but never at the cost of the surface being usable.

    `place_arm_on_table` drops a surface whose best mount reaches nothing, so a cell with
    perfect coverage and no object in the annulus must not win over a slightly worse one
    that can work.
    """
    # One object tucked into a corner: the cells that reach it are worse-covered than the
    # empty middle of the far side.
    target = surface(0.90, 0.45, [(0.70, 0.25)])
    m = mount(target)
    check(
        "usability: the chosen cell can reach something",
        m.n_in_reach > 0,
        f"n_in_reach={m.n_in_reach} coverage={m.coverage:.0%}",
    )


def main() -> int:
    for test in (
        test_narrow_counter_mounts_at_the_back,
        test_island_corner_is_not_chosen_over_an_edge,
        test_l_shaped_counter_ignores_its_own_notch,
        test_a_cell_that_reaches_nothing_is_not_preferred,
    ):
        print(f"\n-- {test.__name__}")
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
