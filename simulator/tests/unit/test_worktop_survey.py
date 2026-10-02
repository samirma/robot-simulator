"""The worktop survey (spec §2.2, §2.3): the reference project's surface ranking and spot
search, on small scenes built for each case. Its equality with the reference itself on
the engines' default scenes is `integration/test_reference_parity.py`."""

import math

import mujoco
import numpy as np

import worktop_survey as ws


def facts(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return ws.Facts("generic", m, d)


def scene(extra=""):
    return f"""<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="table"><geom type="box" size="0.6 0.4 0.375" pos="1.5 0 0.375"/></body>
      {extra}</worldbody></mujoco>"""


APPLE_ON_SIDE = """
      <body name="side_table"><geom type="box" size="0.3 0.3 0.4" pos="-1.5 0 0.4"/></body>
      <body name="apple_1" pos="-1.5 0.1 0.83"><freejoint/>
        <geom type="sphere" size="0.03" mass="0.1"/></body>"""


def test_the_worktop_is_the_largest_table_height_top_when_nothing_is_on_any():
    f = facts(scene('<body name="stool"><geom type="box" size="0.2 0.2 0.3" pos="-1 2 0.3"/></body>'))
    wt = ws.worktop(f)
    assert wt.name == "table" and wt.z == 0.75
    assert wt.contains((1.5, 0.0)) and not wt.contains((-1.0, 2.0))


def test_a_surface_holding_the_task_categories_ranks_first():
    f = facts(scene(APPLE_ON_SIDE))
    assert ws.worktop(f).name == "side_table"
    first = ws.candidates(f, "so101")[0]
    assert first.surface == "side_table" and abs(first.z - 0.8) < 1e-9


def test_spots_are_deterministic_on_the_surface_and_inside_its_edges():
    f = facts(scene())
    a, b = ws.candidates(f, "so101"), ws.candidates(facts(scene()), "so101")
    assert [(s.xy.tolist(), s.z, s.yaw) for s in a] == [(s.xy.tolist(), s.z, s.yaw) for s in b]
    inset = ws.MOUNT["so101"].footprint / 2 + 0.02
    for s in a:
        assert 0.9 + inset - 1e-9 <= s.xy[0] <= 2.1 - inset + 1e-9
        assert -0.4 + inset - 1e-9 <= s.xy[1] <= 0.4 - inset + 1e-9
        assert s.z == 0.75


def test_the_first_spot_faces_the_most_worktop():
    """REF's coverage rule: of the headings at the chosen cell, none puts more of the arm's
    forward workspace over the worktop."""
    f = facts(scene())
    m = ws.MOUNT["so101"]
    s = ws.candidates(f, "so101")[0]
    offsets = ws._workspace_offsets(m.reach, m.workspace_radius)

    def coverage(yaw):
        c, sn = math.cos(yaw), math.sin(yaw)
        pts = s.xy + offsets @ np.array([[c, sn], [-sn, c]])
        return np.mean([(0.9 < x < 2.1) and (-0.4 < y < 0.4) for x, y in pts])

    best = max(coverage(k * 2 * math.pi / 24) for k in range(24))
    # (REF reads coverage off a 5 cm lookup grid; this reads it exactly)
    assert coverage(s.yaw) >= best - 0.06


def test_a_robot_without_a_mount_has_no_spots():
    assert ws.candidates(facts(scene()), "myagv") == []


def test_counter_mount_against_the_back_edge_facing_the_room():
    """REF's RoboCasa rule on a synthetic counter: 1.0 x 0.6 m, its back against a wall
    at y = +0.3, the room at -y."""
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
      <geom name="wall" type="box" size="2 0.05 1.2" pos="0 0.35 1.2"/>
      <body name="counter"><geom type="box" size="0.5 0.3 0.45" pos="0 0 0.45"/></body>
      </worldbody></mujoco>"""
    f = facts(xml)
    f.kind = "robocasa"
    f.regions = [{"name": "counter/top", "centre": np.array([0.0, 0.0]),
                  "half": np.array([0.5, 0.3]), "top_z": 0.9, "rot": 0.0}]
    spots = ws.candidates(f, "so101")
    first = spots[0]
    assert np.allclose(first.xy, [0.0, 0.3 - 0.2]) and abs(first.yaw + math.pi / 2) < 1e-12
    assert first.z == 0.9
    # then REF's tabletop search over the counter, then the front edge facing the wall
    assert any("chosen for" in s.why for s in spots[1:])
    front = [s for s in spots if s.why.startswith("front edge")]
    assert front and np.allclose(front[0].xy, [0.0, -0.3 + ws.MOUNT["so101"].footprint / 2])
    assert abs(front[0].yaw - math.pi / 2) < 1e-12
    assert ws.worktop(f).name == "counter/top"
