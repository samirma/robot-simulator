"""The one placement function (spec §2.3 Placement, §5 Placement), on small scenes built
for each case, with real robot models. (A worktop is a top-level body: the reference
survey looks at bodies, not at loose geoms of the world body.)"""

import math

import mujoco
import numpy as np
import pytest

import placement
import registry
import robot_model
import scenes
import worktop_objects
from world import Instance, World


def scene_xml(extra: str = "", floor: str = '<geom name="floor" type="plane" size="4 4 .1"/>'):
    return f"""<mujoco><option timestep="0.002"/><worldbody>
      <light pos="0 0 3" dir="0 0 -1" directional="true"/>{floor}{extra}</worldbody></mujoco>"""


def make_world(xml):
    sc = scenes.Scene("molmospaces", "test", "x", mujoco.MjSpec.from_string(xml))
    w = World(sc)
    return w, placement.SceneSurfaces(w.model, w.data)


def load(rid):
    return robot_model.load(registry.get(rid))


def place(w, s, rid, where):
    r = registry.get(rid)
    return placement.place(w, s, load(rid), where, r.mobile, rid + "/")


def add(w, rid, pl):
    inst = Instance(rid, load(rid), rid + "/", pl.surface, pl.xyz, pl.yaw)
    w.add(inst, staging=pl.staging)
    return inst


WORKTOP = '<body name="table" pos="0 0 0"><geom type="box" size="0.6 0.4 0.375" pos="1.5 0 0.375"/></body>'
BOXES = "".join(f'<body name="box_{i}_{j}" pos="{1.5 + x:.2f} {y:.2f} 0.80"><freejoint/>'
                f'<geom type="box" size="0.04 0.04 0.04" mass="0.1"/></body>'
                for i, x in enumerate(np.arange(-0.55, 0.56, 0.1))
                for j, y in enumerate(np.arange(-0.35, 0.36, 0.1)))


def test_worktop_is_the_survey_surface():
    w, s = make_world(scene_xml(WORKTOP))
    assert abs(s.floor_z) < 1e-6
    assert s.worktop is not None and s.worktop.name == "table"
    assert abs(s.worktop.z - 0.75) < 1e-9
    d = s.worktop.describe()
    assert abs(d["area_m2"] - 0.96) < 1e-6 and d["bounds"] == [0.9, -0.4, 2.1, 0.4]


def test_no_worktop_outside_table_height():
    # the reference survey's band: a fixed top between 0.35 m and 1.30 m
    low = '<body name="low"><geom type="box" size="0.6 0.4 0.15" pos="1.5 0 0.15"/></body>'
    high = '<body name="high"><geom type="box" size="0.6 0.4 0.7" pos="-1.5 0 0.7"/></body>'
    w, s = make_world(scene_xml(low + high))
    assert s.worktop is None
    with pytest.raises(placement.Refused, match="no worktop"):
        place(w, s, "so101", "worktop")
    assert w.robots == {}


def test_arm_on_the_worktop_at_the_survey_spot_and_repeatable():
    w, s = make_world(scene_xml(WORKTOP))
    a = place(w, s, "so101", "worktop")
    b = place(w, s, "so101", "worktop")
    assert np.array_equal(a.xyz, b.xyz) and a.yaw == b.yaw
    first = s.spots("so101")[0]
    assert np.array_equal(a.xyz[:2], first.xy) and a.yaw == first.yaw
    assert a.surface_z == 0.75 and abs(a.xyz[2] - 0.75) < 0.003
    assert a.staging is not None and a.staging.cleared == []
    assert sorted(a.staging.poses()) == sorted(["apple", "plate", "bowl", "mug", "banana", "lemon"])


def test_occupied_worktop_is_refused():
    w, s = make_world(scene_xml(WORKTOP))
    add(w, "so101", place(w, s, "so101", "worktop"))
    with pytest.raises(placement.Refused, match="already holds so101"):
        place(w, s, "mycobot280", "worktop")
    assert list(w.robots) == ["so101"]


def test_worktop_with_no_clear_spot_is_refused():
    # a robot that brings no staging needs a spot already clear
    w, s = make_world(scene_xml(WORKTOP + BOXES))
    for _ in range(300):
        mujoco.mj_step(w.model, w.data)
    with pytest.raises(placement.Refused, match="no clear spot"):
        place(w, s, "myagv", "worktop")
    assert w.robots == {}


def test_worktop_robot_clears_the_loose_objects_in_its_working_area():
    w, s = make_world(scene_xml(WORKTOP + BOXES))
    for _ in range(300):
        mujoco.mj_step(w.model, w.data)
    pl = place(w, s, "so101", "worktop")
    near = []
    inv = np.linalg.inv(worktop_objects.base_frame(pl.staging.frame_pos, pl.staging.yaw))
    for b in range(w.model.nbody):
        name = w.model.body(b).name
        if name.startswith("box_"):
            local = inv @ np.array([*w.data.xpos[b], 1.0])
            if np.hypot(local[0], local[1]) <= worktop_objects.CLEAR_RADIUS:
                near.append(name)
    assert near and pl.staging.cleared == near


def test_arm_refused_when_every_spot_intersects_the_scene():
    # a slab 0.15 m over the whole table (a fixed world geom, so not a surface the survey
    # would stand the arm on): no arm fits under it
    slab = '<geom name="slab" type="box" size="0.7 0.5 0.01" pos="1.5 0 0.91"/>'
    w, s = make_world(scene_xml(WORKTOP + slab))
    with pytest.raises(placement.Refused, match="no clear spot on the worktop for so101.*slab"):
        place(w, s, "so101", "worktop")
    assert w.robots == {}


def test_arm_refused_when_its_objects_would_not_stand_on_the_worktop():
    # room for the arm, not for the six objects around it
    small = '<body name="stool"><geom type="box" size="0.15 0.15 0.375" pos="1.5 0 0.375"/></body>'
    w, s = make_world(scene_xml(small))
    assert s.worktop is not None
    with pytest.raises(placement.Refused, match="staged .* unsupported"):
        place(w, s, "so101", "worktop")
    assert w.robots == {}


def test_mobile_robot_on_a_worktop_too_small_for_its_travel_is_placed():
    """No travel requirement on the worktop (spec §2.3, amended 2026-10-02): the robot
    stands on a top too small to drive on, supported and clear; on the floor it is held to
    its travel."""
    small = '<body name="small"><geom type="box" size="0.25 0.25 0.375" pos="1.5 0 0.375"/></body>'
    w, s = make_world(scene_xml(small))
    assert s.worktop is not None
    pl = place(w, s, "myagv", "worktop")
    assert pl.surface == "worktop" and abs(pl.surface_z - 0.75) < 1e-6


def test_floor_placement_supports_travel_and_faces_open_floor():
    w, s = make_world(scene_xml(WORKTOP))
    pl = place(w, s, "myagv", "floor")
    assert abs(pl.surface_z) < 1e-6
    fmap = placement.Context(w, s, load("myagv"), robot_model.shape(load("myagv")), True).full_map(0.0)
    runs = {k: placement._open_run(fmap, s.static_map, 0.0, pl.xyz[:2], k * math.pi / 8)
            for k in range(16)}
    assert placement._open_run(fmap, s.static_map, 0.0, pl.xyz[:2], pl.yaw) >= max(runs.values()) - 1e-9
    # the robot compiled in at that pose touches nothing but the floor
    inst = add(w, "myagv", pl)
    mujoco.mj_forward(w.model, w.data)
    own = w.robot_body_ids(inst)
    floor = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for c in w.data.contact[:w.data.ncon]:
        a, b = w.model.geom_bodyid[c.geom1] in own, w.model.geom_bodyid[c.geom2] in own
        if a != b:
            other = c.geom2 if a else c.geom1
            assert other == floor or c.dist > -0.001


def test_floor_without_room_to_travel_is_refused():
    # a 0.6 m x 0.6 m pen: the myAGV fits but cannot drive 0.5 m forward
    walls = "".join(f'<geom type="box" size="{sx} {sy} 0.3" pos="{px} {py} 0.3"/>'
                    for sx, sy, px, py in ((0.05, 0.4, 0.35, 0), (0.05, 0.4, -0.35, 0),
                                           (0.4, 0.05, 0, 0.35), (0.4, 0.05, 0, -0.35)))
    floor = '<geom name="floor" type="box" size="0.4 0.4 0.05" pos="0 0 -0.05"/>'
    w, s = make_world(scene_xml(walls, floor=floor))
    with pytest.raises(placement.Refused, match="travel|support"):
        place(w, s, "myagv", "floor")
    assert w.robots == {}


def test_camera_clearance_counts_furniture_but_not_the_support_surface():
    w, s = make_world(scene_xml())
    rm = load("myagv")
    ctx = placement.Context(w, s, rm, robot_model.shape(rm), True)
    floor = frozenset(g for g in range(w.model.ngeom) if w.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
    assert placement._camera_rays(ctx, np.zeros(3), 0.0, 0.0, floor) == []
    w2, s2 = make_world(scene_xml('<geom name="cupboard" type="box" size="0.05 0.5 0.5" pos="0.6 0 0.5"/>'))
    ctx2 = placement.Context(w2, s2, rm, robot_model.shape(rm), True)
    hits = placement._camera_rays(ctx2, np.zeros(3), 0.0, 0.0, floor)
    assert hits and all(h[1] == "cupboard" and h[2] <= placement.CAMERA_CLEARANCE for h in hits)


def test_camera_clearance_refusal():
    # a pen just large enough for the myAGV's travel paths: every spot and heading that
    # travels leaves a wall within 0.8 m of the camera
    # (a round pen: a square one leaves its diagonals clear of the camera's 0.8 m)
    r, n = 0.70, 32
    walls = "".join(
        f'<geom type="box" size="0.03 {r * math.tan(math.pi / n) + 0.01:.4f} 0.3" '
        f'pos="{r * math.cos(2 * math.pi * k / n):.4f} {r * math.sin(2 * math.pi * k / n):.4f} 0.3" '
        f'euler="0 0 {2 * math.pi * k / n:.5f}"/>' for k in range(n))
    floor = '<geom name="floor" type="box" size="0.9 0.9 0.05" pos="0 0 -0.05"/>'
    w, s = make_world(scene_xml(walls, floor=floor))
    with pytest.raises(placement.Refused, match="camera clearance: .*camera camera_link sees"):
        place(w, s, "myagv", "floor")
    assert w.robots == {}


def test_second_floor_robot_keeps_clear_of_the_first():
    w, s = make_world(scene_xml(WORKTOP))
    a = add(w, "myagv", place(w, s, "myagv", "floor"))
    pl = place(w, s, "rosmaster_x3_plus", "floor")
    assert np.linalg.norm(pl.xyz[:2] - a.xyz[:2]) > 0.3
