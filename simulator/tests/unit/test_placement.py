"""The one placement function (spec §2.3 Placement, §5 Placement), on small scenes built
for each case, with real robot models -- and one synthetic robot whose turning circle
reaches far beyond its straight runs. (A worktop is a top-level body: the reference
survey looks at bodies, not at loose geoms of the world body.)"""

import math
import types

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


def robot_frame_to_world(spot, shape, p):
    """A point of the robot's own frame, with the robot standing at a survey spot."""
    xyz = np.array([spot.xy[0], spot.xy[1], spot.z - shape.lo[2] + 0.0005])
    c, s_ = math.cos(spot.yaw), math.sin(spot.yaw)
    return xyz + np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(p, float)


def peg_distance(rm, p):
    """Signed distance from a 4 mm sphere at robot-frame point `p` to the nearest collision
    geom of the robot alone at home, and whether `p` is inside one of its part boxes."""
    spec = rm.spec.copy()
    spec.worldbody.add_body(name="peg", pos=list(p)).add_geom(
        name="peg", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004, 0, 0])
    m = spec.compile()
    d = mujoco.MjData(m)
    for name, q in rm.home.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = q
    mujoco.mj_forward(m, d)
    peg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "peg")
    dist = min(mujoco.mj_geomDistance(m, d, peg, g, 0.5, np.zeros(6)) for g in range(m.ngeom)
               if g != peg and (m.geom_contype[g] or m.geom_conaffinity[g]))
    boxes = robot_model.shape(rm).boxes
    return dist, any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in boxes)


@pytest.mark.parametrize("peg,taken", [((0.102, 0.021, 0.055), True), (None, False)],
                         ids=["beside-a-part", "inside-a-part"])
def test_worktop_clearance_is_judged_on_the_collision_geometry(peg, taken):
    """Spec §2.3: a worktop robot's clearance is judged on its own collision geometry, not
    on its bounding box: a fixed 4 mm peg inside the box of one of its parts but clear of
    every collision geom leaves the survey's first spot to it; one inside a part does not."""
    rm = load("so101")
    if peg is None:   # a point inside the shoulder: its centre of mass
        m = rm.spec.copy().compile()
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        peg = d.xipos[b]
    dist, in_box = peg_distance(rm, peg)
    assert in_box
    assert (dist > 0) if taken else (dist < -0.001), dist
    _, s0 = make_world(scenes.TEST_SCENE_XML)
    first = s0.spots("so101")[0]
    at = robot_frame_to_world(first, robot_model.shape(rm), peg)
    w, s = make_world(scenes.TEST_SCENE_XML.replace(
        "</worldbody>", f'<geom name="peg" type="sphere" size="0.004" '
                        f'pos="{at[0]} {at[1]} {at[2]}"/></worldbody>'))
    # that spot alone (the survey itself would rank the spots differently with the peg)
    if taken:
        pl = placement.place(w, s, rm, "worktop", False, "so101/", spots=[first])
        assert np.allclose(pl.xyz[:2], first.xy) and pl.yaw == first.yaw
    else:
        with pytest.raises(placement.Refused, match="peg"):
            placement.place(w, s, rm, "worktop", False, "so101/", spots=[first])


def test_the_scene_objects_are_no_camera_obstacle_for_an_arm_among_them():
    """Spec §2.3 camera clearance excludes the robot's own worktop objects: an arm that
    fits where the scene's objects are stands there, even with one of them (moved by
    physics) right in front of its camera."""
    w, s = make_world(scenes.TEST_SCENE_XML)
    rm = load("so101")
    pl = placement.place(w, s, rm, "worktop", False, "so101/")
    w.stage_scene(pl.staging)
    shape = robot_model.shape(rm)
    _name, cpos, cmat, _fovy, _res = shape.cameras[0]
    spot = types.SimpleNamespace(xy=pl.xyz[:2], z=pl.surface_z, yaw=pl.yaw)
    target = robot_frame_to_world(spot, shape, cpos + 0.3 * -cmat[:, 2])
    mug = pl.staging.poses()["mug"][0]
    j = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_JOINT, "task_mug_joint")
    a = w.model.jnt_qposadr[j]
    w.data.qpos[a:a + 3] = [target[0], target[1], mug[2]]       # standing on the top
    mujoco.mj_forward(w.model, w.data)
    ctx = placement.Context(w, s, rm, shape, False)
    hits = placement._camera_rays(ctx, pl.xyz, pl.yaw, pl.surface_z, set(pl.support_geoms))
    assert any(h[1].startswith("task_mug") for h in hits), hits    # the camera does see it
    again = placement.place(w, s, rm, "worktop", False, "so101/")
    assert again.at_scene_objects and again.staging is None
    assert np.allclose(again.xyz[:2], w.scene_staging.frame_pos[:2])


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
    with pytest.raises(placement.Refused, match="placement refused: .*so101.*interpenetration: .*slab"):
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
    again = place(w, s, "myagv", "floor")      # repeatable for identical inputs
    assert np.array_equal(again.xyz, pl.xyz) and again.yaw == pl.yaw
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


def platform(hx, hy):
    """A floor of hx x hy half-extents and nothing around it: beyond it is unsupported."""
    return f'<geom name="floor" type="box" size="{hx} {hy} 0.05" pos="0 0 -0.05"/>'


def test_floor_refused_without_side_travel():
    # a strip 0.66 m wide: the myAGV fits, drives along it and turns on it, but has not
    # 0.25 m to either side
    w, s = make_world(scene_xml(floor=platform(2.0, 0.33)))
    with pytest.raises(placement.Refused, match="travel"):
        place(w, s, "myagv", "floor")
    assert w.robots == {}


def test_floor_refused_without_back_travel():
    # 1.0 x 0.95 m: room for the footprint, 0.5 m forward and 0.25 m to each side, not for
    # 0.25 m back as well, whichever way it faces
    w, s = make_world(scene_xml(floor=platform(0.5, 0.475)))
    with pytest.raises(placement.Refused, match="travel"):
        place(w, s, "myagv", "floor")
    assert w.robots == {}


STICK = """<mujoco><worldbody><body name="base"><freejoint name="root"/>
  <geom type="box" size="0.3 0.05 0.03" pos="0.3 0 0.03" mass="1"/></body></worldbody></mujoco>"""


def test_floor_refused_without_room_to_turn_in_place():
    # a 0.6 m robot turning about one end sweeps a 1.2 m circle; the strip, 0.9 m wide, has
    # room for its forward, back and side runs but not for that circle
    stick = robot_model.RobotModel(robot=types.SimpleNamespace(id="stick"),
                                   spec=mujoco.MjSpec.from_string(STICK), root="base",
                                   floating=True)
    w, s = make_world(scene_xml(floor=platform(3.0, 0.45)))
    with pytest.raises(placement.Refused, match="travel"):
        placement.place(w, s, stick, "floor", True, "stick/")
    assert w.robots == {}


def test_floor_refused_without_support():
    # a grating of 8 cm bars and 8 cm gaps: no footprint is supported anywhere
    bars = "".join(f'<geom type="box" size="0.04 1.5 0.05" pos="{-1.5 + 0.16 * k:.2f} 0 -0.05"/>'
                   for k in range(20))
    w, s = make_world(scene_xml(floor=bars))
    with pytest.raises(placement.Refused, match="support"):
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
