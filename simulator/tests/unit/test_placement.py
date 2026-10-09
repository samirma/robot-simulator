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
import worktop_survey
from world import Instance, World

#: the worktop robot of the robot specification
ARM = "rebot_b601_rs"


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
#: a top with room for the arm, not for the six objects around it
STOOL = '<body name="stool"><geom type="box" size="0.15 0.15 0.375" pos="1.5 0 0.375"/></body>'
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
        place(w, s, ARM, "worktop")
    assert w.robots == {}


def test_arm_on_the_worktop_at_the_survey_spot_and_repeatable():
    """The scene as `start` leaves it, its six objects at the survey's first spot: the arm
    stands there among them, bringing and clearing nothing, the same for identical inputs."""
    w, s = make_world(scene_xml(WORKTOP))
    w.stage_scene(placement.stage_objects(w, s))
    a = place(w, s, ARM, "worktop")
    b = place(w, s, ARM, "worktop")
    assert np.array_equal(a.xyz, b.xyz) and a.yaw == b.yaw
    first = s.spots()[0]
    assert np.array_equal(a.xyz[:2], first.xy) and a.yaw == first.yaw
    assert a.surface_z == 0.75 and abs(a.xyz[2] - 0.75) < 0.003
    assert a.at_scene_objects and a.staging is None
    assert sorted(w.scene_staging.poses()) == sorted(["apple", "plate", "bowl", "mug", "banana",
                                                      "lemon"])


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


@pytest.mark.parametrize("peg,taken", [((0.072, 0.073, 0.114), True), (None, False)],
                         ids=["beside-a-part", "inside-a-part"])
def test_worktop_clearance_is_judged_on_the_collision_geometry(peg, taken):
    """Spec §2.3: a worktop robot's clearance is judged on its own collision geometry, not
    on its bounding box: a fixed 4 mm peg inside the box of one of its parts but clear of
    every collision geom leaves the survey's first spot to it; one inside a part does not."""
    rm = load(ARM)
    if peg is None:   # a point inside the first link: its centre of mass
        m = rm.spec.copy().compile()
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link1")
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        peg = d.xipos[b]
    dist, in_box = peg_distance(rm, peg)
    assert in_box
    assert (dist > 0) if taken else (dist < -0.001), dist
    _, s0 = make_world(scenes.TEST_SCENE_XML)
    first = s0.spots()[0]
    at = robot_frame_to_world(first, robot_model.shape(rm), peg)
    w, s = make_world(scenes.TEST_SCENE_XML.replace(
        "</worldbody>", f'<geom name="peg" type="sphere" size="0.004" '
                        f'pos="{at[0]} {at[1]} {at[2]}"/></worldbody>'))
    # that spot alone (the survey itself would rank the spots differently with the peg)
    if taken:
        pl = placement.place(w, s, rm, "worktop", False, ARM + "/", spots=[first])
        assert np.allclose(pl.xyz[:2], first.xy) and pl.yaw == first.yaw
    else:
        with pytest.raises(placement.Refused, match="peg"):
            placement.place(w, s, rm, "worktop", False, ARM + "/", spots=[first])


def with_camera(rm):
    """The arm with a camera on its base looking forward and down (the reBot Arm B601-RS
    has none; camera clearance applies to every camera an arm has)."""
    rm.spec.body(rm.root).add_camera(name="eye", pos=[0.05, 0.0, 0.25], fovy=60.0,
                                     xyaxes=[0, -1, 0, 0.7071, 0, 0.7071])
    return rm


def test_the_scene_objects_are_no_camera_obstacle_for_an_arm_among_them():
    """Spec §2.3 camera clearance excludes the robot's own worktop objects: an arm that
    fits where the scene's objects are stands there, even with one of them (moved by
    physics) right in front of its camera."""
    w, s = make_world(scenes.TEST_SCENE_XML)
    rm = with_camera(load(ARM))
    w.stage_scene(placement.stage_objects(w, s))
    pl = placement.place(w, s, rm, "worktop", False, ARM + "/")
    assert pl.at_scene_objects
    shape = robot_model.shape(rm)
    _name, cpos, cmat, _fovy, _res = shape.cameras[0]
    spot = types.SimpleNamespace(xy=pl.xyz[:2], z=pl.surface_z, yaw=pl.yaw)
    target = robot_frame_to_world(spot, shape, cpos + 0.3 * -cmat[:, 2])
    mug = w.scene_staging.poses()["mug"][0]
    j = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_JOINT, "task_mug_joint")
    a = w.model.jnt_qposadr[j]
    w.data.qpos[a:a + 3] = [target[0], target[1], mug[2]]       # standing on the top
    mujoco.mj_forward(w.model, w.data)
    # the camera does see it, within the clearance distance (rays that counted every geom
    # would hit it), and that is no hit
    ctx = placement.Context(w, s, rm, shape, False)
    ctx._counted = np.ones(w.model.ngeom, bool)
    hits = placement._camera_rays(ctx, pl.xyz, pl.yaw, pl.surface_z, set(pl.support_geoms))
    assert any(h[1].startswith("task_mug") for h in hits), hits
    ctx = placement.Context(w, s, rm, shape, False)
    assert placement._camera_rays(ctx, pl.xyz, pl.yaw, pl.surface_z, set(pl.support_geoms)) == []
    again = placement.place(w, s, rm, "worktop", False, ARM + "/")
    assert again.at_scene_objects and again.staging is None
    assert np.allclose(again.xyz[:2], w.scene_staging.frame_pos[:2])


def test_occupied_worktop_is_refused():
    w, s = make_world(scene_xml(WORKTOP))
    add(w, ARM, place(w, s, ARM, "worktop"))
    with pytest.raises(placement.Refused, match=f"already holds {ARM}"):
        place(w, s, "myagv", "worktop")
    assert list(w.robots) == [ARM]


def test_worktop_with_no_clear_spot_is_refused():
    # a robot that brings no staging needs a spot already clear
    w, s = make_world(scene_xml(WORKTOP + BOXES))
    for _ in range(300):
        mujoco.mj_step(w.model, w.data)
    with pytest.raises(placement.Refused, match="no clear spot"):
        place(w, s, "myagv", "worktop")
    assert w.robots == {}


def test_staging_clears_the_loose_objects_in_its_working_area():
    """The scene's six objects are staged with the loose objects in their working area
    cleared (spec §2.2): on a top covered with boxes, exactly those within the clearing
    radius; an arm then stands among the objects, clearing nothing more."""
    w, s = make_world(scene_xml(WORKTOP + BOXES))
    for _ in range(300):
        mujoco.mj_step(w.model, w.data)
    st = placement.stage_objects(w, s)
    near = []
    inv = np.linalg.inv(worktop_objects.base_frame(st.frame_pos, st.yaw))
    for b in range(w.model.nbody):
        name = w.model.body(b).name
        if name.startswith("box_"):
            local = inv @ np.array([*w.data.xpos[b], 1.0])
            if np.hypot(local[0], local[1]) <= worktop_objects.CLEAR_RADIUS:
                near.append(name)
    assert near and st.cleared == near
    w.stage_scene(st)
    pl = place(w, s, ARM, "worktop")
    assert pl.at_scene_objects and pl.staging is None


def test_arm_refused_when_every_spot_intersects_the_scene():
    # a slab 0.15 m over the whole table (a fixed world geom, so not a surface the survey
    # would stand the arm on): no arm fits under it. The refusal names the slab, whether as
    # the arm's interpenetration or as what its wrist camera (which looks forward and down
    # over the worktop) sees within the camera clearance.
    slab = '<geom name="slab" type="box" size="0.7 0.5 0.01" pos="1.5 0 0.91"/>'
    w, s = make_world(scene_xml(WORKTOP + slab))
    with pytest.raises(placement.Refused, match=f"placement refused: .*{ARM}.*(interpenetration: .*slab|sees slab)"):
        place(w, s, ARM, "worktop")
    assert w.robots == {}


def block_the_scene_objects_spot(w):
    """A fixed post where the arm's base would stand among the scene's objects, put into
    the live world after `start` staged them: no arm fits there any more."""
    x, y = map(float, w.scene_staging.frame_pos[:2])
    with w.lock:
        w.spec.worldbody.add_body(name="post", pos=[x, y, 0.85]).add_geom(
            name="post", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.09])
        m, d = w.spec.recompile(w.model, w.data)
        mujoco.mj_forward(m, d)
        w._swap(m, d)


def test_arm_refused_when_its_objects_would_not_stand_on_the_worktop():
    """An arm that does not fit where the scene's objects are takes them to a spot where it
    fits (spec §2.3): a spot with room for the arm but not for the six objects around it --
    on a stool beside the table -- is refused."""
    stool = STOOL.replace('pos="1.5 0 0.375"', 'pos="-1.5 0 0.375"')
    alone = make_world(scene_xml(stool))
    on_stool = place(*alone, ARM, "worktop")                 # the arm alone fits there
    assert on_stool.staging is None and on_stool.surface == "worktop"
    spot = worktop_survey.Spot(xy=on_stool.xyz[:2].copy(), z=on_stool.surface_z,
                               yaw=on_stool.yaw, surface="stool", why="the arm alone fits")
    w, s = make_world(scene_xml(WORKTOP + stool))
    w.stage_scene(placement.stage_objects(w, s))
    block_the_scene_objects_spot(w)
    with pytest.raises(placement.Refused, match="staged .* unsupported"):
        placement.place(w, s, load(ARM), "worktop", False, ARM + "/", spots=[spot])
    assert w.robots == {}


def free_state(w, body):
    b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, body)
    j = w.model.body_jntadr[b]
    if w.model.body_jntnum[b] == 0:
        return None
    a, v = w.model.jnt_qposadr[j], w.model.jnt_dofadr[j]
    return w.data.qpos[a:a + 7].copy(), w.data.qvel[v:v + 6].copy()


#: a ledge beside the stool (a geom of the world body: no surface the survey stands
#: anything on) holding a loose box within the area a staging on the stool would clear
LEDGE = ('<geom name="ledge" type="box" size="0.15 0.08 0.01" pos="1.5 -0.45 0.69"/>'
         '<body name="box" pos="1.5 -0.45 0.74"><freejoint name="box_free"/>'
         '<geom type="box" size="0.03 0.03 0.04" mass="0.1"/></body>')


def test_an_arm_in_a_scene_without_worktop_objects_stands_alone():
    """Spec §2.3: a robot's spawn never brings the six objects. A scene whose staging was
    refused at start (§2.2: a stool too small for them) has none, and an arm is placed
    there alone: no objects added, nothing cleared -- the loose box their staging would
    clear keeps its free joint and its state -- and the world takes no staging there."""
    w, s = make_world(scene_xml(STOOL + LEDGE))
    for _ in range(200):
        mujoco.mj_step(w.model, w.data)
    with pytest.raises(placement.Refused, match="six worktop objects"):
        placement.stage_objects(w, s)
    assert w.scene_staging is None
    pl = place(w, s, ARM, "worktop")
    assert pl.staging is None and not pl.at_scene_objects
    assert "none staged, nothing cleared" in pl.report[0]
    frame_pos, frame_yaw = worktop_objects.frame_of(pl.xyz[:2], pl.surface_z, pl.yaw)
    assert worktop_objects.plan_clear(w.model, w.data, frame_pos, frame_yaw, keep=set()) == ["box"]
    nbody, box = w.model.nbody, free_state(w, "box")
    inst = add(w, ARM, pl)
    assert w.model.nbody == nbody + len(w.robot_body_ids(inst))
    for body in worktop_objects.BODIES.values():
        assert mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, body) < 0, body
    now = free_state(w, "box")
    assert now is not None and np.array_equal(now[0], box[0]) and np.array_equal(now[1], box[1])
    other = Instance("other", load(ARM), "other/", "worktop", pl.xyz, pl.yaw)
    model = w.model
    with pytest.raises(ValueError, match="has none"):
        w.add(other, staging=worktop_objects.Staging(frame_pos, frame_yaw, ["box"]))
    assert w.model is model and list(w.robots) == [ARM]


def test_scene_objects_are_staged_at_the_first_clear_survey_spot_or_not_at_all():
    """Spec §2.2: the scene stages its six objects, with no robot, at the first of the
    reference survey's spots where all six stand on the surface free of interpenetration
    -- past a spot a fixed post blocks -- and nowhere when no spot leaves every object
    supported."""
    w, s = make_world(scene_xml(WORKTOP))
    first = s.spots()[0]
    st = placement.stage_objects(w, s)
    assert np.allclose(st.frame_pos[:2], first.xy) and st.yaw == first.yaw and w.robots == {}
    plate = st.poses()["plate"][0]
    post = (f'<geom name="post" type="box" size="0.02 0.02 0.05" '
            f'pos="{plate[0]} {plate[1]} 0.80"/>')
    w2, s2 = make_world(scene_xml(WORKTOP + post))
    st2 = placement.stage_objects(w2, s2)
    assert not np.allclose(st2.frame_pos[:2], first.xy)
    k = next(i for i, sp in enumerate(s2.spots()) if np.allclose(sp.xy, st2.frame_pos[:2])
             and sp.yaw == st2.yaw)
    for sp in s2.spots()[:k]:
        staging = worktop_objects.Staging(*worktop_objects.frame_of(sp.xy, sp.z, sp.yaw))
        assert placement._penetrations(w2, None, "", None, sp.yaw, (), sp.z, staging)
    small = '<body name="stool"><geom type="box" size="0.15 0.15 0.375" pos="1.5 0 0.375"/></body>'
    w3, s3 = make_world(scene_xml(small))
    with pytest.raises(placement.Refused, match="six worktop objects .*unsupported"):
        placement.stage_objects(w3, s3)


def test_the_worktop_searches_are_bounded():
    """Spec §2.2, §2.3: the scene's objects are staged at the first clear spot among the
    survey's first 40, or not at all; an arm is tried at the survey's first 600 spots, and
    its search ends at the 40th supported, camera-clear spot where it or its objects do not
    fit."""
    assert placement.STAGING_SPOTS == 40
    assert placement.MAX_WORKTOP_SPOTS == 600 and placement.MAX_WORKTOP_TRIALS == 40
    w, s = make_world(scene_xml(WORKTOP))
    plate = placement.stage_objects(w, s).poses()["plate"][0]
    post = (f'<geom name="post" type="box" size="0.02 0.02 0.05" '
            f'pos="{plate[0]} {plate[1]} 0.80"/>')
    w2, s2 = make_world(scene_xml(WORKTOP + post))
    staged = placement.stage_objects(w2, s2)
    good = next(sp for sp in s2.spots() if np.allclose(sp.xy, staged.frame_pos[:2])
                and sp.yaw == staged.yaw)
    blocked = s2.spots()[0]                      # the post stands on its plate
    assert not np.allclose(blocked.xy, good.xy)
    away = worktop_survey.Spot(xy=np.array([3.5, 3.5]), z=0.75, yaw=0.0, surface="floor",
                               why="no worktop under it")
    # staging: the first 40 spots
    s2._spots = [blocked] * 39 + [good]
    assert np.allclose(placement.stage_objects(w2, s2).frame_pos[:2], good.xy)
    s2._spots = [blocked] * 40 + [good]
    with pytest.raises(placement.Refused, match="six worktop objects"):
        placement.stage_objects(w2, s2)
    # an arm that does not fit where the scene's objects are (a post where its base would
    # stand), so it takes them along: blocked at that spot, it fits at a later one
    w3, s3 = make_world(scene_xml(WORKTOP))
    w3.stage_scene(placement.stage_objects(w3, s3))
    block_the_scene_objects_spot(w3)
    blocked = s3.spots()[0]
    assert np.allclose(blocked.xy, w3.scene_staging.frame_pos[:2])
    rm = load(ARM)

    def arm(spots):
        pl = placement.place(w3, s3, rm, "worktop", False, ARM + "/", spots=spots)
        assert pl.staging is not None and not pl.at_scene_objects   # the objects come along
        return pl

    def fits(spot):
        try:
            return arm([spot]) is not None
        except placement.Refused:
            return False

    good = next(sp for sp in s3.spots()[1:] if fits(sp))
    assert np.allclose(arm([blocked] * 39 + [good]).xyz[:2], good.xy)
    with pytest.raises(placement.Refused, match="no clear spot"):
        arm([blocked] * 40 + [good])
    assert np.allclose(arm([away] * 599 + [good]).xyz[:2], good.xy)
    with pytest.raises(placement.Refused, match="no clear spot"):
        arm([away] * 600 + [good])
    assert w3.robots == {}


def test_mobile_robot_on_a_worktop_too_small_for_its_travel_is_placed():
    """No travel requirement on the worktop (spec §2.3): the robot
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


def test_camera_clearance_counts_only_furniture_walls_and_earlier_robots():
    """Spec §2.3: a hit on furniture, walls or earlier robots counts; a loose object small
    enough to grasp (an apple, a box) does not, nor do the six worktop objects for a robot
    on the floor; a loose body too big to grasp (a stool) is furniture and counts."""
    rm = load("myagv")
    shape = robot_model.shape(rm)
    floor = frozenset({0})

    def hits(extra, every=False):
        w, s = make_world(scene_xml(extra))
        ctx = placement.Context(w, s, rm, shape, True)
        if every:      # rays that counted every geom
            ctx._counted = np.ones(w.model.ngeom, bool)
        return placement._camera_rays(ctx, np.zeros(3), 0.0, 0.0, floor)

    small = ('<body name="apple" pos="0.45 0 0.12"><freejoint/>'
             '<geom name="apple" type="box" size="0.04 0.14 0.12" mass="0.1"/></body>')
    stool = ('<body name="stool" pos="0.45 0 0.35"><freejoint/>'
             '<geom name="stool_seat" type="box" size="0.2 0.2 0.35" mass="3"/></body>')
    cupboard = '<geom name="cupboard" type="box" size="0.05 0.5 0.5" pos="0.6 0 0.5"/>'
    assert {h[1] for h in hits(small, every=True)} == {"apple"}
    assert hits(small) == []
    assert hits(stool) and {h[1] for h in hits(stool)} == {"stool_seat"}
    assert {h[1] for h in hits(cupboard)} == {"cupboard"}
    # the six worktop objects, staged on a low table right in front of the camera
    table = '<body name="table"><geom type="box" size="0.6 0.4 0.03" pos="1.0 0 0.12"/></body>'
    w, s = make_world(scene_xml(table))
    st = worktop_objects.Staging(*worktop_objects.frame_of(np.array([0.55, 0.0]), 0.15, 0.0))
    w.stage_scene(st)
    ctx = placement.Context(w, s, rm, shape, True)
    ctx_all = placement.Context(w, s, rm, shape, True)
    ctx_all._counted = np.ones(w.model.ngeom, bool)
    support = frozenset({0}) | {g for g in range(w.model.ngeom)
                                if w.model.body(w.model.geom_bodyid[g]).name == "table"}
    seen = {h[1] for h in placement._camera_rays(ctx_all, np.zeros(3), 0.0, 0.0, support)}
    assert any(n.startswith("task_") for n in seen), seen      # the camera does see them
    assert placement._camera_rays(ctx, np.zeros(3), 0.0, 0.0, support) == []
    # an earlier robot (a floating body, as small as a loose object would be) counts
    w, s = make_world(scene_xml())
    other = add(w, "myagv", placement.Placement(
        np.array([0.55, 0.0, -shape.lo[2] + 0.0005]), 0.0, "floor", 0.0, (0,)))
    ctx = placement.Context(w, s, rm, shape, True)
    found = placement._camera_rays(ctx, np.zeros(3), 0.0, 0.0, floor)
    mine = {g for g in range(w.model.ngeom) if w.model.geom_bodyid[g] in w.robot_body_ids(other)}
    names = {w.model.geom(g).name or f"geom {g}" for g in mine}
    assert found and all(h[1] in names for h in found), found


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
