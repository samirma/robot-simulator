"""Adding and removing robots keeps time, poses, velocities and controls of everything that
survives; robots collide with each other; the physics loop keeps real time."""

import time

import mujoco
import numpy as np
import pytest

import placement
import registry
import robot_model
import scenes
import worktop_objects
from world import Instance, World


def make_world(with_object=False):
    spec = mujoco.MjSpec.from_string(scenes.TEST_SCENE_XML)
    if with_object:
        b = spec.worldbody.add_body(name="cup", pos=[-1.0, 1.0, 0.05])
        b.add_freejoint()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.04, 0.04], mass=0.2)
    w = World(scenes.Scene("molmospaces", "test", "1", spec))
    return w, placement.SceneSurfaces(w.model, w.data)


def spawn(w, s, rid, where):
    r = registry.get(rid)
    rm = robot_model.load(r)
    pl = placement.place(w, s, rm, where, r.mobile, rid + "/")
    inst = Instance(rid, rm, rid + "/", where, pl.xyz, pl.yaw)
    w.add(inst, staging=pl.staging)
    return inst


def free_state(w, body):
    b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, body)
    j = w.model.body_jntadr[b]
    if w.model.body_jntnum[b] == 0:
        return None
    a, v = w.model.jnt_qposadr[j], w.model.jnt_dofadr[j]
    return w.data.qpos[a:a + 7].copy(), w.data.qvel[v:v + 6].copy()


def world_with_loose_objects():
    """The test scene with loose objects: two on the worktop (to be cleared by an arm
    there) and one far away on the floor."""
    spec = mujoco.MjSpec.from_string(scenes.TEST_SCENE_XML)
    for name, pos in (("mug", [1.55, 0.05, 0.80]), ("can", [1.35, -0.15, 0.80]),
                      ("cup", [-1.0, 1.0, 0.05])):
        b = spec.worldbody.add_body(name=name, pos=pos)
        b.add_freejoint(name=name + "_free")
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.04], mass=0.2)
    w = World(scenes.Scene("molmospaces", "test", "1", spec))
    for _ in range(200):
        mujoco.mj_step(w.model, w.data)
    return w, placement.SceneSurfaces(w.model, w.data)


def robot_state(w, prefix):
    """Every joint's full qpos and qvel of the robot (free joints included)."""
    q, v = [], []
    for j in range(w.model.njnt):
        if w.model.joint(j).name.startswith(prefix):
            a, d = w.model.jnt_qposadr[j], w.model.jnt_dofadr[j]
            nq = {0: 7, 1: 4}.get(int(w.model.jnt_type[j]), 1)
            q.append(w.data.qpos[a:a + nq].copy())
            v.append(w.data.qvel[d:d + (nq - 1 if nq > 1 else 1)].copy())
    return np.concatenate(q), np.concatenate(v)


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def add_to_world(w, build):
    """Edit the live world's spec (`build(worldbody)`) and recompile, keeping its state."""
    with w.lock:
        build(w.spec.worldbody)
        m, d = w.spec.recompile(w.model, w.data)
        mujoco.mj_forward(m, d)
        w._swap(m, d)


def displaced_world():
    """The test scene as `start` leaves it -- its six worktop objects staged where the
    SO-101 is placed, the loose objects there cleared -- then a fixed post on that spot, so
    no arm fits at the scene's objects any more, and a loose jar that has since come to
    stand behind the spot the arm takes instead (so it is cleared for that arm)."""
    w, s = world_with_loose_objects()
    rm = robot_model.load(registry.get("so101"))
    pl = placement.place(w, s, rm, "worktop", False, "so101/")
    w.stage_scene(pl.staging)
    add_to_world(w, lambda wb: wb.add_body(name="post", pos=[*map(float, pl.xyz[:2]), 0.85])
                 .add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.09]))
    away = placement.place(w, s, rm, "worktop", False, "so101/")
    assert not away.at_scene_objects
    behind = away.xyz[:2] - 0.2 * np.array([np.cos(away.yaw), np.sin(away.yaw)])

    def jar(wb):
        b = wb.add_body(name="jar", pos=[*map(float, behind), 0.80])
        b.add_freejoint(name="jar_free")
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.04], mass=0.2)

    add_to_world(w, jar)
    for _ in range(200):
        mujoco.mj_step(w.model, w.data)
    return w, s, pl.staging


def test_arm_away_from_the_scene_objects_takes_them_and_puts_them_back():
    """Spec §2.3 *Arms and the objects*, §5 Lifecycle: an arm that does not fit where the
    scene's objects are stands at its own survey spot, the six objects staged around it and
    the loose objects there cleared; removing it puts the six back where the scene staged
    them and returns what was cleared for it with the pose and velocity it had when
    cleared, while the other robots and scene objects keep their state bit for bit."""
    w, s, scene = displaced_world()
    scene_poses = scene.poses()
    agv = spawn(w, s, "myagv", "floor")
    # another robot operating, a scene object sliding, the jar nudged: all of it moving
    for name, a in w.robot_elements(agv, mujoco.mjtObj.mjOBJ_ACTUATOR):
        w.data.ctrl[a] = 3.0
    b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    w.data.qvel[w.model.jnt_dofadr[w.model.body_jntadr[b]]] = 0.2
    for _ in range(100):
        mujoco.mj_step(w.model, w.data)
    b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "jar")
    w.data.qvel[w.model.jnt_dofadr[w.model.body_jntadr[b]] + 1] = 0.01
    mujoco.mj_forward(w.model, w.data)
    t0, cup, jar, agv0 = w.data.time, free_state(w, "cup"), free_state(w, "jar"), \
        robot_state(w, "myagv/")
    nbody = w.model.nbody
    arm = spawn(w, s, "so101", "worktop")
    # not at the scene's spot: the arm brought the scene's objects along
    assert arm.staging is not None and arm.info.get("displaced_scene_objects")
    assert not np.allclose(arm.xyz[:2], scene.frame_pos[:2])
    for name, pos in object_poses(w).items():
        assert pos is not None and np.allclose(pos, arm.staging.poses()[name][0], atol=1e-6), name
    assert "jar" in arm.staging.cleared and not set(arm.staging.cleared) & set(scene.cleared), \
        arm.staging.cleared
    for name in arm.staging.cleared:
        b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert free_state(w, name) is None and w.data.xpos[b][2] < -49, name
    # nothing else was touched by the recompile
    assert w.data.time == t0 and same(free_state(w, "cup"), cup)
    assert same(robot_state(w, "myagv/"), agv0)
    for name, a in w.robot_elements(agv, mujoco.mjtObj.mjOBJ_ACTUATOR):
        assert w.data.ctrl[a] == 3.0
    for _ in range(50):
        mujoco.mj_step(w.model, w.data)
    t1, cup1, agv1 = w.data.time, free_state(w, "cup"), robot_state(w, "myagv/")
    w.remove("so101")
    assert w.data.time == t1 and same(free_state(w, "cup"), cup1)
    assert same(robot_state(w, "myagv/"), agv1)
    # the six are back where the scene staged them; the scene's own clearing stays
    for name, pos in object_poses(w).items():
        assert pos is not None and np.allclose(pos, scene_poses[name][0], atol=1e-6), name
    for name in scene.cleared:
        assert free_state(w, name) is None, name
    # what was cleared for the arm is back as it was when cleared
    assert same(free_state(w, "jar"), jar) and w.model.nbody == nbody
    for _ in range(50):
        mujoco.mj_step(w.model, w.data)
    assert np.isfinite(w.data.qpos).all()


def test_a_failed_add_away_from_the_scene_objects_rolls_back_robot_and_staging():
    """A displaced arm whose model fails to compile in: robot, its staging and the scene's
    objects all stay as they were, and the next add works."""
    w, s, scene = displaced_world()
    r = registry.get("so101")
    pl = placement.place(w, s, robot_model.load(r), "worktop", False, "so101/")
    assert pl.staging is not None and "jar" in pl.staging.cleared
    model, version, nbody = w.model, w.version, w.model.nbody
    jar = free_state(w, "jar")
    broken = robot_model.load(r)
    broken.spec.body(broken.root).add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="nowhere")
    with pytest.raises(Exception):
        w.add(Instance("so101", broken, "so101/", "worktop", pl.xyz, pl.yaw), staging=pl.staging)
    assert w.model is model and w.version == version and w.robots == {}
    # the spec is as it was: the next add (the same placement) succeeds and goes away cleanly
    st = worktop_objects.Staging(pl.staging.frame_pos, pl.staging.yaw, pl.staging.cleared)
    w.add(Instance("so101", robot_model.load(r), "so101/", "worktop", pl.xyz, pl.yaw), staging=st)
    assert free_state(w, "jar") is None
    w.remove("so101")
    assert w.model.nbody == nbody and same(free_state(w, "jar"), jar)
    for name, pos in object_poses(w).items():
        assert np.allclose(pos, scene.poses()[name][0], atol=1e-6), name


def test_a_failed_add_rolls_back_robot_and_staging(monkeypatch):
    w, s = world_with_loose_objects()
    r = registry.get("so101")
    rm = robot_model.load(r)
    pl = placement.place(w, s, rm, "worktop", r.mobile, "so101/")
    assert pl.staging.cleared
    gone = pl.staging.cleared[0]
    model, nq, version = w.model, w.model.nq, w.version
    mug = free_state(w, gone)
    # the meshes are not where the staging looks for them: the recompile fails
    monkeypatch.setattr(worktop_objects, "ASSETS", worktop_objects.ASSETS / "nowhere")
    monkeypatch.setattr(worktop_objects, "missing_assets", lambda: [])
    inst = Instance("so101", rm, "so101/", "worktop", pl.xyz, pl.yaw)
    with pytest.raises(Exception):
        w.add(inst, staging=pl.staging)
    assert w.model is model and w.version == version and w.robots == {}
    monkeypatch.undo()
    # the spec is as it was: the next add (the same staging) succeeds
    st = worktop_objects.Staging(pl.staging.frame_pos, pl.staging.yaw, pl.staging.cleared)
    w.add(Instance("so101", rm, "so101/", "worktop", pl.xyz, pl.yaw), staging=st)
    assert free_state(w, gone) is None
    w.remove("so101")
    assert w.model.nq == nq
    now = free_state(w, gone)
    assert np.array_equal(now[0], mug[0]) and np.array_equal(now[1], mug[1])


def joint_state(w, prefix):
    out = {}
    for j in range(w.model.njnt):
        n = w.model.joint(j).name
        if n.startswith(prefix):
            a, v = w.model.jnt_qposadr[j], w.model.jnt_dofadr[j]
            out[n] = (float(w.data.qpos[a]), float(w.data.qvel[v]))
    return out


def test_add_and_remove_preserve_the_survivors_state():
    w, s = make_world(with_object=True)
    agv = spawn(w, s, "myagv", "floor")
    # a scene object has moved: pushed along and still sliding
    cup = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    ja = w.model.body_jntadr[cup]
    w.data.qpos[w.model.jnt_qposadr[ja]:w.model.jnt_qposadr[ja] + 3] = [-0.5, 1.5, 0.05]
    w.data.qvel[w.model.jnt_dofadr[ja]] = 0.3
    # the myAGV operating: wheels spinning, robot moving
    for name, a in w.robot_elements(agv, mujoco.mjtObj.mjOBJ_ACTUATOR):
        w.data.ctrl[a] = 4.0
    for _ in range(300):
        mujoco.mj_step(w.model, w.data)
    t0 = w.data.time
    before = joint_state(w, "myagv/")
    cup_before = (w.data.qpos[w.model.jnt_qposadr[ja]:w.model.jnt_qposadr[ja] + 7].copy(),
                  w.data.qvel[w.model.jnt_dofadr[ja]:w.model.jnt_dofadr[ja] + 6].copy())
    ctrl = {w.model.actuator(a).name: float(w.data.ctrl[a]) for a in range(w.model.nu)}
    arm = spawn(w, s, "so101", "worktop")
    assert w.data.time == t0
    after = joint_state(w, "myagv/")
    for k, v in before.items():
        assert after[k] == v, k
    cup = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    ja = w.model.body_jntadr[cup]
    assert np.array_equal(w.data.qpos[w.model.jnt_qposadr[ja]:w.model.jnt_qposadr[ja] + 7], cup_before[0])
    assert np.array_equal(w.data.qvel[w.model.jnt_dofadr[ja]:w.model.jnt_dofadr[ja] + 6], cup_before[1])
    for k, v in ctrl.items():
        a = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_ACTUATOR, k)
        assert w.data.ctrl[a] == v
    for _ in range(100):
        mujoco.mj_step(w.model, w.data)
    before = joint_state(w, "myagv/")
    t1 = w.data.time
    w.remove("so101")
    assert w.data.time == t1 and joint_state(w, "myagv/") == before
    assert mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "so101/base_link") < 0
    assert "so101" not in w.robots and arm is not None


def test_new_robot_starts_at_home_at_rest():
    w, s = make_world()
    inst = spawn(w, s, "mycobot280", "worktop")
    for name, q in inst.rm.home.items():
        j = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_JOINT, "mycobot280/" + name)
        assert abs(w.data.qpos[w.model.jnt_qposadr[j]] - q) < 1e-12


def test_robots_collide_with_each_other():
    w, s = make_world()
    a = spawn(w, s, "myagv", "floor")
    b = spawn(w, s, "rosmaster_x3_plus", "floor")
    ga = [g for g in range(w.model.ngeom) if w.model.geom_bodyid[g] in w.robot_body_ids(a)
          and w.model.geom_contype[g]]
    gb = [g for g in range(w.model.ngeom) if w.model.geom_bodyid[g] in w.robot_body_ids(b)
          and w.model.geom_contype[g]]
    m = w.model
    pairs = lambda g1, g2: (m.geom_contype[g1] & m.geom_conaffinity[g2]) or \
        (m.geom_contype[g2] & m.geom_conaffinity[g1])
    assert any(pairs(x, y) for x in ga for y in gb)
    # ... while each keeps exactly its own model's self-collision
    rm = robot_model.load(registry.get("myagv"))
    alone = rm.spec.compile()
    self_pairs_alone = sum(1 for x in range(alone.ngeom) for y in range(alone.ngeom)
                           if (alone.geom_contype[x] & alone.geom_conaffinity[y]))
    own = [g for g in range(m.ngeom) if m.geom_bodyid[g] in w.robot_body_ids(a)]
    self_pairs = sum(1 for x in own for y in own if (m.geom_contype[x] & m.geom_conaffinity[y]))
    assert self_pairs == self_pairs_alone


def test_physics_loop_keeps_real_time():
    w, s = make_world()
    w.start()
    try:
        t0, s0 = time.monotonic(), w.data.time
        time.sleep(2.0)
        with w.lock:
            s1 = w.data.time
        rtf = (s1 - s0) / (time.monotonic() - t0)
        assert 0.9 < rtf < 1.1
        snap = w.snapshot()
        assert snap["version"] == w.version and snap["qpos"].shape == (w.model.nq,)
    finally:
        w.stop()


def test_rtf_warning_when_physics_cannot_keep_up(capsys, monkeypatch):
    import world as world_mod

    monkeypatch.setattr(world_mod, "RTF_WINDOW_S", 1.0)
    w, s = make_world()
    logs = []
    w.log = logs.append
    real_step = mujoco.mj_step

    def slow_step(m, d, nstep=1):
        real_step(m, d, nstep=nstep)
        time.sleep(0.004 * nstep)   # 2 ms of physics per 4+ ms of wall time: RTF < 0.5

    monkeypatch.setattr(world_mod.mujoco, "mj_step", slow_step)
    t0 = time.time()
    w.start()
    try:
        time.sleep(2.5)
    finally:
        w.stop()
    assert any("real-time factor" in l and "below 0.90" in l for l in logs)
    # every completed window is kept, in wall time, for a check to know not to claim its
    # rates or bounds for the interval (spec §3 Timing)
    windows = list(w.rtf_windows)
    assert len(windows) >= 2
    for (s, e, r), nxt in zip(windows, windows[1:] + [None]):
        assert t0 - 0.5 <= s < e <= time.time() and 0.9 <= e - s <= 3.0 and r < 0.9
        if nxt is not None:
            assert abs(nxt[0] - e) < 1e-6


def object_poses(w):
    out = {}
    for name in worktop_objects.OBJECTS:
        b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, f"task_{name}")
        out[name] = None if b < 0 else w.data.xpos[b].copy()
    return out


def test_scene_objects_exist_without_a_robot_and_survive_every_robot():
    """Staged at start, the six worktop objects are the scene's: an arm that fits where they
    are stands there and leaves them alone; and after any arm leaves they are where the
    scene staged them."""
    w, s = make_world()
    nbody = w.model.nbody
    rm = robot_model.load(registry.get("so101"))
    pl = placement.place(w, s, rm, "worktop", False, "so101/")
    w.stage_scene(pl.staging)
    assert w.model.nbody > nbody and w.scene_nbody == w.model.nbody
    poses = pl.staging.poses()
    for name, pos in object_poses(w).items():
        assert pos is not None and np.allclose(pos, poses[name][0], atol=1e-6)
    # an arm placed at the scene's spot brings no objects of its own
    inst = spawn(w, s, "so101", "worktop")
    assert inst.staging is None and np.allclose(inst.xyz[:2], pl.staging.frame_pos[:2])
    w.remove("so101")
    for name, pos in object_poses(w).items():
        assert pos is not None and np.allclose(pos, poses[name][0], atol=0.02)


@pytest.mark.parametrize("rid", registry.ids())
def test_every_registry_robot_loads_as_one_body(rid):
    """Every robot is a single body (spec §2.3, amended 2026-10-02): its own model with one
    top body, starting from its `home` keyframe; no assembly machinery is left."""
    r = registry.get(rid)
    if registry.missing_files(r):
        pytest.skip(f"{rid}: required files are missing (run.sh setup)")
    assert not hasattr(robot_model, "ARM_PREFIX") and not hasattr(robot_model, "Component")
    assert "components" not in robot_model.RobotModel.__dataclass_fields__
    rm = robot_model.load(r)
    assert [b.name for b in rm.spec.worldbody.bodies] == [rm.root]
    assert rm.floating == r.mobile
    m = mujoco.MjSpec.from_file(str(robot_model.model_file(r))).compile()
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key >= 0, f"{rid} has no home keyframe"
    mujoco.mj_resetDataKeyframe(m, d, key)
    for j in range(m.njnt):
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            assert rm.home[m.joint(j).name] == d.qpos[m.jnt_qposadr[j]], m.joint(j).name
