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


def test_staging_comes_and_goes_with_its_robot_and_keeps_the_rest():
    w, s = world_with_loose_objects()
    agv = spawn(w, s, "myagv", "floor")
    # the far cup is moving; so is the myAGV
    cup = free_state(w, "cup")
    b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    w.data.qvel[w.model.jnt_dofadr[w.model.body_jntadr[b]]] = 0.2
    for name, a in w.robot_elements(agv, mujoco.mjtObj.mjOBJ_ACTUATOR):
        w.data.ctrl[a] = 3.0
    for _ in range(100):
        mujoco.mj_step(w.model, w.data)
    # the mug on the worktop is nudged: it is cleared with this state
    b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    w.data.qvel[w.model.jnt_dofadr[w.model.body_jntadr[b]] + 1] = 0.01
    mujoco.mj_forward(w.model, w.data)
    nq, nbody, t0 = w.model.nq, w.model.nbody, w.data.time
    mug, can, cup = free_state(w, "mug"), free_state(w, "can"), free_state(w, "cup")
    agv_before = joint_state(w, "myagv/")
    arm = spawn(w, s, "so101", "worktop")
    cleared = arm.staging.cleared
    assert cleared and set(cleared) <= {"mug", "can"}
    assert w.data.time == t0
    # cleared: no free joint any more, parked 50 m down; staged: the six objects
    for name in cleared:
        assert free_state(w, name) is None
        b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert w.data.xpos[b][2] < -49
    for body in worktop_objects.BODIES.values():
        assert mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0
    # the survivors are untouched, bit for bit
    for name, st in (("cup", cup), ("mug", mug), ("can", can)):
        if name in cleared:
            continue
        after = free_state(w, name)
        assert np.array_equal(after[0], st[0]) and np.array_equal(after[1], st[1]), name
    assert joint_state(w, "myagv/") == agv_before
    for _ in range(50):
        mujoco.mj_step(w.model, w.data)
    t1, cup1, agv1 = w.data.time, free_state(w, "cup"), joint_state(w, "myagv/")
    w.remove("so101")
    # the staged objects are gone, the cleared ones are back as they were cleared
    assert w.model.nq == nq and w.model.nbody == nbody and w.data.time == t1
    for body in worktop_objects.BODIES.values():
        assert mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, body) < 0
    for name, st in (("mug", mug), ("can", can)):
        if name in cleared:
            now = free_state(w, name)
            assert np.array_equal(now[0], st[0]) and np.array_equal(now[1], st[1]), name
    after = free_state(w, "cup")
    assert np.array_equal(after[0], cup1[0]) and np.array_equal(after[1], cup1[1])
    assert joint_state(w, "myagv/") == agv1
    # and the world still steps
    for _ in range(50):
        mujoco.mj_step(w.model, w.data)
    assert np.isfinite(w.data.qpos).all()


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
    w.start()
    try:
        time.sleep(2.5)
    finally:
        w.stop()
    assert any("real-time factor" in l and "below 0.90" in l for l in logs)


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
