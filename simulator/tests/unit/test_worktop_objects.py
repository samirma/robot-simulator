"""The worktop objects (spec §2.3 *Worktop objects*): the reference's six objects at its
poses, the clearing of the working area, and the staging's way in and out of a spec."""

import math

import mujoco
import numpy as np
import pytest

import worktop_objects as wo

pytestmark = pytest.mark.skipif(bool(wo.missing_assets()),
                                reason="worktop object meshes not fetched (run.sh setup)")


def test_poses_are_the_references_in_the_robot_frame():
    frame, yaw = wo.frame_of((1.0, 2.0), 0.9, math.pi / 2)
    assert frame == [1.0, 2.0, 0.904]
    p = wo.poses(frame, yaw)
    assert set(p) == {"apple", "plate", "bowl", "mug", "banana", "lemon"}
    # apple at (0.30, 0.10, 0.020) in the base frame, turned by 90 degrees
    assert np.allclose(p["apple"][0], [1.0 - 0.10, 2.0 + 0.30, 0.924])
    assert np.allclose(p["plate"][0], [1.0 + 0.226, 2.0 + 0.226, 0.904])
    assert np.allclose(p["apple"][1], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    # the banana keeps its own 0.785 rad on top of the frame's heading
    w, _, _, z = p["banana"][1]
    assert abs(2 * math.atan2(z, w) - (math.pi / 2 + 0.785)) < 1e-12


def scene():
    return mujoco.MjSpec.from_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="table"><geom type="box" size="0.6 0.6 0.375" pos="0 0 0.375"/></body>
      <body name="near" pos="0.5 0 0.8"><freejoint name="near_j"/>
        <geom type="box" size="0.03 0.03 0.03" mass="0.1"/></body>
      <body name="far" pos="0.6 0 0.8"><freejoint/>
        <geom type="box" size="0.03 0.03 0.03" mass="0.1"/></body>
      <body name="high" pos="0 0.2 1.3"><freejoint/>
        <geom type="box" size="0.03 0.03 0.03" mass="0.1"/></body>
      <body name="tray" pos="-0.7 0 0.8"><freejoint/>
        <geom type="box" size="0.05 0.05 0.01" mass="0.1"/>
        <body name="on_tray" pos="0.2 0 0.03"><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>
      </body>
      <body name="robot_base" pos="0 -0.2 0.8"><freejoint/>
        <geom type="box" size="0.03 0.03 0.03" mass="0.1"/></body>
      </worldbody></mujoco>""")


def compiled(spec):
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_clearing_takes_the_references_working_area():
    m, d = compiled(scene())
    frame, yaw = wo.frame_of((0.0, 0.0), 0.75, 0.0)
    robot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot_base")
    cleared = wo.plan_clear(m, d, frame, yaw, keep={robot})
    # within 0.55 m (near, the tray by its child at 0.5 m), not beyond it (far), not
    # above the band (high), never a robot; a child goes with its parent
    assert cleared == ["near", "tray"]


def test_apply_and_undo_leave_the_spec_as_it_was():
    spec = scene()
    m0, d0 = compiled(spec)
    frame, yaw = wo.frame_of((0.0, 0.0), 0.75, 0.3)
    st = wo.Staging(frame, yaw, ["near", "tray"])
    st.capture(m0, d0)
    st.apply(spec)
    m1, d1 = compiled(spec)
    # five free objects added (the plate is fixed), two cleared objects lost theirs
    assert m1.nq == m0.nq + 5 * 7 - 2 * 7
    assert m1.nbody == m0.nbody + 6
    for name, (pos, quat) in st.poses().items():
        b = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, wo.BODIES[name])
        assert np.allclose(d1.xpos[b], pos, atol=1e-12) and np.allclose(d1.xquat[b], quat, atol=1e-12)
    b = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, "near")
    assert abs(d1.xpos[b][2] - (0.8 - wo.SUNK_DEPTH)) < 1e-9
    # the apple carries the reference's contact tuning, the plate its 24-box rim
    g = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_GEOM, "task_apple_geom")
    assert m1.geom_condim[g] == 6 and np.allclose(m1.geom_friction[g], [2.0, 0.05, 0.001])
    plate = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, "task_plate")
    assert m1.body_jntnum[plate] == 0 and int((m1.geom_bodyid == plate).sum()) == 2 + 24
    st.undo(spec)
    m2, d2 = compiled(spec)
    assert (m2.nq, m2.nbody, m2.ngeom, m2.nmesh, m2.ntex, m2.nmat) == \
        (m0.nq, m0.nbody, m0.ngeom, m0.nmesh, m0.ntex, m0.nmat)
    assert np.array_equal(m2.body_pos, m0.body_pos)
    j = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, "near_j")
    assert j >= 0 and m2.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    # the cleared objects' state comes back as captured
    d0.qvel[:] = 0
    a = m0.jnt_qposadr[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, "near_j")]
    st.restore_state(m2, d2)
    b = m2.jnt_qposadr[j]
    assert np.array_equal(d2.qpos[b:b + 7], d0.qpos[a:a + 7])


def test_missing_meshes_are_refused_naming_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(wo, "ASSETS", tmp_path)
    st = wo.Staging([0, 0, 0.754], 0.0, [])
    with pytest.raises(wo.AssetsMissing, match="run.sh setup"):
        st.apply(scene())
