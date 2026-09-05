"""Re-derive the arm's forward kinematics from MuJoCo and check the transcription.

``robot_console.arm.kinematics`` hand-copies every link's parent-relative ``pos`` and
``quat``, and the ``gripperframe`` site's offset, out of
``so_arm101_description/mjcf/so_arm101.xml``. Nothing checked those numbers. A
slip in any one of them produces a chain that is smooth, self-consistent and
wrong -- the IK converges happily onto a pose the arm does not actually reach,
which is indistinguishable from a physics problem right up until the robot
misses the object.

So these tests load the same MJCF the simulator loads, drive MuJoCo's own
``mj_forward`` to the true site pose, and require the pure-NumPy chain to agree.
The tolerance is 1e-9 m: this is an exact algebraic identity, not an
approximation, so anything looser would hide a real error.

The module docstring in ``kinematics.py`` has promised this file for some time
without it existing. It exists now.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_console.arm.kinematics import (
    ARM_JOINTS,
    JAW_CENTER_OFFSET,
    JOINT_LIMITS,
    approach_pitch,
    fk,
    ik_position,
    jaw_axis,
    jaw_center,
    level_jaw_roll,
    position_jacobian,
    tcp_position,
)

mujoco = pytest.importorskip("mujoco", reason="the offline validator's MuJoCo extra")

#: The site the transcription is anchored to.
TCP_SITE = "gripperframe"
#: Exact identity, not an approximation.
TOLERANCE = 1e-9


#: The shared robot spec every engine loads. Read from the sibling checkout when it is
#: there and skipped otherwise: the console must not *depend* on the simulator tree, but
#: when both are checked out together the arm under test should be the real one.
SHARED_MODEL = (
    Path(__file__).resolve().parents[3] / "simulator" / "shared" / "robots" / "so101" / "model.xml"
)

#: Contract joint name -> the name the MJCF uses. The ROS surface does this translation
#: on the wire; the test has to do it to drive MuJoCo directly.
MJCF_JOINT = {name: name.removesuffix("_joint") for name in ARM_JOINTS}


@pytest.fixture(scope="module")
def model_and_data():
    """Load the shared SO-101 spec, so the arm under test is the real one."""
    if not SHARED_MODEL.exists():
        pytest.skip(f"shared robot spec not present at {SHARED_MODEL}")
    model = mujoco.MjModel.from_xml_path(str(SHARED_MODEL))
    return model, mujoco.MjData(model)


def mujoco_tcp_pose(model, data, angles: np.ndarray) -> np.ndarray:
    """Return MuJoCo's own 4x4 world pose for the ``gripperframe`` site."""
    mujoco.mj_resetData(model, data)
    for name, angle in zip(ARM_JOINTS, angles, strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MJCF_JOINT[name])
        data.qpos[int(model.jnt_qposadr[joint])] = float(angle)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(data.site_xmat[site]).reshape(3, 3)
    pose[:3, 3] = np.asarray(data.site_xpos[site])
    return pose


def sample_poses(seed: int, count: int) -> list[np.ndarray]:
    """Random arm configurations inside the MJCF joint limits."""
    rng = np.random.default_rng(seed)
    low = np.asarray([JOINT_LIMITS[name][0] for name in ARM_JOINTS])
    high = np.asarray([JOINT_LIMITS[name][1] for name in ARM_JOINTS])
    return [rng.uniform(low, high) for _ in range(count)]


def test_the_scene_actually_contains_the_site_we_transcribed(model_and_data) -> None:
    model, _ = model_and_data
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE) >= 0
    for name in ARM_JOINTS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0


def test_zero_pose_matches_mujoco(model_and_data) -> None:
    model, data = model_and_data
    angles = np.zeros(len(ARM_JOINTS))
    np.testing.assert_allclose(fk(angles), mujoco_tcp_pose(model, data, angles), atol=TOLERANCE)


@pytest.mark.parametrize("index", range(len(ARM_JOINTS)))
def test_each_joint_alone_matches_mujoco(model_and_data, index: int) -> None:
    """One joint at a time, so a bad link is attributed to that link."""
    model, data = model_and_data
    low, high = JOINT_LIMITS[ARM_JOINTS[index]]
    for angle in (low * 0.9, -0.3, 0.4, high * 0.9):
        angles = np.zeros(len(ARM_JOINTS))
        angles[index] = angle
        expected = mujoco_tcp_pose(model, data, angles)
        np.testing.assert_allclose(
            fk(angles),
            expected,
            atol=TOLERANCE,
            err_msg=f"{ARM_JOINTS[index]} at {angle:.3f} rad: link {index} is mis-transcribed",
        )


def test_random_configurations_match_mujoco(model_and_data) -> None:
    model, data = model_and_data
    worst = 0.0
    for angles in sample_poses(seed=20260829, count=64):
        expected = mujoco_tcp_pose(model, data, angles)
        np.testing.assert_allclose(fk(angles), expected, atol=TOLERANCE)
        worst = max(worst, float(np.linalg.norm(tcp_position(angles) - expected[:3, 3])))
    assert worst < TOLERANCE


def test_the_jacobian_agrees_with_mujocos_own(model_and_data) -> None:
    """``position_jacobian`` is finite-difference; MuJoCo's is analytic."""
    model, data = model_and_data
    for angles in sample_poses(seed=7, count=8):
        mujoco_tcp_pose(model, data, angles)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site)
        columns = [
            int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in ARM_JOINTS
        ]
        np.testing.assert_allclose(position_jacobian(angles), jacp[:, columns], atol=1e-5)


def test_jaw_axis_is_the_site_z_axis(model_and_data) -> None:
    model, data = model_and_data
    for angles in sample_poses(seed=11, count=8):
        expected = mujoco_tcp_pose(model, data, angles)
        np.testing.assert_allclose(jaw_axis(fk(angles)), expected[:3, 2], atol=TOLERANCE)


def test_jaw_center_is_the_site_offset_along_that_axis(model_and_data) -> None:
    model, data = model_and_data
    for angles in sample_poses(seed=13, count=8):
        expected = mujoco_tcp_pose(model, data, angles)
        np.testing.assert_allclose(
            jaw_center(angles),
            expected[:3, 3] + JAW_CENTER_OFFSET * expected[:3, 2],
            atol=TOLERANCE,
        )


def test_ik_lands_where_it_says_it_does(model_and_data) -> None:
    """An IK solution must be reachable in MuJoCo, not just in our own chain.

    Each target carries the approach pitch the plan actually asks for there --
    a steep pitch over the apple, a shallow one over the plate. Requesting a
    steep top-down pitch at the plate is genuinely unreachable, so pairing the
    two arbitrarily would test the wrong thing.
    """
    model, data = model_and_data
    targets = [
        (np.asarray([0.30, 0.10, 0.12]), -1.15),  # over the apple
        (np.asarray([0.30, 0.10, 0.018]), -1.15),  # at the apple
        (np.asarray([0.220, -0.250, 0.09]), -0.75),  # crossing over the plate
        (np.asarray([0.220, -0.250, 0.045]), -0.50),  # at the release point
        (np.asarray([0.26, 0.0, 0.18]), -0.50),  # home
    ]
    for target, pitch in targets:
        solve = ik_position(target, pitch=pitch, pitch_weight=1.0, max_iterations=600)
        assert solve.position_error < 5e-3, (
            f"IK missed {target} at pitch {pitch} by {solve.position_error:.4f} m"
        )
        actual = mujoco_tcp_pose(model, data, solve.joints)[:3, 3]
        np.testing.assert_allclose(tcp_position(solve.joints), actual, atol=TOLERANCE)


def test_both_layouts_pass_the_preflight_reach_gate() -> None:
    """The grasp and release poses solve from both places the objects may be staged.

    The scripted plan used to prove this on every preflight by solving all 37 of its
    waypoints; with the plan gone the preflight solves the two that decide a
    pick-and-place, and this pins that both layouts in ``task.LAYOUTS`` pass it. A
    layout that failed here would refuse every episode at preflight, which is the
    right failure -- but it should be found by this test, not by a run.
    """
    from robot_console.arm.preflight import _within_reach
    from robot_console.arm.task import LAYOUTS

    for name, (apple, plate) in LAYOUTS.items():
        ok, detail = _within_reach(np.asarray(apple), np.asarray(plate))
        assert ok, f"layout {name!r}: {detail}"


def test_level_jaw_roll_actually_levels_the_jaws(model_and_data) -> None:
    model, data = model_and_data
    for angles in sample_poses(seed=17, count=6):
        angles = angles.copy()
        angles[4] = level_jaw_roll(angles)
        tilt = abs(float(mujoco_tcp_pose(model, data, angles)[2, 2]))
        assert tilt < 0.35, f"jaw axis is {tilt:.3f} off horizontal after levelling"


def test_approach_pitch_reads_a_top_grasp_as_straight_down(model_and_data) -> None:
    solve = ik_position(
        np.asarray([0.30, 0.10, 0.12]), pitch=-np.pi / 2, pitch_weight=1.0, max_iterations=600
    )
    assert approach_pitch(fk(solve.joints)) < -1.2
