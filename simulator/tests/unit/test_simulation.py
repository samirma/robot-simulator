"""The simulation's start path, in-process with no port bound (spec §2.2): the six worktop
objects are part of the scene, staged by `start` at the first of the reference survey's
spots where they stand clear, with no robot involved; a scene with a worktop does not
start when their meshes are missing, and the refusal names `run.sh setup`; a worktop with
no clear spot starts without them."""

import copy
from argparse import Namespace

import numpy as np
import pytest

import placement
import robot_model
import scenes
import simulation
import worktop_objects
from world import World


def make_sim(scene="test:1"):
    """A Simulation with its world and surfaces built the way `load()` builds them."""
    sim = simulation.Simulation(Namespace(engine="molmospaces", scene=scene, sim_port=0,
                                          mujoco=False))
    sim.world = World(scenes.load("molmospaces", scene), log=lambda m: None)
    bare = copy.copy(sim.world.data)
    facts = sim.world.scene.survey_facts(sim.world.model, bare)
    sim.surfaces = placement.SceneSurfaces(sim.world.model, bare, sim.world.scene.geomgroup,
                                           facts=facts)
    return sim


def test_start_stages_the_six_worktop_objects_with_no_robot(monkeypatch):
    if worktop_objects.missing_assets():
        pytest.skip("YCB meshes missing (run.sh setup)")
    sim = make_sim()
    monkeypatch.setattr(simulation.robot_model, "load",
                        lambda robot: pytest.fail("staging loads no robot model"))
    sim._stage_worktop_objects(sim.surfaces.worktop)
    st = sim.world.scene_staging
    assert sorted(st.poses()) == sorted(worktop_objects.OBJECTS) and sim.world.robots == {}
    first = sim.surfaces.spots()[0]
    assert np.allclose(st.frame_pos[:2], first.xy) and st.yaw == first.yaw


def test_start_refuses_without_the_worktop_object_meshes(monkeypatch):
    sim = make_sim()
    monkeypatch.setattr(worktop_objects, "missing_assets", lambda: ["x"])
    with pytest.raises(SystemExit, match="run.sh setup"):
        sim._stage_worktop_objects(sim.surfaces.worktop)


def test_a_worktop_with_no_clear_spot_starts_without_objects(monkeypatch):
    sim = make_sim()
    logs = []
    monkeypatch.setattr(simulation, "log", logs.append)

    def nowhere(world, surfaces):
        raise placement.Refused("no spot on the worktop where the six worktop objects stand clear")

    monkeypatch.setattr(simulation.placement_mod, "stage_objects", nowhere)
    sim._stage_worktop_objects(sim.surfaces.worktop)
    assert sim.world.scene_staging is None
    assert any("no worktop objects staged" in line for line in logs)


def test_a_scene_with_no_worktop_has_no_objects(monkeypatch):
    sim = simulation.Simulation(Namespace(engine="molmospaces", scene="test:1", sim_port=0,
                                          mujoco=False))
    monkeypatch.setattr(simulation.placement_mod, "stage_objects",
                        lambda *a: pytest.fail("no worktop: nothing is staged"))
    monkeypatch.setattr(simulation.robot_model, "load",
                        lambda robot: pytest.fail("no worktop: nothing is loaded"))
    sim._stage_worktop_objects(None)


# ---------------------------------------------------------------- the window's camera

import math  # noqa: E402

import mujoco  # noqa: E402

import registry  # noqa: E402
from world import Instance  # noqa: E402

ARM = "rebot_b601_rs"
_SIMS = {}


def started(scene):
    """The simulation of `scene` as `start` leaves it, its six objects staged (built once
    per scene; the iTHOR house needs MolmoSpaces' assets)."""
    if scene not in _SIMS:
        if worktop_objects.missing_assets():
            pytest.skip("YCB meshes missing (run.sh setup)")
        try:
            sim = make_sim(scene)
        except Exception as exc:     # the house is not installed
            pytest.skip(f"{scene} does not load here: {exc}")
        sim._stage_worktop_objects(sim.surfaces.worktop)
        _SIMS[scene] = sim
    return _SIMS[scene]


def same_view(a, b) -> bool:
    return (a.type == b.type and np.allclose(a.lookat, b.lookat, atol=1e-9)
            and abs(a.distance - b.distance) < 1e-9 and abs(a.azimuth - b.azimuth) < 1e-9
            and abs(a.elevation - b.elevation) < 1e-9)


def test_the_overview_looks_down_on_the_worktop_from_14_m():
    """Spec §2.2: the window opens on a free camera looking at the worktop's centroid on
    the floor from 14 m, azimuth 90 deg, elevation -70 deg (the `test` scene has no
    ceiling)."""
    sim = started("test:1")
    cam = simulation.overview_camera(sim)
    cx, cy = sim.surfaces.worktop.describe()["centroid"]
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert np.allclose(cam.lookat, [cx, cy, sim.surfaces.floor_z])
    assert (cam.distance, cam.azimuth, cam.elevation) == (14.0, 90.0, -70.0)


def test_under_a_ceiling_the_overview_takes_the_clearest_fixed_direction():
    """Spec §2.2: where something hides the worktop from there (iTHOR rooms have ceilings)
    the camera looks at the worktop from the clearest of the fixed directions (elevations
    -70, -50, -35 deg; azimuths every 45 deg), as far back as it still sees it, at most
    14 m."""
    sim = started("ithor:1")
    wt = sim.surfaces.worktop
    cx, cy = wt.describe()["centroid"]
    above = (cx, cy, wt.z + 0.3)
    assert simulation._clear_distance(sim, above, 90.0, -70.0) < 14.0     # the ceiling
    cam = simulation.overview_camera(sim)
    assert np.allclose(cam.lookat, [cx, cy, wt.z + 0.05])
    assert cam.elevation in (-70.0, -50.0, -35.0)
    assert any(abs(math.remainder(cam.azimuth - (90.0 + 45.0 * k), 360.0)) < 1e-9
               for k in range(8))
    reach = {(el, az % 360.0): min(simulation._clear_distance(sim, above, az, el), 14.0)
             for el in (-70.0, -50.0, -35.0) for az in (90.0 + 45.0 * k for k in range(8))}
    best = max(reach.values())
    assert abs(reach[(cam.elevation, cam.azimuth % 360.0)] - best) < 1e-6
    assert 0.5 <= cam.distance <= min(best, 14.0)
    # from where it stands, nothing hides the worktop
    assert simulation._clear_distance(sim, cam.lookat, cam.azimuth, cam.elevation) >= cam.distance


def test_a_scene_with_no_worktop_is_framed_whole():
    sim = make_sim("test:1")
    sim.surfaces.worktop = None
    cam = simulation.overview_camera(sim)
    m = sim.world.model
    assert np.allclose(cam.lookat, m.stat.center)
    assert abs(cam.distance - 1.5 * float(m.stat.extent)) < 1e-9
    assert (cam.azimuth, cam.elevation) == (90.0, -70.0)


@pytest.mark.parametrize("scene", ["test:1", "ithor:1"])
def test_a_spawn_reloading_the_model_keeps_the_users_camera(scene):
    """Spec §2.2: only the opening view is set; a spawn reloading the model keeps the
    camera where the user left it. The window resets its camera to MuJoCo's default after
    each load: the keeper puts the overview back after the first and the user's view back
    after a reload, and leaves a camera the user moved alone."""
    sim = started(scene)
    w = sim.world
    opening = simulation.overview_camera(sim)
    keeper = simulation.CameraKeeper(opening, now=0.0)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(w.model, cam)            # the window's reset of the load
    keeper.settle(w.model, cam, now=1.0)
    assert same_view(cam, opening)
    # the user looks elsewhere; the keeper leaves the view alone
    cam.lookat[:] = [0.3, -0.2, 0.9]
    cam.distance, cam.azimuth, cam.elevation = 2.5, 33.0, -20.0
    user = simulation.sensors.free_camera(cam.lookat, cam.distance, cam.azimuth, cam.elevation)
    keeper.settle(w.model, cam, now=2.0)
    assert same_view(cam, user)
    # a spawn reloads the model: the arm joins the world, the window resets its camera
    rm = robot_model.load(registry.get(ARM))
    pl = placement.place(w, sim.surfaces, rm, "worktop", False, ARM + "/")
    keeper.reloading(cam, now=30.0)
    w.add(Instance(ARM, rm, ARM + "/", "worktop", pl.xyz, pl.yaw), staging=pl.staging)
    try:
        mujoco.mjv_defaultFreeCamera(w.model, cam)
        assert not same_view(cam, user)
        keeper.settle(w.model, cam, now=31.0)
        assert same_view(cam, user)
        # only for a while after the load: later the window's camera is the user's own
        mujoco.mjv_defaultFreeCamera(w.model, cam)
        keeper.settle(w.model, cam, now=30.0 + simulation.CAMERA_WATCH_S + 1.0)
        assert not same_view(cam, user)
    finally:
        w.remove(ARM)


@pytest.mark.parametrize("scene", ["test:1"])
def test_the_rebot_fits_at_the_staging_spot(scene):
    """Spec §2.3: the reBot Arm B601-RS fits at the staging spot of `test` (that of ithor:1
    is integration/test_default_scenes.py): it stands among the scene's six objects, which
    stay where the scene staged them, and nothing more is cleared."""
    sim = started(scene)
    w = sim.world
    st = w.scene_staging
    assert st is not None
    pl = placement.place(w, sim.surfaces, robot_model.load(registry.get(ARM)), "worktop",
                         False, ARM + "/")
    assert pl.at_scene_objects and pl.staging is None
    assert np.allclose(pl.xyz[:2], st.frame_pos[:2]) and abs(pl.yaw - st.yaw) < 1e-12
