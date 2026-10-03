"""The simulation's start path, in-process with no port bound (spec §2.2): the six worktop
objects are part of the scene, staged by `start` around the SO-101's spot; a scene with a
worktop does not start when their meshes or the SO-101's files are missing, and the
refusal names `run.sh setup`."""

import copy
from argparse import Namespace

import pytest

import placement
import registry
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


def test_start_stages_the_six_worktop_objects():
    if registry.missing_files(registry.get("so101")) or worktop_objects.missing_assets():
        pytest.skip("SO-101 files or YCB meshes missing (run.sh setup)")
    sim = make_sim()
    sim._stage_worktop_objects(sim.surfaces.worktop)
    assert sorted(sim.world.scene_staging.poses()) == sorted(worktop_objects.OBJECTS)


def test_start_refuses_without_the_so101_files(monkeypatch):
    def missing(robot):
        raise robot_model.ModelError(
            "so101: required files are missing (x); run simulator/<engine>/run.sh setup")

    sim = simulation.Simulation(Namespace(engine="molmospaces", scene="test:1", sim_port=0,
                                          mujoco=False))
    monkeypatch.setattr(simulation.robot_model, "load", missing)
    with pytest.raises(SystemExit, match="run.sh setup"):
        sim._stage_worktop_objects(object())


def test_start_refuses_without_the_worktop_object_meshes(monkeypatch):
    if registry.missing_files(registry.get("so101")):
        pytest.skip("SO-101 files missing (run.sh setup)")
    sim = make_sim()
    monkeypatch.setattr(worktop_objects, "missing_assets", lambda: ["x"])
    with pytest.raises(SystemExit, match="run.sh setup"):
        sim._stage_worktop_objects(sim.surfaces.worktop)


def test_a_scene_with_no_worktop_has_no_objects(monkeypatch):
    sim = simulation.Simulation(Namespace(engine="molmospaces", scene="test:1", sim_port=0,
                                          mujoco=False))
    monkeypatch.setattr(simulation.robot_model, "load",
                        lambda robot: pytest.fail("no worktop: nothing is loaded"))
    sim._stage_worktop_objects(None)
