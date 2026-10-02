"""--scene: one form on every engine; sources and ranges refused with their message."""

import mujoco
import pytest

import scenes


@pytest.mark.parametrize("engine,default", [("molmospaces", ("ithor", "1")),
                                            ("robocasa", ("robocasa", "1-1"))])
def test_defaults(engine, default):
    assert scenes.parse(engine, None) == default


def test_sources_per_engine():
    assert scenes.parse("molmospaces", "procthor:12") == ("procthor", "12")
    assert scenes.parse("robocasa", "robocasa:60-60") == ("robocasa", "60-60")
    for e in ("molmospaces", "robocasa"):
        assert scenes.parse(e, "test:1") == ("test", "1")


def test_other_engines_source_is_refused_naming_that_engine():
    with pytest.raises(scenes.SceneError, match="it is on robocasa"):
        scenes.parse("molmospaces", "robocasa:1-1")
    with pytest.raises(scenes.SceneError, match="it is on molmospaces"):
        scenes.parse("robocasa", "ithor:1")


@pytest.mark.parametrize("text,msg", [("robocasa:61-1", "layout 61 out of range 1-60"),
                                      ("robocasa:1-0", "style 0 out of range 1-60"),
                                      ("robocasa:1", "<layout>-<style>"),
                                      ("test:2", "range 1-1"),
                                      ("nosuch:1", "unknown source"),
                                      ("ithor", "<source>:<id>")])
def test_refusals(text, msg):
    engine = "molmospaces" if not text.startswith("robocasa") else "robocasa"
    with pytest.raises(scenes.SceneError) as exc:
        scenes.parse(engine, text)
    assert msg in str(exc.value)


def test_test_scene_is_one_mjcf_with_a_worktop():
    spec = mujoco.MjSpec.from_string(scenes.TEST_SCENE_XML)
    m = spec.compile()
    assert m.nbody == 2 and m.nlight == 1
    top = m.geom("worktop_top")
    assert abs((m.body("worktop").pos[2] + top.pos[2] + top.size[2]) - scenes.TEST_WORKTOP_TOP) < 1e-9


def test_help_lists_sources_and_default():
    assert "ithor:<n>" in scenes.help_text("molmospaces")
    assert "default: robocasa:1-1" in scenes.help_text("robocasa")
