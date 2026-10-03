"""Scene sources per engine: naming, validation and loading (spec §2.1, §2.2).

`--scene <source>:<id>` is the one scene flag on every engine. Each engine names its own
sources and default scene:

* molmospaces: `ithor:<n>`, `procthor:<n>`, `test:1`; default `ithor:1`.
* robocasa:    `robocasa:<layout>-<style>`, `test:1`; default `robocasa:1-1`.

The household scenes are loaded exactly as the reference project loaded them (the
iTHOR/ProcTHOR house MJCF resolved and installed by MolmoSpaces and read with
`MjSpec.from_file`; the RoboCasa kitchen built from `KitchenArena` with an empty robot
list inside a robosuite `ManipulationTask`), with no light and no robot added. A RoboCasa
kitchen also gets RoboCasa's own objects on its counters (`robocasa_objects.py`); the
six worktop objects are staged by the simulation at start (`worktop_objects.py`), not
here. `test` is a flat floor with one worktop at a fixed height, the same MJCF on both
engines. `Scene.survey_facts` hands the worktop survey (`worktop_survey.py`) what
each source knows about its surfaces.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ENGINES = {
    "molmospaces": {"sources": ("ithor", "procthor", "test"), "default": "ithor:1"},
    "robocasa": {"sources": ("robocasa", "test"), "default": "robocasa:1-1"},
}

#: RoboCasa v1.0's layout and style ids (`LayoutType`, `StyleType`).
ROBOCASA_LAYOUTS = (1, 60)
ROBOCASA_STYLES = (1, 60)

#: The `test` scene: one MJCF, identical on both engines. A 12 m x 12 m floor and one
#: 1.20 m x 0.80 m worktop whose top is at 0.75 m, 1.5 m in front of the origin.
TEST_WORKTOP_TOP = 0.75
TEST_SCENE_XML = f"""
<mujoco model="test_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.62 0.62 0.60" rgb2="0.52 0.52 0.50"/>
    <material name="floor" texture="grid" texrepeat="12 12" reflectance="0.05"/>
    <material name="worktop" rgba="0.72 0.60 0.45 1"/>
    <material name="wall" rgba="0.85 0.85 0.82 1"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor"
          friction="1.0 0.005 0.0001"/>
    <body name="worktop" pos="1.5 0 0">
      <geom name="worktop_top" type="box" size="0.60 0.40 {TEST_WORKTOP_TOP / 2:.4f}"
            pos="0 0 {TEST_WORKTOP_TOP / 2:.4f}" material="worktop"
            friction="1.0 0.005 0.0001"/>
    </body>
    <geom name="wall_n" type="box" size="6 0.05 1.2" pos="0 6.05 1.2" material="wall"/>
    <geom name="wall_s" type="box" size="6 0.05 1.2" pos="0 -6.05 1.2" material="wall"/>
    <geom name="wall_e" type="box" size="0.05 6 1.2" pos="6.05 0 1.2" material="wall"/>
    <geom name="wall_w" type="box" size="0.05 6 1.2" pos="-6.05 0 1.2" material="wall"/>
  </worldbody>
</mujoco>
"""


class SceneError(ValueError):
    pass


@dataclass
class Scene:
    """A loaded scene: its spec (robots are attached to it), and what placement needs."""

    engine: str
    source: str
    id: str
    spec: object
    path: str | None = None
    #: which MjvOption geom groups show the scene's visuals (RoboCasa draws its
    #: collision hulls in group 0 and its visuals in group 1).
    geomgroup: tuple = (1, 1, 1, 0, 0, 0)
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.source}:{self.id}"

    def survey_facts(self, model, data):
        """What the worktop survey (`worktop_survey.py`) reads off this scene, given its
        bare compiled model and data:

        * iTHOR / ProcTHOR: the house's object metadata, its occupancy map and the THOR
          type sets, through molmo_spaces, exactly as the reference read them;
        * RoboCasa: the counters' free worktop regions, from RoboCasa itself;
        * `test`: nothing but the geometry (the same on both engines).
        """
        import worktop_survey as ws

        if self.source in ("ithor", "procthor") and self.path:
            from molmo_spaces.utils.constants.object_constants import (
                ALL_PICKUP_TYPES_THOR, RECEPTACLE_TYPES_THOR)
            from molmo_spaces.utils.mj_model_and_data_utils import (
                descendant_bodies, descendant_geoms, geom_aabb)
            from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

            with contextlib.redirect_stdout(sys.stderr):
                metadata = (get_scene_metadata(self.path) or {}).get("objects", {})
                thormap = load_scene_map(self.path)
            return ws.Facts("thor", model, data, metadata=metadata, thormap=thormap,
                            pickup_types=tuple(ALL_PICKUP_TYPES_THOR),
                            receptacle_types=tuple(RECEPTACLE_TYPES_THOR),
                            geom_aabb=geom_aabb, descendant_bodies=descendant_bodies,
                            descendant_geoms=descendant_geoms)
        if self.source == "robocasa" and self.extra.get("arena") is not None:
            return ws.Facts("robocasa", model, data, regions=counter_regions(self.extra["arena"]))
        return ws.Facts("generic", model, data)


def load_scene_map(scene_path, agent_radius: float | None = 0.35):
    """The occupancy map shipped beside a THOR house, or None: the reference's
    `load_scene_map` (molmospaces/tools/scene_placement.py:87-116)."""
    from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

    scene_path = Path(str(scene_path).replace("_ceiling", ""))
    text = str(scene_path)
    if "ithor" in text:
        cls = iTHORMap
    elif "procthor" in text or "holodeck" in text:
        cls = ProcTHORMap
    else:
        return None
    png = scene_path.parent / f"{scene_path.stem}_map.png"
    if not png.is_file():
        return None
    return cls.load(path=png.as_posix(), agent_radius=agent_radius)


def counter_regions(arena) -> list[dict]:
    """The free worktop rectangles of every counter, in world coordinates: the reference's
    `counter_regions` (robocasa/tools/spawn_robot.py:183-220), RoboCasa's own
    `Counter.get_reset_regions()`."""
    import numpy as np
    from robocasa.models.fixtures.counter import Counter

    regions = []
    for name, fixture in arena.fixtures.items():
        if not isinstance(fixture, Counter):
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                found = fixture.get_reset_regions(env=None)
        except Exception as exc:  # a fixture variant with a different region model
            print(f"  ({name}: no reset regions -- {exc})", file=sys.stderr)
            continue
        rot = float(getattr(fixture, "rot", 0.0) or 0.0)
        c, s = np.cos(rot), np.sin(rot)
        for region_name, region in found.items():
            ox, oy, oz = region["offset"]
            regions.append({
                "name": f"{name}/{region_name}",
                "centre": np.array([fixture.pos[0] + c * ox - s * oy,
                                    fixture.pos[1] + s * ox + c * oy]),
                "half": np.array(region["size"], dtype=float) / 2.0,
                "top_z": float(fixture.pos[2] + oz),
                "rot": rot,
            })
    return regions


def engine_of(source: str) -> list[str]:
    return [e for e, info in ENGINES.items() if source in info["sources"]]


#: Each source's ids, as `--help` and the refusals name them (iTHOR's and ProcTHOR's exact
#: ranges are MolmoSpaces' training-split index, checked when the scene loads).
SOURCE_RANGES = {
    "ithor": "ithor:<n>, an iTHOR floor plan of the training split (1-12 kitchens, 201-212 "
             "living rooms, 301-312 bedrooms, 401-412 bathrooms)",
    "procthor": "procthor:<n>, a ProcTHOR-10k training house",
    "robocasa": f"robocasa:<layout>-<style>, layout {ROBOCASA_LAYOUTS[0]}-"
                f"{ROBOCASA_LAYOUTS[1]}, style {ROBOCASA_STYLES[0]}-{ROBOCASA_STYLES[1]}",
    "test": "test:1, a flat floor with one worktop (identical on both engines)",
}


def help_text(engine: str) -> str:
    info = ENGINES[engine]
    lines = [f"  {SOURCE_RANGES[s]}" for s in info["sources"]]
    return "\n".join(lines) + f"\n  default: {info['default']}"


def parse(engine: str, text: str | None) -> tuple[str, str]:
    """(source, id) of `--scene`, refused with a message naming what is wrong."""
    if engine not in ENGINES:
        raise SceneError(f"unknown engine {engine!r} (one of: {', '.join(ENGINES)})")
    text = text or ENGINES[engine]["default"]
    source, sep, sid = text.partition(":")
    if not sep or not source or not sid:
        raise SceneError(f"--scene: expected <source>:<id>, got {text!r}; this engine takes\n"
                         + help_text(engine))
    if source not in ENGINES[engine]["sources"]:
        others = engine_of(source)
        if others:
            raise SceneError(f"--scene: source {source!r} is not on the {engine} engine; "
                             f"it is on {', '.join(others)} (--engine {others[0]})")
        raise SceneError(f"--scene: unknown source {source!r}; the {engine} engine has: "
                         + ", ".join(ENGINES[engine]["sources"]))
    if source == "test":
        if sid != "1":
            raise SceneError("--scene: test has one scene, id 1 (range 1-1)")
    elif source == "robocasa":
        m = re.fullmatch(r"(\d+)-(\d+)", sid)
        if not m:
            raise SceneError(f"--scene: robocasa ids are <layout>-<style> (layout "
                             f"{ROBOCASA_LAYOUTS[0]}-{ROBOCASA_LAYOUTS[1]}, style "
                             f"{ROBOCASA_STYLES[0]}-{ROBOCASA_STYLES[1]}), got {sid!r}")
        lay, sty = int(m.group(1)), int(m.group(2))
        if not ROBOCASA_LAYOUTS[0] <= lay <= ROBOCASA_LAYOUTS[1]:
            raise SceneError(f"--scene: layout {lay} out of range "
                             f"{ROBOCASA_LAYOUTS[0]}-{ROBOCASA_LAYOUTS[1]}")
        if not ROBOCASA_STYLES[0] <= sty <= ROBOCASA_STYLES[1]:
            raise SceneError(f"--scene: style {sty} out of range "
                             f"{ROBOCASA_STYLES[0]}-{ROBOCASA_STYLES[1]}")
    else:
        if not re.fullmatch(r"\d+", sid):
            raise SceneError(f"--scene: {source} ids are integers, got {sid!r}; "
                             f"{source} takes {SOURCE_RANGES[source]}")
    return source, sid


# ---------------------------------------------------------------- molmospaces


def _ranges(nums: list[int]) -> str:
    nums = sorted(nums)
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(out)


def molmospaces_index(source: str) -> dict:
    """The house index MolmoSpaces serves for this source (train split, as the
    reference's resolve_scene.py read it)."""
    from molmo_spaces.molmo_spaces_constants import get_scenes

    dataset = "procthor-10k" if source == "procthor" else source
    with contextlib.redirect_stdout(sys.stderr):
        scenes = get_scenes(dataset, "train")
    entries = scenes["train"] if isinstance(scenes, dict) and "train" in scenes else scenes
    if isinstance(entries, dict):
        return {int(k): v for k, v in entries.items() if v is not None}
    return {i: v for i, v in enumerate(entries) if v is not None}


def check_molmospaces_id(source: str, sid: str) -> None:
    index = molmospaces_index(source)
    if int(sid) not in index:
        raise SceneError(f"--scene: {source}:{sid} is out of range; {source} ids are "
                         f"{_ranges(list(index))}")


def resolve_molmospaces(source: str, sid: str) -> str:
    """The house MJCF path, installed on demand -- the reference's resolve_scene.py."""
    import subprocess

    dataset = "procthor-10k" if source == "procthor" else source
    tool = Path(os.environ.get("ENGINE_ROOT", Path(__file__).resolve().parents[1] / "molmospaces")) \
        / "tools" / "resolve_scene.py"
    out = subprocess.run([sys.executable, str(tool), dataset, sid], stdout=subprocess.PIPE,
                         check=False)
    if out.returncode != 0:
        raise SceneError(f"could not resolve scene {source}:{sid} (see messages above)")
    return out.stdout.decode().strip().splitlines()[-1]


def load_molmospaces(source: str, sid: str):
    import mujoco

    check_molmospaces_id(source, sid)
    xml = resolve_molmospaces(source, sid)
    spec = mujoco.MjSpec.from_file(xml)
    return spec, xml


# ---------------------------------------------------------------- robocasa


def point_robocasa_assets() -> None:
    """Point robocasa at this engine's generated asset tree instead of its checkout."""
    import robocasa.models

    root = os.environ.get("ROBOCASA_ASSETS_DIR")
    if root:
        if not (Path(root) / ".complete_fixtures").exists():
            raise SceneError(f"the RoboCasa kitchen assets are not installed in {root}; "
                             "run: simulator/robocasa/run.sh setup")
        robocasa.models.assets_root = root


def build_kitchen_arena(layout: int, style: int, seed: int = 0, objects: bool = False):
    """The reference's kitchen build, unchanged: `KitchenArena` with an empty robot list
    inside a `ManipulationTask`, fixtures only, compiled from its XML. With `objects`,
    RoboCasa's own kitchen objects stand on its counters as well (`robocasa_objects`)."""
    import mujoco
    import numpy as np

    point_robocasa_assets()
    import robocasa  # noqa: F401  -- registers assets_root
    from robocasa.models.scenes.kitchen_arena import KitchenArena
    from robosuite.models.tasks import ManipulationTask

    arena = KitchenArena(layout_id=layout, style_id=style, rng=np.random.default_rng(seed))
    arena.set_origin([0, 0, 0])
    fixtures = [cfg["model"] for cfg in arena.get_fixture_cfgs()]
    print(f"layout {layout}, style {style}: {len(fixtures)} fixtures", file=sys.stderr)
    loose = []
    if objects:
        import robocasa_objects

        for obj, pos, yaw in robocasa_objects.populate(arena, counter_regions(arena),
                                                       seed=layout * 100 + style):
            obj.set_pos(pos)
            obj.set_euler((0.0, 0.0, yaw))
            loose.append(obj)
        print(f"  {len(loose)} RoboCasa objects on the counters", file=sys.stderr)
    task = ManipulationTask(
        mujoco_arena=arena,
        mujoco_robots=[],
        mujoco_objects=fixtures + loose,
        enable_multiccd=True,
        enable_sleeping_islands=False,
    )
    return arena, mujoco.MjSpec.from_string(task.get_xml())


def load_robocasa(sid: str):
    lay, sty = (int(x) for x in sid.split("-"))
    with contextlib.redirect_stdout(sys.stderr):
        arena, spec = build_kitchen_arena(lay, sty, objects=True)
    return spec, arena


# ---------------------------------------------------------------- entry point


def load(engine: str, text: str | None) -> Scene:
    import mujoco

    source, sid = parse(engine, text)
    if source == "test":
        return Scene(engine, source, sid, mujoco.MjSpec.from_string(TEST_SCENE_XML),
                     geomgroup=(1, 1, 1, 0, 0, 0))
    if source in ("ithor", "procthor"):
        spec, xml = load_molmospaces(source, sid)
        return Scene(engine, source, sid, spec, path=xml, geomgroup=(1, 1, 1, 0, 0, 0))
    spec, arena = load_robocasa(sid)
    # RoboCasa: group 0 is collision (random translucent colours), group 1 visual. The
    # reference's `visual_only()` showed groups 1 and 2; robot visuals use group 2.
    return Scene(engine, source, sid, spec, geomgroup=(0, 1, 1, 0, 0, 0),
                 extra={"arena": arena})
