"""Checks of the robot records in robots_specs/ against the robot files robots_specs/<id>.md.

    uv run --with pytest --with pyyaml --with mujoco pytest robots_specs/tests

Meshes must have been fetched first (python3 robots_specs/tools/fetch_meshes.py).
"""

from __future__ import annotations

import hashlib
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fetch_meshes  # noqa: E402
import schema  # noqa: E402
import spec_registry  # noqa: E402

SPECS, ROOT = spec_registry.SPECS, spec_registry.ROOT
ROBOTS = spec_registry.robots()
SINGLE = ROBOTS
IDS = [r.id for r in SINGLE]

#: Motions each robot's evidence case exercises (workspace spec §3).
REQUIRED_MOTIONS = {
    "myagv": {"drive"},
    "so101": {"arm", "gripper"},
    "ainex": {"head", "walk", "action_group"},
    "mycobot280": {"arm", "gripper"},
    "rosmaster_x3_plus": {"drive", "arm", "gripper"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ros_file(robot) -> Path:
    return ROOT / robot.ros


def load(robot) -> dict:
    return yaml.safe_load(ros_file(robot).read_text())


def manifest() -> dict[str, str]:
    return fetch_meshes.read_manifest()


# ------------------------------------------------------------------ registry


def test_registry_lists_the_robots():
    assert {r.id for r in ROBOTS} == {"myagv", "so101", "ainex", "mycobot280",
                                      "rosmaster_x3_plus"}


def test_every_robot_file_is_a_robot_named_by_its_id():
    files = spec_registry.robot_files()
    assert [p.stem for p in files] == sorted(r.id for r in ROBOTS), \
        "every robots_specs/*.md but high_level_spec.md and SCHEMA.md is one robot's file"
    for r in ROBOTS:
        assert r.file.name == f"{r.id}.md"


def test_spec_lists_exactly_the_robot_files():
    text = spec_registry.SPEC.read_text()
    listed = re.findall(r"^\| `([^`]+)` \| (.+?) \| `([^`]+)` \| \[`([^`]+)`\]\(([^)]+)\) \|$", text, re.M)
    assert [(rid, name, kind, f"{rid}.md", f"{rid}.md") for rid, name, kind, *_ in listed] == listed
    assert sorted((rid, name, kind) for rid, name, kind, *_ in listed) == \
        sorted((r.id, r.name, r.kind) for r in ROBOTS)


def test_no_yaml_registry():
    stray = [p.name for p in SPECS.glob("*.y*ml")]
    assert not stray, f"the specification is the registry; remove {stray}"


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_robot_folder_has_its_files(robot):
    folder = SPECS / robot.id
    assert folder.is_dir()
    assert robot.urdf and (ROOT / robot.urdf).is_file(), robot.urdf
    if robot.mjcf:
        assert (ROOT / robot.mjcf).is_file(), robot.mjcf
    ros = [p.name for p in folder.glob("ros*.yml")]
    assert len(ros) == 1, f"exactly one ros.yml/ros2.yml, found {ros}"
    assert robot.ros == f"robots_specs/{robot.id}/{ros[0]}"
    doc = load(robot)
    if doc["model"]["mjcf"] == "model.xml":
        assert (folder / "model.xml").is_file() and (folder / "import.md").is_file()
        assert "derived" in (folder / "model.xml").read_text()[:2000].lower(), \
            "a converted model is labelled derived"
    official = Path(robot.mjcf).name if robot.mjcf else None
    if official:
        assert official in (doc["model"]["mjcf"], doc["model"].get("official_mjcf"))
    assert doc["model"]["mjcf_official"] == (doc["model"]["mjcf"] == official)


# ------------------------------------------------------------------ interface files


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_schema(robot):
    errors = schema.validate(ros_file(robot))
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_every_periodic_topic_has_a_rate(robot):
    for t in load(robot)["topics"]:
        r = t.get("rate")
        assert r == "non_periodic" or (isinstance(r, (int, float)) and r > 0), t["name"]


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_every_velocity_motion_has_a_stop(robot):
    doc = load(robot)
    eps = {t["name"]: t["type"] for t in doc["topics"]}
    eps.update({s["name"]: s["type"] for k in ("services", "actions") for s in doc[k]})
    for m in doc["motions"]:
        if m["id"] in ("drive", "walk"):
            stop = m.get("stop")
            assert stop, f"{m['id']} has no stop command"
            assert eps.get(stop["name"]) == stop["type"], stop
            assert stop.get("message") is not None


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_required_motions_recorded(robot):
    ids = {m["id"] for m in load(robot)["motions"]}
    assert REQUIRED_MOTIONS[robot.id] <= ids


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_watchdog_never_claims_an_unmade_measurement(robot):
    for m in load(robot)["motions"]:
        wd = m["watchdog"]
        if wd["basis"] == "measured":
            assert wd.get("date") and wd.get("method")
        else:
            assert "source" in wd["method"].lower()


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_camera_streams_marked(robot):
    doc = load(robot)
    marked = [t for t in doc["topics"] if t.get("camera")]
    for t in marked:
        assert t["type"].endswith("Image") and not t["type"].endswith("CompressedImage")


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_dialect_matches_the_spec(robot):
    want = "ros1" if robot.ros.endswith("/ros.yml") else "ros2"
    assert load(robot)["dialect"] == want


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_pinned_revisions_recorded(robot):
    revs = {s["revision"] for s in load(robot)["sources"]}
    for rev in robot.revisions:
        assert rev in revs, f"{rev} from the spec is not in {robot.ros} sources"


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_urdf_digest(robot):
    doc = load(robot)
    urdf = ROOT / robot.urdf
    assert doc["model"]["urdf"] == urdf.name
    assert doc["model"]["urdf_sha256"] == sha256(urdf)
    if "URDF SHA-256" in robot.sha256:
        assert sha256(urdf) == robot.sha256["URDF SHA-256"]


def test_fetch_sources_are_the_spec_pins():
    pins = {rev for r in ROBOTS for rev in r.revisions}
    for key, (_url, rev) in fetch_meshes.GIT.items():
        assert rev in pins, f"fetch_meshes.py pins {key} at {rev}, not a spec revision"
    ws = spec_registry.by_id()["rosmaster_x3_plus"].sha256["Workspace archive SHA-256"]
    assert fetch_meshes.DRIVE["yahboom"]["member_sha256"] == ws
    assert "1SRg1aD_u8kyxxjm4vp0cFYxu8Ddj2sXU" in spec_registry.by_id()["rosmaster_x3_plus"].section


# ------------------------------------------------------------------ meshes


def _urdf_meshes(robot) -> set[str]:
    urdf = ROOT / robot.urdf
    out = set()
    for el in ET.parse(urdf).getroot().iter("mesh"):
        fn = el.get("filename")
        m = re.match(r"package://[^/]+/(.*)", fn)
        rel = m.group(1) if m else str((urdf.parent / fn).relative_to(SPECS / robot.id))
        out.add(f"{robot.id}/{rel}")
    return out


def _mjcf_meshes(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    comp = root.find("compiler")
    meshdir = comp.get("meshdir", "") if comp is not None else ""
    base = (path.parent / meshdir).resolve()
    return {str((base / m.get("file")).resolve().relative_to(SPECS.resolve()))
            for m in root.iter("mesh") if m.get("file")}


def _models(robot) -> list[Path]:
    doc = load(robot)
    paths = [SPECS / robot.id / doc["model"]["mjcf"]]
    if robot.mjcf:
        paths.append(ROOT / robot.mjcf)
    return sorted(set(paths))


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_referenced_meshes_are_in_the_manifest(robot):
    listed = set(manifest())
    refs = _urdf_meshes(robot)
    for model in _models(robot):
        refs |= _mjcf_meshes(model)
    missing = sorted(refs - listed)
    assert not missing, f"not in meshes.sha256: {missing}"


def test_manifest_paths_are_repo_relative_and_fetchable():
    for rel in manifest():
        fetch_meshes.origin_of(rel)       # raises when no source provides it
        assert (SPECS / rel.split("/", 1)[0]).is_dir()


def test_meshes_match_the_manifest():
    bad = [p for p, d in manifest().items() if fetch_meshes.sha256(SPECS / p) != d]
    assert not bad, (f"{len(bad)} file(s) missing or mismatched; run "
                     f"python3 robots_specs/tools/fetch_meshes.py: {bad[:5]}")


def test_fetched_meshes_are_not_committed():
    import subprocess
    r = subprocess.run(["git", "ls-files", "robots_specs"], cwd=ROOT, capture_output=True, text=True)
    tracked = set(r.stdout.split())
    assert not tracked & {f"robots_specs/{p}" for p in manifest()}
    r = subprocess.run(["git", "check-ignore", "--no-index", *[f"robots_specs/{p}" for p in manifest()]],
                       cwd=ROOT, capture_output=True, text=True)
    assert len(r.stdout.split()) == len(manifest()), "every fetched file is git-ignored"


# ------------------------------------------------------------------ MuJoCo


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_models_load_in_mujoco(robot):
    mujoco = pytest.importorskip("mujoco")
    for path in _models(robot):
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.nbody > 1 and model.njnt > 0, path.name


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_model_conventions(robot):
    mujoco = pytest.importorskip("mujoco")
    doc = load(robot)
    model = mujoco.MjModel.from_xml_path(str(SPECS / robot.id / doc["model"]["mjcf"]))
    free = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if robot.kind == "arm":
        assert not free, "an arm is fixed where it is placed"
    else:
        assert free == ["root"], "a mobile robot has one freejoint named root"
    cams = {model.camera(i).name for i in range(model.ncam)}
    for c in doc["sensors"]["cameras"]:
        if not c.get("optional"):
            assert c["frame_id"] in cams, f"no <camera name={c['frame_id']!r}>"
            res = [int(v) for v in model.cam_resolution[model.camera(c["frame_id"]).id]]
            assert res == [c["width"], c["height"]], (c["frame_id"], res)
    keys = {model.key(i).name for i in range(model.nkey)}
    assert "home" in keys


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_model_settles_on_a_floor(robot):
    """The model, dropped onto a plane (arms fixed above it), stays finite for 1 s."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np
    doc = load(robot)
    path = SPECS / robot.id / doc["model"]["mjcf"]
    spec = mujoco.MjSpec.from_file(str(path))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos)), "simulation diverged"


#: Robots whose recorded boot (incl. an approved camera boot) serves a required camera.
REQUIRED_CAMERA = {"myagv", "so101", "ainex", "rosmaster_x3_plus"}


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_required_cameras_are_not_optional(robot):
    doc = load(robot)
    required = [c for c in doc["sensors"]["cameras"] if not c.get("optional")]
    assert bool(required) == (robot.id in REQUIRED_CAMERA)
    topics = {t["name"]: t for t in doc["topics"]}
    for c in required:
        assert not topics[c["image_topic"]].get("optional")


# ------------------------------------------------------------------ appearance


def test_stl_splits_are_lossless():
    """Every `<x>.STL.part<k>.stl` concatenates back to the official STL's triangles."""
    import struct
    parts: dict[str, list[str]] = {}
    for rel in manifest():
        m = re.match(r"([^/]+)/derived_meshes/(.+\.stl)\.part(\d+)\.stl$", rel, re.I)
        if m:
            parts.setdefault(f"{m.group(1)}/{m.group(2)}", []).append(rel)
    assert parts
    for src, rels in parts.items():
        rels.sort(key=lambda r: int(re.search(r"part(\d+)\.stl$", r).group(1)))
        data = (SPECS / src).read_bytes()
        body = b"".join((SPECS / r).read_bytes()[84:] for r in rels)
        assert body == data[84:], src
        assert sum(struct.unpack("<I", (SPECS / r).read_bytes()[80:84])[0] for r in rels) == \
            struct.unpack("<I", data[80:84])[0]


def _visual_mesh_files(model_path: Path) -> set[str]:
    root = ET.parse(model_path).getroot()
    files = {m.get("name"): m.get("file") for m in root.iter("mesh")}
    classes = {}
    for d in root.iter("default"):
        g = d.find("geom")
        if d.get("class") and g is not None:
            classes[d.get("class")] = g.get("group")
    out = set()
    for g in root.iter("geom"):
        if g.get("mesh") and (g.get("group") == "2" or classes.get(g.get("class")) == "2"):
            out.add(files[g.get("mesh")])
    return out


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_visuals_are_official_visual_meshes(robot):
    """Visual geoms show the URDF's (or official MJCF's) visual meshes, or files derived
    from exactly those, never a collision-only mesh."""
    doc = load(robot)
    urdf = ET.parse(ROOT / robot.urdf).getroot()
    vis, col = set(), set()
    for link in urdf.iter("link"):
        for kind, acc in (("visual", vis), ("collision", col)):
            for m in link.findall(f"{kind}/geometry/mesh"):
                fn = m.get("filename")
                mm = re.match(r"package://[^/]+/(.*)", fn)
                acc.add(mm.group(1) if mm else fn)
    only_col = col - vis
    for f in _visual_mesh_files(SPECS / robot.id / doc["model"]["mjcf"]):
        base = re.sub(r"^derived_meshes/", "", f)
        base = re.sub(r"(\.m\d+\.obj|\.part\d+\.stl)$", "", base)
        assert base not in only_col, f"visual geom uses collision-only mesh {f}"


@pytest.mark.parametrize("robot", SINGLE, ids=lambda r: r.id)
def test_collada_visuals_keep_their_materials(robot):
    """A COLLADA visual is shown as its per-material OBJ parts with each material's own
    colour and texture (from the derived .mtl)."""
    sys.path.insert(0, str(SPECS / "tools" / "models"))
    from _visual import dae_parts
    doc = load(robot)
    root = ET.parse(SPECS / robot.id / doc["model"]["mjcf"]).getroot()
    meshes = {m.get("name"): m.get("file") for m in root.iter("mesh")}
    mats = {m.get("name"): m for m in root.iter("material")}
    texs = {t.get("name"): t.get("file") for t in root.iter("texture")}
    used = {(meshes[g.get("mesh")], g.get("material")) for g in root.iter("geom")
            if g.get("mesh") and meshes[g.get("mesh")].endswith(".obj")}
    daes = sorted({re.sub(r"\.m\d+\.obj$", "", f)[len("derived_meshes/"):] for f, _ in used})
    if not any(f.endswith(".dae") for f in _urdf_meshes(robot)):
        assert not used
        return
    assert daes, "COLLADA visuals must use the derived per-material OBJ parts"
    for dae in daes:
        for p in dae_parts(SPECS / robot.id, dae):
            hits = [mat for f, mat in used if f == p["obj"]]
            assert hits, p["obj"]
            for mat in hits:
                el = mats[mat]
                rgba = [float(v) for v in el.get("rgba", "1 1 1 1").split()]
                assert np.allclose(rgba, p["rgba"], atol=1e-5), (mat, rgba, p["rgba"])
                if p["texture"]:
                    assert texs[el.get("texture")] == p["texture"]
