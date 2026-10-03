"""Checks of the robot records in robots_specs/ against the robot files robots_specs/<id>.md.

    uv run --with pytest --with pyyaml --with mujoco pytest robots_specs/tests

Meshes must have been fetched first (python3 robots_specs/tools/fetch_meshes.py).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
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

#: Motions each robot's evidence case exercises (workspace spec §3).
REQUIRED_MOTIONS = {
    "myagv": {"drive"},
    "so101": {"arm", "gripper"},
    "ainex": {"head", "walk", "action_group"},
    "mycobot280": {"arm", "gripper"},
    "rosmaster_x3_plus": {"drive", "arm", "gripper"},
}


sha256 = fetch_meshes.sha256
manifest = fetch_meshes.read_manifest


def ros_file(robot) -> Path:
    return ROOT / robot.ros


def load(robot) -> dict:
    return yaml.safe_load(ros_file(robot).read_text())


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


KINDS = {"mobile_base", "mobile_manipulator", "humanoid", "arm"}


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_robot_file_entries(robot):
    """Each robot file gives the id, identity, kind, embodiment, official URL, pinned sources,
    authoritative boot, documentation URL and the paths of its URDF, official MJCF where one
    exists, MuJoCo model and ROS interface file (robot specification §1; workspace spec §1.2, §2)."""
    b = robot.bullets
    for key in ("Robot id", "Folder", "Manufacturer", "Kind", "Embodiment", "Official URDF",
                "MuJoCo model", "ROS interface", "Authoritative boot", "Documentation"):
        assert b.get(key), f"{robot.file.name} has no '{key}' entry"
    assert robot.kind in KINDS, robot.kind
    assert spec_registry._code(b["Folder"]) == f"robots_specs/{robot.id}/"
    assert any(k.startswith("Official ") and "<http" in v for k, v in b.items()), "no official URL"
    assert "<http" in b["Documentation"]
    assert robot.revisions, "no pinned revision"
    doc = load(robot)
    assert robot.model == f"robots_specs/{robot.id}/{doc['model']['mjcf']}"
    if not doc["model"]["mjcf_official"]:
        assert "derived" in b["MuJoCo model"] and f"robots_specs/{robot.id}/import.md" in b["MuJoCo model"]


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_robot_file_names_the_authoritative_boot(robot):
    """The robot file names the boot's launch file and every vendor data file it loads, at
    a pinned revision recorded in the file (robot specification §1)."""
    boot = load(robot)["boot"]
    text = robot.bullets["Authoritative boot"]
    for row in [boot["launch_files"][0], *(boot.get("data_files") or [])]:
        assert Path(row["path"]).name in text, f"{robot.file.name}: boot entry does not name {row['path']}"
    srcs = {s["id"]: s["revision"] for s in load(robot)["sources"]}
    rev = srcs[boot["launch_files"][0]["source"]]
    assert rev in robot.revisions or all(Path(m).name in robot.section for m in rev.split("!/")), rev


def test_no_yaml_registry():
    stray = [p.name for p in SPECS.glob("*.y*ml")]
    assert not stray, f"the specification is the registry; remove {stray}"


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
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


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_schema(robot):
    errors = schema.validate(ros_file(robot))
    assert not errors, "\n".join(errors)


def _broken_copy(tmp_path, robot, change) -> Path:
    doc = load(robot)
    change(doc)
    path = tmp_path / robot.id / ros_file(robot).name
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


@pytest.mark.parametrize("change, error", [
    (lambda d: d["model"]["robot_description"].pop("content"), "robot_description.content"),
    (lambda d: d["sensors"]["imus"][0].update(topic="/not_recorded"), "sensors.imus[0] topic"),
    (lambda d: d.update(camera_boot_authority="manufacturer"), "camera_boot_authority"),
    (lambda d: d["boot"]["launch_files"][0].update({"GPIO20)": None}), "beyond {source, path, role}"),
    (lambda d: next(t for t in d["topics"] if t["name"] == "/scan").pop("frame_id"), "records no frame_id"),
    (lambda d: d["motions"][0]["feedback"].append({"name": "/odom", "field": "pose.pose (x", "y": None}),
     "feedback[2] must be"),
], ids=["no-robot-description-content", "imu-topic-not-recorded", "camera-boot-authority",
        "split-boot-row", "no-frame-id", "split-feedback-row"])
def test_schema_rejects(tmp_path, change, error):
    errors = schema.validate(_broken_copy(tmp_path, spec_registry.by_id()["myagv"], change))
    assert any(error in msg for msg in errors), errors


def test_estimated_action_groups():
    """Estimated action groups are labelled robot data, valid against the interface file and
    URDF; only the AiNex, whose vendor groups are in no pinned source, has them."""
    files = sorted(SPECS.glob("*/action_groups.yml"))
    assert [p.parent.name for p in files] == ["ainex"]
    for p in files:
        errors = schema.validate_action_groups(p)
        assert not errors, "\n".join(errors)


@pytest.mark.parametrize("change, error", [
    (lambda d: d["groups"]["wave"]["frames"][0]["offsets_rad"].update(r_sho_roll=-9.0), "outside the URDF limit"),
    (lambda d: d.update(basis="manufacturer"), "basis: estimate"),
    (lambda d: d.update(smoke_example="left_shot"), "smoke_example"),
], ids=["beyond-limit", "not-an-estimate", "unknown-smoke-example"])
def test_action_groups_rejects(tmp_path, change, error):
    src = SPECS / "ainex"
    dst = tmp_path / "ainex"
    dst.mkdir()
    for name in ("ros.yml", "ainex.urdf"):
        shutil.copy2(src / name, dst / name)
    doc = yaml.safe_load((src / "action_groups.yml").read_text())
    change(doc)
    (dst / "action_groups.yml").write_text(yaml.safe_dump(doc, sort_keys=False))
    errors = schema.validate_action_groups(dst / "action_groups.yml")
    assert any(error in msg for msg in errors), errors


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_import_notes_list_the_estimates(robot):
    """import.md distinguishes estimates from manufacturer data under one heading (robot
    specification §1)."""
    text = (SPECS / robot.id / "import.md").read_text()
    assert "\n## Estimates (not manufacturer data)\n" in text


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_name_matches_robot_file(robot):
    assert load(robot)["name"] == robot.name, "name is the robot file's display name (SCHEMA.md)"


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_required_motions_recorded(robot):
    ids = {m["id"] for m in load(robot)["motions"]}
    assert REQUIRED_MOTIONS[robot.id] <= ids


def _sources(src) -> frozenset:
    return frozenset([src] if isinstance(src, str) else src or [])


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_watchdog_sources_follow_the_command(robot):
    """A watchdog cites the path of its own command: motions with different command
    endpoints do not share one set of watchdog sources."""
    motions = load(robot)["motions"]
    for a in motions:
        for b in motions:
            if a["command"]["name"] < b["command"]["name"]:
                assert _sources(a["watchdog"].get("source")) != _sources(b["watchdog"].get("source")), \
                    (a["id"], b["id"])


def test_mycobot280_rviz2_is_part_of_the_boot():
    """rviz2 is started unconditionally by the authoritative boot launch, so its rows are
    not optional (robot specification §1)."""
    doc = load(spec_registry.by_id()["mycobot280"])
    rviz = [n for n in doc["nodes"] if n["name"] == "/rviz2"]
    assert rviz and not rviz[0].get("optional")
    rows = [t for t in doc["topics"] if "/rviz2" in t["nodes"]]
    rows += [p for p in doc["parameters"] if p.get("node") == "/rviz2"]
    assert {t["name"] for t in rows} >= {"/initialpose", "/goal_pose", "/clicked_point"}
    assert not [r["name"] for r in rows if r.get("optional")]


def test_myagv_mapping_and_navigation_endpoints_are_optional_rows():
    """The endpoints of the separate mapping and navigation launches, and the usb_cam
    plugin topics, are recorded as optional rows (myagv.md; robot specification §1)."""
    doc = load(spec_registry.by_id()["myagv"])
    topics = {t["name"]: t for t in doc["topics"]}
    for name in ("/move_base/global_costmap/costmap", "/move_base/local_costmap/costmap",
                 "/move_base/GlobalPlanner/plan", "/move_base/status", "/usb_cam/image_raw/compressed"):
        assert topics[name].get("optional") is True, name
    params = [p for p in doc["parameters"] if p.get("optional")]
    for prefix in ("/gmapping/", "/amcl/", "/move_base/"):
        assert any(p["name"].startswith(prefix) for p in params), prefix


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_no_row_is_left_unrecorded(robot):
    text = ros_file(robot).read_text()
    assert "not enumerated" not in text.lower(), "record the rows instead"


def _record_texts(rid: str) -> dict[str, str]:
    folder = SPECS / rid
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix in (".md", ".yml", ".xml", ".urdf")]
    files.append(SPECS / "tools" / "models" / f"{rid}.py")
    return {p.name: p.read_text() for p in files if p.is_file()}


def test_no_composite_leftovers():
    """Robots are single bodies (robot specification §2, amended 2026-10-02): the
    myCobot 280 and myAGV records do not describe the removed assembly."""
    for name, text in _record_texts("mycobot280").items():
        assert "myagv" not in text.lower() and "stands alone" not in text, name
    for name, text in _record_texts("myagv").items():
        assert "mycobot" not in text.lower(), name


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_pinned_revisions_recorded(robot):
    revs = {s["revision"] for s in load(robot)["sources"]}
    for rev in robot.revisions:
        assert rev in revs, f"{rev} from the spec is not in {robot.ros} sources"


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
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


def _in_folder(fn: str) -> str:
    """A URDF mesh filename as a path inside the robot folder (`package://<pkg>/` resolves to it)."""
    m = re.match(r"package://[^/]+/(.*)", fn)
    return m.group(1) if m else fn


def _urdf_meshes(robot) -> set[str]:
    """Every mesh the URDF references (a reference inside an XML comment is none)."""
    return {f"{robot.id}/{_in_folder(el.get('filename'))}"
            for el in ET.parse(ROOT / robot.urdf).getroot().iter("mesh")}


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


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_referenced_meshes_are_in_the_manifest(robot):
    listed = set(manifest())
    refs = _urdf_meshes(robot)
    for model in _models(robot):
        refs |= _mjcf_meshes(model)
    missing = sorted(refs - listed)
    assert not missing, f"not in meshes.sha256: {missing}"


#: Upstream folders kept whole: so101.md asks to "preserve the upstream model assets under assets/".
PRESERVED = ("so101/assets/",)


def test_manifest_lists_only_needed_files():
    """meshes.sha256 fetches nothing the robots do not need: a URDF or model reference, a
    texture a referenced COLLADA file loads, the source of a needed derived file, or a
    preserved upstream folder (robot specification §1)."""
    listed = manifest()
    need = set()
    for r in ROBOTS:
        need |= _urdf_meshes(r)
        for model in _models(r):
            need |= _mjcf_meshes(model)
    for rel in sorted(need):
        if rel in listed and fetch_meshes.origin_of(rel)[0] == "derive":
            need.add(fetch_meshes.origin_of(rel)[1])
        if re.search(r"\.dae\.m\d+\.obj$", rel):
            need.add(re.sub(r"\.m\d+\.obj$", ".mtl", rel))
    for rel in sorted(p for p in need if p.endswith(".dae") and (SPECS / p).is_file()):
        need |= {str(Path(rel).parent / img) for img in
                 re.findall(r"<init_from>([^<]+\.(?:png|jpe?g))</init_from>", (SPECS / rel).read_text())}
    extra = sorted(p for p in listed if p not in need and not p.startswith(PRESERVED))
    assert not extra, f"meshes.sha256 lists files nothing needs: {extra}"


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


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_models_load_in_mujoco(robot):
    mujoco = pytest.importorskip("mujoco")
    for path in _models(robot):
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.nbody > 1 and model.njnt > 0, path.name


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
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


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_model_settles_on_a_floor(robot):
    """The model, dropped onto a plane (arms fixed above it), stays finite for 1 s."""
    mujoco = pytest.importorskip("mujoco")
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


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
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


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_visuals_are_official_visual_meshes(robot):
    """Visual geoms show the URDF's (or official MJCF's) visual meshes, or files derived
    from exactly those, never a collision-only mesh."""
    doc = load(robot)
    urdf = ET.parse(ROOT / robot.urdf).getroot()
    vis, col = ({_in_folder(mesh.get("filename")) for link in urdf.iter("link")
                 for mesh in link.findall(f"{kind}/geometry/mesh")} for kind in ("visual", "collision"))
    only_col = col - vis
    for f in _visual_mesh_files(SPECS / robot.id / doc["model"]["mjcf"]):
        base = re.sub(r"^derived_meshes/", "", f)
        base = re.sub(r"(\.m\d+\.obj|\.part\d+\.stl)$", "", base)
        assert base not in only_col, f"visual geom uses collision-only mesh {f}"
        assert "/collision/" not in f, f"visual geom uses a collision mesh file {f}"


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
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


# ------------------------------------------------------------------ derived models


def _stage(src: Path, dst: Path) -> None:
    """Copy the robot folder's own files (a generator rewrites model.xml in place) and hard-link
    the fetched meshes below it (read only), copying where a link is impossible."""
    dst.mkdir(parents=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    for p in src.rglob("*"):
        if p.is_file() and p.parent != src:
            q = dst / p.relative_to(src)
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, q)
            except OSError:
                shutil.copy2(p, q)


#: Generators that print the model instead of writing model.xml.
TO_STDOUT = {"myagv"}
#: Third-party modules a generator needs beyond the standard library.
GENERATOR_NEEDS = {"mycobot280": ("mujoco",), "rosmaster_x3_plus": ("mujoco",), "so101": ("numpy", "scipy")}


@pytest.mark.parametrize("robot", ROBOTS, ids=lambda r: r.id)
def test_derived_model_is_reproducible(robot, tmp_path):
    """A derived model.xml is a reproducible conversion: its generator in tools/models/
    regenerates it byte for byte from the files in the robot folder (robot specification §1)."""
    if load(robot)["model"]["mjcf"] != "model.xml":
        pytest.skip("no derived model")
    for mod in GENERATOR_NEEDS.get(robot.id, ()):
        pytest.importorskip(mod)
    root = tmp_path / "robots_specs"
    shutil.copytree(SPECS / "tools", root / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    _stage(SPECS / robot.id, root / robot.id)
    run = subprocess.run([sys.executable, str(root / "tools" / "models" / f"{robot.id}.py")],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-2000:]
    produced = run.stdout if robot.id in TO_STDOUT else (root / robot.id / "model.xml").read_text()
    assert produced == (SPECS / robot.id / "model.xml").read_text(), \
        f"tools/models/{robot.id}.py does not reproduce {robot.id}/model.xml"
