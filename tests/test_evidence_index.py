"""Workspace spec §3: `evidences/index.md` has a passing case, with its pictures, for
every robot id of `robots_specs/high_level_spec.md` on every engine.

    uv run --no-project --with pytest --with pyyaml --with pillow pytest tests/test_evidence_index.py

The expectations are derived here from the robot registry and its interface files (which
cameras are required), independently of the evidence script that wrote the index.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
EVID = Path(os.environ.get("EVIDENCE_DIR", REPO / "evidences"))  # override for a trial run
SPECS = REPO / "robots_specs"
# The engines (workspace spec §2, simulator spec §2.1) and their default scenes.
ENGINES = {"molmospaces": "ithor:1", "robocasa": "robocasa:1-1"}


def registry():
    text = (SPECS / "high_level_spec.md").read_text()
    out = {}
    for sec in re.split(r"\n## ", text)[1:]:
        m = re.search(r"\*\*Robot id:\*\*\s*`([^`]+)`", sec)
        if not m:
            continue
        kind = re.search(r"\*\*Kind:\*\*\s*`([^`]+)`", sec).group(1)
        comps = [x.group(1) for x in re.finditer(r"\*\*`(?:base|arm)`:\*\*\s*`([^`]+)`", sec)]
        out[m.group(1)] = {"kind": kind, "owners": comps or [m.group(1)]}
    return out


ROBOTS = registry()


def required_camera_files(rid):
    files = []
    for owner in ROBOTS[rid]["owners"]:
        f = next(p for p in (SPECS / owner / "ros.yml", SPECS / owner / "ros2.yml") if p.exists())
        iface = yaml.safe_load(f.read_text())
        for cam in (iface.get("sensors") or {}).get("cameras") or []:
            if not cam.get("optional"):
                files.append("cam__" + cam["image_topic"].strip("/").replace("/", "__") + ".png")
    return files


def index_rows():
    """The case table of index.md: (engine, robot) -> row cells and the links in it."""
    text = (EVID / "index.md").read_text()
    rows = {}
    header = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "Engine":
            header = cells
            continue
        if header is None or set(cells[0]) <= {"-", " "}:
            continue
        row = dict(zip(header, cells))
        robot = row["Robot"].strip("`")
        row["_links"] = re.findall(r"\]\(([^)]+)\)", line)
        rows[(row["Engine"], robot)] = row
    return rows


def test_registry_has_robots():
    assert len(ROBOTS) >= 6 and "myagv_mycobot280" in ROBOTS


def test_index_exists():
    assert (EVID / "index.md").is_file(), "evidences/index.md missing: run tests/evidence.sh"


def _png(path: Path, size=None):
    assert path.is_file(), f"missing picture {path.relative_to(REPO)}"
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            assert im.size == size, f"{path.name}: {im.size}"
        lo, hi = im.getextrema()[0]
        assert hi - lo > 10, f"{path.name}: a flat picture"


@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("rid", list(ROBOTS))
def test_passing_case_with_pictures(engine, rid):
    rows = index_rows()
    assert (engine, rid) in rows, f"no case for {rid} on {engine} in evidences/index.md"
    row = rows[(engine, rid)]
    assert row["Result"] == "PASS", f"{engine}/{rid}: {row['Result']}"
    assert row["Scene"] == ENGINES[engine], f"not the engine's default scene: {row['Scene']}"
    placement = "worktop" if ROBOTS[rid]["kind"] == "arm" else "floor"
    assert row["Placement"] == placement
    links = set(row["_links"])
    d = f"{engine}/{rid}"
    # the scene picture and every required camera's picture, linked from the index
    want = [f"{d}/scene.png"] + [f"{d}/{f}" for f in required_camera_files(rid)]
    for rel in want:
        assert rel in links, f"index row of {d} does not link {rel}"
    _png(EVID / d / "scene.png", size=(1280, 720))
    for f in required_camera_files(rid):
        _png(EVID / d / f)
    # the case record agrees: every item passed
    case = json.loads((EVID / d / "case.json").read_text())
    assert case["pass"] is True and case["robot"] == rid and case["engine"] == engine
    assert all(case["checks"].values()), case["checks"]
    assert case["readiness_line"].startswith("spawn ready:")
    smoke = json.loads((EVID / d / "smoke.json").read_text())
    assert smoke["motions"] and all(m["pass"] for m in smoke["motions"])
    for m in smoke["motions"]:
        for when in ("before", "after"):
            assert (EVID / d / f"motion_{m['name']}_{when}.png").is_file()
    fleet = case["items"]["fleet"]
    profiles = {re.search(r"^id:\s*(\S+)", p.read_text(), re.M).group(1)
                for p in (REPO / "robot_console" / "src" / "robot_console" / "profiles").glob("*.yaml")}
    expected = profiles & ({rid} | set(ROBOTS[rid]["owners"]))
    assert expected and {f["profile"] for f in fleet} == expected, f"{d}: console checks {fleet}"
    for f in fleet:
        text = (EVID / d / f["file"]).read_text()
        assert f["exit_code"] == 0 and "# exit code: 0" in text, f"{d}: {f['file']}"


#: The worktop objects (simulator spec §2.3): staged around every arm on the worktop.
WORKTOP_OBJECTS = {"apple", "plate", "bowl", "mug", "banana", "lemon"}


@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("rid", [r for r, v in ROBOTS.items() if v["kind"] == "arm"])
def test_worktop_cases_record_their_staged_objects(engine, rid):
    """Workspace spec §3: a worktop case records the six objects staged around the arm,
    each with its staged pose, and which scene objects were cleared for them."""
    d = f"{engine}/{rid}"
    case = json.loads((EVID / d / "case.json").read_text())
    spawned = case.get("spawned") or {}
    assert spawned.get("placement") == "worktop"
    staged = spawned.get("staged") or {}
    assert set(staged) == WORKTOP_OBJECTS, f"{d}: staged {sorted(staged)}"
    for name, o in staged.items():
        assert o["body"] == f"task_{name}" and len(o["pos"]) == 3 and len(o["quat"]) == 4, name
        # on the worktop, around the arm (the farthest, the bowl, is 0.42 m out)
        dxy = math.hypot(o["pos"][0] - spawned["xyz"][0], o["pos"][1] - spawned["xyz"][1])
        assert dxy < 0.45, f"{d}: {name} {dxy:.3f} m from the arm"
    assert isinstance(spawned.get("cleared"), list)
    assert case["checks"].get("worktop_objects") is True
    # and the index says so
    text = (EVID / "index.md").read_text()
    section = text.split(f"### {engine} / {rid} ")[1].split("\n### ")[0]
    assert "Worktop objects staged: " in section
    for name in WORKTOP_OBJECTS:
        assert name in section.split("Worktop objects staged: ")[1].splitlines()[0], name
