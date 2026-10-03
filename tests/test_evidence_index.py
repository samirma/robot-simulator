"""Workspace spec §3: `evidences/index.md` has a passing case, with its pictures, for
every robot id of the robot registry -- the robot files `robots_specs/<id>.md` (§2) -- on
every engine, and every worktop case records its six staged objects.

    uv run --no-project --with pytest --with pyyaml --with pillow pytest tests/test_evidence_index.py

The expectations are derived here from the robot files and their interface files (which
cameras and motions are required), independently of the evidence script that wrote the
index. With no `evidences/` (before `tests/evidence.sh` has run) every case test fails,
naming the missing file: a missing case is a defect (§3).
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
    """The robot files: every `robots_specs/*.md` (the specification and the schema aside)
    that records a robot id, with its kind and the file's name."""
    out = {}
    for p in sorted(SPECS.glob("*.md")):
        if p.name in ("high_level_spec.md", "SCHEMA.md"):
            continue
        text = p.read_text()
        m = re.search(r"\*\*Robot id:\*\*\s*`([^`]+)`", text)
        if not m:
            continue
        kind = re.search(r"\*\*Kind:\*\*\s*`([^`]+)`", text)
        out[m.group(1)] = {"kind": kind.group(1) if kind else None, "file": p.name}
    return out


ROBOTS = registry()


def interface_files(rid):
    return [p for p in (SPECS / rid / "ros.yml", SPECS / rid / "ros2.yml") if p.exists()]


def interface(rid):
    return yaml.safe_load(interface_files(rid)[0].read_text())


def required_camera_files(rid):
    return ["cam__" + cam["image_topic"].strip("/").replace("/", "__") + ".png"
            for cam in (interface(rid).get("sensors") or {}).get("cameras") or []
            if not cam.get("optional")]


def required_motions(rid):
    """Workspace spec §3's smoke-run motions, as the robot records them (its interface
    file's `motions`): a wheeled base's drive as forward, back, sideways (left and right)
    and a turn, each head joint, and every other recorded motion -- walk, arm, gripper,
    action group -- by its id."""
    out = set()
    for m in interface(rid).get("motions") or []:
        if m["id"] == "drive":
            out |= {"drive_forward", "drive_back", "drive_left", "drive_right", "drive_turn"}
        elif m["id"] == "head":
            out.add(m["joint"])
        else:
            out.add(m["id"])
    return out


def walk_row(rid):
    return next((m for m in interface(rid).get("motions") or [] if m["id"] == "walk"), None)


def walk_step_periods(rid):
    """Gait periods per walk step, as the robot's record defines it
    (`motions[walk].command.step_definition`; x_move_amplitude in m per step, one step = one
    gait period, amended 2026-10-03)."""
    cmd = walk_row(rid)["command"]
    unit = next(f["unit"] for f in cmd["fields"] if f["field"] == "x_move_amplitude")
    assert cmd.get("step_definition") and "one step = one gait period" in unit, \
        f"{rid}: the record no longer defines a walk step as one gait period ({unit!r})"
    return 1


def check_walk(d, rid, m):
    """A passing walk against the record: its commanded displacement is x_move_amplitude per
    completed gait period, and the measured forward displacement is within the recorded
    walk-displacement tolerance of it (`tolerances`, amended 2026-10-03)."""
    prm = m["command"]["set_param"]
    period = prm["period_time"] / 1000.0
    periods = m["gait_periods_completed"]
    assert periods >= 1 and periods == round(m["walking_time_s"] / period), (d, periods, m["walking_time_s"])
    commanded = prm["x_move_amplitude"] * periods / walk_step_periods(rid)
    assert m["commanded"]["forward_m"] == pytest.approx(commanded, abs=1e-4), (d, m["commanded"])
    tol = next(t["tolerance"] for t in interface(rid)["tolerances"]
               if t["figure"].startswith("walk displacement"))
    bound = max(tol.get("relative", 0.0) * abs(commanded), tol.get("absolute", 0.0))
    fwd = m["measured"]["forward_m"]
    assert fwd > 0 and abs(fwd - commanded) <= bound + 1e-4, \
        f"{d}: walked {fwd} m vs commanded {commanded:.4f} m (bound {bound:.4f})"


def check_action_group(d, rid, m):
    """The action group played is the record's smoke example, with its basis reported: an
    estimated group is labelled as one (`motions[action_group].command.smoke_example`,
    `smoke_example_basis`, amended 2026-10-03)."""
    cmd = next(x for x in interface(rid)["motions"] if x["id"] == "action_group")["command"]
    want = cmd.get("smoke_example") or cmd["example"]
    assert m["command"]["message"] == want, (d, m["command"]["message"], want)
    basis = cmd.get("smoke_example_basis")
    assert m["command"].get("smoke_example_basis") == basis, (d, m["command"])
    if basis == "estimate":
        assert "ESTIMATED" in m.get("note", ""), f"{d}: the estimated action group is not labelled"


def console_profile_ids():
    """The console's packaged profile ids (which profile names the robot is the console's
    own fact; its verdict is the fleet check's exit status)."""
    return {re.search(r"^id:\s*(\S+)", p.read_text(), re.M).group(1)
            for p in (REPO / "robot_console" / "src" / "robot_console" / "profiles").glob("*.yaml")}


def index_text():
    path = EVID / "index.md"
    assert path.is_file(), f"{path} missing: run tests/evidence.sh (a missing case is a defect)"
    return path.read_text()


def case_record(d):
    path = EVID / d / "case.json"
    assert path.is_file(), f"{path} missing: run tests/evidence.sh --engine/--robot for it"
    return json.loads(path.read_text())


def index_rows():
    """The case table of index.md: (engine, robot) -> row cells and the links in it."""
    text = index_text()
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
    """The registry the expectations come from (workspace spec §2): each robot file is named
    by its id, has a kind and its folder, and -- every robot a single body -- exactly one ROS
    interface file."""
    assert ROBOTS
    for rid, v in ROBOTS.items():
        assert v["file"] == f"{rid}.md" and v["kind"] and (SPECS / rid).is_dir(), (rid, v)
        assert len(interface_files(rid)) == 1, (rid, interface_files(rid))


def test_index_exists():
    index_text()


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
    # the case record agrees: every item passed, on the default scene, at the placement
    case = case_record(d)
    assert case["pass"] is True and case["robot"] == rid and case["engine"] == engine
    assert all(case["checks"].values()), case["checks"]
    assert case["scene"] == ENGINES[engine] and (case.get("spawned") or {}).get("placement") == placement
    assert case["readiness_line"].startswith("spawn ready:")
    assert case["spawn_command"].startswith(f"simulator/spawn.sh {rid} --placement {placement}")
    assert re.match(r"simulator/(kitchen\.sh start|\w+/run\.sh start)", case["simulation_command"])
    wire = case["wire"]          # one wire per robot (simulator spec §2.3)
    assert wire["robot"] == rid and wire["url"].startswith("ws://"), f"{d}: wire {wire}"
    # the smoke run: every motion of §3's list for this robot, through its wire, passing; a
    # drive or walk followed by the recorded stop
    smoke = json.loads((EVID / d / "smoke.json").read_text())
    run = {m["name"]: m for m in smoke["motions"]}
    assert run and all(m["pass"] for m in run.values()), {n: m["pass"] for n, m in run.items()}
    missing = required_motions(rid) - set(run)
    assert not missing, f"{d}: smoke run lacks {sorted(missing)}"
    for name, m in run.items():
        assert m.get("wire") == wire["url"], f"{d}: {name} not commanded over the robot's wire"
        if name.startswith("drive_") or name == "walk":
            assert m.get("stop"), f"{d}: {name} not followed by the recorded stop"
        for when in ("before", "after"):
            assert (EVID / d / f"motion_{name}_{when}.png").is_file()
    if walk_row(rid):
        check_walk(d, rid, run["walk"])
    if "action_group" in run:
        check_action_group(d, rid, run["action_group"])
    # the console check for the profile naming the robot, if any
    fleet = case["items"]["fleet"]
    expected = console_profile_ids() & {rid}
    assert {f["profile"] for f in fleet} == expected, f"{d}: console checks {fleet}"
    for f in fleet:
        text = (EVID / d / f["file"]).read_text()
        assert f["exit_code"] == 0 and "# exit code: 0" in text, f"{d}: {f['file']}"


#: The worktop objects (simulator spec §2.3): staged around every arm on the worktop.
WORKTOP_OBJECTS = {"apple", "plate", "bowl", "mug", "banana", "lemon"}


@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("rid", [r for r, v in ROBOTS.items() if v["kind"] == "arm"])
def test_worktop_cases_record_their_staged_objects(engine, rid):
    """Workspace spec §3: a worktop case records the six objects staged around the arm,
    each with its staged pose, and which scene objects were cleared for them; its scene
    picture shows them."""
    d = f"{engine}/{rid}"
    case = case_record(d)
    spawned = case.get("spawned") or {}
    assert spawned.get("placement") == "worktop"
    staged = spawned.get("staged") or {}
    assert set(staged) == WORKTOP_OBJECTS, f"{d}: staged {sorted(staged)}"
    for name, o in staged.items():
        assert o["body"] == f"task_{name}" and len(o["pos"]) == 3 and len(o["quat"]) == 4, name
        # around the arm: inside the staging area, 0.55 m around the arm's base frame
        # (simulator spec §2.3; the farthest reference pose, the bowl's, is 0.42 m out)
        dxy = math.hypot(o["pos"][0] - spawned["xyz"][0], o["pos"][1] - spawned["xyz"][1])
        assert dxy < 0.55, f"{d}: {name} {dxy:.3f} m from the arm"
    assert isinstance(spawned.get("cleared"), list)
    assert case["checks"].get("worktop_objects") is True
    # the scene picture shows them
    assert case["checks"].get("worktop_objects_in_picture") is True, \
        case["items"]["scene"].get("worktop_objects_shown")
    # and the index says so
    text = index_text()
    section = text.split(f"### {engine} / {rid} ")[1].split("\n### ")[0]
    assert "Worktop objects staged: " in section
    for name in WORKTOP_OBJECTS:
        assert name in section.split("Worktop objects staged: ")[1].splitlines()[0], name
