"""The engines' setup and repair (spec §2.1, §4): a moved or copied checkout repairs itself
(the venv's recorded paths, MolmoSpaces' absolute asset links), a fetch refuses a file that
does not match its pinned digest and leaves no unverified file behind, and the vendored
upstreams are never modified."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import SIM, SHARED


def run(*args, env=None):
    return subprocess.run([sys.executable, *map(str, args)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=60, env=env)


def test_a_moved_venv_is_rewritten_to_its_new_place(tmp_path):
    """uv's absolute paths -- `bin/activate`, script shebangs, `.pth` files and editable
    finders -- name the old checkout until `relocate_venv.py` rewrites them."""
    old, new = tmp_path / "a" / "old" / "sim", tmp_path / "b" / "new" / "sim"
    venv = new / ".venv"
    site = venv / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    (venv / "bin").mkdir()
    files = {
        venv / "bin" / "activate": f"VIRTUAL_ENV='{old}/.venv'\nexport VIRTUAL_ENV\n",
        venv / "bin" / "mjpython": f"#!{old}/.venv/bin/python3\nimport sys\n",
        site / "_upstream.pth": f"{old}/upstream\n",
        site / "__editable___pkg_finder.py": f"MAPPING = {{'pkg': '{old}/upstream/pkg'}}\n",
    }
    for path, text in files.items():
        path.write_text(text)
    tool = SHARED / "tools" / "relocate_venv.py"
    assert run(tool, venv, "--check").returncode == 1
    assert run(tool, venv).returncode == 0
    for path in files:
        text = path.read_text()
        assert str(old) not in text and str(new) in text, (path.name, text)
    assert run(tool, venv, "--check").returncode == 0


def test_moved_asset_links_are_pointed_at_this_checkouts_cache(tmp_path):
    """MolmoSpaces' absolute asset links into a cache that moved are re-pointed into this
    checkout's cache; a link this cannot account for is left alone."""
    assets, cache = tmp_path / "assets", tmp_path / "data" / "mujoco"
    (cache / "objects" / "thor").mkdir(parents=True)
    (cache / "objects" / "thor" / "Apple.xml").write_text("<mujoco/>")
    (assets / "objects").mkdir(parents=True)
    gone = tmp_path / "gone" / "data" / "mujoco"
    (assets / "objects" / "thor").symlink_to(gone / "objects" / "thor")
    (assets / "objects" / "Apple.xml").symlink_to(gone / "objects" / "thor" / "Apple.xml")
    (assets / "objects" / "Pear.xml").symlink_to(gone / "objects" / "thor" / "Pear.xml")
    env = dict(os.environ, MLSPACES_ASSETS_DIR=str(assets), MLSPACES_CACHE_DIR=str(cache))
    out = run(SIM / "molmospaces" / "tools" / "relink_assets.py", env=env)
    assert out.returncode == 0, out.stdout
    assert os.readlink(assets / "objects" / "thor") == str(cache / "objects" / "thor")
    assert (assets / "objects" / "Apple.xml").read_text() == "<mujoco/>"
    assert os.readlink(assets / "objects" / "Pear.xml") == str(gone / "objects" / "thor" / "Pear.xml")


def load_fetch_objects():
    spec = importlib.util.spec_from_file_location("fetch_objects_under_test",
                                                  SHARED / "tools" / "fetch_objects.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_objects_refuses_a_mismatch_and_leaves_no_unverified_file(tmp_path, monkeypatch):
    """One fetched file does not match its pinned digest: nothing is moved into place (not
    even the files that did match), a file already there that does not match is removed,
    and no staging directory is left."""
    fo = load_fetch_objects()
    shared = tmp_path / "simulator" / "shared"
    (shared / "objects").mkdir(parents=True)
    source = tmp_path / "source"
    good, bad = b"verified bytes", b"tampered bytes"
    for name, data in (("apple", good), ("mug", bad)):
        (source / name).mkdir(parents=True)
        (source / name / "textured.obj").write_bytes(data)
    ycb = Path("simulator/shared/objects/ycb")
    stale = tmp_path / ycb / "lemon" / "textured.obj"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    digest = fo.hashlib.sha256
    manifest = shared / "objects" / "ycb.sha256"
    manifest.write_text("".join(f"{d}  {rel}\n" for d, rel in (
        (digest(good).hexdigest(), ycb / "apple" / "textured.obj"),
        (digest(good).hexdigest(), ycb / "mug" / "textured.obj"),
        (digest(good).hexdigest(), ycb / "lemon" / "textured.obj"))))
    monkeypatch.setattr(fo, "SHARED", shared)
    monkeypatch.setattr(fo, "ROOT", tmp_path)
    monkeypatch.setattr(fo, "MANIFEST", manifest)
    (source / "lemon").mkdir()
    (source / "lemon" / "textured.obj").write_bytes(good)
    monkeypatch.setattr(fo, "URL", source.as_uri() + "/{name}/{file}")
    assert fo.main([]) == 1
    assert not [p for p in (tmp_path / ycb).rglob("*") if p.is_file()]
    assert not list((shared / "objects").glob(".fetch-objects-*"))
    # with every file matching, the same fetch puts all of them in place
    (source / "mug" / "textured.obj").write_bytes(good)
    assert fo.main([]) == 0
    assert sorted(p.parent.name for p in (tmp_path / ycb).rglob("*.obj")) == ["apple", "lemon", "mug"]
    assert fo.main(["--check"]) == 0


UPSTREAMS = [SIM / "molmospaces" / "upstream", SIM / "robocasa" / "upstream" / "robosuite",
             SIM / "robocasa" / "upstream" / "robocasa"]


@pytest.mark.parametrize("upstream", UPSTREAMS, ids=lambda p: str(p.relative_to(SIM)))
def test_vendored_upstreams_are_unmodified(upstream):
    if not (upstream / ".git").exists():
        pytest.skip(f"{upstream.relative_to(SIM)} is not checked out (run.sh setup)")
    out = subprocess.run(["git", "-C", str(upstream), "status", "--porcelain"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout == "", out.stdout[-2000:]
