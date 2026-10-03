"""fetch_meshes.py keeps only verified files: a fetched file whose sha256 does not match
meshes.sha256 is refused and nothing unverified is left behind (simulator spec §2.1).

Runs on a temporary robots_specs/ tree with the git fetch stubbed; no network.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import fetch_meshes  # noqa: E402

GOOD = b"the pinned upstream bytes"
REL = "so101/assets/part.stl"          # an SO-ARM100 file (fetch_meshes.SOURCES)


@pytest.fixture
def specs(tmp_path, monkeypatch):
    specs = tmp_path / "robots_specs"
    (specs / "so101").mkdir(parents=True)
    manifest = specs / "meshes.sha256"
    manifest.write_text(f"{hashlib.sha256(GOOD).hexdigest()}  robots_specs/{REL}\n")
    monkeypatch.setattr(fetch_meshes, "SPECS", specs)
    monkeypatch.setattr(fetch_meshes, "MANIFEST", manifest)
    return specs


def _serve(monkeypatch, data: bytes) -> list[str]:
    calls = []

    def fetch_git(key, wanted, stage):
        calls.append(key)
        for rel in wanted.values():
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            (stage / rel).write_bytes(data)

    monkeypatch.setattr(fetch_meshes, "fetch_git", fetch_git)
    return calls


def test_a_mismatching_fetch_is_refused_and_nothing_is_kept(specs, monkeypatch):
    calls = _serve(monkeypatch, b"not the pinned bytes")
    (specs / REL).parent.mkdir(parents=True)
    (specs / REL).write_bytes(b"a stale copy")            # an unverified file already in place
    with pytest.raises(fetch_meshes.Refused) as exc:
        fetch_meshes.main([])
    assert exc.value.code == 1
    assert calls == ["SO-ARM100"]
    assert not (specs / REL).exists(), "no unverified file is left in place"
    assert not list(specs.glob(".fetch-*")), "the staging directory is removed"


def test_a_matching_fetch_is_kept(specs, monkeypatch):
    _serve(monkeypatch, GOOD)
    assert fetch_meshes.main([]) == 0
    assert (specs / REL).read_bytes() == GOOD
    assert not list(specs.glob(".fetch-*"))
    assert fetch_meshes.main(["--check"]) == 0
