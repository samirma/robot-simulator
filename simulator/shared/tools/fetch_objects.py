#!/usr/bin/env python3
"""Fetch the worktop objects' YCB meshes the repository does not commit.

    python3 simulator/shared/tools/fetch_objects.py            fetch what is missing
    python3 simulator/shared/tools/fetch_objects.py --check    verify only, fetch nothing

The six objects the worktop staging uses (`worktop_objects.py`: apple, plate, bowl, mug,
banana, lemon), each a `textured.obj`, `textured.mtl` and `texture_map.png`, come from
elpis-lab/YCB_Dataset (MIT wrapper of the YCB Object and Model Set; see
`simulator/shared/objects/LICENSES.md`) at the pinned commit below, whose files are byte
for byte the ones the reference project (github.com/samirma/robot-simulator rev 34547ae,
`simulator/shared/tasks/assets/ycb/`) vendored. `simulator/shared/objects/ycb.sha256`
lists every file (`<sha256>  <path from the repo root>`).

Every file is downloaded to a staging directory first, checked against the manifest, and
only then moved into place. A mismatch is refused (non-zero exit) and leaves no unverified
file behind; a file already in place that does not match is removed. Files already
matching are left alone, so re-running is a no-op. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

SHARED = Path(__file__).resolve().parent.parent          # simulator/shared/
ROOT = SHARED.parent.parent                              # repository root
MANIFEST = SHARED / "objects" / "ycb.sha256"
REPO = "elpis-lab/YCB_Dataset"
COMMIT = "9e8c6488a2ff673d9aa48a91492fb89423c1b106"      # "Fit to OBB", 2025-08-22
URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/ycb/{{name}}/{{file}}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict[Path, str]:
    out = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, rel = line.split(None, 1)
            out[Path(rel.strip())] = digest
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify only, fetch nothing")
    args = ap.parse_args(argv)
    want = manifest()
    todo = []
    for rel, digest in want.items():
        dest = ROOT / rel
        if dest.is_file():
            if sha256(dest) == digest:
                continue
            print(f"removing {rel}: it does not match {MANIFEST.name}", file=sys.stderr)
            dest.unlink()
        todo.append((rel, digest))
    if args.check:
        for rel, _ in todo:
            print(f"missing or wrong: {rel}", file=sys.stderr)
        return 1 if todo else 0
    if not todo:
        print(f"worktop objects: all {len(want)} files present and verified")
        return 0
    stage = Path(tempfile.mkdtemp(prefix=".fetch-objects-", dir=SHARED / "objects"))
    try:
        fetched = []
        for rel, digest in todo:
            name, file = rel.parts[-2], rel.parts[-1]
            url = URL.format(name=name, file=file)
            tmp = stage / name / file
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                    shutil.copyfileobj(r, fh)
            except OSError as exc:
                print(f"error: could not fetch {url}: {exc}", file=sys.stderr)
                return 1
            got = sha256(tmp)
            if got != digest:
                print(f"error: {rel} from {url} has sha256 {got}, expected {digest}; refused",
                      file=sys.stderr)
                return 1
            fetched.append((tmp, ROOT / rel))
        for tmp, dest in fetched:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), dest)
        print(f"worktop objects: fetched and verified {len(fetched)} file(s) from {REPO}@{COMMIT[:8]}")
        return 0
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
