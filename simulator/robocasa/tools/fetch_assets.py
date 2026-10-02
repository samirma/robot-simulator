#!/usr/bin/env python
"""Build and fill the RoboCasa engine's generated asset tree, `simulator/robocasa/assets/`.

    python tools/fetch_assets.py base                 # the tree itself (upstream's own assets)
    python tools/fetch_assets.py tex fixtures ...     # one or more download sources
    python tools/fetch_assets.py --list

robocasa reads every kitchen file relative to `robocasa.models.assets_root`, which is
`<robocasa>/models/assets` inside the upstream checkout. The upstream checkout is never
modified (spec §4), so the scene loader points `assets_root` at this engine's `assets/`
instead, and this tool generates that tree: first a copy-on-write copy of the upstream's
committed `models/assets` (the arena, layouts, styles, base fixtures), then the
downloaded sources extracted over it, exactly where robocasa's own downloader and the
reference setup extracted them inside the checkout. The resulting files are the same, so
the compiled kitchen is the same.

Sources (the same zips robocasa v1.0's `download_kitchen_assets` and the reference
`tools/download_lightwheel_assets.py` fetch):

* `tex`             robocasa/robocasa-assets textures.zip           -> textures/
* `tex_generative`  robocasa/robocasa-assets generative_textures.zip -> generative_textures/
* `fixtures`        robocasa/robocasa-assets fixtures.zip, and every per-category
                    fixtures_lightwheel/*.zip of the renamed nvidia repo -> fixtures/
* `objs_objaverse`  robocasa/robocasa-assets objaverse.zip          -> objects/objaverse/
* `objs_aigen`      robocasa/robocasa-assets aigen_objs.zip         -> objects/aigen_objs/
* `objs_lightwheel` every objects_lightwheel/*.zip of the nvidia repo -> objects/lightwheel/

`setup` fetches `base tex fixtures objs_lightwheel`, which a bare kitchen needs (its
fixtures place lightwheel accessories such as knife blocks); `assets` fetches
every source. Each source leaves a completion marker so a re-run is a no-op.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ENGINE_ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("ROBOCASA_ASSETS_DIR", ENGINE_ROOT / "assets"))
ROBOCASA_DIR = Path(os.environ.get("ROBOCASA_DIR", ENGINE_ROOT / "upstream" / "robocasa"))
UPSTREAM_ASSETS = ROBOCASA_DIR / "robocasa" / "models" / "assets"
DOWNLOADS = ENGINE_ROOT / "data" / "downloads"

RC_REPO = "robocasa/robocasa-assets"
LW_REPO = "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"

# source -> list of (repo, filename-or-prefix, extract parent, is_prefix)
SOURCES = {
    "tex": [(RC_REPO, "textures.zip", ".", False)],
    "tex_generative": [(RC_REPO, "generative_textures.zip", ".", False)],
    "fixtures": [(RC_REPO, "fixtures.zip", ".", False),
                 (LW_REPO, "fixtures_lightwheel/", "fixtures", True)],
    "objs_objaverse": [(RC_REPO, "objaverse.zip", "objects", False)],
    "objs_aigen": [(RC_REPO, "aigen_objs.zip", "objects", False)],
    "objs_lightwheel": [(LW_REPO, "objects_lightwheel/", "objects/lightwheel", True)],
}
DEFAULT = ("base", "tex", "fixtures", "objs_lightwheel")


def marker(name: str) -> Path:
    return ASSETS / f".complete_{name}"


def build_base() -> None:
    """Copy upstream's committed asset tree in (APFS clones on macOS: no extra space)."""
    if marker("base").exists():
        return
    if not UPSTREAM_ASSETS.is_dir():
        raise SystemExit(f"no upstream asset tree at {UPSTREAM_ASSETS}; run ./run.sh setup")
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f">> copying upstream assets {UPSTREAM_ASSETS} -> {ASSETS}", file=sys.stderr)
    flags = ["-Rc"] if sys.platform == "darwin" else ["-R", "--reflink=auto"]
    for child in sorted(UPSTREAM_ASSETS.iterdir()):
        dest = ASSETS / child.name
        if dest.exists():
            continue
        subprocess.run(["cp", *flags, str(child), str(dest)], check=True)
    marker("base").touch()


def download(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            return Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename,
                                        local_dir=str(DOWNLOADS / repo.replace("/", "__"))))
        except Exception as exc:  # network hiccup: retry, then fail loudly
            print(f"   download of {repo}/{filename} failed (try {attempt + 1}): {exc}",
                  file=sys.stderr)
    raise SystemExit(f"could not download {repo}/{filename}")


def extract(zip_path: Path, parent: Path) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as z:
        z.extractall(path=parent)
    zip_path.unlink()


def fetch(name: str) -> None:
    if name == "base":
        build_base()
        return
    if marker(name).exists():
        print(f">> {name}: already fetched", file=sys.stderr)
        return
    build_base()
    for repo, fname, parent, is_prefix in SOURCES[name]:
        if is_prefix:
            from huggingface_hub import list_repo_files

            files = [f for f in list_repo_files(repo, repo_type="dataset")
                     if f.startswith(fname) and f.endswith(".zip")]
        else:
            files = [fname]
        for f in files:
            print(f">> {name}: {repo}/{f}", file=sys.stderr)
            extract(download(repo, f), ASSETS / parent)
    marker(name).touch()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="*", help="base, default, all, or " + ", ".join(SOURCES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(["base", *SOURCES]))
        return 0
    wanted: list[str] = []
    for s in args.sources or ["default"]:
        if s == "default":
            wanted += DEFAULT
        elif s == "all":
            wanted += ["base", *SOURCES]
        elif s == "base" or s in SOURCES:
            wanted.append(s)
        else:
            raise SystemExit(f"unknown asset source {s!r} (one of: base, {', '.join(SOURCES)})")
    for s in dict.fromkeys(wanted):
        fetch(s)
    shutil.rmtree(DOWNLOADS, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
