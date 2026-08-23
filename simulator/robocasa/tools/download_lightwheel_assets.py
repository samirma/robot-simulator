#!/usr/bin/env python
"""Fetch fixture/object assets that robocasa v1.0's downloader misses.

``python -m robocasa.scripts.download_kitchen_assets`` (v1.0) 404s on the
lightwheel fixtures/objects: it still points at ``fixtures_lightwheel.zip`` /
``objects_lightwheel.zip`` in nvidia's HF repo, which has since been renamed
(PhysicalAI-Kitchen-Assets -> PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF)
and restructured into per-category zips. It also never fetches the base
``fixtures.zip`` from robocasa/robocasa-assets (the git clone only carries git
LFS pointers for those). This script downloads all three and extracts them
into robocasa's asset tree. Idempotent: existing files are simply overwritten
by identical content.
"""

import os

import robocasa
from huggingface_hub import hf_hub_download, list_repo_files
from zipfile import ZipFile

LW_REPO = "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"
RC_REPO = "robocasa/robocasa-assets"
ASSETS = os.path.join(robocasa.__path__[0], "models/assets")


def fetch_and_extract(repo, filename, target):
    print(f">> {repo}/{filename} -> {target}")
    p = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename)
    with ZipFile(p) as z:
        z.extractall(path=target)
    os.remove(p)


def main():
    # Base fixtures (cabinet housings, panels, registry), robocasa-hosted.
    fetch_and_extract(RC_REPO, "fixtures.zip", ASSETS)

    # Lightwheel fixtures and objects, per-category zips in the renamed repo.
    targets = {
        "fixtures_lightwheel/": os.path.join(ASSETS, "fixtures"),
        "objects_lightwheel/": os.path.join(ASSETS, "objects/lightwheel"),
    }
    files = list_repo_files(LW_REPO, repo_type="dataset")
    for prefix, target in targets.items():
        os.makedirs(target, exist_ok=True)
        for f in files:
            if f.startswith(prefix) and f.endswith(".zip"):
                fetch_and_extract(LW_REPO, f, target)
    print(">> done")


if __name__ == "__main__":
    main()
