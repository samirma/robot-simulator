#!/usr/bin/env python
"""Re-point the assets/ symlink tree at this checkout's data/ cache.

    python tools/relink_assets.py [--dry-run]      (or: ./run.sh repair)

MolmoSpaces builds assets/ as a tree of *absolute* symlinks into the versioned cache
under data/mujoco/. Move or copy the checkout and every one of them still names the old
location: nothing errors until a scene is resolved, and then it fails as a missing
``FloorPlan1_physics.xml`` inside an installer that believes the house is installed --
its ``.*_complete_links`` markers say so, so it never recreates the links itself.

A link is rewritten only when its target is absolute, does not exist, and has a
``/data/mujoco/`` component whose remainder *does* exist in this checkout's cache.
Anything else is left alone and counted, so a link this cannot account for is reported
rather than guessed at. Absolute targets are kept, because that is what upstream writes.

Stdlib only, so it runs under any python and needs no working venv.
"""

import argparse
import os
import sys
from pathlib import Path

MARKER = "/data/mujoco/"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = ap.parse_args()

    sim_root = Path(__file__).resolve().parent.parent
    assets = Path(os.environ.get("MLSPACES_ASSETS_DIR", sim_root / "assets"))
    cache = Path(os.environ.get("MLSPACES_CACHE_DIR", sim_root / "data" / "mujoco"))
    if not assets.is_dir():
        print(f"no asset tree at {assets}; nothing to relink", file=sys.stderr)
        return 0

    fixed = unresolved = 0
    for dirpath, dirnames, filenames in os.walk(assets):
        # os.walk does not descend into symlinked directories, but it does list them in
        # dirnames -- which is exactly where most of this tree's links are.
        for name in dirnames + filenames:
            link = Path(dirpath) / name
            if not link.is_symlink():
                continue
            target = os.readlink(link)
            if not os.path.isabs(target) or os.path.exists(target):
                continue
            _, found, rest = target.partition(MARKER)
            new = cache / rest if found else None
            if new is None or not new.exists():
                unresolved += 1
                continue
            if not args.dry_run:
                link.unlink()
                link.symlink_to(new)
            fixed += 1

    # Quiet when there is nothing to do: every launch runs this, and it takes ~0.2 s.
    if fixed:
        verb = "would relink" if args.dry_run else "relinked"
        print(f"{verb} {fixed} asset links into {cache}", file=sys.stderr)
    if unresolved:
        print(
            f"warning: {unresolved} broken links point outside {cache} or at files it "
            "does not hold; `./run.sh assets default` rebuilds what it can",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
