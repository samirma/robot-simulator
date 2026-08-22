#!/usr/bin/env python
"""Download and link every scene in a dataset so it is usable offline.

Scenes are installed per-file on demand by default, so a bulk archive fetch is
not enough -- each house has to be run through
``install_scene_with_objects_and_grasps_from_path`` to pull its meshes and
grasps and create the symlinks MuJoCo loads from.

    python tools/prefetch_scenes.py ithor [--split train] [--limit N]
"""

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="ithor, procthor-10k, procthor-objaverse, holodeck-objaverse")
    ap.add_argument("--split", default="train")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--limit", type=int, default=None, help="stop after N scenes")
    ap.add_argument("--start", type=int, default=0, help="skip the first N scenes")
    args = ap.parse_args()

    from molmo_spaces.molmo_spaces_constants import get_scenes
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    scenes = get_scenes(args.dataset, args.split)
    entries = scenes[args.split] if isinstance(scenes, dict) and args.split in scenes else scenes
    # The index map is keyed by scene index (1-based) and is sparse -- iTHOR
    # numbers its floor plans 1-30, 201-230, ... so most slots are None.
    values = list(entries.values() if isinstance(entries, dict) else entries)
    entries = [e for e in values if e is not None][args.start :]
    if args.limit is not None:
        entries = entries[: args.limit]

    total = len(entries)
    print(f"installing {total} {args.dataset} scenes", file=sys.stderr)

    failed: list[tuple[str, str]] = []
    started = time.monotonic()
    for i, entry in enumerate(entries):
        path = Path(entry[args.variant] if isinstance(entry, dict) else entry)
        try:
            install_scene_with_objects_and_grasps_from_path(path)
        except Exception as exc:  # keep going; one bad scene shouldn't stop the batch
            failed.append((path.name, f"{type(exc).__name__}: {exc}"))
        if (i + 1) % 10 == 0 or i + 1 == total:
            rate = (i + 1) / max(1e-6, time.monotonic() - started)
            print(
                f"  {i + 1}/{total}  ({rate:.1f}/s, {len(failed)} failed)",
                file=sys.stderr,
                flush=True,
            )

    if failed:
        print(f"\n{len(failed)} scene(s) failed:", file=sys.stderr)
        for name, err in failed[:20]:
            print(f"  {name}: {err}", file=sys.stderr)
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more", file=sys.stderr)
        return 1

    print(f"installed {total} scenes with no failures", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
