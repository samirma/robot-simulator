#!/usr/bin/env python
"""Resolve a scene reference to an absolute MJCF path, installing it if needed.

    python tools/resolve_scene.py ithor 1 [--split train] [--variant base]

Prints the absolute .xml path on stdout (and nothing else on stdout, so it is
safe to use in ``$(...)``). Progress and errors go to stderr.

This wraps the same two calls that ``scripts/datagen/fetch_assets.py`` uses --
``get_scenes`` plus ``install_scene_with_objects_and_grasps_from_path`` -- but
returns the path instead of only downloading.
"""

import argparse
import contextlib
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="ithor, procthor-10k, procthor-objaverse, holodeck-objaverse")
    ap.add_argument("index", type=int, nargs="?", default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--variant", default="base")
    ap.add_argument(
        "--no-install", action="store_true", help="only resolve, do not download assets"
    )
    args = ap.parse_args()

    # molmo_spaces prints progress ("Using SCENES_ROOT: ...") to stdout, which
    # would corrupt the single path this script contracts to emit. Send all
    # library chatter to stderr and write the result to the real stdout at the end.
    real_stdout = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        return _resolve(args, real_stdout)


def _resolve(args: argparse.Namespace, out) -> int:
    from molmo_spaces.molmo_spaces_constants import get_scenes
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    scenes = get_scenes(args.dataset, args.split)
    # get_scenes returns a split-keyed map; older callers pass the split twice.
    entries = scenes[args.split] if isinstance(scenes, dict) and args.split in scenes else scenes

    try:
        entry = entries[args.index]
    except (IndexError, KeyError):
        print(
            f"index {args.index} out of range for {args.dataset}/{args.split} "
            f"({len(entries)} scenes)",
            file=sys.stderr,
        )
        return 1

    # The index is a sparse map, not a list: an index inside range(len(entries)) can
    # still map to None. Say so plainly instead of dying in Path(None) further down.
    if entry is None:
        usable = [i for i, v in entries.items() if v is not None] if hasattr(
            entries, "items"
        ) else []
        print(
            f"{args.dataset}/{args.split}#{args.index} is empty -- the index is sparse, "
            f"so not every number in range maps to a house."
            + (f" Try one of: {', '.join(str(i) for i in usable[:8])} ..." if usable else ""),
            file=sys.stderr,
        )
        return 1

    # Some datasets expose per-scene variants ({"base": ..., "ceiling": ...}).
    # A variant key can be present but None -- the index lists the variant for the
    # dataset as a whole, not per scene, so not every house has every variant.
    if isinstance(entry, dict):
        chosen = entry.get(args.variant)
        if chosen is None:
            available = sorted(k for k, v in entry.items() if v is not None)
            if not available:
                print(
                    f"scene {args.dataset}/{args.split}#{args.index} has no usable "
                    f"variant (all of {sorted(entry)} are empty)",
                    file=sys.stderr,
                )
                return 1
            # Falling back beats failing: the caller asked for a scene, not a variant,
            # and the default variant is our choice rather than theirs.
            fallback = available[0]
            print(
                f"variant '{args.variant}' not available for this scene; "
                f"using '{fallback}' (have: {', '.join(available)})",
                file=sys.stderr,
            )
            chosen = entry[fallback]
        scene_path = Path(chosen)
    else:
        scene_path = Path(entry)

    if not args.no_install:
        print(f"installing {scene_path.name} ...", file=sys.stderr)
        install_scene_with_objects_and_grasps_from_path(scene_path)

    # Scene XMLs reference their meshes as "../../objects/...", which only resolves
    # inside the unversioned symlink tree -- never in the versioned cache the links
    # point into. get_scenes already hands back the symlink-tree path, so emit it
    # verbatim; do NOT .resolve() it, which would rewrite it to the cache path and
    # break every mesh reference.
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    scenes_root = ASSETS_DIR / "scenes"
    if not scene_path.is_relative_to(scenes_root):
        # A layout we do not know about. The directory under scenes/ is not simply
        # the dataset name (ithor is "ithor", the rest are "<dataset>-<split>"), so
        # guessing one risks pointing MuJoCo at a path that loads but is wrong.
        print(
            f"error: {scene_path} is outside the asset tree at {scenes_root}; "
            "mesh references would not resolve",
            file=sys.stderr,
        )
        return 1

    if not scene_path.exists():
        if not args.no_install:
            # The install above was supposed to produce this file. Fail loudly here
            # rather than let MuJoCo report it as a missing-mesh error three frames
            # deep, which is what makes this class of breakage hard to place.
            print(
                f"error: {scene_path} does not exist after installing; "
                "the scene did not download correctly",
                file=sys.stderr,
            )
            return 1
        # --no-install asked for resolution only, so a missing file is expected
        # rather than an error -- but say so, since the path is not yet loadable.
        print(f"note: {scene_path} is not installed yet", file=sys.stderr)

    print(scene_path, file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
