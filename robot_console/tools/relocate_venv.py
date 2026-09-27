#!/usr/bin/env python3
"""Make a uv venv usable again after its checkout has been moved or copied.

    python3 relocate_venv.py VENV_DIR [--check]

uv writes the venv's absolute path into ``bin/activate`` and into the shebang of every
script it installs, and an editable install's finder names the source tree absolutely
too. Move the checkout and ``bin/python`` still starts -- it is a symlink to the base
interpreter -- while ``mjpython`` dies on a shebang naming a directory that is gone, and
anything that imports an editable package through its finder imports from nowhere.

Rewritten in place rather than rebuilt, because a rebuild re-resolves every dependency:
the two engines run MuJoCo 3.5.0 and 3.3.1 and disagree on spec-editing APIs, and a
fresh resolve is free to move either. This changes paths and nothing else.

The old location is read from ``bin/activate``; the prefix that moved is what is left
after dropping the path components the old and new venv paths end in common, so
``/a/old/sim/.venv`` -> ``/b/new/sim/.venv`` rewrites ``/a/old/`` to ``/b/new/``, which
also covers the editable finders pointing at ``/a/old/sim/upstream``.

``--check`` exits 1 if the venv has moved and 0 if not, changing nothing. Stdlib only:
it has to run when the venv it repairs cannot.

There is a copy of this file in ``simulator/shared/tools/`` -- the console must install
and run with no simulator checkout, so it cannot import that one.
"""

from __future__ import annotations  # macOS's /usr/bin/python3 is 3.9

import argparse
import os
import re
import sys
from pathlib import Path

ACTIVATE_RE = re.compile(r"^VIRTUAL_ENV='(.*)'$", re.M)


def recorded_path(venv: Path) -> Path | None:
    try:
        m = ACTIVATE_RE.search((venv / "bin" / "activate").read_text())
    except OSError:
        return None
    return Path(m.group(1)) if m else None


def has_moved(venv: Path) -> bool:
    was = recorded_path(venv)
    if was is None:
        return False
    try:
        return not os.path.samefile(was, venv)
    except OSError:  # the old location is gone
        return True


def moved_prefix(old: Path, new: Path) -> tuple[str, str]:
    a, b = list(old.parts), list(new.parts)
    while len(a) > 1 and len(b) > 1 and a[-1] == b[-1]:
        a.pop()
        b.pop()
    return str(Path(*a)), str(Path(*b))


def candidates(venv: Path):
    for p in (venv / "bin").iterdir():
        if p.is_file() and not p.is_symlink():
            yield p
    for site in (venv / "lib").glob("python*/site-packages"):
        for pattern in ("*.pth", "*.egg-link", "__editable__*.py", "*.dist-info/direct_url.json"):
            yield from site.glob(pattern)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("venv", type=Path)
    ap.add_argument("--check", action="store_true", help="exit 1 if moved; change nothing")
    args = ap.parse_args()

    venv = args.venv.absolute()
    if not has_moved(venv):
        return 0
    if args.check:
        return 1

    old, new = moved_prefix(recorded_path(venv), venv)
    # The trailing separator is what keeps /a/old from also matching /a/older.
    needle, repl = (old.rstrip("/") + "/").encode(), (new.rstrip("/") + "/").encode()
    tail_needle, tail_repl = f"'{old}'".encode(), f"'{new}'".encode()
    changed = 0
    for path in candidates(venv):
        data = path.read_bytes()
        if b"\0" in data[:8192] or needle not in data and tail_needle not in data:
            continue  # a binary, or nothing of ours in it
        path.write_bytes(data.replace(needle, repl).replace(tail_needle, tail_repl))
        changed += 1
        if path.suffix == ".py":
            # The finder's MAPPING is a constant, so its .pyc carries the old path too;
            # the rewrite changes the source's mtime, but equal-length paths leave its
            # size alone, and a stale cache is not worth reasoning about.
            for pyc in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
                pyc.unlink()
    print(f"relocated {venv}: {old} -> {new} in {changed} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
