#!/usr/bin/env python3
"""Run a vendor ROS node script unchanged, with some of its hardware modules replaced by
the simulator's (the simulated board, GPIO):

    run_with_fakes.py <module>=<fake module> [...] -- <script> [ros args...]

Each `<module>` is put into sys.modules as the named fake (imported from sim_libs/) before
the script runs as __main__, so `from <package> import <module>`-style imports resolve to
the simulated hardware while the rest of the vendor package (messages, helpers) is real.
"""

import importlib
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    argv = sys.argv[1:]
    sep = argv.index("--")
    fakes, rest = argv[:sep], argv[sep + 1:]
    for spec in fakes:
        name, _, fake = spec.partition("=")
        mod = importlib.import_module(fake)
        pkg, _, leaf = name.rpartition(".")
        if pkg:
            parent = importlib.import_module(pkg)
            setattr(parent, leaf, mod)
        sys.modules[name] = mod
    script = rest[0]
    sys.argv = rest
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
