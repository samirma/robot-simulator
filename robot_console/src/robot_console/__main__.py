"""`python -m robot_console` -- the same entry point `bin/teleop.sh` execs."""

import sys

from robot_console.cli import main

if __name__ == "__main__":
    sys.exit(main())
