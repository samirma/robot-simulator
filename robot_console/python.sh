#!/usr/bin/env bash
# The console's venv python, installing the venv first if needed. For example:
#
#   robot_console/python.sh -m robot_console.fleet --url ws://127.0.0.1:9090 --expect myagv
#   robot_console/python.sh -m pytest robot_console/tests
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_bootstrap.sh"
exec "$CONSOLE_PY" "$@"
