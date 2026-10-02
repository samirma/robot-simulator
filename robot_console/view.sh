#!/usr/bin/env bash
# Camera and bounded-control page for the robot on a rosbridge websocket.
#
#   view.sh [--url ws://host:port] [--robot <id>] [--namespace <name>] [--no-open] [--port N]
#
# Serves one static page on 127.0.0.1 that speaks rosbridge to --url from the browser, and
# opens it. Run with --help for details. The first run installs the console's venv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_bootstrap.sh"
exec "$CONSOLE_PY" -m robot_console.view "$@"
