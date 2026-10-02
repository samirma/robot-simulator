#!/usr/bin/env bash
# Keyboard teleoperation of a supported mobile robot (myagv, ainex, rosmaster_x3_plus)
# over a ROS 1 rosbridge websocket, with its cameras shown live.
#
#   teleop.sh [--robot <id>] [--namespace <name>] [--url ws://host:port]
#
# Opens a local window that owns keyboard focus. Keys: W/S forward/back, A/D strafe,
# Q/E rotate, Space stop, Enter enable commands, Esc quit; AiNex: arrow keys move the head.
# Run with --help for details. The first run installs the console's venv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_bootstrap.sh"
exec "$CONSOLE_PY" -m robot_console.teleop "$@"
