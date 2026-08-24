#!/usr/bin/env bash
# Mapping and navigation for a myAGV, simulated or real.
#
#   ./bin/slam.sh explore  --out runs/house     map a space autonomously
#   ./bin/slam.sh map      --out runs/house     drive by hand, map builds live
#   ./bin/slam.sh navigate --map runs/house     click a point, robot drives there
#
#   ./bin/slam.sh explore --host 192.168.1.42   ...against a real myAGV
#   ./bin/slam.sh <mode> --help                 every flag for that mode
#
# Keys (the map window must have focus):
#   W/S forward-back   A/D strafe   Q/E rotate   Space stop/pause
#   M save the map     Esc quit
#   navigate: left click sets a goal, right click cancels
#
# Needs a robot publishing /scan (the YDLidar X2 on hardware; the simulator's ray-cast
# stand-in with `./run.sh view --robot myagv --ros-port 9090`). Maps are written as a
# map_server pgm/yaml pair, so myagv_navigation can load them directly.
#
# The first run creates .venv and installs the package; after that this is just a
# launcher. Flags are forwarded to the matching `python -m robot_console.<mode>`.
set -euo pipefail

# Resolved without cd, so `--out runs/house` still means the caller's cwd. Using
# $(cd .. && pwd) here would also capture any terminal-title escapes the shell emits.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
BIN_DIR="$(dirname "$_self")"
CONSOLE_ROOT="$(dirname "$BIN_DIR")"
CONSOLE_ROOT="$(realpath "$CONSOLE_ROOT" 2>/dev/null || echo "${CONSOLE_ROOT%/.}")"

VENV_DIR="${ROBOT_CONSOLE_VENV:-$CONSOLE_ROOT/.venv}"
PY="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.teleop-stamp"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$_self"
}

bootstrap() {
  command -v uv >/dev/null || die "uv not found; install it with
    curl -LsSf https://astral.sh/uv/install.sh | sh
  or set the venv up by hand:
    python3 -m venv '$VENV_DIR' && '$VENV_DIR/bin/pip' install -e '$CONSOLE_ROOT'"

  if [ ! -x "$PY" ]; then
    echo ">> creating venv ($VENV_DIR)"
    uv venv --python 3.12 "$VENV_DIR" || uv venv "$VENV_DIR"
  fi
  echo ">> installing robot_console"
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$CONSOLE_ROOT"
  touch "$STAMP"
}

# Shared with teleop.sh: one venv, one stamp, so switching between the two launchers
# never triggers a reinstall.
if [ ! -x "$PY" ] || [ ! -f "$STAMP" ] || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$STAMP" ]; then
  bootstrap
fi

for arg in "$@"; do
  if [ "$arg" = "--reinstall" ]; then
    bootstrap
    exit 0
  fi
done

[ $# -ge 1 ] || { usage; exit 2; }

mode="$1"; shift
case "$mode" in
  explore)  module="robot_console.explore" ;;
  map)      module="robot_console.mapping" ;;
  navigate) module="robot_console.navigate" ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage; die "unknown mode '$mode' (expected explore, map or navigate)" ;;
esac

# exec, so Ctrl-C reaches Python directly -- which is what stops a real AGV, since it has
# no command watchdog of its own.
exec "$PY" -m "$module" "$@"
