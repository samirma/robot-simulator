#!/usr/bin/env bash
# CoppeliaSim simulator launcher.
#
#   ./run.sh setup                     install venv + client + download CoppeliaSim EDU
#   ./run.sh view [--scene room]       open a scene with a shared robot spawned in it:
#                 [--robot so101]        so101 | myagv
#                 [--ros-port 9090]      ...on the myAGV's vendor ROS topics (myagv only)
#                 [--control 127.0.0.1:8000] ...or as a generic control server (so101)
#                 [--headless]           ...run CoppeliaSim with no GUI window (-h)
#                 [--gui]                ...force the GUI (default)
#   ./run.sh shell                     interactive shell inside the venv
#
# The robot models and the wire bridge come from simulator/shared, so robot_console
# connects to a CoppeliaSim-hosted robot exactly as it does to a MolmoSpaces or RoboCasa
# one. spawn_robot.py launches CoppeliaSim itself and talks to it over the ZMQ remote API.
#
# Any flags after the subcommand are forwarded to the underlying entry point.
set -euo pipefail

_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
SIM_ROOT="$(dirname "$_self")"
SIM_ROOT="$(realpath "$SIM_ROOT" 2>/dev/null || echo "${SIM_ROOT%/.}")"
# shellcheck source=/dev/null
source "$SIM_ROOT/env.sh"

PY="$VENV_DIR/bin/python"

die() { echo "error: $*" >&2; exit 1; }

find_python311() {
  for c in /opt/homebrew/opt/python@3.11/bin/python3.11 \
           /usr/local/opt/python@3.11/bin/python3.11 \
           "$(command -v python3.11 || true)" \
           "$(command -v python3 || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

# ---------------------------------------------------------------- setup

do_setup() {
  command -v uv >/dev/null || die "uv not found; install from https://docs.astral.sh/uv/"
  command -v curl >/dev/null || die "curl not found"

  if [ ! -x "$PY" ]; then
    py="$(find_python311)" || die "python 3.x not found"
    echo ">> creating venv on $py"
    uv venv --python "$py" "$VENV_DIR"
  fi

  # The client is pure Python; no MuJoCo here. cbor + pyzmq come in as deps of the client.
  echo ">> installing the ZMQ remote API client + bridge deps"
  VIRTUAL_ENV="$VENV_DIR" uv pip install \
    coppeliasim-zmqremoteapi-client numpy opencv-python msgpack-numpy websockets

  if [ ! -x "$COPSIM_BIN" ]; then
    mkdir -p "$APP_DIR"
    echo ">> downloading CoppeliaSim EDU $COPSIM_VER ($COPSIM_OS, ~260 MB)"
    echo "   By downloading you accept the EDU terms at coppeliarobotics.com"
    curl -L -o "$DATA_ROOT/$COPSIM_ZIP" "$COPSIM_URL"
    echo ">> unpacking"
    ( cd "$APP_DIR" && unzip -q -o "$DATA_ROOT/$COPSIM_ZIP" )
    # The zip may contain a top-level folder; normalise so COPSIM_APP is right.
    if [ ! -d "$COPSIM_APP" ]; then
      found="$(/usr/bin/find "$APP_DIR" -maxdepth 3 -name CoppeliaSim.app -type d | head -1)"
      [ -n "$found" ] || die "CoppeliaSim.app not found after unzip"
      [ "$found" = "$COPSIM_APP" ] || ln -sfn "$found" "$COPSIM_APP"
    fi
    rm -f "$DATA_ROOT/$COPSIM_ZIP"
  fi

  # Gatekeeper: the .app is ad-hoc signed. Without clearing quarantine, macOS blocks the
  # executable and its plugins (the ZMQ/URDF dylibs) with confusing "cannot be loaded".
  if [ "$(uname -s)" = "Darwin" ]; then
    echo ">> clearing Gatekeeper quarantine"
    xattr -dr com.apple.quarantine "$COPSIM_APP" 2>/dev/null || true
  fi

  echo ">> setup complete: $COPSIM_BIN"
}

ensure_setup() {
  [ -x "$PY" ] || die "not installed yet - run: ./run.sh setup"
  [ -x "$COPSIM_BIN" ] || die "CoppeliaSim not installed - run: ./run.sh setup"
}

# ---------------------------------------------------------------- view

do_view() {
  ensure_setup
  # spawn_robot.py launches CoppeliaSim and drives it over ZMQ; it reads COPSIM_BIN etc.
  # from the environment sourced above.
  exec "$PY" "$SIM_ROOT/tools/spawn_robot.py" "$@"
}

# ---------------------------------------------------------------- shell

do_shell() {
  ensure_setup
  echo ">> venv: $VENV_DIR   CoppeliaSim: $COPSIM_BIN"
  exec "${SHELL:-/bin/bash}" -i
}

# ---------------------------------------------------------------- dispatch

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  setup)  do_setup "$@" ;;
  view)   do_view "$@" ;;
  shell)  do_shell "$@" ;;
  help|-h|--help)
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    ;;
  *) die "unknown command '$cmd' (try: ./run.sh help)" ;;
esac
