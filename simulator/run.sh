#!/usr/bin/env bash
# MolmoSpaces simulator launcher.
#
#   ./run.sh setup                     install venv + package, fetch default assets
#   ./run.sh assets [ithor|objects|..] pre-fetch bulk asset sources
#   ./run.sh view [--scene ithor:1]    open a house in the MuJoCo viewer
#                 [--robot so101]      ...optionally with an out-of-tree robot in it:
#                                      so101 | myagv | rebot_b601 | ainex
#                 [--ros-port 9090]    ...and on that robot's own vendor ROS topics
#                                      (myagv and ainex only):
#                                      myagv -> cmd_vel in, odom + camera + /scan
#                                      out; ainex -> /walking/* and /app/* in, joint_states
#                                      + camera + /scan out (it has no cmd_vel at all)
#   ./run.sh serve  [--port 8000]      run the external control server, paired with
#                                      `view --robot <name> --control 127.0.0.1:8000`
#   ./run.sh shell                     interactive shell inside the venv
#
# Any flags after the subcommand are forwarded to the underlying entry point.
set -euo pipefail

# Resolved without cd; see the note in env.sh about title-escape capture.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
SIM_ROOT="$(dirname "$_self")"
SIM_ROOT="$(realpath "$SIM_ROOT" 2>/dev/null || echo "${SIM_ROOT%/.}")"
# shellcheck source=/dev/null
source "$SIM_ROOT/env.sh"

PY="$VENV_DIR/bin/python"
# The MuJoCo passive viewer must own the main thread on macOS, which is what
# mjpython provides. Everything else runs under plain python.
MJPY="$VENV_DIR/bin/mjpython"
[ "$(uname -s)" = "Darwin" ] || MJPY="$PY"

die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------- setup

find_python311() {
  for c in /opt/homebrew/opt/python@3.11/bin/python3.11 \
           /usr/local/opt/python@3.11/bin/python3.11 \
           "$(command -v python3.11 || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

do_setup() {
  command -v uv >/dev/null || die "uv not found; install from https://docs.astral.sh/uv/"

  if [ ! -d "$MOLMOSPACES_DIR" ]; then
    echo ">> cloning molmospaces"
    git clone https://github.com/allenai/molmospaces.git "$MOLMOSPACES_DIR"
  fi

  if [ ! -x "$PY" ]; then
    # mjpython needs a shared libpython, which uv's standalone CPython does not
    # ship, so the venv is built on a Homebrew/system framework Python 3.11.
    py311="$(find_python311)" || die "python 3.11 not found. Run: brew install python@3.11"
    echo ">> creating venv on $py311"
    uv venv --python "$py311" "$VENV_DIR"
  fi

  echo ">> installing molmospaces[mujoco]"
  # mujoco-filament is a linux-x86_64-only wheel; the plain mujoco extra is the
  # only option on macOS arm64.
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$MOLMOSPACES_DIR[mujoco]"

  echo ">> installing default assets (robots, scene indices)"
  "$PY" -m molmo_spaces.molmo_spaces_constants

  echo ">> setup complete"
}

ensure_setup() {
  [ -x "$PY" ] || die "not installed yet - run: ./run.sh setup"
}

# ---------------------------------------------------------------- assets

# Bulk sources worth pre-fetching for offline use. Deliberately excludes
# objaverse (~129k objects) and the procthor/holodeck scene sets, which are
# enormous and stream on demand anyway.
declare -a ITHOR_SOURCES=(
  "mujoco/scenes/ithor/20251217_with_occupancy"
  "mujoco/objects/thor/20251117"
  "mujoco/grasps/droid/20251116"
)

do_assets() {
  ensure_setup
  local what="${1:-ithor}"
  case "$what" in
    list)
      "$PY" "$MOLMOSPACES_DIR/scripts/assets/hf_download.py" "$DATA_ROOT" --list
      ;;
    ithor)
      for src in "${ITHOR_SOURCES[@]}"; do
        echo ">> fetching $src"
        "$PY" "$MOLMOSPACES_DIR/scripts/assets/hf_download.py" \
          "$DATA_ROOT" --data_source_dir "$src" --versioned --yes
      done
      echo ">> relinking into $MLSPACES_ASSETS_DIR"
      "$PY" -m molmo_spaces.molmo_spaces_constants
      # Scene files are fetched per-file on demand, so the archive pull above is
      # not enough to make every house usable offline -- walk them explicitly.
      echo ">> installing every iTHOR house (this is the slow part)"
      "$PY" "$SIM_ROOT/tools/prefetch_scenes.py" ithor
      ;;
    default)
      "$PY" -m molmo_spaces.molmo_spaces_constants
      ;;
    *)
      # treat as an explicit source dir
      "$PY" "$MOLMOSPACES_DIR/scripts/assets/hf_download.py" \
        "$DATA_ROOT" --data_source_dir "$what" --versioned --yes
      ;;
  esac
}

# ---------------------------------------------------------------- view

do_view() {
  ensure_setup
  local scene="ithor:1"
  local robot=""
  local -a rest=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --scene) scene="$2"; shift 2 ;;
      --robot) robot="$2"; shift 2 ;;
      *) rest+=("$1"); shift ;;
    esac
  done

  local dataset="${scene%%:*}"
  local index="${scene##*:}"
  [ "$dataset" = "$index" ] && index=0

  echo ">> resolving $dataset scene #$index (downloading if needed)"
  local xml
  xml="$("$PY" "$SIM_ROOT/tools/resolve_scene.py" "$dataset" "$index")" \
    || die "could not resolve scene $scene"
  echo ">> $xml"

  # Deliberately a script, not `-m mujoco.viewer`: see tools/view_scene.py.
  # The path must be absolute, which resolve_scene.py guarantees.
  if [ -n "$robot" ]; then
    # Spawns an out-of-tree robot (robots/<name>/) into the house. Nothing beyond the
    # scene and the robot is involved, so this works for robots that have no grasp
    # library yet.
    exec "$MJPY" "$SIM_ROOT/tools/spawn_robot.py" "$robot" --scene "$xml" \
      "${rest[@]+"${rest[@]}"}"
  fi
  exec "$MJPY" "$SIM_ROOT/tools/view_scene.py" "$xml" "${rest[@]+"${rest[@]}"}"
}

# ---------------------------------------------------------------- serve

do_serve() {
  ensure_setup
  exec "$PY" "$SIM_ROOT/bridge/server.py" "$@"
}

# ---------------------------------------------------------------- shell

do_shell() {
  ensure_setup
  echo ">> venv: $VENV_DIR   assets: $MLSPACES_ASSETS_DIR   MUJOCO_GL=$MUJOCO_GL"
  exec "${SHELL:-/bin/bash}" -i
}

# ---------------------------------------------------------------- dispatch

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  setup)  do_setup "$@" ;;
  assets) do_assets "$@" ;;
  view)   do_view "$@" ;;
  serve)  do_serve "$@" ;;
  shell)  do_shell "$@" ;;
  help|-h|--help)
    # Print the header comment block: everything after the shebang up to the
    # first non-comment line, with the leading "# " stripped.
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    ;;
  *) die "unknown command '$cmd' (try: ./run.sh help)" ;;
esac
