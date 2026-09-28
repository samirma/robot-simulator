#!/usr/bin/env bash
# MolmoSpaces simulator launcher.
#
#   ./run.sh setup                     install venv + package, fetch default assets
#   ./run.sh assets [ithor|objects|..] pre-fetch bulk asset sources
#   ./run.sh view --robot <id> [--scene ithor:1]
#                                      one robot in a house, in the MuJoCo viewer; serves
#                                      no wire (kitchen.sh serve is what serves). <id> is
#                                      a robot robots_specs/robots.yml marks simulated; a
#                                      worktop robot gets the task staged in front of it.
#                                      --scene takes ithor:<n> or procthor:<n>
#                 [--render out.png]   ...or write one frame to a PNG, headless, and exit
#                 [--timeout N]        ...closing the window after N seconds
#   ./run.sh shell                     interactive shell inside the venv
#   ./run.sh repair                    re-point the venv and assets/ at this checkout after
#                                      it has been moved; every command does this anyway
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

  echo ">> fetching the robot meshes into robots_specs/"
  "$SIM_ROOT/../../fetch_robot_assets.sh"

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
  do_repair

  echo ">> installing molmospaces[mujoco]"
  # mujoco-filament is a linux-x86_64-only wheel; the plain mujoco extra is the
  # only option on macOS arm64.
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$MOLMOSPACES_DIR[mujoco]"

  echo ">> installing default assets (robots, scene indices)"
  "$PY" -m molmo_spaces.molmo_spaces_constants
  do_repair

  echo ">> converting the robot meshes MuJoCo cannot read (shared/robot_models.py)"
  "$PY" "$SHARED_ROOT/robot_models.py" --assets

  echo ">> setup complete"
}

# A checkout that has been moved or copied keeps working only after two things are
# re-pointed at it, and neither fails where it breaks. The venv: uv writes absolute paths
# into every script's shebang and the editable finder, so `python` starts while `mjpython`
# does not. The asset tree: assets/ is absolute symlinks into data/, and the installer
# trusts its own completion markers over the links, so a scene then "fails to download"
# into a directory that holds it. Both are no-ops when nothing has moved, so every
# command runs this rather than asking anyone to remember it.
do_repair() {
  [ -x "$PY" ] || return 0
  "$PY" "$SHARED_ROOT/tools/relocate_venv.py" "$VENV_DIR"
  "$PY" "$SIM_ROOT/tools/relink_assets.py"
}

ensure_setup() {
  [ -x "$PY" ] || die "not installed yet - run: ./run.sh setup"
  do_repair
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
  local scene="ithor:1" robot="" py="$MJPY"
  local -a rest=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --robot) [ $# -ge 2 ] || die "--robot needs an id"; robot="$2"; shift 2 ;;
      --scene) [ $# -ge 2 ] || die "--scene needs a value"; scene="$2"; shift 2 ;;
      # Windowless, so not through mjpython: it exists for the passive viewer's
      # main-thread constraint and nothing else.
      --render) [ $# -ge 2 ] || die "--render needs a path"; py="$PY"
                rest+=(--render "$2"); shift 2 ;;
      --timeout) [ $# -ge 2 ] || die "--timeout needs seconds"; rest+=(--timeout "$2"); shift 2 ;;
      --layout|--style) die "$1 is a RoboCasa scene flag; this engine takes --scene ithor:<n> or procthor:<n>" ;;
      *) die "unknown view flag '$1' (try: ./run.sh help)" ;;
    esac
  done
  [ -n "$robot" ] || die "view needs --robot <id> (try: ./run.sh help)"
  case "$robot" in *,*) die "view shows a single robot; --robot takes one id" ;; esac
  "$PY" "$SHARED_ROOT/robots_spec.py" check "$robot" || die "--robot: see ./run.sh help"

  local dataset="${scene%%:*}" index="${scene##*:}"
  case "$scene" in
    ithor:*|procthor:*) ;;
    *) die "--scene: expected ithor:<n> or procthor:<n>, got '$scene'" ;;
  esac
  [ "$dataset" = procthor ] && dataset=procthor-10k
  echo ">> resolving $scene (downloading if needed)"
  local xml
  xml="$("$PY" "$SIM_ROOT/tools/resolve_scene.py" "$dataset" "$index")" \
    || die "could not resolve scene $scene"
  # --ros-port 0: a view serves no wire.
  exec "$py" "$SIM_ROOT/tools/spawn_robot.py" "$robot" --scene "$xml" --ros-port 0 \
    "${rest[@]+"${rest[@]}"}"
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
  shell)  do_shell "$@" ;;
  repair) do_repair ;;
  help|-h|--help)
    # Print the header comment block: everything after the shebang up to the
    # first non-comment line, with the leading "# " stripped.
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    ;;
  *) die "unknown command '$cmd' (try: ./run.sh help)" ;;
esac
