#!/usr/bin/env bash
# RoboCasa simulator launcher (engine #2: MuJoCo + robosuite + RoboCasa kitchens).
#
#   ./run.sh setup                       clone upstream + venv + editable install
#   ./run.sh assets                      download the kitchen assets (~10 GB)
#   ./run.sh view [--layout 1] [--style 3] [--robot PandaOmron]
#                                        open a kitchen in the MuJoCo viewer;
#                                        layout 1-60, style 1-60
#                 [--robot myagv|so101]  ...with a shared robot in it instead, on the
#                                        real hardware's interface:
#                 [--ros-port 9090]      myagv -> cmd_vel in, odom + camera + /scan out
#                 [--control HOST:PORT]  so101 -> generic observation/action server
#                 [--headless]           ...with no window (displayless hosts, checks)
#                 [--render out.png]     ...or just write a PNG and exit
#   ./run.sh --layout 1 --style 3        shorthand for `view --layout 1 --style 3`
#   ./run.sh shell                       interactive shell inside the venv
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

# robocasa v1.0 targets robosuite's master branch (its Kitchen env passes
# lite_physics/load_model_on_init, which no v1.5.x tag accepts).
ROBOSUITE_REF=master
ROBOCASA_REF=v1.0

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

  if [ ! -d "$ROBOSUITE_DIR" ]; then
    echo ">> cloning robosuite $ROBOSUITE_REF"
    git clone --depth 1 --branch "$ROBOSUITE_REF" \
      https://github.com/ARISE-Initiative/robosuite.git "$ROBOSUITE_DIR"
  fi
  if [ ! -d "$ROBOCASA_DIR" ]; then
    echo ">> cloning robocasa $ROBOCASA_REF"
    git clone --depth 1 --branch "$ROBOCASA_REF" \
      https://github.com/robocasa/robocasa.git "$ROBOCASA_DIR"
  fi

  if [ ! -x "$PY" ]; then
    # mjpython needs a shared libpython, which uv's standalone CPython does not
    # ship, so the venv is built on a Homebrew/system framework Python 3.11.
    py311="$(find_python311)" || die "python 3.11 not found. Run: brew install python@3.11"
    echo ">> creating venv on $py311"
    uv venv --python "$py311" "$VENV_DIR"
  fi

  echo ">> installing robosuite + robocasa (editable)"
  # Editable, mirroring the molmospaces engine: out-of-tree robots and engines
  # build on these without forking the upstream clones. Never modify upstream/.
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$ROBOSUITE_DIR" -e "$ROBOCASA_DIR"

  echo ">> setup complete; run './run.sh assets' to fetch the kitchen assets"
}

ensure_setup() {
  [ -x "$PY" ] || die "not installed yet - run: ./run.sh setup"
  [ -d "$ROBOCASA_DIR" ] || die "upstream clones missing - run: ./run.sh setup"
}

# ---------------------------------------------------------------- assets

do_assets() {
  ensure_setup
  echo ">> downloading kitchen assets (~10 GB) into $ROBOCASA_DIR/robocasa/models/assets"
  # The v1.0 script 404s on the lightwheel zips (nvidia renamed the repo) and
  # skips the base fixtures.zip; tools/download_lightwheel_assets.py covers both.
  "$PY" -m robocasa.scripts.download_kitchen_assets "$@" || true
  "$PY" "$SIM_ROOT/tools/download_lightwheel_assets.py"
}

# ---------------------------------------------------------------- view

do_view() {
  ensure_setup
  local robot=""
  local headless=0
  local -a rest=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --robot) robot="$2"; shift 2 ;;
      # Both take no window, so they must not be routed through mjpython: it exists
      # for the main-thread constraint of the passive viewer and nothing else.
      --headless|--render) headless=1; rest+=("$1"); shift ;;
      *) rest+=("$1"); shift ;;
    esac
  done

  # `--robot` is overloaded, and deliberately so: the two robot vocabularies here are
  # disjoint, and making the user learn two flag names to say "put this robot in the
  # kitchen" would be worse. A shared robot goes to spawn_robot.py and gets the vendor
  # wire contracts; anything else is a robosuite robot name for the plain viewer, which
  # is where `--robot PandaOmron` has always gone.
  case "$robot" in
    myagv|so101)
      local py="$MJPY"
      [ "$headless" = 1 ] && py="$PY"
      exec "$py" "$SIM_ROOT/tools/spawn_robot.py" "$robot" "${rest[@]+"${rest[@]}"}"
      ;;
    "") ;;
    *) rest+=(--robot "$robot") ;;
  esac

  # Deliberately a script, not `-m mujoco.viewer`, and under mjpython on macOS:
  # the passive viewer must own the main thread. See tools/view_kitchen.py.
  exec "$MJPY" "$SIM_ROOT/tools/view_kitchen.py" "${rest[@]+"${rest[@]}"}"
}

# ---------------------------------------------------------------- shell

do_shell() {
  ensure_setup
  echo ">> venv: $VENV_DIR   upstream: $SIM_ROOT/upstream   MUJOCO_GL=$MUJOCO_GL"
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
  help|-h|--help)
    # Print the header comment block: everything after the shebang up to the
    # first non-comment line, with the leading "# " stripped.
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    ;;
  -*)
    # `./run.sh --layout 1 --style 3` == `./run.sh view --layout 1 --style 3`
    do_view "$cmd" "$@"
    ;;
  *) die "unknown command '$cmd' (try: ./run.sh help)" ;;
esac
