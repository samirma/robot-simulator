#!/usr/bin/env bash
# RoboCasa simulator launcher (engine #2: MuJoCo + robosuite + RoboCasa kitchens).
#
#   ./run.sh setup                       clone upstream + venv + editable install
#   ./run.sh assets [<source>]           download every kitchen asset source (~10 GB), or
#                                        one: tex, tex_generative, objs_objaverse,
#                                        objs_aigen or lightwheel
#   ./run.sh repair                      re-point the venv at this checkout after it has
#                                        been moved; every command does this anyway
set -euo pipefail

# Resolved without cd; see the note in env.sh about title-escape capture.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
SIM_ROOT="$(dirname "$_self")"
SIM_ROOT="$(realpath "$SIM_ROOT" 2>/dev/null || echo "${SIM_ROOT%/.}")"
# shellcheck source=/dev/null
source "$SIM_ROOT/env.sh"

PY="$VENV_DIR/bin/python"

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

  echo ">> fetching the robot meshes into robots_specs/"
  "$SIM_ROOT/../../fetch_robot_assets.sh"

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
  do_repair

  echo ">> installing robosuite + robocasa (editable)"
  # Editable, mirroring the molmospaces engine: out-of-tree robots and engines
  # build on these without forking the upstream clones. Never modify upstream/.
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$ROBOSUITE_DIR" -e "$ROBOCASA_DIR"
  # The rosbridge server this engine hosts is `websockets`-based, and neither robosuite
  # nor robocasa depends on it. MolmoSpaces gets it transitively, which is exactly why
  # its absence here only shows up as a ModuleNotFoundError at serve time, after a
  # minute of kitchen compile.
  VIRTUAL_ENV="$VENV_DIR" uv pip install websockets

  echo ">> converting the robot meshes MuJoCo cannot read (shared/robot_models.py)"
  "$PY" "$SHARED_ROOT/robot_models.py" --assets

  echo ">> setup complete; run './run.sh assets' to fetch the kitchen assets"
}

# A moved or copied checkout: uv writes absolute paths into every script's shebang and
# into the editable finders for robosuite and robocasa, so `python` still starts while
# `mjpython` does not and both packages import from the old tree. A no-op when nothing
# has moved, so every command runs it rather than asking anyone to remember it.
do_repair() {
  [ -x "$PY" ] || return 0
  "$PY" "$SHARED_ROOT/tools/relocate_venv.py" "$VENV_DIR"
}

ensure_setup() {
  [ -x "$PY" ] || die "not installed yet - run: ./run.sh setup"
  [ -d "$ROBOCASA_DIR" ] || die "upstream clones missing - run: ./run.sh setup"
  do_repair
}

# ---------------------------------------------------------------- assets

# The asset sources this engine knows (spec §2.1: `assets` fetches every one, or only the
# one named): robocasa's own registry types, and `lightwheel`. The v1.0 registry's
# lightwheel zips 404 (nvidia renamed the repo) and it skips the base fixtures.zip, so
# `lightwheel` is tools/download_lightwheel_assets.py, which covers both, in place of the
# registry's `fixtures_lw` and `objs_lw`.
REGISTRY_SOURCES=(tex tex_generative objs_objaverse objs_aigen)

do_assets() {
  ensure_setup
  local what="${1:-}" known
  known="${REGISTRY_SOURCES[*]} lightwheel"
  [ $# -le 1 ] || die "assets takes at most one source (one of: $known)"
  if [ -n "$what" ]; then
    case " $known " in
      *" $what "*) ;;
      *) die "unknown asset source '$what' (one of: $known)" ;;
    esac
  fi
  echo ">> downloading ${what:-every} kitchen asset source into $ROBOCASA_DIR/robocasa/models/assets"
  if [ -z "$what" ]; then
    "$PY" -m robocasa.scripts.download_kitchen_assets --type "${REGISTRY_SOURCES[@]}"
  elif [ "$what" != lightwheel ]; then
    "$PY" -m robocasa.scripts.download_kitchen_assets --type "$what"
  fi
  if [ -z "$what" ] || [ "$what" = lightwheel ]; then
    "$PY" "$SIM_ROOT/tools/download_lightwheel_assets.py"
  fi
}

# ---------------------------------------------------------------- dispatch

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  setup)  do_setup "$@" ;;
  assets) do_assets "$@" ;;
  repair) do_repair ;;
  help|-h|--help)
    # Print the header comment block: everything after the shebang up to the
    # first non-comment line, with the leading "# " stripped.
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    ;;
  *) die "unknown command '$cmd' (try: ./run.sh help)" ;;
esac
