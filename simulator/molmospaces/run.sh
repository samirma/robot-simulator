#!/usr/bin/env bash
# MolmoSpaces engine: MuJoCo with iTHOR and ProcTHOR houses (the default engine).
#
#   ./run.sh setup                   venv, upstream checkout (pinned), default assets,
#                                    robot meshes (fetched and verified into robots_specs/),
#                                    worktop objects (into simulator/shared/objects/ycb/)
#   ./run.sh assets [<source>]       pre-fetch asset sources for offline use: every one, or
#                                    one of: ithor, procthor, objects, grasps, default
#   ./run.sh start [--scene <s>] [--sim-port <p>] [--mujoco]
#                                    start this engine's simulation with the scene alone,
#                                    headless unless --mujoco opens a MuJoCo window
#   ./run.sh repair                  re-point the venv and assets/ after the checkout moved
#                                    (every command does this itself)
#
# start:
#   --scene <source>:<id>   ithor:<n> (an iTHOR floor plan), procthor:<n> (a ProcTHOR-10k
#                           house) or test:1 (flat floor and one worktop); default ithor:1.
#                           Scenes install on demand.
#   --sim-port <p>          the simulation's local control port (default 9080); refused
#                           when taken. spawn.sh adds robots through it.
#   --mujoco                open a MuJoCo window in the simulation process; closing it
#                           ends the simulation and every spawned robot.
set -euo pipefail

_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
_dir="$(dirname "$_self")"
_dir="$(realpath "$_dir" 2>/dev/null || echo "${_dir%/.}")"
# shellcheck source=/dev/null
source "$_dir/env.sh"
# shellcheck source=/dev/null
source "$SHARED_ROOT/engine_common.sh"

MOLMOSPACES_URL="https://github.com/allenai/molmospaces.git"

do_setup() {
  [ $# -eq 0 ] || die "setup takes no arguments"
  need_uv
  fetch_robot_meshes
  fetch_worktop_objects
  checkout "$MOLMOSPACES_URL" "$MOLMOSPACES_REV" "$MOLMOSPACES_DIR"
  make_venv
  do_repair
  echo ">> installing molmospaces[mujoco] (editable, pinned checkout)"
  VIRTUAL_ENV="$VENV_DIR" uv pip install -q -e "${MOLMOSPACES_DIR}[mujoco]" pyyaml scipy pillow
  echo ">> installing default assets (scene indices, object and grasp metadata)"
  "$PY" -m molmo_spaces.molmo_spaces_constants >/dev/null
  do_repair
  echo ">> installing the default scene ithor:1"
  "$PY" "$ENGINE_ROOT/tools/resolve_scene.py" ithor 1 >/dev/null
  echo ">> setup complete"
}

do_repair() {
  [ -x "$PY" ] || return 0
  "$PY" "$SHARED_ROOT/tools/relocate_venv.py" "$VENV_DIR"
  "$PY" "$ENGINE_ROOT/tools/relink_assets.py"
}

declare -a ITHOR_SOURCES=(
  "mujoco/scenes/ithor/20251217_with_occupancy"
  "mujoco/objects/thor/20251117"
  "mujoco/grasps/droid/20251116"
)

do_assets() {
  ensure_setup
  [ $# -le 1 ] || die "assets takes at most one source"
  local what="${1:-all}"
  case "$what" in
    all|ithor|procthor|objects|grasps|default) ;;
    *) die "unknown asset source '$what' (one of: ithor, procthor, objects, grasps, default)" ;;
  esac
  if [ "$what" = default ] || [ "$what" = all ]; then
    "$PY" -m molmo_spaces.molmo_spaces_constants >/dev/null
  fi
  if [ "$what" = objects ] || [ "$what" = all ] || [ "$what" = ithor ]; then
    echo ">> fetching mujoco/objects/thor/20251117"
    "$PY" "$MOLMOSPACES_DIR/scripts/assets/hf_download.py" "$DATA_ROOT" \
      --data_source_dir "mujoco/objects/thor/20251117" --versioned --yes
  fi
  if [ "$what" = grasps ] || [ "$what" = all ]; then
    echo ">> fetching mujoco/grasps/droid/20251116"
    "$PY" "$MOLMOSPACES_DIR/scripts/assets/hf_download.py" "$DATA_ROOT" \
      --data_source_dir "mujoco/grasps/droid/20251116" --versioned --yes
  fi
  if [ "$what" = ithor ] || [ "$what" = all ]; then
    echo ">> fetching mujoco/scenes/ithor/20251217_with_occupancy"
    "$PY" "$MOLMOSPACES_DIR/scripts/assets/hf_download.py" "$DATA_ROOT" \
      --data_source_dir "mujoco/scenes/ithor/20251217_with_occupancy" --versioned --yes
    "$PY" -m molmo_spaces.molmo_spaces_constants >/dev/null
    echo ">> installing every iTHOR house"
    "$PY" "$ENGINE_ROOT/tools/prefetch_scenes.py" ithor
  fi
  if [ "$what" = procthor ] || [ "$what" = all ]; then
    echo ">> installing every ProcTHOR-10k training house (large)"
    "$PY" "$ENGINE_ROOT/tools/prefetch_scenes.py" procthor-10k
  fi
  do_repair
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true
case "$cmd" in
  setup)  do_setup "$@" ;;
  assets) do_assets "$@" ;;
  repair) [ $# -eq 0 ] || die "repair takes no arguments"; do_repair ;;
  start)  start_simulation "$@" ;;
  help|-h|--help) usage ;;
  *) die "unknown command '$cmd' (see ./run.sh --help)" ;;
esac
