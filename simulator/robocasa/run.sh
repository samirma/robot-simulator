#!/usr/bin/env bash
# RoboCasa engine: MuJoCo with RoboCasa kitchens, as a scene provider only (no robosuite
# robot, controller or observation stack enters the model).
#
#   ./run.sh setup                   venv, upstream checkouts (pinned robosuite and robocasa
#                                    v1.0), default kitchen assets, robot meshes (fetched
#                                    and verified into robots_specs/), worktop objects
#                                    (into simulator/shared/objects/ycb/)
#   ./run.sh assets [<source>]       pre-fetch asset sources for offline use: every one, or
#                                    one of: tex, tex_generative, fixtures, objs_objaverse,
#                                    objs_aigen, objs_lightwheel
#   ./run.sh start [--scene <s>] [--sim-port <p>] [--mujoco]
#                                    start this engine's simulation with the scene alone,
#                                    headless unless --mujoco opens a MuJoCo window
#   ./run.sh repair                  re-point the venv after the checkout moved (every
#                                    command does this itself)
#
# start:
#   --scene <source>:<id>   robocasa:<layout>-<style> (layout 1-60, style 1-60) or test:1
#                           (flat floor and one worktop); default robocasa:1-1.
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

do_setup() {
  [ $# -eq 0 ] || die "setup takes no arguments"
  need_uv
  fetch_robot_meshes
  fetch_worktop_objects
  checkout https://github.com/ARISE-Initiative/robosuite.git "$ROBOSUITE_REV" "$ROBOSUITE_DIR"
  checkout https://github.com/robocasa/robocasa.git "$ROBOCASA_REV" "$ROBOCASA_DIR"
  make_venv
  do_repair
  echo ">> installing robosuite + robocasa (editable, pinned checkouts)"
  VIRTUAL_ENV="$VENV_DIR" uv pip install -q -e "$ROBOSUITE_DIR" -e "$ROBOCASA_DIR" \
    pyyaml scipy pillow huggingface_hub
  echo ">> installing the default kitchen assets into $ROBOCASA_ASSETS_DIR"
  "$PY" "$ENGINE_ROOT/tools/fetch_assets.py" default
  echo ">> setup complete"
}

do_repair() {
  [ -x "$PY" ] || return 0
  "$PY" "$SHARED_ROOT/tools/relocate_venv.py" "$VENV_DIR"
}

do_assets() {
  ensure_setup
  [ $# -le 1 ] || die "assets takes at most one source"
  local what="${1:-all}"
  "$PY" "$ENGINE_ROOT/tools/fetch_assets.py" "$what"
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
