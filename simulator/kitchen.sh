#!/usr/bin/env bash
# Start one engine's simulation: a household scene with its six worktop objects (apple,
# plate, bowl, mug, banana, lemon) on the worktop, no robot, headless or in a MuJoCo
# window. Robots are added later, by id, with ./spawn.sh.
#
#   ./kitchen.sh start [--engine molmospaces|robocasa] [--scene <source>:<id>] [--mujoco]
#                      [--sim-port <p>]
#
#   --engine molmospaces   molmospaces (default) or robocasa; one engine per run, and only
#                          that engine needs `run.sh setup`.
#   --scene <source>:<id>  the scene, the same form on every engine:
#                            molmospaces: ithor:<n>, procthor:<n>, test:1   (default ithor:1)
#                            robocasa:    robocasa:<layout>-<style> (1-60 each), test:1
#                                                                           (default robocasa:1-1)
#                          test:1 is a flat floor with one worktop, identical on both.
#   --mujoco               open a MuJoCo window in the simulation process; closing it ends
#                          the simulation and every spawned robot. Headless otherwise.
#   --sim-port 9080        the simulation's local control port; refused when taken.
#
# Then, in another terminal:   ./spawn.sh <id> [--placement worktop|floor]
# Stop with Ctrl-C (or close the window): every spawned robot ends with it.
set -euo pipefail

_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
ROOT="$(dirname "$_self")"
ROOT="$(realpath "$ROOT" 2>/dev/null || echo "${ROOT%/.}")"

die() { echo "error: $*" >&2; exit 1; }
usage() { awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"; }

cmd="${1:-}"
case "$cmd" in
  start) shift ;;
  -h|--help|help|"") usage; [ -n "$cmd" ] && exit 0; exit 2 ;;
  *) die "unknown command '$cmd' (see ./kitchen.sh --help)" ;;
esac

ENGINE="molmospaces"
declare -a PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --engine) [ $# -ge 2 ] || die "--engine needs a value"; ENGINE="$2"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;
    --scene) [ $# -ge 2 ] || die "--scene needs a value <source>:<id>"; PASS+=(--scene "$2"); shift 2 ;;
    --scene=*) PASS+=(--scene "${1#*=}"); shift ;;
    --sim-port) [ $# -ge 2 ] || die "--sim-port needs a value"; PASS+=(--sim-port "$2"); shift 2 ;;
    --sim-port=*) PASS+=(--sim-port "${1#*=}"); shift ;;
    --mujoco) PASS+=(--mujoco); shift ;;
    *) die "start: unknown flag '$1' (see ./kitchen.sh --help)" ;;
  esac
done
case "$ENGINE" in
  molmospaces|robocasa) ;;
  *) die "--engine: expected molmospaces or robocasa, got '$ENGINE'" ;;
esac
[ -x "$ROOT/$ENGINE/.venv/bin/python" ] \
  || die "$ENGINE is not set up yet; run: $ROOT/$ENGINE/run.sh setup"
exec "$ROOT/$ENGINE/run.sh" start ${PASS[@]+"${PASS[@]}"}
