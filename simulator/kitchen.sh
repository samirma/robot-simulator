#!/usr/bin/env bash
# Robots from robots_specs/ in a household scene, one engine at a time, on one wire.
#
#   ./kitchen.sh serve    load a scene, stage the task when a worktop robot is present,
#                         and serve the fleet on one rosbridge port
#
#   ./kitchen.sh serve --help
#
# `serve` owns the world, so it owns the window: MuJoCo builds a viewer from the model and
# data objects in memory, so a window can only exist inside the process holding the
# physics. That is why `--mujoco` is a `serve` flag. Grading an episode is the console's:
#
#   cd ../robot_console && ./run_task.sh [--episodes N]
#
#: serve
# usage: ./kitchen.sh serve [--engine molmospaces|robocasa] [--robots <id>[,<id>...]]
#                           [--ros-namespace <ns>] [--mujoco] [--port <p>]
#                           [--scene <s>]              # MolmoSpaces scene flag
#                           [--layout N --style N]     # RoboCasa scene flags
#                           [<staging flags>]
#
#   --engine molmospaces  molmospaces | robocasa. One per run; only that one need be set up.
#   --robots so101        comma-separated ids of the robots robots_specs/robots.yml marks
#                         `simulated`. They share one scene, one port and one ROS graph,
#                         each under its own id: /so101/*, /myagv/*, /ainex/*. A robot
#                         whose `placement` is `worktop` stands on the worktop, the others
#                         on the floor. With any worktop robot, apple_on_plate is staged
#                         on the worktop in front of it and the camera rig is served on
#                         /scene/*; without one there is no task and no rig. /reset is
#                         served whenever the SO-101 is.
#   --ros-namespace NS    put a lone robot under NS instead of its id; '' serves the bare
#                         vendor interface. Refused with more than one robot (the rig does
#                         not count), and for a name another provider owns (scene, rosapi,
#                         rosbridge_websocket, simulator).
#   --mujoco              open a MuJoCo window in the serving process -- the only one
#                         holding the physics. Closing the window ends the run. Headless
#                         otherwise.
#   --port 9090           rosbridge port
#   --scene ithor:1       MolmoSpaces scene: ithor:<n> or procthor:<n>
#   --layout 1 --style 1  RoboCasa kitchen, both 1-60
#
#   staging flags -- each changes only the staged world, never the wire, and none changes
#   the apple's or the plate's size or colour:
#   --reference-table     the reference rig's 0.92 m wooden slab under the task
#                         objects (default: off)
#   --no-dressing         the apple and the plate only, without the bowl, mug, banana and
#                         lemon (default: dressing on)
#   --reference-lighting  the reference rig's exposure (default: off)
#   --extra-lights        the reference's two lamps (default: off)
#   --swap-objects        the plate at the apple's spawn and the apple where the plate
#                         was (default: off)
#
# The serve warns when its real-time factor over a 10 s window falls below 0.90.
#
# With SIMULATOR_TRUTH_LOG=<path> in the environment (e.g. runs/truth.jsonl), a serve that
# stages the task also appends its true state -- each /reset, and the apple, plate and
# fingers at every rig frame -- to that local file, for robot_console's offline audit
# (python -m robot_console.arm.audit). Nothing of it is served on the wire.
#
# Examples:
#   ./kitchen.sh serve
#   ./kitchen.sh serve --robots so101,myagv --engine robocasa
#   ./kitchen.sh serve --robots myagv --ros-namespace '' --mujoco
#
set -euo pipefail

# Job control, so every backgrounded engine becomes its own process group and can be
# killed as one: the python holding the port is below the PID `$!` reports.
set -m

# Resolved without cd; see the note in each engine's env.sh about title-escape capture.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
ROOT="$(dirname "$_self")"
ROOT="$(realpath "$ROOT" 2>/dev/null || echo "${ROOT%/.}")"

MOLMO="$ROOT/molmospaces"
ROBOCASA="$ROOT/robocasa"

ENGINE="molmospaces"
ROBOTS="so101"
PORT="9090"
MUJOCO=0
NAMESPACE_SET=0
NAMESPACE=""
# Each engine's default scene (spec §2.3), and whether a scene flag was given at all:
# the other engine's is refused by name.
SCENE="ithor:1"
LAYOUT=1
STYLE=1
declare -a MOLMO_FLAGS_SEEN=() ROBOCASA_FLAGS_SEEN=() STAGE_FLAGS=()

die() { echo "error: $*" >&2; exit 1; }

# The header comment is the help, cut into sections by `#:` markers so one block serves
# `help` and `serve --help` and cannot drift from the flags it documents.
usage() {
  awk -v want="${1:-all}" '
    NR == 1        { next }
    !/^#/          { exit }
                   { sub(/^# ?/, "") }
    /^: /          { sect = substr($0, 3); next }
    sect == "" || want == "all" || want == sect { print }
  ' "$0"
}

# ---------------------------------------------------------------- arguments

cmd="serve"
case "${1:-}" in
  serve|help|-h|--help) cmd="$1"; shift || true ;;
esac
case "$cmd" in help|-h|--help) usage; exit 0 ;; esac

value() {  # flag, remaining argument count, next argument
  [ "$2" -ge 2 ] || die "$1 needs a value (see ./kitchen.sh serve --help)"
}
int_in() {  # flag, value, lo, hi
  case "$2" in ''|*[!0-9]*) die "$1: expected an integer, got '$2'" ;; esac
  [ "$2" -ge "$3" ] && [ "$2" -le "$4" ] || die "$1: expected $3-$4, got $2"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)  usage "$cmd"; exit 0 ;;
    --engine)   value "$1" $#; ENGINE="$2"; shift 2 ;;
    --robots)   value "$1" $#; ROBOTS="$2"; shift 2 ;;
    --port)     value "$1" $#; int_in --port "$2" 1 65535; PORT="$2"; shift 2 ;;
    --ros-namespace) value "$1" $#; NAMESPACE_SET=1; NAMESPACE="$2"; shift 2 ;;
    --mujoco)   MUJOCO=1; shift ;;
    --scene)    value "$1" $#; SCENE="$2"; MOLMO_FLAGS_SEEN+=(--scene); shift 2 ;;
    --layout)   value "$1" $#; int_in --layout "$2" 1 60; LAYOUT="$2"
                ROBOCASA_FLAGS_SEEN+=(--layout); shift 2 ;;
    --style)    value "$1" $#; int_in --style "$2" 1 60; STYLE="$2"
                ROBOCASA_FLAGS_SEEN+=(--style); shift 2 ;;
    --reference-table|--no-dressing|--reference-lighting|--extra-lights|--swap-objects)
                STAGE_FLAGS+=("$1"); shift ;;
    *) die "unknown flag '$1' (see ./kitchen.sh $cmd --help)" ;;
  esac
done

case "$ENGINE" in
  molmospaces)
    [ ${#ROBOCASA_FLAGS_SEEN[@]} -eq 0 ] \
      || die "${ROBOCASA_FLAGS_SEEN[0]} is a RoboCasa scene flag; the molmospaces engine takes --scene ithor:<n> or procthor:<n>"
    case "$SCENE" in
      ithor:*|procthor:*) int_in --scene "${SCENE#*:}" 0 99999 ;;
      *) die "--scene: expected ithor:<n> or procthor:<n>, got '$SCENE'" ;;
    esac ;;
  robocasa)
    [ ${#MOLMO_FLAGS_SEEN[@]} -eq 0 ] \
      || die "--scene is a MolmoSpaces scene flag; the robocasa engine takes --layout N --style N" ;;
  *) die "--engine: expected molmospaces or robocasa, got '$ENGINE'" ;;
esac

engine_root() { [ "$1" = molmospaces ] && echo "$MOLMO" || echo "$ROBOCASA"; }

# Only the engine actually being run has to be installed.
need_engine() {
  [ -x "$1/.venv/bin/python" ] \
    || die "$(basename "$1") is not set up yet - run: cd $1 && ./run.sh setup"
  # This script starts the engine's python directly rather than through its run.sh, so
  # it asks for the repair a moved checkout needs itself. A no-op when nothing moved.
  "$1/run.sh" repair || die "could not re-point $(basename "$1") at this checkout"
}

# ---------------------------------------------------------------- engines
#
# Each engine runs in its own subshell: `env.sh` exports VENV_DIR, PYTHONPATH and
# MUJOCO_GL, and the two engines disagree on all three.

molmospaces() {
  (
    # shellcheck source=/dev/null
    source "$MOLMO/env.sh"
    local xml dataset="${SCENE%%:*}"
    [ "$dataset" = procthor ] && dataset=procthor-10k
    # Scene MJCFs reference their meshes through the assets/ symlink tree, so the scene
    # is named by its assets/ path -- and resolved (downloaded on a first run) here.
    xml="$("$MOLMO/.venv/bin/python" "$MOLMO/tools/resolve_scene.py" \
            "$dataset" "${SCENE##*:}")" || die "could not resolve scene $SCENE"
    # exec, so the PID this subshell reports to `$!` IS the engine.
    exec "$1" "$MOLMO/tools/spawn_robot.py" "$ROBOTS" --scene "$xml" "${@:2}"
  )
}

robocasa() {
  (
    # shellcheck source=/dev/null
    source "$ROBOCASA/env.sh"
    exec "$1" "$ROBOCASA/tools/spawn_robot.py" "$ROBOTS" --layout "$LAYOUT" --style "$STYLE" "${@:2}"
  )
}

# The MuJoCo passive viewer must own the main thread on macOS, which is what mjpython
# provides; anything windowless runs under plain python.
engine_python() {
  local root; root="$(engine_root "$ENGINE")"
  if [ "$1" = viewer ] && [ "$(uname -s)" = "Darwin" ]; then
    echo "$root/.venv/bin/mjpython"
  else
    echo "$root/.venv/bin/python"
  fi
}

# Naming the holder of a taken port turns a puzzling failure into an obvious one.
port_free() {
  nc -z 127.0.0.1 "$1" 2>/dev/null || return 0
  local holder
  holder="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -Fc 2>/dev/null | sed -n 's/^c//p' | sort -u | paste -sd, -)"
  die "port $1 is already in use${holder:+ (by: $holder)} - ${2:-pick another port}"
}

# Kill a backgrounded engine and everything it forked, then wait for the port to come free.
stop_engine() {
  local pid="$1" port="$2" i
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  [ -n "$port" ] || return 0
  for i in $(seq 1 20); do
    nc -z 127.0.0.1 "$port" 2>/dev/null || return 0
    sleep 0.5
  done
  echo "warning: port $port still held after stopping the engine" >&2
}

cleanup() {
  if [ -n "${sim_pid:-}" ]; then stop_engine "$sim_pid" "$PORT"; fi
}

# ---------------------------------------------------------------- run

need_engine "$(engine_root "$ENGINE")"
# The ids, the namespace and its collisions, by the one module both engines use -- up
# front, rather than after an engine has spent a minute compiling a kitchen.
declare -a NS_FLAGS=()
[ "$NAMESPACE_SET" -eq 0 ] || NS_FLAGS=(--ros-namespace "$NAMESPACE")
"$(engine_python headless)" "$ROOT/shared/serve_args.py" check --robots "$ROBOTS" \
  ${NS_FLAGS[@]+"${NS_FLAGS[@]}"} || die "see ./kitchen.sh serve --help"
port_free "$PORT" "pick another with --port PORT"
trap cleanup INT TERM EXIT
declare -a HEADLESS=(--headless)
window=""
if [ "$MUJOCO" -eq 1 ]; then
  HEADLESS=()
  window=" in a window"
fi
echo ">> $ENGINE $ROBOTS$window on ws://127.0.0.1:$PORT"
"$ENGINE" "$(engine_python "$([ "$MUJOCO" -eq 1 ] && echo viewer || echo headless)")" \
  ${HEADLESS[@]+"${HEADLESS[@]}"} --ros-port "$PORT" \
  ${NS_FLAGS[@]+"${NS_FLAGS[@]}"} ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
sim_pid=$!
echo
echo "run the task against it from robot_console/:"
echo "  ./run_task.sh --label $ENGINE$([ "$PORT" = 9090 ] || echo " --url ws://127.0.0.1:$PORT") --episodes 6"
wait "$sim_pid"
