#!/usr/bin/env bash
# Robots from robots_specs/ on a kitchen work surface, one engine at a time.
#
#   ./kitchen.sh serve    load a world and serve it on rosbridge, optionally in a window
#
#   ./kitchen.sh serve --help
#
# `serve` owns the world, so it owns the window: MuJoCo builds a viewer from the model and
# data objects in memory -- there is launch, launch_from_path and launch_passive, and no
# connect -- so a window can only exist inside the process holding the physics. That is
# why `--mujoco` is a `serve` flag. Grading an episode is the console's:
#
#   cd ../robot_console && ./run_task.sh [--episodes N]
#
#: serve
# usage: ./kitchen.sh serve [flags]
#
# Loads an engine, stages shared/tasks/apple_on_plate.py into the kitchen it compiled, and
# serves the robots on rosbridge. Headless unless --mujoco asks for a window.
#
#   --robots so101        which robots share the kitchen and the port: comma-separated ids
#                         of the robots robots_specs/robots.yml marks `simulated`. Each
#                         gets its own namespace on one rosbridge -- /so101/*, /myagv/* --
#                         which is one ROS graph with a namespace per robot, as a real
#                         bringup is. A robot whose robots.yml `placement` is `worktop`
#                         stands on the worktop; the others take the floor of the same
#                         room. apple_on_plate is staged when the SO-101 is in the list,
#                         and with it the worktop's camera rig on /scene/* -- the rig is
#                         the scene's, published by the fleet, not by any robot. A
#                         kitchen holding only a base gets the room, the robot and its
#                         camera. Every robot presents all of its cameras. Each renders
#                         inside the physics loop, so each one costs control rate for
#                         everyone on the port.
#   --mujoco              open a MuJoCo window on the world being served. The window is
#                         built from the model and data objects in memory, so it belongs
#                         to the process holding the physics, which is this one: a serve
#                         already running cannot grow a window; restart it with --mujoco.
#                         Rendering it costs control rate for every client on the port,
#                         and closing the window ends the run.
#   --engine molmospaces  molmospaces | robocasa. One per run; only that one need be set up.
#   --scene ithor:1       MolmoSpaces scene
#   --layout 1 --style 1  RoboCasa kitchen, both 1-60
#   --port 9090           rosbridge port
#
#   what the task stages, all off unless noted:
#   --reference-table     the reference rig's 0.92 m wooden slab under the objects. It
#                         sits on the kitchen's own counter and does not move the VLA's
#                         pass count: 2/6 bare against 1/6 with it.
#   --no-dressing         apple and plate only, without the bowl/mug/banana/lemon
#   --reference-lighting  the reference rig's exposure. Against the photometry: it matches
#                         that rig's clipped-pixel fraction almost exactly and costs
#                         MolmoAct2 the task outright, 0/24 episodes against 6/18.
#   --extra-lights        the reference's two lamps; they blow out a lit kitchen
#   --swap-objects        plate at the apple's spawn and vice versa. ON for robocasa, off
#                         for molmospaces; --no-swap-objects reverts. The console reads
#                         the layout off the wire, so it needs no matching flag.
#   --task-objects        the task's own measured YCB pair, not each engine's native one
#   --side-camera-mirror  the side camera on the far side of the worktop; robocasa only
#
# Examples:
#   ./kitchen.sh serve
#   ./kitchen.sh serve --robots so101,myagv --engine robocasa
#   ./kitchen.sh serve --engine robocasa --robots myagv --mujoco
#
set -euo pipefail

# Job control, so every backgrounded engine becomes its own process group and can be
# killed as one. It has to be: `molmospaces ... &` backgrounds a *function* whose body is
# a subshell, so the python that actually holds the port is two forks below the PID `$!`
# reports. Killing that PID alone reaps the shells and orphans the engine, which then
# sits on its port until the next run fails the port check and blames the port.
set -m

# Resolved without cd; see the note in each engine's env.sh about title-escape capture.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
ROOT="$(dirname "$_self")"
ROOT="$(realpath "$ROOT" 2>/dev/null || echo "${ROOT%/.}")"

MOLMO="$ROOT/molmospaces"
ROBOCASA="$ROOT/robocasa"

# The object categories that rank MolmoSpaces work surfaces when the arm is placed: the
# task's own pair. Not a flag -- the task decides what is on the worktop.
TARGET="plate,apple"
SCENE="ithor:1"
LAYOUT=1
STYLE=1
# 9090 is the rosbridge default and what a real bringup for this arm presents.
PORT="9090"
ROBOTS="so101"
ENGINE="molmospaces"
# Whether `serve` also opens a window on the world it is serving.
MUJOCO=0
declare -a STAGE_FLAGS=()
REFERENCE_TABLE=0
# Whether the plate and the apple trade places. Resolved per engine below: on for
# robocasa, off for molmospaces, so the two engines show a policy the same objects in
# different arrangements. The console reads the layout off the wire.
SWAP="auto"

die() { echo "error: $*" >&2; exit 1; }

# The header comment is the help, cut into sections by `#:` markers so one block serves
# `help` and `serve --help` and cannot drift from the flags it documents.
# Everything before the first marker is shared and prints every time.
usage() {
  awk -v want="${1:-all}" '
    NR == 1        { next }              # the shebang
    !/^#/          { exit }              # the comment block ends, and so does the help
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

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)  usage "$cmd"; exit 0 ;;
    --scene)    SCENE="$2";   shift 2 ;;
    --layout)   LAYOUT="$2";  shift 2 ;;
    --style)    STYLE="$2";   shift 2 ;;
    --port)     PORT="$2";    shift 2 ;;
    --engine)   ENGINE="$2";  shift 2 ;;
    --robots)   ROBOTS="$2";  shift 2 ;;
    --mujoco)   MUJOCO=1; shift ;;
    --reference-table)    REFERENCE_TABLE=1; shift ;;
    --no-reference-table) REFERENCE_TABLE=0; shift ;;
    --no-dressing)        STAGE_FLAGS+=(--no-dressing);        shift ;;
    --reference-lighting) STAGE_FLAGS+=(--reference-lighting); shift ;;
    --extra-lights)       STAGE_FLAGS+=(--extra-lights);       shift ;;
    --swap-objects)       SWAP=1; shift ;;
    --no-swap-objects)    SWAP=0; shift ;;
    --task-objects)       STAGE_FLAGS+=(--task-objects);       shift ;;
    --side-camera-mirror) STAGE_FLAGS+=(--side-camera-mirror); shift ;;
    *) die "unknown flag '$1' (try: ./kitchen.sh $cmd --help)" ;;
  esac
done

case "$ENGINE" in
  molmospaces|robocasa) ;;
  *) die "--engine: expected molmospaces or robocasa" ;;
esac

# Every member presents every camera it has: the worktop rig and each robot's own.
CAMERA_FLAGS=(--wrist-camera)

[ "$REFERENCE_TABLE" -eq 1 ] || STAGE_FLAGS+=(--no-reference-table)

# apple_on_plate is the SO-101's task: it stages its objects in the arm's base frame and
# its arbiter grades a jaw closing on an apple. A myAGV takes the floor and has no work
# surface and no gripper, so a kitchen holding only a base gets the room, the robot and
# its camera and no task -- which is also what stops the staging from binding to
# whichever robot happened to be first in the list.
declare -a TASK_FLAGS=()
case ",$ROBOTS," in
  *,so101,*) TASK_FLAGS=(--task apple_on_plate) ;;
  *) ;;
esac

if [ "$SWAP" = auto ]; then
  [ "$ENGINE" = robocasa ] && SWAP=1 || SWAP=0
fi
[ "$SWAP" -eq 0 ] || STAGE_FLAGS+=(--swap-objects)

engine_root() { [ "$1" = molmospaces ] && echo "$MOLMO" || echo "$ROBOCASA"; }

# Only the engine actually being run has to be installed. Setting up the other is a large
# download, and requiring it in order to use this one is a barrier with nothing behind it.
need_engine() {
  [ -x "$1/.venv/bin/python" ] \
    || die "$(basename "$1") is not set up yet - run: cd $1 && ./run.sh setup"
  # This script starts the engine's python directly rather than through its run.sh, so
  # it has to ask for the repair a moved checkout needs itself: without it `--mujoco`
  # dies on mjpython's shebang and MolmoSpaces reports a scene it holds as undownloadable.
  # A no-op when nothing has moved.
  "$1/run.sh" repair || die "could not re-point $(basename "$1") at this checkout"
}

# ---------------------------------------------------------------- engines
#
# Each engine runs in its own subshell. `env.sh` exports VENV_DIR, PYTHONPATH and
# MUJOCO_GL, and the two engines disagree on all three -- sourcing both into one shell
# would put robosuite on MolmoSpaces' path and the wrong interpreter on both.

molmospaces() {
  (
    # shellcheck source=/dev/null
    source "$MOLMO/env.sh"
    local xml
    # Scene MJCFs reference their meshes through the assets/ symlink tree, so the scene
    # has to be named by its assets/ path and not by its realpath. resolve_scene.py is
    # what gets that right -- and it downloads the house if this is a first run.
    xml="$("$MOLMO/.venv/bin/python" "$MOLMO/tools/resolve_scene.py" \
            "${SCENE%%:*}" "${SCENE##*:}")" || die "could not resolve scene $SCENE"
    # exec, so the PID this subshell reports to `$!` IS the engine. Without it the
    # subshell forks python as a child, `kill $!` reaps only the subshell, and the engine
    # is orphaned still holding its port -- which the next run then fails on, blaming the
    # port rather than the leak.
    exec "$1" "$MOLMO/tools/spawn_robot.py" "$ROBOTS" --scene "$xml" --target "$TARGET" "${@:2}"
  )
}

robocasa() {
  (
    # shellcheck source=/dev/null
    source "$ROBOCASA/env.sh"
    # No RoboCasa object sampling: the task brings its own objects at measured positions,
    # and RoboCasa's sampler would add a second apple the jaw cannot close on plus a bowl
    # inside the plate's footprint.
    exec "$1" "$ROBOCASA/tools/spawn_robot.py" "$ROBOTS" --layout "$LAYOUT" --style "$STYLE" "${@:2}"
  )
}

# The MuJoCo passive viewer must own the main thread on macOS, which is what mjpython
# provides; anything windowless runs under plain python. Same rule as both run.sh files.
engine_python() {
  local root; root="$(engine_root "$ENGINE")"
  if [ "$1" = viewer ] && [ "$(uname -s)" = "Darwin" ]; then
    echo "$root/.venv/bin/mjpython"
  else
    echo "$root/.venv/bin/python"
  fi
}

# 9090 is a popular port, and a sibling checkout running its own simulator is the likeliest
# thing holding it. Naming the holder turns a puzzling failure into an obvious one.
port_free() {
  nc -z 127.0.0.1 "$1" 2>/dev/null || return 0
  local holder
  holder="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -Fc 2>/dev/null | sed -n 's/^c//p' | sort -u | paste -sd, -)"
  die "port $1 is already in use${holder:+ (by: $holder)} - ${2:-pick another port}"
}

# Kill a backgrounded engine and everything it forked, then wait for the port to actually
# come free -- the next run's port check is otherwise the first thing that notices.
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
# The ids are robots_specs/robots.yml's, read by the one loader both engines use.
"$(engine_python headless)" "$ROOT/shared/robots_spec.py" check "$ROBOTS" \
  || die "--robots: see ./kitchen.sh serve --help"
# Checked up front, because the failure otherwise arrives as a websockets traceback from
# an engine that has already spent a minute compiling a kitchen.
port_free "$PORT" "pick another with --port PORT"
# EXIT as well as INT/TERM: without it a `die` anywhere below leaves the engine holding
# its port, and the next run fails the port check for no visible reason.
trap cleanup INT TERM EXIT
# The window, if one was asked for, is opened by this process because the physics is here:
# `launch_passive` builds a viewer from the model and data objects in memory. mjpython is
# the macOS main-thread requirement that comes with it, which is the whole of the
# difference between the two branches -- same engine, same flags, same wire.
declare -a HEADLESS=(--headless)
window=""
if [ "$MUJOCO" -eq 1 ]; then
  HEADLESS=()
  window=" in a window"
fi
echo ">> $ENGINE $ROBOTS$window on ws://127.0.0.1:$PORT"
# `--control-hz` is the rate of the members whose contract does not fix their own; a
# member that does (the myAGV, from ros_surfaces/myagv.py) runs at its contract's rates
# whatever this says, and the loop runs at the fastest member's.
"$ENGINE" "$(engine_python "$([ "$MUJOCO" -eq 1 ] && echo viewer || echo headless)")" \
  ${HEADLESS[@]+"${HEADLESS[@]}"} --ros-port "$PORT" --control-hz 10 \
  ${TASK_FLAGS[@]+"${TASK_FLAGS[@]}"} ${CAMERA_FLAGS[@]+"${CAMERA_FLAGS[@]}"} \
  ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
sim_pid=$!
echo
echo "run the task against it from robot_console/:"
echo "  ./run_task.sh --label $ENGINE$([ "$PORT" = 9090 ] || echo " --url ws://127.0.0.1:$PORT") --episodes 6"
wait "$sim_pid"
