#!/usr/bin/env bash
# The SO-101 on a kitchen work surface, in *both* engines, around the same two objects.
#
#   ./kitchen_arm.sh                 render a screenshot from each engine into shots/
#   ./kitchen_arm.sh view            open both engines in the MuJoCo viewer
#   ./kitchen_arm.sh serve           run both headless, each on its own control port
#   ./kitchen_arm.sh help
#
#   --objects bowl,apple   the pair to stage the scene around (default: bowl,apple)
#   --scene ithor:1        MolmoSpaces scene    (default: ithor:1, a kitchen)
#   --layout 1 --style 1   RoboCasa kitchen     (both 1-60)
#   --out DIR              where screenshots go (default: shots/)
#   --ports 8000,8001      control ports for `serve`  (molmospaces, robocasa)
#
# Why bowl + apple. The two engines share no assets, so "the same objects" can only mean
# the same *categories* present in both, and they should be a pair that can actually be
# made to interact -- putting the apple in the bowl is a pick-and-place either arm can be
# asked to do. iTHOR FloorPlan1's island carries a Bowl, an Apple, Bread, a ButterKnife
# and a Tomato; RoboCasa's objaverse registry has bowl, apple, bread, knife, plate, spoon,
# mug and ~70 more. Bowl and apple are the pair that is on a *reachable* surface in both:
# MolmoSpaces is told which of the island's objects to build the mount around (--target),
# and RoboCasa is told which to spawn on the worktop (--objects), because RoboCasa
# kitchens are fixtures and ship with no loose objects at all until something adds them.
#
# The two engines are otherwise driven identically, and that is the point: the same robot
# spec out of shared/robots/so101/, the same molmospaces-control-v1 protocol on the wire,
# and one `robot-console-arm` that cannot tell which of them it is connected to.
set -euo pipefail

# Resolved without cd; see the note in each engine's env.sh about title-escape capture.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
ROOT="$(dirname "$_self")"
ROOT="$(realpath "$ROOT" 2>/dev/null || echo "${ROOT%/.}")"

MOLMO="$ROOT/molmospaces"
ROBOCASA="$ROOT/robocasa"

OBJECTS="bowl,apple"
SCENE="ithor:1"
LAYOUT=1
STYLE=1
OUT="$ROOT/shots"
PORTS="8000,8001"

die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------- arguments

cmd="shot"
case "${1:-}" in
  shot|view|serve|help|-h|--help) cmd="$1"; shift || true ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --objects) OBJECTS="$2"; shift 2 ;;
    --scene)   SCENE="$2";   shift 2 ;;
    --layout)  LAYOUT="$2";  shift 2 ;;
    --style)   STYLE="$2";   shift 2 ;;
    --out)     OUT="$2";     shift 2 ;;
    --ports)   PORTS="$2";   shift 2 ;;
    *) die "unknown flag '$1' (try: ./kitchen_arm.sh help)" ;;
  esac
done

if [ "$cmd" = "help" ] || [ "$cmd" = "-h" ] || [ "$cmd" = "--help" ]; then
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
  exit 0
fi

MOLMO_PORT="${PORTS%%,*}"
ROBOCASA_PORT="${PORTS##*,}"

for engine in "$MOLMO" "$ROBOCASA"; do
  [ -x "$engine/.venv/bin/python" ] \
    || die "$(basename "$engine") is not set up yet - run: cd $engine && ./run.sh setup"
done

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
    "$1" "$MOLMO/tools/spawn_robot.py" so101 --scene "$xml" --target "$OBJECTS" "${@:2}"
  )
}

robocasa() {
  (
    # shellcheck source=/dev/null
    source "$ROBOCASA/env.sh"
    "$1" "$ROBOCASA/tools/spawn_robot.py" so101 --layout "$LAYOUT" --style "$STYLE" \
      --objects "$OBJECTS" "${@:2}"
  )
}

# The MuJoCo passive viewer must own the main thread on macOS, which is what mjpython
# provides; anything windowless runs under plain python. Same rule as both run.sh files.
py()     { echo "$1/.venv/bin/python"; }
viewer() {
  if [ "$(uname -s)" = "Darwin" ]; then echo "$1/.venv/bin/mjpython"; else echo "$1/.venv/bin/python"; fi
}

# ---------------------------------------------------------------- commands

case "$cmd" in
  shot)
    mkdir -p "$OUT"
    echo ">> molmospaces: so101 in $SCENE around $OBJECTS"
    molmospaces "$(py "$MOLMO")" --render "$OUT/molmospaces_so101.png" \
      --width 1600 --height 1000 --distance 1.1 --elevation -22
    echo ">> robocasa: so101 in kitchen layout $LAYOUT style $STYLE around $OBJECTS"
    robocasa "$(py "$ROBOCASA")" --render "$OUT/robocasa_so101.png" \
      --width 1600 --height 1000 --distance 1.1 --elevation -22
    echo
    echo "screenshots:"
    echo "  $OUT/molmospaces_so101.png"
    echo "  $OUT/robocasa_so101.png"
    ;;

  view)
    # Two windows, two processes. Backgrounding the first is safe precisely because they
    # are separate processes: each mjpython owns the main thread of its own.
    echo ">> molmospaces viewer (close the window to quit)"
    molmospaces "$(viewer "$MOLMO")" &
    molmo_pid=$!
    echo ">> robocasa viewer (close the window to quit)"
    robocasa "$(viewer "$ROBOCASA")" &
    robocasa_pid=$!
    trap 'kill "$molmo_pid" "$robocasa_pid" 2>/dev/null || true' INT TERM
    wait
    ;;

  serve)
    # Checked up front, because the failure otherwise arrives as a websockets traceback
    # from inside an engine that has already spent a minute compiling a kitchen. 8000 is
    # a popular port; something else on the machine having it is not a bug here.
    for port in "$MOLMO_PORT" "$ROBOCASA_PORT"; do
      if nc -z 127.0.0.1 "$port" 2>/dev/null; then
        die "port $port is already in use - pick others with --ports A,B"
      fi
    done
    echo ">> molmospaces so101 on ws://127.0.0.1:$MOLMO_PORT"
    molmospaces "$(py "$MOLMO")" --headless --control "127.0.0.1:$MOLMO_PORT" &
    molmo_pid=$!
    echo ">> robocasa so101 on ws://127.0.0.1:$ROBOCASA_PORT"
    robocasa "$(py "$ROBOCASA")" --headless --control "127.0.0.1:$ROBOCASA_PORT" &
    robocasa_pid=$!
    echo
    echo "drive either one with the same client, from robot_console/:"
    echo "  robot-console-arm --control 127.0.0.1:$MOLMO_PORT"
    echo "  robot-console-arm --control 127.0.0.1:$ROBOCASA_PORT"
    trap 'kill "$molmo_pid" "$robocasa_pid" 2>/dev/null || true' INT TERM
    wait
    ;;
esac
