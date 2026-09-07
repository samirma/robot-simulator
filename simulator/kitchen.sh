#!/usr/bin/env bash
# The SO-101, the myAGV and the AiNex on a kitchen work surface, one engine at a time.
#
#   ./kitchen.sh serve    load a world and serve it on rosbridge. Headless, always.
#   ./kitchen.sh view     look at a world: a MuJoCo window, the live camera page, or both.
#
#   ./kitchen.sh serve --help        ./kitchen.sh view --help
#
# Serving is `serve`'s alone and it never opens a window: a run is watched through the
# cameras the robots actually present, and a window costs control rate for every client
# on the port. Grading an episode is neither command's -- that is the console's:
#
#   cd ../robot_console && ./run_task.sh [--episodes N]
#
#: serve
# usage: ./kitchen.sh serve [flags]
#
# Loads an engine, stages shared/tasks/apple_on_plate.py into the kitchen it compiled, and
# serves the robots on rosbridge. No window, ever -- that is `view`.
#
#   --robots so101        which robots share the kitchen and the port: any of `so101`,
#                         `myagv`, `ainex`, comma-separated. Each gets its own namespace
#                         on one rosbridge -- /so101/*, /myagv/*, /ainex/* -- which is one
#                         ROS graph with a namespace per robot, as a real bringup is.
#   --cameras both        which cameras render:
#                           both   the worktop rig and every robot's own
#                           scene  the rig alone, on /scene/overhead and /scene/side
#                           robot  each robot's own alone -- the SO-101's wrist_cam,
#                                  a myAGV's or an AiNex's front_camera
#                         Each one renders inside the physics loop, so each one costs
#                         control rate for everyone on the port: one SO-101 publishes at
#                         9.8 Hz and adding a camera-bearing myAGV takes it to 5.7. Both
#                         narrowing settings take topics off the wire that something
#                         expects -- `robot` drops the rig the arm task is graded from,
#                         `scene` drops a base's camera, part of its vendor contract -- so
#                         the console's fleet check refuses them, by name.
#   --engine molmospaces  molmospaces | robocasa. One per run; only that one need be set up.
#   --scene ithor:1       MolmoSpaces scene
#   --layout 1 --style 1  RoboCasa kitchen, both 1-60
#   --objects plate,apple categories that *rank* MolmoSpaces surfaces; spawns nothing
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
#   ./kitchen.sh serve --engine robocasa --robots ainex --cameras robot
#
#: view
# usage: ./kitchen.sh view [--mujoco] [--live] [world flags]
#
# Both by default; naming one gives that one alone.
#
#   --mujoco      a MuJoCo window, on a world of its own. It has to be its own: MuJoCo
#                 builds a viewer from the model and data objects in memory -- there is
#                 launch, launch_from_path and launch_passive, and no connect -- so a
#                 window belongs to the process holding the physics and cannot be opened
#                 onto one already running. This world is served to nobody: no rosbridge,
#                 no robot on the wire. Say which world with the same --engine / --scene /
#                 --layout / --robots flags `serve` takes; they default the same way.
#   --live        the live camera page, in a browser, on the rosbridge at --port. This
#                 half really does attach -- it is a websocket client -- so it shows what
#                 a `serve` in another terminal is publishing and needs no world of its
#                 own. Nothing serving there is an error.
#   --port 9090       the rosbridge --live watches
#   --http-port 8791  the port the page itself is served on
#
# Examples:
#   ./kitchen.sh view --live                      # watch a serve running elsewhere
#   ./kitchen.sh view --mujoco --engine robocasa  # just the window, own world
#   ./kitchen.sh view --engine robocasa           # both
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

OBJECTS="plate,apple"
SCENE="ithor:1"
LAYOUT=1
STYLE=1
# 9090 is the rosbridge default and what a real bringup for this arm presents.
PORT="9090"
HTTP_PORT=8791
ROBOTS="so101"
ENGINE="molmospaces"
CAMERAS="both"
# What `view` opens. "auto" is both, which is what it means unless one of them is named.
MUJOCO="auto"
LIVE="auto"
declare -a STAGE_FLAGS=()
declare -a CAMERA_FLAGS=()
REFERENCE_TABLE=0
# Whether the plate and the apple trade places. Resolved per engine below: on for
# robocasa, off for molmospaces, so the two engines show a policy the same objects in
# different arrangements. The console reads the layout off the wire.
SWAP="auto"

die() { echo "error: $*" >&2; exit 1; }
say() { printf '\033[1m%s\033[0m\n' "$*"; }

# The header comment is the help, cut into sections by `#:` markers so one block serves
# `help`, `serve --help` and `view --help` and cannot drift from the flags it documents.
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
  view|serve|help|-h|--help) cmd="$1"; shift || true ;;
esac
case "$cmd" in help|-h|--help) usage; exit 0 ;; esac

# One list, because both commands shape a world the same way and differ only in what they
# then do with it. What belongs to one command alone is refused for the other, by name.
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)  usage "$cmd"; exit 0 ;;
    --objects)  OBJECTS="$2"; shift 2 ;;
    --scene)    SCENE="$2";   shift 2 ;;
    --layout)   LAYOUT="$2";  shift 2 ;;
    --style)    STYLE="$2";   shift 2 ;;
    --port)     PORT="$2";    shift 2 ;;
    --http-port) HTTP_PORT="$2"; shift 2 ;;
    --engine)   ENGINE="$2";  shift 2 ;;
    --robots)   ROBOTS="$2";  shift 2 ;;
    --cameras)  CAMERAS="$2"; shift 2 ;;
    --mujoco)   MUJOCO=1; [ "$LIVE" = auto ] && LIVE=0; shift ;;
    --live)     LIVE=1; [ "$MUJOCO" = auto ] && MUJOCO=0; shift ;;
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

# Refused by name rather than accepted and ignored: silently doing nothing with a flag
# somebody typed on purpose is worse than saying where it went.
if [ "$cmd" = serve ]; then
  { [ "$MUJOCO" = auto ] && [ "$LIVE" = auto ]; } \
    || die "--mujoco and --live belong to \`view\`: a serve is headless, always.
    ./kitchen.sh view --help"
else
  [ "$CAMERAS" = both ] \
    || die "--cameras belongs to \`serve\`: it chooses what goes on the wire, and a view
    puts nothing there.  ./kitchen.sh serve --help"
fi

# Naming either one speaks for both, which is what makes `view --live` the page *without*
# a window rather than the page as well as one.
if [ "$MUJOCO" = auto ]; then [ "$cmd" = view ] && MUJOCO=1 || MUJOCO=0; fi
if [ "$LIVE" = auto ];   then [ "$cmd" = view ] && LIVE=1   || LIVE=0;   fi

case "$ENGINE" in
  molmospaces|robocasa) ;;
  *) die "--engine: expected molmospaces or robocasa" ;;
esac

# No per-robot syntax behind these three words: an engine already resolves each base's own
# camera against that robot's MJCF prefix, and the arm's is one flag, so "every robot's
# own" is what happens when neither is overridden, and `--camera none` is the fleet-wide
# off switch `scene` wants. All engine flags, and both engines take them, so the camera
# set is not somewhere the two can drift apart.
case "$CAMERAS" in
  both)  CAMERA_FLAGS=(--wrist-camera) ;;
  scene) CAMERA_FLAGS=(--camera none) ;;
  robot) CAMERA_FLAGS=(--wrist-camera --no-scene-cameras) ;;
  *)     die "--cameras: expected both, scene or robot" ;;
esac

[ "$REFERENCE_TABLE" -eq 1 ] || STAGE_FLAGS+=(--no-reference-table)

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
    exec "$1" "$MOLMO/tools/spawn_robot.py" "$ROBOTS" --scene "$xml" --target "$OBJECTS" "${@:2}"
  )
}

robocasa() {
  (
    # shellcheck source=/dev/null
    source "$ROBOCASA/env.sh"
    # No `--objects`: both commands stage a task, which brings its own objects at measured
    # positions, and RoboCasa's sampler would add a second apple the jaw cannot close on
    # plus a bowl inside the plate's footprint. spawn_robot.py refuses the combination
    # outright; not passing it is what keeps it from being asked for.
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
  if [ -n "${http_pid:-}" ]; then kill "$http_pid" 2>/dev/null || true; fi
  if [ -n "${view_pid:-}" ]; then stop_engine "$view_pid" ""; fi
  if [ -n "${sim_pid:-}" ]; then stop_engine "$sim_pid" "$PORT"; fi
}

# Served over HTTP rather than opened from file://, because browsers refuse a ws://
# connection from a file:// origin and the page then sits there discovering nothing.
serve_page() {
  local page="$ROOT/live_cameras.html"
  [ -f "$page" ] || die "missing $page"
  port_free "$HTTP_PORT" "pick another with --http-port PORT"
  python3 -m http.server "$HTTP_PORT" --directory "$ROOT" --bind 127.0.0.1 >/dev/null 2>&1 &
  http_pid=$!
  # `ns` tells the page which robot's state and command topics the sliders drive. The
  # camera grid does not need it -- it discovers streams from rosapi, so every robot on
  # the port shows up regardless.
  local url="http://127.0.0.1:$HTTP_PORT/live_cameras.html?url=ws://127.0.0.1:$PORT&ns=so101"
  say "camera page: $url"
  command -v open >/dev/null && open "$url" || true
}

# ---------------------------------------------------------------- run

if [ "$cmd" = view ]; then
  trap cleanup INT TERM EXIT

  if [ "$MUJOCO" -eq 1 ]; then
    need_engine "$(engine_root "$ENGINE")"
    echo ">> $ENGINE $ROBOTS in a window (its own world; nothing is served)"
    # No --ros-port: this world is looked at, not served. The task is still staged, so
    # the window shows the apple and the plate rather than a bare counter.
    "$ENGINE" "$(engine_python viewer)" --task apple_on_plate --control-hz 10 \
      ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
    view_pid=$!
  fi

  if [ "$LIVE" -eq 1 ]; then
    nc -z 127.0.0.1 "$PORT" 2>/dev/null \
      || die "nothing is serving on ws://127.0.0.1:$PORT - start one with:
    ./kitchen.sh serve"
    serve_page
    echo "  the page shows what that simulator was started with; Ctrl-C stops serving it"
  fi

  if [ -n "${view_pid:-}" ]; then wait "$view_pid"; fi
  if [ -n "${http_pid:-}" ]; then wait "$http_pid"; fi
  exit 0
fi

need_engine "$(engine_root "$ENGINE")"
# Checked up front, because the failure otherwise arrives as a websockets traceback from
# an engine that has already spent a minute compiling a kitchen.
port_free "$PORT" "pick another with --port PORT"
# EXIT as well as INT/TERM: without it a `die` anywhere below leaves the engine holding
# its port, and the next run fails the port check for no visible reason.
trap cleanup INT TERM EXIT
echo ">> $ENGINE $ROBOTS on ws://127.0.0.1:$PORT (cameras: $CAMERAS)"
"$ENGINE" "$(engine_python headless)" --headless --ros-port "$PORT" \
  --task apple_on_plate --control-hz 10 \
  ${CAMERA_FLAGS[@]+"${CAMERA_FLAGS[@]}"} ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
sim_pid=$!
echo
echo "run the task against it from robot_console/:"
echo "  ./run_task.sh --label $ENGINE$([ "$PORT" = 9090 ] || echo " --url ws://127.0.0.1:$PORT") --episodes 6"
echo "watch its cameras:  ./kitchen.sh view --live$([ "$PORT" = 9090 ] || echo " --port $PORT")"
wait "$sim_pid"
