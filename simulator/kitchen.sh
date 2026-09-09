#!/usr/bin/env bash
# The SO-101 and the myAGV on a kitchen work surface, one engine at a time.
#
#   ./kitchen.sh serve    load a world, serve it on rosbridge, optionally show it
#   ./kitchen.sh view     the live camera page, on whatever is being served
#
#   ./kitchen.sh serve --help        ./kitchen.sh view --help
#
# `serve` owns the world, so it owns the window: MuJoCo builds a viewer from the model and
# data objects in memory -- there is launch, launch_from_path and launch_passive, and no
# connect -- so a window can only exist inside the process holding the physics. That is
# why `--mujoco` is a `serve` flag and `view` has none. `view` is a websocket client and
# nothing else: it attaches to a port and shows what is on it. Grading an episode is
# neither command's -- that is the console's:
#
#   cd ../robot_console && ./run_task.sh [--episodes N]
#
#: serve
# usage: ./kitchen.sh serve [flags]
#
# Loads an engine, stages shared/tasks/apple_on_plate.py into the kitchen it compiled, and
# serves the robots on rosbridge. Headless unless --mujoco asks for a window.
#
#   --robots so101        which robots share the kitchen and the port: `so101`, `myagv`,
#                         comma-separated. Each gets its own namespace on one rosbridge
#                         -- /so101/*, /myagv/* -- which is one ROS graph with a
#                         namespace per robot, as a real bringup is. The arm is bolted
#                         to a worktop; a myAGV takes the floor of the same room.
#                         apple_on_plate is staged when the arm is in the list, and with
#                         it the worktop's camera rig on /scene/* -- the rig is the
#                         scene's, published by the fleet, not by any robot. A kitchen
#                         holding only a base gets the room, the robot and its camera.
#   --cameras both        which cameras render:
#                           both   the worktop rig and every robot's own
#                           scene  the rig alone, on /scene/overhead and /scene/side
#                           robot  each robot's own alone -- the SO-101's wrist_cam,
#                                  a myAGV's front_camera
#                         Each one renders inside the physics loop, so each one costs
#                         control rate for everyone on the port: one SO-101 publishes at
#                         9.8 Hz and adding a camera-bearing myAGV takes it to 5.7. Both
#                         narrowing settings take topics off the wire that something
#                         expects -- `robot` drops the rig the arm task is graded from,
#                         `scene` drops a base's camera, part of its vendor contract -- so
#                         the console's fleet check refuses them, by name.
#   --mujoco              open a MuJoCo window on the world being served. The window is
#                         built from the model and data objects in memory -- MuJoCo has
#                         launch, launch_from_path and launch_passive and no connect -- so
#                         it belongs to the process holding the physics, which is this
#                         one. That is why the flag is here and not on `view`, and why a
#                         serve already running cannot grow a window: restart it with
#                         --mujoco. Rendering it costs control rate for every client on
#                         the port, and closing the window ends the run.
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
#   ./kitchen.sh serve --engine robocasa --robots myagv --cameras robot --mujoco
#
#: view
# usage: ./kitchen.sh view [--port PORT] [--http-port PORT]
#
# Opens the live camera page in a browser on the rosbridge at --port. That is all `view`
# is and all it can be: a websocket client. It holds no physics, compiles no kitchen and
# takes none of the flags that shape a world -- it shows whatever is already on that port,
# discovering the robots and their cameras from the wire, so it works against a serve in
# another terminal without being told what that serve was started with. A window belongs
# to the run that owns the world; ask `serve` for one with --mujoco.
#
#   --live            names what `view` already is, and is accepted for that reason alone
#   --port 9090       the rosbridge to watch. Nothing serving there is an error.
#   --http-port 8791  the port the page itself is served on
#
# Examples:
#   ./kitchen.sh view                 # watch the serve on 9090
#   ./kitchen.sh view --port 9091     # ...or the one on 9091
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
# Whether `serve` also opens a window on the world it is serving.
MUJOCO=0
# Whether --live was typed. It names what `view` already is, so it changes nothing there;
# it exists so that `serve --live` can be refused by name rather than as "unknown flag".
LIVE=0
declare -a STAGE_FLAGS=()
# Flags typed on the command line that shape a world. `view` shapes none, so it names
# these back rather than accepting them and doing nothing with them.
declare -a WORLD_FLAGS=()
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

# One list, because the two commands share the ports and the help machinery. Everything
# that describes a world belongs to `serve` alone now that `view` builds none, and is
# collected into WORLD_FLAGS so a view can name back exactly what was typed.
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)  usage "$cmd"; exit 0 ;;
    --objects)  OBJECTS="$2"; WORLD_FLAGS+=("$1"); shift 2 ;;
    --scene)    SCENE="$2";   WORLD_FLAGS+=("$1"); shift 2 ;;
    --layout)   LAYOUT="$2";  WORLD_FLAGS+=("$1"); shift 2 ;;
    --style)    STYLE="$2";   WORLD_FLAGS+=("$1"); shift 2 ;;
    --port)     PORT="$2";    shift 2 ;;
    --http-port) HTTP_PORT="$2"; shift 2 ;;
    --engine)   ENGINE="$2";  WORLD_FLAGS+=("$1"); shift 2 ;;
    --robots)   ROBOTS="$2";  WORLD_FLAGS+=("$1"); shift 2 ;;
    --cameras)  CAMERAS="$2"; WORLD_FLAGS+=("$1"); shift 2 ;;
    --mujoco)   MUJOCO=1; shift ;;
    # `view` is the page and only the page, so this names what it already does. Kept
    # because it is what `serve` prints as the way to watch a run, and what a decade of
    # muscle memory and every example types.
    --live)     LIVE=1; shift ;;
    --reference-table)    REFERENCE_TABLE=1; WORLD_FLAGS+=("$1"); shift ;;
    --no-reference-table) REFERENCE_TABLE=0; WORLD_FLAGS+=("$1"); shift ;;
    --no-dressing)        STAGE_FLAGS+=(--no-dressing);        WORLD_FLAGS+=("$1"); shift ;;
    --reference-lighting) STAGE_FLAGS+=(--reference-lighting); WORLD_FLAGS+=("$1"); shift ;;
    --extra-lights)       STAGE_FLAGS+=(--extra-lights);       WORLD_FLAGS+=("$1"); shift ;;
    --swap-objects)       SWAP=1; WORLD_FLAGS+=("$1"); shift ;;
    --no-swap-objects)    SWAP=0; WORLD_FLAGS+=("$1"); shift ;;
    --task-objects)       STAGE_FLAGS+=(--task-objects);       WORLD_FLAGS+=("$1"); shift ;;
    --side-camera-mirror) STAGE_FLAGS+=(--side-camera-mirror); WORLD_FLAGS+=("$1"); shift ;;
    *) die "unknown flag '$1' (try: ./kitchen.sh $cmd --help)" ;;
  esac
done

# Refused by name rather than accepted and ignored: silently doing nothing with a flag
# somebody typed on purpose is worse than saying where it went. `view` compiles nothing,
# so every flag that describes a world is one of these -- `view --engine robocasa` used to
# be accepted and change nothing, which reads as the page showing the wrong engine.
if [ "$cmd" = serve ]; then
  [ "$LIVE" -eq 0 ] \
    || die "--live belongs to \`view\`: it names the camera page, and a serve opens none.
    ./kitchen.sh serve [flags] then ./kitchen.sh view"
else
  [ "$MUJOCO" -eq 0 ] \
    || die "--mujoco belongs to \`serve\`: a window is built from the model and data
    objects in memory, so it belongs to the process holding the physics -- there is no way
    to open one onto an engine already running. Ask the run that owns the world:
    ./kitchen.sh serve --mujoco [flags]"
  if [ "${#WORLD_FLAGS[@]}" -gt 0 ]; then
    named="$(printf '%s, ' "${WORLD_FLAGS[@]}")"; named="${named%, }"
    [ "${#WORLD_FLAGS[@]}" -eq 1 ] && verb="belongs" || verb="belong"
    die "$named $verb to \`serve\`: a view builds no world and shows whatever is on
    --port, discovering the robots and their cameras from the wire.
    ./kitchen.sh serve --help"
  fi
fi

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
  # Only the websocket address. The page asks rosapi what is on the graph and builds a
  # panel per robot it finds, so it needs no `ns=` and no --robots echoed at it -- which
  # it could not be given anyway when it is watching a serve in another terminal. `?ns=`
  # is still honoured, as a filter to one robot.
  local url="http://127.0.0.1:$HTTP_PORT/live_cameras.html?url=ws://127.0.0.1:$PORT"
  say "camera page: $url"
  command -v open >/dev/null && open "$url" || true
}

# ---------------------------------------------------------------- run

if [ "$cmd" = view ]; then
  trap cleanup INT TERM EXIT

  nc -z 127.0.0.1 "$PORT" 2>/dev/null \
    || die "nothing is serving on ws://127.0.0.1:$PORT - start one with:
    ./kitchen.sh serve$([ "$PORT" = 9090 ] || echo " --port $PORT")
    add --mujoco there if you also want a window on it"
  serve_page
  echo "  Ctrl-C stops the page; whatever serves ws://127.0.0.1:$PORT keeps running"
  wait "$http_pid"
  exit 0
fi

need_engine "$(engine_root "$ENGINE")"
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
echo ">> $ENGINE $ROBOTS$window on ws://127.0.0.1:$PORT (cameras: $CAMERAS)"
"$ENGINE" "$(engine_python "$([ "$MUJOCO" -eq 1 ] && echo viewer || echo headless)")" \
  ${HEADLESS[@]+"${HEADLESS[@]}"} --ros-port "$PORT" --control-hz 10 \
  ${TASK_FLAGS[@]+"${TASK_FLAGS[@]}"} ${CAMERA_FLAGS[@]+"${CAMERA_FLAGS[@]}"} \
  ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
sim_pid=$!
echo
echo "run the task against it from robot_console/:"
echo "  ./run_task.sh --label $ENGINE$([ "$PORT" = 9090 ] || echo " --url ws://127.0.0.1:$PORT") --episodes 6"
echo "watch its cameras:  ./kitchen.sh view --live$([ "$PORT" = 9090 ] || echo " --port $PORT")"
wait "$sim_pid"
