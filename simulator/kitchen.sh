#!/usr/bin/env bash
# The SO-101 on a kitchen work surface, in one engine at a time.
#
#   ./kitchen.sh                 render a screenshot from each engine into shots/
#   ./kitchen.sh view            open the engine in the MuJoCo viewer
#   ./kitchen.sh serve           stage the task and serve it on rosbridge; --viewer for a window
#   ./kitchen.sh cameras         open the live camera page on a running `serve`
#   ./kitchen.sh help
#
# This script hosts the world. Running a task against it is the console's job:
#
#   cd ../robot_console && ./run_task.sh [--episodes N]
#
# `inspect` used to do both halves here and no longer exists -- grading an episode is not
# simulation configuration, and the shell heredoc it lived in was untested.
#
#   --objects plate,apple  the pair each engine builds its worktop around (default
#                          plate,apple), for shot/view only. Each engine furnishes it from
#                          its *own* assets: RoboCasa samples the pair out of its object
#                          registry, and MolmoSpaces moves the house's own apple and plate
#                          into the arm's working annulus and sinks the clutter that was in
#                          its way. MolmoSpaces used to spawn nothing at all -- --target
#                          only *ranked* the house's surfaces -- so its shot was of a book,
#                          a loaf and a tomato while RoboCasa's was of the pair, and the
#                          two screenshots were not comparable. --target still ranks
#                          surfaces as well; ranking picks the counter, the placement then
#                          furnishes it. Under `serve` neither of these runs: the task
#                          stages its own measured apple and plate 0.32 m from the arm and
#                          clears the same workspace itself. The engine prints both
#                          distances at startup, so "in reach" is read, not assumed.
#   --scene ithor:1        MolmoSpaces scene    (default: ithor:1, a kitchen)
#   --layout 1 --style 1   RoboCasa kitchen     (both 1-60)
#   --out DIR              where screenshots go (default: shots/)
#   --port 9090            rosbridge port for `serve`
#
#   serve:
#   --robots A,B           which robots share the kitchen and the port (default so101).
#                          `so101,myagv` mounts the arm on a work surface and puts the
#                          base on the floor of the same room, both on one rosbridge
#                          under /so101/* and /myagv/* -- one ROS graph, a namespace per
#                          robot, which is how a real multi-robot bringup is arranged.
#                          run_task.sh still grades the arm; pass it the same --robots
#                          and it checks the extra robot is on the wire.
#   --engine E             molmospaces (default) | robocasa. One engine per run: only
#                          the engine named here needs to be set up, and every command
#                          honours it.
#   --wrist                also stream the eye-in-hand camera
#   --viewer               open the engine's own MuJoCo window while it serves. One
#                          process, so the window shows exactly the state on the wire.
#                          It costs camera rate -- measured on MolmoSpaces with three
#                          cameras, 7.6 Hz headless against 5.0 Hz with the window --
#                          and the loop still holds real time at that price.
#   --reference-table      also stage the reference rig's 0.92 m wooden work surface
#                          under the objects. Off by default: it sits on top of the
#                          kitchen's own counter and reads as one table overlapping
#                          another, and measured it does not move the VLA's pass count
#                          (2/6 on the bare counter against 1/6 with the slab). The
#                          scripted policy is verified both ways.
#   --no-reference-table   the default, kept so older invocations still parse
#   --no-dressing          apple and plate only, without the bowl/mug/banana/lemon
#   --reference-lighting   impose the reference rig's exposure on the kitchen. Off by
#                          default, against what the photometry says: it matches the
#                          reference's clipped-pixel fraction almost exactly and costs
#                          MolmoAct2 the task outright (0/24 episodes with it, 6/18
#                          without). For comparing exposure, not for running a policy.
#   --extra-lights         add the reference's two lamps (they blow out a lit kitchen)
#   --swap-objects         stage the plate at the apple's spawn and the apple where the
#                          plate was. ON BY DEFAULT for --engine robocasa and off for
#                          molmospaces, so the two engines show the policy the same
#                          objects in different arrangements; --no-swap-objects turns it
#                          off. The console reads the layout off the wire (the apple's
#                          reset position), so run_task.sh needs no matching flag.
#   --task-objects         stage the task's own measured YCB apple and plate. The default
#                          is each engine's *native* pair -- the iTHOR house's apple and
#                          plate in MolmoSpaces, apple_10 and plate_4 from RoboCasa's
#                          registry -- scaled to the task's radii and with the task's
#                          measured contact block on the apple. See adopt_native_objects
#                          / make_task_objects in each engine's spawn_robot.py.
#   --side-camera-mirror   stage the side camera on the other side of the worktop
#                          (reflected across the arm's x-z plane). In the swapped
#                          layout the reference side view has the plate between it and
#                          the apple; from the other side the apple is the near object.
#                          An experiment flag; the policy is still told the contract
#                          poses in its docs.
#   --http-port 8791       port the `cameras` page is served on
#
# One engine per run. Two at once meant two ports, two of every flag, and two kitchens
# competing for one GPU -- which on this machine is not a theoretical cost: RoboCasa's
# 44-fixture kitchen with its cameras and a viewer window already runs at 0.12x real time
# on its own, and the client refuses to start below 0.10x. Comparing the engines is still
# the point, and running each in turn against the same client is what actually
# demonstrates the thing worth demonstrating -- that the client cannot tell them apart.
#
# What the task brings, and what the engines bring. `serve` stages
# shared/tasks/apple_on_plate.py into whichever kitchen an engine compiled: the reference
# rig's objects -- a 20 mm apple, a white plate, and the bowl, mug, banana and lemon it
# keeps as scenery -- at the poses and contact parameters that were measured there, on
# the engine's own counter. The engine still supplies the room and still mounts the arm;
# the task supplies the geometry, because that is the part a procedurally chosen object
# cannot.
#
# The reference rig's wooden slab is not staged by default (see --reference-table). It
# matches the reference's overhead framing to the figure -- all four corners at a worst
# normalised radius of 0.930 -- and it does not move the VLA's pass count, while it does
# sit visibly on top of the kitchen's own island. Framing that matches a number is not the
# same as a scene a policy can act in; the lighting result taught that already.
#
# `--objects` chooses what RoboCasa puts on its worktop for `shot` and `view`; it is not
# passed to `serve`, because a task that stages its own apple does not want a second one
# the jaw cannot close on. MolmoSpaces gets the same pair as `--target` in every command,
# which only ranks surfaces and spawns nothing.
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
OUT="$ROOT/shots"
# 9090 is the rosbridge default and what a real bringup for this arm presents, so it is
# the right port for whichever engine is running. There is only ever one.
PORT="9090"
# Which robots share the kitchen, and therefore the port. One ROS graph with a namespace
# per robot -- `so101` alone is on /so101/*, and `so101,myagv` adds /myagv/* beside it on
# the same socket. The default is the arm alone, so every existing invocation is
# unchanged.
ROBOTS="so101"
ENGINE="molmospaces"
WRIST=0
VIEWER=0
HTTP_PORT=8791
# The task stages the reference slab unless told not to; this script tells it not to.
declare -a STAGE_FLAGS=(--no-reference-table)
# Whether the plate and the apple trade places. "auto" resolves per engine once the
# engine is known: on for robocasa, off for molmospaces -- one engine keeps the
# contract's arrangement and the other shows the policy the same objects the other way
# round. The console needs no matching flag; it reads the layout off the wire.
SWAP="auto"
# Set by the commands that stage a task, so the engine wrappers know not to also ask the
# engine for loose objects of its own.
STAGING_TASK=0

die() { echo "error: $*" >&2; exit 1; }
say() { printf '\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- arguments

cmd="shot"
case "${1:-}" in
  shot|view|serve|inspect|cameras|help|-h|--help) cmd="$1"; shift || true ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --objects)  OBJECTS="$2"; shift 2 ;;
    --scene)    SCENE="$2";   shift 2 ;;
    --layout)   LAYOUT="$2";  shift 2 ;;
    --style)    STYLE="$2";   shift 2 ;;
    --out)      OUT="$2";     shift 2 ;;
    --port)     PORT="$2";    shift 2 ;;
    --ports)    die "--ports is gone: one engine runs at a time now, so use --port PORT" ;;
    --engine)   ENGINE="$2";  shift 2 ;;
    --robots)   ROBOTS="$2";  shift 2 ;;
    --policy|--steps|--episodes|--log-dir|--)
      die "$1 belongs to robot_console/run_task.sh now: this script only hosts the world" ;;
    --wrist)    WRIST=1;      shift ;;
    --viewer)   VIEWER=1;     shift ;;
    --http-port) HTTP_PORT="$2"; shift 2 ;;
    --no-reference-table) shift ;;
    --reference-table)    STAGE_FLAGS=("${STAGE_FLAGS[@]/--no-reference-table}"); shift ;;
    --no-dressing)        STAGE_FLAGS+=(--no-dressing);        shift ;;
    --reference-lighting) STAGE_FLAGS+=(--reference-lighting); shift ;;
    --extra-lights)       STAGE_FLAGS+=(--extra-lights);       shift ;;
    --swap-objects)       SWAP=1; shift ;;
    --no-swap-objects)    SWAP=0; shift ;;
    --task-objects)       STAGE_FLAGS+=(--task-objects);       shift ;;
    --side-camera-mirror) STAGE_FLAGS+=(--side-camera-mirror); shift ;;
    *) die "unknown flag '$1' (try: ./kitchen.sh help)" ;;
  esac
done

if [ "$cmd" = "help" ] || [ "$cmd" = "-h" ] || [ "$cmd" = "--help" ]; then
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
  exit 0
fi


case "$ENGINE" in
  molmospaces|robocasa) ;;
  both) die "--engine both is gone: run one engine at a time.
    Two engines at once meant two ports, two of every flag and two kitchens sharing one
    GPU -- on this machine that alone cost enough frame rate to stall a run. Compare them
    by running each in turn against the same client; that is the comparison that matters,
    because a client cannot tell them apart." ;;
  *) die "--engine: expected molmospaces or robocasa" ;;
esac

# The swap resolves per engine only once the engine is known -- see SWAP above.
if [ "$SWAP" = auto ]; then
  [ "$ENGINE" = robocasa ] && SWAP=1 || SWAP=0
fi
[ "$SWAP" -eq 0 ] || STAGE_FLAGS+=(--swap-objects)

# Defined here rather than beside the other engine helpers below, because `need_engine`
# calls it 70 lines before that point and bash binds a function only when it executes the
# definition. It used to live below, and every command except `cameras` died on
# `engine_root: command not found` with an empty engine name in the message.
engine_root() { [ "$1" = molmospaces ] && echo "$MOLMO" || echo "$ROBOCASA"; }

need_engine() {
  [ -x "$1/.venv/bin/python" ] \
    || die "$(basename "$1") is not set up yet - run: cd $1 && ./run.sh setup"
}
# Only the engine actually being run has to be installed. Setting up the other is a
# large download, and requiring it in order to use this one is a barrier with nothing
# behind it. `cameras` starts no engine at all.
case "$cmd" in
  cameras) ;;
  *) need_engine "$(engine_root "$ENGINE")" ;;
esac

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
    # `--objects` only for the comparison commands. `serve`/`inspect` stage a task,
    # which brings its own objects at measured positions -- and RoboCasa's sampler would
    # add a second apple the jaw cannot close on plus a bowl inside the plate's
    # footprint. spawn_robot.py refuses the combination outright; this is what keeps it
    # from ever being asked for.
    local objects=()
    [ "$STAGING_TASK" -eq 1 ] || objects=(--objects "$OBJECTS")
    exec "$1" "$ROBOCASA/tools/spawn_robot.py" "$ROBOTS" --layout "$LAYOUT" --style "$STYLE" \
      ${objects[@]+"${objects[@]}"} "${@:2}"
  )
}

# The MuJoCo passive viewer must own the main thread on macOS, which is what mjpython
# provides; anything windowless runs under plain python. Same rule as both run.sh files.
py()     { echo "$1/.venv/bin/python"; }
# --viewer needs mjpython on macOS (the passive viewer must own the Cocoa main thread)
# and drops --headless. The offscreen camera renderers and the on-screen window then
# coexist in one process, which was not obviously safe beforehand -- under mjpython the
# script runs off the main thread and each renderer opens a hidden GLFW window -- and was
# verified before this flag was added. If it ever stops working, both env.sh files name
# MUJOCO_GL=cgl as the escape hatch.
engine_python() {
  local root; root="$(engine_root "$1")"
  if [ "$VIEWER" -eq 1 ]; then viewer "$root"; else py "$root"; fi
}
headless_arg() { [ "$VIEWER" -eq 1 ] || echo "--headless"; }
viewer() {
  if [ "$(uname -s)" = "Darwin" ]; then echo "$1/.venv/bin/mjpython"; else echo "$1/.venv/bin/python"; fi
}

# 9090 is the rosbridge default and what a real bringup for this arm uses, so it is the
# right default here -- but it is also a popular port, and a sibling checkout running its
# own simulator is the likeliest thing holding it. Naming the holder turns a puzzling
# failure into an obvious one.

port_free() {
  nc -z 127.0.0.1 "$1" 2>/dev/null || return 0
  local holder
  holder="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -Fc 2>/dev/null | sed -n 's/^c//p' | sort -u | paste -sd, -)"
  die "port $1 is already in use${holder:+ (by: $holder)} - pick another with --port PORT"
}

# Kill a backgrounded engine and everything it forked, then wait for the port to actually
# come free -- the next run's port check is otherwise the first thing that notices.
stop_engine() {
  local pid="$1" port="$2" i
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  for i in $(seq 1 20); do
    nc -z 127.0.0.1 "$port" 2>/dev/null || return 0
    sleep 0.5
  done
  echo "warning: port $port still held after stopping the engine" >&2
}

# ---------------------------------------------------------------- commands

case "$cmd" in
  shot)
    mkdir -p "$OUT"
    case "$ENGINE" in
      molmospaces) echo ">> molmospaces: so101 in $SCENE around $OBJECTS" ;;
      robocasa)    echo ">> robocasa: so101 in kitchen layout $LAYOUT style $STYLE around $OBJECTS" ;;
    esac
    "$ENGINE" "$(py "$(engine_root "$ENGINE")")" --render "$OUT/${ENGINE}_so101.png" \
      --width 1600 --height 1000 --distance 1.1 --elevation -22
    echo
    echo "screenshot: $OUT/${ENGINE}_so101.png"
    # Comparing the engines is still the point of this command; it is just done one run
    # at a time now, which is also the only way the two shots can share a GPU fairly.
    [ -f "$OUT/molmospaces_so101.png" ] && [ -f "$OUT/robocasa_so101.png" ] \
      && echo "  (both engines rendered - $OUT holds the pair)"
    ;;

  view)
    echo ">> $ENGINE viewer (close the window to quit)"
    "$ENGINE" "$(viewer "$(engine_root "$ENGINE")")"
    ;;

  serve)
    STAGING_TASK=1
    wrist_arg=(); [ "$WRIST" -eq 1 ] && wrist_arg=(--wrist-camera)
    # Checked up front, because the failure otherwise arrives as a websockets traceback
    # from an engine that has already spent a minute compiling a kitchen.
    port_free "$PORT"
    # EXIT as well as INT/TERM: without it a `die` anywhere below leaves the engine
    # holding its port, and the next run fails the port check for no visible reason.
    trap 'stop_engine "$sim_pid" "$PORT"' INT TERM EXIT
    echo ">> $ENGINE $ROBOTS on ws://127.0.0.1:$PORT$([ "$VIEWER" -eq 1 ] && echo ' (with a window)')"
    "$ENGINE" "$(engine_python "$ENGINE")" $(headless_arg) --ros-port "$PORT" \
      --task apple_on_plate --control-hz 10 "${wrist_arg[@]}" \
      ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
    sim_pid=$!
    echo
    echo "run the task against it from robot_console/:"
    echo "  ./run_task.sh --label $ENGINE$([ "$PORT" = 9090 ] || echo " --url ws://127.0.0.1:$PORT") --episodes 6"
    echo "  (layout: $([ "$SWAP" -eq 1 ] && echo 'swapped -- plate at the apple spawn' || echo 'standard'); the console reads it off the wire)"
    echo "watch it:"
    echo "  ./kitchen.sh cameras --engine $ENGINE$([ "$PORT" = 9090 ] || echo " --port $PORT")"
    wait
    ;;

  cameras)
    page="$ROOT/live_cameras.html"
    [ -f "$page" ] || die "missing $page"
    url="ws://127.0.0.1:$PORT"
    nc -z 127.0.0.1 "$PORT" 2>/dev/null \
      || die "nothing is serving on $url - start one with ./kitchen.sh serve --engine $ENGINE"

    # Served over HTTP rather than opened from file://, because browsers refuse a ws://
    # connection from a file:// origin and the page then sits there discovering nothing.
    port_free "$HTTP_PORT"
    python3 -m http.server "$HTTP_PORT" --directory "$ROOT" --bind 127.0.0.1 \
      >/dev/null 2>&1 &
    http_pid=$!
    trap 'kill "$http_pid" 2>/dev/null || true' INT TERM EXIT
    # `ns` tells the page which robot's state and command topics to drive. The camera
    # grid does not need it -- it discovers streams from rosapi, so every robot on the
    # port shows up regardless -- but the arm sliders address one robot and must be told
    # which. The arm is always `so101` here.
    page_url="http://127.0.0.1:$HTTP_PORT/live_cameras.html?url=$url&ns=so101"
    say "camera page: $page_url"
    echo "  cameras are discovered from the wire, so --wrist shows up without a reload"
    echo "  the sliders drive the arm once you tick 'Enable control'"
    echo "  Ctrl-C stops serving the page; the simulator keeps running"
    command -v open >/dev/null && open "$page_url"
    wait "$http_pid"
    ;;

  inspect)
    # Gone, in the tradition of --ports and --engine both: the simulator hosts the world
    # and the console runs the task. Two terminals, two projects.
    die "inspect has moved to robot_console/run_task.sh.
    terminal 1:  ./kitchen.sh serve --engine $ENGINE$([ "$VIEWER" -eq 1 ] && echo ' --viewer')
    terminal 2:  cd ../robot_console && ./run_task.sh --label $ENGINE [--episodes N]"
    ;;
esac
