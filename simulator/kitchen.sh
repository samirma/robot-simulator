#!/usr/bin/env bash
# The SO-101 on a kitchen work surface, in one engine at a time.
#
#   ./kitchen.sh serve           host the world: the engine, the kitchen, the robots and
#                                the cameras. The default command, and the only one that
#                                starts anything.
#   ./kitchen.sh view            open the live camera page on a simulator that is
#                                already serving. Takes --port and --http-port, and
#                                none of the flags that configure a simulator.
#   ./kitchen.sh help
#
# One command hosts the world and one looks at it, and they do not overlap: `view` does
# not stage a task, choose an engine or pick a camera set, because it does not start the
# process those would configure. Running a task against a `serve` is a third thing again,
# and it is the console's:
#
#   cd ../robot_console && ./run_task.sh [--episodes N]
#
# `shot`, `inspect` and `cameras` used to live here and no longer do. `inspect` graded
# episodes, which is the console's half of the split; `shot` rendered a screenshot per
# engine back when the two could be started together, and comparing engines is now done
# by running each in turn against the same client; `cameras` is what `view` is now.
#
#   what a serve also opens (both off by default):
#   --mujoco               the engine's own MuJoCo window. It has to be a `serve` flag
#                          and cannot be a `view` one: MuJoCo renders the window inside
#                          the process holding the model, so a window belongs to the run
#                          that owns the physics and cannot be attached to one already
#                          under way. One process is also what makes the window show
#                          exactly the state on the wire. It costs camera rate --
#                          measured on MolmoSpaces with three cameras, 7.6 Hz headless
#                          against 5.0 Hz with the window -- and the loop still holds
#                          real time at that price.
#   --live                 the live camera page, in a browser, served over HTTP. Saves
#                          the second terminal; `./kitchen.sh view` is that terminal.
#
#   what is on the wire:
#   --cameras SET          which views the arm streams (default `scene`):
#                            scene  the contract's two: /overhead and /side
#                            both   those plus the eye-in-hand /wrist
#                            wrist  the eye-in-hand view ALONE
#                          `wrist` takes the two scene topics off the wire, so the
#                          console will not find the camera set it expects -- it is for
#                          isolating what a policy sees, not for running the task. Every
#                          camera is rendered inside the physics loop, so each one costs
#                          control rate for every client.
#   --robots A,B           which robots share the kitchen and the port (default so101).
#                          `so101,myagv` mounts the arm on a work surface and puts the
#                          base on the floor of the same room, both on one rosbridge
#                          under /so101/* and /myagv/* -- one ROS graph, a namespace per
#                          robot, which is how a real multi-robot bringup is arranged.
#                          run_task.sh still grades the arm; pass it the same --robots
#                          and it checks the extra robot is on the wire.
#
#   which kitchen:
#   --engine E             molmospaces (default) | robocasa. One engine per run: only
#                          the engine named here needs to be set up, and every command
#                          honours it.
#   --scene ithor:1        MolmoSpaces scene    (default: ithor:1, a kitchen)
#   --layout 1 --style 1   RoboCasa kitchen     (both 1-60)
#   --objects plate,apple  categories that *rank* MolmoSpaces surfaces (its `--target`),
#                          which is how the arm ends up on the counter holding the pair
#                          rather than on whichever worktop scored best. It spawns
#                          nothing: every command here stages a task, and the task brings
#                          its own measured apple and plate 0.32 m from the arm and
#                          clears the workspace itself. RoboCasa ignores it, because its
#                          sampler would add a second apple the jaw cannot close on.
#                          The engine prints both reach distances at startup, so "in
#                          reach" is read, not assumed.
#   --port 9090            rosbridge port
#   --http-port 8791       port the live camera page is served on
#
#   what the task stages:
#   --reference-table      also stage the reference rig's 0.92 m wooden work surface
#                          under the objects. Off by default: it sits on top of the
#                          kitchen's own counter and reads as one table overlapping
#                          another, and measured it does not move the VLA's pass count
#                          (2/6 on the bare counter against 1/6 with the slab).
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
#                          (reflected across the arm's x-z plane). RoboCasa only, where
#                          the swapped layout puts the plate between the reference side
#                          view and the apple; from the other side the apple is the near
#                          object. An experiment flag; the policy is still told the
#                          contract poses in its docs.
#
# One engine per run. Two at once meant two ports, two of every flag, and two kitchens
# competing for one GPU -- which on this machine is not a theoretical cost: RoboCasa's
# 44-fixture kitchen with its cameras and a viewer window already runs at 0.12x real time
# on its own, and the client refuses to start below 0.10x. Comparing the engines is still
# the point, and running each in turn against the same client is what actually
# demonstrates the thing worth demonstrating -- that the client cannot tell them apart.
#
# What the task brings, and what the engines bring. Both commands stage
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
# 9090 is the rosbridge default and what a real bringup for this arm presents, so it is
# the right port for whichever engine is running. There is only ever one.
PORT="9090"
HTTP_PORT=8791
# Which robots share the kitchen, and therefore the port. One ROS graph with a namespace
# per robot -- `so101` alone is on /so101/*, and `so101,myagv` adds /myagv/* beside it on
# the same socket. The default is the arm alone, so every existing invocation is
# unchanged.
ROBOTS="so101"
ENGINE="molmospaces"
CAMERAS="scene"
# What a `serve` opens for itself, beyond the wire. Neither, unless asked.
MUJOCO=0
LIVE=0
declare -a STAGE_FLAGS=()
declare -a CAMERA_FLAGS=()
REFERENCE_TABLE=0
# Whether the plate and the apple trade places. "auto" resolves per engine once the
# engine is known: on for robocasa, off for molmospaces -- one engine keeps the
# contract's arrangement and the other shows the policy the same objects the other way
# round. The console needs no matching flag; it reads the layout off the wire.
SWAP="auto"

die() { echo "error: $*" >&2; exit 1; }
say() { printf '\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- arguments

cmd="serve"
case "${1:-}" in
  view|serve|help|-h|--help) cmd="$1"; shift || true ;;
esac

if [ "$cmd" = "help" ] || [ "$cmd" = "-h" ] || [ "$cmd" = "--help" ]; then
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
  exit 0
fi

# Two commands, two argument lists, kept apart rather than one list filtered afterwards.
# `view` starts no process, so every flag that configures one is a `serve` flag, and
# refusing them here is what makes "view is exclusively the view" structural instead of
# a rule someone has to remember.
if [ "$cmd" = view ]; then
  while [ $# -gt 0 ]; do
    case "$1" in
      --port)      PORT="$2";      shift 2 ;;
      --http-port) HTTP_PORT="$2"; shift 2 ;;
      # Exactly what `view` is. Accepted because it is what a `serve --live` habit types,
      # and there is nothing else this command could be asked for.
      --live)      shift ;;
      --mujoco)    die "a MuJoCo window renders inside the process that holds the model, so it
    cannot be opened onto a simulator that is already running. It is a serve flag:
      ./kitchen.sh serve --mujoco" ;;
      *) die "'$1' configures a simulator and \`view\` starts none - it opens the camera
    page on one that is already serving. Configure it where it is started:
      ./kitchen.sh serve $1 ..." ;;
    esac
  done
else
  while [ $# -gt 0 ]; do
    case "$1" in
      --objects)  OBJECTS="$2"; shift 2 ;;
      --scene)    SCENE="$2";   shift 2 ;;
      --layout)   LAYOUT="$2";  shift 2 ;;
      --style)    STYLE="$2";   shift 2 ;;
      --port)     PORT="$2";    shift 2 ;;
      --http-port) HTTP_PORT="$2"; shift 2 ;;
      --engine)   ENGINE="$2";  shift 2 ;;
      --robots)   ROBOTS="$2";  shift 2 ;;
      --cameras)  CAMERAS="$2"; shift 2 ;;
      --mujoco)   MUJOCO=1; shift ;;
      --live)     LIVE=1;   shift ;;
      --reference-table)    REFERENCE_TABLE=1; shift ;;
      --no-reference-table) REFERENCE_TABLE=0; shift ;;
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

  case "$ENGINE" in
    molmospaces|robocasa) ;;
    *) die "--engine: expected molmospaces or robocasa" ;;
  esac

  # The wrist view is a topic on top of the contract's two; `wrist` alone takes those two
  # away, which is the one setting the console cannot run against. Both are engine flags
  # and both engines take them, so the camera set is not somewhere the two can drift apart.
  case "$CAMERAS" in
    scene) ;;
    both)  CAMERA_FLAGS=(--wrist-camera) ;;
    wrist) CAMERA_FLAGS=(--wrist-camera --no-scene-cameras) ;;
    *)     die "--cameras: expected scene, both or wrist" ;;
  esac

  [ "$REFERENCE_TABLE" -eq 1 ] || STAGE_FLAGS+=(--no-reference-table)

  # The swap resolves per engine only once the engine is known -- see SWAP above.
  if [ "$SWAP" = auto ]; then
    [ "$ENGINE" = robocasa ] && SWAP=1 || SWAP=0
  fi
  [ "$SWAP" -eq 0 ] || STAGE_FLAGS+=(--swap-objects)

fi

engine_root() { [ "$1" = molmospaces ] && echo "$MOLMO" || echo "$ROBOCASA"; }

# Only the engine actually being run has to be installed. Setting up the other is a large
# download, and requiring it in order to use this one is a barrier with nothing behind it.
# `view` reaches neither: it starts no engine, so watching another terminal's RoboCasa
# does not require this checkout to have one.
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
    # No `--objects`: every command here stages a task, which brings its own objects at
    # measured positions, and RoboCasa's sampler would add a second apple the jaw cannot
    # close on plus a bowl inside the plate's footprint. spawn_robot.py refuses the
    # combination outright; not passing it is what keeps it from ever being asked for.
    exec "$1" "$ROBOCASA/tools/spawn_robot.py" "$ROBOTS" --layout "$LAYOUT" --style "$STYLE" "${@:2}"
  )
}

# The MuJoCo passive viewer must own the main thread on macOS, which is what mjpython
# provides; anything windowless runs under plain python. Same rule as both run.sh files.
# With a window the offscreen camera renderers and the on-screen one coexist in one
# process, which was not obviously safe beforehand -- under mjpython the script runs off
# the main thread and each renderer opens a hidden GLFW window -- and was verified before
# --mujoco existed. If it ever stops working, both env.sh files name MUJOCO_GL=cgl as the
# escape hatch.
engine_python() {
  local root; root="$(engine_root "$ENGINE")"
  if [ "$MUJOCO" -eq 1 ] && [ "$(uname -s)" = "Darwin" ]; then
    echo "$root/.venv/bin/mjpython"
  else
    echo "$root/.venv/bin/python"
  fi
}
headless_arg() { [ "$MUJOCO" -eq 1 ] || echo "--headless"; }

# 9090 is the rosbridge default and what a real bringup for this arm uses, so it is the
# right default here -- but it is also a popular port, and a sibling checkout running its
# own simulator is the likeliest thing holding it. Naming the holder turns a puzzling
# failure into an obvious one.
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

# Compiling a kitchen takes the better part of a minute, and the page has nothing to
# discover until rosbridge is up -- opening the browser first shows an empty grid and
# reads as a broken page rather than a slow compile.
wait_for_rosbridge() {
  local i
  for i in $(seq 1 240); do
    nc -z 127.0.0.1 "$PORT" 2>/dev/null && return 0
    kill -0 "$sim_pid" 2>/dev/null || die "the engine exited before it served on port $PORT"
    sleep 1
  done
  echo "warning: nothing on port $PORT yet; opening the page anyway" >&2
}

# Served over HTTP rather than opened from file://, because browsers refuse a ws://
# connection from a file:// origin and the page then sits there discovering nothing.
serve_page() {
  local page="$ROOT/live_cameras.html"
  [ -f "$page" ] || die "missing $page"
  port_free "$HTTP_PORT" "pick another with --http-port PORT"
  python3 -m http.server "$HTTP_PORT" --directory "$ROOT" --bind 127.0.0.1 >/dev/null 2>&1 &
  http_pid=$!
  # `ns` tells the page which robot's state and command topics to drive. The camera grid
  # does not need it -- it discovers streams from rosapi, so every robot on the port shows
  # up regardless -- but the arm sliders address one robot and must be told which. The arm
  # is always `so101` here.
  local page_url="http://127.0.0.1:$HTTP_PORT/live_cameras.html?url=ws://127.0.0.1:$PORT&ns=so101"
  say "camera page: $page_url"
  echo "  cameras are discovered from the wire, so --cameras both shows the wrist view"
  echo "  without a reload; the sliders drive the arm once you tick 'Enable control'"
  command -v open >/dev/null && open "$page_url" || true
}

# ---------------------------------------------------------------- run

if [ "$cmd" = view ]; then
  # The page discovers what is on the wire, so it needs the wire and not an engine of its
  # own. Nothing there is the one thing it cannot work around, and it says so rather than
  # opening a browser onto an empty grid.
  nc -z 127.0.0.1 "$PORT" 2>/dev/null \
    || die "nothing is serving on ws://127.0.0.1:$PORT - start one with: ./kitchen.sh serve"
  trap cleanup INT TERM EXIT
  echo ">> the camera page on the simulator already at ws://127.0.0.1:$PORT"
  serve_page
  echo "  Ctrl-C stops serving the page; the simulator keeps running"
  wait "$http_pid"
  exit 0
fi

need_engine "$(engine_root "$ENGINE")"
# Checked up front, because the failure otherwise arrives as a websockets traceback
# from an engine that has already spent a minute compiling a kitchen.
port_free "$PORT" "pick another with --port PORT, or watch the one that is there: ./kitchen.sh view"
[ "$LIVE" -eq 1 ] && port_free "$HTTP_PORT" "pick another with --http-port PORT"
# EXIT as well as INT/TERM: without it a `die` anywhere below leaves the engine
# holding its port, and the next run fails the port check for no visible reason.
trap cleanup INT TERM EXIT
echo ">> $ENGINE $ROBOTS on ws://127.0.0.1:$PORT (cameras: $CAMERAS$([ "$MUJOCO" -eq 1 ] && echo ', with a window'))"
"$ENGINE" "$(engine_python)" $(headless_arg) --ros-port "$PORT" \
  --task apple_on_plate --control-hz 10 \
  ${CAMERA_FLAGS[@]+"${CAMERA_FLAGS[@]}"} ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
sim_pid=$!
if [ "$LIVE" -eq 1 ]; then
  wait_for_rosbridge
  serve_page
fi
echo
echo "run the task against it from robot_console/:"
echo "  ./run_task.sh --label $ENGINE$([ "$PORT" = 9090 ] || echo " --url ws://127.0.0.1:$PORT") --episodes 6"
echo "  (layout: $([ "$SWAP" -eq 1 ] && echo 'swapped -- plate at the apple spawn' || echo 'standard'); the console reads it off the wire)"
[ "$LIVE" -eq 1 ] || echo "watch it:  ./kitchen.sh view$([ "$PORT" = 9090 ] || echo " --port $PORT")"
wait "$sim_pid"
